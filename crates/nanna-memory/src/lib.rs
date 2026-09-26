#![warn(clippy::all)]
#![warn(clippy::pedantic, clippy::nursery)]
// Raised for the same reason as `nanna-daemon`'s: proving a future or closure
// is `Send` walks into wgpu's `Global`/`Hub`/`Registry` graph by way of
// `CosineSimilaritySearch`, which is deeper than the default limit of 128.
// nightly-2026-08-25 turned that overflow into the future-incompatible
// `recursion_depth_exceeding_limit` warning (rust#159228), which is scheduled
// to become a hard error. Solver depth only — no behaviour, no codegen change.
#![recursion_limit = "256"]

//! Memory and embedding system for Nanna
//!
//! Provides vector storage and semantic search with SIMD and GPU acceleration.
//! Implements FSRS-6 for cognitive memory decay and the "dreaming" consolidation model.

mod activity;
pub mod chunk_rank;
pub mod chunking;
mod consolidation;
mod dreaming;
mod fsrs;
pub mod retention;
mod service;

/// Integer <-> `f32` conversions for scores, averages and budgets, where std has
/// no lossless route. Each one is value-for-value the `as` cast it replaced, by
/// way of `nanna_numeric`, which states the rounding once for the workspace.
mod lossy {
    /// An integer as the nearest `f32` (ties to even), exactly as `as f32`.
    pub trait LossyF32 {
        fn lossy_f32(self) -> f32;
    }

    impl LossyF32 for usize {
        fn lossy_f32(self) -> f32 {
            nanna_numeric::f32_from_usize(self)
        }
    }

    impl LossyF32 for u32 {
        fn lossy_f32(self) -> f32 {
            nanna_numeric::f32_from_u32(self)
        }
    }

    impl LossyF32 for u64 {
        fn lossy_f32(self) -> f32 {
            nanna_numeric::f32_from_u64(self)
        }
    }

    impl LossyF32 for i64 {
        fn lossy_f32(self) -> f32 {
            nanna_numeric::f32_from_i64(self)
        }
    }

    /// `x` truncated toward zero, saturating at the bounds and mapping NaN to 0,
    /// exactly as `as usize`.
    pub fn f32_to_usize(x: f32) -> usize {
        nanna_numeric::usize_from_f32(x)
    }

    /// `x` truncated toward zero, saturating at the bounds and mapping NaN to 0,
    /// exactly as `as i64`.
    pub fn f32_to_i64(x: f32) -> i64 {
        nanna_numeric::i64_from_f64(f64::from(x))
    }
}

pub use activity::ActivityClock;

pub use chunk_rank::{collapse_chunk_hits, ChunkHit};
pub use chunking::{chunk_text, derive_chunk_params, Chunk, ChunkParams, CHUNKER_VERSION};

pub use consolidation::{
    ConsolidationConfig, ConsolidationResult, CompressionLevel,
    WeightThresholds, ClusteringWeights, MemoryCluster, cluster_memories,
    create_consolidated_entry, composite_cluster_score, cluster_score_ceiling,
    cluster_content_bytes_for_context, FALLBACK_SUMMARIZER_CONTEXT_WINDOW_TOKENS,
    is_verbatim_pinned, FACT_TYPE_METADATA_KEY,
};
pub use dreaming::{
    dream_trigger, DreamOutcome, DreamTrigger, DreamingConfig, DreamingService, DreamingStats,
    MemoryFeedback, make_summarize_fn, LlmSummarizer,
};
pub use fsrs::{
    FsrsParameters, FsrsState, MemoryState, Rating, IngestAction,
    power_law_retrievability,
};
// The decay exponent's published value and the range a fitted one may take.
// Exported because `FsrsParameters::w20` is public and its doc names them: a
// caller overriding the exponent needs the same bounds the default is held to.
pub use fsrs::{DECAY_MAX, DECAY_MIN, FSRS5_DEFAULT_DECAY, FSRS6_DEFAULT_DECAY};
pub use service::{
    MemoryService, MemoryServiceConfig, RecallResult, RecallReport, RecallTimings, EmbedFn,
    MemoryStats, MemoryListEntry, MemoryExportRecord, ConsolidationBands,
    // Exported so the episodic writer chunks against the same bound the merge
    // path caps against. Those two numbers drifting apart is precisely how a
    // 4096-byte ceiling ended up silently rejecting 3200-char chunks.
    MEMORY_CHUNK_MAX_CHARS, MEMORY_OBSERVATION_MAX_CHARS,
    MEMORY_CHUNK_TARGET_CHARS, chunk_max_chars_for_window,
};
pub use retention::{
    measure_gated_recall, measure_recall, run_retention_cycle, CorpusParams,
    RetentionCorpus, RetentionMeasurement, RetentionProbe, RetentionReport,
    topic_centroid, TOPIC_METADATA_KEY,
};

use async_trait::async_trait;
use nanna_gpu::{CosineSimilaritySearch, GpuContext};
use nanna_simd::{cosine_similarity_f32, normalize_f32};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

#[derive(Error, Debug)]
pub enum MemoryError {
    #[error("Embedding dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("Memory not found: {0}")]
    NotFound(String),
    /// No embedding provider is wired up, so anything needing a vector cannot run.
    ///
    /// This is its own variant because it is the one memory failure a *user* can
    /// fix, and the message is read by the agent: it used to be reported as
    /// `MemoryError::Io(NotConnected, "No embedding function configured")`, which
    /// surfaced to the model as `IO error: ...`. A model cannot act on an IO
    /// error — it retries or gives up — so `recall` looked broken rather than
    /// unconfigured. Keep the text actionable and addressed to the person.
    #[error(
        "no embedding provider configured — set one in Settings, or run a local Ollama \
         with an embedding model pulled, then memory search will work"
    )]
    NoEmbeddingProvider,
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("Persistence error: {0}")]
    Persistence(String),
    /// The clustering configuration would merge memories that are not related.
    ///
    /// Refusing is the conservative direction here and the only safe one:
    /// consolidation *rewrites* memories, replacing several with one gist, so a
    /// cycle run under a config in which cosine similarity has no veto destroys
    /// content it had no evidence was redundant. A warning would be read after
    /// the merge had already happened.
    #[error("invalid clustering configuration: {0}")]
    InvalidClusteringConfig(String),
}

/// Trait for pluggable persistence backends (Turso, etc.)
///
/// Implementors are responsible for durably storing and retrieving memory entries.
/// The in-memory vector cache remains the primary store for search; this layer
/// provides crash-safe persistence.
/// Result of loading the durable store, including any rows skipped as corrupt.
#[derive(Debug, Default)]
pub struct LoadReport {
    pub entries: Vec<MemoryEntry>,
    /// Rows that existed but could not be read (corrupt payload / overflow chain).
    pub corrupt_rows: usize,
    /// Total rows the table reported (`entries.len() + corrupt_rows` on a salvage).
    pub expected: usize,
}

/// Health of the durable memory store after the startup load.
///
/// `degraded` is true when any row was unreadable, so it can be surfaced
/// instead of the silent "loaded 0 of N" that a whole-table corruption used to
/// cause.
#[derive(Debug, Clone, Copy, Default)]
pub struct MemoryStoreHealth {
    pub degraded: bool,
    pub corrupt_rows: usize,
    pub loaded: usize,
    pub expected: usize,
}

/// What a durable batch removal did. See [`VectorStore::remove_many_durable`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DurableRemoval {
    /// Entries whose row the backend confirmed gone, and which left RAM.
    pub removed: usize,
    /// Ids the backend refused to delete. They are still in RAM and on disk.
    pub failed: usize,
    /// The first backend error among `failed`, for the caller to report.
    pub first_error: Option<String>,
}

/// Which memories a recall may return.
///
/// One definition for every recall path, and the only place the rule lives.
/// A second, narrower copy used to sit in the chunk-only pass and silently
/// dropped a global memory that only its chunks matched (2026-09-22 review);
/// and the daemon's `memory.search` could not express "global only" at all, so
/// it answered `scope:"global"` with every memory while `list` and `export`
/// answered it with the global ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecallScope<'a> {
    /// Every memory, whatever its workspace.
    Everything,
    /// Only memories that belong to no workspace.
    GlobalOnly,
    /// That workspace's memories plus the global ones — what a workspace sees.
    Workspace(&'a str),
}

impl<'a> RecallScope<'a> {
    /// The historical `Option<&str>` scope: `None` = everything, `Some(id)` =
    /// that workspace plus the globals.
    #[must_use]
    pub const fn from_workspace(workspace_id: Option<&'a str>) -> Self {
        match workspace_id {
            None => Self::Everything,
            Some(ws) => Self::Workspace(ws),
        }
    }

    /// Whether a memory owned by `memory_workspace` is visible in this scope.
    #[must_use]
    pub fn admits(self, memory_workspace: Option<&str>) -> bool {
        debug_assert!(
            !matches!(self, Self::Workspace("")),
            "a workspace scope names a workspace"
        );
        match self {
            Self::Everything => true,
            Self::GlobalOnly => memory_workspace.is_none(),
            Self::Workspace(ws) => memory_workspace.is_none_or(|owner| owner == ws),
        }
    }

    /// The workspace a SQL-side prefilter may narrow to. `GlobalOnly` has no
    /// SQL form, so it scans unscoped and [`Self::admits`] does the cut.
    #[must_use]
    pub const fn sql_workspace(self) -> Option<&'a str> {
        match self {
            Self::Workspace(ws) => Some(ws),
            Self::Everything | Self::GlobalOnly => None,
        }
    }
}

/// What a search could actually COMPARE, measured during the scan it describes.
///
/// An empty result set has three causes and they are not interchangeable:
/// nothing matched, the rows exist but carry no vector the query can be scored
/// against, or the query itself is the wrong width for the store's binding and
/// nothing was scanned at all. Reported as "No memories found" they are
/// indistinguishable — a total blackout reads as an honest miss, and the reader
/// stops looking.
///
/// Measured in the scan rather than read from a rebind-time counter, because a
/// snapshot taken when the binding changed goes stale the moment the backfill
/// starts draining. The width predicate is already evaluated per row by the
/// store's `cosine_or_stale`, so counting it costs nothing and is exact at
/// answer time.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SearchCoverage {
    /// Entries in the store when the scan ran.
    pub total: usize,
    /// Entries whose stored vector had the query's width — the only ones a
    /// cosine can score. The remainder are queued for backfill (empty vector)
    /// or still bound to a superseded model's width.
    pub comparable: usize,
    /// False when the QUERY's width disagrees with the store's binding, in
    /// which case no row was scanned: every memory is unreachable, not absent.
    pub query_width_matches: bool,
}

impl SearchCoverage {
    /// Entries the scan could not score. Never a "no match" — a row that was
    /// not compared was not consulted.
    #[must_use]
    pub const fn unsearchable(&self) -> usize {
        self.total.saturating_sub(self.comparable)
    }

    /// Whether every entry in the store was actually scored, so an empty
    /// result really does mean "nothing matched".
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        self.query_width_matches && self.unsearchable() == 0
    }
}

#[async_trait]
pub trait MemoryPersistence: Send + Sync {
    /// Persist (insert or update) a single entry.
    async fn save_entry(&self, entry: &MemoryEntry) -> Result<(), MemoryError>;
    /// Remove an entry by its ID.
    async fn remove_entry(&self, id: &str) -> Result<(), MemoryError>;
    /// Remove multiple entries by ID in one batch.
    ///
    /// The default loops [`remove_entry`](Self::remove_entry); a backing store
    /// that can amortize per-delete cost across the batch (e.g. a single WAL
    /// checkpoint for the whole set, rather than one per row) overrides this. The
    /// dream cycle removes a cluster's superseded sources through this path, so a
    /// per-row checkpoint there would fsync once per consolidated memory.
    async fn remove_entries(&self, ids: &[&str]) -> Result<(), MemoryError> {
        for id in ids {
            self.remove_entry(id).await?;
        }
        Ok(())
    }
    /// Update only the FSRS cognitive state for an existing entry.
    async fn update_entry_fsrs(&self, id: &str, fsrs: &FsrsState) -> Result<(), MemoryError>;
    /// Update only the text content for an existing entry.
    async fn update_entry_content(&self, id: &str, content: &str) -> Result<(), MemoryError>;
    /// Replace the stored chunk set for `memory_id` with `chunks`.
    ///
    /// Whole-set replacement, not a patch: chunk boundaries move when content
    /// changes, so an ordinal that meant one span before means a different span
    /// after, and merging the two would interleave text from two different
    /// splittings. Deleting and rewriting is the only operation whose result
    /// does not depend on what was there.
    ///
    /// The default is a no-op, which is correct for a store with no durable
    /// backing: chunks are derived data, reproducible from content at any time.
    /// `model` is the binding these chunks were produced under, and is what
    /// unembedded chunks get queued against. Without it the queue could not
    /// name which bucket the work belongs to.
    async fn replace_chunks(
        &self,
        _memory_id: &str,
        _workspace_id: Option<&str>,
        _model: Option<&str>,
        _chunks: &[ChunkWrite],
    ) -> Result<(), MemoryError> {
        Ok(())
    }

    /// Chunks with no vector for `model` yet, oldest first, at most `limit`.
    ///
    /// "No vector" and "a vector from a different model" are the same state
    /// here, and that is what makes an embedding-model switch a resumable
    /// incremental backfill instead of a full rebuild: vectors from two models
    /// live in unrelated spaces, so a chunk carrying another model's vector is
    /// exactly as unsearchable under `model` as a chunk carrying none.
    ///
    /// Returns `(chunk_id, content)`. Default: nothing to do.
    /// `seed` asks the backing store to first sweep for pre-existing work
    /// (chunks from before this process, or before the queue existed). The
    /// sweep scans the whole chunk table, so callers pass it once per model
    /// per process — new writes enqueue themselves and need no sweep.
    async fn chunks_needing_embedding(
        &self,
        _model: &str,
        _limit: usize,
        _seed: bool,
    ) -> Result<Vec<PendingChunk>, MemoryError> {
        Ok(Vec::new())
    }

    /// Attach `embedding` to the chunk with `chunk_id`, stamped with `model`.
    async fn set_chunk_embedding(
        &self,
        _chunk_id: i64,
        _embedding: &[f32],
        _model: &str,
    ) -> Result<(), MemoryError> {
        Ok(())
    }

    /// Chunk-level nearest neighbours, as `(memory_id, ordinal, similarity)`.
    ///
    /// Restricted to chunks embedded by `model`. Comparing across models is
    /// meaningless — equal width is not equal vector space — and it fails
    /// silently rather than loudly, so the model is a required argument here
    /// instead of an optional filter.
    ///
    /// Similarity, not distance — the caller compares it against the same
    /// calibrated cosine threshold every other score is compared against, and
    /// a path that returned distance here would silently invert every
    /// comparison downstream.
    async fn search_chunks(
        &self,
        _query: &[f32],
        _model: &str,
        _limit: usize,
        _workspace_id: Option<&str>,
    ) -> Result<Vec<(String, i64, f32)>, MemoryError> {
        Ok(Vec::new())
    }

    /// Discard every durable bucket for `memory_id`.
    ///
    /// Called when content is rewritten: every stored vector describes the old
    /// text, so keeping any of them leaves the memory findable by words it no
    /// longer contains.
    async fn clear_memory_buckets(&self, _memory_id: &str) -> Result<(), MemoryError> {
        Ok(())
    }

    /// Re-activate a model's stored chunk vectors after a rebind, returning how
    /// many were restored. This is what makes returning to a provider free.
    async fn restore_chunk_vectors(&self, _model: &str) -> Result<usize, MemoryError> {
        Ok(0)
    }

    /// Remove completed work from the durable queue.
    async fn dequeue_embedding(
        &self,
        _memory_id: &str,
        _ordinal: i64,
        _model: &str,
    ) -> Result<(), MemoryError> {
        Ok(())
    }

    /// Record a failed embedding attempt against queued work, with its reason.
    ///
    /// The reason is what separates a queue that is stuck from one that is
    /// merely slow — a 402 with nothing attached reads as silence.
    async fn record_embedding_failure(
        &self,
        _memory_id: &str,
        _ordinal: i64,
        _model: &str,
        _error: &str,
    ) -> Result<(), MemoryError> {
        Ok(())
    }

    /// Queue depth per model with the most recent error for each.
    async fn embedding_queue_health(
        &self,
    ) -> Result<Vec<(String, usize, Option<String>)>, MemoryError> {
        Ok(Vec::new())
    }

    /// Memory ids that have no chunk rows at all, at most `limit`.
    ///
    /// These are memories written before chunking existed. They are searchable
    /// by their whole-row vector, so this is a quality backfill rather than a
    /// repair — but a store where only new memories are chunk-searchable
    /// retrieves inconsistently, which is worse than either extreme.
    async fn parents_without_chunks(&self, _limit: usize) -> Result<Vec<String>, MemoryError> {
        Ok(Vec::new())
    }

    /// Load all persisted entries (called on startup to populate the in-memory cache).
    async fn load_all(&self) -> Result<Vec<MemoryEntry>, MemoryError>;

    /// Load all entries with a corruption report (rows skipped as unreadable).
    ///
    /// The default delegates to [`load_all`](Self::load_all) and reports zero
    /// corruption; a backing store able to salvage individual rows overrides this
    /// so one bad row no longer aborts the whole load.
    async fn load_all_report(&self) -> Result<LoadReport, MemoryError> {
        let entries = self.load_all().await?;
        let expected = entries.len();
        Ok(LoadReport { entries, corrupt_rows: 0, expected })
    }
}

/// A chunk awaiting a vector, carrying everything needed to embed it AND to
/// report against it.
///
/// The queue key (`memory_id`, `ordinal`) travels with the row id because a
/// failure has to be recorded against the durable queue, not against the
/// chunk table — otherwise a provider that fails forever leaves no trace and
/// the queue looks merely slow.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingChunk {
    pub chunk_id: i64,
    pub memory_id: String,
    pub ordinal: i64,
    pub content: String,
}

/// One chunk on its way to durable storage.
///
/// Mirrors the stored row minus the parent identity, which travels alongside as
/// an argument to [`MemoryPersistence::replace_chunks`].
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkWrite {
    pub ordinal: i64,
    pub content: String,
    pub char_start: i64,
    pub char_end: i64,
    /// Usually `None` — chunk vectors are filled by the backfill pass, not on
    /// the write path, because embedding N chunks inline would put N network
    /// round trips in front of every remember. The exception is a memory that
    /// fits in one chunk: its single chunk IS the parent text, so the parent's
    /// vector already describes it exactly and re-embedding would be waste.
    pub embedding: Option<Vec<f32>>,
    pub embedding_model: Option<String>,
    pub chunk_max_chars: i64,
    pub chunker_version: i64,
}

/// A memory entry with embedding, metadata, and FSRS cognitive state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id: String,
    pub content: String,
    /// The embedding for the CURRENTLY active model — the one search compares
    /// against. Empty means "not embedded by the active model yet", which is a
    /// backfill job, not an error.
    pub embedding: Vec<f32>,
    /// Which model produced [`Self::embedding`].
    ///
    /// Dimension alone cannot answer this: two different 1024-dim models
    /// produce vectors that compare without asserting and mean nothing. Naming
    /// the model is the only way to tell "the same vector space" from "the same
    /// width".
    #[serde(default)]
    pub embedding_model: Option<String>,
    /// Every embedding ever computed for this entry, keyed by model identity
    /// (`provider:model`), including the active one.
    ///
    /// Providers flap — a rate-limited primary falls over and comes back on the
    /// next call. Throwing away the previous vector on every switch meant each
    /// flap cost a full re-embed of the store, and a flap back cost another.
    /// Retained buckets make a switch-back free: rebinding is a lookup, and only
    /// genuinely new (model, entry) pairs need work.
    #[serde(default)]
    pub embeddings: HashMap<String, Vec<f32>>,
    pub metadata: HashMap<String, String>,
    pub timestamp: i64,
    /// FSRS-6 cognitive state (stability, retrievability, etc.)
    #[serde(default)]
    pub fsrs: FsrsState,
    /// Workspace ID if this memory is scoped to a workspace (None = global)
    #[serde(default)]
    pub workspace_id: Option<String>,
}

/// Vector store configuration
#[derive(Debug)]
pub struct VectorStoreConfig {
    /// Expected embedding dimension. Stored as `AtomicUsize` so it can be
    /// updated at runtime (e.g., when the embedding model changes via fallback)
    /// without requiring `&mut self` on the `VectorStore`.
    pub dimension: std::sync::atomic::AtomicUsize,
    /// Chunk budget in chars for the currently bound embedding model, derived
    /// from that model's input window. Atomic for the same reason `dimension`
    /// is: a provider switch has to move it without `&mut self`.
    ///
    /// Zero is the unbound sentinel, NOT a bound of nothing — see
    /// [`Self::get_chunk_max_chars`].
    pub chunk_max_chars: std::sync::atomic::AtomicUsize,
    pub use_f16: bool,  // Store embeddings as f16 to save memory
}

impl Clone for VectorStoreConfig {
    fn clone(&self) -> Self {
        Self {
            dimension: std::sync::atomic::AtomicUsize::new(
                self.dimension.load(std::sync::atomic::Ordering::Relaxed)
            ),
            chunk_max_chars: std::sync::atomic::AtomicUsize::new(
                self.chunk_max_chars.load(std::sync::atomic::Ordering::Relaxed)
            ),
            use_f16: self.use_f16,
        }
    }
}

impl Default for VectorStoreConfig {
    fn default() -> Self {
        Self {
            dimension: std::sync::atomic::AtomicUsize::new(1536),  // OpenAI ada-002 default
            chunk_max_chars: std::sync::atomic::AtomicUsize::new(0),
            use_f16: true,
        }
    }
}

impl VectorStoreConfig {
    /// Create config with specified dimension
    #[must_use]
    pub const fn with_dimension(dim: usize) -> Self {
        Self {
            dimension: std::sync::atomic::AtomicUsize::new(dim),
            chunk_max_chars: std::sync::atomic::AtomicUsize::new(0),
            use_f16: true,
        }
    }

    /// Get the current dimension
    pub fn get_dimension(&self) -> usize {
        self.dimension.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Set a new dimension at runtime
    pub fn set_dimension(&self, dim: usize) {
        self.dimension.store(dim, std::sync::atomic::Ordering::Relaxed);
    }

    /// The chunk budget for the currently bound embedding model, in chars.
    ///
    /// Zero means nothing has bound yet, which resolves to the
    /// retrieval-granularity default rather than to "no limit" — see
    /// [`crate::chunking::derive_chunk_params`].
    pub fn get_chunk_max_chars(&self) -> usize {
        let latched = self.chunk_max_chars.load(std::sync::atomic::Ordering::Relaxed);
        if latched == 0 {
            crate::chunking::derive_chunk_params(None).max_chars
        } else {
            latched
        }
    }

    /// Set the chunk budget at runtime.
    ///
    /// Lives beside the dimension latch, and under the same discipline: both
    /// are only ever moved while `MemoryService`'s binding write-lock is held,
    /// so no reader can observe a new model with an old geometry.
    pub fn set_chunk_max_chars(&self, max_chars: usize) {
        self.chunk_max_chars
            .store(max_chars, std::sync::atomic::Ordering::Relaxed);
    }
}

/// In-memory vector store with SIMD and GPU-accelerated search
pub struct VectorStore {
    config: VectorStoreConfig,
    entries: RwLock<Vec<MemoryEntry>>,
    gpu: Option<Arc<GpuContext>>,
    gpu_pipeline: Option<CosineSimilaritySearch>,
    /// Optional Turso (or other) backing store for durable persistence.
    /// When set, writes (add/remove/update) are mirrored to the backing store.
    /// Search always operates purely in-memory.
    db: Option<Arc<dyn MemoryPersistence + Send + Sync>>,
    /// Durable-store health from the last `load_from_db` (degraded if any row was
    /// unreadable). Surfaced so a corrupt store isn't a silent empty one.
    store_health: RwLock<MemoryStoreHealth>,
    /// Models whose pre-existing-work sweep (adopt + seed) has already run
    /// this process. The sweep scans the whole chunk table, and running it on
    /// every drain batch made the backfill quadratic in store size.
    seeded_models: std::sync::Mutex<std::collections::HashSet<String>>,
    /// `provider:model` of the embedder currently bound.
    ///
    /// Lives HERE, with the width and chunk-geometry latches, rather than
    /// beside them in `MemoryService` — one copy, moved under one lock, so
    /// there is no second value that can go stale. Every write path needs it to
    /// stamp which bucket its work belongs to.
    active_model: RwLock<Option<String>>,
    /// How long recent memories waited between being written without a vector
    /// and receiving their first one — Nanna's staleness number (P13): until
    /// then a memory exists but no similarity search can find it. A bounded
    /// ring of the last [`SEARCHABLE_LATENCY_SAMPLES_MAX`] waits, in seconds.
    searchable_latency: std::sync::Mutex<std::collections::VecDeque<u64>>,
}

/// Samples kept for the queue-to-searchable percentiles.
///
/// Sized for the p95: with 512 samples the 95th percentile has 25 samples above
/// it, so one outlier cannot move it, while the window stays recent — a busy
/// mission defers hundreds of writes an hour. 4 KiB.
pub const SEARCHABLE_LATENCY_SAMPLES_MAX: usize = 512;

/// Queue-to-searchable latency over the recent window: how long memories
/// written without a vector stayed unfindable by similarity search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SearchableLatency {
    /// Waits in the window, at most [`SEARCHABLE_LATENCY_SAMPLES_MAX`].
    pub samples: usize,
    pub p50_secs: u64,
    pub p95_secs: u64,
}

/// Append `sample`, dropping the oldest once the ring is full.
fn push_bounded(ring: &mut std::collections::VecDeque<u64>, sample: u64) {
    if ring.len() >= SEARCHABLE_LATENCY_SAMPLES_MAX {
        ring.pop_front();
    }
    ring.push_back(sample);
    debug_assert!(!ring.is_empty(), "the sample just pushed is there");
    debug_assert!(
        ring.len() <= SEARCHABLE_LATENCY_SAMPLES_MAX,
        "the ring is bounded"
    );
}

/// Nearest-rank percentiles over `samples`; `None` when there are none.
fn latency_percentiles(samples: &[u64]) -> Option<SearchableLatency> {
    if samples.is_empty() {
        return None;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    // Nearest rank: the smallest sample with at least `percent`% of the
    // samples at or below it.
    let rank = |percent: usize| -> u64 {
        let index = (sorted.len() * percent).div_ceil(100).saturating_sub(1);
        sorted.get(index).copied().unwrap_or_default()
    };
    let latency = SearchableLatency {
        samples: sorted.len(),
        p50_secs: rank(50),
        p95_secs: rank(95),
    };
    debug_assert!(
        latency.p50_secs <= latency.p95_secs,
        "percentiles are ordered"
    );
    Some(latency)
}

impl VectorStore {
    #[must_use]
    pub fn new(config: VectorStoreConfig) -> Self {
        Self {
            config,
            entries: RwLock::new(Vec::new()),
            gpu: None,
            gpu_pipeline: None,
            db: None,
            store_health: RwLock::new(MemoryStoreHealth::default()),
            seeded_models: std::sync::Mutex::new(std::collections::HashSet::new()),
            active_model: RwLock::new(None),
            searchable_latency: std::sync::Mutex::new(std::collections::VecDeque::new()),
        }
    }

    /// Queue-to-searchable latency over the recent window, or `None` before any
    /// memory written without a vector has received one in this process.
    #[must_use]
    pub fn searchable_latency(&self) -> Option<SearchableLatency> {
        let samples: Vec<u64> = self
            .searchable_latency
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .copied()
            .collect();
        latency_percentiles(&samples)
    }

    /// Create a vector store with GPU acceleration.
    ///
    /// Falls back to SIMD if GPU initialization fails.
    pub async fn with_gpu(config: VectorStoreConfig) -> Self {
        match GpuContext::new().await {
            Ok(ctx) => {
                let ctx = Arc::new(ctx);
                match CosineSimilaritySearch::new(&ctx) {
                    Ok(pipeline) => {
                        info!("VectorStore using GPU: {}", ctx.adapter_info.name);
                        Self {
                            config,
                            entries: RwLock::new(Vec::new()),
                            gpu: Some(ctx),
                            gpu_pipeline: Some(pipeline),
                            db: None,
                            store_health: RwLock::new(MemoryStoreHealth::default()),
            seeded_models: std::sync::Mutex::new(std::collections::HashSet::new()),
                            active_model: RwLock::new(None),
                            searchable_latency: std::sync::Mutex::new(
                                std::collections::VecDeque::new(),
                            ),
                        }
                    }
                    Err(e) => {
                        warn!("GPU pipeline creation failed, using SIMD: {}", e);
                        Self::new(config)
                    }
                }
            }
            Err(e) => {
                warn!("GPU initialization failed, using SIMD: {}", e);
                Self::new(config)
            }
        }
    }

    /// Attach a persistence backend.  Returns `self` for builder-style chaining.
    ///
    /// Once attached, every mutating operation (`add`, `remove`, `update_fsrs`,
    /// `update_content`) will also write through to the backing store.
    #[must_use]
    pub fn with_persistence(mut self, db: Arc<dyn MemoryPersistence + Send + Sync>) -> Self {
        self.db = Some(db);
        self
    }

    /// Load all entries from the persistence backend into the in-memory cache.
    ///
    /// This replaces any existing in-memory entries.  Call once on startup after
    /// attaching the persistence layer.
    ///
    /// # Errors
    ///
    /// Returns `MemoryError::Persistence` if the backing store fails to load.
    pub async fn load_from_db(&self) -> Result<usize, MemoryError> {
        let Some(ref db) = self.db else {
            return Err(MemoryError::Persistence("No persistence backend attached".to_string()));
        };

        let report = match db.load_all_report().await {
            Ok(r) => r,
            Err(e) => {
                // Whole-store load failed (e.g. btree corruption the salvage could
                // not localize to a single row). Mark the store degraded so
                // `/status` doesn't report a corrupt store as healthy-but-empty,
                // then surface the error as before.
                *self.store_health.write().await = MemoryStoreHealth {
                    degraded: true,
                    corrupt_rows: 0,
                    loaded: 0,
                    expected: 0,
                };
                error!("Memory store load FAILED ({e}); marked degraded (0 loaded)");
                return Err(e);
            }
        };
        let count = report.entries.len();

        // Record + surface durable-store health. A salvage load skips only the
        // corrupt rows rather than dropping the whole table, so make any loss
        // loud instead of the silent "loaded 0 of N" a corrupt store used to give.
        // `degraded` tracks genuine corruption (data loss); benign no-embedding
        // rows (readable, just not searchable) don't count.
        let degraded = report.corrupt_rows > 0;
        *self.store_health.write().await = MemoryStoreHealth {
            degraded,
            corrupt_rows: report.corrupt_rows,
            loaded: count,
            expected: report.expected,
        };
        if degraded {
            error!(
                "Memory store DEGRADED: loaded {} of {} rows; {} unreadable (corrupt). \
                 Those memories are lost until the store is repaired.",
                count, report.expected, report.corrupt_rows
            );
        }

        let mismatched = report.entries.iter()
            .filter(|e| e.embedding.len() != self.config.get_dimension())
            .count();

        if mismatched > 0 {
            let sample_dim = report.entries.iter()
                .find(|e| e.embedding.len() != self.config.get_dimension())
                .map_or(0, |e| e.embedding.len());
            warn!(
                "Dimension mismatch loading from DB: {} of {} entries have {} dims (expected {}). \
                 They will be re-embedded.",
                mismatched, count, sample_dim, self.config.get_dimension()
            );
        }

        *self.entries.write().await = report.entries;
        info!("Loaded {} entries from persistence backend", count);
        Ok(count)
    }

    /// Durable-store health recorded by the last [`load_from_db`](Self::load_from_db).
    /// `degraded` is true if any row was unreadable (corrupt).
    pub async fn store_health(&self) -> MemoryStoreHealth {
        *self.store_health.read().await
    }

    /// Check if GPU acceleration is available.
    #[must_use]
    pub const fn has_gpu(&self) -> bool {
        self.gpu.is_some() && self.gpu_pipeline.is_some()
    }

    /// Add a memory entry.
    ///
    /// An EMPTY active embedding is legal: it is the queued-for-backfill state
    /// ([`MemoryEntry::embedding`] documents it) that a write racing a provider
    /// switch lands in — the entry is kept, unsearchable until backfill embeds
    /// it under the current binding. Only a non-empty vector claims to be
    /// searchable, so only that is held to the bound width.
    ///
    /// # Errors
    ///
    /// Returns `MemoryError::DimensionMismatch` if a non-empty embedding has
    /// the wrong dimension.
    pub async fn add(&self, mut entry: MemoryEntry) -> Result<(), MemoryError> {
        if !entry.embedding.is_empty() && entry.embedding.len() != self.config.get_dimension() {
            return Err(MemoryError::DimensionMismatch {
                expected: self.config.get_dimension(),
                got: entry.embedding.len(),
            });
        }

        // Normalize the embedding for cosine similarity
        normalize_f32(&mut entry.embedding);

        // Write-through to persistence backend before updating in-memory cache
        if let Some(ref db) = self.db
            && let Err(e) = db.save_entry(&entry).await {
                warn!("Failed to persist memory entry {}: {}", entry.id, e);
                // Non-fatal: continue with in-memory add
            }

        // Chunks follow the content, on EVERY path that writes content — not
        // just `remember`. The consolidation and dream paths mutate entries
        // through these primitives directly, and a chunk set that only tracked
        // the ingest API would go stale exactly where memories change most.
        self.write_chunks(
            &entry.id,
            &entry.content,
            entry.workspace_id.as_deref(),
            entry
                .embedding_model
                .as_deref()
                .filter(|_| !entry.embedding.is_empty())
                .map(|m| (m, entry.embedding.as_slice())),
        )
        .await;

        self.entries.write().await.push(entry);
        Ok(())
    }

    /// Search for similar memories using SIMD-accelerated cosine similarity.
    ///
    /// GPU dispatch is available but only engaged for very large vector counts
    /// (>50,000) due to buffer upload/readback overhead dominating compute savings.
    ///
    /// ## Benchmark findings (RTX 4070 Ti SUPER + Zen 4 AVX-512)
    ///
    /// GPU fixed overhead: ~750us per dispatch (buffer alloc + upload + shader + readback).
    /// SIMD single cosine similarity (768-dim): ~0.1us (AVX-512).
    /// GPU never beat SIMD up to 10,000 vectors at any dimension tested (768/1536/3072).
    /// At 10,000 vectors: SIMD 1.5ms vs GPU 5.2ms (GPU still 3.5x slower).
    ///
    /// The crossover point with the current per-search buffer upload model is estimated
    /// at ~50,000+ vectors. To make GPU worthwhile at lower counts, we would need
    /// GPU-resident persistent buffers (upload once, search many times).
    ///
    /// See: `cargo bench -p nanna-gpu --bench gpu_vs_simd` for full results.
    /// Cosine against a stored embedding, tolerating one of the wrong width.
    ///
    /// `cosine_similarity_f32` opens with `assert_eq!(a.len(), b.len())` — a
    /// hard assert, and this crate's release profile sets `panic = "abort"`, so
    /// a single stale row takes the whole daemon down. The search guard above
    /// only validates the QUERY against config; it says nothing about what is
    /// in the store, and the two genuinely can differ: changing embedding
    /// provider changes the width, while every row already persisted keeps the
    /// old one.
    ///
    /// A stale row is not a non-match, it is a row that cannot be compared yet.
    /// Scoring it at the cosine floor keeps it out of every result without
    /// pretending it was evaluated — and, unlike the assert, leaves the process
    /// alive to re-embed it.
    fn cosine_or_stale(query: &[f32], embedding: &[f32]) -> f32 {
        if query.len() == embedding.len() {
            cosine_similarity_f32(query, embedding)
        } else {
            -1.0
        }
    }

    /// Order two `(index, similarity)` candidates: higher similarity first,
    /// and — when two score the same — lower index first.
    ///
    /// The index tiebreak is what makes this a **total** order, and that is
    /// load-bearing rather than tidy. The ranking below used to be a stable
    /// `sort_by` on similarity alone, which resolved ties by original position
    /// implicitly; selection is unstable, so without an explicit tiebreak two
    /// equally-similar memories could swap places between runs. Equal
    /// similarities are ordinary here, not exotic: every stale-width row scores
    /// exactly the floor, and duplicate embeddings score exactly alike.
    ///
    /// NaN is mapped to below the floor instead of being handed to
    /// `partial_cmp`, which reports it as `None`. A NaN score is not a low
    /// score, it is an *incomparable* one — it makes the comparator
    /// intransitive, so it could previously sort anywhere, including into the
    /// returned results. It is reachable: a zero-magnitude embedding of the
    /// right width (a row queued for backfill after an embedder rebind) gives
    /// cosine `0/0`, which the width check in [`cosine_or_stale`] cannot catch.
    /// Sorting it last is the same verdict a stale row gets, and it is
    /// deterministic.
    fn rank_order(a: &(usize, f32), b: &(usize, f32)) -> std::cmp::Ordering {
        let sa = if a.1.is_finite() { a.1 } else { f32::NEG_INFINITY };
        let sb = if b.1.is_finite() { b.1 } else { f32::NEG_INFINITY };
        // Not `is_finite`: the sanitized value is deliberately -inf, which is
        // ordered but not finite. What the total order needs is that neither
        // side is NaN, which is exactly what the two lines above guarantee.
        debug_assert!(
            !sa.is_nan() && !sb.is_nan(),
            "NaN must have been mapped to -inf above"
        );
        sb.partial_cmp(&sa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    }

    /// Reduce `ranked` to its best `top_k` entries, in [`rank_order`].
    ///
    /// This used to `sort_by` all N and then `truncate` to `top_k` — ranking
    /// fifty thousand candidates to keep ten, O(N log N) where the answer needs
    /// only O(N). `select_nth_unstable_by` partitions in linear time and only
    /// the kept prefix is then sorted. At N = `50_000` with `top_k` = 10 that is
    /// roughly 50k comparisons instead of ~780k, and `remember_scoped` runs a
    /// search on **every** ingest, so it is on the write path as well as the
    /// read one.
    ///
    /// Results are identical to the old stable sort element for element
    /// wherever the scores are finite, which [`rank_order`] is what guarantees.
    fn rank_top_k(ranked: &mut Vec<(usize, f32)>, top_k: usize) {
        let before = ranked.len();

        if top_k == 0 {
            ranked.clear();
            return;
        }
        // When `top_k` reaches the end there is nothing to discard and the
        // partition pass would be pure overhead before the same sort.
        if top_k < ranked.len() {
            // After this, everything before index `top_k` orders at or ahead of
            // everything from it on — i.e. the prefix IS the top k, unordered.
            ranked.select_nth_unstable_by(top_k, Self::rank_order);
            ranked.truncate(top_k);
        }
        ranked.sort_by(Self::rank_order);

        debug_assert!(ranked.len() <= top_k, "kept more than top_k");
        debug_assert!(ranked.len() <= before, "ranking must not invent entries");
        debug_assert!(
            ranked
                .windows(2)
                .all(|w| Self::rank_order(&w[0], &w[1]) != std::cmp::Ordering::Greater),
            "ranked prefix must be ordered"
        );
    }

    pub async fn search(&self, query_embedding: &[f32], top_k: usize) -> Vec<(MemoryEntry, f32)> {
        self.search_with_coverage(query_embedding, top_k).await.0
    }

    /// [`search`](Self::search), plus what the scan could actually compare.
    ///
    /// Same ranking, same results — the extra return is measured inside the
    /// scan that produced them, so a caller can tell an empty answer that means
    /// "nothing matched" from one that means "nothing was searchable". See
    /// [`SearchCoverage`].
    pub async fn search_with_coverage(
        &self,
        query_embedding: &[f32],
        top_k: usize,
    ) -> (Vec<(MemoryEntry, f32)>, SearchCoverage) {
        // Benchmark-calibrated threshold: GPU only wins with persistent buffers or
        // very large vector counts. With per-search buffer upload, the ~750us fixed
        // overhead means GPU needs >50k vectors to amortize the cost vs AVX-512.
        //
        // Previous threshold was 1000 — benchmarks showed GPU was 23x SLOWER there.
        // See: docs/benchmarks/gpu-vs-simd-analysis.md
        const GPU_THRESHOLD: usize = 50_000;

        if query_embedding.len() != self.config.get_dimension() {
            // Nothing was scanned. The store is not empty of matches, it is
            // unreachable at this width — say which.
            let total = self.entries.read().await.len();
            return (
                Vec::new(),
                SearchCoverage {
                    total,
                    comparable: 0,
                    query_width_matches: false,
                },
            );
        }

        // Normalize query
        let mut query = query_embedding.to_vec();
        normalize_f32(&mut query);

        let entries = self.entries.read().await;
        let entry_count = entries.len();

        if entry_count == 0 {
            return (
                Vec::new(),
                SearchCoverage {
                    total: 0,
                    comparable: 0,
                    query_width_matches: true,
                },
            );
        }

        // Exact at answer time: the same width predicate `cosine_or_stale`
        // applies row by row, counted once here rather than inferred from a
        // rebind-time snapshot that the backfill invalidates as it drains.
        let comparable = entries
            .iter()
            .filter(|e| e.embedding.len() == query.len())
            .count();
        let coverage = SearchCoverage {
            total: entry_count,
            comparable,
            query_width_matches: true,
        };

        // The GPU path flattens every embedding into one width-uniform buffer,
        // so a single stale or queued-for-backfill row (empty vector after a
        // rebind, old width after a provider switch) would misalign the whole
        // batch. Those rows can exist by design; when any is present, take the
        // SIMD path, which scores them at the stale floor row-by-row.
        let similarities: Vec<f32> = if entry_count >= GPU_THRESHOLD
            && let (Some(gpu), Some(pipeline)) = (self.gpu.as_ref(), self.gpu_pipeline.as_ref())
            && entries.iter().all(|e| e.embedding.len() == query.len())
        {
            // GPU path: batch all vectors together
            debug!("Using GPU for {} vectors (above {} threshold)", entry_count, GPU_THRESHOLD);

            // Flatten all embeddings into a single buffer
            let vectors: Vec<f32> = entries
                .iter()
                .flat_map(|e| e.embedding.iter().copied())
                .collect();

            match pipeline.search(gpu, &query, &vectors).await {
                Ok(sims) => sims,
                Err(e) => {
                    warn!("GPU search failed, falling back to SIMD: {}", e);
                    // Fallback to SIMD
                    entries
                        .iter()
                        .map(|entry| Self::cosine_or_stale(&query, &entry.embedding))
                        .collect()
                }
            }
        } else {
            // SIMD path — fast for all practical memory store sizes.
            // `cosine_or_stale`, not the raw cosine: the raw kernel asserts
            // equal widths and this profile aborts on panic, so one stale or
            // queued row would take the daemon down mid-search.
            debug!("Using SIMD ({}) for {} vectors", nanna_simd::simd_tier(), entry_count);
            entries
                .iter()
                .map(|entry| Self::cosine_or_stale(&query, &entry.embedding))
                .collect()
        };

        // Rank by INDEX, then clone only the winners.
        //
        // This used to clone EVERY entry — embeddings included — into a Vec,
        // sort that, and then throw all but `top_k` away. At 2048 dimensions
        // each discarded clone is ~8 KB of memcpy, and `remember_scoped` runs a
        // search on every single ingest, so the cost was O(store) per write and
        // O(store²) across a run. Capturing every tool call raises the write
        // count roughly forty-fold, which would have turned that from wasteful
        // into a stall.
        //
        // `rank_top_k` is the second half of the same idea: it also stopped
        // *ordering* all N to keep `top_k`. Measured at 50k (`bench/BASELINE.md`
        // Suite 2c), the sort was the larger part of the gap between the raw
        // cosine kernel and this function.
        let mut ranked: Vec<(usize, f32)> = similarities.into_iter().enumerate().collect();
        Self::rank_top_k(&mut ranked, top_k);
        let scored: Vec<(MemoryEntry, f32)> = ranked
            .into_iter()
            .map(|(idx, sim)| (entries[idx].clone(), sim))
            .collect();
        drop(entries);
        (scored, coverage)
    }

    /// Search for similar memories with workspace scope filtering.
    ///
    /// Scope rules:
    /// - `workspace_id = Some(id)`: returns global + that workspace's memories
    /// - `workspace_id = None` (global): returns all memories
    pub async fn search_scoped(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        workspace_id: Option<&str>,
    ) -> Vec<(MemoryEntry, f32)> {
        self.search_scoped_with_coverage(query_embedding, top_k, workspace_id)
            .await
            .0
    }

    /// [`search_scoped`](Self::search_scoped), plus what the scan could compare.
    ///
    /// The coverage describes the SCAN, which is unscoped — a memory the
    /// workspace filter then drops was still searched. That is the number the
    /// caller needs: it separates "your query matched nothing" from "most of
    /// the store could not be consulted".
    pub async fn search_scoped_with_coverage(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        workspace_id: Option<&str>,
    ) -> (Vec<(MemoryEntry, f32)>, SearchCoverage) {
        self.search_in_scope_with_coverage(
            query_embedding,
            top_k,
            RecallScope::from_workspace(workspace_id),
        )
        .await
    }

    /// [`Self::search_scoped_with_coverage`] over any [`RecallScope`].
    pub async fn search_in_scope_with_coverage(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        scope: RecallScope<'_>,
    ) -> (Vec<(MemoryEntry, f32)>, SearchCoverage) {
        // Get more to filter
        let (all_results, coverage) = self.search_with_coverage(query_embedding, top_k * 3).await;

        let filtered: Vec<(MemoryEntry, f32)> = all_results
            .into_iter()
            .filter(|(entry, _)| scope.admits(entry.workspace_id.as_deref()))
            .take(top_k)
            .collect();

        debug_assert!(
            filtered.len() <= top_k,
            "the scoped answer is bounded by top_k"
        );
        (filtered, coverage)
    }

    /// Get entry by ID
    pub async fn get(&self, id: &str) -> Option<MemoryEntry> {
        let entries = self.entries.read().await;
        entries.iter().find(|e| e.id == id).cloned()
    }

    /// Remove entry by ID.
    ///
    /// Durable or refused: the persistence backend is written **first**, and a
    /// failure there leaves the entry in RAM and returns the error. The old
    /// order (RAM, then a logged-and-ignored backend error) reported a delete
    /// that a restart undid — the memory came back from disk.
    ///
    /// # Errors
    ///
    /// Returns `MemoryError::NotFound` if no entry with the given ID exists,
    /// and the backend's error if it could not delete the row.
    pub async fn remove(&self, id: &str) -> Result<(), MemoryError> {
        if !self.entries.read().await.iter().any(|e| e.id == id) {
            return Err(MemoryError::NotFound(id.to_string()));
        }
        if let Some(ref db) = self.db {
            db.remove_entry(id).await?;
        }
        let mut entries = self.entries.write().await;
        let before = entries.len();
        entries.retain(|e| e.id != id);
        debug_assert!(before - entries.len() <= 1, "memory ids are unique");
        drop(entries);
        Ok(())
    }

    /// Remove many entries, durably: an entry leaves RAM only once the backend
    /// has confirmed its row is gone.
    ///
    /// The fast path is the one batched [`MemoryPersistence::remove_entries`]
    /// call. That batch is not one transaction, so when it fails it may have
    /// stopped part-way; the fallback then deletes one id at a time (deleting
    /// an already-deleted row is a no-op) to learn exactly which rows are gone,
    /// and only those leave RAM. So the result can be trusted row by row:
    /// what it reports removed will not come back on a restart, and what it
    /// reports failed is still visible.
    pub async fn remove_many_durable(&self, ids: &[&str]) -> DurableRemoval {
        if ids.is_empty() {
            return DurableRemoval::default();
        }
        let mut failed = 0usize;
        let mut first_error = None;
        let confirmed: Vec<&str> = match self.db {
            None => ids.to_vec(),
            Some(ref db) => match db.remove_entries(ids).await {
                Ok(()) => ids.to_vec(),
                Err(batch_error) => {
                    warn!(
                        "Batch removal of {} memories failed ({batch_error}); retrying one by one",
                        ids.len()
                    );
                    let mut confirmed = Vec::with_capacity(ids.len());
                    for id in ids {
                        match db.remove_entry(id).await {
                            Ok(()) => confirmed.push(*id),
                            Err(e) => {
                                failed += 1;
                                first_error.get_or_insert_with(|| e.to_string());
                            }
                        }
                    }
                    confirmed
                }
            },
        };
        let gone: std::collections::HashSet<&str> = confirmed.iter().copied().collect();
        let mut entries = self.entries.write().await;
        let before = entries.len();
        entries.retain(|e| !gone.contains(e.id.as_str()));
        let removed = before - entries.len();
        drop(entries);
        debug_assert!(
            removed <= ids.len(),
            "removed more entries than ids requested"
        );
        debug_assert!(failed <= ids.len(), "more failures than ids");
        DurableRemoval {
            removed,
            failed,
            first_error,
        }
    }

    /// Remove many entries by ID in one batch.
    ///
    /// Best-effort like [`remove`](Self::remove): an id absent from the in-RAM
    /// cache is skipped rather than erroring, and a persistence failure is logged
    /// but non-fatal (the dream cycle that drives this must not abort a whole
    /// consolidation because one source removal failed). The batch takes the
    /// entries write-lock **once** and issues **one** persistence
    /// [`remove_entries`](MemoryPersistence::remove_entries) call, so a backing
    /// store that batches (Turso) checkpoints once for the set instead of per row.
    ///
    /// Returns the number of entries actually removed from the in-RAM cache.
    pub async fn remove_many(&self, ids: &[&str]) -> usize {
        if ids.is_empty() {
            return 0;
        }
        let requested = ids.len();
        let id_set: std::collections::HashSet<&str> = ids.iter().copied().collect();

        let mut entries = self.entries.write().await;
        let before = entries.len();
        entries.retain(|e| !id_set.contains(e.id.as_str()));
        let removed = before - entries.len();
        drop(entries);

        debug_assert!(
            removed <= requested,
            "removed more entries than ids requested"
        );

        // Write-through: one batched persistence call for the whole set.
        if let Some(ref db) = self.db
            && let Err(e) = db.remove_entries(ids).await {
                warn!(
                    "Failed to batch-remove {} memory entries from persistence: {}",
                    requested, e
                );
                // Non-fatal
            }

        removed
    }

    /// Update FSRS state for an entry.
    ///
    /// # Errors
    ///
    /// Returns `MemoryError::NotFound` if no entry with the given ID exists.
    pub async fn update_fsrs<F>(&self, id: &str, f: F) -> Result<(), MemoryError>
    where
        F: FnOnce(&mut FsrsState),
    {
        let mut entries = self.entries.write().await;
        let entry = entries
            .iter_mut()
            .find(|e| e.id == id)
            .ok_or_else(|| MemoryError::NotFound(id.to_string()))?;
        f(&mut entry.fsrs);
        let new_fsrs = entry.fsrs.clone();
        drop(entries);

        // Write-through to persistence backend
        if let Some(ref db) = self.db
            && let Err(e) = db.update_entry_fsrs(id, &new_fsrs).await {
                warn!("Failed to persist FSRS update for {}: {}", id, e);
                // Non-fatal
            }

        Ok(())
    }

    /// Update content for an entry (used during expansion).
    ///
    /// # Errors
    ///
    /// Returns `MemoryError::NotFound` if no entry with the given ID exists.
    pub async fn update_content(&self, id: &str, content: &str) -> Result<(), MemoryError> {
        let mut entries = self.entries.write().await;
        let entry = entries
            .iter_mut()
            .find(|e| e.id == id)
            .ok_or_else(|| MemoryError::NotFound(id.to_string()))?;
        entry.content = content.to_string();
        // Every vector this entry holds describes the text that just stopped
        // existing. Clearing them is what `update_content_and_embedding` does
        // for the same reason — and this path skipping it was found live: the
        // stale buckets survived in RAM and on disk, a later rebind reactivated
        // them, and the backfill then skipped the entry because a bucket
        // existed for the active model. The memory stayed findable by its old
        // words, permanently. Cleared, the entry is the ordinary
        // queued-for-backfill state and the drain re-embeds the new text.
        entry.embedding = Vec::new();
        entry.embedding_model = None;
        entry.embeddings.clear();
        let workspace_id = entry.workspace_id.clone();
        drop(entries);

        // Write-through to persistence backend
        if let Some(ref db) = self.db {
            // Durable buckets first, for the same reason as the RAM map above.
            if let Err(e) = db.clear_memory_buckets(id).await {
                warn!("Could not clear stale buckets for {id}: {e}");
            }
            if let Err(e) = db.update_entry_content(id, content).await {
                warn!("Failed to persist content update for {}: {}", id, e);
                // Non-fatal
            }
        }

        // New text, new boundaries. The parent's vector is NOT reused as a
        // single-chunk seed here: this path changes content without touching
        // the embedding, so that vector still describes the OLD text and
        // seeding it would assert a chunk had been embedded when it had not.
        self.write_chunks(id, content, workspace_id.as_deref(), None).await;

        Ok(())
    }

    /// Replace an entry's content **and** embedding in place (merge/dreaming path).
    ///
    /// Keeps the entry's ID and FSRS history; the caller reinforces FSRS
    /// separately. Persists the full updated entry (upsert) so the durable
    /// content and embedding never diverge from the in-memory cache.
    ///
    /// `model` names the binding that produced `embedding`. Rewriting content
    /// invalidates EVERY previously computed vector for this entry, so the
    /// bucket map is reset to just the incoming vector — leaving other models'
    /// buckets in place would let a later rebind resurrect a vector for text
    /// that no longer exists.
    ///
    /// An empty `embedding` is the queued-for-backfill state (same convention
    /// as [`Self::add`]): the content lands, the vectors are cleared, and the
    /// entry is unsearchable until backfill re-embeds it under the current
    /// binding.
    ///
    /// # Errors
    ///
    /// Returns `DimensionMismatch` if a non-empty `embedding` has the wrong
    /// dimension, or `NotFound` if no entry has the given `id`.
    pub async fn update_content_and_embedding(
        &self,
        id: &str,
        content: &str,
        embedding: Vec<f32>,
        model: Option<&str>,
    ) -> Result<(), MemoryError> {
        self.update_content_and_embedding_if(id, None, content, embedding, model)
            .await
            .map(|applied| {
                debug_assert!(applied, "an unguarded rewrite always applies");
            })
    }

    /// [`Self::update_content_and_embedding`] guarded by a compare-and-swap on
    /// the entry's CURRENT content. Returns `Ok(false)` — touching nothing —
    /// when `expected_content` is `Some` and the entry's live content differs.
    ///
    /// This is the primitive that makes a dream-cycle rewrite safe against a
    /// concurrent live write. The dream paths (dedup fold, expansion) decide
    /// what to write from a SNAPSHOT taken seconds earlier — long enough for a
    /// `remember` on the same memory to have folded new content in — and an
    /// unguarded rewrite would replace that fresh content with a merge of the
    /// stale snapshot: a silent lost update. The check runs INSIDE the entries
    /// write lock, the same critical section as the mutation, so there is no
    /// window between verify and write. On `false` the caller skips its fold
    /// (the pair is simply re-examined by the next cycle — lossless), never
    /// retries blindly.
    ///
    /// # Errors
    ///
    /// Same as [`Self::update_content_and_embedding`].
    pub async fn update_content_and_embedding_if(
        &self,
        id: &str,
        expected_content: Option<&str>,
        content: &str,
        mut embedding: Vec<f32>,
        model: Option<&str>,
    ) -> Result<bool, MemoryError> {
        debug_assert!(!id.is_empty(), "memory id must not be empty");
        debug_assert!(!content.is_empty(), "merged content must not be empty");
        if !embedding.is_empty() && embedding.len() != self.config.get_dimension() {
            return Err(MemoryError::DimensionMismatch {
                expected: self.config.get_dimension(),
                got: embedding.len(),
            });
        }

        // Normalize for cosine similarity, matching `add`.
        normalize_f32(&mut embedding);

        // Update the in-memory entry, snapshot it, then release the lock.
        let mut entries = self.entries.write().await;
        let entry = entries
            .iter_mut()
            .find(|e| e.id == id)
            .ok_or_else(|| MemoryError::NotFound(id.to_string()))?;
        if let Some(expected) = expected_content
            && entry.content != expected {
                return Ok(false);
            }
        entry.content = content.to_string();
        entry.embeddings.clear();
        if embedding.is_empty() {
            entry.embedding_model = None;
        } else {
            entry.embedding_model = model.map(str::to_string);
            if let Some(model) = model {
                entry.embeddings.insert(model.to_string(), embedding.clone());
            }
        }
        entry.embedding = embedding;
        let snapshot = entry.clone();
        drop(entries);

        // Write-through the whole entry (content + embedding stay consistent).
        //
        // Buckets are cleared FIRST. This path rewrote the content, so every
        // previously stored vector describes text that no longer exists —
        // `entry.embeddings` was cleared above for exactly that reason, and
        // leaving the durable copies behind would let a restart resurrect what
        // the rewrite discarded. `save_entry` then writes back only the vector
        // that actually matches the new content.
        if let Some(ref db) = self.db {
            if let Err(e) = db.clear_memory_buckets(id).await {
                warn!("Could not clear stale buckets for {id}: {e}");
            }
            if let Err(e) = db.save_entry(&snapshot).await {
                warn!("Failed to persist merged memory {}: {}", id, e);
                // Non-fatal: in-memory cache already updated.
            }
        }

        self.write_chunks(
            id,
            content,
            snapshot.workspace_id.as_deref(),
            snapshot
                .embedding_model
                .as_deref()
                .filter(|_| !snapshot.embedding.is_empty())
                .map(|m| (m, snapshot.embedding.as_slice())),
        )
        .await;

        debug_assert_eq!(snapshot.id, id, "merged entry id must be unchanged");
        Ok(true)
    }

    /// Remove entries whose content STILL MATCHES the snapshot the caller
    /// folded from, in one batch. `pairs` is `(id, expected_content)`.
    ///
    /// The destructive half of a dream fold: the survivor was rewritten to
    /// contain each source's text, and then the sources are removed. But
    /// removal decided from a snapshot is only safe while the source is
    /// unchanged — a live `remember` may have folded NEW content into a
    /// source between the snapshot and this batch, and unconditionally
    /// removing it would delete text that exists nowhere else. A changed
    /// source is simply kept (skipped), which leaves a transient
    /// near-duplicate for the next cycle to re-fold — lossless, like every
    /// other fold fallback.
    ///
    /// Returns the number of entries actually removed. Persistence is one
    /// batched call for the removed set, matching [`Self::remove_many`].
    pub async fn remove_many_matching(&self, pairs: &[(&str, &str)]) -> usize {
        if pairs.is_empty() {
            return 0;
        }
        let expected: std::collections::HashMap<&str, &str> = pairs.iter().copied().collect();

        // Decide and remove inside ONE write-lock critical section, so no
        // write can land between the content check and the removal.
        let mut entries = self.entries.write().await;
        let before = entries.len();
        let mut removed_ids: Vec<String> = Vec::with_capacity(pairs.len());
        entries.retain(|e| {
            let matches = expected
                .get(e.id.as_str())
                .is_some_and(|want| e.content == **want);
            if matches {
                removed_ids.push(e.id.clone());
            }
            !matches
        });
        let removed = before - entries.len();
        drop(entries);

        debug_assert_eq!(removed, removed_ids.len(), "one id per removed entry");
        debug_assert!(removed <= pairs.len(), "cannot remove more than requested");

        if removed < pairs.len() {
            debug!(
                requested = pairs.len(),
                removed,
                "matched removal skipped concurrently-rewritten entries; they \
                 stay live for the next cycle"
            );
        }

        // Write-through: one batched persistence call for the whole set.
        if let Some(ref db) = self.db
            && !removed_ids.is_empty() {
                let id_refs: Vec<&str> = removed_ids.iter().map(String::as_str).collect();
                if let Err(e) = db.remove_entries(&id_refs).await {
                    warn!(
                        "Failed to batch-remove {} memory entries from persistence: {}",
                        id_refs.len(),
                        e
                    );
                    // Non-fatal
                }
            }

        removed
    }

    /// Get all entries (for consolidation)
    pub async fn all_entries(&self) -> Vec<MemoryEntry> {
        self.entries.read().await.clone()
    }

    /// Project every entry through `project` under one read lock, without
    /// cloning the entries themselves.
    ///
    /// [`Self::all_entries`] clones every vector in every model bucket — for
    /// an export that wants content and FSRS state, that is the dominant cost
    /// for nothing (a 1536-dim bucket is 6 KiB per memory per model). One
    /// output per entry, so the result is bounded by the store.
    pub async fn map_entries<T>(&self, project: impl Fn(&MemoryEntry) -> T) -> Vec<T> {
        let entries = self.entries.read().await;
        let projected: Vec<T> = entries.iter().map(project).collect();
        debug_assert_eq!(projected.len(), entries.len(), "one output per entry");
        projected
    }

    /// Get total number of entries
    pub async fn len(&self) -> usize {
        self.entries.read().await.len()
    }

    /// Check if store is empty
    pub async fn is_empty(&self) -> bool {
        self.entries.read().await.is_empty()
    }

    /// Clear all entries
    pub async fn clear(&self) {
        self.entries.write().await.clear();
    }

    /// Save to file.
    ///
    /// # Deprecated
    ///
    /// This method is retained only for one-time JSON→Turso migration.
    /// Use [`VectorStore::with_persistence`] and [`VectorStore::load_from_db`] instead.
    ///
    /// # Errors
    /// Returns `MemoryError` if serialization or the file write/rename fails.
    pub async fn save(&self, path: &std::path::Path) -> Result<(), MemoryError> {
        warn!("VectorStore::save() is deprecated. Use Turso persistence instead.");
        // Serialize under the read lock, then release it before the file IO so the
        // lock is never held across an `.await`.
        let (json, count) = {
            let entries = self.entries.read().await;
            (serde_json::to_string_pretty(&*entries)?, entries.len())
        };
        // Atomic write: write to a sibling temp file, then rename over the target.
        // `fs::write` truncates in place, so a crash mid-write would leave a
        // corrupt/empty store; rename-into-place is atomic on the same filesystem.
        let tmp_path = path.with_extension("json.tmp");
        tokio::fs::write(&tmp_path, json).await?;
        tokio::fs::rename(&tmp_path, path).await?;
        info!("Saved {} entries to {:?}", count, path);
        Ok(())
    }

    /// Load from file.
    ///
    /// # Deprecated
    ///
    /// This method is retained only for one-time JSON→Turso migration.
    /// Use [`VectorStore::with_persistence`] and [`VectorStore::load_from_db`] instead.
    ///
    /// Loads all entries regardless of embedding dimension. If the embedding
    /// model has changed, call [`MemoryService::probe_and_align_dimension`]
    /// after loading to re-embed mismatched entries.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::Io`] when the file cannot be read and
    /// [`MemoryError::Serialization`] when it is not a JSON array of memory
    /// entries. The in-memory store is left untouched on either error.
    pub async fn load(&self, path: &std::path::Path) -> Result<(), MemoryError> {
        let json = tokio::fs::read_to_string(path).await?;
        let loaded: Vec<MemoryEntry> = serde_json::from_str(&json)?;
        
        info!("Parsing {} entries from {:?}, expecting {} dimensions", 
              loaded.len(), path, self.config.get_dimension());

        let mismatched = loaded.iter()
            .filter(|e| e.embedding.len() != self.config.get_dimension())
            .count();

        if mismatched > 0 {
            let sample_dim = loaded.iter()
                .find(|e| e.embedding.len() != self.config.get_dimension())
                .map_or(0, |e| e.embedding.len());
            warn!(
                "Dimension mismatch: {} of {} entries have {} dims (expected {}). \
                 They will be re-embedded after dimension probe.",
                mismatched, loaded.len(), sample_dim, self.config.get_dimension()
            );
        }

        let count = loaded.len();
        *self.entries.write().await = loaded;
        info!("Loaded {} entries from {:?}", count, path);
        Ok(())
    }

    /// Flush all in-memory entries to the persistence backend.
    ///
    /// Used during one-time JSON → Turso migration: after `load()` populates
    /// the in-memory cache from JSON, call this to persist every entry to Turso.
    ///
    /// # Errors
    ///
    /// Returns `MemoryError::Persistence` if the backing store is not set or a write fails.
    pub async fn flush_to_db(&self) -> Result<usize, MemoryError> {
        let db = self.db.as_ref().ok_or_else(|| {
            MemoryError::Persistence("No persistence backend attached".to_string())
        })?;

        let entries = self.entries.read().await;
        let total = entries.len();
        let mut saved = 0usize;

        for entry in entries.iter() {
            if let Err(e) = db.save_entry(entry).await {
                warn!("Failed to flush entry {} to DB: {}", entry.id, e);
            } else {
                saved += 1;
            }
        }
        drop(entries);

        info!("Flushed {}/{} entries to Turso", saved, total);
        Ok(saved)
    }

    /// Re-embed all entries whose dimension doesn't match the expected dimension.
    ///
    /// Returns the number of entries re-embedded. Entries that fail to re-embed
    /// are removed (content was likely empty or the embed function errored).
    /// Point every entry at its bucket for `model`, keeping all other buckets.
    ///
    /// This is what a provider switch costs now: a hash lookup per entry, no
    /// network, no LLM. An entry that has been embedded by this model before —
    /// because the router used it earlier and flapped away — comes back
    /// instantly. An entry that has not is left with an empty active vector,
    /// which search skips and [`Self::entries_missing_model`] reports for
    /// backfill.
    ///
    /// Returns `(rebound, missing)`.
    pub async fn rebind_to_model(&self, model: &str) -> (usize, usize) {
        let mut entries = self.entries.write().await;
        let (mut rebound, mut missing) = (0usize, 0usize);
        for entry in entries.iter_mut() {
            // Retain whatever the active vector currently is, so switching AWAY
            // from a model never discards the work that produced it.
            //
            // Rows written before buckets existed carry a vector but no model
            // name. Dropping those on the first rebind would throw away the
            // entire store's existing embeddings — so they are kept under a
            // width-tagged legacy key. That key can never be *matched* by a real
            // provider (we do not know which model made it, and equal width is
            // not equal vector space), but keeping it costs almost nothing and
            // discarding it is irreversible.
            if !entry.embedding.is_empty() {
                let key = entry
                    .embedding_model
                    .clone()
                    .unwrap_or_else(|| format!("legacy-{}d", entry.embedding.len()));
                entry.embeddings.entry(key).or_insert_with(|| entry.embedding.clone());
            }
            if let Some(vector) = entry.embeddings.get(model) {
                entry.embedding = vector.clone();
                entry.embedding_model = Some(model.to_string());
                rebound += 1;
            } else {
                entry.embedding.clear();
                entry.embedding_model = None;
                missing += 1;
            }
        }
        drop(entries);
        (rebound, missing)
    }

    /// Ids and content of entries with no bucket for `model`, oldest first.
    ///
    /// `limit` bounds one backfill pass so the filler yields between batches
    /// instead of monopolising a rate-limited provider.
    pub async fn entries_missing_model(&self, model: &str, limit: usize) -> Vec<(String, String)> {
        let entries = self.entries.read().await;
        entries
            .iter()
            .filter(|e| !e.embeddings.contains_key(model))
            .take(limit)
            .map(|e| (e.id.clone(), e.content.clone()))
            .collect()
    }

    /// Record `embedding` as `model`'s vector for `id`, activating it when
    /// `model` is the one currently bound.
    ///
    /// # Errors
    ///
    /// Returns `DimensionMismatch` when `activate` is set and the vector does
    /// not match the bound width — an activated vector goes straight into the
    /// search path, so this is the last gate against a backfill racing a
    /// provider switch.
    pub async fn set_embedding_for_model(
        &self,
        id: &str,
        model: &str,
        embedding: Vec<f32>,
        activate: bool,
    ) -> Result<(), MemoryError> {
        if activate && embedding.len() != self.config.get_dimension() {
            return Err(MemoryError::DimensionMismatch {
                expected: self.config.get_dimension(),
                got: embedding.len(),
            });
        }
        let mut entries = self.entries.write().await;
        let Some(entry) = entries.iter_mut().find(|e| e.id == id) else {
            return Err(MemoryError::NotFound(id.to_string()));
        };
        // A memory's FIRST vector ever is the moment it becomes findable by
        // similarity search. A re-embed after a provider switch keeps its old
        // buckets, so it is not a wait — counting it would read a days-old
        // memory as a days-long queue.
        let first_vector = entry.embeddings.is_empty() && entry.embedding.is_empty();
        entry.embeddings.insert(model.to_string(), embedding.clone());
        if activate {
            entry.embedding = embedding;
            entry.embedding_model = Some(model.to_string());
        }
        let snapshot = entry.clone();
        drop(entries);
        if activate && first_vector {
            // Seconds, because `timestamp` is; a clock that went backwards
            // reads as no wait rather than a negative one.
            let waited =
                u64::try_from(chrono_timestamp().saturating_sub(snapshot.timestamp)).unwrap_or(0);
            push_bounded(
                &mut self
                    .searchable_latency
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner),
                waited,
            );
        }
        // Non-fatal, like every other write-through here: the in-memory cache is
        // already correct, and a backfill that fails to persist just repeats on
        // the next restart rather than losing the memory.
        if let Some(ref db) = self.db
            && let Err(e) = db.save_entry(&snapshot).await
        {
            warn!("Failed to persist backfilled embedding for {}: {}", snapshot.id, e);
        }
        Ok(())
    }

    /// Get the current configured dimension
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.config.get_dimension()
    }

    /// Update the expected embedding dimension at runtime.
    ///
    /// Called by [`MemoryService::probe_and_align_dimension`] when the
    /// embedding model changes and returns a different dimension.
    /// After this call, `add()` accepts entries with the new dimension.
    pub fn set_dimension(&self, new_dim: usize) {
        self.config.set_dimension(new_dim);
    }

    /// The bound embedding model, if the daemon has named one.
    pub async fn active_model(&self) -> Option<String> {
        self.active_model.read().await.clone()
    }

    /// Bind `model`. Only called under `MemoryService`'s binding write-lock,
    /// beside `set_dimension` and `set_chunk_params`.
    pub async fn set_active_model(&self, model: &str) {
        *self.active_model.write().await = Some(model.to_string());
    }

    /// Chunk geometry for the currently bound embedding model.
    #[must_use]
    pub fn chunk_params(&self) -> crate::chunking::ChunkParams {
        crate::chunking::ChunkParams {
            max_chars: self.config.get_chunk_max_chars(),
            version: crate::chunking::CHUNKER_VERSION,
        }
    }

    /// Move the chunk budget. Only called under `MemoryService`'s binding
    /// write-lock, beside `set_dimension`.
    pub fn set_chunk_params(&self, params: crate::chunking::ChunkParams) {
        self.config.set_chunk_max_chars(params.max_chars);
    }

    /// Chunk-level nearest neighbours for `query`, collapsed to one entry per
    /// parent memory.
    ///
    /// Over-fetches deliberately: `limit` is the number of MEMORIES wanted, but
    /// several of the top chunks routinely belong to the same memory, so asking
    /// the index for exactly `limit` chunks would return fewer memories than
    /// asked for whenever a memory matched more than once — and would do it
    /// most often for exactly the memories that matched best.
    pub async fn search_chunks(
        &self,
        query: &[f32],
        model: &str,
        limit: usize,
        workspace_id: Option<&str>,
        min_score: f32,
    ) -> HashMap<String, crate::chunk_rank::ChunkHit> {
        /// Chunks fetched per memory wanted. Four is the point past which the
        /// over-fetch stops changing which memories come back in practice,
        /// while keeping the SQL scan bounded.
        const CHUNK_OVERFETCH: usize = 4;

        let Some(ref db) = self.db else { return HashMap::new() };
        // No bound model means no way to say which vector space the query
        // lives in, and chunk vectors carry the model that made them. Refusing
        // to search is the only honest answer; guessing would compare across
        // spaces.
        if query.is_empty() || model.is_empty() || limit == 0 {
            return HashMap::new();
        }
        let hits = db
            .search_chunks(query, model, limit.saturating_mul(CHUNK_OVERFETCH), workspace_id)
            .await
            .unwrap_or_else(|e| {
                // Degrading to row-only search is correct; doing it silently is
                // not — recall would just quietly get coarser.
                warn!("Chunk search failed ({e}); this recall uses row vectors only");
                Vec::new()
            });
        crate::chunk_rank::collapse_chunk_hits(&hits, min_score)
    }

    /// Chunks awaiting a vector under `model`, as `(chunk_id, content)`.
    pub async fn chunks_needing_embedding(
        &self,
        model: &str,
        limit: usize,
    ) -> Vec<PendingChunk> {
        // First call for a model this process pays the pre-existing-work
        // sweep; after that the queue alone is consulted. Marked seeded only
        // on success, so a failed sweep is retried rather than skipped forever.
        // A poisoned lock still holds a valid set: its critical sections are a
        // single `contains`/`insert`, which leave it consistent even if they
        // panicked.
        let seed = !self
            .seeded_models
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains(model);
        match self.db {
            Some(ref db) => match db.chunks_needing_embedding(model, limit, seed).await {
                Ok(work) => {
                    if seed {
                        self.seeded_models
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .insert(model.to_string());
                    }
                    work
                }
                Err(e) => {
                    // An error absorbed to an empty batch reads as "queue
                    // complete" and stops the drain — with pending work still
                    // in the table and nothing anywhere saying so.
                    warn!("Could not read the embedding queue ({e}); the drain will retry next pass");
                    Vec::new()
                }
            },
            None => Vec::new(),
        }
    }

    /// Attach a backfilled vector to a chunk.
    ///
    /// # Errors
    ///
    /// Returns `MemoryError::Persistence` if the write fails.
    pub async fn set_chunk_embedding(
        &self,
        chunk_id: i64,
        embedding: &[f32],
        model: &str,
    ) -> Result<(), MemoryError> {
        match self.db {
            Some(ref db) => db.set_chunk_embedding(chunk_id, embedding, model).await,
            None => Ok(()),
        }
    }

    /// Re-activate a model's durable chunk vectors after a rebind.
    pub async fn restore_chunk_vectors(&self, model: &str) -> usize {
        match self.db {
            Some(ref db) => db.restore_chunk_vectors(model).await.unwrap_or_else(|e| {
                warn!("Could not restore '{model}' chunk vectors ({e}); they will be re-embedded by the drain instead");
                0
            }),
            None => 0,
        }
    }

    /// Mark queued work complete.
    ///
    /// Best-effort: a lost dequeue costs one redundant re-embed on the next
    /// drain, which is strictly better than a lost vector.
    pub async fn dequeue_embedding(&self, memory_id: &str, ordinal: i64, model: &str) {
        if let Some(ref db) = self.db
            && let Err(e) = db.dequeue_embedding(memory_id, ordinal, model).await {
                debug!("Could not dequeue {memory_id}#{ordinal}: {e}");
            }
    }

    /// Note a failed embedding attempt against the durable queue.
    ///
    /// Best-effort: losing the note must not abort the drain, but the note is
    /// the difference between a stuck queue and a silent one.
    pub async fn record_embedding_failure(
        &self,
        memory_id: &str,
        ordinal: i64,
        model: &str,
        error: &str,
    ) {
        if let Some(ref db) = self.db
            && let Err(e) = db
                .record_embedding_failure(memory_id, ordinal, model, error)
                .await
            {
                debug!("Could not record embedding failure for {memory_id}#{ordinal}: {e}");
            }
    }

    /// Queue depth per model with the most recent error for each.
    pub async fn embedding_queue_health(&self) -> Vec<(String, usize, Option<String>)> {
        match self.db {
            Some(ref db) => db.embedding_queue_health().await.unwrap_or_else(|e| {
                warn!("Could not read embedding queue health: {e}");
                Vec::new()
            }),
            None => Vec::new(),
        }
    }

    /// Memories with no chunk rows yet — written before chunking existed.
    pub async fn parents_without_chunks(&self, limit: usize) -> Vec<String> {
        match self.db {
            Some(ref db) => db.parents_without_chunks(limit).await.unwrap_or_else(|e| {
                warn!("Could not list unchunked memories ({e}); the chunk backfill will retry next pass");
                Vec::new()
            }),
            None => Vec::new(),
        }
    }

    /// Re-split and store the chunk set for the in-RAM entry `id`.
    ///
    /// Used by the backfill to chunk memories that predate chunking. Reads the
    /// entry from the cache rather than the database because the cache is the
    /// authority on current content — a memory updated in this session has its
    /// new text here before the write-through completes.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::NotFound`] when no in-memory entry has `id`.
    /// Failing to write the chunks is logged, not returned.
    pub async fn rechunk(&self, id: &str) -> Result<(), MemoryError> {
        let snapshot = {
            let entries = self.entries.read().await;
            entries
                .iter()
                .find(|e| e.id == id)
                .cloned()
                .ok_or_else(|| MemoryError::NotFound(id.to_string()))?
        };
        self.write_chunks(
            &snapshot.id,
            &snapshot.content,
            snapshot.workspace_id.as_deref(),
            snapshot
                .embedding_model
                .as_deref()
                .filter(|_| !snapshot.embedding.is_empty())
                .map(|m| (m, snapshot.embedding.as_slice())),
        )
        .await;
        Ok(())
    }

    /// Re-split `content` and replace the stored chunk set for `id`.
    ///
    /// Called after every mutation that changes a memory's text. Chunks are
    /// derived data, so a failure here is logged and swallowed exactly like the
    /// entry write-throughs around it: losing the retrieval index for one
    /// memory must not fail the write that produced it, and the backfill pass
    /// rebuilds what is missing.
    ///
    /// `parent_vector` is the entry's own embedding, if it has one under the
    /// active model. When the content fits in a single chunk that vector
    /// already describes the chunk exactly — same text, same model — so it is
    /// reused and the backfill has nothing to do. Multi-chunk memories get
    /// text-only rows and are embedded by the backfill pass; doing it inline
    /// would put one network round trip per chunk in front of every write.
    async fn write_chunks(
        &self,
        id: &str,
        content: &str,
        workspace_id: Option<&str>,
        parent_vector: Option<(&str, &[f32])>,
    ) {
        let Some(ref db) = self.db else { return };
        let params = self.chunk_params();
        let pieces = crate::chunking::chunk_text(content, &params);

        let single = pieces.len() == 1;
        let writes: Vec<ChunkWrite> = pieces
            .into_iter()
            .map(|c| {
                let (embedding, embedding_model) = match parent_vector {
                    Some((model, vector)) if single => {
                        (Some(vector.to_vec()), Some(model.to_string()))
                    }
                    _ => (None, None),
                };
                ChunkWrite {
                    ordinal: c.ordinal,
                    content: c.content,
                    char_start: c.char_start,
                    char_end: c.char_end,
                    embedding,
                    embedding_model,
                    chunk_max_chars: i64::try_from(params.max_chars).unwrap_or(i64::MAX),
                    chunker_version: params.version,
                }
            })
            .collect();

        let model = self.active_model().await;
        if let Err(e) = db
            .replace_chunks(id, workspace_id, model.as_deref(), &writes)
            .await
        {
            warn!("Failed to write chunks for memory {}: {} (retrieval for this memory falls back to its whole-row vector until the backfill rebuilds them)", id, e);
        }
    }
}

/// Conversation memory for maintaining chat context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationMemory {
    pub session_id: String,
    pub messages: Vec<ConversationMessage>,
    pub max_messages: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationMessage {
    pub role: String,
    pub content: String,
    pub timestamp: i64,
}

impl ConversationMemory {
    pub fn new(session_id: impl Into<String>, max_messages: usize) -> Self {
        Self {
            session_id: session_id.into(),
            messages: Vec::new(),
            max_messages,
        }
    }

    pub fn add(&mut self, role: impl Into<String>, content: impl Into<String>) {
        let msg = ConversationMessage {
            role: role.into(),
            content: content.into(),
            timestamp: chrono_timestamp(),
        };
        self.messages.push(msg);

        // Trim old messages if over limit
        if self.messages.len() > self.max_messages {
            let to_remove = self.messages.len() - self.max_messages;
            self.messages.drain(0..to_remove);
        }
    }

    pub fn clear(&mut self) {
        self.messages.clear();
    }

    #[must_use] 
    pub const fn len(&self) -> usize {
        self.messages.len()
    }

    #[must_use] 
    pub const fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }
}

fn chrono_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        // Seconds since the epoch exceed i64::MAX only ~292 billion years from now.
        .map_or(0, |d| i64::try_from(d.as_secs()).unwrap_or(i64::MAX))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The one scope rule every recall path shares.
    #[test]
    fn a_workspace_scope_sees_its_own_and_the_global_memories() {
        let ws = RecallScope::Workspace("a");
        assert!(ws.admits(None), "global is visible in a workspace");
        assert!(ws.admits(Some("a")), "own workspace");
        assert!(!ws.admits(Some("b")), "never another workspace");

        assert!(RecallScope::Everything.admits(Some("b")));
        assert!(RecallScope::Everything.admits(None));

        assert!(RecallScope::GlobalOnly.admits(None));
        assert!(
            !RecallScope::GlobalOnly.admits(Some("a")),
            "global-only means no workspace's memories"
        );

        assert_eq!(RecallScope::from_workspace(None), RecallScope::Everything);
        assert_eq!(RecallScope::from_workspace(Some("a")), ws);
        assert_eq!(RecallScope::GlobalOnly.sql_workspace(), None);
        assert_eq!(ws.sql_workspace(), Some("a"));
    }

    /// What the ranking did before `rank_top_k`: a STABLE sort of all N on
    /// similarity alone, then `truncate`. Kept verbatim as the oracle, so the
    /// equivalence claim is checked against the real previous behaviour rather
    /// than against a restatement of the new one.
    fn legacy_rank(similarities: &[f32], top_k: usize) -> Vec<(usize, f32)> {
        let mut ranked: Vec<(usize, f32)> = similarities.iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(top_k);
        ranked
    }

    fn new_rank(similarities: &[f32], top_k: usize) -> Vec<(usize, f32)> {
        let mut ranked: Vec<(usize, f32)> = similarities.iter().copied().enumerate().collect();
        VectorStore::rank_top_k(&mut ranked, top_k);
        ranked
    }

    #[test]
    fn selection_ranking_matches_the_stable_sort_it_replaced() {
        // Deliberately tie-heavy: only eight distinct scores across 200 slots,
        // so nearly every comparison is a tie and an unstable selection without
        // the index tiebreak would diverge immediately. Includes the stale
        // floor (-1.0), which a real store produces for every wrong-width row.
        const TIED: [f32; 8] = [0.99, 0.75, 0.75, 0.50, 0.50, 0.0, -1.0, -1.0];
        let scores: Vec<f32> = (0..200).map(|i| TIED[i % TIED.len()]).collect();

        for top_k in [1, 2, 3, 10, 31, 199, 200, 201, 1000] {
            assert_eq!(
                new_rank(&scores, top_k),
                legacy_rank(&scores, top_k),
                "selection diverged from the stable sort at top_k = {top_k}"
            );
        }
    }

    #[test]
    fn selection_ranking_handles_the_degenerate_sizes() {
        let scores = [0.3_f32, 0.9, 0.1];

        // top_k = 0 returns nothing rather than panicking in the selection.
        assert_eq!(new_rank(&scores, 0), Vec::new());
        // An empty candidate list is fine at any top_k.
        assert_eq!(new_rank(&[], 5), Vec::new());
        // top_k at and above the length keeps everything, fully ordered.
        assert_eq!(new_rank(&scores, 3), vec![(1, 0.9), (0, 0.3), (2, 0.1)]);
        assert_eq!(new_rank(&scores, 9), vec![(1, 0.9), (0, 0.3), (2, 0.1)]);
        // And the ordinary case keeps exactly top_k, best first.
        assert_eq!(new_rank(&scores, 2), vec![(1, 0.9), (0, 0.3)]);
    }

    #[test]
    fn ties_resolve_by_ascending_index_so_the_ranking_is_deterministic() {
        // Every score identical: the order is decided entirely by the tiebreak.
        let scores = [0.5_f32; 64];
        let ranked = new_rank(&scores, 5);
        assert_eq!(
            ranked.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4],
            "an all-ties ranking must return the first top_k by index"
        );
    }

    #[test]
    fn a_nan_score_ranks_last_instead_of_anywhere() {
        // A zero-magnitude embedding of the right width yields cosine 0/0.
        // The old comparator handed NaN to `partial_cmp`, got `None`, and
        // treated it as Equal — an intransitive comparator, so NaN could land
        // in the results. It must now rank below even the stale floor.
        let scores = [f32::NAN, -1.0, 0.8, f32::NAN, 0.2];

        let ranked = new_rank(&scores, 3);
        assert_eq!(
            ranked.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
            vec![2, 4, 1],
            "NaN must not displace a real score"
        );

        // And when NaN cannot be avoided it is last, in index order.
        let all = new_rank(&scores, 5);
        assert_eq!(
            all.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
            vec![2, 4, 1, 0, 3]
        );
    }

    fn store_of_width(width: usize) -> VectorStore {
        VectorStore::new(VectorStoreConfig {
            dimension: std::sync::atomic::AtomicUsize::new(width),
            chunk_max_chars: std::sync::atomic::AtomicUsize::new(0),
            use_f16: false,
        })
    }

    /// A row the way `store_unembedded` writes one: no vector, no buckets.
    fn unembedded(id: &str, timestamp: i64) -> MemoryEntry {
        MemoryEntry {
            id: id.to_string(),
            content: format!("memory {id}"),
            embedding: Vec::new(),
            embedding_model: None,
            embeddings: HashMap::new(),
            metadata: HashMap::new(),
            timestamp,
            fsrs: FsrsState::default(),
            workspace_id: None,
        }
    }

    /// P13 staleness: a memory written without a vector records how long it
    /// waited once it receives its first one.
    #[tokio::test]
    async fn a_first_vector_records_the_queue_to_searchable_wait() {
        let store = store_of_width(3);
        assert_eq!(store.searchable_latency(), None, "nothing has waited yet");
        store
            .add(unembedded("m", chrono_timestamp() - 7))
            .await
            .expect("stored unembedded");
        store
            .set_embedding_for_model("m", "prov:a", vec![1.0, 0.0, 0.0], true)
            .await
            .expect("filled");
        let latency = store.searchable_latency().expect("one wait recorded");
        assert_eq!(latency.samples, 1);
        assert!(latency.p50_secs >= 7, "waited at least 7 s: {latency:?}");
        assert_eq!(
            latency.p50_secs, latency.p95_secs,
            "one sample is both percentiles"
        );
    }

    /// A re-embed after a provider switch was already findable — it keeps its
    /// old bucket — so it is not a wait and must not skew the number.
    #[tokio::test]
    async fn a_re_embed_is_not_counted_as_a_wait() {
        let store = store_of_width(3);
        store.add(unembedded("m", 0)).await.expect("stored");
        store
            .set_embedding_for_model("m", "prov:a", vec![1.0, 0.0, 0.0], true)
            .await
            .expect("first vector");
        store
            .set_embedding_for_model("m", "prov:b", vec![0.0, 1.0, 0.0], true)
            .await
            .expect("provider switch");
        assert_eq!(
            store.searchable_latency().map(|l| l.samples),
            Some(1),
            "only the first vector counts"
        );
    }

    /// A fill into a bucket that is not activated leaves the memory
    /// unsearchable, so it does not end the wait either.
    #[tokio::test]
    async fn an_inactive_bucket_fill_does_not_end_the_wait() {
        let store = store_of_width(3);
        store.add(unembedded("m", 0)).await.expect("stored");
        store
            .set_embedding_for_model("m", "prov:a", vec![1.0, 0.0, 0.0], false)
            .await
            .expect("bucket only");
        assert_eq!(store.searchable_latency(), None);
    }

    #[test]
    fn latency_percentiles_use_nearest_rank() {
        let samples: Vec<u64> = (1..=100).collect();
        let latency = latency_percentiles(&samples).expect("samples");
        assert_eq!(latency.samples, 100);
        assert_eq!(latency.p50_secs, 50);
        assert_eq!(latency.p95_secs, 95);
        assert_eq!(latency_percentiles(&[]), None);
    }

    #[test]
    fn the_latency_ring_is_bounded_and_keeps_the_newest() {
        let mut ring = std::collections::VecDeque::new();
        for sample in 0..(SEARCHABLE_LATENCY_SAMPLES_MAX as u64 + 88) {
            push_bounded(&mut ring, sample);
        }
        assert_eq!(ring.len(), SEARCHABLE_LATENCY_SAMPLES_MAX);
        assert_eq!(ring.front(), Some(&88), "the oldest were dropped");
    }

    #[tokio::test]
    async fn test_vector_store() {
        let config = VectorStoreConfig {
            dimension: std::sync::atomic::AtomicUsize::new(8),
            chunk_max_chars: std::sync::atomic::AtomicUsize::new(0),
            use_f16: false,
        };
        let store = VectorStore::new(config);

        let entry = MemoryEntry {
            embedding_model: None,
            embeddings: HashMap::new(),
            id: "test1".to_string(),
            content: "Hello world".to_string(),
            embedding: vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            metadata: HashMap::new(),
            timestamp: 0,
            fsrs: FsrsState::default(),
            workspace_id: None,
        };

        store.add(entry).await.unwrap();
        assert_eq!(store.len().await, 1);

        let results = store
            .search(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 10)
            .await;
        assert_eq!(results.len(), 1);
        assert!(results[0].1 > 0.99);  // Should be very similar
    }

    fn health_test_entry(id: &str) -> MemoryEntry {
        MemoryEntry {
            embedding_model: None,
            embeddings: HashMap::new(),
            id: id.to_string(),
            content: "c".to_string(),
            embedding: vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            metadata: HashMap::new(),
            timestamp: 0,
            fsrs: FsrsState::default(),
            workspace_id: None,
        }
    }

    // A persistence whose salvage load reports corruption — proves store_health
    // surfacing without needing a real corrupt DB.
    struct DegradedDb;
    #[async_trait]
    impl MemoryPersistence for DegradedDb {
        async fn save_entry(&self, _e: &MemoryEntry) -> Result<(), MemoryError> { Ok(()) }
        async fn remove_entry(&self, _id: &str) -> Result<(), MemoryError> { Ok(()) }
        async fn update_entry_fsrs(&self, _id: &str, _f: &FsrsState) -> Result<(), MemoryError> { Ok(()) }
        async fn update_entry_content(&self, _id: &str, _c: &str) -> Result<(), MemoryError> { Ok(()) }
        async fn load_all(&self) -> Result<Vec<MemoryEntry>, MemoryError> {
            Ok(vec![health_test_entry("ok0"), health_test_entry("ok1")])
        }
        async fn load_all_report(&self) -> Result<LoadReport, MemoryError> {
            Ok(LoadReport {
                entries: vec![health_test_entry("ok0"), health_test_entry("ok1")],
                corrupt_rows: 3,
                expected: 5,
            })
        }
    }

    // A persistence with no salvage override — the default load_all_report reports
    // zero corruption.
    struct CleanDb;
    #[async_trait]
    impl MemoryPersistence for CleanDb {
        async fn save_entry(&self, _e: &MemoryEntry) -> Result<(), MemoryError> { Ok(()) }
        async fn remove_entry(&self, _id: &str) -> Result<(), MemoryError> { Ok(()) }
        async fn update_entry_fsrs(&self, _id: &str, _f: &FsrsState) -> Result<(), MemoryError> { Ok(()) }
        async fn update_entry_content(&self, _id: &str, _c: &str) -> Result<(), MemoryError> { Ok(()) }
        async fn load_all(&self) -> Result<Vec<MemoryEntry>, MemoryError> {
            Ok(vec![health_test_entry("ok")])
        }
    }

    /// Records every chunk-set replacement so a test can assert what the
    /// mutation paths actually wrote.
    struct ChunkRecordingDb {
        calls: std::sync::Mutex<Vec<(String, Vec<ChunkWrite>)>>,
    }

    impl ChunkRecordingDb {
        fn new() -> Self {
            Self { calls: std::sync::Mutex::new(Vec::new()) }
        }
        fn last(&self) -> (String, Vec<ChunkWrite>) {
            self.calls.lock().unwrap().last().cloned().expect("a chunk write happened")
        }
        fn call_count(&self) -> usize {
            self.calls.lock().unwrap().len()
        }
    }

    #[async_trait::async_trait]
    impl MemoryPersistence for ChunkRecordingDb {
        async fn save_entry(&self, _e: &MemoryEntry) -> Result<(), MemoryError> { Ok(()) }
        async fn remove_entry(&self, _id: &str) -> Result<(), MemoryError> { Ok(()) }
        async fn update_entry_fsrs(&self, _id: &str, _f: &FsrsState) -> Result<(), MemoryError> { Ok(()) }
        async fn update_entry_content(&self, _id: &str, _c: &str) -> Result<(), MemoryError> { Ok(()) }
        async fn load_all(&self) -> Result<Vec<MemoryEntry>, MemoryError> { Ok(Vec::new()) }
        async fn replace_chunks(
            &self,
            memory_id: &str,
            _workspace_id: Option<&str>,
            _model: Option<&str>,
            chunks: &[ChunkWrite],
        ) -> Result<(), MemoryError> {
            self.calls.lock().unwrap().push((memory_id.to_string(), chunks.to_vec()));
            Ok(())
        }
    }

    fn chunking_store(db: &Arc<ChunkRecordingDb>, max_chars: usize) -> VectorStore {
        VectorStore::new(VectorStoreConfig {
            dimension: std::sync::atomic::AtomicUsize::new(4),
            chunk_max_chars: std::sync::atomic::AtomicUsize::new(max_chars),
            use_f16: false,
        })
        .with_persistence(db.clone())
    }

    fn chunked_entry(id: &str, content: &str, embedding: Vec<f32>, model: Option<&str>) -> MemoryEntry {
        MemoryEntry {
            id: id.to_string(),
            content: content.to_string(),
            embeddings: HashMap::new(),
            embedding_model: model.map(str::to_string),
            embedding,
            metadata: HashMap::new(),
            timestamp: 0,
            fsrs: FsrsState::default(),
            workspace_id: None,
        }
    }

    /// Chunks have to be written by the STORE primitive, not by the ingest
    /// API. Consolidation and dreaming mutate entries through these directly,
    /// so a chunk set maintained only by `remember` would go stale exactly
    /// where memories change most.
    #[tokio::test]
    async fn adding_an_entry_writes_its_chunks() {
        let db = Arc::new(ChunkRecordingDb::new());
        let store = chunking_store(&db, 40);
        let content = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu";
        store
            .add(chunked_entry("m1", content, vec![1.0, 0.0, 0.0, 0.0], Some("prov:a")))
            .await
            .unwrap();

        let (id, chunks) = db.last();
        assert_eq!(id, "m1");
        assert!(chunks.len() > 1, "content longer than the budget must split");
        let rebuilt: String = chunks.iter().map(|c| c.content.as_str()).collect();
        assert_eq!(rebuilt, content, "the stored chunks must rebuild the memory");
        assert!(
            chunks.iter().all(|c| c.embedding.is_none()),
            "multi-chunk vectors are backfill work, not inline network calls"
        );
        assert!(chunks.iter().all(|c| c.chunk_max_chars == 40));
    }

    /// A memory that fits in one chunk already has a vector for exactly that
    /// text under exactly that model. Re-embedding it would be pure waste, so
    /// the parent vector seeds the single chunk.
    #[tokio::test]
    async fn a_single_chunk_memory_reuses_the_parent_vector() {
        let db = Arc::new(ChunkRecordingDb::new());
        let store = chunking_store(&db, 4000);
        store
            .add(chunked_entry("m2", "short enough", vec![1.0, 0.0, 0.0, 0.0], Some("prov:a")))
            .await
            .unwrap();

        let (_, chunks) = db.last();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].embedding.as_deref(), Some(&[1.0, 0.0, 0.0, 0.0][..]));
        assert_eq!(chunks[0].embedding_model.as_deref(), Some("prov:a"));
    }

    /// `update_content` changes text WITHOUT touching the embedding, so the
    /// parent vector now describes the old text. Seeding it into the new
    /// single chunk would assert that chunk had been embedded when it had not
    /// — a vector claiming to describe text it never saw.
    #[tokio::test]
    async fn a_content_only_update_rechunks_without_seeding_a_stale_vector() {
        let db = Arc::new(ChunkRecordingDb::new());
        let store = chunking_store(&db, 4000);
        store
            .add(chunked_entry("m3", "the harbor light was green", vec![1.0, 0.0, 0.0, 0.0], Some("prov:a")))
            .await
            .unwrap();
        store.update_content("m3", "the harbor light was red").await.unwrap();

        let (id, chunks) = db.last();
        assert_eq!(id, "m3");
        assert_eq!(db.call_count(), 2, "the update must rewrite the chunk set");
        assert_eq!(chunks[0].content, "the harbor light was red");
        assert!(
            chunks[0].embedding.is_none(),
            "the parent vector describes the OLD text and must not be reused"
        );
    }

    /// The merge/dream path replaces content and embedding together, so the
    /// incoming vector genuinely describes the new text.
    #[tokio::test]
    async fn a_merge_rechunks_and_may_seed_the_new_vector() {
        let db = Arc::new(ChunkRecordingDb::new());
        let store = chunking_store(&db, 4000);
        store
            .add(chunked_entry("m4", "first", vec![1.0, 0.0, 0.0, 0.0], Some("prov:a")))
            .await
            .unwrap();
        store
            .update_content_and_embedding("m4", "merged text", vec![0.0, 1.0, 0.0, 0.0], Some("prov:a"))
            .await
            .unwrap();

        let (_, chunks) = db.last();
        assert_eq!(db.call_count(), 2);
        assert_eq!(chunks[0].content, "merged text");
        assert_eq!(chunks[0].embedding_model.as_deref(), Some("prov:a"));
    }

    #[tokio::test]
    async fn store_health_reflects_salvage_report() {
        let config = VectorStoreConfig {
            dimension: std::sync::atomic::AtomicUsize::new(8),
            chunk_max_chars: std::sync::atomic::AtomicUsize::new(0),
            use_f16: false,
        };
        let store = VectorStore::new(config).with_persistence(Arc::new(DegradedDb));
        let loaded = store.load_from_db().await.unwrap();
        assert_eq!(loaded, 2);
        let h = store.store_health().await;
        assert!(h.degraded);
        assert_eq!(h.corrupt_rows, 3);
        assert_eq!(h.loaded, 2);
        assert_eq!(h.expected, 5);
    }

    #[tokio::test]
    async fn store_health_clean_when_no_corruption() {
        let config = VectorStoreConfig {
            dimension: std::sync::atomic::AtomicUsize::new(8),
            chunk_max_chars: std::sync::atomic::AtomicUsize::new(0),
            use_f16: false,
        };
        let store = VectorStore::new(config).with_persistence(Arc::new(CleanDb));
        store.load_from_db().await.unwrap();
        let h = store.store_health().await;
        assert!(!h.degraded);
        assert_eq!(h.corrupt_rows, 0);
        assert_eq!(h.loaded, 1);
        assert_eq!(h.expected, 1);
    }

    // A persistence whose whole load fails (e.g. btree corruption the salvage
    // can't localize to a row).
    struct FailingDb;
    #[async_trait]
    impl MemoryPersistence for FailingDb {
        async fn save_entry(&self, _e: &MemoryEntry) -> Result<(), MemoryError> { Ok(()) }
        async fn remove_entry(&self, _id: &str) -> Result<(), MemoryError> { Ok(()) }
        async fn update_entry_fsrs(&self, _id: &str, _f: &FsrsState) -> Result<(), MemoryError> { Ok(()) }
        async fn update_entry_content(&self, _id: &str, _c: &str) -> Result<(), MemoryError> { Ok(()) }
        async fn load_all(&self) -> Result<Vec<MemoryEntry>, MemoryError> {
            Err(MemoryError::Persistence("inconsistent overflow chain".into()))
        }
    }

    #[tokio::test]
    async fn store_health_degraded_when_whole_load_fails() {
        // A whole-store load failure must still mark the store degraded, so
        // /status can't report a corrupt store as healthy-but-empty.
        let config = VectorStoreConfig {
            dimension: std::sync::atomic::AtomicUsize::new(8),
            chunk_max_chars: std::sync::atomic::AtomicUsize::new(0),
            use_f16: false,
        };
        let store = VectorStore::new(config).with_persistence(Arc::new(FailingDb));
        assert!(store.load_from_db().await.is_err());
        let h = store.store_health().await;
        assert!(h.degraded, "a whole-store load failure must mark degraded");
        assert_eq!(h.loaded, 0);
    }

    // Counts single vs batched removals to prove `remove_many` routes through the
    // batched persistence path (one call) instead of N single `remove_entry`s.
    #[derive(Default)]
    struct CountingRemovalDb {
        single_calls: std::sync::atomic::AtomicUsize,
        batch_calls: std::sync::atomic::AtomicUsize,
        batch_ids_total: std::sync::atomic::AtomicUsize,
    }
    #[async_trait]
    impl MemoryPersistence for CountingRemovalDb {
        async fn save_entry(&self, _e: &MemoryEntry) -> Result<(), MemoryError> { Ok(()) }
        async fn remove_entry(&self, _id: &str) -> Result<(), MemoryError> {
            self.single_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        }
        async fn remove_entries(&self, ids: &[&str]) -> Result<(), MemoryError> {
            self.batch_calls
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.batch_ids_total
                .fetch_add(ids.len(), std::sync::atomic::Ordering::SeqCst);
            Ok(())
        }
        async fn update_entry_fsrs(&self, _id: &str, _f: &FsrsState) -> Result<(), MemoryError> { Ok(()) }
        async fn update_entry_content(&self, _id: &str, _c: &str) -> Result<(), MemoryError> { Ok(()) }
        async fn load_all(&self) -> Result<Vec<MemoryEntry>, MemoryError> { Ok(vec![]) }
    }

    fn entry_dim8(id: &str) -> MemoryEntry {
        MemoryEntry {
            embedding_model: None,
            embeddings: HashMap::new(),
            id: id.to_string(),
            content: format!("content {id}"),
            embedding: vec![0.0; 8],
            metadata: HashMap::new(),
            timestamp: 0,
            fsrs: FsrsState::default(),
            workspace_id: None,
        }
    }

    /// A backend that refuses to delete one named row, and whose batch delete
    /// always fails part-way — the shape of a real `bulk_delete`, which is not
    /// one transaction.
    struct RefusingDb {
        refuse: &'static str,
    }
    #[async_trait]
    impl MemoryPersistence for RefusingDb {
        async fn save_entry(&self, _e: &MemoryEntry) -> Result<(), MemoryError> {
            Ok(())
        }
        async fn remove_entry(&self, id: &str) -> Result<(), MemoryError> {
            if id == self.refuse {
                return Err(MemoryError::Persistence("database is locked".into()));
            }
            Ok(())
        }
        async fn remove_entries(&self, _ids: &[&str]) -> Result<(), MemoryError> {
            Err(MemoryError::Persistence("batch stopped part-way".into()))
        }
        async fn update_entry_fsrs(&self, _id: &str, _f: &FsrsState) -> Result<(), MemoryError> {
            Ok(())
        }
        async fn update_entry_content(&self, _id: &str, _c: &str) -> Result<(), MemoryError> {
            Ok(())
        }
        async fn load_all(&self) -> Result<Vec<MemoryEntry>, MemoryError> {
            Ok(vec![])
        }
    }

    fn refusing_store(refuse: &'static str) -> VectorStore {
        VectorStore::new(VectorStoreConfig {
            dimension: std::sync::atomic::AtomicUsize::new(8),
            chunk_max_chars: std::sync::atomic::AtomicUsize::new(0),
            use_f16: false,
        })
        .with_persistence(Arc::new(RefusingDb { refuse }))
    }

    /// A delete the backend refused used to leave RAM first and log the
    /// failure, so the caller was told "deleted" and a restart brought the
    /// memory back. Now it is refused, and RAM still matches disk.
    #[tokio::test]
    async fn a_delete_the_backend_refuses_is_refused() {
        let store = refusing_store("stuck");
        store.add(entry_dim8("stuck")).await.unwrap();
        store.add(entry_dim8("free")).await.unwrap();

        let refused = store.remove("stuck").await;
        assert!(
            matches!(refused, Err(MemoryError::Persistence(_))),
            "the backend's error is returned: {refused:?}"
        );
        assert!(
            store.get("stuck").await.is_some(),
            "kept, because disk kept it"
        );

        store
            .remove("free")
            .await
            .expect("an ordinary delete still works");
        assert!(store.get("free").await.is_none());
        assert!(matches!(
            store.remove("free").await,
            Err(MemoryError::NotFound(_))
        ));
    }

    /// The batch is not one transaction, so its failure says nothing about
    /// which rows went. The durable path finds out one id at a time and removes
    /// from RAM exactly the rows the backend confirmed gone.
    #[tokio::test]
    async fn a_failed_batch_removes_exactly_what_the_backend_confirmed() {
        let store = refusing_store("m1");
        for i in 0..3 {
            store.add(entry_dim8(&format!("m{i}"))).await.unwrap();
        }

        let outcome = store.remove_many_durable(&["m0", "m1", "m2"]).await;

        assert_eq!(outcome.removed, 2);
        assert_eq!(outcome.failed, 1);
        assert!(
            outcome
                .first_error
                .as_deref()
                .is_some_and(|e| e.contains("locked")),
            "the refused row's own error is reported: {outcome:?}"
        );
        assert!(
            store.get("m1").await.is_some(),
            "the refused row stays visible"
        );
        assert_eq!(store.len().await, 1);
        assert_eq!(
            store.remove_many_durable(&[]).await,
            DurableRemoval::default()
        );
    }

    #[tokio::test]
    async fn remove_many_uses_one_batched_persistence_call() {
        use std::sync::atomic::Ordering::SeqCst;
        let db = Arc::new(CountingRemovalDb::default());
        let store = VectorStore::new(VectorStoreConfig {
            dimension: std::sync::atomic::AtomicUsize::new(8),
            chunk_max_chars: std::sync::atomic::AtomicUsize::new(0),
            use_f16: false,
        })
        .with_persistence(db.clone());
        for i in 0..3 {
            store.add(entry_dim8(&format!("m{i}"))).await.unwrap();
        }

        let removed = store.remove_many(&["m0", "m1", "m2"]).await;

        assert_eq!(removed, 3, "all three present entries removed from cache");
        assert_eq!(
            db.batch_calls.load(SeqCst),
            1,
            "exactly one batched persistence call"
        );
        assert_eq!(
            db.batch_ids_total.load(SeqCst),
            3,
            "the batch carried all three ids"
        );
        assert_eq!(
            db.single_calls.load(SeqCst),
            0,
            "no per-row remove_entry calls"
        );
        assert_eq!(store.len().await, 0);
    }

    #[tokio::test]
    async fn remove_many_empty_is_a_noop() {
        let db = Arc::new(CountingRemovalDb::default());
        let store = VectorStore::new(VectorStoreConfig {
            dimension: std::sync::atomic::AtomicUsize::new(8),
            chunk_max_chars: std::sync::atomic::AtomicUsize::new(0),
            use_f16: false,
        })
        .with_persistence(db.clone());
        let removed = store.remove_many(&[]).await;
        assert_eq!(removed, 0);
        assert_eq!(
            db.batch_calls.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "an empty batch makes no persistence call"
        );
    }

    #[tokio::test]
    async fn test_save_is_atomic_and_roundtrips() {
        let config = VectorStoreConfig {
            dimension: std::sync::atomic::AtomicUsize::new(8),
            chunk_max_chars: std::sync::atomic::AtomicUsize::new(0),
            use_f16: false,
        };
        let store = VectorStore::new(config);
        store
            .add(MemoryEntry {
                embedding_model: None,
                embeddings: HashMap::new(),
                id: "s1".to_string(),
                content: "persist me".to_string(),
                embedding: vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                metadata: HashMap::new(),
                timestamp: 0,
                fsrs: FsrsState::default(),
                workspace_id: None,
            })
            .await
            .unwrap();

        let dir = std::env::temp_dir().join(format!("nanna_mem_save_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("memories.json");

        store.save(&path).await.unwrap();
        // Target written; the temp file was renamed away (not left behind).
        assert!(path.exists());
        assert!(!dir.join("memories.json.tmp").exists());

        // The store round-trips through load into a fresh store.
        let reloaded = VectorStore::new(VectorStoreConfig {
            dimension: std::sync::atomic::AtomicUsize::new(8),
            chunk_max_chars: std::sync::atomic::AtomicUsize::new(0),
            use_f16: false,
        });
        reloaded.load(&path).await.unwrap();
        assert_eq!(reloaded.len().await, 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_conversation_memory() {
        let mut memory = ConversationMemory::new("test", 3);
        memory.add("user", "Hello");
        memory.add("assistant", "Hi there!");
        memory.add("user", "How are you?");
        memory.add("assistant", "I'm good!");

        assert_eq!(memory.len(), 3);  // Trimmed to max
        assert_eq!(memory.messages[0].role, "assistant");  // First message was trimmed
    }

    /// The queued-for-backfill state must be writable: an entry whose vector
    /// raced a provider switch arrives with an empty embedding, and rejecting
    /// it (the old `0 != dim` mismatch) is exactly the write-failure loop the
    /// incident produced.
    #[tokio::test]
    async fn add_accepts_a_queued_entry_awaiting_backfill() {
        let store = VectorStore::new(VectorStoreConfig {
            dimension: std::sync::atomic::AtomicUsize::new(8),
            chunk_max_chars: std::sync::atomic::AtomicUsize::new(0),
            use_f16: false,
        });
        let mut queued = entry_dim8("queued");
        queued.embedding = Vec::new();
        store.add(queued).await.expect("a queued entry is a legal write");

        // A non-empty wrong-width vector is still a hard error — only the
        // explicit queued state is exempt from the width gate.
        let mut wrong = entry_dim8("wrong");
        wrong.embedding = vec![0.5; 3];
        let err = store.add(wrong).await.expect_err("wrong width still rejected");
        assert!(matches!(err, MemoryError::DimensionMismatch { expected: 8, got: 3 }));
    }

    /// A queued row in the store must be skipped by search, not panic it: the
    /// raw SIMD cosine asserts equal widths and this crate aborts on panic, so
    /// before the fix one queued row took the daemon down on the next recall.
    /// The ranking properties, proven through the **shipped** `search` rather
    /// than through `rank_top_k` in isolation. The unit tests above pin the
    /// ranking function; this one pins that `search` actually uses it, and that
    /// `top_k` really does bound what comes back out of a larger store.
    #[tokio::test]
    async fn search_ranks_ties_deterministically_and_honours_top_k() {
        let store = VectorStore::new(VectorStoreConfig {
            dimension: std::sync::atomic::AtomicUsize::new(8),
            chunk_max_chars: std::sync::atomic::AtomicUsize::new(0),
            use_f16: false,
        });

        // Twenty rows, all with the SAME embedding, so every score ties and the
        // order is decided entirely by the index tiebreak. Insertion order is
        // the index order, so the answer must be the first `top_k` inserted.
        for i in 0..20 {
            store
                .add(MemoryEntry {
                    embedding: vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ..entry_dim8(&format!("tied-{i:02}"))
                })
                .await
                .unwrap();
        }

        let query = [1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let ranked_ids = |r: &[(MemoryEntry, f32)]| -> Vec<String> {
            r.iter().map(|(e, _)| e.id.clone()).collect()
        };

        let first = store.search(&query, 5).await;
        assert_eq!(first.len(), 5, "top_k bounds the result");
        assert_eq!(
            ranked_ids(&first),
            ["tied-00", "tied-01", "tied-02", "tied-03", "tied-04"],
        );

        // Repeat it: an unstable selection without the tiebreak would be free
        // to return a different five, or the same five in another order.
        for _ in 0..8 {
            assert_eq!(
                ranked_ids(&store.search(&query, 5).await),
                ranked_ids(&first),
                "an all-ties ranking must be stable across calls"
            );
        }
    }

    /// A zero-magnitude embedding of the RIGHT width yields cosine `0/0` = NaN.
    /// The width check cannot catch it — the vector is the correct length, it
    /// is the content that is degenerate — so the ranking is the only thing
    /// standing between a NaN and the results. It must lose to a real match.
    #[tokio::test]
    async fn search_never_returns_a_nan_scored_row_ahead_of_a_real_match() {
        let store = VectorStore::new(VectorStoreConfig {
            dimension: std::sync::atomic::AtomicUsize::new(8),
            chunk_max_chars: std::sync::atomic::AtomicUsize::new(0),
            use_f16: false,
        });
        // Right width, zero magnitude — this is the row that produces NaN.
        store
            .add(MemoryEntry {
                embedding: vec![0.0; 8],
                ..entry_dim8("zero-magnitude")
            })
            .await
            .unwrap();
        store
            .add(MemoryEntry {
                embedding: vec![0.2, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ..entry_dim8("weak-match")
            })
            .await
            .unwrap();

        let query = [1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        // Asked for one: it must be the real row, never the degenerate one.
        let top = store.search(&query, 1).await;
        assert_eq!(top.len(), 1);
        assert_eq!(
            top[0].0.id, "weak-match",
            "a NaN-scoring row must not outrank a real, if weak, match"
        );

        // Asked for both: the degenerate row comes last, and its score is not
        // a NaN leaking out to callers that compare it against a threshold.
        let all = store.search(&query, 10).await;
        assert_eq!(
            all.iter().map(|(e, _)| e.id.as_str()).collect::<Vec<_>>(),
            ["weak-match", "zero-magnitude"],
        );
    }

    #[tokio::test]
    async fn search_skips_queued_rows_instead_of_panicking() {
        let store = VectorStore::new(VectorStoreConfig {
            dimension: std::sync::atomic::AtomicUsize::new(8),
            chunk_max_chars: std::sync::atomic::AtomicUsize::new(0),
            use_f16: false,
        });
        let good = MemoryEntry {
            embedding: vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ..entry_dim8("good")
        };
        store.add(good).await.unwrap();
        let mut queued = entry_dim8("queued");
        queued.embedding = Vec::new();
        store.add(queued).await.unwrap();

        let results = store
            .search(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 10)
            .await;
        assert_eq!(results.len(), 2, "both rows are ranked");
        assert_eq!(results[0].0.id, "good");
        assert!(results[0].1 > 0.99);
        assert!(
            results[1].1 <= -1.0 + f32::EPSILON,
            "the queued row scores at the stale floor, below any min_score"
        );
    }

    /// An empty answer must be able to say WHY. The three states are not
    /// interchangeable: nothing matched, N of M rows carry no vector this query
    /// can be scored against, and the query width does not match the binding at
    /// all — the last one is a total blackout, and reported as "no memories
    /// found" it reads as an honest miss.
    ///
    /// Measured in the scan, not from a rebind-time snapshot: the backfill
    /// drains continuously, so any counter taken when the binding changed is
    /// stale by the time a question is asked.
    #[tokio::test]
    async fn search_reports_what_it_could_actually_compare() {
        let store = VectorStore::new(VectorStoreConfig {
            dimension: std::sync::atomic::AtomicUsize::new(8),
            chunk_max_chars: std::sync::atomic::AtomicUsize::new(0),
            use_f16: false,
        });
        let good = MemoryEntry {
            embedding: vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ..entry_dim8("good")
        };
        store.add(good).await.unwrap();
        let mut queued = entry_dim8("queued");
        queued.embedding = Vec::new();
        store.add(queued).await.unwrap();
        let mut stale = entry_dim8("stale");
        stale.embedding = Vec::new();
        store.add(stale).await.unwrap();

        let (_, coverage) = store
            .search_with_coverage(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 10)
            .await;
        assert_eq!(coverage.total, 3);
        assert_eq!(coverage.comparable, 1, "only the embedded row was scored");
        assert_eq!(coverage.unsearchable(), 2);
        assert!(coverage.query_width_matches);
        assert!(
            !coverage.is_complete(),
            "an empty answer from this scan would not mean 'nothing matched'"
        );

        // The blackout: the query is the wrong width for the binding, so the
        // scan never ran. Every memory is unreachable, not absent.
        let (results, coverage) = store.search_with_coverage(&[1.0, 0.0], 10).await;
        assert!(results.is_empty());
        assert!(!coverage.query_width_matches);
        assert_eq!(coverage.total, 3, "the store is not empty — it is unreachable");
        assert_eq!(coverage.comparable, 0);

        // And a store that could be read whole says so, so a genuine miss is
        // still reportable as a genuine miss.
        let searchable = VectorStore::new(VectorStoreConfig {
            dimension: std::sync::atomic::AtomicUsize::new(8),
            chunk_max_chars: std::sync::atomic::AtomicUsize::new(0),
            use_f16: false,
        });
        searchable.add(entry_dim8("only")).await.unwrap();
        let (_, coverage) = searchable.search_with_coverage(&[0.0; 8], 10).await;
        assert!(coverage.is_complete());
    }

    /// Rewriting content invalidates every previously computed vector, so the
    /// bucket map must reset to just the producing model's — a later rebind
    /// must never resurrect a vector for text that no longer exists.
    #[tokio::test]
    async fn update_content_and_embedding_resets_stale_buckets() {
        let store = VectorStore::new(VectorStoreConfig {
            dimension: std::sync::atomic::AtomicUsize::new(8),
            chunk_max_chars: std::sync::atomic::AtomicUsize::new(0),
            use_f16: false,
        });
        let mut entry = entry_dim8("m");
        entry.embedding_model = Some("prov:a".into());
        entry.embeddings.insert("prov:a".into(), vec![0.0; 8]);
        entry.embeddings.insert("prov:old".into(), vec![0.0; 4]);
        store.add(entry).await.unwrap();

        store
            .update_content_and_embedding("m", "rewritten", vec![1.0; 8], Some("prov:a"))
            .await
            .unwrap();
        let updated = store.get("m").await.unwrap();
        assert_eq!(updated.content, "rewritten");
        assert_eq!(updated.embedding_model.as_deref(), Some("prov:a"));
        assert_eq!(
            updated.embeddings.keys().collect::<Vec<_>>(),
            vec!["prov:a"],
            "only the producing model's bucket survives a rewrite"
        );

        // The queued form of the same rewrite: content lands, vectors clear.
        store
            .update_content_and_embedding("m", "rewritten again", Vec::new(), None)
            .await
            .unwrap();
        let queued = store.get("m").await.unwrap();
        assert_eq!(queued.content, "rewritten again");
        assert!(queued.embedding.is_empty(), "queued for backfill");
        assert!(queued.embeddings.is_empty());
        assert_eq!(queued.embedding_model, None);
    }

    /// The compare-and-swap half of the fold-vs-live-write fix: a guarded
    /// rewrite whose expectation went stale must decline and touch nothing —
    /// applying a merge computed from a stale snapshot is a silent lost
    /// update (the dream-cycle vs `remember` interleave, 2026-08-10 class).
    #[tokio::test]
    async fn guarded_rewrite_declines_when_the_content_moved() {
        let store = VectorStore::new(VectorStoreConfig {
            dimension: std::sync::atomic::AtomicUsize::new(8),
            chunk_max_chars: std::sync::atomic::AtomicUsize::new(0),
            use_f16: false,
        });
        store.add(entry_dim8("m")).await.unwrap();
        let snapshot = store.get("m").await.unwrap().content;

        // A concurrent writer moves the content…
        store
            .update_content_and_embedding("m", "the live write", vec![1.0; 8], Some("prov:a"))
            .await
            .unwrap();

        // …so a rewrite guarded by the OLD snapshot must decline whole.
        let applied = store
            .update_content_and_embedding_if(
                "m",
                Some(snapshot.as_str()),
                "a merge of the stale snapshot",
                vec![0.5; 8],
                Some("prov:a"),
            )
            .await
            .unwrap();
        assert!(!applied, "a stale expectation must decline the swap");
        let live = store.get("m").await.unwrap();
        assert_eq!(
            live.content, "the live write",
            "a declined swap must leave the live write intact"
        );
        assert_eq!(
            live.embedding_model.as_deref(),
            Some("prov:a"),
            "a declined swap must not touch the binding either"
        );

        // A fresh expectation applies normally.
        let applied = store
            .update_content_and_embedding_if(
                "m",
                Some("the live write"),
                "a merge of the live content",
                vec![0.25; 8],
                Some("prov:a"),
            )
            .await
            .unwrap();
        assert!(applied);
        assert_eq!(
            store.get("m").await.unwrap().content,
            "a merge of the live content"
        );
    }

    /// The matched-removal half: removal authorized against a snapshot must
    /// only delete entries whose content is STILL that snapshot. A source a
    /// live write rewrote after the fold decision keeps its fresh content.
    #[tokio::test]
    async fn matched_removal_keeps_a_rewritten_entry() {
        let store = VectorStore::new(VectorStoreConfig {
            dimension: std::sync::atomic::AtomicUsize::new(8),
            chunk_max_chars: std::sync::atomic::AtomicUsize::new(0),
            use_f16: false,
        });
        store.add(entry_dim8("stale")).await.unwrap();
        store.add(entry_dim8("moved")).await.unwrap();
        let stale_snapshot = store.get("stale").await.unwrap().content;
        let moved_snapshot = store.get("moved").await.unwrap().content;

        // "moved" is rewritten between the fold decision and the batch removal.
        store
            .update_content_and_embedding("moved", "fresh content the fold never saw", vec![1.0; 8], None)
            .await
            .unwrap();

        let removed = store
            .remove_many_matching(&[
                ("stale", stale_snapshot.as_str()),
                ("moved", moved_snapshot.as_str()),
            ])
            .await;

        assert_eq!(removed, 1, "only the unchanged entry may be removed");
        assert!(store.get("stale").await.is_none(), "the unchanged source is folded away");
        let kept = store.get("moved").await.expect("the rewritten source must survive");
        assert_eq!(kept.content, "fresh content the fold never saw");

        // Negative space: an empty batch is a no-op.
        assert_eq!(store.remove_many_matching(&[]).await, 0);
    }
}
