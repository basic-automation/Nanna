//! Turso-backed `MemoryPersistence` adapter
//!
//! Bridges `nanna_memory::MemoryPersistence` ↔ `nanna_storage::MemoryRepository`,
//! converting between `MemoryEntry` (in-memory type) and `Memory`/`NewMemory`
//! (storage models).  All FSRS fields are round-tripped losslessly.

use async_trait::async_trait;
use nanna_memory::{ChunkWrite, FsrsState, LoadReport, MemoryEntry, MemoryError, MemoryPersistence, PendingChunk};
use nanna_storage::{MemoryFsrsUpdate, MemoryRepository, NewMemory, NewMemoryChunk};
use std::collections::HashMap;
use tracing::{info, warn};

/// Implements `MemoryPersistence` using the `MemoryRepository` from `nanna-storage`.
pub struct TursoMemoryPersistence {
    repo: MemoryRepository,
}

impl TursoMemoryPersistence {
    #[must_use]
    pub const fn new(repo: MemoryRepository) -> Self {
        Self { repo }
    }
}

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

/// Convert a `MemoryEntry` to a `NewMemory` for INSERT/UPSERT.
fn entry_to_new_memory(entry: &MemoryEntry) -> NewMemory {
    // Serialize metadata HashMap<String,String> as JSON
    let metadata_value: Option<serde_json::Value> = if entry.metadata.is_empty() {
        None
    } else {
        serde_json::to_value(&entry.metadata).ok()
    };

    // Extract session_id from metadata if present
    let session_id = entry.metadata.get("session_id").cloned();

    NewMemory {
        memory_id: entry.id.clone(),
        content: entry.content.clone(),
        embedding: Some(entry.embedding.clone()),
        // Now tracked. A NULL here is why a same-width vector from a different
        // model was undetectable: the column existed, nothing ever filled it.
        embedding_model: entry.embedding_model.clone(),
        session_id,
        metadata: metadata_value,
        tags: Vec::new(), // Tags are stored inside metadata for MemoryEntry
        workspace_id: entry.workspace_id.clone(),
        fsrs_stability: entry.fsrs.stability,
        fsrs_difficulty: entry.fsrs.difficulty,
        fsrs_last_access: entry.fsrs.last_access,
        fsrs_access_count: i64::from(entry.fsrs.access_count),
        fsrs_importance: entry.fsrs.importance,
        fsrs_storage_strength: entry.fsrs.storage_strength,
        fsrs_generation: i64::from(entry.fsrs.generation),
    }
}

/// Attach every persisted bucket to `entry`, and activate the one matching its
/// recorded model.
///
/// Buckets used to live only in RAM, so a restart collapsed a memory's whole
/// embedding history down to the single vector in the row. Restoring them is
/// what makes a provider flap cost nothing across process boundaries.
fn attach_buckets(entry: &mut MemoryEntry, buckets: Vec<(String, Vec<f32>)>) {
    for (model, vector) in buckets {
        if vector.is_empty() {
            continue;
        }
        // The row's own vector wins for the model it names — it is the copy the
        // search path materialized, and disagreeing with it here would make the
        // active vector and its bucket differ for the same model.
        entry.embeddings.entry(model).or_insert(vector);
    }
    if entry.embedding.is_empty()
        && let Some(model) = entry.embedding_model.clone()
        && let Some(vector) = entry.embeddings.get(&model)
    {
        entry.embedding = vector.clone();
    }
}

/// Convert a `nanna_storage::Memory` back to a `MemoryEntry`.
///
/// A row with no vector converts like any other, carrying an empty embedding.
/// That is the documented meaning of an empty `MemoryEntry::embedding` — "not
/// embedded by the active model yet, which is a backfill job, not an error" —
/// and the store is built for it: `add` skips the width check on an empty
/// vector, `search` drops to the SIMD path rather than misaligning the GPU
/// batch, and `entries_missing_model` treats exactly these rows as the backfill
/// queue.
///
/// This used to `return None`, and the caller counted it as a benign skip. It
/// was not benign. A memory written while the embedding provider was down
/// persisted its content fine and then vanished on the next restart: never
/// loaded, so never searchable, and — because the backfill queue is built from
/// loaded entries — never backfilled either. The row stayed in the database
/// forever, invisible. The log line said "skipped", which reads as "nothing to
/// see", not "this memory is now unreachable for good".
#[must_use]
pub fn db_memory_to_entry(mem: nanna_storage::Memory) -> MemoryEntry {
    let embedding = mem.embedding.unwrap_or_default();

    // Rebuild metadata HashMap from JSON
    let mut metadata: HashMap<String, String> = mem.metadata
        .as_ref()
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();

    // Re-inject session_id into metadata if it was stored in session_id column
    if let Some(ref sid) = mem.session_id {
        metadata.entry("session_id".to_string()).or_insert_with(|| sid.clone());
    }

    // Parse timestamp from created_at ISO string (fallback: 0)
    let timestamp = chrono::DateTime::parse_from_rfc3339(&mem.created_at)
        .or_else(|_| {
            // Try Turso datetime format 'YYYY-MM-DD HH:MM:SS'
            chrono::NaiveDateTime::parse_from_str(&mem.created_at, "%Y-%m-%d %H:%M:%S")
                .map(|ndt| ndt.and_utc().fixed_offset())
        })
        .map_or(0, |dt| dt.timestamp());

    let fsrs = FsrsState {
        stability: mem.fsrs_stability,
        difficulty: mem.fsrs_difficulty,
        last_access: mem.fsrs_last_access,
        access_count: crate::numeric::u32_clamped(mem.fsrs_access_count),
        importance: mem.fsrs_importance,
        storage_strength: mem.fsrs_storage_strength,
        generation: crate::numeric::u32_clamped(mem.fsrs_generation),
    };

    MemoryEntry {
        id: mem.memory_id,
        content: mem.content,
        // The DB keeps one vector per row plus the model that made it. Retained
        // buckets live in RAM for the life of the daemon, so a provider that
        // flaps mid-session costs nothing; across a restart only the persisted
        // one survives and the rest are backfilled lazily.
        //
        // An empty vector gets NO bucket: a bucket keyed by the model name
        // asserts "this model has embedded this entry", and an empty vector
        // asserts the opposite. Recording one would lift the row straight out
        // of the backfill queue it belongs in.
        embeddings: mem
            .embedding_model
            .clone()
            .filter(|_| !embedding.is_empty())
            .map(|model| HashMap::from([(model, embedding.clone())]))
            .unwrap_or_default(),
        embedding_model: mem.embedding_model.clone(),
        embedding,
        metadata,
        timestamp,
        fsrs,
        workspace_id: mem.workspace_id,
    }
}

// ---------------------------------------------------------------------------
// Trait implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl MemoryPersistence for TursoMemoryPersistence {
    async fn save_entry(&self, entry: &MemoryEntry) -> Result<(), MemoryError> {
        let new_mem = entry_to_new_memory(entry);
        // Persist every bucket the entry carries, not only the active one.
        // `memories.embedding` remains the materialized active copy for the
        // existing search path; `memory_vectors` is the durable set, and it is
        // what makes returning to a provider after an outage a lookup instead
        // of a re-embed.
        for (model, vector) in &entry.embeddings {
            if vector.is_empty() || model.is_empty() {
                continue;
            }
            if let Err(e) = self.repo.put_memory_vector(&entry.id, model, vector).await {
                warn!("Could not persist bucket '{model}' for {}: {e}", entry.id);
            }
        }

        // Try INSERT; if a duplicate memory_id exists, update all fields instead.
        // We use a two-step approach because turso INSERT OR REPLACE would reset
        // the autoincrement id — instead we attempt insert and fall back to update.
        match self.repo.create(new_mem).await {
            Ok(_) => Ok(()),
            Err(nanna_storage::StorageError::Database(ref e))
                if e.to_string().contains("UNIQUE") || e.to_string().contains("unique") =>
            {
                // Entry exists — update FSRS + content + EMBEDDING.
                //
                // The embedding was missing here, which is why a backfilled
                // vector never survived a restart: `set_embedding_for_model`
                // saved the entry, the save fell into this branch, and the one
                // field it had just computed was the one field not written.
                //
                // Each write's error is returned, not discarded: the three
                // `let _ =` here reported a save that had not happened, so the
                // caller kept the RAM copy and a restart silently rolled it back.
                // The order leaves any failure in the safe state — content first
                // (which clears the stored vector), so a failure after it leaves
                // the row queued for re-embedding rather than wrongly searchable.
                let failed = |what: &str, e: nanna_storage::StorageError| {
                    MemoryError::Persistence(format!("updating the {what} of {}: {e}", entry.id))
                };
                self.repo
                    .update_content(&entry.id, &entry.content)
                    .await
                    .map_err(|e| failed("content", e))?;
                if !entry.embedding.is_empty() {
                    self.repo
                        .update_embedding(
                            &entry.id,
                            &entry.embedding,
                            entry.embedding_model.as_deref(),
                        )
                        .await
                        .map_err(|e| failed("embedding", e))?;
                }
                self.repo
                    .update_fsrs(
                        &entry.id,
                        &MemoryFsrsUpdate {
                            stability: entry.fsrs.stability,
                            difficulty: entry.fsrs.difficulty,
                            last_access: entry.fsrs.last_access,
                            access_count: i64::from(entry.fsrs.access_count),
                            importance: entry.fsrs.importance,
                            storage_strength: entry.fsrs.storage_strength,
                            generation: i64::from(entry.fsrs.generation),
                        },
                    )
                    .await
                    .map_err(|e| failed("FSRS state", e))?;
                Ok(())
            }
            Err(e) => Err(MemoryError::Persistence(e.to_string())),
        }
    }

    /// Whole-set replacement, which is what `MemoryRepository::replace_chunks`
    /// does under one connection lock — the delete and the inserts are one
    /// transaction, so a crash mid-write cannot leave a parent holding half of
    /// its old chunking and half of its new one.
    async fn replace_chunks(
        &self,
        memory_id: &str,
        workspace_id: Option<&str>,
        active_model: Option<&str>,
        chunks: &[ChunkWrite],
    ) -> Result<(), MemoryError> {
        let rows: Vec<NewMemoryChunk> = chunks
            .iter()
            .map(|c| NewMemoryChunk {
                memory_id: memory_id.to_string(),
                ordinal: c.ordinal,
                content: c.content.clone(),
                char_start: c.char_start,
                char_end: c.char_end,
                embedding: c.embedding.clone(),
                embedding_model: c.embedding_model.clone(),
                chunk_max_chars: c.chunk_max_chars,
                chunker_version: c.chunker_version,
                // Denormalized so scoped search filters inline instead of
                // joining back to the parent on every candidate.
                workspace_id: workspace_id.map(str::to_string),
            })
            .collect();

        self.repo
            .replace_chunks(memory_id, &rows)
            .await
            .map_err(|e| MemoryError::Persistence(e.to_string()))?;

        // Queue what still needs a vector, durably. A chunk written with a
        // seeded vector (single-chunk memories reuse the parent's) is already
        // complete and is not queued; everything else is work that must survive
        // a restart, because the process that would have done it may not.
        for c in chunks {
            match (&c.embedding, &c.embedding_model) {
                (Some(v), Some(model)) if !v.is_empty() && !model.is_empty() => {
                    if let Err(e) = self
                        .repo
                        .put_chunk_vector(memory_id, c.ordinal, model, v)
                        .await
                    {
                        warn!("Could not persist chunk bucket for {memory_id}#{}: {e}", c.ordinal);
                    }
                }
                _ => {
                    if let Some(model) = active_model
                        && let Err(e) = self
                            .repo
                            .enqueue_embedding(memory_id, c.ordinal, model)
                            .await
                    {
                        warn!("Could not queue chunk {memory_id}#{} : {e}", c.ordinal);
                    }
                }
            }
        }
        Ok(())
    }

    async fn chunks_needing_embedding(
        &self,
        model: &str,
        limit: usize,
        seed: bool,
    ) -> Result<Vec<PendingChunk>, MemoryError> {
        self.repo
            .chunks_needing_embedding(model, limit, seed)
            .await
            .map(|rows| {
                rows.into_iter()
                    .map(|c| PendingChunk {
                        chunk_id: c.id,
                        memory_id: c.memory_id,
                        ordinal: c.ordinal,
                        content: c.content,
                    })
                    .collect()
            })
            .map_err(|e| MemoryError::Persistence(e.to_string()))
    }

    async fn set_chunk_embedding(
        &self,
        chunk_id: i64,
        embedding: &[f32],
        model: &str,
    ) -> Result<(), MemoryError> {
        self.repo
            .set_chunk_embedding(chunk_id, embedding, model)
            .await
            .map(|_| ())
            .map_err(|e| MemoryError::Persistence(e.to_string()))
    }

    async fn clear_memory_buckets(&self, memory_id: &str) -> Result<(), MemoryError> {
        self.repo
            .clear_memory_buckets(memory_id)
            .await
            .map_err(|e| MemoryError::Persistence(e.to_string()))
    }

    async fn restore_chunk_vectors(&self, model: &str) -> Result<usize, MemoryError> {
        self.repo
            .restore_chunk_vectors(model)
            .await
            .map_err(|e| MemoryError::Persistence(e.to_string()))
    }

    async fn dequeue_embedding(
        &self,
        memory_id: &str,
        ordinal: i64,
        model: &str,
    ) -> Result<(), MemoryError> {
        self.repo
            .dequeue_embedding(memory_id, ordinal, model)
            .await
            .map_err(|e| MemoryError::Persistence(e.to_string()))
    }

    async fn record_embedding_failure(
        &self,
        memory_id: &str,
        ordinal: i64,
        model: &str,
        error: &str,
    ) -> Result<(), MemoryError> {
        self.repo
            .record_embedding_failure(memory_id, ordinal, model, error)
            .await
            .map_err(|e| MemoryError::Persistence(e.to_string()))
    }

    async fn embedding_queue_health(&self) -> Result<Vec<(String, usize, Option<String>)>, MemoryError> {
        self.repo
            .embedding_queue_health()
            .await
            .map_err(|e| MemoryError::Persistence(e.to_string()))
    }

    async fn search_chunks(
        &self,
        query: &[f32],
        model: &str,
        limit: usize,
        workspace_id: Option<&str>,
    ) -> Result<Vec<(String, i64, f32)>, MemoryError> {
        if query.is_empty() || model.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        self.repo
            .search_chunks_by_embedding_sql(query, model, limit, workspace_id)
            .await
            .map(|rows| {
                rows.into_iter()
                    // `vector_distance_cos` returns a DISTANCE; every threshold
                    // downstream is stated as a similarity. Converting here,
                    // once, at the boundary, is the only place that cannot be
                    // forgotten by a later caller.
                    .map(|(memory_id, ordinal, distance)| {
                        (memory_id, ordinal, 1.0 - crate::numeric::f32_from_f64(distance))
                    })
                    .collect()
            })
            .map_err(|e| MemoryError::Persistence(e.to_string()))
    }

    async fn parents_without_chunks(&self, limit: usize) -> Result<Vec<String>, MemoryError> {
        self.repo
            .parents_without_chunks(limit)
            .await
            .map_err(|e| MemoryError::Persistence(e.to_string()))
    }

    async fn remove_entry(&self, id: &str) -> Result<(), MemoryError> {
        self.repo
            .delete(id)
            .await
            .map(|_| ())
            .map_err(|e| MemoryError::Persistence(e.to_string()))
    }

    /// Batch removal via `MemoryRepository::bulk_delete`, which overwrites every
    /// row's sensitive columns and then checkpoint-truncates the WAL **once** for
    /// the whole set — one fsync per dream cycle's cluster instead of one per
    /// consolidated source memory.
    async fn remove_entries(&self, ids: &[&str]) -> Result<(), MemoryError> {
        self.repo
            .bulk_delete(ids)
            .await
            .map(|_| ())
            .map_err(|e| MemoryError::Persistence(e.to_string()))
    }

    async fn update_entry_fsrs(&self, id: &str, fsrs: &FsrsState) -> Result<(), MemoryError> {
        self.repo
            .update_fsrs(
                id,
                &MemoryFsrsUpdate {
                    stability: fsrs.stability,
                    difficulty: fsrs.difficulty,
                    last_access: fsrs.last_access,
                    access_count: i64::from(fsrs.access_count),
                    importance: fsrs.importance,
                    storage_strength: fsrs.storage_strength,
                    generation: i64::from(fsrs.generation),
                },
            )
            .await
            .map(|_| ())
            .map_err(|e| MemoryError::Persistence(e.to_string()))
    }

    async fn update_entry_content(&self, id: &str, content: &str) -> Result<(), MemoryError> {
        self.repo
            .update_content(id, content)
            .await
            .map(|_| ())
            .map_err(|e| MemoryError::Persistence(e.to_string()))
    }

    async fn load_all(&self) -> Result<Vec<MemoryEntry>, MemoryError> {
        let db_memories = self
            .repo
            .bulk_load()
            .await
            .map_err(|e| MemoryError::Persistence(e.to_string()))?;

        let (mut entries, awaiting_embedding) = Self::convert_rows(db_memories);
        self.attach_all_buckets(&mut entries).await;
        if awaiting_embedding > 0 {
            info!(
                "Loaded {awaiting_embedding} memories with no vector yet; they are queued for \
                 backfill and unsearchable until it runs — their content is intact"
            );
        }

        Ok(entries)
    }

    /// Load with a corruption report.
    ///
    /// Fast path first: the single-scan `bulk_load`. Only when it fails with a
    /// **corruption** error (`is_corruption_error`) do we pay for the row-by-row
    /// `bulk_load_salvage` (one `SELECT id` + one point lookup per row) — so a
    /// healthy store of N memories stays a single scan, not N+1 queries. A
    /// non-corruption failure propagates unchanged. `corrupt_rows` counts genuine
    /// unreadable rows (data loss); benign no-embedding rows are just skipped.
    async fn load_all_report(&self) -> Result<LoadReport, MemoryError> {
        match self.repo.bulk_load().await {
            Ok(db_memories) => {
                let expected = db_memories.len();
                let (mut entries, awaiting_embedding) = Self::convert_rows(db_memories);
                self.attach_all_buckets(&mut entries).await;
                if awaiting_embedding > 0 {
                    info!(
                        "Loaded {awaiting_embedding} memories with no vector yet; they are queued \
                         for backfill and unsearchable until it runs — their content is intact"
                    );
                }
                Ok(LoadReport { entries, corrupt_rows: 0, expected })
            }
            Err(e) if nanna_storage::is_corruption_error(&e) => {
                warn!("Bulk memory load hit corruption ({e}); falling back to row-by-row salvage");
                let report = self
                    .repo
                    .bulk_load_salvage()
                    .await
                    .map_err(|se| MemoryError::Persistence(se.to_string()))?;
                let expected = report.expected;
                let corrupt_rows = report.corrupt_ids.len();
                let (mut entries, awaiting_embedding) = Self::convert_rows(report.memories);
                self.attach_all_buckets(&mut entries).await;
                if awaiting_embedding > 0 {
                    info!(
                        "Salvaged {awaiting_embedding} memories with no vector yet; they are \
                         queued for backfill, not lost"
                    );
                }
                warn!(
                    "Salvage load recovered {} of {} rows; {} corrupt rows skipped",
                    entries.len(),
                    expected,
                    corrupt_rows
                );
                Ok(LoadReport { entries, corrupt_rows, expected })
            }
            Err(e) => Err(MemoryError::Persistence(e.to_string())),
        }
    }
}

impl TursoMemoryPersistence {
    /// Convert stored `Memory` rows to `MemoryEntry`.
    ///
    /// Returns every row — none are dropped — plus a count of how many arrived
    /// without a vector, so the caller can say so. Those are backfill work, not
    /// losses, and the distinction is the whole point: the previous version
    /// discarded them and reported the discard as a skip.
    fn convert_rows(rows: Vec<nanna_storage::Memory>) -> (Vec<MemoryEntry>, usize) {
        let mut entries = Vec::with_capacity(rows.len());
        let mut awaiting_embedding = 0usize;
        for mem in rows {
            let entry = db_memory_to_entry(mem);
            if entry.embedding.is_empty() {
                awaiting_embedding += 1;
            }
            entries.push(entry);
        }
        (entries, awaiting_embedding)
    }

    /// Attach persisted buckets to already-converted entries.
    ///
    /// ONE scan of `memory_vectors`, grouped in memory, rather than a lookup
    /// per entry: the loader needs every bucket for every row, and N+1 point
    /// queries over a few thousand memories is a startup stall on the path
    /// that already has to finish before the GUI's ready timeout.
    async fn attach_all_buckets(&self, entries: &mut [MemoryEntry]) {
        let all = match self.repo.all_memory_vectors().await {
            Ok(all) => all,
            Err(e) => {
                warn!(
                    "Could not load embedding buckets ({e}); memories keep only their active \
                     vector and a provider switch will re-embed instead of reusing"
                );
                return;
            }
        };
        if all.is_empty() {
            return;
        }
        let mut by_id: HashMap<String, Vec<(String, Vec<f32>)>> = HashMap::new();
        for (id, model, vector) in all {
            by_id.entry(id).or_default().push((model, vector));
        }
        let mut restored = 0usize;
        for entry in entries.iter_mut() {
            if let Some(buckets) = by_id.remove(&entry.id) {
                restored += buckets.len();
                attach_buckets(entry, buckets);
            }
        }
        if restored > 0 {
            info!("Restored {restored} embedding buckets across {} memories", entries.len());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stored_row(memory_id: &str, embedding: Option<Vec<f32>>, model: Option<&str>) -> nanna_storage::Memory {
        nanna_storage::Memory {
            id: 1,
            memory_id: memory_id.to_string(),
            content: "the harbor light was green".to_string(),
            embedding,
            embedding_model: model.map(str::to_string),
            session_id: None,
            created_at: "2026-08-04 00:00:00".to_string(),
            updated_at: "2026-08-04 00:00:00".to_string(),
            metadata: None,
            tags: Vec::new(),
            workspace_id: None,
            fsrs_stability: 1.0,
            fsrs_difficulty: 5.0,
            fsrs_last_access: 0,
            fsrs_access_count: 0,
            fsrs_importance: 3.0,
            fsrs_storage_strength: 1.0,
            fsrs_generation: 0,
        }
    }

    /// A memory written while the embedding provider was down persists its
    /// content fine, and used to vanish on the next restart: the loader
    /// returned `None` for it, so it was never in the store, never searchable,
    /// and never backfilled — because the backfill queue is built from loaded
    /// entries. The row stayed in the database, unreachable, forever.
    #[test]
    fn a_row_awaiting_its_vector_survives_the_load() {
        let entry = db_memory_to_entry(stored_row("m1", None, None));
        assert_eq!(entry.id, "m1");
        assert_eq!(entry.content, "the harbor light was green");
        assert!(entry.embedding.is_empty(), "no vector yet, by definition");
        assert!(
            entry.embeddings.is_empty(),
            "an empty vector must not be bucketed under any model — that would \
             claim the model had embedded it and drop it out of the queue"
        );
    }

    /// The `None` case must not be reachable by writing an empty vector either:
    /// a row can carry `Some(vec![])` from an older write path, and bucketing
    /// that under its recorded model name would be the same lie.
    #[test]
    fn an_empty_stored_vector_is_not_bucketed_under_its_model() {
        let entry = db_memory_to_entry(stored_row("m2", Some(Vec::new()), Some("ollama:nomic")));
        assert_eq!(entry.embedding, Vec::<f32>::new());
        assert!(
            entry.embeddings.is_empty(),
            "an empty vector claims no model, whatever the column says"
        );
    }

    #[test]
    fn a_row_with_a_vector_round_trips_into_its_model_bucket() {
        let entry = db_memory_to_entry(stored_row("m3", Some(vec![0.1, 0.2]), Some("ollama:nomic")));
        assert_eq!(entry.embedding, vec![0.1, 0.2]);
        assert_eq!(entry.embeddings.get("ollama:nomic"), Some(&vec![0.1, 0.2]));
    }

    /// The count exists so the load can SAY what it deferred. Silence is what
    /// made the old behaviour survive: "skipped N" read as housekeeping.
    #[test]
    fn conversion_reports_how_many_await_a_vector_and_drops_none() {
        let rows = vec![
            stored_row("a", Some(vec![0.1, 0.2]), Some("ollama:nomic")),
            stored_row("b", None, None),
            stored_row("c", None, None),
        ];
        let (entries, awaiting) = TursoMemoryPersistence::convert_rows(rows);
        assert_eq!(entries.len(), 3, "every row must survive the conversion");
        assert_eq!(awaiting, 2);
    }
}
