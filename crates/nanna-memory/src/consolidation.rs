//! Memory Consolidation ("Dreaming")
//!
//! Periodic processing that mimics biological memory consolidation during sleep:
//! - Scores memories by a composite of similarity, recall frequency, importance, and age
//! - Clusters memories that should be merged together
//! - Compresses low-weight memories (summarization)
//! - Expands high-weight memories (enrichment)
//! - Merges related memories
//! - Respects a maximum compression ratio to avoid over-aggressive consolidation

use crate::lossy::LossyF32;
use crate::{MemoryEntry, FsrsParameters, FsrsState};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Upper bound on how many memories may be folded into a single consolidation
/// cluster (and therefore a single summarization prompt). A degenerate weight
/// band of thousands of mutually-similar memories would otherwise collapse into
/// one cluster whose prompt overflows a small local model's context window
/// (P12). Over-cap members stay unassigned and re-cluster among themselves, so
/// no content is dropped — the band is just consolidated in several passes.
pub const DEFAULT_MAX_CLUSTER_MEMORIES: usize = 64;

/// Context window (tokens) assumed when the summarizer model is unknown.
///
/// The safe low end (a small 8k-token local model). Callers that know the model
/// should size the budget to it via
/// [`ConsolidationConfig::with_summarizer_context_window`].
pub const FALLBACK_SUMMARIZER_CONTEXT_WINDOW_TOKENS: usize = 8_192;

/// Fallback per-cluster content-byte budget when the summarizer model is unknown.
///
/// Derived from [`FALLBACK_SUMMARIZER_CONTEXT_WINDOW_TOKENS`] via
/// [`cluster_content_bytes_for_context`] — not an "8 GB tier" constant — so
/// `default()` and the model-aware path share one formula.
pub const DEFAULT_MAX_CLUSTER_CONTENT_BYTES: usize =
    cluster_content_bytes_for_context(FALLBACK_SUMMARIZER_CONTEXT_WINDOW_TOKENS);

/// Convert a summarizer model's context window (in tokens) into the per-cluster
/// content-byte budget, so the consolidation prompt built from a cluster always
/// fits that model's window.
///
/// Reserves headroom for the summarization instruction/framing and the generated
/// summary, then converts the remaining token budget to bytes at the token
/// estimator's **worst-case density**: `nanna_llm::estimate_tokens` counts every
/// non-ASCII char as one token, and the smallest non-ASCII UTF-8 char is 2 bytes,
/// so content can reach at most 0.5 tokens/byte. Budgeting **2 bytes per token**
/// therefore guarantees the concatenated content never exceeds the token budget
/// for any script (ASCII simply under-fills — safety over utilization).
#[must_use]
pub const fn cluster_content_bytes_for_context(context_window_tokens: usize) -> usize {
    /// Instruction + framing + per-memory `---` separators.
    const PROMPT_OVERHEAD_TOKENS: usize = 512;
    /// Room for the generated consolidated summary.
    const OUTPUT_RESERVE_TOKENS: usize = 1024;
    /// Worst-case bytes-per-token that cannot overflow the token budget.
    const SAFE_BYTES_PER_TOKEN: usize = 2;
    /// Never collapse to nothing on a tiny/misreported window.
    const MIN_CONTENT_BYTES: usize = 2_048;

    let content_tokens =
        context_window_tokens.saturating_sub(PROMPT_OVERHEAD_TOKENS + OUTPUT_RESERVE_TOKENS);
    let bytes = content_tokens * SAFE_BYTES_PER_TOKEN;
    if bytes > MIN_CONTENT_BYTES {
        bytes
    } else {
        MIN_CONTENT_BYTES
    }
}

const fn default_max_cluster_memories() -> usize {
    DEFAULT_MAX_CLUSTER_MEMORIES
}

const fn default_max_cluster_content_bytes() -> usize {
    DEFAULT_MAX_CLUSTER_CONTENT_BYTES
}

/// Consolidation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationConfig {
    /// Minimum composite score to consider two memories for the same cluster (0.0-1.0).
    /// This blends similarity, recall overlap, importance, and age — not just cosine similarity.
    pub cluster_threshold: f32,
    /// Minimum cluster size to consolidate (singletons below this are skipped or expanded)
    pub min_cluster_size: usize,
    /// Maximum number of memories a single cluster may hold, bounding the size of
    /// the consolidation prompt handed to the (possibly small, local) summarizer.
    #[serde(default = "default_max_cluster_memories")]
    pub max_cluster_memories: usize,
    /// Maximum total `content` bytes a single cluster may hold, bounding the
    /// consolidation prompt to fit the summarizer model's context window. Defaults
    /// to the small-model fallback; size it to the real model with
    /// [`ConsolidationConfig::with_summarizer_context_window`].
    #[serde(default = "default_max_cluster_content_bytes")]
    pub max_cluster_content_bytes: usize,
    /// Maximum fraction of total memories that can be removed in a single run (0.0-1.0).
    /// E.g. 0.50 means at most 50% of memories can be merged away.
    pub max_compression_ratio: f32,
    /// Minimum number of memories to leave after consolidation (floor).
    /// Consolidation stops merging once the store would drop below this count.
    pub min_remaining_memories: usize,
    /// Weight thresholds for compression levels
    pub weight_thresholds: WeightThresholds,
    /// Weights for the composite clustering score
    pub clustering_weights: ClusteringWeights,
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        Self {
            // 0.75, and both digits are measured — see
            // [`ClusteringWeights::non_similarity_floor`],
            // [`ConsolidationConfig::validate`] and the
            // `clustering_threshold_sweep` bench.
            //
            // The old 0.45 was below the shipped weights' non-semantic floor
            // (0.50), so a pair cleared this bar on non-semantic terms ALONE:
            // semantic dissimilarity could not veto a merge and two orthogonal
            // memories clustered. Anything above 0.50 restores the veto.
            //
            // 0.75 rather than the bare minimum because the sweep says the
            // range is free: 0.55, 0.65 and 0.75 produce *identical* outcomes
            // on the loose corpus (3 clusters, 27 merged, compression 0.450,
            // recall 1.000) while the cosine each actually demands rises
            // 0.10 → 0.30 → 0.50. Taking the top of the flat range buys a 5×
            // stricter semantic bar at zero measured cost. Above it the trade
            // becomes real: 0.85 (cosine 0.70, the bar the 2026 literature
            // uses) drops compression to 0.383 — still at recall 1.000 — and
            // 0.95 stops clustering entirely.
            cluster_threshold: 0.75,
            min_cluster_size: 2,
            max_cluster_memories: DEFAULT_MAX_CLUSTER_MEMORIES,
            max_cluster_content_bytes: DEFAULT_MAX_CLUSTER_CONTENT_BYTES,
            max_compression_ratio: 0.50,
            min_remaining_memories: 20,
            weight_thresholds: WeightThresholds::default(),
            clustering_weights: ClusteringWeights::default(),
        }
    }
}

impl ConsolidationConfig {
    /// Reject a configuration in which cosine similarity cannot veto a merge.
    ///
    /// The invariant: **the non-semantic terms must not, on their own, clear
    /// [`Self::cluster_threshold`]** — that is,
    /// [`ClusteringWeights::non_similarity_floor`] `< cluster_threshold`.
    /// Otherwise two memories that are equally unimportant, equally
    /// never-recalled and written near each other cluster regardless of what
    /// they are *about*, and a dream cycle summarizes unrelated memories into
    /// one gist.
    ///
    /// This is a relationship between four weights and a threshold, which is
    /// exactly the kind of thing that goes stale when either side is tuned
    /// alone — it was violated by the shipped defaults from the day the weights
    /// were last retuned until 2026-09-09. Hence a check rather than a comment.
    ///
    /// # Errors
    /// Returns a sentence naming the floor, the threshold, and the two ways to
    /// fix it.
    pub fn validate(&self) -> Result<(), String> {
        let floor = self.clustering_weights.non_similarity_floor();
        if floor >= self.cluster_threshold {
            return Err(format!(
                "clustering config lets non-semantic terms alone form a cluster: the \
                 recall/importance/age floor is {floor:.3} but cluster_threshold is \
                 {:.3}, so a pair with ZERO cosine similarity still clusters and \
                 semantic dissimilarity cannot veto a merge. Raise cluster_threshold \
                 above {floor:.3}, or lower recall_affinity/importance_proximity/\
                 age_proximity relative to similarity.",
                self.cluster_threshold
            ));
        }
        Ok(())
    }

    /// The minimum cosine similarity this configuration really demands — the
    /// honest reading of [`Self::cluster_threshold`]. See
    /// [`ClusteringWeights::min_required_similarity`].
    #[must_use]
    pub fn min_required_similarity(&self) -> f32 {
        self.clustering_weights
            .min_required_similarity(self.cluster_threshold)
    }

    /// Size [`Self::max_cluster_content_bytes`] to the summarizer model's context
    /// window (in tokens) so an automatically-built consolidation prompt always
    /// fits it — replacing the small-model fallback with the real budget. Leaves
    /// the member-count cap ([`Self::max_cluster_memories`], a coarse safety
    /// limit) untouched, since the byte budget is what maps to the context window.
    #[must_use]
    pub const fn with_summarizer_context_window(mut self, context_window_tokens: usize) -> Self {
        self.max_cluster_content_bytes = cluster_content_bytes_for_context(context_window_tokens);
        self
    }
}

/// Weights for the composite clustering score.
/// The final score is a weighted combination of these signals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringWeights {
    /// How much cosine similarity matters (semantic relatedness)
    pub similarity: f32,
    /// How much overlapping recall frequency matters (memories accessed together)
    pub recall_affinity: f32,
    /// How much similar importance levels matter (merge peers, not outliers)
    pub importance_proximity: f32,
    /// How much similar age matters (merge memories from the same era)
    pub age_proximity: f32,
    /// The timescale the age term is measured against, in minutes: how long the
    /// store being consolidated has been accumulating.
    ///
    /// "Close in time" is only meaningful relative to how much time there is. An
    /// hour apart is most of the history of a workspace three hours old and a
    /// rounding error in one six months old, so a fixed half-life is wrong for
    /// every store except the one it was tuned on — the previous 30-day constant
    /// scored every pair inside a working session at ~0.99, which is the same as
    /// having no age term at all.
    ///
    /// Set per run from the observed span of the memories themselves. The
    /// default is a placeholder for callers that never set it.
    #[serde(default = "default_time_span_minutes")]
    pub time_span_minutes: f32,
}

/// One day, for callers that never set an observed span.
const fn default_time_span_minutes() -> f32 {
    1440.0
}

impl ClusteringWeights {
    /// The sum of every weight — the divisor [`composite_cluster_score`] uses.
    #[must_use]
    pub fn total(&self) -> f32 {
        self.similarity + self.recall_affinity + self.importance_proximity + self.age_proximity
    }

    /// The composite score a pair receives with **zero semantic support**.
    ///
    /// This is not a corner case; it is the commonest pair in a real store.
    /// `recall_affinity` and `importance_proximity` are 1.0 *by construction*
    /// whenever the two values are equal — including `0 == 0`, two memories
    /// neither of which has ever been recalled — and `age_prox` is ~1.0 for any
    /// two memories written inside one session. So for two fresh, equally
    /// unimportant, contemporaneous memories the score is pinned at this floor
    /// no matter what their embeddings say.
    ///
    /// If this reaches [`ConsolidationConfig::cluster_threshold`], cosine
    /// similarity has **no veto** and the clusterer groups by "written around
    /// the same time and equally unremarkable". Measured against the weights
    /// shipped before 2026-09-09 (floor 0.50, threshold 0.45): two orthogonal
    /// unit vectors scored 0.500, two anti-correlated ones also scored 0.500,
    /// and four mutually unrelated memories formed a single cluster of four.
    #[must_use]
    pub fn non_similarity_floor(&self) -> f32 {
        let total = self.total();
        if total <= 0.0 {
            // `composite_cluster_score` falls back to pure similarity here, so
            // there is no floor at all.
            return 0.0;
        }
        (self.recall_affinity + self.importance_proximity + self.age_proximity) / total
    }

    /// The minimum cosine similarity a pair actually needs to reach `threshold`
    /// when every non-semantic term is maximal — i.e. the real semantic bar
    /// this configuration enforces, as opposed to the one `threshold` looks
    /// like it enforces.
    ///
    /// Returns 0.0 when the floor already clears the threshold (the broken
    /// case: no similarity is required at all).
    #[must_use]
    pub fn min_required_similarity(&self, threshold: f32) -> f32 {
        let total = self.total();
        if total <= 0.0 || self.similarity <= 0.0 {
            return threshold.max(0.0);
        }
        let deficit = threshold - self.non_similarity_floor();
        (deficit * total / self.similarity).max(0.0)
    }
}

impl Default for ClusteringWeights {
    fn default() -> Self {
        // Similarity and time carry 80% between them: those are the two axes a
        // dream actually groups along — what is this about, and when did it
        // happen. Access counts and importance are real signals but secondary,
        // and at 0.20 each they were able to out-vote a weak topical match and
        // pull unrelated memories into the same cluster simply because both had
        // never been recalled.
        Self {
            similarity: 0.50,
            recall_affinity: 0.10,
            importance_proximity: 0.10,
            age_proximity: 0.30,
            time_span_minutes: default_time_span_minutes(),
        }
    }
}

/// Weight thresholds that determine compression level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightThresholds {
    /// Below this: compress to essence (1 word/sentence)
    pub essence: f32,
    /// Below this: moderate compression
    pub compressed: f32,
    /// Below this: standard detail
    pub standard: f32,
    /// Below this: full detail (no compression)
    pub detailed: f32,
    // Above detailed threshold: expand/research
}

impl Default for WeightThresholds {
    fn default() -> Self {
        Self {
            essence: 0.2,
            compressed: 0.5,
            standard: 0.8,
            detailed: 1.0,
        }
    }
}

/// Compression level for summarization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionLevel {
    /// Compress to absolute essence (1 word to 1 sentence)
    Essence,
    /// Moderate compression (key points only)
    Compressed,
    /// Standard detail (minor trimming)
    Standard,
    /// Keep full detail
    Detailed,
    /// Expand and enrich with more context
    Expand,
}

impl CompressionLevel {
    /// Determine compression level from weight
    #[must_use]
    pub fn from_weight(weight: f32, thresholds: &WeightThresholds) -> Self {
        if weight < thresholds.essence {
            Self::Essence
        } else if weight < thresholds.compressed {
            Self::Compressed
        } else if weight < thresholds.standard {
            Self::Standard
        } else if weight <= thresholds.detailed {
            Self::Detailed
        } else {
            Self::Expand
        }
    }

    /// Get the consolidation prompt for this compression level.
    ///
    /// Compression IS the goal — the result must be materially smaller than
    /// the material it replaces, or the store never stops growing. What
    /// changes here is HOW: not detail-stripping ("remove examples, keep the
    /// main point"), which leaves a thinner transcript, but generalisation —
    /// finding the throughline across fragments and stating the general shape
    /// once instead of the instances many times. That is what makes the
    /// output both shorter AND worth more than its inputs.
    ///
    /// The one hard rule is honesty about provenance: interpretation is
    /// welcome, invented facts are not.
    #[must_use]
    pub const fn summarization_prompt(&self) -> &'static str {
        match self {
            Self::Essence => {
                "Distil these related memories to the single idea underneath them. Not a \
                label — the insight. One short line that someone who lived through all of \
                them would nod at."
            }
            Self::Compressed => {
                "These memories are related. Write the short version of what was going on \
                and what it added up to, in 2-3 sentences. Generalise across them rather \
                than listing them: name the pattern, not every instance. Keep only the \
                details that would change a future decision."
            }
            Self::Standard => {
                "These memories belong together. Replace them with one shorter account that \
                says what was going on and what it amounts to. Generalise: where several \
                fragments are instances of the same pattern, state the pattern ONCE instead \
                of repeating the instances — that is where the compression comes from. Keep \
                the specifics that would change a future decision (names, paths, numbers, \
                outcomes) and drop the ones that would not. Flag anything here that \
                contradicts something else; a contradiction noticed is worth more than a \
                smooth story. Interpret freely, never invent, and say when you are inferring. \
                Your output MUST be substantially shorter than the material you were given."
            }
            Self::Detailed => {
                "Keep this memory as-is. No compression needed."
            }
            Self::Expand => {
                "This material matters, so spend words where they earn their keep: make the \
                connections explicit — what caused what, what it resembles, what it predicts, \
                what is still unresolved. Even here the result should be no longer than the \
                material it replaces: buy the space by generalising the repetitive parts. \
                Mark speculation as speculation. Never invent facts."
            }
        }
    }
}

/// A cluster of related memories to be consolidated
#[derive(Debug, Clone)]
pub struct MemoryCluster {
    /// Memories in this cluster
    pub memories: Vec<MemoryEntry>,
    /// Average embedding (centroid)
    pub centroid: Vec<f32>,
    /// Compression level for this cluster
    pub compression_level: CompressionLevel,
    /// Average weight of memories in cluster
    pub avg_weight: f32,
}

impl MemoryCluster {
    /// Create a new cluster from memories
    #[must_use]
    pub fn new(memories: Vec<MemoryEntry>, compression_level: CompressionLevel, fsrs_params: &FsrsParameters) -> Self {
        let centroid = Self::compute_centroid(&memories);
        let avg_weight = memories.iter()
            .map(|m| m.fsrs.weight(fsrs_params))
            .sum::<f32>() / memories.len().max(1).lossy_f32();
        
        Self {
            memories,
            centroid,
            compression_level,
            avg_weight,
        }
    }

    /// Compute centroid (average) of embeddings
    fn compute_centroid(memories: &[MemoryEntry]) -> Vec<f32> {
        if memories.is_empty() {
            return Vec::new();
        }
        
        let dim = memories[0].embedding.len();
        let mut centroid = vec![0.0; dim];
        
        for memory in memories {
            for (i, &val) in memory.embedding.iter().enumerate() {
                if i < dim {
                    centroid[i] += val;
                }
            }
        }
        
        let count = memories.len().lossy_f32();
        for val in &mut centroid {
            *val /= count;
        }
        
        centroid
    }

    /// Get concatenated content for summarization
    #[must_use]
    pub fn concatenated_content(&self) -> String {
        self.memories
            .iter()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n\n---\n\n")
    }

    /// Build the prompt for LLM consolidation
    #[must_use]
    pub fn build_consolidation_prompt(&self) -> String {
        let instruction = self.compression_level.summarization_prompt();
        let content = self.concatenated_content();
        let count = self.memories.len();
        
        format!(
            "You are consolidating {count} memories into one.\n\n\
            Instruction: {instruction}\n\n\
            Memories to consolidate:\n\n{content}\n\n\
            Consolidated memory:"
        )
    }
}

/// Result of a consolidation run
#[derive(Debug, Clone, Default)]
pub struct ConsolidationResult {
    /// Number of memories processed
    pub memories_processed: usize,
    /// Number of clusters formed
    pub clusters_formed: usize,
    /// Number of memories merged (reduced by)
    pub memories_merged: usize,
    /// Number of near-duplicate memories folded away **without** an LLM call
    /// (dream phase (b) — see `MemoryService::fold_near_duplicates`). Counted
    /// separately from `memories_merged` so the token-free share of a cycle's
    /// compression is visible rather than hidden inside the summarized total.
    pub memories_deduped: usize,
    /// Number of memories expanded
    pub memories_expanded: usize,
    /// Errors encountered (non-fatal)
    pub errors: Vec<String>,
}

/// Compute a composite clustering score between two memories.
///
/// Blends semantic similarity, recall affinity, importance proximity, and age proximity.
/// Returns a value in [0, 1] where higher means more likely to cluster together.
#[must_use]
pub fn composite_cluster_score(
    a: &MemoryEntry,
    b: &MemoryEntry,
    weights: &ClusteringWeights,
) -> f32 {
    composite_cluster_score_with_norms(
        a,
        l2_norm(&a.embedding),
        b,
        l2_norm(&b.embedding),
        weights,
    )
}

/// [`composite_cluster_score`] when both embeddings' L2 norms are already known.
///
/// Exists for one caller — [`cluster_memories`]'s O(N²) inner loop — and for one
/// reason: the norms are **loop-invariant** and the cosine kernel recomputes
/// them on every pair. `nanna_simd::cosine_similarity_f32` accumulates three
/// FMAs per element (`a·b`, `a·a`, `b·b`), so two thirds of its arithmetic is
/// re-deriving magnitudes that do not change. Hoisting them out makes each pair
/// a single dot product against two cached scalars.
///
/// **The result is bit-identical, not merely close**, and that is a property of
/// how the two kernels are written rather than a tolerance: `dot_product_f32`
/// and `cosine_similarity_f32` walk the same 16-wide chunks in the same order
/// with the same `fmadd`, reduce the same way, and finish with the same scalar
/// remainder — so `dot_product_f32(a, b)` reproduces the cosine kernel's own
/// `dot_sum` exactly, and `dot_product_f32(a, a)` reproduces its `mag_a`.
/// Dividing by `norm_a * norm_b` is then the same final expression.
/// `norm_hoisting_is_bit_identical_to_the_cosine_kernel` pins that.
fn composite_cluster_score_with_norms(
    a: &MemoryEntry,
    norm_a: f32,
    b: &MemoryEntry,
    norm_b: f32,
    weights: &ClusteringWeights,
) -> f32 {
    // `f32::NEG_INFINITY` as the threshold makes the bound below unreachable,
    // so this always scores in full.
    score_or_prune(a, norm_a, b, norm_b, weights, f32::NEG_INFINITY)
        .unwrap_or(f32::NEG_INFINITY)
}

/// The composite score, or `None` when the pair **provably** cannot reach
/// `threshold` — decided without computing the cosine.
///
/// ## Why this exists
///
/// Suite 3b's follow-on measurement closed the question of whether the
/// clustering pass can be made faster by computing each pair more cheaply: it
/// cannot. Hoisting the norms removed two thirds of the kernel's arithmetic and
/// bought 13%, because the kernel is memory-bandwidth bound. The only remaining
/// lever is **comparing fewer pairs**, and this is the one that needs no index
/// and loses nothing.
///
/// ## The bound, and why it is exact rather than heuristic
///
/// The score is a weighted mean of four terms, and three of them are scalar
/// arithmetic on fields already in cache while the fourth streams two whole
/// embeddings. Each of the three is bounded above by 1.0 *by construction*
/// (`recall_affinity` and `importance_proximity` are `1 - ratio` forms,
/// `age_prox` is explicitly clamped), and so is `sim` (cosine, floored at 0).
/// So the largest score this pair could possibly achieve, whatever its
/// embeddings say, is the mean taken with `sim = 1`:
///
/// ```text
/// ceiling = (w_sim*1 + w_recall*ra + w_imp*ip + w_age*ap) / total
/// ```
///
/// If `ceiling < threshold`, no cosine value can rescue the pair, so the
/// cosine is never computed. This is an **identity, not an approximation** —
/// the skipped pairs are exactly the pairs that would have been rejected — and
/// `pruning_never_changes_a_cluster` checks that clustering with and without it
/// produces identical clusters over a randomized corpus.
///
/// Two cases correctly do not prune: a non-positive total weight (the score
/// falls back to pure similarity, where no such bound holds) and a threshold of
/// `f32::NEG_INFINITY`, which is how the unconditional
/// [`composite_cluster_score_with_norms`] is expressed in terms of this.
///
/// How much it saves is **data-dependent and honestly modest** on a store
/// written in one sitting, where all three cheap terms sit near 1.0. It pays on
/// a long-lived store, where `age_prox` falls as pairs spread across the store's
/// lifetime — which is the regime the roadmap calls the realistic one. The
/// bench reports the skip count so the effect is visible rather than assumed.
fn score_or_prune(
    a: &MemoryEntry,
    norm_a: f32,
    b: &MemoryEntry,
    norm_b: f32,
    weights: &ClusteringWeights,
    threshold: f32,
) -> Option<f32> {
    let cheap = cheap_terms(a, b, weights);
    let total_weight = weights.total();

    if total_weight > 0.0 {
        // The best this pair could score with a perfect cosine. Computed from
        // `cheap` rather than by calling `cluster_score_ceiling`, which would
        // evaluate the same three terms a second time — but it is the same
        // expression, and `the_public_ceiling_matches_the_one_the_prune_uses`
        // holds the two together.
        let ceiling = (weights.similarity + cheap.weighted_sum) / total_weight;
        debug_assert!(
            ceiling >= -f32::EPSILON,
            "the ceiling is a mean of non-negative terms"
        );
        if ceiling < threshold {
            return None;
        }
    }

    // 1. Semantic similarity (cosine) — only now, once it can matter.
    let sim = cosine_with_norms(&a.embedding, norm_a, &b.embedding, norm_b).max(0.0);
    if total_weight <= 0.0 {
        return Some(sim); // fallback to pure similarity
    }
    let score = weights.similarity.mul_add(sim, cheap.weighted_sum) / total_weight;
    debug_assert!(
        score <= (weights.similarity + cheap.weighted_sum) / total_weight + f32::EPSILON,
        "a real score must not exceed the ceiling the prune trusted"
    );
    Some(score)
}

/// The best score a pair could possibly reach, whatever its embeddings say.
///
/// The upper bound [`score_or_prune`] decides on: the composite score with
/// `sim` at its maximum of 1.0. Public so a benchmark can count how many pairs
/// the prune rejects without restating the formula — a second copy of this
/// arithmetic in a bench would be free to drift from the one that ships.
///
/// Returns 1.0 when the weights sum to zero, where the score falls back to
/// pure similarity and no bound of this shape holds.
#[must_use]
pub fn cluster_score_ceiling(a: &MemoryEntry, b: &MemoryEntry, weights: &ClusteringWeights) -> f32 {
    let total = weights.total();
    if total <= 0.0 {
        return 1.0;
    }
    (weights.similarity + cheap_terms(a, b, weights).weighted_sum) / total
}

/// The three non-semantic terms, already multiplied by their weights.
struct CheapTerms {
    weighted_sum: f32,
}

/// Everything in the composite score that does not touch an embedding.
///
/// Split out so the ceiling and the final score are computed from **one**
/// evaluation of these terms rather than two — paying for them twice would hand
/// back part of what the prune saves on every pair it does not skip.
fn cheap_terms(a: &MemoryEntry, b: &MemoryEntry, weights: &ClusteringWeights) -> CheapTerms {

    // 2. Recall affinity: memories accessed a similar number of times are "peers"
    //    and memories both accessed recently are likely contextually related.
    let recall_a = a.fsrs.access_count.lossy_f32();
    let recall_b = b.fsrs.access_count.lossy_f32();
    let max_recall = recall_a.max(recall_b).max(1.0);
    // Equal access counts are peers (affinity 1), including two never-accessed
    // memories — `min/max` wrongly scored (0,0) as 0. Divergence lowers affinity.
    let recall_affinity = 1.0 - (recall_a - recall_b).abs() / max_recall; // 0..1

    // 3. Importance proximity: prefer merging memories of similar importance.
    //    Two low-importance memories should merge; a high+low pair should not.
    let imp_diff = (a.fsrs.importance - b.fsrs.importance).abs();
    let importance_prox = (1.0 - imp_diff / 5.0).max(0.0); // normalize: max diff ~5

    // 4. Age proximity, measured against how long this store has existed.
    //
    //    This was `exp(-days / 30)` — a fixed 30-day half-life — so every pair
    //    inside one working session scored ~0.99 and the age term contributed
    //    nothing to the decision. But no fixed timescale can be right: an hour
    //    apart is most of the history of a workspace three hours old, and a
    //    rounding error in one six months old. The same absolute gap has to mean
    //    different things in the two stores.
    //
    //    So the gap is expressed as a FRACTION of the store's own lifetime.
    //    Simultaneous memories score 1.0, memories at opposite ends of the
    //    store's history score 0.0, and everything in between is linear — which
    //    makes a young workspace discriminate at minute scale and a mature one
    //    at week scale, with no tuning constant to go stale.
    let age_diff_minutes = (a.timestamp - b.timestamp).unsigned_abs().lossy_f32() / 60.0;
    // A store with no span yet (one memory, or all written in the same second)
    // has no time axis to judge on — treat everything as contemporaneous rather
    // than dividing by zero.
    let span = weights.time_span_minutes.max(1.0);
    let age_prox = (1.0 - age_diff_minutes / span).clamp(0.0, 1.0);

    debug_assert!(
        (0.0..=1.0).contains(&recall_affinity)
            && (0.0..=1.0).contains(&importance_prox)
            && (0.0..=1.0).contains(&age_prox),
        "every cheap term must be in 0..=1, or the prune's ceiling is not an upper bound"
    );

    CheapTerms {
        weighted_sum: weights.age_proximity.mul_add(
            age_prox,
            weights.importance_proximity.mul_add(
                importance_prox,
                weights.recall_affinity * recall_affinity,
            ),
        ),
    }
}

/// Two memories may only ever consolidate together when they share the exact
/// same scope. A memory's `workspace_id` is `None` for a global memory and
/// `Some(id)` for a workspace-scoped one; recall returns "global + this
/// workspace" (see [`crate::MemoryService::recall`]), so both a workspace pair
/// and a global pair are legitimately co-visible and mergeable — but a
/// cross-workspace pair, or a global↔workspace pair, is not. Merging across a
/// scope boundary would either **leak** workspace-private content into another
/// scope or **lose** a memory's scope by re-homing it, silently defeating the
/// per-workspace memory scoping the `remember`/`recall` tools enforce. Exact
/// `Option` equality (`None == None`, `Some(a) == Some(a)`) is the safe rule.
#[must_use]
pub fn same_scope(a: &MemoryEntry, b: &MemoryEntry) -> bool {
    a.workspace_id == b.workspace_id
}

/// The metadata key carrying a memory's provenance.
///
/// Written at extraction time from `nanna_agent::MemoryProvenance::as_str` —
/// `"stated"` when the user asserted the fact, `"observed"` when the agent
/// inferred it.
pub const FACT_TYPE_METADATA_KEY: &str = "fact_type";

/// The one `fact_type` value that pins a memory verbatim.
const STATED_FACT_TYPE: &str = "stated";

/// Is the memory carrying this metadata pinned verbatim — i.e. must a dream
/// cycle never paraphrase it?
///
/// Repeated summarization erases low-frequency detail: the clause that appears
/// once in a band is exactly the one a gist drops. `crate::retention`'s drift
/// fixture reproduces that against this pipeline — a rare safety-critical clause
/// is gone after a single summarizing cycle while the topic stays perfectly
/// recallable, so recall metrics cannot see the loss.
///
/// A fact the **user asserted** is the case where that loss is least acceptable
/// and least excusable: it was not inferred from anything, it was said, and it
/// is stored in the words it was said in. So provenance is the gate, not a tuned
/// importance threshold — a categorical rule with no magic number in it.
///
/// Conservative by construction, mirroring `MemoryProvenance`'s own default: a
/// missing, empty or unrecognised `fact_type` is **not** pinned. Absence of a
/// provenance signal is not evidence that the user stated something, so an
/// unlabeled memory can never pin itself out of consolidation.
/// Takes the metadata map rather than a whole entry so the one policy answers
/// for every shape a memory is read back in (`MemoryEntry` while consolidating,
/// `MemoryListEntry` when listing), with no second copy to drift.
#[must_use]
pub fn is_verbatim_pinned<S: std::hash::BuildHasher>(
    metadata: &HashMap<String, String, S>,
) -> bool {
    metadata
        .get(FACT_TYPE_METADATA_KEY)
        .is_some_and(|fact_type| fact_type.trim().eq_ignore_ascii_case(STATED_FACT_TYPE))
}

/// Cluster memories using the composite score (not just cosine similarity).
///
/// Uses greedy clustering: for each unassigned memory, find all unassigned memories
/// whose composite score exceeds the threshold and group them.
///
/// A cluster is always **scope-homogeneous**: only memories sharing the seed's
/// `workspace_id` can join it (see `same_scope`), so a dream cycle can never
/// merge across a workspace boundary or fold a global and a workspace memory
/// into one entry.
///
/// Each cluster is bounded by `config.max_cluster_memories` (member count) and
/// `config.max_cluster_content_bytes` (total content) so the consolidation prompt
/// built from it can never overflow a small local model's context window. A
/// candidate that would breach either bound is left unassigned and re-clustered
/// on a later seed — nothing is dropped, the band just consolidates in more passes.
#[must_use]
pub fn cluster_memories(
    memories: &[MemoryEntry],
    config: &ConsolidationConfig,
) -> Vec<Vec<MemoryEntry>> {
    if memories.is_empty() {
        return Vec::new();
    }
    // A cap of 0 would make every cluster a skipped singleton — a config bug.
    debug_assert!(config.max_cluster_memories >= 1, "max_cluster_memories must be >= 1");

    let cap_count = config.max_cluster_memories.max(1);
    let cap_bytes = config.max_cluster_content_bytes;

    // Each memory's L2 norm, computed once. The greedy pass below is O(N²) in
    // pair comparisons and the cosine kernel re-derives both magnitudes on
    // every one of them, so this is N vector passes bought to remove up to
    // N(N-1)/2 of them. Scores are unchanged bit for bit — see
    // `composite_cluster_score_with_norms`.
    let norms: Vec<f32> = memories.iter().map(|m| l2_norm(&m.embedding)).collect();
    debug_assert_eq!(norms.len(), memories.len(), "one norm per memory");

    let mut clusters: Vec<Vec<MemoryEntry>> = Vec::new();
    let mut assigned = vec![false; memories.len()];

    for i in 0..memories.len() {
        if assigned[i] {
            continue;
        }

        // The seed is always admitted (a single over-sized memory forms a lone,
        // sub-`min_cluster_size` cluster that consolidation then skips).
        let mut cluster = vec![memories[i].clone()];
        let mut cluster_bytes = memories[i].content.len();
        assigned[i] = true;

        for j in (i + 1)..memories.len() {
            if cluster.len() >= cap_count {
                break; // count bound reached — remaining matches seed later clusters
            }
            if assigned[j] {
                continue;
            }
            // Hard scope barrier: never cluster across a workspace boundary
            // (cheaper than the composite score, so check it first).
            if !same_scope(&memories[i], &memories[j]) {
                continue;
            }

            // Prune before the cosine where the pair provably cannot reach
            // the threshold; `None` and "scored below" are the same verdict,
            // reached at very different cost.
            let Some(score) = score_or_prune(
                &memories[i],
                norms[i],
                &memories[j],
                norms[j],
                &config.clustering_weights,
                config.cluster_threshold,
            ) else {
                continue;
            };
            if score < config.cluster_threshold {
                continue;
            }
            // Byte bound: skip (don't consume) a candidate that would overflow the
            // prompt budget; a smaller later candidate may still fit.
            if cluster_bytes.saturating_add(memories[j].content.len()) > cap_bytes {
                continue;
            }

            cluster_bytes += memories[j].content.len();
            cluster.push(memories[j].clone());
            assigned[j] = true;
        }

        clusters.push(cluster);
    }

    // Postconditions: every cluster honors the count bound, and every multi-member
    // cluster honors the byte bound (a lone seed may exceed it by itself).
    debug_assert!(
        clusters.iter().all(|c| c.len() <= cap_count),
        "a cluster exceeded max_cluster_memories"
    );
    debug_assert!(
        clusters.iter().all(|c| {
            c.len() == 1 || c.iter().map(|m| m.content.len()).sum::<usize>() <= cap_bytes
        }),
        "a multi-member cluster exceeded max_cluster_content_bytes"
    );
    // Every cluster is scope-homogeneous: all members share one workspace_id, so
    // consolidating a cluster can never mix or re-home a memory's scope.
    debug_assert!(
        clusters
            .iter()
            .all(|c| c.iter().all(|m| m.workspace_id == c[0].workspace_id)),
        "a cluster mixed memories from different workspaces"
    );

    clusters
}

/// Cosine similarity between two vectors
/// L2 norm of an embedding, in the exact form the cosine kernel derives it.
///
/// `sqrt` of `dot_product_f32(v, v)`, which reproduces
/// `cosine_similarity_f32`'s internal `mag_a` bit for bit (same chunking, same
/// FMA order, same reduction, same scalar remainder). An empty vector has no
/// magnitude; 0.0 is the value that makes [`cosine_with_norms`] return this
/// module's "0.0 on empty" contract rather than a NaN.
fn l2_norm(v: &[f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    nanna_simd::dot_product_f32(v, v).sqrt()
}

/// [`cosine_similarity`] with both magnitudes supplied by the caller.
///
/// Same contract as [`cosine_similarity`] — 0.0 on a width mismatch, on an
/// empty vector, and on anything non-finite (a zero-magnitude embedding divides
/// by zero and must not leak an infinity or a NaN into the score).
fn cosine_with_norms(a: &[f32], norm_a: f32, b: &[f32], norm_b: f32) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let sim = nanna_simd::dot_product_f32(a, b) / (norm_a * norm_b);
    if sim.is_finite() { sim } else { 0.0 }
}

/// The pre-hoist cosine: one `cosine_similarity_f32` call per pair, magnitudes
/// and all.
///
/// **Test-only, and deliberately kept rather than deleted.** Production now
/// goes through [`cosine_with_norms`], which claims to reproduce this function
/// *bit for bit* — and a claim like that needs the thing it is compared
/// against to still exist. `#[cfg(test)]` states that status instead of
/// leaving a function that looks like a live code path and has no callers.
#[cfg(test)]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    // Guards preserve this clusterer's "0.0 on mismatch/empty" contract:
    // `nanna_simd::cosine_similarity_f32` *panics* on unequal lengths (memories
    // from different embedding-dimension eras can co-occur during migration)
    // and yields NaN for a zero-magnitude vector.
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    // Delegate to the workspace SIMD primitive (AVX-512/AVX2/NEON) — the same
    // one the vector-search path uses — instead of a private scalar loop, since
    // this runs O(N^2) times per band during a dream cycle.
    let sim = nanna_simd::cosine_similarity_f32(a, b);
    if sim.is_finite() { sim } else { 0.0 }
}

/// Metadata keys that locate ONE specific source record, and therefore can
/// never be true of a summary that replaced several.
///
/// `source_id` groups the chunks of one tool result and `chunk` is a position
/// (`"3/17"`) inside that group — together they are how `assemble_handle_content`
/// reassembles the whole text behind a memory handle. A consolidated entry that
/// inherited them would be pulled into that reassembly **at the inherited
/// position**, splicing a paraphrase into the middle of a result the model was
/// promised was stored verbatim. A summary has no position in anyone's byte
/// stream; there is no value of these keys it could truthfully carry.
const SOURCE_LOCATOR_METADATA_KEYS: [&str; 2] = ["source_id", "chunk"];

/// Metadata for a consolidated entry: only what is true of the whole cluster.
///
/// The merge used to be first-writer-wins across the cluster, which asserts one
/// arbitrary source's value about all of them — a gist of five different tools'
/// output claiming `tool=exec, outcome=ok, target=./build.sh` because that
/// source sorted first. Two rules replace it, and neither needs a threshold:
///
/// 1. A **source locator** ([`SOURCE_LOCATOR_METADATA_KEYS`]) is never
///    inherited, however unanimous.
/// 2. Every other key is inherited only when **every source that carries it
///    agrees** on the value. Unanimity is the exact condition under which the
///    claim survives the merge; disagreement means the merged entry has no
///    truthful value to state, so it states none.
///
/// Pure, so the rule is testable without a store.
#[must_use]
fn consolidated_metadata(memories: &[MemoryEntry]) -> HashMap<String, String> {
    debug_assert!(
        !memories.is_empty(),
        "a cluster with no members has no metadata to merge"
    );

    let mut merged: HashMap<String, String> = HashMap::new();
    let mut contested: Vec<String> = Vec::new();

    for memory in memories {
        for (key, value) in &memory.metadata {
            if SOURCE_LOCATOR_METADATA_KEYS.contains(&key.as_str()) {
                continue;
            }
            match merged.get(key) {
                None if !contested.iter().any(|k| k == key) => {
                    merged.insert(key.clone(), value.clone());
                }
                Some(existing) if existing != value => {
                    merged.remove(key);
                    contested.push(key.clone());
                }
                _ => {}
            }
        }
    }

    debug_assert!(
        merged
            .keys()
            .all(|k| !SOURCE_LOCATOR_METADATA_KEYS.contains(&k.as_str())),
        "a source locator survived the merge"
    );
    merged
}

/// Create a consolidated memory entry from a cluster
#[must_use]
pub fn create_consolidated_entry(
    cluster: &MemoryCluster,
    consolidated_content: String,
    new_embedding: Vec<f32>,
) -> MemoryEntry {
    // Only what is true of the whole cluster — see `consolidated_metadata`.
    let mut metadata = consolidated_metadata(&cluster.memories);
    
    // Provenance is MONOTONE across a merge: if any source was a user
    // assertion, the entry that replaces it still contains a user assertion and
    // must still read as one. The metadata merge above keeps a key only when
    // every source agrees, so a cluster mixing a stated and an observed memory
    // would carry NO `fact_type` at all — and a missing `fact_type` reads as
    // "unknown", demoting the user's assertion. The verbatim pin makes this unreachable today (a stated memory is
    // partitioned out before it can join a cluster), so this is the invariant
    // stated explicitly rather than left to hold by reachability.
    if cluster.memories.iter().any(|m| is_verbatim_pinned(&m.metadata)) {
        metadata.insert(
            FACT_TYPE_METADATA_KEY.to_string(),
            STATED_FACT_TYPE.to_string(),
        );
    }

    // Track consolidation
    let source_ids: Vec<_> = cluster.memories.iter().map(|m| m.id.clone()).collect();
    metadata.insert("consolidated_from".to_string(), source_ids.join(","));
    metadata.insert("consolidation_level".to_string(), format!("{:?}", cluster.compression_level));

    // Precondition: consolidation is only meaningful for a non-empty cluster.
    debug_assert!(
        !cluster.memories.is_empty(),
        "create_consolidated_entry on empty cluster"
    );

    // Create new FSRS state inheriting the best traits from the cluster.
    // Use `max_finite_or` (not `max_by(partial_cmp).unwrap()`) so a stray NaN in
    // a stored FSRS value can't panic the dreaming cycle.
    let mut fsrs = FsrsState::new();
    // Importance: take the max (don't average away a high-importance memory)
    fsrs.importance = max_finite_or(cluster.memories.iter().map(|m| m.fsrs.importance), 1.0);
    fsrs.storage_strength = max_finite_or(
        cluster.memories.iter().map(|m| m.fsrs.storage_strength),
        0.1,
    );
    // Sum access counts (consolidated memory inherits all recall history)
    fsrs.access_count = cluster.memories.iter()
        .map(|m| m.fsrs.access_count)
        .sum();
    fsrs.generation = cluster.memories.iter()
        .map(|m| m.fsrs.generation)
        .max()
        .unwrap_or(0) + 1;

    // Inherit workspace_id from the cluster. `cluster_memories` guarantees a
    // cluster is scope-homogeneous (all members share one workspace_id), so
    // taking the first is exact, not a lossy pick — asserted below.
    let workspace_id = cluster.memories.first()
        .and_then(|m| m.workspace_id.clone());
    debug_assert!(
        cluster
            .memories
            .iter()
            .all(|m| m.workspace_id == workspace_id),
        "consolidating a cluster that spans workspaces would re-home or leak a memory"
    );

    // Postcondition: the merged FSRS scalars are always finite (never NaN/inf),
    // so downstream weight math can't be poisoned.
    debug_assert!(
        fsrs.importance.is_finite() && fsrs.storage_strength.is_finite(),
        "consolidated FSRS scalars must be finite"
    );

    MemoryEntry {
        id: uuid::Uuid::new_v4().to_string(),
        content: consolidated_content,
        // A consolidated memory inherits no bucket: it is NEW text, so its
        // vector belongs to whichever model just embedded it. The daemon stamps
        // the model on write; until then it is simply unlabelled.
        embedding_model: None,
        embeddings: HashMap::new(),
        embedding: new_embedding,
        metadata,
        timestamp: now(),
        fsrs,
        workspace_id,
    }
}

/// Maximum of the finite values in `values`, or `default` when none are finite.
///
/// NaN-safe replacement for `values.max_by(|a, b| a.partial_cmp(b).unwrap())`,
/// whose `unwrap` panics the moment any value is NaN. Non-finite inputs
/// (NaN/±inf) are skipped rather than compared, so a single corrupt stored FSRS
/// scalar can't crash the dreaming cycle.
fn max_finite_or(values: impl Iterator<Item = f32>, default: f32) -> f32 {
    debug_assert!(default.is_finite(), "default must be finite");

    let result = values
        .filter(|v| v.is_finite())
        .fold(None, |acc: Option<f32>, v| {
            Some(acc.map_or(v, |a| a.max(v)))
        })
        .unwrap_or(default);

    debug_assert!(
        result.is_finite(),
        "max_finite_or must return a finite value"
    );
    result
}

fn now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_secs()).unwrap_or(i64::MAX))
}

#[cfg(test)]
mod tests {

    /// The defect this invariant exists to prevent, pinned as a test so it
    /// cannot return by a weight tweak.
    #[test]
    fn non_semantic_terms_alone_must_not_form_a_cluster() {
        let config = ConsolidationConfig::default();
        config
            .validate()
            .expect("the shipped defaults must satisfy their own invariant");

        let floor = config.clustering_weights.non_similarity_floor();
        assert!(
            floor < config.cluster_threshold,
            "floor {floor} must sit below threshold {}",
            config.cluster_threshold
        );

        // The concrete consequence: two ORTHOGONAL memories, both fresh, both
        // never accessed, written at the same instant — the commonest pair
        // shape in a real store — must not cluster.
        let mut a_emb = vec![0.0_f32; 16];
        a_emb[0] = 1.0;
        let mut b_emb = vec![0.0_f32; 16];
        b_emb[1] = 1.0;
        let a = entry_with("a", a_emb, 1_700_000_000);
        let b = entry_with("b", b_emb, 1_700_000_000);

        let score = composite_cluster_score(&a, &b, &config.clustering_weights);
        assert!(
            score < config.cluster_threshold,
            "two orthogonal memories scored {score}, at or above threshold {} — \
             cosine similarity has no veto",
            config.cluster_threshold
        );

        let clusters = cluster_memories(&[a, b], &config);
        assert!(
            clusters.iter().all(|c| c.len() == 1),
            "unrelated memories were clustered together: sizes {:?}",
            clusters.iter().map(Vec::len).collect::<Vec<_>>()
        );
    }

    /// The exact configuration that shipped before 2026-09-09 must now be
    /// rejected, and the message must say how to fix it.
    #[test]
    fn the_pre_fix_default_threshold_is_now_rejected() {
        let broken = ConsolidationConfig {
            cluster_threshold: 0.45,
            ..ConsolidationConfig::default()
        };
        let err = broken
            .validate()
            .expect_err("floor 0.50 >= threshold 0.45 must not validate");
        assert!(err.contains("0.500"), "message must name the floor: {err}");
        assert!(err.contains("cluster_threshold"), "must name the knob: {err}");
        assert_eq!(
            broken.min_required_similarity(),
            0.0,
            "the broken config demands no similarity at all — that is the defect"
        );
    }

    /// `min_required_similarity` reports what a config really asks for, which
    /// is not what `cluster_threshold` looks like it asks for.
    #[test]
    fn min_required_similarity_is_the_honest_reading_of_the_threshold() {
        let config = ConsolidationConfig::default();
        let need = config.min_required_similarity();
        assert!(need > 0.0, "the fixed default must demand some similarity");

        // Derived, not asserted against a literal: a pair sitting exactly at
        // the reported bar, with every non-semantic term maximal, must land at
        // the threshold.
        let w = &config.clustering_weights;
        let reconstructed = (w.similarity * need) / w.total() + w.non_similarity_floor();
        assert!(
            (reconstructed - config.cluster_threshold).abs() < 1e-5,
            "reported bar {need} reconstructs to {reconstructed}, not {}",
            config.cluster_threshold
        );

        // The drift fixture's own config (threshold 0.65) always satisfied the
        // invariant — it is the shipped default that did not.
        let drift = ConsolidationConfig {
            cluster_threshold: 0.65,
            ..ConsolidationConfig::default()
        };
        assert!(drift.validate().is_ok());
        assert!((drift.min_required_similarity() - 0.30).abs() < 1e-5);
    }

    /// A weights table with no similarity weight has no floor to speak of —
    /// `composite_cluster_score` falls back to pure similarity — so the
    /// invariant must not fire spuriously there.
    #[test]
    fn a_zero_total_weight_table_has_no_floor() {
        let w = ClusteringWeights {
            similarity: 0.0,
            recall_affinity: 0.0,
            importance_proximity: 0.0,
            age_proximity: 0.0,
            time_span_minutes: 1440.0,
        };
        assert_eq!(w.non_similarity_floor(), 0.0);
        let config = ConsolidationConfig {
            clustering_weights: w,
            ..ConsolidationConfig::default()
        };
        assert!(
            config.validate().is_ok(),
            "pure-similarity fallback is fine"
        );
    }

    /// Helper: a fresh, never-accessed memory at a given instant.
    fn entry_with(id: &str, embedding: Vec<f32>, timestamp: i64) -> MemoryEntry {
        MemoryEntry {
            id: id.to_string(),
            content: format!("content {id}"),
            embedding,
            embedding_model: None,
            embeddings: HashMap::new(),
            metadata: HashMap::new(),
            timestamp,
            fsrs: FsrsState::default(),
            workspace_id: None,
        }
    }
    use super::*;

    #[test]
    fn test_compression_levels() {
        let thresholds = WeightThresholds::default();
        
        assert_eq!(CompressionLevel::from_weight(0.1, &thresholds), CompressionLevel::Essence);
        assert_eq!(CompressionLevel::from_weight(0.3, &thresholds), CompressionLevel::Compressed);
        assert_eq!(CompressionLevel::from_weight(0.6, &thresholds), CompressionLevel::Standard);
        assert_eq!(CompressionLevel::from_weight(0.9, &thresholds), CompressionLevel::Detailed);
        assert_eq!(CompressionLevel::from_weight(1.5, &thresholds), CompressionLevel::Expand);
    }

    /// `cluster_score_ceiling` exists so a benchmark can count pruned pairs
    /// without a second copy of the formula — which only helps if it really is
    /// the same formula. This pins the two together at the decision boundary:
    /// a pair prunes at threshold `t` exactly when its public ceiling is below
    /// `t`.
    #[test]
    fn the_public_ceiling_matches_the_one_the_prune_uses() {
        let weights = ClusteringWeights::default();
        let a = entry_with("a", vec![1.0, 0.0, 0.0, 0.0], 1_700_000_000);

        for (offset, access, importance) in [
            (0_i64, 0_u32, 1.0_f32),
            (3_600, 4, 2.0),
            (86_400, 12, 4.0),
            (30 * 86_400, 0, 1.0),
        ] {
            let mut b = entry_with("b", vec![0.0, 1.0, 0.0, 0.0], 1_700_000_000 + offset);
            b.fsrs.access_count = access;
            b.fsrs.importance = importance;

            let ceiling = cluster_score_ceiling(&a, &b, &weights);
            let (na, nb) = (l2_norm(&a.embedding), l2_norm(&b.embedding));

            // Just under the ceiling: must not prune. Just over: must prune.
            let below = ceiling - 1e-3;
            let above = ceiling + 1e-3;
            assert!(
                score_or_prune(&a, na, &b, nb, &weights, below).is_some(),
                "pruned at a threshold below the ceiling {ceiling}"
            );
            assert!(
                score_or_prune(&a, na, &b, nb, &weights, above).is_none(),
                "did not prune at a threshold above the ceiling {ceiling}"
            );
            // And the ceiling really bounds the real score.
            let actual = composite_cluster_score_with_norms(&a, na, &b, nb, &weights);
            assert!(
                actual <= ceiling + f32::EPSILON,
                "score {actual} exceeded its own ceiling {ceiling}"
            );
        }
    }

    /// **The prune must skip only pairs that would have been rejected anyway —
    /// checked by clustering the same corpus twice and comparing the output,
    /// not by re-deriving the inequality in the test.**
    ///
    /// `score_or_prune` declines to compute the cosine when the pair's best
    /// possible score is under the threshold. That is only lossless if the
    /// ceiling really is an upper bound, which rests on all three cheap terms
    /// being in `0..=1`. Rather than assert the algebra, this reproduces the
    /// unpruned pass (threshold `NEG_INFINITY` never prunes) and requires the
    /// resulting clusters to match exactly, over a corpus built to straddle the
    /// threshold from both sides.
    #[test]
    fn pruning_never_changes_a_cluster() {
        const DIM: usize = 64;
        let config = ConsolidationConfig::default();

        // A corpus spread across time, importance and access count, so the
        // ceiling lands above the threshold for some pairs and below it for
        // others. A corpus where the prune never fires would pass vacuously.
        let mut memories = Vec::new();
        for i in 0..240_usize {
            let topic = i % 12;
            let mut embedding = vec![0.0_f32; DIM];
            for (d, slot) in embedding.iter_mut().enumerate() {
                let raw = u16::try_from((d.wrapping_mul(2_654_435_761) ^ (topic * 7717)) % 1000)
                    .expect("bounded by the modulus");
                *slot = f32::from(raw) / 500.0 - 1.0;
            }
            // Deterministic jitter so members of a topic are near but not equal.
            let jitter = u16::try_from((i.wrapping_mul(40_503)) % 97).expect("bounded");
            embedding[i % DIM] += f32::from(jitter) / 500.0;

            let mut entry = entry_with(
                &format!("m{i}"),
                embedding,
                // Spread over ~40 days so age_proximity genuinely varies.
                1_700_000_000 + i64::try_from(i).expect("fits") * 14_400,
            );
            entry.fsrs.access_count = u32::try_from(i % 17).expect("fits");
            entry.fsrs.importance = 1.0 + f32::from(u8::try_from(i % 5).expect("fits"));
            memories.push(entry);
        }

        // Reference: the same greedy pass with pruning made unreachable.
        let norms: Vec<f32> = memories.iter().map(|m| l2_norm(&m.embedding)).collect();
        let mut pruned_pairs = 0_usize;
        let mut total_pairs = 0_usize;
        for i in 0..memories.len() {
            for j in (i + 1)..memories.len() {
                if !same_scope(&memories[i], &memories[j]) {
                    continue;
                }
                total_pairs += 1;
                let with_prune = score_or_prune(
                    &memories[i],
                    norms[i],
                    &memories[j],
                    norms[j],
                    &config.clustering_weights,
                    config.cluster_threshold,
                );
                let full = composite_cluster_score_with_norms(
                    &memories[i],
                    norms[i],
                    &memories[j],
                    norms[j],
                    &config.clustering_weights,
                );
                match with_prune {
                    // Pruned: the full score must indeed be below threshold.
                    None => {
                        pruned_pairs += 1;
                        assert!(
                            full < config.cluster_threshold,
                            "pruned a pair that actually scored {full} >= {}",
                            config.cluster_threshold
                        );
                    }
                    // Not pruned: the score must be exactly the unpruned one.
                    Some(scored) => assert_eq!(
                        scored.to_bits(),
                        full.to_bits(),
                        "pruning path changed the score: {scored} vs {full}"
                    ),
                }
            }
        }

        assert!(
            pruned_pairs > 0,
            "the prune never fired on this corpus, so the test proves nothing — \
             make the fixture spread further across time/importance/recall"
        );
        assert!(
            pruned_pairs < total_pairs,
            "the prune rejected every pair, so the fixture cannot show it is lossless"
        );

        // And the end-to-end shape: clusters must be identical.
        let clusters = cluster_memories(&memories, &config);
        let shape: Vec<Vec<String>> = clusters
            .iter()
            .map(|c| c.iter().map(|m| m.id.clone()).collect())
            .collect();
        // Re-running must be deterministic, and every memory must appear once.
        let again = cluster_memories(&memories, &config);
        let shape_again: Vec<Vec<String>> = again
            .iter()
            .map(|c| c.iter().map(|m| m.id.clone()).collect())
            .collect();
        assert_eq!(shape, shape_again, "clustering is not deterministic");
        assert_eq!(
            shape.iter().map(Vec::len).sum::<usize>(),
            memories.len(),
            "clustering lost or duplicated a memory"
        );
    }

    /// **The hoist must change the arithmetic, not the answer — and "close
    /// enough" is not the claim being made.**
    ///
    /// `cluster_memories` no longer calls `cosine_similarity_f32`; it computes
    /// each memory's magnitude once and divides a dot product by two cached
    /// scalars. That is only safe to do silently if the result is *identical*,
    /// because the score is compared against `cluster_threshold` and a pair
    /// sitting exactly on it would otherwise flip between releases for reasons
    /// nobody could see.
    ///
    /// It is identical because the two SIMD kernels are written the same way —
    /// same 16-wide chunks, same order, same `fmadd`, same reduction, same
    /// scalar remainder. So this asserts exact `f32` equality, on vectors
    /// deliberately chosen to produce long, association-sensitive sums.
    #[test]
    fn norm_hoisting_is_bit_identical_to_the_cosine_kernel() {
        // Widths that do and do not divide the 16-wide SIMD chunk, so the
        // scalar-remainder tail is exercised too.
        for dim in [1_usize, 15, 16, 17, 127, 384, 768] {
            // `u16` bounded by the modulus, so the widening to f32 is exact
            // and clippy's precision-loss lint has nothing to warn about.
            let draw = |i: usize, mul: usize, add: usize, modulus: u16| -> f32 {
                let raw = u16::try_from(i.wrapping_mul(mul).wrapping_add(add) % usize::from(modulus))
                    .expect("bounded by the modulus");
                f32::from(raw)
            };
            let a: Vec<f32> = (0..dim).map(|i| draw(i, 2_654_435_761, 0, 1000) / 500.0 - 1.0).collect();
            let b: Vec<f32> = (0..dim).map(|i| draw(i, 40_503, 7, 997) / 498.5 - 1.0).collect();

            let reference = cosine_similarity(&a, &b);
            let hoisted = cosine_with_norms(&a, l2_norm(&a), &b, l2_norm(&b));
            assert_eq!(
                reference.to_bits(),
                hoisted.to_bits(),
                "dim {dim}: hoisted cosine {hoisted} != kernel cosine {reference}"
            );
        }
    }

    /// The hoist must also inherit the degenerate-input contract, not just the
    /// arithmetic: 0.0 on a width mismatch, on empty, and on a zero-magnitude
    /// vector (which divides by zero and would otherwise leak inf or NaN into
    /// a score that is about to be compared against a threshold).
    #[test]
    fn norm_hoisting_keeps_the_zero_on_degenerate_input_contract() {
        let v = vec![1.0_f32, 2.0, 3.0];
        let zero = vec![0.0_f32; 3];

        assert_eq!(cosine_with_norms(&v, l2_norm(&v), &zero, l2_norm(&zero)), 0.0);
        assert_eq!(cosine_with_norms(&zero, l2_norm(&zero), &v, l2_norm(&v)), 0.0);
        assert_eq!(cosine_with_norms(&zero, l2_norm(&zero), &zero, l2_norm(&zero)), 0.0);
        // Width mismatch: memories from two embedding eras can co-occur.
        let w = vec![1.0_f32, 2.0];
        assert_eq!(cosine_with_norms(&v, l2_norm(&v), &w, l2_norm(&w)), 0.0);
        // Empty on both sides.
        assert_eq!(cosine_with_norms(&[], 0.0, &[], 0.0), 0.0);
        assert_eq!(l2_norm(&[]), 0.0);

        // And each matches what the kernel path returns for the same input.
        assert_eq!(cosine_similarity(&v, &zero), 0.0);
        assert_eq!(cosine_similarity(&v, &w), 0.0);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &c) - 0.0).abs() < 0.001);

        let d = vec![0.707, 0.707, 0.0];
        assert!((cosine_similarity(&a, &d) - 0.707).abs() < 0.01);
    }

    // Reference scalar implementation kept only in the test, to prove the
    // SIMD-backed `cosine_similarity` matches it across random vectors.
    fn cosine_reference(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        let mut dot = 0.0f32;
        let (mut na, mut nb) = (0.0f32, 0.0f32);
        for (x, y) in a.iter().zip(b.iter()) {
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
        if na == 0.0 || nb == 0.0 {
            return 0.0;
        }
        dot / (na.sqrt() * nb.sqrt())
    }

    #[test]
    fn cosine_matches_scalar_reference_on_embedding_sized_vectors() {
        // Deterministic pseudo-random 768-dim pairs (typical embedding width).
        for seed in 0..8u32 {
            let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
            let mut next = || {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                // Doubling is exact in binary floating point, so the fused
                // multiply-add rounds once where `x * 2.0 - 1.0` did: the same
                // bits for every `u32` state (checked exhaustively).
                (s.lossy_f32() / u32::MAX.lossy_f32()).mul_add(2.0, -1.0)
            };
            let a: Vec<f32> = (0..768).map(|_| next()).collect();
            let b: Vec<f32> = (0..768).map(|_| next()).collect();
            let simd = cosine_similarity(&a, &b);
            let scalar = cosine_reference(&a, &b);
            assert!(
                (simd - scalar).abs() < 1e-4,
                "seed {seed}: simd {simd} vs scalar {scalar}"
            );
        }
    }

    #[test]
    fn cosine_edge_cases_return_zero_not_nan() {
        // Zero-magnitude vector: SIMD yields NaN; the guard must return 0.0.
        let zero = vec![0.0f32; 4];
        let v = vec![1.0f32, 2.0, 3.0, 4.0];
        let z = cosine_similarity(&zero, &v);
        assert!(z.is_finite() && z.abs() < 1e-6, "zero-vector → 0.0, got {z}");
        // Length mismatch must not panic (SIMD would); return 0.0.
        assert!(cosine_similarity(&[1.0, 2.0], &[1.0, 2.0, 3.0]).abs() < f32::EPSILON);
        // Empty input.
        assert!(cosine_similarity(&[], &[]).abs() < f32::EPSILON);
    }

    #[test]
    fn test_composite_score_identical() {
        let weights = ClusteringWeights::default();
        let entry = MemoryEntry {
            embedding_model: None,
            embeddings: HashMap::new(),
            id: "1".into(),
            content: "test".into(),
            embedding: vec![1.0, 0.0, 0.0],
            metadata: HashMap::new(),
            timestamp: 1_000_000,
            fsrs: FsrsState::default(),
            workspace_id: None,
        };
        let score = composite_cluster_score(&entry, &entry, &weights);
        assert!(score > 0.9, "identical memories should score high: {score}");
    }

    #[test]
    fn test_composite_score_different() {
        let weights = ClusteringWeights::default();
        let a = MemoryEntry {
            embedding_model: None,
            embeddings: HashMap::new(),
            id: "1".into(),
            content: "A".into(),
            embedding: vec![1.0, 0.0, 0.0],
            metadata: HashMap::new(),
            timestamp: 0,
            fsrs: FsrsState { importance: 1.0, access_count: 0, ..FsrsState::default() },
            workspace_id: None,
        };
        let b = MemoryEntry {
            embedding_model: None,
            embeddings: HashMap::new(),
            id: "2".into(),
            content: "B".into(),
            embedding: vec![0.0, 1.0, 0.0],
            metadata: HashMap::new(),
            timestamp: 86400 * 365, // 1 year apart
            fsrs: FsrsState { importance: 5.0, access_count: 100, ..FsrsState::default() },
            workspace_id: None,
        };
        let score = composite_cluster_score(&a, &b, &weights);
        assert!(score < 0.3, "very different memories should score low: {score}");
    }

    #[test]
    fn test_clustering_with_composite() {
        let config = ConsolidationConfig {
            cluster_threshold: 0.5,
            ..Default::default()
        };

        let memories = vec![
            MemoryEntry {
                embedding_model: None,
                embeddings: HashMap::new(),
                id: "1".into(),
                content: "A".into(),
                embedding: vec![1.0, 0.0, 0.0],
                metadata: HashMap::new(),
                timestamp: 1000,
                fsrs: FsrsState { importance: 2.0, access_count: 5, ..FsrsState::default() },
                workspace_id: None,
            },
            MemoryEntry {
                embedding_model: None,
                embeddings: HashMap::new(),
                id: "2".into(),
                content: "B".into(),
                embedding: vec![0.99, 0.1, 0.0],
                metadata: HashMap::new(),
                timestamp: 1100,
                fsrs: FsrsState { importance: 2.0, access_count: 4, ..FsrsState::default() },
                workspace_id: None,
            },
            MemoryEntry {
                embedding_model: None,
                embeddings: HashMap::new(),
                id: "3".into(),
                content: "C".into(),
                embedding: vec![0.0, 1.0, 0.0],
                metadata: HashMap::new(),
                timestamp: 86400 * 60,
                fsrs: FsrsState { importance: 5.0, access_count: 0, ..FsrsState::default() },
                workspace_id: None,
            },
        ];

        let clusters = cluster_memories(&memories, &config);

        // 1 and 2 are similar in embedding, importance, age, and recall → should cluster
        // 3 is different on every axis → separate
        assert_eq!(clusters.len(), 2);
    }

    /// A memory that clusters with every other `similar_entry` (identical
    /// embedding/importance/recall, adjacent timestamps → composite score ≈ 1).
    fn similar_entry(id: &str, content: &str) -> MemoryEntry {
        MemoryEntry {
            embedding_model: None,
            embeddings: HashMap::new(),
            id: id.to_string(),
            content: content.to_string(),
            embedding: vec![1.0, 0.0, 0.0],
            metadata: HashMap::new(),
            timestamp: 1000,
            fsrs: FsrsState { importance: 2.0, access_count: 5, ..FsrsState::default() },
            workspace_id: None,
        }
    }


    fn entry_with_fact_type(fact_type: Option<&str>) -> MemoryEntry {
        let mut entry = similar_entry("pin", "the deploy key lives in 1Password");
        if let Some(fact_type) = fact_type {
            entry
                .metadata
                .insert(FACT_TYPE_METADATA_KEY.to_string(), fact_type.to_string());
        }
        entry
    }

    /// Positive space: only an explicit `stated` provenance pins a memory, in
    /// whatever casing/padding the extraction path happened to write.
    #[test]
    fn stated_provenance_pins_a_memory_verbatim() {
        for label in ["stated", "STATED", "  Stated  "] {
            assert!(
                is_verbatim_pinned(&entry_with_fact_type(Some(label)).metadata),
                "{label:?} names a user assertion"
            );
        }
    }

    /// Negative space, and the load-bearing half: absence of a provenance signal
    /// is NOT evidence the user stated something, so an unlabeled memory must
    /// never be able to pin itself out of consolidation. Mirrors
    /// `MemoryProvenance`'s own `Observed` default.
    #[test]
    fn everything_that_is_not_stated_stays_consolidatable() {
        for label in [None, Some(""), Some("observed"), Some("statedly"), Some("user-said")] {
            assert!(
                !is_verbatim_pinned(&entry_with_fact_type(label).metadata),
                "{label:?} is not an explicit user assertion"
            );
        }
    }

    /// The partition the band loop performs must be lossless and exact: every
    /// memory lands on exactly one side, and neither side holds one of the
    /// other's.
    #[test]
    fn the_provenance_partition_is_lossless() {
        let band = vec![
            entry_with_fact_type(Some("stated")),
            entry_with_fact_type(None),
            entry_with_fact_type(Some("observed")),
            entry_with_fact_type(Some("STATED")),
        ];
        let band_count = band.len();

        let (pinned, unpinned): (Vec<_>, Vec<_>) = band
            .into_iter()
            .partition(|entry: &MemoryEntry| is_verbatim_pinned(&entry.metadata));

        assert_eq!(pinned.len() + unpinned.len(), band_count);
        assert_eq!(pinned.len(), 2);
        assert!(pinned.iter().all(|e| is_verbatim_pinned(&e.metadata)));
        assert!(!unpinned.iter().any(|e| is_verbatim_pinned(&e.metadata)));
    }



    fn chunk_entry(id: &str, source_id: &str, chunk: &str, tool: &str, outcome: &str) -> MemoryEntry {
        let mut entry = similar_entry(id, "some captured output");
        entry.metadata.insert("source_id".to_string(), source_id.to_string());
        entry.metadata.insert("chunk".to_string(), chunk.to_string());
        entry.metadata.insert("tool".to_string(), tool.to_string());
        entry.metadata.insert("outcome".to_string(), outcome.to_string());
        entry.metadata.insert("category".to_string(), "tool_result".to_string());
        entry
    }

    /// The bug this rule exists for. `assemble_handle_content` gathers every
    /// memory sharing a `source_id` and orders them by their `chunk` position,
    /// so a consolidated entry that inherited both would be spliced into the
    /// middle of a tool result the model was promised was stored verbatim —
    /// and the sources it replaced are gone, so nothing else fills that slot.
    ///
    /// A summary has no position in anyone's byte stream. It carries neither
    /// key, however unanimous the cluster was about them.
    #[test]
    fn a_summary_never_inherits_a_source_locator() {
        let cluster = MemoryCluster::new(
            vec![
                chunk_entry("c1", "abc123", "1/17", "exec", "ok"),
                chunk_entry("c2", "abc123", "2/17", "exec", "ok"),
            ],
            CompressionLevel::Essence,
            &FsrsParameters::default(),
        );

        let consolidated = create_consolidated_entry(&cluster, "a gist".to_string(), vec![0.0; 4]);

        assert!(
            !consolidated.metadata.contains_key("source_id"),
            "a summary is not a chunk of any one result"
        );
        assert!(
            !consolidated.metadata.contains_key("chunk"),
            "and it has no position inside one"
        );
    }

    /// Every other key survives only when the whole cluster agrees. A gist of
    /// two different tools' output must not claim to be from one of them.
    #[test]
    fn a_summary_inherits_only_what_every_source_agrees_on() {
        let cluster = MemoryCluster::new(
            vec![
                chunk_entry("c1", "abc123", "1/2", "exec", "ok"),
                chunk_entry("c2", "def456", "1/2", "read_file", "FAILED"),
            ],
            CompressionLevel::Essence,
            &FsrsParameters::default(),
        );

        let consolidated = create_consolidated_entry(&cluster, "a gist".to_string(), vec![0.0; 4]);

        assert!(
            !consolidated.metadata.contains_key("tool"),
            "two tools disagree, so the merged entry names neither"
        );
        assert!(
            !consolidated.metadata.contains_key("outcome"),
            "one succeeded and one failed — the merge has no truthful verdict"
        );
        assert_eq!(
            consolidated.metadata.get("category").map(String::as_str),
            Some("tool_result"),
            "a key the whole cluster agrees on is still true of the merge"
        );
    }

    /// Negative space for the unanimity rule: a key present on only SOME
    /// sources is still unanimous among those that carry it, and a key that is
    /// dropped by disagreement must stay dropped even if a later source would
    /// have re-introduced the first value.
    #[test]
    fn unanimity_survives_partial_presence_and_never_re_admits_a_contested_key() {
        let mut only_one = similar_entry("a", "x");
        only_one.metadata.insert("topic".to_string(), "deploys".to_string());
        let bare = similar_entry("b", "y");
        let mut disagrees = similar_entry("c", "z");
        disagrees.metadata.insert("tool".to_string(), "exec".to_string());
        let mut agrees_with_first = similar_entry("d", "w");
        agrees_with_first.metadata.insert("tool".to_string(), "read_file".to_string());
        let mut back_to_exec = similar_entry("e", "v");
        back_to_exec.metadata.insert("tool".to_string(), "exec".to_string());

        let cluster = MemoryCluster::new(
            vec![only_one, bare, disagrees, agrees_with_first, back_to_exec],
            CompressionLevel::Essence,
            &FsrsParameters::default(),
        );
        let consolidated = create_consolidated_entry(&cluster, "a gist".to_string(), vec![0.0; 4]);

        assert_eq!(
            consolidated.metadata.get("topic").map(String::as_str),
            Some("deploys"),
            "nothing contradicted it"
        );
        assert!(
            !consolidated.metadata.contains_key("tool"),
            "a contested key stays dropped — the third source agreeing with the \
             first does not make the second one go away"
        );
    }

    /// Provenance must be monotone across a merge: an entry that replaces a
    /// user assertion still contains one, so it must still read as one. The
    /// metadata merge keeps only unanimous keys, so a mixed cluster would lose
    /// `fact_type` entirely; both orders are asserted so no ordering can
    /// sneak back in.
    #[test]
    fn a_consolidated_entry_inherits_stated_provenance_from_any_source() {
        for stated_first in [true, false] {
            let mut stated = similar_entry("stated", "the deploy key lives in 1Password");
            stated
                .metadata
                .insert(FACT_TYPE_METADATA_KEY.to_string(), "stated".to_string());
            let mut observed = similar_entry("observed", "the deploy job read a key from somewhere");
            observed
                .metadata
                .insert(FACT_TYPE_METADATA_KEY.to_string(), "observed".to_string());

            let memories = if stated_first {
                vec![stated, observed]
            } else {
                vec![observed, stated]
            };
            let cluster = MemoryCluster::new(
                memories,
                CompressionLevel::Standard,
                &FsrsParameters::default(),
            );

            let consolidated =
                create_consolidated_entry(&cluster, "a gist".to_string(), vec![0.0; 4]);

            assert!(
                is_verbatim_pinned(&consolidated.metadata),
                "stated_first={stated_first}: a merge containing a user assertion stays stated"
            );
        }
    }

    /// Negative space: a cluster of purely agent-observed memories must NOT
    /// come back pinned, or every consolidated entry would pin itself and
    /// consolidation would stop after one pass.
    #[test]
    fn a_consolidated_entry_of_observed_sources_stays_consolidatable() {
        let cluster = MemoryCluster::new(
            vec![
                similar_entry("a", "the build took 40 seconds"),
                similar_entry("b", "the build took 41 seconds"),
            ],
            CompressionLevel::Standard,
            &FsrsParameters::default(),
        );

        let consolidated = create_consolidated_entry(&cluster, "a gist".to_string(), vec![0.0; 4]);

        assert!(
            !is_verbatim_pinned(&consolidated.metadata),
            "nothing here was ever asserted by the user"
        );
    }

    #[test]
    fn cluster_member_count_is_bounded_and_lossless() {
        // 25 mutually-similar memories with a cap of 10 must split into bounded
        // clusters (10, 10, 5) with every memory preserved exactly once.
        let config = ConsolidationConfig {
            cluster_threshold: 0.3,
            max_cluster_memories: 10,
            ..Default::default()
        };
        let memories: Vec<_> = (0..25).map(|i| similar_entry(&i.to_string(), "x")).collect();

        let clusters = cluster_memories(&memories, &config);

        assert!(
            clusters.iter().all(|c| c.len() <= 10),
            "no cluster may exceed the member cap"
        );
        let total: usize = clusters.iter().map(Vec::len).sum();
        assert_eq!(total, 25, "no memory may be dropped");
        let mut ids: Vec<_> = clusters.iter().flatten().map(|m| m.id.clone()).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), 25, "every memory appears exactly once");
    }

    /// Like [`similar_entry`] but with an explicit scope, so the workspace
    /// barrier can be exercised without touching any other cluster signal.
    fn scoped_entry(id: &str, workspace_id: Option<&str>) -> MemoryEntry {
        MemoryEntry {
            workspace_id: workspace_id.map(str::to_string),
            ..similar_entry(id, "x")
        }
    }

    #[test]
    fn cluster_never_merges_across_workspaces() {
        // Two memories identical on every clustering signal (embedding,
        // importance, recall, age) but in DIFFERENT workspaces must land in
        // separate clusters — a dream cycle must never fold workspace B's
        // memory into workspace A's consolidated entry.
        let config = ConsolidationConfig {
            cluster_threshold: 0.3,
            ..Default::default()
        };
        let memories = vec![
            scoped_entry("a", Some("ws-A")),
            scoped_entry("b", Some("ws-B")),
        ];

        let clusters = cluster_memories(&memories, &config);

        assert_eq!(clusters.len(), 2, "cross-workspace memories must not cluster");
        assert!(clusters.iter().all(|c| c.len() == 1));
    }

    #[test]
    fn cluster_merges_within_one_workspace() {
        // The barrier must not block a legitimate same-workspace merge: two
        // similar memories in the SAME workspace still cluster together.
        let config = ConsolidationConfig {
            cluster_threshold: 0.3,
            ..Default::default()
        };
        let memories = vec![
            scoped_entry("a", Some("ws-A")),
            scoped_entry("b", Some("ws-A")),
        ];

        let clusters = cluster_memories(&memories, &config);

        assert_eq!(clusters.len(), 1, "same-workspace memories must cluster");
        assert_eq!(clusters[0].len(), 2);
        assert!(clusters[0].iter().all(|m| m.workspace_id.as_deref() == Some("ws-A")));
    }

    #[test]
    fn cluster_never_merges_global_with_workspace() {
        // A global memory (None) and a workspace memory (Some) are co-visible in
        // recall but must NOT merge: folding them would either leak the
        // workspace memory into global scope or hide the global fact inside one
        // workspace. Both directions are unsafe → exact-Option equality.
        let config = ConsolidationConfig {
            cluster_threshold: 0.3,
            ..Default::default()
        };
        let memories = vec![
            scoped_entry("global", None),
            scoped_entry("scoped", Some("ws-A")),
        ];

        let clusters = cluster_memories(&memories, &config);

        assert_eq!(clusters.len(), 2, "global and workspace memories must not merge");
    }

    #[test]
    fn cluster_partitions_by_workspace_losslessly() {
        // A mixed pool of three scopes (+ globals) must partition cleanly:
        // every memory preserved exactly once, and no cluster spans scopes.
        let config = ConsolidationConfig {
            cluster_threshold: 0.3,
            ..Default::default()
        };
        let mut memories = Vec::new();
        for (ws, n) in [(None, 3), (Some("ws-A"), 4), (Some("ws-B"), 2)] {
            for i in 0..n {
                memories.push(scoped_entry(&format!("{ws:?}-{i}"), ws));
            }
        }
        let expected_total = memories.len();

        let clusters = cluster_memories(&memories, &config);

        // No cluster spans scopes.
        assert!(
            clusters
                .iter()
                .all(|c| c.iter().all(|m| m.workspace_id == c[0].workspace_id)),
            "a cluster mixed workspaces"
        );
        // Nothing dropped, nothing duplicated.
        let total: usize = clusters.iter().map(Vec::len).sum();
        assert_eq!(total, expected_total, "no memory may be dropped");
        let mut ids: Vec<_> = clusters.iter().flatten().map(|m| m.id.clone()).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), expected_total, "every memory appears exactly once");
    }

    #[test]
    fn cluster_content_bytes_are_bounded_and_lossless() {
        // Ten 100-byte memories with a 300-byte budget → at most 3 per cluster.
        let config = ConsolidationConfig {
            cluster_threshold: 0.3,
            max_cluster_memories: 1000,
            max_cluster_content_bytes: 300,
            ..Default::default()
        };
        let body = "a".repeat(100);
        let memories: Vec<_> = (0..10).map(|i| similar_entry(&i.to_string(), &body)).collect();

        let clusters = cluster_memories(&memories, &config);

        assert!(
            clusters
                .iter()
                .all(|c| c.len() == 1 || c.iter().map(|m| m.content.len()).sum::<usize>() <= 300),
            "a multi-member cluster exceeded the byte budget"
        );
        assert_eq!(
            clusters.iter().map(Vec::len).sum::<usize>(),
            10,
            "no memory may be dropped by the byte bound"
        );
    }

    #[test]
    fn config_deserializes_without_new_bound_fields() {
        // A ConsolidationConfig serialized before the bound fields existed must
        // still load, defaulting the new caps (serde backward compatibility).
        let legacy = r#"{
            "cluster_threshold": 0.45,
            "min_cluster_size": 2,
            "max_compression_ratio": 0.5,
            "min_remaining_memories": 20,
            "weight_thresholds": {"essence":0.2,"compressed":0.5,"standard":0.8,"detailed":1.0},
            "clustering_weights": {"similarity":0.45,"recall_affinity":0.2,"importance_proximity":0.2,"age_proximity":0.15}
        }"#;
        let cfg: ConsolidationConfig = serde_json::from_str(legacy).expect("legacy config must load");
        assert_eq!(cfg.max_cluster_memories, DEFAULT_MAX_CLUSTER_MEMORIES);
        assert_eq!(cfg.max_cluster_content_bytes, DEFAULT_MAX_CLUSTER_CONTENT_BYTES);
    }

    #[test]
    fn content_budget_scales_with_context_window() {
        // Bigger context → bigger (or equal) content budget, and each budget
        // fits the token window: at the estimator's worst-case 0.5 tok/byte the
        // content is at most `bytes / 2` tokens, which must stay under `window`.
        let small = cluster_content_bytes_for_context(8_192);
        let large = cluster_content_bytes_for_context(200_000);
        assert!(large > small, "budget must grow with the window");
        for window in [8_192_usize, 32_000, 128_000, 200_000] {
            let bytes = cluster_content_bytes_for_context(window);
            let worst_case_tokens = bytes / 2; // 0.5 tok/byte upper bound
            assert!(
                worst_case_tokens < window,
                "budget {bytes}B (~{worst_case_tokens} tok) must fit the {window}-tok window"
            );
        }
    }

    #[test]
    fn content_budget_floors_on_tiny_window() {
        // A tiny or misreported window must not collapse the budget to zero, or
        // consolidation would lock up (every candidate rejected).
        assert!(cluster_content_bytes_for_context(0) >= 2_048);
        assert!(cluster_content_bytes_for_context(100) >= 2_048);
    }

    #[test]
    fn with_summarizer_context_window_sizes_the_byte_budget() {
        // The builder swaps the small-model fallback for the real model's budget
        // and leaves the member-count cap alone.
        let base = ConsolidationConfig::default();
        let large = base.clone().with_summarizer_context_window(200_000);
        assert_eq!(
            large.max_cluster_content_bytes,
            cluster_content_bytes_for_context(200_000)
        );
        assert!(large.max_cluster_content_bytes > base.max_cluster_content_bytes);
        assert_eq!(large.max_cluster_memories, base.max_cluster_memories);
    }

    #[test]
    fn default_content_budget_matches_fallback_window() {
        // default() must equal the fallback-window formula (one source of truth).
        assert_eq!(
            DEFAULT_MAX_CLUSTER_CONTENT_BYTES,
            cluster_content_bytes_for_context(FALLBACK_SUMMARIZER_CONTEXT_WINDOW_TOKENS)
        );
    }

    #[test]
    fn test_cluster_prompt() {
        let memories = vec![
            MemoryEntry {
                embedding_model: None,
                embeddings: HashMap::new(),
                id: "1".to_string(),
                content: "The user prefers dark mode".to_string(),
                embedding: vec![1.0, 0.0],
                metadata: HashMap::new(),
                timestamp: 0,
                fsrs: FsrsState::default(),
                workspace_id: None,
            },
            MemoryEntry {
                embedding_model: None,
                embeddings: HashMap::new(),
                id: "2".to_string(),
                content: "User likes dark themes in apps".to_string(),
                embedding: vec![0.95, 0.1],
                metadata: HashMap::new(),
                timestamp: 0,
                fsrs: FsrsState::default(),
                workspace_id: None,
            },
        ];

        let cluster = MemoryCluster::new(memories, CompressionLevel::Compressed, &FsrsParameters::default());
        let prompt = cluster.build_consolidation_prompt();
        
        assert!(prompt.contains("2 memories"));
        assert!(prompt.contains("dark mode"));
        assert!(prompt.contains("dark themes"));
    }

    fn entry_with_fsrs(id: &str, importance: f32, storage: f32, access: u32) -> MemoryEntry {
        let fsrs = FsrsState {
            importance,
            storage_strength: storage,
            access_count: access,
            ..FsrsState::default()
        };
        MemoryEntry {
            embedding_model: None,
            embeddings: HashMap::new(),
            id: id.to_string(),
            content: format!("memory {id}"),
            embedding: vec![1.0, 0.0],
            metadata: HashMap::new(),
            timestamp: 0,
            fsrs,
            workspace_id: None,
        }
    }

    #[test]
    fn max_finite_or_skips_nan_and_inf() {
        // A NaN or ±inf must be ignored, never compared (the old
        // `max_by(partial_cmp).unwrap()` panicked on NaN).
        let vals = [1.0_f32, f32::NAN, 3.5, f32::INFINITY, 2.0];
        assert!((max_finite_or(vals.into_iter(), 1.0) - 3.5).abs() < 1e-6);
        // All non-finite → fall back to the default.
        let none = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
        assert!((max_finite_or(none.into_iter(), 0.1) - 0.1).abs() < 1e-6);
        // Empty → default.
        assert!((max_finite_or(std::iter::empty(), 1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn consolidated_entry_takes_max_importance_and_sums_access() {
        let cluster = MemoryCluster {
            memories: vec![
                entry_with_fsrs("a", 2.0, 0.3, 4),
                entry_with_fsrs("b", 5.0, 0.1, 7),
            ],
            centroid: vec![1.0, 0.0],
            compression_level: CompressionLevel::Standard,
            avg_weight: 0.5,
        };
        let out = create_consolidated_entry(&cluster, "merged".into(), vec![1.0, 0.0]);
        assert!((out.fsrs.importance - 5.0).abs() < 1e-6, "importance = max");
        assert!(
            (out.fsrs.storage_strength - 0.3).abs() < 1e-6,
            "storage = max"
        );
        assert_eq!(out.fsrs.access_count, 11, "access counts summed");
        assert_eq!(
            out.metadata.get("consolidated_from").map(String::as_str),
            Some("a,b")
        );
    }

    #[test]
    fn consolidated_entry_inherits_cluster_workspace() {
        // A cluster scoped to a workspace produces a consolidated entry in that
        // same workspace — the merged memory stays put, never re-homed to global.
        let mut a = entry_with_fsrs("a", 2.0, 0.3, 4);
        let mut b = entry_with_fsrs("b", 5.0, 0.1, 7);
        a.workspace_id = Some("ws-A".to_string());
        b.workspace_id = Some("ws-A".to_string());
        let cluster = MemoryCluster {
            memories: vec![a, b],
            centroid: vec![1.0, 0.0],
            compression_level: CompressionLevel::Standard,
            avg_weight: 0.5,
        };
        let out = create_consolidated_entry(&cluster, "merged".into(), vec![1.0, 0.0]);
        assert_eq!(out.workspace_id.as_deref(), Some("ws-A"));
    }

    #[test]
    fn consolidated_entry_survives_nan_fsrs() {
        // A corrupt stored FSRS scalar must not panic dreaming; the result
        // still has finite scalars and inherits the finite sibling's importance.
        let cluster = MemoryCluster {
            memories: vec![
                entry_with_fsrs("a", f32::NAN, f32::NAN, 1),
                entry_with_fsrs("b", 3.0, 0.2, 2),
            ],
            centroid: vec![1.0, 0.0],
            compression_level: CompressionLevel::Standard,
            avg_weight: 0.5,
        };
        let out = create_consolidated_entry(&cluster, "merged".into(), vec![1.0, 0.0]);
        assert!(out.fsrs.importance.is_finite() && out.fsrs.storage_strength.is_finite());
        assert!((out.fsrs.importance - 3.0).abs() < 1e-6);
        assert!((out.fsrs.storage_strength - 0.2).abs() < 1e-6);
    }
}
