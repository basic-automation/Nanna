//! Dreaming Service - Periodic Memory Consolidation
//!
//! Implements the biological "dreaming" process where memories are:
//! - Compressed based on fading importance
//! - Expanded when highly valuable
//! - Clustered with similar memories
//! - Automatically promoted/demoted based on feedback

use crate::{
    ActivityClock, ConsolidationConfig, ConsolidationResult, EmbedFn, MemoryError, MemoryService,
    MemoryServiceConfig,
};
use crate::lossy::LossyF32;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Configuration for the dreaming service
#[derive(Debug, Clone)]
pub struct DreamingConfig {
    /// Memory service configuration
    pub memory: MemoryServiceConfig,
    /// Consolidation configuration
    pub consolidation: ConsolidationConfig,
    /// Auto-promote memories that are accessed frequently (threshold)
    pub auto_promote_access_threshold: u32,
    /// Auto-demote memories that are never accessed after N days
    pub auto_demote_days: f32,
    /// Minimum memories before consolidation runs
    pub min_memories_for_consolidation: usize,
    /// How long the system must be idle before a gated dream cycle may run,
    /// in seconds. Used by [`DreamingService::dream_if_idle`] — dreaming during
    /// active use competes with the agent for the GPU/LLM, so we wait for a lull.
    pub idle_threshold_secs: u64,
    /// Live memory count at/above which a gated dream cycle runs **regardless**
    /// of idle time (memory-pressure relief). `0` disables the pressure trigger.
    pub memory_pressure_count: usize,
}

impl Default for DreamingConfig {
    fn default() -> Self {
        Self {
            memory: MemoryServiceConfig::default(),
            consolidation: ConsolidationConfig::default(),
            auto_promote_access_threshold: 5,
            auto_demote_days: 30.0,
            min_memories_for_consolidation: 10,
            // 5 min idle before dreaming; relieve pressure past 5k live memories.
            idle_threshold_secs: 300,
            memory_pressure_count: 5000,
        }
    }
}

/// Why a gated dream cycle did (or did not) run — makes the gate observable and
/// unit-testable without an LLM in the loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DreamTrigger {
    /// Ran because the system was idle at least `idle_threshold_secs`.
    Idle,
    /// Ran because the live memory count reached `memory_pressure_count`
    /// (even though the system was not idle).
    MemoryPressure,
    /// Skipped: not idle yet and no memory pressure.
    Skipped,
}

/// Cross-check a gate decision against its inputs — the guard against a
/// future edit desyncing the policy from its two triggers. One copy, for both
/// entry points: it used to be pasted verbatim into each, where one could be
/// edited and the other not.
fn debug_check_gate(
    trigger: DreamTrigger,
    idle: Duration,
    memory_count: usize,
    config: &DreamingConfig,
) {
    debug_assert!(
        trigger != DreamTrigger::Skipped
            || (idle < Duration::from_secs(config.idle_threshold_secs)
                && (config.memory_pressure_count == 0
                    || memory_count < config.memory_pressure_count)),
        "Skipped must imply not-idle-yet AND below memory pressure"
    );
    debug_assert!(
        trigger != DreamTrigger::MemoryPressure
            || (config.memory_pressure_count > 0 && memory_count >= config.memory_pressure_count),
        "MemoryPressure must imply the pressure ceiling was reached"
    );
}

/// Pure gate decision: should a dream cycle run given how long the system has
/// been idle and how many memories are live?
///
/// Memory pressure takes priority so a continuously-busy system still
/// consolidates before the store grows without bound; otherwise idle time
/// governs. Pure (no clock/IO) so the policy is exhaustively testable.
///
/// Exported so the daemon scheduler gates its (standalone-`MemoryService`)
/// dream cycle on the **same** policy `DreamingService::dream_if_idle` uses —
/// one source of truth, no drift between the two consolidation paths.
#[must_use]
pub fn dream_trigger(idle: Duration, memory_count: usize, cfg: &DreamingConfig) -> DreamTrigger {
    if cfg.memory_pressure_count > 0 && memory_count >= cfg.memory_pressure_count {
        DreamTrigger::MemoryPressure
    } else if idle >= Duration::from_secs(cfg.idle_threshold_secs) {
        DreamTrigger::Idle
    } else {
        DreamTrigger::Skipped
    }
}

/// Feedback type for automatic promotion/demotion
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryFeedback {
    /// User indicated this was helpful (👍, "thanks", positive reaction)
    Helpful,
    /// User indicated this was wrong/unhelpful (👎, "that's wrong", correction)
    Unhelpful,
    /// Memory was successfully used in a task
    UsedSuccessfully,
    /// Memory led to an error or bad outcome
    CausedError,
}

/// FSRS weight adjustment for one feedback signal (positive promotes, negative
/// demotes). Single source of truth for both the immediate ([`DreamingService::apply_feedback`])
/// and the deferred dream-time aggregation paths, which must stay in lock-step.
#[must_use]
const fn feedback_boost(feedback: MemoryFeedback) -> f32 {
    match feedback {
        MemoryFeedback::Helpful => 0.3,
        MemoryFeedback::UsedSuccessfully => 0.5,
        MemoryFeedback::Unhelpful => -0.3,
        MemoryFeedback::CausedError => -0.5,
    }
}

/// Bounded per-memory tally of feedback signals pending the next dream cycle.
///
/// The dream loop only ever consumes the **aggregate** boost (a commutative
/// sum), so the signals themselves never need retaining: per-variant counts are
/// sufficient and exactly reproduce the sum. This bounds the accumulator **by
/// construction** — a fixed 16 bytes per memory regardless of flood volume
/// (Tiger Style: bound everything; the distinct-memory axis is bounded by the
/// store) — with no drop policy. A retain-N cap would mis-aggregate a
/// mixed-direction flood: N positives followed by many negatives would drop the
/// negatives and flip the applied sign. Counters saturate at `u32::MAX`
/// (~4.3 B same-variant signals between two dream cycles — unreachable).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct FeedbackTally {
    helpful: u32,
    unhelpful: u32,
    used_successfully: u32,
    caused_error: u32,
}

impl FeedbackTally {
    /// Count one feedback signal (saturating).
    const fn record(&mut self, feedback: MemoryFeedback) {
        let counter = match feedback {
            MemoryFeedback::Helpful => &mut self.helpful,
            MemoryFeedback::Unhelpful => &mut self.unhelpful,
            MemoryFeedback::UsedSuccessfully => &mut self.used_successfully,
            MemoryFeedback::CausedError => &mut self.caused_error,
        };
        *counter = counter.saturating_add(1);
    }

    /// Total signals tallied (for logging).
    const fn signal_count(&self) -> u64 {
        self.helpful as u64
            + self.unhelpful as u64
            + self.used_successfully as u64
            + self.caused_error as u64
    }

    /// Aggregate boost: Σ count × per-variant boost — identical to summing the
    /// individual signals, since the sum is commutative (fused multiply-add for
    /// one rounding per term).
    // Counts are exact in f32 below 2^24; beyond that the rounding error is
    // ~1e-7 relative, absorbed by the ±1.0 clamp the dream loop applies.
    fn total_boost(&self) -> f32 {
        let total = self.helpful.lossy_f32().mul_add(feedback_boost(MemoryFeedback::Helpful), 0.0);
        let total =
            self.unhelpful.lossy_f32().mul_add(feedback_boost(MemoryFeedback::Unhelpful), total);
        let total = self
            .used_successfully
            .lossy_f32()
            .mul_add(feedback_boost(MemoryFeedback::UsedSuccessfully), total);
        self.caused_error.lossy_f32().mul_add(feedback_boost(MemoryFeedback::CausedError), total)
    }
}

/// Statistics from a dreaming run
#[derive(Debug, Clone, Default)]
pub struct DreamingStats {
    /// Consolidation results
    pub consolidation: ConsolidationResult,
    /// Memories auto-promoted
    pub auto_promoted: usize,
    /// Memories auto-demoted
    pub auto_demoted: usize,
    /// Total memories after dreaming
    pub total_memories: usize,
}

/// Outcome of a gated dream cycle: which trigger fired, and what it did.
///
/// Returned instead of a bare `DreamingStats` so a caller can log *why* the
/// cycle ran (idle lull vs memory pressure) without re-deriving the decision.
#[derive(Debug, Clone)]
pub struct DreamOutcome {
    /// Why the cycle ran. Never [`DreamTrigger::Skipped`] — a skip is reported
    /// as `Ok(None)`, so this type cannot represent "ran because it didn't".
    pub trigger: DreamTrigger,
    /// What the cycle accomplished.
    pub stats: DreamingStats,
}

/// The dreaming service - manages memory consolidation and auto-feedback
pub struct DreamingService {
    config: DreamingConfig,
    memory: Arc<MemoryService>,
    /// Pending feedback to apply (`memory_id` -> per-variant signal tally)
    pending_feedback: RwLock<HashMap<String, FeedbackTally>>,
    /// Monotonic record of the most recent user/agent activity, driving the idle
    /// gate in [`DreamingService::dream_if_idle`]. An `Arc` so the host (the
    /// daemon's control plane) can stamp the **same** clock this service reads —
    /// see [`Self::with_activity_clock`].
    activity: Arc<ActivityClock>,
}

impl DreamingService {
    /// Create a new dreaming service over a **private** memory store.
    ///
    /// Prefer [`Self::with_shared_memory`] in the daemon so dreaming operates on
    /// the same live store the agent's `remember`/`recall` tools use — a store
    /// created here is isolated and will not see those memories.
    #[must_use]
    pub fn new(config: DreamingConfig) -> Self {
        let memory = Arc::new(MemoryService::new(config.memory.clone()));
        Self::from_parts(config, memory)
    }

    /// Create a dreaming service that consolidates an **existing, shared**
    /// memory store — the single source of truth for the running app.
    ///
    /// This is the seam that unifies the two memory stacks (P13): the daemon
    /// hands its live `Arc<MemoryService>` here so a dream cycle merges the same
    /// memories the agent wrote, instead of a throwaway private store. The
    /// shared store should already carry its own `embed_fn`/persistence — do not
    /// call [`Self::with_embed_fn`] on a service built this way.
    #[must_use]
    pub fn with_shared_memory(config: DreamingConfig, memory: Arc<MemoryService>) -> Self {
        Self::from_parts(config, memory)
    }

    /// Shared field initializer for both constructors (single mutation site).
    fn from_parts(config: DreamingConfig, memory: Arc<MemoryService>) -> Self {
        let service = Self {
            config,
            memory,
            pending_feedback: RwLock::new(HashMap::new()),
            activity: Arc::new(ActivityClock::new()),
        };
        // The store handle is retained (not dropped) for the service's lifetime.
        debug_assert!(Arc::strong_count(&service.memory) >= 1);
        // A fresh clock reads as active-at-construction, so a service built at
        // boot does not immediately consider itself idle-forever.
        debug_assert!(service.activity.idle() < Duration::from_secs(1));
        service
    }

    /// Read the idle signal from an **externally shared** clock instead of this
    /// service's private one.
    ///
    /// This is the other half of the P13 unification: the daemon's control plane
    /// already stamps an [`ActivityClock`] on every chat request, so handing that
    /// same `Arc` here means the service and its host cannot disagree about
    /// whether the system is in use. Without it the host would have to call
    /// [`Self::record_activity`] as a *second* stamp on every request and the two
    /// notions of "last activity" could drift.
    #[must_use]
    pub fn with_activity_clock(mut self, clock: Arc<ActivityClock>) -> Self {
        self.activity = clock;
        self
    }

    /// Clone the activity-clock handle this service gates on, so a host can stamp
    /// the same clock it reads.
    #[must_use]
    pub fn activity_clock(&self) -> Arc<ActivityClock> {
        Arc::clone(&self.activity)
    }

    /// Clone the shared memory-store handle (e.g. to hand the daemon's tools the
    /// very store this service dreams over).
    #[must_use]
    pub fn memory_arc(&self) -> Arc<MemoryService> {
        Arc::clone(&self.memory)
    }

    /// Mark "now" as the most recent activity, resetting the idle timer.
    ///
    /// Call this whenever the agent handles a message or runs a tool so a dream
    /// cycle only fires during a genuine lull (see [`Self::dream_if_idle`]).
    /// Lock-free — safe to call from a hot request path.
    pub fn record_activity(&self) {
        self.activity.record();
    }

    /// How long the system has been idle since the last [`Self::record_activity`]
    /// (or since the shared clock was last stamped by the host).
    #[must_use]
    pub fn idle_duration(&self) -> Duration {
        self.activity.idle()
    }

    /// Run a dream cycle **only if** the idle/memory-pressure gate says so,
    /// using an **externally supplied** idle duration and consolidation budget.
    ///
    /// This is the gated entry point for a host that already owns an activity
    /// clock — the daemon stamps a lock-free [`AtomicU64`-backed] clock on every
    /// chat, and must not pay for this service's `RwLock<Instant>` on that hot
    /// path. Passing `idle` in keeps **one** activity source of truth while the
    /// *policy* ([`dream_trigger`]) and the *orchestration* (feedback flush →
    /// FSRS flush → consolidation) stay here, so the scheduled and manual paths
    /// cannot drift apart.
    ///
    /// `consolidation` is a parameter rather than `self.config.consolidation`
    /// because each caller sizes the cluster byte budget to **its own**
    /// summarizer model's context window.
    ///
    /// Returns `Ok(Some((trigger, stats)))` if a cycle ran, `Ok(None)` if the
    /// gate skipped it.
    ///
    /// [`AtomicU64`-backed]: std::sync::atomic::AtomicU64
    ///
    /// # Errors
    ///
    /// Returns `MemoryError` if a triggered consolidation fails.
    pub async fn dream_if_triggered<F, Fut>(
        &self,
        idle: Duration,
        consolidation: &ConsolidationConfig,
        summarize_fn: F,
    ) -> Result<Option<(DreamTrigger, DreamingStats)>, MemoryError>
    where
        F: Fn(String) -> Fut + Send + Sync,
        Fut: std::future::Future<Output = Result<String, String>> + Send,
    {
        let memory_count = self.memory.stats().await.total;
        let trigger = dream_trigger(idle, memory_count, &self.config);

        debug_check_gate(trigger, idle, memory_count, &self.config);

        if trigger == DreamTrigger::Skipped {
            debug!(
                "dream gate: skipped (idle {:?} < {}s, {} memories)",
                idle, self.config.idle_threshold_secs, memory_count
            );
            return Ok(None);
        }

        info!("dream gate: running ({trigger:?}; idle {idle:?}, {memory_count} memories)");
        let stats = self
            .dream_with_consolidation(consolidation, summarize_fn)
            .await?;
        Ok(Some((trigger, stats)))
    }

    /// Run a dream cycle **only if** the idle/memory-pressure gate says so.
    ///
    /// Returns `Ok(Some(stats))` with the trigger reason if a cycle ran, or
    /// `Ok(None)` if it was skipped because the system is still active and the
    /// store isn't under pressure. This is the idle-gated entry point the
    /// scheduler should call instead of the unconditional [`Self::dream`].
    ///
    /// Reads this service's own activity clock; a host with its own clock
    /// should call [`Self::dream_if_triggered`] instead.
    ///
    /// # Errors
    ///
    /// Returns `MemoryError` if a triggered consolidation fails.
    pub async fn dream_if_idle<F, Fut>(
        &self,
        summarize_fn: F,
    ) -> Result<Option<DreamOutcome>, MemoryError>
    where
        F: Fn(String) -> Fut + Send + Sync,
        Fut: std::future::Future<Output = Result<String, String>> + Send,
    {
        self.dream_if_idle_with(&self.config.consolidation, summarize_fn)
            .await
    }

    /// [`Self::dream_if_idle`] with the consolidation limits supplied per run.
    ///
    /// The cluster byte budget must be sized to the *summarizer model's* context
    /// window, which a long-lived service cannot know at construction time (the
    /// model is resolved from the router when the cycle fires). Taking the
    /// config as an argument keeps that decision at the call site instead of
    /// freezing a stale budget into the service.
    ///
    /// # Errors
    ///
    /// Returns `MemoryError` if a triggered consolidation fails.
    pub async fn dream_if_idle_with<F, Fut>(
        &self,
        consolidation: &ConsolidationConfig,
        summarize_fn: F,
    ) -> Result<Option<DreamOutcome>, MemoryError>
    where
        F: Fn(String) -> Fut + Send + Sync,
        Fut: std::future::Future<Output = Result<String, String>> + Send,
    {
        let idle = self.idle_duration();
        let memory_count = self.memory.stats().await.total;
        let trigger = dream_trigger(idle, memory_count, &self.config);

        debug_check_gate(trigger, idle, memory_count, &self.config);

        if trigger == DreamTrigger::Skipped {
            debug!(
                "dream_if_idle: skipped (idle {:?} < {}s, {} memories)",
                idle, self.config.idle_threshold_secs, memory_count
            );
            return Ok(None);
        }

        info!("dream_if_idle: running ({trigger:?}; idle {idle:?}, {memory_count} memories)");
        let stats = self.dream_with(consolidation, summarize_fn).await?;
        debug_assert!(
            trigger != DreamTrigger::Skipped,
            "a reported outcome must carry the trigger that actually ran"
        );
        Ok(Some(DreamOutcome { trigger, stats }))
    }

    /// Set the embedding function on the underlying memory store.
    ///
    /// Only valid when this service **owns** its memory store uniquely (i.e. was
    /// built with [`Self::new`], before any [`Self::memory_arc`] clone escaped).
    /// A service built with [`Self::with_shared_memory`] must configure its
    /// `embed_fn` on the shared store before sharing it — setting it here would
    /// be a no-op against a `&mut` we cannot obtain, so that is asserted against.
    #[must_use]
    pub fn with_embed_fn(mut self, f: EmbedFn) -> Self {
        match Arc::get_mut(&mut self.memory) {
            Some(mem) => mem.set_embed_fn(f),
            None => debug_assert!(
                false,
                "with_embed_fn requires unique ownership; configure the shared \
                 store before with_shared_memory"
            ),
        }
        self
    }

    /// Get reference to the underlying memory service
    #[must_use]
    pub fn memory(&self) -> &MemoryService {
        &self.memory
    }

    /// Record feedback for a memory (will be applied during dreaming).
    ///
    /// Tallied into a fixed-size per-memory `FeedbackTally` — every signal
    /// counts toward the aggregate the dream cycle applies, and the accumulator
    /// cannot grow with flood volume.
    pub async fn record_feedback(&self, memory_id: &str, feedback: MemoryFeedback) {
        let mut pending = self.pending_feedback.write().await;
        pending
            .entry(memory_id.to_string())
            .or_default()
            .record(feedback);
        // Release the write guard before logging (nothing below touches it).
        drop(pending);

        debug!("Recorded {:?} feedback for memory {}", feedback, memory_id);
    }

    /// The boost the next dream cycle will apply to `memory_id` from feedback
    /// recorded so far; `None` when nothing is pending for it.
    pub async fn pending_feedback_boost(&self, memory_id: &str) -> Option<f32> {
        self.pending_feedback
            .read()
            .await
            .get(memory_id)
            .map(FeedbackTally::total_boost)
    }

    /// Apply pending feedback immediately (doesn't wait for dreaming)
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::NotFound`] when no memory has `memory_id`.
    pub async fn apply_feedback(
        &self,
        memory_id: &str,
        feedback: MemoryFeedback,
    ) -> Result<(), MemoryError> {
        let boost = feedback_boost(feedback);

        if boost > 0.0 {
            self.memory.promote(memory_id, boost).await?;
        } else {
            self.memory.demote(memory_id, boost.abs()).await?;
        }

        info!(
            "Applied {:?} feedback to memory {} (boost: {})",
            feedback, memory_id, boost
        );
        Ok(())
    }

    /// Run the dreaming process (memory consolidation) with this service's
    /// configured consolidation budget.
    ///
    /// This should be called periodically (e.g., hourly or during idle times).
    ///
    /// # Arguments
    /// * `summarize_fn` - Async function that takes a prompt and returns summarized text (LLM call)
    ///
    /// # Errors
    ///
    /// Returns `MemoryError` if consolidation fails.
    pub async fn dream<F, Fut>(&self, summarize_fn: F) -> Result<DreamingStats, MemoryError>
    where
        F: Fn(String) -> Fut + Send + Sync,
        Fut: std::future::Future<Output = Result<String, String>> + Send,
    {
        self.dream_with_consolidation(&self.config.consolidation, summarize_fn)
            .await
    }

    /// Run the dreaming process against a caller-supplied consolidation budget.
    ///
    /// The full cycle, in order: **(1)** apply tallied feedback (promote/demote),
    /// **(2)** flush pending FSRS updates — the *testing effect* queued by every
    /// `recall`, which is only ever drained here — **(3)** bail out below
    /// `min_memories_for_consolidation`, **(4)** consolidate (rank-similar →
    /// concatenate → summarize).
    ///
    /// This is the single orchestration body every dreaming entry point runs
    /// through. Calling [`MemoryService::consolidate`] directly **skips steps
    /// 1–3** and leaks the pending-update queue, so hosts must come through here.
    ///
    /// `consolidation` is a parameter so each caller can size the cluster byte
    /// budget to its own summarizer model's context window.
    ///
    /// # Errors
    ///
    /// Returns `MemoryError` if consolidation fails.
    pub async fn dream_with_consolidation<F, Fut>(
        &self,
        consolidation: &ConsolidationConfig,
        summarize_fn: F,
    ) -> Result<DreamingStats, MemoryError>
    where
        F: Fn(String) -> Fut + Send + Sync,
        Fut: std::future::Future<Output = Result<String, String>> + Send,
    {
        self.dream_with(consolidation, summarize_fn).await
    }

    /// [`Self::dream`] with the consolidation limits supplied per run — the
    /// single implementation both entry points delegate to.
    ///
    /// # Errors
    ///
    /// Returns `MemoryError` if consolidation fails.
    pub async fn dream_with<F, Fut>(
        &self,
        consolidation: &ConsolidationConfig,
        summarize_fn: F,
    ) -> Result<DreamingStats, MemoryError>
    where
        F: Fn(String) -> Fut + Send + Sync,
        Fut: std::future::Future<Output = Result<String, String>> + Send,
    {
        let mut stats = DreamingStats::default();

        // 1. Apply pending feedback
        let pending = {
            let mut pending = self.pending_feedback.write().await;
            std::mem::take(&mut *pending)
        };

        for (memory_id, tally) in pending {
            // Aggregate feedback (Σ count × per-variant boost)
            let total_boost = tally.total_boost();
            debug!(
                "Applying {} tallied feedback signals to {} (boost {total_boost})",
                tally.signal_count(),
                memory_id
            );

            // Apply aggregated feedback
            if total_boost > 0.0 {
                if let Err(e) = self.memory.promote(&memory_id, total_boost.min(1.0)).await {
                    warn!("Failed to apply feedback to {}: {}", memory_id, e);
                } else {
                    stats.auto_promoted += 1;
                }
            } else if total_boost < 0.0 {
                if let Err(e) = self
                    .memory
                    .demote(&memory_id, total_boost.abs().min(1.0))
                    .await
                {
                    warn!("Failed to apply feedback to {}: {}", memory_id, e);
                } else {
                    stats.auto_demoted += 1;
                }
            }
        }

        // 2. Apply FSRS updates (testing effect from recalls).
        //    `MemoryService::recall` queues one review per hit and this is the
        //    ONLY drain — skipping it both inerts the testing effect and lets
        //    the queue grow unbounded for the host's whole uptime.
        self.memory.apply_pending_updates().await;

        // 3. Check if we have enough memories for consolidation
        let count = self.memory.count().await;
        if count < self.config.min_memories_for_consolidation {
            info!(
                "Skipping consolidation: only {} memories (need {})",
                count, self.config.min_memories_for_consolidation
            );
            stats.total_memories = count;
            return Ok(stats);
        }

        // 4. Run consolidation (the actual "dreaming")
        info!("Starting memory consolidation ({} memories)...", count);
        stats.consolidation = self.memory.consolidate(consolidation, summarize_fn).await?;

        stats.total_memories = self.memory.count().await;

        info!(
            "Dreaming complete: {} processed, {} merged, {} expanded, {} promoted, {} demoted",
            stats.consolidation.memories_processed,
            stats.consolidation.memories_merged,
            stats.consolidation.memories_expanded,
            stats.auto_promoted,
            stats.auto_demoted,
        );

        Ok(stats)
    }

    /// Remember something (delegates to memory service)
    ///
    /// # Errors
    ///
    /// Returns the errors of [`crate::MemoryService::remember`]: the store write
    /// fails (e.g. a vector whose width does not match the store). A missing or
    /// failing embedding provider is not an error.
    pub async fn remember(
        &self,
        content: &str,
        metadata: HashMap<String, String>,
    ) -> Result<String, MemoryError> {
        self.memory.remember(content, metadata).await
    }

    /// Recall memories (delegates to memory service)
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::NoEmbeddingProvider`] when no embedding function is
    /// configured, and [`MemoryError::Io`] when embedding the query fails.
    pub async fn recall(&self, query: &str) -> Result<Vec<crate::RecallResult>, MemoryError> {
        self.memory.recall(query).await
    }

    /// Forget a memory (delegates to memory service)
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::NotFound`] when no memory has `id`, and the
    /// backend's error when it could not delete the row (the memory is then
    /// kept, so RAM and disk still agree).
    pub async fn forget(&self, id: &str) -> Result<(), MemoryError> {
        self.memory.forget(id).await
    }

    /// Get memory statistics
    pub async fn stats(&self) -> crate::MemoryStats {
        self.memory.stats().await
    }

    /// Save memories to file
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::Serialization`] when the entries cannot be encoded
    /// and [`MemoryError::Io`] when the temporary file cannot be written or
    /// renamed over `path`.
    pub async fn save(&self, path: &std::path::Path) -> Result<(), MemoryError> {
        self.memory.save(path).await
    }

    /// Load memories from file
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::Io`] when the file cannot be read and
    /// [`MemoryError::Serialization`] when it is not a JSON array of memory
    /// entries.
    pub async fn load(&self, path: &std::path::Path) -> Result<(), MemoryError> {
        self.memory.load(path).await
    }
}

/// Create a summarization function from an LLM client
///
/// This is a helper to create the `summarize_fn` argument for `dream()`.
pub fn make_summarize_fn<C>(
    llm: Arc<C>,
    model: String,
) -> impl Fn(String) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String, String>> + Send>>
where
    C: LlmSummarizer + Send + Sync + 'static,
{
    move |prompt: String| {
        let llm = llm.clone();
        let model = model.clone();
        Box::pin(async move { llm.summarize(&model, &prompt).await })
    }
}

/// Trait for LLM summarization (implemented by `LlmClient`)
///
/// This is a simple trait that any LLM client can implement for memory consolidation.
pub trait LlmSummarizer: Send + Sync {
    /// Summarize/consolidate the given prompt into a condensed memory.
    fn summarize(
        &self,
        model: &str,
        prompt: &str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String, String>> + Send + '_>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_dreaming_service_creation() {
        let service = DreamingService::new(DreamingConfig::default());
        let stats = service.stats().await;
        assert_eq!(stats.total, 0);
    }

    #[tokio::test]
    async fn test_feedback_recording() {
        let service = DreamingService::new(DreamingConfig::default());

        service
            .record_feedback("mem-1", MemoryFeedback::Helpful)
            .await;
        service
            .record_feedback("mem-1", MemoryFeedback::UsedSuccessfully)
            .await;
        service
            .record_feedback("mem-2", MemoryFeedback::Unhelpful)
            .await;

        let pending = service.pending_feedback.read().await;
        assert_eq!(
            pending.get("mem-1").map(FeedbackTally::signal_count),
            Some(2)
        );
        assert_eq!(
            pending.get("mem-2").map(FeedbackTally::signal_count),
            Some(1)
        );
    }

    #[tokio::test]
    async fn feedback_flood_is_fixed_size_and_exactly_aggregated() {
        let service = DreamingService::new(DreamingConfig::default());

        // The mixed-direction flood a retain-N cap gets WRONG: 16 positives
        // followed by 20 strong negatives. Dropping the negatives past a cap
        // would leave the aggregate positive; the true sum is
        // 16(0.3) − 20(0.5) = −5.2, i.e. firmly negative.
        for _ in 0..16 {
            service
                .record_feedback("hot", MemoryFeedback::Helpful)
                .await;
        }
        for _ in 0..20 {
            service
                .record_feedback("hot", MemoryFeedback::CausedError)
                .await;
        }

        let tally = {
            let pending = service.pending_feedback.read().await;
            *pending.get("hot").expect("tally must exist")
        };
        // Every signal counted — nothing dropped…
        assert_eq!(tally.signal_count(), 36);
        // …the accumulator is a fixed-size tally regardless of volume…
        assert_eq!(std::mem::size_of::<FeedbackTally>(), 16);
        // …and the aggregate is the exact sum, with the correct (negative) sign.
        assert!((tally.total_boost() - (-5.2)).abs() < 1e-4);
    }

    #[test]
    fn feedback_tally_counters_saturate_instead_of_wrapping() {
        // A counter at u32::MAX must stay there (saturating), never wrap to 0
        // and silently zero out billions of signals.
        let mut tally = FeedbackTally {
            helpful: u32::MAX,
            ..FeedbackTally::default()
        };
        tally.record(MemoryFeedback::Helpful);
        assert_eq!(tally.helpful, u32::MAX);
    }

    #[test]
    fn feedback_boost_signs_match_semantics() {
        // Positive signals promote, negative signals demote — the two paths that
        // consume this table (immediate + dream-time) depend on the signs.
        assert!(feedback_boost(MemoryFeedback::Helpful) > 0.0);
        assert!(feedback_boost(MemoryFeedback::UsedSuccessfully) > 0.0);
        assert!(feedback_boost(MemoryFeedback::Unhelpful) < 0.0);
        assert!(feedback_boost(MemoryFeedback::CausedError) < 0.0);
    }

    #[test]
    fn tally_total_boost_matches_signal_by_signal_sum() {
        // The tally must aggregate exactly like summing the individual signals
        // (the representation swap must not change the applied result).
        let signals = [
            MemoryFeedback::Helpful,
            MemoryFeedback::Helpful,
            MemoryFeedback::CausedError,
            MemoryFeedback::UsedSuccessfully,
            MemoryFeedback::Unhelpful,
        ];
        let mut tally = FeedbackTally::default();
        let mut reference = 0.0_f32;
        for s in signals {
            tally.record(s);
            reference += feedback_boost(s);
        }
        assert!((tally.total_boost() - reference).abs() < 1e-6);
    }

    fn gate_cfg(idle_secs: u64, pressure: usize) -> DreamingConfig {
        DreamingConfig {
            idle_threshold_secs: idle_secs,
            memory_pressure_count: pressure,
            ..DreamingConfig::default()
        }
    }

    #[test]
    fn dream_trigger_idle_boundary() {
        let cfg = gate_cfg(300, 0);
        // Just under the threshold → skipped.
        assert_eq!(
            dream_trigger(Duration::from_secs(299), 0, &cfg),
            DreamTrigger::Skipped
        );
        // Exactly at the threshold → idle-triggered.
        assert_eq!(
            dream_trigger(Duration::from_secs(300), 0, &cfg),
            DreamTrigger::Idle
        );
    }

    #[test]
    fn dream_trigger_memory_pressure_overrides_activity() {
        let cfg = gate_cfg(300, 100);
        // Not idle at all, but the store is at the pressure ceiling → run anyway.
        assert_eq!(
            dream_trigger(Duration::from_secs(0), 100, &cfg),
            DreamTrigger::MemoryPressure
        );
        // Below the ceiling and not idle → skipped.
        assert_eq!(
            dream_trigger(Duration::from_secs(1), 99, &cfg),
            DreamTrigger::Skipped
        );
    }

    #[test]
    fn dream_trigger_pressure_disabled_by_zero() {
        let cfg = gate_cfg(300, 0);
        // memory_pressure_count == 0 disables the pressure trigger regardless of count.
        assert_eq!(
            dream_trigger(Duration::from_secs(1), 1_000_000, &cfg),
            DreamTrigger::Skipped
        );
    }

    #[tokio::test]
    async fn dream_if_idle_skips_when_active() {
        use std::sync::atomic::{AtomicBool, Ordering};

        // High idle threshold so a freshly-created service is never "idle".
        let service = DreamingService::new(gate_cfg(3600, 0));
        service.record_activity();

        let called = AtomicBool::new(false);
        let result = service
            .dream_if_idle(|_prompt| {
                called.store(true, Ordering::SeqCst);
                async { Ok(String::new()) }
            })
            .await
            .expect("gate must not error");

        assert!(result.is_none(), "an active service must not dream");
        assert!(
            !called.load(Ordering::SeqCst),
            "summarize_fn must not be invoked when the cycle is skipped"
        );
    }

    /// A dimension-3 embedder good enough to let `remember` run in tests.
    fn test_embed_fn() -> EmbedFn {
        Arc::new(|_text: &str| Box::pin(async { Ok(vec![1.0_f32, 0.0, 0.0]) }))
    }

    /// A dimension-3 embedder that maps texts ending in `0..=3` onto four
    /// mutually non-similar directions (pairwise cosine ≤ 0), so `remember`
    /// stores them as **distinct** memories instead of folding them into one by
    /// the >0.9-similarity dedup rule. Anything else lands on a fifth direction.
    fn distinct_embed_fn() -> EmbedFn {
        Arc::new(|text: &str| {
            let vector = match text.chars().last() {
                Some('0') => vec![1.0_f32, 0.0, 0.0],
                Some('1') => vec![0.0_f32, 1.0, 0.0],
                Some('2') => vec![0.0_f32, 0.0, 1.0],
                Some('3') => vec![-1.0_f32, 0.0, 0.0],
                _ => vec![0.0_f32, -1.0, 0.0],
            };
            Box::pin(async move { Ok(vector) })
        })
    }

    fn dim3_config() -> DreamingConfig {
        DreamingConfig {
            memory: MemoryServiceConfig {
                dimension: 3,
                ..MemoryServiceConfig::default()
            },
            ..DreamingConfig::default()
        }
    }

    #[tokio::test]
    async fn with_shared_memory_sees_writes_through_the_shared_arc() {
        // Build a shared store, configure it, and hand it to the dreaming service.
        let shared =
            Arc::new(MemoryService::new(dim3_config().memory).with_embed_fn(test_embed_fn()));
        let service = DreamingService::with_shared_memory(dim3_config(), Arc::clone(&shared));

        // A memory written directly through the shared handle (as the daemon's
        // `remember` tool would) must be visible to the dream service's store —
        // proving they are the *same* store, not two isolated ones.
        assert_eq!(service.memory().count().await, 0);
        shared
            .remember("shared fact", HashMap::new())
            .await
            .expect("remember must succeed");
        assert_eq!(
            service.memory().count().await,
            1,
            "dreaming must observe writes made through the shared Arc"
        );
        // The handle round-trips back out identical.
        assert!(Arc::ptr_eq(&service.memory_arc(), &shared));
    }

    /// The regression this whole unification exists for.
    ///
    /// `recall` queues one FSRS "review" per hit (the testing effect) and
    /// `apply_pending_updates` is the ONLY drain. The daemon used to call
    /// `MemoryService::consolidate` directly, which never drains it — so the
    /// testing effect was inert and the queue grew for the process's whole
    /// uptime. A dream cycle must drain it.
    #[tokio::test]
    async fn dream_drains_the_pending_fsrs_queue_that_consolidate_alone_leaks() {
        let shared =
            Arc::new(MemoryService::new(dim3_config().memory).with_embed_fn(test_embed_fn()));
        let service = DreamingService::with_shared_memory(dim3_config(), Arc::clone(&shared));

        shared
            .remember("a durable fact", HashMap::new())
            .await
            .expect("remember must succeed");

        // Nothing recalled yet → nothing queued (negative space).
        assert_eq!(shared.pending_update_count().await, 0);

        // Three recalls enqueue reviews.
        for _ in 0..3 {
            shared.recall("a durable fact").await.expect("recall works");
        }
        let queued = shared.pending_update_count().await;
        assert!(queued > 0, "recall must queue testing-effect reviews");

        // The OLD shape: consolidating directly leaves every review queued.
        shared
            .consolidate(&ConsolidationConfig::default(), |_p| async {
                Ok(String::from("summary"))
            })
            .await
            .expect("consolidate must succeed");
        assert_eq!(
            shared.pending_update_count().await,
            queued,
            "MemoryService::consolidate must NOT drain the queue — if this \
             starts passing, the leak moved and this test guards the wrong seam"
        );

        // The NEW shape: a dream cycle drains it.
        service
            .dream_with_consolidation(&ConsolidationConfig::default(), |_p| async {
                Ok(String::from("summary"))
            })
            .await
            .expect("dream must succeed");
        assert_eq!(
            shared.pending_update_count().await,
            0,
            "a dream cycle must flush every queued testing-effect review"
        );
    }

    #[tokio::test]
    async fn dream_if_triggered_uses_the_callers_idle_not_its_own_clock() {
        use std::sync::atomic::{AtomicBool, Ordering};

        // 1 h threshold, and the service's OWN clock was just stamped — so
        // `dream_if_idle` would skip. Passing a larger idle must still run,
        // proving the daemon's lock-free ActivityClock is the source of truth.
        let service = DreamingService::new(gate_cfg(3600, 0));
        service.record_activity();

        let called = AtomicBool::new(false);
        let ran = service
            .dream_if_triggered(
                Duration::from_secs(7200),
                &ConsolidationConfig::default(),
                |_p| {
                    called.store(true, Ordering::SeqCst);
                    async { Ok(String::new()) }
                },
            )
            .await
            .expect("gate must not error");

        let (trigger, _stats) = ran.expect("a 2 h idle must trigger a cycle");
        assert_eq!(trigger, DreamTrigger::Idle);
        // Store is empty, so it stops at the min-memories guard before any LLM call.
        assert!(
            !called.load(Ordering::SeqCst),
            "an empty store must not reach the summarizer"
        );
    }

    #[tokio::test]
    async fn dream_if_triggered_skips_when_the_caller_reports_activity() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let service = DreamingService::new(gate_cfg(3600, 0));
        let called = AtomicBool::new(false);

        let ran = service
            .dream_if_triggered(
                Duration::from_secs(5),
                &ConsolidationConfig::default(),
                |_p| {
                    called.store(true, Ordering::SeqCst);
                    async { Ok(String::new()) }
                },
            )
            .await
            .expect("gate must not error");

        assert!(ran.is_none(), "an active system must not dream");
        assert!(!called.load(Ordering::SeqCst));
    }

    /// Each daemon path sizes the cluster byte budget to *its own* summarizer's
    /// context window, so the per-call config must win over `self.config`.
    #[tokio::test]
    async fn dream_with_consolidation_honors_the_caller_supplied_budget() {
        let base = dim3_config();
        // Service-level config carries a distinctive budget…
        let service = DreamingService::new(DreamingConfig {
            consolidation: ConsolidationConfig {
                min_remaining_memories: 999,
                ..base.consolidation.clone()
            },
            ..base
        });
        // …and the caller supplies a different one.
        let caller = ConsolidationConfig {
            min_remaining_memories: 7,
            ..ConsolidationConfig::default()
        };
        assert_ne!(
            service.config.consolidation.min_remaining_memories,
            caller.min_remaining_memories
        );

        // Empty store: the cycle returns before consolidating, but the call
        // must compile+run against the caller's config, not the service's.
        let stats = service
            .dream_with_consolidation(&caller, |_p| async { Ok(String::new()) })
            .await
            .expect("dream must succeed");
        assert_eq!(stats.total_memories, 0);
        // `dream()` still routes through the service-level config.
        let via_default = service
            .dream(|_p| async { Ok(String::new()) })
            .await
            .expect("dream must succeed");
        assert_eq!(via_default.total_memories, 0);
    }

    #[tokio::test]
    async fn shared_activity_clock_opens_and_shuts_the_gate_from_the_host_side() {
        // The unification invariant: a host that stamps its OWN clock must move
        // this service's idle gate, without ever calling `record_activity`.
        let clock = Arc::new(ActivityClock::new());
        // Threshold 0 ⇒ any idle duration counts as idle, so the gate is open…
        let open = DreamingService::new(gate_cfg(0, 0))
            .with_activity_clock(Arc::clone(&clock))
            .with_embed_fn(test_embed_fn());
        assert!(
            open.idle_duration() < Duration::from_secs(1),
            "a freshly stamped shared clock reads as recently active"
        );

        // …and a service with a high threshold sharing the same clock is shut,
        // because the host's stamp is what it reads.
        let shut = DreamingService::new(gate_cfg(3600, 0)).with_activity_clock(Arc::clone(&clock));
        clock.record();
        let called = std::sync::atomic::AtomicBool::new(false);
        let skipped = shut
            .dream_if_idle(|_p| {
                called.store(true, std::sync::atomic::Ordering::SeqCst);
                async { Ok(String::new()) }
            })
            .await
            .expect("gate must not error");
        assert!(
            skipped.is_none(),
            "the host's stamp must hold the gate shut without record_activity()"
        );
        assert!(!called.load(std::sync::atomic::Ordering::SeqCst));

        // The handle round-trips, so a host can adopt the service's clock too.
        assert!(Arc::ptr_eq(&shut.activity_clock(), &clock));
    }

    #[tokio::test]
    async fn gated_cycle_reports_the_trigger_that_fired() {
        // Memory pressure must be distinguishable from an idle lull in the
        // outcome, so a caller can log *why* a cycle ran.
        let shared =
            Arc::new(MemoryService::new(dim3_config().memory).with_embed_fn(distinct_embed_fn()));
        for i in 0..3 {
            shared
                .remember(&format!("fact {i}"), HashMap::new())
                .await
                .expect("remember must succeed");
        }
        // Never idle (1h threshold), but pressure at 1 memory ⇒ MemoryPressure.
        let cfg = DreamingConfig {
            memory: dim3_config().memory,
            idle_threshold_secs: 3600,
            memory_pressure_count: 1,
            // Below the consolidation floor, so the cycle short-circuits before
            // ever needing an LLM — we are asserting the trigger, not the merge.
            min_memories_for_consolidation: 1_000,
            ..DreamingConfig::default()
        };
        let service = DreamingService::with_shared_memory(cfg, shared);
        service.record_activity();

        let outcome = service
            .dream_if_idle(|_p| async { Ok(String::new()) })
            .await
            .expect("cycle must not error")
            .expect("memory pressure must force a run despite recent activity");
        assert_eq!(outcome.trigger, DreamTrigger::MemoryPressure);
        assert_eq!(outcome.stats.total_memories, 3);
    }

    #[tokio::test]
    async fn per_run_consolidation_config_overrides_the_construction_time_one() {
        // The service is built with a permissive budget but the call supplies a
        // restrictive one; the call site must win, since only it knows the
        // summarizer model's real context window.
        let permissive = ConsolidationConfig {
            min_remaining_memories: 0,
            ..ConsolidationConfig::default()
        };
        let restrictive = ConsolidationConfig {
            // A floor above the store size leaves zero removal budget, so the
            // cycle provably cannot merge anything.
            min_remaining_memories: usize::MAX,
            ..ConsolidationConfig::default()
        };
        let cfg = DreamingConfig {
            memory: dim3_config().memory,
            consolidation: permissive,
            min_memories_for_consolidation: 1,
            idle_threshold_secs: 0,
            ..DreamingConfig::default()
        };
        let shared =
            Arc::new(MemoryService::new(dim3_config().memory).with_embed_fn(distinct_embed_fn()));
        for i in 0..4 {
            shared
                .remember(&format!("same-ish fact {i}"), HashMap::new())
                .await
                .expect("remember must succeed");
        }
        let service = DreamingService::with_shared_memory(cfg, Arc::clone(&shared));

        let stats = service
            .dream_with(&restrictive, |_p| async { Ok("merged".to_string()) })
            .await
            .expect("cycle must not error");
        assert_eq!(
            stats.consolidation.memories_merged, 0,
            "the per-run floor must gate the merge, not the construction-time config"
        );
        assert_eq!(shared.count().await, 4, "no memory may be removed");
    }

    #[tokio::test]
    async fn new_builds_an_isolated_store() {
        // A privately-constructed service must NOT share state with an unrelated
        // store — writes elsewhere are invisible.
        let service = DreamingService::new(dim3_config()).with_embed_fn(test_embed_fn());
        let other =
            Arc::new(MemoryService::new(dim3_config().memory).with_embed_fn(test_embed_fn()));
        other.remember("elsewhere", HashMap::new()).await.unwrap();

        assert_eq!(service.memory().count().await, 0);
        assert!(!Arc::ptr_eq(&service.memory_arc(), &other));
    }
}
