//! Cognitive memory decay, in FSRS's *shape* — not an FSRS-6 implementation.
//!
//! The heading this module carried for a long time said "FSRS-6 implementation,
//! based on the FSRS-6 algorithm", and that overclaimed in a way that made the
//! weight table below look like a transcription bug. What is actually borrowed
//! is FSRS-6's **power-law forgetting curve**, `R(t, S) = (1 + factor*t/S)^-decay`
//! — the one formula whose published constant therefore transfers directly, and
//! the reason `w20` carries FSRS-6's decay while the rest of the table does not.
//! See <https://github.com/open-spaced-repetition/fsrs4anki> for the real thing.
//!
//! The stability and difficulty updates are Nanna's own: FSRS schedules *study
//! reviews* of cards a person grades, while this schedules the decay of
//! memories whose "review" is an incidental recall. Same curve, different
//! update rules — so FSRS's fitted weights for those rules are not this
//! module's to adopt, and the ones sitting in [`FsrsParameters`] unread are
//! evidence of that history rather than of a table waiting to be finished.
//! `only_the_live_weights_change_any_fsrs_output` pins exactly which eight of
//! the twenty-one are wired to anything.
//!
//! Key concepts:
//! - Stability (S): Time (in days) for retrievability to drop to 90%
//! - Retrievability (R): Probability of successful recall (0-1)
//! - Difficulty (D): Inherent difficulty of the memory (1-10)

use crate::lossy::LossyF32;
use serde::{Deserialize, Serialize};

/// The 21 FSRS weights, in FSRS-6's slot order.
///
/// [`Default`] is **not** FSRS-6's published weight table — only `w20` is (see
/// its field doc). The rest are FSRS-5 values with six slots zeroed.
///
/// **Only eight slots are read by anything: `w6..=w12` and `w20`.** The other
/// thirteen are public, serializable fields that look like tuning knobs and
/// turn nothing — and several are non-zero (`w0 = 0.4072`, `w16 = 2.2035`),
/// which is what makes the appearance convincing. That is measured, not
/// asserted: `only_the_live_weights_change_any_fsrs_output` perturbs every
/// slot and checks whether any observable number moves, so this doc cannot go
/// stale without a test failing.
///
/// They are kept rather than deleted because the slot *order* is FSRS's and
/// renumbering it would make every reference to "w20" ambiguous against the
/// published algorithm. Adopting the published table for the unread ones is an
/// open decision (ROADMAP P13), not an oversight: Nanna's stability update is
/// not FSRS's, so transcribing its constants into a differently-shaped formula
/// would be cargo-culting, not correctness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FsrsParameters {
    /// Initial stability for first review (w0-w3 for different ratings)
    pub w0: f32, // Again
    pub w1: f32, // Hard
    pub w2: f32, // Good
    pub w3: f32, // Easy
    
    /// Difficulty parameters
    pub w4: f32, // Initial difficulty mean
    pub w5: f32, // Initial difficulty variance
    pub w6: f32, // Difficulty change on failure
    pub w7: f32, // Difficulty change on success
    
    /// Stability parameters
    pub w8: f32,  // Stability increase factor
    pub w9: f32,  // Stability decrease on failure
    pub w10: f32, // Stability modifier
    pub w11: f32, // Stability hard penalty
    pub w12: f32, // Stability easy bonus
    pub w13: f32, // Stability relearn factor
    pub w14: f32, // Hard interval factor
    pub w15: f32, // Easy interval factor
    
    /// Forgetting curve parameters
    pub w16: f32, // Short-term stability factor
    pub w17: f32, // Long-term stability factor
    pub w18: f32, // Forgetting curve decay
    pub w19: f32, // Reserved
    /// Forgetting-curve decay exponent — the only weight in this struct that
    /// feeds a formula shaped exactly like FSRS's own, so it is the only one
    /// whose published default transfers directly.
    ///
    /// Defaults to [`FSRS6_DEFAULT_DECAY`] (`0.1542`).
    ///
    /// Two corrections are folded into that number, in order:
    /// 1. It was `0.5` — FSRS-4.5/5's hardcoded `DECAY` — paired with the
    ///    FSRS-6 curve, which decayed retrievability far too fast and made
    ///    aged-but-valid memories fall below the recall weight gate and vanish.
    /// 2. The fix for (1) took `0.0658`, which is **`w[19]` of the FSRS-6
    ///    parameter vector, not `w[20]`** — an off-by-one into the published
    ///    array. `fsrs-rs` names the real value `FSRS6_DEFAULT_DECAY = 0.1542`
    ///    and clamps this parameter to <code>[DECAY_MIN]..=[DECAY_MAX]</code>
    ///    (`0.1..=0.8`), a range `0.0658` sits **below** — so the optimizer
    ///    could never have produced it.
    ///
    /// Both flips are gated on the retention harness ([`crate::retention`]);
    /// its `w20` experiment measures aged recall at all three exponents so the
    /// correction is shown not to reintroduce (1). Making this exponent
    /// *trainable* (fit from access history) is the eventual goal — ROADMAP P13.
    pub w20: f32,
}

/// FSRS-6's published forgetting-curve decay, matching `fsrs-rs`'s
/// `FSRS6_DEFAULT_DECAY` — the last element of its 21-value `DEFAULT_PARAMETERS`.
pub const FSRS6_DEFAULT_DECAY: f32 = 0.1542;

/// FSRS-5's hardcoded decay, kept named so the comparison in
/// [`crate::retention`] reads as the documented constant it is rather than a
/// magic number.
pub const FSRS5_DEFAULT_DECAY: f32 = 0.5;

/// Lower clamp `fsrs-rs`'s optimizer applies to the decay exponent. A default
/// outside this range cannot be a fitted FSRS parameter.
pub const DECAY_MIN: f32 = 0.1;

/// Upper clamp `fsrs-rs`'s optimizer applies to the decay exponent.
pub const DECAY_MAX: f32 = 0.8;

impl Default for FsrsParameters {
    fn default() -> Self {
        // NOT the FSRS-6 weight table, despite the module heading — `w0..w18`
        // are FSRS-5 values and six entries are zeroed, matching neither
        // published table. Left as-is on purpose: only `w20` feeds a formula
        // shaped like FSRS's own, so only `w20`'s published default transfers
        // without an A/B on the retention harness. Adopting the rest is its
        // own decision, open in ROADMAP P13.
        Self {
            w0: 0.4072,
            w1: 1.1829,
            w2: 3.1262,
            w3: 15.4722,
            w4: 7.2102,
            w5: 0.5316,
            w6: 1.0651,
            w7: 0.0234,
            w8: 1.616,
            w9: 0.1544,
            w10: 1.0,
            w11: 1.9395,
            w12: 0.1176,
            w13: 0.0,
            w14: 0.0,
            w15: 0.0,
            w16: 2.2035,
            w17: 0.0,
            w18: 0.0,
            w19: 0.0,
            // FSRS-6's published decay. Was 0.5 (the FSRS-5 constant), then
            // 0.0658 (w[19] of the FSRS-6 vector, read one slot short) — see
            // the field doc for both corrections.
            w20: FSRS6_DEFAULT_DECAY,
        }
    }
}

/// Memory state based on accessibility
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryState {
    /// High retention (≥70%), immediately retrievable
    Active,
    /// Medium retention (40-70%), retrievable with effort
    Dormant,
    /// Low retention (10-40%), rarely surfaces
    Silent,
    /// Below threshold (<10%), effectively forgotten
    Unavailable,
}

impl MemoryState {
    /// Determine state from accessibility score
    #[must_use]
    pub fn from_accessibility(accessibility: f32) -> Self {
        if accessibility >= 0.7 {
            Self::Active
        } else if accessibility >= 0.4 {
            Self::Dormant
        } else if accessibility >= 0.1 {
            Self::Silent
        } else {
            Self::Unavailable
        }
    }
}

/// FSRS state for a memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FsrsState {
    /// Stability: days until retrievability drops to 90%
    pub stability: f32,
    /// Difficulty: inherent difficulty (1-10)
    pub difficulty: f32,
    /// Last access timestamp (Unix seconds)
    pub last_access: i64,
    /// Number of times this memory has been accessed
    pub access_count: u32,
    /// Importance multiplier (default 1.0, can be boosted)
    pub importance: f32,
    /// Storage strength (only increases, never decreases)
    pub storage_strength: f32,
    /// Number of consolidation generations
    pub generation: u32,
}

impl Default for FsrsState {
    fn default() -> Self {
        Self {
            stability: 1.0,      // 1 day initial stability
            difficulty: 5.0,    // Medium difficulty
            last_access: now(),
            access_count: 0,
            importance: 1.0,
            storage_strength: 0.1,
            generation: 0,
        }
    }
}

impl FsrsState {
    /// Create new FSRS state
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate retrievability (probability of recall)
    /// 
    /// Uses power law forgetting: R(t, S) = (1 + factor × t / S)^(-decay)
    #[must_use]
    pub fn retrievability(&self, params: &FsrsParameters) -> f32 {
        let elapsed_days = self.elapsed_days();
        power_law_retrievability(elapsed_days, self.stability, params.w20)
    }

    /// Calculate retrieval strength (decays, restored by access)
    #[must_use]
    pub fn retrieval_strength(&self, params: &FsrsParameters) -> f32 {
        // Retrieval strength decays faster than stability
        let r = self.retrievability(params);
        // Weight by access frequency (more accesses = stronger retrieval paths)
        let access_factor = (self.access_count.lossy_f32() / 10.0).min(1.0);
        r * 0.5f32.mul_add(access_factor, 0.5)
    }

    /// Calculate overall accessibility score
    /// 
    /// accessibility = 0.5 × retention + 0.3 × `retrieval_strength` + 0.2 × `storage_strength`
    #[must_use]
    pub fn accessibility(&self, params: &FsrsParameters) -> f32 {
        let retention = self.retrievability(params);
        let retrieval = self.retrieval_strength(params);
        let storage = self.storage_strength.min(1.0);
        
        0.2f32.mul_add(storage, 0.3f32.mul_add(retrieval, 0.5 * retention))
    }

    /// Get memory state based on accessibility
    #[must_use]
    pub fn state(&self, params: &FsrsParameters) -> MemoryState {
        MemoryState::from_accessibility(self.accessibility(params))
    }

    /// Calculate weight for summarization
    /// 
    /// weight < 0.2: compress to essence
    /// weight 0.2-0.8: moderate compression
    /// weight > 0.8: full detail
    /// weight > 1.0: expand/research
    #[must_use]
    pub fn weight(&self, params: &FsrsParameters) -> f32 {
        self.retrievability(params) * self.importance
    }

    /// Days elapsed since last access
    #[must_use]
    pub fn elapsed_days(&self) -> f32 {
        let elapsed_secs = now() - self.last_access;
        elapsed_secs.lossy_f32() / 86400.0
    }

    /// Record an access (the testing effect)
    /// 
    /// Every retrieval strengthens the memory
    pub fn record_access(&mut self, params: &FsrsParameters, rating: Rating) {
        self.record_access_at(params, rating, now());
    }

    /// [`Self::record_access`] for a review that happened at `at` (unix
    /// seconds). The dream applies queued recalls, possibly days after they
    /// happened: measured from the dream, a recall two days earlier got two
    /// days' extra stability boost, its decay clock restarted at the dream,
    /// and K queued recalls became one review plus K−1 near no-ops. The clock
    /// never moves backwards: a review older than the last access is measured
    /// as immediate and leaves `last_access` where it is.
    pub fn record_access_at(&mut self, params: &FsrsParameters, rating: Rating, at: i64) {
        // Update stability based on current state and rating
        let elapsed_days = (at - self.last_access).max(0).lossy_f32() / 86400.0;
        let r = power_law_retrievability(elapsed_days, self.stability, params.w20);
        debug_assert!(r.is_finite(), "retrievability is finite");
        
        // Stability increase formula from FSRS
        let stability_modifier = match rating {
            Rating::Again => params.w9, // Decrease on failure
            Rating::Hard => params.w8.mul_add(params.w11.powf(-self.difficulty / 10.0), 1.0),
            Rating::Good => params.w8.mul_add((1.0 - r).powf(params.w10), 1.0),
            Rating::Easy => params.w8.mul_add(params.w12.powf(self.difficulty / 10.0), 1.0),
        };
        
        self.stability = (self.stability * stability_modifier).max(0.1);
        
        // Update difficulty
        let difficulty_delta = match rating {
            Rating::Again => params.w6,
            Rating::Hard => params.w6 * 0.5,
            Rating::Good => -params.w7,
            Rating::Easy => -params.w7 * 2.0,
        };
        self.difficulty = (self.difficulty + difficulty_delta).clamp(1.0, 10.0);
        
        // Storage strength only increases
        self.storage_strength = (self.storage_strength + 0.1).min(1.0);
        
        // Update access tracking
        self.last_access = self.last_access.max(at);
        self.access_count += 1;
    }

    /// Promote memory (mark as helpful/important)
    pub fn promote(&mut self, boost: f32) {
        self.importance = (self.importance + boost).min(3.0);
        self.storage_strength = (self.storage_strength + 0.2).min(1.0);
    }

    /// Demote memory (mark as wrong/unhelpful)
    pub fn demote(&mut self, penalty: f32) {
        self.importance = (self.importance - penalty).max(0.1);
    }
}

/// Rating for memory access quality
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[derive(Default)]
pub enum Rating {
    /// Complete failure to recall
    Again,
    /// Recalled with significant difficulty
    Hard,
    /// Recalled correctly
    #[default]
    Good,
    /// Recalled easily
    Easy,
}


/// Calculate retrievability using power law forgetting curve
/// 
/// R(t, S) = (1 + factor × t / S)^(-decay)
/// where factor = 0.9^(-1/decay) - 1
#[must_use]
pub fn power_law_retrievability(elapsed_days: f32, stability: f32, decay: f32) -> f32 {
    if elapsed_days <= 0.0 {
        return 1.0;
    }
    if stability <= 0.0 {
        return 0.0;
    }
    
    let factor = 0.9_f32.powf(-1.0 / decay) - 1.0;
    (1.0 + factor * elapsed_days / stability).powf(-decay).clamp(0.0, 1.0)
}

/// Calculate similarity threshold for prediction error gating
/// 
/// Returns action based on similarity:
/// - >0.92: Reinforce existing
/// - >0.75: Update existing  
/// - <0.75: Create new
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IngestAction {
    /// Almost identical - just strengthen existing memory
    Reinforce,
    /// Related - merge/update existing memory
    Update,
    /// Novel - create new memory
    Create,
}

impl IngestAction {
    #[must_use]
    pub fn from_similarity(similarity: f32) -> Self {
        if similarity > 0.92 {
            Self::Reinforce
        } else if similarity > 0.75 {
            Self::Update
        } else {
            Self::Create
        }
    }
}

/// Get current Unix timestamp
/// The current time in unix seconds — the clock `last_access` is kept in.
pub fn unix_now() -> i64 {
    now()
}

fn now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_secs()).unwrap_or(i64::MAX))
}

#[cfg(test)]
mod review_time_tests {
    use super::{FsrsParameters, FsrsState, Rating, now};

    /// A recall queued two days before the dream is measured at the recall:
    /// the same stability gain as had it been applied then, and the decay
    /// clock starts from the recall, not the dream.
    #[test]
    fn a_queued_recall_is_dated_by_the_recall_not_the_dream() {
        let params = FsrsParameters::default();
        let recalled_at = now() - 2 * 86_400;
        let base = FsrsState {
            last_access: recalled_at - 86_400,
            ..FsrsState::default()
        };

        let mut on_time = base.clone();
        on_time.record_access_at(&params, Rating::Good, recalled_at);
        let mut late = base.clone();
        late.record_access(&params, Rating::Good);

        assert_eq!(on_time.last_access, recalled_at, "the clock starts at the recall");
        assert!(late.last_access > recalled_at);
        assert!(
            late.stability > on_time.stability,
            "measured at the dream, the boost was inflated: {} vs {}",
            late.stability,
            on_time.stability
        );

        // A review older than the last access never moves the clock back.
        let mut newer = base;
        newer.last_access = now();
        let kept = newer.last_access;
        newer.record_access_at(&params, Rating::Good, recalled_at);
        assert_eq!(newer.last_access, kept);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retrievability_decay() {
        let params = FsrsParameters::default();
        let mut state = FsrsState::new();
        state.stability = 1.0; // 1 day stability
        
        // Simulate time passing by manipulating last_access
        state.last_access = now() - 86400; // 1 day ago
        let r1 = state.retrievability(&params);
        
        state.last_access = now() - 86400 * 7; // 7 days ago
        let r7 = state.retrievability(&params);
        
        state.last_access = now() - 86400 * 30; // 30 days ago
        let r30 = state.retrievability(&params);
        
        // Should decay over time
        assert!(r1 > r7);
        assert!(r7 > r30);
        
        // Should follow power law (not exponential - decay is slower)
        println!("R at 1 day: {r1:.3}");
        println!("R at 7 days: {r7:.3}");
        println!("R at 30 days: {r30:.3}");
    }

    #[test]
    fn test_memory_states() {
        assert_eq!(MemoryState::from_accessibility(0.9), MemoryState::Active);
        assert_eq!(MemoryState::from_accessibility(0.5), MemoryState::Dormant);
        assert_eq!(MemoryState::from_accessibility(0.2), MemoryState::Silent);
        assert_eq!(MemoryState::from_accessibility(0.05), MemoryState::Unavailable);
    }

    /// The weight slots this crate's formulas actually read.
    ///
    /// Not a grep result — [`only_the_live_weights_change_any_fsrs_output`]
    /// re-derives it behaviourally on every run.
    const LIVE_WEIGHT_SLOTS: [usize; 8] = [6, 7, 8, 9, 10, 11, 12, 20];

    /// Set slot `i` to `value`. Exists so a test can sweep the table without
    /// twenty-one hand-written cases going quietly out of step with the struct.
    fn set_weight_slot(p: &mut FsrsParameters, i: usize, value: f32) {
        match i {
            0 => p.w0 = value,
            1 => p.w1 = value,
            2 => p.w2 = value,
            3 => p.w3 = value,
            4 => p.w4 = value,
            5 => p.w5 = value,
            6 => p.w6 = value,
            7 => p.w7 = value,
            8 => p.w8 = value,
            9 => p.w9 = value,
            10 => p.w10 = value,
            11 => p.w11 = value,
            12 => p.w12 = value,
            13 => p.w13 = value,
            14 => p.w14 = value,
            15 => p.w15 = value,
            16 => p.w16 = value,
            17 => p.w17 = value,
            18 => p.w18 = value,
            19 => p.w19 = value,
            20 => p.w20 = value,
            other => panic!("FsrsParameters has 21 slots, asked for w{other}"),
        }
    }

    /// Every number `FsrsParameters` can influence, for one fixed starting
    /// state: the five read-only scores, plus the stability and difficulty
    /// `record_access` produces for each of the four ratings.
    ///
    /// `anchor` fixes `last_access`, so the only thing that varies between two
    /// fingerprints is the parameter table.
    fn fsrs_fingerprint(params: &FsrsParameters, anchor: i64) -> Vec<f32> {
        let fresh = || FsrsState {
            stability: 2.5,
            difficulty: 6.0,
            last_access: anchor,
            access_count: 3,
            importance: 1.2,
            storage_strength: 0.4,
            generation: 1,
        };

        let base = fresh();
        let mut out = vec![
            base.retrievability(params),
            base.retrieval_strength(params),
            base.accessibility(params),
            base.weight(params),
            // `state` is a bucketing of accessibility; include it as a number
            // so a weight that only moves a memory across a band still counts.
            match base.state(params) {
                MemoryState::Active => 3.0,
                MemoryState::Dormant => 2.0,
                MemoryState::Silent => 1.0,
                MemoryState::Unavailable => 0.0,
            },
        ];

        for rating in [Rating::Again, Rating::Hard, Rating::Good, Rating::Easy] {
            let mut s = fresh();
            s.record_access(params, rating);
            out.push(s.stability);
            out.push(s.difficulty);
        }
        out
    }

    /// The largest relative move between two fingerprints.
    fn largest_relative_move(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "fingerprints must be the same shape");
        a.iter()
            .zip(b)
            .map(|(x, y)| {
                let scale = x.abs().max(y.abs()).max(1.0);
                (x - y).abs() / scale
            })
            .fold(0.0_f32, f32::max)
    }

    /// **13 of the 21 FSRS weights change nothing, and this proves it by
    /// perturbing them rather than by reading the code.**
    ///
    /// `FsrsParameters` presents twenty-one public, serializable `wN` fields —
    /// a tuning surface. Only eight of them are wired to a formula. The rest
    /// are a table transcribed from FSRS's published parameters for a
    /// stability update that this crate does not implement, so they look like
    /// knobs and turn nothing. Several are *non-zero* (`w0 = 0.4072`,
    /// `w16 = 2.2035`), which is what makes the appearance convincing.
    ///
    /// A grep would answer this too, and would rot the first time someone
    /// wired one up. This sweeps every slot, moves it, and asks whether any
    /// observable number moved with it — so the day a formula starts reading
    /// `w16`, this test fails and names the slot.
    /// The value every slot is moved to during the sweep. Distinct from every
    /// default in the table, and inside the decay clamp so the `w20` case
    /// probes the real path instead of a rejected parameter — checked below at
    /// compile time, so editing either the probe or the clamps cannot silently
    /// turn that case into a no-op.
    const PROBE: f32 = 0.37;
    const _: () = assert!(
        PROBE > DECAY_MIN && PROBE < DECAY_MAX,
        "the probe must be a legal decay, or the w20 case tests nothing"
    );

    #[test]
    fn only_the_live_weights_change_any_fsrs_output() {
        // Far enough in the past that retrievability is well off its ceiling,
        // so a decay change has room to show. Fixed, so wall-clock drift
        // between fingerprints cannot masquerade as a parameter effect.
        let anchor = now() - 30 * 86400;
        let baseline = fsrs_fingerprint(&FsrsParameters::default(), anchor);

        let mut live = Vec::new();
        let mut dead = Vec::new();
        for slot in 0..21 {
            let mut params = FsrsParameters::default();
            set_weight_slot(&mut params, slot, PROBE);
            let moved = largest_relative_move(&baseline, &fsrs_fingerprint(&params, anchor));
            if moved > 1e-4 {
                live.push(slot);
            } else {
                dead.push(slot);
            }
        }

        assert_eq!(
            live, LIVE_WEIGHT_SLOTS,
            "the set of FSRS weights that affect behaviour changed.\n             live now: {live:?}\ndead now: {dead:?}\n             If a formula started reading a new slot, update LIVE_WEIGHT_SLOTS \
             and the `FsrsParameters` doc that says which weights are read. If a \
             slot went dead, say why in the doc rather than leaving a field that \
             looks like a knob and turns nothing."
        );

        // The harness itself must be stable, or "unchanged" means nothing:
        // re-fingerprint the untouched defaults and require an exact match on
        // the same tolerance the sweep used.
        let baseline_again = fsrs_fingerprint(&FsrsParameters::default(), anchor);
        assert!(
            largest_relative_move(&baseline, &baseline_again) <= 1e-4,
            "the fingerprint is not reproducible, so no verdict above is trustworthy"
        );
    }

    #[test]
    fn test_testing_effect() {
        let params = FsrsParameters::default();
        let mut state = FsrsState::new();
        state.last_access = now() - 86400; // 1 day ago
        
        let stability_before = state.stability;
        state.record_access(&params, Rating::Good);
        let stability_after = state.stability;
        
        // Stability should increase after successful recall
        assert!(stability_after > stability_before);
        assert_eq!(state.access_count, 1);
    }

    #[test]
    fn test_weight_calculation() {
        let params = FsrsParameters::default();
        let mut state = FsrsState::new();
        
        // Fresh memory should have high weight
        let weight_fresh = state.weight(&params);
        assert!(weight_fresh > 0.8);
        
        // Old memory should have low weight
        state.last_access = now() - 86400 * 30;
        let weight_old = state.weight(&params);
        assert!(weight_old < weight_fresh);
        
        // Promoted memory should have boosted weight
        state.promote(1.0);
        let weight_promoted = state.weight(&params);
        assert!(weight_promoted > weight_old);
    }

    #[test]
    fn test_ingest_action() {
        assert_eq!(IngestAction::from_similarity(0.95), IngestAction::Reinforce);
        assert_eq!(IngestAction::from_similarity(0.85), IngestAction::Update);
        assert_eq!(IngestAction::from_similarity(0.5), IngestAction::Create);
    }

    /// The default decay must be a value the reference optimizer could have
    /// produced. This is the check that would have caught reading `w[19]`
    /// (`0.0658`) as `w[20]`: it is below `fsrs-rs`'s clamp floor, so no fitted
    /// FSRS-6 parameter set can contain it in that slot.
    #[test]
    fn default_decay_is_inside_the_reference_clamp_range() {
        let default_decay = FsrsParameters::default().w20;
        assert_eq!(default_decay, FSRS6_DEFAULT_DECAY);
        assert!(
            default_decay >= DECAY_MIN,
            "decay {default_decay} is below the optimizer's floor {DECAY_MIN} — \
             a value this small is not a fitted FSRS parameter"
        );
        assert!(
            default_decay <= DECAY_MAX,
            "decay {default_decay} is above the optimizer's ceiling {DECAY_MAX}"
        );
        // Negative space: the two values this default has wrongly held are
        // exactly the ones the range rejects and accepts-but-is-not.
        const {
            assert!(
                0.0658_f32 < DECAY_MIN,
                "the w[19] misread must be out of range"
            );
            assert!(
                FSRS5_DEFAULT_DECAY <= DECAY_MAX,
                "FSRS-5's decay is in range, just wrong for FSRS-6"
            );
            assert!(FSRS5_DEFAULT_DECAY != FSRS6_DEFAULT_DECAY);
        }
    }

    /// The curve is anchored: whatever the exponent, retrievability is 0.9 at
    /// `t == S`. That is the definition of stability, and it is what makes the
    /// `factor = 0.9^(-1/decay) - 1` term correct rather than a fudge — so a
    /// decay change moves the *tail*, never the anchor.
    #[test]
    fn every_decay_keeps_the_ninety_percent_anchor_at_one_stability() {
        for decay in [
            FSRS5_DEFAULT_DECAY,
            FSRS6_DEFAULT_DECAY,
            DECAY_MIN,
            DECAY_MAX,
        ] {
            let r = power_law_retrievability(10.0, 10.0, decay);
            assert!(
                (r - 0.9).abs() < 1e-3,
                "decay {decay} broke the anchor: R(t=S) = {r}, expected 0.9"
            );
        }
    }

    /// A smaller exponent is a flatter tail. Stated as an ordering rather than
    /// as fixed numbers so it documents the *property* the decay controls.
    #[test]
    fn smaller_decay_retains_aged_memories_longer() {
        // 800 days at 1-day stability — the retention harness's aged corpus.
        let aged = |decay: f32| power_law_retrievability(800.0, 1.0, decay);
        let fsrs5 = aged(FSRS5_DEFAULT_DECAY);
        let fsrs6 = aged(FSRS6_DEFAULT_DECAY);
        let misread = aged(0.0658);
        assert!(
            fsrs5 < fsrs6,
            "FSRS-5's 0.5 must decay faster: {fsrs5} vs {fsrs6}"
        );
        assert!(
            fsrs6 < misread,
            "the w[19] misread was flatter still: {fsrs6} vs {misread}"
        );
        // The correction must not undo what the first flip bought: the default
        // recall gate is `min_weight = 0.1`, and FSRS-6's exponent clears it at
        // this age while FSRS-5's does not.
        assert!(
            fsrs5 < 0.1,
            "FSRS-5's decay drops an 800-day memory below the gate: {fsrs5}"
        );
        assert!(
            fsrs6 > 0.1,
            "FSRS-6's decay must keep it retrievable: {fsrs6}"
        );
    }
}
