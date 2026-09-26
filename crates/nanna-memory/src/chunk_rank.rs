//! Collapsing per-chunk search hits back to the memories that own them.
//!
//! Chunk search returns pieces; recall returns memories. Something has to
//! decide what a memory's score is when several of its chunks matched, and the
//! choice is load-bearing in a way that is easy to get wrong.
//!
//! **Averaging is wrong.** A memory whose chunk 7 of 40 answers the query
//! exactly scores 0.9 on that chunk and ~0.1 on the other 39; the mean buries
//! it under a memory that is vaguely on-topic throughout. That is the exact
//! failure chunking exists to fix — a long memory embedded whole becomes a
//! centroid of everything it contains — so averaging the chunks reintroduces it
//! one level up.
//!
//! **Summing is worse.** It makes score scale with length, so the longest
//! memory wins every query regardless of relevance.
//!
//! So: **max**. A memory is as relevant as its most relevant piece. But a
//! memory that matched in six places is genuinely better evidence than one that
//! matched in a single place at the same peak, and max alone cannot express
//! that. Hence the two-part rule below.

use std::collections::HashMap;

use crate::lossy::LossyF32;

/// How much corroboration may improve a hit's rank, as a fraction of the
/// distance from its best score to a perfect 1.0.
///
/// Corroboration is a tiebreaker, not evidence of its own. A quarter of the
/// remaining headroom means it can reorder hits that are already close and
/// cannot promote a weak match over a strong one: a 0.50 hit corroborated
/// infinitely often reaches 0.625, still below an uncorroborated 0.65.
const CORROBORATION_HEADROOM: f32 = 0.25;

/// One memory's chunk hits, collapsed.
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkHit {
    /// The best cosine similarity among this memory's matching chunks. This is
    /// a real, calibrated similarity — it is what the threshold is tested
    /// against.
    pub best: f32,
    /// Ordinal of the chunk that produced [`Self::best`]. Lets a reader page
    /// straight to the part that matched instead of from the top.
    pub best_ordinal: i64,
    /// How many OTHER chunks of this memory also cleared the threshold.
    pub corroborating: usize,
}

impl ChunkHit {
    /// The value to sort by.
    ///
    /// Deliberately NOT the value to threshold against — see
    /// [`collapse_chunk_hits`]. It is `best` plus a saturating share of the
    /// headroom above it, so it is monotonic in both inputs, never reaches 1.0
    /// from corroboration alone, and never drops below `best`.
    #[must_use]
    pub fn rank(&self) -> f32 {
        let saturating = 1.0 - 1.0 / (1.0 + self.corroborating.lossy_f32());
        ((1.0 - self.best) * CORROBORATION_HEADROOM).mul_add(saturating, self.best)
    }
}

/// Collapse `(memory_id, ordinal, similarity)` chunk hits into one entry per
/// memory, keeping only those whose BEST chunk clears `min_score`.
///
/// The split between admitting and ranking is the whole point. `min_score` is
/// a calibrated cosine threshold — 0.40 means "this text is actually related",
/// tuned against real embeddings. If corroboration were folded into the number
/// the threshold tests, that calibration would silently change meaning: a
/// memory with eight weakly-matching chunks could clear a bar that no single
/// piece of it deserves to clear, and the same threshold would admit different
/// things depending on how long the memory happened to be.
///
/// So admission is decided on the true best cosine, and corroboration only ever
/// reorders what was already admitted.
#[must_use]
pub fn collapse_chunk_hits(
    hits: &[(String, i64, f32)],
    min_score: f32,
) -> HashMap<String, ChunkHit> {
    let mut collapsed: HashMap<String, ChunkHit> = HashMap::new();
    for (memory_id, ordinal, similarity) in hits {
        // Chunks below the bar are not evidence of anything and must not
        // corroborate. Counting them would let a long memory accumulate rank
        // from pieces that do not match at all. NaN (a zero vector) is checked
        // by name: it compares false against the floor and would otherwise be
        // admitted as a hit and poison the ordering downstream.
        if !similarity.is_finite() || *similarity < min_score {
            continue;
        }
        collapsed
            .entry(memory_id.clone())
            .and_modify(|hit| {
                if *similarity > hit.best {
                    hit.best = *similarity;
                    hit.best_ordinal = *ordinal;
                }
                hit.corroborating += 1;
            })
            .or_insert(ChunkHit {
                best: *similarity,
                best_ordinal: *ordinal,
                corroborating: 0,
            });
    }
    collapsed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(best: f32, corroborating: usize) -> ChunkHit {
        ChunkHit { best, best_ordinal: 0, corroborating }
    }

    /// The property that makes corroboration safe to add: it can reorder
    /// near-ties, and it can never lift a weak match past a strong one.
    #[test]
    fn corroboration_never_overturns_a_clearly_better_match() {
        let weak_but_repeated = hit(0.50, 10_000);
        let strong_and_alone = hit(0.65, 0);
        assert!(
            weak_but_repeated.rank() < strong_and_alone.rank(),
            "{} should not outrank {}",
            weak_but_repeated.rank(),
            strong_and_alone.rank()
        );
    }

    #[test]
    fn corroboration_breaks_a_tie_in_favour_of_more_evidence() {
        assert!(hit(0.80, 5).rank() > hit(0.80, 0).rank());
    }

    /// Rank must never fall below the score it is built from, or sorting by
    /// rank would demote a hit relative to its own admitted similarity.
    #[test]
    fn rank_is_monotonic_and_never_below_best() {
        for best in [0.0_f32, 0.4, 0.75, 0.99, 1.0] {
            let mut previous = f32::NEG_INFINITY;
            for corroborating in [0usize, 1, 2, 5, 50, 5000] {
                let r = hit(best, corroborating).rank();
                assert!(r >= best - f32::EPSILON, "rank {r} fell below best {best}");
                assert!(r <= 1.0 + f32::EPSILON, "rank {r} exceeded 1.0");
                assert!(r >= previous, "rank must not decrease with more evidence");
                previous = r;
            }
        }
    }

    /// A memory is as relevant as its best piece — the one thing averaging
    /// gets wrong, and the reason chunking exists at all.
    #[test]
    fn one_excellent_chunk_beats_many_mediocre_ones() {
        let hits = vec![
            ("needle".to_string(), 7, 0.91),
            ("haystack".to_string(), 0, 0.55),
            ("haystack".to_string(), 1, 0.56),
            ("haystack".to_string(), 2, 0.54),
            ("haystack".to_string(), 3, 0.57),
        ];
        let collapsed = collapse_chunk_hits(&hits, 0.40);
        assert!(collapsed["needle"].rank() > collapsed["haystack"].rank());
        assert_eq!(collapsed["needle"].best_ordinal, 7, "points at the chunk that matched");
    }

    /// Admission is decided on the true cosine, never on the corroborated
    /// rank — otherwise the threshold would mean something different for a
    /// long memory than for a short one.
    #[test]
    fn admission_uses_the_true_best_not_the_corroborated_rank() {
        let hits: Vec<(String, i64, f32)> =
            (0..20).map(|i| ("long".to_string(), i, 0.39)).collect();
        let collapsed = collapse_chunk_hits(&hits, 0.40);
        assert!(
            collapsed.is_empty(),
            "twenty near-misses must not add up to a hit"
        );
    }

    /// Sub-threshold chunks are not evidence and must not corroborate, or a
    /// long memory would accumulate rank from pieces that do not match.
    #[test]
    fn sub_threshold_chunks_do_not_corroborate() {
        let hits = vec![
            ("m".to_string(), 0, 0.80),
            ("m".to_string(), 1, 0.10),
            ("m".to_string(), 2, 0.05),
        ];
        let collapsed = collapse_chunk_hits(&hits, 0.40);
        assert_eq!(collapsed["m"].corroborating, 0);
        assert!((collapsed["m"].best - 0.80).abs() < 1e-6);
    }

    /// A NaN chunk (a zero vector) is not evidence: alone it admits nothing,
    /// and beside a real hit it neither corroborates nor wins.
    #[test]
    fn a_nan_chunk_is_not_evidence() {
        let alone = collapse_chunk_hits(&[("z".to_string(), 0, f32::NAN)], 0.40);
        assert!(
            alone.is_empty(),
            "NaN must not clear the floor by comparing false"
        );

        let hits = vec![("m".to_string(), 0, 0.80), ("m".to_string(), 1, f32::NAN)];
        let collapsed = collapse_chunk_hits(&hits, 0.40);
        assert_eq!(collapsed["m"].corroborating, 0, "NaN never corroborates");
        assert_eq!(collapsed["m"].best_ordinal, 0);
    }

    #[test]
    fn the_best_ordinal_tracks_the_best_score_regardless_of_arrival_order() {
        let hits = vec![
            ("m".to_string(), 3, 0.60),
            ("m".to_string(), 9, 0.85),
            ("m".to_string(), 1, 0.70),
        ];
        let collapsed = collapse_chunk_hits(&hits, 0.40);
        assert_eq!(collapsed["m"].best_ordinal, 9);
        assert_eq!(collapsed["m"].corroborating, 2);
    }
}
