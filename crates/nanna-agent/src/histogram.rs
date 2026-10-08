//! Fixed-bucket latency histograms behind the `/metrics` `_bucket` series.
//!
//! The stats trackers keep a ring of the last 200 latencies, which is what the
//! GUI's p50/p95/p99 read. A ring cannot be aggregated across scrapes or
//! daemons, so Prometheus-style histograms are kept beside it: a count per
//! bucket plus the running sum, in memory only. They describe the process,
//! not the history the trackers restore from the store, and reset when the
//! daemon restarts — the counter-reset semantics a scraper already handles.
//!
//! **Bounded by construction.** One histogram is a fixed array (13 counters
//! and a sum, 112 bytes) per tool or model, and tools and models are bounded
//! by configuration, so nothing here grows with traffic.

use serde::{Deserialize, Serialize};

/// Bucket count, not counting the implicit `+Inf` bucket.
pub const LATENCY_BUCKETS: usize = 12;

/// Upper bounds (inclusive, ms) for a tool call: 10 ms covers an in-process
/// read, 180 s is `exec`'s ceiling, so the last finite bound sits below it
/// and a command that runs to the ceiling lands in `+Inf`.
pub const TOOL_LATENCY_BOUNDS_MS: [u64; LATENCY_BUCKETS] = [
    10, 25, 50, 100, 250, 500, 1_000, 2_500, 5_000, 10_000, 30_000, 120_000,
];

/// Upper bounds (inclusive, ms) for a model request: from a cached or tiny
/// completion (250 ms) to a long streamed turn on a local model (10 min).
pub const MODEL_LATENCY_BOUNDS_MS: [u64; LATENCY_BUCKETS] = [
    250, 500, 1_000, 2_500, 5_000, 10_000, 20_000, 30_000, 60_000, 120_000, 300_000, 600_000,
];

/// Per-bucket counts (non-cumulative; the last slot is `+Inf`) and the sum of
/// every observed latency, both saturating.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LatencyHistogram {
    counts: [u64; LATENCY_BUCKETS + 1],
    sum_ms: u64,
}

impl LatencyHistogram {
    /// Count one latency of `latency_ms` against `bounds`. A histogram must
    /// always be observed and read with the same bounds.
    pub fn observe(&mut self, bounds: &[u64; LATENCY_BUCKETS], latency_ms: u64) {
        debug_assert!(bounds.is_sorted(), "bucket bounds ascend");
        let slot = bounds.partition_point(|&bound| bound < latency_ms);
        debug_assert!(slot <= LATENCY_BUCKETS, "the last slot is +Inf");
        self.counts[slot] = self.counts[slot].saturating_add(1);
        self.sum_ms = self.sum_ms.saturating_add(latency_ms);
    }

    /// Cumulative counts, Prometheus `le` order: entry `i` counts every
    /// latency `<= bounds[i]`, and the last entry (`+Inf`) is [`Self::count`].
    #[must_use]
    pub fn cumulative(&self) -> [u64; LATENCY_BUCKETS + 1] {
        let mut running = 0u64;
        let mut out = [0u64; LATENCY_BUCKETS + 1];
        for (slot, count) in self.counts.iter().enumerate() {
            running = running.saturating_add(*count);
            out[slot] = running;
        }
        debug_assert!(out.is_sorted(), "cumulative counts never fall");
        out
    }

    /// Every observation (the `_count` series).
    #[must_use]
    pub fn count(&self) -> u64 {
        self.counts
            .iter()
            .fold(0u64, |total, c| total.saturating_add(*c))
    }

    /// Sum of every observed latency in ms (the `_sum` series).
    #[must_use]
    pub const fn sum_ms(&self) -> u64 {
        self.sum_ms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_latency_lands_in_the_first_bucket_that_holds_it() {
        let mut h = LatencyHistogram::default();
        for ms in [0, 10, 11, 120_000, 120_001, 180_000] {
            h.observe(&TOOL_LATENCY_BOUNDS_MS, ms);
        }
        let cumulative = h.cumulative();
        assert_eq!(cumulative[0], 2, "0 and 10 are <= 10 (le is inclusive)");
        assert_eq!(cumulative[1], 3, "11 is <= 25");
        assert_eq!(
            cumulative[LATENCY_BUCKETS - 1],
            4,
            "120 000 is <= the last bound"
        );
        assert_eq!(cumulative[LATENCY_BUCKETS], 6, "+Inf counts everything");
        assert_eq!(h.count(), 6);
        assert_eq!(h.sum_ms(), 10 + 11 + 120_000 + 120_001 + 180_000);
    }

    #[test]
    fn counters_saturate_instead_of_wrapping() {
        let mut h = LatencyHistogram::default();
        h.observe(&MODEL_LATENCY_BOUNDS_MS, u64::MAX);
        h.observe(&MODEL_LATENCY_BOUNDS_MS, u64::MAX);
        assert_eq!(h.sum_ms(), u64::MAX);
        assert_eq!(h.count(), 2);
        assert_eq!(h.cumulative()[LATENCY_BUCKETS], 2);
    }

    #[test]
    fn both_bound_sets_ascend() {
        assert!(TOOL_LATENCY_BOUNDS_MS.is_sorted());
        assert!(MODEL_LATENCY_BOUNDS_MS.is_sorted());
        assert!(
            TOOL_LATENCY_BOUNDS_MS.windows(2).all(|w| w[0] < w[1]),
            "no repeated bound"
        );
        assert!(
            MODEL_LATENCY_BOUNDS_MS.windows(2).all(|w| w[0] < w[1]),
            "no repeated bound"
        );
    }
}
