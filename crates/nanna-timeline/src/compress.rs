//! Folding one episode's event series into a single episode — the DSP
//! compression step (P25, "DSP timeline compression", step 2).
//!
//! Deterministic and model-free by design. A card's thread is a time series of
//! posts and transitions; what makes it worth remembering is its *shape* —
//! where it changed course and where it peaked — not every progress line in
//! between. So compression is piecewise, like decimating a signal while
//! keeping its extrema:
//!
//! - **Always kept:** the first and last event (the series' endpoints), every
//!   `outcome` (a transition — the card changed state), and every local
//!   salience peak at or above [`PEAK_FLOOR`].
//! - **Decimated:** everything else, evenly, into whatever budget is left.
//! - **Elision is visible:** every run of dropped events leaves a
//!   `[… N events elided …]` marker in place, so a reader of the folded
//!   episode can tell a quiet stretch from a compressed one — the timeline
//!   crate's third property, carried through.
//!
//! No randomness, no model call, no clock: the same series always folds to
//! the same episode, which is what lets a dream cycle re-run it safely.

use crate::{Episode, EventKind, MAX_EVENT_SOURCE_IDS};
use nanna_storage::MemoryEventRow;

/// Salience at or above which a local maximum is a peak worth keeping.
///
/// The middle of the normalized range: the board writes 0.5 for an ordinary
/// comment, so a peak must be at least as notable as that to be protected —
/// a local maximum among progress lines (0.3) is noise, not a peak.
pub const PEAK_FLOOR: f32 = 0.5;

/// Smallest meaningful budget: the two endpoints.
pub const MIN_BUDGET: usize = 2;

/// Events a dream keeps when it folds one card's series.
///
/// Bound justification: two dozen lines is a readable account of one card's
/// life, and at the timeline's per-event content cap it stays within what a
/// single memory is meant to hold. Shared by the daemon's fold phase and the
/// `dreaming.timeline_*` budgets, so the number gated is the number shipped.
pub const DREAM_FOLD_BUDGET: usize = 24;

/// One series folded into one episode, with the arithmetic of the fold.
#[derive(Debug, Clone, PartialEq)]
pub struct CompressedEpisode {
    pub episode: Episode,
    /// Events whose content survives in the folded episode.
    pub kept: usize,
    /// Events represented only by an elision marker.
    pub elided: usize,
}

/// Fold `events` (oldest first, one card's series) into one episode keeping at
/// most `budget` of them. `None` for an empty series.
///
/// A `budget` below [`MIN_BUDGET`] is raised to it: a fold that cannot keep
/// both endpoints cannot say when the series began and ended.
#[must_use]
pub fn compress_episode(events: &[MemoryEventRow], budget: usize) -> Option<CompressedEpisode> {
    let first = events.first()?;
    let budget = budget.max(MIN_BUDGET);
    debug_assert!(
        events
            .windows(2)
            .all(|w| w[0].ts_unix_ms <= w[1].ts_unix_ms),
        "a series is folded oldest first"
    );
    let kept = kept_indices(events, budget);
    debug_assert!(kept.len() <= budget, "the fold honours its budget");
    debug_assert!(kept.windows(2).all(|w| w[0] < w[1]), "kept in series order");

    let mut lines = Vec::with_capacity(kept.len() * 2);
    let mut next = 0usize;
    for &index in &kept {
        if index > next {
            lines.push(elision_marker(index - next));
        }
        let event = &events[index];
        lines.push(format!("[{}] {}", event.kind, event.content));
        next = index + 1;
    }
    if next < events.len() {
        lines.push(elision_marker(events.len() - next));
    }

    let kind = if events.iter().any(|e| e.kind == EventKind::Outcome.as_str()) {
        EventKind::Outcome
    } else {
        EventKind::Message
    };
    let salience = events.iter().map(|e| e.salience).fold(0.0_f32, f32::max);
    let episode = Episode {
        kind,
        ts_unix_ms: first.ts_unix_ms,
        workspace_id: first.workspace_id.clone(),
        content: lines.join("\n"),
        salience,
        source_ids: lineage(events, &kept),
    };
    Some(CompressedEpisode {
        episode,
        kept: kept.len(),
        elided: events.len() - kept.len(),
    })
}

/// The indices a fold keeps, ascending, at most `budget` of them.
fn kept_indices(events: &[MemoryEventRow], budget: usize) -> Vec<usize> {
    let n = events.len();
    if n <= budget {
        return (0..n).collect();
    }
    let (mandatory, optional): (Vec<usize>, Vec<usize>) =
        (0..n).partition(|&i| is_mandatory(events, i));

    let mut kept = if mandatory.len() > budget {
        // More transitions and peaks than room: the endpoints, then
        // transitions before peaks (a change of state outranks a bump in
        // salience), each by salience, earlier first on ties — a total order,
        // so the fold stays deterministic.
        let is_outcome = |i: usize| events[i].kind == EventKind::Outcome.as_str();
        let mut inner: Vec<usize> = mandatory
            .iter()
            .copied()
            .filter(|&i| i != 0 && i != n - 1)
            .collect();
        inner.sort_by(|&a, &b| {
            is_outcome(b)
                .cmp(&is_outcome(a))
                .then(events[b].salience.total_cmp(&events[a].salience))
                .then(a.cmp(&b))
        });
        inner.truncate(budget - MIN_BUDGET);
        inner.extend([0, n - 1]);
        inner
    } else {
        // Room left over: decimate the rest evenly into it.
        let room = budget - mandatory.len();
        let mut kept = mandatory;
        let m = optional.len();
        kept.extend((0..room.min(m)).map(|j| optional[j * m / room.min(m).max(1)]));
        kept
    };
    kept.sort_unstable();
    kept.dedup();
    kept
}

/// Whether event `i` must survive the fold: an endpoint, a transition, or a
/// local salience peak at or above [`PEAK_FLOOR`].
fn is_mandatory(events: &[MemoryEventRow], i: usize) -> bool {
    if i == 0 || i + 1 == events.len() || events[i].kind == EventKind::Outcome.as_str() {
        return true;
    }
    let here = events[i].salience;
    here >= PEAK_FLOOR && here >= events[i - 1].salience && here >= events[i + 1].salience
}

/// The folded episode's lineage: every kept event's sources, first occurrence
/// first, bounded like any episode's.
fn lineage(events: &[MemoryEventRow], kept: &[usize]) -> Vec<String> {
    let mut ids: Vec<String> = Vec::new();
    for &index in kept {
        for id in &events[index].source_ids {
            if ids.len() == MAX_EVENT_SOURCE_IDS {
                return ids;
            }
            if !ids.contains(id) {
                ids.push(id.clone());
            }
        }
    }
    ids
}

fn elision_marker(count: usize) -> String {
    debug_assert!(count > 0, "an elision stands for at least one event");
    let noun = if count == 1 { "event" } else { "events" };
    format!("[… {count} {noun} elided …]")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `SplitMix64`: a fixed-seed corpus, so every assertion below is about the
    /// same series on every run and every machine.
    struct Corpus(u64);

    impl Corpus {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        /// A salience in `[0, 1)` from the top 8 bits, exactly representable.
        fn salience(&mut self) -> f32 {
            let [top, ..] = self.next().to_be_bytes();
            f32::from(top) / 256.0
        }
    }

    fn series(seed: u64, n: i64) -> Vec<MemoryEventRow> {
        let mut corpus = Corpus(seed);
        (0..n)
            .map(|i| {
                let outcome = corpus.next().is_multiple_of(9);
                MemoryEventRow {
                    id: i,
                    event_id: format!("e{i}"),
                    ts_unix_ms: 1_000 * i,
                    kind: if outcome { "outcome" } else { "message" }.to_string(),
                    workspace_id: Some("ws1".to_string()),
                    content: format!("event {i}"),
                    content_len_chars: 8,
                    embedding: None,
                    embedding_model: None,
                    salience: corpus.salience(),
                    source_ids: vec!["task:7".to_string(), format!("task_note:{i}")],
                    created_at: String::new(),
                }
            })
            .collect()
    }

    fn contains_event(folded: &CompressedEpisode, i: usize) -> bool {
        folded
            .episode
            .content
            .lines()
            .any(|line| line.ends_with(&format!(" event {i}")))
    }

    /// The `dreaming.timeline_compression` and
    /// `dreaming.timeline_transition_retention` budgets (bench/budgets.toml):
    /// a 100-event series folded at the shipped budget drops at least 75% of
    /// its events and keeps every transition.
    #[test]
    fn budget_gate_timeline_fold() {
        let events = series(2026, 100);
        let transitions: Vec<usize> = (0..events.len())
            .filter(|&i| events[i].kind == "outcome")
            .collect();
        assert!(
            transitions.len() + MIN_BUDGET <= DREAM_FOLD_BUDGET,
            "the corpus must leave room for every transition"
        );
        let folded = compress_episode(&events, DREAM_FOLD_BUDGET).expect("folds");
        let compression = 1.0 - f64::from(u32::try_from(folded.kept).unwrap_or(u32::MAX)) / 100.0;
        println!(
            "dreaming.timeline_compression = {compression:.2}; transitions kept {}/{}",
            transitions
                .iter()
                .filter(|&&i| contains_event(&folded, i))
                .count(),
            transitions.len()
        );
        assert!(compression >= 0.75, "compression {compression}");
        for &i in &transitions {
            assert!(contains_event(&folded, i), "transition {i} was dropped");
        }
    }

    #[test]
    fn an_empty_series_folds_to_nothing() {
        assert!(compress_episode(&[], 8).is_none());
    }

    #[test]
    fn a_series_within_budget_is_kept_whole() {
        let events = series(1, 6);
        let folded = compress_episode(&events, 6).expect("folds");
        assert_eq!((folded.kept, folded.elided), (6, 0));
        assert!(!folded.episode.content.contains("elided"));
    }

    /// The contract: transitions and peaks survive, the budget holds, and
    /// every event is accounted for either as content or inside a marker.
    #[test]
    fn transitions_and_peaks_survive_and_every_event_is_accounted_for() {
        let events = series(42, 200);
        // Room for every transition and peak plus some decimated filler, so
        // this exercises the decimation branch rather than the triage one.
        let mandatory = (0..events.len())
            .filter(|&i| is_mandatory(&events, i))
            .count();
        let budget = mandatory + 10;
        assert!(
            budget < events.len(),
            "the series must actually need folding"
        );
        let folded = compress_episode(&events, budget).expect("folds");
        assert_eq!(folded.kept, budget, "{}", folded.kept);
        assert_eq!(folded.kept + folded.elided, events.len());
        for (i, event) in events.iter().enumerate() {
            if is_mandatory(&events, i) {
                assert!(
                    contains_event(&folded, i),
                    "event {i} ({}) was dropped",
                    event.kind
                );
            }
        }
        let marked: usize = folded
            .episode
            .content
            .lines()
            .filter_map(|line| line.strip_prefix("[… "))
            .filter_map(|rest| rest.split(' ').next()?.parse::<usize>().ok())
            .sum();
        assert_eq!(
            marked, folded.elided,
            "the markers add up to what was dropped"
        );
        assert!(folded.episode.source_ids.len() <= MAX_EVENT_SOURCE_IDS);
        assert_eq!(
            folded.episode.source_ids[0], "task:7",
            "lineage names the card first"
        );
    }

    #[test]
    fn folding_is_deterministic_and_keeps_series_order() {
        let events = series(7, 120);
        let a = compress_episode(&events, 30).expect("folds");
        let b = compress_episode(&events, 30).expect("folds");
        assert_eq!(a, b);
        let order: Vec<usize> = a
            .episode
            .content
            .lines()
            .filter_map(|line| line.rsplit(' ').next()?.parse::<usize>().ok())
            .collect();
        assert!(order.windows(2).all(|w| w[0] < w[1]), "{order:?}");
    }

    /// More transitions than room: the endpoints stay, and the most salient of
    /// the rest fill the budget.
    #[test]
    fn a_budget_smaller_than_the_transitions_keeps_endpoints_and_the_most_salient() {
        let mut events = series(3, 50);
        for event in &mut events {
            event.kind = "outcome".to_string();
        }
        // One ordinary message among the transitions must lose to them.
        events[20].kind = "message".to_string();
        events[20].salience = 0.99;
        let folded = compress_episode(&events, 5).expect("folds");
        assert_eq!(folded.kept, 5);
        assert!(contains_event(&folded, 0) && contains_event(&folded, 49));
        assert!(
            !contains_event(&folded, 20),
            "a peak never displaces a transition"
        );
        assert_eq!(folded.episode.kind, EventKind::Outcome);
        let floor = compress_episode(&events, 0).expect("folds");
        assert_eq!(
            floor.kept, MIN_BUDGET,
            "a zero budget still keeps the endpoints"
        );
    }
}
