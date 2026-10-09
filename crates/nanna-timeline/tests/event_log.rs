#![warn(clippy::pedantic, clippy::nursery, clippy::all)]
//! The properties the episodic log must hold against a real database, not a
//! mock: the wall-clock ordering, the half-open window that lets adjacent
//! buckets tile the axis, idempotent append under redelivery, workspace
//! scoping, and the caps that keep one event from costing more than a day of
//! them.

use nanna_storage::{Storage, StorageConfig};
use nanna_timeline::{
    Episode, EventKind, MAX_EVENT_CONTENT_CHARS, MAX_EVENT_PAGE, MAX_EVENT_SOURCE_IDS, Timeline,
    TimelineError, was_truncated,
};

fn temp_db_path(tag: &str) -> String {
    let dir = std::env::temp_dir().join(format!(
        "nanna_timeline_{tag}_{}_{:p}",
        std::process::id(),
        &raw const tag
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir.join("timeline.db").to_string_lossy().to_string()
}

async fn open(tag: &str) -> Storage {
    Storage::new(&StorageConfig {
        path: temp_db_path(tag),
    })
    .await
    .expect("storage opens")
}

fn episode(kind: EventKind, ts_unix_ms: i64, content: &str) -> Episode {
    Episode {
        kind,
        ts_unix_ms,
        workspace_id: None,
        content: content.to_string(),
        salience: 0.5,
        source_ids: Vec::new(),
    }
}

#[tokio::test]
async fn migration_014_creates_a_usable_event_log() {
    let storage = open("migrate").await;
    let timeline = Timeline::new(&storage);
    // A fresh database answers a query rather than erroring on a missing
    // table, which is the only thing that proves the migration ran.
    let events = timeline.recent(10).await.expect("query a fresh log");
    assert!(events.is_empty());
    assert_eq!(timeline.count_in_range(0, i64::MAX).await.unwrap(), 0);
}

#[tokio::test]
async fn events_come_back_on_the_wall_clock_axis_not_insertion_order() {
    let storage = open("order").await;
    let timeline = Timeline::new(&storage);

    // Appended out of order on purpose: a channel replaying a backlog stamps
    // when things happened, not when they were ingested.
    for ts in [3_000_i64, 1_000, 2_000] {
        timeline
            .append(&episode(EventKind::Message, ts, &format!("at {ts}")))
            .await
            .expect("append");
    }

    let events = timeline.range(0, 10_000, None, 100).await.expect("range");
    let stamps: Vec<i64> = events.iter().map(|e| e.ts_unix_ms).collect();
    assert_eq!(stamps, vec![1_000, 2_000, 3_000]);

    let newest_first: Vec<i64> = timeline
        .recent(100)
        .await
        .expect("recent")
        .iter()
        .map(|e| e.ts_unix_ms)
        .collect();
    assert_eq!(newest_first, vec![3_000, 2_000, 1_000]);
}

#[tokio::test]
async fn adjacent_windows_tile_the_axis_exactly_once() {
    let storage = open("tile").await;
    let timeline = Timeline::new(&storage);
    for ts in [0_i64, 100, 200, 300] {
        timeline
            .append(&episode(EventKind::Outcome, ts, "x"))
            .await
            .expect("append");
    }

    // The boundary event at 200 must appear in exactly one of these.
    let left = timeline.range(0, 200, None, 100).await.unwrap();
    let right = timeline.range(200, 400, None, 100).await.unwrap();
    assert_eq!(left.len(), 2, "0 and 100");
    assert_eq!(right.len(), 2, "200 and 300");
    assert_eq!(
        left.len() + right.len(),
        4,
        "no event counted twice or lost"
    );
    assert_eq!(timeline.count_in_range(0, 200).await.unwrap(), 2);
    assert_eq!(timeline.count_in_range(200, 400).await.unwrap(), 2);
}

#[tokio::test]
async fn redelivering_the_same_event_id_does_not_duplicate_it() {
    let storage = open("idempotent").await;
    let row = nanna_storage::NewMemoryEvent {
        event_id: "stable-id".to_string(),
        ts_unix_ms: 42,
        kind: "message".to_string(),
        workspace_id: None,
        content: "once".to_string(),
        content_len_chars: 4,
        embedding: None,
        embedding_model: None,
        salience: 0.25,
        source_ids: Vec::new(),
    };

    let first = storage.memory_events().append(&row).await.expect("append");
    let second = storage
        .memory_events()
        .append(&row)
        .await
        .expect("re-append");
    assert!(first, "the first delivery inserts");
    assert!(!second, "the redelivery is ignored, not duplicated");
    assert_eq!(
        storage
            .memory_events()
            .count_in_range(0, 100)
            .await
            .unwrap(),
        1
    );
}

#[tokio::test]
async fn a_scoped_range_sees_only_its_workspace() {
    let storage = open("scope").await;
    let timeline = Timeline::new(&storage);

    for (ws, ts) in [(Some("alpha"), 10_i64), (Some("beta"), 20), (None, 30)] {
        timeline
            .append(&Episode {
                workspace_id: ws.map(str::to_string),
                ..episode(EventKind::Recall, ts, "scoped")
            })
            .await
            .expect("append");
    }

    let alpha = timeline.range(0, 100, Some("alpha"), 100).await.unwrap();
    assert_eq!(alpha.len(), 1);
    assert_eq!(alpha[0].ts_unix_ms, 10);

    // Unscoped means the whole axis, including the global event.
    assert_eq!(timeline.range(0, 100, None, 100).await.unwrap().len(), 3);
}

#[tokio::test]
async fn oversized_content_is_capped_and_says_so() {
    let storage = open("cap").await;
    let timeline = Timeline::new(&storage);

    // Multi-byte throughout: a byte-wise cut here would panic mid-character.
    let huge = "—".repeat(MAX_EVENT_CONTENT_CHARS + 250);
    timeline
        .append(&episode(EventKind::ToolCall, 1, &huge))
        .await
        .expect("append survives oversized multi-byte content");

    let stored = &timeline.recent(1).await.unwrap()[0];
    assert_eq!(stored.content.chars().count(), MAX_EVENT_CONTENT_CHARS);
    assert_eq!(
        stored.content_len_chars,
        i64::try_from(MAX_EVENT_CONTENT_CHARS + 250).expect("length fits i64"),
        "the pre-truncation length is what makes the loss visible"
    );
    assert!(was_truncated(stored));
}

#[tokio::test]
async fn lineage_is_capped_and_otherwise_round_trips() {
    let storage = open("lineage").await;
    let timeline = Timeline::new(&storage);

    timeline
        .append(&Episode {
            source_ids: vec!["mem-1".into(), "mem-2".into()],
            ..episode(EventKind::Outcome, 1, "derived")
        })
        .await
        .expect("append");
    timeline
        .append(&Episode {
            source_ids: (0..MAX_EVENT_SOURCE_IDS + 40)
                .map(|i| format!("mem-{i}"))
                .collect(),
            ..episode(EventKind::Outcome, 2, "over-derived")
        })
        .await
        .expect("append");

    let events = timeline.range(0, 10, None, 10).await.unwrap();
    assert_eq!(events[0].source_ids, vec!["mem-1", "mem-2"]);
    assert_eq!(events[1].source_ids.len(), MAX_EVENT_SOURCE_IDS);
}

#[tokio::test]
async fn a_page_larger_than_the_cap_is_refused_rather_than_served() {
    let storage = open("page").await;
    let timeline = Timeline::new(&storage);
    let err = timeline
        .recent(MAX_EVENT_PAGE + 1)
        .await
        .expect_err("the cap must be enforced, not rounded down");
    assert!(matches!(err, TimelineError::PageTooLarge { .. }));

    let err = timeline
        .range(0, 1, None, MAX_EVENT_PAGE + 1)
        .await
        .expect_err("range is capped too");
    assert!(matches!(err, TimelineError::PageTooLarge { .. }));
}

#[tokio::test]
async fn an_inverted_window_is_refused_rather_than_silently_empty() {
    let storage = open("inverted").await;
    let timeline = Timeline::new(&storage);
    let err = timeline
        .range(500, 100, None, 10)
        .await
        .expect_err("a backwards window is a caller bug, not an empty result");
    assert!(matches!(err, TimelineError::InvertedWindow { .. }));
    assert!(timeline.count_in_range(500, 100).await.is_err());
}

#[tokio::test]
async fn salience_outside_the_normalized_range_is_refused() {
    let storage = open("salience").await;
    let timeline = Timeline::new(&storage);
    for bad in [1.5_f32, -0.1, f32::NAN, f32::INFINITY] {
        let err = timeline
            .append(&Episode {
                salience: bad,
                ..episode(EventKind::Message, 1, "x")
            })
            .await
            .expect_err("out-of-range salience must not be clamped silently");
        assert!(matches!(err, TimelineError::SalienceOutOfRange { .. }));
    }
    // And nothing was written by the rejected attempts.
    assert_eq!(timeline.count_in_range(0, 10).await.unwrap(), 0);
}

#[tokio::test]
async fn every_kind_survives_a_round_trip_through_the_database() {
    let storage = open("kinds").await;
    let timeline = Timeline::new(&storage);
    let kinds = [
        EventKind::Message,
        EventKind::ToolCall,
        EventKind::Recall,
        EventKind::Outcome,
    ];
    for (i, kind) in kinds.iter().enumerate() {
        timeline
            .append(&episode(*kind, i64::try_from(i).expect("index fits i64"), "k"))
            .await
            .expect("append");
    }
    let stored = timeline.range(0, 100, None, 100).await.unwrap();
    let round_tripped: Vec<EventKind> = stored
        .iter()
        .map(|e| EventKind::from_str_opt(&e.kind).expect("a stored kind must parse back"))
        .collect();
    assert_eq!(round_tripped, kinds);
}

/// One card's series, by lineage: `task:1` must not pull in `task:10`'s
/// events (the id is matched with its own quotes), and the series comes back
/// oldest first, the order `compress_episode` folds in.
#[tokio::test]
async fn a_cards_series_is_found_by_exact_lineage_oldest_first() {
    let storage = open("for_source").await;
    let timeline = Timeline::new(&storage);
    for (ts, card) in [(30, "task:1"), (10, "task:1"), (20, "task:10")] {
        let mut e = episode(EventKind::Message, ts, card);
        e.source_ids = vec![card.to_string()];
        timeline.append(&e).await.expect("append");
    }
    let series = storage
        .memory_events()
        .for_source("task:1", 10)
        .await
        .expect("series");
    let stamps: Vec<i64> = series.iter().map(|r| r.ts_unix_ms).collect();
    assert_eq!(stamps, [10, 30], "task:10's event is not task:1's");
    assert!(storage.memory_events().for_source("", 10).await.is_err());
    assert!(
        storage
            .memory_events()
            .for_source("a\"b", 10)
            .await
            .is_err()
    );
}
