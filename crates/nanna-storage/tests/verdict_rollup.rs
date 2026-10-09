//! Per-member, per-label verdict rollup (P25 Stage 1).
//!
//! The router reads this to pick a member for a card and to adjust capability
//! tags at verdict time. The property that matters most is attribution: a
//! verdict belongs to the member who held the card when it was judged, because
//! a failed verdict sends the card back to the router, which may reassign it.

#![warn(clippy::all)]
#![warn(clippy::pedantic, clippy::nursery)]

use nanna_storage::{
    MemberKind, MemberOwner, MemberStatus, NewMember, NewTask, Storage, TaskPatch, VerdictTally,
};
use serde_json::json;

async fn storage_with_agents(ids: &[&str]) -> Storage {
    let storage = Storage::in_memory().await.expect("in-memory storage");
    for id in ids {
        storage
            .members()
            .create(NewMember {
                id: (*id).to_string(),
                name: (*id).to_string(),
                avatar: None,
                kind: MemberKind::Agent,
                owner_kind: MemberOwner::Workspace,
                owner_id: Some("ws1".to_string()),
                status: MemberStatus::Idle,
                profile: json!({}),
            })
            .await
            .expect("create member");
    }
    storage
}

async fn card(storage: &Storage, labels: &[&str], assignee: &str) -> i64 {
    storage
        .tasks()
        .create(NewTask {
            scope: "workspace".to_string(),
            scope_id: Some("ws1".to_string()),
            title: "card".to_string(),
            priority: 3,
            labels: labels.iter().map(|l| (*l).to_string()).collect(),
            assignee: Some(assignee.to_string()),
            ..NewTask::default()
        })
        .await
        .expect("create card")
        .id
}

async fn judge(storage: &Storage, id: i64, detail: serde_json::Value) {
    storage
        .tasks()
        .log_activity(id, Some("harness"), "acceptance_checked", Some(detail))
        .await
        .expect("log verdict");
}

fn tally(rollup: &[VerdictTally], member: &str, label: Option<&str>) -> (u64, u64) {
    rollup
        .iter()
        .find(|t| t.member_id == member && t.label.as_deref() == label)
        .map_or((0, 0), |t| (t.passed, t.failed))
}

/// The load-bearing case. Reading the CURRENT assignee would charge
/// `agent-a`'s failure to `agent-b`, who was handed the card afterwards —
/// exactly the reassignment a failed verdict triggers.
#[tokio::test]
async fn a_verdict_stays_with_the_member_who_was_judged() {
    let storage = storage_with_agents(&["agent-a", "agent-b"]).await;
    let id = card(&storage, &["rust"], "agent-a").await;
    judge(
        &storage,
        id,
        json!({ "passed": false, "detail": "tests failed" }),
    )
    .await;

    storage
        .tasks()
        .update(
            id,
            TaskPatch {
                assignee: Some(Some("agent-b".to_string())),
                ..TaskPatch::default()
            },
            Some("router"),
        )
        .await
        .expect("reassign");
    judge(&storage, id, json!({ "passed": true })).await;

    let rollup = storage.tasks().verdict_rollup(100).await.expect("rollup");
    assert_eq!(
        tally(&rollup, "agent-a", Some("rust")),
        (0, 1),
        "{rollup:?}"
    );
    assert_eq!(
        tally(&rollup, "agent-b", Some("rust")),
        (1, 0),
        "{rollup:?}"
    );

    let history = storage.tasks().activity(id, 50).await.expect("activity");
    let stamped: Vec<Option<&str>> = history
        .iter()
        .filter(|e| e.action == "acceptance_checked")
        .map(|e| e.assignee.as_deref())
        .collect();
    assert_eq!(
        stamped,
        [Some("agent-a"), Some("agent-b")],
        "the store stamps each row"
    );
}

/// A card counts once toward each of its labels and once toward the member's
/// total; an unlabelled card still counts toward the total.
#[tokio::test]
async fn labels_and_the_total_are_tallied_separately() {
    let storage = storage_with_agents(&["agent-a"]).await;
    let both = card(&storage, &["rust", "docs"], "agent-a").await;
    let bare = card(&storage, &[], "agent-a").await;
    judge(&storage, both, json!({ "passed": true })).await;
    judge(&storage, bare, json!({ "passed": false })).await;

    let rollup = storage.tasks().verdict_rollup(100).await.expect("rollup");
    assert_eq!(
        tally(&rollup, "agent-a", None),
        (1, 1),
        "the total sees both cards"
    );
    assert_eq!(tally(&rollup, "agent-a", Some("rust")), (1, 0));
    assert_eq!(tally(&rollup, "agent-a", Some("docs")), (1, 0));
    assert_eq!(
        rollup.len(),
        3,
        "no row for a label nobody was judged on: {rollup:?}"
    );
}

/// Verdicts that say nothing about the member are left out, not guessed at: a
/// check that could not tell (`unknown`), a row without a boolean `passed`, and
/// an unassigned card.
#[tokio::test]
async fn verdicts_that_say_nothing_about_a_member_are_left_out() {
    let storage = storage_with_agents(&["agent-a"]).await;
    let id = card(&storage, &["rust"], "agent-a").await;
    judge(&storage, id, json!({ "passed": false, "unknown": true })).await;
    judge(&storage, id, json!({ "detail": "no outcome recorded" })).await;

    let unassigned = storage
        .tasks()
        .create(NewTask {
            scope: "workspace".to_string(),
            scope_id: Some("ws1".to_string()),
            title: "nobody's".to_string(),
            priority: 3,
            ..NewTask::default()
        })
        .await
        .expect("create")
        .id;
    judge(&storage, unassigned, json!({ "passed": true })).await;
    // Not a verdict at all.
    storage
        .tasks()
        .log_activity(
            id,
            Some("harness"),
            "replanned",
            Some(json!({ "passed": true })),
        )
        .await
        .expect("log");

    let rollup = storage.tasks().verdict_rollup(100).await.expect("rollup");
    assert!(rollup.is_empty(), "{rollup:?}");
}

/// The window keeps the most recent verdicts: recent outcomes are the ones
/// that predict, and the scan is bounded however long the history grows.
#[tokio::test]
async fn the_window_keeps_the_most_recent_verdicts() {
    let storage = storage_with_agents(&["agent-a"]).await;
    let id = card(&storage, &[], "agent-a").await;
    for _ in 0..3 {
        judge(&storage, id, json!({ "passed": false })).await;
    }
    judge(&storage, id, json!({ "passed": true })).await;
    judge(&storage, id, json!({ "passed": true })).await;

    let recent = storage.tasks().verdict_rollup(2).await.expect("rollup");
    assert_eq!(tally(&recent, "agent-a", None), (2, 0), "{recent:?}");
    let all = storage.tasks().verdict_rollup(10).await.expect("rollup");
    assert_eq!(tally(&all, "agent-a", None), (2, 3), "{all:?}");
}
