//! Task lifecycle events (P25 Stage 1).
//!
//! These events are emitted from [`TaskRepository`] rather than from a caller
//! because the repository is the only layer that sees every writer. The tests
//! that matter here are therefore the ones a caller-level emitter would fail:
//! an ancestor auto-completed by somebody else's `complete`, and a change that
//! did not actually change anything.

#![warn(clippy::all)]
#![warn(clippy::pedantic, clippy::nursery)]

use nanna_storage::{
    HUMAN_MEMBER_ID, NewTask, Storage, TaskEvent, TaskEventKind, TaskEventSink, TaskNoteKind,
    TaskPatch,
};
use std::sync::{Arc, Mutex};

/// Records everything published, so a test can assert on the whole sequence
/// rather than on one event in isolation — ordering is part of the contract
/// the router will read.
#[derive(Default)]
struct Recorder {
    events: Mutex<Vec<TaskEvent>>,
}

impl Recorder {
    fn kinds(&self) -> Vec<TaskEventKind> {
        self.events
            .lock()
            .expect("recorder not poisoned")
            .iter()
            .map(|e| e.kind)
            .collect()
    }

    fn of_kind(&self, kind: TaskEventKind) -> Vec<TaskEvent> {
        self.events
            .lock()
            .expect("recorder not poisoned")
            .iter()
            .filter(|e| e.kind == kind)
            .cloned()
            .collect()
    }

    fn clear(&self) {
        self.events.lock().expect("recorder not poisoned").clear();
    }
}

impl TaskEventSink for Recorder {
    fn publish(&self, event: TaskEvent) {
        self.events
            .lock()
            .expect("recorder not poisoned")
            .push(event);
    }
}

async fn storage_with_recorder() -> (Storage, Arc<Recorder>) {
    let storage = Storage::in_memory().await.expect("in-memory storage");
    let recorder = Arc::new(Recorder::default());
    assert!(
        storage.set_task_events(recorder.clone()),
        "first sink attaches"
    );
    (storage, recorder)
}

fn card(title: &str) -> NewTask {
    NewTask {
        scope: "workspace".to_string(),
        scope_id: Some("ws1".to_string()),
        title: title.to_string(),
        priority: 3,
        ..NewTask::default()
    }
}

#[tokio::test]
async fn creating_a_card_announces_it() {
    let (storage, recorder) = storage_with_recorder().await;
    let task = storage
        .tasks()
        .create(card("write the thing"))
        .await
        .expect("created");

    assert_eq!(recorder.kinds(), vec![TaskEventKind::Created]);
    let events = recorder.of_kind(TaskEventKind::Created);
    let event = events.first().expect("one created event");
    assert_eq!(event.task_id, task.id);
    assert_eq!(event.scope, "workspace");
    assert_eq!(event.scope_id.as_deref(), Some("ws1"));
    assert_eq!(event.detail["title"], "write the thing");
}

/// The `created` actor is whoever made the card, never its assignee (the
/// activity row stamps the assignee in its own column). Before, a card made
/// by one member for another named the wrong member as its author, and the
/// board router could not tell cards it split off itself from new work.
#[tokio::test]
async fn the_created_actor_is_the_creator_not_the_assignee() {
    let (storage, recorder) = storage_with_recorder().await;
    let task = storage
        .tasks()
        .create(NewTask {
            assignee: Some("human".to_string()),
            created_by: Some("router:global".to_string()),
            ..card("clarify the brief")
        })
        .await
        .expect("created");
    let events = recorder.of_kind(TaskEventKind::Created);
    assert_eq!(events[0].actor.as_deref(), Some("router:global"));
    let activity = storage
        .tasks()
        .activity(task.id, 10)
        .await
        .expect("activity");
    assert_eq!(activity[0].actor.as_deref(), Some("router:global"));
    assert_eq!(
        activity[0].assignee.as_deref(),
        Some("human"),
        "the assignee has its own column"
    );

    storage
        .tasks()
        .create(NewTask {
            assignee: Some("human".to_string()),
            ..card("unknown author")
        })
        .await
        .expect("created");
    let events = recorder.of_kind(TaskEventKind::Created);
    assert_eq!(
        events[1].actor, None,
        "an unknown creator is not guessed from the assignee"
    );
}

#[tokio::test]
async fn a_status_change_is_announced_once_and_only_when_it_changes() {
    let (storage, recorder) = storage_with_recorder().await;
    let repo = storage.tasks();
    let task = repo.create(card("a card")).await.expect("created");
    recorder.clear();

    repo.update(
        task.id,
        TaskPatch {
            status: Some("in_progress".to_string()),
            ..TaskPatch::default()
        },
        Some("gui"),
    )
    .await
    .expect("status updated");
    assert_eq!(recorder.kinds(), vec![TaskEventKind::StatusChanged]);

    // Re-writing the same status is not a transition, and a consumer woken by
    // one would be reacting to nothing.
    recorder.clear();
    repo.update(
        task.id,
        TaskPatch {
            status: Some("in_progress".to_string()),
            ..TaskPatch::default()
        },
        Some("gui"),
    )
    .await
    .expect("no-op status update");
    assert_eq!(recorder.kinds(), Vec::<TaskEventKind>::new());
}

#[tokio::test]
async fn reassignment_is_announced_but_a_no_op_reassign_is_not() {
    let (storage, recorder) = storage_with_recorder().await;
    let repo = storage.tasks();
    let task = repo.create(card("a card")).await.expect("created");
    recorder.clear();

    repo.update(
        task.id,
        TaskPatch {
            assignee: Some(Some(HUMAN_MEMBER_ID.to_string())),
            ..TaskPatch::default()
        },
        Some("router"),
    )
    .await
    .expect("assigned");
    let events = recorder.of_kind(TaskEventKind::Assigned);
    assert_eq!(events.len(), 1, "one assignment");
    assert_eq!(events[0].detail["assignee"], HUMAN_MEMBER_ID);
    assert_eq!(events[0].actor.as_deref(), Some("router"));

    // The regression this guards: `apply_patch` used to record an assignee
    // change whenever the field was present, with no old/new comparison, so
    // re-writing the same member looked like a hand-off and would have woken
    // the router for nothing.
    recorder.clear();
    repo.update(
        task.id,
        TaskPatch {
            assignee: Some(Some(HUMAN_MEMBER_ID.to_string())),
            ..TaskPatch::default()
        },
        Some("router"),
    )
    .await
    .expect("no-op reassign");
    assert_eq!(recorder.kinds(), Vec::<TaskEventKind>::new());
}

#[tokio::test]
async fn every_thread_post_is_announced() {
    let (storage, recorder) = storage_with_recorder().await;
    let repo = storage.tasks();
    let task = repo.create(card("a card")).await.expect("created");
    recorder.clear();

    let note = repo
        .post(
            task.id,
            Some("harness"),
            None,
            TaskNoteKind::Question,
            "which one?",
        )
        .await
        .expect("posted");
    let events = recorder.of_kind(TaskEventKind::Posted);
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].detail["note_id"], note.id);
    assert_eq!(events[0].detail["kind"], "question");

    // `add_note` is the thin Comment wrapper over `post`; it must announce too,
    // or the harness's own findings would be the one silent writer.
    recorder.clear();
    repo.add_note(task.id, Some("harness"), "a finding")
        .await
        .expect("noted");
    assert_eq!(recorder.kinds(), vec![TaskEventKind::Posted]);
}

#[tokio::test]
async fn a_failed_post_announces_nothing() {
    let (storage, recorder) = storage_with_recorder().await;
    let repo = storage.tasks();
    let task = repo.create(card("a card")).await.expect("created");
    recorder.clear();

    repo.post(task.id, Some("harness"), None, TaskNoteKind::Comment, "   ")
        .await
        .expect_err("empty content is rejected");
    assert_eq!(
        recorder.kinds(),
        Vec::<TaskEventKind>::new(),
        "an event for a post nobody can read would be a lie"
    );
}

#[tokio::test]
async fn completing_announces_a_verdict() {
    let (storage, recorder) = storage_with_recorder().await;
    let repo = storage.tasks();
    let task = repo.create(card("a card")).await.expect("created");
    recorder.clear();

    repo.complete(task.id, Some("gui"), None)
        .await
        .expect("completed");
    let events = recorder.of_kind(TaskEventKind::Verdict);
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].task_id, task.id);
    assert_eq!(events[0].detail["auto"], false);
}

/// The case that decides where these events live.
///
/// Completing the last child auto-completes the parent, and **no caller ever
/// names the parent**. An emitter sitting above the repository — in the IPC
/// handler or a tool service — would announce the child and let the parent go
/// done in silence.
#[tokio::test]
async fn an_ancestor_closed_by_a_cascade_gets_its_own_verdict() {
    let (storage, recorder) = storage_with_recorder().await;
    let repo = storage.tasks();
    let parent = repo
        .create(card("the parent"))
        .await
        .expect("parent created");
    let child = repo
        .create(NewTask {
            parent_id: Some(parent.id),
            ..card("the child")
        })
        .await
        .expect("child created");
    recorder.clear();

    let outcome = repo
        .complete(child.id, Some("harness"), None)
        .await
        .expect("completed");
    assert_eq!(
        outcome.auto_completed,
        vec![parent.id],
        "parent auto-completes"
    );

    let verdicts = recorder.of_kind(TaskEventKind::Verdict);
    assert_eq!(verdicts.len(), 2, "child and parent both announced");
    assert_eq!(verdicts[0].task_id, child.id);
    assert_eq!(verdicts[0].detail["auto"], false);
    assert_eq!(verdicts[1].task_id, parent.id);
    assert_eq!(verdicts[1].detail["auto"], true);
    assert_eq!(verdicts[1].detail["trigger"], child.id);
}

#[tokio::test]
async fn completing_an_already_done_card_announces_nothing() {
    let (storage, recorder) = storage_with_recorder().await;
    let repo = storage.tasks();
    let task = repo.create(card("a card")).await.expect("created");
    repo.complete(task.id, Some("gui"), None)
        .await
        .expect("completed");
    recorder.clear();

    let outcome = repo
        .complete(task.id, Some("gui"), None)
        .await
        .expect("idempotent");
    assert!(outcome.already_done);
    assert_eq!(
        recorder.kinds(),
        Vec::<TaskEventKind>::new(),
        "completion is idempotent, so its announcement must be too"
    );
}

#[tokio::test]
async fn a_store_with_no_sink_still_works() {
    // Every existing caller and every other test runs without a sink; the
    // events must be an addition, not a requirement.
    let storage = Storage::in_memory().await.expect("in-memory storage");
    let repo = storage.tasks();
    let task = repo.create(card("a card")).await.expect("created");
    repo.add_note(task.id, Some("harness"), "a note")
        .await
        .expect("noted");
    repo.complete(task.id, Some("gui"), None)
        .await
        .expect("completed");
}

#[tokio::test]
async fn a_second_sink_is_refused_rather_than_silently_replacing_the_first() {
    let (storage, recorder) = storage_with_recorder().await;
    let second = Arc::new(Recorder::default());
    assert!(
        !storage.set_task_events(second.clone()),
        "attaching twice is a wiring bug, and the first sink keeps receiving"
    );

    storage
        .tasks()
        .create(card("a card"))
        .await
        .expect("created");
    assert_eq!(recorder.kinds(), vec![TaskEventKind::Created]);
    assert_eq!(second.kinds(), Vec::<TaskEventKind>::new());
}

// ---------------------------------------------------------------------------
// Derived transitions: `blocked` / `unblocked`
//
// `blocked` is computed on read and never stored, so these events are a
// difference between two derivations rather than a column that changed. The
// tests below are written around the four mutations that can move it.
// ---------------------------------------------------------------------------

/// `b` depends on `a`, so `b` starts blocked.
async fn blocked_pair(storage: &Storage) -> (i64, i64) {
    let repo = storage.tasks();
    let a = repo
        .create(card("the dependency"))
        .await
        .expect("a created");
    let b = repo
        .create(NewTask {
            depends_on: vec![a.id],
            ..card("the dependent")
        })
        .await
        .expect("b created");
    let fetched = repo.get(b.id).await.expect("read back");
    assert!(fetched.blocked, "b starts blocked");
    (a.id, b.id)
}

#[tokio::test]
async fn completing_a_dependency_unblocks_its_dependent() {
    let (storage, recorder) = storage_with_recorder().await;
    let (a, b) = blocked_pair(&storage).await;
    recorder.clear();

    storage
        .tasks()
        .complete(a, Some("harness"), None)
        .await
        .expect("dependency completed");

    let unblocked = recorder.of_kind(TaskEventKind::Unblocked);
    assert_eq!(unblocked.len(), 1, "exactly the one dependent");
    assert_eq!(unblocked[0].task_id, b);
    assert!(
        !storage.tasks().get(b).await.expect("read back").blocked,
        "and the derived flag agrees with the event"
    );
}

#[tokio::test]
async fn cancelling_a_dependency_also_unblocks() {
    // `is_blocked` treats cancelled as closed, so a cancel releases dependents
    // exactly as a completion does — and a consumer that only watched
    // completions would miss it.
    let (storage, recorder) = storage_with_recorder().await;
    let (a, b) = blocked_pair(&storage).await;
    recorder.clear();

    storage
        .tasks()
        .update(
            a,
            TaskPatch {
                status: Some("cancelled".to_string()),
                ..TaskPatch::default()
            },
            Some("gui"),
        )
        .await
        .expect("dependency cancelled");

    let unblocked = recorder.of_kind(TaskEventKind::Unblocked);
    assert_eq!(unblocked.len(), 1);
    assert_eq!(unblocked[0].task_id, b);
}

#[tokio::test]
async fn reopening_a_dependency_blocks_its_dependent_again() {
    let (storage, recorder) = storage_with_recorder().await;
    let (a, b) = blocked_pair(&storage).await;
    let repo = storage.tasks();
    repo.complete(a, Some("harness"), None)
        .await
        .expect("completed");
    recorder.clear();

    repo.reopen(a, Some("recurrence")).await.expect("reopened");

    let blocked = recorder.of_kind(TaskEventKind::Blocked);
    assert_eq!(blocked.len(), 1, "the transition runs both ways");
    assert_eq!(blocked[0].task_id, b);
    assert!(repo.get(b).await.expect("read back").blocked);
}

#[tokio::test]
async fn deleting_a_dependency_unblocks_and_does_not_announce_the_deleted_card() {
    let (storage, recorder) = storage_with_recorder().await;
    let (a, b) = blocked_pair(&storage).await;
    recorder.clear();

    storage
        .tasks()
        .delete(a, Some("gui"))
        .await
        .expect("deleted");

    let unblocked = recorder.of_kind(TaskEventKind::Unblocked);
    assert_eq!(unblocked.len(), 1, "only the survivor transitions");
    assert_eq!(
        unblocked[0].task_id, b,
        "a deleted card did not become unblocked, it stopped existing"
    );
}

#[tokio::test]
async fn dropping_the_dependency_edge_unblocks_without_closing_anything() {
    let (storage, recorder) = storage_with_recorder().await;
    let (_a, b) = blocked_pair(&storage).await;
    recorder.clear();

    storage
        .tasks()
        .update(
            b,
            TaskPatch {
                depends_on: Some(Vec::new()),
                ..TaskPatch::default()
            },
            Some("router"),
        )
        .await
        .expect("edge removed");

    let unblocked = recorder.of_kind(TaskEventKind::Unblocked);
    assert_eq!(unblocked.len(), 1);
    assert_eq!(unblocked[0].task_id, b);
}

#[tokio::test]
async fn gaining_an_open_dependency_blocks() {
    let (storage, recorder) = storage_with_recorder().await;
    let repo = storage.tasks();
    let a = repo
        .create(card("the dependency"))
        .await
        .expect("a created");
    let b = repo.create(card("the dependent")).await.expect("b created");
    recorder.clear();

    repo.update(
        b.id,
        TaskPatch {
            depends_on: Some(vec![a.id]),
            ..TaskPatch::default()
        },
        Some("router"),
    )
    .await
    .expect("edge added");

    let blocked = recorder.of_kind(TaskEventKind::Blocked);
    assert_eq!(blocked.len(), 1);
    assert_eq!(blocked[0].task_id, b.id);
}

#[tokio::test]
async fn a_dependent_with_two_dependencies_unblocks_only_on_the_last_one() {
    // The whole point of deriving the flag rather than storing it: partial
    // progress is not a transition, and a consumer woken at the halfway point
    // would start work that is still blocked.
    let (storage, recorder) = storage_with_recorder().await;
    let repo = storage.tasks();
    let a = repo.create(card("first")).await.expect("a");
    let b = repo.create(card("second")).await.expect("b");
    let c = repo
        .create(NewTask {
            depends_on: vec![a.id, b.id],
            ..card("the dependent")
        })
        .await
        .expect("c");
    recorder.clear();

    repo.complete(a.id, Some("harness"), None)
        .await
        .expect("first done");
    assert_eq!(
        recorder.of_kind(TaskEventKind::Unblocked).len(),
        0,
        "one of two is not an unblock"
    );

    repo.complete(b.id, Some("harness"), None)
        .await
        .expect("second done");
    let unblocked = recorder.of_kind(TaskEventKind::Unblocked);
    assert_eq!(unblocked.len(), 1, "the last one releases it");
    assert_eq!(unblocked[0].task_id, c.id);
}

#[tokio::test]
async fn a_mutation_that_changes_no_blocking_announces_no_transition() {
    let (storage, recorder) = storage_with_recorder().await;
    let (_a, _b) = blocked_pair(&storage).await;
    let task = storage
        .tasks()
        .create(card("unrelated"))
        .await
        .expect("created");
    recorder.clear();

    storage
        .tasks()
        .update(
            task.id,
            TaskPatch {
                title: Some("renamed".to_string()),
                ..TaskPatch::default()
            },
            Some("gui"),
        )
        .await
        .expect("renamed");

    assert_eq!(recorder.of_kind(TaskEventKind::Blocked).len(), 0);
    assert_eq!(recorder.of_kind(TaskEventKind::Unblocked).len(), 0);
}

// ---------------------------------------------------------------------------
// Time-driven: `due` / `overdue`
//
// These have no write to hang off — a sweep notices them — so the property
// that matters is that a sweep running every five minutes announces a card
// ONCE per crossing rather than once per pass.
// ---------------------------------------------------------------------------

fn dated(title: &str, due_at: Option<&str>, deadline_at: Option<&str>) -> NewTask {
    NewTask {
        due_at: due_at.map(str::to_string),
        deadline_at: deadline_at.map(str::to_string),
        ..card(title)
    }
}

#[tokio::test]
async fn a_defer_date_that_has_arrived_is_announced_due() {
    let (storage, recorder) = storage_with_recorder().await;
    let repo = storage.tasks();
    let task = repo
        .create(dated("starts today", Some("2026-07-20"), None))
        .await
        .expect("created");
    recorder.clear();

    let (due, overdue) = repo.announce_due("2026-07-20").await.expect("swept");
    assert_eq!((due, overdue), (1, 0));
    let events = recorder.of_kind(TaskEventKind::Due);
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].task_id, task.id);
}

#[tokio::test]
async fn a_future_defer_date_is_not_announced() {
    let (storage, recorder) = storage_with_recorder().await;
    let repo = storage.tasks();
    repo.create(dated("starts next week", Some("2026-07-27"), None))
        .await
        .expect("created");
    recorder.clear();

    let (due, overdue) = repo.announce_due("2026-07-20").await.expect("swept");
    assert_eq!((due, overdue), (0, 0));
    assert_eq!(recorder.kinds(), Vec::<TaskEventKind>::new());
}

/// The property the marker columns exist for.
#[tokio::test]
async fn a_repeated_sweep_announces_nothing_twice() {
    let (storage, recorder) = storage_with_recorder().await;
    let repo = storage.tasks();
    repo.create(dated("late", Some("2026-07-01"), Some("2026-07-10")))
        .await
        .expect("created");
    recorder.clear();

    let first = repo.announce_due("2026-07-20").await.expect("first sweep");
    assert_eq!(first, (1, 1), "due and overdue on the first crossing");
    recorder.clear();

    for _ in 0..3 {
        let again = repo.announce_due("2026-07-20").await.expect("later sweep");
        assert_eq!(again, (0, 0));
    }
    assert_eq!(
        recorder.kinds(),
        Vec::<TaskEventKind>::new(),
        "a 5-minute sweep must not re-announce the same card forever"
    );
}

#[tokio::test]
async fn a_deadline_is_not_overdue_on_the_day_it_falls() {
    // Dates are compared at day granularity throughout the store, so a card due
    // "2026-07-20" has all of that day; it is late on the 21st.
    let (storage, _recorder) = storage_with_recorder().await;
    let repo = storage.tasks();
    repo.create(dated("due today", None, Some("2026-07-20")))
        .await
        .expect("created");

    let (_, overdue) = repo.announce_due("2026-07-20").await.expect("swept");
    assert_eq!(overdue, 0, "still has the day");

    let (_, overdue) = repo.announce_due("2026-07-21").await.expect("swept");
    assert_eq!(overdue, 1, "late the next day");
}

#[tokio::test]
async fn a_day_only_deadline_and_a_full_timestamp_behave_the_same() {
    // A raw string compare would make `2026-07-20` overdue against
    // `2026-07-20T00:01:00Z` — the exact bug day-granularity avoids.
    let (storage, _recorder) = storage_with_recorder().await;
    let repo = storage.tasks();
    repo.create(dated("timestamped", None, Some("2026-07-20T23:59:00Z")))
        .await
        .expect("created");

    let (_, overdue) = repo
        .announce_due("2026-07-20T00:01:00Z")
        .await
        .expect("swept");
    assert_eq!(overdue, 0);
}

#[tokio::test]
async fn a_closed_card_is_never_late() {
    let (storage, recorder) = storage_with_recorder().await;
    let repo = storage.tasks();
    let task = repo
        .create(dated("finished early", None, Some("2026-07-10")))
        .await
        .expect("created");
    repo.complete(task.id, Some("gui"), None)
        .await
        .expect("done");
    recorder.clear();

    let (due, overdue) = repo.announce_due("2026-07-20").await.expect("swept");
    assert_eq!((due, overdue), (0, 0), "a card that is done is not late");
}

#[tokio::test]
async fn moving_the_date_re_arms_the_announcement() {
    let (storage, _recorder) = storage_with_recorder().await;
    let repo = storage.tasks();
    let task = repo
        .create(dated("slipping", Some("2026-07-01"), Some("2026-07-10")))
        .await
        .expect("created");
    assert_eq!(
        repo.announce_due("2026-07-20").await.expect("swept"),
        (1, 1)
    );

    // Pushed out: both must be able to fire again against the new dates.
    repo.update(
        task.id,
        TaskPatch {
            due_at: Some(Some("2026-08-01".to_string())),
            deadline_at: Some(Some("2026-08-10".to_string())),
            ..TaskPatch::default()
        },
        Some("gui"),
    )
    .await
    .expect("dates moved");
    assert_eq!(
        repo.announce_due("2026-07-20").await.expect("swept"),
        (0, 0),
        "not yet — the new dates are in the future"
    );
    assert_eq!(
        repo.announce_due("2026-08-11").await.expect("swept"),
        (1, 1),
        "and they announce again once reached"
    );
}

#[tokio::test]
async fn reopening_re_arms_both_announcements() {
    // A recurring card comes back around and must be able to fall due again;
    // a marker left set would mean it silently never does.
    let (storage, _recorder) = storage_with_recorder().await;
    let repo = storage.tasks();
    let task = repo
        .create(dated("weekly", Some("2026-07-01"), Some("2026-07-10")))
        .await
        .expect("created");
    assert_eq!(
        repo.announce_due("2026-07-20").await.expect("swept"),
        (1, 1)
    );

    repo.complete(task.id, Some("harness"), None)
        .await
        .expect("done");
    repo.reopen(task.id, Some("recurrence"))
        .await
        .expect("reopened");

    assert_eq!(
        repo.announce_due("2026-07-20").await.expect("swept"),
        (1, 1),
        "the reopened card announces again"
    );
}
