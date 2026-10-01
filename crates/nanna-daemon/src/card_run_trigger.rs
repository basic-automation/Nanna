//! Starts a member's run when a board card is assigned to it (P25 Stage 3).
//!
//! [`crate::tasks::TaskRunManager::start_card`] runs one member on one card;
//! this module decides *when*. Like the board router's wakes it is fed by the
//! store's own event sink, not by a bus subscriber (a lagging `broadcast`
//! receiver loses events, and a lost `assigned` is a card nobody ever works):
//! [`crate::task_event_bridge::TaskEventBridge`] hands every card that may
//! have become workable to a bounded queue, and one worker starts the runs.
//!
//! **Four things make a card workable**, so four things wake the worker:
//! - it was assigned to an agent ([`RunWake::Card`] from `assigned`);
//! - the card it waited on closed ([`RunWake::Card`] from `unblocked`);
//! - its date arrived — P25 decision 10, a date defers a card
//!   ([`RunWake::Card`] from `due`);
//! - its member finished another card — decision 3, a member works one card
//!   at a time, so a card assigned to a busy member waits for this
//!   ([`RunWake::MemberFree`], sent by the run manager).
//!
//! **Only cards the board client created are started, for now** — the same
//! rule, for the same reason, as the router's wakes (see
//! [`crate::board_router_trigger`]): while the chat harness lives, its own
//! workspace-scoped cards must not grow runs of their own.

use crate::board_router_trigger::{AGENT_MEMBER_PREFIX, created_by_board_client};
use crate::control::ControlPlane;
use nanna_storage::{Storage, Task, TaskEvent, TaskEventKind};
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

/// Wakes the run queue holds before the sink starts reporting drops.
///
/// Bound justification: the largest burst is one event per card of a scope
/// (a bulk assignment, or every deferred card of a board coming due at
/// once), which the store caps at [`nanna_storage::TASKS_PER_SCOPE_MAX`];
/// a queued wake is a card id or a member id (≤ 128 bytes), so a whole
/// scope's worth stays around a megabyte.
pub const RUN_QUEUE_MAX: usize = nanna_storage::TASKS_PER_SCOPE_MAX;

/// How many of a freed member's waiting cards are read to find the next one.
///
/// Bound justification: a member works one card at a time, so a long wait
/// list is cards that were refused (blocked, deferred) and stay refused;
/// reading 64 of them, best first, finds the next workable one in any board
/// a person could keep in their head, as one bounded indexed read.
pub const MEMBER_NEXT_CARDS_MAX: usize = 64;

/// How many of the newest run markers are read at boot to find the card runs
/// a dead daemon left unfinished.
///
/// Bound justification: a run leaves two markers, and the runs a daemon dies
/// inside are the last ones it started — at most one per member, since a
/// member works one card at a time. 1 024 rows reach back 512 runs, beyond
/// any roster a board holds, in one bounded read.
pub const RESUME_SCAN_ROWS: usize = 1024;

/// Why the run worker was woken.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunWake {
    /// Card `id` may have become workable.
    Card(i64),
    /// This member's card run ended: start its next card, if one waits.
    MemberFree(String),
}

/// The wake `event` produces, if any. Pure, so the sink can call it on the
/// task store's write path; the worker does every read.
#[must_use]
pub fn run_wake_for(event: &TaskEvent) -> Option<RunWake> {
    if event.scope == "session" {
        return None;
    }
    let wakes = match event.kind {
        TaskEventKind::Assigned => event
            .detail
            .get("assignee")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|assignee| assignee.starts_with(AGENT_MEMBER_PREFIX)),
        // A card that waited on another (a clarification, a dependency) may
        // be workable now — including one the router gave back to the member
        // it already had, which emits no `assigned`.
        TaskEventKind::Due | TaskEventKind::Unblocked => true,
        _ => false,
    };
    debug_assert!(event.task_id > 0, "store ids start at 1");
    wakes.then_some(RunWake::Card(event.task_id))
}

/// Whether `card` can be worked by its assignee right now: open and not yet
/// picked up, not waiting on another card, assigned to an agent, on a board,
/// and not deferred to a later date (see [`is_deferred`]).
#[must_use]
pub fn is_startable(card: &Task, now: chrono::DateTime<chrono::Utc>) -> bool {
    card.status == "pending"
        && card.completed_at.is_none()
        && !card.blocked
        && card.scope != "session"
        && card
            .assignee
            .as_deref()
            .is_some_and(|assignee| assignee.starts_with(AGENT_MEMBER_PREFIX))
        && !is_deferred(card, now)
}

/// P25 decision 10: a *date* defers a card until it arrives. A date that does
/// not parse defers nothing — the card is shown and worked, never hidden by a
/// value nobody can read.
#[must_use]
pub fn is_deferred(card: &Task, now: chrono::DateTime<chrono::Utc>) -> bool {
    card.due_at
        .as_deref()
        .and_then(crate::tasks::parse_db_time)
        .is_some_and(|date| date > now)
}

/// Drain the run queue until every sender is gone.
///
/// One wake at a time: starting a run is a few store reads and a spawn, and
/// the run manager — not this loop — enforces one card per member and one run
/// per card, so ordering here only decides which card a freed member takes.
pub async fn run(
    mut queue: mpsc::Receiver<RunWake>,
    control: Arc<ControlPlane>,
    storage: Arc<Storage>,
) {
    info!("Card runs armed: a board card assigned to an agent is worked by that agent");
    let resumed = resume_interrupted(&control, &storage).await;
    if resumed > 0 {
        info!(
            resumed,
            "card runs the last daemon died inside were resumed"
        );
    }
    while let Some(wake) = queue.recv().await {
        match wake {
            RunWake::Card(card_id) => {
                try_start(&control, &storage, card_id).await;
            }
            RunWake::MemberFree(member_id) => start_next_for(&control, &storage, &member_id).await,
        }
    }
    debug!("card run queue closed; worker exiting");
}

/// Resume the card runs the previous daemon died inside; returns how many
/// started again.
///
/// Called once, when the worker starts — nothing can be running yet, so a
/// card whose newest run marker is a start was interrupted, not paused (a
/// cancelled run ends with its marker). Each is closed out with an end
/// marker, its member shown idle again, the card put back to `pending` and
/// started like any assignment — the task store is the run's checkpoint,
/// exactly as for the run manager's own provider-incident resumes.
pub async fn resume_interrupted(control: &ControlPlane, storage: &Storage) -> usize {
    use crate::tasks::{RUN_ENDED_ACTION, RUN_STARTED_ACTION};
    let tasks = storage.tasks();
    let interrupted = match tasks
        .cards_with_unended(RUN_STARTED_ACTION, RUN_ENDED_ACTION, RESUME_SCAN_ROWS)
        .await
    {
        Ok(interrupted) => interrupted,
        Err(e) => {
            warn!(error = %e, "card runs: interrupted runs unreadable; none resumed");
            return 0;
        }
    };
    let mut resumed = 0usize;
    for card_id in interrupted {
        let detail = serde_json::json!({ "stop": "the daemon stopped mid-run" });
        if let Err(e) = tasks
            .log_activity(card_id, Some("daemon"), RUN_ENDED_ACTION, Some(detail))
            .await
        {
            warn!(card_id, error = %e, "card runs: interrupted run not closed out; not resumed");
            continue;
        }
        let Ok(card) = tasks.get(card_id).await else {
            continue;
        };
        let Some(member_id) = card.assignee.clone() else {
            continue;
        };
        if let Err(e) = storage
            .members()
            .set_status(&member_id, nanna_storage::MemberStatus::Idle)
            .await
        {
            debug!(card_id, member = %member_id, error = %e, "card runs: member status not reset");
        }
        if card.status == "in_progress" {
            let wait = nanna_storage::TaskPatch {
                status: Some("pending".to_string()),
                ..nanna_storage::TaskPatch::default()
            };
            if let Err(e) = tasks.update(card_id, wait, Some(&member_id)).await {
                warn!(card_id, error = %e, "card runs: interrupted card left in progress");
                continue;
            }
        }
        if try_start(control, storage, card_id).await {
            resumed += 1;
        }
    }
    resumed
}

/// Start the best waiting card assigned to `member_id`, if any can start.
async fn start_next_for(control: &ControlPlane, storage: &Storage, member_id: &str) {
    let waiting = match storage
        .tasks()
        .open_board_cards_assigned_to(member_id, MEMBER_NEXT_CARDS_MAX)
        .await
    {
        Ok(waiting) => waiting,
        Err(e) => {
            warn!(member = member_id, error = %e, "card runs: waiting cards unreadable");
            return;
        }
    };
    for card in waiting {
        if try_start(control, storage, card.id).await {
            return;
        }
    }
    debug!(
        member = member_id,
        "card runs: nothing waiting for a freed member"
    );
}

/// Start card `card_id`'s run if it is workable now; `true` when it started.
///
/// Every refusal is ordinary — the card is closed, blocked, deferred, held
/// by a human, from the chat harness, or its member is busy (it is picked up
/// again when the member frees) — and is logged at debug.
async fn try_start(control: &ControlPlane, storage: &Storage, card_id: i64) -> bool {
    let tasks = storage.tasks();
    let card = match tasks.get(card_id).await {
        Ok(card) => card,
        Err(e) => {
            warn!(card_id, error = %e, "card runs: card unreadable; not started");
            return false;
        }
    };
    if !is_startable(&card, chrono::Utc::now()) {
        debug!(card_id, status = %card.status, "card runs: not workable now");
        return false;
    }
    match crate::tasks::subtree_has_work(storage, &card).await {
        Ok(true) => {}
        Ok(false) => {
            debug!(
                card_id,
                "card runs: nothing in the card's subtree can be worked yet"
            );
            return false;
        }
        Err(e) => {
            warn!(card_id, error = %e, "card runs: subtree unreadable; not started");
            return false;
        }
    }
    match created_by_board_client(&tasks, card_id).await {
        Ok(true) => {}
        Ok(false) => {
            debug!(card_id, "card runs: not a board-client card; not started");
            return false;
        }
        Err(e) => {
            warn!(card_id, error = %e, "card runs: creator unreadable; not started");
            return false;
        }
    }
    let reply = control.start_assigned_card(card_id).await;
    let started = reply.get("started") == Some(&serde_json::Value::Bool(true));
    if started {
        info!(card_id, member = ?card.assignee, "card run started on assignment");
    } else {
        debug!(card_id, %reply, "card runs: start refused");
    }
    started
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn event(kind: TaskEventKind, scope: &str, detail: serde_json::Value) -> TaskEvent {
        TaskEvent {
            kind,
            task_id: 5,
            scope: scope.to_string(),
            scope_id: None,
            actor: Some("router:global".to_string()),
            detail,
        }
    }

    #[test]
    fn an_agent_assignment_or_an_arriving_date_wakes_the_worker() {
        let to_agent = event(
            TaskEventKind::Assigned,
            "global",
            json!({"assignee": "agent:builder"}),
        );
        assert_eq!(run_wake_for(&to_agent), Some(RunWake::Card(5)));
        let due = event(TaskEventKind::Due, "workspace", json!({}));
        assert_eq!(run_wake_for(&due), Some(RunWake::Card(5)));
        let unblocked = event(TaskEventKind::Unblocked, "global", json!({}));
        assert_eq!(run_wake_for(&unblocked), Some(RunWake::Card(5)));
        // A person's card is theirs to do; a released card is the router's.
        for assignee in [json!("human"), json!(null)] {
            let not_an_agent = event(
                TaskEventKind::Assigned,
                "global",
                json!({"assignee": assignee}),
            );
            assert_eq!(run_wake_for(&not_an_agent), None);
        }
        // Chat scaffolding never grows a card run; other kinds are other triggers.
        let chat = event(
            TaskEventKind::Assigned,
            "session",
            json!({"assignee": "agent:builder"}),
        );
        assert_eq!(run_wake_for(&chat), None);
        let created = event(
            TaskEventKind::Created,
            "global",
            json!({"assignee": "agent:builder"}),
        );
        assert_eq!(run_wake_for(&created), None);
    }

    fn card(status: &str, assignee: Option<&str>, due_at: Option<&str>) -> Task {
        Task {
            id: 5,
            parent_id: None,
            scope: "global".to_string(),
            scope_id: None,
            project: None,
            title: "a card".to_string(),
            description: None,
            status: status.to_string(),
            priority: 3,
            labels: Vec::new(),
            tool_scope: Vec::new(),
            due_at: due_at.map(str::to_string),
            deadline_at: None,
            recurrence: None,
            depends_on: Vec::new(),
            acceptance: None,
            assignee: assignee.map(str::to_string),
            sort_order: 0,
            created_at: "2026-10-01 10:00:00".to_string(),
            updated_at: "2026-10-01 10:00:00".to_string(),
            completed_at: None,
            blocked: false,
            due_announced_at: None,
            overdue_announced_at: None,
        }
    }

    #[test]
    fn only_an_open_unblocked_agent_card_whose_date_has_come_starts() {
        let now = chrono::DateTime::parse_from_rfc3339("2026-10-01T12:00:00Z")
            .expect("time")
            .with_timezone(&chrono::Utc);
        assert!(is_startable(
            &card("pending", Some("agent:builder"), None),
            now
        ));
        assert!(is_startable(
            &card(
                "pending",
                Some("agent:builder"),
                Some("2026-10-01 11:59:00")
            ),
            now
        ));
        assert!(
            !is_startable(
                &card(
                    "pending",
                    Some("agent:builder"),
                    Some("2026-10-02T09:00:00Z")
                ),
                now
            ),
            "deferred to tomorrow"
        );
        assert!(
            !is_startable(&card("in_progress", Some("agent:builder"), None), now),
            "picked up"
        );
        assert!(
            !is_startable(&card("pending", Some("human"), None), now),
            "a person's"
        );
        assert!(!is_startable(&card("pending", None, None), now), "nobody's");
        let mut blocked = card("pending", Some("agent:builder"), None);
        blocked.blocked = true;
        assert!(!is_startable(&blocked, now), "waits on another card");
        let unreadable = card("pending", Some("agent:builder"), Some("next tuesday"));
        assert!(
            is_startable(&unreadable, now),
            "an unreadable date hides nothing"
        );
    }

    /// A run the last daemon died inside is closed out at boot — end marker,
    /// member idle, card pending — even when it cannot start again here (this
    /// control plane has no agent), so nothing stays stuck as "busy".
    #[tokio::test]
    async fn an_interrupted_card_run_is_closed_out_at_boot() {
        use crate::tasks::{RUN_ENDED_ACTION, RUN_STARTED_ACTION};
        let storage = Arc::new(Storage::in_memory().await.expect("storage"));
        let control = ControlPlane::new(Arc::new(crate::session::SessionManager::new()))
            .with_storage(Arc::clone(&storage))
            .await;
        storage
            .members()
            .create(nanna_storage::NewMember {
                id: "agent:builder".to_string(),
                name: "Builder".to_string(),
                avatar: None,
                kind: nanna_storage::MemberKind::Agent,
                owner_kind: nanna_storage::MemberOwner::Workspace,
                owner_id: None,
                status: nanna_storage::MemberStatus::Busy,
                profile: json!({}),
            })
            .await
            .expect("member");
        let tasks = storage.tasks();
        let card = tasks
            .create(nanna_storage::NewTask {
                title: "half done".to_string(),
                scope: "global".to_string(),
                priority: 3,
                assignee: Some("agent:builder".to_string()),
                ..nanna_storage::NewTask::default()
            })
            .await
            .expect("card");
        let picked_up = nanna_storage::TaskPatch {
            status: Some("in_progress".to_string()),
            ..nanna_storage::TaskPatch::default()
        };
        tasks
            .update(card.id, picked_up, None)
            .await
            .expect("picked up");
        tasks
            .log_activity(card.id, Some("agent:builder"), RUN_STARTED_ACTION, None)
            .await
            .expect("marker");

        assert_eq!(
            resume_interrupted(&control, &storage).await,
            0,
            "no agent here to run it"
        );
        let card = tasks.get(card.id).await.expect("card");
        assert_eq!(card.status, "pending");
        let member = storage
            .members()
            .get("agent:builder")
            .await
            .expect("member");
        assert_eq!(member.status, nanna_storage::MemberStatus::Idle);
        let unended = tasks
            .cards_with_unended(RUN_STARTED_ACTION, RUN_ENDED_ACTION, RESUME_SCAN_ROWS)
            .await
            .expect("scan");
        assert!(unended.is_empty(), "closed out: {unended:?}");
    }
}
