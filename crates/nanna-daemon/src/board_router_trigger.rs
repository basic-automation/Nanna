//! Wakes the board router when the human puts a card on a board (P25 Stage 2).
//!
//! [`crate::board_router::route_card`] decides what happens to one card; this
//! module decides *which* cards reach it and runs the calls one at a time.
//!
//! **Fed by the store's own event sink, not by a bus subscriber** — the same
//! reasoning as [`crate::memory_write_through`]: a lagging `broadcast`
//! receiver loses events, and a lost `created` here is a card nobody ever
//! routes. [`crate::task_event_bridge::TaskEventBridge`] hands every waking
//! event to a bounded `mpsc` queue as a [`Wake`]; one worker drains it in
//! order. The sink never waits: a full queue is counted and logged.
//!
//! **The wakes so far** (P25 decisions 4, 6 and 7):
//! - [`WakeReason::Created`] — a card was put on a board;
//! - [`WakeReason::ClarificationAnswered`] — a card the router had blocked on
//!   a clarification is unblocked because the human completed it, so the
//!   router decides again with the answer in hand;
//! - [`WakeReason::RecurringReopened`] — the recurrence sweep reopened a
//!   recurring card, which goes back to the router (decision 7) rather than
//!   silently to whoever held it last time;
//! - [`WakeReason::HandedBack`] — an agent's run gave the card back
//!   unfinished (decision 7's failed verdict); past [`HAND_BACKS_MAX`] in a
//!   row the router asks the human instead of routing it again;
//! - [`WakeReason::Stalled`] — the stall sweep took back an agent's card
//!   that nothing was working ([`crate::card_run_trigger::release_stalled`]);
//!   it counts toward the same bound.
//!
//! **Only cards the board client created are routed, for now.** While the
//! chat harness lives, its model writes workspace-scoped cards of its own
//! (`tasks.add`), and routing those would fight it — a `clarify` blocks the
//! harness's own work on a human card mid-mission. A card is routed only if
//! its creator is [`BOARD_CLIENT_ACTOR`], the actor the IPC create
//! (`TaskAction::Create`) stamps: checked on the event itself for `created`,
//! and against the card's `created` activity row for every later wake. The
//! router's own splits and clarifications carry the router's id, so they never
//! wake it. A stalled card is board work by any board creator
//! ([`crate::card_run_trigger::is_board_creator`]) — the sweep only ever takes
//! back board work, and a released card nobody routes would sit unowned. The
//! remaining wake (heartbeat) is a separate roadmap item.

use crate::agent_service::AgentServiceConfig;
use crate::board_router::{RouterComplete, answered_clarifications, route_card, router_for};
use crate::llm_router::LlmRouter;
use nanna_storage::{Storage, StorageError, Task, TaskEvent, TaskEventKind, TaskRepository};
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use tracing::{debug, info, warn};

/// The `created` actor whose cards wake the router: the board client's IPC
/// create. See the module docs for why no other creator does yet.
pub const BOARD_CLIENT_ACTOR: &str = "gui";

/// The actor the recurrence sweep reopens a card as
/// (`crate::tasks::sweep_recurrences`).
pub const RECURRENCE_ACTOR: &str = "recurrence";

/// The actor the stall sweep releases a card as
/// (`crate::card_run_trigger::release_stalled`); a release by it is a
/// [`WakeReason::Stalled`].
pub const STALL_ACTOR: &str = "stall";

/// The id prefix of every agent member (`agent:<slug>`), the actor a
/// member's card run writes as.
pub const AGENT_MEMBER_PREFIX: &str = "agent:";

/// Hand-backs in a row the router may answer by routing the card again;
/// the next one asks the human instead (decision 7's "bounded retries, then a
/// clarification to the human").
///
/// Bound justification: two is one retry. A second member (or the same one
/// with what the first run posted) gets a real second attempt; a third
/// unattended attempt would spend a model run on a card two runs could not
/// finish, with nothing new to go on. Hermes Kanban's dispatcher blocks at
/// the same count (`failure_limit`, default 2).
pub const HAND_BACKS_MAX: usize = 2;

/// The activity row that resets the hand-back count: the router asked the
/// human after [`HAND_BACKS_MAX`] hand-backs, so whatever comes after the
/// answer is a fresh attempt.
pub const HAND_BACK_LIMIT_ACTION: &str = "hand_back_limit";

/// The activity row `TaskRepository::complete` writes when a card is marked
/// done. A hand-back before it is not part of the current row.
const COMPLETED_ACTION: &str = "completed";

/// How many of a card's come-back markers are read to count its hand-backs.
///
/// Bound justification: only the four marker actions are read, and the count
/// is broken by the human being asked once it reaches [`HAND_BACKS_MAX`], so
/// a real run of hand-backs is at most a few rows; the store's page ceiling
/// leaves room for a count that a crash interrupted before it asked.
const HAND_BACK_SCAN_ROWS: usize = nanna_storage::ACTIVITY_MARKER_ROWS_MAX;

/// Member-profile key holding the router's own model list, walked in order
/// with failover. Absent or empty means the agent's chat models.
pub const PROFILE_MODELS_KEY: &str = "model_priority";

/// Wakes the route queue holds before the sink starts reporting drops.
///
/// Bound justification: a queued [`Wake`] is 16 bytes, and the largest burst
/// the store can produce is one event per card of a scope (a board client
/// filling it, or a cascade unblocking all of it), which the store caps at
/// [`nanna_storage::TASKS_PER_SCOPE_MAX`] cards — so a whole scope's worth
/// fits (~16 KiB) even while the worker is busy with a slow model.
pub const ROUTE_QUEUE_MAX: usize = nanna_storage::TASKS_PER_SCOPE_MAX;

/// Why the router was woken for a card.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WakeReason {
    /// The board client created the card.
    Created,
    /// The card became unblocked. Queued for every board card; the worker
    /// routes it only when a completed clarification is among its
    /// dependencies, which takes store reads the sink must not do.
    ClarificationAnswered,
    /// The recurrence sweep reopened the card.
    RecurringReopened,
    /// An agent's run gave the card back unfinished (decision 7's failed
    /// verdict). Queued for every agent that clears its own assignment; the
    /// worker routes it only when the card's activity says it was handed back.
    HandedBack,
    /// The stall sweep took the card back from a member that was not working
    /// it (P25 decision 7's `stalled`).
    Stalled,
}

/// One queued wake: route card `task_id` because of `reason`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Wake {
    pub task_id: i64,
    pub reason: WakeReason,
}

/// The wake `event` produces, if any. Pure, so the sink can call it on the
/// task store's write path; everything that needs a read is the worker's.
#[must_use]
pub fn wake_for(event: &TaskEvent) -> Option<Wake> {
    if event.scope == "session" {
        return None;
    }
    let reason = match event.kind {
        TaskEventKind::Created if event.actor.as_deref() == Some(BOARD_CLIENT_ACTOR) => {
            WakeReason::Created
        }
        TaskEventKind::Unblocked => WakeReason::ClarificationAnswered,
        TaskEventKind::StatusChanged
            if event.actor.as_deref() == Some(RECURRENCE_ACTOR)
                && event.detail.get("reopened") == Some(&serde_json::Value::Bool(true)) =>
        {
            WakeReason::RecurringReopened
        }
        TaskEventKind::Assigned
            if event.actor.as_deref() == Some(STALL_ACTOR)
                && event.detail.get("assignee") == Some(&serde_json::Value::Null) =>
        {
            WakeReason::Stalled
        }
        TaskEventKind::Assigned
            if event
                .actor
                .as_deref()
                .is_some_and(|actor| actor.starts_with(AGENT_MEMBER_PREFIX))
                && event.detail.get("assignee") == Some(&serde_json::Value::Null) =>
        {
            WakeReason::HandedBack
        }
        _ => return None,
    };
    debug_assert!(event.task_id > 0, "store ids start at 1");
    Some(Wake {
        task_id: event.task_id,
        reason,
    })
}

/// The models the router walks: its profile's [`PROFILE_MODELS_KEY`] list
/// when that names at least one model, else `fallback` (the agent's chat
/// models). Blank entries are not models in either list.
#[must_use]
pub fn router_models(profile: &serde_json::Value, fallback: &[String]) -> Vec<String> {
    let listed: Vec<String> = profile
        .get(PROFILE_MODELS_KEY)
        .and_then(serde_json::Value::as_array)
        .map(|models| {
            models
                .iter()
                .filter_map(serde_json::Value::as_str)
                .filter(|m| !m.trim().is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    let models = if listed.is_empty() {
        crate::agent_service::named_models(fallback)
    } else {
        listed
    };
    debug_assert!(
        models.iter().all(|m| !m.trim().is_empty()),
        "a blank entry names no model"
    );
    models
}

/// Whether the router may take `card` up at all: open and not picked up.
///
/// P25 decision 7: a card being worked is never reassigned, and a card is
/// `in_progress` exactly while something has picked it up. A stalled
/// `in_progress` card is the `stalled` trigger's, not a wake's.
#[must_use]
pub fn card_is_routable(card: &Task) -> bool {
    card.status == "pending" && card.completed_at.is_none()
}

/// Drain the route queue until every sender is gone, routing each card.
///
/// One card at a time: the router is one member, and its decisions read the
/// board they change (a split creates cards the next decision may see).
pub async fn run(
    mut queue: mpsc::Receiver<Wake>,
    storage: Arc<Storage>,
    llm: Arc<LlmRouter>,
    agent_config: Arc<RwLock<AgentServiceConfig>>,
) {
    info!("Board router running: cards created on a board are routed by its router member");
    while let Some(wake) = queue.recv().await {
        let fallback = {
            let config = agent_config.read().await;
            crate::agent_service::configured_models(&config.model, &config.model_priority)
        };
        let Some(card) = card_to_route(&storage, wake).await else {
            continue;
        };
        if matches!(wake.reason, WakeReason::HandedBack | WakeReason::Stalled) {
            match hand_backs_in_a_row(&storage.tasks(), card.id).await {
                Ok((0, _)) => {
                    debug!(
                        task_id = card.id,
                        "board router: unassigned, not handed back; skipped"
                    );
                    continue;
                }
                Ok((count, latest)) if count >= HAND_BACKS_MAX => {
                    ask_the_human(&storage, &card, count, &latest).await;
                    continue;
                }
                Ok(_) => {}
                Err(e) => {
                    warn!(task_id = card.id, error = %e, "board router: hand-backs unreadable; not routed");
                    continue;
                }
            }
        }
        route_one(&storage, &llm, &fallback, &card).await;
    }
    debug!("board router queue closed; worker exiting");
}

/// Whether card `task_id` was created by the board client — the rule every
/// wake after `created` is held to while the chat harness lives (module docs).
///
/// # Errors
/// The store failure reading the card's `created` row.
pub async fn created_by_board_client(
    tasks: &TaskRepository,
    task_id: i64,
) -> Result<bool, StorageError> {
    Ok(tasks.created_by(task_id).await?.as_deref() == Some(BOARD_CLIENT_ACTOR))
}

/// The card `wake` names, if the router should decide on it now.
///
/// Every refusal is ordinary (the card closed, was picked up, came from the
/// chat harness, or was unblocked by something other than an answered
/// clarification) and is logged at debug; only a store failure warns.
pub async fn card_to_route(storage: &Storage, wake: Wake) -> Option<Task> {
    let tasks = storage.tasks();
    let task_id = wake.task_id;
    let card = match tasks.get(task_id).await {
        Ok(card) => card,
        Err(e) => {
            warn!(task_id, error = %e, "board router: card could not be read; not routed");
            return None;
        }
    };
    if !card_is_routable(&card) {
        debug!(task_id, status = %card.status, "board router: card closed or taken; skipped");
        return None;
    }
    if wake.reason == WakeReason::Created {
        return Some(card);
    }
    if wake.reason == WakeReason::RecurringReopened {
        // Only the board client's recurring cards are released at reopen
        // (`reopen_for_next_round`); any other keeps its assignee.
        return match created_by_board_client(&tasks, task_id).await {
            Ok(true) => Some(card),
            Ok(false) => {
                debug!(task_id, "board router: not a board-client card; skipped");
                None
            }
            Err(e) => {
                warn!(task_id, error = %e, "board router: creator unreadable; not routed");
                None
            }
        };
    }
    // A card that came back (a hand-back, a stall) or was answered is board
    // work whoever on the board made it: a router's split or an agent's
    // hand-on is released exactly like a card the human typed, and nothing
    // but the router takes up a pending card nobody holds. Gating these on
    // the board client stranded such cards unowned forever.
    match tasks.created_by(task_id).await {
        Ok(creator) if crate::card_run_trigger::is_board_creator(creator.as_deref()) => {}
        Ok(_) => {
            debug!(task_id, "board router: not board work; skipped");
            return None;
        }
        Err(e) => {
            warn!(task_id, error = %e, "board router: creator unreadable; not routed");
            return None;
        }
    }
    if matches!(wake.reason, WakeReason::Stalled | WakeReason::HandedBack) {
        return Some(card);
    }
    debug_assert_eq!(wake.reason, WakeReason::ClarificationAnswered);
    if let Some(assignee) = card.assignee.as_deref() {
        // A member asked this one itself (`ask_user`, a provider refusal):
        // the answer restarts that member's run, which keeps its card. Only
        // a card nobody holds — the router's own clarify — goes back to it.
        debug!(
            task_id,
            assignee, "board router: answered card still has its member; skipped"
        );
        return None;
    }
    match answered_clarifications(&tasks, &card).await {
        Ok(answers) if !answers.is_empty() => Some(card),
        Ok(_) => {
            debug!(
                task_id,
                "board router: unblocked by no clarification; skipped"
            );
            None
        }
        Err(why) => {
            warn!(task_id, %why, "board router: clarifications unreadable; not routed");
            None
        }
    }
}

/// How many times in a row card `task_id` came back unfinished, and why.
///
/// A card comes back when its member's run hands it back or the stall sweep
/// takes it back. Counted newest first until the last time the human was
/// asked about it, or the last time the card was completed: a recurring card
/// that came back once last week and was then finished, or a done card the
/// human reopened, starts a new row. Without that break a second hand-back
/// weeks later asked the human that "2 runs could not finish it" at once.
/// The reason is the newest one's.
///
/// # Errors
/// The store failure reading the card's activity.
pub async fn hand_backs_in_a_row(
    tasks: &TaskRepository,
    task_id: i64,
) -> Result<(usize, String), StorageError> {
    let activity = tasks
        .newest_activities_of(
            task_id,
            &[
                crate::tasks::HANDED_BACK_ACTION,
                crate::tasks::STALLED_ACTION,
                HAND_BACK_LIMIT_ACTION,
                COMPLETED_ACTION,
            ],
            HAND_BACK_SCAN_ROWS,
        )
        .await?;
    let mut count = 0usize;
    let mut latest: Option<String> = None;
    // Newest first, and only the markers: an edit or a run's notes between
    // two hand-backs can no longer push the earlier one out of the count.
    for row in &activity {
        if row.action == HAND_BACK_LIMIT_ACTION || row.action == COMPLETED_ACTION {
            break;
        }
        if row.action == crate::tasks::HANDED_BACK_ACTION
            || row.action == crate::tasks::STALLED_ACTION
        {
            count += 1;
            if latest.is_none() {
                latest = row
                    .detail
                    .as_ref()
                    .and_then(|d| d.get("reason"))
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string);
            }
        }
    }
    debug_assert!(count <= activity.len(), "counted from the rows read");
    Ok((count, latest.unwrap_or_default()))
}

/// The retry bound is spent: block the card on a question for the human
/// rather than spend another run (decision 7). Model-free — the router's own
/// `clarify`, applied as the router, so it reads on the thread like any other
/// decision and the human's answer wakes the router as usual.
async fn ask_the_human(storage: &Storage, card: &Task, count: usize, latest: &str) {
    debug_assert!(count >= HAND_BACKS_MAX, "only past the bound");
    let task_id = card.id;
    let Some(router_id) = router_for(card) else {
        debug!(task_id, "board router: card is not on a board; skipped");
        return;
    };
    let latest = crate::tasks::truncate_post(latest);
    let latest: String = latest.chars().take(HAND_BACK_REASON_CHARS_MAX).collect();
    let decision = nanna_storage::routing::RouterDecision::Clarify {
        question: format!(
            "Card #{task_id} came back unfinished {count} times in a row. The last \
             run said: {latest}\nWhat should change — the card, its acceptance check, or \
             who does it?"
        ),
        reason: format!(
            "{count} runs could not finish it; another would spend a run on the same guess"
        ),
    };
    let tasks = storage.tasks();
    match nanna_storage::routing::apply_decision(
        &tasks,
        &storage.members(),
        &router_id,
        task_id,
        &decision,
        false,
    )
    .await
    {
        Ok(applied) => {
            info!(task_id, router = %router_id, count, ?applied, "board router asked the human after hand-backs");
            let detail = serde_json::json!({ "count": count });
            if let Err(e) = tasks
                .log_activity(
                    task_id,
                    Some(&router_id),
                    HAND_BACK_LIMIT_ACTION,
                    Some(detail),
                )
                .await
            {
                warn!(task_id, error = %e, "board router: hand-back count not reset");
            }
        }
        Err(e) => {
            warn!(task_id, router = %router_id, error = %e, "board router: could not ask the human");
        }
    }
}

/// How much of the last run's reason the human's question quotes. Bound
/// justification: the question is a card title's source and one post; ~600
/// characters say what failed without burying the question.
const HAND_BACK_REASON_CHARS_MAX: usize = 600;

/// Route `card`, logging the outcome. Every failure is operational (no
/// model, the model or store failed) and leaves the card as it was, so it is
/// logged and the worker moves on.
async fn route_one(storage: &Storage, llm: &Arc<LlmRouter>, fallback: &[String], card: &Task) {
    debug_assert!(card_is_routable(card), "checked by card_to_route");
    let task_id = card.id;
    let Some(router_id) = router_for(card) else {
        debug!(task_id, "board router: card is not on a board; skipped");
        return;
    };
    let profile = match storage.members().get(&router_id).await {
        Ok(member) => member.profile,
        Err(e) => {
            // A board without its router member still routes on the chat
            // models; the member's absence is worth a line, not a stop.
            warn!(task_id, router = %router_id, error = %e, "board router member unreadable");
            serde_json::Value::Null
        }
    };
    let models = router_models(&profile, fallback);
    if models.is_empty() {
        warn!(
            task_id,
            "board router: {}; card left unrouted",
            crate::agent_service::NO_MODEL_CONFIGURED
        );
        return;
    }
    let walk =
        crate::dream_summarizer::complete_with_failover(Arc::clone(llm), models, "Board router");
    let complete: Box<RouterComplete> = Box::new(walk);
    // Not being worked: `card_to_route` just said so. A pickup in the window
    // since is the same race any other member's edit has.
    match route_card(storage, &complete, task_id, false).await {
        Ok(applied) => info!(task_id, router = %router_id, ?applied, "board router decided"),
        Err(why) => warn!(task_id, router = %router_id, %why, "board router: card not routed"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn event(kind: TaskEventKind, scope: &str, actor: Option<&str>) -> TaskEvent {
        TaskEvent {
            kind,
            task_id: 3,
            scope: scope.to_string(),
            scope_id: Some("ws1".to_string()),
            actor: actor.map(str::to_string),
            detail: json!({"title": "a card"}),
        }
    }

    fn reason(kind: TaskEventKind, scope: &str, actor: Option<&str>) -> Option<WakeReason> {
        wake_for(&event(kind, scope, actor)).map(|wake| {
            assert_eq!(wake.task_id, 3, "the event's card");
            wake.reason
        })
    }

    #[test]
    fn a_card_the_board_client_creates_on_a_board_wakes_the_router() {
        for scope in ["workspace", "global"] {
            assert_eq!(
                reason(TaskEventKind::Created, scope, Some(BOARD_CLIENT_ACTOR)),
                Some(WakeReason::Created)
            );
        }
    }

    #[test]
    fn an_unblocked_board_card_is_a_candidate_whoever_unblocked_it() {
        // The creator and the dependency are the worker's to check: the sink
        // cannot read the store.
        for actor in [Some(BOARD_CLIENT_ACTOR), Some("harness"), None] {
            assert_eq!(
                reason(TaskEventKind::Unblocked, "workspace", actor),
                Some(WakeReason::ClarificationAnswered)
            );
        }
        assert_eq!(reason(TaskEventKind::Unblocked, "session", None), None);
    }

    #[test]
    fn the_recurrence_sweeps_reopen_wakes_it_and_no_other_status_change_does() {
        let mut reopened = event(
            TaskEventKind::StatusChanged,
            "workspace",
            Some(RECURRENCE_ACTOR),
        );
        reopened.detail = json!({"status": "pending", "reopened": true});
        assert_eq!(
            wake_for(&reopened).map(|w| w.reason),
            Some(WakeReason::RecurringReopened)
        );
        // A replan reopening a wrong verdict is not the sweep.
        let mut by_hand = reopened.clone();
        by_hand.actor = Some(BOARD_CLIENT_ACTOR.to_string());
        assert_eq!(wake_for(&by_hand), None);
        // An ordinary status move by the sweep's actor is not a reopen.
        let mut moved = reopened.clone();
        moved.detail = json!({"status": "in_progress"});
        assert_eq!(wake_for(&moved), None);
        reopened.scope = "session".to_string();
        assert_eq!(wake_for(&reopened), None);
    }

    #[test]
    fn the_stall_sweeps_release_wakes_it_and_its_other_writes_do_not() {
        let mut released = event(TaskEventKind::Assigned, "global", Some(STALL_ACTOR));
        released.detail = json!({"assignee": null});
        assert_eq!(
            wake_for(&released).map(|w| w.reason),
            Some(WakeReason::Stalled)
        );
        // Only a release: the sweep never assigns, and nobody else's release
        // is a stall.
        let mut assigned = released.clone();
        assigned.detail = json!({"assignee": "agent:builder"});
        assert_eq!(wake_for(&assigned), None);
        let mut by_hand = released.clone();
        by_hand.actor = Some(BOARD_CLIENT_ACTOR.to_string());
        assert_eq!(wake_for(&by_hand), None);
        let status = event(TaskEventKind::StatusChanged, "global", Some(STALL_ACTOR));
        assert_eq!(
            wake_for(&status),
            None,
            "the release's status half is not a second wake"
        );
        released.scope = "session".to_string();
        assert_eq!(wake_for(&released), None);
    }

    #[test]
    fn nothing_else_wakes_it_while_the_chat_harness_lives() {
        // The chat model's own cards, the router's own splits and
        // clarifications, and a creator nobody recorded.
        for actor in [Some("harness"), Some("agent"), Some("router:ws1"), None] {
            assert_eq!(reason(TaskEventKind::Created, "workspace", actor), None);
        }
        // Session scope is chat scaffolding, never a board.
        assert_eq!(
            reason(TaskEventKind::Created, "session", Some(BOARD_CLIENT_ACTOR)),
            None
        );
        // Other kinds are other triggers, not these.
        for kind in [
            TaskEventKind::Assigned,
            TaskEventKind::Blocked,
            TaskEventKind::Posted,
            TaskEventKind::Verdict,
            TaskEventKind::StatusChanged,
            TaskEventKind::Due,
            TaskEventKind::Overdue,
        ] {
            assert_eq!(reason(kind, "workspace", Some(BOARD_CLIENT_ACTOR)), None);
        }
    }

    /// A global card that waits on one dependency, created by `creator`.
    async fn waiting_card(
        storage: &Storage,
        creator: &str,
        dependency_labels: Vec<String>,
    ) -> (i64, i64) {
        let tasks = storage.tasks();
        let dependency = tasks
            .create(nanna_storage::NewTask {
                scope: "global".to_string(),
                title: "What colour?".to_string(),
                priority: 3,
                labels: dependency_labels,
                assignee: Some(nanna_storage::HUMAN_MEMBER_ID.to_string()),
                created_by: Some("router:global".to_string()),
                ..nanna_storage::NewTask::default()
            })
            .await
            .unwrap();
        let card = tasks
            .create(nanna_storage::NewTask {
                scope: "global".to_string(),
                title: "Paint the shed".to_string(),
                priority: 3,
                depends_on: vec![dependency.id],
                created_by: Some(creator.to_string()),
                ..nanna_storage::NewTask::default()
            })
            .await
            .unwrap();
        (card.id, dependency.id)
    }

    fn answered(task_id: i64) -> Wake {
        Wake {
            task_id,
            reason: WakeReason::ClarificationAnswered,
        }
    }

    #[tokio::test]
    async fn an_answered_clarification_hands_the_card_back_to_the_router() {
        let storage = Storage::in_memory().await.unwrap();
        let (card, clarification) = waiting_card(
            &storage,
            BOARD_CLIENT_ACTOR,
            vec![nanna_storage::routing::CLARIFICATION_LABEL.to_string()],
        )
        .await;
        let tasks = storage.tasks();
        tasks
            .post(
                clarification,
                Some(BOARD_CLIENT_ACTOR),
                Some(nanna_storage::HUMAN_MEMBER_ID),
                nanna_storage::TaskNoteKind::Comment,
                "Green, the same as the fence.",
            )
            .await
            .unwrap();
        tasks
            .complete(clarification, Some(BOARD_CLIENT_ACTOR), None)
            .await
            .unwrap();

        let routed = card_to_route(&storage, answered(card)).await;
        assert_eq!(
            routed.map(|c| c.id),
            Some(card),
            "the work card is routed again"
        );
        let answers = answered_clarifications(&tasks, &tasks.get(card).await.unwrap())
            .await
            .unwrap();
        assert_eq!(answers.len(), 1);
        assert_eq!(answers[0].card_id, clarification);
        assert_eq!(
            answers[0].answer.as_deref(),
            Some("Green, the same as the fence."),
            "the human's post is the answer the router reads"
        );
    }

    #[tokio::test]
    async fn an_unblock_that_answers_nothing_does_not_wake_the_router() {
        let storage = Storage::in_memory().await.unwrap();
        // Unblocked by an ordinary dependency, not a clarification.
        let (card, dependency) = waiting_card(&storage, BOARD_CLIENT_ACTOR, Vec::new()).await;
        storage
            .tasks()
            .complete(dependency, Some(BOARD_CLIENT_ACTOR), None)
            .await
            .unwrap();
        assert!(card_to_route(&storage, answered(card)).await.is_none());

        // The chat harness's own card, even when a clarification unblocks it.
        let (harness_card, clarification) = waiting_card(
            &storage,
            "harness",
            vec![nanna_storage::routing::CLARIFICATION_LABEL.to_string()],
        )
        .await;
        storage
            .tasks()
            .complete(clarification, Some(BOARD_CLIENT_ACTOR), None)
            .await
            .unwrap();
        assert!(
            card_to_route(&storage, answered(harness_card))
                .await
                .is_none()
        );
    }

    #[tokio::test]
    async fn a_reopened_recurring_card_is_routed_only_if_the_board_client_made_it() {
        let storage = Storage::in_memory().await.unwrap();
        let tasks = storage.tasks();
        let mut routed = Vec::new();
        for creator in [BOARD_CLIENT_ACTOR, "harness"] {
            let card = tasks
                .create(nanna_storage::NewTask {
                    scope: "global".to_string(),
                    title: format!("Weekly review by {creator}"),
                    priority: 3,
                    recurrence: Some("0 9 * * 1".to_string()),
                    created_by: Some(creator.to_string()),
                    ..nanna_storage::NewTask::default()
                })
                .await
                .unwrap();
            tasks.complete(card.id, Some(creator), None).await.unwrap();
            let done = tasks.get(card.id).await.unwrap();
            assert!(crate::tasks::reopen_for_next_round(&tasks, &done, None).await);
            let wake = Wake {
                task_id: card.id,
                reason: WakeReason::RecurringReopened,
            };
            routed.push(card_to_route(&storage, wake).await.is_some());
        }
        assert_eq!(routed, vec![true, false]);
    }

    #[tokio::test]
    async fn a_card_already_picked_up_or_closed_is_not_routed() {
        let storage = Storage::in_memory().await.unwrap();
        let tasks = storage.tasks();
        let card = tasks
            .create(nanna_storage::NewTask {
                scope: "global".to_string(),
                title: "Taken".to_string(),
                priority: 3,
                created_by: Some(BOARD_CLIENT_ACTOR.to_string()),
                ..nanna_storage::NewTask::default()
            })
            .await
            .unwrap();
        let created = Wake {
            task_id: card.id,
            reason: WakeReason::Created,
        };
        assert!(card_to_route(&storage, created).await.is_some());
        tasks
            .update(
                card.id,
                nanna_storage::TaskPatch {
                    status: Some("in_progress".to_string()),
                    ..nanna_storage::TaskPatch::default()
                },
                Some("harness"),
            )
            .await
            .unwrap();
        assert!(
            card_to_route(&storage, created).await.is_none(),
            "being worked"
        );
        tasks
            .complete(card.id, Some("harness"), None)
            .await
            .unwrap();
        assert!(card_to_route(&storage, created).await.is_none(), "closed");
    }

    #[test]
    fn the_router_walks_its_own_models_before_the_chat_models() {
        let chat = vec!["chat-a".to_string(), "chat-b".to_string()];
        assert_eq!(
            router_models(
                &json!({"role": "router", "model_priority": ["small", " ", "big"]}),
                &chat
            ),
            vec!["small".to_string(), "big".to_string()],
            "the profile list, in order, blanks dropped"
        );
        assert_eq!(
            router_models(&json!({"role": "router"}), &chat),
            chat,
            "no list: the chat models"
        );
        assert_eq!(
            router_models(&json!({"model_priority": ["", "  "]}), &chat),
            chat,
            "a list of blanks names no model"
        );
        assert_eq!(router_models(&serde_json::Value::Null, &chat), chat);
        assert_eq!(
            router_models(&json!({}), &[" ".to_string()]),
            Vec::<String>::new()
        );
    }

    #[test]
    fn an_agent_clearing_its_own_assignment_is_a_hand_back() {
        let mut released = event(TaskEventKind::Assigned, "global", Some("agent:builder"));
        released.detail = json!({"assignee": null});
        assert_eq!(
            wake_for(&released).map(|w| w.reason),
            Some(WakeReason::HandedBack)
        );
        // The human unassigning a card is theirs to do; it routes nothing.
        let mut by_hand = released.clone();
        by_hand.actor = Some(BOARD_CLIENT_ACTOR.to_string());
        assert_eq!(wake_for(&by_hand), None);
        // An agent handing a card ON is an assignment, not a hand-back.
        let mut onward = released.clone();
        onward.detail = json!({"assignee": "agent:reviewer"});
        assert_eq!(wake_for(&onward), None);
        released.scope = "session".to_string();
        assert_eq!(wake_for(&released), None);
    }

    /// Decision 7's bound: hand-backs are counted newest first up to the last
    /// time the human was asked, and the count past the bound asks the human
    /// instead of routing — once, after which the count starts over.
    #[tokio::test]
    async fn hand_backs_past_the_bound_ask_the_human_and_start_the_count_over() {
        let storage = Storage::in_memory().await.unwrap();
        let tasks = storage.tasks();
        let card = tasks
            .create(nanna_storage::NewTask {
                scope: "global".to_string(),
                title: "Fix the flaky test".to_string(),
                priority: 3,
                created_by: Some(BOARD_CLIENT_ACTOR.to_string()),
                ..nanna_storage::NewTask::default()
            })
            .await
            .unwrap();
        let hand_back = |reason: &'static str| {
            let tasks = storage.tasks();
            async move {
                tasks
                    .log_activity(
                        card.id,
                        Some("agent:builder"),
                        crate::tasks::HANDED_BACK_ACTION,
                        Some(json!({"member": "agent:builder", "reason": reason})),
                    )
                    .await
                    .unwrap();
            }
        };
        assert_eq!(hand_backs_in_a_row(&tasks, card.id).await.unwrap().0, 0);
        hand_back("the check never passed").await;
        hand_back("still red: test_io timed out").await;
        let (count, latest) = hand_backs_in_a_row(&tasks, card.id).await.unwrap();
        assert_eq!(count, HAND_BACKS_MAX);
        assert_eq!(latest, "still red: test_io timed out", "the newest reason");

        let current = tasks.get(card.id).await.unwrap();
        ask_the_human(&storage, &current, count, &latest).await;
        let asked = tasks.get(card.id).await.unwrap();
        assert!(asked.blocked, "the card waits on the human: {asked:?}");
        let question = tasks.get(asked.depends_on[0]).await.unwrap();
        assert_eq!(
            question.assignee.as_deref(),
            Some(nanna_storage::HUMAN_MEMBER_ID)
        );
        let thread = tasks.notes(card.id, 10).await.unwrap();
        assert!(
            thread
                .iter()
                .any(|n| n.author_member_id.as_deref() == Some("router:global")
                    && n.content.contains("2 runs could not finish it")),
            "the router says why on the thread: {thread:?}"
        );
        assert_eq!(
            hand_backs_in_a_row(&tasks, card.id).await.unwrap().0,
            0,
            "asking the human starts the count over"
        );
    }

    /// A card the router split off, or an agent handed on, comes back to the
    /// router like the human's own: before, only `gui` cards were routed on a
    /// hand-back, so these sat pending with nobody on them forever.
    #[tokio::test]
    async fn a_handed_back_card_is_routed_whoever_on_the_board_made_it() {
        let storage = Storage::in_memory().await.unwrap();
        let tasks = storage.tasks();
        let mut routed = Vec::new();
        for creator in [
            BOARD_CLIENT_ACTOR,
            "router:global",
            "agent:builder",
            "harness",
        ] {
            let card = tasks
                .create(nanna_storage::NewTask {
                    scope: "global".to_string(),
                    title: format!("Made by {creator}"),
                    priority: 3,
                    created_by: Some(creator.to_string()),
                    ..nanna_storage::NewTask::default()
                })
                .await
                .unwrap();
            for reason in [WakeReason::HandedBack, WakeReason::Stalled] {
                let wake = Wake {
                    task_id: card.id,
                    reason,
                };
                routed.push((
                    creator,
                    reason,
                    card_to_route(&storage, wake).await.is_some(),
                ));
            }
        }
        for (creator, reason, was_routed) in routed {
            assert_eq!(
                was_routed,
                creator != "harness",
                "{creator} card on {reason:?}"
            );
        }
    }

    /// A member that asked the human itself keeps its card: the answer
    /// restarts its run, and the router must not reassign the card under it.
    #[tokio::test]
    async fn an_answer_to_a_members_own_question_is_not_the_routers() {
        let storage = Storage::in_memory().await.unwrap();
        let (card, clarification) = waiting_card(
            &storage,
            "router:global",
            vec![nanna_storage::routing::CLARIFICATION_LABEL.to_string()],
        )
        .await;
        let tasks = storage.tasks();
        tasks
            .complete(clarification, Some(BOARD_CLIENT_ACTOR), None)
            .await
            .unwrap();
        assert!(
            card_to_route(&storage, answered(card)).await.is_some(),
            "a router-made card nobody holds goes back to the router"
        );
        tasks
            .update(
                card,
                nanna_storage::TaskPatch {
                    assignee: Some(Some(nanna_storage::HUMAN_MEMBER_ID.to_string())),
                    ..nanna_storage::TaskPatch::default()
                },
                Some(BOARD_CLIENT_ACTOR),
            )
            .await
            .unwrap();
        assert!(
            card_to_route(&storage, answered(card)).await.is_none(),
            "a card its member still holds is the member's"
        );
    }

    /// The count reads markers by action: a busy card's edits between two
    /// hand-backs must not push the first out of the window. Read through a
    /// 256-row page of every action, 300 edits left the count at 1, and the
    /// router routed again instead of asking the human — without end.
    #[tokio::test]
    async fn edits_between_hand_backs_do_not_hide_one() {
        let storage = Storage::in_memory().await.unwrap();
        let tasks = storage.tasks();
        let card = tasks
            .create(nanna_storage::NewTask {
                scope: "global".to_string(),
                title: "Busy card".to_string(),
                priority: 3,
                created_by: Some(BOARD_CLIENT_ACTOR.to_string()),
                ..nanna_storage::NewTask::default()
            })
            .await
            .unwrap();
        let hand_back = || async {
            tasks
                .log_activity(
                    card.id,
                    Some("agent:builder"),
                    crate::tasks::HANDED_BACK_ACTION,
                    Some(json!({"member": "agent:builder", "reason": "ran out"})),
                )
                .await
                .unwrap();
        };
        hand_back().await;
        for _ in 0..300 {
            tasks
                .log_activity(card.id, Some(BOARD_CLIENT_ACTOR), "updated", None)
                .await
                .unwrap();
        }
        hand_back().await;
        let (count, _) = hand_backs_in_a_row(&tasks, card.id).await.unwrap();
        assert_eq!(count, HAND_BACKS_MAX, "both hand-backs are counted");
    }

    /// Finishing a card ends its row of hand-backs: a recurring card handed
    /// back once a round must not reach the bound across rounds.
    #[tokio::test]
    async fn a_completion_starts_the_hand_back_count_over() {
        let storage = Storage::in_memory().await.unwrap();
        let tasks = storage.tasks();
        let card = tasks
            .create(nanna_storage::NewTask {
                scope: "global".to_string(),
                title: "Weekly report".to_string(),
                priority: 3,
                created_by: Some(BOARD_CLIENT_ACTOR.to_string()),
                ..nanna_storage::NewTask::default()
            })
            .await
            .unwrap();
        let hand_back = || async {
            tasks
                .log_activity(
                    card.id,
                    Some("agent:builder"),
                    crate::tasks::HANDED_BACK_ACTION,
                    Some(json!({"member": "agent:builder", "reason": "ran out"})),
                )
                .await
                .unwrap();
        };
        hand_back().await;
        tasks
            .complete(card.id, Some("agent:helper"), None)
            .await
            .unwrap();
        tasks.reopen(card.id, Some(RECURRENCE_ACTOR)).await.unwrap();
        hand_back().await;
        let (count, latest) = hand_backs_in_a_row(&tasks, card.id).await.unwrap();
        assert_eq!(count, 1, "last round's hand-back is not this round's");
        assert_eq!(latest, "ran out");
    }
}
