//! Wakes the board router when the human puts a card on a board (P25 Stage 2).
//!
//! [`crate::board_router::route_card`] decides what happens to one card; this
//! module decides *which* cards reach it and runs the calls one at a time.
//!
//! **Fed by the store's own event sink, not by a bus subscriber** — the same
//! reasoning as [`crate::memory_write_through`]: a lagging `broadcast`
//! receiver loses events, and a lost `created` here is a card nobody ever
//! routes. [`crate::task_event_bridge::TaskEventBridge`] hands every waking
//! event's card id to a bounded `mpsc` queue; one worker drains it in order.
//! The sink never waits: a full queue is counted and logged.
//!
//! **Only cards the board client created wake the router, for now.** While
//! the chat harness lives, its model writes workspace-scoped cards of its own
//! (`tasks.add`), and routing those would fight it — a `clarify` blocks the
//! harness's own work on a human card mid-mission. So the wake is restricted
//! to `created` events whose actor is [`BOARD_CLIENT_ACTOR`], the actor the
//! IPC create (`TaskAction::Create`) stamps. The router's own splits and
//! clarifications carry the router's id, so they never wake it either. The
//! remaining wakes (failed verdict, clarification done, recurring reopen,
//! stalled, heartbeat) are separate roadmap items.

use crate::agent_service::AgentServiceConfig;
use crate::board_router::{RouterComplete, route_card, router_for};
use crate::llm_router::LlmRouter;
use nanna_storage::{Storage, Task, TaskEvent, TaskEventKind};
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use tracing::{debug, info, warn};

/// The `created` actor whose cards wake the router: the board client's IPC
/// create. See the module docs for why no other creator does yet.
pub const BOARD_CLIENT_ACTOR: &str = "gui";

/// Member-profile key holding the router's own model list, walked in order
/// with failover. Absent or empty means the agent's chat models.
pub const PROFILE_MODELS_KEY: &str = "model_priority";

/// Card ids the route queue holds before the sink starts reporting drops.
///
/// Bound justification: a queued entry is one `i64`, and the largest burst a
/// board client can produce is filling one scope, which the store caps at
/// [`nanna_storage::TASKS_PER_SCOPE_MAX`] cards — so a whole scope's worth of
/// creates fits (~8 KiB) even while the worker is busy with a slow model.
pub const ROUTE_QUEUE_MAX: usize = nanna_storage::TASKS_PER_SCOPE_MAX;

/// Whether `event` should wake the router: a card the board client created on
/// a board (workspace or global scope). Pure, so the sink can call it on the
/// task store's write path.
#[must_use]
pub fn wakes_router(event: &TaskEvent) -> bool {
    event.kind == TaskEventKind::Created
        && event.scope != "session"
        && event.actor.as_deref() == Some(BOARD_CLIENT_ACTOR)
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

/// Whether the router may decide anything about `card` beyond parking it.
///
/// P25 decision 7: a card being worked is never reassigned. Only the card's
/// own status says so here — a card is `in_progress` exactly while something
/// has picked it up — which is conservative: a stalled `in_progress` card
/// can only be parked until the `stalled` trigger exists.
#[must_use]
pub fn card_is_being_worked(card: &Task) -> bool {
    card.status == "in_progress"
}

/// Drain the route queue until every sender is gone, routing each card.
///
/// One card at a time: the router is one member, and its decisions read the
/// board they change (a split creates cards the next decision may see).
pub async fn run(
    mut queue: mpsc::Receiver<i64>,
    storage: Arc<Storage>,
    llm: Arc<LlmRouter>,
    agent_config: Arc<RwLock<AgentServiceConfig>>,
) {
    info!("Board router running: cards created on a board are routed by its router member");
    while let Some(task_id) = queue.recv().await {
        debug_assert!(task_id > 0, "store ids start at 1");
        let fallback = {
            let config = agent_config.read().await;
            crate::agent_service::configured_models(&config.model, &config.model_priority)
        };
        route_one(&storage, &llm, &fallback, task_id).await;
    }
    debug!("board router queue closed; worker exiting");
}

/// Route card `task_id`, logging the outcome. Every failure is operational
/// (the card is gone, no model, the model or store failed) and leaves the
/// card as it was, so it is logged and the worker moves on.
async fn route_one(storage: &Storage, llm: &Arc<LlmRouter>, fallback: &[String], task_id: i64) {
    let card = match storage.tasks().get(task_id).await {
        Ok(card) => card,
        Err(e) => {
            warn!(task_id, error = %e, "board router: card could not be read; not routed");
            return;
        }
    };
    if card.completed_at.is_some() || card.status == "done" || card.status == "cancelled" {
        debug!(task_id, status = %card.status, "board router: card already closed; skipped");
        return;
    }
    let Some(router_id) = router_for(&card) else {
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
    match route_card(storage, &complete, task_id, card_is_being_worked(&card)).await {
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

    #[test]
    fn a_card_the_board_client_creates_on_a_board_wakes_the_router() {
        assert!(wakes_router(&event(
            TaskEventKind::Created,
            "workspace",
            Some(BOARD_CLIENT_ACTOR)
        )));
        assert!(wakes_router(&event(
            TaskEventKind::Created,
            "global",
            Some(BOARD_CLIENT_ACTOR)
        )));
    }

    #[test]
    fn nothing_else_wakes_it_while_the_chat_harness_lives() {
        // The chat model's own cards: it would route the harness's plan.
        assert!(!wakes_router(&event(
            TaskEventKind::Created,
            "workspace",
            Some("harness")
        )));
        assert!(!wakes_router(&event(
            TaskEventKind::Created,
            "workspace",
            Some("agent")
        )));
        // The router's own splits and clarifications.
        assert!(!wakes_router(&event(
            TaskEventKind::Created,
            "workspace",
            Some("router:ws1")
        )));
        // A creator nobody recorded is not the board client.
        assert!(!wakes_router(&event(
            TaskEventKind::Created,
            "workspace",
            None
        )));
        // Session scope is chat scaffolding, never a board.
        assert!(!wakes_router(&event(
            TaskEventKind::Created,
            "session",
            Some(BOARD_CLIENT_ACTOR)
        )));
        // Other kinds are other triggers, not this one.
        for kind in [
            TaskEventKind::Assigned,
            TaskEventKind::Posted,
            TaskEventKind::Verdict,
            TaskEventKind::StatusChanged,
        ] {
            assert!(!wakes_router(&event(
                kind,
                "workspace",
                Some(BOARD_CLIENT_ACTOR)
            )));
        }
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
}
