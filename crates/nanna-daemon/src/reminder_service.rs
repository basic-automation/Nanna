//! The `schedule.{add,list,cancel}` services and reminder delivery.
//!
//! The bundled `remind`, `list_reminders` and `cancel_reminder` skills declared
//! these and nothing registered them, so all three were withheld at every boot.
//! They were deliberately left unwired on 2026-09-15 because the obvious wiring
//! — a `Delayed` task whose payload runs as an agent prompt — delivers nowhere:
//! the scheduler's generic arm runs the prompt in a throwaway `scheduled-{id}`
//! session and the result lands in the log. The `remind` skill promises the
//! model the message "will be injected into the conversation", so a reminder
//! that fired into a log file would have been a broken promise nothing reports.
//!
//! **Delivery is a message, not a model turn.** A reminder is text the user (or
//! the agent on their behalf) already wrote; firing it needs no model. The
//! executor appends it to the originating session as an assistant message —
//! persisted, so it is in the history of the next turn and survives a restart —
//! and broadcasts [`Event::SessionMessageAdded`] so a client viewing the session
//! shows it without a reload. That also makes delivery verifiable on a host
//! with no model at all.
//!
//! **The session is the route.** `ScheduledTask::target_session` was declared,
//! persisted and read by nobody; this is its first writer (`schedule.add`) and
//! its first reader ([`deliver_reminder`]). The skill supplies it from
//! `Nanna.sessionId()`, the run-scoped binding, so a reminder set in one chat
//! cannot be filed under another that happened to be running at the same time.
//!
//! **Absolute time.** Reminders are [`TaskType::At`] one-shots, persisted as a
//! wall-clock instant: a restart neither re-arms nor postpones them, and one
//! that came due while the daemon was down is delivered on the first tick after
//! boot with the lateness stated.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use chrono::{DateTime, TimeDelta, Utc};
use nanna_core::{ScheduledTask, Scheduler, TaskType};
use nanna_scripting::ServiceFn;
use serde_json::{Value, json};
use tokio::sync::{RwLock, broadcast};
use tracing::info;

use crate::protocol::Event;
use crate::session::SessionManager;

/// Scheduler task name every reminder carries.
///
/// The executor routes on it, and
/// `schedule.list` / `schedule.cancel` touch nothing else — the model must not
/// be able to list or cancel `memory_consolidation` through a reminder tool.
pub const REMINDER_TASK_NAME: &str = "reminder";

/// Most reminders that may be pending at once.
///
/// Derived from the listing, which is the one place they all enter a model's
/// context together: a `list_reminders` line is ~100 characters (~25 tokens),
/// so 100 reminders is ~2.5k tokens — a tool result a small local model can
/// still read whole. Beyond that the list stops being usable before storage
/// or the 30s tick (which clones every task) would notice.
pub const REMINDERS_PENDING_MAX: usize = 100;

/// Longest reminder message accepted, in bytes.
///
/// A delivered reminder is re-read as history on every later turn of its
/// session, so it is priced like conversation, not like storage: 4 KiB is ~1k
/// tokens, already far past a note-to-self. Longer text belongs in memory or a
/// file, and the refusal says so.
pub const REMINDER_TEXT_BYTES_MAX: usize = 4096;

/// The scheduler, reached late.
///
/// Script services are built (and skills loaded against them) before the
/// scheduler exists — its executor captures the agent those services feed. The
/// slot is filled once, right after the scheduler is built; a call that finds
/// it empty is answered with an error rather than a panic.
pub type SchedulerSlot = Arc<OnceLock<Arc<RwLock<Scheduler>>>>;

/// Build `schedule.add`, `schedule.list` and `schedule.cancel`.
pub fn build_reminder_services(
    scheduler: SchedulerSlot,
    sessions: Arc<SessionManager>,
) -> HashMap<String, ServiceFn> {
    let mut services: HashMap<String, ServiceFn> = HashMap::new();

    let add_scheduler = Arc::clone(&scheduler);
    services.insert(
        "schedule.add".to_string(),
        Arc::new(move |params: Value| {
            let scheduler = Arc::clone(&add_scheduler);
            let sessions = Arc::clone(&sessions);
            Box::pin(async move {
                let scheduler = scheduler_from(&scheduler)?;
                add_reminder(&scheduler, &sessions, &params, Utc::now()).await
            })
        }),
    );

    let list_scheduler = Arc::clone(&scheduler);
    services.insert(
        "schedule.list".to_string(),
        Arc::new(move |_params: Value| {
            let scheduler = Arc::clone(&list_scheduler);
            Box::pin(async move {
                let scheduler = scheduler_from(&scheduler)?;
                let scheduler = scheduler.read().await;
                Ok(list_reminders(&scheduler.list_tasks().await, Utc::now()))
            })
        }),
    );

    services.insert(
        "schedule.cancel".to_string(),
        Arc::new(move |params: Value| {
            let scheduler = Arc::clone(&scheduler);
            Box::pin(async move {
                let scheduler = scheduler_from(&scheduler)?;
                cancel_reminder(&scheduler, &params).await
            })
        }),
    );

    debug_assert_eq!(services.len(), 3, "exactly the three schedule.* services");
    services
}

fn scheduler_from(slot: &SchedulerSlot) -> Result<Arc<RwLock<Scheduler>>, String> {
    slot.get().cloned().ok_or_else(|| {
        "The scheduler has not started yet, so no reminder was set. Try again in a moment."
            .to_string()
    })
}

/// A validated `schedule.add` request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReminderRequest {
    pub message: String,
    pub session_id: String,
    pub fire_at: DateTime<Utc>,
}

/// Validate `schedule.add` params against `now`. Pure.
///
/// # Errors
///
/// A sentence for the model naming what was wrong and that nothing was set:
/// an empty or oversized message, a delay that is not a positive whole number
/// of seconds or overflows the calendar, or no session to deliver into.
pub fn parse_reminder_request(
    params: &Value,
    now: DateTime<Utc>,
) -> Result<ReminderRequest, String> {
    let message = params
        .get("message")
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or_default();
    if message.is_empty() {
        return Err("No reminder was set: `message` is empty. Say what to be reminded of.".into());
    }
    if message.len() > REMINDER_TEXT_BYTES_MAX {
        return Err(format!(
            "No reminder was set: the message is {} bytes and a reminder holds at most {REMINDER_TEXT_BYTES_MAX}. \
             Keep the reminder short and put the detail in memory or a file it can point to.",
            message.len()
        ));
    }
    // Lenient on purpose: a script engine hands every JS number over as a
    // float (`90` arrives as `90.0`), so a strict `as_i64` refused every
    // reminder the skill ever sent — found by driving the real daemon, not by
    // a unit test. Whole floats and digit strings are read; `1.5` is not.
    let delay_secs = crate::tasks::opt_i64(params, "delay_secs")
        .map_err(|e| format!("No reminder was set: {e}"))?
        .unwrap_or(0);
    if delay_secs <= 0 {
        return Err(
            "No reminder was set: `delay_secs` must be a whole number of seconds greater than zero."
                .into(),
        );
    }
    let fire_at = TimeDelta::try_seconds(delay_secs)
        .and_then(|delay| now.checked_add_signed(delay))
        .ok_or_else(|| {
            format!(
                "No reminder was set: a delay of {delay_secs}s is past the end of the calendar."
            )
        })?;
    let session_id = params
        .get("session_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or_default();
    if session_id.is_empty() {
        return Err(
            "No reminder was set: this call is not running inside a conversation, so there is \
             nowhere to deliver it."
                .into(),
        );
    }
    debug_assert!(fire_at > now, "a positive delay lands in the future");
    Ok(ReminderRequest {
        message: message.to_string(),
        session_id: session_id.to_string(),
        fire_at,
    })
}

async fn add_reminder(
    scheduler: &RwLock<Scheduler>,
    sessions: &SessionManager,
    params: &Value,
    now: DateTime<Utc>,
) -> Result<Value, String> {
    let request = parse_reminder_request(params, now)?;
    if !sessions.exists(&request.session_id).await {
        return Err(format!(
            "No reminder was set: conversation `{}` does not exist, so there is nowhere to deliver it.",
            request.session_id
        ));
    }
    // The WRITE guard, so the count below and the add after it are one step:
    // under a read guard two concurrent adds both saw room and both added,
    // and the bound held only while nobody raced it.
    let scheduler = scheduler.write().await;
    if !scheduler.runtime().enabled() {
        return Err(
            "No reminder was set: the scheduler is switched off in Settings, so it would never \
             fire. Turn the scheduler on and try again."
                .into(),
        );
    }
    let pending = pending_reminders(&scheduler.list_tasks().await).count();
    if pending >= REMINDERS_PENDING_MAX {
        return Err(format!(
            "No reminder was set: {pending} reminders are already pending, the most allowed. \
             Cancel one you no longer need first."
        ));
    }

    let mut task = nanna_core::at_task(REMINDER_TASK_NAME, request.fire_at, &request.message);
    task.target_session = Some(request.session_id.clone());
    let id = task.id.clone();
    scheduler.add_task(task).await;
    info!(
        "Reminder {id} set for session {} at {}",
        request.session_id, request.fire_at
    );
    Ok(json!({
        "id": id,
        "fire_at": request.fire_at.to_rfc3339(),
        "delay_secs": (request.fire_at - now).num_seconds(),
        "session_id": request.session_id,
        "resolution_secs": scheduler.check_interval().as_secs(),
    }))
}

fn pending_reminders(tasks: &[ScheduledTask]) -> impl Iterator<Item = &ScheduledTask> {
    tasks
        .iter()
        .filter(|task| task.name == REMINDER_TASK_NAME && task.enabled)
}

/// The pending reminders, soonest first. Pure.
#[must_use]
pub fn list_reminders(tasks: &[ScheduledTask], now: DateTime<Utc>) -> Value {
    let mut pending: Vec<(&ScheduledTask, DateTime<Utc>)> = pending_reminders(tasks)
        .filter_map(|task| task.task_type.next_run().map(|fire_at| (task, fire_at)))
        .collect();
    pending.sort_by_key(|(task, fire_at)| (*fire_at, task.id.clone()));
    debug_assert!(pending.len() <= tasks.len());
    Value::Array(
        pending
            .into_iter()
            .map(|(task, fire_at)| {
                json!({
                    "id": task.id,
                    "message": task.payload,
                    "fire_at": fire_at.to_rfc3339(),
                    "remaining_secs": (fire_at - now).num_seconds().max(0),
                    "session_id": task.target_session,
                })
            })
            .collect(),
    )
}

async fn cancel_reminder(scheduler: &RwLock<Scheduler>, params: &Value) -> Result<Value, String> {
    let id = params
        .get("id")
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or_default();
    if id.is_empty() {
        return Err("Nothing was cancelled: `id` is empty. `list_reminders` shows the ids.".into());
    }
    let guard = scheduler.read().await;
    let is_reminder = guard
        .get_task(id)
        .await
        .is_some_and(|task| task.name == REMINDER_TASK_NAME);
    // Only reminders. An id belonging to any other scheduled job is reported
    // exactly like an unknown one, so this tool cannot remove dreaming or the
    // recurrence sweep.
    let cancelled = is_reminder && guard.remove_task(id).await;
    drop(guard);
    Ok(json!({ "id": id, "cancelled": cancelled }))
}

/// Deliver a due reminder into its session. Returns the delivered text.
///
/// Fails — and the scheduler keeps the reminder, disabled, where the failure
/// can be seen — when the reminder names no session or its session is gone.
///
/// # Errors
///
/// The reminder names no session, or its session no longer exists. Nothing is
/// appended and nothing is announced.
///
/// # Panics
///
/// When handed a task that is not a reminder — a routing bug in the executor.
pub async fn deliver_reminder(
    sessions: &SessionManager,
    event_tx: &broadcast::Sender<Event>,
    task: &ScheduledTask,
    now: DateTime<Utc>,
    tick: std::time::Duration,
) -> Result<String, String> {
    assert_eq!(
        task.name, REMINDER_TASK_NAME,
        "only reminders are delivered here"
    );
    let session_id = task
        .target_session
        .as_deref()
        .ok_or_else(|| format!("reminder {} names no session to deliver into", task.id))?;
    let fire_at = match task.task_type {
        TaskType::At { fire_at } => Some(fire_at),
        _ => None,
    };
    let content = reminder_text(&task.payload, fire_at, now, tick);
    sessions
        .post_assistant_message(event_tx, session_id, content.clone())
        .await
        .ok_or_else(|| {
            format!(
                "reminder {} could not be delivered: session {session_id} no longer exists",
                task.id
            )
        })?;
    info!("Reminder {} delivered into session {session_id}", task.id);
    Ok(content)
}

/// The text a reminder is delivered as. Pure.
///
/// Lateness is stated only when it exceeds two scheduler ticks: up to one tick
/// late is the scheduler's normal resolution, and anything past two means the
/// daemon was not running when it came due — which the user should be told
/// rather than left to wonder why "in 5 minutes" arrived an hour later.
#[must_use]
pub fn reminder_text(
    message: &str,
    fire_at: Option<DateTime<Utc>>,
    now: DateTime<Utc>,
    tick: std::time::Duration,
) -> String {
    let text = format!("⏰ Reminder: {message}");
    let Some(fire_at) = fire_at else {
        return text;
    };
    let late_secs = (now - fire_at).num_seconds();
    let tick_secs = i64::try_from(tick.as_secs()).unwrap_or(i64::MAX);
    if late_secs <= tick_secs.saturating_mul(2) {
        return text;
    }
    debug_assert!(late_secs > 0, "only a late reminder reaches here");
    let late_mins = late_secs.unsigned_abs().div_ceil(60);
    format!(
        "{text}\n\n(This was due at {} UTC and is {late_mins} min late — Nanna was not running when it came due.)",
        fire_at.format("%Y-%m-%d %H:%M")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    const TICK: Duration = Duration::from_secs(30);

    fn at(secs: i64) -> DateTime<Utc> {
        DateTime::from_timestamp(1_800_000_000 + secs, 0).expect("in range")
    }

    fn params(message: &str, delay_secs: i64, session_id: &str) -> Value {
        json!({ "message": message, "delay_secs": delay_secs, "session_id": session_id })
    }

    #[test]
    fn a_valid_request_fires_after_its_delay_in_its_session() {
        let request = parse_reminder_request(&params("  stretch  ", 90, "chat-1"), at(0)).unwrap();
        assert_eq!(request.message, "stretch");
        assert_eq!(request.session_id, "chat-1");
        assert_eq!(request.fire_at, at(90));
    }

    #[test]
    fn requests_that_cannot_be_delivered_are_refused_with_a_reason() {
        for (bad, needle) in [
            (params("", 60, "s"), "`message` is empty"),
            (params("x", 0, "s"), "greater than zero"),
            (params("x", -5, "s"), "greater than zero"),
            (params("x", 60, ""), "not running inside a conversation"),
            (
                json!({ "message": "x", "delay_secs": 60 }),
                "not running inside a conversation",
            ),
            (params("x", i64::MAX, "s"), "past the end of the calendar"),
        ] {
            let error = parse_reminder_request(&bad, at(0)).unwrap_err();
            assert!(error.contains(needle), "{bad} -> {error}");
            assert!(error.starts_with("No reminder was set"), "{error}");
        }
    }

    #[test]
    fn a_delay_arrives_in_every_dialect_a_script_engine_speaks() {
        // Boa passes JS numbers as JSON floats; a model sometimes quotes them.
        for delay in [json!(90), json!(90.0), json!("90"), json!(" 90 ")] {
            let request = json!({ "message": "x", "delay_secs": delay, "session_id": "s" });
            let parsed = parse_reminder_request(&request, at(0));
            assert_eq!(parsed.map(|r| r.fire_at), Ok(at(90)), "{delay}");
        }
        for delay in [json!(1.5), json!("soon"), json!(true)] {
            let request = json!({ "message": "x", "delay_secs": delay, "session_id": "s" });
            let error = parse_reminder_request(&request, at(0)).unwrap_err();
            assert!(
                error.starts_with("No reminder was set: delay_secs must be an integer"),
                "{error}"
            );
        }
    }

    #[test]
    fn the_message_bound_is_inclusive() {
        let exactly = "a".repeat(REMINDER_TEXT_BYTES_MAX);
        assert!(parse_reminder_request(&params(&exactly, 1, "s"), at(0)).is_ok());
        let over = "a".repeat(REMINDER_TEXT_BYTES_MAX + 1);
        let error = parse_reminder_request(&params(&over, 1, "s"), at(0)).unwrap_err();
        assert!(error.contains("at most 4096"), "{error}");
    }

    fn reminder(message: &str, fire_at: DateTime<Utc>) -> ScheduledTask {
        let mut task = nanna_core::at_task(REMINDER_TASK_NAME, fire_at, message);
        task.target_session = Some("chat-1".into());
        task
    }

    #[test]
    fn the_listing_holds_only_pending_reminders_soonest_first() {
        let later = reminder("later", at(600));
        let sooner = reminder("sooner", at(60));
        let mut delivered_failed = reminder("failed", at(10));
        delivered_failed.enabled = false;
        let dreaming = nanna_core::consolidation_task(None);

        let listed = list_reminders(&[later, dreaming, delivered_failed, sooner], at(0));
        let listed = listed.as_array().unwrap();
        assert_eq!(listed.len(), 2, "{listed:?}");
        assert_eq!(listed[0]["message"], "sooner");
        assert_eq!(listed[0]["remaining_secs"], 60);
        assert_eq!(listed[1]["message"], "later");
        assert_eq!(listed[1]["session_id"], "chat-1");
    }

    #[test]
    fn an_overdue_reminder_lists_zero_remaining_not_a_negative_time() {
        let listed = list_reminders(&[reminder("overdue", at(0))], at(500));
        assert_eq!(listed[0]["remaining_secs"], 0);
    }

    #[test]
    fn lateness_is_stated_only_past_two_ticks() {
        let on_time = reminder_text("stretch", Some(at(0)), at(29), TICK);
        assert_eq!(on_time, "⏰ Reminder: stretch");
        let within_resolution = reminder_text("stretch", Some(at(0)), at(60), TICK);
        assert_eq!(within_resolution, "⏰ Reminder: stretch");
        let after_downtime = reminder_text("stretch", Some(at(0)), at(3600), TICK);
        assert!(
            after_downtime.starts_with("⏰ Reminder: stretch\n\n"),
            "{after_downtime}"
        );
        assert!(after_downtime.contains("60 min late"), "{after_downtime}");
    }

    #[tokio::test]
    async fn cancel_refuses_every_job_that_is_not_a_reminder() {
        let scheduler = Arc::new(RwLock::new(Scheduler::new(
            nanna_core::SchedulerConfig::default(),
        )));
        let dreaming = nanna_core::consolidation_task(None);
        let dreaming_id = dreaming.id.clone();
        let pending = reminder("stretch", at(60));
        let pending_id = pending.id.clone();
        let guard = scheduler.read().await;
        guard.add_task(dreaming).await;
        guard.add_task(pending).await;
        drop(guard);

        let refused = cancel_reminder(&scheduler, &json!({ "id": dreaming_id }))
            .await
            .unwrap();
        assert_eq!(refused["cancelled"], false);
        assert!(
            scheduler
                .read()
                .await
                .get_task(&dreaming_id)
                .await
                .is_some()
        );

        let done = cancel_reminder(&scheduler, &json!({ "id": pending_id }))
            .await
            .unwrap();
        assert_eq!(done["cancelled"], true);
        assert!(scheduler.read().await.get_task(&pending_id).await.is_none());

        let again = cancel_reminder(&scheduler, &json!({ "id": pending_id }))
            .await
            .unwrap();
        assert_eq!(again["cancelled"], false);
        assert!(cancel_reminder(&scheduler, &json!({})).await.is_err());
    }

    #[tokio::test]
    async fn add_checks_the_session_the_switch_and_the_bound() {
        let scheduler = Arc::new(RwLock::new(Scheduler::new(
            nanna_core::SchedulerConfig::default(),
        )));
        let sessions = SessionManager::new();
        let session = sessions.create(None).await;

        let missing = add_reminder(&scheduler, &sessions, &params("x", 60, "nope"), at(0)).await;
        assert!(missing.unwrap_err().contains("does not exist"));

        let set = add_reminder(&scheduler, &sessions, &params("x", 60, &session.id), at(0))
            .await
            .unwrap();
        assert_eq!(set["fire_at"], at(60).to_rfc3339());
        assert_eq!(set["resolution_secs"], 30);
        let stored = scheduler
            .read()
            .await
            .get_task(set["id"].as_str().unwrap())
            .await
            .unwrap();
        assert_eq!(stored.target_session.as_deref(), Some(session.id.as_str()));
        assert!(matches!(stored.task_type, TaskType::At { fire_at } if fire_at == at(60)));

        for _ in 1..REMINDERS_PENDING_MAX {
            add_reminder(&scheduler, &sessions, &params("x", 60, &session.id), at(0))
                .await
                .unwrap();
        }
        let full = add_reminder(&scheduler, &sessions, &params("x", 60, &session.id), at(0)).await;
        assert!(
            full.unwrap_err()
                .contains("100 reminders are already pending")
        );

        let switched_off = Arc::new(RwLock::new(Scheduler::new(nanna_core::SchedulerConfig {
            enabled: false,
            ..Default::default()
        })));
        let off = add_reminder(
            &switched_off,
            &sessions,
            &params("x", 60, &session.id),
            at(0),
        )
        .await;
        assert!(off.unwrap_err().contains("switched off"));
    }

    #[tokio::test]
    async fn delivery_appends_to_the_session_and_announces_it() {
        let sessions = SessionManager::new();
        let session = sessions.create(None).await;
        let (event_tx, mut events) = broadcast::channel(8);
        let mut task = reminder("stretch", at(0));
        task.target_session = Some(session.id.clone());

        let delivered = deliver_reminder(&sessions, &event_tx, &task, at(10), TICK)
            .await
            .unwrap();
        assert_eq!(delivered, "⏰ Reminder: stretch");

        let stored = sessions.get(&session.id).await.unwrap();
        let last = stored.messages.last().unwrap();
        assert_eq!(last.content, delivered);
        assert_eq!(last.role, crate::session::MessageRole::Assistant);

        match events.try_recv().unwrap() {
            Event::SessionMessageAdded {
                session_id,
                message_id,
                role,
                content,
            } => {
                assert_eq!(session_id, session.id);
                assert_eq!(message_id, last.id);
                assert_eq!(role, "assistant");
                assert_eq!(content, delivered);
            }
            other => panic!("unexpected event {other:?}"),
        }
    }

    #[tokio::test]
    async fn delivery_into_a_deleted_session_fails_loudly() {
        let sessions = SessionManager::new();
        let (event_tx, mut events) = broadcast::channel(8);
        let task = reminder("stretch", at(0));

        let error = deliver_reminder(&sessions, &event_tx, &task, at(10), TICK)
            .await
            .unwrap_err();
        assert!(error.contains("no longer exists"), "{error}");
        assert!(
            events.try_recv().is_err(),
            "nothing delivered, nothing announced"
        );

        let mut unrouted = reminder("stretch", at(0));
        unrouted.target_session = None;
        let error = deliver_reminder(&sessions, &event_tx, &unrouted, at(10), TICK)
            .await
            .unwrap_err();
        assert!(error.contains("names no session"), "{error}");
    }
}
