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
//! **A card nobody is working is taken back** — P25 decision 7's `stalled`
//! trigger. Every [`STALL_SWEEP_INTERVAL`] the worker also runs
//! [`release_stalled`]: an agent's board card that is `in_progress`, that no
//! live run serves, that its member was not stopped on, and that has shown no
//! sign of life for [`STALL_AFTER`] is put back to `pending` with nobody on
//! it, which wakes the board router ([`crate::board_router_trigger`]).
//!
//! **A rate-limited card is tried again once its wait is over.** A run the
//! provider rate-limits leaves a `rate_limited {until}` row and keeps its
//! card ([`crate::tasks::RATE_LIMITED_ACTION`]); [`try_start`] refuses it
//! until then ([`cooling_down_until`]), and the same tick restarts every card
//! whose wait has passed ([`restart_cooled_down`]).
//!
//! **Only board work is started** — cards the board client, a router or an
//! agent member made ([`is_board_creator`]); never the chat harness's own
//! workspace-scoped cards while chat lives (the same reason as the router's
//! wakes, see [`crate::board_router_trigger`]). A closed card also wakes its
//! parent ([`RunWake::ChildClosed`]): a parent waiting on a sub-card handed
//! to another member resumes when that member is done.

use crate::board_router_trigger::{AGENT_MEMBER_PREFIX, BOARD_CLIENT_ACTOR};
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

/// How long an agent's `in_progress` card may go with no run serving it and
/// no sign of life (no write, no activity row, no post) before it is stalled.
///
/// Bound justification: the sweep sees run liveness directly (the run
/// manager's registry), so this is not a heartbeat timeout like Hermes
/// Kanban's 4 h `dispatch_stale_timeout` — a live run is never stalled however
/// quiet it is. What it guards is the gap between a card being picked up and
/// its run registering (sub-second), and a person moving an agent's card to
/// *In progress* by hand meaning to start it themselves. Half an hour is six
/// sweeps: long past any start race, short enough that a card does not sit
/// orphaned through a working session.
pub const STALL_AFTER: std::time::Duration = std::time::Duration::from_mins(30);

/// How often the worker looks for stalled cards.
///
/// Bound justification: the same five-minute cadence as the task time sweep (`due`/`overdue`); a stall
/// is measured in tens of minutes, so a finer clock buys nothing.
pub const STALL_SWEEP_INTERVAL: std::time::Duration = std::time::Duration::from_mins(5);

/// How many in-progress agent cards one sweep reads.
///
/// Bound justification: a member works one card at a time and a roster holds
/// at most [`nanna_storage::MEMBERS_MAX`] members, so this many cards is every
/// one a correct board can have in progress; each costs a handful of indexed
/// reads, once per sweep.
pub const STALL_SCAN_CARDS_MAX: usize = nanna_storage::MEMBERS_MAX;

/// Why the run worker was woken.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunWake {
    /// Card `id` may have become workable.
    Card(i64),
    /// This member's card run ended: start its next card, if one waits.
    MemberFree(String),
    /// Card `id` closed: its parent, waiting on it, may be workable again —
    /// a sub-card handed to another member is not the parent run's to serve,
    /// so the parent's run ends while it is open and resumes here.
    ChildClosed(i64),
}

/// The wake `event` produces, if any. Pure, so the sink can call it on the
/// task store's write path; the worker does every read.
#[must_use]
pub fn run_wake_for(event: &TaskEvent) -> Option<RunWake> {
    if event.scope == "session" {
        return None;
    }
    if event.kind == TaskEventKind::Verdict {
        return Some(RunWake::ChildClosed(event.task_id));
    }
    let wakes = match event.kind {
        // Assigned to an agent, or created already assigned to one (a
        // sub-card the router split off or a member handed on).
        TaskEventKind::Assigned | TaskEventKind::Created => event
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

/// Whether a card made by `creator` is board work a member may be started on.
///
/// That is the board client's, a router's (its splits) or an agent member's
/// (the sub-cards a card run hands on). The chat harness's own cards are not —
/// the rule the router's wakes keep while chat lives.
#[must_use]
pub fn is_board_creator(creator: Option<&str>) -> bool {
    creator.is_some_and(|creator| {
        creator == BOARD_CLIENT_ACTOR
            || creator.starts_with(nanna_storage::ROUTER_MEMBER_PREFIX)
            || creator.starts_with(AGENT_MEMBER_PREFIX)
    })
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
    let mut stall_clock = tokio::time::interval(STALL_SWEEP_INTERVAL);
    stall_clock.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    // The first tick fires at once; nothing can have stalled while this
    // worker was starting, and boot's interrupted runs were just resumed.
    stall_clock.tick().await;
    loop {
        let wake = tokio::select! {
            wake = queue.recv() => wake,
            _ = stall_clock.tick() => {
                let now = chrono::Utc::now();
                let released = release_stalled(&control, &storage, now).await;
                if released > 0 {
                    info!(released, "stalled board cards went back to their router");
                }
                let restarted = restart_cooled_down(&control, &storage).await;
                if restarted > 0 {
                    info!(restarted, "rate-limited board cards started again");
                }
                continue;
            }
        };
        let Some(wake) = wake else {
            break;
        };
        match wake {
            RunWake::Card(card_id) => {
                try_start(&control, &storage, card_id).await;
            }
            RunWake::MemberFree(member_id) => start_next_for(&control, &storage, &member_id).await,
            RunWake::ChildClosed(child_id) => {
                if let Ok(Some(parent_id)) =
                    storage.tasks().get(child_id).await.map(|c| c.parent_id)
                {
                    try_start(&control, &storage, parent_id).await;
                }
            }
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

/// A card's newest run marker, read from its activity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunMarker {
    /// No run marker among the rows read: no member run ever started here.
    None,
    /// A run started and has not ended — a run the daemon lost.
    Started,
    /// The last run ended because the human stopped it: the card is paused
    /// with its member ("Stop = stop"), never stalled.
    Paused,
    /// The last run ended any other way.
    Ended,
}

/// The newest run marker among `activity` (oldest first, as
/// [`nanna_storage::TaskRepository::activity`] returns it).
#[must_use]
pub fn newest_run_marker(activity: &[nanna_storage::TaskActivityEntry]) -> RunMarker {
    use crate::tasks::{RUN_ENDED_ACTION, RUN_STARTED_ACTION};
    let Some(row) = activity
        .iter()
        .rev()
        .find(|row| row.action == RUN_STARTED_ACTION || row.action == RUN_ENDED_ACTION)
    else {
        return RunMarker::None;
    };
    if row.action == RUN_STARTED_ACTION {
        return RunMarker::Started;
    }
    let stop = row
        .detail
        .as_ref()
        .and_then(|detail| detail.get("stop"))
        .and_then(serde_json::Value::as_str);
    // `finish_card_run` records the stop as `format!("{stop:?}")`.
    if stop == Some("Cancelled") {
        RunMarker::Paused
    } else {
        RunMarker::Ended
    }
}

/// The newest readable time among `times` (stored timestamps); `None` when
/// none parses.
#[must_use]
pub fn last_touch(times: &[String]) -> Option<chrono::DateTime<chrono::Utc>> {
    let newest = times
        .iter()
        .filter_map(|time| crate::tasks::parse_db_time(time))
        .max();
    debug_assert!(
        newest.is_none() || !times.is_empty(),
        "a time comes from the input"
    );
    newest
}

/// Minutes `card` has gone untouched at `now`, when that is past
/// [`STALL_AFTER`]. A card whose times cannot be read is never stalled:
/// nothing is taken from a member on a value nobody can read.
#[must_use]
pub fn stalled_minutes(
    last_touch: Option<chrono::DateTime<chrono::Utc>>,
    now: chrono::DateTime<chrono::Utc>,
) -> Option<i64> {
    let idle = now.signed_duration_since(last_touch?);
    let threshold = chrono::Duration::from_std(STALL_AFTER).ok()?;
    debug_assert!(threshold > chrono::Duration::zero(), "a positive threshold");
    (idle >= threshold).then(|| idle.num_minutes())
}

/// Take back every agent's board card nobody is working (P25 decision 7's
/// `stalled` trigger); returns how many were released.
///
/// A card is stalled when it is `in_progress`, assigned to an agent, board
/// work ([`is_board_creator`]), not paused by the human ([`RunMarker::Paused`]),
/// served by no live run (its own, an ancestor's, or one over its board), and
/// untouched for [`STALL_AFTER`]. Each is handed to its board's router:
/// a `stalled` activity row (which the router counts toward its retry bound,
/// as Hermes Kanban counts stale reclaims toward its failure breaker), a post
/// saying why, and a release to `pending` with nobody on it — the release,
/// written as [`crate::board_router_trigger::STALL_ACTOR`], is the router's
/// wake. Without a run manager nothing can be judged live, so nothing is
/// released.
pub async fn release_stalled(
    control: &ControlPlane,
    storage: &Storage,
    now: chrono::DateTime<chrono::Utc>,
) -> usize {
    let Some(runs) = control.task_runs() else {
        return 0;
    };
    let tasks = storage.tasks();
    let candidates = match tasks
        .in_progress_board_cards(AGENT_MEMBER_PREFIX, STALL_SCAN_CARDS_MAX)
        .await
    {
        Ok(candidates) => candidates,
        Err(e) => {
            warn!(error = %e, "stall sweep: in-progress cards unreadable");
            return 0;
        }
    };
    let mut released = 0usize;
    for card in candidates {
        match stall_check(storage, runs, &card, now).await {
            Ok(Some((minutes, marker))) => {
                if stall_card(storage, runs, &card, minutes, marker).await {
                    released += 1;
                }
            }
            Ok(None) => {}
            Err(e) => warn!(card_id = card.id, error = %e, "stall sweep: card not judged"),
        }
    }
    debug_assert!(
        released <= STALL_SCAN_CARDS_MAX,
        "one release per card read"
    );
    released
}

/// Whether `card` is stalled at `now`: its idle minutes and newest run
/// marker when it is, `None` when it is not. Cheapest checks first.
async fn stall_check(
    storage: &Storage,
    runs: &crate::tasks::TaskRunManager,
    card: &Task,
    now: chrono::DateTime<chrono::Utc>,
) -> Result<Option<(i64, RunMarker)>, nanna_storage::StorageError> {
    debug_assert_eq!(card.status, "in_progress", "the scan reads picked-up cards");
    let tasks = storage.tasks();
    let Some(minutes) = stalled_minutes(last_touch(&tasks.touch_times(card.id).await?), now) else {
        return Ok(None);
    };
    if !is_board_creator(tasks.created_by(card.id).await?.as_deref()) {
        return Ok(None);
    }
    // By action, not within a window of the newest rows: 64 edits or board
    // reorders after a Stop used to push its marker out of view, and the
    // paused card was then taken back as stalled.
    let marker_row = tasks
        .newest_activity_of(
            card.id,
            &[
                crate::tasks::RUN_STARTED_ACTION,
                crate::tasks::RUN_ENDED_ACTION,
            ],
        )
        .await?;
    let marker = newest_run_marker(marker_row.as_slice());
    if marker == RunMarker::Paused {
        return Ok(None);
    }
    let mut lineage = vec![card.id];
    let mut parent = card.parent_id;
    while let Some(parent_id) = parent {
        if lineage.len() > nanna_storage::TASK_DEPTH_MAX || lineage.contains(&parent_id) {
            break;
        }
        lineage.push(parent_id);
        parent = tasks.get(parent_id).await?.parent_id;
    }
    if runs
        .serves_card(&card.scope, card.scope_id.as_deref(), &lineage)
        .await
    {
        return Ok(None);
    }
    Ok(Some((minutes, marker)))
}

/// Hand stalled `card` back to its router; `true` when it was released.
///
/// Ordered like a member's own hand-back: the `stalled` row first (the
/// router's worker counts it when the release wakes it), then the post, then
/// the release that is the wake.
async fn stall_card(
    storage: &Storage,
    runs: &crate::tasks::TaskRunManager,
    card: &Task,
    minutes: i64,
    marker: RunMarker,
) -> bool {
    use crate::board_router_trigger::STALL_ACTOR;
    use crate::tasks::{RUN_ENDED_ACTION, STALLED_ACTION};
    debug_assert!(minutes >= 0, "idle time is not negative");
    let tasks = storage.tasks();
    let card_id = card.id;
    let Some(member_id) = card.assignee.clone() else {
        return false;
    };
    if marker == RunMarker::Started {
        // A run that started and never ended, while this daemon lives: close
        // it out so the card is not also "interrupted" at the next boot.
        let detail = serde_json::json!({ "stop": "stalled" });
        if let Err(e) = tasks
            .log_activity(card_id, Some("daemon"), RUN_ENDED_ACTION, Some(detail))
            .await
        {
            warn!(card_id, error = %e, "stall sweep: lost run not closed out; card kept");
            return false;
        }
    }
    let detail = serde_json::json!({
        "member": member_id,
        "idle_minutes": minutes,
        "reason": format!("nothing worked it for {minutes} minutes (stalled)"),
    });
    if let Err(e) = tasks
        .log_activity(card_id, Some(STALL_ACTOR), STALLED_ACTION, Some(detail))
        .await
    {
        warn!(card_id, error = %e, "stall sweep: stall not recorded; card kept");
        return false;
    }
    if !runs.member_is_working(&member_id).await
        && let Err(e) = storage
            .members()
            .set_status(&member_id, nanna_storage::MemberStatus::Idle)
            .await
    {
        debug!(card_id, member = %member_id, error = %e, "stall sweep: member status not reset");
    }
    if let Some(router_id) = crate::board_router::router_for(card) {
        let post = format!(
            "Stalled — this card was in progress with {member_id} but nothing has worked it \
             for {minutes} minutes. Taking it back from them for the router to decide again."
        );
        if let Err(e) = tasks
            .post(
                card_id,
                Some(&router_id),
                Some(&router_id),
                nanna_storage::TaskNoteKind::Comment,
                &post,
            )
            .await
        {
            debug!(card_id, error = %e, "stall sweep: stall not posted");
        }
    }
    let release = nanna_storage::TaskPatch {
        status: Some("pending".to_string()),
        assignee: Some(None),
        ..nanna_storage::TaskPatch::default()
    };
    match tasks.update(card_id, release, Some(STALL_ACTOR)).await {
        Ok(_) => {
            info!(card_id, member = %member_id, minutes, "stalled card released to its router");
            true
        }
        Err(e) => {
            warn!(card_id, error = %e, "stall sweep: stalled card not released");
            false
        }
    }
}

/// When a card the provider rate-limited may be tried again, if not yet.
///
/// That is the `until` of its newest [`crate::tasks::RATE_LIMITED_ACTION`] row, if that row is newer
/// than the card's last run start and `until` is still ahead of `now`.
/// `activity` is oldest first. An unreadable `until` holds nothing back.
#[must_use]
pub fn cooling_down_until(
    activity: &[nanna_storage::TaskActivityEntry],
    now: chrono::DateTime<chrono::Utc>,
) -> Option<chrono::DateTime<chrono::Utc>> {
    use crate::tasks::{RATE_LIMITED_ACTION, RUN_STARTED_ACTION};
    let row = activity
        .iter()
        .rev()
        .find(|row| row.action == RATE_LIMITED_ACTION || row.action == RUN_STARTED_ACTION)?;
    if row.action != RATE_LIMITED_ACTION {
        return None;
    }
    let until = row
        .detail
        .as_ref()
        .and_then(|detail| detail.get("until"))
        .and_then(serde_json::Value::as_str)
        .and_then(crate::tasks::parse_db_time)?;
    (until > now).then_some(until)
}

/// Start every rate-limited card whose wait is over; returns how many started.
///
/// The candidates are the cards whose newest `rate_limited` row is newer
/// than their newest `run_started` (rate-limited and not tried since), among
/// the newest [`RESUME_SCAN_ROWS`] such rows — each member works one card,
/// so the waiting ones are few and recent. [`try_start`] makes every other
/// check, including the wait itself.
pub async fn restart_cooled_down(control: &ControlPlane, storage: &Storage) -> usize {
    use crate::tasks::{RATE_LIMITED_ACTION, RUN_STARTED_ACTION};
    let waiting = match storage
        .tasks()
        .cards_with_unended(RATE_LIMITED_ACTION, RUN_STARTED_ACTION, RESUME_SCAN_ROWS)
        .await
    {
        Ok(waiting) => waiting,
        Err(e) => {
            warn!(error = %e, "card runs: rate-limited cards unreadable");
            return 0;
        }
    };
    let mut started = 0usize;
    for card_id in waiting {
        if try_start(control, storage, card_id).await {
            started += 1;
        }
    }
    debug_assert!(started <= RESUME_SCAN_ROWS, "one start per card read");
    started
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

/// The actor a paused card is put back to `pending` as when it has been
/// given to another member (see [`take_over_paused`]).
pub const HANDOVER_ACTOR: &str = "handover";

/// Whether a card's newest run marker `marker` is a pause (Stop) by a member
/// other than `assignee`, an agent it has been given to since.
///
/// The pause post says the card "stays with me until it is restarted or
/// reassigned"; a pause therefore holds only for the member it stopped. A
/// marker that does not name who held the card (written before that was
/// recorded) is taken as the current assignee's: never un-pause on a guess.
#[must_use]
pub fn paused_for_another_member(
    marker: Option<&nanna_storage::TaskActivityEntry>,
    assignee: Option<&str>,
) -> bool {
    let Some(row) = marker else {
        return false;
    };
    let Some(paused_member) = row.assignee.as_deref() else {
        return false;
    };
    newest_run_marker(std::slice::from_ref(row)) == RunMarker::Paused
        && assignee.is_some_and(|a| a.starts_with(AGENT_MEMBER_PREFIX) && a != paused_member)
}

/// `card`, put back to `pending` when it is a paused card given to another
/// member, so that member's run can start it; else `card` unchanged.
///
/// Stop leaves a card `in_progress` with its member, and a run starts only
/// on a `pending` card. Reassigning a stopped card used to leave it there for
/// good: not startable, and never stalled either, since the sweep spares a
/// paused card.
async fn take_over_paused(
    tasks: &nanna_storage::TaskRepository,
    card: Task,
) -> Result<Task, nanna_storage::StorageError> {
    if card.status != "in_progress" {
        return Ok(card);
    }
    let marker = tasks
        .newest_activity_of(
            card.id,
            &[
                crate::tasks::RUN_STARTED_ACTION,
                crate::tasks::RUN_ENDED_ACTION,
            ],
        )
        .await?;
    if !paused_for_another_member(marker.as_ref(), card.assignee.as_deref()) {
        return Ok(card);
    }
    let pending = nanna_storage::TaskPatch {
        status: Some("pending".to_string()),
        ..nanna_storage::TaskPatch::default()
    };
    let taken = tasks.update(card.id, pending, Some(HANDOVER_ACTOR)).await?;
    debug_assert_eq!(taken.status, "pending", "the patch set it");
    info!(
        card_id = card.id,
        assignee = card.assignee.as_deref().unwrap_or_default(),
        "card runs: a stopped card was given to another member; it is theirs to start"
    );
    Ok(taken)
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
    let card = match take_over_paused(&tasks, card).await {
        Ok(card) => card,
        Err(e) => {
            warn!(card_id, error = %e, "card runs: paused card not handed over; not started");
            return false;
        }
    };
    if !is_startable(&card, chrono::Utc::now()) {
        debug!(card_id, status = %card.status, "card runs: not workable now");
        return false;
    }
    let newest_wait_or_start = tasks
        .newest_activity_of(
            card_id,
            &[
                crate::tasks::RATE_LIMITED_ACTION,
                crate::tasks::RUN_STARTED_ACTION,
            ],
        )
        .await;
    match newest_wait_or_start {
        Ok(row) => {
            if let Some(until) = cooling_down_until(row.as_slice(), chrono::Utc::now()) {
                debug!(card_id, %until, "card runs: rate-limited; waits");
                return false;
            }
        }
        Err(e) => {
            warn!(card_id, error = %e, "card runs: activity unreadable; not started");
            return false;
        }
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
    match tasks.created_by(card_id).await {
        Ok(creator) if is_board_creator(creator.as_deref()) => {}
        Ok(_) => {
            debug!(card_id, "card runs: not a board card; not started");
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
        assert_eq!(
            run_wake_for(&created),
            Some(RunWake::Card(5)),
            "made already assigned"
        );
        let closed = event(TaskEventKind::Verdict, "global", json!({}));
        assert_eq!(run_wake_for(&closed), Some(RunWake::ChildClosed(5)));
        let posted = event(TaskEventKind::Posted, "global", json!({}));
        assert_eq!(run_wake_for(&posted), None);
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
    fn board_work_is_the_board_clients_a_routers_or_an_agents() {
        for creator in ["gui", "router:global", "router:ws-1", "agent:builder"] {
            assert!(is_board_creator(Some(creator)), "{creator}");
        }
        // The chat harness's own cards, and a creator nobody recorded.
        for creator in [Some("harness"), Some("agent"), Some("recurrence"), None] {
            assert!(!is_board_creator(creator), "{creator:?}");
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

    #[test]
    fn a_card_is_stalled_only_past_the_threshold_and_only_on_a_readable_time() {
        let now = chrono::Utc::now();
        let minutes = |m: i64| Some(now - chrono::Duration::minutes(m));
        assert_eq!(stalled_minutes(minutes(31), now), Some(31));
        assert_eq!(
            stalled_minutes(minutes(30), now),
            Some(30),
            "the threshold itself"
        );
        assert_eq!(stalled_minutes(minutes(29), now), None);
        assert_eq!(
            stalled_minutes(None, now),
            None,
            "unreadable times take nothing"
        );
        let times = vec![
            "2026-10-01 10:00:00".to_string(),
            "not a time".to_string(),
            "2026-10-01T11:30:00Z".to_string(),
        ];
        assert_eq!(
            last_touch(&times),
            crate::tasks::parse_db_time("2026-10-01T11:30:00Z"),
            "the newest readable one"
        );
        assert_eq!(last_touch(&[]), None);
    }

    #[test]
    fn the_newest_run_marker_tells_a_pause_from_a_lost_run() {
        use crate::tasks::{RUN_ENDED_ACTION, RUN_STARTED_ACTION};
        let row =
            |action: &str, detail: Option<serde_json::Value>| nanna_storage::TaskActivityEntry {
                id: 1,
                task_id: 5,
                actor: None,
                action: action.to_string(),
                detail,
                created_at: "2026-10-01 10:00:00".to_string(),
                assignee: None,
            };
        assert_eq!(newest_run_marker(&[row("created", None)]), RunMarker::None);
        let started = row(RUN_STARTED_ACTION, None);
        let paused = row(RUN_ENDED_ACTION, Some(json!({"stop": "Cancelled"})));
        let ended = row(RUN_ENDED_ACTION, Some(json!({"stop": "AllTasksDone"})));
        assert_eq!(
            newest_run_marker(std::slice::from_ref(&started)),
            RunMarker::Started
        );
        assert_eq!(
            newest_run_marker(&[started.clone(), paused.clone()]),
            RunMarker::Paused
        );
        assert_eq!(
            newest_run_marker(&[paused, started.clone(), row("acceptance_checked", None)]),
            RunMarker::Started,
            "the newest marker wins, whatever follows it"
        );
        assert_eq!(newest_run_marker(&[started, ended]), RunMarker::Ended);
    }

    /// A global card assigned to `agent:builder`, made by `creator`, picked up.
    async fn picked_up_card(storage: &Storage, creator: &str, title: &str) -> i64 {
        let tasks = storage.tasks();
        let card = tasks
            .create(nanna_storage::NewTask {
                title: title.to_string(),
                scope: "global".to_string(),
                priority: 3,
                assignee: Some("agent:builder".to_string()),
                created_by: Some(creator.to_string()),
                ..nanna_storage::NewTask::default()
            })
            .await
            .expect("card");
        let status = nanna_storage::TaskPatch {
            status: Some("in_progress".to_string()),
            ..nanna_storage::TaskPatch::default()
        };
        tasks
            .update(card.id, status, None)
            .await
            .expect("picked up");
        card.id
    }

    #[test]
    fn a_rate_limit_holds_a_card_back_until_its_wait_or_its_next_run() {
        use crate::tasks::{RATE_LIMITED_ACTION, RUN_STARTED_ACTION};
        let now = chrono::DateTime::parse_from_rfc3339("2026-10-03T12:00:00Z")
            .expect("time")
            .with_timezone(&chrono::Utc);
        let row = |action: &str, until: Option<&str>| nanna_storage::TaskActivityEntry {
            id: 1,
            task_id: 5,
            actor: None,
            action: action.to_string(),
            detail: until.map(|u| json!({ "until": u })),
            created_at: "2026-10-03 11:59:00".to_string(),
            assignee: None,
        };
        let limited = row(RATE_LIMITED_ACTION, Some("2026-10-03T12:05:00Z"));
        assert!(cooling_down_until(std::slice::from_ref(&limited), now).is_some());
        let expired = row(RATE_LIMITED_ACTION, Some("2026-10-03T11:55:00Z"));
        assert_eq!(
            cooling_down_until(&[expired], now),
            None,
            "the wait is over"
        );
        let retried = [limited, row(RUN_STARTED_ACTION, None)];
        assert_eq!(cooling_down_until(&retried, now), None, "tried again since");
        let unreadable = row(RATE_LIMITED_ACTION, Some("soon"));
        assert_eq!(
            cooling_down_until(&[unreadable], now),
            None,
            "holds nothing back"
        );
        assert_eq!(cooling_down_until(&[], now), None);
    }

    /// The `stalled` trigger end to end in the store: an agent's board card
    /// left in progress with nothing working it goes back to its router; a
    /// paused card, a fresh one and the chat harness's own are left alone.
    #[tokio::test]
    async fn a_card_nobody_is_working_is_taken_back_for_the_router() {
        use crate::tasks::{RUN_ENDED_ACTION, RUN_STARTED_ACTION, STALLED_ACTION};
        let storage = Arc::new(Storage::in_memory().await.expect("storage"));
        let control = ControlPlane::new(Arc::new(crate::session::SessionManager::new()))
            .with_task_runs(Arc::new(crate::tasks::TaskRunManager::new()))
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
        let lost = picked_up_card(&storage, "gui", "lost run").await;
        tasks
            .log_activity(lost, Some("agent:builder"), RUN_STARTED_ACTION, None)
            .await
            .expect("marker");
        let paused = picked_up_card(&storage, "gui", "paused").await;
        tasks
            .log_activity(
                paused,
                Some("agent:builder"),
                RUN_ENDED_ACTION,
                Some(json!({"stop": "Cancelled"})),
            )
            .await
            .expect("marker");
        // Edits after the Stop (renames, board reorders) each log a row; 70
        // of them used to push the pause marker out of the 64 rows read, and
        // the paused card was taken back as stalled.
        for edit in 0..70 {
            tasks
                .log_activity(
                    paused,
                    Some("gui"),
                    "updated",
                    Some(json!({ "edit": edit })),
                )
                .await
                .expect("edit");
        }
        let chat = picked_up_card(&storage, "harness", "chat's own").await;

        assert_eq!(
            release_stalled(&control, &storage, chrono::Utc::now()).await,
            0,
            "nothing is stalled the moment it is picked up"
        );
        let later = chrono::Utc::now() + chrono::Duration::minutes(31);
        assert_eq!(release_stalled(&control, &storage, later).await, 1);

        let card = tasks.get(lost).await.expect("card");
        assert_eq!(card.status, "pending");
        assert_eq!(card.assignee, None, "back to the router");
        let activity = tasks.activity(lost, 32).await.expect("activity");
        assert_eq!(
            newest_run_marker(&activity),
            RunMarker::Ended,
            "the lost run is closed out"
        );
        assert!(activity.iter().any(|row| row.action == STALLED_ACTION));
        let posts = tasks.notes(lost, 8).await.expect("posts");
        assert!(
            posts
                .iter()
                .any(|p| p.author.as_deref() == Some("router:global")
                    && p.content.starts_with("Stalled")),
            "the router says why: {posts:?}"
        );
        let (count, reason) = crate::board_router_trigger::hand_backs_in_a_row(&tasks, lost)
            .await
            .expect("count");
        assert_eq!(count, 1, "a stall counts toward the retry bound");
        assert!(reason.contains("stalled"), "{reason}");
        let member = storage
            .members()
            .get("agent:builder")
            .await
            .expect("member");
        assert_eq!(member.status, nanna_storage::MemberStatus::Idle);

        for kept in [paused, chat] {
            let card = tasks.get(kept).await.expect("card");
            assert_eq!(card.status, "in_progress", "#{kept} kept");
            assert_eq!(card.assignee.as_deref(), Some("agent:builder"));
        }
        assert_eq!(
            release_stalled(&control, &storage, later).await,
            0,
            "a released card is not released twice"
        );
    }

    #[test]
    fn a_pause_holds_only_for_the_member_it_stopped() {
        use crate::tasks::{RUN_ENDED_ACTION, RUN_STARTED_ACTION};
        let row = |action: &str, stop: Option<&str>, held_by: Option<&str>| {
            nanna_storage::TaskActivityEntry {
                id: 1,
                task_id: 5,
                actor: None,
                action: action.to_string(),
                detail: stop.map(|stop| json!({ "stop": stop })),
                created_at: "2026-10-05 10:00:00".to_string(),
                assignee: held_by.map(str::to_string),
            }
        };
        let paused_by_a = row(RUN_ENDED_ACTION, Some("Cancelled"), Some("agent:a"));
        assert!(paused_for_another_member(
            Some(&paused_by_a),
            Some("agent:b")
        ));
        assert!(
            !paused_for_another_member(Some(&paused_by_a), Some("agent:a")),
            "still its own"
        );
        assert!(
            !paused_for_another_member(Some(&paused_by_a), Some("human")),
            "a person starts it"
        );
        assert!(!paused_for_another_member(Some(&paused_by_a), None));

        let unnamed = row(RUN_ENDED_ACTION, Some("Cancelled"), None);
        assert!(
            !paused_for_another_member(Some(&unnamed), Some("agent:b")),
            "never on a guess"
        );
        let ended = row(RUN_ENDED_ACTION, Some("AllTasksDone"), Some("agent:a"));
        assert!(!paused_for_another_member(Some(&ended), Some("agent:b")));
        let started = row(RUN_STARTED_ACTION, None, Some("agent:a"));
        assert!(!paused_for_another_member(Some(&started), Some("agent:b")));
        assert!(!paused_for_another_member(None, Some("agent:b")));
    }

    /// Stop, then give the card to someone else: it goes back to `pending`
    /// for them, where it used to stay `in_progress` with no run, forever.
    #[tokio::test]
    async fn a_stopped_card_given_to_another_member_is_theirs_to_start() {
        use crate::tasks::RUN_ENDED_ACTION;
        let storage = Storage::in_memory().await.expect("storage");
        for id in ["agent:builder", "agent:helper"] {
            storage
                .members()
                .create(nanna_storage::NewMember {
                    id: id.to_string(),
                    name: id.trim_start_matches("agent:").to_string(),
                    avatar: None,
                    kind: nanna_storage::MemberKind::Agent,
                    owner_kind: nanna_storage::MemberOwner::Workspace,
                    owner_id: None,
                    status: nanna_storage::MemberStatus::Idle,
                    profile: json!({}),
                })
                .await
                .expect("member");
        }
        let tasks = storage.tasks();
        let mut cards = Vec::new();
        for title in ["handed on", "kept"] {
            let card = picked_up_card(&storage, "gui", title).await;
            tasks
                .log_activity(
                    card,
                    Some("agent:builder"),
                    RUN_ENDED_ACTION,
                    Some(json!({"stop": "Cancelled"})),
                )
                .await
                .expect("pause");
            cards.push(card);
        }
        let reassign = nanna_storage::TaskPatch {
            assignee: Some(Some("agent:helper".to_string())),
            ..nanna_storage::TaskPatch::default()
        };
        tasks
            .update(cards[0], reassign, Some("gui"))
            .await
            .expect("reassigned");

        let handed_on = take_over_paused(&tasks, tasks.get(cards[0]).await.expect("card"))
            .await
            .expect("take over");
        assert_eq!(handed_on.status, "pending");
        assert_eq!(handed_on.assignee.as_deref(), Some("agent:helper"));
        assert!(
            is_startable(&handed_on, chrono::Utc::now()),
            "the new member's to run"
        );

        let kept = take_over_paused(&tasks, tasks.get(cards[1]).await.expect("card"))
            .await
            .expect("read");
        assert_eq!(
            kept.status, "in_progress",
            "Stop still means stop for its own member"
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
