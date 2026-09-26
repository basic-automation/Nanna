//! Long-horizon chat (P19): every chat turn is a harness run.
//!
//! Before this, chat and the long-horizon harness were two execution paths.
//! Chat called [`crate::agent_service::AgentService::chat_with_options`] — one
//! growing context, no continuation machinery, and a follow-up message
//! serialized behind a `tokio::Mutex` until the turn finished. The harness had
//! the continuation machinery (re-anchored O(1) steps, acceptance verdicts,
//! progress-or-replan) but nothing to execute, because nothing turned a
//! request into a plan.
//!
//! [`nanna_agent::planner`] closes that gap, so this module makes the harness
//! THE chat path — there is no other:
//!
//! 1. **Plan.** The turn's message becomes a plan. Conversation and questions
//!    yield one task with no acceptance check, which the harness runs as
//!    exactly one step — the same cost as the old path.
//! 2. **Run — fire and forget.** The Send is `ACKed` immediately and the run is
//!    driven by a spawned task; a run can last hours, and holding the IPC
//!    request open that long was observed to outlive the GUI client's grace
//!    period ("Received response for unknown request"). Every step streams
//!    into the transcript through the existing `MessageDelta` / `ToolStart` /
//!    `ToolEnd` events.
//! 3. **Recoverable.** The run registers with
//!    [`crate::agent_service::AgentService::register_external_run`], so
//!    navigation away and back rebuilds the live view (`get_run_state`), the
//!    Stop button works (`cancel` wakes the shared token, aborting the
//!    in-flight LLM stream and tool calls, and skipping planning/seeding),
//!    and the full run timeline is persisted with the final message instead
//!    of evaporating with the stream.
//! 4. **Interject.** A message sent while a run is live does not queue behind
//!    it: it is admitted at the next step boundary and jumps the plan, so the
//!    user is answered at the first available opportunity rather than in
//!    however many hours the run takes.
//!
//! **It must feel like chat.** The harness is machinery, not UI: a
//! single-task turn renders as a plain reply — no step banner, no run-stats
//! line — and the `TASK COMPLETE` claim marker the harness verdicts on is
//! stripped before anything is persisted. Run mechanics surface only when
//! there is a genuine multi-item run to attribute work to.

use super::{ControlPlane, Value, json};
use crate::agent_service::{NO_MODEL_CONFIGURED, named_models};
use crate::session::{MessageRole, SessionMessage, TimelineItem};
use crate::tasks::{
    AgentPlanner, AgentStepRunner, ChatSink, PendingMessages, SessionInterjector, TursoTaskSource,
    closed_task_ids, demote_in_progress, seed_continuation, seed_plan,
};
use nanna_agent::AgentConfig;
use nanna_agent::harness::{Interjector, LongHorizonConfig};
use nanna_storage::Storage;
use nanna_tools::ToolRegistry;
use nanna_agent::planner::{PLAN_DESCRIPTION_MAX_BYTES, PLAN_GOAL_MAX_BYTES};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Continuation rounds one mission turn may run before it is forcibly ended.
///
/// Bound justification: the dry-round counter is the real exit; this only
/// brakes a planner that keeps inventing genuinely new titles forever, and 60
/// rounds of real new work is a session the user should be steering anyway.
const CONTINUATION_ROUNDS_MAX: usize = 60;

/// Consecutive rounds that make no progress before the mission is done.
///
/// Bound justification: one dry round can be a planner hiccup; two in a row
/// means re-planning from the current store state finds nothing left — the
/// goal is as done as this planner can make it. A round is dry when it plans
/// no new work OR when it makes none ([`round_made_progress`]).
const CONTINUATION_DRY_ROUNDS: usize = 2;

/// Rounds a run may end in an ERROR stop and still be retried.
///
/// `AllTasksDone` used to be the only continuable stop, which meant one
/// transient Turso error in `TursoTaskSource::next`, or three consecutive
/// 502s tripping the runner's error cap, silently ended an overnight run. The
/// mission's termination criterion should be "no more work can be planned",
/// not "the last step happened to exit by the happy path".
const CONTINUATION_ERROR_ROUNDS: usize = 3;

/// Did a continuation round actually PROGRESS the mission?
///
/// The duplicate-title filter (`tasks::same_title`) is a narrow safety net,
/// not the terminator: it only refuses a title it can prove is the same
/// string modulo case, punctuation and plurals, because a duplicate filter
/// that guesses can DELETE work nobody will ever do. Rephrasing walks
/// straight past it — observed live 2026-08-03 (goal "fix conflicts and merge
/// all open prs"), the planner re-proposed finished work as "check the
/// current PR state", then "enumerate the currently open PRs", then "check
/// which PRs are still open", three different verbs for one job.
///
/// So convergence is decided by what a round DID, not by what it was called.
/// Two signals, both title-blind:
///
/// - The **acceptance pre-check**
///   (`LongHorizonConfig::precheck_acceptance_items`) is the decisive one, and
///   the environment proves it: an item THIS ROUND SEEDED whose done-condition
///   already passes is completed with no step run and counted in
///   `items_already_satisfied`. That is what would have ended the live run,
///   whose condition (`gh pr list --state open … | grep -qx 0`) passed on
///   every round. Those completions are therefore NOT progress — they are
///   proof the goal was already met. It is scoped to the round's seeded ids
///   precisely so it can never swallow an interjected user message.
/// - The **structural** one: a round that made no side-effectful tool call (no
///   write, no edit, no shell — the same work-evidence set the
///   completion-claim rung uses) and closed nothing left the world and the
///   plan exactly as it found them.
///
/// The structural signal has a documented limit: `exec` is dual-use, so a
/// read-only `gh pr list` through the shell counts as side-effectful and the
/// round reads as progress. That is the case the pre-check above covers, and
/// it is why the pre-check is the decisive rung rather than a second opinion.
///
/// Genuine multi-round missions are untouched: writing a file or closing an
/// item the run actually did resets the counter exactly as before.
const fn round_made_progress(round: &nanna_agent::harness::LongHorizonReport) -> bool {
    round.side_effect_tool_calls > 0 || round.items_completed > round.items_already_satisfied
}

/// WHY a chat turn's mission ended — one closed enum, set at every exit of
/// the continuation loop and carried to all three places an ending is
/// reported: the terminal log line, the liveness [`crate::liveness::StopMark`]
/// (so `session.liveness` can tell a converged mission from an abandoned
/// one), and the user-visible mission-end sentence.
///
/// Before this, every continuation exit was silent. The `chat harness run
/// finished` line fired BEFORE the loop, so a mission that ran twenty rounds
/// logged the first round's numbers and nothing else; and the planner-
/// starvation give-up left `report.stop` reading `AllTasksDone`, so the
/// failure notice found nothing to announce and the user saw a run simply
/// stop. Recovering why took a database query; it should take a grep, and
/// the user should not have to grep at all.
#[derive(Debug, Clone, PartialEq, Eq)]
enum MissionEnd {
    /// The continuation loop never ran — an ordinary conversational turn.
    SingleRun,
    /// The run's own stop is terminal and was asked for (a budget stop, a
    /// cancel the harness itself recorded). Carries the stop's wire kind.
    DeliberateStop(String),
    /// Stop was pressed.
    Cancelled,
    /// Re-planning found nothing new to do, twice running
    /// ([`CONTINUATION_DRY_ROUNDS`]).
    DryRoundsExhausted,
    /// The mission hit [`CONTINUATION_ROUNDS_MAX`] rounds of genuinely new work.
    RoundsMaxExhausted,
    /// Runs kept failing past [`CONTINUATION_ERROR_ROUNDS`].
    ErrorRoundsExhausted,
    /// The planner kept falling back and seeding nothing past
    /// [`CONTINUATION_ERROR_ROUNDS`] — planning starved, which is NOT the
    /// same ending as "the goal is done" even though `report.stop` still
    /// reads `AllTasksDone`.
    PlannerStarvation,
    /// A transient provider outage ended the turn, and the work is PARKED
    /// rather than given up on. Carries the provider-facing reason.
    ParkedTransient(String),
}

impl MissionEnd {
    /// Snake-case wire spelling for the log field and the liveness ledger.
    /// `deliberate_stop:<kind>` keeps the underlying stop greppable without a
    /// second field.
    fn cause(&self) -> String {
        match self {
            Self::SingleRun => "single_run".to_string(),
            Self::DeliberateStop(kind) => format!("deliberate_stop:{kind}"),
            Self::Cancelled => "cancelled".to_string(),
            Self::DryRoundsExhausted => "dry_rounds_exhausted".to_string(),
            Self::RoundsMaxExhausted => "rounds_max_exhausted".to_string(),
            Self::ErrorRoundsExhausted => "error_rounds_exhausted".to_string(),
            Self::PlannerStarvation => "planner_starvation".to_string(),
            Self::ParkedTransient(_) => "parked_transient".to_string(),
        }
    }

    /// Did the mission end because it RAN OUT of something, rather than
    /// because it finished? These endings are always announced: the goal is
    /// not verified complete and the user is the only one who can decide what
    /// to do next.
    const fn gave_up(&self) -> bool {
        matches!(
            self,
            Self::ErrorRoundsExhausted
                | Self::PlannerStarvation
                | Self::RoundsMaxExhausted
                | Self::ParkedTransient(_)
                | Self::DeliberateStop(_)
        )
    }
}

/// Per-session interjection intake, shared between the chat handler (which
/// pushes) and the live run's [`SessionInterjector`] (which drains).
///
/// Keyed by session id. An entry outlives its run so a message that lands in
/// the gap between two turns is not dropped.
/// A turn that ended because the PROVIDER was down, not because the work was
/// finished or hopeless.
///
/// A transient outage is the one give-up that is a lie: the mission did not
/// fail, the daemon simply could not reach the model. Parking records what
/// the resumed turn needs — which scope's work is unfinished, which provider
/// it was waiting on, and how much of the error budget was already spent —
/// so a resume continues the SAME budget rather than handing a flapping
/// provider a fresh one. Resume is evidence-driven (the next successful
/// completion on that model), never a timer.
#[derive(Debug, Clone)]
pub struct ParkedTurn {
    pub scope: String,
    pub scope_id: Option<String>,
    /// The model whose recovery is the resume trigger.
    pub model: String,
    /// The user's own words, so a resumed turn re-plans the same goal.
    pub goal: String,
    /// Error rounds already spent — carried, never reset, so
    /// `CONTINUATION_ERROR_ROUNDS` stays the single terminal brake.
    pub error_rounds: usize,
    /// How many times this work has already been resumed from a park.
    ///
    /// A flapping provider can park the same goal repeatedly; each resume
    /// carries the count forward so the SAME terminal brake that bounds
    /// error rounds also bounds park/resume cycles. Nothing new to tune.
    pub resumes: usize,
    /// The provider-facing reason, as told to the user.
    pub reason: String,
    pub parked_at: String,
}

#[derive(Debug, Default)]
pub struct ChatRunRegistry {
    pending: RwLock<HashMap<String, Arc<PendingMessages>>>,
    active: RwLock<std::collections::HashSet<String>>,
    /// Sessions whose turn ended PARKED on a transient provider outage.
    ///
    /// At most one park per session — a second park replaces the first,
    /// because the newer one describes the same unfinished work with fresher
    /// evidence. The bound is therefore the session count, which already
    /// bounds `pending` and `active`; no cap of its own.
    parked: RwLock<HashMap<String, ParkedTurn>>,
    /// Resume counts handed from a park waiter to the turn it is starting.
    ///
    /// Bounded by the parked-session count for the same reason `parked` is:
    /// an entry exists only between a resume decision and the turn that
    /// consumes it, and the turn always consumes it.
    resumed: RwLock<HashMap<String, usize>>,
    /// Wakes admission-gate waiters ([`Self::wait_idle`] / [`Self::wait_active`])
    /// on every claim AND release. One channel for both edges — waiters
    /// re-check their own condition on wake, so a spurious wake costs a read
    /// and a missed edge cannot happen (interest is registered before the
    /// condition is checked).
    changed: tokio::sync::Notify,
}

impl ChatRunRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The intake queue for a session, created on first use.
    pub async fn pending_for(&self, session_id: &str) -> Arc<PendingMessages> {
        if let Some(queue) = self.pending.read().await.get(session_id) {
            return queue.clone();
        }
        let mut pending = self.pending.write().await;
        pending
            .entry(session_id.to_string())
            .or_insert_with(|| Arc::new(PendingMessages::new()))
            .clone()
    }

    /// Whether a harness run is currently live for this session.
    pub async fn is_active(&self, session_id: &str) -> bool {
        self.active.read().await.contains(session_id)
    }

    /// Whether ANY harness run is live. The dream gate consults this: a
    /// mission mid-flight is the opposite of idle however old the last user
    /// message is, and a consolidation that rewrites the run's scoped
    /// memories mid-step deadlocked a live mission (observed 2026-08-10:
    /// dreams opened 16 minutes into a run because idleness was counted
    /// from the unanswered user message, folded 316 of the run's tool-result
    /// memories, and the step never returned).
    pub async fn any_active(&self) -> bool {
        !self.active.read().await.is_empty()
    }

    /// How many sessions have a harness run live — the `/metrics` gauge.
    pub async fn active_count(&self) -> usize {
        self.active.read().await.len()
    }

    /// Claim the run slot. Returns false when one is already live — the
    /// caller must then interject instead of starting a second run.
    pub async fn try_claim(&self, session_id: &str) -> bool {
        let claimed = {
            let mut active = self.active.write().await;
            if active.contains(session_id) {
                false
            } else {
                active.insert(session_id.to_string());
                true
            }
        };
        if claimed {
            // A claim is the "yield NOW" edge for background work holding the
            // local provider — see [`Self::wait_active`].
            self.changed.notify_waiters();
        }
        claimed
    }

    /// Release the slot. Must run on every exit path, including errors.
    pub async fn release(&self, session_id: &str) {
        self.active.write().await.remove(session_id);
        // The "resume" edge for admission-gate waiters parked in
        // [`Self::wait_idle`].
        self.changed.notify_waiters();
    }

    /// Record that this session's turn ended PARKED on a provider outage.
    pub async fn park(&self, session_id: &str, park: ParkedTurn) {
        tracing::warn!(
            session_id,
            model = %park.model,
            error_rounds = park.error_rounds,
            "turn parked on a provider outage — waiting for the provider to answer again"
        );
        self.parked.write().await.insert(session_id.to_string(), park);
    }

    /// Drop a session's park without resuming it — what a NEW user turn does:
    /// that turn already carries the pending items and the established
    /// context, so resuming behind it would duplicate the work.
    pub async fn clear_park(&self, session_id: &str) -> Option<ParkedTurn> {
        self.parked.write().await.remove(session_id)
    }

    /// Whether a session is still waiting on a provider to come back.
    pub async fn is_parked(&self, session_id: &str) -> bool {
        self.parked.read().await.contains_key(session_id)
    }

    /// Remember how many times this session's parked work has been resumed, so
    /// the count survives into the turn that is about to run (and back into
    /// its park, if the provider drops again).
    pub async fn note_resume(&self, session_id: &str, resumes: usize) {
        self.resumed.write().await.insert(session_id.to_string(), resumes);
    }

    /// The resume count a starting turn inherits, and clears as it takes it.
    pub async fn take_resume_count(&self, session_id: &str) -> usize {
        self.resumed.write().await.remove(session_id).unwrap_or(0)
    }

    /// Every session parked on `model`, claimed for resume.
    ///
    /// The resume TRIGGER is recovery evidence — a later successful
    /// completion on the same model — never a timer, so the caller is
    /// whatever observed that success. Claiming removes the park, so two
    /// observers cannot resume the same turn twice.
    pub async fn claim_parked_for_model(&self, model: &str) -> Vec<(String, ParkedTurn)> {
        // Selection and removal happen under one write guard, so two
        // observers cannot both claim the same park.
        self.parked
            .write()
            .await
            .extract_if(|_, park| park.model == model)
            .collect()
    }

    /// Park until NO harness run is live.
    ///
    /// The admission gate for background work on the shared local provider
    /// (P22 Tier 4): the embedding backfill takes its next slot, a yielded
    /// heartbeat resumes, and dream summarization proceeds only through
    /// here. Event-driven — wakes on every claim/release edge, no polling
    /// interval to tune — and PRIORITY, not a quota: the moment the last run
    /// releases, waiters proceed.
    ///
    /// Wakeup-loss safety: interest in the next edge is registered (`enable`)
    /// BEFORE the condition is read, so a release landing between the read
    /// and the await still wakes the waiter.
    /// Wait until `session_id` has no live harness run. Interest is
    /// registered before the read so a release between the check and the
    /// await still wakes us — same wakeup-loss shape as [`Self::wait_idle`].
    pub async fn wait_until_idle(&self, session_id: &str) {
        loop {
            let waiter = self.changed.notified();
            tokio::pin!(waiter);
            waiter.as_mut().enable();
            if !self.active.read().await.contains(session_id) {
                return;
            }
            waiter.await;
        }
    }

    pub async fn wait_idle(&self) {
        loop {
            let waiter = self.changed.notified();
            tokio::pin!(waiter);
            waiter.as_mut().enable();
            if !self.any_active().await {
                return;
            }
            waiter.await;
        }
    }

    /// Park until SOME harness run is live — the preemption signal for
    /// background work already running on the local provider: a scheduled
    /// heartbeat/cron run select-races its own agent turn against this and
    /// yields the generation slot when a user turn arrives (P22 Tier 4;
    /// evidence: a heartbeat held the single GPU slot 157s into a mission's
    /// opening turn while the planner timed out behind it).
    pub async fn wait_active(&self) {
        loop {
            let waiter = self.changed.notified();
            tokio::pin!(waiter);
            waiter.as_mut().enable();
            if self.any_active().await {
                return;
            }
            waiter.await;
        }
    }
}

impl ControlPlane {
    /// Start one chat turn as a long-horizon harness run.
    ///
    /// `conversation` is the bounded rendering of the session so far (see
    /// [`conversation_context`]); it is handed to the planner so a follow-up
    /// turn ("double it") plans against what was actually said.
    ///
    /// Returns `Ok(Some(message_id))` when a run was started — the run itself
    /// proceeds in a spawned task and the caller should ACK immediately.
    /// Returns `Ok(None)` when a run is already live for the session: the
    /// message was admitted to that run instead.
    pub(crate) async fn run_chat_turn(
        self: &Arc<Self>,
        session_id: &str,
        content: &str,
    ) -> Result<Option<String>, String> {
        self.run_chat_turn_with(session_id, content, Vec::new())
            .await
    }

    /// [`Self::run_chat_turn`] for a message that carried images, as
    /// `(base64_data, media_type)` — every step of the turn sends them.
    /// A message admitted into a run already in flight joins that run's
    /// plan as text: its images are not carried (logged).
    pub(crate) async fn run_chat_turn_with(
        self: &Arc<Self>,
        session_id: &str,
        content: &str,
        attachments: Vec<(String, String)>,
    ) -> Result<Option<String>, String> {
        let (Some(agent), Some(router), Some(tools), Some(storage), Some(event_tx)) = (
            self.agent.clone(),
            self.router.clone(),
            self.tools.clone(),
            self.storage.clone(),
            self.event_tx.clone(),
        ) else {
            return Err("long-horizon chat requires agent, router, tools and storage".to_string());
        };

        let registry = self.chat_runs.clone();
        let pending = registry.pending_for(session_id).await;

        // A live run owns this session: the message joins it at the next step
        // boundary rather than starting a competing run.
        if !registry.try_claim(session_id).await {
            warn_uncarried_images(session_id, attachments.len());
            let depth = pending.push(content.to_string()).await;
            tracing::info!(
                session_id,
                depth,
                "message admitted to the live run at the next step boundary"
            );
            return Ok(None);
        }

        // A live user turn supersedes any park: this turn seeds the pending
        // items and rebuilds the established context by itself, so resuming
        // the parked one behind it would do the same work twice.
        if let Some(park) = registry.clear_park(session_id).await {
            tracing::info!(
                session_id,
                model = %park.model,
                parked_at = %park.parked_at,
                "a new user turn supersedes the parked turn — dropping the park"
            );
        }

        let message_id = uuid::Uuid::new_v4().to_string();
        let _ = event_tx.send(crate::protocol::Event::MessageStart {
            session_id: session_id.to_string(),
            message_id: message_id.clone(),
        });

        // Register the run so navigation recovery, Stop, and timeline
        // persistence work exactly as for the in-service chat path.
        let run_handle = agent.register_external_run(session_id).await;

        // P22 liveness: one ledger per session, stamped by the sink below,
        // read by the beat task and the `session.liveness` verb. The entry
        // outlives the turn so cross-turn state (last stop, the repeat-
        // completion streak) survives between turns.
        //
        // The turn opens HERE, before the spawn: the beat task quits the
        // moment it sees a non-running ledger, so opening inside the spawned
        // task would let a delayed first poll of that task kill the beat for
        // the whole turn. Prep (recall over possibly-benched embedders,
        // workspace reload, memory writes) runs inside the spawn and is
        // exactly the silent stretch the beat must cover — the ministral
        // mission's first tool call came 2m28s after send, all of it
        // pre-model. The fingerprint keys the repeat-completion escalation
        // to THIS request; the model is attached once the runner config
        // exists.
        let live = self.liveness.handle(session_id);
        live.begin_turn(None, crate::liveness::content_fingerprint(content));

        let sink = ChatSink {
            session_id: session_id.to_string(),
            message_id: message_id.clone(),
            event_tx: event_tx.clone(),
            run: Some(run_handle.clone()),
            // Parity with the retired direct path: chat tool calls feed the
            // stats tracker and the Turso time-series.
            tool_stats: Some(self.tool_stats.clone()),
            storage: Some(storage.clone()),
            liveness: Some(live.clone()),
            quiet_item: Arc::new(std::sync::Mutex::new(None)),
            text_join: Arc::default(),
        };
        // The finalizer needs the sink after the step runner takes ownership;
        // ChatSink is a bundle of shared handles, so a clone IS the same sink.
        let final_sink = sink.clone();

        // P22: the step runner and planner are built INSIDE the spawned turn,
        // after `prepare_chat_turn` — their system prompt and workspace come
        // out of that prep, and nothing before the spawn may block the
        // delivery ack. `this` is the owned control-plane handle the spawned
        // task preps through.
        let this = Arc::clone(self);

        // Opt-in assistant auto-remember, matching the user-message side of
        // the Send handler.
        let auto_remember = self.config.read().await.memory.auto_remember_messages;
        let memory = self.memory.clone();

        let sessions = self.sessions.clone();
        // The SAME registry `build_task_services` was given, so the baseline
        // this turn publishes is the one `tasks.add` reads.
        let turn_baselines = self.turn_baselines.clone();
        // Zero for an ordinary turn; carried forward when a park waiter
        // started this one, so repeated provider outages spend a single
        // budget instead of a fresh one each time.
        let resumed_from_park = registry.take_resume_count(session_id).await;

        // The interactive turn is a run like any other, so it binds its own
        // session for the whole future instead of leaning on the shared slot.
        // Chat was the last holdout here and it is the case that actually
        // overlaps: turn B's `prepare_chat_turn` re-points the shared binding
        // the moment it is admitted, while turn A is still streaming and still
        // resolving relative paths — so A's tools would start answering with
        // B's root. `with_run_session` gives each turn its own copy, which no
        // other turn can reach.
        //
        // Boxed for the reason the sub-agent and scheduled paths box: this
        // future is large, and inlining it in the spawned task's state machine
        // puts the whole of it on the stack at spawn time.
        //
        // CONSTRAINT the registry's own doc states: a task-local does not cross
        // `tokio::spawn`. The scope therefore has to wrap the future INSIDE the
        // spawn, as it does here, and any tool execution moved onto a task of
        // its own would silently fall back to the shared slot with no compile
        // error.
        // Handles for the liveness beat task below; `live` itself moves into
        // the turn task, which owns begin/finish.
        let beat_live = live.clone();
        let beat_event_tx = event_tx.clone();
        let beat_session = session_id.to_string();
        let chat_turn = ChatTurn {
            this,
            agent,
            router,
            tools,
            storage,
            event_tx,
            registry,
            pending,
            run_handle,
            live,
            final_sink,
            auto_remember,
            memory,
            sessions,
            turn_baselines,
            session_id: session_id.to_string(),
            content: content.to_string(),
            attachments: Arc::new(attachments),
            message_id: message_id.clone(),
            resumed_from_park,
            scope: "session".to_string(),
            scope_id: Some(session_id.to_string()),
        };
        // Handles for the death watcher below — the originals move into the
        // turn task, and the watcher must be able to run the release tail
        // without them.
        let watcher = TurnWatcher::for_turn(&chat_turn);
        let scope_session_id = session_id.to_string();
        // The root of the turn's span tree (P6): every line the turn logs —
        // prep, each harness step, each LLM and tool call — names the session
        // and message it belongs to. Outside `with_run_session` so the span is
        // entered on every poll of the whole turn.
        let turn_span =
            tracing::info_span!("chat_turn", session_id = %session_id, message_id = %message_id);
        let turn = tokio::spawn(tracing::Instrument::instrument(
            ToolRegistry::with_run_session(scope_session_id, Box::pin(chat_turn.run(sink))),
            turn_span,
        ));

        // Death watcher: a turn that dies before its release tail must not
        // leak its registrations. The task above is fire-and-forget, releases
        // are MANUAL at its tail, and there is no process-wide panic hook —
        // so a panic anywhere in the turn used to vanish without a log line
        // while the run claim stayed held. That claim is what admits the
        // session's next message AND what the dream gate consults (PR #207),
        // so the leak wedges the session forever and freezes dreaming with
        // it. The 2026-08-10 incident had exactly this signature: the step's
        // last log line, then 50+ minutes of silence from a live session
        // while every other subsystem ran normally.
        //
        // The watcher only acts on an abnormal death (`JoinError`: panic or
        // abort); a normal return has already run the tail above, and
        // re-running release on an already-released id is a harmless no-op
        // anyway.
        tokio::spawn(watcher.watch(turn));

        // P22 liveness beat: while this turn is in flight, say so — in the
        // log AND over IPC — at the derived cadence (`beat_interval_secs`,
        // ~30s today; see its doc for the derivation from the silence
        // budgets). A dead daemon and a slow model look identical from
        // outside: the 2026-08-10 ministral leg was 3h59m of silence scored
        // as a model result, and the GUI spinner asks the same question every
        // session. The beat makes "alive and lawfully waiting" a positive
        // signal, so the ABSENCE of beats finally means something. The task
        // exits by itself: `beat()` returns None once the release tail or the
        // death watcher closes the ledger's turn.
        tokio::spawn(liveness_beat(beat_live, beat_event_tx, beat_session));

        Ok(Some(message_id))
    }
}

/// A message admitted into a run already in flight joins its plan as text;
/// any images it carried are not carried along. Say so in the log.
fn warn_uncarried_images(session_id: &str, count: usize) {
    if count > 0 {
        tracing::warn!(
            session_id,
            count,
            "images sent into a run already in flight are not carried — only the text joins it"
        );
    }
}

/// One chat turn: the handles its spawned task owns, and its identity.
struct ChatTurn {
    this: Arc<ControlPlane>,
    agent: Arc<crate::agent_service::AgentService>,
    router: Arc<crate::llm_router::LlmRouter>,
    tools: Arc<ToolRegistry>,
    storage: Arc<Storage>,
    event_tx: tokio::sync::broadcast::Sender<crate::protocol::Event>,
    registry: Arc<ChatRunRegistry>,
    pending: Arc<PendingMessages>,
    run_handle: crate::agent_service::ExternalRunHandle,
    live: Arc<crate::liveness::SessionLiveness>,
    /// The finalizer needs the sink after the step runner takes ownership;
    /// `ChatSink` is a bundle of shared handles, so a clone IS the same sink.
    final_sink: ChatSink,
    /// Opt-in assistant auto-remember, matching the user-message side of
    /// the Send handler.
    auto_remember: bool,
    memory: Option<Arc<nanna_memory::MemoryService>>,
    sessions: Arc<crate::session::SessionManager>,
    /// The SAME registry `build_task_services` was given, so the baseline
    /// this turn publishes is the one `tasks.add` reads.
    turn_baselines: Option<Arc<crate::tasks::TurnBaselines>>,
    session_id: String,
    content: String,
    /// Images the message carried, `(base64_data, media_type)`; every step
    /// of the turn sends them (see `AgentStepRunner::attachments`).
    attachments: Arc<Vec<(String, String)>>,
    message_id: String,
    /// Zero for an ordinary turn; carried forward when a park waiter
    /// started this one, so repeated provider outages spend a single
    /// budget instead of a fresh one each time.
    resumed_from_park: usize,
    scope: String,
    scope_id: Option<String>,
}

/// What a turn runs on once prep succeeded.
struct TurnRunners {
    step_runner: AgentStepRunner,
    planner: Arc<AgentPlanner>,
    conversation: Option<String>,
    workspace_root: Option<PathBuf>,
}

/// Everything every harness round of one turn runs against.
struct TurnHarness {
    step_runner: AgentStepRunner,
    planner: Arc<AgentPlanner>,
    source: TursoTaskSource,
    interjector: SessionInterjector,
    workdir: PathBuf,
    config: LongHorizonConfig,
    conversation: Option<String>,
    /// Kept beside the runner because `workspace_root` itself is consumed
    /// into the harness workdir, and the reseed path has to re-read the
    /// artifact ledger from the same root.
    artifact_root: Option<PathBuf>,
    conflicts: Vec<ClaimConflict>,
    conflict_block: Option<String>,
    /// The turn-start facts every planning call in this turn carries.
    standing_context: Vec<String>,
    closed_before_turn: std::collections::HashSet<i64>,
    /// Did the first run ACT? See `ChatTurn::run_mission`.
    run_evidence: bool,
}

/// The continuation loop's accounting.
#[derive(Default)]
struct RoundBudget {
    error_rounds: usize,
    dry_rounds: usize,
    continuations: usize,
    // The loop's own verdict, set wherever the loop
    // DECIDES to stop. `None` means one of the plain
    // conjuncts below ended it, and the tail derives the
    // cause from the counters.
    mission_end: Option<MissionEnd>,
    // A stop is charged to the error budget EXACTLY ONCE.
    // `report.stop` is only replaced when a round runs; the
    // planner-fallback path `continue`s without running one,
    // so the same stop was re-matched and re-charged on the
    // next iteration — observed live, error_rounds jumping
    // 2→4 across one 30-second round, halving a budget that
    // is supposed to buy three real retries.
    stop_charged: bool,
    // The newest provider-health verdict, so the ending can
    // say "the provider is unreachable" instead of "the run
    // keeps failing" when those are different facts.
    last_probe: Option<ProviderProbe>,
    // A round cut short by a provider fault leaves its tool
    // effects on disk while the transcript shows none of
    // them. The next round plans against that gap unless it
    // is told — the same re-anchor the step ladder appends
    // on its own retries ([`crate::tasks::transient_retry_note`]),
    // carried one round forward and dropped the moment a
    // round completes cleanly.
    transient_note: Option<String>,
    // The verified state the last reseed was armed against
    // — a later reseed needs CHANGED evidence, which is
    // what makes one reseed per distinct wall terminate.
    reseed_fingerprint: Option<u64>,
}

/// Where a continuation round leaves the loop.
enum RoundFlow {
    /// Re-evaluate the loop condition.
    Next,
    /// End the loop.
    Stop,
}

impl ChatTurn {
    /// The spawned turn: prep, the mission, then the release tail.
    async fn run(self, sink: ChatSink) {
        // The liveness ledger was opened before this task was spawned
        // (see the `begin_turn` call above). How this turn ended, for
        // the ledger: overwritten wherever a more precise verdict
        // exists; `no_run` covers the paths that never reach the
        // harness (prep failure, seed failure).
        let mut turn_stop_kind = "no_run".to_string();
        // …and WHY the turn itself ended, which is a different question
        // once the continuation loop exists (see [`MissionEnd`]). `None`
        // for the paths that never reach the loop.
        let mut turn_exit_cause: Option<String> = None;

        // Prep, then build the runners. `None` = the turn cannot run at
        // all — the reason is already announced in the transcript, and
        // the release tail below still runs. `acted` is whether the run
        // ACTED (see `TurnHarness::run_evidence`); a turn that never
        // reached the harness did not.
        let acted = if let Some(runners) = self.build_runners(sink, &mut turn_stop_kind).await
            && let Some((stop_kind, exit_cause, run_evidence)) = self.run_mission(runners).await
        {
            turn_exit_cause = Some(exit_cause);
            turn_stop_kind = stop_kind;
            run_evidence
        } else {
            false
        };

        self.finish_turn(&turn_stop_kind, turn_exit_cause.as_deref(), acted)
            .await;
    }

    /// Prep the turn and build its runners; `None` when it cannot run (the
    /// reason is already in the transcript).
    async fn build_runners(&self, sink: ChatSink, turn_stop_kind: &mut String) -> Option<TurnRunners> {
        match self.this.prepare_chat_turn(&self.session_id, &self.content).await {
            Err(message) => {
                tracing::warn!(%message, "chat turn preparation failed — nothing was run");
                self.final_sink.delta(&format!("_could not start the run: {message}_"));
                None
            }
            Ok(prep) => self.runners_for(prep, sink, turn_stop_kind).await,
        }
    }

    /// The step runner and planner for a prepared turn, gated on a provider
    /// serving the resolved model.
    async fn runners_for(
        &self,
        prep: super::chat::ChatTurnPrep,
        sink: ChatSink,
        turn_stop_kind: &mut String,
    ) -> Option<TurnRunners> {
        let super::chat::ChatTurnPrep {
            system_prompt,
            conversation,
            workspace_root,
            workspace_context,
            chat_model,
            chat_tools,
        } = prep;

        // The active workspace scopes stored memories, so a run's
        // observations belong to the workspace they happened in.
        // `services_workspace_id` is the same handle the tool
        // services use (prep just updated it), so tools and
        // memory agree.
        let active_workspace_id = match &self.this.services_workspace_id {
            Some(ws) => ws.read().await.clone(),
            None => None,
        };

        // THE one place a chat turn's model is decided. Hoisted
        // above the runner literal so both runners below are built
        // from the same resolved value: the planner takes
        // `step_runner.agent_config.clone()`, so planning and
        // stepping cannot end up on different models, and the
        // fail-fast further down checks the model that will
        // actually run.
        //
        // `agent_config()` hands back a fresh clone per call, and
        // that clone is the whole isolation mechanism — mutating it
        // here reaches this turn and nothing else. The pin must
        // never be written into the shared `AgentServiceConfig`,
        // which the sub-agent spawner and the dream summarizer read
        // live.
        let is_pinned = chat_model.is_some();
        let agent_config =
            turn_agent_config(self.agent.agent_config().await, chat_model);
        // The fallback walk down the priority list. A pinned chat gets none:
        // answering on a model the user did not pick, with the pin still
        // showing, is the silent substitution the pin exists to rule out.
        let model_chain = if is_pinned {
            None
        } else {
            let mut models = self.agent.chat_model_chain().await;
            if models.first() != Some(&agent_config.model) {
                models.retain(|m| m != &agent_config.model);
                models.insert(0, agent_config.model.clone());
            }
            Some(Arc::new(
                crate::tasks::ModelChain::new(models, &self.router).announcing_to(sink.clone()),
            ))
        };

        // SAY which model won, for the same reason the workspace
        // resolution above says which workspace won: an override
        // nobody can see is indistinguishable from a bug, and "the
        // pin is set but the turn ran on the global model" is
        // exactly the failure this wiring exists to end.
        tracing::info!(
            session_id = %self.session_id,
            model = %agent_config.model,
            source = if is_pinned { "chat pin" } else { "global [llm] default" },
            "chat turn model resolved"
        );

        let step_runner = AgentStepRunner {
            discovered_tools: Arc::new(tokio::sync::RwLock::new(
                std::collections::HashSet::new(),
            )),
            // The tools the user picked for this chat, verbatim.
            user_selected_tools: chat_tools,
            // One ledger for the whole turn: the breakers' streaks
            // must outlive the step boundary that discards every
            // other RunState field, or their thresholds are
            // unreachable.
            repeat_ledger: Arc::new(nanna_agent::RepeatLedger::new()),
            router: self.router.clone(),
            tools: self.tools.clone(),
            agent_config,
            system_prompt,
            workspace_root: workspace_root.clone(),
            workspace_context,
            stats: Some(self.this.model_stats.clone()),
            chat_sink: Some(sink),
            // Tool results go to memory, a stub goes to context.
            memory: self.this.memory.clone(),
            workspace_id: active_workspace_id,
            gpu_fault_count: Arc::new(std::sync::atomic::AtomicU32::new(0)),
            // Capability transitions reach the model once, in the
            // next tool result (P22 Tier 4).
            degradations: self.this.degradations.clone(),
            attachments: Arc::clone(&self.attachments),
            model_chain,
        };
        // The planner shares the step runner's provider handling
        // but must not stream its JSON into the transcript —
        // planning is not work to show.
        let planner_runner = planner_runner_for(&step_runner);
        let planner = Arc::new(AgentPlanner::new(Arc::new(planner_runner)));
        self.gate_unserved_model(
            TurnRunners {
                step_runner,
                planner,
                conversation,
                workspace_root,
            },
            is_pinned,
            turn_stop_kind,
        )
    }

    /// Deterministic fail-fast on a model no provider serves.
    fn gate_unserved_model(
        &self,
        runners: TurnRunners,
        is_pinned: bool,
        turn_stop_kind: &mut String,
    ) -> Option<TurnRunners> {
        let step_runner = &runners.step_runner;
        // Deterministic fail-fast: the planner and every harness
        // step resolve the provider the same way, so a model no
        // provider serves has already decided the whole turn.
        // Without this check the turn still "runs": the planner
        // falls back to a single-task plan, both step attempts
        // fail identically, poison containment cancels the item,
        // and the run exits AllTasksDone with zero steps and
        // nothing streamed — the user sees their prompt struck
        // through as a cancelled task and no reply at all
        // (observed live 2026-07-31, priority set to bare
        // "claude-fable-5" with only OpenRouter configured). Say
        // why instead, and seed no task that is born dead.
        //
        // A pinned chat gets a DIFFERENT sentence, because the
        // Settings wording is a lie for it: the model is not the
        // one Settings names, so a user sent there finds a
        // perfectly-served global model and no explanation. Falling
        // back to that global model would be worse still — the chat
        // would answer on a model the user did not pick, with
        // nothing saying so, which is the silent-substitution class
        // this project has already been bitten by. So name the pin,
        // say it is THIS chat's, and say how to drop it.
        // The model the run will actually START on: the first one in the
        // priority list a provider serves (see `ModelChain::new`).
        let model = step_runner.active_model();
        self.live.set_model(&model);
        // A blank name is no model at all. It must be caught here, not by the
        // provider check below: an unprefixed name resolves to whichever
        // provider claims those, so "" passed that check whenever that
        // provider was registered and every step sent a request naming no
        // model. A pin is never blank (clearing one removes it), so this is
        // always the Settings model.
        if named_models(std::slice::from_ref(&model)).is_empty() {
            *turn_stop_kind = "no_model".to_string();
            tracing::warn!("chat turn cannot run: no model is configured");
            self.final_sink.delta(&format!("_could not run: {NO_MODEL_CONFIGURED}._"));
            return None;
        }
        if step_runner.router.client_for_model(&model).is_none() {
            *turn_stop_kind = "no_provider".to_string();
            tracing::warn!(
                %model,
                pinned = is_pinned,
                "chat turn cannot run: no provider serves the model it must run on"
            );
            self.final_sink.delta(&if is_pinned {
                format!(
                    "_could not run: this chat is pinned to model \
                     '{model}', and no provider is configured for it. \
                     The pin is this conversation's own — it overrides \
                     the model in Settings, so changing Settings will \
                     not help. Either add that provider's credential \
                     (it registers live, no restart needed), pick \
                     another model for this chat, or clear the pin to \
                     fall back to the Settings default._"
                )
            } else {
                format!(
                    "_could not run: no provider is configured for model \
                     '{model}'. Add the provider's credential in Settings \
                     — it registers live, no restart needed — or pick a \
                     model from an available provider in Settings → \
                     Models._"
                )
            });
            None
        } else {
            Some(runners)
        }
    }

    /// Plan and run the mission, then let it continue while there is work.
    /// Returns the ledger's stop kind and the turn's exit cause, or `None`
    /// when the plan could not be seeded.
    /// Snapshot the tasks already closed at turn start and publish that
    /// boundary to every in-run item-creation path (see the comments inside).
    async fn open_turn_baseline(&self) -> std::collections::HashSet<i64> {
        // The turn-start boundary for continuation dedup: tasks
        // already closed NOW belong to history, and only titles
        // closed AFTER this snapshot count as work this turn did
        // (see `seed_continuation`). On a store error the baseline
        // degrades to empty, which OVER-filters — continuation rounds
        // then dedup against all of history and the mission ends
        // early rather than treadmilling.
        let closed_before_turn =
            closed_task_ids(&self.storage, &self.scope, self.scope_id.as_deref())
                .await
                .unwrap_or_default();

        // Publish the boundary so EVERY in-run item-creation path
        // shares it, not just the continuation planner below. The
        // harness's replan step decomposes a stalled item by telling
        // the model to add subtasks through the todo tool, which
        // lands in `tasks.add` — a path this snapshot used not to
        // reach, so an abandoned title came straight back (#2059 →
        // #2060, observed live 2026-08-02). Dropped again on the exit
        // tail that releases the run claim.
        if let Some(ref baselines) = self.turn_baselines {
            baselines
                .open_turn(
                    &self.scope,
                    self.scope_id.as_deref(),
                    closed_before_turn.clone(),
                )
                .await;
        }
        closed_before_turn
    }

    /// Which open items this turn may work: its own, plus leftovers the
    /// planner re-adopts (see [`crate::tasks::TurnAdmission`]). A turn started
    /// by the park waiter IS a resume, so it keeps the whole scope (`None`).
    /// On a store error the admission is skipped — the old whole-scope
    /// behaviour — rather than failing the turn.
    async fn turn_admission(&self) -> Option<Arc<crate::tasks::TurnAdmission>> {
        if self.resumed_from_park > 0 {
            return None;
        }
        let admission = crate::tasks::TurnAdmission::at_turn_start(
            &self.storage,
            &self.scope,
            self.scope_id.as_deref(),
        )
        .await
        .inspect_err(|message| {
            tracing::warn!(%message, "could not snapshot leftovers — admitting the whole scope");
        })
        .ok()
        .map(Arc::new)?;
        // Published beside the turn's baseline (and dropped with it), so a
        // `tasks.add` that reuses an open leftover admits it into this turn.
        if let Some(ref baselines) = self.turn_baselines {
            baselines
                .register_admission(
                    &self.scope,
                    self.scope_id.as_deref(),
                    Arc::clone(&admission),
                )
                .await;
        }
        Some(admission)
    }

    /// Runs the mission; `Some((stop_kind, exit_cause, run_evidence))` when
    /// it reached the harness.
    async fn run_mission(&self, runners: TurnRunners) -> Option<(String, String, bool)> {
        let TurnRunners {
            step_runner,
            planner,
            conversation,
            workspace_root,
        } = runners;
        // Kept beside the runner because `workspace_root` itself is
        // consumed into the harness workdir below, and the reseed
        // path has to re-read the artifact ledger from the same root.
        let artifact_root = workspace_root.clone();
        // Unfinished work from an earlier turn is INFORMATION FOR THE
        // MODEL, not an instruction to the harness. Owner directive
        // (2026-07-25): *"the model should decide to resume or answer
        // another question by the user … i don't think we should assume
        // that the user wants to resume."* Previously the store decided
        // silently: leftover items sorted ahead of the new plan, so a
        // fresh question waited behind stale work nobody re-confirmed.
        // Now the planner is shown what is outstanding and chooses.
        let outstanding = open_work_context(&self.storage, &self.scope, self.scope_id.as_deref()).await;
        // Resume = continue, not restart (P22): a re-send after a
        // run self-terminated must seed the new turn with what the
        // previous one PROVED — closed items and their verified
        // verdicts (which name the commands run and the artifact
        // state they confirmed). The store carries them across the
        // turn boundary; without this block the new planner started
        // from zero and re-seeded the mission's opening sentence
        // verbatim (observed twice in one leg, with six state
        // re-assessments in four hours and zero items ever completed
        // by work).
        let established_state =
            established_rows(&self.storage, &self.scope, self.scope_id.as_deref()).await;
        let established = established_work_context(&established_state);

        // Ground truth about the ARTIFACT, re-read from disk right
        // now — never from prose, never from the model's memory of
        // what it wrote. See [`artifact_state_block`]: continuation
        // turns kept reconstructing files they had already built
        // because nothing in the context said "this file exists and
        // holds verified work".
        let artifact_state =
            artifact_state_block(artifact_root.as_deref(), &established_state).await;

        // The user says something is broken that this session
        // VERIFIED working. Two pieces of evidence disagree, and the
        // one thing that must not happen is rewriting the artifact on
        // the strength of whichever was heard last.
        let conflicts = claim_conflicts(&self.content, &established_state);
        let conflict_block = claim_conflict_block(&conflicts);
        if !conflicts.is_empty() {
            tracing::warn!(
                session_id = %self.session_id,
                conflicts = conflicts.len(),
                subjects = ?conflicts.iter().map(|c| c.subject.as_str())
                    .collect::<Vec<_>>(),
                "the message contradicts a verified pass — reconciling before mutating"
            );
        }

        // The turn-start facts every planning call in this turn
        // carries. Rebuilt (never appended to) on a reseed, so the
        // block is an idempotent snapshot rather than a growing log.
        let standing_context: Vec<String> = [
            conversation.clone(),
            artifact_state,
            conflict_block.clone(),
            established,
        ]
        .into_iter()
        .flatten()
        .collect();
        let context = {
            let mut parts: Vec<&str> =
                standing_context.iter().map(String::as_str).collect();
            parts.extend(outstanding.as_deref());
            let joined = parts.join("\n\n");
            Some(joined).filter(|c| !c.is_empty())
        };

        let closed_before_turn = self.open_turn_baseline().await;
        let admission = self.turn_admission().await;

        let seeded = self
            .plan_and_seed(
                &planner,
                context.as_deref(),
                &conflicts,
                admission.as_deref(),
            )
            .await;
        let ids = match seeded {
            Err(message) => {
                tracing::warn!(%message, "could not seed the chat plan");
                self.final_sink.delta(&format!("_could not start the run: {message}_"));
                return None;
            }
            Ok(ids) => ids,
        };
        // A one-task plan is a conversation-shaped turn: mark its
        // item quiet so the transcript reads as a plain reply,
        // with no step banner. Items added later (interjections,
        // replans) get banners — by then there IS a run to show.
        if let (1, Some(id)) = (ids.len(), ids.first())
            && let Ok(mut quiet) = self.final_sink.quiet_item.lock()
        {
            *quiet = Some(*id);
        }

        let (source, interjector, config) = self.round_io(&planner);
        let source = source.with_admission(admission);
        let workdir = workspace_root.unwrap_or_else(|| PathBuf::from("."));

        let mut report = self
            .first_run(&source, &interjector, &config, &step_runner, &workdir)
            .await;
        let mut harness = TurnHarness {
            step_runner,
            planner,
            source,
            interjector,
            workdir,
            config,
            conversation,
            artifact_root,
            conflicts,
            conflict_block,
            standing_context,
            closed_before_turn,
            run_evidence: false,
        };

        // Keep a MISSION alive while there is still work to plan.
        //
        // `AllTasksDone` means "the current plan drained", not
        // "the goal is met". A 9B planner decomposes a 42-feature
        // build into a handful of tasks, so the turn used to end
        // minutes in with the goal barely started — observed live
        // 2026-07-27: ended normally at 15 minutes, 2 of 42
        // acceptance checks passing. The multi-hour runs this is
        // meant to match came from a fully seeded ladder, which
        // chat never gets.
        //
        // So: re-plan from the CURRENT state (the store is the
        // truth, and the workspace is on disk for the model to
        // inspect) and keep going. Termination is loop-until-dry
        // rather than a step count: when a planning round adds no
        // new open work twice running, the goal is as done as this
        // planner can make it.
        //
        // Only missions continue — but "mission" is about WORK
        // DONE, not plan size. Gating on `ids.len() > 1` was
        // wrong: this planner routinely emits ONE task for a
        // 42-feature build ("Build minidb CLI by iterating
        // through all 42 test files"), so the gate stayed shut on
        // exactly the runs it exists for — observed 2026-07-27, a
        // one-task plan ran 67 tool calls over 12 minutes and then
        // ended at 4/42 with continuation never firing.
        //
        // `steps_taken > 1` was still not the honest signal, and it
        // failed the same way one rung down. Observed 2026-07-28: a
        // 42-feature build produced a ONE-task plan whose FIRST step
        // made 13 tool calls, wrote a file, ran a test, and then
        // marked its single task done. `ids.len() > 1` was false and
        // `steps_taken > 1` was false, so the continuation loop
        // never ran a single round and a mission meant to last hours
        // ended 90 seconds in, at 1/42.
        //
        // Counting steps and items cannot separate a mission from a
        // greeting, because both can be one of each. What separates
        // them is whether the run ACTED: a conversational turn
        // answers from the model's head and calls no tools, while
        // anything that touched the world may have left work behind.
        // So the question asked here is "did this run do anything?",
        // and if it did, the loop below is allowed to ask whether
        // more remains. It is a cheap question — the dry-round
        // counter closes it out after two empty rounds.
        // Say how the FIRST run ended, always. Every exit
        // produces a report and it was simply discarded, so
        // recovering why took a database query. It should
        // take a grep. This line is deliberately NOT the
        // turn's terminal line — it fires before the
        // continuation loop, so a mission that ran twenty
        // rounds would otherwise be described by round one's
        // numbers. The cumulative terminal line is
        // `chat harness mission finished`, emitted once at
        // loop exit below; exactly one line per turn says the
        // MISSION finished.
        tracing::info!(
            stop = ?report.stop,
            steps = report.steps_taken,
            tool_calls = report.tool_calls,
            items = report.items_completed,
            false_success = report.false_success_claims,
            "chat harness first run finished"
        );

        // Run evidence: did this run ACT? See the reasoning
        // above — a conversational turn answers from the
        // model's head and calls no tools.
        harness.run_evidence =
            ids.len() > 1 || report.steps_taken > 1 || report.tool_calls > 0;
        let mut budget = RoundBudget::default();
        self.continue_mission(&mut harness, &mut report, &mut budget).await;
        let (stop_kind, exit_cause) = self.finish_mission(&harness, &report, budget).await;
        Some((stop_kind, exit_cause, harness.run_evidence))
    }

    /// The task source, interjector and harness config every round of this
    /// turn runs with.
    fn round_io(
        &self,
        planner: &Arc<AgentPlanner>,
    ) -> (TursoTaskSource, SessionInterjector, LongHorizonConfig) {
        let source = TursoTaskSource::new(
            self.storage.clone(),
            self.scope.clone(),
            self.scope_id.clone(),
            "chat".to_string(),
            Some(self.event_tx.clone()),
        );
        let interjector = SessionInterjector {
            storage: self.storage.clone(),
            scope: self.scope.clone(),
            scope_id: self.scope_id.clone(),
            pending: self.pending.clone(),
            planner: planner.clone(),
            actor: "chat".to_string(),
            event_tx: Some(self.event_tx.clone()),
            cancel: Some(self.run_handle.cancel.clone()),
        };
        let config = LongHorizonConfig {
            actor: "chat".to_string(),
            ..LongHorizonConfig::default()
        };
        (source, interjector, config)
    }

    /// Plan the turn's goal against its context and seed the plan, unless
    /// Stop was pressed while planning ran.
    async fn plan_and_seed(
        &self,
        planner: &AgentPlanner,
        context: Option<&str>,
        conflicts: &[ClaimConflict],
        admission: Option<&crate::tasks::TurnAdmission>,
    ) -> Result<Vec<i64>, String> {
        self.live.on_planning();
        let mut plan = planner
            .plan(&self.content, context, Some(&self.run_handle.cancel))
            .await;
        // Reconcile BEFORE mutating: one task per contradicted
        // outcome, at the head of the plan (seeding is in plan order
        // and sorts strictly below every existing item, so the head
        // is genuinely first). Whatever the planner proposed to
        // change about that subject now runs after the verdict that
        // says which side's evidence holds.
        prepend_reconciliation_tasks(&mut plan, conflicts);
        tracing::info!(
            session_id = %self.session_id,
            tasks = plan.tasks.len(),
            origin = ?plan.origin,
            reconciliations = conflicts.len(),
            "planned a chat turn"
        );

        // Stop pressed while planning ran: seed nothing and start no
        // work. An empty seed makes the harness run below a no-op — its
        // first cancel check fires before any step — so the turn falls
        // straight through to the persist/release tail.
        if self.run_handle.cancel.is_cancelled() {
            tracing::info!(
                session_id = %self.session_id,
                "cancelled during planning — skipping the run"
            );
            Ok(Vec::new())
        } else {
            // Work the planner proposed again that is already open is
            // ADOPTED, not duplicated: the existing item joins this turn.
            let adopted =
                admission.map_or_else(Vec::new, |admission| admission.adopt_from(&mut plan));
            if !adopted.is_empty() {
                tracing::info!(
                    session_id = %self.session_id,
                    adopted = ?adopted,
                    "the plan re-adopted open work from an earlier turn"
                );
            }
            if plan.tasks.is_empty() {
                return Ok(adopted);
            }
            let mut ids = seed_plan(
                &self.storage,
                &self.scope,
                self.scope_id.as_deref(),
                &plan,
                false,
            )
            .await?;
            ids.splice(0..0, adopted);
            Ok(ids)
        }
    }

    /// The first harness run of the turn, plus the post-run drain for
    /// messages that arrived after its last step boundary.
    async fn first_run(
        &self,
        source: &TursoTaskSource,
        interjector: &SessionInterjector,
        config: &LongHorizonConfig,
        step_runner: &AgentStepRunner,
        workdir: &std::path::Path,
    ) -> nanna_agent::harness::LongHorizonReport {
        const POST_RUN_DRAIN_MAX: usize = 4;

        // Drain-before-release. The interjector is only polled
        // INSIDE the harness loop, before `next()`. A message that
        // arrives after the final poll therefore lands in
        // `pending` with no loop left to notice it: the run exits,
        // the claim is released, and nothing ever starts a run for
        // it — the user's message silently disappears. Observed
        // live: "the queue doesn't seem to ever make it to the
        // model even after the model reaches a stopping point."
        //
        // So: after the harness returns, re-check the queue and
        // run again for whatever arrived late. Bounded by
        // POST_RUN_DRAIN_MAX rather than `while !empty` — a user
        // typing steadily could otherwise keep one turn alive
        // forever, and any message past the bound is still safe in
        // `pending` for the next turn to claim.
        let mut report = nanna_agent::harness::LongHorizonRunner::new(config.clone())
            .run_with_interjector(
                &self.content,
                source,
                step_runner,
                workdir,
                Some(self.run_handle.cancel.clone()),
                Some(interjector),
            )
            .await;

        for sweep in 0..POST_RUN_DRAIN_MAX {
            if self.run_handle.cancel.is_cancelled() {
                break; // the user pressed Stop; do not start more work
            }
            let admitted = match interjector.interject().await {
                Ok(0) => break,
                Ok(n) => n,
                Err(message) => {
                    tracing::warn!(%message, "post-run drain could not seed");
                    break;
                }
            };
            tracing::info!(
                sweep,
                admitted,
                "message arrived after the last step boundary — running it now"
            );
            let extra = nanna_agent::harness::LongHorizonRunner::new(config.clone())
                .run_with_interjector(
                    &self.content,
                    source,
                    step_runner,
                    workdir,
                    Some(self.run_handle.cancel.clone()),
                    Some(interjector),
                )
                .await;
            fold_drain_segment(&mut report, extra, admitted);
        }
        report
    }

    /// The continuation loop: keep a mission going while it has work to plan.
    async fn continue_mission(
        &self,
        h: &mut TurnHarness,
        report: &mut nanna_agent::harness::LongHorizonReport,
        b: &mut RoundBudget,
    ) {
        loop {
            if !self.stop_permits_round(h, report, b).await {
                break;
            }
            if b.dry_rounds >= CONTINUATION_DRY_ROUNDS
                && !self.reseed_before_dry_end(h, report, b).await
            {
                break;
            }
            if b.continuations >= CONTINUATION_ROUNDS_MAX || self.run_handle.cancel.is_cancelled() {
                break;
            }
            b.continuations += 1;
            if matches!(self.continuation_round(h, report, b).await, RoundFlow::Stop) {
                break;
            }
        }
    }

    /// Whether the last run's stop lets the mission continue, charging an
    /// error stop to the budget exactly once.
    async fn stop_permits_round(
        &self,
        h: &TurnHarness,
        report: &nanna_agent::harness::LongHorizonReport,
        b: &mut RoundBudget,
    ) -> bool {
        use nanna_agent::harness::StopReason;
        match report.stop {
            // A completed plan continues only on RUN
            // evidence: a conversational turn the
            // turn-start planner deliberately answered
            // without touching anything must not
            // auto-resume work it chose to defer.
            StopReason::AllTasksDone => h.run_evidence,
            // Transient: the store hiccuped or the model
            // failed a few times in a row. Worth another
            // round, but bounded so a hard fault cannot spin.
            StopReason::SourceError { .. } | StopReason::RunnerErrors { .. } => {
                // A CRASHED run proves nothing about
                // whether a mission exists: it may have
                // died before its first step. So the
                // mission test here also consults the
                // store — open items in this scope are
                // work someone planned and nobody
                // finished, which is exactly what the
                // error budget exists to get back to.
                let open_items = self.storage
                    .tasks()
                    .counts(&self.scope, self.scope_id.as_deref())
                    .await
                    .map_or(0, |(open, _closed)| open);
                let mission = h.run_evidence || open_items > 0;
                tracing::info!(
                    run_evidence = h.run_evidence,
                    open_items,
                    mission,
                    stop = ?report.stop,
                    "error-stop mission test"
                );
                if !mission {
                    false
                } else if b.stop_charged {
                    // Already paid for; the loop is
                    // simply coming back around.
                    true
                } else {
                    // An error round is the budget for
                    // "the run keeps failing while the
                    // provider is up". Spending it on a
                    // provider that is DOWN buys the
                    // mission nothing but a faster
                    // give-up: three rounds of 30-second
                    // planner timeouts burned the whole
                    // budget in a minute. So charge it
                    // against evidence, not the clock.
                    let health = provider_answers(
                        &h.step_runner.router,
                        &h.step_runner.agent_config.model,
                    )
                    .await;
                    b.error_rounds += 1;
                    b.stop_charged = true;
                    b.last_probe = Some(health.clone());
                    // Transient faults are the ones that
                    // cut a step mid-flight; a run that
                    // failed for any other reason has no
                    // orphaned effects to warn about.
                    if let Some(msg) = stop_message(&report.stop)
                        && crate::tasks::is_transient_llm_error(msg)
                    {
                        b.transient_note =
                            Some(crate::tasks::transient_retry_note(
                                b.error_rounds,
                                crate::tasks::transient_fault_kind(msg),
                            ));
                    }
                    if b.error_rounds <= CONTINUATION_ERROR_ROUNDS {
                        tracing::warn!(
                            stop = ?report.stop,
                            round = b.error_rounds,
                            provider_answered = health.answered,
                            probe_secs = health.elapsed_secs,
                            "run ended on an error — retrying rather \
                             than abandoning the mission"
                        );
                        true
                    } else {
                        tracing::error!(
                            stop = ?report.stop,
                            error_rounds = b.error_rounds,
                            budget = CONTINUATION_ERROR_ROUNDS,
                            provider_answered = health.answered,
                            "the error budget is spent — giving up"
                        );
                        b.mission_end = Some(giveup_end(
                            report,
                            Some(&health),
                            MissionEnd::ErrorRoundsExhausted,
                        ));
                        false
                    }
                }
            }
            // Deliberate: the user stopped it, or the budget
            // is genuinely spent. Do not paper over these.
            // The tail names the cause from the stop
            // itself, so nothing is set here.
            _ => false,
        }
    }

    /// ONE fresh-context reseed before a dry ending that would walk away from
    /// checks the environment says still FAIL. Returns whether it reseeded.
    async fn reseed_before_dry_end(
        &self,
        h: &mut TurnHarness,
        report: &nanna_agent::harness::LongHorizonReport,
        b: &mut RoundBudget,
    ) -> bool {
        // ONE fresh-context reseed before a dry ending
        // that would walk away from checks the
        // environment says still FAIL.
        //
        // "Dry" means re-planning found nothing left to
        // do. When walked-away done-conditions are
        // still failing, that is not what happened:
        // the run went blind. The verified wedge was a
        // run-scoped byte-identity breaker
        // short-circuiting the run's own reads of its
        // own artifact, so the model could no longer
        // SEE what it had built and re-planning had
        // nothing to plan against. A fresh runner
        // (clean breaker ledger, clean tool
        // discovery), a re-read of the artifact and a
        // re-based dedup baseline restore turn-start
        // conditions inside the same run — while
        // verified outcomes and cumulative accounting
        // are KEPT, because the knowledge is not what
        // went stale.
        //
        // Armed at most once per DISTINCT verified
        // state: reaching the dry terminal again with
        // the same failing checks and the same verdicts
        // ends the run exactly as today. Total rounds
        // stay bounded by CONTINUATION_ROUNDS_MAX; no
        // new constant exists.
        let fingerprint = verified_state_fingerprint(report);
        if report.abandoned_unmet.is_empty()
            || b.reseed_fingerprint == Some(fingerprint)
        {
            false
        } else {
            b.reseed_fingerprint = Some(fingerprint);
            h.step_runner = fresh_step_runner(&h.step_runner);
            h.closed_before_turn =
                closed_task_ids(&self.storage, &self.scope, self.scope_id.as_deref())
                    .await
                    .unwrap_or_default();
            if let Some(ref baselines) = self.turn_baselines {
                baselines
                    .open_turn(
                        &self.scope,
                        self.scope_id.as_deref(),
                        h.closed_before_turn.clone(),
                    )
                    .await;
            }
            let rows =
                established_rows(&self.storage, &self.scope, self.scope_id.as_deref())
                    .await;
            let artifact =
                artifact_state_block(h.artifact_root.as_deref(), &rows).await;
            h.standing_context = [
                h.conversation.clone(),
                artifact,
                h.conflict_block.clone(),
                established_work_context(&rows),
            ]
            .into_iter()
            .flatten()
            .collect();
            b.dry_rounds = 0;
            tracing::warn!(
                continuations = b.continuations,
                unmet = report.abandoned_unmet.len(),
                "re-planning came up empty while done-conditions still \
                 FAIL — reseeding the run from a fresh context instead \
                 of ending dry"
            );
            self.final_sink.delta(&format!(
                "\n\n_re-planning came up empty, but {} done-condition{} \
                 still FAIL, so this is not a finish. Starting over from \
                 a fresh context — clean tool discovery and a re-read of \
                 the artifact on disk — while keeping everything already \
                 verified. Nothing was undone; disk is truth._\n\n",
                report.abandoned_unmet.len(),
                if report.abandoned_unmet.len() == 1 { "" } else { "s" },
            ));
            true
        }
    }

    /// One continuation round: plan, seed, run.
    async fn continuation_round(
        &self,
        h: &TurnHarness,
        report: &mut nanna_agent::harness::LongHorizonReport,
        b: &mut RoundBudget,
    ) -> RoundFlow {
        let Some(next_plan) = self.plan_continuation(h, report, b).await else {
            return RoundFlow::Stop;
        };
        let seeded = match self.seed_round(h, report, b, &next_plan).await {
            Ok(seeded) => seeded,
            Err(flow) => return flow,
        };
        self.run_round(h, report, b, &seeded).await;
        RoundFlow::Next
    }

    /// Re-plan from the current state; `None` when Stop was pressed while
    /// planning ran.
    async fn plan_continuation(
        &self,
        h: &TurnHarness,
        report: &nanna_agent::harness::LongHorizonReport,
        b: &RoundBudget,
    ) -> Option<nanna_agent::Plan> {
        let outstanding =
            open_work_context(&self.storage, &self.scope, self.scope_id.as_deref()).await;
        // Walked-away work whose done-condition STILL
        // fails is the strongest planning signal the run
        // holds: it names exactly where the goal is
        // unmet, in the environment's own words. Without
        // it the continuation planner re-plans blind and
        // proposes either nothing or the same wall
        // (observed 2026-08-09: turn ended "dry" at 5/42
        // with the failing verdict sitting unread on the
        // drain sweep).
        let unmet_block = if report.abandoned_unmet.is_empty() {
            None
        } else {
            let mut lines = vec![
                "UNMET WORK — these items were given up on, but their \
                 done-conditions STILL FAIL (the goal is not achieved; \
                 plan a different approach to each):"
                    .to_string(),
            ];
            for u in report.abandoned_unmet.iter().take(UNMET_SHOWN_MAX) {
                lines.push(format!(
                    "- #{} {}: {}",
                    u.id, u.title, u.detail
                ));
            }
            Some(lines.join("\n"))
        };
        // The knowledge half, same shape as the unmet
        // half: what this turn VERIFIED done, in the
        // environment's own words. Without it the planner
        // re-plans blind and re-proposes finished work —
        // each re-proposal closing as "already satisfied"
        // and, before P22, counting the mission DRY for
        // discovering its own progress.
        let established_block =
            established_block(&report.verified_outcomes);
        // The turn-start facts (conversation, ARTIFACT
        // STATE, any claim conflict, what earlier turns
        // proved) ride EVERY planning call in the turn,
        // not just the first — a continuation planner that
        // cannot see the artifact plans as if it were not
        // there.
        let ctx = {
            let mut parts: Vec<&str> =
                h.standing_context.iter().map(String::as_str).collect();
            parts.extend(outstanding.as_deref());
            parts.extend(established_block.as_deref());
            parts.extend(unmet_block.as_deref());
            parts.extend(b.transient_note.as_deref());
            parts.join("\n\n")
        };
        let ctx = Some(ctx).filter(|c| !c.is_empty());
        let next_plan = h.planner
            .plan(&self.content, ctx.as_deref(), Some(&self.run_handle.cancel))
            .await;
        // Stop pressed while the continuation round planned:
        // the mission is over — seed nothing.
        if self.run_handle.cancel.is_cancelled() {
            return None;
        }
        Some(next_plan)
    }

    /// Seed the round's plan. An empty round is accounted here: `Err` says
    /// whether the loop re-checks its condition or ends.
    async fn seed_round(
        &self,
        h: &TurnHarness,
        report: &nanna_agent::harness::LongHorizonReport,
        b: &mut RoundBudget,
        next_plan: &nanna_agent::Plan,
    ) -> Result<Vec<i64>, RoundFlow> {
        // Dedup against titles this turn already CLOSED: the
        // finished copy cannot be reused (only OPEN titles
        // dedupe at add), so a planner that re-emits the work
        // it just completed would seed a fresh clone every
        // round and the dry detector would never trip —
        // observed live 2026-08-02 (session 7ccc455a), one
        // question re-planned eleven times to ROUNDS_MAX. An
        // all-duplicate plan seeds nothing and counts dry.
        let seeded = seed_continuation(
            &self.storage,
            &self.scope,
            self.scope_id.as_deref(),
            next_plan,
            &h.closed_before_turn,
        )
        .await
        .unwrap_or_default();
        // Open work is counted, never peeked with `next()` —
        // that CLAIMS an item, and a probe must not consume
        // the work it is probing for.
        let has_work = self.storage
            .tasks()
            .counts(&self.scope, self.scope_id.as_deref())
            .await
            .is_ok_and(|(open, _closed)| open > 0);
        if seeded.is_empty() || !has_work {
            // A FALLBACK plan that seeded nothing is not
            // "re-planning found nothing left" — the
            // planner never spoke (empty/degraded output)
            // and its regenerated monolith title always
            // collides with the closed-title dedup by
            // construction. Observed live 2026-08-08: two
            // such rounds read as dry and ended a mission
            // at 13/42 six minutes in. Planner silence is
            // an ERROR round: bounded by the error budget,
            // never proof the goal is done.
            if next_plan.origin == nanna_agent::planner::PlanOrigin::Fallback {
                // Charge it against the same evidence the
                // stop-match charge uses: a planner that
                // cannot speak because the provider is
                // unreachable is not a planner that keeps
                // failing, and the ending must be able to
                // tell the user which one happened.
                let health = provider_answers(
                    &h.step_runner.router,
                    &h.step_runner.agent_config.model,
                )
                .await;
                b.error_rounds += 1;
                b.last_probe = Some(health.clone());
                tracing::warn!(
                    continuations = b.continuations,
                    error_rounds = b.error_rounds,
                    provider_answered = health.answered,
                    probe_secs = health.elapsed_secs,
                    "continuation planner fell back and seeded nothing — \
                     counting an error round, not a dry one"
                );
                if b.error_rounds > CONTINUATION_ERROR_ROUNDS {
                    tracing::error!(
                        continuations = b.continuations,
                        error_rounds = b.error_rounds,
                        budget = CONTINUATION_ERROR_ROUNDS,
                        provider_answered = health.answered,
                        "continuation planner keeps falling back — planning \
                         starved, giving up"
                    );
                    b.mission_end = Some(giveup_end(
                        report,
                        Some(&health),
                        MissionEnd::PlannerStarvation,
                    ));
                    return Err(RoundFlow::Stop);
                }
                return Err(RoundFlow::Next);
            }
            // A round that planned nothing is only DRY when
            // the environment agrees the goal is met. Its
            // sibling below has always known this; this
            // branch did not, so a mission whose walked-away
            // done-conditions were still FAILING could end
            // "dry" — declaring victory over its own
            // evidence. Reopening the standing wall is the
            // action the invariant implies: the item goes
            // back to pending BY ID (never re-seeded by
            // title — the closed-title dedup eats that) so
            // the next round targets the stored failing
            // verdict instead of re-planning blind.
            //
            // The round is then charged to ROUNDS_MAX, not
            // to the two-strike dry budget, exactly as the
            // post-round unmet branch charges it. Once every
            // unmet item is ALREADY open, reopening changes
            // nothing and the round falls through to dry
            // accounting below — which is what makes this
            // terminate.
            let reopened = reopen_top_unmet(
                &self.storage,
                &report.abandoned_unmet,
            )
            .await;
            if let Some((id, title)) = reopened {
                tracing::warn!(
                    continuations = b.continuations,
                    item = id,
                    %title,
                    "round planned nothing but this item's done-condition \
                     still FAILS — reopening it rather than calling the \
                     mission dry"
                );
                return Err(RoundFlow::Next);
            }
            b.dry_rounds += 1;
            tracing::info!(
                continuations = b.continuations,
                dry_rounds = b.dry_rounds,
                unmet = report.abandoned_unmet.len(),
                "mission continuation planned no new work"
            );
            return Err(RoundFlow::Next);
        }
        Ok(seeded)
    }

    /// Run a seeded round and fold what it did into the turn's report.
    async fn run_round(
        &self,
        h: &TurnHarness,
        report: &mut nanna_agent::harness::LongHorizonReport,
        b: &mut RoundBudget,
        seeded: &[i64],
    ) {
        tracing::info!(
            continuations = b.continuations,
            new_tasks = seeded.len(),
            "mission continues — the goal is not done yet"
        );
        // Continuation rounds ask the ENVIRONMENT first, but
        // only about THIS ROUND'S SEEDED ITEMS. By this point
        // the turn has already run its plan and acted on the
        // world, so "this re-proposal's done-condition already
        // passes" means the work is done — not that the planner
        // wrote a weak condition before anything happened, which
        // is why the first round above runs without the
        // pre-check.
        //
        // Scoped to `seeded` rather than switched on for the
        // round, because the round is not the only source of
        // items: the harness polls the interjector before every
        // selection, so a message the USER sends mid-round is
        // planned into a new item and would be selected under a
        // round-wide flag. An interjected ask whose acceptance
        // happened to pass already would then close with zero
        // steps and the user would never be answered. Leftovers
        // and replan subtasks are outside the set for the same
        // reason: the pre-check may only skip what the
        // continuation planner just re-proposed.
        //
        // And while a claim conflict is UNRESOLVED, the
        // pre-check is suspended entirely for the turn: a
        // disputed pass is not established knowledge, so
        // "the done-condition already passes" is exactly
        // the sentence the user just contradicted. Closing
        // on it would settle the argument in favour of the
        // side nobody re-checked. Suspension costs one
        // step per re-proposal and only on turns where the
        // user reported something broken.
        let round_config = LongHorizonConfig {
            precheck_acceptance_items: if h.conflicts.is_empty() {
                seeded.iter().copied().collect()
            } else {
                std::collections::HashSet::new()
            },
            ..h.config.clone()
        };
        let runner = nanna_agent::harness::LongHorizonRunner::new(round_config);
        let more = runner
            .run_with_interjector(
                &self.content,
                &h.source,
                &h.step_runner,
                &h.workdir,
                Some(self.run_handle.cancel.clone()),
                Some(&h.interjector),
            )
            .await;
        // The dry counter is decided by what the round DID,
        // not by what it was called — see
        // [`round_made_progress`]. Seeding a title the
        // duplicate filter let through is not yet progress:
        // the round has to change something, and items the
        // acceptance pre-check closed for free changed
        // nothing (they prove the goal was ALREADY met).
        //
        // A round that ended in a retryable ERROR is
        // accounted by `error_rounds` alone. It changed
        // nothing either, but feeding both counters would
        // silently cut the error-retry budget from
        // CONTINUATION_ERROR_ROUNDS to
        // CONTINUATION_DRY_ROUNDS: two bounds, two failure
        // modes, neither shortening the other.
        account_round(b, report, &more);
        fold_round(report, more);
        // A NEW run failure is a new charge; the guard only
        // exists to stop one failure being billed twice.
        b.stop_charged = false;
        // The round that just ran saw the re-anchor; whatever
        // it did is in the transcript now, so the warning has
        // done its job and must not repeat.
        b.transient_note = None;
    }

    /// Name how the mission ended, park a transient outage, and say why in
    /// the transcript. Returns the ledger's stop kind and the exit cause.
    async fn finish_mission(
        &self,
        h: &TurnHarness,
        report: &nanna_agent::harness::LongHorizonReport,
        b: RoundBudget,
    ) -> (String, String) {
        let RoundBudget {
            error_rounds,
            dry_rounds,
            continuations,
            mission_end,
            last_probe,
            ..
        } = b;
        // The steps are over: the summary and notices that follow are the
        // harness's, not step text (see `StepTextJoin::end_steps`).
        if let Ok(mut join) = self.final_sink.text_join.lock() {
            join.end_steps();
        }
        // ONE cumulative terminal line per user turn, at the
        // single site every exit path crosses. Non-mission
        // turns cross it too (continuations = 0, cause =
        // single_run), so "how did that turn end" is always
        // exactly one grep — `mission finished` — and never a
        // database query.
        let mission_end = mission_end.unwrap_or_else(|| {
            if !h.run_evidence {
                MissionEnd::SingleRun
            } else if self.run_handle.cancel.is_cancelled() {
                MissionEnd::Cancelled
            } else if dry_rounds >= CONTINUATION_DRY_ROUNDS {
                MissionEnd::DryRoundsExhausted
            } else if continuations >= CONTINUATION_ROUNDS_MAX {
                MissionEnd::RoundsMaxExhausted
            } else {
                MissionEnd::DeliberateStop(stop_kind(&report.stop))
            }
        });
        let exit_cause = mission_end.cause();
        tracing::info!(
            session_id = %self.session_id,
            exit_cause = %exit_cause,
            stop = ?report.stop,
            continuations,
            dry_rounds,
            error_rounds,
            steps = report.steps_taken,
            tool_calls = report.tool_calls,
            side_effect_tool_calls = report.side_effect_tool_calls,
            items_completed = report.items_completed,
            items_abandoned = report.items_abandoned,
            interjected_items = report.interjected_items,
            unmet = report.abandoned_unmet.len(),
            acceptance_unknown = report.acceptance_unknown,
            "chat harness mission finished"
        );

        // The ledger records the FINAL round's verdict — the
        // stop the user actually experienced. The turn's own
        // exit cause rides beside it (see `MissionEnd`), so a
        // starved mission is no longer indistinguishable from
        // a converged one.
        let turn_stop_kind = stop_kind(&report.stop);

        // Park a transient outage rather than demoting the
        // session's work to "gave up": the run is registered
        // with the shared registry so a later recovery can
        // pick it back up, and the ending SAYS so.
        if let MissionEnd::ParkedTransient(ref why) = mission_end {
            self.registry
                .park(
                    &self.session_id,
                    ParkedTurn {
                        scope: self.scope.clone(),
                        scope_id: self.scope_id.clone(),
                        model: h.step_runner.agent_config.model.clone(),
                        goal: self.content.clone(),
                        error_rounds,
                        resumes: self.resumed_from_park,
                        reason: why.clone(),
                        parked_at: chrono::Utc::now().to_rfc3339(),
                    },
                )
                .await;
            // The park is only half a promise until something
            // watches for the recovery it names. One waiter per
            // park, and the waiting IS the probe: a provider
            // that is down cannot answer inside the transport's
            // own silence window, so the probe's latency is the
            // retry cadence — no polling interval to pick.
            spawn_park_waiter(
                Arc::clone(&self.this),
                Arc::clone(&self.registry),
                self.session_id.clone(),
                h.step_runner.router.clone(),
                h.step_runner.agent_config.model.clone(),
            );
        }

        self.announce_mission_end(report, &mission_end, last_probe.as_ref());
        (turn_stop_kind, exit_cause)
    }

    /// The failure notice, the mission-end notice, and the run mechanics line.
    fn announce_mission_end(
        &self,
        report: &nanna_agent::harness::LongHorizonReport,
        mission_end: &MissionEnd,
        last_probe: Option<&ProviderProbe>,
    ) {
        // A run that failed must SAY it failed. Poison
        // containment can drain the whole plan through
        // abandonment and exit `AllTasksDone` having streamed
        // nothing — which rendered as no reply at all, the
        // user's prompt just struck through as a cancelled
        // task (observed live 2026-07-31). Announce WHAT
        // stopped the run and WHY, in the transcript the user
        // is actually looking at.
        if let Some(notice) = failure_notice(report) {
            tracing::warn!(
                stop = ?report.stop,
                last_runner_error = report.last_runner_error.as_deref(),
                "chat run failed — surfacing the reason in the transcript"
            );
            self.final_sink.delta(&notice);
        }

        // …and a MISSION that ended without finishing must say
        // WHY, with the evidence it already holds. The
        // failure notice above only fires on an error-shaped
        // stop, so the planner-starvation give-up — whose
        // `report.stop` still reads `AllTasksDone` — used to
        // surface nothing at all: the run simply stopped.
        //
        // A CANCEL still says nothing about HOW it ended —
        // it was asked for — but it surfaces the failing
        // checks it walked away from, dated with the turn's
        // own age because nothing re-measured them at the
        // stop. That evidence is unrecoverable next turn.
        if let Some(notice) = mission_end_notice(
            report,
            mission_end,
            last_probe,
            self.live.snapshot().elapsed_s,
        ) {
            self.final_sink.delta(&notice);
        }

        // Run mechanics are shown only when there was a real run:
        // a single-step reply stays a plain reply.
        let multi_step = report.steps_taken > 1
            || report.items_completed > 1
            || report.interjected_items > 0;
        if multi_step {
            self.final_sink.delta(&format!(
                "\n\n_{} step{} · {} item{} completed{}{}_",
                report.steps_taken,
                if report.steps_taken == 1 { "" } else { "s" },
                report.items_completed,
                if report.items_completed == 1 { "" } else { "s" },
                // Abandoned work is part of the run's honest
                // arithmetic — "3 items completed" from a
                // 5-item plan must not read as done.
                if report.items_abandoned > 0 {
                    format!(" · {} abandoned", report.items_abandoned)
                } else {
                    String::new()
                },
                if report.interjected_items > 0 {
                    format!(" · {} interjected", report.interjected_items)
                } else {
                    String::new()
                },
            ));
        }
    }

    /// The release tail every exit crosses: state a repeat completion, demote
    /// in-flight items, persist the reply, and release the registrations.
    async fn finish_turn(&self, turn_stop_kind: &str, turn_exit_cause: Option<&str>, acted: bool) {
        // P22: close the liveness ledger. When this exit is a REPEAT —
        // the same request ending `all_tasks_done` again with zero
        // side-effecting work in between — the repeat is STATED in the
        // transcript instead of completing silently. The lfm leg
        // declared itself done 28 times over four hours with nothing on
        // disk, and every declaration looked identical to a genuine
        // finish; a confident "done" repeated after every nudge with
        // nothing to show for it is the most corrosive shape the product
        // has. This delta runs before the transcript is persisted below,
        // so the escalation is part of the assistant message itself.
        //
        // Only for a turn that ACTED. A conversational turn — one step, one
        // item, no tool calls — completes with no side effects every time by
        // design, so asking the same question twice, or pressing Regenerate,
        // used to append "⚠️ repeat completion … nothing was written … If you
        // expected something to exist by now, it does not" to a plain answer.
        // The ledger still records the stop either way.
        let repeat = self.live.finish_turn(turn_stop_kind, turn_exit_cause);
        if let Some(repeats) = repeat.filter(|_| acted) {
            tracing::warn!(
                session_id = %self.session_id,
                repeats,
                "run ended all_tasks_done again for the same request with zero new \
                 side effects — stating the repeat in the transcript"
            );
            self.final_sink.delta(&format!(
                "\n\n_⚠️ repeat completion #{repeats}: this same request has now ended \
                 \"all tasks done\" {} times in a row with no side-effecting work in \
                 between — nothing was written, edited, or executed, so the world is \
                 exactly as it was. If you expected something to exist by now, it does \
                 not. Name the missing outcome (a file, a command, a change) and I \
                 will target it directly instead of re-verifying._",
                repeats + 1,
            ));
        }

        // The turn is over and the chat is back to waiting on input, so it
        // no longer holds anything: every in-flight item returns to
        // pending. `in_progress` means "a running turn is working this
        // right now" — nothing is running now.
        //
        // This ran only for CANCELLED turns before, which left a turn that
        // ended normally holding its items forever. `next()` sorts
        // `in_progress` first ("resume what you were doing"), so those
        // leftovers outranked the user's next message and the chat resumed
        // stale work instead of answering it — the same hijack observed on
        // 2026-08-02 after a cancel, just reached by the ordinary path.
        // Left to accumulate it reads as tasks that follow you around and
        // never die, which is what a long-lived chat per workspace looks
        // like from outside.
        //
        // Demotion is not closure: the items stay open and the planner is
        // still shown them as unfinished work per the owner directive
        // above. They simply stop being at the head of the queue, so the
        // next message decides what happens to them.
        match demote_in_progress(&self.storage, &self.scope, self.scope_id.as_deref(), "chat").await {
            Ok(0) => {}
            Ok(demoted) => tracing::info!(
                demoted,
                cancelled = self.run_handle.cancel.is_cancelled(),
                "turn ended — in-flight items returned to pending"
            ),
            Err(message) => {
                tracing::warn!(%message, "could not release the turn's in-flight items");
            }
        }

        self.state_a_silent_finish(turn_stop_kind);
        self.persist_reply().await;

        // The turn's workdir binding is turn-scoped and dies with the turn.
        // Kept, it would grow the map one permanent entry per session the
        // daemon ever chats in, and — worse — a session whose workspace is
        // later closed would keep a stale entry that now WINS over the
        // global default, so tools would resolve against a directory the
        // session no longer has. This mirrors what the sub-agent path
        // already does at its own tail (`control/session.rs`). Before
        // `registry.release`, so the next turn for this session cannot be
        // admitted until the stale binding is gone; its own
        // `prepare_chat_turn` binds a fresh one.
        self.agent.tools().clear_session_workdir(&self.session_id).await;

        // Every exit path releases all three registrations — a leaked
        // entry would make the session look busy forever, and a leaked
        // turn baseline would keep filtering titles into the next turn,
        // where re-asking is legitimate.
        self.agent.unregister_external_run(&self.session_id).await;
        self.registry.release(&self.session_id).await;
        if let Some(ref baselines) = self.turn_baselines {
            baselines.close_turn(&self.scope, self.scope_id.as_deref()).await;
        }
    }

    /// A turn that finished with nothing to show says so.
    ///
    /// A model that replies to a message with the bare `TASK COMPLETE` marker
    /// closes its item — the claim is honoured — and the marker is then
    /// stripped from the reply, which leaves a successful turn with no text
    /// and no tool calls: an empty message the GUI hides, i.e. silence. An
    /// empty *completion* is already reported (`_could not run: empty
    /// completion…_`); an empty *claimed* completion gets the same honesty.
    /// Never on a cancel: an empty stopped turn is exactly what Stop asked
    /// for.
    fn state_a_silent_finish(&self, turn_stop_kind: &str) {
        if turn_stop_kind != "all_tasks_done" || self.run_handle.cancel.is_cancelled() {
            return;
        }
        let said_nothing = strip_harness_markers(&crate::agent_service::lock_buf(
            &self.run_handle.accumulated_text,
        ))
        .trim()
        .is_empty();
        let did_nothing =
            crate::agent_service::lock_buf(&self.run_handle.completed_tool_calls).is_empty();
        if said_nothing && did_nothing {
            tracing::warn!(
                session_id = %self.session_id,
                "turn finished with no reply text and no tool calls — stating it"
            );
            self.final_sink.delta(SILENT_FINISH_NOTICE);
        }
    }

    /// Persist the WHOLE run as the assistant message, auto-remember it when
    /// opted in, and close the message stream.
    async fn persist_reply(&self) {
        // Persist the WHOLE run: the message content is everything that
        // streamed (so conversation history and exports read like chat),
        // and the timeline journal carries the interleaved record.
        // Harness plumbing (the TASK COMPLETE claim marker) is stripped
        // from both — it is a verdict signal, not conversation.
        let mut full_text =
            crate::agent_service::lock_buf(&self.run_handle.accumulated_text).clone();
        let repeated = self
            .final_sink
            .text_join
            .lock()
            .ok()
            .and_then(|join| join.repeated_last_step());
        if let Some((ref repeat, ref span)) = repeated {
            drop_repeated_span(&mut full_text, repeat, span);
        }
        let mut content = strip_harness_markers(&full_text);
        let mut timeline = sanitize_timeline(
            self.run_handle
                .timeline
                .lock()
                .map(|journal| journal.clone())
                .unwrap_or_default(),
        );
        if let Some((ref repeat, _)) = repeated {
            drop_repeat_from_timeline(&mut timeline, repeat);
        }
        if self.run_handle.cancel.is_cancelled() {
            mark_stopped(&mut content, &mut timeline);
        }
        self.sessions
            .add_full_message(
                &self.session_id,
                MessageRole::Assistant,
                &content,
                crate::session::MessageDetails {
                    timeline,
                    ..crate::session::MessageDetails::default()
                },
            )
            .await;

        if self.auto_remember
            && let Some(memory) = &self.memory
            && content.split_whitespace().count() >= 3
            && let Err(e) = memory
                .remember_with_importance(&content, HashMap::new(), 1.0)
                .await
        {
            tracing::debug!("Failed to auto-remember assistant response: {e}");
        }

        let _ = self.event_tx.send(crate::protocol::Event::MessageEnd {
            session_id: self.session_id.clone(),
            message_id: self.message_id.clone(),
            content,
        });
    }
}

/// Cut the converging repeat — the last step's verbatim copy of the step
/// before it — out of the reply, at the span `StepTextJoin` recorded.
///
/// A model that answers without saying `TASK COMPLETE` closes its item by
/// repeating the answer (`answer_converged`). The live stream has shown both
/// copies; the reply the user keeps, and the `message_end` the GUI promotes
/// the live bubble to, needs one. A no-op unless the span really holds the
/// repeat (a dropped delta would shift it), and never the reply's only copy.
fn drop_repeated_span(text: &mut String, repeat: &str, span: &std::ops::Range<usize>) {
    let Some(slice) = text.get(span.clone()) else {
        return;
    };
    if slice.trim() != repeat.trim() || span.start == 0 {
        return;
    }
    debug_assert!(span.end <= text.len());
    text.replace_range(span.clone(), "");
}

/// The timeline twin of [`drop_repeated_span`]: a conversation-shaped turn
/// merges its steps' text into one entry, so the repeat is that entry's
/// trailing copy (the run summary is not in the timeline).
fn drop_repeat_from_timeline(timeline: &mut [TimelineItem], repeat: &str) {
    let Some(TimelineItem::Text { content, .. }) = timeline
        .iter_mut()
        .rev()
        .find(|item| matches!(item, TimelineItem::Text { .. }))
    else {
        return;
    };
    let tail = repeat.trim();
    let body = content.trim_end();
    if tail.is_empty() || !body.ends_with(tail) || body.len() == tail.len() {
        return;
    }
    let kept = content[..body.len() - tail.len()].trim_end().len();
    content.truncate(kept);
}

/// The marker the GUI shows on a bubble the user stopped (`stopSession` in
/// `pages/index.vue`) — the same text, so the two agree.
const STOPPED_MARKER: &str = "[Stopped by user]";

/// Persist a stopped turn the way the GUI showed it.
///
/// The GUI appends `[Stopped by user]` to the live bubble and then lets the
/// turn's `message_end` replace it, expecting the daemon to have persisted the
/// same marker. The harness path persisted only what had streamed — so a turn
/// stopped before any text ended as `""`: the marker vanished from the live
/// bubble, history kept an empty assistant message, and later turns' context
/// read nothing where the user had said "stop".
fn mark_stopped(content: &mut String, timeline: &mut Vec<TimelineItem>) {
    if content.contains(STOPPED_MARKER) {
        return;
    }
    let appended = if content.trim().is_empty() {
        STOPPED_MARKER.to_string()
    } else {
        format!("\n\n{STOPPED_MARKER}")
    };
    content.push_str(&appended);
    timeline.push(TimelineItem::Text {
        content: appended,
        at: chrono::Utc::now().to_rfc3339(),
    });
    debug_assert!(
        content.ends_with(STOPPED_MARKER),
        "the marker closes the reply"
    );
}

/// What a turn that finished with nothing to show says instead of nothing.
const SILENT_FINISH_NOTICE: &str = "_finished without a reply: the model marked this done but said nothing and ran \
     no tools. Ask again, or rephrase if the request was unclear._";

/// The planner shares the step runner's provider handling but must not
/// stream its JSON into the transcript — planning is not work to show.
fn planner_runner_for(step_runner: &AgentStepRunner) -> AgentStepRunner {
    AgentStepRunner {
        discovered_tools: Arc::new(tokio::sync::RwLock::new(
            std::collections::HashSet::new(),
        )),
        // Planning calls no tools, so user picks are noise here.
        user_selected_tools: Vec::new(),
        chat_sink: None,
        router: step_runner.router.clone(),
        tools: step_runner.tools.clone(),
        agent_config: step_runner.agent_config.clone(),
        system_prompt: step_runner.system_prompt.clone(),
        workspace_root: step_runner.workspace_root.clone(),
        // The planner sees the same bounded reference: acting
        // on ROADMAP items nobody asked for is precisely a
        // planning failure.
        workspace_context: step_runner.workspace_context.clone(),
        stats: step_runner.stats.clone(),
        // Planning calls no tools, so it has nothing to remember.
        memory: None,
        workspace_id: None,
        // One run, one fault tally: a GPU fault seen while
        // planning and one seen while stepping are the same
        // repeat evidence. The breaker ledger is shared for
        // the same reason — planning calls no tools today, so
        // this costs nothing and cannot drift if that ever
        // changes.
        gpu_fault_count: step_runner.gpu_fault_count.clone(),
        repeat_ledger: Arc::clone(&step_runner.repeat_ledger),
        degradations: step_runner.degradations.clone(),
        // The planner answers in JSON about the request's text; the images
        // are for the steps that do the work.
        attachments: Arc::default(),
        // One walk per run: a model the planner saw fail is not retried
        // by the next step, and the other way round.
        model_chain: step_runner.model_chain.clone(),
    }
}

/// Fold a post-run drain segment into the turn's report.
fn fold_drain_segment(
    report: &mut nanna_agent::harness::LongHorizonReport,
    extra: nanna_agent::harness::LongHorizonReport,
    admitted: usize,
) {
    report.steps_taken += extra.steps_taken;
    report.tool_calls += extra.tool_calls;
    report.side_effect_tool_calls += extra.side_effect_tool_calls;
    report.items_completed += extra.items_completed;
    report.items_already_satisfied += extra.items_already_satisfied;
    report.items_abandoned += extra.items_abandoned;
    report.items_completed_unverified +=
        extra.items_completed_unverified;
    report.items_revived += extra.items_revived;
    report.replans += extra.replans;
    report.false_success_claims += extra.false_success_claims;
    report.interjected_items += admitted + extra.interjected_items;
    // Union by id, like the knowledge half below: a
    // drain segment's dropped work must survive to the
    // closing message, which NAMES abandonments rather
    // than counting them. The harness only ever
    // appends to this list — no sweep revives an item
    // that had no check — so a segment can add to it
    // and never correct it.
    for a in &extra.abandoned_unverifiable {
        report.abandoned_unverifiable.retain(|p| p.id != a.id);
    }
    report
        .abandoned_unverifiable
        .extend(extra.abandoned_unverifiable.clone());
    for v in &extra.verified_outcomes {
        report.verified_outcomes.retain(|p| p.id != v.id);
    }
    report.verified_outcomes.extend(extra.verified_outcomes.clone());
    report.acceptance_unknown += extra.acceptance_unknown;
    if extra.last_runner_error.is_some() {
        report.last_runner_error = extra.last_runner_error;
    }
    // The drain segment ran last, so its stop describes
    // where the turn actually ended up — same rule as
    // `fold_reports`.
    report.stop = extra.stop;
}

/// Decide what a finished continuation round counts as — see
/// [`round_made_progress`].
fn account_round(
    b: &mut RoundBudget,
    report: &nanna_agent::harness::LongHorizonReport,
    more: &nanna_agent::harness::LongHorizonReport,
) {
    let errored = matches!(
        more.stop,
        nanna_agent::harness::StopReason::SourceError { .. }
            | nanna_agent::harness::StopReason::RunnerErrors { .. }
    );
    if round_made_progress(more) {
        b.dry_rounds = 0;
    } else if !errored {
        // A failing done-condition on walked-away work
        // REFUTES "the goal is done": this round found
        // nothing, but the environment says the mission
        // is unmet, so the round consumes the bounded
        // continuation budget (ROUNDS_MAX) instead of
        // the two-strike dry budget. Dryness may only
        // conclude a mission the evidence permits.
        if more.items_already_satisfied > 0 {
            // Knowledge, not a dry round (P22): the
            // round PROVED work is done — that fact
            // now rides `verified_outcomes` into the
            // next planning context, so an informed
            // planner either proposes genuinely new
            // work or seeds nothing, and THAT round
            // counts dry. Discovering "already done"
            // must make the mission faster, never
            // push it toward giving up (observed:
            // three runs died at dry_rounds=2 with
            // already_satisfied closures on the
            // books). Bounded by ROUNDS_MAX and by
            // the closed-title dedup either way.
            tracing::info!(
                continuations = b.continuations,
                already_satisfied = more.items_already_satisfied,
                "round closed items by evidence — knowledge, \
                 not a dry round; feeding facts to the planner"
            );
        } else if more.abandoned_unmet.is_empty()
            && report.abandoned_unmet.is_empty()
        {
            b.dry_rounds += 1;
            tracing::info!(
                continuations = b.continuations,
                dry_rounds = b.dry_rounds,
                steps = more.steps_taken,
                "mission continuation changed nothing and closed nothing"
            );
        } else {
            tracing::warn!(
                continuations = b.continuations,
                unmet = more
                    .abandoned_unmet
                    .len()
                    .max(report.abandoned_unmet.len()),
                "round found nothing new but abandoned checks still \
                 fail — the goal is provably unmet; not a dry round"
            );
        }
    }
}

/// Fold a continuation round's report into the turn's.
fn fold_round(
    report: &mut nanna_agent::harness::LongHorizonReport,
    more: nanna_agent::harness::LongHorizonReport,
) {
    report.steps_taken += more.steps_taken;
    report.tool_calls += more.tool_calls;
    report.side_effect_tool_calls += more.side_effect_tool_calls;
    report.items_completed += more.items_completed;
    report.items_already_satisfied += more.items_already_satisfied;
    report.items_abandoned += more.items_abandoned;
    report.items_completed_unverified +=
        more.items_completed_unverified;
    report.items_revived += more.items_revived;
    report.replans += more.replans;
    report.false_success_claims += more.false_success_claims;
    report.interjected_items += more.interjected_items;
    // Union by id: a round's sweep only re-checks the
    // items THAT round abandoned, so earlier rounds'
    // standing walls must persist (same-id entries
    // refresh to the newest verdict). A wall that has
    // since fallen is bounded by ROUNDS_MAX, and the
    // per-round context pushes the model straight at
    // it, which is the fastest way to find out.
    for u in &more.abandoned_unmet {
        report.abandoned_unmet.retain(|p| p.id != u.id);
    }
    report.abandoned_unmet.extend(more.abandoned_unmet.clone());
    // Same union for the UNCHECKED half of the same
    // story — the majority of abandonments. It is
    // append-only in the harness (no sweep can revive
    // an item that had no check), so a later round can
    // only add to it, and dropping it here would leave
    // the closing message able to count round two's
    // dropped work but never name it.
    for a in &more.abandoned_unverifiable {
        report.abandoned_unverifiable.retain(|p| p.id != a.id);
    }
    report
        .abandoned_unverifiable
        .extend(more.abandoned_unverifiable.clone());
    // Same union for the knowledge half: newest
    // verdict per id wins, earlier rounds' facts
    // persist so the planner context accumulates.
    for v in &more.verified_outcomes {
        report.verified_outcomes.retain(|p| p.id != v.id);
    }
    report.verified_outcomes.extend(more.verified_outcomes.clone());
    report.acceptance_unknown += more.acceptance_unknown;
    if more.last_runner_error.is_some() {
        report.last_runner_error = more.last_runner_error;
    }
    report.stop = more.stop;
}

/// The handles the death watcher needs to run a crashed turn's release tail.
struct TurnWatcher {
    registry: Arc<ChatRunRegistry>,
    agent: Arc<crate::agent_service::AgentService>,
    event_tx: tokio::sync::broadcast::Sender<crate::protocol::Event>,
    sink: ChatSink,
    baselines: Option<Arc<crate::tasks::TurnBaselines>>,
    session: String,
    message_id: String,
    live: Arc<crate::liveness::SessionLiveness>,
    tools: Arc<ToolRegistry>,
}

impl TurnWatcher {
    fn for_turn(turn: &ChatTurn) -> Self {
        Self {
            registry: turn.registry.clone(),
            agent: turn.agent.clone(),
            event_tx: turn.event_tx.clone(),
            sink: turn.final_sink.clone(),
            baselines: turn.turn_baselines.clone(),
            session: turn.session_id.clone(),
            message_id: turn.message_id.clone(),
            live: turn.live.clone(),
            tools: turn.agent.tools().clone(),
        }
    }

    /// Release a turn that died (panic or abort) before its release tail.
    async fn watch(self, turn: tokio::task::JoinHandle<()>) {
        let Self {
            registry: watcher_registry,
            agent: watcher_agent,
            event_tx: watcher_event_tx,
            sink: watcher_sink,
            baselines: watcher_baselines,
            session: watcher_session,
            message_id: watcher_message_id,
            live: watcher_live,
            tools: watcher_tools,
        } = self;
        let Err(join_error) = turn.await else {
            return;
        };
        tracing::error!(
            session_id = %watcher_session,
            panicked = join_error.is_panic(),
            "chat turn task died before its release tail: {join_error}; \
             releasing its registrations"
        );
        watcher_sink.delta(
            "\n\n_internal error: this turn crashed before finishing; the \
             session has been released — see the daemon log._",
        );
        let _ = watcher_event_tx.send(crate::protocol::Event::MessageEnd {
            session_id: watcher_session.clone(),
            message_id: watcher_message_id,
            content: String::new(),
        });
        // The tail's workdir clear is exactly as unreachable on this path as
        // the releases below, and leaving the binding behind is the worse
        // half of the leak: the next thing to resolve a path as this
        // session would silently use a dead turn's root.
        watcher_tools.clear_session_workdir(&watcher_session).await;
        watcher_agent.unregister_external_run(&watcher_session).await;
        watcher_registry.release(&watcher_session).await;
        if let Some(baselines) = watcher_baselines {
            baselines.close_turn("session", Some(&watcher_session)).await;
        }
        // Close the liveness ledger too, or the beat task would keep
        // beating for a turn that no longer exists. `crashed` also gives
        // the `session.liveness` verb an honest stop state.
        // No exit cause: the turn never reached the continuation loop's
        // exit, and `crashed` already says everything known.
        let _ = watcher_live.finish_turn("crashed", None);
    }
}

/// The P22 liveness beat for one turn — see its call site.
async fn liveness_beat(
    beat_live: Arc<crate::liveness::SessionLiveness>,
    beat_event_tx: tokio::sync::broadcast::Sender<crate::protocol::Event>,
    beat_session: String,
) {
    let period =
        std::time::Duration::from_secs(crate::liveness::beat_interval_secs());
    loop {
        tokio::time::sleep(period).await;
        let Some(snap) = beat_live.beat() else { break };
        tracing::info!(
            session_id = %beat_session,
            elapsed_s = snap.elapsed_s,
            quiet_s = snap.quiet_s,
            phase = snap.phase.as_str(),
            awaiting = %snap.awaiting,
            step_index = ?snap.step_index,
            last_tool = snap.last_tool.as_ref().map(|t| t.name.as_str()),
            beat = snap.beats,
            "liveness beat"
        );
        let _ = beat_event_tx.send(crate::protocol::Event::LivenessBeat {
            session_id: beat_session.clone(),
            elapsed_s: snap.elapsed_s,
            phase: snap.phase.as_str().to_string(),
            awaiting: snap.awaiting.clone(),
            quiet_s: snap.quiet_s,
            step_index: snap.step_index,
            last_tool: snap.last_tool.map(|t| t.name),
            beat: snap.beats,
        });
    }
}

/// The [`AgentConfig`] THIS chat turn runs on: the chat's own pin if it has
/// one, otherwise the global `[llm]` default exactly as it arrived.
///
/// This is the single point where the precedence rule is executed. `base` is
/// always a fresh per-turn clone from
/// [`crate::agent_service::AgentService::agent_config`] — that freshness is the
/// entire isolation mechanism, so this function may mutate what it is given and
/// must never be handed shared state.
///
/// The unpinned path returns `base` untouched rather than reconstructing it, so
/// a chat that never picked a model is byte-for-byte what it was before per-chat
/// models existed. That is not a nicety: the global default carries a model
/// routing tier table, provider keys and the whole nudge policy, and "unpinned
/// behaves as before" is only checkable if nothing on that path is rebuilt.
fn turn_agent_config(base: AgentConfig, chat_model: Option<String>) -> AgentConfig {
    let Some(model) = chat_model else {
        return base;
    };
    let mut config = base;
    // The one helper that knows what a pin moves and — just as importantly —
    // what it must clear (`model_routing`, or the loop re-picks a global model
    // from iteration 2 onward). Never re-implemented here; see its doc.
    crate::agent_service::apply_chat_model_override(&mut config, model);
    config
}

/// Shape the chat handler returns when a message joined a live run.
/// `content` is present (empty) because the GUI command layer requires the
/// field on every non-error response.
#[must_use]
pub fn interjected_response(session_id: &str, depth: usize) -> Value {
    json!({
        "status": "interjected",
        "session_id": session_id,
        "pending": depth,
        "content": "",
        "message": "admitted to the run in progress at the next step boundary",
        // P22 delivery contract — see `started_response`.
        "delivery": "accepted",
        "accepted_at": chrono::Utc::now().to_rfc3339(),
    })
}

/// Shape the chat handler returns the moment a NEW run is admitted: the
/// DELIVERY ack (P22).
///
/// `delivery: "accepted"` + `accepted_at` are the
/// explicit contract that this response certifies delivery only — the user
/// message is persisted and a run owns it. Run completion arrives as events
/// (`message_end`), never in this response, and nothing slower than the
/// claim decision may run before it: the bench driver twice reported a
/// mission as un-sent after a 120s `chat.send` timeout while the daemon had
/// in fact accepted it and was grinding through recall (2026-08-10), and the
/// GUI has the same ambiguity for any first turn slow to produce a token.
#[must_use]
pub fn started_response(message_id: &str) -> Value {
    json!({
        "status": "started",
        "message_id": message_id,
        "content": "",
        "delivery": "accepted",
        "accepted_at": chrono::Utc::now().to_rfc3339(),
    })
}

/// Snake-case wire kind of a stop reason, extracted from the enum's own serde
/// tag (`#[serde(tag = "reason", rename_all = "snake_case")]`) so the
/// liveness ledger's spelling can never drift from the enum.
fn stop_kind(stop: &nanna_agent::harness::StopReason) -> String {
    serde_json::to_value(stop)
        .ok()
        .and_then(|v| v.get("reason").and_then(|r| r.as_str().map(str::to_string)))
        .unwrap_or_else(|| "unknown".to_string())
}

/// The message an ERROR stop carries, if any. Deliberate stops carry none.
const fn stop_message(stop: &nanna_agent::harness::StopReason) -> Option<&str> {
    use nanna_agent::harness::StopReason;
    match stop {
        StopReason::RunnerErrors { message } | StopReason::SourceError { message } => {
            Some(message.as_str())
        }
        _ => None,
    }
}

/// What one probe of the run's provider found.
#[derive(Debug, Clone)]
struct ProviderProbe {
    model: String,
    /// The provider produced a completion — it is up, whatever the run did.
    answered: bool,
    elapsed_secs: u64,
    error: Option<String>,
}

/// Ask the run's own model for the smallest possible completion: is the
/// provider ANSWERING right now?
///
/// An error round is the budget for "the run keeps failing while the provider
/// is up". Spending it on a provider that is DOWN buys the mission nothing —
/// observed live, three fallback rounds of planner timeouts consumed the whole
/// budget inside a minute and the turn gave up on a mission whose only problem
/// was an unreachable endpoint. So each round is charged against evidence:
/// the probe says which of "the provider is unreachable" and "the run keeps
/// failing" actually happened, and the ending says the right one.
///
/// Bounds — both existing, neither new:
/// - the deadline is [`nanna_llm::STREAM_READ_TIMEOUT_SECS`], the transport's
///   OWN declared silence tolerance. A provider that cannot deliver a single
///   token inside the window the transport already refuses to wait past is
///   not answering by the daemon's existing definition. It also paces the
///   give-up honestly: an outage now costs a full transport window per error
///   round instead of a 30-second planner timeout, so the budget is spent
///   over minutes of demonstrated silence rather than seconds.
/// - `max_tokens: 1` — the question is "does it answer", not "what does it
///   say", and one token is the smallest answer that exists.
///
/// Side benefit: a successful probe re-warms a model that a runner reset
/// unloaded (`keep_alive = 0`), so the next planning call is not spent on a
/// cold load — part of the observed fallback cascade was self-inflicted.
///
/// Never returns an error: a probe that cannot run IS the "not answering"
/// verdict.
/// Watch one parked turn until its provider answers again, then resume it.
///
/// The park's own promise ("this resumes when the provider is back") is only
/// true if something is waiting, and the wait must cost what the evidence
/// costs: [`provider_answers`] returns in about a second when the provider is
/// up and holds for the transport's own silence window when it is down, so the
/// probe IS the retry cadence — there is no interval to invent, and a healthy
/// provider is never polled after the first answer.
///
/// A resume re-enters through the ordinary turn path, so the resumed work sees
/// exactly what a user-sent continuation would: established context, the
/// artifact-state preamble, and the scope's still-open items. The park's
/// `resumes` count rides along and is bounded by the same
/// [`CONTINUATION_ERROR_ROUNDS`] brake that bounds error rounds, so a flapping
/// provider cannot loop forever.
fn spawn_park_waiter(
    control: Arc<super::ControlPlane>,
    registry: Arc<ChatRunRegistry>,
    session_id: String,
    router: Arc<crate::llm_router::LlmRouter>,
    model: String,
) {
    tokio::spawn(async move {
        loop {
            // The park may have been claimed by a user message in the
            // meantime — that turn already carries the work forward.
            if !registry.is_parked(&session_id).await {
                return;
            }
            let probe = provider_answers(&router, &model).await;
            if !probe.answered {
                continue;
            }
            let Some(park) = registry.clear_park(&session_id).await else {
                return;
            };
            if park.resumes >= CONTINUATION_ERROR_ROUNDS {
                tracing::warn!(
                    session_id = %session_id,
                    model = %model,
                    resumes = park.resumes,
                    "parked turn has already been resumed as often as the error \
                     budget allows — leaving it for the user"
                );
                return;
            }
            tracing::info!(
                session_id = %session_id,
                model = %model,
                resumes = park.resumes + 1,
                probe_secs = probe.elapsed_secs,
                "provider answered again — resuming the parked turn"
            );
            registry.note_resume(&session_id, park.resumes + 1).await;
            if let Err(e) = control.run_chat_turn(&session_id, &park.goal).await {
                tracing::warn!(session_id = %session_id, error = %e, "parked turn could not resume");
            }
            return;
        }
    });
}

async fn provider_answers(
    router: &Arc<crate::llm_router::LlmRouter>,
    model: &str,
) -> ProviderProbe {
    let started = std::time::Instant::now();
    let mut probe = ProviderProbe {
        model: model.to_string(),
        answered: false,
        elapsed_secs: 0,
        error: None,
    };
    let Some(client) = router.client_for_model(model) else {
        probe.error = Some(format!("no provider serves model '{model}'"));
        return probe;
    };
    let request = nanna_llm::CompletionRequest {
        model: crate::llm_router::LlmRouter::strip_model_prefix(model),
        messages: vec![nanna_llm::Message::user("ok")],
        max_tokens: Some(1),
        ..Default::default()
    };
    let deadline = std::time::Duration::from_secs(nanna_llm::STREAM_READ_TIMEOUT_SECS);
    match tokio::time::timeout(deadline, client.complete(&request)).await {
        Ok(Ok(_)) => probe.answered = true,
        Ok(Err(error)) => probe.error = Some(error.to_string()),
        Err(_) => {
            probe.error = Some(format!(
                "no answer within the transport's {}s silence tolerance",
                nanna_llm::STREAM_READ_TIMEOUT_SECS
            ));
        }
    }
    probe.elapsed_secs = started.elapsed().as_secs();
    probe
}

/// Terminal give-up, or a PARK?
///
/// A mission whose budget ran out against a provider that is demonstrably
/// down has not failed — it is waiting, and calling that "gave up" is the one
/// give-up that is a lie. The evidence is the same classification the step
/// runner heals on ([`crate::tasks::is_transient_llm_error`]) plus the health
/// probe: either the recorded fault is a transient transport class, or the
/// probe found the provider unreachable. Anything else keeps today's terminal
/// give-up unchanged.
fn giveup_end(
    report: &nanna_agent::harness::LongHorizonReport,
    probe: Option<&ProviderProbe>,
    terminal: MissionEnd,
) -> MissionEnd {
    let recorded = stop_message(&report.stop)
        .map(str::to_string)
        .or_else(|| report.last_runner_error.clone())
        .unwrap_or_default();
    let transient_fault = crate::tasks::is_transient_llm_error(&recorded);
    let provider_down = probe.is_some_and(|p| !p.answered);
    if !transient_fault && !provider_down {
        return terminal;
    }
    let model = probe.map_or("the configured model", |p| p.model.as_str());
    let detail = probe
        .and_then(|p| p.error.clone())
        .or(Some(recorded))
        .filter(|d| !d.is_empty())
        .map(|d| format!(" ({d})"))
        .unwrap_or_default();
    MissionEnd::ParkedTransient(format!("{model} stopped answering{detail}"))
}

/// Put the top standing wall back on the board.
///
/// Returns the reopened item when one actually MOVED (it was closed and is
/// now pending). An item already open is not reopened — nothing changed, and
/// reporting a change that did not happen is how a loop stops terminating.
/// Reopening is by ID on purpose: re-seeding the title would be swallowed by
/// the closed-title dedup, which is exactly why the failing verdict went
/// unread in the first place.
async fn reopen_top_unmet(
    storage: &Arc<Storage>,
    unmet: &[nanna_agent::harness::AbandonedUnmet],
) -> Option<(i64, String)> {
    for item in unmet {
        let Ok(task) = storage.tasks().get(item.id).await else {
            continue;
        };
        if task.status != "done" && task.status != "cancelled" {
            continue;
        }
        match storage.tasks().reopen(item.id, Some("chat")).await {
            Ok(reopened) => return Some((reopened.id, reopened.title)),
            Err(message) => {
                tracing::warn!(
                    item = item.id,
                    %message,
                    "could not reopen a standing unmet item"
                );
            }
        }
    }
    None
}

/// Standing walls, rendered for a human: at most [`UNMET_SHOWN_MAX`] of them,
/// with the environment's own verdict.
const UNMET_SHOWN_MAX: usize = 5;

/// The unresolved evidence a stopped turn is still holding — the standing
/// walls and the environment's own verdict on each — with no "stopping —"
/// banner around it.
///
/// Split out of [`mission_end_notice`] because the CANCEL path needs exactly
/// this and none of the banner. A cancel was asked for and must never be
/// dressed up as a fault, so the sentence stays suppressed; but suppressing
/// the sentence used to suppress the evidence with it, and that evidence is
/// unrecoverable next turn — cancelled items are filtered out of every
/// context path, and these verdicts live only on the in-memory report. The
/// banner cannot simply be reused one branch higher: it is built around a
/// `why` string the cancel arm has no honest way to produce.
///
/// `stale_for_secs` dates the verdicts on the endings that did NOT re-measure
/// them. Every entry here was recorded by a drain sweep at some round's plan
/// exhaustion; an ending that drained ran its sweep AT the stop, so its
/// verdicts are current and carry no date. A cancel stops mid-plan with no
/// sweep at all, so the newest verdict on the report is at most as old as the
/// turn — which the liveness ledger already measures (`snapshot().elapsed_s`).
fn unresolved_evidence(
    report: &nanna_agent::harness::LongHorizonReport,
    stale_for_secs: Option<u64>,
) -> Option<String> {
    if report.abandoned_unmet.is_empty() && report.abandoned_unverifiable.is_empty() {
        return None;
    }
    let mut out = String::new();
    if !report.abandoned_unmet.is_empty() {
        out.push_str("\n\n_Still unmet:_");
        for item in report.abandoned_unmet.iter().take(UNMET_SHOWN_MAX) {
            let _ = write!(out, "\n_· #{} {}: {}_", item.id, item.title, item.detail);
        }
        if report.abandoned_unmet.len() > UNMET_SHOWN_MAX {
            let _ = write!(
                out,
                "\n_· …and {} more_",
                report.abandoned_unmet.len() - UNMET_SHOWN_MAX
            );
        }
        // Only the CHECKED half is a measurement, so only it can go stale.
        // The unchecked half below records what happened at the moment of
        // abandonment, which no later minute changes.
        if let Some(secs) = stale_for_secs {
            let ago = if secs >= 60 {
                format!("{}m", secs / 60)
            } else {
                format!("{secs}s")
            };
            let _ = write!(
                out,
                "\n_· last measured up to {ago} ago — the stop re-checked nothing._"
            );
        }
    }
    // The other half of what a stopped turn walked away from: items with no
    // done-condition to run at all. Disjoint from the list above by
    // construction, and the MAJORITY of abandonments — these used to leave a
    // count and no name, and in one observed session the item that vanished
    // that way was the root goal itself.
    if !report.abandoned_unverifiable.is_empty() {
        out.push_str("\n\n_Dropped, with no check to run:_");
        for item in report.abandoned_unverifiable.iter().take(UNMET_SHOWN_MAX) {
            // With no check, the item's last step result is the only evidence
            // that exists for why the harness stopped trying — clamped at the
            // width this file already uses for a stored verdict (see
            // `established_rows`), because it arrives at the harness's own
            // step-result bound and this is a chat message, not a log.
            let last = item
                .last_result
                .as_deref()
                .map(without_model_notices)
                .filter(|said| !said.is_empty())
                .map(|said| format!(" — last said: {}", clamp_display(&said, 200)))
                .unwrap_or_default();
            let _ = write!(
                out,
                "\n_· #{} {}: {}{}_",
                item.id,
                item.title,
                item.reason,
                last
            );
        }
        if report.abandoned_unverifiable.len() > UNMET_SHOWN_MAX {
            let _ = write!(
                out,
                "\n_· …and {} more_",
                report.abandoned_unverifiable.len() - UNMET_SHOWN_MAX
            );
        }
    }
    Some(out)
}

/// One sentence saying HOW a mission ended and on what evidence — the
/// user-visible sibling of [`failure_notice`].
///
/// `failure_notice` only fires on an error-shaped stop, so the endings that
/// most need explaining surfaced nothing: the planner-starvation give-up
/// keeps `report.stop == AllTasksDone`, and a dry ending that left failing
/// checks behind read as a clean finish. Both are the same product failure —
/// a run that stops and does not say why.
///
/// Deliberately quiet for the two endings that need no sentence:
/// - a CANCEL was asked for — but its unresolved evidence is still printed,
///   bannerless, by [`unresolved_evidence`]: the sentence is what the cancel
///   makes dishonest, not the failing checks it walked away from;
/// - a mission that converged with nothing failing, nothing unknown and
///   nothing abandoned has an honest ending already (the run-stats line), and
///   a "stopping:" banner on every ordinary chat turn that happened to read a
///   file would be noise, not honesty.
///
/// `turn_elapsed_s` is the liveness ledger's own measure of how long this turn
/// has been running, used only to date the cancel path's evidence — see
/// [`unresolved_evidence`].
fn mission_end_notice(
    report: &nanna_agent::harness::LongHorizonReport,
    end: &MissionEnd,
    probe: Option<&ProviderProbe>,
    turn_elapsed_s: u64,
) -> Option<String> {
    let unresolved = !report.abandoned_unmet.is_empty()
        || report.acceptance_unknown > 0
        || report.items_abandoned > 0;
    // A give-up always speaks. Anything else speaks only when it left
    // something unresolved behind — a converged mission is already described
    // by the run-stats line, and a "stopping:" banner on every ordinary chat
    // turn that happened to read a file would be noise, not honesty.
    if !end.gave_up() && !unresolved {
        return None;
    }
    let why = match end {
        MissionEnd::SingleRun => return None,
        // The evidence WITHOUT the sentence: see [`unresolved_evidence`].
        MissionEnd::Cancelled => return unresolved_evidence(report, Some(turn_elapsed_s)),
        MissionEnd::DryRoundsExhausted if !report.abandoned_unmet.is_empty() => {
            "re-planning found no new work, but the evidence below is still unmet".to_string()
        }
        MissionEnd::DryRoundsExhausted if !report.abandoned_unverifiable.is_empty() => {
            "re-planning found no new work, and the work below was dropped with nothing to \
             check it against"
                .to_string()
        }
        MissionEnd::DryRoundsExhausted => {
            // Same ending with no list to point at — the remaining way to end
            // dry-but-unresolved is a check that never returned a verdict,
            // which the count below names. Promising "the evidence below" and
            // then printing nothing is the shape observed live: a dry ending
            // that read "0 items verified done, 0 checks still failing" under
            // a sentence that had just announced evidence.
            "re-planning found no new work, and nothing it left behind proves the goal met"
                .to_string()
        }
        MissionEnd::RoundsMaxExhausted => format!(
            "the mission ran its full budget of {CONTINUATION_ROUNDS_MAX} planning rounds"
        ),
        MissionEnd::ErrorRoundsExhausted => {
            // Two different facts, and the ending must not conflate them.
            match probe {
                Some(p) if !p.answered => format!(
                    "the provider stopped answering — '{}' produced nothing in {}s",
                    p.model, p.elapsed_secs
                ),
                _ => format!(
                    "runs kept failing across {CONTINUATION_ERROR_ROUNDS} error rounds while \
                     the provider was answering"
                ),
            }
        }
        MissionEnd::PlannerStarvation => format!(
            "planning starved — the planner fell back and proposed nothing for \
             {CONTINUATION_ERROR_ROUNDS} rounds, so the goal is NOT verified complete"
        ),
        MissionEnd::ParkedTransient(reason) => format!(
            "{reason}. The work is PARKED, not abandoned: it resumes when that provider \
             answers again, or the moment you send another message"
        ),
        MissionEnd::DeliberateStop(kind) => format!("the run stopped: {kind}"),
    };

    let mut out = format!(
        "\n\n_stopping — {why}. {} item{} verified done, {} check{} still failing",
        report.items_completed,
        if report.items_completed == 1 { "" } else { "s" },
        report.abandoned_unmet.len(),
        if report.abandoned_unmet.len() == 1 { "" } else { "s" },
    );
    if report.acceptance_unknown > 0 {
        let _ = write!(
            out,
            ", {} check{} timed out without a verdict (UNKNOWN, not failed)",
            report.acceptance_unknown,
            if report.acceptance_unknown == 1 { "" } else { "s" }
        );
    }
    // The count is the LAST resort, and only when nothing below names the
    // work: both abandonment lists together cover every abandonment this
    // harness records, so this fires for a report that carries a count with
    // no names — an older run's report, whose list fields default to empty on
    // deserialize. Gated on both lists so it can never be read as a second
    // count of items already named below.
    if report.abandoned_unmet.is_empty()
        && report.abandoned_unverifiable.is_empty()
        && report.items_abandoned > 0
    {
        // Says only that they are not named HERE, never that they had no
        // check. Both lists are populated exclusively inside the drain sweep,
        // which runs on one stop reason — so a run that abandons a CHECKED
        // item and then stops on repeated runner errors, wall clock, token
        // budget or a source error arrives here with both lists empty and a
        // check that simply never ran. Asserting "no check" there would be the
        // same unverified claim this whole pass exists to remove.
        let _ = write!(
            out,
            ", {} item{} abandoned, not named here",
            report.items_abandoned,
            if report.items_abandoned == 1 { "" } else { "s" }
        );
    }
    // Verified work is on disk and stays there — say so, so a stopped
    // mission is never mistaken for a rolled-back one.
    out.push_str(". The work already verified stands — disk is truth._");
    // The endings that reach here drained their plan, so the sweep that
    // produced these verdicts ran at the stop: current, and undated.
    if let Some(evidence) = unresolved_evidence(report, None) {
        out.push_str(&evidence);
    }
    Some(out)
}

/// The user-visible line for a run that ended in failure, or `None` when the
/// ending needs no announcement (success, or a deliberate stop like Cancelled
/// or an exhausted budget — those were asked for).
///
/// Two failure shapes are announced, per "summaries must announce
/// themselves" — WHAT stopped and WHY, stated where the user is looking:
/// - an error stop (`RunnerErrors` / `SourceError`);
/// - `AllTasksDone` that completed nothing while abandoning something — the
///   poison-containment path, where a deterministic runner fault drains the
///   plan through abandonment and exits by the happy path.
///
/// Partial runs (some items completed, some abandoned) are NOT an error
/// banner; the run-stats line carries the abandoned count instead.
fn failure_notice(report: &nanna_agent::harness::LongHorizonReport) -> Option<String> {
    use nanna_agent::harness::StopReason;
    let why = match &report.stop {
        StopReason::RunnerErrors { message } => format!("the model kept failing: {message}"),
        StopReason::SourceError { message } => format!("the task store failed: {message}"),
        StopReason::AllTasksDone
            if report.items_completed == 0 && report.items_abandoned > 0 =>
        {
            report.last_runner_error.clone().unwrap_or_else(|| {
                "every planned task was abandoned without completing".to_string()
            })
        }
        _ => return None,
    };
    // Separate from streamed step text when there was any; a run that never
    // took a step starts its (only) message cleanly.
    let sep = if report.steps_taken > 0 { "\n\n" } else { "" };
    let verb = if report.steps_taken == 0 {
        "could not run"
    } else {
        "could not finish"
    };
    Some(format!("{sep}_{verb}: {why}_"))
}

/// Render this scope's still-open work so the PLANNER can decide what to do
/// with it — resume it, fold the new request into it, or leave it parked and
/// answer the question that was actually asked.
///
/// Deliberately framed as a decision for the model rather than a directive:
/// the store must not silently resume stale work just because it sorts first.
/// `None` when nothing is outstanding, so an ordinary turn carries no extra
/// prompt weight.
///
/// Bounded like every other injected context: at most
/// [`OPEN_WORK_MAX`] items, titles clamped, so a scope with a runaway plan
/// cannot crowd out the request itself.
/// Render this turn's verified outcomes for the continuation planner — the
/// knowledge mirror of the `UNMET WORK` block. Bounded like
/// [`open_work_context`]: enough to inform, never to displace the request.
fn established_block(verified: &[nanna_agent::harness::VerifiedOutcome]) -> Option<String> {
    if verified.is_empty() {
        return None;
    }
    let mut lines = vec![
        "ESTABLISHED — these done-conditions already PASS, verified by running them this \
         turn. Do NOT re-propose or re-assess this work; plan only what builds beyond it:"
            .to_string(),
    ];
    for v in verified.iter().rev().take(ESTABLISHED_MAX) {
        lines.push(format!("- #{} {}: {}", v.id, v.title, v.detail));
    }
    if verified.len() > ESTABLISHED_MAX {
        lines.push(format!("- …and {} more", verified.len() - ESTABLISHED_MAX));
    }
    Some(lines.join("\n"))
}

/// Bound for every block this module injects into the planner's context.
///
/// Same rationale wherever it is used: convey what is known without
/// displacing the request itself. The planner clamps its WHOLE context slot
/// to [`PLAN_GOAL_MAX_BYTES`] and the clamp cuts the tail, so an unbounded
/// block does not add information — it deletes whatever came after it.
/// Newest facts win the bounded slots and every cut announces itself.
const ESTABLISHED_MAX: usize = 10;

/// One closed item's verdict, read back from the store.
///
/// The rows are read ONCE per turn and consumed by every block that needs
/// them ([`established_work_context`], [`artifact_state_block`],
/// [`claim_conflicts`]) — three readers, one query set.
#[derive(Debug, Clone)]
pub(crate) struct EstablishedRow {
    pub id: i64,
    pub title: String,
    /// The completion verdict's text, when the completion was verified.
    pub verdict: Option<String>,
    /// RFC3339 completion time, empty when the store recorded none.
    pub when: String,
    /// The machine-checkable done-condition the item closed on, verbatim —
    /// the check a disputed claim has to be reconciled against.
    pub acceptance: Option<Value>,
}

/// What earlier turns in this scope PROVED done — closed items with their
/// completion verdicts, read back from the store at turn start.
///
/// Newest completions first: the freshest verdicts describe the current
/// artifact best. Bounded by [`ESTABLISHED_MAX`], with one bounded activity
/// query per shown row.
pub(crate) async fn established_rows(
    storage: &Arc<Storage>,
    scope: &str,
    scope_id: Option<&str>,
) -> Vec<EstablishedRow> {
    let Ok(all) = storage.tasks().list(scope, scope_id, true).await else {
        return Vec::new();
    };
    let mut done: Vec<_> = all.iter().filter(|t| t.status == "done").collect();
    if done.is_empty() {
        return Vec::new();
    }
    done.sort_by(|a, b| b.completed_at.cmp(&a.completed_at));
    let mut rows = Vec::with_capacity(done.len().min(ESTABLISHED_MAX));
    for task in done.iter().take(ESTABLISHED_MAX) {
        // The completion verdict lives in the task's activity log (action
        // "completed", detail {verified, verdict}). One bounded query per
        // shown task, at most ESTABLISHED_MAX per turn start.
        let verdict = storage.tasks().activity(task.id, 25).await.ok().and_then(|entries| {
            entries
                .iter()
                .rev()
                .find(|e| e.action == "completed")
                .and_then(|e| e.detail.as_ref())
                .and_then(|d| {
                    d.get("verdict")
                        .and_then(serde_json::Value::as_str)
                        .map(str::to_string)
                })
        });
        rows.push(EstablishedRow {
            id: task.id,
            title: clamp_display(&task.title, 120),
            verdict: verdict.map(|v| clamp_display(&v, 200)),
            when: task.completed_at.clone().unwrap_or_default(),
            acceptance: task.acceptance.clone(),
        });
    }
    rows
}

/// Notices the harness writes TO THE MODEL, by their opening bracket. They
/// ride tool results and step text so the model reads them; a user-facing
/// report quoting that text verbatim showed the user lines like
/// `frobnicate FAILED ([HARNESS NOTE — you are mid-task: …] [REPEAT-FAILURE
/// BREAKER …`.
const MODEL_FACING_NOTICE_OPENERS: &[&str] = &[
    "[HARNESS NOTE",
    "[REPEAT-FAILURE BREAKER",
    "[ZERO-INFORMATION BREAKER",
    "[CONTEXT NOTICE",
];

/// `text` cut before the first model-facing notice (see
/// [`MODEL_FACING_NOTICE_OPENERS`]), with the cut marked; empty when the
/// notice was all there was. Notices run to the end of the entry they are
/// attached to, so everything after the first one is harness text.
fn without_model_notices(text: &str) -> String {
    let Some(at) = MODEL_FACING_NOTICE_OPENERS
        .iter()
        .filter_map(|opener| text.find(opener))
        .min()
    else {
        return text.to_string();
    };
    let kept = text[..at].trim_end().trim_end_matches('(').trim_end();
    debug_assert!(
        MODEL_FACING_NOTICE_OPENERS
            .iter()
            .all(|o| !kept.contains(o)),
        "the cut is before every notice"
    );
    if kept.is_empty() {
        String::new()
    } else {
        format!("{kept}…")
    }
}

/// Clamp a stored string for display, announcing the cut.
fn clamp_display(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_string();
    }
    format!("{}…", &text[..text.floor_char_boundary(max)])
}

/// Render [`established_rows`] as the turn-start knowledge block. This is the
/// continuation context that makes a driver/user re-send after a
/// self-terminated run CONTINUE the mission instead of restarting it: the
/// verdicts name the commands that passed and when, i.e. the artifact state
/// the environment last confirmed.
///
/// Verified and unverified completions are rendered under separate headers.
/// An unverified item closed on the model's own word (`TASK COMPLETE`, or an
/// answer that converged) with no check run at all, so listing it under
/// "done-condition PASSING — do not redo or re-assess" told the planner a
/// check had confirmed something no check ever looked at — and told it not
/// to revisit exactly the work a follow-up like "that's wrong" is about.
fn established_work_context(rows: &[EstablishedRow]) -> Option<String> {
    let (verified, unverified): (Vec<&EstablishedRow>, Vec<&EstablishedRow>) =
        rows.iter().partition(|row| row.verdict.is_some());
    debug_assert_eq!(verified.len() + unverified.len(), rows.len());
    if rows.is_empty() {
        return None;
    }
    let mut out = String::new();
    if !verified.is_empty() {
        out.push_str(
            "## Verified done in earlier work this session\n\
             These closed with their done-condition PASSING (the verdict shows what the \
             environment confirmed, and when). Do not redo or re-assess them — continue \
             from this state:\n",
        );
        for row in &verified {
            let verdict = row.verdict.as_deref().unwrap_or_default();
            let _ = writeln!(
                out,
                "- #{} {} — verified {}: {verdict}",
                row.id, row.title, row.when
            );
        }
    }
    if !unverified.is_empty() {
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(
            "## Closed earlier on the model's own word (no check ran)\n\
             Nothing verified these; they are what was said, not what was confirmed. \
             Build on them, but re-assess one if the request questions it:\n",
        );
        for row in &unverified {
            let _ = writeln!(
                out,
                "- #{} {} — closed {} (unverified)",
                row.id, row.title, row.when
            );
        }
    }
    debug_assert!(!out.is_empty(), "a non-empty row set always renders");
    Some(out)
}

/// The workspace-local ledger every write tool ratchets against
/// (`write_file`, `edit_file`, `file_buffer`, `exec`). Its entries are
/// `{hi, last, at, good?, goodAt?, chk?}` keyed by the canonical
/// workspace-relative path — one file, one entry (P22 Tier 3).
const HIWATER_LEDGER: &str = ".nanna/write_hiwater.json";

/// ARTIFACT STATE — what exists on disk RIGHT NOW, re-read at turn start.
///
/// The failure this closes: a continuation turn's planner was told what had
/// been *done* (verdicts) but never what *existed*, so the model planned as
/// if the workspace were empty and rebuilt files it had already built —
/// losing the parts it did not remember. Prose about past work is not a
/// substitute for the artifact, and the model's own memory of what it wrote
/// is exactly the thing under suspicion.
///
/// Two halves, both ground truth, both rebuilt (never appended to) per turn:
/// 1. every file the write ratchet is tracking, with a FRESH `stat` (the
///    ledger records what a write left; only the stat says what is there
///    now) plus the ledger's own high-water and last structurally-good
///    sizes;
/// 2. the scope's verified verdicts — the same rows
///    [`established_work_context`] renders, so one store read serves both.
///
/// Bounds: [`ESTABLISHED_MAX`] entries, newest-written first, with the cut
/// announced. Every entry cost a real write and every verdict cost a real
/// execution, so the block's size is evidence-derived; the cap exists only
/// because the planner's context slot is itself clamped and an unbounded
/// block would silently delete the request.
///
/// Fails open at every step: no workspace, no ledger, unreadable ledger, or
/// a file that has since been deleted each degrade to less content, never to
/// an error.
pub(crate) async fn artifact_state_block(
    workspace_root: Option<&std::path::Path>,
    established: &[EstablishedRow],
) -> Option<String> {
    let mut files: Vec<String> = Vec::new();
    let mut tracked = 0usize;
    if let Some(root) = workspace_root {
        let raw = tokio::fs::read_to_string(root.join(HIWATER_LEDGER))
            .await
            .unwrap_or_default();
        if let Ok(Value::Object(map)) = serde_json::from_str::<Value>(&raw) {
            // Newest-written first: the freshest entries describe the work
            // this turn is continuing.
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            entries.sort_by(|a, b| {
                let at = |v: &Value| v.get("at").and_then(Value::as_i64).unwrap_or(0);
                at(b.1).cmp(&at(a.1))
            });
            tracked = entries.len();
            for (key, entry) in entries.iter().take(ESTABLISHED_MAX) {
                let path = root.join(key.as_str());
                let (size, modified) =
                    tokio::fs::metadata(&path).await.map_or((None, None), |meta| {
                        (
                            Some(meta.len()),
                            meta.modified().ok().map(|t| {
                                chrono::DateTime::<chrono::Utc>::from(t)
                                    .to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
                            }),
                        )
                    });
                let mut line = size.map_or_else(
                    // Tracked but gone: that IS the state, and saying it is
                    // how a rebuild becomes a decision instead of an accident.
                    || format!("- {key} — NOT PRESENT on disk right now"),
                    |bytes| format!("- {key} — {bytes} bytes on disk"),
                );
                if let Some(when) = modified {
                    let _ = write!(line, ", last modified {when}");
                }
                if let Some(hi) = entry.get("hi").and_then(Value::as_u64) {
                    let _ = write!(line, "; largest version this session {hi} bytes");
                }
                if let Some(good) = entry.get("good").and_then(Value::as_u64) {
                    let _ = write!(
                        line,
                        "; last version that passed a structural check {good} bytes"
                    );
                    if let Some(at) = entry.get("goodAt").and_then(Value::as_i64) {
                        let _ = write!(line, " (at {at})");
                    }
                }
                if let Some(chk) = entry.get("chk").and_then(Value::as_str) {
                    let _ = write!(line, "; latest check verdict: {chk}");
                }
                files.push(line);
            }
        }
    }

    let verified: Vec<String> = established
        .iter()
        .filter_map(|row| {
            row.verdict
                .as_ref()
                .map(|v| format!("- #{} {} — verified {}: {v}", row.id, row.title, row.when))
        })
        .collect();

    if files.is_empty() && verified.is_empty() {
        return None;
    }

    let mut out = String::from("## ARTIFACT STATE (re-read from disk at the start of this turn)\n");
    if !files.is_empty() {
        out.push_str(
            "These files hold verified work: read before writing; extend, do not reconstruct.\n",
        );
        out.push_str(&files.join("\n"));
        out.push('\n');
        if tracked > files.len() {
            let _ = writeln!(
                out,
                "- …and {} more tracked file(s) not shown",
                tracked - files.len()
            );
        }
    }
    if !verified.is_empty() {
        out.push_str(
            "These done-conditions were VERIFIED by execution — the outcome and when it was \
             last confirmed:\n",
        );
        out.push_str(&verified.join("\n"));
        out.push('\n');
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// Claim conflict: the user says it is broken, the environment said it passed
// ---------------------------------------------------------------------------

/// Phrases in which a user ASSERTS that something is failing.
///
/// Deliberately small and literal: this list only decides whether to LOOK for
/// a contradiction, and a message that does not assert failure changes
/// nothing at all. Substring matching on the lowercased message covers the
/// inflections ("fail" covers fails/failed/failing) without a stemmer.
const FAILURE_ASSERTIONS: &[&str] = &[
    "fail",
    "broken",
    "broke",
    "doesn't work",
    "does not work",
    "not working",
    "no longer works",
    "stopped working",
    "regress",
    "crash",
];

/// A user claim that CONTRADICTS a verdict the environment already rendered.
#[derive(Debug, Clone)]
pub(crate) struct ClaimConflict {
    pub task_id: i64,
    pub title: String,
    pub verdict: String,
    pub when: String,
    /// The identity in the user's message that named the disputed subject —
    /// a file or path the verified check itself referenced.
    pub subject: String,
    /// The disputed done-condition, so the reconciliation task re-runs the
    /// SAME check under the existing acceptance ceiling.
    pub acceptance: Option<Value>,
}

/// Keep only path-like identities: a token that names a FILE or a path.
///
/// Bare words are not identities — matching "test" or "build" against a
/// user's sentence would fire on half of English, and a false conflict costs
/// a real reproduction step. A dot or a slash is the cheap, conservative
/// evidence that a token names something on disk.
fn push_identity(out: &mut Vec<String>, raw: &str) {
    let keep = |c: char| c.is_alphanumeric() || matches!(c, '.' | '/' | '_' | '-');
    let token = raw.trim_matches(|c: char| !keep(c)).to_ascii_lowercase();
    if token.len() < 4 || !(token.contains('.') || token.contains('/')) {
        return;
    }
    if !token.chars().any(char::is_alphanumeric) {
        return;
    }
    if !out.contains(&token) {
        out.push(token);
    }
}

/// Every on-disk identity a closed item's evidence names: from its verdict
/// text, from the done-condition it closed on, and from its title.
fn identity_tokens(row: &EstablishedRow) -> Vec<String> {
    fn scan(text: &str, out: &mut Vec<String>) {
        let split = |c: char| {
            c.is_whitespace() || matches!(c, '`' | '"' | '\'' | '(' | ')' | ',' | ';' | ':' | '|')
        };
        for part in text.split(split) {
            push_identity(out, part);
        }
    }
    let mut out = Vec::new();
    if let Some(verdict) = &row.verdict {
        scan(verdict, &mut out);
    }
    if let Some(acceptance) = &row.acceptance {
        for field in ["command", "path"] {
            if let Some(text) = acceptance.get(field).and_then(Value::as_str) {
                scan(text, &mut out);
            }
        }
    }
    scan(&row.title, &mut out);
    out
}

/// Does this message assert the failure of something the environment already
/// verified passing?
///
/// Conservative by construction, in both directions: the message must assert
/// a failure AND name an identity the verified check itself referenced. No
/// match means today's behaviour, unchanged — this is not a filter on
/// ordinary chat, it is a detector for the one case where two pieces of
/// evidence disagree.
///
/// At most one conflict per SUBJECT: the same file named by two closed items
/// is one argument, not two, and the bound is therefore the evidence held —
/// no cap of its own.
fn claim_conflicts(message: &str, rows: &[EstablishedRow]) -> Vec<ClaimConflict> {
    let lower = message.to_ascii_lowercase();
    if !FAILURE_ASSERTIONS.iter().any(|a| lower.contains(a)) {
        return Vec::new();
    }
    let mut out: Vec<ClaimConflict> = Vec::new();
    for row in rows {
        let Some(verdict) = row.verdict.as_deref() else {
            continue;
        };
        let Some(subject) = identity_tokens(row)
            .into_iter()
            .find(|token| lower.contains(token.as_str()))
        else {
            continue;
        };
        if out.iter().any(|c| c.subject == subject) {
            continue;
        }
        out.push(ClaimConflict {
            task_id: row.id,
            title: row.title.clone(),
            verdict: verdict.to_string(),
            when: row.when.clone(),
            subject,
            acceptance: row.acceptance.clone(),
        });
    }
    out
}

/// The planner-facing rendering of a claim conflict: BOTH sides, named, with
/// neither presumed right.
fn claim_conflict_block(conflicts: &[ClaimConflict]) -> Option<String> {
    if conflicts.is_empty() {
        return None;
    }
    let mut out = String::from(
        "## CLAIM-CONFLICT — two pieces of evidence disagree\n\
         The message reports a failure of work this session VERIFIED passing. Neither side \
         is assumed right. Reproduce first, then reconcile: plan the reproduction BEFORE any \
         task that changes the subject, and do not rewrite the artifact on the strength of \
         whichever account was heard last.\n",
    );
    for conflict in conflicts {
        let _ = writeln!(
            out,
            "- `{}`: the message reports it failing. Verified PASSING {} by #{} {} — {}",
            conflict.subject,
            conflict.when,
            conflict.task_id,
            conflict.title,
            conflict.verdict
        );
    }
    Some(out)
}

/// Put one reproduction/reconciliation task at the HEAD of the plan for each
/// contradicted outcome.
///
/// Seeding is in plan order and sorts strictly below every existing item, so
/// the head really is first: whatever the planner proposed to change about
/// that subject now runs after the verdict that says which side's evidence
/// holds. The task carries the DISPUTED check as its own acceptance, so the
/// re-run happens through the shipped acceptance machinery and its existing
/// timeout ceiling — no new runner, no new bound.
fn prepend_reconciliation_tasks(
    plan: &mut nanna_agent::planner::Plan,
    conflicts: &[ClaimConflict],
) {
    if conflicts.is_empty() {
        return;
    }
    let mut head: Vec<nanna_agent::planner::PlannedTask> = Vec::with_capacity(conflicts.len());
    for conflict in conflicts {
        let description = format!(
            "The message reports that `{subject}` is FAILING. A check run in this session \
             recorded it PASSING: {verdict} (verified {when}, item #{id} \"{title}\").\n\n\
             Do NOT change `{subject}` or anything it depends on in this task. This task's \
             only product is a VERDICT about which piece of evidence holds, and WHY they \
             differ.\n\n\
             1. Re-run the disputed check exactly as recorded, in a fresh scratch directory \
             under the workspace — follow the described steps and order when any were given, \
             otherwise the check's own command.\n\
             2. Note the provenance of each file the check references that resolves on disk: \
             its current size and modification time, whether it changed since {when}, and \
             (when a `.__prev__` copy sits beside it) how it differs. Note the working \
             directory and any environment the check depends on.\n\
             3. Say plainly which side reproduced. If the failure reproduces, that is the \
             standing verdict and the fix proceeds from it. If it does NOT reproduce, report \
             BOTH pieces of evidence and ask ONE clarifying question — rewriting an artifact \
             on an unreproduced report is how verified work gets destroyed.",
            subject = conflict.subject,
            verdict = conflict.verdict,
            when = conflict.when,
            id = conflict.task_id,
            title = conflict.title,
        );
        head.push(nanna_agent::planner::PlannedTask {
            title: format!(
                "Reproduce the reported failure of {} before changing it",
                conflict.subject
            ),
            description: Some(nanna_agent::planner::clamp_bytes(
                &description,
                PLAN_DESCRIPTION_MAX_BYTES,
            )),
            acceptance: conflict.acceptance.clone(),
            tool_scope: Vec::new(),
        });
    }
    head.append(&mut plan.tasks);
    plan.tasks = head;
}

/// The run's verified state, as one order-stable value: which checks are
/// standing walls and what every verdict currently says.
///
/// The reseed rung arms once per DISTINCT value. If a reseed produces no
/// change in what the environment says, the next dry terminal ends the run
/// exactly as today — which is what keeps "do not end dry while your checks
/// still fail" terminating rather than looping.
fn verified_state_fingerprint(report: &nanna_agent::harness::LongHorizonReport) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::hash::DefaultHasher::new();
    let mut unmet: Vec<(i64, &str)> = report
        .abandoned_unmet
        .iter()
        .map(|u| (u.id, u.detail.as_str()))
        .collect();
    unmet.sort_unstable();
    unmet.as_slice().hash(&mut hasher);
    let mut verified: Vec<(i64, &str)> = report
        .verified_outcomes
        .iter()
        .map(|v| (v.id, v.detail.as_str()))
        .collect();
    verified.sort_unstable();
    verified.as_slice().hash(&mut hasher);
    hasher.finish()
}

/// A step runner with the SAME configuration and RESET run-scoped state.
///
/// What resets is what went stale: the byte-identity breaker ledger (which
/// had been short-circuiting the run's own re-reads of its own artifact, so
/// the model could no longer see what it had built) and tool discovery. What
/// carries is what is still true: the provider, the tools, the prompt, the
/// workspace, the transcript sink, the memory sink, and the GPU fault tally —
/// a hardware fault is a fact about the machine, not about the context.
fn fresh_step_runner(previous: &AgentStepRunner) -> AgentStepRunner {
    AgentStepRunner {
        discovered_tools: Arc::new(tokio::sync::RwLock::new(
            std::collections::HashSet::new(),
        )),
        // The user's picks are a fact about the SESSION, not run-scoped
        // state: a runner reset discards stale discovery, never the
        // selection the user made.
        user_selected_tools: previous.user_selected_tools.clone(),
        repeat_ledger: Arc::new(nanna_agent::RepeatLedger::new()),
        router: previous.router.clone(),
        tools: previous.tools.clone(),
        agent_config: previous.agent_config.clone(),
        system_prompt: previous.system_prompt.clone(),
        workspace_root: previous.workspace_root.clone(),
        workspace_context: previous.workspace_context.clone(),
        stats: previous.stats.clone(),
        chat_sink: previous.chat_sink.clone(),
        memory: previous.memory.clone(),
        workspace_id: previous.workspace_id.clone(),
        gpu_fault_count: previous.gpu_fault_count.clone(),
        degradations: previous.degradations.clone(),
        attachments: Arc::clone(&previous.attachments),
        model_chain: previous.model_chain.clone(),
    }
}

async fn open_work_context(
    storage: &Arc<Storage>,
    scope: &str,
    scope_id: Option<&str>,
) -> Option<String> {
    /// Enough to convey what is parked without displacing the request. A
    /// model that needs the full list can read it with the todo tool.
    const OPEN_WORK_MAX: usize = 10;

    let open = storage.tasks().list(scope, scope_id, false).await.ok()?;
    if open.is_empty() {
        return None;
    }
    let total = open.len();
    let mut out = String::from(
        "## Unfinished work from earlier in this session\n\
         These tasks are still open. Decide for yourself whether the user's new message means \
         to continue them, to change them, or to set them aside and answer something else — \
         do NOT assume a resume was requested:\n",
    );
    for task in open.iter().take(OPEN_WORK_MAX) {
        let title = if task.title.len() > 120 {
            let end = task.title.floor_char_boundary(120);
            format!("{}…", &task.title[..end])
        } else {
            task.title.clone()
        };
        let _ = writeln!(out, "- #{} [{}] {}", task.id, task.status, title);
    }
    if total > OPEN_WORK_MAX {
        let _ = writeln!(
            out,
            "- …and {} more (use the todo tool to see them all)",
            total - OPEN_WORK_MAX
        );
    }
    Some(out)
}

/// Render recent conversation as role-tagged lines for prompt injection.
///
/// The harness re-anchors every step from the task store, so unlike the
/// retired direct path (which passed history as a message array) the
/// conversation must ride in the system prompt — without it, "double it"
/// after "what is 2+2?" plans against nothing.
///
/// Bounds — both derived from the planner's own limits so every consumer of
/// this rendering sees the same window:
/// - total ≤ [`PLAN_GOAL_MAX_BYTES`] (8 KiB): the planner clamps its context
///   slot to this constant, and anything larger would displace the step's
///   working context;
/// - each message ≤ [`PLAN_DESCRIPTION_MAX_BYTES`] (2 KiB): one giant paste
///   must not occupy the whole window — this guarantees at least the last
///   four turns always fit.
///
/// Newest messages win; a dropped prefix and clamped messages announce
/// themselves so a partial view is never mistaken for the whole.
pub(super) fn conversation_context(messages: &[SessionMessage]) -> Option<String> {
    const OMISSION_NOTE: &str = "[earlier conversation omitted]";

    let mut lines: Vec<String> = Vec::new();
    let mut used = 0usize;
    let mut truncated = false;

    for message in messages.iter().rev() {
        let speaker = match message.role {
            MessageRole::User => "User",
            MessageRole::Assistant => "Nanna",
            // System/tool records are plumbing, not conversation.
            MessageRole::System | MessageRole::Tool => continue,
        };
        let text = message.content.trim();
        if text.is_empty() {
            continue;
        }

        let clamped = if text.len() > PLAN_DESCRIPTION_MAX_BYTES {
            let end = text.floor_char_boundary(PLAN_DESCRIPTION_MAX_BYTES);
            format!("{}… [message truncated]", &text[..end])
        } else {
            text.to_string()
        };
        let line = format!("{speaker}: {clamped}");

        if used + line.len() + 1 > PLAN_GOAL_MAX_BYTES {
            truncated = true;
            break;
        }
        used += line.len() + 1;
        lines.push(line);
    }

    if lines.is_empty() {
        return None;
    }
    if truncated {
        lines.push(OMISSION_NOTE.to_string());
    }
    lines.reverse();
    Some(lines.join("\n"))
}

/// The claim marker `nanna_agent::harness::step_claims_completion` verdicts
/// on. The model emits it, so matching is case-insensitive.
const CLAIM_MARKER: &str = "TASK COMPLETE";

/// Loop-recovery steering markers the agent loop may inject into a turn.
/// Steering is harness-to-model, never conversation: everything from the
/// marker to the end of its line is harness-fabricated, so the line is cut at
/// the marker. Emitted by our own code, so matching is exact.
const NUDGE_MARKERS: &[&str] = &["[THINKING SPIRAL DETECTED]"];

/// Case-insensitive `strip_prefix` (ASCII markers only). `get` guards the
/// UTF-8 boundary a byte-length slice could split.
fn strip_prefix_ci<'a>(s: &'a str, prefix: &str) -> Option<&'a str> {
    let head = s.get(..prefix.len())?;
    head.eq_ignore_ascii_case(prefix).then(|| &s[prefix.len()..])
}

/// Case-insensitive `strip_suffix` (ASCII markers only).
fn strip_suffix_ci<'a>(s: &'a str, suffix: &str) -> Option<&'a str> {
    let split = s.len().checked_sub(suffix.len())?;
    let tail = s.get(split..)?;
    tail.eq_ignore_ascii_case(suffix).then(|| &s[..split])
}

/// Per-line strip: `None` drops the line, `Some` keeps (possibly peeled)
/// text. Recursion terminates because every recursive call shrinks the line
/// by at least one marker length.
fn strip_marker_line(line: &str) -> Option<&str> {
    for marker in NUDGE_MARKERS {
        if let Some(pos) = line.find(marker) {
            let head = line[..pos].trim_end();
            return if head.is_empty() { None } else { strip_marker_line(head) };
        }
    }
    let trimmed = line.trim();
    if trimmed.eq_ignore_ascii_case(CLAIM_MARKER) {
        return None;
    }
    // Stream glue: the claim marker fused with a neighboring line when the
    // newline between them was lost (observed live 2026-08-02: "TASK
    // COMPLETE[THINKING SPIRAL DETECTED]..."). Glue never introduces spaces,
    // so a whitespace boundary reads as genuine prose and survives.
    if let Some(rest) = strip_prefix_ci(trimmed, CLAIM_MARKER)
        && !rest.starts_with(char::is_whitespace)
    {
        return strip_marker_line(rest);
    }
    if let Some(rest) = strip_suffix_ci(trimmed, CLAIM_MARKER)
        && !rest.ends_with(char::is_whitespace)
    {
        return strip_marker_line(rest);
    }
    Some(line)
}

/// Remove harness plumbing from user-visible text: the `TASK COMPLETE`
/// claim marker the harness verdicts on (a line matching
/// `nanna_agent::harness::step_claims_completion`'s predicate — trimmed,
/// case-insensitive, on its own line, plus the glued prefix/suffix shapes
/// stream fusion produces) and loop-steering nudge markers. Space-separated
/// inline mentions are left alone.
pub(super) fn strip_harness_markers(text: &str) -> String {
    let mut out: Vec<&str> = text.lines().filter_map(strip_marker_line).collect();
    // Marker lines at the end often leave a dangling blank line behind them.
    while out.last().is_some_and(|line| line.trim().is_empty()) {
        out.pop();
    }
    out.join("\n")
}

/// Apply [`strip_harness_markers`] to the journal's text entries and drop
/// every prose entry left with nothing to show. Tool, fault and step entries
/// pass through untouched — they are records and run mechanics, not prose.
pub(super) fn sanitize_timeline(items: Vec<TimelineItem>) -> Vec<TimelineItem> {
    items
        .into_iter()
        .filter_map(|item| match item {
            TimelineItem::Text { content, at } => {
                let stripped = strip_harness_markers(&content);
                if stripped.trim().is_empty() {
                    None
                } else {
                    Some(TimelineItem::Text {
                        content: stripped,
                        at,
                    })
                }
            }
            // A burst that closed with nothing in it — a lone newline delta
            // between two tool calls opens and closes its own segment —
            // rendered as its own "💭 Thinking · 1 words" card (observed
            // live 2026-08-03). Markers are NOT stripped from thinking: it
            // is the model's own record, and emptiness is the only thing
            // that makes the card meaningless.
            TimelineItem::Thinking { content, at } => {
                if content.trim().is_empty() {
                    None
                } else {
                    Some(TimelineItem::Thinking { content, at })
                }
            }
            other => Some(other),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use nanna_agent::harness::{LongHorizonReport, StopReason};

    /// P22 delivery-ack contract: both admission shapes certify delivery
    /// explicitly and are distinguishable from any run-completion payload.
    /// A client that sees `delivery: "accepted"` knows the message is
    /// persisted and owned by a run — anything else is "not delivered".
    #[test]
    fn started_ack_certifies_delivery_not_completion() {
        let ack = started_response("msg-1");
        assert_eq!(ack["status"], "started");
        assert_eq!(ack["message_id"], "msg-1");
        assert_eq!(ack["delivery"], "accepted");
        assert!(ack["accepted_at"].as_str().is_some_and(|t| !t.is_empty()));
        // Empty content: the transcript is driven by events, never by this ack.
        assert_eq!(ack["content"], "");
    }

    /// A prep for a chat with no pin, so the "unpinned is unchanged" assertion
    /// below is written against the same shape the real turn destructures.
    fn prep(chat_model: Option<&str>) -> super::super::chat::ChatTurnPrep {
        super::super::chat::ChatTurnPrep {
            system_prompt: "you are nanna".to_string(),
            conversation: None,
            workspace_root: None,
            workspace_context: None,
            chat_model: chat_model.map(str::to_string),
            chat_tools: Vec::new(),
        }
    }

    /// A realistic global default: a primary model, a populated routing tier
    /// table, and the summarization/provider settings a pin must not touch.
    fn global_default() -> AgentConfig {
        crate::agent_service::agent_config_from(&crate::agent_service::AgentServiceConfig {
            model: "claude-sonnet-4".to_string(),
            model_priority: vec!["claude-sonnet-4".to_string()],
            model_routing: vec![
                "claude-haiku-3-5:simple".to_string(),
                "claude-sonnet-4:complex".to_string(),
            ],
            summarization_priority: vec!["ollama/lfm2.5:latest".to_string()],
            ..Default::default()
        })
    }

    /// THE wiring assertion. Before the turn read `chat_model` off the prep,
    /// the pin was stored, shown in the GUI and shipped over IPC while every
    /// turn still ran on the global model — the feature was inert. A prep
    /// carrying a pin must produce a turn config pointed at that model.
    #[test]
    fn a_prep_carrying_a_pin_produces_a_turn_on_the_pinned_model() {
        let config = turn_agent_config(global_default(), prep(Some("ollama/qwen3:14b")).chat_model);

        assert_eq!(config.model, "ollama/qwen3:14b");
        // The loop re-picks a model every iteration after the first while the
        // tier table is populated, so a pin that left it in place would run
        // steps 2..n on the global models anyway — the inert bug with extra
        // steps.
        assert!(
            config.model_routing.is_empty(),
            "a pinned chat must not be re-routed off its model mid-run"
        );
        // The pin names the CHAT model only.
        assert_eq!(
            config.summarization_priority,
            vec!["ollama/lfm2.5:latest".to_string()]
        );
    }

    /// The other half of the contract: a chat that never picked a model must
    /// behave EXACTLY as it did before per-chat models existed. Compared
    /// through `Debug` because `AgentConfig` is not `PartialEq` — the point is
    /// that every field is identical, not just the two a pin moves, so a
    /// whole-value comparison is what is wanted here.
    #[test]
    fn a_prep_with_no_pin_is_the_global_default_untouched() {
        let before = format!("{:?}", global_default());
        let after = format!("{:?}", turn_agent_config(global_default(), prep(None).chat_model));

        assert_eq!(after, before, "an unpinned chat must not be reshaped at all");
    }

    /// The planner is built from `step_runner.agent_config.clone()`, so it
    /// inherits the pin by construction. Asserted here because planning and
    /// stepping disagreeing about the model is the failure that would make a
    /// pinned chat plan on one model and work on another, invisibly.
    #[test]
    fn the_planner_inherits_the_pinned_model_from_the_step_runner() {
        let step = turn_agent_config(global_default(), prep(Some("ollama/qwen3:14b")).chat_model);
        let planner = step.clone();

        assert_eq!(planner.model, step.model);
        assert!(planner.model_routing.is_empty());
    }

    #[test]
    fn interjected_ack_carries_the_same_delivery_contract() {
        let ack = interjected_response("session-9", 3);
        assert_eq!(ack["status"], "interjected");
        assert_eq!(ack["pending"], 3);
        assert_eq!(ack["delivery"], "accepted");
        assert!(ack["accepted_at"].as_str().is_some_and(|t| !t.is_empty()));
    }

    /// `stop_kind` reads the enum's own serde tag, so the liveness ledger's
    /// spelling tracks the enum by construction.
    #[test]
    fn stop_kind_matches_the_serde_tag() {
        assert_eq!(stop_kind(&StopReason::AllTasksDone), "all_tasks_done");
        assert_eq!(stop_kind(&StopReason::Cancelled), "cancelled");
        assert_eq!(
            stop_kind(&StopReason::RunnerErrors { message: "x".into() }),
            "runner_errors"
        );
        assert_eq!(
            stop_kind(&StopReason::SourceError { message: "x".into() }),
            "source_error"
        );
        assert_eq!(stop_kind(&StopReason::WallClockExhausted), "wall_clock_exhausted");
        assert_eq!(stop_kind(&StopReason::TokenBudgetExhausted), "token_budget_exhausted");
    }

    fn msg(role: MessageRole, content: &str) -> SessionMessage {
        SessionMessage {
            id: uuid::Uuid::new_v4().to_string(),
            role,
            content: content.to_string(),
            timestamp: Utc::now(),
            tool_calls: Vec::new(),
            attachments: Vec::new(),
            reasoning: None,
            timeline: Vec::new(),
            usage: None,
        }
    }

    fn report(
        stop: StopReason,
        steps: usize,
        completed: usize,
        abandoned: usize,
        last_err: Option<&str>,
    ) -> LongHorizonReport {
        LongHorizonReport {
            verified_outcomes: Vec::new(),
            acceptance_unknown: 0,
            stop,
            steps_taken: steps,
            tool_calls: 0,
            side_effect_tool_calls: 0,
            items_completed: completed,
            items_completed_unverified: 0,
            items_revived: 0,
            abandoned_unmet: Vec::new(),
            abandoned_unverifiable: Vec::new(),
            items_regressed_reopened: 0,
            items_already_satisfied: 0,
            items_abandoned: abandoned,
            last_runner_error: last_err.map(String::from),
            replans: 0,
            false_success_claims: 0,
            input_tokens: 0,
            output_tokens: 0,
            wall_clock_secs: 0,
            tokens_per_completed_item: None,
            interjected_items: 0,
        }
    }

    fn unmet(id: i64, detail: &str) -> nanna_agent::harness::AbandonedUnmet {
        nanna_agent::harness::AbandonedUnmet {
            id,
            title: format!("feature {id}"),
            detail: detail.to_string(),
        }
    }

    fn dropped(id: i64, reason: &str, last: Option<&str>) -> nanna_agent::harness::AbandonedUnverifiable {
        nanna_agent::harness::AbandonedUnverifiable {
            id,
            title: format!("feature {id}"),
            reason: reason.to_string(),
            last_result: last.map(str::to_string),
        }
    }

    // ------------------------------------------------------------------
    // Exit honesty: one cause, one line, one sentence
    // ------------------------------------------------------------------

    /// Every exit cause has a distinct, greppable wire spelling, and the
    /// underlying stop stays visible on the deliberate one.
    #[test]
    fn mission_end_causes_are_distinct_and_greppable() {
        let causes = [
            MissionEnd::SingleRun.cause(),
            MissionEnd::Cancelled.cause(),
            MissionEnd::DryRoundsExhausted.cause(),
            MissionEnd::RoundsMaxExhausted.cause(),
            MissionEnd::ErrorRoundsExhausted.cause(),
            MissionEnd::PlannerStarvation.cause(),
            MissionEnd::ParkedTransient("ollama down".into()).cause(),
            MissionEnd::DeliberateStop("wall_clock_exhausted".into()).cause(),
        ];
        let unique: std::collections::HashSet<&String> = causes.iter().collect();
        assert_eq!(unique.len(), causes.len(), "{causes:?}");
        assert_eq!(causes[7], "deliberate_stop:wall_clock_exhausted");
        // A give-up is never silently indistinguishable from a finish.
        assert!(MissionEnd::PlannerStarvation.gave_up());
        assert!(MissionEnd::ErrorRoundsExhausted.gave_up());
        assert!(!MissionEnd::SingleRun.gave_up());
        assert!(!MissionEnd::DryRoundsExhausted.gave_up());
    }

    /// The planner-starvation ending is the one `failure_notice` cannot see:
    /// `report.stop` still reads `AllTasksDone`, so today it surfaced nothing
    /// at all. It must now say the goal is NOT complete, name the standing
    /// walls, and keep the verified work standing.
    #[test]
    fn planner_starvation_announces_itself_with_its_evidence() {
        let mut r = report(StopReason::AllTasksDone, 12, 4, 2, None);
        r.abandoned_unmet = vec![unmet(7, "`sh tests/test_07.sh` exited 1")];
        r.acceptance_unknown = 2;
        assert!(
            failure_notice(&r).is_none(),
            "this is exactly the ending the failure notice cannot see"
        );
        let notice = mission_end_notice(&r, &MissionEnd::PlannerStarvation, None, 0)
            .expect("a starved mission must say so");
        assert!(notice.contains("planning starved"), "{notice}");
        assert!(notice.contains("NOT verified complete"), "{notice}");
        assert!(notice.contains("4 items verified done"), "{notice}");
        assert!(notice.contains("1 check still failing"), "{notice}");
        assert!(notice.contains("2 checks timed out"), "{notice}");
        assert!(notice.contains("disk is truth"), "{notice}");
        assert!(notice.contains("#7 feature 7: `sh tests/test_07.sh` exited 1"), "{notice}");
    }

    /// Chat generality: an ordinary turn that happened to call a tool and then
    /// converged must NOT grow a "stopping:" banner. The sentence is for
    /// endings that left something unresolved or gave up.
    #[test]
    fn a_clean_ending_stays_a_plain_reply() {
        let clean = report(StopReason::AllTasksDone, 2, 1, 0, None);
        assert!(mission_end_notice(&clean, &MissionEnd::SingleRun, None, 0).is_none());
        assert!(mission_end_notice(&clean, &MissionEnd::DryRoundsExhausted, None, 0).is_none());
        // A cancel was asked for — never dressed up as a fault, and a cancel
        // with nothing unresolved to show says nothing at all.
        let cancelled = report(StopReason::Cancelled, 3, 0, 1, None);
        assert!(mission_end_notice(&cancelled, &MissionEnd::Cancelled, None, 0).is_none());
        // …but a dry ending that walked away from a failing check speaks up.
        let mut dry = report(StopReason::AllTasksDone, 5, 2, 1, None);
        dry.abandoned_unmet = vec![unmet(3, "`cargo test` exited 101")];
        let notice = mission_end_notice(&dry, &MissionEnd::DryRoundsExhausted, None, 0)
            .expect("failing checks make an ending loud");
        assert!(notice.contains("still unmet") || notice.contains("still failing"), "{notice}");
    }

    /// A cancel suppresses the SENTENCE, not the evidence. The failing checks
    /// a stopped turn walked away from live only on the in-memory report —
    /// cancelled items are filtered out of every context path, so anything
    /// not printed here is gone. It carries the turn's age, because a cancel
    /// stops mid-plan and re-measures nothing.
    #[test]
    fn a_cancel_prints_its_unmet_evidence_without_a_banner() {
        let mut cancelled = report(StopReason::Cancelled, 3, 0, 1, None);
        cancelled.abandoned_unmet = vec![unmet(1, "`cargo test` exited 101")];
        let notice = mission_end_notice(&cancelled, &MissionEnd::Cancelled, None, 4_500)
            .expect("a cancel that walked away from a failing check must still show it");
        assert!(
            !notice.contains("stopping —"),
            "a cancel was asked for and is never dressed up as a fault: {notice}"
        );
        assert!(notice.contains("#1 feature 1: `cargo test` exited 101"), "{notice}");
        assert!(notice.contains("75m ago"), "a cancel's verdict is dated: {notice}");
        assert!(notice.contains("re-checked nothing"), "{notice}");

        // …and a cancel that only dropped unchecked work still names it: the
        // record is what happened at the abandonment, so it is not dated.
        let mut cancelled = report(StopReason::Cancelled, 3, 0, 1, None);
        cancelled.abandoned_unverifiable = vec![dropped(6, "the user stopped the run", None)];
        let notice = mission_end_notice(&cancelled, &MissionEnd::Cancelled, None, 4_500)
            .expect("dropped work is named on a cancel too");
        assert!(!notice.contains("stopping —"), "{notice}");
        assert!(notice.contains("#6 feature 6: the user stopped the run"), "{notice}");
        assert!(!notice.contains("ago"), "an abandonment record is not a measurement: {notice}");

        // The endings that drained ran their sweep AT the stop, so their
        // evidence is current and must NOT be dated.
        let mut drained = report(StopReason::AllTasksDone, 9, 1, 1, None);
        drained.abandoned_unmet = vec![unmet(1, "`cargo test` exited 101")];
        let notice = mission_end_notice(&drained, &MissionEnd::PlannerStarvation, None, 4_500)
            .expect("a starved mission announces itself");
        assert!(!notice.contains("ago"), "a fresh sweep needs no date: {notice}");
    }

    /// The dry ending must never promise evidence it has none of. Observed
    /// live: "re-planning found no new work, but the evidence below is still
    /// unmet. 0 items verified done, 0 checks still failing" — rendered with
    /// no list under it, because the work it walked away from carried no
    /// machine-checkable done-condition at all.
    #[test]
    fn a_dry_ending_with_no_list_promises_none_and_names_what_it_dropped() {
        // The majority path: abandoned work with no done-condition. It is
        // NAMED, and the sentence points at the list that actually prints.
        let mut unchecked = report(StopReason::AllTasksDone, 4, 0, 2, None);
        unchecked.abandoned_unverifiable = vec![
            dropped(11, "no progress after 3 steps", Some("still trying to open the file")),
            dropped(12, "the run ended first", None),
        ];
        let notice = mission_end_notice(&unchecked, &MissionEnd::DryRoundsExhausted, None, 0)
            .expect("work abandoned without a check still ends the turn unresolved");
        assert!(
            !notice.contains("evidence below"),
            "there is no unmet check below — that promise is the defect: {notice}"
        );
        assert!(notice.contains("nothing to check it against"), "{notice}");
        assert!(notice.contains("#11 feature 11: no progress after 3 steps"), "{notice}");
        assert!(notice.contains("last said: still trying to open the file"), "{notice}");
        assert!(notice.contains("#12 feature 12: the run ended first"), "{notice}");
        assert!(!notice.contains("Still unmet"), "{notice}");
        // Named, so never re-reported as a bare count.
        assert!(!notice.contains("abandoned, not named here"), "{notice}");

        // The bare-count fallback: a report that carries the count and no
        // names at all (an older run's, whose list fields default empty).
        let counted = report(StopReason::AllTasksDone, 4, 0, 2, None);
        assert!(counted.abandoned_unmet.is_empty());
        let notice = mission_end_notice(&counted, &MissionEnd::DryRoundsExhausted, None, 0)
            .expect("a count with no names is still unresolved");
        assert!(!notice.contains("evidence below"), "{notice}");
        assert!(notice.contains("nothing it left behind proves the goal met"), "{notice}");
        assert!(notice.contains("2 items abandoned, not named here"), "{notice}");
        assert!(!notice.contains("Still unmet"), "{notice}");

        // …and the dry ending that DOES have a list keeps its promise.
        let mut listed = report(StopReason::AllTasksDone, 4, 0, 1, None);
        listed.abandoned_unmet = vec![unmet(2, "`sh check.sh` exited 1")];
        let notice = mission_end_notice(&listed, &MissionEnd::DryRoundsExhausted, None, 0)
            .expect("a failing check makes the ending loud");
        assert!(notice.contains("evidence below is still unmet"), "{notice}");
        assert!(notice.contains("#2 feature 2: `sh check.sh` exited 1"), "{notice}");
        // One count, not two: the listed item is not re-counted as unnamed.
        assert!(!notice.contains("abandoned with no check"), "{notice}");
    }

    /// "The provider is unreachable" and "the run keeps failing" are different
    /// facts, and the ending must not conflate them.
    #[test]
    fn error_round_ending_distinguishes_outage_from_a_failing_run() {
        let r = report(
            StopReason::RunnerErrors { message: "API error: 500".into() },
            3,
            0,
            1,
            Some("API error: 500"),
        );
        let down = ProviderProbe {
            model: "ollama/qwen3.5:9b".to_string(),
            answered: false,
            elapsed_secs: 120,
            error: Some("no answer".to_string()),
        };
        let notice = mission_end_notice(&r, &MissionEnd::ErrorRoundsExhausted, Some(&down), 0)
            .expect("an exhausted error budget announces itself");
        assert!(notice.contains("provider stopped answering"), "{notice}");
        assert!(notice.contains("ollama/qwen3.5:9b"), "{notice}");

        let up = ProviderProbe { answered: true, ..down };
        let notice = mission_end_notice(&r, &MissionEnd::ErrorRoundsExhausted, Some(&up), 0)
            .expect("still announces itself");
        assert!(notice.contains("while the provider was answering"), "{notice}");
    }

    /// A transient fault is a PARK, not a give-up; a deterministic one keeps
    /// today's terminal ending.
    #[test]
    fn transient_giveups_park_and_hard_ones_do_not() {
        let transient = report(
            StopReason::RunnerErrors { message: "API error: 502 bad gateway".into() },
            2,
            0,
            1,
            None,
        );
        let end = giveup_end(&transient, None, MissionEnd::ErrorRoundsExhausted);
        assert!(matches!(end, MissionEnd::ParkedTransient(_)), "{end:?}");
        let notice = mission_end_notice(&transient, &end, None, 0).expect("a park announces itself");
        assert!(notice.contains("PARKED, not abandoned"), "{notice}");
        assert!(notice.contains("resumes when that provider answers"), "{notice}");

        let hard = report(
            StopReason::RunnerErrors { message: "API error: 400 - context length exceeded".into() },
            2,
            0,
            1,
            None,
        );
        assert_eq!(
            giveup_end(&hard, None, MissionEnd::ErrorRoundsExhausted),
            MissionEnd::ErrorRoundsExhausted,
            "a deterministic fault is not a park"
        );

        // An unreachable provider parks even when the recorded fault is not
        // itself transport-shaped: the probe is the newer evidence.
        let probe = ProviderProbe {
            model: "ollama/qwen3.5:9b".to_string(),
            answered: false,
            elapsed_secs: 120,
            error: Some("no answer within the transport's 120s silence tolerance".to_string()),
        };
        assert!(matches!(
            giveup_end(&hard, Some(&probe), MissionEnd::ErrorRoundsExhausted),
            MissionEnd::ParkedTransient(_)
        ));
    }

    /// A crashed run proves nothing about whether a mission exists. Store
    /// evidence — open items nobody finished — grants the error rounds; an
    /// empty store forfeits them.
    #[tokio::test]
    async fn error_rounds_follow_session_state_not_just_run_evidence() {
        let storage = Arc::new(Storage::in_memory().await.expect("storage"));
        let open_items = |storage: Arc<Storage>| async move {
            storage
                .tasks()
                .counts("session", Some("s1"))
                .await
                .map_or(0, |(open, _closed)| open)
        };
        // RunnerErrors with steps=0, tool_calls=0 and an empty store: no
        // mission, no rounds.
        let run_evidence = false;
        assert_eq!(open_items(storage.clone()).await, 0);
        assert!(
            !(run_evidence || open_items(storage.clone()).await > 0),
            "a run that seeded nothing still gets zero rounds"
        );

        storage
            .tasks()
            .create(nanna_storage::NewTask {
                scope: "session".to_string(),
                scope_id: Some("s1".to_string()),
                title: "Implement DEL".to_string(),
                priority: 2,
                ..Default::default()
            })
            .await
            .expect("seed");
        assert_eq!(open_items(storage.clone()).await, 1);
        assert!(
            run_evidence || open_items(storage.clone()).await > 0,
            "a crashed run that seeded work it never touched still holds a mission"
        );
    }

    /// The dry branch that seeded nothing must consult the failing evidence
    /// exactly like its sibling: reopen the standing wall by ID (a re-seeded
    /// title is eaten by the closed-title dedup), and stop reopening once
    /// nothing moves — which is what makes the guard terminate.
    #[tokio::test]
    async fn a_standing_unmet_item_is_reopened_by_id_exactly_once() {
        let storage = Arc::new(Storage::in_memory().await.expect("storage"));
        let task = storage
            .tasks()
            .create(nanna_storage::NewTask {
                scope: "session".to_string(),
                scope_id: Some("s1".to_string()),
                title: "Implement test_07".to_string(),
                priority: 2,
                ..Default::default()
            })
            .await
            .expect("seed");
        storage
            .tasks()
            .complete(task.id, Some("test"), None)
            .await
            .expect("close");

        let walls = vec![unmet(task.id, "`sh tests/test_07.sh` exited 1")];
        let first = reopen_top_unmet(&storage, &walls).await;
        assert_eq!(first.as_ref().map(|(id, _)| *id), Some(task.id));
        assert_eq!(
            storage.tasks().get(task.id).await.expect("read back").status,
            "pending"
        );
        assert!(
            reopen_top_unmet(&storage, &walls).await.is_none(),
            "an already-open item is not reopened again — the loop must terminate"
        );
        // Nothing to reopen at all is also quiet.
        assert!(reopen_top_unmet(&storage, &[]).await.is_none());
    }

    /// One reseed per DISTINCT verified state: the fingerprint changes when
    /// the environment's verdicts change, and only then.
    #[test]
    fn reseed_fingerprint_tracks_the_environments_verdicts() {
        let mut a = report(StopReason::AllTasksDone, 1, 0, 1, None);
        a.abandoned_unmet = vec![unmet(1, "exit 1"), unmet(2, "exit 2")];
        let mut reordered = a.clone();
        reordered.abandoned_unmet.reverse();
        assert_eq!(
            verified_state_fingerprint(&a),
            verified_state_fingerprint(&reordered),
            "order is not evidence"
        );
        let mut moved = a.clone();
        moved.abandoned_unmet[0].detail = "exit 0".to_string();
        assert_ne!(
            verified_state_fingerprint(&a),
            verified_state_fingerprint(&moved),
            "a changed verdict is changed evidence — a later reseed is armed"
        );
    }

    // ------------------------------------------------------------------
    // Claim conflict
    // ------------------------------------------------------------------

    fn verified_row(id: i64, title: &str, verdict: &str, command: &str) -> EstablishedRow {
        EstablishedRow {
            id,
            title: title.to_string(),
            verdict: Some(verdict.to_string()),
            when: "2026-08-14T10:00:00Z".to_string(),
            acceptance: Some(json!({ "kind": "command", "command": command })),
        }
    }

    #[test]
    fn a_contradicted_pass_is_detected_and_only_a_contradicted_one() {
        let rows = vec![verified_row(
            12,
            "Implement SET and GET",
            "`sh tests/test_02.sh` exited 0 — ok",
            "sh tests/test_02.sh",
        )];
        // Asserts failure AND names the verified subject.
        let hit = claim_conflicts("tests/test_02.sh is failing again", &rows);
        assert_eq!(hit.len(), 1, "{hit:?}");
        assert_eq!(hit[0].subject, "tests/test_02.sh");
        assert_eq!(hit[0].task_id, 12);

        // Names the subject but asserts nothing — ordinary chat.
        assert!(claim_conflicts("what does tests/test_02.sh cover?", &rows).is_empty());
        // Asserts failure about something else entirely.
        assert!(claim_conflicts("the deploy script is broken", &rows).is_empty());
        // No verdict on the row: nothing to contradict.
        let unverified = vec![EstablishedRow {
            verdict: None,
            ..rows[0].clone()
        }];
        assert!(claim_conflicts("tests/test_02.sh is failing", &unverified).is_empty());
    }

    /// Bare words are not identities: "test" or "build" would fire on half of
    /// English, and a false conflict costs a real reproduction step.
    #[test]
    fn only_path_like_identities_can_collide() {
        let rows = vec![verified_row(3, "Build it", "`cargo build` exited 0", "cargo build")];
        assert!(
            claim_conflicts("the build is broken", &rows).is_empty(),
            "a bare word must never be treated as an on-disk identity"
        );
        let rows = vec![verified_row(
            4,
            "Write the config",
            "`src/config.rs` exists",
            "test -f src/config.rs",
        )];
        assert_eq!(claim_conflicts("src/config.rs is broken", &rows).len(), 1);
    }

    /// The conflict renders BOTH sides, and the reconciliation task lands at
    /// the HEAD of the plan carrying the disputed check as its own acceptance.
    #[test]
    fn reconciliation_is_planned_before_any_mutation() {
        use nanna_agent::planner::{Plan, PlannedTask};

        let rows = vec![verified_row(
            12,
            "Implement SET and GET",
            "`sh tests/test_02.sh` exited 0 — ok",
            "sh tests/test_02.sh",
        )];
        let conflicts = claim_conflicts("tests/test_02.sh fails now", &rows);
        let block = claim_conflict_block(&conflicts).expect("a conflict renders");
        assert!(block.contains("CLAIM-CONFLICT"), "{block}");
        assert!(block.contains("tests/test_02.sh"), "{block}");
        assert!(block.contains("`sh tests/test_02.sh` exited 0"), "{block}");
        assert!(block.contains("Reproduce first"), "{block}");

        let mut plan = Plan {
            tasks: vec![PlannedTask {
                title: "Rewrite tests/test_02.sh handling in the parser".to_string(),
                description: None,
                acceptance: None,
                tool_scope: Vec::new(),
            }],
            origin: nanna_agent::planner::PlanOrigin::Model,
        };
        prepend_reconciliation_tasks(&mut plan, &conflicts);
        assert_eq!(plan.tasks.len(), 2);
        assert!(plan.tasks[0].title.starts_with("Reproduce the reported failure"));
        assert_eq!(
            plan.tasks[0].acceptance,
            Some(json!({ "kind": "command", "command": "sh tests/test_02.sh" })),
            "the reconciliation re-runs the DISPUTED check through the shipped machinery"
        );
        let description = plan.tasks[0].description.as_deref().expect("a protocol");
        assert!(description.contains("Do NOT change"), "{description}");
        assert!(description.contains("ask ONE clarifying question"), "{description}");
        assert!(plan.tasks[1].title.starts_with("Rewrite"), "the mutation still runs, second");

        // No conflict, no change at all.
        let mut untouched = plan.clone();
        let before = untouched.clone();
        prepend_reconciliation_tasks(&mut untouched, &[]);
        assert_eq!(untouched, before);
    }

    // ------------------------------------------------------------------
    // Artifact state
    // ------------------------------------------------------------------

    /// The headline block: ground truth re-read from disk, under the contract
    /// sentence, with the ledger's own evidence beside a FRESH stat.
    #[tokio::test]
    async fn artifact_state_reads_the_ledger_and_stats_the_files() {
        let root = std::env::temp_dir().join(format!("nanna-artifact-{}", uuid::Uuid::new_v4()));
        tokio::fs::create_dir_all(root.join(".nanna")).await.expect("mkdir");
        tokio::fs::write(root.join("minidb.sh"), "echo hi\n").await.expect("write");
        tokio::fs::write(
            root.join(HIWATER_LEDGER),
            serde_json::to_string(&json!({
                "minidb.sh": {
                    "hi": 4096, "last": 8, "at": 200,
                    "good": 3335, "goodAt": 190, "chk": "ok",
                },
                "gone.sh": { "hi": 12, "last": 12, "at": 100 },
            }))
            .expect("json"),
        )
        .await
        .expect("write ledger");

        let rows = vec![verified_row(
            5,
            "Implement SET",
            "`sh tests/test_02.sh` exited 0",
            "sh tests/test_02.sh",
        )];
        let block = artifact_state_block(Some(root.as_path()), &rows)
            .await
            .expect("a tracked workspace renders");
        assert!(block.contains("ARTIFACT STATE"), "{block}");
        assert!(
            block.contains("read before writing; extend, do not reconstruct"),
            "the contract sentence IS the point: {block}"
        );
        // Fresh stat, not the ledger's `last`.
        assert!(block.contains("minidb.sh — 8 bytes on disk"), "{block}");
        assert!(block.contains("last modified"), "{block}");
        assert!(block.contains("largest version this session 4096 bytes"), "{block}");
        assert!(block.contains("structural check 3335 bytes"), "{block}");
        assert!(block.contains("latest check verdict: ok"), "{block}");
        // Newest-written first, and a tracked file that is gone says so.
        assert!(
            block.find("minidb.sh").unwrap() < block.find("gone.sh").unwrap(),
            "{block}"
        );
        assert!(block.contains("gone.sh — NOT PRESENT on disk"), "{block}");
        // The verdict half rides along.
        assert!(block.contains("#5 Implement SET — verified"), "{block}");

        // Fails open everywhere: no workspace, no ledger, no evidence.
        assert!(artifact_state_block(None, &[]).await.is_none());
        assert!(
            artifact_state_block(Some(root.join("nope").as_path()), &[])
                .await
                .is_none(),
            "an unreadable ledger degrades to no block, never to an error"
        );
        // …but stored verdicts alone still render.
        assert!(artifact_state_block(None, &rows).await.is_some());

        let _ = tokio::fs::remove_dir_all(&root).await;
    }

    /// P22: verified outcomes render as the planner's ESTABLISHED block —
    /// each fact with its id, title and the environment's verdict — bounded
    /// with an announced overflow, newest first.
    #[test]
    fn established_block_renders_verified_facts_bounded() {
        assert!(established_block(&[]).is_none(), "no facts, no block");
        let outcomes: Vec<nanna_agent::harness::VerifiedOutcome> = (0..13)
            .map(|i| nanna_agent::harness::VerifiedOutcome {
                id: i,
                title: format!("feature {i}"),
                detail: format!("`test_{i}.sh` exited 0"),
                already_satisfied: i % 2 == 0,
            })
            .collect();
        let block = established_block(&outcomes).expect("facts must render");
        assert!(block.starts_with("ESTABLISHED"));
        assert!(block.contains("do NOT re-propose") || block.contains("Do NOT re-propose"));
        // Newest facts win the bounded slots…
        assert!(block.contains("#12 feature 12: `test_12.sh` exited 0"));
        assert!(!block.contains("#0 feature 0"), "oldest must yield: {block}");
        // …and the cut announces itself.
        assert!(block.contains("…and 3 more"), "{block}");
    }

    /// P22 resume-as-continue: a NEW turn's planner context carries what
    /// earlier turns PROVED — closed items with their completion verdicts,
    /// read back from the store — so a re-send after self-termination builds
    /// on the artifact state instead of re-seeding the mission from zero.
    #[tokio::test]
    async fn established_work_context_reads_verdicts_back_from_the_store() {
        let storage = Arc::new(Storage::in_memory().await.unwrap());
        assert!(
            established_work_context(&established_rows(&storage, "session", Some("s1")).await)
                .is_none(),
            "nothing closed, no block"
        );
        let task = storage
            .tasks()
            .create(nanna_storage::NewTask {
                scope: "session".to_string(),
                scope_id: Some("s1".to_string()),
                title: "Implement SET and GET".to_string(),
                priority: 2,
                ..Default::default()
            })
            .await
            .unwrap();
        storage
            .tasks()
            .complete(
                task.id,
                Some("chat"),
                Some(serde_json::json!({
                    "verified": true,
                    "verdict": "`sh tests/test_02.sh` exited 0 — ok",
                })),
            )
            .await
            .unwrap();
        // An open sibling must not leak into the established block.
        storage
            .tasks()
            .create(nanna_storage::NewTask {
                scope: "session".to_string(),
                scope_id: Some("s1".to_string()),
                title: "Implement DEL".to_string(),
                priority: 2,
                ..Default::default()
            })
            .await
            .unwrap();

        let rows = established_rows(&storage, "session", Some("s1")).await;
        let block =
            established_work_context(&rows).expect("a verified completion must render");
        assert!(block.contains("Verified done in earlier work"));
        assert!(block.contains("Implement SET and GET"));
        assert!(
            block.contains("`sh tests/test_02.sh` exited 0"),
            "the verdict IS the artifact state: {block}"
        );
        assert!(!block.contains("Implement DEL"), "open work stays out: {block}");
    }

    fn row(id: i64, title: &str, verdict: Option<&str>) -> EstablishedRow {
        EstablishedRow {
            id,
            title: title.to_string(),
            verdict: verdict.map(str::to_string),
            when: "2026-09-21T11:10:44Z".to_string(),
            acceptance: None,
        }
    }

    /// An item closed on the model's word was listed under "done-condition
    /// PASSING — do not redo or re-assess", which no check had confirmed.
    #[test]
    fn unverified_completions_are_not_presented_as_passing_checks() {
        let only_unverified = established_work_context(&[row(1, "Answer the question", None)])
            .expect("a closed item renders");
        assert!(!only_unverified.contains("PASSING"), "{only_unverified}");
        assert!(
            !only_unverified.contains("Do not redo"),
            "{only_unverified}"
        );
        assert!(
            only_unverified.contains("no check ran"),
            "{only_unverified}"
        );
        assert!(
            only_unverified.contains("#1 Answer the question"),
            "{only_unverified}"
        );

        let mixed = established_work_context(&[
            row(2, "Implement SET", Some("`sh test.sh` exited 0")),
            row(1, "Answer the question", None),
        ])
        .expect("closed items render");
        let passing = mixed.find("PASSING").expect("the verified section is kept");
        let unchecked = mixed
            .find("no check ran")
            .expect("the unverified section is kept");
        let set = mixed.find("#2 Implement SET").expect("verified row");
        let answer = mixed
            .find("#1 Answer the question")
            .expect("unverified row");
        assert!(
            passing < set && set < unchecked && unchecked < answer,
            "{mixed}"
        );
    }

    #[test]
    fn a_user_facing_excerpt_drops_notices_written_for_the_model() {
        let step = "[step report synthesized from the tool record — the model emitted no final text]\n\
                    - frobnicate FAILED ([HARNESS NOTE — you are mid-task: x] [REPEAT-FAILURE BREAKER — stop)";
        let kept = without_model_notices(step);
        assert!(kept.ends_with("- frobnicate FAILED…"), "{kept}");
        assert!(
            !kept.contains("HARNESS NOTE") && !kept.contains("BREAKER"),
            "{kept}"
        );
        assert_eq!(without_model_notices("[HARNESS NOTE — only this]"), "");
        assert_eq!(without_model_notices("plain words"), "plain words");
    }

    #[test]
    fn failure_notice_surfaces_a_fully_abandoned_plan() {
        // The live 2026-07-31 shape: provider fault → poison containment →
        // AllTasksDone with zero steps. The notice must carry the error.
        let r = report(
            StopReason::AllTasksDone,
            0,
            0,
            1,
            Some("No provider available for model 'claude-fable-5'"),
        );
        let notice = failure_notice(&r).expect("a drained-by-abandonment run must announce itself");
        assert!(notice.contains("could not run"), "{notice}");
        assert!(
            notice.contains("No provider available for model 'claude-fable-5'"),
            "{notice}"
        );
        assert!(
            !notice.starts_with('\n'),
            "nothing streamed, so no separator: {notice:?}"
        );
    }

    #[test]
    fn failure_notice_falls_back_when_no_runner_error_was_recorded() {
        // Fruitless-step abandonment records no runner error; the notice must
        // still say the plan was abandoned rather than staying silent.
        let notice = failure_notice(&report(StopReason::AllTasksDone, 0, 0, 2, None))
            .expect("abandonment without an error still announces itself");
        assert!(notice.contains("abandoned"), "{notice}");
    }

    #[test]
    fn failure_notice_reports_error_stops_and_separates_from_streamed_text() {
        let r = report(
            StopReason::RunnerErrors {
                message: "step error: 502".to_string(),
            },
            4,
            1,
            0,
            Some("step error: 502"),
        );
        let notice = failure_notice(&r).expect("an error stop is a failure");
        assert!(
            notice.starts_with("\n\n"),
            "steps streamed text — the notice needs a separator: {notice:?}"
        );
        assert!(notice.contains("could not finish"), "{notice}");
        assert!(notice.contains("502"), "{notice}");

        let r = report(
            StopReason::SourceError {
                message: "storage exploded".to_string(),
            },
            0,
            0,
            0,
            None,
        );
        let notice = failure_notice(&r).expect("a source error is a failure");
        assert!(notice.contains("task store"), "{notice}");
    }

    #[test]
    fn failure_notice_stays_quiet_on_success_and_deliberate_stops() {
        // Clean success.
        assert!(failure_notice(&report(StopReason::AllTasksDone, 3, 2, 0, None)).is_none());
        // The user pressed Stop — do not dress it up as a fault.
        assert!(failure_notice(&report(StopReason::Cancelled, 0, 0, 1, Some("x"))).is_none());
        // Budget stops were asked for.
        assert!(failure_notice(&report(StopReason::WallClockExhausted, 9, 3, 0, None)).is_none());
        // Partial completion: the run-stats line carries the abandoned count;
        // an error banner would overstate the failure.
        assert!(failure_notice(&report(StopReason::AllTasksDone, 5, 2, 1, Some("x"))).is_none());
    }

    /// REGRESSION (live 2026-08-02, session 7ccc455a): one conversational
    /// question ("difference between a mutex and a semaphore?") became an
    /// eleven-round treadmill — every continuation round re-planned the same
    /// goal, the previous copy was CLOSED so seeding created a fresh clone
    /// (tasks #2033..#2044), and the dry detector never tripped until
    /// `CONTINUATION_ROUNDS_MAX`. This drives the loop's exact round
    /// arithmetic against a real store with a planner that re-emits the same
    /// goal every round: with closed-title dedup, two consecutive
    /// all-duplicate rounds end the mission and the goal is worked ONCE.
    #[tokio::test]
    async fn two_all_duplicate_continuation_rounds_end_the_mission() {
        use crate::tasks::{closed_task_ids, seed_continuation, seed_plan};
        use nanna_agent::planner::Plan;

        let storage = Arc::new(Storage::in_memory().await.expect("storage"));
        let scope = "session";
        let scope_id = Some("s1");
        let goal = "Explain the difference between a mutex and a semaphore";

        // Turn start: snapshot the closed baseline, seed the initial plan,
        // and let the "run" complete its single task.
        let closed_before_turn = closed_task_ids(&storage, scope, scope_id)
            .await
            .expect("baseline");
        let ids = seed_plan(&storage, scope, scope_id, &Plan::single(goal), false)
            .await
            .expect("initial seed");
        for id in &ids {
            storage
                .tasks()
                .complete(*id, Some("test"), None)
                .await
                .expect("complete");
        }

        // The continuation loop's round accounting, with the planner
        // behaviour observed live: the same single-task plan every round.
        let mut dry_rounds = 0usize;
        let mut continuations = 0usize;
        while dry_rounds < CONTINUATION_DRY_ROUNDS && continuations < CONTINUATION_ROUNDS_MAX {
            continuations += 1;
            let seeded =
                seed_continuation(&storage, scope, scope_id, &Plan::single(goal), &closed_before_turn)
                    .await
                    .unwrap_or_default();
            if seeded.is_empty() {
                dry_rounds += 1;
                continue;
            }
            dry_rounds = 0;
            for id in &seeded {
                storage
                    .tasks()
                    .complete(*id, Some("test"), None)
                    .await
                    .expect("complete");
            }
        }

        assert_eq!(
            continuations, CONTINUATION_DRY_ROUNDS,
            "the mission ends after two dry rounds, not at ROUNDS_MAX"
        );
        let all = storage
            .tasks()
            .list(scope, scope_id, true)
            .await
            .expect("list");
        assert_eq!(
            all.len(),
            1,
            "the goal was worked once, never treadmilled: {:?}",
            all.iter().map(|t| &t.title).collect::<Vec<_>>()
        );
    }

    /// One continuation round's report: `side_effects` work-evidence calls
    /// and `completed` items closed. `tool_calls` is deliberately larger than
    /// `side_effects` — a round that only READ still called tools, and that
    /// is exactly the distinction the total counter cannot make.
    fn round(side_effects: usize, completed: usize) -> LongHorizonReport {
        LongHorizonReport {
            tool_calls: side_effects + 4,
            side_effect_tool_calls: side_effects,
            ..report(StopReason::AllTasksDone, 1, completed, 0, None)
        }
    }

    /// A round whose items were all closed by the acceptance PRE-CHECK: the
    /// environment said the work was already there, so nothing ran.
    fn already_satisfied_round(items: usize) -> LongHorizonReport {
        LongHorizonReport {
            tool_calls: 0,
            side_effect_tool_calls: 0,
            items_already_satisfied: items,
            ..report(StopReason::AllTasksDone, 0, items, 0, None)
        }
    }

    /// REGRESSION (live 2026-08-03, "fix conflicts and merge all open prs"):
    /// the planner rephrased finished work every round, so the title filter
    /// saw new work forever. Progress is judged by what a round CHANGED.
    #[test]
    fn a_round_that_changes_nothing_makes_no_progress() {
        // Read-only inspection: tools were called, nothing moved.
        assert!(!round_made_progress(&round(0, 0)));
        // Writing/editing/shelling is progress…
        assert!(round_made_progress(&round(1, 0)));
        // …and so is closing an item the round actually worked.
        assert!(round_made_progress(&round(0, 1)));
    }

    /// The decisive rung. The live mission's done-condition (`gh pr list
    /// --state open … | grep -qx 0`) passed on EVERY continuation round, so
    /// every re-proposal the planner made was already satisfied. Those
    /// completions prove the goal was already met — they are not progress, and
    /// two of them in a row end the mission.
    #[test]
    fn items_the_environment_had_already_satisfied_are_not_progress() {
        assert!(!round_made_progress(&already_satisfied_round(1)));
        assert!(!round_made_progress(&already_satisfied_round(3)));
        // One genuinely worked item among pre-satisfied ones IS progress: the
        // comparison is on the excess, not on the raw count.
        let mixed = LongHorizonReport {
            items_already_satisfied: 2,
            ..round(0, 3)
        };
        assert!(round_made_progress(&mixed));
    }

    /// End to end over the counter the loop actually runs: a planner that
    /// re-proposes finished work under NEW verbs every round — the exact live
    /// transcript, which `same_title` can no longer match and must not — still
    /// converges, because the pre-check closes each proposal for free and the
    /// dry counter reaches its bound.
    #[test]
    fn a_rephrasing_planner_converges_on_the_pre_check() {
        let mut dry_rounds = 0usize;
        let mut continuations = 0usize;
        while dry_rounds < CONTINUATION_DRY_ROUNDS && continuations < CONTINUATION_ROUNDS_MAX {
            // Every round: one seeded task, pre-check passes, nothing run.
            if round_made_progress(&already_satisfied_round(1)) {
                dry_rounds = 0;
            } else {
                dry_rounds += 1;
            }
            continuations += 1;
        }
        assert_eq!(
            continuations, CONTINUATION_DRY_ROUNDS,
            "the mission ends two rounds after the environment says it is done"
        );
    }

    /// The structural counter must work where the title filter cannot: rounds
    /// with genuinely different titles that nonetheless change nothing end the
    /// mission after `CONTINUATION_DRY_ROUNDS`, not at `ROUNDS_MAX`.
    #[tokio::test]
    async fn two_rounds_that_change_nothing_end_the_mission() {
        use crate::tasks::{closed_task_ids, seed_continuation};
        use nanna_agent::planner::Plan;

        let storage = Arc::new(Storage::in_memory().await.expect("storage"));
        let scope = "session";
        let scope_id = Some("s1");
        let closed_before_turn = closed_task_ids(&storage, scope, scope_id)
            .await
            .expect("baseline");

        // Titles the duplicate filter has no grounds to refuse — different
        // subjects every round.
        let titles = [
            "inventory the alpha module",
            "inventory the beta module",
            "inventory the gamma module",
        ];
        let mut dry_rounds = 0usize;
        let mut continuations = 0usize;
        while dry_rounds < CONTINUATION_DRY_ROUNDS && continuations < CONTINUATION_ROUNDS_MAX {
            let seeded = seed_continuation(
                &storage,
                scope,
                scope_id,
                &Plan::single(titles[continuations % titles.len()]),
                &closed_before_turn,
            )
            .await
            .unwrap_or_default();
            continuations += 1;
            assert!(
                !seeded.is_empty(),
                "a genuinely new title seeds — only the structural signal can end this"
            );
            if round_made_progress(&round(0, 0)) {
                dry_rounds = 0;
            } else {
                dry_rounds += 1;
            }
        }

        assert_eq!(
            continuations, CONTINUATION_DRY_ROUNDS,
            "two rounds that changed nothing end the mission"
        );
    }

    /// A genuine multi-round mission is untouched: a round that writes files
    /// or closes items resets the counter exactly as before.
    #[test]
    fn a_productive_round_resets_the_dry_counter() {
        // Observed shape of a real build: a quiet round, then work, then two
        // quiet ones. The mission survives the first quiet round and ends on
        // the SECOND consecutive one — four rounds, not two.
        let script = [round(0, 0), round(3, 1), round(0, 0), round(0, 0)];
        let mut dry_rounds = 0usize;
        let mut continuations = 0usize;
        while dry_rounds < CONTINUATION_DRY_ROUNDS && continuations < script.len() {
            if round_made_progress(&script[continuations]) {
                dry_rounds = 0;
            } else {
                dry_rounds += 1;
            }
            continuations += 1;
        }
        assert_eq!(continuations, 4, "the productive round bought more rounds");
        assert_eq!(dry_rounds, CONTINUATION_DRY_ROUNDS);
    }

    #[test]
    fn conversation_context_renders_roles_in_order() {
        let history = vec![
            msg(MessageRole::User, "what is 2+2?"),
            msg(MessageRole::Assistant, "4"),
        ];
        let rendered = conversation_context(&history).expect("non-empty history renders");
        assert_eq!(rendered, "User: what is 2+2?\nNanna: 4");
    }

    #[test]
    fn conversation_context_skips_system_tool_and_empty_messages() {
        let history = vec![
            msg(MessageRole::System, "internal prompt"),
            msg(MessageRole::User, "   "),
            msg(MessageRole::Tool, "tool record"),
        ];
        assert!(conversation_context(&history).is_none());
        assert!(conversation_context(&[]).is_none());
    }

    #[test]
    fn conversation_context_clamps_one_giant_message() {
        let giant = "x".repeat(PLAN_GOAL_MAX_BYTES * 2);
        let history = vec![msg(MessageRole::User, &giant), msg(MessageRole::User, "next")];
        let rendered = conversation_context(&history).expect("renders");
        // The giant message is clamped to the per-message bound, so BOTH
        // messages fit — one paste must not occupy the whole window.
        assert!(rendered.contains("… [message truncated]"));
        assert!(rendered.ends_with("User: next"));
        assert!(rendered.len() <= PLAN_GOAL_MAX_BYTES + "[earlier conversation omitted]\n".len());
    }

    #[test]
    fn conversation_context_keeps_newest_and_announces_dropped_prefix() {
        // Enough medium messages to overflow the total budget.
        let filler = "y".repeat(PLAN_DESCRIPTION_MAX_BYTES / 2);
        let history: Vec<SessionMessage> = (0..20)
            .map(|i| msg(MessageRole::User, &format!("m{i} {filler}")))
            .collect();
        let rendered = conversation_context(&history).expect("renders");
        assert!(rendered.starts_with("[earlier conversation omitted]"));
        // Newest message always survives.
        assert!(rendered.contains("m19 "));
        // Oldest was dropped.
        assert!(!rendered.contains("m0 "));
    }

    #[test]
    fn strip_harness_markers_removes_only_whole_marker_lines() {
        let text = "did the work\ntask complete\nTASK COMPLETE\nalmost TASK COMPLETE inline\n";
        let stripped = strip_harness_markers(text);
        assert_eq!(stripped, "did the work\nalmost TASK COMPLETE inline");
    }

    #[test]
    fn strip_harness_markers_trims_dangling_trailing_blanks() {
        assert_eq!(strip_harness_markers("answer: 4\n\nTASK COMPLETE\n"), "answer: 4");
        assert_eq!(strip_harness_markers("TASK COMPLETE"), "");
    }

    #[test]
    fn strip_harness_markers_cuts_lines_at_nudge_markers() {
        // The whole nudge line is fabricated steering — dropped outright.
        assert_eq!(
            strip_harness_markers(
                "[THINKING SPIRAL DETECTED] I was overthinking. Let me act instead of deliberate."
            ),
            ""
        );
        // Stream glue onto real prose: the fabricated tail is cut, prose stays.
        assert_eq!(
            strip_harness_markers("checking the cache[THINKING SPIRAL DETECTED] I was overthinking."),
            "checking the cache"
        );
    }

    #[test]
    fn strip_harness_markers_peels_glued_claim_markers() {
        // The live 2026-08-02 shape: claim marker fused to a nudge that the
        // lost newline glued onto the same line.
        let text = "ran both checks concurrently.\n\nTASK COMPLETE[THINKING SPIRAL DETECTED] \
                    I was overthinking. Let me act instead of deliberate.";
        assert_eq!(strip_harness_markers(text), "ran both checks concurrently.");
        // Suffix glue: prose that lost its newline before the claim line.
        assert_eq!(
            strip_harness_markers("all checks green.TASK COMPLETE"),
            "all checks green."
        );
        // Doubled marker collapses to nothing.
        assert_eq!(strip_harness_markers("TASK COMPLETEtask complete"), "");
    }

    #[test]
    fn strip_harness_markers_keeps_space_separated_mentions() {
        // Glue never introduces spaces, so a whitespace boundary is prose —
        // even at the start or end of a line.
        let text = "TASK COMPLETE is the marker I emit\nwe are almost TASK COMPLETE";
        assert_eq!(strip_harness_markers(text), text);
    }

    #[test]
    fn sanitize_timeline_strips_text_and_drops_emptied_entries() {
        let at = Utc::now().to_rfc3339();
        let items = vec![
            TimelineItem::Text {
                content: "hello\nTASK COMPLETE".to_string(),
                at: at.clone(),
            },
            TimelineItem::Text {
                content: "\nTASK COMPLETE\n".to_string(),
                at: at.clone(),
            },
            TimelineItem::Tool {
                call_id: "c1".to_string(),
                name: "exec".to_string(),
                input: None,
                output: Some("ok".to_string()),
                success: Some(true),
                duration_ms: Some(3),
                tokens: None,
                total_tokens: None,
                short_circuited: None,
                diff: None,
                at,
            },
        ];
        let sanitized = sanitize_timeline(items);
        assert_eq!(sanitized.len(), 2);
        assert!(matches!(
            &sanitized[0],
            TimelineItem::Text { content, .. } if content == "hello"
        ));
        assert!(matches!(&sanitized[1], TimelineItem::Tool { .. }));
    }

    #[test]
    fn sanitize_timeline_drops_empty_thinking_and_keeps_records() {
        let at = Utc::now().to_rfc3339();
        let items = vec![
            TimelineItem::Thinking {
                content: "  \n\t\n".to_string(),
                at: at.clone(),
            },
            TimelineItem::Thinking {
                content: String::new(),
                at: at.clone(),
            },
            TimelineItem::Thinking {
                content: "  the file is missing a newline  ".to_string(),
                at: at.clone(),
            },
            TimelineItem::Tool {
                call_id: "c1".to_string(),
                name: "read_file".to_string(),
                input: None,
                output: Some("ok".to_string()),
                success: Some(true),
                duration_ms: Some(3),
                tokens: None,
                total_tokens: None,
                short_circuited: None,
                diff: None,
                at: at.clone(),
            },
            TimelineItem::Fault {
                message: "stream ended without done=true".to_string(),
                at,
            },
        ];
        let sanitized = sanitize_timeline(items);
        assert_eq!(sanitized.len(), 3);
        // Real thinking survives verbatim — no trimming, no marker strip.
        assert!(matches!(
            &sanitized[0],
            TimelineItem::Thinking { content, .. }
                if content == "  the file is missing a newline  "
        ));
        assert!(matches!(&sanitized[1], TimelineItem::Tool { .. }));
        assert!(matches!(&sanitized[2], TimelineItem::Fault { .. }));
    }

    /// The idle-backfill supervisor's bound is "one pass per turn", and it
    /// gets that from `wait_active` — NOT from a timer. If `wait_active`
    /// returned on an idle registry, the supervisor would spin as fast as the
    /// scheduler allows and the "one probe per turn" claim would be false.
    #[tokio::test]
    async fn wait_active_does_not_return_while_nothing_is_running() {
        let runs = ChatRunRegistry::new();
        assert!(!runs.any_active().await, "a fresh registry has no live run");
        let waited = tokio::time::timeout(
            std::time::Duration::from_millis(50),
            runs.wait_active(),
        )
        .await;
        assert!(
            waited.is_err(),
            "wait_active must park on an idle registry — returning here is what \
             would turn the supervisor's edge into a spin"
        );
    }

    /// One claim/release pair drives exactly one active→idle cycle — the
    /// supervisor's whole loop body — PROVIDED the observer is parked on
    /// `wait_active` when the claim lands. The handshake is not test
    /// scaffolding for its own sake: without it this test fails by timeout,
    /// which is the missed-edge property `supervise_idle_backfill` documents
    /// (and which `a_turn_that_ends_before_the_observer_parks_is_missed` pins
    /// deliberately below).
    #[tokio::test]
    async fn one_turn_drives_one_active_then_idle_cycle() {
        let runs = Arc::new(ChatRunRegistry::new());
        let observer = runs.clone();
        let saw_active = Arc::new(tokio::sync::Notify::new());
        let saw_active_signal = saw_active.clone();
        let cycle = tokio::spawn(async move {
            observer.wait_active().await;
            // `notify_one` stores a permit when nobody is waiting yet, so the
            // main task cannot miss this regardless of poll order.
            saw_active_signal.notify_one();
            observer.wait_idle().await;
        });

        tokio::task::yield_now().await;
        assert!(runs.try_claim("session-1").await, "the slot is free");
        assert!(runs.any_active().await, "the claim is visible");

        // Release only once the observer has actually consumed the active edge.
        tokio::time::timeout(std::time::Duration::from_secs(5), saw_active.notified())
            .await
            .expect("the observer must see the active edge it was parked on");
        runs.release("session-1").await;

        tokio::time::timeout(std::time::Duration::from_secs(5), cycle)
            .await
            .expect("the release must complete the cycle")
            .expect("the observer task must not panic");
        assert!(!runs.any_active().await, "the cycle ends with the registry idle");
    }

    /// The missed edge, pinned on purpose rather than left to be rediscovered.
    ///
    /// `wait_active` registers interest and THEN reads the flag, so a turn that
    /// begins and ends before the observer is polled leaves no edge behind and
    /// the observer stays parked. `supervise_idle_backfill` inherits exactly
    /// this: such a turn's queued rows wait for the NEXT turn rather than
    /// draining at the end of their own.
    ///
    /// That is a bounded, deliberate cost — the rows are durable and reachable
    /// by their `source_id` handle throughout — and the alternative (probing
    /// before parking) would make the supervisor spin on an idle daemon. This
    /// test exists so that if anyone ever "fixes" the loop, they are told which
    /// property they traded away.
    #[tokio::test]
    async fn a_turn_that_ends_before_the_observer_parks_is_missed() {
        let runs = Arc::new(ChatRunRegistry::new());
        let observer = runs.clone();
        let cycle = tokio::spawn(async move {
            observer.wait_active().await;
            observer.wait_idle().await;
        });
        tokio::task::yield_now().await;

        // A whole turn, start to finish, with no chance for the observer to run
        // in between.
        assert!(runs.try_claim("session-1").await);
        runs.release("session-1").await;

        let outcome =
            tokio::time::timeout(std::time::Duration::from_millis(200), cycle).await;
        assert!(
            outcome.is_err(),
            "a turn that opens and closes between polls leaves no edge — if this now \
             completes, the registry gained edge buffering and supervise_idle_backfill's \
             documented one-turn worst case is stale"
        );
    }

    /// A release with other runs still live is NOT an idle edge. The drain
    /// must not start while a second session is still holding the local
    /// provider — that is the contention this gate exists to prevent.
    #[tokio::test]
    async fn wait_idle_holds_until_the_last_run_releases() {
        let runs = Arc::new(ChatRunRegistry::new());
        assert!(runs.try_claim("session-1").await);
        assert!(runs.try_claim("session-2").await);

        runs.release("session-1").await;
        let early = tokio::time::timeout(
            std::time::Duration::from_millis(50),
            runs.wait_idle(),
        )
        .await;
        assert!(
            early.is_err(),
            "one of two runs releasing is not an idle edge"
        );

        runs.release("session-2").await;
        tokio::time::timeout(std::time::Duration::from_secs(5), runs.wait_idle())
            .await
            .expect("the last release opens the gate");
    }
}
