//! Task scheduler for autonomous behavior
//!
//! Provides heartbeats, cron jobs, and background task execution.
//! Supports persistent storage of cron jobs via nanna-storage.

use crate::cron::{CronError, CronExpr};
use chrono::{DateTime, Utc};
use nanna_storage::{NewCronJob, Storage};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;
use tokio::sync::{mpsc, RwLock};
use tokio::time::{interval, interval_at, Instant};
use tracing::{debug, error, info, warn};

/// Floor on the heartbeat interval.
///
/// Two independent constraints meet at the same number. The scheduler resolves
/// *all* due work on its `check_interval` tick (30s), so a heartbeat period
/// finer than that is finer than the scheduler's own resolution — it cannot be
/// honored, only rounded. And a heartbeat is a full agent turn against the chat
/// model, which takes far longer than 30s on a local backend, so a shorter
/// period would queue heartbeats faster than they drain. `interval()` also
/// panics outright on a zero period, which this floor rules out by
/// construction.
pub const MIN_HEARTBEAT_INTERVAL_SECS: u64 = 30;

/// Scheduled task types
#[derive(Debug, Clone)]
pub enum TaskType {
    /// Heartbeat - periodic self-check
    Heartbeat,
    /// Cron job with parsed schedule
    Cron {
        schedule: String,
        /// Boxed: five field sets make a `CronExpr` several times larger than
        /// every other variant.
        parsed: Option<Box<CronExpr>>,
        next_run: Option<DateTime<Utc>>,
    },
    /// One-shot delayed task
    ///
    /// The delay is measured from a monotonic `Instant`, which cannot be
    /// persisted: a reloaded `Delayed` task restarts its whole delay at boot.
    /// Anything that must fire at a wall-clock moment across a restart is an
    /// [`TaskType::At`] instead.
    Delayed { delay: Duration, created: Instant },
    /// One-shot task due at an absolute wall-clock instant.
    ///
    /// Persisted as `at_<rfc3339>`, so a restart neither re-arms nor drifts it:
    /// a moment that passed while the daemon was down is simply due on the
    /// first tick after boot.
    At { fire_at: DateTime<Utc> },
    /// Recurring task at fixed interval
    Recurring { interval: Duration },
}

/// Prefix of the persisted schedule string for [`TaskType::At`].
const AT_SCHEDULE_PREFIX: &str = "at_";

impl TaskType {
    /// The schedule string this task type is persisted and reported as.
    ///
    /// One spelling, used by storage and by every client listing, so the
    /// string a client shows is the string `job_to_task` parses back.
    #[must_use]
    pub fn schedule_label(&self) -> String {
        match self {
            Self::Heartbeat => "heartbeat".to_string(),
            Self::Cron { schedule, .. } => schedule.clone(),
            Self::Recurring { interval } => format!("every_{}s", interval.as_secs()),
            Self::Delayed { delay, .. } => format!("delay_{}s", delay.as_secs()),
            Self::At { fire_at } => format!("{AT_SCHEDULE_PREFIX}{}", fire_at.to_rfc3339()),
        }
    }

    /// The next wall-clock moment this task is due, where one is known.
    #[must_use]
    pub const fn next_run(&self) -> Option<DateTime<Utc>> {
        match self {
            Self::Cron { next_run, .. } => *next_run,
            Self::At { fire_at } => Some(*fire_at),
            Self::Heartbeat | Self::Recurring { .. } | Self::Delayed { .. } => None,
        }
    }

    /// Whether a task of this type runs at most once.
    #[must_use]
    pub const fn is_one_shot(&self) -> bool {
        matches!(self, Self::Delayed { .. } | Self::At { .. })
    }
}

/// Whether `task` is due at `now`. Pure: the loop's only scheduling decision.
///
/// Heartbeats are never "due" here — they run on their own timer.
#[must_use]
pub fn is_task_due(task: &ScheduledTask, now: DateTime<Utc>) -> bool {
    if !task.enabled {
        return false;
    }
    match &task.task_type {
        TaskType::Cron { next_run, .. } => next_run.is_some_and(|next| next <= now),
        TaskType::Recurring { interval } => task.last_run.is_none_or(|last| {
            now.signed_duration_since(last)
                >= chrono::Duration::from_std(*interval).unwrap_or_default()
        }),
        TaskType::Delayed { delay, created } => task.run_count == 0 && created.elapsed() >= *delay,
        TaskType::At { fire_at } => task.run_count == 0 && *fire_at <= now,
        TaskType::Heartbeat => false,
    }
}

/// A scheduled task
#[derive(Debug, Clone)]
pub struct ScheduledTask {
    pub id: String,
    pub name: String,
    pub task_type: TaskType,
    pub payload: String,
    pub enabled: bool,
    pub last_run: Option<DateTime<Utc>>,
    pub run_count: u64,
    /// Timezone for cron evaluation (default: UTC)
    pub timezone: String,
    /// Channel to send results to (optional)
    pub target_channel: Option<String>,
    /// Session to run in (optional)
    pub target_session: Option<String>,
}

/// Task execution result
#[derive(Debug, Clone)]
pub struct TaskResult {
    pub task_id: String,
    pub task_name: String,
    pub success: bool,
    pub output: Option<String>,
    pub error: Option<String>,
    pub duration_ms: u64,
    pub started_at: DateTime<Utc>,
    pub finished_at: DateTime<Utc>,
}

/// Job run history entry
#[derive(Debug, Clone)]
pub struct JobRun {
    pub id: i64,
    pub job_id: String,
    pub started_at: DateTime<Utc>,
    pub finished_at: Option<DateTime<Utc>>,
    pub success: bool,
    pub output: Option<String>,
    pub error: Option<String>,
}

/// Callback for task execution
pub type TaskExecutor = Arc<
    dyn Fn(ScheduledTask) -> std::pin::Pin<Box<dyn std::future::Future<Output = TaskResult> + Send>>
        + Send
        + Sync,
>;

/// Scheduler configuration
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Master switch. `false` keeps the scheduler loaded — jobs stay in the
    /// store and remain listable/editable — but fires nothing: no heartbeat,
    /// no cron, no consolidation, no recurrence sweep.
    pub enabled: bool,
    /// Heartbeat interval. Values below [`MIN_HEARTBEAT_INTERVAL_SECS`] are
    /// clamped up when the running loop reads them.
    pub heartbeat_interval: Duration,
    /// Whether heartbeats are enabled.
    ///
    /// A heartbeat drives a full agent turn against the *same* model chat uses.
    /// On a single-slot local backend a heartbeat firing mid-conversation
    /// time-shares the slot and the in-flight generation gets cancelled, which
    /// reaches the client as a stream that ended without `done=true`. Being
    /// able to turn this off is a prerequisite for a clean local benchmark run.
    pub heartbeat_enabled: bool,
    /// Heartbeat prompt/payload
    pub heartbeat_prompt: String,
    /// Maximum concurrent tasks
    pub max_concurrent: usize,
    /// Check interval for cron jobs (how often to check for due jobs)
    pub check_interval: Duration,
    /// Default timezone for cron expressions
    pub default_timezone: String,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            heartbeat_interval: Duration::from_mins(30), // 30 minutes
            heartbeat_enabled: true,
            // Do not command a `Read HEARTBEAT.md` here — that drove a read_file
            // tool call that hard-errored on a missing file (resolved to ~ with
            // no active workspace), and P17 retired the bespoke HEARTBEAT.md
            // entirely (recurrence lives in scheduled-task config now).
            heartbeat_prompt: "Heartbeat check-in. Run any due scheduled tasks. Do not read files from disk looking for instructions, and do not infer or repeat old tasks from prior chats. If nothing needs attention, reply HEARTBEAT_OK.".to_string(),
            max_concurrent: 4,
            check_interval: Duration::from_secs(30),
            default_timezone: "UTC".to_string(),
        }
    }
}

/// The scheduler settings a user may change while the loop is running.
///
/// [`Scheduler::start`] moves a *clone* of [`SchedulerConfig`] into its spawned
/// task, so everything captured there is frozen for the life of the process.
/// These three knobs are user-facing (Settings → Scheduler) and one of them —
/// the heartbeat — actively interferes with chat on a single-slot local model,
/// so flipping it has to take effect now, not at the next daemon restart. They
/// therefore live behind atomics that the loop re-reads on every tick.
///
/// All accesses are `Relaxed`: each knob is read independently on its own tick
/// and nothing is ordered against anything else, so the only guarantee needed
/// is that a write eventually becomes visible.
#[derive(Debug)]
pub struct SchedulerRuntime {
    enabled: AtomicBool,
    heartbeat_enabled: AtomicBool,
    heartbeat_interval_secs: AtomicU64,
}

impl SchedulerRuntime {
    fn from_config(config: &SchedulerConfig) -> Self {
        Self {
            enabled: AtomicBool::new(config.enabled),
            heartbeat_enabled: AtomicBool::new(config.heartbeat_enabled),
            heartbeat_interval_secs: AtomicU64::new(clamp_heartbeat_secs(
                config.heartbeat_interval.as_secs(),
            )),
        }
    }

    /// Whether the scheduler fires anything at all.
    #[must_use]
    pub fn enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    /// Whether the periodic heartbeat prompt fires. Independent of
    /// [`Self::enabled`]: both must be true for a heartbeat to run.
    #[must_use]
    pub fn heartbeat_enabled(&self) -> bool {
        self.heartbeat_enabled.load(Ordering::Relaxed)
    }

    pub fn set_heartbeat_enabled(&self, enabled: bool) {
        self.heartbeat_enabled.store(enabled, Ordering::Relaxed);
    }

    /// The heartbeat period, never below [`MIN_HEARTBEAT_INTERVAL_SECS`].
    #[must_use]
    pub fn heartbeat_interval(&self) -> Duration {
        Duration::from_secs(self.heartbeat_interval_secs.load(Ordering::Relaxed))
    }

    /// Set the heartbeat period, clamping up to [`MIN_HEARTBEAT_INTERVAL_SECS`].
    /// Clamping at the setter (not the reader) means the stored value is always
    /// the value the loop will actually use.
    pub fn set_heartbeat_interval(&self, interval: Duration) {
        self.heartbeat_interval_secs
            .store(clamp_heartbeat_secs(interval.as_secs()), Ordering::Relaxed);
    }
}

/// Raise a heartbeat period to the floor the scheduler can actually honor.
/// See [`MIN_HEARTBEAT_INTERVAL_SECS`] for why the floor exists.
#[must_use]
pub fn clamp_heartbeat_secs(secs: u64) -> u64 {
    secs.max(MIN_HEARTBEAT_INTERVAL_SECS)
}

/// The main scheduler
pub struct Scheduler {
    config: SchedulerConfig,
    tasks: Arc<RwLock<HashMap<String, ScheduledTask>>>,
    executor: Option<TaskExecutor>,
    shutdown_tx: Option<mpsc::Sender<()>>,
    storage: Option<Arc<Storage>>,
    /// Job run history (in-memory, last N runs per job)
    history: Arc<RwLock<HashMap<String, Vec<JobRun>>>>,
    /// The knobs the running loop re-reads. Shared with the spawned task, so
    /// this is the authority for the three live settings — `config` keeps the
    /// boot-time values and is updated alongside by [`Self::apply_settings`].
    runtime: Arc<SchedulerRuntime>,
}

impl Scheduler {
    #[must_use]
    pub fn new(config: SchedulerConfig) -> Self {
        let runtime = Arc::new(SchedulerRuntime::from_config(&config));
        Self {
            config,
            tasks: Arc::new(RwLock::new(HashMap::new())),
            executor: None,
            shutdown_tx: None,
            storage: None,
            history: Arc::new(RwLock::new(HashMap::new())),
            runtime,
        }
    }

    /// Handle onto the live knobs. Cloneable and independent of any lock around
    /// the scheduler itself.
    #[must_use]
    pub fn runtime(&self) -> Arc<SchedulerRuntime> {
        self.runtime.clone()
    }

    /// Apply user-facing scheduler settings to the *running* loop.
    ///
    /// Takes effect within one `check_interval` tick — no restart. `config` is
    /// updated in lock-step so the struct never disagrees with the atomics
    /// about what the scheduler is doing.
    pub fn apply_settings(
        &mut self,
        enabled: bool,
        heartbeat_enabled: bool,
        heartbeat_interval: Duration,
    ) {
        self.runtime.set_enabled(enabled);
        self.runtime.set_heartbeat_enabled(heartbeat_enabled);
        self.runtime.set_heartbeat_interval(heartbeat_interval);

        self.config.enabled = enabled;
        self.config.heartbeat_enabled = heartbeat_enabled;
        // Mirror the *clamped* value, not the requested one.
        self.config.heartbeat_interval = self.runtime.heartbeat_interval();

        info!(
            "Scheduler settings applied: enabled={enabled}, heartbeat_enabled={heartbeat_enabled}, \
             heartbeat_interval={:?}",
            self.config.heartbeat_interval
        );
    }

    /// Set persistent storage for cron jobs.
    #[must_use]
    pub fn with_storage(mut self, storage: Arc<Storage>) -> Self {
        self.storage = Some(storage);
        self
    }

    /// Load persisted cron jobs from storage.
    ///
    /// Returns 0 without touching anything when no storage is configured.
    ///
    /// # Errors
    ///
    /// Returns the storage error when listing the persisted cron jobs fails.
    pub async fn load_jobs(&self) -> Result<usize, nanna_storage::StorageError> {
        let Some(storage) = &self.storage else {
            return Ok(0);
        };

        let jobs = storage.cron_jobs().list_all().await?;
        let count = jobs.len();

        let mut tasks = self.tasks.write().await;
        for job in jobs {
            let task = self.job_to_task(&job);
            tasks.insert(job.job_id, task);
        }
        drop(tasks);

        info!("Loaded {} cron jobs from storage", count);
        Ok(count)
    }

    /// Convert storage job to scheduler task
    fn job_to_task(&self, job: &nanna_storage::CronJob) -> ScheduledTask {
        let task_type = if job.schedule == "heartbeat" {
            TaskType::Heartbeat
        } else if let Some(interval_secs) = parse_interval(&job.schedule) {
            TaskType::Recurring {
                interval: Duration::from_secs(interval_secs),
            }
        } else if let Some(fire_at) = parse_at(&job.schedule) {
            TaskType::At { fire_at }
        } else if let Some(delay_secs) = parse_delay(&job.schedule) {
            TaskType::Delayed {
                delay: Duration::from_secs(delay_secs),
                created: Instant::now(),
            }
        } else {
            // Try to parse as cron expression
            let parsed = CronExpr::parse(&job.schedule).ok().map(Box::new);
            let next_run = parsed.as_deref().and_then(super::cron::CronExpr::next_from_now);
            TaskType::Cron {
                schedule: job.schedule.clone(),
                parsed,
                next_run,
            }
        };

        let payload = job
            .task
            .get("payload")
            .and_then(|v| v.as_str())
            .or_else(|| job.task.get("text").and_then(|v| v.as_str()))
            .unwrap_or("")
            .to_string();

        let name = job
            .task
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or(&job.job_id)
            .to_string();

        let target_channel = job
            .task
            .get("channel")
            .and_then(|v| v.as_str())
            .map(String::from);

        let target_session = job
            .task
            .get("session")
            .and_then(|v| v.as_str())
            .map(String::from);

        let timezone = job
            .task
            .get("timezone")
            .and_then(|v| v.as_str())
            .unwrap_or(&self.config.default_timezone)
            .to_string();

        ScheduledTask {
            id: job.job_id.clone(),
            name,
            task_type,
            payload,
            enabled: job.enabled,
            last_run: job.last_run.as_ref().and_then(|s| s.parse().ok()),
            run_count: 0,
            timezone,
            target_channel,
            target_session,
        }
    }

    /// Save a task to persistent storage.
    async fn persist_task(&self, task: &ScheduledTask) -> Result<(), nanna_storage::StorageError> {
        let Some(storage) = &self.storage else {
            return Ok(());
        };

        let schedule = task.task_type.schedule_label();
        let next_run = task.task_type.next_run().map(|dt| dt.to_rfc3339());

        let job = NewCronJob {
            job_id: task.id.clone(),
            schedule,
            task: serde_json::json!({
                "name": task.name,
                "payload": task.payload,
                "timezone": task.timezone,
                "channel": task.target_channel,
                "session": task.target_session,
            }),
            enabled: task.enabled,
            next_run,
            metadata: None,
        };

        storage.cron_jobs().create(job).await?;
        Ok(())
    }

    /// Set the task executor callback (builder pattern).
    #[must_use]
    pub fn with_executor(mut self, executor: TaskExecutor) -> Self {
        self.executor = Some(executor);
        self
    }

    /// Set the task executor callback (mutable reference).
    pub fn set_executor(&mut self, executor: TaskExecutor) {
        self.executor = Some(executor);
    }

    /// Create a cron task from a schedule expression
    ///
    /// # Errors
    ///
    /// Returns the [`CronError`] from [`CronExpr::parse`] when `schedule` is not
    /// a valid cron expression.
    pub fn cron_task(
        name: &str,
        schedule: &str,
        payload: &str,
    ) -> Result<ScheduledTask, CronError> {
        let parsed = CronExpr::parse(schedule)?;
        let next_run = parsed.next_from_now();

        Ok(ScheduledTask {
            id: format!("{}-{}", name, uuid::Uuid::new_v4()),
            name: name.to_string(),
            task_type: TaskType::Cron {
                schedule: schedule.to_string(),
                parsed: Some(Box::new(parsed)),
                next_run,
            },
            payload: payload.to_string(),
            enabled: true,
            last_run: None,
            run_count: 0,
            timezone: "UTC".to_string(),
            target_channel: None,
            target_session: None,
        })
    }

    /// Add a task (persisted if storage is configured)
    pub async fn add_task(&self, task: ScheduledTask) {
        // Persist first if storage is available
        if let Err(e) = self.persist_task(&task).await {
            warn!("Failed to persist task {}: {}", task.id, e);
        }

        let mut tasks = self.tasks.write().await;
        info!("Scheduled task: {} ({})", task.name, task.id);
        tasks.insert(task.id.clone(), task);
    }

    /// Remove a task (from memory and storage)
    pub async fn remove_task(&self, task_id: &str) -> bool {
        // Remove from storage
        if let Some(storage) = &self.storage
            && let Err(e) = storage.cron_jobs().delete(task_id).await {
                warn!("Failed to delete task {} from storage: {}", task_id, e);
            }

        let mut tasks = self.tasks.write().await;
        tasks.remove(task_id).is_some()
    }

    /// Update a task's schedule
    ///
    /// Returns `Ok(false)` when no task has `task_id`. The new schedule is
    /// persisted like a new task is (a failure is logged, as in
    /// [`Self::add_task`]). It used to write only `next_run` — and an empty
    /// `last_run` over the real one — so the new expression never reached the
    /// store and a restart put the old schedule back.
    ///
    /// # Errors
    ///
    /// Returns the [`CronError`] from [`CronExpr::parse`] when `schedule` is not
    /// a valid cron expression; the task is left unchanged.
    pub async fn update_schedule(&self, task_id: &str, schedule: &str) -> Result<bool, CronError> {
        let parsed = CronExpr::parse(schedule)?;
        let next_run = parsed.next_from_now();
        let updated = self
            .modify_task(task_id, |task| {
                task.task_type = TaskType::Cron {
                    schedule: schedule.to_string(),
                    parsed: Some(Box::new(parsed)),
                    next_run,
                };
            })
            .await;
        Ok(updated)
    }

    /// Enable/disable a task (persisted). `false` when no task has `task_id`.
    pub async fn set_task_enabled(&self, task_id: &str, enabled: bool) -> bool {
        self.modify_task(task_id, |task| task.enabled = enabled)
            .await
    }

    /// Replace a task's payload (persisted). `false` when no task has
    /// `task_id`.
    pub async fn update_payload(&self, task_id: &str, payload: &str) -> bool {
        self.modify_task(task_id, |task| task.payload = payload.to_string())
            .await
    }

    /// Apply `change` to the task in RAM and persist the whole row, the same
    /// upsert [`Self::add_task`] writes, so every field changed together
    /// reaches the store together. `false` when no task has `task_id`.
    async fn modify_task(&self, task_id: &str, change: impl FnOnce(&mut ScheduledTask)) -> bool {
        let mut tasks = self.tasks.write().await;
        let Some(task) = tasks.get_mut(task_id) else {
            return false;
        };
        change(task);
        let snapshot = task.clone();
        drop(tasks);
        debug_assert_eq!(snapshot.id, task_id, "the change kept the task's id");
        if let Err(e) = self.persist_task(&snapshot).await {
            warn!("Failed to persist the change to task {task_id}: {e}");
        }
        true
    }

    /// Get all tasks
    pub async fn list_tasks(&self) -> Vec<ScheduledTask> {
        let tasks = self.tasks.read().await;
        tasks.values().cloned().collect()
    }

    /// Get a specific task
    pub async fn get_task(&self, task_id: &str) -> Option<ScheduledTask> {
        let tasks = self.tasks.read().await;
        tasks.get(task_id).cloned()
    }

    /// Find tasks by name
    pub async fn find_by_name(&self, name: &str) -> Vec<ScheduledTask> {
        let tasks = self.tasks.read().await;
        tasks.values().filter(|t| t.name == name).cloned().collect()
    }

    /// Check if a task with the given name exists
    pub async fn has_task_named(&self, name: &str) -> bool {
        let tasks = self.tasks.read().await;
        tasks.values().any(|t| t.name == name)
    }

    /// Remove all tasks with a given name (useful for cleaning up duplicates)
    pub async fn remove_tasks_by_name(&self, name: &str) -> usize {
        let task_ids: Vec<String> = {
            let tasks = self.tasks.read().await;
            tasks.values()
                .filter(|t| t.name == name)
                .map(|t| t.id.clone())
                .collect()
        };

        let count = task_ids.len();
        for task_id in task_ids {
            self.remove_task(&task_id).await;
        }
        count
    }

    /// Remove all tasks except one with a given name (keep the most recent one)
    pub async fn deduplicate_by_name(&self, name: &str) -> usize {
        let task_ids: Vec<String> = {
            let tasks = self.tasks.read().await;
            let mut matching: Vec<_> = tasks.values()
                .filter(|t| t.name == name)
                .collect();

            // Keep one (the first alphabetically by ID)
            matching.sort_by(|a, b| a.id.cmp(&b.id));

            // Skip the first one, return the rest for deletion
            let ids = matching.into_iter()
                .skip(1)
                .map(|t| t.id.clone())
                .collect();
            drop(tasks);
            ids
        };

        let count = task_ids.len();
        for task_id in task_ids {
            self.remove_task(&task_id).await;
        }
        count
    }

    /// Get job run history
    pub async fn get_history(&self, job_id: &str, limit: usize) -> Vec<JobRun> {
        let history = self.history.read().await;
        history
            .get(job_id)
            .map_or_default(|runs| runs.iter().rev().take(limit).cloned().collect())
    }

    /// Record a job run
    async fn record_run(&self, result: &TaskResult) {
        record_run_in(&self.history, &result.task_id, result).await;
    }

    /// How often the loop looks for due work — the resolution of every
    /// non-heartbeat schedule, reminders included.
    #[must_use]
    pub const fn check_interval(&self) -> Duration {
        self.config.check_interval
    }

    /// Run a task immediately (bypass schedule)
    pub async fn run_now(&self, task_id: &str) -> Option<TaskResult> {
        let executor = self.executor.as_ref()?;
        let task = {
            let tasks = self.tasks.read().await;
            tasks.get(task_id).cloned()?
        };

        let result = executor(task).await;

        // Record the run
        self.record_run(&result).await;

        // Update last_run
        {
            let mut tasks = self.tasks.write().await;
            if let Some(t) = tasks.get_mut(task_id) {
                t.last_run = Some(result.finished_at);
                t.run_count += 1;

                // Update next_run for cron tasks
                if let TaskType::Cron {
                    parsed: Some(ref p),
                    ref mut next_run,
                    ..
                } = t.task_type
                {
                    *next_run = p.next_from_now();
                }
            }
        }

        Some(result)
    }

    /// Start the scheduler.
    pub fn start(&mut self) {
        let executor = if let Some(e) = &self.executor {
            e.clone()
        } else {
            warn!("No executor set, scheduler will not run tasks");
            return;
        };

        let (shutdown_tx, mut shutdown_rx) = mpsc::channel::<()>(1);
        self.shutdown_tx = Some(shutdown_tx);

        let config = self.config.clone();
        let tasks = self.tasks.clone();
        let storage = self.storage.clone();
        let history = self.history.clone();
        let runtime = self.runtime.clone();
        let in_flight = InFlight::default();

        // Spawn the scheduler loop
        tokio::spawn(async move {
            // The heartbeat period is retunable at runtime, so track the value
            // this timer was built from and rebuild when it changes.
            let mut heartbeat_secs = runtime.heartbeat_interval().as_secs();
            // A full period out, like the retune below: `interval()` fires its
            // first tick immediately, which ran a model turn at every boot.
            let mut heartbeat_interval = {
                let period = Duration::from_secs(heartbeat_secs);
                interval_at(Instant::now() + period, period)
            };
            let mut check_interval = interval(config.check_interval);

            info!(
                "Scheduler started (enabled: {}, heartbeat every {}s, check every {:?})",
                runtime.enabled(),
                heartbeat_secs,
                config.check_interval
            );

            loop {
                tokio::select! {
                    _ = shutdown_rx.recv() => {
                        info!("Scheduler shutting down");
                        break;
                    }
                    _ = heartbeat_interval.tick() => {
                        if runtime.enabled() && runtime.heartbeat_enabled() {
                            // Spawned like every due task, never awaited here:
                            // a heartbeat is a model turn, and awaiting it in
                            // this `select!` stalled every other due task (and
                            // shutdown) behind it. The claim keeps a slow one
                            // from overlapping the next tick.
                            let Some(claim) = InFlightClaim::take(&in_flight, HEARTBEAT_CLAIM) else {
                                debug!("Heartbeat still running; skipping this tick");
                                continue;
                            };
                            tokio::spawn(run_heartbeat(
                                claim,
                                executor.clone(),
                                history.clone(),
                                config.heartbeat_prompt.clone(),
                            ));
                        }
                    }
                    _ = check_interval.tick() => {
                        // Pick up a live heartbeat-interval change. Rebuilt only
                        // when the value actually differs, so steady state never
                        // resets the timer's phase; the replacement starts a full
                        // period out so retuning the interval cannot itself fire a
                        // heartbeat.
                        let desired_secs = runtime.heartbeat_interval().as_secs();
                        if desired_secs != heartbeat_secs {
                            heartbeat_secs = desired_secs;
                            let period = Duration::from_secs(heartbeat_secs);
                            heartbeat_interval = interval_at(Instant::now() + period, period);
                            info!("Heartbeat interval changed to {heartbeat_secs}s");
                        }

                        // Master switch: leave jobs loaded, fire nothing.
                        if !runtime.enabled() {
                            continue;
                        }

                        // Check for due tasks
                        let now = Utc::now();
                        let tasks_snapshot = {
                            let tasks = tasks.read().await;
                            tasks.values().cloned().collect::<Vec<_>>()
                        };

                        for task in tasks_snapshot {
                            if !is_task_due(&task, now) {
                                continue;
                            }
                            // Claimed BEFORE it is spawned, released only after
                            // its state is settled. The tick is 30s and nothing
                            // bounds how long a run takes, while "due" is read
                            // from state a run updates only when it FINISHES —
                            // so a run still going at the next tick was due
                            // again and started a second copy of itself.
                            let Some(claim) = InFlightClaim::take(&in_flight, &task.id) else {
                                debug!("Task {} still running; not starting it twice", task.id);
                                continue;
                            };
                            tokio::spawn(run_due_task(
                                claim,
                                task,
                                executor.clone(),
                                tasks.clone(),
                                storage.clone(),
                                history.clone(),
                            ));
                        }
                    }
                }
            }
        });
    }

    /// Stop the scheduler
    pub async fn stop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(()).await;
        }
    }
}

/// Run one due task and settle its state: history, `last_run`, storage.
///
/// One-shots are settled by outcome. A delivered one-shot is REMOVED — memory
/// and storage — because a disabled row that is never re-enabled is dead
/// weight that accumulates one row per reminder forever. A failed one stays,
/// disabled and persisted disabled, so the failure is inspectable and a
/// restart does not silently retry it. (Before this, a fired `Delayed` task
/// was disabled in memory only; storage kept `enabled = 1`, so every daemon
/// restart re-armed it and it fired again.)
/// In-flight key for the heartbeat. Not a task id: the heartbeat is built
/// fresh on every tick, so its own id would never repeat and never collide.
const HEARTBEAT_CLAIM: &str = "\0heartbeat";

/// One heartbeat run, holding `claim` until it has been recorded.
async fn run_heartbeat(
    claim: InFlightClaim,
    executor: TaskExecutor,
    history: Arc<RwLock<HashMap<String, Vec<JobRun>>>>,
    prompt: String,
) {
    debug_assert_eq!(
        claim.task_id, HEARTBEAT_CLAIM,
        "the heartbeat holds its own claim"
    );
    debug!("Heartbeat tick");
    let result = executor(heartbeat_task(&prompt)).await;
    record_run_in(&history, "heartbeat", &result).await;
    if !result.success {
        warn!("Heartbeat failed: {:?}", result.error);
    }
    drop(claim);
}

async fn run_due_task(
    claim: InFlightClaim,
    task: ScheduledTask,
    executor: TaskExecutor,
    tasks: Arc<RwLock<HashMap<String, ScheduledTask>>>,
    storage: Option<Arc<Storage>>,
    history: Arc<RwLock<HashMap<String, Vec<JobRun>>>>,
) {
    let task_id = task.id.clone();
    debug_assert_eq!(
        claim.task_id, task_id,
        "a run holds the claim for its own task"
    );
    let one_shot = task.task_type.is_one_shot();
    debug!("Running task: {} ({})", task.name, task_id);
    let result = executor(task).await;
    debug_assert_eq!(
        result.task_id, task_id,
        "executor must answer for the task it ran"
    );
    debug_assert!(
        result.finished_at >= result.started_at,
        "a run cannot finish before it starts"
    );

    record_run_in(&history, &result.task_id, &result).await;

    let next_run = {
        let mut tasks_guard = tasks.write().await;
        if one_shot && result.success {
            tasks_guard.remove(&task_id);
            None
        } else if let Some(t) = tasks_guard.get_mut(&task_id) {
            t.last_run = Some(result.finished_at);
            t.run_count += 1;
            if let TaskType::Cron {
                parsed: Some(ref p),
                ref mut next_run,
                ..
            } = t.task_type
            {
                *next_run = p.next_from_now();
            }
            if one_shot {
                t.enabled = false;
            }
            t.task_type.next_run().map(|dt| dt.to_rfc3339())
        } else {
            None
        }
    };

    if let Some(storage) = storage {
        settle_in_storage(&storage, &task_id, one_shot, &result, next_run.as_deref()).await;
    }
    // Released only now: the in-memory state above is what the next tick reads.
    drop(claim);

    if result.success {
        info!("Task {} completed in {}ms", task_id, result.duration_ms);
    } else {
        error!("Task {} failed: {:?}", task_id, result.error);
    }
}

/// Ids of tasks with a run in progress. Bounded by the task map it indexes.
type InFlight = Arc<std::sync::Mutex<std::collections::HashSet<String>>>;

/// Proof that a task's run is in progress; releases the id when dropped, so an
/// executor that panics (in a build that unwinds) does not wedge its task.
struct InFlightClaim {
    in_flight: InFlight,
    task_id: String,
}

impl InFlightClaim {
    /// Claim `task_id`, or `None` when a run of it is already in progress.
    fn take(in_flight: &InFlight, task_id: &str) -> Option<Self> {
        let mut ids = in_flight
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !ids.insert(task_id.to_string()) {
            return None;
        }
        debug_assert!(ids.contains(task_id));
        drop(ids);
        Some(Self {
            in_flight: Arc::clone(in_flight),
            task_id: task_id.to_string(),
        })
    }
}

impl Drop for InFlightClaim {
    fn drop(&mut self) {
        let released = self
            .in_flight
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&self.task_id);
        debug_assert!(released, "a claim releases exactly the id it took");
    }
}

/// Mirror a finished run into storage. See [`run_due_task`] for the one-shot rule.
async fn settle_in_storage(
    storage: &Storage,
    task_id: &str,
    one_shot: bool,
    result: &TaskResult,
    next_run: Option<&str>,
) {
    let jobs = storage.cron_jobs();
    if one_shot && result.success {
        if let Err(e) = jobs.delete(task_id).await {
            warn!("Failed to delete delivered one-shot {task_id}: {e}");
        }
        return;
    }
    let finished_at = result.finished_at.to_rfc3339();
    if let Err(e) = jobs.update_last_run(task_id, &finished_at, next_run).await {
        warn!("Failed to persist the last run of task {task_id}: {e}");
    }
    if one_shot && let Err(e) = jobs.set_enabled(task_id, false).await {
        warn!("Failed to persist failed one-shot {task_id} as disabled: {e}");
    }
}

/// Append a run to the bounded per-job history, under `job_id`.
///
/// The id is explicit because the heartbeat files its runs under `heartbeat`
/// rather than under a task id — it used to keep its own copy of this function
/// for that, with its own bound.
async fn record_run_in(
    history: &RwLock<HashMap<String, Vec<JobRun>>>,
    job_id: &str,
    result: &TaskResult,
) {
    let mut hist = history.write().await;
    let runs = hist.entry(job_id.to_string()).or_default();
    runs.push(JobRun {
        id: i64::try_from(runs.len()).unwrap_or(i64::MAX - 1) + 1,
        job_id: job_id.to_string(),
        started_at: result.started_at,
        finished_at: Some(result.finished_at),
        success: result.success,
        output: result.output.clone(),
        error: result.error.clone(),
    });
    if runs.len() > JOB_HISTORY_RUNS_MAX {
        runs.remove(0);
    }
    debug_assert!(runs.len() <= JOB_HISTORY_RUNS_MAX);
    // Held across the push and the trim together: a reader between them would
    // see the history one run over its bound.
    drop(hist);
}

/// Runs kept per job. The GUI's history view pages ten at a time; a hundred is
/// ten pages, and the list is in memory only.
const JOB_HISTORY_RUNS_MAX: usize = 100;

/// Parse an absolute one-shot schedule like `at_2026-09-17T10:00:00+00:00`.
fn parse_at(schedule: &str) -> Option<DateTime<Utc>> {
    let stamp = schedule.strip_prefix(AT_SCHEDULE_PREFIX)?;
    DateTime::parse_from_rfc3339(stamp)
        .ok()
        .map(|dt| dt.with_timezone(&Utc))
}

/// Parse interval string like `every_300s` into seconds
fn parse_interval(schedule: &str) -> Option<u64> {
    if schedule.starts_with("every_") && schedule.ends_with('s') {
        let num_str = &schedule[6..schedule.len() - 1];
        num_str.parse().ok()
    } else {
        None
    }
}

/// Parse delay string like "`delay_60s`" into seconds
fn parse_delay(schedule: &str) -> Option<u64> {
    if schedule.starts_with("delay_") && schedule.ends_with('s') {
        let num_str = &schedule[6..schedule.len() - 1];
        num_str.parse().ok()
    } else {
        None
    }
}

/// Helper to create a heartbeat task
#[must_use]
pub fn heartbeat_task(prompt: &str) -> ScheduledTask {
    ScheduledTask {
        id: format!("heartbeat-{}", uuid::Uuid::new_v4()),
        name: "heartbeat".to_string(),
        task_type: TaskType::Heartbeat,
        payload: prompt.to_string(),
        enabled: true,
        last_run: None,
        run_count: 0,
        timezone: "UTC".to_string(),
        target_channel: None,
        target_session: None,
    }
}

/// Helper to create a recurring task
#[must_use]
pub fn recurring_task(name: &str, interval: Duration, payload: &str) -> ScheduledTask {
    ScheduledTask {
        id: format!("{}-{}", name, uuid::Uuid::new_v4()),
        name: name.to_string(),
        task_type: TaskType::Recurring { interval },
        payload: payload.to_string(),
        enabled: true,
        last_run: None,
        run_count: 0,
        timezone: "UTC".to_string(),
        target_channel: None,
        target_session: None,
    }
}

/// Helper to create a delayed one-shot task
#[must_use]
pub fn delayed_task(name: &str, delay: Duration, payload: &str) -> ScheduledTask {
    ScheduledTask {
        id: format!("{}-{}", name, uuid::Uuid::new_v4()),
        name: name.to_string(),
        task_type: TaskType::Delayed {
            delay,
            created: Instant::now(),
        },
        payload: payload.to_string(),
        enabled: true,
        last_run: None,
        run_count: 0,
        timezone: "UTC".to_string(),
        target_channel: None,
        target_session: None,
    }
}

/// Helper to create a one-shot task due at an absolute wall-clock instant
#[must_use]
pub fn at_task(name: &str, fire_at: DateTime<Utc>, payload: &str) -> ScheduledTask {
    ScheduledTask {
        id: format!("{}-{}", name, uuid::Uuid::new_v4()),
        name: name.to_string(),
        task_type: TaskType::At { fire_at },
        payload: payload.to_string(),
        enabled: true,
        last_run: None,
        run_count: 0,
        timezone: "UTC".to_string(),
        target_channel: None,
        target_session: None,
    }
}

/// Helper to create a memory consolidation ("dreaming") task
#[must_use]
pub fn consolidation_task(interval: Option<Duration>) -> ScheduledTask {
    let interval = interval.unwrap_or(Duration::from_secs(3600)); // 1 hour default
    ScheduledTask {
        id: format!("consolidation-{}", uuid::Uuid::new_v4()),
        name: "memory_consolidation".to_string(),
        task_type: TaskType::Recurring { interval },
        payload: "Run memory consolidation (dreaming): compress fading memories, expand important ones.".to_string(),
        enabled: true,
        last_run: None,
        run_count: 0,
        timezone: "UTC".to_string(),
        target_channel: None,
        target_session: None,
    }
}

/// Task type marker for the dreaming task
pub const DREAMING_TASK_NAME: &str = "memory_consolidation";

/// Check if a task is the dreaming/consolidation task
#[must_use]
pub fn is_dreaming_task(task: &ScheduledTask) -> bool {
    task.name == DREAMING_TASK_NAME
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_scheduler_add_task() {
        let scheduler = Scheduler::new(SchedulerConfig::default());

        let task = recurring_task("test", Duration::from_secs(60), "test payload");
        scheduler.add_task(task.clone()).await;

        let tasks = scheduler.list_tasks().await;
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].name, "test");
    }

    #[tokio::test]
    async fn test_cron_task_creation() {
        let task = Scheduler::cron_task("daily-backup", "0 3 * * *", "Run backup").unwrap();
        assert_eq!(task.name, "daily-backup");
        assert!(matches!(task.task_type, TaskType::Cron { .. }));
    }

    fn fixed(secs: i64) -> DateTime<Utc> {
        DateTime::from_timestamp(1_800_000_000 + secs, 0).expect("in range")
    }

    #[test]
    fn an_at_schedule_parses_back_to_the_instant_it_was_written_from() {
        let fire_at = fixed(0);
        let label = TaskType::At { fire_at }.schedule_label();
        assert!(label.starts_with("at_"), "{label}");
        assert_eq!(parse_at(&label), Some(fire_at));
        assert_eq!(parse_at("at_not-a-time"), None);
        assert_eq!(parse_at("delay_60s"), None);
    }

    #[test]
    fn an_at_task_is_due_once_at_its_instant_and_never_while_disabled() {
        let mut task = at_task("reminder", fixed(60), "stretch");
        assert!(!is_task_due(&task, fixed(59)));
        assert!(is_task_due(&task, fixed(60)));
        assert!(
            is_task_due(&task, fixed(6000)),
            "overdue after downtime is still due"
        );
        task.enabled = false;
        assert!(!is_task_due(&task, fixed(60)));
        task.enabled = true;
        task.run_count = 1;
        assert!(!is_task_due(&task, fixed(60)));
        assert!(task.task_type.is_one_shot());
        assert_eq!(task.task_type.next_run(), Some(fixed(60)));
    }

    #[tokio::test]
    async fn a_reloaded_at_task_keeps_its_instant_instead_of_rearming() {
        let storage = Arc::new(Storage::in_memory().await.expect("in-memory storage"));
        let before = Scheduler::new(SchedulerConfig::default()).with_storage(storage.clone());
        let mut task = at_task("reminder", fixed(90), "stretch");
        task.target_session = Some("chat-1".into());
        let id = task.id.clone();
        before.add_task(task).await;

        let after = Scheduler::new(SchedulerConfig::default()).with_storage(storage);
        assert_eq!(after.load_jobs().await.expect("load"), 1);
        let reloaded = after.get_task(&id).await.expect("reloaded");
        assert!(matches!(reloaded.task_type, TaskType::At { fire_at } if fire_at == fixed(90)));
        assert_eq!(reloaded.target_session.as_deref(), Some("chat-1"));
        assert_eq!(reloaded.payload, "stretch");
    }

    /// Wait (bounded) for `condition`, polling the scheduler's own state.
    async fn eventually<F, Fut>(mut condition: F) -> bool
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = bool>,
    {
        for _ in 0..200 {
            if condition().await {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        false
    }

    #[test]
    fn an_in_flight_claim_excludes_a_second_run_until_released() {
        let in_flight = InFlight::default();
        let first = InFlightClaim::take(&in_flight, "job-a").expect("free");
        assert!(InFlightClaim::take(&in_flight, "job-a").is_none());
        let other = InFlightClaim::take(&in_flight, "job-b").expect("independent ids");
        drop(first);
        assert!(InFlightClaim::take(&in_flight, "job-a").is_some());
        drop(other);
        assert!(in_flight.lock().expect("ids").is_empty());
    }

    #[tokio::test]
    async fn a_recurring_run_slower_than_the_tick_never_overlaps_itself() {
        let running = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let peak = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let runs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (running_in, peak_in, runs_in) = (running.clone(), peak.clone(), runs.clone());
        let executor: TaskExecutor = Arc::new(move |task: ScheduledTask| {
            let (running, peak, runs) = (running_in.clone(), peak_in.clone(), runs_in.clone());
            Box::pin(async move {
                let now_running = running.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(now_running, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(100)).await;
                running.fetch_sub(1, Ordering::SeqCst);
                runs.fetch_add(1, Ordering::SeqCst);
                TaskResult {
                    task_id: task.id.clone(),
                    task_name: task.name.clone(),
                    success: true,
                    output: None,
                    error: None,
                    duration_ms: 100,
                    started_at: Utc::now(),
                    finished_at: Utc::now(),
                }
            })
        });
        let config = SchedulerConfig {
            heartbeat_enabled: false,
            check_interval: Duration::from_millis(10),
            ..SchedulerConfig::default()
        };
        let mut scheduler = Scheduler::new(config).with_executor(executor);
        // Due on every tick: the interval is far below both tick and run time.
        scheduler
            .add_task(recurring_task("sweep", Duration::from_millis(1), "x"))
            .await;
        scheduler.start();
        let ran_twice = eventually(|| async { runs.load(Ordering::SeqCst) >= 3 }).await;
        scheduler.stop().await;

        assert!(ran_twice, "a recurring task keeps running after each run");
        assert_eq!(
            peak.load(Ordering::SeqCst),
            1,
            "never two runs of one task at once"
        );
    }

    /// The heartbeat is a model turn. It used to be awaited inside the loop's
    /// `select!`, so one that took long — here, one that never returns — froze
    /// every other due task; and `interval()` fired it once at boot. Paused
    /// clock: the 30 s period passes instantly.
    #[tokio::test(start_paused = true)]
    async fn a_heartbeat_never_fires_at_boot_and_never_stalls_due_tasks() {
        let heartbeats = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let runs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (heartbeats_in, runs_in) = (heartbeats.clone(), runs.clone());
        let executor: TaskExecutor = Arc::new(move |task: ScheduledTask| {
            let (heartbeats, runs) = (heartbeats_in.clone(), runs_in.clone());
            Box::pin(async move {
                if task.name == "heartbeat" {
                    heartbeats.fetch_add(1, Ordering::SeqCst);
                    std::future::pending::<()>().await;
                }
                runs.fetch_add(1, Ordering::SeqCst);
                TaskResult {
                    task_id: task.id.clone(),
                    task_name: task.name.clone(),
                    success: true,
                    output: None,
                    error: None,
                    duration_ms: 0,
                    started_at: Utc::now(),
                    finished_at: Utc::now(),
                }
            })
        });
        let config = SchedulerConfig {
            heartbeat_enabled: true,
            heartbeat_interval: Duration::from_secs(MIN_HEARTBEAT_INTERVAL_SECS),
            check_interval: Duration::from_millis(10),
            ..SchedulerConfig::default()
        };
        let mut scheduler = Scheduler::new(config).with_executor(executor);
        scheduler
            .add_task(recurring_task("sweep", Duration::from_nanos(1), "x"))
            .await;
        scheduler.start();

        tokio::time::sleep(Duration::from_secs(1)).await;
        assert_eq!(heartbeats.load(Ordering::SeqCst), 0, "no heartbeat at boot");

        tokio::time::sleep(Duration::from_secs(MIN_HEARTBEAT_INTERVAL_SECS + 1)).await;
        assert_eq!(
            heartbeats.load(Ordering::SeqCst),
            1,
            "one heartbeat per period"
        );
        let before = runs.load(Ordering::SeqCst);
        tokio::time::sleep(Duration::from_secs(1)).await;
        let after = runs.load(Ordering::SeqCst);
        scheduler.stop().await;
        assert!(
            after > before,
            "due tasks keep running beside a stuck heartbeat ({before} → {after})"
        );
    }

    /// An edited job must come back edited after a restart. `update_schedule`
    /// wrote only `next_run` (and blanked `last_run`), so the new expression
    /// never reached the store; a payload edit was not even attempted.
    #[tokio::test]
    async fn an_edited_job_survives_a_reload() {
        let storage = Arc::new(Storage::in_memory().await.expect("in-memory storage"));
        let scheduler = Scheduler::new(SchedulerConfig::default()).with_storage(storage.clone());
        let task = Scheduler::cron_task("digest", "0 9 * * *", "old prompt").expect("cron");
        let id = task.id.clone();
        scheduler.add_task(task).await;

        assert!(
            scheduler
                .update_schedule(&id, "30 18 * * *")
                .await
                .expect("valid")
        );
        assert!(scheduler.update_payload(&id, "new prompt").await);
        assert!(scheduler.set_task_enabled(&id, false).await);
        assert!(
            !scheduler.set_task_enabled("no-such-job", true).await,
            "unknown is not updated"
        );
        assert!(!scheduler.update_payload("no-such-job", "x").await);
        assert!(
            !scheduler
                .update_schedule("no-such-job", "0 1 * * *")
                .await
                .expect("valid")
        );

        let reloaded = Scheduler::new(SchedulerConfig::default()).with_storage(storage);
        reloaded.load_jobs().await.expect("load");
        let job = reloaded
            .get_task(&id)
            .await
            .expect("the job is still there");
        assert!(
            matches!(&job.task_type, TaskType::Cron { schedule, .. } if schedule == "30 18 * * *"),
            "{:?}",
            job.task_type
        );
        assert_eq!(job.payload, "new prompt");
        assert!(!job.enabled);
    }

    #[tokio::test]
    async fn one_shots_fire_once_then_vanish_on_success_and_stay_disabled_on_failure() {
        let storage = Arc::new(Storage::in_memory().await.expect("in-memory storage"));
        let calls: Arc<std::sync::Mutex<HashMap<String, usize>>> = Arc::default();
        let calls_in_executor = calls.clone();
        let executor: TaskExecutor = Arc::new(move |task: ScheduledTask| {
            let calls = calls_in_executor.clone();
            Box::pin(async move {
                *calls
                    .lock()
                    .expect("calls")
                    .entry(task.id.clone())
                    .or_default() += 1;
                // Slower than several ticks: without the claim, the next tick
                // would see the one-shot still due and fire it again.
                tokio::time::sleep(Duration::from_millis(150)).await;
                let success = task.payload == "deliverable";
                TaskResult {
                    task_id: task.id.clone(),
                    task_name: task.name.clone(),
                    success,
                    output: None,
                    error: (!success).then(|| "session gone".to_string()),
                    duration_ms: 150,
                    started_at: Utc::now(),
                    finished_at: Utc::now(),
                }
            })
        });
        let config = SchedulerConfig {
            heartbeat_enabled: false,
            check_interval: Duration::from_millis(20),
            ..SchedulerConfig::default()
        };
        let mut scheduler = Scheduler::new(config)
            .with_storage(storage.clone())
            .with_executor(executor);
        let delivered = at_task("reminder", Utc::now(), "deliverable");
        let failed = at_task("reminder", Utc::now(), "undeliverable");
        let (delivered_id, failed_id) = (delivered.id.clone(), failed.id.clone());
        scheduler.add_task(delivered).await;
        scheduler.add_task(failed).await;
        scheduler.start();

        let settled = eventually(|| async {
            scheduler.get_task(&delivered_id).await.is_none()
                && scheduler
                    .get_task(&failed_id)
                    .await
                    .is_some_and(|t| t.run_count == 1)
        })
        .await;
        scheduler.stop().await;
        assert!(settled, "both one-shots should have run and settled");

        let calls = calls.lock().expect("calls").clone();
        assert_eq!(calls.get(&delivered_id), Some(&1), "fired exactly once");
        assert_eq!(calls.get(&failed_id), Some(&1), "fired exactly once");
        let kept = scheduler
            .get_task(&failed_id)
            .await
            .expect("failed one-shot kept");
        assert!(!kept.enabled);

        // Storage agrees, so a restart neither re-fires nor resurrects.
        let restarted = Scheduler::new(SchedulerConfig::default()).with_storage(storage);
        assert_eq!(restarted.load_jobs().await.expect("load"), 1);
        assert!(restarted.get_task(&delivered_id).await.is_none());
        let reloaded = restarted
            .get_task(&failed_id)
            .await
            .expect("failed one-shot persisted");
        assert!(
            !reloaded.enabled,
            "a failed one-shot must not re-arm on restart"
        );
    }

    #[test]
    fn test_parse_interval() {
        assert_eq!(parse_interval("every_300s"), Some(300));
        assert_eq!(parse_interval("every_60s"), Some(60));
        assert_eq!(parse_interval("invalid"), None);
    }

    #[test]
    fn default_heartbeat_prompt_does_not_command_file_read() {
        // Guards against reintroducing the imperative "Read HEARTBEAT.md" that
        // hard-errored on a missing file (os error 2) during the heartbeat cycle.
        let p = SchedulerConfig::default().heartbeat_prompt.to_lowercase();
        assert!(!p.contains("read heartbeat"), "must not command a file read: {p}");
        assert!(!p.contains(".md"), "must not reference a bespoke .md file: {p}");
        assert!(p.contains("heartbeat_ok"), "must keep the HEARTBEAT_OK sentinel: {p}");
    }

    #[test]
    fn runtime_mirrors_the_config_it_was_built_from() {
        let config = SchedulerConfig {
            enabled: false,
            heartbeat_enabled: false,
            heartbeat_interval: Duration::from_mins(10),
            ..SchedulerConfig::default()
        };
        let scheduler = Scheduler::new(config);
        let runtime = scheduler.runtime();

        assert!(!runtime.enabled());
        assert!(!runtime.heartbeat_enabled());
        assert_eq!(runtime.heartbeat_interval(), Duration::from_mins(10));
    }

    #[test]
    fn heartbeat_interval_is_clamped_to_the_scheduler_tick() {
        // Below the scheduler's own 30s resolution the period cannot be
        // honored, and zero would panic `interval()` outright.
        assert_eq!(clamp_heartbeat_secs(0), MIN_HEARTBEAT_INTERVAL_SECS);
        assert_eq!(clamp_heartbeat_secs(1), MIN_HEARTBEAT_INTERVAL_SECS);
        assert_eq!(
            clamp_heartbeat_secs(MIN_HEARTBEAT_INTERVAL_SECS),
            MIN_HEARTBEAT_INTERVAL_SECS
        );
        // At or above the floor the user's value is used verbatim.
        assert_eq!(clamp_heartbeat_secs(1800), 1800);

        // The clamp applies through every entry point, so a running loop can
        // never read a zero period out of the atomics.
        let scheduler = Scheduler::new(SchedulerConfig {
            heartbeat_interval: Duration::ZERO,
            ..SchedulerConfig::default()
        });
        assert_eq!(
            scheduler.runtime().heartbeat_interval(),
            Duration::from_secs(MIN_HEARTBEAT_INTERVAL_SECS)
        );
        scheduler
            .runtime()
            .set_heartbeat_interval(Duration::from_secs(5));
        assert_eq!(
            scheduler.runtime().heartbeat_interval(),
            Duration::from_secs(MIN_HEARTBEAT_INTERVAL_SECS)
        );
    }

    #[test]
    fn apply_settings_updates_both_the_runtime_and_the_config() {
        // The struct must never disagree with the atomics about what the
        // scheduler is doing — `config` is what a reader inspects, `runtime` is
        // what the loop obeys.
        let mut scheduler = Scheduler::new(SchedulerConfig::default());
        let runtime = scheduler.runtime();
        assert!(runtime.enabled() && runtime.heartbeat_enabled());

        scheduler.apply_settings(false, false, Duration::from_mins(15));

        assert!(!runtime.enabled());
        assert!(!runtime.heartbeat_enabled());
        assert_eq!(runtime.heartbeat_interval(), Duration::from_mins(15));
        assert!(!scheduler.config.enabled);
        assert!(!scheduler.config.heartbeat_enabled);
        assert_eq!(
            scheduler.config.heartbeat_interval,
            Duration::from_mins(15)
        );
    }

    #[test]
    fn apply_settings_mirrors_the_clamped_interval_not_the_requested_one() {
        let mut scheduler = Scheduler::new(SchedulerConfig::default());
        scheduler.apply_settings(true, true, Duration::from_secs(1));
        assert_eq!(
            scheduler.config.heartbeat_interval,
            Duration::from_secs(MIN_HEARTBEAT_INTERVAL_SECS)
        );
    }

    #[test]
    fn a_handle_taken_before_a_change_observes_it() {
        // The running loop clones the handle once at startup; a settings change
        // that arrives later must be visible through that same clone, or the
        // toggles would only take effect on a daemon restart.
        let mut scheduler = Scheduler::new(SchedulerConfig::default());
        let handle_held_by_the_loop = scheduler.runtime();

        scheduler.apply_settings(true, false, Duration::from_mins(30));

        assert!(!handle_held_by_the_loop.heartbeat_enabled());
        assert!(handle_held_by_the_loop.enabled());
    }
}
