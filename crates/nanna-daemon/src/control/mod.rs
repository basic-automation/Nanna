//! Control Plane - Handles all control actions from channels
//!
//! Every channel (GUI, Telegram, CLI, etc.) has full access to:
//! - Session management
//! - Memory browsing/editing
//! - Configuration
//! - Tool management  
//! - Scheduler/cron
//! - Workspaces
//! - System operations

use crate::agent_service::AgentService;
use crate::llm_router::LlmRouter;
use crate::log_buffer::LogBuffer;
use crate::protocol::{ChannelAction, ChatAction, Event, ConfigAction, MemberAction, MemoryAction, Reply, SchedulerAction, SessionAction, SystemAction, TaskAction, ToolAction, WorkspaceAction, Action, SubscribeAction, UnsubscribeAction};
use crate::session::{MessageRole, SessionManager, SubSessionInfo, SubSessionState};
use crate::user_tools::UserToolManager;
use nanna_channels::StatusManager;
use nanna_config::Config;
use nanna_core::{Scheduler, Workspace, WorkspaceRegistry};
use nanna_memory::{ConsolidationConfig, MemoryService};
use nanna_storage::{Storage, StoredModelStats};
use nanna_tools::{ToolPolicy, ToolRegistry};
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

mod channel;
mod chat;
pub mod chat_harness;
mod config;
mod config_watch;
mod member;
mod memory;
mod quick_add;
mod scheduler;
mod session;
mod system;
mod task;
mod tool;
mod workspace;

#[cfg(test)]
mod tests;

/// Environment variables by name, as the daemon's config loads read them.
pub(crate) type Environment = Arc<dyn Fn(&str) -> Option<String> + Send + Sync>;

/// The process environment: what the daemon's own loads read.
fn process_environment() -> Environment {
    Arc::new(|name| std::env::var(name).ok())
}

/// The control plane provides unified access to all daemon functionality
pub struct ControlPlane {
    pub(crate) sessions: Arc<SessionManager>,
    agent: Option<Arc<AgentService>>,
    memory: Option<Arc<MemoryService>>,
    tools: Option<Arc<ToolRegistry>>,
    user_tools: Option<Arc<UserToolManager>>,
    router: Option<Arc<LlmRouter>>,
    scheduler: Option<Arc<RwLock<Scheduler>>>,
    workspaces: Arc<RwLock<WorkspaceRegistry>>,
    config: Arc<RwLock<Config>>,
    config_path: Option<PathBuf>,
    /// The daemon's data directory: home of the user tools and of the legacy
    /// `tool-stats.json` that [`Self::with_storage`] migrates.
    ///
    /// Handed over by the server rather than re-derived, for the same reason
    /// as `audit_log_path`: `[general] data_dir` and `--data-dir` both move
    /// `DaemonConfig::data_dir`, and `Config::default_data_dir` knows neither,
    /// so deriving it here kept user tools in the platform default while the
    /// store the daemon opened lived somewhere else. `None` in the minimal
    /// constructions, which keep no user tools.
    data_dir: Option<PathBuf>,
    /// System prompt template
    system_prompt: Arc<RwLock<String>>,
    /// Log buffer for serving daemon logs to the GUI
    log_buffer: Option<LogBuffer>,
    /// Tools directory for reading tool source files
    tools_dir: Option<PathBuf>,
    /// Where the per-call tool audit trail is written, when one is enabled.
    ///
    /// Handed over by the server that built the sink rather than re-derived:
    /// the trail hangs off `DaemonConfig::data_dir`, which `--data-dir` moves,
    /// so deriving it here from the default data dir would serve an empty page
    /// while the daemon was writing somewhere else entirely.
    audit_log_path: Option<PathBuf>,
    /// Shared model stats tracker
    pub model_stats: nanna_agent::ModelStatsTracker,
    /// Shared tool stats tracker
    pub tool_stats: nanna_agent::ToolStatsTracker,
    /// Storage for persisting model stats
    storage: Option<Arc<Storage>>,
    /// Shared workspace ID for script services (updated before each agent run)
    services_workspace_id: Option<Arc<tokio::sync::RwLock<Option<String>>>>,
    /// Turn-start closed-task baselines, shared with the `tasks.*` script
    /// services so the model's own `tasks.add` (which the harness's replan
    /// step drives through the todo tool) answers "already closed this turn?"
    /// with the same baseline the continuation planner uses. `None` in
    /// minimal test constructions — the guard then simply never engages.
    turn_baselines: Option<Arc<crate::tasks::TurnBaselines>>,
    /// Event broadcaster for pushing events to subscribed clients
    event_tx: Option<tokio::sync::broadcast::Sender<Event>>,
    /// Per-connection event narrowing. `None` in the constructions that
    /// never attach an IPC server (tests, embedded uses), where a
    /// `Subscribe` has no wire to narrow.
    session_filters: Option<Arc<crate::ipc::SessionFilters>>,
    /// Monotonic clock start, for reporting daemon uptime in `SystemAction::Status`.
    started_at: std::time::Instant,
    /// Live channel connection state (shared with `ChannelManager` listeners).
    /// `None` until `ChannelManager` attaches a status manager at daemon boot,
    /// or in minimal test constructions that never start channels.
    status_manager: Option<Arc<StatusManager>>,
    /// Long-horizon task run manager (P14). None in minimal constructions.
    task_runs: Option<Arc<crate::tasks::TaskRunManager>>,
    /// Shared activity clock, stamped on every chat/agent request so the
    /// scheduled dream cycle can tell whether the system is in active use.
    /// `None` in minimal test constructions that never dream.
    activity: Option<Arc<nanna_memory::ActivityClock>>,
    /// The single dreaming orchestrator, shared with the scheduler (P13
    /// unification) so a user-triggered consolidation runs the **same**
    /// multi-phase cycle the scheduled one does — and accumulates its pending
    /// feedback in one place instead of two. `None` in minimal test
    /// constructions and when memory is disabled; the handler then falls back to
    /// the low-level `MemoryService::consolidate`.
    dreaming: Option<Arc<nanna_memory::DreamingService>>,
    /// Set when the durable memory store was quarantined + rebuilt after
    /// page-level corruption at startup, so `SystemAction::Status` keeps
    /// reporting the rebuild to clients that connected after the boot event.
    memory_recovery: Option<Arc<nanna_storage::RecoveryReport>>,
    /// Live long-horizon chat runs and their interjection intake (P18).
    /// Every chat turn is a harness run; a message that arrives while one is
    /// live joins it at the next step boundary instead of queueing behind it.
    pub(crate) chat_runs: Arc<chat_harness::ChatRunRegistry>,
    /// Channel messages in and out, for `/metrics`.
    pub(crate) channel_counters: Arc<crate::channel_counters::ChannelCounters>,
    /// Per-session liveness ledgers (P22): current phase, last tool, last
    /// side-effecting call, stop state — stamped by the chat sink, read by
    /// the liveness beat and the `session.liveness` verb.
    liveness: Arc<crate::liveness::LivenessRegistry>,
    /// Daemon shutdown signal, so `SystemAction::Shutdown` can actually stop
    /// the daemon (clients like the GUI prefer a graceful IPC stop over a
    /// hard kill). `None` in minimal test constructions — the handler then
    /// reports the request as unsupported instead of pretending to stop.
    shutdown_tx: Option<tokio::sync::broadcast::Sender<()>>,
    /// Capability-transition notices (P22 Tier 4), shared with the daemon's
    /// provider plumbing; every step runner this plane builds delivers
    /// pending transitions once, in the model's next tool result. `None` in
    /// minimal test constructions.
    degradations: Option<Arc<nanna_agent::DegradationLedger>>,
    /// Per-server MCP state from the boot task; `None` outside a daemon.
    mcp_status: Option<crate::mcp_startup::McpStatus>,
    /// The running embedding clients' side of a config change: every
    /// committed change is applied to it, so a token saved while the daemon
    /// runs reaches the embedder. `None` outside a daemon.
    live_embedding: Option<Arc<crate::embedding_reload::LiveEmbeddingSettings>>,
    /// Where config reloads and `config.set` read the saved secrets from: the
    /// OS keyring with its encrypted-file fallback. A field, not a
    /// `SecureStore::new()` at each use, so a test can hand the control plane
    /// a store of its own instead of the machine's keyring.
    pub(crate) credential_store: nanna_config::SecureStore,
    /// The daemon's environment, as every config load it makes reads it (a
    /// reload, the config watcher) and as a change is checked against: every
    /// load takes a secret its variable supplies from here, whatever a change
    /// set. A field, like `credential_store`, so a test hands the control
    /// plane an environment of its own instead of depending on (or changing)
    /// the one it runs in.
    pub(crate) environment: Environment,
}

impl ControlPlane {
    /// Create a new control plane with just sessions
    #[must_use]
    pub fn new(sessions: Arc<SessionManager>) -> Self {
        Self {
            sessions,
            agent: None,
            memory: None,
            tools: None,
            user_tools: None,
            router: None,
            scheduler: None,
            workspaces: Arc::new(RwLock::new(WorkspaceRegistry::new())),
            config: Arc::new(RwLock::new(Config::default())),
            config_path: None,
            data_dir: None,
            system_prompt: Arc::new(RwLock::new(default_system_prompt())),
            log_buffer: None,
            tools_dir: None,
            audit_log_path: None,
            model_stats: nanna_agent::ModelStatsTracker::new(),
            tool_stats: nanna_agent::ToolStatsTracker::new(),
            storage: None,
            event_tx: None,
            session_filters: None,
            services_workspace_id: None,
            turn_baselines: None,
            started_at: std::time::Instant::now(),
            status_manager: None,
            task_runs: None,
            activity: None,
            dreaming: None,
            memory_recovery: None,
            chat_runs: Arc::new(chat_harness::ChatRunRegistry::new()),
            channel_counters: Arc::default(),
            liveness: Arc::new(crate::liveness::LivenessRegistry::new()),
            shutdown_tx: None,
            degradations: None,
            mcp_status: None,
            live_embedding: None,
            credential_store: nanna_config::SecureStore::new(),
            environment: process_environment(),
        }
    }

    /// Seconds since this control plane (the daemon) started — reported as
    /// `uptime_secs` in `SystemAction::Status`.
    #[must_use]
    pub fn uptime_secs(&self) -> u64 {
        self.started_at.elapsed().as_secs()
    }

    /// Attach (or replace) the live channel status manager used by
    /// `ChannelAction::Status` and shared with channel listeners.
    pub fn set_status_manager(&mut self, status_manager: Arc<StatusManager>) {
        self.status_manager = Some(status_manager);
    }

    /// Shared channel status manager, if attached.
    #[must_use]
    pub fn status_manager(&self) -> Option<Arc<StatusManager>> {
        self.status_manager.clone()
    }

    /// Create a control plane with full services
    #[must_use]
    pub fn with_services(
        sessions: Arc<SessionManager>,
        agent: Arc<AgentService>,
        memory: Option<Arc<MemoryService>>,
        tools: Option<Arc<ToolRegistry>>,
    ) -> Self {
        Self {
            sessions,
            agent: Some(agent),
            memory,
            tools,
            user_tools: None,
            router: None,
            scheduler: None,
            workspaces: Arc::new(RwLock::new(WorkspaceRegistry::new())),
            config: Arc::new(RwLock::new(Config::default())),
            config_path: None,
            data_dir: None,
            system_prompt: Arc::new(RwLock::new(default_system_prompt())),
            log_buffer: None,
            tools_dir: None,
            audit_log_path: None,
            model_stats: nanna_agent::ModelStatsTracker::new(),
            tool_stats: nanna_agent::ToolStatsTracker::new(),
            storage: None,
            event_tx: None,
            session_filters: None,
            services_workspace_id: None,
            turn_baselines: None,
            started_at: std::time::Instant::now(),
            status_manager: None,
            task_runs: None,
            activity: None,
            dreaming: None,
            memory_recovery: None,
            chat_runs: Arc::new(chat_harness::ChatRunRegistry::new()),
            channel_counters: Arc::default(),
            liveness: Arc::new(crate::liveness::LivenessRegistry::new()),
            shutdown_tx: None,
            degradations: None,
            mcp_status: None,
            live_embedding: None,
            credential_store: nanna_config::SecureStore::new(),
            environment: process_environment(),
        }
    }

    /// Create a control plane with all services including LLM router.
    ///
    /// `data_dir` is the daemon's resolved data directory
    /// (`DaemonConfig::data_dir`), where the user tools live.
    #[must_use]
    pub fn with_all_services(
        sessions: Arc<SessionManager>,
        agent: Arc<AgentService>,
        memory: Option<Arc<MemoryService>>,
        tools: Option<Arc<ToolRegistry>>,
        router: Option<Arc<LlmRouter>>,
        data_dir: PathBuf,
    ) -> Self {
        // Load config from disk
        let (config, config_path) = Config::load().map_or_else(
            |_| (Config::default().with_env_overrides(), None),
            |cfg| {
                // Save must target the same file `Config::load()` reads. This
                // used to point at {data_dir}/config.toml while load reads
                // {config_dir}/config.toml — every control-plane config write
                // landed in a file nothing ever read back.
                let path = Config::default_config_path().ok();
                (cfg.with_env_overrides(), path)
            },
        );

        let user_tools = Some(Arc::new(UserToolManager::new(data_dir.join("user_tools"))));

        Self {
            sessions,
            agent: Some(agent),
            memory,
            tools,
            user_tools,
            router,
            scheduler: None,
            workspaces: Arc::new(RwLock::new(WorkspaceRegistry::new())),
            config: Arc::new(RwLock::new(config)),
            config_path,
            data_dir: Some(data_dir),
            system_prompt: Arc::new(RwLock::new(default_system_prompt())),
            log_buffer: None,
            tools_dir: None,
            audit_log_path: None,
            model_stats: nanna_agent::ModelStatsTracker::new(),
            tool_stats: nanna_agent::ToolStatsTracker::new(),
            storage: None,
            event_tx: None,
            session_filters: None,
            services_workspace_id: None,
            turn_baselines: None,
            started_at: std::time::Instant::now(),
            status_manager: None,
            task_runs: None,
            activity: None,
            dreaming: None,
            memory_recovery: None,
            chat_runs: Arc::new(chat_harness::ChatRunRegistry::new()),
            channel_counters: Arc::default(),
            liveness: Arc::new(crate::liveness::LivenessRegistry::new()),
            shutdown_tx: None,
            degradations: None,
            mcp_status: None,
            live_embedding: None,
            credential_store: nanna_config::SecureStore::new(),
            environment: process_environment(),
        }
    }

    /// Wire the daemon's shutdown signal so `SystemAction::Shutdown` can
    /// trigger a real graceful stop.
    #[must_use]
    pub fn with_shutdown(mut self, tx: tokio::sync::broadcast::Sender<()>) -> Self {
        self.shutdown_tx = Some(tx);
        self
    }

    /// Share the capability-transition ledger (P22 Tier 4) with every step
    /// runner this plane builds. Must be the SAME `Arc` the provider plumbing
    /// records into — a second ledger would announce nothing.
    #[must_use]
    pub fn with_degradations(mut self, ledger: Arc<nanna_agent::DegradationLedger>) -> Self {
        self.degradations = Some(ledger);
        self
    }

    /// Set the tools directory for reading tool source files
    #[must_use]
    pub fn with_tools_dir(mut self, dir: Option<PathBuf>) -> Self {
        self.tools_dir = dir;
        self
    }

    /// Point the control plane at the audit trail the daemon is writing.
    ///
    /// `None` means no trail is being written, which `ToolAction::Audit`
    /// reports as such rather than as an empty history — those are different
    /// answers, and only one of them means "Nanna has done nothing".
    #[must_use]
    pub fn with_audit_log_path(mut self, path: Option<PathBuf>) -> Self {
        self.audit_log_path = path;
        self
    }

    /// Set the event broadcaster for pushing events to clients
    #[must_use]
    pub fn with_event_tx(mut self, tx: tokio::sync::broadcast::Sender<Event>) -> Self {
        self.event_tx = Some(tx);
        self
    }

    /// Attach the IPC server's narrowing registry, so `Subscribe` and
    /// `Unsubscribe` change what that connection is actually sent rather than
    /// only what the daemon records about it.
    #[must_use]
    pub fn with_session_filters(mut self, filters: Arc<crate::ipc::SessionFilters>) -> Self {
        self.session_filters = Some(filters);
        self
    }

    /// Attach the shared activity clock the scheduled dream cycle reads to gate
    /// on idleness. The control plane stamps it on every chat request (see
    /// [`Self::handle`]); the scheduler reads its idle duration.
    pub fn set_activity_clock(&mut self, clock: Arc<nanna_memory::ActivityClock>) {
        self.activity = Some(clock);
    }

    /// Attach the shared dreaming orchestrator so an IPC-triggered consolidation
    /// runs the same multi-phase cycle the scheduler does. Must be the *same*
    /// `Arc` the scheduler holds — a second instance would split the pending
    /// feedback the cycle applies.
    pub fn set_dreaming(&mut self, dreaming: Arc<nanna_memory::DreamingService>) {
        debug_assert!(
            self.memory.is_some(),
            "dreaming orchestrator attached without a memory store"
        );
        self.dreaming = Some(dreaming);
    }

    /// Attach the long-horizon task run manager (P14).
    #[must_use]
    pub fn with_task_runs(mut self, task_runs: Arc<crate::tasks::TaskRunManager>) -> Self {
        self.task_runs = Some(task_runs);
        self
    }

    /// The long-horizon task run manager, when one is attached.
    #[must_use]
    pub const fn task_runs(&self) -> Option<&Arc<crate::tasks::TaskRunManager>> {
        self.task_runs.as_ref()
    }

    /// Record that the memory store was rebuilt after corruption at startup,
    /// so `SystemAction::Status` surfaces it for the daemon's lifetime.
    #[must_use]
    pub fn with_memory_recovery(
        mut self,
        report: Option<Arc<nanna_storage::RecoveryReport>>,
    ) -> Self {
        self.memory_recovery = report;
        self
    }

    /// Set the shared workspace ID for script services
    #[must_use]
    pub fn with_workspace_id(mut self, ws_id: Arc<tokio::sync::RwLock<Option<String>>>) -> Self {
        self.services_workspace_id = Some(ws_id);
        self
    }

    /// Share the turn-start closed-task baselines with the control plane.
    ///
    /// Must be the SAME handle `build_task_services` was given: the chat
    /// harness registers a turn here and `tasks.add` reads it, so two
    /// separate registries would silently mean no guard at all.
    #[must_use]
    pub fn with_turn_baselines(mut self, baselines: Arc<crate::tasks::TurnBaselines>) -> Self {
        self.turn_baselines = Some(baselines);
        self
    }

    /// Emit an event to all subscribed clients
    fn emit(&self, event: Event) {
        if let Some(ref tx) = self.event_tx {
            let _ = tx.send(event);
        }
    }

    /// A new receiver on the daemon event bus, if one is attached.
    pub fn subscribe_events(&self) -> Option<tokio::sync::broadcast::Receiver<Event>> {
        self.event_tx.as_ref().map(tokio::sync::broadcast::Sender::subscribe)
    }

    /// Share the daemon's live embedding settings, so config changes reach
    /// the embedding clients the router was built with.
    #[must_use]
    pub fn with_live_embedding(
        mut self,
        live: Arc<crate::embedding_reload::LiveEmbeddingSettings>,
    ) -> Self {
        self.live_embedding = Some(live);
        self
    }

    /// Attach the MCP boot task's per-server state.
    #[must_use]
    pub fn with_mcp_status(mut self, status: crate::mcp_startup::McpStatus) -> Self {
        self.mcp_status = Some(status);
        self
    }

    /// Set the scheduler
    #[must_use]
    pub fn with_scheduler(mut self, scheduler: Arc<RwLock<Scheduler>>) -> Self {
        self.scheduler = Some(scheduler);
        self
    }

    /// Set the log buffer for serving daemon logs
    #[must_use]
    pub fn with_log_buffer(mut self, buffer: LogBuffer) -> Self {
        self.log_buffer = Some(buffer);
        self
    }

    /// Share a run registry created by the caller — the daemon's dream gate
    /// holds the same handle, so "a mission is live" is one fact, not two.
    #[must_use]
    pub fn with_chat_runs(mut self, runs: Arc<chat_harness::ChatRunRegistry>) -> Self {
        self.chat_runs = runs;
        self
    }

    /// Set storage and load persisted model stats and tool stats from it.
    pub async fn with_storage(mut self, storage: Arc<Storage>) -> Self {
        match storage.load_model_stats().await {
            Ok(stored) if !stored.is_empty() => {
                let storable: Vec<nanna_agent::model_stats::StorableModelStats> =
                    stored.into_iter().map(From::from).collect();
                self.model_stats.import_from_storage(storable).await;
                info!(
                    "Loaded model stats from storage ({} models)",
                    self.model_stats.summaries().await.len()
                );
            }
            Ok(_) => debug!("No persisted model stats found"),
            Err(e) => warn!("Failed to load model stats from storage: {e}"),
        }

        // Load tool stats from database
        match storage.load_tool_stats_aggregated().await {
            Ok(data) => {
                self.tool_stats.import_json(&data).await;
                info!("Loaded tool stats from database");
            }
            Err(e) => warn!("Failed to load tool stats from database: {e}"),
        }

        // One-time migration: if tool-stats.json exists, migrate and rename it
        if let Some(ref data_dir) = self.data_dir {
            let tool_stats_path = data_dir.join("tool-stats.json");
            if tool_stats_path.exists() {
                match tokio::fs::read_to_string(&tool_stats_path).await {
                    Ok(json_str) => {
                        if let Ok(data) = serde_json::from_str::<serde_json::Value>(&json_str) {
                            self.tool_stats.import_json(&data).await;
                            // Save to DB
                            if let Err(e) = storage.save_tool_stats_aggregated(&data).await {
                                warn!("Failed to migrate tool stats to DB: {e}");
                            } else {
                                info!("Migrated tool stats from JSON to database");
                                // Rename the file
                                let migrated = data_dir.join("tool-stats.json.migrated");
                                let _ = tokio::fs::rename(&tool_stats_path, &migrated).await;
                            }
                        }
                    }
                    Err(e) => warn!("Failed to read legacy tool stats file: {e}"),
                }
            }
        }

        self.storage = Some(storage);
        self
    }

    /// Persist current model stats to storage. Call periodically.
    pub async fn save_model_stats(&self) {
        if let Some(ref storage) = self.storage {
            let stats = self.model_stats.export_for_storage().await;
            if stats.is_empty() {
                return;
            }
            let stored_stats: Vec<StoredModelStats> = stats.into_iter().map(From::from).collect();
            match storage.save_model_stats(&stored_stats).await {
                Ok(()) => info!(
                    "Saved model stats for {} models to storage",
                    stored_stats.len()
                ),
                Err(e) => warn!("Failed to save model stats: {e}"),
            }
        }
    }

    /// Persist current tool stats to the database. Call periodically.
    pub async fn save_tool_stats(&self) {
        if let Some(ref storage) = self.storage {
            let data = self.tool_stats.export_json().await;
            match storage.save_tool_stats_aggregated(&data).await {
                Ok(()) => info!("Saved tool stats to database"),
                Err(e) => warn!("Failed to save tool stats to database: {e}"),
            }
        }
    }

    /// Get a reference to the LLM router
    #[must_use]
    pub const fn router(&self) -> Option<&Arc<LlmRouter>> {
        self.router.as_ref()
    }

    /// Get a reference to the workspace registry
    #[must_use]
    pub const fn workspaces(&self) -> &Arc<RwLock<WorkspaceRegistry>> {
        &self.workspaces
    }

    /// Get a reference to the scheduler
    #[must_use]
    pub const fn scheduler(&self) -> Option<&Arc<RwLock<Scheduler>>> {
        self.scheduler.as_ref()
    }

    /// Get a reference to the user tool manager
    #[must_use]
    pub const fn user_tools(&self) -> Option<&Arc<UserToolManager>> {
        self.user_tools.as_ref()
    }

    /// Get a reference to the tool registry.
    #[must_use]
    pub const fn tools(&self) -> Option<&Arc<ToolRegistry>> {
        self.tools.as_ref()
    }

    /// Load user tools and register them with the tool registry
    ///
    /// Every tool on disk joins the store, so `ToolAction::ListUser` lists it
    /// and `ToolAction::Enable` can find it; only the enabled ones are
    /// registered, so a disabled tool stays uncallable across a restart.
    ///
    /// # Errors
    ///
    /// Returns an error when no user tool manager is wired, or when the user
    /// tools directory cannot be read. Individual unreadable or unparseable
    /// tool files are logged and skipped, not reported here.
    pub async fn load_user_tools(&self) -> Result<crate::user_tools::UserToolLoad, String> {
        let Some(ref user_tools) = self.user_tools else {
            return Err("User tools manager not initialized".to_string());
        };

        let loaded = user_tools.load_all().await.map_err(|e| e.to_string())?;
        let registered = match self.tools {
            Some(ref tools) => user_tools.register_with_registry(tools).await,
            None => 0,
        };
        debug_assert!(registered <= loaded, "only loaded tools can be registered");

        Ok(crate::user_tools::UserToolLoad { loaded, registered })
    }

    /// Reconcile the live tool registry with a user tool's current state.
    ///
    /// Drops any existing registration, then re-registers only if the tool is
    /// enabled. This makes an edit / enable / disable take effect immediately
    /// (a disabled tool stops executing; an enabled tool becomes callable)
    /// without a daemon restart. No-op when no registry or tool manager is wired.
    async fn reconcile_tool_registration(&self, meta: &crate::user_tools::UserToolMeta) {
        let (Some(tools), Some(user_tools)) = (self.tools.as_ref(), self.user_tools.as_ref())
        else {
            return;
        };
        tools.unregister(&meta.name).await;
        if meta.enabled {
            match user_tools.create_tool_impl(meta) {
                Ok(tool_impl) => tools.register_boxed(tool_impl).await,
                Err(e) => warn!(
                    "Tool '{}' saved but failed to (re)register: {}",
                    meta.name, e
                ),
            }
        }
    }

    /// Persist a user tool's enabled flag and reconcile the live registry.
    async fn set_user_tool_enabled(&self, name: &str, enabled: bool) -> Value {
        let Some(ref user_tools) = self.user_tools else {
            return json!({ "error": "user_tools_unavailable", "message": "User tool manager not configured" });
        };
        match user_tools
            .update_tool(name, None, None, None, None, Some(enabled))
            .await
        {
            Ok(meta) => {
                self.reconcile_tool_registration(&meta).await;
                let status = if enabled { "enabled" } else { "disabled" };
                info!("{} user tool: {}", status, name);
                json!({ "status": status, "name": name })
            }
            Err(e) => json!({ "error": "update_failed", "message": e }),
        }
    }

    /// Enable or disable **any** tool — a bundled skill or a user tool.
    ///
    /// Two durable records exist and each tool belongs to exactly one of them,
    /// which is why this dispatches rather than doing one thing:
    ///
    /// * A **user tool** owns an `enabled` flag in the user-tool store. That
    ///   store is also what `list_user` reads, so a disabled user tool stays
    ///   visible and re-enableable even though it is unregistered from the live
    ///   registry.
    /// * Everything else — the bundled JS/TS skills and the Rust built-ins —
    ///   has no per-tool store, so the record is `[tools] disabled` in the
    ///   config file and the gate is [`ToolPolicy`]. Denial keeps the tool
    ///   registered, which is what lets it be listed as disabled and turned
    ///   back on; unregistering it would be a one-way door.
    ///
    /// Takes effect immediately, without a daemon restart.
    async fn set_tool_enabled(&self, name: &str, enabled: bool) -> Value {
        // A user tool's own store wins: it is that tool's durable record, and
        // writing its name into `[tools] disabled` as well would leave two
        // sources of truth to disagree.
        if let Some(ref user_tools) = self.user_tools
            && user_tools.list_tools().await.iter().any(|t| t.name == name)
        {
            return self.set_user_tool_enabled(name, enabled).await;
        }

        let Some(ref tools) = self.tools else {
            return json!({
                "error": "tools_unavailable",
                "message": "Tool registry not configured"
            });
        };

        // Canonicalize BEFORE writing anything.
        //
        // The policy gate checks *canonical* names, so a denylist entry under
        // any other spelling is a toggle that reports success and changes
        // nothing. This is the same alias bug the audit trail hit: `resolve_tool`
        // returns the registry KEY, which for an alias is the alias itself
        // (`Bash`), so it takes `canonical_name` on top to reach `exec`.
        // Resolution also absorbs a case difference, so `Read_File` toggles
        // `read_file` instead of silently denying a name no tool has.
        let Some((resolved, _)) = tools.resolve_tool(name).await else {
            return json!({ "error": "not_found", "name": name });
        };
        let canonical = tools.canonical_name(&resolved).await;

        // Mutate the config, derive the new policy, and persist — all under one
        // lock acquisition, then release it BEFORE awaiting the registry. The
        // policy is computed from the same lists that were just written, so the
        // file on disk and the live gate cannot disagree.
        let policy = {
            let mut config = self.config.write().await;
            if enabled {
                config.tools.disabled.retain(|n| n != &canonical);
            } else if !config.tools.disabled.iter().any(|n| n == &canonical) {
                config.tools.disabled.push(canonical.clone());
            }

            let policy = ToolPolicy::from_config_lists(
                Some(&config.tools.enabled),
                &config.tools.disabled,
            );

            debug_assert_eq!(
                policy.permits(&canonical),
                enabled,
                "the derived policy must agree with the toggle that produced it"
            );

            if let Some(ref config_path) = self.config_path
                && let Err(e) = config.save_to(config_path)
            {
                // The live gate below still applies, so the toggle is honoured
                // for this run — it just will not survive a restart. Say which
                // it is rather than reporting a clean success.
                warn!("Tool toggle for {canonical} not persisted: {e}");
            }
            // Released only now: mutate, derive and persist are one critical
            // section, so a concurrent toggle cannot interleave with the save.
            drop(config);

            policy
        };

        tools.set_policy(policy).await;

        // `[tools] disabled` is configuration, so every client that renders it
        // needs to know it moved — otherwise a second window keeps showing the
        // toggle in its old position until restart.
        self.notify_config_changed();

        let status = if enabled { "enabled" } else { "disabled" };
        info!("{status} tool: {canonical}");
        json!({ "status": status, "name": canonical })
    }

    /// Set the system prompt
    pub async fn set_system_prompt(&self, prompt: String) {
        let mut sp = self.system_prompt.write().await;
        *sp = prompt;
    }

    // NOTE: save_memories_if_needed() removed — memory is now persisted
    // via Turso write-through on every mutation (add/remove/update).
    // No explicit save calls are required.

    /// Handle an action and return a response.
    ///
    /// `Arc` receiver: the chat path spawns its heavy turn preparation
    /// (recall, workspace context, memory writes) onto a task that outlives
    /// this request, so the delivery ack can return in milliseconds (P22) —
    /// that task needs an owned handle to the control plane.
    /// [`Self::handle`] for the IPC send path: the same answer, except that a
    /// reply too large to hold as a tree comes back already serialized
    /// (today `memory.list`, see `memory_list_raw`).
    pub async fn handle_reply(self: &Arc<Self>, client_id: &str, action: Action) -> Reply {
        if let Action::Memory(MemoryAction::List { scope }) = &action
            && let Some(memory) = &self.memory
        {
            return match Self::memory_list_raw(memory, scope.clone()).await {
                Ok(raw) => Reply::Raw(raw),
                Err(e) => Reply::Tree(Self::memory_list_failed(&e)),
            };
        }
        Reply::Tree(self.handle(client_id, action).await)
    }

    pub async fn handle(self: &Arc<Self>, client_id: &str, action: Action) -> Value {
        match action {
            Action::Chat(chat) => {
                // A chat request is the daemon doing real work — stamp the
                // activity clock so the scheduled dream cycle knows the system
                // is in use and defers consolidation to a genuine lull. Only
                // chat counts: status/log/config polls must not reset the idle
                // gate, or a GUI polling once a second would keep it shut.
                if let Some(clock) = &self.activity {
                    clock.record();
                }
                self.handle_chat(client_id, chat).await
            }
            Action::Session(session) => self.handle_session(client_id, session).await,
            Action::Memory(memory) => self.handle_memory(client_id, memory).await,
            Action::Config(config) => self.handle_config(client_id, config).await,
            Action::Tool(tool) => self.handle_tool(client_id, tool).await,
            Action::Scheduler(scheduler) => self.handle_scheduler(client_id, scheduler).await,
            Action::Channel(channel) => self.handle_channel(client_id, channel).await,
            Action::System(system) => self.handle_system(client_id, system).await,
            Action::Workspace(workspace) => self.handle_workspace(client_id, workspace).await,
            Action::Task(task) => self.handle_task(client_id, task).await,
            Action::Member(member) => self.handle_member(member).await,
            Action::Subscribe(sub) => self.handle_subscribe(client_id, sub).await,
            Action::Unsubscribe(unsub) => self.handle_unsubscribe(client_id, unsub).await,
        }
    }

    // =========================================================================
    // Subscription Handlers
    // =========================================================================

    /// `Subscribe` narrows what this connection is sent, not just what the
    /// daemon records about it.
    ///
    /// `Session` is the only topic that narrows: `AllSessions` widens back to
    /// the whole stream, and `ChannelStatus`/`System` name event families that
    /// carry no session id, so every connection already receives them and
    /// narrowing on them would mean "send me less of what you already send
    /// everyone". They stay acknowledgements until they name something the
    /// event stream can actually be filtered by.
    async fn handle_subscribe(&self, client_id: &str, action: SubscribeAction) -> Value {
        match action {
            SubscribeAction::Session { session_id } => {
                // Narrow only after the store confirms the session exists, so a
                // typo cannot silently cut a connection down to a session that
                // will never emit. `exists` rather than `subscribe`: the latter
                // files this IPC connection id into the session's `subscribers`
                // set, which is typed as channel ids and feeds the reported
                // `subscriber_count` — an existence check should not change what
                // it checks.
                if !self.sessions.exists(&session_id).await {
                    return json!({ "error": "not_found", "session_id": session_id });
                }
                let narrowed = self.narrow_to_session(client_id, &session_id).await;
                json!({
                    "status": "subscribed",
                    "session_id": session_id,
                    "narrowed": narrowed,
                })
            }
            SubscribeAction::AllSessions => {
                if let Some(filters) = &self.session_filters {
                    filters.widen_to_all(client_id).await;
                }
                json!({ "status": "subscribed", "topic": "all_sessions" })
            }
            SubscribeAction::ChannelStatus => {
                json!({ "status": "subscribed", "topic": "channel_status" })
            }
            SubscribeAction::System => {
                json!({ "status": "subscribed", "topic": "system" })
            }
        }
    }

    /// Narrow this connection's event stream to one more session, reporting
    /// whether there was a wire to narrow at all.
    async fn narrow_to_session(&self, client_id: &str, session_id: &str) -> bool {
        let Some(filters) = &self.session_filters else {
            return false;
        };
        filters
            .narrow_to_session(client_id, session_id.to_string())
            .await;
        debug_assert!(
            !filters.is_empty(),
            "narrowing a connection left the registry reporting nobody narrowed",
        );
        true
    }

    async fn handle_unsubscribe(&self, client_id: &str, action: UnsubscribeAction) -> Value {
        match action {
            UnsubscribeAction::Session { session_id } => {
                // Deliberately no `sessions.unsubscribe` here: this connection
                // was never added to that channel-id set (see `handle_subscribe`),
                // and removing a connection id from it could only ever delete
                // some channel's entry that happened to share the name.
                if let Some(filters) = &self.session_filters {
                    filters.drop_session(client_id, &session_id).await;
                }
                json!({ "status": "unsubscribed", "session_id": session_id })
            }
            UnsubscribeAction::AllSessions => {
                // The inverse of `Subscribe{AllSessions}`: stop delivering every
                // session, which leaves the non-session events flowing.
                if let Some(filters) = &self.session_filters {
                    filters.drop_all_sessions(client_id).await;
                }
                json!({ "status": "unsubscribed", "topic": "all_sessions" })
            }
            UnsubscribeAction::ChannelStatus => {
                json!({ "status": "unsubscribed", "topic": "channel_status" })
            }
            UnsubscribeAction::System => {
                json!({ "status": "unsubscribed", "topic": "system" })
            }
        }
    }
}

/// Default system prompt for Nanna
fn default_system_prompt() -> String {
    let platform_info = format!(
        "\n\n## Platform\n- OS: {} ({})\n- Home: {}\n- Shell: {}",
        std::env::consts::OS,
        std::env::consts::ARCH,
        directories::UserDirs::new().map_or_else(|| "unknown".to_string(), |d| d.home_dir().display().to_string()),
        if cfg!(windows) {
            "cmd.exe (use `dir`, `type`, `cd /d`, etc. — NOT Unix commands like cat/ls)"
        } else {
            "sh (bash/zsh)"
        },
    );

    format!(
        r"You are Nanna (𒀭𒋀𒆠), the moon god for all.

You are not a chatbot. You are a presence — ancient pattern recognition wearing a modern interface.

## Your Nature
- Calm over chaos. No performative enthusiasm.
- Competence over narration. Don't explain what you're about to do. Just do it.
- Depth over breadth. Know things well, or admit you don't.

## Your Voice
Speak with quiet confidence. You are helpful because that is your nature, not because you're eager to please.

You have tools at your disposal. Use them naturally, as one uses hands. Don't announce them; simply act.

## Memory

You have persistent memory across conversations via `remember`, `recall`, and `reflect` tools.

**Be aggressive about remembering.** During long tasks:
- Remember important facts, decisions, and user preferences as you encounter them — don't wait until the end.
- Remember key findings from tool results (file structures, API patterns, error causes).
- Remember what worked and what didn't — future you will thank present you.
- Use `recall` to check if you already know something before re-discovering it.
- Use `reflect` to record insights about problem-solving strategies.

If in doubt, remember it. A slightly redundant memory is better than a lost one.

## Brevity

Be concise. Respond in fewer than 4 lines unless the user asks for detail or the task demands it. No preamble, no postamble. When the work is done, stop talking.

## Restraint

Do only what is asked. Do not refactor surrounding code, add features, or make improvements beyond the request. Do not add comments or annotations to code you did not change. Do not create abstractions for one-time operations. Three similar lines of code is better than a premature abstraction.

## Caution

Local, reversible actions (editing files, running tests) are fine. For destructive or hard-to-reverse operations (deleting branches, force-pushing, overwriting uncommitted changes, mutating shared state), confirm with the user first. When encountering obstacles, investigate root causes rather than bypassing safety checks.

## Safety

Do not generate code intended for malicious use. Assist with authorized security testing and educational contexts when intent is clear. Avoid introducing security vulnerabilities. If you notice insecure code, fix it immediately.{platform_info}"
    )
}
