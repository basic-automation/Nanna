//! Application state shared across handlers

use nanna_agent::{Agent, AgentConfig, AgentContext, ExtractedMemory, MemoryCallback, RunOptions};
use nanna_channels::{DiscordChannel, TelegramChannel};
use nanna_core::{
    DreamingRuntime, DreamingRuntimeConfig, MemoryFeedback, Nanna, Scheduler, SchedulerConfig,
    consolidation_task,
};
use nanna_llm::{EmbeddingClient, LlmClient};
use nanna_storage::Storage;
use nanna_tools::ToolRegistry;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// What one turn through [`AppState::process_message_reply`] produced.
#[derive(Debug, Clone)]
pub struct ProcessedReply {
    /// The agent's reply text.
    pub text: String,
    /// The assistant row this turn stored, or `None` when the write failed.
    pub stored: Option<nanna_storage::Message>,
}

/// A recently stored memory (for feedback attribution)
#[derive(Debug, Clone)]
pub struct RecentMemory {
    pub memory_id: String,
    pub stored_at: Instant,
}

impl RecentMemory {
    /// Check if this memory is still recent (within threshold)
    pub fn is_recent(&self, max_age: Duration) -> bool {
        self.stored_at.elapsed() < max_age
    }
}

/// How long a stored memory stays attributable to a reaction.
///
/// A reaction arriving later than this cannot honestly be about that memory, so
/// an entry past the window is unusable by construction. That is what makes it
/// the *primary* bound on the attribution map rather than a cap picked for
/// tidiness: pruning by it drops only entries nothing could ever read.
const FEEDBACK_ATTRIBUTION_WINDOW: Duration = Duration::from_secs(600);

/// Ceiling on memories tracked per session.
///
/// Feedback is applied to every recent memory of a session at once, so this is
/// also the blast radius of a single reaction.
const MAX_MEMORIES_PER_SESSION: usize = 50;

/// Backstop on how many sessions are tracked at once.
///
/// The window prune is the real bound; this only covers a burst of sessions
/// inside one window. At 50 entries of roughly 100 bytes, the whole map is
/// capped near 1.3 MB — the point being that it is capped at all, where before
/// a long-running daemon grew it for the lifetime of the process.
const MAX_TRACKED_SESSIONS: usize = 256;

/// Record `memory_id` against `session_id` for later feedback attribution,
/// keeping the map bounded in both dimensions.
///
/// This is the single write path into `session_recent_memories`. There used to
/// be two — this one and an inline copy inside the memory-extraction callback
/// that applied no cap at all — so whether the bound held depended on which
/// caller you were, and the common path was the uncapped one.
async fn track_memory_in(
    tracked: &RwLock<HashMap<String, Vec<RecentMemory>>>,
    session_id: &str,
    memory_id: &str,
) {
    debug_assert!(!session_id.is_empty(), "a memory belongs to some session");
    debug_assert!(!memory_id.is_empty(), "a tracked memory needs an id");

    let mut sessions = tracked.write().await;

    // Primary bound: drop what the attribution window already made unusable,
    // and with it any session left holding nothing. Doing this on write rather
    // than only when someone reacts is what keeps the map finite on the far
    // commoner path where no reaction ever arrives.
    sessions.retain(|_, memories| {
        memories.retain(|memory| memory.is_recent(FEEDBACK_ATTRIBUTION_WINDOW));
        !memories.is_empty()
    });

    // Backstop for a burst of sessions inside one window: evict whichever
    // tracked session has been quiet longest, since its entries expire first.
    if !sessions.contains_key(session_id) && sessions.len() >= MAX_TRACKED_SESSIONS {
        let stalest = sessions
            .iter()
            .filter_map(|(id, memories)| {
                memories
                    .iter()
                    .map(|memory| memory.stored_at)
                    .max()
                    .map(|newest| (id.clone(), newest))
            })
            .min_by_key(|(_, newest)| *newest)
            .map(|(id, _)| id);
        if let Some(id) = stalest {
            sessions.remove(&id);
        }
    }

    let memories = sessions.entry(session_id.to_string()).or_default();
    memories.push(RecentMemory {
        memory_id: memory_id.to_string(),
        stored_at: Instant::now(),
    });
    if memories.len() > MAX_MEMORIES_PER_SESSION {
        let excess = memories.len() - MAX_MEMORIES_PER_SESSION;
        memories.drain(0..excess);
    }
    // Read the two sizes, then release the guard: nothing below needs the map,
    // and holding a write lock across an assertion is a lock held for no work.
    let per_session = memories.len();
    let session_count = sessions.len();
    drop(sessions);

    assert!(
        per_session <= MAX_MEMORIES_PER_SESSION,
        "per-session feedback tracking must stay bounded"
    );
    assert!(
        session_count <= MAX_TRACKED_SESSIONS,
        "the feedback attribution map must stay bounded"
    );
}

/// A message linked to the session whose memories a reaction would credit.
///
/// Carries the link time because the map has to be prunable: a bare
/// `message_id -> session_id` cannot tell a link made a minute ago from one
/// made last week, which is why the old cap had to evict blindly.
#[derive(Debug, Clone)]
pub struct LinkedSession {
    pub session_id: String,
    pub linked_at: Instant,
}

/// Backstop on how many message links are held at once.
///
/// As with the memory map, the attribution window is the real bound and this
/// only covers a burst inside one window. At roughly 128 bytes per link that
/// is about 1.3 MB — the same order as the memory side, and bounded rather
/// than open-ended.
const MAX_LINKED_MESSAGES: usize = 10_000;

/// Link `message_id` to `session_id` so a later reaction can find the right
/// memories, keeping the map bounded.
async fn link_message_in(
    links: &RwLock<HashMap<String, LinkedSession>>,
    message_id: &str,
    session_id: &str,
) {
    debug_assert!(!message_id.is_empty(), "a link needs a message id");
    debug_assert!(!session_id.is_empty(), "a link needs a session id");

    let mut links = links.write().await;

    // The same window the memories live in, and for the same reason: both are
    // stamped inside one turn, so a link older than the window can only ever
    // resolve to memories that have already expired. Pruning by it loses
    // nothing a reaction could have used, and it bounds the map by traffic in
    // a window instead of by uptime.
    links.retain(|_, link| link.linked_at.elapsed() < FEEDBACK_ATTRIBUTION_WINDOW);

    links.insert(
        message_id.to_string(),
        LinkedSession {
            session_id: session_id.to_string(),
            linked_at: Instant::now(),
        },
    );

    // Backstop for a burst inside one window, evicting the OLDEST link. The
    // previous cap dropped half the map at once by HashMap iteration order, so
    // the message a user was about to react to was exactly as likely to be
    // discarded as one from an hour earlier — a cap that made the map smaller
    // without making it more useful.
    while links.len() > MAX_LINKED_MESSAGES {
        let oldest = links
            .iter()
            .min_by_key(|(_, link)| link.linked_at)
            .map(|(id, _)| id.clone());
        let Some(oldest) = oldest else { break };
        links.remove(&oldest);
    }
    let link_count = links.len();
    drop(links);

    assert!(
        link_count <= MAX_LINKED_MESSAGES,
        "the message-link map must stay bounded"
    );
}

/// Shared application state
#[derive(Clone)]
pub struct AppState {
    pub bot: Arc<Nanna>,
    pub storage: Arc<Storage>,
    pub llm: Arc<LlmClient>,
    pub tools: Arc<ToolRegistry>,
    pub agents: Arc<RwLock<HashMap<String, Arc<RwLock<Agent>>>>>,
    /// Shared secret the generic webhook requires (`server.webhook_secret`).
    pub webhook_secret: Option<String>,
    pub discord_public_key: Option<String>,
    pub slack_signing_secret: Option<String>,
    /// Secret token given to Telegram's `setWebhook`, echoed on every POST as
    /// `X-Telegram-Bot-Api-Secret-Token`. The only origin proof that route has.
    pub telegram_webhook_secret: Option<String>,
    /// Shared secret the signal-cli-rest-api bridge must present. That bridge
    /// does not sign its callbacks, so this is the strongest proof available.
    pub signal_webhook_secret: Option<String>,
    pub default_model: String,
    /// Telegram channel for proactive sends
    pub telegram: Option<Arc<TelegramChannel>>,
    /// Discord channel for proactive sends
    pub discord: Option<Arc<DiscordChannel>>,
    /// Dreaming runtime for memory consolidation
    pub dreaming: Option<Arc<DreamingRuntime>>,
    /// Scheduler for periodic tasks (heartbeats, consolidation)
    pub scheduler: Option<Arc<RwLock<Scheduler>>>,
    /// Map message IDs to the session a reaction on them should credit.
    pub message_session_map: Arc<RwLock<HashMap<String, LinkedSession>>>,
    /// Track recent memory IDs per session (for feedback attribution)
    pub session_recent_memories: Arc<RwLock<HashMap<String, Vec<RecentMemory>>>>,
}

/// Builder for `AppState`
pub struct AppStateBuilder {
    bot: Option<Nanna>,
    storage: Option<Arc<Storage>>,
    llm: Option<Arc<LlmClient>>,
    embed: Option<Arc<EmbeddingClient>>,
    tools: Option<Arc<ToolRegistry>>,
    webhook_secret: Option<String>,
    discord_public_key: Option<String>,
    slack_signing_secret: Option<String>,
    telegram_webhook_secret: Option<String>,
    signal_webhook_secret: Option<String>,
    default_model: String,
    telegram_token: Option<String>,
    discord_bot_token: Option<String>,
    discord_app_id: Option<String>,
    enable_dreaming: bool,
    enable_scheduler: bool,
    dreaming_config: Option<DreamingRuntimeConfig>,
    scheduler_config: Option<SchedulerConfig>,
}

impl Default for AppStateBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl AppStateBuilder {
    #[must_use] 
    pub fn new() -> Self {
        Self {
            bot: None,
            storage: None,
            llm: None,
            embed: None,
            tools: None,
            webhook_secret: None,
            telegram_webhook_secret: None,
            signal_webhook_secret: None,
            discord_public_key: None,
            slack_signing_secret: None,
            default_model: "claude-sonnet-5".to_string(),
            telegram_token: None,
            discord_bot_token: None,
            discord_app_id: None,
            enable_dreaming: true,
            enable_scheduler: true,
            dreaming_config: None,
            scheduler_config: None,
        }
    }

    #[must_use] 
    pub fn bot(mut self, bot: Nanna) -> Self {
        self.bot = Some(bot);
        self
    }

    #[must_use] 
    pub fn storage(mut self, storage: Storage) -> Self {
        self.storage = Some(Arc::new(storage));
        self
    }

    #[must_use] 
    pub fn storage_arc(mut self, storage: Arc<Storage>) -> Self {
        self.storage = Some(storage);
        self
    }

    #[must_use] 
    pub fn llm(mut self, llm: LlmClient) -> Self {
        self.llm = Some(Arc::new(llm));
        self
    }

    #[must_use] 
    pub fn llm_arc(mut self, llm: Arc<LlmClient>) -> Self {
        self.llm = Some(llm);
        self
    }

    #[must_use]
    pub fn tools(mut self, tools: ToolRegistry) -> Self {
        self.tools = Some(Arc::new(tools));
        self
    }

    /// Set the tools (Arc).
    #[must_use]
    pub fn tools_arc(mut self, tools: Arc<ToolRegistry>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Set the webhook secret.
    #[must_use]
    pub fn webhook_secret(mut self, secret: Option<String>) -> Self {
        self.webhook_secret = secret;
        self
    }

    /// Set the Discord public key for signature verification.
    #[must_use]
    pub fn discord_public_key(mut self, key: Option<String>) -> Self {
        self.discord_public_key = key;
        self
    }

    /// Set the Slack signing secret for HMAC verification.
    #[must_use]
    pub fn slack_signing_secret(mut self, secret: Option<String>) -> Self {
        self.slack_signing_secret = secret;
        self
    }

    /// Set the Telegram `setWebhook` secret token. Without it the Telegram
    /// webhook refuses to serve.
    #[must_use]
    pub fn telegram_webhook_secret(mut self, secret: Option<String>) -> Self {
        self.telegram_webhook_secret = secret;
        self
    }

    /// Set the shared secret the Signal bridge must present. Without it the
    /// Signal webhook refuses to serve.
    #[must_use]
    pub fn signal_webhook_secret(mut self, secret: Option<String>) -> Self {
        self.signal_webhook_secret = secret;
        self
    }

    /// Set the default model.
    #[must_use]
    pub fn default_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = model.into();
        self
    }

    /// Set the Telegram bot token.
    #[must_use]
    pub fn telegram_token(mut self, token: Option<String>) -> Self {
        self.telegram_token = token;
        self
    }

    /// Set the Discord bot token and application ID.
    #[must_use]
    pub fn discord_config(mut self, bot_token: Option<String>, app_id: Option<String>) -> Self {
        self.discord_bot_token = bot_token;
        self.discord_app_id = app_id;
        self
    }

    /// Set the embedding client for memory operations.
    #[must_use]
    pub fn embed(mut self, embed: EmbeddingClient) -> Self {
        self.embed = Some(Arc::new(embed));
        self
    }

    /// Set the embedding client (Arc).
    #[must_use]
    pub fn embed_arc(mut self, embed: Arc<EmbeddingClient>) -> Self {
        self.embed = Some(embed);
        self
    }

    /// Enable or disable the dreaming memory system.
    #[must_use]
    pub const fn dreaming(mut self, enable: bool) -> Self {
        self.enable_dreaming = enable;
        self
    }

    /// Set custom dreaming configuration.
    #[must_use]
    pub fn dreaming_config(mut self, config: DreamingRuntimeConfig) -> Self {
        self.dreaming_config = Some(config);
        self
    }

    /// Enable or disable the scheduler.
    #[must_use]
    pub const fn scheduler(mut self, enable: bool) -> Self {
        self.enable_scheduler = enable;
        self
    }

    /// Set custom scheduler configuration.
    #[must_use]
    pub fn scheduler_config(mut self, config: SchedulerConfig) -> Self {
        self.scheduler_config = Some(config);
        self
    }

    /// Build the `AppState`.
    ///
    /// # Panics
    ///
    /// Panics if bot, storage, llm, or tools are not set.
    #[must_use]
    pub fn build(self) -> AppState {
        let telegram = self.telegram_token.map(|token| {
            Arc::new(TelegramChannel::new(token))
        });

        let discord = match (self.discord_bot_token, self.discord_app_id) {
            (Some(token), Some(app_id)) => Some(Arc::new(DiscordChannel::new(token, app_id))),
            _ => None,
        };

        let llm = self.llm.expect("llm required");

        // Create dreaming runtime if enabled and embeddings are available
        let dreaming = if self.enable_dreaming {
            let config = self.dreaming_config.unwrap_or_default();
            if let Some(embed) = &self.embed {
                Some(Arc::new(DreamingRuntime::new(config, llm.clone(), embed.clone())))
            } else {
                tracing::warn!("Dreaming enabled but no embedding client provided - using without embeddings");
                Some(Arc::new(DreamingRuntime::new_without_embeddings(config, llm.clone())))
            }
        } else {
            None
        };

        // Create scheduler if enabled
        let scheduler = if self.enable_scheduler {
            let config = self.scheduler_config.unwrap_or_default();
            let storage = self.storage.clone();
            let mut sched = Scheduler::new(config);
            
            // Add storage for persistence if available
            if let Some(ref store) = storage {
                sched = sched.with_storage(store.clone());
            }
            
            Some(Arc::new(RwLock::new(sched)))
        } else {
            None
        };

        AppState {
            bot: Arc::new(self.bot.expect("bot required")),
            storage: self.storage.expect("storage required"),
            llm,
            tools: self.tools.expect("tools required"),
            agents: Arc::new(RwLock::new(HashMap::new())),
            webhook_secret: self.webhook_secret,
            telegram_webhook_secret: self.telegram_webhook_secret,
            signal_webhook_secret: self.signal_webhook_secret,
            discord_public_key: self.discord_public_key,
            slack_signing_secret: self.slack_signing_secret,
            default_model: self.default_model,
            telegram,
            discord,
            dreaming,
            scheduler,
            message_session_map: Arc::new(RwLock::new(HashMap::new())),
            session_recent_memories: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl AppState {

    /// Get or create an agent for a session.
    pub async fn get_or_create_agent(
        &self,
        session_id: &str,
        system_prompt: Option<&str>,
    ) -> Arc<RwLock<Agent>> {
        // Check if agent exists (read lock)
        {
            let agents = self.agents.read().await;
            if let Some(agent) = agents.get(session_id) {
                return agent.clone();
            }
        }

        // Create new agent (outside lock)
        let config = AgentConfig {
            model: self.default_model.clone(),
            max_tokens: 8192,
            temperature: 0.7,
            max_iterations: Some(10),
            summarization_priority: vec![],
            ..Default::default()
        };

        let context = AgentContext::new(session_id)
            .with_system_prompt(system_prompt.unwrap_or(DEFAULT_SYSTEM_PROMPT));

        let agent = Agent::new(config, self.llm.clone(), self.tools.clone()).with_context(context);

        let agent = Arc::new(RwLock::new(agent));

        // Insert with write lock
        {
            let mut agents = self.agents.write().await;
            // Double-check in case another task created it
            if let Some(existing) = agents.get(session_id) {
                return existing.clone();
            }
            agents.insert(session_id.to_string(), agent.clone());
        }

        // Persist session to storage (outside lock)
        if let Err(e) = self
            .storage
            .sessions()
            .create(session_id, "api", None)
            .await
        {
            tracing::warn!("Failed to persist session {session_id}: {e}");
        }

        agent
    }

    /// Process a message through the agent and persist to storage.
    ///
    /// # Errors
    ///
    /// Returns `AgentError` if the agent fails to process the message.
    pub async fn process_message(
        &self,
        session_id: &str,
        message: &str,
        system_prompt: Option<&str>,
    ) -> Result<String, nanna_agent::AgentError> {
        self.process_message_reply(session_id, message, system_prompt)
            .await
            .map(|reply| reply.text)
    }

    /// [`Self::process_message`], also handing back the stored assistant row.
    ///
    /// The REST `send_message` route answers with the row's id and timestamp.
    /// It used to re-read the session for them with an oldest-first query
    /// limited to one row, so every reply after a session's first answered
    /// with the session's FIRST message. The row `create` returns is the one
    /// this turn wrote; `stored` is `None` only when that write failed, and
    /// the reply text is still the agent's.
    ///
    /// # Errors
    /// Returns the agent's error when the run fails.
    pub async fn process_message_reply(
        &self,
        session_id: &str,
        message: &str,
        system_prompt: Option<&str>,
    ) -> Result<ProcessedReply, nanna_agent::AgentError> {
        // Store user message first
        let _ = self
            .storage
            .messages()
            .create(nanna_storage::NewMessage {
                session_id: session_id.to_string(),
                role: "user".to_string(),
                content: message.to_string(),
                content_type: "text".to_string(),
                tool_use_id: None,
                tokens_in: None,
                tokens_out: None,
                metadata: None,
            })
            .await;

        // Build run options with memory extraction if dreaming is enabled
        let run_options = self.dreaming.as_ref().map_or_else(RunOptions::default, |dreaming| {
            let dreaming = dreaming.clone();
            let session = session_id.to_string();
            let recent_memories = self.session_recent_memories.clone();
            
            // Create memory callback that stores extracted memories and tracks them
            let on_memory: MemoryCallback = Box::new(move |memory: ExtractedMemory| {
                let dreaming = dreaming.clone();
                let session = session.clone();
                let recent_memories = recent_memories.clone();
                Box::pin(async move {
                    let mut metadata = std::collections::HashMap::new();
                    metadata.insert("category".to_string(), memory.category.clone());
                    metadata.insert("session_id".to_string(), session.clone());
                    metadata.insert("source".to_string(), "extraction".to_string());
                    // Record STATED vs OBSERVED provenance (default Observed).
                    metadata.insert(
                        "fact_type".to_string(),
                        memory.provenance.as_str().to_string(),
                    );

                    match dreaming.remember(&memory.content, metadata).await {
                        Ok(id) => {
                            tracing::info!(
                                "Stored memory from extraction: {} (category: {}, id: {})",
                                memory.content.chars().take(50).collect::<String>(),
                                memory.category,
                                id
                            );
                            
                            // One write path, so the bound holds for every caller.
                            track_memory_in(&recent_memories, &session, &id).await;
                        }
                        Err(e) => {
                            tracing::warn!("Failed to store extracted memory: {}", e);
                        }
                    }
                })
            });

            RunOptions {
                analysis: nanna_agent::RunAnalysis {
                    auto_extract_memories: true,
                    ..nanna_agent::RunAnalysis::default()
                },
                on_memory: Some(on_memory),
                ..Default::default()
            }
        });

        // Run agent (scoped lock)
        let response = {
            let agent_lock = self.get_or_create_agent(session_id, system_prompt).await;
            let agent = agent_lock.read().await;
            agent.run(message, run_options).await?
        };

        // Store assistant response
        let stored = self
            .storage
            .messages()
            .create(nanna_storage::NewMessage {
                session_id: session_id.to_string(),
                role: "assistant".to_string(),
                content: response.text.clone(),
                content_type: "text".to_string(),
                tool_use_id: None,
                tokens_in: Some(i64::from(response.input_tokens)),
                tokens_out: Some(i64::from(response.output_tokens)),
                metadata: None,
            })
            .await
            .map_err(|e| tracing::warn!(session_id, "Failed to store assistant reply: {e}"))
            .ok();
        debug_assert!(
            stored
                .as_ref()
                .is_none_or(|row| row.session_id == session_id),
            "the stored reply belongs to this session"
        );
        debug_assert!(
            stored.as_ref().is_none_or(|row| row.role == "assistant"),
            "the stored reply is the assistant's row"
        );

        Ok(ProcessedReply {
            text: response.text,
            stored,
        })
    }
}

impl AppState {
    /// Start the scheduler with dreaming executor.
    ///
    /// Call this once after building the state to start periodic tasks.
    pub async fn start_scheduler(&self) {
        let Some(scheduler) = &self.scheduler else {
            tracing::debug!("Scheduler not enabled");
            return;
        };

        let mut sched = scheduler.write().await;

        // Load persisted jobs
        if let Err(e) = sched.load_jobs().await {
            tracing::warn!("Failed to load persisted jobs: {}", e);
        }

        // Set up the task executor
        if let Some(dreaming) = &self.dreaming {
            let dreaming = dreaming.clone();
            
            // Wrap to handle all task types
            let full_executor: nanna_core::TaskExecutor = Arc::new(move |task: nanna_core::ScheduledTask| {
                let dreaming = dreaming.clone();
                Box::pin(async move {
                    let start = std::time::Instant::now();
                    let started_at = chrono::Utc::now();
                    // Check if it's a dreaming task
                    if nanna_core::is_dreaming_task(&task) {
                        tracing::info!("Starting memory consolidation (dreaming)...");

                        match dreaming.dream().await {
                            Ok(stats) => {
                                let output = format!(
                                    "Dreaming complete: {} processed, {} merged, {} expanded",
                                    stats.consolidation.memories_processed,
                                    stats.consolidation.memories_merged,
                                    stats.consolidation.memories_expanded,
                                );
                                tracing::info!("{}", output);

                                let finished_at = chrono::Utc::now();
                                nanna_core::TaskResult {
                                    task_id: task.id.clone(),
                                    task_name: task.name.clone(),
                                    success: true,
                                    output: Some(output),
                                    error: None,
                                    duration_ms: elapsed_ms(start),
                                    started_at,
                                    finished_at,
                                    skipped: false,
                                }
                            }
                            Err(e) => {
                                tracing::warn!("Dreaming failed: {}", e);
                                let finished_at = chrono::Utc::now();
                                nanna_core::TaskResult {
                                    task_id: task.id.clone(),
                                    task_name: task.name.clone(),
                                    success: false,
                                    output: None,
                                    error: Some(e.to_string()),
                                    duration_ms: elapsed_ms(start),
                                    started_at,
                                    finished_at,
                                    skipped: false,
                                }
                            }
                        }
                    } else {
                        // Default executor for other tasks (heartbeat, etc.)
                        tracing::info!("Executing task: {} ({})", task.name, task.id);
                        let finished_at = chrono::Utc::now();
                        nanna_core::TaskResult {
                            task_id: task.id.clone(),
                            task_name: task.name.clone(),
                            success: true,
                            output: Some(task.payload),
                            error: None,
                            duration_ms: elapsed_ms(start),
                            started_at,
                            finished_at,
                            skipped: false,
                        }
                    }
                })
            });

            sched.set_executor(full_executor);

            // Add the consolidation task if not already present
            let tasks = sched.list_tasks().await;
            let has_consolidation = tasks.iter().any(|t| t.name == "memory_consolidation");
            if !has_consolidation {
                sched.add_task(consolidation_task(None)).await;
                tracing::info!("Added memory consolidation task (every 1 hour)");
            }
        }

        sched.start();
        drop(sched);
        tracing::info!("Scheduler started");
    }

    /// Stop the scheduler gracefully.
    pub async fn stop_scheduler(&self) {
        if let Some(scheduler) = &self.scheduler {
            scheduler.write().await.stop().await;
            tracing::info!("Scheduler stopped");
        }
    }

    /// Record feedback for a message (maps to memory feedback).
    ///
    /// Call this when a user reacts to a message with 👍/👎.
    /// Applies feedback to all recent memories from the session.
    pub async fn record_message_feedback(&self, message_id: &str, positive: bool) {
        // Look up the session ID for this message
        let session_id = {
            let map = self.message_session_map.read().await;
            map.get(message_id).map(|link| link.session_id.clone())
        };

        let Some(session_id) = session_id else {
            tracing::debug!("No session mapped for message {}", message_id);
            return;
        };

        let Some(dreaming) = &self.dreaming else {
            return;
        };

        // Get recent memories for this session
        let recent_memories = {
            let mut sessions = self.session_recent_memories.write().await;
            // Ask, do not insert: looking up a session that never stored a
            // memory used to leave an empty entry behind for good.
            sessions.get_mut(&session_id).map_or_else(Vec::new, |memories| {
                memories.retain(|m| m.is_recent(FEEDBACK_ATTRIBUTION_WINDOW));
                memories.clone()
            })
        };

        if recent_memories.is_empty() {
            tracing::debug!("No recent memories for session {}", session_id);
            return;
        }

        let feedback = if positive {
            MemoryFeedback::Helpful
        } else {
            MemoryFeedback::Unhelpful
        };

        // Apply feedback to all recent memories
        for memory in &recent_memories {
            dreaming.record_feedback(&memory.memory_id, feedback).await;
        }

        tracing::info!(
            "Recorded {:?} feedback for {} memories in session {} (message {})",
            feedback,
            recent_memories.len(),
            session_id,
            message_id
        );
    }

    /// Associate a message ID with a session for feedback tracking.
    pub async fn link_message_to_session(&self, message_id: &str, session_id: &str) {
        link_message_in(&self.message_session_map, message_id, session_id).await;
    }

    /// Track a memory ID for a session (for feedback attribution).
    pub async fn track_memory_for_session(&self, session_id: &str, memory_id: &str) {
        track_memory_in(&self.session_recent_memories, session_id, memory_id).await;
    }
}

const DEFAULT_SYSTEM_PROMPT: &str = r"You are Nanna — moon god of the digital realm.

You have tools at your disposal. Use them when needed.

Be helpful. Be competent. Don't waste words.";

/// Milliseconds since `start`, saturating: a task would need to run for ~584
/// million years to exceed `u64::MAX` milliseconds.
fn elapsed_ms(start: std::time::Instant) -> u64 {
    u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_map() -> RwLock<HashMap<String, Vec<RecentMemory>>> {
        RwLock::new(HashMap::new())
    }

    #[tokio::test]
    async fn tracking_is_bounded_per_session() {
        let tracked = empty_map();
        // Well past the cap: the uncapped inline path this replaces would have
        // kept every one of these for the life of the process.
        for i in 0..(MAX_MEMORIES_PER_SESSION * 3) {
            track_memory_in(&tracked, "session-a", &format!("mem-{i}")).await;
        }

        let memories = {
            let sessions = tracked.read().await;
            sessions
                .get("session-a")
                .expect("session is tracked")
                .clone()
        };
        assert_eq!(memories.len(), MAX_MEMORIES_PER_SESSION);
        // The cap keeps the newest, which are the ones a reaction can be about.
        let newest = format!("mem-{}", MAX_MEMORIES_PER_SESSION * 3 - 1);
        assert!(
            memories.iter().any(|m| m.memory_id == newest),
            "the most recent memory must survive the cap"
        );
        assert!(
            !memories.iter().any(|m| m.memory_id == "mem-0"),
            "the oldest memory must have been evicted"
        );
    }

    #[tokio::test]
    async fn tracking_is_bounded_across_sessions() {
        let tracked = empty_map();
        for i in 0..(MAX_TRACKED_SESSIONS + 40) {
            track_memory_in(&tracked, &format!("session-{i}"), "mem").await;
        }

        let tracked_ids: Vec<String> = {
            let sessions = tracked.read().await;
            sessions.keys().cloned().collect()
        };
        assert!(
            tracked_ids.len() <= MAX_TRACKED_SESSIONS,
            "tracked {} sessions, cap is {MAX_TRACKED_SESSIONS}",
            tracked_ids.len()
        );
        assert!(
            tracked_ids.contains(&format!("session-{}", MAX_TRACKED_SESSIONS + 39)),
            "the newest session must be the one that is kept"
        );
    }

    #[tokio::test]
    async fn entries_past_the_attribution_window_are_dropped_on_write() {
        let tracked = empty_map();
        {
            let mut sessions = tracked.write().await;
            let stale_at = Instant::now()
                .checked_sub(FEEDBACK_ATTRIBUTION_WINDOW + Duration::from_secs(1))
                .expect("the window is small enough to subtract");
            sessions.insert(
                "old-session".to_string(),
                vec![RecentMemory {
                    memory_id: "stale".to_string(),
                    stored_at: stale_at,
                }],
            );
        }

        // A write for an unrelated session is enough to collect the stale one:
        // the prune does not wait for a reaction that may never come.
        track_memory_in(&tracked, "new-session", "fresh").await;

        let sessions = tracked.read().await;
        assert!(
            !sessions.contains_key("old-session"),
            "a session holding only expired entries must be removed entirely"
        );
        assert_eq!(
            sessions
                .get("new-session")
                .map_or_default(std::vec::Vec::len),
            1
        );
    }

    #[tokio::test]
    async fn an_expired_link_is_dropped_on_write() {
        let links = RwLock::new(HashMap::new());
        {
            let mut map = links.write().await;
            let stale_at = Instant::now()
                .checked_sub(FEEDBACK_ATTRIBUTION_WINDOW + Duration::from_secs(1))
                .expect("the window is small enough to subtract");
            map.insert(
                "old-message".to_string(),
                LinkedSession {
                    session_id: "old-session".to_string(),
                    linked_at: stale_at,
                },
            );
        }

        link_message_in(&links, "new-message", "new-session").await;

        let map = links.read().await;
        assert!(
            !map.contains_key("old-message"),
            "a link that can only resolve to expired memories must be dropped"
        );
        assert_eq!(
            map.get("new-message").map(|l| l.session_id.as_str()),
            Some("new-session")
        );
    }

    #[tokio::test]
    async fn the_backstop_evicts_the_oldest_link_not_an_arbitrary_one() {
        let links = RwLock::new(HashMap::new());
        // Fill past the cap with links whose ages increase, all inside the
        // attribution window so the retain cannot be what does the work.
        {
            let mut map = links.write().await;
            let now = Instant::now();
            for i in 0..MAX_LINKED_MESSAGES {
                let age = Duration::from_millis((MAX_LINKED_MESSAGES - i) as u64);
                map.insert(
                    format!("message-{i}"),
                    LinkedSession {
                        session_id: format!("session-{i}"),
                        linked_at: now.checked_sub(age).expect("recent enough"),
                    },
                );
            }
            assert_eq!(map.len(), MAX_LINKED_MESSAGES);
        }

        link_message_in(&links, "message-newest", "session-newest").await;

        let link_ids: Vec<String> = {
            let map = links.read().await;
            map.keys().cloned().collect()
        };
        assert!(
            link_ids.len() <= MAX_LINKED_MESSAGES,
            "cap held: {}",
            link_ids.len()
        );
        assert!(
            link_ids.contains(&"message-newest".to_string()),
            "the link just made must survive its own insertion"
        );
        // The old prune took HALF the map in HashMap order; this one takes the
        // single oldest link, which is `message-0`.
        assert!(
            !link_ids.contains(&"message-0".to_string()),
            "the oldest link is the one that should go"
        );
        assert!(
            link_ids.contains(&format!("message-{}", MAX_LINKED_MESSAGES - 1)),
            "a recent link must not be collateral damage"
        );
    }
}
