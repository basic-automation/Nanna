//! Shared component wiring: API key checks, scheduler, LLM/tools/storage setup.

use crate::onboarding;
use chrono::Utc;
use nanna_agent::{Agent, AgentConfig, AgentContext, RunOptions};
use nanna_config::Config;
use nanna_core::{LlmClient, Scheduler, SchedulerConfig, ScheduledTask, TaskResult};
use nanna_daemon::llm_router::ProviderId;
use nanna_storage::{Storage, StorageConfig};
use nanna_tools::{
    CancelReminderTool, EchoTool, ExecTool, ExploreTool, ListDirTool, ListRemindersTool,
    ReadFileTool, RecallTool, ReflectTool, ReminderStore, RememberTool, RemindTool, StatusTool,
    ToolRegistry, TursoMemoryStorage, WebFetchTool, WebSearchTool, WonderTool, WriteFileTool,
};
use std::sync::Arc;
use tracing::{error, info};

/// Ensure API key is configured, prompt if not.
pub fn ensure_api_key(mut config: Config) -> anyhow::Result<Config> {
    if !onboarding::has_api_key(&config) {
        onboarding::quick_setup(&mut config)?;
    }
    Ok(config)
}

/// Create the scheduler with a task executor that runs tasks through an agent.
/// What the `nanna serve` scheduler does with a due task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduledRun {
    /// A memory consolidation: run a dream cycle, never a prompt.
    Dream,
    /// Leave it alone, reporting why.
    Skip(&'static str),
    /// A user job or heartbeat: run its payload as an agent prompt.
    Prompt,
}

/// Classify `task` for the `nanna serve` scheduler. Pure.
///
/// That scheduler shares the persisted job table with the daemon, which keeps
/// its own machinery there too. Before this, every row was run as an agent
/// prompt — the daemon's recurrence sweep and reminders included — and the
/// server's second scheduler "completed" the same rows by echoing their text.
#[must_use]
pub fn classify_scheduled(task: &ScheduledTask) -> ScheduledRun {
    if nanna_core::is_dreaming_task(task) {
        ScheduledRun::Dream
    } else if nanna_daemon::server::DAEMON_SYSTEM_TASKS.contains(&task.name.as_str()) {
        ScheduledRun::Skip("owned by the daemon")
    } else if task.payload.trim().is_empty() {
        ScheduledRun::Skip("empty payload")
    } else {
        ScheduledRun::Prompt
    }
}

/// Run one dream cycle for the scheduler, or report that there is no runtime.
async fn run_scheduled_dream(
    task: &ScheduledTask,
    dreaming: Option<Arc<nanna_core::DreamingRuntime>>,
) -> TaskResult {
    let started_at = Utc::now();
    let (success, output, error) = match dreaming {
        Some(dreaming) => match dreaming.dream().await {
            Ok(stats) => (
                true,
                Some(format!(
                    "Dreaming complete: {} processed, {} merged, {} expanded",
                    stats.consolidation.memories_processed,
                    stats.consolidation.memories_merged,
                    stats.consolidation.memories_expanded,
                )),
                None,
            ),
            Err(e) => (false, None, Some(e.to_string())),
        },
        None => (
            true,
            Some("Skipped (no dreaming runtime)".to_string()),
            None,
        ),
    };
    TaskResult {
        task_id: task.id.clone(),
        task_name: task.name.clone(),
        success,
        output,
        error,
        duration_ms: 0,
        started_at,
        finished_at: Utc::now(),
    }
}

/// A due task the scheduler deliberately did not run, recorded as such.
fn skipped_result(task: &ScheduledTask, why: &str) -> TaskResult {
    let now = Utc::now();
    TaskResult {
        task_id: task.id.clone(),
        task_name: task.name.clone(),
        success: true,
        output: Some(format!("Skipped ({why})")),
        error: None,
        duration_ms: 0,
        started_at: now,
        finished_at: now,
    }
}

/// The `nanna serve` scheduler's settings, from `[scheduler]`.
fn scheduler_config(config: &Config) -> SchedulerConfig {
    SchedulerConfig {
        enabled: config.scheduler.enabled,
        heartbeat_interval: std::time::Duration::from_secs(
            nanna_core::clamp_heartbeat_secs(config.scheduler.heartbeat_interval_secs),
        ),
        heartbeat_enabled: config.scheduler.heartbeat_enabled,
        heartbeat_prompt: "Heartbeat: check in and review state".to_string(),
        max_concurrent: 4,
        check_interval: std::time::Duration::from_secs(30),
        default_timezone: "UTC".to_string(),
    }
}

/// Build the `nanna serve` scheduler: the ONE scheduler that command runs.
pub fn create_scheduler(
    config: &Config,
    llm: Arc<LlmClient>,
    tools: Arc<ToolRegistry>,
    storage: Arc<Storage>,
    dreaming: Option<Arc<nanna_core::DreamingRuntime>>,
) -> Scheduler {
    let scheduler_config = scheduler_config(config);

    // Clone storage for the scheduler's persistence
    let scheduler_storage = storage.clone();

    // Create a task executor that runs tasks through an agent
    let model = config.llm.model.clone();
    let executor: nanna_core::TaskExecutor = Arc::new(move |task: ScheduledTask| {
        let llm = llm.clone();
        let tools = tools.clone();
        let storage = storage.clone();
        let model = model.clone();
        let dreaming = dreaming.clone();

        Box::pin(async move {
            match classify_scheduled(&task) {
                ScheduledRun::Dream => return run_scheduled_dream(&task, dreaming).await,
                ScheduledRun::Skip(why) => return skipped_result(&task, why),
                ScheduledRun::Prompt => {}
            }
            let start = std::time::Instant::now();
            let task_id = task.id.clone();

            info!("Executing scheduled task: {} ({})", task.name, task_id);

            // Create a dedicated agent for the task
            let agent_config = AgentConfig {
                model,
                max_tokens: 4096,
                temperature: 0.7,
                max_iterations: Some(5),
                summarization_priority: vec![],
                ..Default::default()
            };

            let session_id = format!("scheduler:{}", task.name);
            let system_prompt = match task.task_type {
                nanna_core::TaskType::Heartbeat => {
                    "You are Nanna in heartbeat mode. Check in, review your state, \
                     and do any proactive work that needs attention. Be concise."
                }
                _ => {
                    "You are Nanna executing a scheduled task. Complete the task efficiently."
                }
            };

            let context = AgentContext::new(&session_id).with_system_prompt(system_prompt);
            let agent = Agent::new(agent_config, llm, tools).with_context(context);

            let started_at = Utc::now();
            let task_name = task.name.clone();

            // Run the task
            match agent.run(&task.payload, RunOptions::default()).await {
                Ok(response) => {
                    // Store the result
                    let _ = storage
                        .messages()
                        .create(nanna_storage::NewMessage {
                            session_id,
                            role: "assistant".to_string(),
                            content: response.text.clone(),
                            content_type: "text".to_string(),
                            tool_use_id: None,
                            tokens_in: Some(i64::from(response.input_tokens)),
                            tokens_out: Some(i64::from(response.output_tokens)),
                            metadata: Some(serde_json::json!({"task_id": task_id})),
                        })
                        .await;

                    let finished_at = Utc::now();
                    TaskResult {
                        task_id,
                        task_name,
                        success: true,
                        output: Some(response.text),
                        error: None,
                        duration_ms: u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
                        started_at,
                        finished_at,
                    }
                }
                Err(e) => {
                    tracing::warn!("Scheduled task {} failed: {}", task_id, e);
                    let finished_at = Utc::now();
                    TaskResult {
                        task_id,
                        task_name,
                        success: false,
                        output: None,
                        error: Some(e.to_string()),
                        duration_ms: u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
                        started_at,
                        finished_at,
                    }
                }
            }
        })
    });

    Scheduler::new(scheduler_config)
        .with_storage(scheduler_storage)
        .with_executor(executor)
}

/// The provider the CLI's chat client belongs to: `[llm].provider`, with an
/// unrecognised name served by Anthropic — the same fallback
/// [`nanna_config::LlmConfig::provider_api_key`] reads the key by.
#[must_use]
fn chat_provider(provider: &str) -> ProviderId {
    match provider {
        "openai" => ProviderId::OpenAI,
        "openrouter" => ProviderId::OpenRouter,
        _ => ProviderId::Anthropic,
    }
}

/// The variable that supplies `provider`'s key when its own field is empty.
const fn chat_key_env(provider: ProviderId) -> &'static str {
    match provider {
        ProviderId::OpenAI => "OPENAI_API_KEY",
        ProviderId::OpenRouter => "OPENROUTER_API_KEY",
        _ => "ANTHROPIC_API_KEY", // anthropic or unknown
    }
}

/// The key the CLI's chat provider runs on: [`chat_api_key`] for
/// [`chat_provider`] and its variable. One definition for the two clients
/// built from it, [`init_components`]'s and `nanna serve`'s bot.
///
/// # Errors
///
/// When that provider has no key, naming the variable that would supply one.
pub fn provider_chat_key(
    config: &Config,
    read_env: impl Fn(&str) -> Option<String>,
) -> anyhow::Result<String> {
    chat_api_key(
        config,
        chat_key_env(chat_provider(&config.llm.provider)),
        read_env,
    )
}

/// The chat provider's key: `[llm].provider`'s own field, else `env_var`
/// read through `read_env`.
fn chat_api_key(
    config: &Config,
    env_var: &str,
    read_env: impl Fn(&str) -> Option<String>,
) -> anyhow::Result<String> {
    config
        .llm
        .provider_api_key()
        .map(str::to_string)
        .or_else(|| read_env(env_var).filter(|key| !key.trim().is_empty()))
        .ok_or_else(|| anyhow::anyhow!(
            "API key required. Run 'nanna init' or set {env_var} environment variable"
        ))
}

/// Initialize common components
pub async fn init_components(
    config: &Config,
) -> anyhow::Result<(Arc<LlmClient>, Arc<ToolRegistry>, Arc<Storage>)> {
    let chat_provider = chat_provider(&config.llm.provider);
    let env_var = chat_key_env(chat_provider);
    let api_key = provider_chat_key(config, |name| std::env::var(name).ok())?;

    // Create LLM client - default to Anthropic
    let llm = Arc::new(match chat_provider {
        ProviderId::OpenAI => LlmClient::openai(&api_key),
        ProviderId::OpenRouter => LlmClient::openrouter(&api_key),
        _ => {
            let provider = &config.llm.provider;
            if provider != "anthropic" {
                error!("Unknown LLM provider: {provider}, defaulting to anthropic");
            }
            LlmClient::anthropic(&api_key)
        }
    });

    // Validate API key early
    info!("Validating API key...");
    if let Err(e) = llm.validate().await {
        return Err(anyhow::anyhow!("API key validation failed: {e}. Check your config or {env_var} environment variable."));
    }
    info!("API key valid");

    // Create tool registry
    let tools = Arc::new(ToolRegistry::new());

    let workspace = config
        .general
        .workspace
        .clone()
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());

    // Initialize storage first (needed for memory tools)
    let storage_path = cli_storage_path(config);

    let storage_config = StorageConfig {
        path: storage_path.to_string_lossy().to_string(),
    };
    let storage = Arc::new(Storage::new(&storage_config).await?);
    info!("Storage initialized at {}", storage_path.display());

    // Register tools
    tools.register(EchoTool).await;
    tools
        .register(ExecTool::new().with_workdir(workspace.display().to_string()))
        .await;
    tools.register(ReadFileTool::new()).await;
    tools.register(WriteFileTool::new()).await;
    tools.register(ListDirTool::new()).await;
    tools.register(WebFetchTool::new()).await;

    // Web search with Brave API (if configured)
    if let Ok(brave_key) = std::env::var("BRAVE_API_KEY") {
        tools.register(WebSearchTool::new().with_api_key(brave_key)).await;
        info!("Web search enabled (Brave API)");
    }

    // Memory tools backed by Turso with optional embeddings
    let memory_storage: Arc<dyn nanna_tools::MemoryStorage + Send + Sync> = {
        let base = TursoMemoryStorage::new(storage.clone());

        // Try to enable semantic search with OpenAI embeddings
        if let Ok(openai_key) = std::env::var("OPENAI_API_KEY") {
            use nanna_llm::EmbeddingClient;
            let embed_client = Arc::new(EmbeddingClient::openai(&openai_key));
            let embed_fn: nanna_tools::EmbedFn = Arc::new(move |text: String| {
                let client = embed_client.clone();
                Box::pin(async move {
                    client.embed_one(&text).await.map_err(|e| e.to_string())
                })
            });
            info!("Semantic search enabled (OpenAI embeddings)");
            Arc::new(base.with_embeddings(embed_fn, "text-embedding-3-small"))
        } else {
            info!("Semantic search disabled (no OPENAI_API_KEY)");
            Arc::new(base)
        }
    };
    tools.register(RememberTool::new(memory_storage.clone())).await;
    tools.register(RecallTool::new(memory_storage.clone())).await;
    tools.register(ReflectTool::new(memory_storage.clone())).await;

    // Scheduling tools
    let scheduler_state = Arc::new(tokio::sync::RwLock::new(ReminderStore::default()));
    tools.register(RemindTool::new(scheduler_state.clone())).await;
    tools.register(ListRemindersTool::new(scheduler_state.clone())).await;
    tools.register(CancelReminderTool::new(scheduler_state.clone())).await;

    // Curiosity/autonomy tools
    tools.register(ExploreTool).await;
    tools.register(WonderTool).await;
    tools.register(StatusTool).await;

    register_discover_tools(&tools, config).await;

    info!("{} tools ready", tools.definitions().await.len());

    Ok((llm, tools, storage))
}

/// The database the CLI opens: `[memory].storage_path`, else `nanna.db` in
/// the data directory THIS configuration selects.
///
/// `Config::resolve_data_dir` honours `[general] data_dir`; the CLI used
/// `Config::default_data_dir`, which does not — so with a custom data dir set,
/// `nanna chat`, `run` and `sessions` opened a different database from the
/// daemon's, and a user's sessions seemed to vanish.
#[must_use]
pub fn cli_storage_path(config: &Config) -> std::path::PathBuf {
    config.memory.storage_path.clone().unwrap_or_else(|| {
        config
            .resolve_data_dir()
            .unwrap_or_else(|_| std::env::current_dir().unwrap_or_default())
            .join("nanna.db")
    })
}

/// Register the `discover_tools` JS/TS skill with the tool registry.
///
/// The skill's source lives in the user's tools directory, where it can be
/// edited — so a source that no longer parses is logged and skipped, never a
/// panic that takes the whole CLI down with it.
async fn register_discover_tools(tools: &Arc<ToolRegistry>, config: &Config) {
    let tools_dir = nanna_tools::skills::defaults::resolve_tools_dir(
        config.tools.tools_dir.as_deref()
    );
    if let Some(ref dir) = tools_dir
        && let Some(source) = nanna_tools::skills::defaults::load_discover_tools_source(dir) {
            match nanna_tools::skills::ScriptedToolWrapper::from_source("discover_tools", &source) {
                Ok(wrapper) => tools.register(wrapper.with_registry(Arc::downgrade(tools))).await,
                Err(e) => tracing::warn!(
                    "discover_tools in {} does not parse ({e}); continuing without it",
                    dir.display()
                ),
            }
        }
}

#[cfg(test)]
mod tests {
    use super::chat_api_key;
    use nanna_config::Config;

    fn no_env(_: &str) -> Option<String> {
        None
    }

    /// Every key field set, each to its own provider's name.
    fn config_with_every_key(provider: &str) -> Config {
        let mut config = Config::default();
        config.llm.provider = provider.to_string();
        config.llm.api_key = Some("anthropic".to_string());
        config.llm.openai_api_key = Some("openai".to_string());
        config.llm.openrouter_api_key = Some("openrouter".to_string());
        config
    }

    /// An `OpenRouter` or `OpenAI` chat never runs on the Anthropic key.
    #[test]
    fn the_chat_client_gets_its_providers_own_key() {
        for (provider, env_var) in [
            ("openai", "OPENAI_API_KEY"),
            ("openrouter", "OPENROUTER_API_KEY"),
            ("anthropic", "ANTHROPIC_API_KEY"),
        ] {
            let key = chat_api_key(&config_with_every_key(provider), env_var, no_env)
                .expect("the provider has a key");
            assert_eq!(key, provider);
        }
    }

    #[test]
    fn a_missing_own_key_falls_back_to_the_providers_variable_only() {
        let mut config = config_with_every_key("openrouter");
        config.llm.openrouter_api_key = None;
        let env = |name: &str| (name == "OPENROUTER_API_KEY").then(|| "from-env".to_string());
        assert_eq!(
            chat_api_key(&config, "OPENROUTER_API_KEY", env).expect("the variable is set"),
            "from-env"
        );

        let error = chat_api_key(&config, "OPENROUTER_API_KEY", no_env)
            .expect_err("the Anthropic key is not OpenRouter's");
        assert!(error.to_string().contains("OPENROUTER_API_KEY"), "{error}");
    }
}

#[cfg(test)]
mod storage_path_tests {
    use super::cli_storage_path;
    use nanna_config::Config;

    /// The CLI and the daemon must open the same database: with `[general]
    /// data_dir` set, the CLI used the platform default instead.
    #[test]
    fn the_cli_opens_the_database_in_the_configured_data_dir() {
        let mut config = Config::default();
        config.general.data_dir = Some(std::path::PathBuf::from("/srv/nanna-data"));
        assert_eq!(
            cli_storage_path(&config),
            std::path::Path::new("/srv/nanna-data/nanna.db")
        );
        config.memory.storage_path = Some(std::path::PathBuf::from("/elsewhere/x.db"));
        assert_eq!(
            cli_storage_path(&config),
            std::path::Path::new("/elsewhere/x.db"),
            "an explicit storage path still wins"
        );
    }
}

#[cfg(test)]
mod scheduled_run_tests {
    use super::{ScheduledRun, classify_scheduled};

    /// The rows `nanna serve` shares with the daemon: its own machinery is
    /// never run as a prompt, dreaming is a dream, and only a real payload is
    /// handed to the model.
    #[test]
    fn only_user_jobs_are_run_as_prompts() {
        let task = |name: &str, payload: &str| {
            let mut task =
                nanna_core::recurring_task(name, std::time::Duration::from_secs(60), payload);
            task.name = name.to_string();
            task
        };
        assert_eq!(
            classify_scheduled(&nanna_core::consolidation_task(None)),
            ScheduledRun::Dream
        );
        for name in nanna_daemon::server::DAEMON_SYSTEM_TASKS {
            assert_eq!(
                classify_scheduled(&task(
                    name,
                    "Reopen recurring tasks whose next occurrence has arrived."
                )),
                ScheduledRun::Skip("owned by the daemon"),
                "{name}"
            );
        }
        assert_eq!(
            classify_scheduled(&task("nightly", "  ")),
            ScheduledRun::Skip("empty payload")
        );
        assert_eq!(
            classify_scheduled(&task("nightly", "Summarise my inbox")),
            ScheduledRun::Prompt
        );
    }
}
