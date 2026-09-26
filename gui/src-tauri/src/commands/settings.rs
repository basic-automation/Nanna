//! Settings, credentials, and model configuration commands.
//!
//! The daemon owns the live LLM clients, tool registry, and memory service.
//! These commands mutate the local `config` cache and persist it to
//! `config.toml` (which the daemon also reads); model/tool listings that need
//! live state are fetched from the daemon.

use crate::backend::Backend;
use crate::state::{backend_handle, AppConfig, AppState, ModelStatusEvent};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tauri::{AppHandle, Emitter, State};
use tokio::sync::RwLock;
use tracing::{error, info, warn};

/// Tool names as seen by the daemon's registry (empty if the daemon is down).
async fn daemon_tool_names(backend: &Backend) -> Vec<String> {
    backend
        .tool_list()
        .await
        .ok()
        .and_then(|r| {
            r.get("tools").and_then(|v| v.as_array()).map(|arr| {
                arr.iter()
                    .filter_map(|t| t.get("name").and_then(|v| v.as_str()).map(String::from))
                    .collect()
            })
        })
        .unwrap_or_default()
}

/// The tools in a daemon `tool.list` reply; `None` when it has no `tools`
/// array. Entries without a string `name` are skipped.
pub(crate) fn tool_infos(reply: &serde_json::Value) -> Option<Vec<ToolInfo>> {
    reply.get("tools").and_then(|v| v.as_array()).map(|arr| {
        arr.iter()
            .filter_map(|t| {
                Some(ToolInfo {
                    name: t.get("name")?.as_str()?.to_string(),
                    description: t.get("description").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    enabled: t.get("enabled").and_then(serde_json::Value::as_bool).unwrap_or(true),
                    is_user_tool: t.get("is_user_tool").and_then(serde_json::Value::as_bool).unwrap_or(false),
                })
            })
            .collect()
    })
}

/// The chat models the settings pickers offer before a provider's live list
/// has been fetched.
fn known_chat_models() -> Vec<String> {
    vec![
        // Anthropic
        "claude-opus-4-20250514".to_string(),
        "claude-sonnet-4-20250514".to_string(),
        "claude-3-5-sonnet-20241022".to_string(),
        "claude-3-5-haiku-20241022".to_string(),
        // OpenAI
        "gpt-4o".to_string(),
        "gpt-4o-mini".to_string(),
        "gpt-4-turbo".to_string(),
        "o1".to_string(),
        "o1-mini".to_string(),
        // OpenRouter
        "deepseek/deepseek-chat".to_string(),
        "google/gemini-2.5-flash-preview-05-20".to_string(),
        "google/gemini-2.5-pro-preview-05-06".to_string(),
        // Ollama (local)
        "llama3.2".to_string(),
        "llama3.1".to_string(),
        "mistral".to_string(),
        "mixtral".to_string(),
        "codellama".to_string(),
        "qwen2.5".to_string(),
        "deepseek-coder-v2".to_string(),
    ]
}

/// Format Claude model IDs into friendly names.
fn format_claude_model_name(id: &str) -> String {
    match id {
        "claude-opus-4-5-20251101" => "Claude Opus 4.5".to_string(),
        "claude-opus-4-20250514" => "Claude Opus 4".to_string(),
        "claude-sonnet-4-20250514" => "Claude Sonnet 4".to_string(),
        "claude-3-5-sonnet-20241022" => "Claude Sonnet 3.5".to_string(),
        "claude-3-5-haiku-20241022" => "Claude Haiku 3.5".to_string(),
        _ => id.to_string(),
    }
}

/// Get application config
///
/// # Errors
///
/// Never returns `Err`: an unreachable daemon lists no tools.
#[tauri::command]
pub async fn get_config(state: State<'_, Arc<RwLock<AppState>>>) -> Result<AppConfig, String> {
    let tool_names: Vec<String> = daemon_tool_names(&*backend_handle(&state).await).await;

    let state_guard = state.read().await;

    Ok(AppConfig {
        theme: "dark".to_string(),
        model: state_guard.config.llm.model.clone(),
        // Any provider counts, not just Anthropic (config field or its env var).
        api_key_set: state_guard.config.llm.has_configured_api_key()
            || [
                "ANTHROPIC_API_KEY",
                "OPENAI_API_KEY",
                "OPENROUTER_API_KEY",
                "GITHUB_TOKEN",
            ]
            .iter()
            .any(|var| std::env::var(var).is_ok_and(|v| !v.trim().is_empty())),
        available_models: known_chat_models(),
        available_tools: tool_names,
    })
}

/// Update model setting
///
/// # Errors
///
/// Never returns `Err`; the `Result` is what Tauri requires of an async
/// command that borrows `State`. The change is not saved to `config.toml`.
#[tauri::command]
pub async fn set_model(state: State<'_, Arc<RwLock<AppState>>>, model: String) -> Result<(), String> {
    state.write().await.config.llm.model = model;
    Ok(())
}

/// Extended settings for the settings page
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtendedSettings {
    // API Keys (masked for display). The groups are flattened, so every key
    // stays top-level on the wire.
    #[serde(flatten)]
    pub llm_api_keys: LlmApiKeys,
    /// GitHub Models token (config or `GITHUB_TOKEN`).
    pub github_key_set: bool,
    #[serde(flatten)]
    pub claude_proxy: ClaudeProxyStatus,
    /// Brave Search key (`BRAVE_API_KEY`).
    pub brave_key_set: bool,

    // Anthropic OAuth status
    #[serde(flatten)]
    pub anthropic_oauth: AnthropicOauthStatus,

    // Chat Provider
    pub provider: String,
    pub available_providers: Vec<String>,

    // Chat Model
    pub model: String,
    pub available_models: Vec<String>,

    // Embedding Provider (separate from chat)
    pub embedding_provider: String,
    pub embedding_model: String,
    pub available_embedding_providers: Vec<String>,
    pub available_embedding_models: Vec<String>,
    pub embedding_enabled: bool,

    // Memory extraction model (empty = use chat model)
    pub extraction_model: String,
    pub available_extraction_models: Vec<String>,

    // Ollama configuration
    pub ollama_host: String,
    /// Whether a bearer token is saved and which server it is for — never
    /// the token itself, which the webview has no use for.
    #[serde(flatten)]
    pub ollama_token: OllamaTokenStatus,

    // Generation params
    pub temperature: f32,
    pub top_p: f32,
    pub max_tokens: u32,

    // Tools
    pub tools: Vec<ToolInfo>,

    // Memory & Scheduling
    #[serde(flatten)]
    pub memory: MemorySettings,
    #[serde(flatten)]
    pub scheduler: SchedulerSettings,

    // Agent loop (long-horizon worker). `agent_max_iterations` None = unlimited.
    pub agent_max_iterations: Option<usize>,
    pub agent_nudge_after_iterations: usize,
    pub agent_nudge_interval_iterations: usize,
}

/// Whether a key is configured — in the config cache or the provider's
/// environment variable — for each pay-per-token LLM API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlmApiKeys {
    pub anthropic_key_set: bool,
    pub openai_key_set: bool,
    pub openrouter_key_set: bool,
}

/// Where the local Claude proxy is expected when nothing says otherwise.
const CLAUDE_PROXY_DEFAULT_URL: &str = "http://localhost:3456";

/// This GUI process's Claude proxy setting: enabled, and the URL, if one was set.
///
/// Seeded once from `CLAUDE_PROXY_ENABLED` / `CLAUDE_PROXY_URL` at first use, so
/// a launch-time environment still configures it, and changed in memory after
/// that. The commands used to write the process environment instead —
/// `unsafe set_var` from a command on the multi-threaded runtime, which races
/// every concurrent `getenv` (glibc's included) — only for this same process
/// to read it back; nothing else ever saw those variables.
static CLAUDE_PROXY: std::sync::LazyLock<std::sync::RwLock<(bool, Option<String>)>> =
    std::sync::LazyLock::new(|| {
        std::sync::RwLock::new((
            std::env::var_os("CLAUDE_PROXY_ENABLED").is_some(),
            std::env::var("CLAUDE_PROXY_URL").ok(),
        ))
    });

/// The current proxy setting as `(enabled, url)`, the URL defaulted.
fn claude_proxy_setting() -> (bool, String) {
    let (enabled, url) = CLAUDE_PROXY
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    (
        enabled,
        url.unwrap_or_else(|| CLAUDE_PROXY_DEFAULT_URL.to_string()),
    )
}

/// Change the proxy setting; `url: None` keeps the current one.
fn set_claude_proxy_setting(enabled: bool, url: Option<String>) {
    let mut guard = CLAUDE_PROXY
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    guard.0 = enabled;
    if url.is_some() {
        guard.1 = url;
    }
}

/// The local Claude proxy: whether it is enabled, and the URL it is expected
/// at (see [`CLAUDE_PROXY`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaudeProxyStatus {
    pub claude_proxy_enabled: bool,
    pub claude_proxy_url: String,
}

/// Anthropic OAuth: whether a token is loaded, and whether requests use it
/// instead of an API key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnthropicOauthStatus {
    pub anthropic_oauth_logged_in: bool,
    pub anthropic_use_oauth: bool,
}

/// The saved Ollama bearer token, described without revealing it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OllamaTokenStatus {
    pub ollama_token_saved: bool,
    /// The server the token is sent to — and only it. `None` when no token
    /// is saved, or the store cannot say which server it is for (it is sent
    /// to none then).
    pub ollama_token_host: Option<String>,
    /// `OLLAMA_API_KEY` is set: that token goes to whatever address is
    /// configured, ahead of any saved one — so what the page says about the
    /// saved token is not the whole story.
    pub ollama_token_from_env: bool,
}

/// What the store says about the saved Ollama token, and whether
/// `OLLAMA_API_KEY` overrides it. A token saved by an older build recorded no
/// server; it is the configured one's, as the daemon reads it.
fn ollama_token_status(
    store: &nanna_config::SecureStore,
    configured_host: &str,
    env_token: Option<&str>,
) -> OllamaTokenStatus {
    let saved = store.ollama_token().is_some();
    OllamaTokenStatus {
        ollama_token_saved: saved,
        ollama_token_host: saved
            .then(|| match store.ollama_token_host() {
                Ok(Some(host)) => Some(host),
                Ok(None) => Some(nanna_config::normalize_ollama_host(configured_host)),
                Err(_) => None,
            })
            .flatten(),
        ollama_token_from_env: env_token.is_some_and(|token| !token.trim().is_empty()),
    }
}

/// Memory consolidation and capture settings.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MemorySettings {
    pub auto_remember_messages: bool,
    pub max_compression_ratio: f32,
    pub min_remaining_memories: usize,
}

/// The whole-scheduler toggles mirrored from `[scheduler]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchedulerSettings {
    pub scheduler_enabled: bool,
    pub heartbeat_enabled: bool,
    pub heartbeat_interval_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInfo {
    pub name: String,
    pub description: String,
    pub enabled: bool,
    /// A user-authored tool rather than a bundled skill. The two are toggled
    /// through the same action but live in different stores, so the surface
    /// labels them differently — a user tool can also be edited and deleted.
    #[serde(default)]
    pub is_user_tool: bool,
}

/// Get extended settings
///
/// # Errors
///
/// Never returns `Err`: an unreachable daemon lists no tools, and everything
/// else is read from the config cache and the environment.
#[tauri::command]
pub async fn get_extended_settings(
    state: State<'_, Arc<RwLock<AppState>>>,
) -> Result<ExtendedSettings, String> {
    // Tools come from the daemon's live registry.
    let tools: Vec<ToolInfo> = backend_handle(&state)
        .await
        .tool_list()
        .await
        .ok()
        .and_then(|r| tool_infos(&r))
        .unwrap_or_default();

    let config = state.read().await.config.clone();
    // Keyring reads, made after the state lock is released and off the async
    // runtime: they can block on an unlock prompt.
    let configured_host = config.memory.ollama_host.clone();
    let ollama_token = tokio::task::spawn_blocking(move || {
        let env_token = std::env::var("OLLAMA_API_KEY").ok();
        ollama_token_status(
            &nanna_config::SecureStore::new(),
            &configured_host,
            env_token.as_deref(),
        )
    })
    .await
    .unwrap_or(OllamaTokenStatus {
        ollama_token_saved: false,
        ollama_token_host: None,
        ollama_token_from_env: false,
    });
    Ok(build_extended_settings(&config, tools, ollama_token))
}

/// The settings page's view of `config`, with `tools` from the daemon.
///
/// Pure apart from environment reads, so what reaches the webview is
/// testable without a running app.
fn build_extended_settings(
    config: &nanna_config::Config,
    tools: Vec<ToolInfo>,
    ollama_token: OllamaTokenStatus,
) -> ExtendedSettings {
    ExtendedSettings {
        llm_api_keys: LlmApiKeys {
            anthropic_key_set: config.llm.api_key.is_some()
                || std::env::var("ANTHROPIC_API_KEY").is_ok(),
            openai_key_set: config.llm.openai_api_key.is_some()
                || std::env::var("OPENAI_API_KEY").is_ok(),
            openrouter_key_set: config.llm.openrouter_api_key.is_some()
                || std::env::var("OPENROUTER_API_KEY").is_ok(),
        },
        github_key_set: config.llm.github_token.is_some()
            || std::env::var("GITHUB_TOKEN").is_ok(),
        claude_proxy: {
            let (claude_proxy_enabled, claude_proxy_url) = claude_proxy_setting();
            ClaudeProxyStatus {
                claude_proxy_enabled,
                claude_proxy_url,
            }
        },
        brave_key_set: config.tools.brave_api_key.is_some()
            || std::env::var("BRAVE_API_KEY").is_ok(),

        // Anthropic OAuth status
        anthropic_oauth: AnthropicOauthStatus {
            anthropic_oauth_logged_in: config.llm.anthropic_oauth_token.is_some(),
            anthropic_use_oauth: config.llm.anthropic_use_oauth,
        },

        provider: config.llm.provider.clone(),
        available_providers: vec![
            "anthropic".to_string(),
            "openai".to_string(),
            "openrouter".to_string(),
            "github".to_string(),
            "claude-proxy".to_string(),
            "ollama".to_string(),
        ],

        model: config.llm.model.clone(),
        available_models: known_chat_models(),

        // Embedding settings come from the config cache (the daemon reads the
        // same file), separate from chat.
        embedding_provider: config.memory.embedding_provider.clone(),
        embedding_model: config.memory.embedding_model.clone(),
        embedding_enabled: config.memory.enabled,
        available_embedding_providers: vec![
            "openai".to_string(),
            "ollama".to_string(),
            "disabled".to_string(),
        ],
        available_embedding_models: vec![
            // OpenAI
            "text-embedding-3-small".to_string(),  // 1536 dims
            "text-embedding-3-large".to_string(),  // 3072 dims
            // Ollama (dynamic list fetched separately)
            "nomic-embed-text".to_string(),        // 768 dims
            "mxbai-embed-large".to_string(),       // 1024 dims
            "all-minilm".to_string(),              // 384 dims
        ],

        ollama_host: config.memory.ollama_host.clone(),
        ollama_token,

        // Memory extraction model
        extraction_model: config.memory.extraction_model.clone(),
        available_extraction_models: vec![
            String::new(), // Empty = use chat model
            "claude-3-5-haiku-20241022".to_string(),
            "claude-3-5-sonnet-20241022".to_string(),
            "gpt-4o-mini".to_string(),
            "gpt-4o".to_string(),
        ],

        temperature: 1.0,
        top_p: 0.95,
        max_tokens: 8192,

        tools,

        memory: MemorySettings {
            auto_remember_messages: config.memory.auto_remember_messages,
            max_compression_ratio: config.memory.max_compression_ratio,
            min_remaining_memories: config.memory.min_remaining_memories,
        },
        // Scheduler toggles are real settings now — read them back from the
        // config the daemon also reads, so the switches reflect what the daemon
        // is doing instead of hardcoded `true`s that never moved.
        scheduler: SchedulerSettings {
            scheduler_enabled: config.scheduler.enabled,
            heartbeat_enabled: config.scheduler.heartbeat_enabled,
            heartbeat_interval_seconds: config.scheduler.heartbeat_interval_secs,
        },

        // Agent-loop iteration policy
        agent_max_iterations: config.agent.max_iterations,
        agent_nudge_after_iterations: config.agent.nudge_after_iterations,
        agent_nudge_interval_iterations: config.agent.nudge_interval_iterations,
    }
}

/// Set memory extraction model (empty string = use chat model)
///
/// # Errors
///
/// Never returns `Err`: a failed `config.toml` save is logged, and the daemon
/// reload is best-effort.
#[tauri::command]
pub async fn set_extraction_model(
    state: State<'_, Arc<RwLock<AppState>>>,
    model: String,
) -> Result<(), String> {
    let mut state_guard = state.write().await;

    // Persist to config (the daemon reads the same file).
    state_guard.config.memory.extraction_model.clone_from(&model);
    if let Err(e) = state_guard.config.save() {
        warn!("Failed to save extraction model to config: {}", e);
    }
    // Never hold AppState across a daemon round trip: every other
    // command waits on this lock for as long as the reload takes.
    let backend = Arc::clone(&state_guard.backend);
    drop(state_guard);
    let _ = backend.config_reload().await;

    if model.is_empty() {
        info!("Extraction model set to: (use chat model)");
    } else {
        info!("Extraction model set to: {}", model);
    }
    Ok(())
}

/// Set a specific API key
///
/// # Errors
///
/// Returns `Unknown provider: …` for a provider other than `anthropic`,
/// `openai`, `brave`, `openrouter`, `github` or `claude-proxy`, before anything
/// changes. Returns `failed to store API key securely: …` when the key cannot
/// be written to the OS keyring; it then stays set in this process's config
/// cache and environment only. A failed `config.toml` save is only logged.
#[tauri::command]
pub async fn set_provider_api_key(
    state: State<'_, Arc<RwLock<AppState>>>,
    provider: String,
    api_key: String,
) -> Result<(), String> {
    let mut state_guard = state.write().await;

    // The daemon owns the live LLM clients and tool registry; persist the key to
    // the secure store and let the daemon reload. This process reads keys from
    // `state_guard.config` (refilled from the store below), not from its own
    // environment: `set_var` from a command running on the multi-threaded
    // runtime races every concurrent `getenv`, and an env copy was gone after a
    // restart anyway — `get_openai_models` read only env, so a stored OpenAI key
    // stopped listing models once the GUI restarted.
    match provider.as_str() {
        "anthropic" => state_guard.config.llm.api_key = Some(api_key.clone()),
        "openai" => state_guard.config.llm.openai_api_key = Some(api_key.clone()),
        "brave" => state_guard.config.tools.brave_api_key = Some(api_key.clone()),
        "openrouter" => state_guard.config.llm.openrouter_api_key = Some(api_key.clone()),
        "github" => state_guard.config.llm.github_token = Some(api_key.clone()),
        "claude-proxy" => {
            // For claude-proxy, the "api_key" is actually the proxy URL
            set_claude_proxy_setting(true, Some(api_key.clone()));
        }
        _ => return Err(format!("Unknown provider: {provider}")),
    }

    // Durable storage is the OS keyring; config.toml never receives secrets.
    // (claude-proxy is a URL, not a secret — strip_secrets leaves it alone.)
    // store_secrets() refills what it stores, so in-memory state (and the OAuth
    // badge) keeps the session's credentials.
    if provider != "claude-proxy"
        && let Err(e) = state_guard.config.store_secrets()
    {
        error!("Failed to store API key in keyring: {e}");
        return Err(format!("failed to store API key securely: {e}"));
    }
    if let Err(e) = state_guard.config.save() {
        error!("Failed to save config: {}", e);
        // Non-fatal - key is hydrated in-process for this session
    }
    // Never hold AppState across a daemon round trip: every other
    // command waits on this lock for as long as the reload takes.
    let backend = Arc::clone(&state_guard.backend);
    drop(state_guard);
    let _ = backend.config_reload().await;

    info!("API key set for provider: {} (secure store)", provider);
    Ok(())
}

// =============================================================================
// Anthropic OAuth Login (via `claude setup-token`)
// =============================================================================

/// Durably persist an OAuth login, then update live state and nudge the daemon.
///
/// The `SecureStore` is written FIRST and a failure aborts the login: the config
/// cache is session-only (`strip_secrets_for_disk` blanks the token in
/// config.toml on every save), so a login that never reaches the store is
/// exactly the restart-logout bug this path exists to prevent.
async fn persist_oauth_login(
    state: &Arc<RwLock<AppState>>,
    credential: &nanna_config::OAuthCredential,
) -> Result<(), String> {
    nanna_config::SecureStore::new()
        .save_anthropic_oauth(credential)
        .map_err(|e| format!("Failed to store OAuth token securely: {e}"))?;

    let mut state_guard = state.write().await;
    state_guard.config.llm.anthropic_oauth_token = Some(credential.access_token.clone());
    state_guard.config.llm.anthropic_use_oauth = true;
    if let Err(e) = state_guard.config.save() {
        error!("Failed to save config: {e}");
    }
    // Never hold AppState across a daemon round trip: every other
    // command waits on this lock for as long as the reload takes.
    let backend = Arc::clone(&state_guard.backend);
    drop(state_guard);
    let _ = backend.config_reload().await;
    Ok(())
}

/// Run `claude setup-token` to authenticate via Claude Code CLI
/// This opens a browser for OAuth, then persists the resulting credential
///
/// # Errors
///
/// Fails when the Claude Code CLI is not installed; when `claude setup-token`
/// cannot be run (or the task running it panics); when it prints no token and
/// no CLI credentials can be loaded, or the loaded ones are expired with no
/// refresh token, or refreshing them fails; and with `Failed to store OAuth
/// token securely: …` when the credential cannot be written to the secure
/// store.
#[tauri::command]
pub async fn run_claude_setup_token(
    state: State<'_, Arc<RwLock<AppState>>>,
) -> Result<String, String> {
    use nanna_config::ClaudeCredentialManager;

    // Check if Claude CLI is available
    if !ClaudeCredentialManager::is_claude_cli_available() {
        return Err(
            "Claude Code CLI not found. Please install it first:\n\
             npm install -g @anthropic-ai/claude-code\n\n\
             Or paste your token from `claude setup-token` directly.".to_string()
        );
    }

    info!("Running claude setup-token...");

    // `claude setup-token` PRINTS the minted token — it does NOT write the CLI
    // credential file (verified on Windows, claude 2.1.71) — so capture its
    // output and parse the token out. The child is a blocking process.
    let captured = tokio::task::spawn_blocking(ClaudeCredentialManager::run_setup_token_captured)
        .await
        .map_err(|e| format!("setup-token task failed: {e}"))?
        .map_err(|e| {
            format!(
                "Failed to run claude setup-token: {e}\n\n\
                 Run `claude setup-token` in a terminal and paste the token instead."
            )
        })?;

    let credential = if let Some(cred) = captured {
        info!("claude setup-token completed (token captured from output)");
        cred
    } else {
        // No token in the output — fall back to the CLI credential store
        // (`claude login`), refusing to silently import a stale token.
        let manager = ClaudeCredentialManager::new();
        let loaded = manager.load().map_err(|e| {
            format!(
                "setup-token printed no token and no CLI credentials were found: {e}. \
                 Run `claude setup-token` in a terminal and paste the token instead."
            )
        })?;
        if loaded.credential.is_expired() {
            if !loaded.credential.can_refresh() {
                return Err(
                    "setup-token printed no token and the CLI credential store is stale. \
                     Run `claude setup-token` in a terminal and paste the token instead."
                        .to_string(),
                );
            }
            let refreshed = manager
                .refresh_token(&loaded.credential)
                .await
                .map_err(|e| format!("CLI credentials expired and refresh failed: {e}"))?;
            if let Err(e) = manager.save(&refreshed, loaded.source) {
                warn!("Failed to save refreshed token to its CLI source: {e}");
            }
            refreshed
        } else {
            loaded.credential
        }
    };

    let subscription = credential
        .subscription_type
        .clone()
        .unwrap_or_else(|| "unknown".to_string());

    persist_oauth_login(state.inner(), &credential).await?;

    info!("Successfully authenticated via claude setup-token (subscription: {})", subscription);
    Ok(format!("Successfully authenticated! Subscription: {subscription}"))
}

/// Import credentials from Claude Code CLI (~/.claude/.credentials.json)
/// This uses the token that Claude Code CLI obtained, which is whitelisted
///
/// # Errors
///
/// Fails with `No credentials found: …` when the CLI credential store has none;
/// for an expired token, with `Token expired and cannot auto-refresh…` when it
/// has no refresh token and `Token expired and refresh failed: …` when
/// refreshing fails; and with `Failed to store OAuth token securely: …` when
/// the credential cannot be written to the secure store.
#[tauri::command]
pub async fn import_claude_code_credentials(
    state: State<'_, Arc<RwLock<AppState>>>,
) -> Result<(), String> {
    use nanna_config::ClaudeCredentialManager;

    let manager = ClaudeCredentialManager::new();

    // Load credentials (checks file and keychain)
    let loaded = manager.load()
        .map_err(|e| format!("No credentials found: {e}. Please run `claude login` first."))?;

    // Check if token is expired
    if loaded.credential.is_expired() {
        if loaded.credential.can_refresh() {
            info!("Token expired, attempting auto-refresh...");
            let refreshed = manager.refresh_token(&loaded.credential).await
                .map_err(|e| format!("Token expired and refresh failed: {e}. Please run `claude login`."))?;

            // Save refreshed token back to source
            if let Err(e) = manager.save(&refreshed, loaded.source) {
                warn!("Failed to save refreshed token: {}", e);
            }

            persist_oauth_login(state.inner(), &refreshed).await?;

            info!("Token refreshed and imported (subscription: {:?})", refreshed.subscription_type);
            return Ok(());
        }
        return Err("Token expired and cannot auto-refresh. Please run `claude login`.".to_string());
    }

    info!(
        "Imported Claude Code credentials (subscription: {:?})",
        loaded.credential.subscription_type
    );

    persist_oauth_login(state.inner(), &loaded.credential).await?;

    info!("Successfully imported Claude Code credentials");
    Ok(())
}

/// Save an Anthropic OAuth token directly (from `claude setup-token`)
///
/// # Errors
///
/// Returns `Token cannot be empty` for a blank token, and `Failed to store
/// OAuth token securely: …` when the credential cannot be written to the secure
/// store.
#[tauri::command]
pub async fn save_anthropic_oauth_token(
    state: State<'_, Arc<RwLock<AppState>>>,
    token: String,
) -> Result<(), String> {
    let token = token.trim().to_string();
    if token.is_empty() {
        return Err("Token cannot be empty".to_string());
    }

    // A pasted `claude setup-token` token is long-lived and carries no
    // refresh token or expiry.
    let credential = nanna_config::OAuthCredential {
        access_token: token,
        refresh_token: None,
        expires_at: None,
        subscription_type: None,
        account_id: None,
        organization_id: None,
    };
    persist_oauth_login(state.inner(), &credential).await?;

    info!("Anthropic OAuth token saved");
    Ok(())
}

/// Log out of Anthropic OAuth (clear token and switch to API key mode)
///
/// # Errors
///
/// Returns `Failed to remove stored OAuth token: …` when the secure-store entry
/// cannot be deleted; nothing else changes then. A failed `config.toml` save
/// afterwards is only logged.
#[tauri::command]
pub async fn logout_anthropic_oauth(state: State<'_, Arc<RwLock<AppState>>>) -> Result<(), String> {
    // Remove the durable credential first — if this fails the user would be
    // silently logged back in at next launch, so surface it instead.
    nanna_config::SecureStore::new()
        .delete_anthropic_oauth()
        .map_err(|e| format!("Failed to remove stored OAuth token: {e}"))?;

    let mut state_guard = state.write().await;

    state_guard.config.llm.anthropic_oauth_token = None;
    state_guard.config.llm.anthropic_use_oauth = false;

    // Persist to config (the daemon rebuilds its LLM providers on reload)
    if let Err(e) = state_guard.config.save() {
        error!("Failed to save config after logout: {}", e);
    }
    // Never hold AppState across a daemon round trip: every other
    // command waits on this lock for as long as the reload takes.
    let backend = Arc::clone(&state_guard.backend);
    drop(state_guard);
    let _ = backend.config_reload().await;

    info!("Anthropic OAuth logout successful");
    Ok(())
}

/// Get Claude CLI credential status
#[derive(serde::Serialize)]
pub struct CredentialStatus {
    cli_available: bool,
    credentials_found: bool,
    source: Option<String>,
    /// Flattened, so its keys stay top-level on the wire.
    #[serde(flatten)]
    expiry: TokenExpiry,
    subscription_type: Option<String>,
}

/// Where a CLI access token stands against its expiry.
#[derive(serde::Serialize)]
pub struct TokenExpiry {
    is_expired: bool,
    can_refresh: bool,
    seconds_until_expiry: Option<i64>,
}

/// Report whether Claude CLI credentials are available, where they live, and
/// how close they are to expiry.
///
/// # Errors
///
/// Never returns `Err`: missing or unreadable CLI credentials are reported as
/// `credentials_found: false`.
#[tauri::command]
pub async fn get_credential_status() -> Result<CredentialStatus, String> {
    use nanna_config::{ClaudeCredentialManager, CredentialSource};

    let cli_available = ClaudeCredentialManager::is_claude_cli_available();
    let manager = ClaudeCredentialManager::new();

    match manager.load() {
        Ok(loaded) => {
            let source = match loaded.source {
                CredentialSource::File => "file",
                CredentialSource::MacOsKeychain => "macos_keychain",
                CredentialSource::WindowsCredentialManager => "windows_credential_manager",
                CredentialSource::LinuxSecretService => "linux_secret_service",
            };
            Ok(CredentialStatus {
                cli_available,
                credentials_found: true,
                source: Some(source.to_string()),
                expiry: TokenExpiry {
                    is_expired: loaded.credential.is_expired(),
                    can_refresh: loaded.credential.can_refresh(),
                    seconds_until_expiry: loaded.credential.seconds_until_expiry(),
                },
                subscription_type: loaded.credential.subscription_type,
            })
        }
        Err(_) => {
            Ok(CredentialStatus {
                cli_available,
                credentials_found: false,
                source: None,
                expiry: TokenExpiry {
                    is_expired: false,
                    can_refresh: false,
                    seconds_until_expiry: None,
                },
                subscription_type: None,
            })
        }
    }
}

/// Providers the daemon's LLM router can route to right now.
///
/// The model picker gates native provider entries on this list, not on
/// GUI-local login state — the two can disagree (e.g. an OAuth login the
/// daemon hasn't registered yet), and only the daemon actually routes chat.
/// Errors when the daemon is unreachable or predates `llm_providers`, so the
/// frontend can fall back to local gating instead of showing an empty picker.
///
/// # Errors
///
/// Fails when the daemon cannot be reached or the `system.status` request is
/// dropped or times out, and with `daemon did not report llm_providers` when
/// the status has no such array.
#[tauri::command]
pub async fn get_daemon_providers(
    state: State<'_, Arc<RwLock<AppState>>>,
) -> Result<Vec<String>, String> {
    let status = backend_handle(&state).await.system_status().await?;
    status
        .get("llm_providers")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|p| p.as_str().map(String::from))
                .collect()
        })
        .ok_or_else(|| "daemon did not report llm_providers".to_string())
}

/// Each configured MCP server's state, from the daemon's `system.status`.
///
/// Empty — not an error — when the daemon predates the field or has no MCP
/// servers configured: the Tools page then simply shows no MCP section.
///
/// # Errors
///
/// Fails when the daemon cannot be reached or the `system.status` request is
/// dropped or times out. A `mcp_servers` field that is missing or not an array
/// lists nothing rather than failing.
#[tauri::command]
pub async fn get_mcp_servers(
    state: State<'_, Arc<RwLock<AppState>>>,
) -> Result<Vec<serde_json::Value>, String> {
    let status = backend_handle(&state).await.system_status().await?;
    Ok(status
        .get("mcp_servers")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default())
}

/// Refresh the OAuth token if expired or expiring soon
///
/// # Errors
///
/// Fails with `No credentials found: …` when the CLI credential store has none,
/// `Cannot refresh: no refresh token available`, or `Token refresh failed: …`.
/// Failures to write the refreshed token back — to the CLI store, the secure
/// store or `config.toml` — are only logged.
#[tauri::command]
pub async fn refresh_oauth_token(
    state: State<'_, Arc<RwLock<AppState>>>,
) -> Result<String, String> {
    use nanna_config::ClaudeCredentialManager;

    let manager = ClaudeCredentialManager::new();
    let loaded = manager.load()
        .map_err(|e| format!("No credentials found: {e}"))?;

    if !loaded.credential.can_refresh() {
        return Err("Cannot refresh: no refresh token available".to_string());
    }

    let refreshed = manager.refresh_token(&loaded.credential).await
        .map_err(|e| format!("Token refresh failed: {e}"))?;

    // Save back to source
    if let Err(e) = manager.save(&refreshed, loaded.source) {
        warn!("Failed to save refreshed token to source: {}", e);
    }

    // Keep the durable store current so the refreshed token (not the stale
    // one) is what the next launch rehydrates. Preserves anthropic_use_oauth
    // as-is: refreshing does not opt the user into OAuth mode.
    if let Err(e) = nanna_config::SecureStore::new().save_anthropic_oauth(&refreshed) {
        warn!("Failed to persist refreshed OAuth token to secure store: {e}");
    }

    // Update the config cache and let the daemon rebuild its providers on reload.
    let mut state_guard = state.write().await;
    state_guard.config.llm.anthropic_oauth_token = Some(refreshed.access_token.clone());

    if let Err(e) = state_guard.config.save() {
        error!("Failed to save config: {}", e);
    }
    // Never hold AppState across a daemon round trip: every other
    // command waits on this lock for as long as the reload takes.
    let backend = Arc::clone(&state_guard.backend);
    drop(state_guard);
    let _ = backend.config_reload().await;

    let hours = refreshed.seconds_until_expiry().map_or(0, |s| s / 3600);
    info!("OAuth token refreshed, expires in {}h", hours);

    Ok(format!("Token refreshed! Expires in {hours}h"))
}

/// Set the active LLM provider
///
/// # Errors
///
/// Returns `Unknown provider: …` for a provider other than `anthropic`,
/// `openai`, `openrouter`, `github`, `claude-proxy` or `ollama`. For
/// `anthropic` it returns `No OAuth token available…` when no OAuth token is
/// loaded, and for `openai`, `openrouter` and `github` it returns `No API key
/// set for <provider>` when neither the config nor the provider's environment
/// variable has one. Nothing changes on error; a failed `config.toml` save is
/// only logged.
#[tauri::command]
pub async fn set_provider(
    state: State<'_, Arc<RwLock<AppState>>>,
    provider: String,
) -> Result<(), String> {
    let mut state_guard = state.write().await;

    // Validate that the selected provider has usable credentials, so we fail
    // here with a clear message rather than at chat time in the daemon. The
    // daemon owns the actual LLM client; we just persist the choice.
    match provider.as_str() {
        "anthropic" => {
            if state_guard.config.llm.anthropic_oauth_token.is_none() {
                return Err("No OAuth token available. Run `claude setup-token` or paste your token.".to_string());
            }
        }
        "openai" => {
            if state_guard.config.llm.openai_api_key.is_none() && std::env::var("OPENAI_API_KEY").is_err() {
                return Err("No API key set for openai".to_string());
            }
        }
        "openrouter" => {
            if state_guard.config.llm.openrouter_api_key.is_none() && std::env::var("OPENROUTER_API_KEY").is_err() {
                return Err("No API key set for openrouter".to_string());
            }
        }
        "github" => {
            if state_guard.config.llm.github_token.is_none() && std::env::var("GITHUB_TOKEN").is_err() {
                return Err("No API key set for github".to_string());
            }
        }
        "claude-proxy" | "ollama" => {}
        _ => return Err(format!("Unknown provider: {provider}")),
    }

    state_guard.config.llm.provider.clone_from(&provider);
    if let Err(e) = state_guard.config.save() {
        error!("Failed to save config: {}", e);
    }
    // Never hold AppState across a daemon round trip: every other
    // command waits on this lock for as long as the reload takes.
    let backend = Arc::clone(&state_guard.backend);
    drop(state_guard);
    let _ = backend.config_reload().await;

    info!("Provider changed to: {}", provider);
    Ok(())
}

/// Set the embedding provider and model (requires restart to take effect)
///
/// # Errors
///
/// Returns `Unknown embedding provider: …` for anything but `openai`, `ollama`
/// or `disabled`, and `Unknown OpenAI embedding model: …` for an `OpenAI` model
/// other than `text-embedding-3-small` or `text-embedding-3-large`; nothing
/// changes then. A failed `config.toml` save is only logged.
#[tauri::command]
pub async fn set_embedding_config(
    state: State<'_, Arc<RwLock<AppState>>>,
    provider: String,
    model: String,
) -> Result<String, String> {
    let mut state_guard = state.write().await;

    // Validate provider
    if !["openai", "ollama", "disabled"].contains(&provider.as_str()) {
        return Err(format!("Unknown embedding provider: {provider}"));
    }

    let model = if provider == "disabled" { "none".to_string() } else { model };

    // Validate OpenAI models (Ollama accepts any installed model)
    if provider == "openai" {
        let valid_openai = ["text-embedding-3-small", "text-embedding-3-large"];
        if !valid_openai.contains(&model.as_str()) {
            return Err(format!("Unknown OpenAI embedding model: {model}"));
        }
    }

    // Save to config file (the daemon reads the same file at startup)
    state_guard.config.memory.embedding_provider.clone_from(&provider);
    state_guard.config.memory.embedding_model.clone_from(&model);
    state_guard.config.memory.enabled = provider != "disabled";
    if let Err(e) = state_guard.config.save() {
        error!("Failed to save embedding config: {}", e);
    }
    drop(state_guard);

    info!("Embedding config changed to: {} / {}", provider, model);

    // Return warning about restart
    Ok("Embedding settings updated. Restart required for changes to take effect. Note: Changing embedding dimensions will make existing memories incompatible.".to_string())
}

/// Get env var status (for checking if keys are set)
///
/// # Errors
///
/// Never returns `Err`.
#[tauri::command]
pub async fn check_env_var(name: String) -> Result<bool, String> {
    Ok(std::env::var(&name).is_ok())
}

/// Set Ollama host URL
///
/// The saved bearer token does not follow the address: it is bound to the
/// server it was saved for, and a token saved by an older build (no server
/// recorded) is first bound to the address being replaced.
///
/// # Errors
///
/// Returns `Ollama host must start with http:// or https://` for any other URL,
/// and `Failed to record which server the saved Ollama token belongs to: …`
/// when an unbound token cannot be bound, before anything changes. Returns
/// `Failed to save config: …` when `config.toml` cannot be written; the cached
/// host has already changed then, but the daemon is not told.
#[tauri::command]
pub async fn set_ollama_host(
    state: State<'_, Arc<RwLock<AppState>>>,
    host: String,
) -> Result<String, String> {
    let mut state_guard = state.write().await;

    // A pasted address often carries whitespace; the scheme check must see
    // past it, and so must what gets saved.
    let host = host.trim();
    // Validate URL format
    if !host.starts_with("http://") && !host.starts_with("https://") {
        return Err("Ollama host must start with http:// or https://".to_string());
    }

    // Remove trailing slash — every call appends `/api/...` itself.
    let host = nanna_config::normalize_ollama_host(host);

    // A token with no server recorded counts as the configured server's;
    // unrecorded, it would count as the new address's too the moment it is
    // saved. Record the server it has been going to before the switch.
    let previous = state_guard.config.memory.ollama_host.clone();
    let store = nanna_config::SecureStore::new();
    if let Err(e) = store.bind_unbound_ollama_token(&previous) {
        let err_msg = format!("Failed to record which server the saved Ollama token belongs to: {e}");
        error!("{err_msg}");
        return Err(err_msg);
    }

    // Save to config file (the daemon reads the same file)
    state_guard.config.memory.ollama_host.clone_from(&host);
    // The cached token was the old server's; keep what the daemon will load.
    state_guard
        .config
        .rebind_ollama_token_if_moved(&previous, &store);
    match state_guard.config.save() {
        Ok(()) => {
            info!("Ollama host saved to config: {}", host);
        }
        Err(e) => {
            let err_msg = format!("Failed to save config: {e}");
            error!("{}", err_msg);
            return Err(err_msg);
        }
    }
    let _ = state_guard.backend.config_reload().await;
    // Held from the validation through the save and the reload notice, so a
    // concurrent config write cannot land between them.
    drop(state_guard);
    // No `OLLAMA_HOST` copy in this process's environment: nothing in the GUI
    // reads it (the daemon is a separate process and reads the config file),
    // and `set_var` here raced concurrent `getenv` on the multi-threaded runtime.

    Ok(format!("Ollama host saved: {host}"))
}

/// Save `key` as the Ollama token for the server at `host`, the only one it
/// will be sent to; blank — whitespace included — removes the saved token
/// and its server. Returns whether a token is saved now.
fn store_ollama_token(
    store: &nanna_config::SecureStore,
    key: &str,
    host: &str,
) -> Result<bool, String> {
    let saving = !key.trim().is_empty();
    store.save_ollama_token(key, host).map_err(|e| {
        if saving {
            format!("Failed to store Ollama API key securely: {e}")
        } else {
            format!("Failed to remove stored Ollama API key: {e}")
        }
    })?;
    Ok(saving)
}

/// Set Ollama API key (for remote/authenticated instances)
///
/// Saved for the configured `[memory].ollama_host` — the only server it is
/// ever sent to. A blank key (whitespace included) removes the saved token.
///
/// # Errors
///
/// Returns `Failed to store Ollama API key securely: …` when the OS keyring
/// write fails (no token is saved then — not even the old one),
/// `Failed to remove stored Ollama API key: …` when clearing the key cannot
/// delete its stored entry, and `Failed to save config: …` when `config.toml`
/// cannot be written.
#[tauri::command]
pub async fn set_ollama_api_key(
    state: State<'_, Arc<RwLock<AppState>>>,
    key: String,
) -> Result<String, String> {
    let mut state_guard = state.write().await;
    let host = state_guard.config.memory.ollama_host.clone();
    let saved = store_ollama_token(&nanna_config::SecureStore::new(), &key, &host).inspect_err(|e| {
        error!("{e}");
    })?;
    // Refill from the store (and environment), so the cache holds what the
    // daemon will load: the new token, or none once removed.
    state_guard.config.llm.ollama_api_key = None;
    state_guard.config.load_secrets_from_store();
    match state_guard.config.save() {
        Ok(()) => {
            info!("Ollama API key saved to OS keychain");
        }
        Err(e) => {
            let err_msg = format!("Failed to save config: {e}");
            error!("{}", err_msg);
            return Err(err_msg);
        }
    }
    // Never hold AppState across a daemon round trip: every other
    // command waits on this lock for as long as the reload takes.
    let backend = Arc::clone(&state_guard.backend);
    drop(state_guard);
    let _ = backend.config_reload().await;
    Ok(if saved { "Ollama token saved" } else { "Ollama token removed" }.to_string())
}

/// Fetch available models from Ollama.
///
/// Answers from the daemon's `system.probe_ollama` — the one hardened probe
/// (bounded body, model cap, connect timeout, "answered but is not Ollama")
/// — rather than a private `GET /api/tags`. The GUI links no `nanna-llm`
/// (P16), so the daemon is the only place the probe can run. A server that
/// is down is an `Err` naming why; a server that is up lists what it has.
///
/// # Errors
///
/// `Failed to connect to Ollama at …: <reason>` when the daemon reports the
/// server unreachable, or the daemon's own error text when the probe request
/// itself failed.
#[tauri::command]
pub async fn get_ollama_models(
    state: State<'_, Arc<RwLock<AppState>>>,
) -> Result<Vec<OllamaModelInfo>, String> {
    // No address: the daemon probes the server it is configured for. This
    // process's copy was read at startup and goes stale after a hand edit of
    // `config.toml`, which the daemon applies live.
    let backend = backend_handle(&state).await;
    let report = backend.system_probe_ollama(None, Vec::new()).await?;
    ollama_models_from_report(&report)
}

/// Turn a `system.probe_ollama` answer into the picker's model list, or the
/// reason nothing could be listed. Pure, so the daemon's wire shape is
/// pinned by a test on this side too.
fn ollama_models_from_report(report: &serde_json::Value) -> Result<Vec<OllamaModelInfo>, String> {
    if let Some(error) = report.get("error").and_then(|e| e.as_str()) {
        return Err(error.to_string());
    }
    let base_url = report
        .get("base_url")
        .and_then(|u| u.as_str())
        .unwrap_or("the configured Ollama host");
    if report.get("reachable").and_then(serde_json::Value::as_bool) != Some(true) {
        let reason = report
            .get("reason")
            .and_then(|r| r.as_str())
            .unwrap_or("no usable answer");
        return Err(format!("Failed to connect to Ollama at {base_url}: {reason}"));
    }
    let models = report
        .get("models")
        .and_then(|m| m.as_array())
        .map_or_default(|models| {
            models
                .iter()
                .filter_map(|m| {
                    let name = m.get("name").and_then(|n| n.as_str())?.to_string();
                    let size_bytes = m.get("size_bytes").and_then(serde_json::Value::as_u64).unwrap_or(0);
                    Some(OllamaModelInfo {
                        is_embedding_model: is_ollama_embedding_model(&name),
                        size_mb: size_bytes / 1_000_000,
                        name,
                    })
                })
                .collect()
        });
    Ok(models)
}

/// Does a model name look like an embedding model? Name-based, because
/// `/api/tags` says nothing about a model's purpose.
fn is_ollama_embedding_model(name: &str) -> bool {
    // Comprehensive list of known embedding model name patterns
    let embedding_patterns = [
        // BGE family
        "bge-m3", "bge-large", "bge-small", "bge-base",
        // Nomic
        "nomic-embed",
        // MixedBread
        "mxbai-embed",
        // Sentence transformers / all-minilm
        "all-minilm", "minilm",
        // Snowflake
        "snowflake-arctic-embed",
        // E5 family
        "e5-small", "e5-base", "e5-large", "e5-mistral",
        // GTE family
        "gte-small", "gte-base", "gte-large", "gte-qwen",
        // Jina
        "jina-embed",
        // Voyage
        "voyage",
        // Cohere
        "embed-english", "embed-multilingual",
        // Generic patterns (catch-all)
        "-embed-", "-embed:",
    ];
    let name_lower = name.to_lowercase();
    let base_name = name.split(':').next().unwrap_or(name).to_lowercase();
    name_lower.contains("embed") || embedding_patterns.iter().any(|p| base_name.contains(p))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaModelInfo {
    pub name: String,
    pub size_mb: u64,
    pub is_embedding_model: bool,
}

/// One model the daemon says is configured but not installed, with the
/// command that installs it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaMissingModel {
    pub name: String,
    pub pull: String,
}

/// The onboarding wizard's two questions, answered separately: is the
/// server running, and is each configured model pulled?
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaProbeResult {
    /// The server that was asked.
    pub base_url: String,
    /// Did anything usable answer `GET /api/tags`?
    pub reachable: bool,
    /// Why not, in words — only when `reachable` is false.
    pub reason: Option<String>,
    /// Every model installed on a reachable server.
    pub models: Vec<OllamaModelInfo>,
    /// The models that were checked for, as Ollama names them.
    pub wanted: Vec<String>,
    /// Configured models the reachable server does not have. Empty when the
    /// server is down: a dead server has said nothing about models.
    pub missing: Vec<OllamaMissingModel>,
}

/// Probe an Ollama server for the onboarding wizard / setup assistant.
///
/// `base_url` blank/absent = the configured host; `models` empty = every
/// Ollama model the config names. Unlike `get_ollama_models`, a down server
/// is a *result* (`reachable: false`), not an error — the wizard renders
/// "start Ollama" and "pull these models" as different next steps.
///
/// # Errors
///
/// Fails when the daemon cannot be asked (no connection, a dropped or
/// timed-out request) or answers with an error instead of a report.
#[tauri::command]
pub async fn probe_ollama(
    state: State<'_, Arc<RwLock<AppState>>>,
    base_url: Option<String>,
    models: Option<Vec<String>>,
) -> Result<OllamaProbeResult, String> {
    // The probe is a network round-trip (up to its connect timeout), so the
    // app-state lock is released before it rather than held across it.
    let backend = backend_handle(&state).await;
    let report = backend
        .system_probe_ollama(base_url.as_deref(), models.unwrap_or_default())
        .await?;
    ollama_probe_from_report(&report)
}

/// Decode the daemon's `system.probe_ollama` answer. Pure.
fn ollama_probe_from_report(report: &serde_json::Value) -> Result<OllamaProbeResult, String> {
    if let Some(error) = report.get("error").and_then(|e| e.as_str()) {
        return Err(error.to_string());
    }
    let str_list = |key: &str| -> Vec<String> {
        report
            .get(key)
            .and_then(|v| v.as_array())
            .map_or_default(|a| a.iter().filter_map(|s| s.as_str().map(str::to_string)).collect())
    };
    let reachable = report.get("reachable").and_then(serde_json::Value::as_bool) == Some(true);
    let models = if reachable {
        ollama_models_from_report(report)?
    } else {
        Vec::new()
    };
    let missing = report
        .get("missing")
        .and_then(|v| v.as_array())
        .map_or_default(|a| {
            a.iter()
                .filter_map(|m| {
                    let name = m.get("name").and_then(|n| n.as_str())?.to_string();
                    let pull = m
                        .get("pull")
                        .and_then(|p| p.as_str())
                        .map_or_else(|| format!("ollama pull {name}"), str::to_string);
                    Some(OllamaMissingModel { name, pull })
                })
                .collect()
        });
    Ok(OllamaProbeResult {
        base_url: report
            .get("base_url")
            .and_then(|u| u.as_str())
            .unwrap_or_default()
            .to_string(),
        reachable,
        reason: report.get("reason").and_then(|r| r.as_str()).map(str::to_string),
        models,
        wanted: str_list("wanted"),
        missing,
    })
}

#[cfg(test)]
mod ollama_probe_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_down_server_is_an_error_for_the_picker_and_a_result_for_the_wizard() {
        let report = json!({
            "base_url": "http://localhost:11434",
            "reachable": false,
            "reason": "connection refused or host unreachable",
            "models": [], "wanted": ["qwen3.5:9b"], "missing": []
        });
        let err = ollama_models_from_report(&report).expect_err("down is an error here");
        assert!(err.contains("localhost:11434") && err.contains("connection refused"), "{err}");

        let probe = ollama_probe_from_report(&report).expect("down is a result here");
        assert!(!probe.reachable);
        assert_eq!(probe.reason.as_deref(), Some("connection refused or host unreachable"));
        assert!(probe.models.is_empty());
        // Down is not "every model is missing".
        assert!(probe.missing.is_empty());
        assert_eq!(probe.wanted, vec!["qwen3.5:9b"]);
    }

    #[test]
    fn a_live_server_lists_models_with_sizes_and_names_what_is_missing() {
        let report = json!({
            "base_url": "http://localhost:11434",
            "reachable": true,
            "models": [
                { "name": "qwen3.5:9b", "size_bytes": 6_000_000_000_u64 },
                { "name": "nomic-embed-text:latest", "size_bytes": 274_000_000_u64 }
            ],
            "wanted": ["qwen3.5:9b", "gemma4:12b"],
            "missing": [{ "name": "gemma4:12b", "pull": "ollama pull gemma4:12b" }]
        });
        let models = ollama_models_from_report(&report).expect("listed");
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].name, "qwen3.5:9b");
        assert_eq!(models[0].size_mb, 6000);
        assert!(!models[0].is_embedding_model);
        assert_eq!(models[1].size_mb, 274);
        assert!(models[1].is_embedding_model);

        let probe = ollama_probe_from_report(&report).expect("decoded");
        assert!(probe.reachable && probe.reason.is_none());
        assert_eq!(probe.models.len(), 2);
        assert_eq!(probe.missing.len(), 1);
        assert_eq!(probe.missing[0].name, "gemma4:12b");
        assert_eq!(probe.missing[0].pull, "ollama pull gemma4:12b");
    }

    #[test]
    fn a_daemon_error_envelope_is_surfaced_not_read_as_down() {
        let report = json!({ "error": "Config not available" });
        assert_eq!(ollama_models_from_report(&report).unwrap_err(), "Config not available");
        assert_eq!(ollama_probe_from_report(&report).unwrap_err(), "Config not available");
    }

    #[test]
    fn embedding_models_are_recognised_by_name() {
        assert!(is_ollama_embedding_model("nomic-embed-text:latest"));
        assert!(is_ollama_embedding_model("BGE-M3:latest"));
        assert!(is_ollama_embedding_model("all-minilm:22m"));
        assert!(!is_ollama_embedding_model("qwen3.5:9b"));
        assert!(!is_ollama_embedding_model("gemma4:12b"));
    }
}

/// Fetch available models from Anthropic
///
/// # Errors
///
/// Returns `OAuth enabled but no token available` or `No Anthropic API key
/// configured` when the selected authentication has no credential. Fails with
/// the HTTP client builder's error when the client cannot be built, `Failed to
/// fetch Anthropic models: …` when the request fails, `Anthropic API error
/// <status>: <body>` for a non-success status, and `Failed to parse Anthropic
/// response: …` when the body is not the expected model list.
#[tauri::command]
pub async fn get_anthropic_models(
    state: State<'_, Arc<RwLock<AppState>>>,
) -> Result<Vec<ModelInfo>, String> {
    #[derive(Deserialize)]
    struct AnthropicModelsResponse {
        data: Vec<AnthropicModel>,
    }

    #[derive(Deserialize)]
    struct AnthropicModel {
        id: String,
        display_name: Option<String>,
    }

    // A snapshot of the credentials: the fetch below is a network call, which
    // must not hold the state lock.
    let (use_oauth, oauth_token, config_api_key) = {
        let state_guard = state.read().await;
        (
            state_guard.config.llm.anthropic_use_oauth,
            state_guard.config.llm.anthropic_oauth_token.clone(),
            state_guard.config.llm.api_key.clone(),
        )
    };

    // Check if OAuth is configured, otherwise use API key
    let (auth_header, auth_value) = if use_oauth {
        let token = oauth_token
            .ok_or("OAuth enabled but no token available")?;
        ("Authorization".to_string(), format!("Bearer {token}"))
    } else {
        let api_key = config_api_key
            .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok())
            .ok_or("No Anthropic API key configured")?;
        ("x-api-key".to_string(), api_key)
    };

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .map_err(|e| e.to_string())?;

    let mut request = client
        .get("https://api.anthropic.com/v1/models")
        .header(&auth_header, &auth_value)
        .header("anthropic-version", "2023-06-01");

    // Add OAuth-specific headers if using OAuth
    if use_oauth {
        request = request
            .header("anthropic-beta", "claude-code-20250219,oauth-2025-04-20")
            .header("user-agent", "claude-code/2.1.2");
    }

    let response = request
        .send()
        .await
        .map_err(|e| format!("Failed to fetch Anthropic models: {e}"))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!("Anthropic API error {status}: {body}"));
    }

    let models: AnthropicModelsResponse = response.json().await
        .map_err(|e| format!("Failed to parse Anthropic response: {e}"))?;

    Ok(models.data.into_iter().map(|m| ModelInfo {
        id: m.id.clone(),
        name: m.display_name.unwrap_or(m.id),
    }).collect())
}

/// Fetch available models from `OpenAI`
///
/// # Errors
///
/// Returns `No OpenAI API key configured` when neither the config nor
/// `OPENAI_API_KEY` has a key. Fails with the HTTP client builder's error when
/// the client cannot be built, `Failed to fetch OpenAI models: …` when the
/// request fails, `OpenAI API error <status>: <body>` for a non-success status,
/// and `Failed to parse OpenAI response: …` when the body is not the expected
/// model list.
#[tauri::command]
pub async fn get_openai_models(
    state: State<'_, Arc<RwLock<AppState>>>,
) -> Result<Vec<ModelInfo>, String> {
    #[derive(Deserialize)]
    struct OpenAIModelsResponse {
        data: Vec<OpenAIModel>,
    }

    #[derive(Deserialize)]
    struct OpenAIModel {
        id: String,
    }

    // Config first: it holds the keyring copy, which survives a restart.
    let api_key = state.read().await.config.llm.openai_api_key.clone()
        .or_else(|| std::env::var("OPENAI_API_KEY").ok())
        .ok_or("No OpenAI API key configured")?;

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .map_err(|e| e.to_string())?;

    let response = client
        .get("https://api.openai.com/v1/models")
        .header("Authorization", format!("Bearer {api_key}"))
        .send()
        .await
        .map_err(|e| format!("Failed to fetch OpenAI models: {e}"))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!("OpenAI API error {status}: {body}"));
    }

    let models: OpenAIModelsResponse = response.json().await
        .map_err(|e| format!("Failed to parse OpenAI response: {e}"))?;

    // Filter to chat models (gpt-*, o1-*, chatgpt-*)
    let chat_prefixes = ["gpt-4", "gpt-3.5", "o1", "o3", "chatgpt"];
    let embedding_prefixes = ["text-embedding"];

    let mut result: Vec<ModelInfo> = models.data.into_iter()
        .filter(|m| {
            chat_prefixes.iter().any(|p| m.id.starts_with(p)) ||
            embedding_prefixes.iter().any(|p| m.id.starts_with(p))
        })
        .map(|m| ModelInfo {
            id: m.id.clone(),
            name: m.id,
        })
        .collect();

    // Sort by name
    result.sort_by(|a, b| a.id.cmp(&b.id));

    Ok(result)
}

/// Fetch available models from `OpenRouter`
///
/// # Errors
///
/// Returns `No OpenRouter API key configured` when neither the config nor
/// `OPENROUTER_API_KEY` has a key. Fails with the HTTP client builder's error
/// when the client cannot be built, `Failed to fetch OpenRouter models: …` when
/// the request fails, `OpenRouter API error <status>: <body>` for a non-success
/// status, and `Failed to parse OpenRouter response: …` when the body is not
/// the expected model list.
#[tauri::command]
pub async fn get_openrouter_models(
    state: State<'_, Arc<RwLock<AppState>>>,
) -> Result<Vec<ModelInfo>, String> {
    #[derive(Deserialize)]
    struct OpenRouterModelsResponse {
        data: Vec<OpenRouterModel>,
    }

    #[derive(Deserialize)]
    struct OpenRouterModel {
        id: String,
        name: Option<String>,
    }

    let api_key = state.read().await.config.llm.openrouter_api_key.clone()
        .or_else(|| std::env::var("OPENROUTER_API_KEY").ok())
        .ok_or("No OpenRouter API key configured")?;

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .map_err(|e| e.to_string())?;

    let response = client
        .get("https://openrouter.ai/api/v1/models")
        .header("Authorization", format!("Bearer {api_key}"))
        .send()
        .await
        .map_err(|e| format!("Failed to fetch OpenRouter models: {e}"))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!("OpenRouter API error {status}: {body}"));
    }

    let models: OpenRouterModelsResponse = response.json().await
        .map_err(|e| format!("Failed to parse OpenRouter response: {e}"))?;

    // Priority prefixes for sorting (these appear first)
    let priority_prefixes = [
        "anthropic/claude",
        "openai/gpt",
        "openai/o1",
        "openai/o3",
        "openai/chatgpt",
        "google/gemini",
        "deepseek/",
        "meta-llama/",
        "mistralai/",
        "qwen/",
        "cohere/",
        "perplexity/",
    ];

    // Include ALL models (no filtering)
    let mut result: Vec<ModelInfo> = models.data.into_iter()
        .map(|m| ModelInfo {
            name: m.name.unwrap_or_else(|| m.id.clone()),
            id: m.id,
        })
        .collect();

    // Sort: priority models first, then alphabetically
    result.sort_by(|a, b| {
        let a_priority = priority_prefixes.iter().position(|p| a.id.starts_with(p)).unwrap_or(999);
        let b_priority = priority_prefixes.iter().position(|p| b.id.starts_with(p)).unwrap_or(999);
        a_priority.cmp(&b_priority).then_with(|| a.id.cmp(&b.id))
    });

    Ok(result)
}

/// Fetch available embedding models from `OpenRouter`'s dedicated embeddings endpoint
///
/// # Errors
///
/// Returns `No OpenRouter API key configured` when neither the config nor
/// `OPENROUTER_API_KEY` has a key. Fails with the HTTP client builder's error
/// when the client cannot be built, `Failed to fetch OpenRouter embedding
/// models: …` when the request fails, `OpenRouter embeddings API error
/// <status>: <body>` for a non-success status, and `Failed to parse OpenRouter
/// embeddings response: …` when the body is not the expected model list.
#[tauri::command]
pub async fn get_openrouter_embedding_models(
    state: State<'_, Arc<RwLock<AppState>>>,
) -> Result<Vec<ModelInfo>, String> {
    #[derive(Deserialize)]
    struct OpenRouterModelsResponse {
        data: Vec<OpenRouterEmbeddingModel>,
    }

    #[derive(Deserialize)]
    struct OpenRouterEmbeddingModel {
        id: String,
        name: Option<String>,
    }

    let api_key = state.read().await.config.llm.openrouter_api_key.clone()
        .or_else(|| std::env::var("OPENROUTER_API_KEY").ok())
        .ok_or("No OpenRouter API key configured")?;

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .map_err(|e| e.to_string())?;

    let response = client
        .get("https://openrouter.ai/api/v1/embeddings/models")
        .header("Authorization", format!("Bearer {api_key}"))
        .send()
        .await
        .map_err(|e| format!("Failed to fetch OpenRouter embedding models: {e}"))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!("OpenRouter embeddings API error {status}: {body}"));
    }

    let models: OpenRouterModelsResponse = response.json().await
        .map_err(|e| format!("Failed to parse OpenRouter embeddings response: {e}"))?;

    let result: Vec<ModelInfo> = models.data.into_iter()
        .map(|m| ModelInfo {
            name: m.name.unwrap_or_else(|| m.id.clone()),
            id: m.id,
        })
        .collect();

    Ok(result)
}

/// Fetch available models from GitHub Models API
///
/// # Errors
///
/// Returns `No GitHub token configured` when neither the config nor
/// `GITHUB_TOKEN` has a token. Fails with the HTTP client builder's error when
/// the client cannot be built, `Failed to fetch GitHub models: …` when the
/// request fails, `GitHub Models API error <status>: <body>` for a non-success
/// status, `Failed to read GitHub response: …` when the body cannot be read,
/// and `Failed to parse GitHub response: <body>` when it is neither a model
/// array nor an object carrying one.
#[tauri::command]
pub async fn get_github_models(
    state: State<'_, Arc<RwLock<AppState>>>,
) -> Result<Vec<ModelInfo>, String> {
    #[derive(Deserialize)]
    struct GitHubModelsResponse {
        data: Option<Vec<GitHubModel>>,
        #[serde(default)]
        models: Vec<GitHubModel>,
    }

    #[derive(Deserialize)]
    struct GitHubModel {
        id: Option<String>,
        name: Option<String>,
        #[serde(default)]
        model_name: Option<String>,
    }

    let api_key = state.read().await.config.llm.github_token.clone()
        .or_else(|| std::env::var("GITHUB_TOKEN").ok())
        .ok_or("No GitHub token configured")?;

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .map_err(|e| e.to_string())?;

    // GitHub Models catalog endpoint
    let response = client
        .get("https://models.inference.ai.azure.com/models")
        .header("Authorization", format!("Bearer {api_key}"))
        .send()
        .await
        .map_err(|e| format!("Failed to fetch GitHub models: {e}"))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!("GitHub Models API error {status}: {body}"));
    }

    let text = response.text().await
        .map_err(|e| format!("Failed to read GitHub response: {e}"))?;

    // Try to parse as JSON array or object with data/models field
    let models: Vec<GitHubModel> = if let Ok(arr) = serde_json::from_str::<Vec<GitHubModel>>(&text) {
        arr
    } else if let Ok(resp) = serde_json::from_str::<GitHubModelsResponse>(&text) {
        resp.data.unwrap_or(resp.models)
    } else {
        return Err(format!("Failed to parse GitHub response: {text}"));
    };

    // Filter and map models
    let result: Vec<ModelInfo> = models.into_iter()
        .filter_map(|m| {
            let id = m.id.or(m.model_name)?;
            let name = m.name.unwrap_or_else(|| id.clone());
            Some(ModelInfo { id, name })
        })
        .collect();

    Ok(result)
}

/// Fetch available models from Anthropic API for use with Claude Proxy
/// This queries Anthropic directly to get models available on your subscription
///
/// # Errors
///
/// Fails with the HTTP client builder's error when the client cannot be built,
/// `Failed to fetch models: …` when the request fails, `Anthropic API error
/// <status>: <body>` for a non-success status, and `Failed to parse Anthropic
/// response: …` for an unexpected body. With no Anthropic credential at all it
/// returns a fixed default list instead of an error.
#[tauri::command]
pub async fn get_claude_proxy_models(
    state: State<'_, Arc<RwLock<AppState>>>,
) -> Result<Vec<ModelInfo>, String> {
    #[derive(Deserialize)]
    struct AnthropicModelsResponse {
        data: Vec<AnthropicModel>,
    }

    #[derive(Deserialize)]
    struct AnthropicModel {
        id: String,
        display_name: Option<String>,
    }

    // A snapshot of the credentials: the fetch below is a network call, which
    // must not hold the state lock.
    let (oauth_token, config_api_key) = {
        let state_guard = state.read().await;
        (
            state_guard.config.llm.anthropic_oauth_token.clone(),
            state_guard.config.llm.api_key.clone(),
        )
    };

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .map_err(|e| e.to_string())?;

    // Try OAuth first (for Pro/Max subscription), then API key
    let response = if let Some(token) = oauth_token {
        client
            .get("https://api.anthropic.com/v1/models")
            .header("Authorization", format!("Bearer {token}"))
            .header("anthropic-version", "2023-06-01")
            .header("anthropic-beta", "claude-code-20250219,oauth-2025-04-20")
            .header("user-agent", "claude-code/2.1.2")
            .send()
            .await
            .map_err(|e| format!("Failed to fetch models: {e}"))?
    } else if let Some(ref api_key) = config_api_key {
        client
            .get("https://api.anthropic.com/v1/models")
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .send()
            .await
            .map_err(|e| format!("Failed to fetch models: {e}"))?
    } else if let Ok(api_key) = std::env::var("ANTHROPIC_API_KEY") {
        client
            .get("https://api.anthropic.com/v1/models")
            .header("x-api-key", &api_key)
            .header("anthropic-version", "2023-06-01")
            .send()
            .await
            .map_err(|e| format!("Failed to fetch models: {e}"))?
    } else {
        // No Anthropic credentials - return default Claude models that the proxy supports
        return Ok(vec![
            ModelInfo { id: "claude-sonnet-4-20250514".to_string(), name: "Claude Sonnet 4".to_string() },
            ModelInfo { id: "claude-opus-4-20250514".to_string(), name: "Claude Opus 4".to_string() },
            ModelInfo { id: "claude-3-5-sonnet-20241022".to_string(), name: "Claude Sonnet 3.5".to_string() },
            ModelInfo { id: "claude-3-5-haiku-20241022".to_string(), name: "Claude Haiku 3.5".to_string() },
        ]);
    };

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!("Anthropic API error {status}: {body}"));
    }

    let models: AnthropicModelsResponse = response.json().await
        .map_err(|e| format!("Failed to parse Anthropic response: {e}"))?;

    // Filter to chat models only (exclude embedding models, etc.)
    let result: Vec<ModelInfo> = models.data.into_iter()
        .filter(|m| m.id.starts_with("claude-"))
        .map(|m| {
            let name = m.display_name.unwrap_or_else(|| format_claude_model_name(&m.id));
            ModelInfo { id: m.id, name }
        })
        .collect();

    Ok(result)
}

/// Enable or disable Claude Proxy
///
/// # Errors
///
/// Never returns `Err`.
#[tauri::command]
pub async fn set_claude_proxy(enabled: bool, url: Option<String>) -> Result<(), String> {
    // Disabling keeps the URL, as clearing only the flag always did.
    set_claude_proxy_setting(enabled, url.filter(|_| enabled));
    Ok(())
}

/// Check if Claude Proxy is running and reachable
///
/// # Errors
///
/// Returns the HTTP client builder's error when the client cannot be built. An
/// unreachable or unhealthy proxy is `Ok(false)`.
#[tauri::command]
pub async fn check_claude_proxy_health() -> Result<bool, String> {
    let (_, proxy_url) = claude_proxy_setting();

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()
        .map_err(|e| e.to_string())?;

    Ok(client
        .get(format!("{proxy_url}/health"))
        .send()
        .await
        .is_ok_and(|resp| resp.status().is_success()))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
}

// =============================================================================
// System Prompt & Agent Settings
// =============================================================================

/// Get the custom system prompt (returns None if using default)
///
/// # Errors
///
/// Never returns `Err`; the `Result` is what Tauri requires of an async command
/// that borrows `State`.
#[tauri::command]
pub async fn get_system_prompt(
    state: State<'_, Arc<RwLock<AppState>>>,
) -> Result<Option<String>, String> {
    let state_guard = state.read().await;
    Ok(state_guard.config.agent.system_prompt.clone())
}

/// Set a custom system prompt (pass null to reset to default)
///
/// # Errors
///
/// Returns `Failed to save config: …` when `config.toml` cannot be written; the
/// cached value has already changed by then.
#[tauri::command]
pub async fn set_system_prompt(
    state: State<'_, Arc<RwLock<AppState>>>,
    prompt: Option<String>,
) -> Result<(), String> {
    let mut state_guard = state.write().await;
    state_guard.config.agent.system_prompt.clone_from(&prompt);

    // Save to disk
    state_guard.config.save()
        .map_err(|e| format!("Failed to save config: {e}"))?;
    drop(state_guard);

    info!("System prompt {}", if prompt.is_some() { "updated" } else { "reset to default" });
    Ok(())
}

/// Set agent name
///
/// # Errors
///
/// Returns `Failed to save config: …` when `config.toml` cannot be written; the
/// cached value has already changed by then.
#[tauri::command]
pub async fn set_agent_name(
    state: State<'_, Arc<RwLock<AppState>>>,
    name: String,
) -> Result<(), String> {
    let mut state_guard = state.write().await;
    state_guard.config.agent.name.clone_from(&name);
    state_guard.config.save()
        .map_err(|e| format!("Failed to save config: {e}"))?;
    drop(state_guard);
    info!("Agent name set to: {}", name);
    Ok(())
}

/// Set personality mode
///
/// # Errors
///
/// Returns `Failed to save config: …` when `config.toml` cannot be written; the
/// cached value has already changed by then.
#[tauri::command]
pub async fn set_personality_mode(
    state: State<'_, Arc<RwLock<AppState>>>,
    mode: String,
) -> Result<(), String> {
    let mut state_guard = state.write().await;
    state_guard.config.agent.personality_mode.clone_from(&mode);
    state_guard.config.save()
        .map_err(|e| format!("Failed to save config: {e}"))?;
    drop(state_guard);
    info!("Personality mode set to: {}", mode);
    Ok(())
}

// NOTE: `set_thinking_enabled` was removed 2026-08-04 (owner directive:
// thinking is always on, and the Settings switch that turned it off is gone).
// Thinking mode is now `nanna_agent::ThinkingMode::default()` everywhere, with
// the only remaining control the internal per-run `RunOptions::thinking_mode`
// override.

/// Set streaming enabled
///
/// # Errors
///
/// Returns `Failed to save config: …` when `config.toml` cannot be written; the
/// cached value has already changed by then.
#[tauri::command]
pub async fn set_streaming_enabled(
    state: State<'_, Arc<RwLock<AppState>>>,
    enabled: bool,
) -> Result<(), String> {
    let mut state_guard = state.write().await;
    state_guard.config.agent.streaming_enabled = enabled;
    state_guard.config.save()
        .map_err(|e| format!("Failed to save config: {e}"))?;
    drop(state_guard);
    info!("Streaming: {}", if enabled { "enabled" } else { "disabled" });
    Ok(())
}

/// Set max tokens for responses
///
/// # Errors
///
/// Returns `Failed to save config: …` when `config.toml` cannot be written; the
/// cached value has already changed by then.
#[tauri::command]
pub async fn set_max_tokens(
    state: State<'_, Arc<RwLock<AppState>>>,
    tokens: u32,
) -> Result<(), String> {
    let mut state_guard = state.write().await;
    state_guard.config.llm.max_tokens = tokens;
    state_guard.config.save()
        .map_err(|e| format!("Failed to save config: {e}"))?;
    drop(state_guard);
    info!("Max tokens set to: {}", tokens);
    Ok(())
}

/// Set the agent-loop iteration policy.
///
/// The loop is a long-horizon worker: `max_iterations` is an optional absolute
/// backstop (`None`/0 = unlimited — only Stop/cancel or the model finishing ends
/// it). Escalating soft nudges begin at `nudge_after` and repeat every
/// `nudge_interval` iterations; they steer a possibly-stuck model but never stop it.
///
/// # Errors
///
/// Returns `Failed to save config: …` when `config.toml` cannot be written; the
/// cached value has already changed by then.
#[tauri::command]
pub async fn set_agent_iteration_policy(
    state: State<'_, Arc<RwLock<AppState>>>,
    max_iterations: Option<usize>,
    nudge_after: usize,
    nudge_interval: usize,
) -> Result<(), String> {
    // Treat 0 (or absent) max as "unlimited". Floor the nudge knobs at 1 so the
    // schedule is always well-defined.
    let max_iterations = max_iterations.filter(|&m| m > 0);
    let nudge_after = nudge_after.max(1);
    let nudge_interval = nudge_interval.max(1);

    let mut state_guard = state.write().await;
    state_guard.config.agent.max_iterations = max_iterations;
    state_guard.config.agent.nudge_after_iterations = nudge_after;
    state_guard.config.agent.nudge_interval_iterations = nudge_interval;
    state_guard.config.save()
        .map_err(|e| format!("Failed to save config: {e}"))?;
    drop(state_guard);
    info!(
        "Agent iteration policy set: max={:?}, nudge_after={}, nudge_interval={}",
        max_iterations, nudge_after, nudge_interval
    );
    Ok(())
}

/// Export config as TOML string, without secrets
///
/// # Errors
///
/// Returns `Failed to serialize config: …` when the config cannot be rendered
/// as TOML.
#[tauri::command]
pub async fn export_config(
    state: State<'_, Arc<RwLock<AppState>>>,
) -> Result<String, String> {
    let state_guard = state.read().await;
    exported_config_toml(&state_guard.config)
}

/// `config` as the TOML an export hands the webview and a downloaded file:
/// without secrets, as `config.toml` itself is written. The cached copy holds
/// the ones hydrated from the keychain; the keychain is where they live.
fn exported_config_toml(config: &nanna_config::Config) -> Result<String, String> {
    let mut exported = config.clone();
    exported.strip_secrets_for_disk();
    toml::to_string_pretty(&exported).map_err(|e| format!("Failed to serialize config: {e}"))
}

/// The config an import of `text` makes the cached copy, replacing `running`.
///
/// The Ollama token is the running one, kept only if the imported address is
/// the same server, and otherwise re-derived for it as for any address change
/// — never one carried in the text. The file is saved without secrets and the
/// daemon loads it from there, so a carried token never reaches the daemon;
/// left in this copy, the next key save would file it under the imported
/// address, over the token saved for the server it belongs to.
fn imported_config(
    text: &str,
    running: &nanna_config::Config,
    store: &nanna_config::SecureStore,
) -> Result<nanna_config::Config, String> {
    let mut imported: nanna_config::Config =
        toml::from_str(text).map_err(|e| format!("Failed to parse config: {e}"))?;
    imported
        .llm
        .ollama_api_key
        .clone_from(&running.llm.ollama_api_key);
    imported.rebind_ollama_token_if_moved(&running.memory.ollama_host, store);
    Ok(imported)
}

/// Import config from TOML string
///
/// The daemon performs the import (`config.import`): it files any secret the
/// text brings in, keeps every stored one the text leaves out, saves, and
/// applies the result live. This command used to only write `config.toml` —
/// the running daemon never saw the import until a restart, and this process's
/// cached copy lost every keychain secret, because an export carries none.
/// The cached copy is then refilled from the store the daemon just settled.
///
/// # Errors
///
/// Returns `Failed to parse config: …` when the text is not a valid config,
/// the daemon's refusal when it refuses, or the transport error when it cannot
/// be reached; nothing changes in any of those cases.
#[tauri::command]
pub async fn import_config(
    state: State<'_, Arc<RwLock<AppState>>>,
    config: String,
) -> Result<(), String> {
    let store = nanna_config::SecureStore::new();
    let (running_host, mut new_config) = {
        let state_guard = state.read().await;
        let running_host = state_guard.config.memory.ollama_host.clone();
        (
            running_host,
            imported_config(&config, &state_guard.config, &store)?,
        )
    };
    // The daemon gets the text's own config — secrets it brings in included,
    // for the daemon to file — not this copy, whose Ollama token is ours.
    let as_written: nanna_config::Config =
        toml::from_str(&config).map_err(|e| format!("Failed to parse config: {e}"))?;
    let wire =
        serde_json::to_value(&as_written).map_err(|e| format!("Failed to encode config: {e}"))?;
    let reply = backend_handle(&state).await.config_import(wire).await?;
    import_refusal(&reply)?;

    new_config.refill_secrets_replacing(&running_host, &store);
    state.write().await.config = new_config;
    info!("Config imported from TOML");
    Ok(())
}

/// The daemon's refusal in a `config.import` reply, if it refused.
fn import_refusal(reply: &serde_json::Value) -> Result<(), String> {
    let Some(error) = reply.get("error") else {
        return Ok(());
    };
    let reason = reply
        .get("message")
        .and_then(serde_json::Value::as_str)
        .map_or_else(|| error.to_string(), str::to_string);
    Err(format!("config import refused: {reason}"))
}

// =============================================================================
// Model Priority (Fallback Chains)
// =============================================================================

/// Get chat model priority list
///
/// # Errors
///
/// Never returns `Err`; the `Result` is what Tauri requires of an async command
/// that borrows `State`.
#[tauri::command]
pub async fn get_chat_model_priority(
    state: State<'_, Arc<RwLock<AppState>>>,
) -> Result<Vec<String>, String> {
    let state_guard = state.read().await;
    Ok(state_guard.config.llm.model_priority.clone())
}

/// Set chat model priority list
///
/// # Errors
///
/// Returns `Failed to save config: …` when `config.toml` cannot be written; the
/// cached priority, primary model and badge have already changed then, but the
/// daemon is not told and no `model-status` event is emitted. The push to the
/// daemon after a successful save is best-effort.
#[tauri::command]
pub async fn set_chat_model_priority(
    app: AppHandle,
    state: State<'_, Arc<RwLock<AppState>>>,
    priority: Vec<String>,
) -> Result<(), String> {
    let mut state_guard = state.write().await;
    state_guard.config.llm.model_priority.clone_from(&priority);

    // Also set the primary model to the first in the list for backwards compatibility
    let new_active = priority.first().cloned().unwrap_or_default();
    if !new_active.is_empty() {
        state_guard.config.llm.model.clone_from(&new_active);
    }

    // Update active_model so the badge reflects the change immediately
    state_guard.active_model.write().await.clone_from(&new_active);

    state_guard.config.save()
        .map_err(|e| format!("Failed to save config: {e}"))?;

    // Propagate to the daemon so changes take effect without restart
    // Clone the handle and release the lock before the round trip: the
    // save above stays ordered under it, the daemon call does not need it.
    let backend = state_guard.backend.clone();
    drop(state_guard);
    let _ = backend.config_set(
        "llm.model_priority",
        serde_json::to_value(&priority).unwrap_or_default(),
    ).await;

    // Emit model-status event so the GUI badge updates
    let _ = app.emit("model-status", ModelStatusEvent {
        active_model: new_active,
        fallback_reason: None,
        rate_limited_models: vec![],
    });

    info!("Chat model priority set: {:?}", priority);
    Ok(())
}

/// Get embedding model priority list
///
/// # Errors
///
/// Never returns `Err`; the `Result` is what Tauri requires of an async command
/// that borrows `State`.
#[tauri::command]
pub async fn get_embedding_model_priority(
    state: State<'_, Arc<RwLock<AppState>>>,
) -> Result<Vec<String>, String> {
    let state_guard = state.read().await;
    Ok(state_guard.config.memory.embedding_priority.clone())
}

/// Set embedding model priority list
///
/// # Errors
///
/// Returns `Failed to save config: …` when `config.toml` cannot be written; the
/// cached value has already changed by then.
#[tauri::command]
pub async fn set_embedding_model_priority(
    state: State<'_, Arc<RwLock<AppState>>>,
    priority: Vec<String>,
) -> Result<(), String> {
    let mut state_guard = state.write().await;
    state_guard.config.memory.embedding_priority.clone_from(&priority);

    // Update the primary embedding config for backwards compatibility
    if let Some(first) = priority.first() {
        if let Some((provider, model)) = first.split_once('/') {
            state_guard.config.memory.embedding_provider = provider.to_string();
            state_guard.config.memory.embedding_model = model.to_string();
        }
    } else {
        state_guard.config.memory.embedding_provider = "disabled".to_string();
    }

    state_guard.config.save()
        .map_err(|e| format!("Failed to save config: {e}"))?;
    drop(state_guard);

    info!("Embedding model priority set: {:?}", priority);
    Ok(())
}

/// Get summarization model priority list
///
/// # Errors
///
/// Never returns `Err`; the `Result` is what Tauri requires of an async command
/// that borrows `State`.
#[tauri::command]
pub async fn get_summarization_model_priority(
    state: State<'_, Arc<RwLock<AppState>>>,
) -> Result<Vec<String>, String> {
    let state_guard = state.read().await;
    Ok(state_guard.config.llm.summarization_priority.clone())
}

/// Set summarization model priority list
///
/// # Errors
///
/// Returns `Failed to save config: …` when `config.toml` cannot be written; the
/// cached value has already changed by then. The daemon reload is best-effort.
#[tauri::command]
pub async fn set_summarization_model_priority(
    state: State<'_, Arc<RwLock<AppState>>>,
    priority: Vec<String>,
) -> Result<(), String> {
    let mut state_guard = state.write().await;
    state_guard.config.llm.summarization_priority.clone_from(&priority);

    state_guard.config.save()
        .map_err(|e| format!("Failed to save config: {e}"))?;
    // Tell the daemon now, as the other settings saves do. It would find the
    // change by itself — its file watcher polls every ~2 s — but a turn that
    // starts inside that window would summarize on the old list.
    // Never hold AppState across a daemon round trip: every other
    // command waits on this lock for as long as the reload takes.
    let backend = Arc::clone(&state_guard.backend);
    drop(state_guard);
    let _ = backend.config_reload().await;

    info!("Summarization model priority set: {:?}", priority);
    Ok(())
}

// =============================================================================
// OCR Configuration Commands
// =============================================================================

/// Get OCR model priority list (vision-capable models used for text extraction)
///
/// # Errors
///
/// Never returns `Err`; the `Result` is what Tauri requires of an async command
/// that borrows `State`.
#[tauri::command]
pub async fn get_ocr_model_priority(
    state: State<'_, Arc<RwLock<AppState>>>,
) -> Result<Vec<String>, String> {
    let state_guard = state.read().await;
    Ok(state_guard.config.memory.ocr_model_priority.clone())
}

/// Set OCR model priority list
///
/// # Errors
///
/// Returns `Failed to save config: …` when `config.toml` cannot be written; the
/// cached value has already changed by then.
#[tauri::command]
pub async fn set_ocr_model_priority(
    state: State<'_, Arc<RwLock<AppState>>>,
    priority: Vec<String>,
) -> Result<(), String> {
    let mut state_guard = state.write().await;
    state_guard.config.memory.ocr_model_priority.clone_from(&priority);

    state_guard.config.save()
        .map_err(|e| format!("Failed to save config: {e}"))?;
    drop(state_guard);

    info!("OCR model priority set: {:?}", priority);
    Ok(())
}

/// Get whether embedded OCR (ocrs) is enabled
///
/// # Errors
///
/// Never returns `Err`; the `Result` is what Tauri requires of an async command
/// that borrows `State`.
#[tauri::command]
pub async fn get_use_embedded_ocr(
    state: State<'_, Arc<RwLock<AppState>>>,
) -> Result<bool, String> {
    let state_guard = state.read().await;
    Ok(state_guard.config.memory.use_embedded_ocr)
}

/// Set whether embedded OCR (ocrs) is enabled
///
/// # Errors
///
/// Returns `Failed to save config: …` when `config.toml` cannot be written; the
/// cached value has already changed by then.
#[tauri::command]
pub async fn set_use_embedded_ocr(
    state: State<'_, Arc<RwLock<AppState>>>,
    enabled: bool,
) -> Result<(), String> {
    let mut state_guard = state.write().await;
    state_guard.config.memory.use_embedded_ocr = enabled;

    state_guard.config.save()
        .map_err(|e| format!("Failed to save config: {e}"))?;
    drop(state_guard);

    info!("Embedded OCR (ocrs) set to: {}", enabled);
    Ok(())
}

// =============================================================================
// Model Routing Commands
// =============================================================================

/// Get model routing configuration
///
/// # Errors
///
/// Never returns `Err`; the `Result` is what Tauri requires of an async command
/// that borrows `State`.
#[tauri::command]
pub async fn get_model_routing(
    state: State<'_, Arc<RwLock<AppState>>>,
) -> Result<Vec<String>, String> {
    let state_guard = state.read().await;
    Ok(state_guard.config.llm.model_routing.clone())
}

/// Set model routing configuration
/// Each entry is "model:tier" where tier is simple|medium|complex
///
/// # Errors
///
/// Returns `Failed to save config: …` when `config.toml` cannot be written; the
/// cached value has already changed by then. The push to the daemon
/// (`config.set` of `llm.model_routing`) happens only after a successful save
/// and is best-effort: it never fails the command.
#[tauri::command]
pub async fn set_model_routing(
    state: State<'_, Arc<RwLock<AppState>>>,
    routes: Vec<String>,
) -> Result<(), String> {
    let mut state_guard = state.write().await;
    state_guard.config.llm.model_routing.clone_from(&routes);

    state_guard.config.save()
        .map_err(|e| format!("Failed to save config: {e}"))?;

    // Propagate to the daemon
    // Clone the handle and release the lock before the round trip: the
    // save above stays ordered under it, the daemon call does not need it.
    let backend = state_guard.backend.clone();
    drop(state_guard);
    let _ = backend.config_set(
        "llm.model_routing",
        serde_json::to_value(&routes).unwrap_or_default(),
    ).await;

    info!("Model routing set: {:?}", routes);
    Ok(())
}

/// Get `routing_first_turn_primary` setting
///
/// # Errors
///
/// Never returns `Err`; the `Result` is what Tauri requires of an async command
/// that borrows `State`.
#[tauri::command]
pub async fn get_routing_first_turn_primary(
    state: State<'_, Arc<RwLock<AppState>>>,
) -> Result<bool, String> {
    let state_guard = state.read().await;
    Ok(state_guard.config.llm.routing_first_turn_primary)
}

/// Set `routing_first_turn_primary` setting
///
/// # Errors
///
/// Returns `Failed to save config: …` when `config.toml` cannot be written; the
/// cached value has already changed by then. The push to the daemon
/// (`config.set` of `llm.routing_first_turn_primary`) happens only after a
/// successful save and is best-effort: it never fails the command.
#[tauri::command]
pub async fn set_routing_first_turn_primary(
    state: State<'_, Arc<RwLock<AppState>>>,
    enabled: bool,
) -> Result<(), String> {
    let mut state_guard = state.write().await;
    state_guard.config.llm.routing_first_turn_primary = enabled;

    state_guard.config.save()
        .map_err(|e| format!("Failed to save config: {e}"))?;

    // Propagate to the daemon
    // Clone the handle and release the lock before the round trip: the
    // save above stays ordered under it, the daemon call does not need it.
    let backend = state_guard.backend.clone();
    drop(state_guard);
    let _ = backend.config_set(
        "llm.routing_first_turn_primary",
        serde_json::Value::Bool(enabled),
    ).await;

    info!("Routing first turn primary set: {}", enabled);
    Ok(())
}

/// Get the sub-agent model priority list (raw stored value; the legacy single
/// `sub_agent_model` is folded in so pre-list configs show what they run).
///
/// # Errors
///
/// Never returns `Err`; the `Result` is what Tauri requires of an async command
/// that borrows `State`.
#[tauri::command]
pub async fn get_sub_agent_models(
    state: State<'_, Arc<RwLock<AppState>>>,
) -> Result<Vec<String>, String> {
    let state_guard = state.read().await;
    let llm = &state_guard.config.llm;
    let models = match &llm.sub_agent_model {
        Some(legacy) if llm.sub_agent_models.is_empty() && !legacy.is_empty() => {
            vec![legacy.clone()]
        }
        _ => llm.sub_agent_models.clone(),
    };
    drop(state_guard);
    Ok(models)
}

/// Set the sub-agent model priority list. Empty = sub-agents use the main
/// chat model list. Saving migrates away the legacy single-model field.
///
/// # Errors
///
/// Returns `Failed to save config: …` when `config.toml` cannot be written; the
/// cached list has already changed and the legacy single model been cleared by
/// then. The pushes to the daemon after a successful save are best-effort.
#[tauri::command]
pub async fn set_sub_agent_models(
    state: State<'_, Arc<RwLock<AppState>>>,
    models: Vec<String>,
) -> Result<(), String> {
    let mut state_guard = state.write().await;
    state_guard.config.llm.sub_agent_models.clone_from(&models);
    // The list is now the source of truth — a lingering single would
    // resurrect itself whenever the list is cleared.
    state_guard.config.llm.sub_agent_model = None;

    state_guard.config.save()
        .map_err(|e| format!("Failed to save config: {e}"))?;

    // Propagate to the daemon
    // Clone the handle and release the lock before the round trip: the
    // save above stays ordered under it, the daemon call does not need it.
    let backend = state_guard.backend.clone();
    drop(state_guard);
    let _ = backend.config_set(
        "llm.sub_agent_models",
        serde_json::json!(models),
    ).await;
    let _ = backend.config_set(
        "llm.sub_agent_model",
        serde_json::Value::Null,
    ).await;

    info!("Sub-agent models set: {:?}", models);
    Ok(())
}

// =============================================================================
// Data storage location (`[general] data_dir`)
// =============================================================================

/// Where the daemon keeps its store, as the GUI can report it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataDirInfo {
    /// The directory the current config selects (override or platform default).
    pub effective: String,
    /// The platform default, so the UI can offer "reset to default".
    pub default: String,
    /// Whether `[general] data_dir` names somewhere other than the default.
    pub is_custom: bool,
}

/// Report the configured data directory.
///
/// Reads the in-memory config the GUI already holds; nothing here touches the
/// daemon, because the GUI is a pure client and the value the daemon *booted*
/// with may differ from the value on disk until it restarts — which is exactly
/// what the UI tells the user.
///
/// # Errors
///
/// Fails when the platform data directory cannot be determined, or the
/// configured one cannot be resolved.
#[tauri::command]
pub async fn get_data_dir(
    state: State<'_, Arc<RwLock<AppState>>>,
) -> Result<DataDirInfo, String> {
    let state_guard = state.read().await;
    let default = nanna_config::Config::default_data_dir()
        .map_err(|e| format!("Cannot determine the platform data directory: {e}"))?;
    let effective = state_guard
        .config
        .resolve_data_dir()
        .map_err(|e| format!("Cannot resolve the data directory: {e}"))?;
    Ok(DataDirInfo {
        effective: effective.display().to_string(),
        default: default.display().to_string(),
        is_custom: state_guard.config.has_custom_data_dir(),
    })
}

/// Set (or, with `None` / blank, clear) the configured data directory.
///
/// Validates the folder first — absolute, a directory, creatable, writable —
/// and refuses rather than persisting a path the daemon cannot use. Writes
/// `config.toml` only; **no data is moved**. The daemon reads this at boot,
/// so the change takes effect on its next restart, and the returned info is
/// what the UI shows while stating that.
///
/// # Errors
///
/// Fails when the chosen folder is refused by `validate_data_dir` (relative,
/// a file, not creatable or not writable), when `config.toml` cannot be
/// saved, or when the platform or configured data directory cannot be
/// resolved afterwards.
#[tauri::command]
pub async fn set_data_dir(
    state: State<'_, Arc<RwLock<AppState>>>,
    path: Option<String>,
) -> Result<DataDirInfo, String> {
    let chosen = path
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty())
        .map(std::path::PathBuf::from);

    if let Some(dir) = &chosen {
        nanna_config::validate_data_dir(dir).map_err(|e| e.to_string())?;
    }

    let mut state_guard = state.write().await;
    state_guard.config.general.data_dir.clone_from(&chosen);
    state_guard
        .config
        .save()
        .map_err(|e| format!("Failed to save config: {e}"))?;

    if let Some(dir) = &chosen {
        info!(
            "Data directory set to {} — takes effect when the daemon restarts; existing data is not moved",
            dir.display()
        );
    } else {
        info!("Data directory reset to the platform default — takes effect when the daemon restarts");
    }

    let default = nanna_config::Config::default_data_dir()
        .map_err(|e| format!("Cannot determine the platform data directory: {e}"))?;
    let effective = state_guard
        .config
        .resolve_data_dir()
        .map_err(|e| format!("Cannot resolve the data directory: {e}"))?;
    Ok(DataDirInfo {
        effective: effective.display().to_string(),
        default: default.display().to_string(),
        is_custom: state_guard.config.has_custom_data_dir(),
    })
}

// =============================================================================
// Config Persistence Commands
// =============================================================================

/// Save config to disk
///
/// # Errors
///
/// Returns `Failed to save config: …` when `config.toml` cannot be written.
#[tauri::command]
pub async fn save_config(
    state: State<'_, Arc<RwLock<AppState>>>,
) -> Result<(), String> {
    state.read().await.config.save()
        .map_err(|e| format!("Failed to save config: {e}"))?;

    info!("Config saved to disk");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A refusal is surfaced with the daemon's message; an import is not.
    #[test]
    fn an_import_refusal_carries_the_daemons_message() {
        assert_eq!(
            import_refusal(&serde_json::json!({ "status": "imported" })),
            Ok(())
        );
        let refused =
            serde_json::json!({ "error": "secret_store_failed", "message": "keyring locked" });
        assert_eq!(
            import_refusal(&refused),
            Err("config import refused: keyring locked".to_string())
        );
        let bare = serde_json::json!({ "error": "import_failed" });
        assert!(import_refusal(&bare).is_err_and(|e| e.contains("import_failed")));
    }

    /// The proxy setting lives in memory, not the process environment, and
    /// keeps the old command semantics: enabling may set a URL, disabling
    /// keeps it, and no URL means the default. The only test touching the
    /// process-wide setting, so it cannot race another.
    #[tokio::test]
    async fn the_claude_proxy_setting_is_in_memory() {
        set_claude_proxy(true, Some("http://127.0.0.1:9".into()))
            .await
            .expect("never fails");
        assert_eq!(
            claude_proxy_setting(),
            (true, "http://127.0.0.1:9".to_string())
        );
        set_claude_proxy(false, Some("http://ignored".into()))
            .await
            .expect("never fails");
        assert_eq!(
            claude_proxy_setting(),
            (false, "http://127.0.0.1:9".to_string()),
            "disabling keeps the URL"
        );
        assert!(
            std::env::var_os("CLAUDE_PROXY_URL").is_none_or(|v| v != "http://127.0.0.1:9"),
            "the environment is not written"
        );
    }

    fn sample_settings() -> ExtendedSettings {
        ExtendedSettings {
            llm_api_keys: LlmApiKeys {
                anthropic_key_set: true,
                openai_key_set: false,
                openrouter_key_set: true,
            },
            github_key_set: false,
            claude_proxy: ClaudeProxyStatus {
                claude_proxy_enabled: true,
                claude_proxy_url: "http://localhost:3456".to_string(),
            },
            brave_key_set: true,
            anthropic_oauth: AnthropicOauthStatus {
                anthropic_oauth_logged_in: false,
                anthropic_use_oauth: true,
            },
            provider: "ollama".to_string(),
            available_providers: vec!["ollama".to_string()],
            model: "qwen2.5".to_string(),
            available_models: vec!["qwen2.5".to_string()],
            embedding_provider: "ollama".to_string(),
            embedding_model: "nomic-embed-text".to_string(),
            available_embedding_providers: vec!["disabled".to_string()],
            available_embedding_models: vec!["all-minilm".to_string()],
            embedding_enabled: true,
            extraction_model: String::new(),
            available_extraction_models: vec![String::new()],
            ollama_host: "http://127.0.0.1:11434".to_string(),
            ollama_token: OllamaTokenStatus {
                ollama_token_saved: true,
                ollama_token_host: Some("http://127.0.0.1:11434".to_string()),
                ollama_token_from_env: false,
            },
            temperature: 1.0,
            top_p: 0.95,
            max_tokens: 8192,
            tools: vec![ToolInfo {
                name: "exec".to_string(),
                description: "run".to_string(),
                enabled: true,
                is_user_tool: false,
            }],
            memory: MemorySettings {
                auto_remember_messages: false,
                max_compression_ratio: 0.5,
                min_remaining_memories: 20,
            },
            scheduler: SchedulerSettings {
                scheduler_enabled: false,
                heartbeat_enabled: true,
                heartbeat_interval_seconds: 1800,
            },
            agent_max_iterations: None,
            agent_nudge_after_iterations: 40,
            agent_nudge_interval_iterations: 10,
        }
    }

    /// The flattened groups must serialize to exactly the bytes the flat
    /// struct did — every key top-level, in the original order — and read
    /// back to the same value.
    #[test]
    fn extended_settings_wire_shape_is_unchanged() {
        let flat = concat!(
            r#"{"anthropic_key_set":true,"openai_key_set":false,"openrouter_key_set":true,"#,
            r#""github_key_set":false,"claude_proxy_enabled":true,"claude_proxy_url":"http://localhost:3456","#,
            r#""brave_key_set":true,"anthropic_oauth_logged_in":false,"anthropic_use_oauth":true,"#,
            r#""provider":"ollama","available_providers":["ollama"],"model":"qwen2.5","available_models":["qwen2.5"],"#,
            r#""embedding_provider":"ollama","embedding_model":"nomic-embed-text","#,
            r#""available_embedding_providers":["disabled"],"available_embedding_models":["all-minilm"],"#,
            r#""embedding_enabled":true,"extraction_model":"","available_extraction_models":[""],"#,
            r#""ollama_host":"http://127.0.0.1:11434","ollama_token_saved":true,"#,
            r#""ollama_token_host":"http://127.0.0.1:11434","ollama_token_from_env":false,"#,
            r#""temperature":1.0,"top_p":0.95,"#,
            r#""max_tokens":8192,"tools":[{"name":"exec","description":"run","enabled":true,"is_user_tool":false}],"#,
            r#""auto_remember_messages":false,"max_compression_ratio":0.5,"#,
            r#""min_remaining_memories":20,"scheduler_enabled":false,"heartbeat_enabled":true,"#,
            r#""heartbeat_interval_seconds":1800,"agent_max_iterations":null,"#,
            r#""agent_nudge_after_iterations":40,"agent_nudge_interval_iterations":10}"#,
        );
        let settings = sample_settings();
        assert_eq!(serde_json::to_string(&settings).expect("serializes"), flat);

        let read_back: ExtendedSettings = serde_json::from_str(flat).expect("deserializes");
        assert_eq!(read_back.llm_api_keys, settings.llm_api_keys);
        assert_eq!(read_back.claude_proxy, settings.claude_proxy);
        assert_eq!(read_back.anthropic_oauth, settings.anthropic_oauth);
        assert_eq!(read_back.memory, settings.memory);
        assert_eq!(read_back.scheduler, settings.scheduler);
        assert_eq!(read_back.ollama_token, settings.ollama_token);
        assert_eq!(serde_json::to_string(&read_back).expect("serializes"), flat);
    }

    /// A hermetic credential store: its own directory, never the OS keyring.
    fn file_store() -> (tempfile::TempDir, nanna_config::SecureStore) {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = nanna_config::SecureStore::file_only_at(dir.path().to_path_buf());
        (dir, store)
    }

    #[test]
    fn the_settings_page_is_never_sent_the_ollama_token() {
        let (_dir, store) = file_store();
        store_ollama_token(&store, "s3cret-bearer-token", "https://host/ollama/").expect("save");
        let mut config = nanna_config::Config::default();
        config.memory.ollama_host = "https://host/ollama".to_string();
        config.llm.ollama_api_key = Some("s3cret-bearer-token".to_string());

        let status = ollama_token_status(&store, &config.memory.ollama_host, None);
        let settings = build_extended_settings(&config, Vec::new(), status);
        let wire = serde_json::to_string(&settings).expect("serializes");
        assert!(
            !wire.contains("s3cret-bearer-token"),
            "the webview must never receive the token: {wire}"
        );
        // What it gets instead: that one is saved, and for which server.
        assert!(wire.contains(r#""ollama_token_saved":true,"ollama_token_host":"https://host/ollama""#), "{wire}");
    }

    #[test]
    fn the_token_status_names_its_server_or_the_configured_one_for_a_legacy_token() {
        let (_dir, store) = file_store();
        assert_eq!(
            ollama_token_status(&store, "http://localhost:11434", None),
            OllamaTokenStatus {
                ollama_token_saved: false,
                ollama_token_host: None,
                ollama_token_from_env: false
            }
        );
        store_ollama_token(&store, "token-for-a", "https://a.example/ollama").expect("save");
        assert_eq!(
            ollama_token_status(&store, "https://b.example/ollama", None)
                .ollama_token_host
                .as_deref(),
            Some("https://a.example/ollama"),
            "bound elsewhere: the page says where, so it can say it is not sent here"
        );
        store
            .delete(nanna_config::credentials::keys::OLLAMA_API_KEY_HOST)
            .expect("unbind");
        assert_eq!(
            ollama_token_status(&store, "https://b.example/ollama/", None)
                .ollama_token_host
                .as_deref(),
            Some("https://b.example/ollama"),
            "a legacy token is the configured server's"
        );
    }

    #[test]
    fn an_exported_config_carries_no_secrets() {
        // Settings -> Data -> Export hands this text to the webview and to a
        // downloaded file.
        let mut config = nanna_config::Config::default();
        config.memory.ollama_host = "https://host/ollama".to_string();
        config.llm.ollama_api_key = Some("s3cret-bearer-token".to_string());
        config.llm.api_key = Some("sk-ant-s3cret".to_string());
        let text = exported_config_toml(&config).expect("serializes");
        assert!(
            !text.contains("s3cret-bearer-token") && !text.contains("sk-ant-s3cret"),
            "an export must not carry secrets: {text}"
        );
        assert!(
            text.contains("https://host/ollama"),
            "the rest is exported: {text}"
        );
    }

    #[test]
    fn an_imported_config_brings_no_ollama_token_along() {
        // The running config holds server A's token; the imported text names
        // server B and carries a token of its own. The file is saved without
        // it, so the daemon never sees it — but left in this copy, the next
        // key save would file it under B over A's.
        let (_dir, store) = file_store();
        store_ollama_token(&store, "token-for-a", "https://a.example/ollama").expect("save");
        let mut running = nanna_config::Config::default();
        running.memory.ollama_host = "https://a.example/ollama".to_string();
        running.llm.ollama_api_key = Some("token-for-a".to_string());

        let mut text = nanna_config::Config::default();
        text.memory.ollama_host = "https://b.example/ollama".to_string();
        text.llm.ollama_api_key = Some("carried-token".to_string());
        let imported = imported_config(
            &toml::to_string_pretty(&text).expect("toml"),
            &running,
            &store,
        )
        .expect("parses");
        assert_eq!(imported.memory.ollama_host, "https://b.example/ollama");
        assert!(
            !matches!(
                imported.llm.ollama_api_key.as_deref(),
                Some("carried-token" | "token-for-a")
            ),
            "B gets neither the carried token nor A's: {:?}",
            imported.llm.ollama_api_key
        );

        // Importing the same server keeps the token it has.
        text.memory.ollama_host = "https://A.example/ollama/".to_string();
        let imported = imported_config(
            &toml::to_string_pretty(&text).expect("toml"),
            &running,
            &store,
        )
        .expect("parses");
        assert_eq!(imported.llm.ollama_api_key.as_deref(), Some("token-for-a"));
    }

    #[test]
    fn the_token_status_says_when_the_environment_supplies_the_token() {
        // `OLLAMA_API_KEY` goes to whatever address is configured, ahead of
        // the saved token: a page that only described the saved one would say
        // "not sent to this address" while the daemon sends a token there.
        let (_dir, store) = file_store();
        store_ollama_token(&store, "token-for-a", "https://a.example/ollama").expect("save");
        let status = ollama_token_status(&store, "https://b.example", Some("env-token"));
        assert!(status.ollama_token_from_env);
        assert_eq!(
            status.ollama_token_host.as_deref(),
            Some("https://a.example/ollama")
        );
        let wire = serde_json::to_string(&status).expect("serializes");
        assert!(
            !wire.contains("env-token"),
            "never the token itself: {wire}"
        );
        assert!(
            !ollama_token_status(&store, "https://b.example", Some("  ")).ollama_token_from_env,
            "a blank variable supplies nothing"
        );
    }

    #[test]
    fn a_whitespace_token_removes_the_saved_one() {
        let (_dir, store) = file_store();
        store_ollama_token(&store, "old-token", "https://host/ollama").expect("save");
        let saved = store_ollama_token(&store, "   ", "https://host/ollama").expect("clear");
        assert!(!saved, "whitespace is no token");
        assert_eq!(store.ollama_token(), None, "the old token must be gone");
        assert_eq!(
            store.ollama_token_host().expect("a file store answers"),
            None,
            "and the record of its server"
        );
    }

    #[test]
    fn a_saved_token_is_trimmed_and_bound_to_the_configured_server() {
        let (_dir, store) = file_store();
        assert!(store_ollama_token(&store, "  s3cret \n", " https://host/ollama/ ").expect("save"));
        assert_eq!(store.ollama_token().as_deref(), Some("s3cret"));
        assert_eq!(
            store
                .ollama_token_host()
                .expect("a file store answers")
                .as_deref(),
            Some("https://host/ollama")
        );
    }

    #[test]
    fn credential_status_wire_shape_is_unchanged() {
        let status = CredentialStatus {
            cli_available: true,
            credentials_found: true,
            source: Some("file".to_string()),
            expiry: TokenExpiry {
                is_expired: false,
                can_refresh: true,
                seconds_until_expiry: Some(3600),
            },
            subscription_type: None,
        };
        assert_eq!(
            serde_json::to_string(&status).expect("serializes"),
            r#"{"cli_available":true,"credentials_found":true,"source":"file","is_expired":false,"can_refresh":true,"seconds_until_expiry":3600,"subscription_type":null}"#
        );
    }
}
