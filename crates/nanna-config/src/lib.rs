#![warn(clippy::all)]
#![warn(clippy::pedantic, clippy::nursery)]

//! Configuration management for Nanna

pub mod credentials;
/// Local on-device inference config (`[infer]`) and the boot-time decisions
/// derived from it.
pub mod infer;
pub use infer::{
    DEFAULT_LOCAL_EMBEDDING_MODEL, InferConfig, InferDevice, InferPrecision, LocalPlan,
    PrecisionReason, ResolvedPrecision,
};
pub mod bind;
/// MCP servers started at boot (`[mcp]`).
pub mod mcp;
pub use mcp::{MCP_SERVERS_MAX, McpConfig, McpServerEntry, mcp_secret_key};
/// Which Ollama server an address names — what the bearer token is bound to.
pub mod ollama;
pub use ollama::{normalize_ollama_host, ollama_server_changed, same_ollama_server};
/// Each provider's API key in its own `[llm]` field and keyring entry, and
/// the move of keys the old layout filed as Anthropic's.
mod provider_key;
/// Channel secrets (bot tokens, signing and webhook secrets) in the secure
/// store, and out of `config.toml`.
mod channel_secrets;
/// Filing the secrets a change brings in before a save strips them.
mod secret_filing;
pub use secret_filing::EnvSuppliedSecret;
/// `[server].webhook_secret` in the secure store, and out of `config.toml`.
mod server_secret;
/// The `[llm]` keys and `[tools].brave_api_key` a `config.toml` holds, filed
/// in the secure store as it loads.
mod api_keys;
/// Forgetting a secret someone clears, and keeping the ones a change lacks.
mod secret_clearing;

/// Canonical application identity for [`directories::ProjectDirs`].
///
/// Every Nanna surface (config, credentials, daemon data, model cache, GUI skills)
/// MUST use these three components so a single uninstall removes the whole tree and
/// so secrets never end up orphaned under a different vendor slug.
pub const APP_QUALIFIER: &str = "com";
pub const APP_ORGANIZATION: &str = "nanna";
pub const APP_NAME: &str = "nanna";

/// Build the canonical [`ProjectDirs`] for Nanna.
///
/// # Errors
///
/// Returns `None` only when the host has no home directory.
#[must_use]
pub fn project_dirs() -> Option<ProjectDirs> {
    ProjectDirs::from(APP_QUALIFIER, APP_ORGANIZATION, APP_NAME)
}

/// Legacy pre-unification identity (`bot/clawd/Nanna`).
///
/// Kept solely so the first
/// boot after upgrade can migrate existing config and credential files into the
/// canonical tree instead of stranding a user's data under the old vendor slug.
#[must_use]
pub fn legacy_clawd_project_dirs() -> Option<ProjectDirs> {
    ProjectDirs::from("bot", "clawd", "Nanna")
}


pub use bind::{DEFAULT_IPC_PORT, LOOPBACK_HOST, default_daemon_ws_url, is_loopback_host};
pub use credentials::{
    ClaudeCredentialManager, CredentialError, CredentialSource, LoadedCredential, OAuthCredential,
    SecureStore, resolve_anthropic_oauth,
};

use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;
use tracing::{info, warn};

#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("TOML parse error: {0}")]
    TomlParse(#[from] toml::de::Error),
    #[error("TOML serialize error: {0}")]
    TomlSerialize(#[from] toml::ser::Error),
    #[error("Config directory not found")]
    NoDirFound,
    #[error("Missing required field: {0}")]
    MissingField(String),
}

/// Main configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct Config {
    /// General settings
    pub general: GeneralConfig,
    /// LLM provider settings
    pub llm: LlmConfig,
    /// Agent personality settings
    pub agent: AgentConfig,
    /// Server settings
    pub server: ServerConfig,
    /// Channel configurations
    pub channels: ChannelsConfig,
    /// Tool settings
    pub tools: ToolsConfig,
    /// Memory settings
    pub memory: MemoryConfig,
    /// Background scheduler settings (heartbeat + cron runner)
    pub scheduler: SchedulerConfig,
    /// Local (on-device) inference settings — the Mummu runner.
    pub infer: InferConfig,
    /// MCP servers the daemon starts at boot.
    pub mcp: McpConfig,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GeneralConfig {
    pub name: String,
    pub log_level: String,
    pub workspace: Option<PathBuf>,
    /// Where Nanna keeps its data: the Turso database, the tools directory,
    /// logs, screenshots and generated audio. `None` (the default) means the
    /// platform location from [`project_dirs`].
    ///
    /// This lives in `config.toml` rather than in the data dir itself for the
    /// obvious reason — a pointer stored inside the thing it points at cannot
    /// be read before you know where it is. Config and data are separate
    /// directories under [`project_dirs`], so there is no circularity.
    ///
    /// Changing this does **not** move an existing store. The daemon opens
    /// whatever is at the new path and creates an empty one if nothing is
    /// there; the old directory is left untouched. Callers that surface this
    /// setting must say so — silently starting empty looks exactly like data
    /// loss to the person it happens to.
    ///
    /// The daemon's `--data-dir` flag still wins over this value, so an
    /// isolated run (`NANNA_DEV_DATA_DIR`, a test harness) is never captured by
    /// the operator's configured location.
    #[serde(default)]
    pub data_dir: Option<PathBuf>,
}

impl Default for GeneralConfig {
    fn default() -> Self {
        Self {
            name: "Nanna".to_string(),
            log_level: "info".to_string(),
            workspace: None,
            data_dir: None,
        }
    }
}

/// Why a proposed data directory was refused.
///
/// Separate from [`ConfigError`] because every variant here is shown directly
/// to a person choosing a folder, so each one has to name the actual problem
/// and not "IO error". A `ConfigError::Io` tells a user nothing they can act on.
#[derive(Debug, thiserror::Error)]
pub enum DataDirError {
    #[error("No folder was given")]
    Empty,
    #[error(
        "'{0}' is a relative path. Use an absolute path — the daemon and the app \
         run from different working directories, so a relative path would mean two \
         different folders."
    )]
    NotAbsolute(String),
    #[error("'{0}' is a file, not a folder")]
    NotADirectory(String),
    #[error("'{path}' could not be created: {reason}")]
    CannotCreate { path: String, reason: String },
    #[error("'{path}' is not writable: {reason}")]
    NotWritable { path: String, reason: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LlmConfig {
    pub provider: String,
    pub model: String,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub max_tokens: u32,
    pub temperature: f32,
    /// `OpenAI` API key for embeddings (semantic memory)
    pub openai_api_key: Option<String>,
    /// `OpenRouter` API key for multi-provider access
    pub openrouter_api_key: Option<String>,
    /// GitHub token for GitHub Models
    pub github_token: Option<String>,
    /// Model priority list for fallback (first working model is used)
    /// Format: `["claude-opus-5", "claude-sonnet-5", "ollama/llama3.2"]`
    pub model_priority: Vec<String>,
    /// Anthropic OAuth access token (alternative to API key)
    pub anthropic_oauth_token: Option<String>,
    /// Whether to use OAuth token instead of API key for Anthropic
    pub anthropic_use_oauth: bool,
    /// Model priority list for summarization, tried in order: the next model
    /// answers when one cannot. Each entry is routed like a chat model, with
    /// chat's credentials — `ollama/<model>` goes to `[memory].ollama_host`
    /// with its bound token.
    /// Format: `["ollama/llama3.2", "openrouter/<vendor>/<model>", "anthropic/claude-haiku-4-5"]`
    /// If empty, truncates instead of summarizing
    pub summarization_priority: Vec<String>,
    // NOTE: `ollama_url` was retired 2026-09-18 (owner decision: "summarization
    // should follow the summarization model selection in settings, with
    // fallbacks"). It was the summarizers' own Ollama address — localhost by
    // default, with no token — so summaries went to a different server than
    // chat whenever chat moved. Summaries now reach Ollama through chat's
    // router. Existing config.toml files still carrying it load unchanged:
    // nothing here uses `#[serde(deny_unknown_fields)]`, so serde ignores the
    // stale key and the next save drops it. Covered by
    // `legacy_llm_ollama_url_key_still_loads`.
    /// Ollama API key (optional — for remote/authenticated Ollama instances)
    pub ollama_api_key: Option<String>,
    /// Model routing priority for cost optimization.
    /// Format: `["model:tier", ...]` where tier is simple|medium|complex.
    /// Cheapest models first. Empty = disabled (always use primary model).
    /// Example: `["claude-haiku-4-5:simple", "claude-opus-5:complex"]`
    pub model_routing: Vec<String>,
    /// Whether to always use the primary model for the first iteration. Default: true.
    pub routing_first_turn_primary: bool,
    /// Legacy single sub-agent model. Superseded by `sub_agent_models`; kept
    /// so configs saved before the list existed keep working — see
    /// [`LlmConfig::effective_sub_agent_models`]. The GUI clears it when the
    /// list is saved.
    pub sub_agent_model: Option<String>,
    /// Model priority list for sub-agents spawned via the `task` tool: first
    /// working model wins, failures fall back to the next in the list.
    /// Empty = sub-agents use the main chat list (`model_priority`).
    /// Format: `["ollama/qwen3:4b", "claude-haiku-3-5"]`
    pub sub_agent_models: Vec<String>,
    /// Anthropic prompt-cache lifetime: `"5m"` (default) or `"1h"`. A 1-hour cache write
    /// costs 2x input instead of 1.25x and pays only when requests sharing a prompt start
    /// 5-60 minutes apart (heartbeats, cron, a reply after a break). Applies to every
    /// cache breakpoint of a request.
    pub prompt_cache_ttl: PromptCacheTtl,
}

/// `[llm] prompt_cache_ttl` — the two lifetimes Anthropic's prompt cache offers. Any other
/// value fails config parsing with an error naming the two accepted spellings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum PromptCacheTtl {
    #[default]
    #[serde(rename = "5m")]
    FiveMinutes,
    #[serde(rename = "1h")]
    OneHour,
}

impl LlmConfig {
    /// The model list sub-agents actually run with, in fallback order:
    /// `sub_agent_models` when set, else the legacy single `sub_agent_model`,
    /// else the main chat list, else the single primary `model`. Never empty.
    #[must_use]
    pub fn effective_sub_agent_models(&self) -> Vec<String> {
        if !self.sub_agent_models.is_empty() {
            return self.sub_agent_models.clone();
        }
        if let Some(ref legacy) = self.sub_agent_model
            && !legacy.is_empty() {
                return vec![legacy.clone()];
            }
        if !self.model_priority.is_empty() {
            return self.model_priority.clone();
        }
        vec![self.model.clone()]
    }
}

impl LlmConfig {
    /// The key of `[llm].provider`, from that provider's own field; `None`
    /// when it is unset or blank.
    ///
    /// `openai` and `openrouter` keep theirs in `openai_api_key` and
    /// `openrouter_api_key`. `api_key` is Anthropic's alone, and serves any
    /// other provider name too: the CLI's chat falls back to Anthropic for a
    /// name it does not know.
    #[must_use]
    pub fn provider_api_key(&self) -> Option<&str> {
        provider_key::NonAnthropicProvider::of(&self.provider)
            .map_or(self.api_key.as_deref(), |provider| provider.key(self))
            .filter(|key| !key.trim().is_empty())
    }

    /// The field [`Self::provider_api_key`] reads, to store an entered key in.
    pub fn provider_api_key_mut(&mut self) -> &mut Option<String> {
        match provider_key::NonAnthropicProvider::of(&self.provider) {
            Some(provider) => provider.slot_mut(self),
            None => &mut self.api_key,
        }
    }
}

impl LlmConfig {
    /// Whether **any** LLM API credential is configured, across every provider —
    /// not just Anthropic.
    ///
    /// Onboarding uses this to decide whether to prompt for a key. The old check
    /// looked only at `api_key` (the Anthropic slot), so a user who had entered an
    /// `OpenAI` / `OpenRouter` / GitHub-Models key, or was using Anthropic OAuth, was
    /// wrongly told they had no key. Ollama is intentionally excluded here: it is a
    /// keyless local backend, so "needs a key at all" is a separate question the
    /// onboarding tracks with `needsKey`.
    #[must_use]
    pub fn has_configured_api_key(&self) -> bool {
        let has = |k: &Option<String>| k.as_deref().is_some_and(|s| !s.trim().is_empty());
        has(&self.api_key)
            || has(&self.anthropic_oauth_token)
            || has(&self.openai_api_key)
            || has(&self.openrouter_api_key)
            || has(&self.github_token)
    }
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            provider: "anthropic".to_string(),
            // A DATED id is a pinned snapshot with a retirement date, and this
            // one was already retired: a daemon booted with no configured
            // model_priority sent every scheduled heartbeat to a 404 (observed
            // live 2026-08-27). Undated ids are the current-generation aliases.
            model: "claude-sonnet-5".to_string(),
            api_key: None,
            base_url: None,
            max_tokens: 8192,
            temperature: 0.7,
            openai_api_key: None,
            openrouter_api_key: None,
            github_token: None,
            model_priority: vec![], // Empty - dynamically populated from available providers
            anthropic_oauth_token: None,
            anthropic_use_oauth: false,
            summarization_priority: vec![], // Empty = truncate instead of summarize
            ollama_api_key: None,
            model_routing: vec![], // Empty = disabled (always use primary model)
            routing_first_turn_primary: true,
            sub_agent_model: None, // Legacy — see sub_agent_models
            sub_agent_models: vec![], // Empty = fall back to model_priority
            prompt_cache_ttl: PromptCacheTtl::FiveMinutes, // the API default
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AgentConfig {
    /// Agent name (displayed in responses)
    pub name: String,
    /// Custom system prompt (None = use default)
    pub system_prompt: Option<String>,
    /// Personality mode: balanced, professional, casual, minimal
    pub personality_mode: String,
    /// Who the agent is (formerly per-workspace SOUL.md / IDENTITY.md).
    /// Injected into every session independent of workspace.
    #[serde(default)]
    pub persona: Option<String>,
    /// Who the user is (formerly per-workspace USER.md).
    /// Injected into every session independent of workspace.
    #[serde(default)]
    pub user_profile: Option<String>,
    // NOTE: `thinking_enabled` was removed 2026-08-04 (owner directive:
    // "thinking should be on by default and remove the option in settings to
    // turn it off"). It was a second, disconnected knob beside
    // `nanna_agent::ThinkingMode`, which now defaults to a real budget.
    // Existing config.toml files still carrying `thinking_enabled = true`
    // load unchanged: nothing in this file uses
    // `#[serde(deny_unknown_fields)]`, so serde ignores the stale key.
    // Covered by `legacy_thinking_enabled_key_still_loads`.
    /// Enable streaming responses
    pub streaming_enabled: bool,
    /// Absolute cap on agent-loop iterations (tool-call rounds). `None` = unlimited
    /// (the default) — the agent is a long-horizon worker; only Stop/cancel ends it.
    /// A value here is a pure runaway backstop for unattended runs.
    #[serde(default)]
    pub max_iterations: Option<usize>,
    /// Iteration at which the first escalating "wrap-up" soft nudge is injected.
    /// The loop is NOT stopped — the nudge only steers a possibly-stuck model.
    /// Default: 500.
    #[serde(default = "default_nudge_after")]
    pub nudge_after_iterations: usize,
    /// After the first nudge, inject a further (more urgent) nudge every N
    /// iterations. Default: 100.
    #[serde(default = "default_nudge_interval")]
    pub nudge_interval_iterations: usize,
}

const fn default_nudge_after() -> usize {
    500
}

const fn default_nudge_interval() -> usize {
    100
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            name: "Nanna".to_string(),
            system_prompt: None,
            personality_mode: "balanced".to_string(),
            persona: None,
            user_profile: None,
            streaming_enabled: true,
            max_iterations: None,
            nudge_after_iterations: default_nudge_after(),
            nudge_interval_iterations: default_nudge_interval(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    pub enabled: bool,
    // NOTE: `host` was removed 2026-09-11. Nothing ever read it — `nanna
    // server` binds its `--host` flag (default loopback) — so it was a field
    // shaped exactly like a security control that controlled nothing: setting
    // it to 127.0.0.1 secured nothing, and its shipped `0.0.0.0` default read
    // as an exposure that never happened. Deleted rather than wired, because
    // wiring it under that default would have exposed an unauthenticated HTTP
    // surface. Existing config.toml files still carrying `host = …` load
    // unchanged (no `#[serde(deny_unknown_fields)]`, so serde ignores the stale
    // key). Covered by `legacy_server_host_key_still_loads`.
    pub port: u16,
    /// The shared secret `nanna server`'s generic webhook requires; without it
    /// the endpoint refuses to serve.
    ///
    /// Secret: in the secure store, not `config.toml` (`server_secret.rs`).
    /// From `NANNA_WEBHOOK_SECRET` when the file does not set it.
    pub webhook_secret: Option<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            port: 3000,
            webhook_secret: None,
        }
    }
}

/// The channels Nanna answers on.
///
/// A channel's section in `config.toml` turns it on and holds its settings; its
/// secrets (bot and app tokens, signing and webhook secrets) are held in the
/// secure store and never written to the file (`channel_secrets.rs`). Each is
/// empty until the load that fills it.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ChannelsConfig {
    pub telegram: Option<TelegramConfig>,
    pub discord: Option<DiscordConfig>,
    pub slack: Option<SlackConfig>,
    pub signal: Option<SignalConfig>,
    pub whatsapp: Option<WhatsAppConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelegramConfig {
    /// Secret: in the secure store, not `config.toml`. Empty when unset.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub bot_token: String,
    pub webhook_url: Option<String>,
    pub allowed_users: Option<Vec<i64>>,
    /// Secret token passed to Telegram's `setWebhook` and echoed back on every
    /// inbound POST as `X-Telegram-Bot-Api-Secret-Token`.
    ///
    /// This is the ONLY origin proof the Telegram webhook has: the route is a
    /// fixed path with no bot token in it, so without this value anyone who can
    /// reach the port can drive the agent. The endpoint refuses to serve until
    /// it is set.
    ///
    /// Secret: in the secure store, not `config.toml`.
    #[serde(default)]
    pub webhook_secret: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscordConfig {
    /// Secret: in the secure store, not `config.toml`. Empty when unset.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub bot_token: String,
    pub application_id: String,
    pub public_key: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlackConfig {
    /// Secret: in the secure store, not `config.toml`. Empty when unset.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub bot_token: String,
    /// Secret: in the secure store, not `config.toml`.
    pub app_token: Option<String>,
    /// Secret: in the secure store, not `config.toml`. Empty when unset.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub signing_secret: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalConfig {
    /// Shared secret the signal-cli-rest-api bridge must present on every
    /// inbound webhook, as `Authorization: Bearer <secret>` or
    /// `X-Webhook-Secret: <secret>`.
    ///
    /// signal-cli-rest-api does not sign its callbacks, so a shared secret is
    /// the strongest proof available on this path. Without it the endpoint
    /// cannot tell the bridge from any other caller and refuses to serve.
    ///
    /// Secret: in the secure store, not `config.toml`.
    #[serde(default)]
    pub webhook_secret: Option<String>,
    /// Phone number registered with Signal (e.g., "+1234567890")
    pub phone_number: String,
    /// URL of signal-cli-rest-api instance
    pub api_url: Option<String>,
    /// Allowed phone numbers (None = allow all)
    pub allowed_numbers: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhatsAppConfig {
    /// Connection method: "cloud-api" or "web"
    pub connection_method: String,
    /// Phone Number ID (for Cloud API)
    pub phone_number_id: Option<String>,
    /// Access token (for Cloud API). Secret: in the secure store, not
    /// `config.toml`.
    pub access_token: Option<String>,
    /// Webhook verify token (for Cloud API — the GET subscription handshake).
    /// Secret: in the secure store, not `config.toml`.
    pub verify_token: Option<String>,
    /// App secret (for Cloud API — HMAC key for the `X-Hub-Signature-256` on
    /// inbound POST payloads). Without it, POSTs to the webhook are unauthenticated.
    /// Secret: in the secure store, not `config.toml`.
    #[serde(default)]
    pub app_secret: Option<String>,
    /// Session name (for Web bridge)
    pub session_name: Option<String>,
    /// Allowed phone numbers (None = allow all)
    pub allowed_contacts: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ToolsConfig {
    pub enabled: Vec<String>,
    pub disabled: Vec<String>,
    pub exec_allowlist: Option<Vec<String>>,
    pub file_sandbox: Option<PathBuf>,
    /// Brave Search API key for `web_search` tool
    pub brave_api_key: Option<String>,
    /// Use TypeScript skill implementations instead of Rust builtins
    pub use_script_tools: bool,
    /// Directory containing tool scripts (default: {`data_dir}/tools`/)
    /// Can be overridden with `NANNA_TOOLS_DIR` environment variable
    pub tools_dir: Option<PathBuf>,
    /// Append one JSON line per tool call to `{data_dir}/logs/tool-audit.jsonl`.
    ///
    /// On by default: the daemon runs unattended, and "what did it do while I
    /// was asleep" has no other answer — the aggregate counters live only on the
    /// agent-loop path, and the per-call `debug!` lines are off at the default
    /// level. The trail is size-capped with one generation of rollover, so
    /// leaving it on cannot grow without bound.
    pub audit_log: bool,
    /// Include a bounded preview of tool *arguments* in the audit trail.
    ///
    /// Off by default, and deliberately separate from [`Self::audit_log`]:
    /// arguments carry secrets (an API key in a request, the body of a file
    /// being written), and the trail is durable plaintext that outlives the run
    /// that produced it. Key *names* are always recorded, which answers most
    /// audit questions without creating a secret sink.
    pub audit_log_values: bool,
}

impl Default for ToolsConfig {
    fn default() -> Self {
        Self {
            enabled: vec!["*".to_string()], // All tools enabled by default
            disabled: Vec::new(),
            exec_allowlist: None,
            file_sandbox: None,
            brave_api_key: None,
            use_script_tools: true,
            tools_dir: None,
            audit_log: true,
            audit_log_values: false,
        }
    }
}

/// Background scheduler settings.
///
/// Since P16 the daemon is the only mode, so the daemon's scheduler is *the*
/// scheduler and this section is the only thing that configures it. These are
/// the three knobs behind Settings → Scheduler in the GUI; the daemon applies
/// them to the running loop on config reload, so no restart is needed.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SchedulerConfig {
    /// Master switch. `false` leaves cron jobs loaded and editable but fires
    /// nothing: no heartbeat, no cron, no memory consolidation, no recurrence
    /// sweep.
    pub enabled: bool,

    /// Whether the periodic heartbeat prompt runs.
    ///
    /// A heartbeat is a full agent turn against the *same* model chat uses.
    /// With a single-slot local backend, a heartbeat firing mid-conversation
    /// time-shares the slot, the in-flight generation is cancelled, and the
    /// client sees a stream that ended without `done=true` — a fault the
    /// harness then spends its retry budget healing. Turning this off is a
    /// prerequisite for a clean local benchmark run.
    pub heartbeat_enabled: bool,

    /// Seconds between heartbeats. Clamped up to
    /// `nanna_core::MIN_HEARTBEAT_INTERVAL_SECS` when applied — below the
    /// scheduler's own 30s tick resolution the period cannot be honored.
    pub heartbeat_interval_secs: u64,
}

const fn default_heartbeat_interval_secs() -> u64 {
    1800
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            heartbeat_enabled: true,
            heartbeat_interval_secs: default_heartbeat_interval_secs(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MemoryConfig {
    pub enabled: bool,
    /// Embedding provider: "openai", "ollama", or "disabled"
    pub embedding_provider: String,
    /// Embedding model name
    pub embedding_model: String,
    pub vector_dimension: usize,
    pub storage_path: Option<PathBuf>,
    /// Ollama server URL (used for both chat and embeddings)
    pub ollama_host: String,
    /// Model to use for memory extraction (empty = use chat model)
    pub extraction_model: String,
    /// Embedding model priority list for fallback
    /// Format: `["openai/text-embedding-3-small", "ollama/nomic-embed-text"]`
    pub embedding_priority: Vec<String>,
    /// Maximum fraction of memories that can be removed in a single consolidation run (0.0-1.0).
    /// Default: 0.50 (50%)
    #[serde(default = "default_max_compression_ratio")]
    pub max_compression_ratio: f32,
    /// Minimum number of memories to retain after consolidation (hard floor).
    /// Default: 20
    #[serde(default = "default_min_remaining_memories")]
    pub min_remaining_memories: usize,

    /// Seconds the daemon must be idle (no chat activity) before the scheduled
    /// dream cycle is allowed to run. Dreaming competes with the live agent for
    /// the summarizer model and rewrites the store mid-conversation, so it waits
    /// for a genuine lull. Default: 300 (5 min).
    /// When true, every user/assistant turn (≥3 words) is written into
    /// long-term memory automatically. **Default true.**
    ///
    /// Remembering is what this system IS: a memory that only holds what
    /// someone thought to save is a notebook, not a memory. Dreaming is the
    /// counterweight — it consolidates and generalises so the store stays
    /// useful rather than merely large — and it has nothing to work with if
    /// conversation never lands. Defaulting this off left the store with 3
    /// entries after a four-hour run (observed 2026-07-27), one of which was
    /// stale enough to send a later run chasing a file that no longer existed.
    ///
    /// The switch REMAINS, and remains honest, because chat content is the
    /// user's own words rather than the agent's working record: tool calls and
    /// their results are captured unconditionally (that is the harness doing
    /// its job), but a person who does not want their conversation persisted
    /// locally can still say so. See PRIVACY.md.
    #[serde(default = "default_auto_remember_messages")]
    pub auto_remember_messages: bool,

    #[serde(default = "default_dream_idle_threshold_secs")]
    pub dream_idle_threshold_secs: u64,

    /// Live memory count at/above which the scheduled dream cycle runs
    /// **regardless** of idle time (memory-pressure relief, so a continuously
    /// busy daemon still consolidates before the store grows unbounded). `0`
    /// disables the pressure override. Default: 5000.
    #[serde(default = "default_dream_memory_pressure_count")]
    pub dream_memory_pressure_count: usize,

    // -----------------------------------------------------------------------
    // OCR settings
    // -----------------------------------------------------------------------

    /// OCR model priority list for document text extraction.
    ///
    /// Format: `["ollama/llava", "anthropic/claude-opus-4-6"]`
    ///
    /// Models are tried in order after embedded OCR (if enabled).
    /// Only vision-capable models should be listed here.
    #[serde(default)]
    pub ocr_model_priority: Vec<String>,

    /// Whether to use the embedded `ocrs` pure-Rust OCR engine before
    /// falling through to the model priority list.
    ///
    /// Default: `true`.  The embedded engine handles Latin-script images
    /// offline with no API cost; disable it to force model-based OCR.
    #[serde(default = "default_use_embedded_ocr")]
    pub use_embedded_ocr: bool,
}

const fn default_max_compression_ratio() -> f32 { 0.50 }
const fn default_min_remaining_memories() -> usize { 20 }
const fn default_dream_idle_threshold_secs() -> u64 { 300 }

/// Remembering conversation is the default: see the field docs.
const fn default_auto_remember_messages() -> bool { true }
const fn default_dream_memory_pressure_count() -> usize { 5000 }
const fn default_use_embedded_ocr() -> bool { true }

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            embedding_provider: "openai".to_string(),
            embedding_model: "text-embedding-3-small".to_string(),
            vector_dimension: 1536,
            storage_path: None,
            ollama_host: "http://localhost:11434".to_string(),
            extraction_model: String::new(), // Empty = use chat model
            // EMPTY means "not configured", which falls back to the
            // `embedding_provider`/`embedding_model` pair.
            //
            // It must stay empty. `#[serde(default)]` is on the CONTAINER, so
            // every config.toml that never wrote an `embedding_priority` key
            // gets this value — and since the daemon treats a non-empty list as
            // authoritative, a default entry here silently overrides the
            // provider the user actually chose. With
            // `openai/text-embedding-3-small` as the default and no
            // OPENAI_API_KEY set, a user who picked Ollama in Settings resolved
            // ZERO providers and the memory subsystem switched off entirely.
            // The Settings dropdown writes provider/model and never touches
            // this list, so that state was one click away.
            embedding_priority: Vec::new(),
            max_compression_ratio: default_max_compression_ratio(),
            min_remaining_memories: default_min_remaining_memories(),
            auto_remember_messages: default_auto_remember_messages(),
            dream_idle_threshold_secs: default_dream_idle_threshold_secs(),
            dream_memory_pressure_count: default_dream_memory_pressure_count(),
            ocr_model_priority: vec![],
            use_embedded_ocr: default_use_embedded_ocr(),
        }
    }
}

/// Whose a stored Ollama token with no recorded server is — one saved by a
/// build that recorded none.
#[derive(Debug, Clone, Copy)]
enum UnboundOllamaToken<'a> {
    /// The configured server's: a process's first load, where it goes where
    /// the older build sent it.
    Configured,
    /// The server a running configuration was sending it to, which the one
    /// being loaded replaces. Without this a reload after the address was
    /// edited would read the token as the edited address's.
    RunningServer(&'a str),
}

/// The process environment, as the hydration seams take it.
fn process_env(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

/// What to tell someone whose `config.toml` still carries the retired
/// `[llm].ollama_url` pointed at another server than `config` uses; `None`
/// when it is absent or names the same server.
///
/// Read from the raw text because serde drops the key on the way in, and the
/// next save drops it from disk — so this load is the last moment anything
/// can see that summaries used to go somewhere else. The shipped default
/// (this machine, like `[memory].ollama_host`'s) moves nothing and says
/// nothing.
fn retired_ollama_url_notice(content: &str, config: &Config) -> Option<String> {
    let raw: toml::Table = content.parse().ok()?;
    let retired = raw.get("llm")?.get("ollama_url")?.as_str()?.trim();
    let host = config.memory.ollama_host.trim();
    if retired.is_empty() || same_ollama_server(retired, host) {
        return None;
    }
    // Redacted as every other Ollama address in the log is: a URL can carry a
    // password in its user info or a token in its query.
    let (retired, host) = (
        ollama::redacted_ollama_host(retired),
        ollama::redacted_ollama_host(host),
    );
    Some(format!(
        "config.toml still sets [llm].ollama_url = \"{retired}\", which is no longer read: \
         summaries now reach Ollama through chat's server, [memory].ollama_host = \"{host}\". \
         If your summarization models are on {retired}, set it as the Ollama server in \
         Settings -> Models (it is then chat's and the embedders' server too); otherwise \
         delete the key."
    ))
}

/// File in `store` under `key` a secret `config.toml` itself holds — `held`,
/// trimmed and not blank — as the file loads, so that the save which next
/// strips it from the file loses nothing ([`channel_secrets::adopt`],
/// [`server_secret::adopt`], [`api_keys::adopt`]). A different stored value is
/// replaced: every load of the file runs with the file's, so the one kept must
/// be the file's, or that save would switch to another. Each load says the
/// line can be deleted.
///
/// `field` names the secret in `config.toml` and `env_var` is where else it can
/// come from, for the messages (never the value).
fn adopt_file_secret(field: &str, env_var: &str, key: &str, held: &str, store: &SecureStore) {
    let stored = match store.get(key) {
        Ok(stored) if stored == held => StoredCopy::Same,
        Ok(_) => StoredCopy::Other,
        Err(_) => StoredCopy::None,
    };
    adopt_file_secret_with(field, env_var, stored, || store.set(key, held));
}

/// What the secure store holds of a secret `config.toml` holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StoredCopy {
    /// Nothing.
    None,
    /// Something else, or the same filed otherwise than the file's would be.
    Other,
    /// The file's, filed as it would be.
    Same,
}

/// [`adopt_file_secret`] for a secret filed by `file`, not under one store
/// key: unless `stored` is already the file's, `file` files it, and each load
/// says the line can be deleted.
fn adopt_file_secret_with(
    field: &str,
    env_var: &str,
    stored: StoredCopy,
    file: impl FnOnce() -> Result<(), crate::credentials::CredentialError>,
) {
    if stored == StoredCopy::Same {
        warn!(
            "config.toml holds {field} in plain text. It is in the secure store, so the line \
             can be deleted; the next save of the settings removes it."
        );
        return;
    }
    match file() {
        Ok(()) => warn!(
            "config.toml holds {field} in plain text. It is now filed in the secure store{}, so \
             the line can be deleted; the next save of the settings removes it.",
            if stored == StoredCopy::Other {
                " in place of the one there"
            } else {
                ""
            }
        ),
        Err(e) => tracing::error!(
            "config.toml holds {field} in plain text and it cannot be filed in the secure store \
             ({e}). This process runs with it; once the settings are next saved it is no longer \
             in config.toml, and must be set again or supplied as {env_var}."
        ),
    }
}

impl Config {
    /// Load config from default location.
    ///
    /// # Errors
    ///
    /// Returns `ConfigError` if the config path cannot be determined or the file cannot be read.
    pub fn load() -> Result<Self, ConfigError> {
        let path = Self::default_config_path()?;
        if path.exists() {
            Self::load_from(&path)
        } else {
            let mut cfg = Self::default();
            cfg.load_secrets_from_store();
            Ok(cfg)
        }
    }

    /// Load config from a specific path.
    ///
    /// # Errors
    ///
    /// Returns `ConfigError::Io` if the file cannot be read.
    /// Returns `ConfigError::Parse` if the TOML is invalid.
    pub fn load_from(path: &PathBuf) -> Result<Self, ConfigError> {
        Self::load_from_with(path, &crate::credentials::SecureStore::new(), process_env)
    }

    /// [`Self::load_from`] against a given store and environment.
    fn load_from_with(
        path: impl AsRef<Path>,
        store: &crate::credentials::SecureStore,
        env: impl Fn(&str) -> Option<String>,
    ) -> Result<Self, ConfigError> {
        let (mut config, content) = Self::parse_file(path.as_ref(), store)?;
        if let Some(notice) = retired_ollama_url_notice(&content, &config) {
            warn!("{notice}");
        }
        config.load_secrets_from(store, env);
        Ok(config)
    }

    /// Read and parse the config file at `path`, with a non-Anthropic
    /// `[llm].provider`'s key filed under its own name
    /// ([`provider_key::refile_provider_key`]) and each channel secret, the
    /// webhook secret and each `[llm]` and `[tools]` key the file itself holds
    /// filed in `store` ([`channel_secrets::adopt`], [`server_secret::adopt`],
    /// [`api_keys::adopt`]) — before any secret is hydrated, so only a key the
    /// old layout left behind is moved and only the file's own secrets are
    /// filed. The file's text comes back too, for what only the raw text can
    /// show.
    fn parse_file(
        path: &Path,
        store: &crate::credentials::SecureStore,
    ) -> Result<(Self, String), ConfigError> {
        let content = std::fs::read_to_string(path)?;
        let mut config: Self = toml::from_str(&content)?;
        info!("Loaded config from {path:?}");
        provider_key::refile_provider_key(&mut config.llm, store);
        channel_secrets::adopt(&mut config.channels, store);
        server_secret::adopt(&mut config.server, store);
        api_keys::adopt(&mut config, store);
        Ok((config, content))
    }

    /// Save config to default location.
    ///
    /// # Errors
    ///
    /// Returns `ConfigError` if the config path cannot be determined or the file cannot be written.
    pub fn save(&self) -> Result<(), ConfigError> {
        let path = Self::default_config_path()?;
        self.save_to(&path)
    }

    /// Save config to a specific path.
    ///
    /// # Errors
    ///
    /// Returns `ConfigError::Io` if the directory cannot be created or the file cannot be written.
    /// Returns `ConfigError::Parse` if the config cannot be serialized.
    pub fn save_to(&self, path: &Path) -> Result<(), ConfigError> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        // Never write secrets into config.toml. The secure store is the only
        // durable home for API keys; the in-memory Config may still hold them
        // for the running process (loaded from keyring/env at boot).
        let mut disk = self.clone();
        disk.strip_secrets_for_disk();
        let contents = toml::to_string_pretty(&disk)?;
        // Temp file + rename, never an in-place write: the daemon watches this
        // file and applies what it reads, so a truncate-then-write could be
        // read half-written — a config missing everything past the cut.
        let tmp = path.with_extension("toml.tmp");
        fs::write(&tmp, contents)?;
        fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Blank every secret field so a serialized config never contains
    /// credentials. Called by [`Self::save_to`].
    pub fn strip_secrets_for_disk(&mut self) {
        self.llm.api_key = None;
        self.llm.openai_api_key = None;
        self.llm.openrouter_api_key = None;
        self.llm.github_token = None;
        self.llm.anthropic_oauth_token = None;
        self.llm.ollama_api_key = None;
        self.tools.brave_api_key = None;
        self.server.webhook_secret = None;
        channel_secrets::strip(&mut self.channels);
    }

    /// Persist any secret fields currently held in-memory into the OS keyring
    /// (or encrypted file fallback), then blank them on this Config. A caller
    /// that goes on using this Config wants [`Self::store_secrets`], which
    /// refills it. (`save()` never writes secrets to disk either way.)
    ///
    /// # Errors
    ///
    /// Returns the first [`SecureStore::set`](crate::credentials::SecureStore::set)
    /// failure: the keyring entry cannot be opened, or the keyring write fails and
    /// the encrypted file fallback fails too. Secrets are processed in a fixed
    /// order and each field is blanked before its write, so on error the failing
    /// secret and those already stored are gone from this `Config`; later ones
    /// are left in place.
    pub fn migrate_secrets_to_keyring(&mut self) -> Result<(), crate::credentials::CredentialError> {
        self.migrate_secrets_to(&crate::credentials::SecureStore::new())
    }

    /// [`Self::migrate_secrets_to_keyring`] into a given store.
    fn migrate_secrets_to(
        &mut self,
        store: &crate::credentials::SecureStore,
    ) -> Result<(), crate::credentials::CredentialError> {
        use crate::credentials::keys;
        let put = |key: &str, val: &mut Option<String>| -> Result<(), crate::credentials::CredentialError> {
            if let Some(v) = val.take() {
                let trimmed = v.trim();
                if !trimmed.is_empty() {
                    store.set(key, trimmed)?;
                }
            }
            Ok(())
        };
        put(keys::ANTHROPIC_API_KEY, &mut self.llm.api_key)?;
        put(keys::OPENAI_API_KEY, &mut self.llm.openai_api_key)?;
        put(keys::OPENROUTER_API_KEY, &mut self.llm.openrouter_api_key)?;
        put(keys::GITHUB_TOKEN, &mut self.llm.github_token)?;
        put(keys::BRAVE_API_KEY, &mut self.tools.brave_api_key)?;
        put(keys::SERVER_WEBHOOK_SECRET, &mut self.server.webhook_secret)?;
        // OAuth token + ollama key share the store under theirs-named keys too.
        if let Some(v) = self.llm.anthropic_oauth_token.take() {
            let trimmed = v.trim();
            if !trimmed.is_empty() {
                store.set(keys::ANTHROPIC_OAUTH_TOKEN, trimmed)?;
            }
        }
        // The Ollama token held here is the configured server's (see
        // `hydrate_ollama_token`), and is stored bound to it: written bare, it
        // would be filed under whichever server the store last recorded. The
        // stored token itself (loaded from there) is left as it is filed —
        // re-saving it would re-file it under this copy's address, which in a
        // long-lived process need not be the server it was loaded for.
        if let Some(v) = self.llm.ollama_api_key.take() {
            let trimmed = v.trim();
            let already_stored = store.ollama_token().as_deref() == Some(trimmed);
            if !trimmed.is_empty() && !already_stored {
                store.save_ollama_token(trimmed, &self.memory.ollama_host)?;
            }
        }
        channel_secrets::file(&mut self.channels, store)
    }

    /// File every secret held in memory in the OS keyring (or encrypted file
    /// fallback) and go on holding it: [`Self::migrate_secrets_to_keyring`],
    /// then [`Self::load_secrets_from_store`]. This `Config` is left with its
    /// secrets as the next [`Self::load`] gets them back, so a caller that goes
    /// on using it (a chat started right after the key was entered) has the
    /// key; [`Self::save`] still writes none of them to disk.
    ///
    /// # Errors
    ///
    /// As [`Self::migrate_secrets_to_keyring`]; this `Config` is then not
    /// refilled.
    pub fn store_secrets(&mut self) -> Result<(), crate::credentials::CredentialError> {
        self.store_secrets_in(&crate::credentials::SecureStore::new(), process_env)
    }

    /// [`Self::store_secrets`] into a given store, refilling from it and `env`
    /// (read as a load reads the process environment).
    ///
    /// # Errors
    ///
    /// As [`Self::store_secrets`].
    pub fn store_secrets_in(
        &mut self,
        store: &crate::credentials::SecureStore,
        env: impl Fn(&str) -> Option<String>,
    ) -> Result<(), crate::credentials::CredentialError> {
        self.migrate_secrets_to(store)?;
        self.load_secrets_from(store, env);
        Ok(())
    }

    /// Hydrate secret fields from `SecureStore` + environment if they are unset.
    /// Safe to call repeatedly; never overwrites a value already present.
    pub fn load_secrets_from_store(&mut self) {
        self.load_secrets_from(&crate::credentials::SecureStore::new(), process_env);
    }

    /// [`Self::load_secrets_from_store`] against a given store and environment,
    /// so the hydration rules are testable without the OS keyring or the
    /// process environment.
    fn load_secrets_from(
        &mut self,
        store: &crate::credentials::SecureStore,
        env: impl Fn(&str) -> Option<String>,
    ) {
        self.load_secrets_with(store, &env, UnboundOllamaToken::Configured);
    }

    /// Fill every unset secret from `env`, then `store`; `unbound` says whose
    /// a stored Ollama token with no recorded server is.
    fn load_secrets_with(
        &mut self,
        store: &crate::credentials::SecureStore,
        env: &impl Fn(&str) -> Option<String>,
        unbound: UnboundOllamaToken<'_>,
    ) {
        self.fill_secrets(Some(store), env, unbound);
    }

    /// [`Self::load_secrets_with`], where no `store` is a store holding
    /// nothing: every unset secret is filled from `env` alone, and nothing
    /// is read or written anywhere else.
    fn fill_secrets(
        &mut self,
        store: Option<&crate::credentials::SecureStore>,
        env: &impl Fn(&str) -> Option<String>,
        unbound: UnboundOllamaToken<'_>,
    ) {
        use crate::credentials::keys;
        let fill = |slot: &mut Option<String>, key: &str, env_name: &str| {
            if slot.as_ref().is_some_and(|s| !s.trim().is_empty()) {
                return;
            }
            if let Some(v) = env(env_name)
                && !v.trim().is_empty() {
                    *slot = Some(v);
                    return;
                }
            if let Some(store) = store
                && let Ok(v) = store.get(key)
                && !v.trim().is_empty()
            {
                *slot = Some(v);
            }
        };
        fill(&mut self.llm.api_key, keys::ANTHROPIC_API_KEY, "ANTHROPIC_API_KEY");
        fill(&mut self.llm.openai_api_key, keys::OPENAI_API_KEY, "OPENAI_API_KEY");
        fill(&mut self.llm.openrouter_api_key, keys::OPENROUTER_API_KEY, "OPENROUTER_API_KEY");
        fill(&mut self.llm.github_token, keys::GITHUB_TOKEN, "GITHUB_TOKEN");
        fill(&mut self.tools.brave_api_key, keys::BRAVE_API_KEY, "BRAVE_API_KEY");
        fill(
            &mut self.server.webhook_secret,
            keys::SERVER_WEBHOOK_SECRET,
            server_secret::WEBHOOK_SECRET_ENV,
        );
        fill(&mut self.llm.anthropic_oauth_token, keys::ANTHROPIC_OAUTH_TOKEN, "ANTHROPIC_OAUTH_TOKEN");
        self.hydrate_ollama_token(store, env, unbound);
        channel_secrets::fill(&mut self.channels, store, env);
    }

    /// Fill `llm.ollama_api_key` for the configured Ollama server.
    ///
    /// Unlike the other secrets, the stored token is bound to a server — the
    /// one it was saved for ([`keys::OLLAMA_API_KEY_HOST`](crate::credentials::keys::OLLAMA_API_KEY_HOST)).
    /// Loaded for any other `[memory].ollama_host` it would be handed to
    /// whatever address the config names next: a new Server URL, or a hand
    /// edit of `config.toml` (which the daemon applies live, and which the
    /// agent can make). A token saved with no server recorded (an older
    /// build) is `unbound`'s. A record that cannot be read binds the token to
    /// no server: it is not sent until the store answers.
    ///
    /// Not bound: `OLLAMA_API_KEY` from the environment, and a token already
    /// present (written into `config.toml` itself). Both are the operator
    /// naming a token for this configuration outright.
    fn hydrate_ollama_token(
        &mut self,
        store: Option<&crate::credentials::SecureStore>,
        env: &impl Fn(&str) -> Option<String>,
        unbound: UnboundOllamaToken<'_>,
    ) {
        if self.llm.ollama_api_key.as_ref().is_some_and(|s| !s.trim().is_empty()) {
            return;
        }
        if let Some(token) = env("OLLAMA_API_KEY")
            && !token.trim().is_empty()
        {
            self.llm.ollama_api_key = Some(token);
            return;
        }
        let Some(store) = store else {
            return;
        };
        let Some(token) = store.ollama_token() else {
            return;
        };
        let bound = match (store.ollama_token_host(), unbound) {
            (Ok(Some(bound)), _) => bound,
            (Ok(None), UnboundOllamaToken::Configured) => {
                self.llm.ollama_api_key = Some(token);
                return;
            }
            (Ok(None), UnboundOllamaToken::RunningServer(running)) if running.trim().is_empty() => {
                // No server is running to attribute it to. A blank record is
                // still a record — it names no server and matches none — so
                // writing one here bound the token to nothing, forever: it was
                // never sent again, even after a real Server URL was set. Left
                // unbound, it goes to the first real server instead.
                return;
            }
            (Ok(None), UnboundOllamaToken::RunningServer(running)) => {
                // Recorded, so that later loads (a restart, the GUI) agree it
                // is that server's. Attributed here whether or not the write
                // lands: a store that refuses it must not free the token.
                if let Err(e) = store.bind_unbound_ollama_token(running) {
                    tracing::warn!(
                        "Could not record {} as the server the saved Ollama token belongs to ({e})",
                        ollama::redacted_ollama_host(running)
                    );
                }
                normalize_ollama_host(running)
            }
            (Err(e), _) => {
                tracing::warn!(
                    "Cannot read which server the saved Ollama token belongs to ({e}); \
                     it is not sent until that can be read"
                );
                return;
            }
        };
        if !same_ollama_server(&bound, &self.memory.ollama_host) {
            ollama::report_withheld_token(&bound, &self.memory.ollama_host);
            return;
        }
        self.llm.ollama_api_key = Some(token);
    }

    /// Re-derive `llm.ollama_api_key` when `[memory].ollama_host` was changed
    /// in memory (not through a load) away from `previous_host`: the token
    /// held was loaded for that server, and is dropped unless it is also this
    /// one's. A stored token with no recorded server was `previous_host`'s,
    /// and is recorded as such. The same server keeps what it holds, and the
    /// store is not read.
    pub fn rebind_ollama_token_if_moved(
        &mut self,
        previous_host: &str,
        store: &crate::credentials::SecureStore,
    ) {
        self.rebind_ollama_token_if_moved_with(previous_host, store, process_env);
    }

    /// [`Self::rebind_ollama_token_if_moved`] against a given store and
    /// environment: the one a process loads with, when it is not the
    /// process environment (a test's daemon control plane).
    pub fn rebind_ollama_token_if_moved_with(
        &mut self,
        previous_host: &str,
        store: &crate::credentials::SecureStore,
        env: impl Fn(&str) -> Option<String>,
    ) {
        if !ollama_server_changed(previous_host, &self.memory.ollama_host) {
            return;
        }
        self.llm.ollama_api_key = None;
        self.hydrate_ollama_token(
            Some(store),
            &env,
            UnboundOllamaToken::RunningServer(previous_host),
        );
    }

    /// Load config from the default location to replace a configuration that
    /// is running with Ollama server `running_ollama_host`: a reload.
    ///
    /// [`Self::load`] reads a stored Ollama token with no recorded server (an
    /// older build's) as the configured server's — right for a process's
    /// first load, wrong for a reload after the address was edited, which
    /// would hand the running server's token to the edited address. Here that
    /// token stays the running server's and is recorded as such.
    ///
    /// # Errors
    ///
    /// As [`Self::load`].
    pub fn load_replacing(
        running_ollama_host: &str,
        store: &crate::credentials::SecureStore,
    ) -> Result<Self, ConfigError> {
        Self::load_replacing_with(running_ollama_host, store, process_env)
    }

    /// [`Self::load_replacing`] against a given environment.
    ///
    /// # Errors
    ///
    /// As [`Self::load`].
    pub fn load_replacing_with(
        running_ollama_host: &str,
        store: &crate::credentials::SecureStore,
        env: impl Fn(&str) -> Option<String>,
    ) -> Result<Self, ConfigError> {
        let path = Self::default_config_path()?;
        if path.exists() {
            Self::load_from_replacing_with(&path, running_ollama_host, store, env)
        } else {
            let mut cfg = Self::default();
            cfg.load_secrets_with(
                store,
                &env,
                UnboundOllamaToken::RunningServer(running_ollama_host),
            );
            Ok(cfg)
        }
    }

    /// [`Self::load_replacing`] from a specific path.
    ///
    /// # Errors
    ///
    /// As [`Self::load_from`].
    pub fn load_from_replacing(
        path: &Path,
        running_ollama_host: &str,
        store: &crate::credentials::SecureStore,
    ) -> Result<Self, ConfigError> {
        Self::load_from_replacing_with(path, running_ollama_host, store, process_env)
    }

    /// [`Self::load_from_replacing`] against a given environment.
    ///
    /// # Errors
    ///
    /// As [`Self::load_from`].
    pub fn load_from_replacing_with(
        path: &Path,
        running_ollama_host: &str,
        store: &crate::credentials::SecureStore,
        env: impl Fn(&str) -> Option<String>,
    ) -> Result<Self, ConfigError> {
        let (mut config, _) = Self::parse_file(path, store)?;
        config.load_secrets_with(
            store,
            &env,
            UnboundOllamaToken::RunningServer(running_ollama_host),
        );
        Ok(config)
    }

    /// Environment variable that redirects config resolution to an explicit file.
    ///
    /// Exists because there was previously **no way to run Nanna against a
    /// config it does not own**. `--data-dir` isolates the database but not the
    /// settings, and the settings path is resolved through `directories`, which
    /// on Windows reads the known-folder API — so `%APPDATA%` cannot redirect it
    /// either. Any boot-time behaviour that depends on configuration (which
    /// providers resolve, whether embeddings are enabled at all) was therefore
    /// only reachable by editing the operator's real `config.toml`, which an
    /// unattended run must never do. One variable, honoured in the single place
    /// every consumer already funnels through, makes those paths testable
    /// without touching anything the operator owns.
    pub const CONFIG_PATH_ENV: &'static str = "NANNA_CONFIG_PATH";

    /// Get default config path.
    ///
    /// [`Self::CONFIG_PATH_ENV`] wins when it is set to a non-empty value. The
    /// override is deliberately taken BEFORE the legacy migration: a caller
    /// naming an explicit file is not asking for the `bot/clawd/Nanna` tree to
    /// be copied into the canonical one as a side effect, and a test harness
    /// least of all.
    ///
    /// A path is returned whether or not it exists — the same contract the
    /// unset case has always had, since [`Self::load`] treats a missing file as
    /// "use defaults" rather than as an error. Whitespace-only is treated as
    /// unset, so an empty variable in a shell profile cannot silently redirect
    /// every consumer to `""`.
    ///
    /// # Errors
    ///
    /// Returns `ConfigError::NoDirFound` if the system config directory cannot be determined.
    pub fn default_config_path() -> Result<PathBuf, ConfigError> {
        if let Some(path) = Self::config_path_override() {
            info!("Config path overridden by {}: {path:?}", Self::CONFIG_PATH_ENV);
            return Ok(path);
        }
        Self::migrate_legacy_config_if_needed();
        let dirs = project_dirs().ok_or(ConfigError::NoDirFound)?;
        Ok(dirs.config_dir().join("config.toml"))
    }

    /// The explicit config path, if one is set and meaningful.
    ///
    /// Pure apart from the environment read, so the trimming policy is testable
    /// on its own: see [`config_path_override_ignores_blank`].
    fn config_path_override() -> Option<PathBuf> {
        let raw = std::env::var(Self::CONFIG_PATH_ENV).ok()?;
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return None;
        }
        Some(PathBuf::from(trimmed))
    }

    /// Get default data directory.
    ///
    /// # Errors
    ///
    /// Returns `ConfigError::NoDirFound` if the system data directory cannot be determined.
    pub fn default_data_dir() -> Result<PathBuf, ConfigError> {
        Self::migrate_legacy_config_if_needed();
        let dirs = project_dirs().ok_or(ConfigError::NoDirFound)?;
        Ok(dirs.data_dir().to_path_buf())
    }

    /// The data directory this configuration actually selects.
    ///
    /// `[general] data_dir` when the user set one, otherwise the platform
    /// default. A blank or whitespace-only value counts as unset — a text input
    /// that submits `""` must not redirect the whole store to a path of `""`,
    /// which is the same policy `Self::config_path_override` applies.
    ///
    /// This is the single place the override is honoured, so every consumer
    /// that already funnels through the daemon's `data_dir` picks it up at
    /// once. The daemon's `--data-dir` flag is applied *after* this and still
    /// wins.
    ///
    /// # Errors
    ///
    /// Returns `ConfigError::NoDirFound` only in the fallback case, when no
    /// override is set and the system data directory cannot be determined.
    pub fn resolve_data_dir(&self) -> Result<PathBuf, ConfigError> {
        if let Some(dir) = Self::meaningful_data_dir(self.general.data_dir.as_deref()) {
            return Ok(dir);
        }
        Self::default_data_dir()
    }

    /// The configured override, if it is one. Pure, so the blank-is-unset
    /// policy is testable without touching the filesystem or the environment.
    #[must_use]
    fn meaningful_data_dir(configured: Option<&Path>) -> Option<PathBuf> {
        let path = configured?;
        if path.as_os_str().is_empty() {
            return None;
        }
        // A path that is only whitespace is a half-filled form, not a location.
        if path.to_str().is_some_and(|s| s.trim().is_empty()) {
            return None;
        }
        Some(path.to_path_buf())
    }

    /// Whether `[general] data_dir` selects somewhere other than the platform
    /// default — i.e. whether this install has been deliberately relocated.
    #[must_use]
    pub fn has_custom_data_dir(&self) -> bool {
        Self::meaningful_data_dir(self.general.data_dir.as_deref()).is_some()
    }

    /// The data directory this install selects, read straight off disk
    /// **without touching the OS keyring**.
    ///
    /// Exists for one caller shape: code that must know where data lives
    /// *before* it is safe to do anything expensive or blocking — the daemon
    /// resolves its log directory from this at startup, before the tracing
    /// subscriber is installed.
    ///
    /// [`Self::load`] is deliberately not used here. It calls
    /// [`Self::load_secrets_from_store`], which reads the OS keyring, and a
    /// keyring read can block on a desktop unlock prompt. A daemon parked on
    /// that prompt during logging setup would emit **nothing at all** to say
    /// why, because the subscriber does not exist yet — a silent hang, which
    /// is the worst possible failure for a background service.
    ///
    /// Best-effort by design: a missing, unreadable or malformed config file
    /// falls through to the platform default rather than failing. The real
    /// parse error still surfaces moments later, with logging up, when the
    /// daemon loads the config properly.
    ///
    /// # Errors
    ///
    /// Returns `ConfigError::NoDirFound` only in the fallback case, when no
    /// override is readable and the system data directory cannot be determined.
    pub fn data_dir_from_disk() -> Result<PathBuf, ConfigError> {
        if let Ok(path) = Self::default_config_path()
            && let Ok(content) = fs::read_to_string(&path)
            && let Ok(parsed) = toml::from_str::<Self>(&content)
            && let Some(dir) = Self::meaningful_data_dir(parsed.general.data_dir.as_deref())
        {
            return Ok(dir);
        }
        Self::default_data_dir()
    }
}

/// Check that `path` can serve as a data directory, creating it if it does not
/// exist yet, and report precisely why not when it cannot.
///
/// Writability is settled by **writing a file**, not by reading permission
/// bits: `metadata().permissions().readonly()` says nothing useful about a
/// directory on Windows and ignores ACLs, mount flags and a full disk. The only
/// honest answer to "can Nanna write here" is to try, which is what the daemon
/// will do seconds later anyway — better to fail in a dialog the user is
/// looking at than at the next boot.
///
/// Creating the directory is deliberate: picking a not-yet-existing folder in a
/// file dialog is an ordinary thing to do, and refusing it would send the user
/// out to a file manager to do it by hand.
///
/// # Errors
///
/// Returns the [`DataDirError`] naming the specific problem: empty, relative,
/// an existing file, uncreatable, or unwritable.
pub fn validate_data_dir(path: &Path) -> Result<(), DataDirError> {
    let display = path.display().to_string();

    if path.as_os_str().is_empty() || path.to_str().is_some_and(|s| s.trim().is_empty()) {
        return Err(DataDirError::Empty);
    }

    // A relative path resolves against the process working directory, and the
    // GUI-spawned sidecar does not share one with a daemon started from a
    // terminal — so the same setting would silently mean two different stores.
    if !path.is_absolute() {
        return Err(DataDirError::NotAbsolute(display));
    }

    if path.exists() && !path.is_dir() {
        return Err(DataDirError::NotADirectory(display));
    }

    fs::create_dir_all(path).map_err(|e| DataDirError::CannotCreate {
        path: display.clone(),
        reason: e.to_string(),
    })?;

    // Probe with a uniquely-named file so two concurrent probes cannot collide,
    // and remove it whether or not the write succeeded.
    let probe = path.join(format!(
        ".nanna-write-probe-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos())
    ));
    let write_result = fs::write(&probe, b"nanna");
    let _ = fs::remove_file(&probe);
    write_result.map_err(|e| DataDirError::NotWritable {
        path: display,
        reason: e.to_string(),
    })?;

    Ok(())
}

impl Config {
    /// Copy config.toml from the legacy `bot/clawd/Nanna` tree into the
    /// canonical `com/nanna/nanna` tree when the latter does not yet exist.
    /// Best-effort and silent on failure — a failed migrate leaves the user on
    /// defaults rather than refusing to start.
    fn migrate_legacy_config_if_needed() {
        let Some(new_dirs) = project_dirs() else { return };
        let new_cfg = new_dirs.config_dir().join("config.toml");
        if new_cfg.exists() {
            return;
        }
        let Some(old_dirs) = legacy_clawd_project_dirs() else { return };
        let old_cfg = old_dirs.config_dir().join("config.toml");
        if !old_cfg.exists() {
            return;
        }
        if let Some(parent) = new_cfg.parent() {
            let _ = fs::create_dir_all(parent);
        }
        match fs::copy(&old_cfg, &new_cfg) {
            Ok(_) => tracing::info!(
                from = %old_cfg.display(),
                to = %new_cfg.display(),
                "Migrated config.toml from legacy clawd path to com.nanna.nanna"
            ),
            Err(e) => tracing::warn!(
                error = %e,
                "Failed to migrate legacy config.toml; continuing with defaults"
            ),
        }
        // Best-effort: also migrate the data dir contents the first time.
        let old_data = old_dirs.data_dir();
        let new_data = new_dirs.data_dir();
        if old_data.exists() && !new_data.exists() {
            if let Err(e) = copy_dir_recursive(old_data, new_data) {
                tracing::warn!(error = %e, "Failed to migrate legacy data dir");
            } else {
                tracing::info!(
                    from = %old_data.display(),
                    to = %new_data.display(),
                    "Migrated data directory from legacy clawd path"
                );
            }
        }
    }

    /// Override config with environment variables.
    ///
    /// **Secret precedence, end to end.** At load, each secret is filled only
    /// if `config.toml` left it blank, from the environment first and then the
    /// secure store (`fill_secrets`). Every entry point (the daemon, the
    /// control plane, the CLI) then calls this, where a non-blank environment
    /// variable replaces whatever was loaded. So the effective order is
    /// **environment > `config.toml` > secure store**, and a blank variable
    /// counts as unset everywhere. The one exception is the Ollama token from
    /// the store, which is only ever used for the server it was saved for.
    #[must_use]
    pub fn with_env_overrides(self) -> Self {
        self.with_env_overrides_from(process_env)
    }

    /// [`Self::with_env_overrides`] against a given environment, so the
    /// override rules are testable without the process environment.
    #[must_use]
    pub fn with_env_overrides_from(mut self, env: impl Fn(&str) -> Option<String>) -> Self {
        self.override_llm_keys(&env);

        // Server config
        if let Some(port) = env("PORT")
            && let Ok(p) = port.parse()
        {
            self.server.port = p;
        }

        self.override_channels(&env);
        self
    }

    /// The channel settings [`Self::with_env_overrides`] takes from `env`.
    ///
    /// A channel's bot token opens its overrides, and each variable replaces
    /// only the field it names: a configured section keeps the rest. Only a
    /// missing section is built from the environment alone. Until 2026-09-21
    /// the token replaced the whole section, so exporting it dropped
    /// Telegram's `allowed_users` (anyone could then drive the bot) and its
    /// webhook secret (the webhook then refused every request).
    ///
    /// A blank variable names nothing: an env-file line left empty must not
    /// blank a configured token or secret.
    fn override_channels(&mut self, env: impl Fn(&str) -> Option<String>) {
        let var = |name: &str| env(name).filter(|value| !value.trim().is_empty());

        if let Some(token) = var("TELEGRAM_BOT_TOKEN") {
            let telegram = self
                .channels
                .telegram
                .get_or_insert_with(|| TelegramConfig {
                    bot_token: String::new(),
                    webhook_url: None,
                    allowed_users: None,
                    webhook_secret: None,
                });
            telegram.bot_token = token;
            if let Some(url) = var("TELEGRAM_WEBHOOK_URL") {
                telegram.webhook_url = Some(url);
            }
            if let Some(secret) = var("TELEGRAM_WEBHOOK_SECRET") {
                telegram.webhook_secret = Some(secret);
            }
        }

        if let Some(token) = var("DISCORD_BOT_TOKEN") {
            let application_id = var("DISCORD_APPLICATION_ID");
            let public_key = var("DISCORD_PUBLIC_KEY");
            if let Some(discord) = &mut self.channels.discord {
                discord.bot_token = token;
                if let Some(id) = application_id {
                    discord.application_id = id;
                }
                if let Some(key) = public_key {
                    discord.public_key = key;
                }
            } else if let (Some(application_id), Some(public_key)) = (application_id, public_key) {
                // Built only whole: a Discord bot is nothing without all three.
                self.channels.discord = Some(DiscordConfig {
                    bot_token: token,
                    application_id,
                    public_key,
                });
            }
        }
    }

    /// The LLM keys [`Self::with_env_overrides`] takes from `env`: each
    /// provider's variable overrides that provider's own field. A blank
    /// variable is unset, as it is at load, and overrides nothing: an
    /// exported-but-empty `ANTHROPIC_API_KEY` used to blank the stored key.
    fn override_llm_keys(&mut self, env: impl Fn(&str) -> Option<String>) {
        let set = |field: &mut Option<String>, name: &str| {
            if let Some(key) = env(name).filter(|key| !key.trim().is_empty()) {
                *field = Some(key);
            }
        };
        set(&mut self.llm.api_key, "ANTHROPIC_API_KEY");
        // Until 2026-09-18 this went to `api_key` when `[llm].provider` was
        // `openai` — the CLI's chat key then — which the daemon registers as
        // the Anthropic credential.
        set(&mut self.llm.openai_api_key, "OPENAI_API_KEY");
        set(&mut self.llm.openrouter_api_key, "OPENROUTER_API_KEY");
    }
}

/// Generate a default config file content
#[must_use] 
pub fn generate_default_config() -> String {
    let config = Config::default();
    toml::to_string_pretty(&config).unwrap_or_default()
}

/// Recursively copy a directory tree. Used only for the one-shot legacy-path migrate.
fn copy_dir_recursive(src: &Path, dst: &Path) -> std::io::Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        let target = dst.join(entry.file_name());
        if ty.is_dir() {
            copy_dir_recursive(&entry.path(), &target)?;
        } else if ty.is_file() {
            fs::copy(entry.path(), target)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------
    // Any-provider API key detection (onboarding gate)
    // -----------------------------------------------------------------

    #[test]
    fn no_credentials_means_no_api_key() {
        assert!(!LlmConfig::default().has_configured_api_key());
    }

    #[test]
    fn each_provider_key_counts_on_its_own() {
        for set in [
            |c: &mut LlmConfig| c.api_key = Some("k".into()),
            |c: &mut LlmConfig| c.anthropic_oauth_token = Some("k".into()),
            |c: &mut LlmConfig| c.openai_api_key = Some("k".into()),
            |c: &mut LlmConfig| c.openrouter_api_key = Some("k".into()),
            |c: &mut LlmConfig| c.github_token = Some("k".into()),
        ] {
            let mut llm = LlmConfig::default();
            set(&mut llm);
            assert!(
                llm.has_configured_api_key(),
                "a single provider credential must satisfy the gate"
            );
        }
    }

    #[test]
    fn blank_or_whitespace_key_does_not_count() {
        let mut llm = LlmConfig {
            openai_api_key: Some("   ".into()),
            ..Default::default()
        };
        assert!(
            !llm.has_configured_api_key(),
            "a whitespace-only key is not a real credential"
        );
        llm.openai_api_key = Some(String::new());
        assert!(
            !llm.has_configured_api_key(),
            "an empty key is not a credential"
        );
    }

    // -----------------------------------------------------------------
    // The chat provider's key, in its own field
    // -----------------------------------------------------------------

    /// Every key field set, each to its own provider's name.
    fn llm_with_every_key(provider: &str) -> LlmConfig {
        LlmConfig {
            provider: provider.to_string(),
            api_key: Some("anthropic".into()),
            openai_api_key: Some("openai".into()),
            openrouter_api_key: Some("openrouter".into()),
            ..LlmConfig::default()
        }
    }

    #[test]
    fn each_provider_reads_its_own_key() {
        for (provider, key) in [
            ("openai", "openai"),
            ("openrouter", "openrouter"),
            ("anthropic", "anthropic"),
            // The CLI's chat falls back to Anthropic for a name it does not know.
            ("something-else", "anthropic"),
        ] {
            assert_eq!(
                llm_with_every_key(provider).provider_api_key(),
                Some(key),
                "{provider}"
            );
        }
    }

    #[test]
    fn an_entered_key_is_stored_in_the_providers_own_field() {
        for provider in ["openai", "openrouter", "anthropic"] {
            let mut llm = LlmConfig {
                provider: provider.to_string(),
                ..LlmConfig::default()
            };
            *llm.provider_api_key_mut() = Some("entered".into());
            let stored_in = [
                ("anthropic", &llm.api_key),
                ("openai", &llm.openai_api_key),
                ("openrouter", &llm.openrouter_api_key),
            ];
            for (field, value) in stored_in {
                let expected = (field == provider).then_some("entered");
                assert_eq!(value.as_deref(), expected, "{provider}: the {field} field");
            }
        }
    }

    #[test]
    fn a_blank_own_key_is_no_key() {
        let mut llm = llm_with_every_key("openrouter");
        llm.openrouter_api_key = Some("  ".into());
        assert_eq!(llm.provider_api_key(), None);
    }

    /// `OPENAI_API_KEY` is `OpenAI`'s key: it overrides `openai_api_key`, and
    /// never lands in `api_key`, which the daemon hands to Anthropic.
    #[test]
    fn each_environment_key_overrides_its_own_field() {
        let env = |name: &str| match name {
            "ANTHROPIC_API_KEY" => Some("env-anthropic".to_string()),
            "OPENAI_API_KEY" => Some("env-openai".to_string()),
            _ => None,
        };
        for provider in ["openai", "anthropic", "openrouter"] {
            let mut config = Config::default();
            config.llm.provider = provider.to_string();
            config.override_llm_keys(env);
            assert_eq!(config.llm.api_key.as_deref(), Some("env-anthropic"), "{provider}");
            assert_eq!(config.llm.openai_api_key.as_deref(), Some("env-openai"), "{provider}");
        }

        let mut config = Config::default();
        config.llm.provider = "openai".to_string();
        config.override_llm_keys(|name| {
            (name == "OPENAI_API_KEY").then(|| "env-openai".to_string())
        });
        assert_eq!(config.llm.api_key, None);
    }

    #[test]
    fn the_openrouter_variable_overrides_its_own_field() {
        let mut config = Config::default();
        config.llm.provider = "openrouter".to_string();
        config.llm.openrouter_api_key = Some("from-config".to_string());
        config.override_llm_keys(|name| {
            (name == "OPENROUTER_API_KEY").then(|| "env-openrouter".to_string())
        });
        assert_eq!(
            config.llm.openrouter_api_key.as_deref(),
            Some("env-openrouter")
        );
        assert_eq!(config.llm.api_key, None);
    }

    /// An exported-but-empty variable is unset, as it is at load: it must not
    /// blank a key the keyring or config.toml supplied.
    #[test]
    fn a_blank_environment_key_overrides_nothing() {
        let mut config = Config::default();
        config.llm.api_key = Some("stored-anthropic".to_string());
        config.llm.openai_api_key = Some("stored-openai".to_string());
        config.llm.openrouter_api_key = Some("stored-openrouter".to_string());
        config.override_llm_keys(|name| match name {
            "ANTHROPIC_API_KEY" => Some(String::new()),
            "OPENAI_API_KEY" => Some("  ".to_string()),
            "OPENROUTER_API_KEY" => Some("\t".to_string()),
            _ => None,
        });
        assert_eq!(config.llm.api_key.as_deref(), Some("stored-anthropic"));
        assert_eq!(config.llm.openai_api_key.as_deref(), Some("stored-openai"));
        assert_eq!(
            config.llm.openrouter_api_key.as_deref(),
            Some("stored-openrouter")
        );
    }

    // -----------------------------------------------------------------
    // Sub-agent model fallback chain
    // -----------------------------------------------------------------

    #[test]
    fn sub_agent_models_win_when_set() {
        let llm = LlmConfig {
            model_priority: vec!["chat-a".into(), "chat-b".into()],
            sub_agent_models: vec!["sub-a".into(), "sub-b".into()],
            ..Default::default()
        };
        assert_eq!(
            llm.effective_sub_agent_models(),
            vec!["sub-a".to_string(), "sub-b".to_string()]
        );
    }

    #[test]
    fn an_empty_sub_agent_list_falls_back_to_the_chat_list() {
        let llm = LlmConfig {
            model_priority: vec!["chat-a".into(), "chat-b".into()],
            ..Default::default()
        };
        assert_eq!(
            llm.effective_sub_agent_models(),
            vec!["chat-a".to_string(), "chat-b".to_string()],
            "no dedicated sub-agent models = follow the main chat chain"
        );
    }

    #[test]
    fn the_legacy_single_model_still_works_and_is_outranked_by_the_list() {
        let mut llm = LlmConfig {
            model_priority: vec!["chat-a".into()],
            sub_agent_model: Some("legacy-sub".into()),
            ..Default::default()
        };
        assert_eq!(
            llm.effective_sub_agent_models(),
            vec!["legacy-sub".to_string()],
            "a config saved before the list existed keeps its behaviour"
        );

        llm.sub_agent_models = vec!["new-sub".into()];
        assert_eq!(
            llm.effective_sub_agent_models(),
            vec!["new-sub".to_string()],
            "the list supersedes the legacy field"
        );
    }

    #[test]
    fn the_chain_is_never_empty() {
        let llm = LlmConfig::default();
        assert_eq!(
            llm.effective_sub_agent_models(),
            vec![llm.model],
            "with nothing configured, sub-agents still get the primary model"
        );
        // An empty-string legacy value must not become a bogus candidate.
        let blank = LlmConfig {
            sub_agent_model: Some(String::new()),
            ..Default::default()
        };
        assert_eq!(blank.effective_sub_agent_models(), vec![blank.model]);
    }

    #[test]
    fn save_to_strips_secrets_from_disk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let mut cfg = Config::default();
        cfg.llm.api_key = Some("sk-secret-anthropic".into());
        cfg.llm.openai_api_key = Some("sk-secret-openai".into());
        cfg.llm.openrouter_api_key = Some("sk-secret-or".into());
        cfg.llm.github_token = Some("ghp_secret".into());
        cfg.llm.ollama_api_key = Some("ollama-secret".into());
        cfg.llm.anthropic_oauth_token = Some("oauth-secret".into());
        cfg.tools.brave_api_key = Some("brave-secret".into());
        cfg.save_to(&path).unwrap();

        let on_disk = std::fs::read_to_string(&path).unwrap();
        for needle in [
            "sk-secret-anthropic",
            "sk-secret-openai",
            "sk-secret-or",
            "ghp_secret",
            "ollama-secret",
            "oauth-secret",
            "brave-secret",
        ] {
            assert!(
                !on_disk.contains(needle),
                "secret {needle:?} leaked into config.toml: {on_disk}"
            );
        }
        // In-memory config is untouched so the running process still has the keys.
        assert_eq!(cfg.llm.api_key.as_deref(), Some("sk-secret-anthropic"));
    }

    #[test]
    fn storing_secrets_leaves_them_held_as_a_load_gets_them_back() {
        // Onboarding stores the key just entered and builds the chat client from
        // the same Config; the migrate alone blanked it, so that chat had no key.
        let dir = tempfile::tempdir().unwrap();
        let store = crate::credentials::SecureStore::file_only_at(dir.path().to_path_buf());
        let no_env = |_: &str| None;
        let mut cfg = Config::default();
        cfg.llm.api_key = Some(" sk-entered ".into());
        cfg.llm.openai_api_key = Some("sk-openai".into());
        cfg.llm.ollama_api_key = Some("ollama-token".into());
        cfg.tools.brave_api_key = Some("brave".into());

        cfg.store_secrets_in(&store, no_env).unwrap();

        assert_eq!(
            cfg.llm.api_key.as_deref(),
            Some("sk-entered"),
            "held as stored"
        );
        let mut next_run = Config::default();
        next_run.load_secrets_from(&store, no_env);
        assert_eq!(cfg.llm.api_key, next_run.llm.api_key);
        assert_eq!(cfg.llm.openai_api_key, next_run.llm.openai_api_key);
        assert_eq!(cfg.llm.ollama_api_key, next_run.llm.ollama_api_key);
        assert_eq!(cfg.tools.brave_api_key, next_run.tools.brave_api_key);
        assert!(
            cfg.llm.ollama_api_key.is_some(),
            "bound to this server, so held"
        );
    }

    #[test]
    fn conversation_is_remembered_by_default() {
        // A memory that only holds what someone thought to save is a notebook.
        assert!(MemoryConfig::default().auto_remember_messages);
        // A config that predates the field gets the same behaviour.
        let parsed: MemoryConfig = toml::from_str("").unwrap();
        assert!(parsed.auto_remember_messages);
    }

    #[test]
    fn opting_out_of_conversation_capture_is_still_honoured() {
        // The switch stays real: chat content is the user's own words, and
        // "on by default" must not quietly become "cannot be turned off".
        let parsed: MemoryConfig =
            toml::from_str("auto_remember_messages = false").unwrap();
        assert!(!parsed.auto_remember_messages);
    }

    #[test]
    fn legacy_thinking_enabled_key_still_loads() {
        // `thinking_enabled` was removed 2026-08-04 (thinking is always on).
        // Every install that ever touched Settings → Agent has the key on
        // disk, and a config that refuses to parse is a dead app — so the
        // stale key must be ignored, not rejected. This is only true while
        // nothing in the chain uses `#[serde(deny_unknown_fields)]`.
        let legacy = r#"
[agent]
name = "Nanna"
personality_mode = "balanced"
thinking_enabled = true
streaming_enabled = true
"#;
        let config: Config = toml::from_str(legacy).expect("legacy config must still parse");
        assert_eq!(config.agent.name, "Nanna");
        assert!(config.agent.streaming_enabled);

        // The opposite disk state (an install that turned thinking OFF) must
        // parse too — and cannot turn anything off any more, by construction:
        // there is no field left for it to land in.
        let legacy_off = "[agent]\nthinking_enabled = false\n";
        let off: Config = toml::from_str(legacy_off).expect("legacy off-state must still parse");
        assert_eq!(off.agent.name, AgentConfig::default().name);
    }

    #[test]
    fn legacy_server_host_key_still_loads() {
        // `[server].host` was removed 2026-09-11: nothing ever read it (the
        // bind is `nanna server --host`, default loopback). Every config ever
        // written from the old defaults carries `host = "0.0.0.0"` on disk,
        // and a config that refuses to parse is a dead app — so the stale key
        // must be ignored, and the keys beside it must still land.
        let legacy = r#"
[server]
enabled = true
host = "0.0.0.0"
port = 4100
webhook_secret = "s3cret"
"#;
        let config: Config = toml::from_str(legacy).expect("legacy config must still parse");
        assert_eq!(config.server.port, 4100, "the keys beside it still land");
        assert_eq!(config.server.webhook_secret.as_deref(), Some("s3cret"));

        // The webhook secret beside it is hand-written, the only way it was
        // ever set. It still takes effect, and as a secret it is filed in the
        // secure store as the file loads (`server_secret.rs`): the next save
        // drops it from config.toml and the load after that still has it.
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        std::fs::write(&path, legacy).expect("write");
        let store = SecureStore::file_only_at(dir.path().join("store"));
        let no_env = |_: &str| None;
        let loaded = Config::load_from_with(&path, &store, no_env).expect("legacy config loads");
        assert_eq!(loaded.server.port, 4100);
        assert_eq!(loaded.server.webhook_secret.as_deref(), Some("s3cret"));
        loaded.save_to(&path).expect("save");
        let saved = std::fs::read_to_string(&path).expect("config.toml");
        assert!(!saved.contains("s3cret"), "the secret left config.toml: {saved}");
        let next = Config::load_from_with(&path, &store, no_env).expect("saved config loads");
        assert_eq!(next.server.port, 4100);
        assert_eq!(
            next.server.webhook_secret.as_deref(),
            Some("s3cret"),
            "the save lost nothing"
        );

        // And it is gone for good: a config written from today's defaults no
        // longer carries a key that looks like it controls the bind.
        let written = toml::to_string(&Config::default()).expect("default config serializes");
        let server = written
            .split("[server]")
            .nth(1)
            .expect("the default config writes a [server] table");
        let server = server.split("\n[").next().unwrap_or_default();
        assert!(!server.contains("host"), "no host key is written: {server}");
    }

    #[test]
    fn legacy_llm_ollama_url_key_still_loads() {
        // `[llm].ollama_url` was retired 2026-09-18 (owner decision: summaries
        // follow the Settings list through chat's providers). It was the
        // summarizers' own Ollama address — localhost by default, no token —
        // so every config ever saved carries it, and a config that refuses to
        // parse is a dead app. The stale key must be ignored, and the keys
        // beside it must still land.
        let legacy = r#"
[llm]
summarization_priority = ["ollama/qwen3:4b"]
ollama_url = "http://localhost:11434"
model = "qwen3.5:9b"
"#;
        let config: Config = toml::from_str(legacy).expect("legacy config must still parse");
        assert_eq!(config.llm.summarization_priority, vec!["ollama/qwen3:4b"]);
        assert_eq!(config.llm.model, "qwen3.5:9b", "the keys beside it still land");

        // And it is gone for good: a saved config no longer carries a key
        // that looks like it points the summarizer somewhere.
        let written = toml::to_string(&config).expect("config serializes");
        assert!(!written.contains("ollama_url"), "no ollama_url is written: {written}");
    }

    /// A leftover `[llm].ollama_url` that named another server than chat's is
    /// the one case where the retirement moves summaries somewhere new, and
    /// serde drops the key before anything else could see it. Loading says
    /// so, naming both servers and what to do.
    #[test]
    fn a_leftover_ollama_url_naming_another_server_is_announced() {
        let legacy = r#"
[llm]
ollama_url = "http://gpu-box:11434"

[memory]
ollama_host = "http://localhost:11434"
"#;
        let config: Config = toml::from_str(legacy).expect("legacy config must still parse");
        let notice =
            retired_ollama_url_notice(legacy, &config).expect("a moved summarizer is announced");
        assert!(
            notice.contains("http://gpu-box:11434") && notice.contains("http://localhost:11434"),
            "names both servers: {notice}"
        );
        assert!(notice.contains("Settings"), "says where to set it: {notice}");
    }

    /// The notice goes to the log, so a password or token in either address
    /// must not.
    #[test]
    fn a_leftover_ollama_url_notice_carries_no_credentials() {
        let legacy = r#"
[llm]
ollama_url = "https://user:hunter2@gpu-box/ollama?token=abc"

[memory]
ollama_host = "https://me:pw@remote.example/ollama"
"#;
        let config: Config = toml::from_str(legacy).expect("parses");
        let notice = retired_ollama_url_notice(legacy, &config).expect("announced");
        for secret in ["hunter2", "user:", "token=abc", "me:pw"] {
            assert!(!notice.contains(secret), "{secret} leaked: {notice}");
        }
        assert!(notice.contains("gpu-box") && notice.contains("remote.example"), "{notice}");
    }

    /// The same server, however it is spelled, moves nothing — the shipped
    /// default case — and neither does a config without the key.
    #[test]
    fn a_leftover_ollama_url_naming_the_same_server_is_quiet() {
        let same = r#"
[llm]
ollama_url = "http://127.0.0.1:11434/"

[memory]
ollama_host = "http://localhost:11434"
"#;
        let config: Config = toml::from_str(same).expect("parses");
        assert_eq!(retired_ollama_url_notice(same, &config), None);

        let without = "[memory]\nollama_host = \"http://gpu-box:11434\"\n";
        let config: Config = toml::from_str(without).expect("parses");
        assert_eq!(retired_ollama_url_notice(without, &config), None);
    }

    // -----------------------------------------------------------------
    // Channel sections under environment overrides
    // -----------------------------------------------------------------

    /// An environment holding exactly `vars`.
    fn env_of<'a>(vars: &'a [(&str, &str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |name| {
            vars.iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| (*value).to_string())
        }
    }

    fn with_telegram() -> Config {
        let mut config = Config::default();
        config.channels.telegram = Some(TelegramConfig {
            bot_token: "cfg-token".into(),
            webhook_url: Some("https://cfg.example/hook".into()),
            allowed_users: Some(vec![42]),
            webhook_secret: Some("cfg-secret".into()),
        });
        config
    }

    fn with_discord() -> Config {
        let mut config = Config::default();
        config.channels.discord = Some(DiscordConfig {
            bot_token: "cfg-token".into(),
            application_id: "cfg-app".into(),
            public_key: "cfg-key".into(),
        });
        config
    }

    /// The token names the bot, not the section: the allowlist, the webhook
    /// URL and the webhook secret stay as configured.
    #[test]
    fn a_telegram_token_from_the_environment_keeps_the_configured_section() {
        let config =
            with_telegram().with_env_overrides_from(env_of(&[("TELEGRAM_BOT_TOKEN", "env-token")]));
        let telegram = config.channels.telegram.expect("section kept");
        assert_eq!(telegram.bot_token, "env-token");
        assert_eq!(
            telegram.allowed_users,
            Some(vec![42]),
            "the allowlist survives"
        );
        assert_eq!(
            telegram.webhook_url.as_deref(),
            Some("https://cfg.example/hook")
        );
        assert_eq!(telegram.webhook_secret.as_deref(), Some("cfg-secret"));
    }

    #[test]
    fn telegram_webhook_variables_override_only_their_own_fields() {
        let config = with_telegram().with_env_overrides_from(env_of(&[
            ("TELEGRAM_BOT_TOKEN", "env-token"),
            ("TELEGRAM_WEBHOOK_URL", "https://env.example/hook"),
            ("TELEGRAM_WEBHOOK_SECRET", "env-secret"),
        ]));
        let telegram = config.channels.telegram.expect("section kept");
        assert_eq!(telegram.bot_token, "env-token");
        assert_eq!(
            telegram.webhook_url.as_deref(),
            Some("https://env.example/hook")
        );
        assert_eq!(telegram.webhook_secret.as_deref(), Some("env-secret"));
        assert_eq!(
            telegram.allowed_users,
            Some(vec![42]),
            "the allowlist survives"
        );
    }

    /// With no `[channels.telegram]` the environment is the whole section.
    #[test]
    fn a_telegram_section_is_built_from_the_environment_when_absent() {
        let config = Config::default().with_env_overrides_from(env_of(&[
            ("TELEGRAM_BOT_TOKEN", "env-token"),
            ("TELEGRAM_WEBHOOK_URL", "https://env.example/hook"),
            ("TELEGRAM_WEBHOOK_SECRET", "env-secret"),
        ]));
        let telegram = config.channels.telegram.expect("section built");
        assert_eq!(telegram.bot_token, "env-token");
        assert_eq!(
            telegram.webhook_url.as_deref(),
            Some("https://env.example/hook")
        );
        assert_eq!(telegram.webhook_secret.as_deref(), Some("env-secret"));
        assert_eq!(telegram.allowed_users, None);

        let config = Config::default()
            .with_env_overrides_from(env_of(&[("TELEGRAM_BOT_TOKEN", "env-token")]));
        let telegram = config.channels.telegram.expect("section built");
        assert_eq!(telegram.bot_token, "env-token");
        assert_eq!(telegram.webhook_url, None);
        assert_eq!(telegram.webhook_secret, None);
    }

    #[test]
    fn a_discord_token_from_the_environment_keeps_the_configured_section() {
        let config =
            with_discord().with_env_overrides_from(env_of(&[("DISCORD_BOT_TOKEN", "env-token")]));
        let discord = config.channels.discord.expect("section kept");
        assert_eq!(discord.bot_token, "env-token");
        assert_eq!(discord.application_id, "cfg-app");
        assert_eq!(discord.public_key, "cfg-key");

        let config = with_discord().with_env_overrides_from(env_of(&[
            ("DISCORD_BOT_TOKEN", "env-token"),
            ("DISCORD_PUBLIC_KEY", "env-key"),
        ]));
        let discord = config.channels.discord.expect("section kept");
        assert_eq!(discord.bot_token, "env-token");
        assert_eq!(discord.application_id, "cfg-app");
        assert_eq!(discord.public_key, "env-key");

        let config = with_discord().with_env_overrides_from(env_of(&[
            ("DISCORD_BOT_TOKEN", "env-token"),
            ("DISCORD_APPLICATION_ID", "env-app"),
        ]));
        let discord = config.channels.discord.expect("section kept");
        assert_eq!(discord.application_id, "env-app");
        assert_eq!(discord.public_key, "cfg-key");
    }

    /// With no `[channels.discord]` the environment must name all three
    /// fields: a section missing its application ID or public key is no
    /// Discord bot.
    #[test]
    fn a_discord_section_is_built_from_the_environment_only_when_complete() {
        let config = Config::default().with_env_overrides_from(env_of(&[
            ("DISCORD_BOT_TOKEN", "env-token"),
            ("DISCORD_APPLICATION_ID", "env-app"),
            ("DISCORD_PUBLIC_KEY", "env-key"),
        ]));
        let discord = config.channels.discord.expect("section built");
        assert_eq!(discord.bot_token, "env-token");
        assert_eq!(discord.application_id, "env-app");
        assert_eq!(discord.public_key, "env-key");

        let config = Config::default().with_env_overrides_from(env_of(&[
            ("DISCORD_BOT_TOKEN", "env-token"),
            ("DISCORD_APPLICATION_ID", "env-app"),
        ]));
        assert!(config.channels.discord.is_none());
    }

    /// The bot token opens a channel's overrides: without it the environment
    /// neither builds a section nor edits the configured one.
    #[test]
    fn without_a_bot_token_the_environment_leaves_the_channel_alone() {
        let env = [
            ("TELEGRAM_WEBHOOK_URL", "https://env.example/hook"),
            ("TELEGRAM_WEBHOOK_SECRET", "env-secret"),
            ("DISCORD_APPLICATION_ID", "env-app"),
            ("DISCORD_PUBLIC_KEY", "env-key"),
        ];
        let config = Config::default().with_env_overrides_from(env_of(&env));
        assert!(config.channels.telegram.is_none());
        assert!(config.channels.discord.is_none());

        let mut config = with_telegram();
        config.channels.discord = with_discord().channels.discord;
        let config = config.with_env_overrides_from(env_of(&env));
        let telegram = config.channels.telegram.expect("section kept");
        assert_eq!(
            telegram.webhook_url.as_deref(),
            Some("https://cfg.example/hook")
        );
        assert_eq!(telegram.webhook_secret.as_deref(), Some("cfg-secret"));
        let discord = config.channels.discord.expect("section kept");
        assert_eq!(discord.application_id, "cfg-app");
        assert_eq!(discord.public_key, "cfg-key");
    }

    /// A variable exported empty (`TELEGRAM_WEBHOOK_SECRET=` in an env file)
    /// names nothing. Taken as a value it would blank the configured one — a
    /// blank webhook secret shuts the webhook — or build a bot around an
    /// empty token.
    #[test]
    fn a_blank_channel_variable_names_nothing() {
        let config = with_telegram().with_env_overrides_from(env_of(&[
            ("TELEGRAM_BOT_TOKEN", "env-token"),
            ("TELEGRAM_WEBHOOK_URL", ""),
            ("TELEGRAM_WEBHOOK_SECRET", "  "),
        ]));
        let telegram = config.channels.telegram.expect("section kept");
        assert_eq!(
            telegram.webhook_url.as_deref(),
            Some("https://cfg.example/hook")
        );
        assert_eq!(telegram.webhook_secret.as_deref(), Some("cfg-secret"));

        let config = with_telegram().with_env_overrides_from(env_of(&[("TELEGRAM_BOT_TOKEN", "")]));
        assert_eq!(
            config.channels.telegram.expect("section kept").bot_token,
            "cfg-token"
        );
        let config =
            Config::default().with_env_overrides_from(env_of(&[("TELEGRAM_BOT_TOKEN", " ")]));
        assert!(config.channels.telegram.is_none());

        let config = with_discord().with_env_overrides_from(env_of(&[
            ("DISCORD_BOT_TOKEN", ""),
            ("DISCORD_APPLICATION_ID", "env-app"),
        ]));
        let discord = config.channels.discord.expect("section kept");
        assert_eq!(discord.bot_token, "cfg-token");
        assert_eq!(discord.application_id, "cfg-app");
        let config = with_discord().with_env_overrides_from(env_of(&[
            ("DISCORD_BOT_TOKEN", "env-token"),
            ("DISCORD_PUBLIC_KEY", ""),
        ]));
        assert_eq!(
            config.channels.discord.expect("section kept").public_key,
            "cfg-key"
        );
    }

    // -----------------------------------------------------------------
    // Data storage location (`[general] data_dir`)
    // -----------------------------------------------------------------

    #[test]
    fn no_configured_data_dir_means_the_platform_default() {
        let config = Config::default();
        assert!(
            !config.has_custom_data_dir(),
            "a stock install is not relocated"
        );
        assert_eq!(
            config.resolve_data_dir().ok(),
            Config::default_data_dir().ok(),
            "with nothing configured, resolution must agree with the platform default"
        );
    }

    #[test]
    fn a_configured_data_dir_wins_over_the_platform_default() {
        let mut config = Config::default();
        let chosen = PathBuf::from(if cfg!(windows) {
            r"D:\nanna-store"
        } else {
            "/srv/nanna-store"
        });
        config.general.data_dir = Some(chosen.clone());

        assert!(config.has_custom_data_dir());
        assert_eq!(
            config.resolve_data_dir().expect("an explicit path resolves"),
            chosen,
            "an explicit override is returned verbatim"
        );
        assert_ne!(
            config.resolve_data_dir().ok(),
            Config::default_data_dir().ok(),
            "and it must not collapse back to the platform location"
        );
    }

    #[test]
    fn a_blank_data_dir_is_treated_as_unset() {
        // A text input that submits "" must not redirect the entire store to a
        // path of "" — the same policy the config-path override applies.
        for blank in ["", "   ", "\t\n"] {
            let mut config = Config::default();
            config.general.data_dir = Some(PathBuf::from(blank));
            assert!(
                !config.has_custom_data_dir(),
                "{blank:?} is a half-filled form, not a location"
            );
            assert_eq!(
                config.resolve_data_dir().ok(),
                Config::default_data_dir().ok(),
                "{blank:?} must fall through to the default"
            );
        }
    }

    #[test]
    fn a_config_written_before_the_field_existed_still_loads() {
        // Every install that predates this setting has no `data_dir` key, and a
        // config that refuses to parse is a dead app.
        let legacy = r#"
[general]
name = "Nanna"
log_level = "info"
"#;
        let config: Config = toml::from_str(legacy).expect("legacy config must still parse");
        assert_eq!(config.general.name, "Nanna");
        assert_eq!(
            config.general.data_dir, None,
            "an absent key means the platform default, not a panic"
        );
        assert!(!config.has_custom_data_dir());
    }

    #[test]
    fn a_data_dir_survives_a_save_load_round_trip() {
        // It is not a secret, so unlike the API keys it must still be on disk
        // after `save_to` — otherwise the setting silently forgets itself.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let chosen = dir.path().join("store");

        let mut cfg = Config::default();
        cfg.general.data_dir = Some(chosen.clone());
        cfg.save_to(&path).unwrap();

        let reloaded: Config =
            toml::from_str(&std::fs::read_to_string(&path).unwrap()).expect("round-trips");
        assert_eq!(reloaded.general.data_dir, Some(chosen));
    }

    /// The keyring-free reader must agree with the full config load about
    /// where data lives — otherwise the daemon writes its logs somewhere other
    /// than its database, which is precisely the split this reader exists to
    /// close.
    ///
    /// Env mutation is kept in ONE `#[test]` on purpose: `std::env` is
    /// process-wide and Rust runs tests in threads, so splitting these would
    /// let a sibling observe the variable mid-mutation.
    #[test]
    fn the_keyring_free_reader_agrees_with_a_full_load() {
        let restore = std::env::var(Config::CONFIG_PATH_ENV).ok();
        let dir = tempfile::tempdir().unwrap();
        let cfg_path = dir.path().join("config.toml");
        let chosen = dir.path().join("relocated-store");

        // A config that names a data dir: the reader must find it.
        let mut cfg = Config::default();
        cfg.general.data_dir = Some(chosen.clone());
        cfg.save_to(&cfg_path).unwrap();

        // SAFETY: single-threaded section; the variable is restored below.
        unsafe { std::env::set_var(Config::CONFIG_PATH_ENV, &cfg_path) };
        assert_eq!(
            Config::data_dir_from_disk().ok(),
            Some(chosen),
            "the override on disk must be honoured without a keyring read"
        );

        // A config that names none: fall through to the platform default,
        // matching `resolve_data_dir` on a stock install.
        let plain = dir.path().join("plain.toml");
        Config::default().save_to(&plain).unwrap();
        unsafe { std::env::set_var(Config::CONFIG_PATH_ENV, &plain) };
        assert_eq!(
            Config::data_dir_from_disk().ok(),
            Config::default_data_dir().ok(),
            "no override means the platform default, not an empty path"
        );

        // Malformed TOML must not take the daemon down during logging setup;
        // the real parse error surfaces later, with logging up.
        let broken = dir.path().join("broken.toml");
        std::fs::write(&broken, "this is not = = valid toml [[[").unwrap();
        unsafe { std::env::set_var(Config::CONFIG_PATH_ENV, &broken) };
        assert_eq!(
            Config::data_dir_from_disk().ok(),
            Config::default_data_dir().ok(),
            "an unparseable config falls back rather than failing"
        );

        // A config file that does not exist at all — a fresh install.
        unsafe { std::env::set_var(Config::CONFIG_PATH_ENV, dir.path().join("absent.toml")) };
        assert_eq!(
            Config::data_dir_from_disk().ok(),
            Config::default_data_dir().ok(),
            "a missing config means defaults, not an error"
        );

        match restore {
            Some(prev) => unsafe { std::env::set_var(Config::CONFIG_PATH_ENV, prev) },
            None => unsafe { std::env::remove_var(Config::CONFIG_PATH_ENV) },
        }
    }

    // -----------------------------------------------------------------
    // Data-directory validation — each refusal names its own reason
    // -----------------------------------------------------------------
    #[test]
    fn validation_accepts_a_writable_directory() {
        let dir = tempfile::tempdir().unwrap();
        validate_data_dir(dir.path()).expect("a fresh temp dir is writable");
    }

    #[test]
    fn validation_creates_a_directory_that_does_not_exist_yet() {
        // Picking a not-yet-existing folder in a file dialog is ordinary;
        // refusing it would send the user out to a file manager.
        let dir = tempfile::tempdir().unwrap();
        let fresh = dir.path().join("does").join("not").join("exist");
        assert!(!fresh.exists());

        validate_data_dir(&fresh).expect("a creatable path is accepted");
        assert!(fresh.is_dir(), "and it really exists afterwards");
    }

    #[test]
    fn validation_leaves_no_probe_file_behind() {
        // The probe proves writability by writing; if it did not clean up, every
        // validation would litter the user's chosen folder.
        let dir = tempfile::tempdir().unwrap();
        validate_data_dir(dir.path()).unwrap();

        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(Result::ok)
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        assert!(
            leftovers.is_empty(),
            "the write probe must clean up after itself, found: {leftovers:?}"
        );
    }

    #[test]
    fn validation_refuses_an_empty_path() {
        for blank in ["", "   "] {
            let err = validate_data_dir(Path::new(blank))
                .expect_err("an empty choice is not a location");
            assert!(
                matches!(err, DataDirError::Empty),
                "{blank:?} should report Empty, got {err}"
            );
        }
    }

    #[test]
    fn validation_refuses_a_relative_path() {
        // The GUI sidecar and a terminal daemon have different working
        // directories, so a relative path means two different stores.
        let err = validate_data_dir(Path::new("nanna-data"))
            .expect_err("a relative path is ambiguous and must be refused");
        assert!(
            matches!(err, DataDirError::NotAbsolute(_)),
            "expected NotAbsolute, got {err}"
        );
        assert!(
            err.to_string().contains("absolute"),
            "the message tells the user what to do instead: {err}"
        );
    }

    #[test]
    fn validation_refuses_an_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("i-am-a-file.txt");
        std::fs::write(&file, b"not a directory").unwrap();

        let err = validate_data_dir(&file).expect_err("a file cannot hold the store");
        assert!(
            matches!(err, DataDirError::NotADirectory(_)),
            "expected NotADirectory, got {err}"
        );
    }

    /// The writability probe must actually probe. A permission bit check would
    /// pass here on many platforms; only a real write catches it.
    #[cfg(unix)]
    #[test]
    fn validation_refuses_a_read_only_directory() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let locked = dir.path().join("read-only");
        std::fs::create_dir(&locked).unwrap();
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o555)).unwrap();

        // Does the mode bit actually bite on this host? Running as root (or on a
        // filesystem that ignores modes) defeats it, and asserting there would
        // be asserting a falsehood about the code under test.
        let mode_bits_apply = std::fs::write(locked.join(".probe"), b"x").is_err();
        let _ = std::fs::remove_file(locked.join(".probe"));

        let result = validate_data_dir(&locked);

        // Restore before asserting so the tempdir can always clean itself up.
        let _ = std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755));

        if !mode_bits_apply {
            return;
        }
        let err = result.expect_err("a directory we cannot write to must be refused");
        assert!(
            matches!(err, DataDirError::NotWritable { .. }),
            "expected NotWritable, got {err}"
        );
    }

    #[test]
    fn project_dirs_uses_canonical_identity() {        let dirs = project_dirs().expect("home dir");
        let cfg = dirs.config_dir().to_string_lossy().to_lowercase();
        // Windows: .../nanna/nanna ; Unix: .../nanna
        assert!(
            cfg.contains("nanna"),
            "config dir should contain nanna, got {cfg}"
        );
        assert!(
            !cfg.contains("clawd"),
            "canonical path must not use legacy clawd slug: {cfg}"
        );
    }

    /// The whole point of the override: an explicit path wins over the
    /// known-folder location, so a harness can run against a config the
    /// operator does not own.
    ///
    /// These three share one `#[test]` deliberately. `std::env` is process-wide
    /// and Rust runs tests in threads, so splitting them would let a sibling
    /// observe this variable mid-mutation — the classic env-var test flake.
    #[test]
    fn config_path_override_wins_and_ignores_blank() {
        // Save and restore: this variable is process-global, and leaking it
        // would redirect every later test in this binary.
        let restore = std::env::var(Config::CONFIG_PATH_ENV).ok();

        // SAFETY: single-threaded section of this test; the variable is
        // restored before returning.
        unsafe { std::env::set_var(Config::CONFIG_PATH_ENV, "") };
        let blank = Config::default_config_path().expect("blank must fall through, not fail");
        assert!(
            blank.ends_with("config.toml"),
            "an empty override must be treated as unset, not as a path of \"\": {blank:?}"
        );

        unsafe { std::env::set_var(Config::CONFIG_PATH_ENV, "   ") };
        let spaces = Config::default_config_path().expect("whitespace must fall through");
        assert_eq!(
            spaces, blank,
            "whitespace-only must resolve identically to unset"
        );

        unsafe { std::env::set_var(Config::CONFIG_PATH_ENV, "D:/somewhere/custom.toml") };
        let overridden = Config::default_config_path().expect("an explicit path must resolve");
        assert_eq!(
            overridden,
            PathBuf::from("D:/somewhere/custom.toml"),
            "an explicit override must be returned verbatim"
        );
        assert_ne!(
            overridden, blank,
            "the override must not collapse to the known-folder path"
        );

        match restore {
            Some(prev) => unsafe { std::env::set_var(Config::CONFIG_PATH_ENV, prev) },
            None => unsafe { std::env::remove_var(Config::CONFIG_PATH_ENV) },
        }
    }

    /// The shipped default model must not be a DATED snapshot id.
    ///
    /// Anthropic publishes two shapes: an undated family alias
    /// (`claude-sonnet-5`) that tracks the live model, and a dated snapshot
    /// (`claude-sonnet-4-20250514`) pinned to one release — which is retired on
    /// a schedule. A dated default therefore has an expiry date built in, and
    /// this one had already passed it: a daemon booted with no configured
    /// `model_priority` sent every scheduled heartbeat to a `404 not_found_error`
    /// (observed on a real boot, 2026-08-27). Nothing failed loudly, because a
    /// heartbeat failure is logged and swallowed.
    ///
    /// Pinning a snapshot is a legitimate choice for a USER to make in their own
    /// config; it is not a legitimate default for the product to ship.
    #[test]
    fn the_default_model_is_not_a_dated_snapshot() {
        let default_model = LlmConfig::default().model;
        assert!(
            !default_model.is_empty(),
            "a default model must exist — the empty string reaches the router as a bare name"
        );
        assert!(
            !has_date_suffix(&default_model),
            "default model `{default_model}` is a dated snapshot, which will be retired and              turn every unconfigured boot's heartbeat into a 404. Use the undated family alias."
        );
    }

    /// `-YYYYMMDD` at the end of a model id, which is how Anthropic spells a
    /// pinned snapshot. Written as a scan rather than a regex so the guard
    /// carries no dependency of its own.
    fn has_date_suffix(model: &str) -> bool {
        let Some((_, tail)) = model.rsplit_once('-') else {
            return false;
        };
        tail.len() == 8 && tail.bytes().all(|b| b.is_ascii_digit())
    }

    #[test]
    fn date_suffix_detection_reads_both_shapes() {
        assert!(has_date_suffix("claude-sonnet-4-20250514"));
        assert!(has_date_suffix("claude-3-5-haiku-20241022"));
        assert!(!has_date_suffix("claude-sonnet-5"));
        assert!(!has_date_suffix("claude-haiku-4-5"));
        // Negative space: a trailing number that is not a date must not trip it,
        // and neither must a local model whose tag contains a colon.
        assert!(!has_date_suffix("claude-opus-4-1"));
        assert!(!has_date_suffix("ollama/qwen3:14b"));
        assert!(!has_date_suffix("nodashes"));
    }
}

#[cfg(test)]
mod prompt_cache_ttl_tests {
    use super::{Config, PromptCacheTtl};

    #[test]
    fn prompt_cache_ttl_defaults_to_five_minutes_and_accepts_one_hour() {
        let absent: Config = toml::from_str("[llm]\nmodel = \"claude-sonnet-5\"\n")
            .expect("a config without the key must load");
        assert_eq!(absent.llm.prompt_cache_ttl, PromptCacheTtl::FiveMinutes);

        let hour: Config = toml::from_str("[llm]\nprompt_cache_ttl = \"1h\"\n")
            .expect("\"1h\" is an accepted spelling");
        assert_eq!(hour.llm.prompt_cache_ttl, PromptCacheTtl::OneHour);
    }

    #[test]
    fn an_unknown_prompt_cache_ttl_is_rejected_by_name() {
        let error = toml::from_str::<Config>("[llm]\nprompt_cache_ttl = \"2h\"\n")
            .expect_err("only the two lifetimes Anthropic offers are accepted");
        let message = error.to_string();
        assert!(
            message.contains("2h"),
            "the error names the bad value: {message}"
        );
        assert!(message.contains("5m"), "and the accepted ones: {message}");
        assert!(message.contains("1h"), "and the accepted ones: {message}");
    }
}

#[cfg(test)]
mod ollama_token_binding_tests {
    use super::Config;
    use crate::credentials::{SecureStore, keys};

    /// A hermetic store: its own directory, never the OS keyring.
    fn store() -> (tempfile::TempDir, SecureStore) {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = SecureStore::file_only_at(dir.path().to_path_buf());
        (dir, store)
    }

    fn no_env(_: &str) -> Option<String> {
        None
    }

    /// The server the store records for its token; the test store always answers.
    fn recorded_server(store: &SecureStore) -> Option<String> {
        store.ollama_token_host().expect("a file store answers")
    }

    fn configured_for(host: &str) -> Config {
        let mut config = Config::default();
        config.memory.ollama_host = host.to_string();
        config
    }

    #[test]
    fn a_token_saved_for_another_server_is_not_loaded() {
        let (_dir, store) = store();
        store.set(keys::OLLAMA_API_KEY, "token-for-a").expect("set");
        store
            .set(keys::OLLAMA_API_KEY_HOST, "https://a.example/ollama")
            .expect("set");

        let mut config = configured_for("https://b.example/ollama");
        config.load_secrets_from(&store, no_env);
        assert_eq!(
            config.llm.ollama_api_key, None,
            "server B must never be handed server A's token"
        );

        // The server it was saved for, however it is spelled, still gets it.
        let mut config = configured_for("https://A.example:443/ollama/");
        config.load_secrets_from(&store, no_env);
        assert_eq!(config.llm.ollama_api_key.as_deref(), Some("token-for-a"));
    }

    #[test]
    fn a_legacy_token_with_no_recorded_server_still_loads() {
        // Saved by a build that recorded no server: it was being sent to the
        // configured one, and keeps being sent there.
        let (_dir, store) = store();
        store.set(keys::OLLAMA_API_KEY, "legacy-token").expect("set");

        let mut config = configured_for("https://b.example/ollama");
        config.load_secrets_from(&store, no_env);
        assert_eq!(config.llm.ollama_api_key.as_deref(), Some("legacy-token"));
    }

    #[test]
    fn the_environment_token_is_not_bound_to_a_server() {
        // `OLLAMA_API_KEY` is the operator saying "use this", wherever the
        // config points; a stored token bound elsewhere does not veto it.
        let (_dir, store) = store();
        store.set(keys::OLLAMA_API_KEY, "token-for-a").expect("set");
        store
            .set(keys::OLLAMA_API_KEY_HOST, "https://a.example/ollama")
            .expect("set");

        let mut config = configured_for("https://b.example/ollama");
        config.load_secrets_from(&store, |name| {
            (name == "OLLAMA_API_KEY").then(|| "env-token".to_string())
        });
        assert_eq!(config.llm.ollama_api_key.as_deref(), Some("env-token"));
    }

    #[test]
    fn a_saved_token_records_its_server_and_a_blank_one_removes_both() {
        let (_dir, store) = store();
        store
            .save_ollama_token("  s3cret ", " https://a.example/ollama/ ")
            .expect("save");
        assert_eq!(store.ollama_token().as_deref(), Some("s3cret"));
        assert_eq!(
            recorded_server(&store).as_deref(),
            Some("https://a.example/ollama")
        );

        // Whitespace is no token: it clears, it does not keep the old one.
        store.save_ollama_token("   ", "https://a.example/ollama").expect("clear");
        assert_eq!(store.ollama_token(), None);
        assert_eq!(recorded_server(&store), None);
        assert!(!store.exists(keys::OLLAMA_API_KEY));
        assert!(!store.exists(keys::OLLAMA_API_KEY_HOST));
    }

    /// With no server address to attribute it to, a legacy token must stay
    /// unbound. Recorded against a blank address it matched no server ever
    /// again — the token was silently retired.
    #[test]
    fn a_blank_running_address_never_binds_the_token() {
        let (dir, store) = store();
        store
            .set(keys::OLLAMA_API_KEY, "legacy-token")
            .expect("set");
        // A scratch config file: a test must never read the owner's.
        let path = dir.path().join("config.toml");
        Config::default().save_to(&path).expect("save");
        let cfg = Config::load_from_replacing_with(&path, "  ", &store, no_env).expect("load");
        assert_eq!(
            recorded_server(&store),
            None,
            "nothing recorded for no server"
        );
        assert_eq!(cfg.llm.ollama_api_key, None);
        assert!(
            !store.bind_unbound_ollama_token("").expect("bind"),
            "refused at the store too"
        );

        // The first real server then gets it.
        assert!(
            store
                .bind_unbound_ollama_token("https://a.example/ollama")
                .expect("bind")
        );
        assert_eq!(
            recorded_server(&store).as_deref(),
            Some("https://a.example/ollama")
        );
    }

    #[test]
    fn switching_servers_does_not_take_an_unbound_token_along() {
        // A legacy token, loaded for server A as its own.
        let (_dir, store) = store();
        store.set(keys::OLLAMA_API_KEY, "legacy-token").expect("set");
        let mut config = configured_for("https://a.example/ollama");
        config.load_secrets_from(&store, no_env);
        assert_eq!(config.llm.ollama_api_key.as_deref(), Some("legacy-token"));

        // Settings binds it to A before the address changes to B.
        assert!(store.bind_unbound_ollama_token(&config.memory.ollama_host).expect("bind"));
        assert!(
            !store.bind_unbound_ollama_token("https://b.example").expect("bind"),
            "a bound token is never rebound"
        );
        config.memory.ollama_host = "https://b.example/ollama".to_string();
        config.rebind_ollama_token_if_moved_with("https://a.example/ollama", &store, no_env);
        assert_eq!(config.llm.ollama_api_key, None, "B must not get A's token");

        // A fresh load (the daemon's) agrees; and A still gets it.
        let mut reloaded = configured_for("https://b.example/ollama");
        reloaded.load_secrets_from(&store, no_env);
        assert_eq!(reloaded.llm.ollama_api_key, None);
        config.memory.ollama_host = "https://a.example/ollama".to_string();
        config.rebind_ollama_token_if_moved_with("https://b.example/ollama", &store, no_env);
        assert_eq!(config.llm.ollama_api_key.as_deref(), Some("legacy-token"));
    }

    #[test]
    fn an_in_memory_address_change_drops_the_old_servers_token() {
        // The daemon's `config.set memory.ollama_host`: the running config
        // holds A's token and is edited in place to point at B.
        let (_dir, store) = store();
        store
            .save_ollama_token("token-for-a", "https://a.example/ollama")
            .expect("save");
        let mut config = configured_for("https://a.example/ollama");
        config.load_secrets_from(&store, no_env);
        assert_eq!(config.llm.ollama_api_key.as_deref(), Some("token-for-a"));

        // Re-spelling the same server is not a move: the token stays.
        config.memory.ollama_host = "https://A.example:443/ollama/".to_string();
        config.rebind_ollama_token_if_moved_with("https://a.example/ollama", &store, no_env);
        assert_eq!(config.llm.ollama_api_key.as_deref(), Some("token-for-a"));

        config.memory.ollama_host = "https://b.example/ollama".to_string();
        config.rebind_ollama_token_if_moved_with("https://a.example/ollama", &store, no_env);
        assert_eq!(config.llm.ollama_api_key, None);
    }

    #[test]
    fn a_legacy_token_does_not_follow_an_in_memory_move() {
        // The daemon's `config.set memory.ollama_host` with a token saved by
        // an older build: nothing has recorded yet that it was A's, and no
        // Settings page bound it first.
        let (_dir, store) = store();
        store
            .set(keys::OLLAMA_API_KEY, "legacy-token")
            .expect("set");
        let mut config = configured_for("https://a.example/ollama");
        config.load_secrets_from(&store, no_env);
        assert_eq!(config.llm.ollama_api_key.as_deref(), Some("legacy-token"));

        config.memory.ollama_host = "https://b.example/ollama".to_string();
        config.rebind_ollama_token_if_moved_with("https://a.example/ollama", &store, no_env);
        assert_eq!(
            config.llm.ollama_api_key, None,
            "B must not get the token that was going to A"
        );
        assert_eq!(
            recorded_server(&store).as_deref(),
            Some("https://a.example/ollama")
        );

        // It is A's from now on: a restart configured for B agrees, and
        // moving back to A gets it again.
        let mut restarted = configured_for("https://b.example/ollama");
        restarted.load_secrets_from(&store, no_env);
        assert_eq!(restarted.llm.ollama_api_key, None);
        config.memory.ollama_host = "https://a.example/ollama".to_string();
        config.rebind_ollama_token_if_moved_with("https://b.example/ollama", &store, no_env);
        assert_eq!(config.llm.ollama_api_key.as_deref(), Some("legacy-token"));
    }

    #[test]
    fn a_reload_does_not_hand_a_legacy_token_to_the_edited_address() {
        // A hand edit of `config.toml` (config_watch) or `config.reload`: the
        // file now names B while the running config was sending a token
        // saved by an older build to A.
        let (dir, store) = store();
        store
            .set(keys::OLLAMA_API_KEY, "legacy-token")
            .expect("set");
        let path = dir.path().join("config.toml");
        configured_for("https://b.example/ollama")
            .save_to(&path)
            .expect("save");

        let loaded =
            Config::load_from_replacing_with(&path, "https://a.example/ollama", &store, no_env)
                .expect("load");
        assert_eq!(
            loaded.llm.ollama_api_key, None,
            "the edited address must not get the running server's token"
        );

        // The running server, however the file spells it, keeps it.
        configured_for("https://A.example:443/ollama/")
            .save_to(&path)
            .expect("save");
        let loaded =
            Config::load_from_replacing_with(&path, "https://a.example/ollama", &store, no_env)
                .expect("load");
        assert_eq!(loaded.llm.ollama_api_key.as_deref(), Some("legacy-token"));
    }

    #[test]
    fn an_unrelated_write_moves_no_token() {
        // An address that names no server matches none — but a config write
        // that leaves it as it is must not count as a move, or it would file
        // a legacy token under that non-address for good.
        let (_dir, store) = store();
        store
            .set(keys::OLLAMA_API_KEY, "legacy-token")
            .expect("set");
        let mut config = configured_for("");
        config.load_secrets_from(&store, no_env);
        config.rebind_ollama_token_if_moved_with("", &store, no_env);
        assert_eq!(config.llm.ollama_api_key.as_deref(), Some("legacy-token"));
        assert_eq!(recorded_server(&store), None, "nothing recorded");
    }

    #[cfg(unix)]
    #[test]
    fn a_legacy_token_stays_home_when_its_server_cannot_be_recorded() {
        use std::os::unix::fs::PermissionsExt;
        let (dir, store) = store();
        store
            .set(keys::OLLAMA_API_KEY, "legacy-token")
            .expect("set");
        let mut config = configured_for("https://a.example/ollama");
        config.load_secrets_from(&store, no_env);

        // A store that reads but refuses writes.
        let read_only = std::fs::Permissions::from_mode(0o555);
        std::fs::set_permissions(dir.path(), read_only).expect("chmod");
        let writable = store.set("probe", "x").is_ok();
        config.memory.ollama_host = "https://b.example/ollama".to_string();
        config.rebind_ollama_token_if_moved_with("https://a.example/ollama", &store, no_env);
        std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o755))
            .expect("chmod back");
        if writable {
            // Permissions are not enforced for this user (root): nothing to show.
            return;
        }
        assert_eq!(recorded_server(&store), None, "the write was refused");
        assert_eq!(
            config.llm.ollama_api_key, None,
            "unrecorded, the token is still A's and not B's"
        );
    }

    #[test]
    fn migrating_leaves_the_stored_token_filed_where_it_is() {
        // The copy held is the stored token itself, loaded from there. An API
        // key save elsewhere in Settings migrates every secret; it must not
        // re-file this one under the copy's address, which in a long-lived
        // GUI can be stale after a hand edit the daemon has applied.
        let (_dir, store) = store();
        store
            .set(keys::OLLAMA_API_KEY, "legacy-token")
            .expect("set");
        let mut config = configured_for("https://stale.example/ollama");
        config.llm.ollama_api_key = Some("legacy-token".to_string());
        config.migrate_secrets_to(&store).expect("migrate");
        assert_eq!(store.ollama_token().as_deref(), Some("legacy-token"));
        assert_eq!(
            recorded_server(&store),
            None,
            "not filed under the copy's address"
        );
    }

    #[test]
    fn a_token_recorded_for_a_blank_address_goes_nowhere() {
        let (_dir, store) = store();
        store.set(keys::OLLAMA_API_KEY, "token").expect("set");
        store.set(keys::OLLAMA_API_KEY_HOST, "").expect("set");
        let mut config = configured_for("http://localhost:11434");
        config.load_secrets_from(&store, no_env);
        assert_eq!(config.llm.ollama_api_key, None);
    }

    #[test]
    fn migrating_secrets_files_the_token_under_the_configured_server() {
        // The store holds another server's token; the one in memory is this
        // server's (typed in, or from the environment) and must not inherit
        // the other's record.
        let (_dir, store) = store();
        store
            .save_ollama_token("token-for-a", "https://a.example/ollama")
            .expect("save");
        let mut config = configured_for("https://b.example/ollama");
        config.llm.ollama_api_key = Some(" token-for-b ".to_string());
        config.migrate_secrets_to(&store).expect("migrate");

        assert_eq!(config.llm.ollama_api_key, None, "blanked once stored");
        assert_eq!(store.ollama_token().as_deref(), Some("token-for-b"));
        assert_eq!(
            recorded_server(&store).as_deref(),
            Some("https://b.example/ollama")
        );
    }
}
