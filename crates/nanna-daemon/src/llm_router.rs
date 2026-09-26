//! Multi-provider LLM router
//!
//! Routes model requests to the appropriate LLM client based on model name/prefix.
//! Supports fallback across multiple providers.
//! Includes health-aware model selection (stats-informed routing).

use nanna_agent::ModelStatsTracker;
use nanna_config::credentials::{SecureStore, keys};
use nanna_llm::{LlmClient, ModelInfo, ModelInfoCache, CompletionRequest, LlmError};
use serde::Serialize;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{info, debug, warn};

/// Provider identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ProviderId {
    Anthropic,
    OpenAI,
    OpenRouter,
    GitHubModels,
    Ollama,
}

impl ProviderId {
    /// Stable lowercase name, matching the provider ids the GUI uses
    /// (`anthropic`, `openai`, `openrouter`, `github`, `ollama`).
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Anthropic => "anthropic",
            Self::OpenAI => "openai",
            Self::OpenRouter => "openrouter",
            Self::GitHubModels => "github",
            Self::Ollama => "ollama",
        }
    }

    /// Explicit routing prefixes, each with the provider it names.
    ///
    /// `openrouter/` comes first: an `OpenRouter` id carries its upstream
    /// vendor after the prefix (`openrouter/anthropic/claude-haiku-4.5`), and
    /// that id must stay on `OpenRouter` rather than be read as Anthropic's.
    /// `anthropic/` and `openai/` are what the Settings summarization picker
    /// writes; the chat picker writes bare family names instead, which the
    /// family rules in [`Self::from_model`] route.
    const PREFIXES: [(&'static str, Self); 5] = [
        ("openrouter/", Self::OpenRouter),
        ("github/", Self::GitHubModels),
        ("ollama/", Self::Ollama),
        ("anthropic/", Self::Anthropic),
        ("openai/", Self::OpenAI),
    ];

    /// The explicit routing prefix `model` starts with, matched without regard
    /// to case, and the provider it names.
    fn explicit_prefix(model: &str) -> Option<(&'static str, Self)> {
        Self::PREFIXES.into_iter().find(|(prefix, _)| {
            model
                .get(..prefix.len())
                .is_some_and(|head| head.eq_ignore_ascii_case(prefix))
        })
    }

    /// Parse provider from model string prefix
    #[must_use]
    pub fn from_model(model: &str) -> Self {
        if let Some((_, provider)) = Self::explicit_prefix(model) {
            return provider;
        }
        let lower = model.to_lowercase();

        if lower.starts_with("gpt-") || lower.starts_with("o1") || lower.starts_with("o3") {
            Self::OpenAI
        } else if lower.starts_with("claude") {
            Self::Anthropic
        } else if lower.contains(':') {
            // Tag notation (e.g., "deepseek-r1:14b", "llama3.2:latest") = local Ollama model
            Self::Ollama
        } else {
            // Default to Anthropic for unknown models
            Self::Anthropic
        }
    }

    /// Strip provider prefix from model name (e.g., "ollama/deepseek-r1:14b" -> "deepseek-r1:14b")
    ///
    /// Exactly the prefix [`Self::from_model`] routed on, in the same case
    /// rule, so a spec is never sent to a provider under a name that still
    /// carries the provider's own prefix.
    #[must_use]
    pub fn strip_prefix(model: &str) -> &str {
        Self::explicit_prefix(model).map_or(model, |(prefix, _)| &model[prefix.len()..])
    }
}

/// Model health status for routing decisions
#[derive(Debug, Clone, Serialize)]
pub enum ModelHealth {
    /// Model is working normally
    Healthy,
    /// Model has elevated errors or latency but is still usable
    Degraded(String),
    /// Model is failing consistently and should be skipped
    Unhealthy(String),
    /// Model was unhealthy but is in a cooldown/recovery period
    Cooldown {
        reason: String,
        /// Timestamp (epoch ms) after which to retry
        retry_after_ms: u64,
    },
}

impl ModelHealth {
    /// Whether this model should be used for new requests
    #[must_use]
    pub fn is_usable(&self) -> bool {
        match self {
            Self::Healthy | Self::Degraded(_) => true,
            Self::Unhealthy(_) => false,
            Self::Cooldown { retry_after_ms, .. } => {
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map_or(0, crate::numeric::millis_u64);
                now >= *retry_after_ms
            }
        }
    }
}

/// Consecutive failures at which a model enters cooldown.
const COOLDOWN_FAILURES: u32 = 5;

/// When a model with `consecutive_failures` may be tried again: the last
/// failure plus an exponential backoff (30 s doubling per further failure,
/// capped at 10 min). `None` below the cooldown threshold.
///
/// Anchored to the failure, not to the moment of asking. It used to be
/// `now + backoff`, recomputed on every call, so the deadline moved with the
/// clock and never arrived — and because a model in cooldown is skipped, no
/// request could ever succeed and reset the streak. Five failures in a row
/// retired a model until the daemon restarted.
#[must_use]
pub fn cooldown_retry_after_ms(
    consecutive_failures: u32,
    last_failure_epoch_ms: u64,
) -> Option<u64> {
    if consecutive_failures < COOLDOWN_FAILURES {
        return None;
    }
    let exponent = consecutive_failures
        .saturating_sub(COOLDOWN_FAILURES)
        .min(8);
    let backoff_ms = 30_000u64.saturating_mul(1u64 << exponent).min(600_000);
    debug_assert!(
        (30_000..=600_000).contains(&backoff_ms),
        "backoff stays in its band"
    );
    Some(last_failure_epoch_ms.saturating_add(backoff_ms))
}

/// Multi-provider LLM router
pub struct LlmRouter {
    /// Available providers and their clients.
    ///
    /// Behind a lock so the provider set can be rebuilt at runtime (config
    /// reload after the user authenticates a new provider) while the router is
    /// shared as `Arc<LlmRouter>` across the daemon. Guards are held only for
    /// map access — never across an await — so a std lock suffices.
    providers: RwLock<HashMap<ProviderId, Arc<LlmClient>>>,
    /// Model info cache
    model_cache: Option<ModelInfoCache>,
    /// Shared model stats tracker for health-aware routing (set post-init)
    stats: Arc<tokio::sync::RwLock<Option<ModelStatsTracker>>>,
    /// Why a provider is absent, recorded at the rebuild that left it out.
    ///
    /// Bounded by the provider enum — at most one short sentence per variant —
    /// so this cannot grow with traffic. Replaced wholesale on every rebuild,
    /// never appended to, so a provider that comes back leaves no stale excuse
    /// behind.
    absent_reasons: RwLock<HashMap<ProviderId, String>>,
}

impl LlmRouter {
    /// Create a new router with no providers
    #[must_use]
    pub fn new() -> Self {
        let model_cache = ModelInfoCache::default_location();
        Self {
            providers: RwLock::new(HashMap::new()),
            model_cache,
            stats: Arc::new(tokio::sync::RwLock::new(None)),
            absent_reasons: RwLock::new(HashMap::new()),
        }
    }

    /// Set the stats tracker (can be called after construction, even behind Arc)
    pub async fn set_stats(&self, stats: ModelStatsTracker) {
        *self.stats.write().await = Some(stats);
    }

    fn insert_provider(&self, provider: ProviderId, client: LlmClient) {
        self.providers
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(provider, Arc::new(client));
    }

    /// Snapshot the client for one provider (guard dropped before return, so
    /// callers can await on the client freely).
    fn client_for(&self, provider: ProviderId) -> Option<Arc<LlmClient>> {
        self.providers
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&provider)
            .cloned()
    }

    /// The Ollama client chat is using right now — the configured server and
    /// its token, as of the last rebuild. Snapshotted per call, so a caller
    /// that asks again after a config reload gets the new server.
    #[must_use]
    pub fn ollama_client(&self) -> Option<Arc<LlmClient>> {
        self.client_for(ProviderId::Ollama)
    }

    /// Rebuild the provider set from resolved credentials, replacing the
    /// current map. Safe to call on a shared `Arc<LlmRouter>`; requests
    /// in flight keep the client `Arc` they already snapshotted.
    ///
    /// Returns `(added, removed)` provider ids (sorted) and logs the diff —
    /// this is the runtime path that config reload drives, so a provider the
    /// user authenticates after boot registers without a daemon restart.
    pub fn rebuild(&self, creds: &ProviderCredentials) -> (Vec<ProviderId>, Vec<ProviderId>) {
        let mut new_map: HashMap<ProviderId, Arc<LlmClient>> = HashMap::new();
        let mut reasons: HashMap<ProviderId, String> = HashMap::new();

        match &creds.anthropic {
            Some(AnthropicCredential::OAuth(token)) => {
                new_map.insert(
                    ProviderId::Anthropic,
                    Arc::new(LlmClient::anthropic_oauth(token)),
                );
            }
            Some(AnthropicCredential::ApiKey(key)) => {
                new_map.insert(ProviderId::Anthropic, Arc::new(LlmClient::anthropic(key)));
            }
            None => {
                // The one provider whose absence has a diagnosable cause the
                // user can act on. The others are absent because no key was
                // configured, which the message below already implies.
                if let Some(reason) = creds.anthropic_absent_reason.as_deref() {
                    reasons.insert(ProviderId::Anthropic, reason.to_string());
                }
            }
        }
        if let Some(ref key) = creds.openai_api_key {
            new_map.insert(ProviderId::OpenAI, Arc::new(LlmClient::openai(key)));
        }
        if let Some(ref key) = creds.openrouter_api_key {
            new_map.insert(ProviderId::OpenRouter, Arc::new(LlmClient::openrouter(key)));
        }
        if let Some(ref token) = creds.github_token {
            new_map.insert(
                ProviderId::GitHubModels,
                Arc::new(LlmClient::github_models(token)),
            );
        }
        // Ollama needs no credential: a local instance is always addressable.
        let ollama = creds.ollama_api_key.as_ref().map_or_else(
            || LlmClient::ollama(&creds.ollama_host),
            |key| LlmClient::ollama_with_key(&creds.ollama_host, key),
        );
        new_map.insert(ProviderId::Ollama, Arc::new(ollama));

        {
            let mut guard = self
                .absent_reasons
                .write()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            *guard = reasons;
        }

        let (mut added, mut removed) = {
            let mut guard = self
                .providers
                .write()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let added: Vec<ProviderId> = new_map
                .keys()
                .filter(|p| !guard.contains_key(p))
                .copied()
                .collect();
            let removed: Vec<ProviderId> = guard
                .keys()
                .filter(|p| !new_map.contains_key(p))
                .copied()
                .collect();
            *guard = new_map;
            drop(guard);
            (added, removed)
        };
        added.sort_unstable();
        removed.sort_unstable();

        if added.is_empty() && removed.is_empty() {
            debug!("LLM provider rebuild: no membership change");
        } else {
            info!(
                "LLM provider rebuild: added={:?}, removed={:?}, now={:?}",
                added,
                removed,
                self.available_providers_sorted()
            );
        }
        (added, removed)
    }

    /// Add an Anthropic provider
    #[must_use]
    pub fn with_anthropic(self, api_key: &str) -> Self {
        info!("Adding Anthropic provider to router");
        self.insert_provider(ProviderId::Anthropic, LlmClient::anthropic(api_key));
        self
    }

    /// Add an Anthropic provider with OAuth
    #[must_use]
    pub fn with_anthropic_oauth(self, oauth_token: &str) -> Self {
        info!("Adding Anthropic OAuth provider to router");
        self.insert_provider(
            ProviderId::Anthropic,
            LlmClient::anthropic_oauth(oauth_token),
        );
        self
    }

    /// Add an `OpenAI` provider
    #[must_use]
    pub fn with_openai(self, api_key: &str) -> Self {
        info!("Adding OpenAI provider to router");
        self.insert_provider(ProviderId::OpenAI, LlmClient::openai(api_key));
        self
    }

    /// Add an `OpenRouter` provider
    #[must_use]
    pub fn with_openrouter(self, api_key: &str) -> Self {
        info!("Adding OpenRouter provider to router");
        self.insert_provider(ProviderId::OpenRouter, LlmClient::openrouter(api_key));
        self
    }

    /// Add a GitHub Models provider
    #[must_use]
    pub fn with_github_models(self, token: &str) -> Self {
        info!("Adding GitHub Models provider to router");
        self.insert_provider(ProviderId::GitHubModels, LlmClient::github_models(token));
        self
    }

    /// Add an Ollama provider
    #[must_use]
    pub fn with_ollama(self, host: &str) -> Self {
        info!("Adding Ollama provider to router");
        self.insert_provider(ProviderId::Ollama, LlmClient::ollama(host));
        self
    }

    /// Add an Ollama provider with API key authentication
    #[must_use]
    pub fn with_ollama_authenticated(self, host: &str, api_key: &str) -> Self {
        info!("Adding Ollama provider to router (authenticated)");
        self.insert_provider(
            ProviderId::Ollama,
            LlmClient::ollama_with_key(host, api_key),
        );
        self
    }

    /// Check if a provider is available
    /// Why `provider` is not registered, if the last rebuild recorded a reason.
    ///
    /// `None` means either the provider IS registered or its absence was simply
    /// an unconfigured credential — nothing that needs explaining.
    #[must_use]
    pub fn absent_reason(&self, provider: ProviderId) -> Option<String> {
        self.absent_reasons
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&provider)
            .cloned()
    }

    pub fn has_provider(&self, provider: ProviderId) -> bool {
        self.providers
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains_key(&provider)
    }

    /// Get all available providers
    pub fn available_providers(&self) -> Vec<ProviderId> {
        self.providers
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .keys()
            .copied()
            .collect()
    }

    /// All available providers, sorted — for stable log lines and IPC payloads.
    pub fn available_providers_sorted(&self) -> Vec<ProviderId> {
        let mut providers = self.available_providers();
        providers.sort_unstable();
        providers
    }

    /// Check if we can handle a given model
    pub fn can_handle(&self, model: &str) -> bool {
        self.client_for(ProviderId::from_model(model)).is_some()
    }

    /// Get the client for a model
    pub fn client_for_model(&self, model: &str) -> Option<Arc<LlmClient>> {
        self.client_for(ProviderId::from_model(model))
    }

    /// Strip provider prefix from a model name.
    /// Public convenience method for use by `agent_service` and other consumers.
    /// e.g., "ollama/deepseek-r1:14b" -> "deepseek-r1:14b"
    #[must_use]
    pub fn strip_model_prefix(model: &str) -> String {
        ProviderId::strip_prefix(model).to_string()
    }

    /// The client and bare model id a summarization-model spec resolves to,
    /// through the provider map chat uses right now.
    ///
    /// An `ollama/` entry therefore gets chat's server and the token bound to
    /// it, an `anthropic/` entry chat's Anthropic credential, and so on — one
    /// grammar ([`ProviderId::from_model`]) and one set of credentials for chat
    /// and every summarizer. Snapshotted per call, so a caller that asks again
    /// after a config reload gets the rebuilt provider.
    ///
    /// # Errors
    ///
    /// Returns a sentence naming the provider and why it is absent (the
    /// rebuild's recorded reason when it kept one) when no client is
    /// registered for the spec's provider, and refuses a blank spec, which
    /// names no model.
    pub fn summarizer_client(&self, spec: &str) -> Result<(LlmClient, String), String> {
        let spec = spec.trim();
        if spec.is_empty() {
            return Err("a blank summarization entry names no model".to_string());
        }
        let provider = ProviderId::from_model(spec);
        let client = self.client_for(provider).ok_or_else(|| {
            let why = self.absent_reason(provider).unwrap_or_else(|| {
                format!("no {} credential is configured", provider.name())
            });
            format!("`{spec}` needs the {} provider, which is not available: {why}", provider.name())
        })?;
        // `LlmClient` clones share their HTTP pool; the num_ctx latch is
        // process-wide and keyed by (server, model), so a summarizer's clone
        // and chat's own client learn from each other's demotions.
        Ok(((*client).clone(), ProviderId::strip_prefix(spec).to_string()))
    }

    /// Get the primary LLM client (first available, preferring Anthropic).
    /// Used for sub-agent spawning where we need a client but don't know the model yet.
    pub fn primary_client(&self) -> Option<Arc<LlmClient>> {
        // Priority order: Anthropic > OpenAI > OpenRouter > GitHub > Ollama
        for provider in [
            ProviderId::Anthropic,
            ProviderId::OpenAI,
            ProviderId::OpenRouter,
            ProviderId::GitHubModels,
            ProviderId::Ollama,
        ] {
            if let Some(client) = self.client_for(provider) {
                return Some(client);
            }
        }
        None
    }

    /// Get model info for a model (routing to correct provider)
    pub async fn get_model_info(&self, model: &str) -> ModelInfo {
        let provider = ProviderId::from_model(model);
        let actual_model = ProviderId::strip_prefix(model);

        debug!("Getting model info for {} via {:?}", actual_model, provider);

        if let Some(client) = self.client_for(provider) {
            client.get_model_info(actual_model, self.model_cache.as_ref()).await
        } else {
            // Provider client missing: cache first, else universal floor (no name table).
            if let Some(cache) = self.model_cache.as_ref()
                && let Some(info) = cache.get(actual_model)
            {
                return info;
            }
            nanna_llm::unknown_model_info(model, &format!("{provider:?}"))
        }
    }

    /// Check the health of a model based on recent stats.
    ///
    /// Thresholds:
    /// - Unhealthy: `success_rate` < 50% with 5+ requests, or 5+ consecutive failures
    /// - Degraded: `success_rate` < 80% or avg latency > 30s
    /// - Cooldown: was unhealthy, apply exponential backoff before retry
    pub async fn model_health(&self, model: &str) -> ModelHealth {
        // A clone of the tracker shares its state, so the lock is needed only
        // to read which tracker is installed, not while summarizing.
        let tracker = self.stats.read().await.clone();
        let Some(stats) = tracker else {
            return ModelHealth::Healthy; // No stats tracker, assume healthy
        };

        let summaries = stats.summaries().await;
        let Some(summary) = summaries.iter().find(|s| s.model == model) else {
            return ModelHealth::Healthy; // No data yet
        };

        // Not enough data to judge
        if summary.total_requests < 3 {
            return ModelHealth::Healthy;
        }

        let error_rate = 1.0 - summary.success_rate;
        let total_errors = crate::numeric::u64_from_f64(
            (crate::numeric::f64_from_u64(summary.total_requests) * error_rate).round(),
        );

        // Check for consecutive failures (unhealthy → cooldown)
        if let Some(retry_after_ms) =
            cooldown_retry_after_ms(summary.consecutive_failures, summary.last_failure_epoch_ms)
        {
            return ModelHealth::Cooldown {
                reason: format!("{} consecutive failures", summary.consecutive_failures),
                retry_after_ms,
            };
        }

        // High error rate → unhealthy
        if error_rate > 0.5 && summary.total_requests >= 5 {
            return ModelHealth::Unhealthy(format!(
                "Success rate {:.0}% ({} errors in {} requests)",
                summary.success_rate * 100.0,
                total_errors,
                summary.total_requests
            ));
        }

        // Moderate error rate or high latency → degraded
        if error_rate > 0.2 {
            return ModelHealth::Degraded(format!(
                "Success rate {:.0}% ({} errors in {} requests)",
                summary.success_rate * 100.0,
                total_errors,
                summary.total_requests
            ));
        }

        if summary.avg_latency_ms > 30_000 {
            return ModelHealth::Degraded(format!(
                "High latency: {:.1}s avg",
                crate::numeric::f64_from_u64(summary.avg_latency_ms) / 1000.0
            ));
        }

        ModelHealth::Healthy
    }

    /// Filter a model priority list to prefer healthy models.
    ///
    /// Returns the reordered list: healthy models first (in original order),
    /// then degraded models, then cooldown-eligible models. Unhealthy models
    /// and models still in cooldown are excluded.
    pub async fn health_sorted_models(&self, models: &[String]) -> Vec<String> {
        let mut healthy = Vec::new();
        let mut degraded = Vec::new();
        let mut cooldown_ready = Vec::new();

        for model in models {
            let health = self.model_health(model).await;
            match health {
                ModelHealth::Healthy => healthy.push(model.clone()),
                ModelHealth::Degraded(_) => degraded.push(model.clone()),
                ModelHealth::Cooldown { .. } if health.is_usable() => {
                    cooldown_ready.push(model.clone());
                }
                ModelHealth::Cooldown { reason, .. } => {
                    debug!("Skipping model {} (cooldown: {})", model, reason);
                }
                ModelHealth::Unhealthy(reason) => {
                    warn!("Skipping unhealthy model {}: {}", model, reason);
                }
            }
        }

        // If all models are unhealthy, fall back to the original list
        // (better to try something than give up entirely)
        if healthy.is_empty() && degraded.is_empty() && cooldown_ready.is_empty() {
            warn!("All models unhealthy, falling back to original priority list");
            return models.to_vec();
        }

        let mut result = healthy;
        result.extend(degraded);
        result.extend(cooldown_ready);
        result
    }

    /// Complete a request (routing to correct provider)
    ///
    /// # Errors
    ///
    /// Returns [`LlmError::MissingApiKey`] when no client is registered for the
    /// model's provider, and otherwise whatever the provider's completion call
    /// returns.
    pub async fn complete(&self, model: &str, request: CompletionRequest) -> Result<String, LlmError> {
        let provider = ProviderId::from_model(model);
        let actual_model = ProviderId::strip_prefix(model);

        debug!("Routing completion for {} to {:?}", actual_model, provider);

        let client = self
            .client_for(provider)
            .ok_or_else(|| LlmError::MissingApiKey(format!("{provider:?}")))?;

        // Update model in request
        let mut request = request;
        request.model = actual_model.to_string();

        client.complete(&request).await
    }
}

impl Default for LlmRouter {
    fn default() -> Self {
        Self::new()
    }
}

/// The resolver every agent this process builds summarizes through: each
/// `summarization_priority` entry becomes a client by
/// [`LlmRouter::summarizer_client`] on `router`, asked afresh on every use.
///
/// Holding the router rather than a client is the point. A config reload
/// rebuilds the router's providers in place, so a new `[memory].ollama_host`,
/// a newly saved token or key reaches the very next summarization — even in a
/// turn that started before the change — with no restart.
#[must_use]
pub fn summarizer_clients(router: &Arc<LlmRouter>) -> nanna_agent::SummarizerClients {
    let router = Arc::clone(router);
    nanna_agent::SummarizerClients::new(move |spec| router.summarizer_client(spec))
}

/// The one resolved credential the Anthropic provider will use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnthropicCredential {
    /// OAuth access token (Claude Pro/Max via `claude setup-token` / CLI login)
    OAuth(String),
    /// Plain API key
    ApiKey(String),
}

/// Credentials resolved for provider construction.
///
/// Resolution (config → keyring → Claude CLI) is separated from client
/// construction ([`LlmRouter::rebuild`]) so construction is deterministic and
/// unit-testable, and so boot and config-reload share one chain instead of the
/// reload silently skipping registration (the boot-only split-brain this
/// module used to have).
#[derive(Debug, Clone)]
pub struct ProviderCredentials {
    pub anthropic: Option<AnthropicCredential>,
    /// Why `anthropic` is `None`, in one non-secret sentence.
    ///
    /// The resolution chain already knows this — it logs it once, at WARN, at
    /// boot — and then throws it away. Every later request then fails with
    /// `No provider for model: claude-sonnet-5 (detected: Anthropic, available:
    /// [Ollama])`, which names the model and the provider list and not the
    /// cause, so it reads as "your model name is wrong". Keeping the sentence
    /// is what lets the failure say what actually happened.
    pub anthropic_absent_reason: Option<String>,
    pub openai_api_key: Option<String>,
    pub openrouter_api_key: Option<String>,
    pub github_token: Option<String>,
    pub ollama_host: String,
    pub ollama_api_key: Option<String>,
}

/// `Some(trimmed)` only for a non-blank value — a `Some("")` credential must
/// not register a provider that then fails every call.
fn non_empty(value: Option<&String>) -> Option<String> {
    value
        .map(|v| v.trim())
        .filter(|v| !v.is_empty())
        .map(String::from)
}

/// Keyring lookup with the same non-blank rule as config values.
fn store_credential(store: &SecureStore, key: &str) -> Option<String> {
    store.get(key).ok().and_then(|v| non_empty(Some(&v)))
}

/// Longest absence reason worth carrying.
///
/// The reason embeds a provider's own error body — third-party text of
/// unbounded length — and it reaches a log line, `last_error`, and from there
/// the user. Bounding it is the Tiger-Style rule, and the bound is derived, not
/// magic: 240 bytes is a little under two 132-column terminal lines, which is
/// what a cause sentence gets before it stops being read.
const ABSENT_REASON_BYTES_MAX: usize = 240;

/// Truncate on a char boundary, never mid-codepoint — a `&s[..n]` here would
/// panic on the first non-ASCII byte a provider ever returns.
fn bounded_reason(reason: String) -> String {
    if reason.len() <= ABSENT_REASON_BYTES_MAX {
        return reason;
    }
    let end = reason.floor_char_boundary(ABSENT_REASON_BYTES_MAX);
    debug_assert!(end <= reason.len(), "a boundary cannot exceed the string");
    debug_assert!(reason.is_char_boundary(end), "must cut on a char boundary");
    format!("{}…", &reason[..end])
}

/// Resolve the Anthropic credential.
///
/// OAuth mode: env override (`ANTHROPIC_OAUTH_TOKEN`, matching
/// `Config::load_secrets_from_store` precedence) → durable stores via
/// [`nanna_config::resolve_anthropic_oauth`] (envelope-aware: refreshes a
/// stale token and mirrors Claude CLI logins into the nanna store, instead
/// of registering a provider whose every call 401s) → config-held token →
/// fall through to an API key. Non-OAuth mode: config/keyring API key first,
/// then any durable OAuth credential (nanna store or Claude CLI login).
async fn resolve_anthropic(
    llm: &crate::server::LlmConfig,
    store: &SecureStore,
) -> (Option<AnthropicCredential>, Option<String>) {
    // Set by whichever step got closest to a credential; reported only if every
    // step fails. `_` prefixed nothing: the last failure is the informative one.
    let mut why_not: Option<String> = None;
    if llm.anthropic_use_oauth {
        if let Ok(v) = std::env::var("ANTHROPIC_OAUTH_TOKEN")
            && let Some(token) = non_empty(Some(&v))
        {
            debug!("Anthropic credential: OAuth token from env");
            return (Some(AnthropicCredential::OAuth(token)), None);
        }
        match nanna_config::resolve_anthropic_oauth(store).await {
            Ok(cred) => {
                debug!("Anthropic credential: OAuth from durable store (refreshed if stale)");
                return (Some(AnthropicCredential::OAuth(cred.access_token)), None);
            }
            Err(e) => {
                if let Some(token) = non_empty(llm.anthropic_oauth_token.as_ref()) {
                    warn!("OAuth resolution failed ({e}); using config-provided token");
                    return (Some(AnthropicCredential::OAuth(token)), None);
                }
                warn!("Anthropic OAuth enabled but no usable credential ({e}); trying API key");
                why_not = Some(format!("the stored OAuth credential is unusable ({e})"));
            }
        }
    }
    if let Some(key) = non_empty(llm.anthropic_api_key.as_ref())
        .or_else(|| store_credential(store, keys::ANTHROPIC_API_KEY))
    {
        debug!("Anthropic credential: API key from config/keyring");
        return (Some(AnthropicCredential::ApiKey(key)), None);
    }
    if !llm.anthropic_use_oauth {
        // The error here is the informative one for the common case: a stored
        // login DOES exist and could not be made usable. Discarding it and
        // falling through to "nothing is configured" is how an expired token
        // came to be reported as a missing one.
        match nanna_config::resolve_anthropic_oauth(store).await {
            Ok(cred) => {
                debug!("Anthropic credential: durable OAuth fallback (nanna store / Claude CLI)");
                return (Some(AnthropicCredential::OAuth(cred.access_token)), None);
            }
            Err(e) => {
                why_not.get_or_insert_with(|| {
                    format!("a stored OAuth login exists but could not be used ({e})")
                });
            }
        }
    }
    (
        None,
        Some(bounded_reason(why_not.unwrap_or_else(|| {
            "no Anthropic credential is configured (no API key, and no stored OAuth login)"
                .to_string()
        }))),
    )
}

impl ProviderCredentials {
    /// Resolve provider credentials from the daemon's LLM config, falling back
    /// to the OS keyring and finally the Claude CLI's own credentials.
    ///
    /// Per provider, first source wins:
    /// - Anthropic: see `resolve_anthropic` — OAuth env/durable-store/config
    ///   chain (refreshing stale tokens) → config API key → keyring API key →
    ///   durable OAuth fallback. An enabled OAuth flag with a missing token
    ///   falls through — the boot chain used to dead-end there, registering no
    ///   Anthropic provider even though the CLI held valid credentials.
    /// - `OpenAI` / `OpenRouter` / GitHub: config key → keyring key.
    /// - Ollama: always present (host needs no credential; blank key = anonymous).
    pub async fn resolve(llm: &crate::server::LlmConfig) -> Self {
        let store = SecureStore::new();

        let (anthropic, anthropic_absent_reason) = resolve_anthropic(llm, &store).await;
        debug_assert!(
            anthropic.is_none() || anthropic_absent_reason.is_none(),
            "a resolved credential must not also carry a reason for its absence"
        );

        Self {
            anthropic,
            anthropic_absent_reason,
            openai_api_key: non_empty(llm.openai_api_key.as_ref())
                .or_else(|| store_credential(&store, keys::OPENAI_API_KEY)),
            openrouter_api_key: non_empty(llm.openrouter_api_key.as_ref())
                .or_else(|| store_credential(&store, keys::OPENROUTER_API_KEY)),
            github_token: non_empty(llm.github_token.as_ref())
                .or_else(|| store_credential(&store, keys::GITHUB_TOKEN)),
            ollama_host: llm.ollama_host.clone(),
            ollama_api_key: non_empty(llm.ollama_api_key.as_ref()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AnthropicCredential, LlmRouter, ModelHealth, ProviderCredentials, ProviderId,
        cooldown_retry_after_ms,
    };

    #[test]
    fn a_cooldown_is_measured_from_the_last_failure() {
        assert_eq!(
            cooldown_retry_after_ms(4, 1_000),
            None,
            "below the threshold"
        );
        assert_eq!(cooldown_retry_after_ms(5, 1_000), Some(31_000));
        assert_eq!(cooldown_retry_after_ms(6, 1_000), Some(61_000), "doubles");
        assert_eq!(
            cooldown_retry_after_ms(40, 1_000),
            Some(601_000),
            "capped at 10 min"
        );
        // A failure long enough ago is a cooldown that has ended.
        let ended = ModelHealth::Cooldown {
            reason: "old".to_string(),
            retry_after_ms: cooldown_retry_after_ms(5, 1_000).expect("in cooldown"),
        };
        assert!(
            ended.is_usable(),
            "the model is tried again once the backoff has passed"
        );
    }

    /// The live bug: `retry_after` was `now + backoff`, recomputed on every
    /// call, so asking twice gave two different deadlines and neither ever
    /// arrived. Five failures retired a model until restart.
    #[tokio::test]
    async fn a_cooldown_deadline_does_not_move_between_calls() {
        let tracker = nanna_agent::ModelStatsTracker::new();
        for _ in 0..5 {
            tracker
                .record(nanna_agent::RequestObservation {
                    model: "flaky".to_string(),
                    success: false,
                    latency: std::time::Duration::from_millis(5),
                    input_tokens: 0,
                    output_tokens: 0,
                    cache_read_tokens: 0,
                    cache_creation_tokens: 0,
                    cache_creation_1h_tokens: 0,
                    tier: None,
                    escalated: false,
                })
                .await;
        }
        let router = LlmRouter::new();
        router.set_stats(tracker).await;

        let deadline = |health: ModelHealth| match health {
            ModelHealth::Cooldown { retry_after_ms, .. } => retry_after_ms,
            other => panic!("five failures must be a cooldown, got {other:?}"),
        };
        let first = deadline(router.model_health("flaky").await);
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        let second = deadline(router.model_health("flaky").await);
        assert_eq!(first, second, "the deadline is a fixed point in time");
    }

    /// The live failure this guards: a provider authenticated after boot must
    /// register on rebuild (no daemon restart), and one whose credential goes
    /// away must drop — with the untouched providers staying registered.
    #[test]
    fn rebuild_registers_and_removes_providers_at_runtime() {
        let router = LlmRouter::new();
        assert!(!router.has_provider(ProviderId::Anthropic));
        assert!(!router.can_handle("claude-fable-5"));

        let creds = ProviderCredentials {
            anthropic: Some(AnthropicCredential::OAuth("test-token".into())),
            anthropic_absent_reason: None,
            openai_api_key: None,
            openrouter_api_key: Some("sk-or-test".into()),
            github_token: None,
            ollama_host: "http://localhost:11434".into(),
            ollama_api_key: None,
        };
        let (added, removed) = router.rebuild(&creds);
        assert_eq!(
            added,
            vec![
                ProviderId::Anthropic,
                ProviderId::OpenRouter,
                ProviderId::Ollama
            ]
        );
        assert_eq!(removed, Vec::<ProviderId>::new());
        // The exact live symptom: a bare Claude model name must now route.
        assert!(router.can_handle("claude-fable-5"));

        assert_eq!(
            router.absent_reason(ProviderId::Anthropic),
            None,
            "a registered provider has nothing to explain"
        );

        let creds = ProviderCredentials {
            anthropic: None,
            anthropic_absent_reason: Some("the stored OAuth credential is unusable".into()),
            ..creds
        };
        let (added, removed) = router.rebuild(&creds);
        assert_eq!(added, Vec::<ProviderId>::new());
        assert_eq!(removed, vec![ProviderId::Anthropic]);
        assert!(!router.can_handle("claude-fable-5"));
        assert!(router.has_provider(ProviderId::OpenRouter));
        assert!(router.has_provider(ProviderId::Ollama));
        assert_eq!(
            router.absent_reason(ProviderId::Anthropic).as_deref(),
            Some("the stored OAuth credential is unusable"),
            "the cause must survive the rebuild that dropped the provider"
        );
        // A provider that is simply not configured needs no excuse.
        assert_eq!(router.absent_reason(ProviderId::OpenAI), None);
    }

    /// A provider that comes back must leave no stale excuse behind — the
    /// reasons map is replaced by each rebuild, never appended to.
    #[test]
    fn an_absence_reason_does_not_outlive_the_absence() {
        let router = LlmRouter::new();
        let absent = ProviderCredentials {
            anthropic: None,
            anthropic_absent_reason: Some("expired 54h ago".into()),
            openai_api_key: None,
            openrouter_api_key: None,
            github_token: None,
            ollama_host: "http://localhost:11434".into(),
            ollama_api_key: None,
        };
        router.rebuild(&absent);
        assert_eq!(
            router.absent_reason(ProviderId::Anthropic).as_deref(),
            Some("expired 54h ago")
        );

        let restored = ProviderCredentials {
            anthropic: Some(AnthropicCredential::ApiKey("sk-test".into())),
            anthropic_absent_reason: None,
            ..absent
        };
        router.rebuild(&restored);
        assert!(router.has_provider(ProviderId::Anthropic));
        assert_eq!(
            router.absent_reason(ProviderId::Anthropic),
            None,
            "the old excuse must be gone once the provider is back"
        );
    }

    /// A provider's error body is third-party text of unbounded length, and it
    /// reaches a log line and the user. It is bounded, and bounded on a CHAR
    /// boundary — the repo has panicked on `&s[..n]` before.
    #[test]
    fn an_absence_reason_is_bounded_without_splitting_a_codepoint() {
        let short = "expired 54h ago".to_string();
        assert_eq!(super::bounded_reason(short.clone()), short);

        // Multibyte right up to the cut: every char is 3 bytes, so the limit
        // lands mid-codepoint unless the boundary is respected.
        let wide: String = "字".repeat(400);
        let bounded = super::bounded_reason(wide);
        assert!(bounded.len() <= super::ABSENT_REASON_BYTES_MAX + "…".len());
        assert!(bounded.ends_with('…'));
        // The real assertion: it is still valid UTF-8 with no replacement char.
        assert!(!bounded.contains('\u{fffd}'));
    }

    #[test]
    fn blank_credentials_never_count() {
        // A Some("") key must not register a provider that 401s on every call.
        assert_eq!(super::non_empty(Some(&"   ".to_string())), None);
        assert_eq!(super::non_empty(None), None);
        assert_eq!(
            super::non_empty(Some(&" sk-x ".to_string())),
            Some("sk-x".to_string())
        );
    }

    #[test]
    fn from_model_infers_provider_by_prefix() {
        // Bare model names route by family prefix (this is exactly what the GUI's
        // parse_model_id historically got wrong for OpenAI models).
        assert_eq!(ProviderId::from_model("gpt-4o"), ProviderId::OpenAI);
        assert_eq!(ProviderId::from_model("gpt-5"), ProviderId::OpenAI);
        assert_eq!(ProviderId::from_model("o1-preview"), ProviderId::OpenAI);
        assert_eq!(ProviderId::from_model("o3-mini"), ProviderId::OpenAI);
        assert_eq!(
            ProviderId::from_model("claude-opus-4"),
            ProviderId::Anthropic
        );
        // Case-insensitive.
        assert_eq!(
            ProviderId::from_model("Claude-Sonnet-4"),
            ProviderId::Anthropic
        );
    }

    #[test]
    fn from_model_recognizes_explicit_and_tagged_prefixes() {
        assert_eq!(
            ProviderId::from_model("openrouter/meta-llama/llama-3"),
            ProviderId::OpenRouter
        );
        assert_eq!(
            ProviderId::from_model("github/gpt-4o"),
            ProviderId::GitHubModels
        );
        assert_eq!(ProviderId::from_model("ollama/qwen3"), ProviderId::Ollama);
        // A `:tag` (with no known prefix) is a local Ollama model.
        assert_eq!(
            ProviderId::from_model("deepseek-r1:14b"),
            ProviderId::Ollama
        );
        assert_eq!(
            ProviderId::from_model("llama3.2:latest"),
            ProviderId::Ollama
        );
        // Unknown, prefix-less models fall back to Anthropic.
        assert_eq!(
            ProviderId::from_model("some-unknown-model"),
            ProviderId::Anthropic
        );
    }

    #[test]
    fn strip_prefix_removes_only_routing_prefixes() {
        assert_eq!(
            ProviderId::strip_prefix("ollama/deepseek-r1:14b"),
            "deepseek-r1:14b"
        );
        assert_eq!(ProviderId::strip_prefix("openrouter/x/y"), "x/y");
        assert_eq!(ProviderId::strip_prefix("github/gpt-4o"), "gpt-4o");
        // Family-named models keep their name (the family IS the model id).
        assert_eq!(ProviderId::strip_prefix("gpt-4o"), "gpt-4o");
        assert_eq!(ProviderId::strip_prefix("claude-opus-4"), "claude-opus-4");
    }

    /// The Settings summarization picker writes `anthropic/<id>` and
    /// `openai/<id>`. Unknown to the router, both went to the Anthropic client
    /// with the prefix still on: `openai/gpt-4o-mini` was sent to Anthropic,
    /// and `anthropic/claude-haiku-4-5` named a model Anthropic does not have.
    /// Every dream cycle, IPC consolidation and `memory.summarize` call walked
    /// past such an entry as a failure.
    #[test]
    fn the_settings_pickers_provider_prefixes_route_and_strip() {
        assert_eq!(
            ProviderId::from_model("anthropic/claude-haiku-4-5"),
            ProviderId::Anthropic
        );
        assert_eq!(
            ProviderId::strip_prefix("anthropic/claude-haiku-4-5"),
            "claude-haiku-4-5"
        );
        assert_eq!(
            ProviderId::from_model("openai/gpt-4o-mini"),
            ProviderId::OpenAI
        );
        assert_eq!(ProviderId::strip_prefix("openai/gpt-4o-mini"), "gpt-4o-mini");

        // OpenRouter ids carry the upstream vendor after the router prefix;
        // `openrouter/` is matched first, so they stay on OpenRouter and keep
        // the vendor half of the id.
        assert_eq!(
            ProviderId::from_model("openrouter/anthropic/claude-haiku-4.5"),
            ProviderId::OpenRouter
        );
        assert_eq!(
            ProviderId::strip_prefix("openrouter/anthropic/claude-haiku-4.5"),
            "anthropic/claude-haiku-4.5"
        );
        assert_eq!(
            ProviderId::from_model("openrouter/openai/gpt-4o-mini"),
            ProviderId::OpenRouter
        );
    }

    /// A summarizer spec resolves through the provider map chat uses, to the
    /// bare id that provider knows; an absent provider is explained with the
    /// reason the rebuild recorded, not reported as a bad model name.
    #[test]
    fn a_summarizer_spec_resolves_like_chat_and_explains_an_absence() {
        let router = LlmRouter::new();
        router.rebuild(&ProviderCredentials {
            anthropic: None,
            anthropic_absent_reason: Some("the stored OAuth credential expired 4h ago".into()),
            openai_api_key: Some("sk-test".into()),
            openrouter_api_key: None,
            github_token: None,
            ollama_host: "http://gpu-box:11434".into(),
            ollama_api_key: None,
        });

        let (client, model) = router
            .summarizer_client("ollama/qwen3:4b")
            .expect("Ollama always registers");
        assert_eq!(model, "qwen3:4b");
        assert_eq!(client.base_url(), "http://gpu-box:11434");
        assert_eq!(
            router.summarizer_client(" openai/gpt-4o-mini ").map(|(_, m)| m),
            Ok("gpt-4o-mini".to_string())
        );

        let Err(err) = router.summarizer_client("anthropic/claude-haiku-4-5") else {
            panic!("no Anthropic credential is registered");
        };
        assert!(err.contains("the stored OAuth credential expired 4h ago"), "{err}");
        let Err(err) = router.summarizer_client("openrouter/meta-llama/llama-3") else {
            panic!("no OpenRouter key is registered");
        };
        assert!(err.contains("openrouter"), "{err}");
        assert!(router.summarizer_client("  ").is_err(), "a blank entry names no model");
    }

    /// `from_model` has always matched prefixes without regard to case, and
    /// `strip_prefix` did not: `Ollama/qwen3:4b` went to Ollama under the name
    /// `Ollama/qwen3:4b`, which no server has. One grammar means the two agree.
    #[test]
    fn a_prefix_strips_in_whatever_case_it_routes_in() {
        for (spec, provider, bare) in [
            ("Ollama/qwen3:4b", ProviderId::Ollama, "qwen3:4b"),
            ("Anthropic/claude-haiku-4-5", ProviderId::Anthropic, "claude-haiku-4-5"),
            ("OPENAI/gpt-4o-mini", ProviderId::OpenAI, "gpt-4o-mini"),
            ("OpenRouter/meta-llama/llama-3", ProviderId::OpenRouter, "meta-llama/llama-3"),
            ("GitHub/gpt-4o", ProviderId::GitHubModels, "gpt-4o"),
        ] {
            assert_eq!(ProviderId::from_model(spec), provider, "{spec}");
            assert_eq!(ProviderId::strip_prefix(spec), bare, "{spec}");
        }
    }
}
