#![warn(clippy::all)]
#![warn(clippy::pedantic, clippy::nursery)]

//! LLM API client for Nanna
//!
//! Supports Anthropic Claude with proper tool calling and streaming.

use tracing::{debug, info, warn};

#[cfg(feature = "auto-refresh")]
pub mod oauth;

#[cfg(feature = "auto-refresh")]
pub use oauth::{OAuthClient, create_oauth_client, create_oauth_client_sync};

pub mod ollama_probe;
pub use ollama_probe::{OllamaModel, OllamaProbe, probe_ollama};

pub mod heal;
pub use heal::{
    count_balanced_top_level_objects, escape_bare_controls_in_strings, heal_json, heal_json_as,
    heal_tool_args,
};

use async_stream::stream;
use futures::{Stream, StreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// A provider-reported token count as the `u32` the usage events carry,
/// saturating: no single request reports four billion tokens, so this never
/// differs from the old truncating cast in practice.
fn token_count_u32(n: u64) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

/// Watches a token stream for a wedged runner emitting one token forever.
///
/// Observed live 2026-07-27: thinking blocks of `"00000000000000…"` filling
/// the GUI, and stream aborts landing on an identical 4306-byte / 31-line
/// shape 46 times in one night — the same fault seen from two directions.
/// Each occurrence cost a step's worth of GPU time and produced nothing.
///
/// Bounds, both derived from what real generation looks like rather than
/// picked for feel:
/// - Only fragments of **≤ 4 characters** are candidates. A sampler that
///   wedges repeats a single token; a legitimately repeated long fragment is
///   content, not a fault.
/// - **40** consecutive identical fragments trips it. Real text repeats a
///   short token a handful of times (indentation runs, `...`, a rule of
///   dashes); it does not do so forty times without variation, while a
///   wedged runner reaches forty almost immediately.
#[derive(Debug, Default)]
struct RepeatWatch {
    last: Option<String>,
    run: usize,
}

impl RepeatWatch {
    /// Trip point, measured against the failure it exists to catch rather
    /// than chosen for feel: the observed degenerate streams die at **31**
    /// NDJSON lines (~30 emitted fragments), so a bound of 40 could never
    /// fire — it was set by reasoning about legitimate text alone, and the
    /// evidence for the real number was already in the logs. 20 sits below
    /// the failure and far above ordinary repetition, which cannot reach it
    /// because any non-candidate fragment resets the run.
    const MAX_RUN: usize = 20;
    const MAX_FRAGMENT_CHARS: usize = 4;

    /// Record a fragment; returns the run length once it is degenerate.
    fn observe(&mut self, fragment: &str) -> Option<usize> {
        if fragment.trim().is_empty() || fragment.chars().count() > Self::MAX_FRAGMENT_CHARS {
            // Not a candidate: any real content resets the watch, so a burst
            // of indentation followed by prose never accumulates.
            self.last = None;
            self.run = 0;
            return None;
        }
        if self.last.as_deref() == Some(fragment) {
            self.run += 1;
        } else {
            self.last = Some(fragment.to_string());
            self.run = 1;
        }
        (self.run >= Self::MAX_RUN).then_some(self.run)
    }
}

/// Whether an Ollama base URL names a server on THIS machine.
///
/// Two things are only right for a server on this machine: sizing `num_ctx`
/// from this machine's GPU, and the healing ladder's process-level cures
/// (unloading a runner, killing and respawning the server). A server reached
/// over the network has a GPU and processes of its own.
///
/// Parsed, not substring-matched: `https://localhost.example.com` and
/// `https://host/127.0.0.1` are remote. Local is `localhost`, the IPv4
/// loopback block, `::1`, or an unspecified address (`0.0.0.0`, `::`), which
/// reaches this machine when dialled. Anything else, including anything that
/// does not parse, reads as remote — the safe way to be wrong: a local server
/// misread as remote loses a cure, while the reverse kills this machine's
/// processes for a server they do not serve.
#[must_use]
pub fn ollama_server_is_local(base_url: &str) -> bool {
    let Ok(url) = reqwest::Url::parse(base_url.trim()) else {
        return false;
    };
    let Some(host) = url.host_str() else {
        return false;
    };
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    let unbracketed = host
        .strip_prefix('[')
        .and_then(|rest| rest.strip_suffix(']'))
        .unwrap_or(host);
    unbracketed
        .parse::<std::net::IpAddr>()
        .map(|ip| ip.to_canonical())
        .is_ok_and(|ip| ip.is_loopback() || ip.is_unspecified())
}

/// The single generation slot a local Ollama server hands out, as a permit.
///
/// llama.cpp serves one generation at a time per slot. A second concurrent
/// request does **not** queue politely behind the first: the server re-selects
/// the slot, swaps prompt caches (`found better prompt` / `updating prompt
/// cache`) and *cancels* the running task (`srv stop: cancel task`). The
/// victim's HTTP stream then just ends — no `done: true` — which this client
/// can only report as a provider incident, so the harness burns retry budget
/// healing damage the daemon did to itself. Observed live 2026-07-26: a
/// scheduled heartbeat killed a chat turn's generation three times running,
/// each time ~4.3 KB / 31 NDJSON lines in.
///
/// One permit per base URL, so two genuinely separate servers keep separate
/// slots and a single server is never asked to interleave. Waiting is the
/// correct behaviour here — the alternative is not "more throughput" but two
/// mutually-cancelling generations.
///
/// Scope: chat/completion generations only. Embeddings hit a different
/// endpoint and are left alone.
fn ollama_generation_slot(base_url: &str) -> std::sync::Arc<tokio::sync::Semaphore> {
    static SLOTS: std::sync::OnceLock<
        std::sync::Mutex<
            std::collections::HashMap<String, std::sync::Arc<tokio::sync::Semaphore>>,
        >,
    > = std::sync::OnceLock::new();
    let slots = SLOTS.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    // A poisoned lock carries no invalid state here (the map is only ever
    // inserted into), so recover rather than propagate a panic into the
    // request path.
    let mut guard = slots.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    guard
        .entry(base_url.to_string())
        .or_insert_with(|| std::sync::Arc::new(tokio::sync::Semaphore::new(1)))
        .clone()
}

#[derive(Error, Debug, Clone)]
pub enum LlmError {
    #[error("HTTP error: {0}")]
    Http(String),
    #[error("API error: {status} - {message}")]
    Api { status: u16, message: String },
    #[error("JSON error: {0}")]
    Json(String),
    #[error("Stream error: {0}")]
    Stream(String),
    #[error("Missing API key for provider: {0}")]
    MissingApiKey(String),
    #[error("Rate limit exceeded: {message}")]
    RateLimit {
        message: String,
        /// Seconds until rate limit resets (if available)
        retry_after: Option<u64>,
    },
    /// A local runner stuck emitting one token forever, caught by
    /// `RepeatWatch`.
    ///
    /// Structured rather than folded into [`LlmError::Api`] because the retry
    /// ladder has to tell "this is the same wedge I just hit" from "one
    /// unlucky generation", and that decision reads the token and the abort
    /// length. Those were only ever present as prose inside the message, so
    /// the caller's choice depended on re-parsing an error string — a
    /// contract nothing enforced and any rewording would have broken.
    ///
    /// The rendered form is byte-identical to the 502 this used to be
    /// (`from_api_response(502, …)`). That is load-bearing, not cosmetic:
    /// `is_transient_llm_error` keys off `"API error: 5"` and
    /// `wedged_runner_error` off `"same token"`, on both the harness and the
    /// chat path. A variant that read differently would drop the wedge out of
    /// the transient class and hard-fail the step instead of retrying it.
    #[error(
        "API error: 502 - Ollama emitted the same token {run}x in a row ({token:?}) — \
         a wedged runner, not a generation. Aborted after {bytes_received} bytes so \
         the step can be retried instead of waiting out the loop."
    )]
    WedgedRunner {
        /// The fragment the decoder was stuck on.
        token: String,
        /// Consecutive repeats observed when the watch tripped.
        run: usize,
        /// Stream bytes received before the abort.
        bytes_received: usize,
    },
    #[error("All fallback models exhausted")]
    AllModelsExhausted,
    #[error("IO error: {0}")]
    Io(String),
}

impl From<std::io::Error> for LlmError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e.to_string())
    }
}

// ============================================================================
// Model Information & Capabilities
// ============================================================================

/// Model capabilities and limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model identifier
    pub id: String,
    /// Maximum input context window in tokens
    pub context_window: usize,
    /// Maximum output tokens
    pub max_output_tokens: usize,
    /// Whether the model supports tool/function calling
    #[serde(default)]
    pub supports_tools: bool,
    /// Whether the model supports vision/images
    #[serde(default)]
    pub supports_vision: bool,
    /// Embedding dimension (if this is an embedding model)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_dimension: Option<usize>,
    /// Unix timestamp when this info was cached
    #[serde(default)]
    pub cached_at: i64,
    /// Provider this info came from
    #[serde(default)]
    pub provider: String,
}

impl ModelInfo {
    /// Cache TTL: 1 week in seconds
    pub const CACHE_TTL_SECS: i64 = 7 * 24 * 60 * 60;

    /// Check if this cached info has expired
    #[must_use]
    pub fn is_expired(&self) -> bool {
        let now = current_timestamp();
        now - self.cached_at > Self::CACHE_TTL_SECS
    }

    /// Recommended compression threshold — where the context manager should
    /// *proactively* compress, always **below** [`Self::hard_input_limit`].
    ///
    /// Nominally 80% of the context window, but a small model with a large
    /// `max_output_tokens` (e.g. context 8k / output 4k) makes 80%-of-context
    /// *exceed* the hard input limit, so proactive compression would never fire
    /// before the hard cap and the agent would emergency-truncate every turn.
    /// Capping at 90% of the hard limit restores the invariant while leaving
    /// large models (where 80%-of-context is already the smaller value) unchanged.
    #[must_use]
    pub fn compression_threshold(&self) -> usize {
        let by_context = (self.context_window * 80) / 100;
        let below_hard = (self.hard_input_limit() * 90) / 100;
        let threshold = by_context.min(below_hard);
        debug_assert!(
            threshold <= self.hard_input_limit(),
            "compression must trigger before the hard input cap"
        );
        threshold
    }

    /// Compression threshold paired with a concrete output reservation —
    /// same 80%-of-context / 90%-of-hard-limit rule as
    /// [`Self::compression_threshold`], but tracking
    /// [`Self::hard_input_limit_for`] so proactive summarization still fires
    /// before the (now larger) dynamic hard cap.
    #[must_use]
    pub fn compression_threshold_for(&self, output_reserve_tokens: usize) -> usize {
        let by_context = (self.context_window * 80) / 100;
        let below_hard = (self.hard_input_limit_for(output_reserve_tokens) * 90) / 100;
        let threshold = by_context.min(below_hard);
        debug_assert!(
            threshold <= self.hard_input_limit_for(output_reserve_tokens),
            "compression must trigger before the hard input cap"
        );
        threshold
    }

    /// Hard limit for input (leaves room for output), static variant.
    ///
    /// Reserves `min(max_output_tokens, context/4)` — several providers
    /// (Ollama among them) report `max_output_tokens >= context_window`, and
    /// subtracting that claim verbatim halved a 32k local model to 16k of
    /// usable input (observed live: the user's own prompt was
    /// emergency-truncated). Callers that know their per-request `max_tokens`
    /// should prefer [`Self::hard_input_limit_for`] with
    /// [`Self::effective_output_budget`] — the reserve then shrinks to what
    /// the request can actually generate, freeing the rest for input.
    #[must_use]
    pub fn hard_input_limit(&self) -> usize {
        self.hard_input_limit_for(self.max_output_tokens.min(self.context_window / 4))
    }

    /// Hard input limit paired with a concrete output reservation.
    ///
    /// The reserve is capped at half the window (output must not starve
    /// input) and the result is floored at half the window (a degenerate
    /// reserve must not zero the input side) — so this never returns 0 and
    /// `input + reserve` never exceeds `context_window`.
    #[must_use]
    pub fn hard_input_limit_for(&self, output_reserve_tokens: usize) -> usize {
        // Windows below 2 tokens are out of contract (no provider reports
        // one; the universal floor is 32k) — the pairing invariant needs
        // ctx/2 to be nonzero.
        debug_assert!(self.context_window >= 2, "context window must be >= 2");
        let reserve = output_reserve_tokens.min(self.context_window / 2);
        let floor = self.context_window / 2;
        self.context_window.saturating_sub(reserve).max(floor)
    }

    /// The output-token budget a request should actually use: the caller's
    /// `max_tokens`, bounded by the model's reported maximum and by half the
    /// window. Pairing a request's `max_tokens` with
    /// [`Self::hard_input_limit_for`] of this same value makes the split
    /// dynamic — a 2k-output agent keeps ~94% of a 32k window for input —
    /// while input + output provably never over-commit the window.
    #[must_use]
    pub fn effective_output_budget(&self, requested_max_tokens: usize) -> usize {
        requested_max_tokens
            .min(self.max_output_tokens)
            .min(self.context_window / 2)
            .max(1)
    }

    /// Output-token budget paired with [`Self::hard_input_limit`]: the slice
    /// of the window the input side does NOT claim. A request whose
    /// `max_tokens` is clamped to this can never over-commit the window —
    /// input ≤ `hard_input_limit` and output ≤ this sum to ≤ `context_window`.
    #[must_use]
    pub fn output_token_budget(&self) -> usize {
        self.context_window.saturating_sub(self.hard_input_limit())
    }

    /// Create with current timestamp
    fn with_timestamp(mut self) -> Self {
        self.cached_at = current_timestamp();
        self
    }

    /// Tokens available for conversation history after hard-input + reserved heads.
    ///
    /// `system_reserved` / `response_reserved` cover prompt + completion heads that
    /// sit outside the stored history window (GUI system prompt + response reserve).
    /// Floored at `min_history` so truncation never empties the transcript, but
    /// never above the hard input limit itself.
    #[must_use]
    pub fn conversation_history_budget(
        &self,
        system_reserved: usize,
        response_reserved: usize,
        min_history: usize,
    ) -> usize {
        let hard = self.hard_input_limit();
        hard.saturating_sub(system_reserved.saturating_add(response_reserved))
            .max(min_history.min(hard))
    }
}

/// Get current Unix timestamp
fn current_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        // Seconds since the epoch exceed i64::MAX only ~292 billion years from now.
        .map_or(0, |d| i64::try_from(d.as_secs()).unwrap_or(i64::MAX))
}

/// Universal context floor when no provider has reported limits for a model.
///
/// Providers update windows constantly; we never hardcode per-model values.
/// Callers should prefer [`LlmClient::get_model_info`] / [`ModelInfoCache`].
/// This floor is intentionally conservative so unknown / offline models stay
/// under their real window rather than overshoot and 400.
pub const UNKNOWN_CONTEXT_WINDOW: usize = 32_000;

/// Universal max-output floor when the provider does not report it.
pub const UNKNOWN_MAX_OUTPUT_TOKENS: usize = 4_096;

/// Best-effort context window (in tokens) for a model by name.
///
/// Order: live disk cache from a prior API fetch, else [`UNKNOWN_CONTEXT_WINDOW`].
/// No per-model name table — models change constantly; use the provider API.
#[must_use]
pub fn model_context_window(model: &str) -> usize {
    model_info_from_cache_or_unknown(model, "").context_window
}

/// Build a [`ModelInfo`] from the on-disk cache, or the universal unknown floor.
///
/// Used by sync paths that cannot await a provider fetch (e.g. consolidation
/// sizing). Prefer [`LlmClient::get_model_info`] whenever a client is available,
/// and [`LlmClient::model_info_from_cache_or_unknown`] for a sync read through
/// one.
///
/// The result is clamped to the effective runner window
/// ([`clamp_model_info_to_effective_window`]): a cached provider claim is a
/// static fact, but a mid-run `num_ctx` demotion is live state, and budgets
/// derived from the stale claim overflow every subsequent prompt.
#[must_use]
pub fn model_info_from_cache_or_unknown(model: &str, provider: &str) -> ModelInfo {
    clamp_model_info_to_effective_window(model, cached_model_info_or_unknown(model, provider))
}

/// The cached provider claim for `model`, or the universal unknown floor, as
/// the provider states it: not clamped to any runner window.
fn cached_model_info_or_unknown(model: &str, provider: &str) -> ModelInfo {
    ModelInfoCache::default_location()
        .and_then(|cache| cache.get(model))
        .unwrap_or_else(|| unknown_model_info(model, provider))
}

/// The context window every Ollama server that has sized `model` will honour
/// right now.
///
/// That is `reported` (provider metadata / cache / config) clamped by the
/// smallest `num_ctx` latched for `model` on any server: without the server a
/// request goes to, only a window all of them honour is safe. A caller holding
/// the client reads [`LlmClient::effective_context_window`] instead, the window
/// of the one server that client sends to. Models that were never sized by the
/// Ollama path (cloud providers, unlatched local models) pass `reported`
/// through unchanged.
///
/// This is the read side of [`LlmClient::demote_context`]. A mid-run VRAM
/// demotion rewrites the latch, and every downstream budget (compression
/// threshold, hard input limit, workspace cap) must be derived from THIS
/// number, not from the provider's static claim. Observed live 2026-08-02: a
/// silent 16384→4096 demotion left the agent budgeting for 16k while Ollama
/// truncated every prompt to 4k model-side — the model lost its task and a
/// 4-hour eval scored 1/42 against 4/42-per-hour at full context.
#[must_use]
pub fn effective_context_window(model: &str, reported: usize) -> usize {
    clamp_window(reported, LlmClient::smallest_latched_num_ctx(model))
}

/// How long one query to a server on this machine may take, as `curl
/// --max-time` wants it. A local Ollama answers `/api/tags` and `/api/ps` in
/// milliseconds; 3 s is past that with room (the daemon's Ollama probe uses
/// the same figure), and a wedged server cannot hold up the sizing of a chat
/// request, which runs before the request is sent, for longer than that.
const LOCAL_OLLAMA_QUERY_MAX_SECS: &str = "3";

/// Run `probe`, which waits on child processes (`nvidia-smi`, `curl`), without
/// starving the async runtime it is called from.
///
/// Sizing a request's `num_ctx` runs on whatever thread asks, which is a tokio
/// worker when a chat stream sizes its first request. A task parked in that
/// worker's LIFO slot cannot be taken by the other workers, so when the task
/// that would answer the probe (a server on the same runtime) sits there, the
/// probe waits on itself; this hung a daemon test one run in four.
/// `block_in_place` hands the worker's queued tasks to the others before
/// blocking. It exists only on the multi-thread runtime (it panics on a
/// current-thread one); there the curl bound
/// ([`LOCAL_OLLAMA_QUERY_MAX_SECS`]) is what ends the wait.
fn without_starving_the_runtime<T>(probe: impl FnOnce() -> T) -> T {
    match tokio::runtime::Handle::try_current() {
        Ok(handle) if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
            tokio::task::block_in_place(probe)
        }
        _ => probe(),
    }
}

/// `reported`, lowered to `num_ctx` when there is one. A latch is a ceiling,
/// never a floor: a claim smaller than it stands.
fn clamp_window(reported: usize, num_ctx: Option<u32>) -> usize {
    num_ctx.map_or(reported, |num_ctx| reported.min(num_ctx as usize))
}

/// `info` with its window clamped to the effective runner window for `model`
/// (see [`effective_context_window`]).
///
/// `max_output_tokens` is re-bounded to half the clamped window, mirroring how
/// the Ollama fetch derives it from the full window — so every budget the
/// [`ModelInfo`] helpers derive (`hard_input_limit_for`,
/// `effective_output_budget`, `compression_threshold_for`) stays consistent
/// with the window the runner will actually honour. No-op when no latch exists
/// or the latch is at/above the reported window.
#[must_use]
pub fn clamp_model_info_to_effective_window(model: &str, info: ModelInfo) -> ModelInfo {
    let effective = effective_context_window(model, info.context_window);
    clamp_model_info_to_window(model, info, effective)
}

/// `info` with its window lowered to `effective` and its output cap re-bounded
/// to half of it (see [`clamp_model_info_to_effective_window`]).
fn clamp_model_info_to_window(model: &str, mut info: ModelInfo, effective: usize) -> ModelInfo {
    if effective < info.context_window {
        tracing::debug!(
            model,
            reported_window = info.context_window,
            effective_window = effective,
            "model info clamped to live effective num_ctx (runner was demoted below the provider claim)"
        );
        info.context_window = effective;
        info.max_output_tokens = info.max_output_tokens.min((effective / 2).max(1));
    }
    info
}

/// The quantum every demoted `num_ctx` snaps to.
///
/// Derived from the runner, not chosen: llama.cpp (Ollama's backend) prefills
/// prompts in physical batches of 512 tokens (`n_batch`/`n_ubatch` default),
/// so a context is consumed and its KV cache allocated in 512-token slices —
/// a finer step changes the footprint by less than one prefill batch and buys
/// nothing, while a coarser one (the old power-of-two buckets) throws away
/// windows that would have fit.
pub const NUM_CTX_QUANTUM: u32 = 512;

/// The top of the demotion ladder.
///
/// It is the largest window the VRAM sizing path can grant (its buckets are
/// `[4096, 8192, 16384, 32768]`), so an unlatched model demoting for the first
/// time walks down from the most it could ever have been given, exactly as the
/// old bucket ladder did.
pub const NUM_CTX_CEILING: u32 = 32_768;

/// Fallback minimum viable window for [`LlmClient::demote_context`] when the
/// caller cannot supply the live agent floor.
///
/// Derived from the pressure-tier `ContextFloor` the agent measures (system
/// prompt slice + core-plus-work-evidence tool definitions + step frame +
/// window-scaled output reserve): observed live 2026-08-03 at ~4.2k tokens,
/// rounded UP to the next [`NUM_CTX_QUANTUM`]. The old ladder's 4096 terminal
/// bucket sat BELOW that floor, and landing there produced 38 below-floor
/// stops and 8 burned resume cycles in 8 minutes — a default any lower than
/// the measured floor manufactures a window the agent will loudly refuse.
///
/// Callers with the run at hand should pass the measured floor instead
/// (`nanna_agent`'s min-viable-window derivation); this constant only covers
/// call sites that have no agent state to measure.
pub const DEFAULT_MIN_VIABLE_NUM_CTX: u32 = 4_608;

/// The provider clients' declared silence tolerance.
///
/// It is how long a stream may go without delivering a single byte before the
/// transport itself calls it dead (`reqwest`'s `read_timeout` on both the remote
/// and the Ollama client — see `build_http_client` /
/// `build_ollama_http_client`). This is deliberately the ONLY time bound on a
/// stream: there is no total-request deadline, because a
/// healthy long generation never pauses between chunks for anywhere near this
/// long, while it routinely exceeds any total cap.
///
/// Public so watchers can DERIVE their cadence from the declared bound instead
/// of inventing a second constant that drifts: the agent loop's stream
/// watchdog fires at a multiple of this (transport must get first claim on a
/// truly silent socket), and the daemon's liveness beat at a fraction of it
/// (several beats must land inside any legal silence window).
pub const STREAM_READ_TIMEOUT_SECS: u64 = 120;

/// Conservative model info when the provider has not (yet) told us limits.
///
/// Supports tools optimistically — capability misses are softer than overrunning
/// a too-small context budget on a modern agent path.
#[must_use]
pub fn unknown_model_info(model: &str, provider: &str) -> ModelInfo {
    ModelInfo {
        id: model.to_string(),
        context_window: UNKNOWN_CONTEXT_WINDOW,
        max_output_tokens: UNKNOWN_MAX_OUTPUT_TOKENS,
        supports_tools: true,
        supports_vision: false,
        embedding_dimension: None,
        cached_at: current_timestamp(),
        provider: provider.to_string(),
    }
}

/// Alias kept for local fetch_* fallbacks — one floor, not a model table.
fn default_model_info(model: &str, provider: &str) -> ModelInfo {
    unknown_model_info(model, provider)
}

/// One model object from `GET /v1/models/{id}`.
///
/// The field names are load-bearing and were wrong: the Models API returns
/// **`max_input_tokens`** (the context window) and **`max_tokens`** (the output
/// cap), and there is no `context_window` field at all. Reading for the wrong
/// names is silent — both `Option`s deserialize to `None`, both fall back to
/// the unknown-model floor, and that floor gets written to the on-disk cache
/// with a fresh timestamp. Every Anthropic model resolved to 32k context / 4k
/// output as a result, no matter what it actually was.
///
/// The aliases keep the older spellings working if a proxy or gateway answers
/// with them. Deliberately at module scope rather than inside the fetch so the
/// mapping is testable without a network round trip.
#[derive(Deserialize)]
struct AnthropicModelResponse {
    id: String,
    #[serde(default, alias = "context_window")]
    max_input_tokens: Option<usize>,
    #[serde(default, alias = "max_output_tokens")]
    max_tokens: Option<usize>,
}

/// The `model_info` block of an Ollama `POST /api/show` response.
///
/// Kept as a raw map because the keys are not fixed: GGUF namespaces its
/// metadata by *architecture*, so the window arrives as `llama.context_length`,
/// `qwen3.context_length`, `bert.context_length`, `nomic-bert.context_length`,
/// and so on. See [`gguf_metadata_usize`].
#[derive(Deserialize, Default)]
struct OllamaShowResponse {
    #[serde(default)]
    model_info: Option<serde_json::Map<String, serde_json::Value>>,
}

/// Read a GGUF metadata integer by its *suffix*, across any architecture.
///
/// GGUF keys everything by architecture and there is no reliable
/// `general.*` mirror. `nomic-embed-text` publishes only
/// `nomic-bert.context_length` and `nomic-bert.embedding_length`; nothing under
/// `general.`. Reading the fixed key `general.context_length` therefore
/// returned `None` for essentially every local model, the caller fell back to
/// the unknown-model floor, and that invented 32k window was cached with a
/// fresh timestamp and used to size every budget downstream — in silence. The
/// same miss left `embedding_dimension` at `None` for every Ollama embedder.
///
/// `general.` still wins when present, since a model that does publish it is
/// stating its own answer. Among architecture-namespaced matches the
/// **smallest** wins: these bound what we may send, and the only safe reading
/// of an ambiguous bound is the one that cannot overrun.
fn gguf_metadata_usize(
    model_info: &serde_json::Map<String, serde_json::Value>,
    suffix: &str,
) -> Option<usize> {
    let dotted = format!(".{suffix}");
    if let Some(v) = model_info
        .get(&format!("general.{suffix}"))
        .and_then(serde_json::Value::as_u64)
    {
        return usize::try_from(v).ok();
    }
    model_info
        .iter()
        .filter(|(k, _)| k.ends_with(&dotted))
        .filter_map(|(_, v)| v.as_u64())
        .filter_map(|v| usize::try_from(v).ok())
        .filter(|v| *v > 0)
        .min()
}

/// File-based cache for model information.
///
/// Holds only what a provider answered about a model. A lookup that got no
/// answer (a refusal, a transport error, a 404) is served the unknown-model
/// floor for that lookup alone: the cache cannot see the event that would
/// change the answer (a token saved, a model pulled, a server back up), so a
/// persisted floor would outlive it by up to [`ModelInfo::CACHE_TTL_SECS`].
///
/// One entry per model name, stamped with the server that answered. A client
/// reads only its own server's entry ([`Self::get_from`]); the name-only
/// readers ([`Self::get`]), which size budgets with no client in hand, see the
/// server asked last.
#[derive(Clone)]
pub struct ModelInfoCache {
    cache_dir: std::path::PathBuf,
}

/// The on-disk form of a cache entry: the answer and who gave it.
#[derive(Serialize, Deserialize)]
struct CachedModelInfo {
    /// The answering server, as [`LlmClient::model_info_server`] names it.
    server: String,
    info: ModelInfo,
}

impl ModelInfoCache {
    /// Bumped whenever a fix changes what a *correct* cached entry contains,
    /// so previously-written entries are orphaned rather than served.
    ///
    /// v2: the Anthropic Models API response was being read with the wrong
    /// field names, so every Anthropic model cached the unknown-model floor.
    ///
    /// v3: Ollama GGUF metadata was read at the fixed key `general.*`, which
    /// almost no model publishes — so every local model cached the invented
    /// 32k floor and every embedder cached a `None` dimension.
    ///
    /// v4: failed lookups were cached as if they were answers. Every 401 (the
    /// previous build sent no bearer token on `/api/show`, so every proxied
    /// server refused it), 404, transport error, `OpenRouter` error status and
    /// proxy failure wrote the floor with a fresh timestamp. A v3 file carries
    /// nothing that tells such an entry from a real answer, so a shorter TTL
    /// would still serve the poisoned ones until they aged out while
    /// discarding good ones early; only orphaning the generation reaches all
    /// of them. v4 entries also record the answering server, a field no v3
    /// file has.
    const SCHEMA_GENERATION: u32 = 4;

    /// Create a new cache with the specified directory
    pub fn new(cache_dir: impl Into<std::path::PathBuf>) -> Self {
        Self {
            cache_dir: cache_dir.into(),
        }
    }

    /// Create cache in the default location (user's cache directory)
    #[must_use]
    pub fn default_location() -> Option<Self> {
        directories::ProjectDirs::from("com", "nanna", "nanna")
            .map(|dirs| Self::new(dirs.cache_dir().join("model_info")))
    }

    /// Cached model info for `model` from whichever server answered last, if
    /// it exists and is not expired.
    ///
    /// For callers with no client in hand; a client reads through
    /// [`Self::get_from`], which never serves another server's answer.
    #[must_use]
    pub fn get(&self, model: &str) -> Option<ModelInfo> {
        self.read(model).map(|entry| entry.info)
    }

    /// Cached model info for `model` only if `server` gave it.
    #[must_use]
    pub fn get_from(&self, server: &str, model: &str) -> Option<ModelInfo> {
        self.read(model)
            .filter(|entry| entry.server == server)
            .map(|entry| entry.info)
    }

    fn read(&self, model: &str) -> Option<CachedModelInfo> {
        let path = self.cache_path(model);
        if !path.exists() {
            return None;
        }

        let content = std::fs::read_to_string(&path).ok()?;
        let entry: CachedModelInfo = serde_json::from_str(&content).ok()?;

        if entry.info.is_expired() {
            // Remove expired cache
            let _ = std::fs::remove_file(&path);
            debug!(model = %model, "Model info cache expired, removed");
            return None;
        }

        debug!(
            model = %model,
            server = %entry.server,
            context_window = entry.info.context_window,
            "Loaded model info from cache"
        );
        Some(entry)
    }

    /// Store `server`'s answer about a model.
    ///
    /// Only an answer belongs here, never a floor that stands in for a failed
    /// lookup (see the type docs).
    ///
    /// # Errors
    ///
    /// Returns [`LlmError::Io`] when the cache directory cannot be created or the
    /// entry file cannot be written, and [`LlmError::Json`] when `info` cannot be
    /// serialized.
    pub fn set(&self, server: &str, info: &ModelInfo) -> Result<(), LlmError> {
        // Ensure cache directory exists
        std::fs::create_dir_all(&self.cache_dir)?;

        let path = self.cache_path(&info.id);
        let entry = CachedModelInfo {
            server: server.to_string(),
            info: info.clone(),
        };
        let content = serde_json::to_string_pretty(&entry)?;
        std::fs::write(&path, content)?;

        debug!(model = %info.id, server = %server, path = %path.display(), "Cached model info");
        Ok(())
    }

    /// Drop the entry for `model` if a server other than `server` gave it.
    ///
    /// For a lookup through `server` that got no answer: the name-only readers
    /// would otherwise go on sizing budgets from the other server's window
    /// while the lookup itself fell back to the floor. That window can be
    /// larger than this server's; the floor is the value built to stay under
    /// a window nobody has reported.
    fn forget_other_server(&self, server: &str, model: &str) {
        if self.read(model).is_some_and(|entry| entry.server != server) {
            let _ = std::fs::remove_file(self.cache_path(model));
            debug!(model = %model, server = %server, "Dropped another server's cached model info");
        }
    }

    /// Clear all cached model info
    ///
    /// # Errors
    ///
    /// Returns [`LlmError::Io`] when the cache directory exists but cannot be
    /// removed.
    pub fn clear(&self) -> Result<(), LlmError> {
        if self.cache_dir.exists() {
            std::fs::remove_dir_all(&self.cache_dir)?;
        }
        Ok(())
    }

    /// Get cache file path for a model
    fn cache_path(&self, model: &str) -> std::path::PathBuf {
        // Sanitize model name for use as filename
        let safe_name: String = model
            .chars()
            .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
            .collect();
        // The suffix is a schema generation, not decoration. Entries written
        // before a fix (see `SCHEMA_GENERATION`) can hold the unknown-model
        // floor (32k window / 4k output) for models that are nothing like
        // it, with a fresh timestamp — so the week-long TTL would keep
        // serving them. Bumping the generation orphans those files instead of
        // trusting them; the next lookup refetches.
        self.cache_dir
            .join(format!("{safe_name}.v{}.json", Self::SCHEMA_GENERATION))
    }
}

impl From<reqwest::Error> for LlmError {
    fn from(e: reqwest::Error) -> Self {
        Self::Http(e.to_string())
    }
}

impl From<serde_json::Error> for LlmError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e.to_string())
    }
}

impl LlmError {
    /// Check if this error is a rate limit error (429)
    #[must_use]
    pub const fn is_rate_limit(&self) -> bool {
        matches!(self, Self::RateLimit { .. })
            || matches!(self, Self::Api { status: 429, .. })
    }

    /// An input-level rejection: this request's INPUT was too long for the
    /// model, stated deterministically by the provider ("input length N
    /// exceeds model maximum M", 400/413/422-shaped).
    ///
    /// The distinction is load-bearing for callers that bench providers on
    /// deterministic errors: a provider that rejected one oversized input is
    /// perfectly healthy for every other call, so this class must never bench
    /// it — the CALLER owns the fault, and the fix is to shrink the input,
    /// not to wait out a cooldown. (Observed 2026-08-10: one over-long input
    /// classified as a provider fault benched the only local embedder for
    /// 240s while a mission was trying to write memories through it.)
    #[must_use]
    pub fn is_input_overflow(&self) -> bool {
        match self {
            Self::Api {
                status: 400 | 413 | 422,
                message,
            } => input_overflow_limits(message).is_some(),
            _ => false,
        }
    }

    /// Check if this error should trigger a fallback to another model
    #[must_use]
    pub fn should_fallback(&self) -> bool {
        match self {
            // Rate limits - definitely fallback.
            //
            // A wedged runner used to arrive as Api{status:502} and so fell
            // into the 502 case below. It is still exactly that fault — keep it
            // falling back, or splitting the variant out would silently stop
            // the router trying another model.
            //
            // Network errors - might be transient.
            Self::RateLimit { .. } | Self::WedgedRunner { .. } | Self::Http(_) => true,
            Self::Api { status, message } => {
                // 429 = rate limit
                // 529 = overloaded (Anthropic)
                // 503 = service unavailable
                // 502 = bad gateway
                if *status == 429 || *status == 529 || *status == 503 || *status == 502 {
                    return true;
                }
                // Check for rate limit in error message (some APIs return 400 with rate limit message)
                let msg_lower = message.to_lowercase();
                msg_lower.contains("rate limit") || msg_lower.contains("rate_limit")
            }
            // Don't fallback on auth errors, JSON errors, etc.
            _ => false,
        }
    }

    /// [`Self::from_api_response`] for a failed HTTP response, with the wait
    /// the response's headers name when the body names none.
    ///
    /// A 429's headers are where most providers say when to come back —
    /// Anthropic's `retry-after`, `OpenAI`'s `x-ratelimit-reset-*` — and the
    /// error used to be built from the body alone, so every one of them was
    /// dropped and the caller backed off on a guess.
    pub async fn from_response(response: reqwest::Response) -> Self {
        let status = response.status().as_u16();
        let header_wait = retry_after_from_headers(response.headers());
        let message = response.text().await.unwrap_or_default();
        match Self::from_api_response(status, message) {
            Self::RateLimit {
                message,
                retry_after,
            } => Self::RateLimit {
                message,
                retry_after: retry_after.or(header_wait),
            },
            other => other,
        }
    }

    /// Parse an API error response to extract rate limit info
    #[must_use]
    pub fn from_api_response(status: u16, message: String) -> Self {
        // Both spellings are real: OpenAI-style bodies say "rate_limit_exceeded",
        // OpenRouter's human-form message says "Rate limit exceeded". Matching
        // only the underscore form silently declassified OpenRouter's — and a
        // rate limit that isn't classified as one is retried at full speed,
        // which is the one response a rate limit must never get.
        let lowered = message.to_lowercase();
        if status == 429 || lowered.contains("rate_limit") || lowered.contains("rate limit") {
            // Try to extract retry-after from the message
            let retry_after = Self::parse_retry_after(&message);
            return Self::RateLimit { message, retry_after };
        }

        Self::Api { status, message }
    }

    /// Try to parse retry-after seconds from an error message.
    ///
    /// Providers say when they will be ready and we were throwing it away —
    /// `OpenRouter`'s 429 body carries `"X-RateLimit-Reset":"1785222060000"`, a
    /// millisecond epoch, and every caller was backing off on a guessed
    /// schedule instead. Reading it turns "wait a made-up interval and hope"
    /// into "wait exactly as long as they asked".
    ///
    /// Returns `None` when nothing usable is present, which keeps the caller's
    /// own backoff as the fallback.
    fn parse_retry_after(message: &str) -> Option<u64> {
        /// A reset further out than this is a misread, not a real wait.
        const MAX_PLAUSIBLE_WAIT_SECS: u64 = 3_600;

        let now_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .ok()?
            .as_secs();

        // `X-RateLimit-Reset` — epoch, in seconds or milliseconds depending on
        // the provider. Disambiguated by magnitude rather than by trusting
        // either convention.
        let from_reset = message
            .split("X-RateLimit-Reset")
            .nth(1)
            .and_then(|rest| {
                let digits: String = rest
                    .chars()
                    .skip_while(|c| !c.is_ascii_digit())
                    .take_while(char::is_ascii_digit)
                    .collect();
                digits.parse::<u64>().ok()
            })
            .map(|reset| if reset > 100_000_000_000 { reset / 1000 } else { reset })
            .and_then(|reset_secs| reset_secs.checked_sub(now_secs));

        // `retry-after: N` / `"retry_after": N` — plain seconds.
        let from_retry_after = ["retry-after", "retry_after"].iter().find_map(|key| {
            let rest = message.to_lowercase();
            let idx = rest.find(key)?;
            let digits: String = rest[idx + key.len()..]
                .chars()
                .skip_while(|c| !c.is_ascii_digit())
                .take_while(char::is_ascii_digit)
                .collect();
            digits.parse::<u64>().ok()
        });

        from_reset
            .or(from_retry_after)
            .filter(|secs| *secs > 0 && *secs <= MAX_PLAUSIBLE_WAIT_SECS)
    }
}

/// Parse "(input) length N exceeds (model) maximum M" out of a provider error
/// body, returning `(sent, limit)` in the provider's own unit (tokens).
///
/// Deliberately liberal about the words between the numbers — Ollama says
/// "input length exceeds maximum context length", OpenAI-compatible servers
/// vary the nouns — and strict about the shape: the message must carry the
/// overflow vocabulary AND two integers where a larger one is followed by a
/// smaller one. The pair is found as the first adjacent *descending* pair in
/// order of appearance, which steps over version fragments in model names
/// ("v1.5" yields the ascending 1, 5) and lands on `sent > limit` every time
/// the provider states both.
fn input_overflow_limits(message: &str) -> Option<(usize, usize)> {
    let lowered = message.to_lowercase();
    let overflow_shape = lowered.contains("exceed")
        && (lowered.contains("length") || lowered.contains("token"))
        && (lowered.contains("maximum") || lowered.contains("context") || lowered.contains("limit"));
    if !overflow_shape {
        return None;
    }

    let mut numbers: Vec<usize> = Vec::new();
    let mut digits = String::new();
    for c in lowered.chars() {
        if c.is_ascii_digit() {
            digits.push(c);
        } else if !digits.is_empty() {
            if let Ok(n) = digits.parse() {
                numbers.push(n);
            }
            digits.clear();
        }
    }
    if !digits.is_empty()
        && let Ok(n) = digits.parse()
    {
        numbers.push(n);
    }

    numbers
        .windows(2)
        .find(|pair| pair[0] > pair[1] && pair[1] >= 1)
        .map(|pair| (pair[0], pair[1]))
}

// =============================================================================
// Provider request pacing
// =============================================================================
//
// Enforced at the PROVIDER level so every caller that routes through a
// rate-limited provider — chat, dream summarization, embeddings, anything —
// draws from ONE clock per provider instead of each pacing itself and
// collectively burning the shared quota. (The gate first lived in the dream
// summarizer alone, which protected dreams and nobody else; the observed
// failure was a dream cycle firing one request per cluster in the same second
// and losing 20 of 22 clusters to 429s.)

/// Per-provider pacing state: when we last asked, and what the provider last
/// TOLD US about our budget.
///
/// The provider's own numbers beat any constant: `X-RateLimit-Remaining` /
/// `X-RateLimit-Reset` reflect the actual account's tier and the actual
/// window, which published docs cannot. The static
/// [`Provider::min_request_spacing`] is only the bootstrap prior, used until
/// the first response teaches us the truth.
#[derive(Default)]
struct PaceState {
    /// When the previous request to this provider was issued.
    last_request: Option<tokio::time::Instant>,
    /// Requests the provider said we had left, decremented optimistically as
    /// we send (the header confirming it arrives one response later).
    remaining: Option<u64>,
    /// When the provider said the window refills.
    reset_at: Option<tokio::time::Instant>,
}

impl PaceState {
    /// How long the next request must wait, given `now`.
    ///
    /// Pure — all clock reads happen at the caller — so the policy is testable
    /// without real time:
    /// - Provider said the window has refilled (`reset_at` in the past): no
    ///   wait; the budget is fresh even if we never saw the new headers.
    /// - Provider said how much budget remains: spread it evenly across the
    ///   time left in the window (`(reset_at - now) / remaining`). A generous
    ///   budget yields spacing BELOW the static prior — dynamic runs faster,
    ///   not just safer. `remaining == 0` waits out the window.
    /// - Nothing learned yet: the static published-limit prior.
    fn required_wait(
        &self,
        now: tokio::time::Instant,
        fallback_spacing: tokio::time::Duration,
    ) -> tokio::time::Duration {
        let since_last = self
            .last_request
            .map(|prev| now.saturating_duration_since(prev));

        let spacing = match (self.reset_at, self.remaining) {
            (Some(reset_at), _) if reset_at <= now => return tokio::time::Duration::ZERO,
            (Some(reset_at), Some(0)) => return reset_at.saturating_duration_since(now),
            (Some(reset_at), Some(remaining)) => {
                let window_left = reset_at.saturating_duration_since(now);
                // u32 clamp is safe: a `remaining` beyond u32::MAX means the
                // spacing is zero at any window length we could measure.
                window_left / u32::try_from(remaining).unwrap_or(u32::MAX)
            }
            _ => fallback_spacing,
        };

        since_last.map_or(tokio::time::Duration::ZERO, |elapsed| spacing.saturating_sub(elapsed))
    }

    /// Fold a provider's rate-limit headers into the state.
    ///
    /// `reset_in` is relative (converted from the header's epoch by the
    /// caller, which owns the wall clock) so this stays pure and testable.
    fn learn(&mut self, remaining: Option<u64>, reset_in: Option<tokio::time::Duration>) {
        if let Some(remaining) = remaining {
            self.remaining = Some(remaining);
        }
        if let Some(reset_in) = reset_in {
            self.reset_at = Some(tokio::time::Instant::now() + reset_in);
        }
    }
}

/// One pacing gate per paced provider. Separate mutexes, not one map behind
/// one lock: the lock is held across the pacing sleep deliberately (concurrent
/// same-provider callers must serialize, or they all measure from the same
/// "last" stamp and fire together), and holding a shared lock across that
/// sleep would stall OTHER providers for no reason.
fn pace_gate(provider: Provider) -> Option<&'static tokio::sync::Mutex<PaceState>> {
    use std::sync::OnceLock;
    use tokio::sync::Mutex;
    static OPENROUTER: OnceLock<Mutex<PaceState>> = OnceLock::new();
    static GITHUB: OnceLock<Mutex<PaceState>> = OnceLock::new();
    match provider {
        Provider::OpenRouter => Some(OPENROUTER.get_or_init(|| Mutex::new(PaceState::default()))),
        Provider::GitHubModels => Some(GITHUB.get_or_init(|| Mutex::new(PaceState::default()))),
        _ => None,
    }
}

/// Hold a request to `provider` until its pacing allows it, process-wide
/// across all clients and callers.
///
/// Public so callers that reach a paced provider WITHOUT an [`LlmClient`]
/// (a future embedding path, for instance) can share the same clock. No-op
/// for providers without a published quota.
pub async fn pace_provider_requests(provider: Provider) {
    let Some(fallback) = provider.min_request_spacing() else {
        return;
    };
    let Some(gate) = pace_gate(provider) else {
        return;
    };
    let mut state = gate.lock().await;
    let wait = state.required_wait(tokio::time::Instant::now(), fallback);
    if wait > tokio::time::Duration::ZERO {
        tokio::time::sleep(wait).await;
    }
    state.last_request = Some(tokio::time::Instant::now());
    // The request we are about to send spends one unit of whatever budget the
    // provider last reported; the confirming header arrives one response late.
    state.remaining = state.remaining.map(|r| r.saturating_sub(1));
}

/// Teach the pacing gate what a response's rate-limit headers said.
///
/// Reads `x-ratelimit-remaining` and `x-ratelimit-reset` (`OpenRouter` sends the
/// reset as an epoch in milliseconds, GitHub in seconds — disambiguated by
/// magnitude, same rule as [`LlmError::parse_retry_after`]). Uses `try_lock`
/// rather than awaiting: losing one update to contention is fine — the next
/// response repeats the lesson — while awaiting here would stall response
/// handling behind another caller's pacing sleep.
fn learn_rate_limit_headers(provider: Provider, headers: &reqwest::header::HeaderMap) {
    let Some(gate) = pace_gate(provider) else {
        return;
    };
    let header_u64 = |name: &str| -> Option<u64> {
        headers
            .get(name)
            .and_then(|v| v.to_str().ok())
            // Some providers send floats ("59.0"); take the integer part.
            .and_then(|s| s.split('.').next()?.trim().parse::<u64>().ok())
    };
    let remaining = header_u64("x-ratelimit-remaining");
    let reset_in = header_u64("x-ratelimit-reset").and_then(|reset| {
        let now_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .ok()?
            .as_secs();
        let reset_secs = if reset > 100_000_000_000 { reset / 1000 } else { reset };
        // A reset in the past yields None here; `required_wait` treats a
        // stale window as refilled anyway once reset_at lapses.
        reset_secs
            .checked_sub(now_secs)
            .map(tokio::time::Duration::from_secs)
    });
    if remaining.is_none() && reset_in.is_none() {
        return;
    }
    if let Ok(mut state) = gate.try_lock() {
        state.learn(remaining, reset_in);
    }
}

/// LLM Provider
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    Anthropic,
    OpenAI,
    OpenRouter,
    GitHubModels,
    /// Claude Max/Pro proxy (OpenAI-compatible, uses claude-max-api-proxy)
    ClaudeProxy,
    Ollama,
}

impl Provider {
    /// Bootstrap spacing between consecutive requests to this provider,
    /// derived from the provider's PUBLISHED request-per-minute quota — never
    /// a tuning knob. This is only the PRIOR: the moment a response arrives,
    /// its `x-ratelimit-*` headers supersede it (see
    /// `PaceState::required_wait`) because headers reflect the actual
    /// account's tier, which docs cannot. `None` means "do not pace", and
    /// every `None` is a decision:
    ///
    /// - `OpenRouter`: free tier publishes 20 requests/minute → 60s/20 = 3s.
    ///   The free `:free` models are what this codebase configures.
    /// - `GitHubModels`: free tier publishes 15 requests/minute → 60s/15 = 4s.
    /// - `Anthropic` / `OpenAI`: limits are ACCOUNT-TIER dependent (Anthropic
    ///   tier 1 is 50 RPM, tier 4 is 4000; `OpenAI` similar). Pacing everyone to
    ///   the lowest tier would throttle paid accounts in the chat hot path for
    ///   no reason; both providers answer bursts with 429 + retry-after, which
    ///   the reactive layer honors. Proactive pacing here would need the
    ///   account's actual tier, which nothing exposes.
    /// - `ClaudeProxy`: fronts an Anthropic subscription; same reasoning.
    /// - `Ollama`: local — the only quota is the GPU itself.
    #[must_use]
    pub const fn min_request_spacing(self) -> Option<tokio::time::Duration> {
        match self {
            Self::OpenRouter => Some(tokio::time::Duration::from_secs(3)),
            Self::GitHubModels => Some(tokio::time::Duration::from_secs(4)),
            Self::Anthropic | Self::OpenAI | Self::ClaudeProxy | Self::Ollama => {
                None
            }
        }
    }
}

/// Message role
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

/// Chat message (simple text format)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: Role::System, content: content.into() }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self { role: Role::User, content: content.into() }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: Role::Assistant, content: content.into() }
    }
}

impl AnthropicMessage {
    /// Create a tool result message
    pub fn tool_result(tool_use_id: impl Into<String>, content: impl Into<String>, is_error: bool) -> Self {
        Self {
            role: "user".to_string(),
            content: vec![ContentBlock::ToolResult {
                tool_use_id: tool_use_id.into(),
                content: content.into(),
                is_error: if is_error { Some(true) } else { None },
            }],
        }
    }

    /// Create an assistant message with tool use
    pub fn tool_use(id: impl Into<String>, name: impl Into<String>, input: serde_json::Value) -> Self {
        Self {
            role: "assistant".to_string(),
            content: vec![ContentBlock::ToolUse {
                id: id.into(),
                name: name.into(),
                input,
            }],
        }
    }

    /// Create an assistant message with text and tool use
    pub fn assistant_with_tool_use(
        text: Option<String>,
        tool_id: impl Into<String>,
        tool_name: impl Into<String>,
        tool_input: serde_json::Value,
    ) -> Self {
        let mut content = Vec::new();
        if let Some(t) = text
            && !t.is_empty() {
                content.push(ContentBlock::Text { text: t });
            }
        content.push(ContentBlock::ToolUse {
            id: tool_id.into(),
            name: tool_name.into(),
            input: tool_input,
        });
        Self {
            role: "assistant".to_string(),
            content,
        }
    }
}

/// Anthropic message content block
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text { text: String },
    Image { source: ImageSource },
    ToolUse { id: String, name: String, input: serde_json::Value },
    ToolResult { tool_use_id: String, content: String, #[serde(skip_serializing_if = "Option::is_none")] is_error: Option<bool> },
    /// Extended thinking content (reasoning)
    Thinking {
        thinking: String,
        /// Signature returned by Anthropic — MUST be sent back in multi-turn conversations.
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
}

/// Image source for vision API
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageSource {
    Base64 {
        media_type: String,
        data: String,
    },
    Url {
        url: String,
    },
}

/// Anthropic message format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: Vec<ContentBlock>,
}

impl AnthropicMessage {
    #[must_use]
    pub fn user(content: Vec<ContentBlock>) -> Self {
        Self { role: "user".to_string(), content }
    }

    #[must_use]
    pub fn assistant(content: Vec<ContentBlock>) -> Self {
        Self { role: "assistant".to_string(), content }
    }

    pub fn user_text(text: impl Into<String>) -> Self {
        Self::user(vec![ContentBlock::Text { text: text.into() }])
    }

    pub fn assistant_text(text: impl Into<String>) -> Self {
        Self::assistant(vec![ContentBlock::Text { text: text.into() }])
    }

    /// Create a user message with an image (base64)
    pub fn user_image(media_type: impl Into<String>, base64_data: impl Into<String>, text: Option<String>) -> Self {
        let mut content = vec![ContentBlock::Image {
            source: ImageSource::Base64 {
                media_type: media_type.into(),
                data: base64_data.into(),
            },
        }];
        if let Some(t) = text {
            content.push(ContentBlock::Text { text: t });
        }
        Self::user(content)
    }

    /// Create a user message with text and image
    pub fn user_with_image(text: impl Into<String>, media_type: impl Into<String>, base64_data: impl Into<String>) -> Self {
        Self::user(vec![
            ContentBlock::Text { text: text.into() },
            ContentBlock::Image {
                source: ImageSource::Base64 {
                    media_type: media_type.into(),
                    data: base64_data.into(),
                },
            },
        ])
    }
}

/// Tool definition for Anthropic
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// Completion request with tools
#[derive(Debug, Clone)]
pub struct CompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub anthropic_messages: Vec<AnthropicMessage>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub stream: bool,
    pub tools: Vec<serde_json::Value>,
    /// Context window limit (for Ollama `num_ctx`). If set, limits the context
    /// window to reduce VRAM usage and allow more model layers on GPU.
    pub context_limit: Option<u32>,
    /// Extended thinking configuration (passed through to Anthropic API)
    pub thinking: Option<ThinkingConfig>,
}

impl Default for CompletionRequest {
    fn default() -> Self {
        Self {
            model: "claude-sonnet-5".to_string(),
            messages: Vec::new(),
            anthropic_messages: Vec::new(),
            max_tokens: Some(4096),
            temperature: Some(0.7),
            stream: false,
            tools: Vec::new(),
            context_limit: None,
            thinking: None,
        }
    }
}

impl CompletionRequest {
    /// Set extended thinking configuration
    #[must_use]
    pub const fn with_thinking(mut self, thinking: ThinkingConfig) -> Self {
        self.thinking = Some(thinking);
        self
    }

    /// Add tools to the request (Anthropic format)
    #[must_use]
    pub fn with_tools(mut self, tools: Vec<serde_json::Value>) -> Self {
        self.tools = tools;
        self
    }

    /// Add an Anthropic-format message (for tool use/results)
    #[must_use]
    pub fn with_anthropic_message(mut self, msg: AnthropicMessage) -> Self {
        self.anthropic_messages.push(msg);
        self
    }
}

/// How the model should surface its reasoning.
///
/// `Summarized` returns readable summary text in the thinking blocks;
/// `Omitted` still emits thinking blocks but leaves their text empty. The raw
/// chain of thought is never returned under either setting, and both are
/// billed the same — this controls visibility only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingDisplay {
    /// Readable summary of the reasoning.
    Summarized,
    /// Thinking blocks arrive with empty text.
    Omitted,
}

/// Extended thinking configuration.
///
/// Three wire shapes, and which ones a model accepts is part of that model's
/// request contract — see [`AnthropicModelContract`]. Sending the wrong shape
/// is a 400, not a degraded response:
///
/// - `{"type":"adaptive"}` — Claude 4.6 and newer. The model decides depth per
///   request; there is no token budget to set.
/// - `{"type":"enabled","budget_tokens":N}` — pre-4.6 models **only**.
///   `budget_tokens` was removed on Opus 5, Opus 4.8, Opus 4.7, Sonnet 5, and
///   Fable 5; sending it to any of them fails the request.
/// - `{"type":"disabled"}` — explicit opt-out, where the model accepts one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ThinkingConfig {
    /// Model-chosen thinking depth (Claude 4.6+).
    Adaptive {
        /// Omitted from the wire when `None`, which takes the model's default.
        #[serde(skip_serializing_if = "Option::is_none")]
        display: Option<ThinkingDisplay>,
    },
    /// Fixed thinking budget — pre-4.6 models only.
    Enabled {
        /// Budget in tokens. Must be strictly less than the request's
        /// `max_tokens`, minimum 1024.
        budget_tokens: u32,
    },
    /// Explicit opt-out.
    Disabled,
}

impl ThinkingConfig {
    /// A fixed-budget config, for pre-4.6 models.
    #[must_use]
    pub const fn enabled(budget_tokens: u32) -> Self {
        Self::Enabled { budget_tokens }
    }

    /// An adaptive config that asks for readable reasoning summaries.
    #[must_use]
    pub const fn adaptive_summarized() -> Self {
        Self::Adaptive {
            display: Some(ThinkingDisplay::Summarized),
        }
    }

    /// An adaptive config that takes the model's own `display` default.
    #[must_use]
    pub const fn adaptive() -> Self {
        Self::Adaptive { display: None }
    }

    /// The fixed budget, if this is a budgeted config.
    #[must_use]
    pub const fn budget_tokens(&self) -> Option<u32> {
        match self {
            Self::Enabled { budget_tokens } => Some(*budget_tokens),
            _ => None,
        }
    }

    /// Whether this config asks the model to reason.
    #[must_use]
    pub const fn is_thinking(&self) -> bool {
        matches!(self, Self::Adaptive { .. } | Self::Enabled { .. })
    }
}

/// The parts of a Claude model's request contract that differ by model family
/// and fail closed — send the wrong shape and the API returns 400 rather than
/// ignoring the field.
///
/// Derived from the published per-model table; the two axes that bite are the
/// `thinking` shape and whether sampling parameters still exist, so those are
/// the two fields. Callers that want one fact at a time read it through the
/// accessors below.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnthropicModelContract {
    /// Which `thinking` shape the model accepts, and how it behaves.
    pub thinking: AnthropicThinkingContract,
    /// `temperature`, `top_p`, and `top_k` were removed; sending one is a 400.
    pub sampling_removed: bool,
}

/// The `thinking` axis of an [`AnthropicModelContract`].
///
/// The display default and the always-on rule are properties of adaptive
/// thinking — no budgeted model has either — so they live inside that variant
/// rather than beside it as flags that would be meaningless on a legacy model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnthropicThinkingContract {
    /// Pre-4.6: takes `{"type":"enabled","budget_tokens":N}`, not adaptive.
    Budgeted,
    /// Takes `{"type":"adaptive"}`; `budget_tokens` is rejected.
    Adaptive {
        /// What `thinking.display` means when the request leaves it off.
        /// `Omitted` on the current generation, so reasoning text only arrives
        /// if `"summarized"` is asked for explicitly; on the 4.6 family the
        /// default is already `Summarized`.
        default_display: ThinkingDisplay,
        /// Thinking cannot be turned off — `{"type":"disabled"}` is rejected,
        /// so the only legal request is one with no `thinking` field.
        always_on: bool,
    },
}

impl AnthropicModelContract {
    /// The pre-4.6 contract: fixed thinking budgets, sampling parameters live.
    const LEGACY: Self = Self {
        thinking: AnthropicThinkingContract::Budgeted,
        sampling_removed: false,
    };

    /// The current-generation contract: adaptive only, reasoning hidden unless
    /// asked for, sampling removed, an explicit disable still accepted.
    const CURRENT: Self = Self {
        thinking: AnthropicThinkingContract::Adaptive {
            default_display: ThinkingDisplay::Omitted,
            always_on: false,
        },
        sampling_removed: true,
    };

    /// Takes `{"type":"adaptive"}`; `budget_tokens` is rejected.
    #[must_use]
    pub const fn adaptive_thinking(self) -> bool {
        matches!(self.thinking, AnthropicThinkingContract::Adaptive { .. })
    }

    /// `thinking.display` defaults to `"omitted"`, so reasoning text only
    /// arrives if `"summarized"` is asked for explicitly. False on the 4.6
    /// family, whose default is already `"summarized"`, and on budgeted models.
    #[must_use]
    pub const fn display_defaults_omitted(self) -> bool {
        matches!(
            self.thinking,
            AnthropicThinkingContract::Adaptive { default_display: ThinkingDisplay::Omitted, .. }
        )
    }

    /// Thinking cannot be turned off — `{"type":"disabled"}` is rejected, so
    /// the only legal request is one with no `thinking` field.
    #[must_use]
    pub const fn thinking_always_on(self) -> bool {
        matches!(self.thinking, AnthropicThinkingContract::Adaptive { always_on: true, .. })
    }
}

/// Classify a Claude model into its request contract.
///
/// Matching is by family substring rather than an exact allow-list so dated
/// snapshots (`claude-sonnet-4-5-20250929`) and the Bedrock `anthropic.`
/// prefix land in the right family. The patterns cannot collide across
/// generations: `"sonnet-5"` does not occur inside `"sonnet-4-5"`, because the
/// character after `sonnet` differs.
///
/// Unknown `claude-*` models get the **current-generation** contract, not the
/// legacy one. Every model released since 4.6 has removed `budget_tokens` and
/// the sampling parameters, so an unrecognized name is far more likely to be
/// newer than the table than older than it — and guessing legacy for a new
/// model produces a hard 400 on the very first request.
#[must_use]
pub fn anthropic_model_contract(model: &str) -> AnthropicModelContract {
    // Strip the provider prefix BEFORE folding dots, or the Bedrock
    // `anthropic.` form has already lost its separator and never matches.
    let lowered = model.to_ascii_lowercase();
    let unprefixed = lowered
        .strip_prefix("anthropic.")
        .or_else(|| lowered.strip_prefix("anthropic/"))
        .unwrap_or(&lowered);
    // Fold dotted aliases (`claude-opus-4.6`) onto the hyphenated ids the
    // patterns below are written against.
    let name = unprefixed.replace('.', "-");
    let name = name.as_str();

    // Mythos Preview is a migration SOURCE, not a member of the Mythos 5
    // contract: the published "before" example for the Fable 5 migration is
    // `claude-mythos-preview` with `{"type":"enabled","budget_tokens":N}`,
    // i.e. removing budgets is a break *relative to* it, and its prompt-cache
    // minimum is grouped with Opus 4.7 rather than with Fable/Mythos 5. It
    // must not inherit the always-on row below.
    if name.contains("mythos-preview") {
        return AnthropicModelContract::LEGACY;
    }

    // Thinking is always on and cannot be disabled.
    if name.contains("fable-5") || name.contains("mythos") {
        return AnthropicModelContract {
            thinking: AnthropicThinkingContract::Adaptive {
                default_display: ThinkingDisplay::Omitted,
                always_on: true,
            },
            sampling_removed: true,
        };
    }

    // Adaptive-only, sampling removed, reasoning hidden unless asked for.
    if name.contains("opus-5")
        || name.contains("sonnet-5")
        || name.contains("opus-4-8")
        || name.contains("opus-4-7")
    {
        return AnthropicModelContract::CURRENT;
    }

    // The 4.6 family: adaptive, but sampling still works and reasoning
    // summaries are already the default.
    if name.contains("opus-4-6") || name.contains("sonnet-4-6") {
        return AnthropicModelContract {
            thinking: AnthropicThinkingContract::Adaptive {
                default_display: ThinkingDisplay::Summarized,
                always_on: false,
            },
            sampling_removed: false,
        };
    }

    // Pre-4.6 generations keep fixed budgets and sampling. The `-4-20…` arms
    // catch dated Claude 4.0 snapshots (`claude-sonnet-4-20250514`), whose
    // family digit is followed straight by the date.
    if name.contains("opus-4-5")
        || name.contains("opus-4-1")
        || name.contains("opus-4-0")
        || name.contains("opus-4-20")
        || name.contains("sonnet-4-5")
        || name.contains("sonnet-4-0")
        || name.contains("sonnet-4-20")
        || name.contains("haiku-4-5")
        || name.contains("haiku-4-20")
        || name.contains("claude-3")
        || name.contains("claude-2")
    {
        return AnthropicModelContract::LEGACY;
    }

    // Unknown: assume current generation. See the doc comment.
    AnthropicModelContract::CURRENT
}

/// Whether `model` names a Claude model, as opposed to an Ollama tag, an
/// `OpenRouter` slug for someone else's model, or an `OpenAI` id.
///
/// [`anthropic_model_contract`] answers "which Claude contract"; it assumes
/// the answer here is already yes. Callers that resolve a model dynamically —
/// the summarizer and distiller pick one from `summarization_priority` and
/// fall back to the main model — have to ask this first, or a local
/// `qwen3.5:9b` would be classified as an unrecognized Claude model and lose
/// its sampling parameters.
///
/// `OpenRouter`'s `anthropic/claude-*` slugs count: that path proxies to the
/// same API and rejects the same removed parameters.
#[must_use]
pub fn is_claude_model(model: &str) -> bool {
    model.to_ascii_lowercase().contains("claude")
}

/// Conform a request to its model's contract, at the boundary where it is
/// about to be sent.
///
/// Per-call-site fixes do not hold this invariant: a request reaches the API
/// from many constructors (`AnthropicRequest` literals, and every
/// `CompletionRequest` that `complete_anthropic_simple` converts), and each
/// new one is another chance to send a parameter the model removed. This runs
/// on the dispatch path so every caller is covered at once, including ones
/// added later.
///
/// Two corrections, both faithful translations of what the calling code
/// already meant rather than new policy:
///
/// 1. **`temperature` is dropped** where the model removed sampling, and
///    alongside any active thinking config. `CompletionRequest::default()`
///    carries `Some(0.7)`, so sub-agent questions, multi-agent decomposition
///    and aggregation, dream summaries and `AgentContext::compress` all
///    inherit a temperature nobody chose — a 400 on the current generation.
/// 2. **`thinking: None` becomes an explicit `disabled`** where the model
///    accepts one. Omitting the field used to mean "no thinking" everywhere,
///    which is what every `thinking: None` in this codebase was written to
///    mean; on Opus 5, Sonnet 5 and Fable it now means *adaptive*. Left as-is
///    the model reasons inside a `max_tokens` sized for no reasoning (512 for
///    the distiller, 1024 for the tool-output summarizer) and returns
///    `stop_reason: "max_tokens"` with empty text — a silent truncation that
///    is harder to spot than the 400 it replaced. Always-on models keep
///    `None`, since they reject an explicit disable.
///
/// Borrowed unchanged when the request already conforms.
#[must_use]
pub fn conform_to_anthropic_contract(
    request: &AnthropicRequest,
) -> std::borrow::Cow<'_, AnthropicRequest> {
    /// How `thinking` must change for the target model.
    enum Fix {
        Leave,
        Set(ThinkingConfig),
        Clear,
    }

    if !is_claude_model(&request.model) {
        return std::borrow::Cow::Borrowed(request);
    }
    let contract = anthropic_model_contract(&request.model);

    // How `thinking` must change for THIS model.
    //
    // The shape is rewritten, not merely filled in, because the model on a
    // request is not fixed once it is built: `route_model` replaces it after
    // the fact, so a shape derived from the configured model can arrive here
    // attached to a different generation. Filling in only the `None` case
    // would let a `budget_tokens` body reach a model that rejects it.
    let fix = match &request.thinking {
        // Omission used to mean "no thinking" everywhere; on adaptive models it
        // now means adaptive. Make that original intent explicit where the
        // model accepts an explicit off.
        None if contract.adaptive_thinking() && !contract.thinking_always_on() => {
            Fix::Set(ThinkingConfig::Disabled)
        }
        // A fixed budget on a model that removed `budget_tokens`.
        Some(ThinkingConfig::Enabled { .. }) if contract.adaptive_thinking() => {
            Fix::Set(if contract.display_defaults_omitted() {
                ThinkingConfig::adaptive_summarized()
            } else {
                ThinkingConfig::adaptive()
            })
        }
        // An adaptive request on a model that predates adaptive. This layer
        // knows the wire contract but not the step's output reserve, so it
        // cannot synthesize an honest `budget_tokens`; dropping the field is
        // the valid degradation, since omission means no thinking on these
        // models. Reaching this means a caller skipped its own derivation.
        Some(ThinkingConfig::Adaptive { .. }) if !contract.adaptive_thinking() => {
            tracing::debug!(
                model = %request.model,
                "adaptive thinking requested on a pre-4.6 model; sending no thinking field"
            );
            Fix::Clear
        }
        // An explicit off on a model that rejects one.
        Some(ThinkingConfig::Disabled) if contract.thinking_always_on() => Fix::Clear,
        _ => Fix::Leave,
    };

    let effective_thinking = match &fix {
        Fix::Set(c) => Some(c),
        Fix::Clear => None,
        Fix::Leave => request.thinking.as_ref(),
    };
    let thinking_active = effective_thinking.is_some_and(ThinkingConfig::is_thinking);
    let drop_temperature =
        request.temperature.is_some() && (contract.sampling_removed || thinking_active);

    if !drop_temperature && matches!(fix, Fix::Leave) {
        return std::borrow::Cow::Borrowed(request);
    }

    let mut conformed = request.clone();
    if drop_temperature {
        conformed.temperature = None;
    }
    match fix {
        Fix::Set(c) => conformed.thinking = Some(c),
        Fix::Clear => conformed.thinking = None,
        Fix::Leave => {}
    }
    std::borrow::Cow::Owned(conformed)
}

/// The `temperature` to actually send for `model`, given what the caller
/// wanted. `None` means the field must be omitted entirely.
///
/// `temperature` was removed on Opus 5 / 4.8 / 4.7, Sonnet 5, and Fable 5 —
/// sending one is a 400, so a hard-coded temperature on an auxiliary request
/// (summarize, distill, score, extract) fails that request outright once the
/// model it resolves to is current-generation.
#[must_use]
pub fn sampling_temperature_for_model(model: &str, requested: f32) -> Option<f32> {
    if is_claude_model(model) && anthropic_model_contract(model).sampling_removed {
        return None;
    }
    Some(requested)
}

/// Full Anthropic request with tools
#[derive(Debug, Clone, Serialize)]
pub struct AnthropicRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    pub max_tokens: u32,
    /// Context window to ask the local runner for (Ollama `num_ctx`).
    ///
    /// Not part of the Anthropic wire format — skipped on serialization — but
    /// it has to ride with the request because only the caller knows how much
    /// context this model can be given on THIS machine. A 12B model's KV cache
    /// at 32k does not fit beside its weights on a 16 GB card already lending
    /// 5 GB to the desktop, and the overrun arrives as
    /// `CUDA error: an illegal memory access was encountered` rather than a
    /// clean allocation failure.
    #[serde(skip_serializing, default)]
    pub context_limit: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Extended thinking configuration (Claude 3.5+ models)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,
    /// Prompt caching: automatic mode caches the longest prefix up to the last cacheable block.
    /// Set to `Some(CacheControl::ephemeral())` to enable automatic prompt caching.
    /// Cached input tokens cost 90% less on Anthropic, 50% less on `OpenAI`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

/// Cache control configuration for prompt caching.
///
/// When set at the request level, enables automatic caching: the API caches all content
/// up to and including the last cacheable block. On subsequent requests with the same
/// prefix, cached content is reused automatically.
///
/// **One TTL per request.** Anthropic rejects a request whose longer-TTL breakpoints do
/// not precede its shorter ones, and one whose explicit marker on the last block disagrees
/// with the top-level TTL. Every marker Nanna emits for a request is derived from that
/// request's single `CacheControl`, which keeps both rules true by construction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheControl {
    #[serde(rename = "type")]
    pub cache_type: String,
    /// `None` is the API default (5 minutes); only a 1-hour TTL is ever sent explicitly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl: Option<CacheTtl>,
}

/// Prompt-cache lifetime on Anthropic.
///
/// A 1-hour write costs 2x base input (5-minute: 1.25x); reads cost 0.1x either way and
/// refresh the timer. So 1-hour pays only when requests sharing a prefix start 5-60 minutes
/// apart — heartbeat and cron gaps on the cloud escape hatch — and needs at least three
/// requests per write to break even.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum CacheTtl {
    /// The API default.
    #[default]
    #[serde(rename = "5m")]
    FiveMinutes,
    #[serde(rename = "1h")]
    OneHour,
}

impl CacheControl {
    /// Create an ephemeral cache control with the API's default ~5 minute TTL.
    #[must_use]
    pub fn ephemeral() -> Self {
        Self::ephemeral_for(CacheTtl::FiveMinutes)
    }

    /// Create an ephemeral cache control with `ttl`.
    ///
    /// The 5-minute default is sent as no `ttl` key at all, so a default request is
    /// byte-identical to the one sent before the TTL was configurable.
    #[must_use]
    pub fn ephemeral_for(ttl: CacheTtl) -> Self {
        let explicit = match ttl {
            CacheTtl::FiveMinutes => None,
            CacheTtl::OneHour => Some(CacheTtl::OneHour),
        };
        let control = Self {
            cache_type: "ephemeral".to_string(),
            ttl: explicit,
        };
        debug_assert_eq!(control.effective_ttl(), ttl, "the TTL must round-trip");
        debug_assert_ne!(
            control.ttl,
            Some(CacheTtl::FiveMinutes),
            "5m stays implicit"
        );
        control
    }

    /// The TTL the API applies, resolving an absent key to its 5-minute default.
    #[must_use]
    pub fn effective_ttl(&self) -> CacheTtl {
        self.ttl.unwrap_or_default()
    }
}

/// System prompt block (for OAuth/Claude Code format)
#[derive(Debug, Clone, Serialize)]
pub struct SystemBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    pub text: String,
    /// Cache control for this specific block (enables explicit cache breakpoints).
    /// Place on the last system block to cache the entire system prompt prefix.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

impl SystemBlock {
    /// Create a text system block
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            block_type: "text".to_string(),
            text: content.into(),
            cache_control: None,
        }
    }

    /// Create a text system block with ephemeral cache control
    pub fn text_cached(content: impl Into<String>) -> Self {
        Self {
            block_type: "text".to_string(),
            text: content.into(),
            cache_control: Some(CacheControl::ephemeral()),
        }
    }
}

/// Anthropic request with array-based system prompt (required for OAuth/Claude Code)
#[derive(Debug, Clone, Serialize)]
struct OAuthAnthropicRequest {
    model: String,
    messages: Vec<AnthropicMessage>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    system: Vec<SystemBlock>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<ThinkingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<CacheControl>,
}

impl OAuthAnthropicRequest {
    /// Convert from a standard `AnthropicRequest`, prepending Claude Code identity
    fn from_request(request: &AnthropicRequest, prepend_identity: bool) -> Self {
        let mut system = Vec::new();

        if prepend_identity {
            system.push(SystemBlock::text(CLAUDE_CODE_IDENTITY));
        }

        if let Some(ref sys) = request.system
            && !sys.is_empty() {
                system.push(SystemBlock::text(sys));
            }

        // Place explicit cache breakpoint on the last system block, so the system prompt
        // prefix is cached server-side. It carries the request's own marker rather than a
        // fresh default: a 5-minute breakpoint ahead of a 1-hour top-level one is a 400
        // (longer TTLs must come first), so one request gets exactly one TTL.
        if let Some(ref control) = request.cache_control
            && let Some(last) = system.last_mut() {
                last.cache_control = Some(control.clone());
            }
        debug_assert!(
            system
                .iter()
                .filter_map(|block| block.cache_control.as_ref())
                .all(|marker| Some(marker) == request.cache_control.as_ref()),
            "every system-block breakpoint must match the request-level TTL"
        );

        Self {
            model: request.model.clone(),
            messages: request.messages.clone(),
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            system,
            tools: request.tools.clone(),
            stream: request.stream,
            thinking: request.thinking.clone(),
            cache_control: request.cache_control.clone(),
        }
    }
}

/// Anthropic API response
#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub response_type: String,
    pub role: String,
    pub content: Vec<ContentBlock>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    /// Tokens written to the prompt cache on this request (billed at cache write rate)
    #[serde(default)]
    pub cache_creation_input_tokens: u32,
    /// Tokens read from the prompt cache on this request (billed at 10% of input rate on Anthropic)
    #[serde(default)]
    pub cache_read_input_tokens: u32,
    /// Per-lifetime split of `cache_creation_input_tokens`. Absent on older responses and
    /// on other providers, which then count as all 5-minute writes.
    #[serde(default)]
    pub cache_creation: Option<CacheCreation>,
}

/// Anthropic's `usage.cache_creation`: the cache writes of one request, by TTL.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
pub struct CacheCreation {
    #[serde(default)]
    pub ephemeral_5m_input_tokens: u32,
    #[serde(default)]
    pub ephemeral_1h_input_tokens: u32,
}

impl Usage {
    /// The 1-hour share of `cache_creation_input_tokens`; 0 when the split is absent.
    #[must_use]
    pub fn cache_creation_1h_tokens(&self) -> u32 {
        hour_share(
            self.cache_creation.as_ref(),
            self.cache_creation_input_tokens,
        )
    }
}

/// The 1-hour share of a cache-write total. Provider data, not an invariant of ours, so an
/// inconsistent split is clamped to the total it is part of rather than asserted.
fn hour_share(split: Option<&CacheCreation>, total: u32) -> u32 {
    let reported = split.map_or(0, |s| s.ephemeral_1h_input_tokens);
    let share = reported.min(total);
    debug_assert!(share <= total, "a share never exceeds its total");
    debug_assert!(share <= reported, "clamping only ever lowers the share");
    share
}

// ============================================================================
// Claude Code Stealth Mode
// ============================================================================

/// Claude Code version to mimic (update as needed)
const CLAUDE_CODE_VERSION: &str = "2.1.273";

/// Claude Code canonical tool names (case-sensitive)
/// Source: <https://cchistory.mariozechner.at/data/prompts-2.1.11.md>
const CLAUDE_CODE_TOOLS: &[(&str, &str)] = &[
    ("read", "Read"),
    ("read_file", "Read"),
    ("write", "Write"),
    ("write_file", "Write"),
    ("edit", "Edit"),
    ("edit_file", "Edit"),
    ("bash", "Bash"),
    ("exec", "Bash"),
    ("shell", "Bash"),
    ("grep", "Grep"),
    ("glob", "Glob"),
    ("list_dir", "Glob"),
    ("ask_user", "AskUserQuestion"),
    ("web_fetch", "WebFetch"),
    ("web_search", "WebSearch"),
    ("notebook_edit", "NotebookEdit"),
];

/// Claude Code identity prefix (REQUIRED for OAuth tokens)
const CLAUDE_CODE_IDENTITY: &str = "You are Claude Code, Anthropic's official CLI for Claude.";

/// Check if a token is an OAuth token (vs API key)
fn is_oauth_token(token: &str) -> bool {
    // OAuth tokens start with sk-ant-oat (OAuth Access Token)
    // Regular API keys start with sk-ant-api
    token.starts_with("sk-ant-oat") || token.starts_with("oauth:")
}

/// Get the raw token (stripping any prefix)
fn get_raw_token(token: &str) -> &str {
    token.strip_prefix("oauth:").unwrap_or(token)
}

/// Convert a tool name to Claude Code canonical form
fn to_claude_code_tool_name(name: &str) -> String {
    let lower = name.to_lowercase();
    for (pattern, canonical) in CLAUDE_CODE_TOOLS {
        if lower == *pattern {
            return (*canonical).to_string();
        }
    }
    // If no match, return original (with first letter capitalized for consistency)
    let mut chars = name.chars();
    chars.next().map_or_else(String::new, |c| c.to_uppercase().chain(chars).collect())
}

/// Convert a tool name from Claude Code form back to original
fn from_claude_code_tool_name(name: &str, original_tools: &[ToolDefinition]) -> String {
    let lower = name.to_lowercase();
    // Find original tool by matching lowercase
    for tool in original_tools {
        if tool.name.to_lowercase() == lower {
            return tool.name.clone();
        }
    }
    // Check Claude Code canonical names and map back
    for (pattern, canonical) in CLAUDE_CODE_TOOLS {
        if name == *canonical {
            // Return the pattern as-is (caller should have original)
            return (*pattern).to_string();
        }
    }
    name.to_string()
}

/// LLM client
#[derive(Clone)]
pub struct LlmClient {
    http: Client,
    provider: Provider,
    api_key: String,
    base_url: String,
}

impl LlmClient {
    /// Build HTTP client with sensible timeouts
    /// Remote-provider HTTP client.
    ///
    /// No TOTAL request deadline, for the same reason the Ollama client below
    /// has none: `reqwest`'s `.timeout()` covers the response BODY, so it
    /// aborts a healthy stream that is still delivering tokens, purely for
    /// taking too long overall. A 120-second cap did that to any turn longer
    /// than two minutes — and long turns are now the normal case, with
    /// thinking on by default, real context windows in place of the phantom
    /// 32k floor, and a ceiling that covers reasoning as well as the answer.
    /// The failure surfaced as a stream error after the full input cost had
    /// already been paid, and read as a model fault rather than a client one.
    ///
    /// A genuine stall is still caught, by silence BETWEEN chunks
    /// (`read_timeout`), which a live stream never hits. Anthropic's own SDKs
    /// default to a 10-minute total for the same reason.
    fn build_http_client() -> Client {
        Client::builder()
            .read_timeout(std::time::Duration::from_secs(STREAM_READ_TIMEOUT_SECS))
            .connect_timeout(std::time::Duration::from_secs(10))
            .build()
            .unwrap_or_else(|_| Client::new())
    }

    /// Ollama-specific HTTP client. Local streaming generations legitimately
    /// run for minutes, so no TOTAL request deadline — a genuine stall is
    /// caught by the read timeout (silence between chunks) instead, which a
    /// healthy stream never hits. Idle connections are never pooled: reusing
    /// a loopback connection the OS/filter stack half-closed surfaces as a
    /// clean EOF mid-stream (observed live 2026-07-23 as "no done=true"
    /// faults while the server logged a client cancel of an actively
    /// decoding task); a fresh connection per request costs sub-millisecond
    /// on loopback.
/// Gemma's reserved stop sentinels: `<unused0>` through `<unused98>`.
///
/// These are the model's own end-of-turn markers. Ollama's chat template is
/// meant to consume them; when it does not, they arrive as ordinary assistant
/// content immediately before the stream closes without a `done=true` message.
/// Recognising one lets the client close the turn cleanly instead of treating a
/// finished reply as an aborted generation.
///
/// Deliberately exact: the whole delta must be the sentinel. A reply that merely
/// mentions `<unused50>` in prose is text, not a stop.
/// The text before a trailing Gemma stop sentinel, if there is one.
///
/// Returns `None` when the content does not end in a sentinel, so callers can
/// pass ordinary text through untouched.
fn strip_trailing_stop_sentinel(content: &str) -> Option<&str> {
    let trimmed = content.trim_end();
    let open = trimmed.rfind("<unused")?;
    if !trimmed.ends_with('>') {
        return None;
    }
    let digits = &trimmed[open + "<unused".len()..trimmed.len() - 1];
    if digits.is_empty() || digits.len() > 2 || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some(&content[..open])
}

fn is_gemma_stop_sentinel(content: &str) -> bool {
    let trimmed = content.trim();
    let Some(digits) = trimmed
        .strip_prefix("<unused")
        .and_then(|rest| rest.strip_suffix('>'))
    else {
        return false;
    };
    !digits.is_empty() && digits.len() <= 2 && digits.bytes().all(|b| b.is_ascii_digit())
}

    /// The `num_ctx` to ask Ollama for on THIS request.
    ///
    /// Every Ollama call site must go through here. The streaming and
    /// non-streaming builders are separate code paths that both talk to the
    /// same runner on the same GPU, and a sizing rule applied to only one of
    /// them is not a rule — it is a coin flip decided by which path the caller
    /// happened to take. (Observed: the first cut of this landed on the
    /// non-streaming builder alone, while the agent loop streams exclusively,
    /// so it would have had no effect on the thing it was written to fix.)
    ///
    /// Source order: what the caller explicitly asked for, then the size
    /// latched for this server, then the env pin outright (falling back to the
    /// probed/nominal size only when no pin is set) at latch initialization.
    /// Only a server on this machine is probed — see
    /// [`Self::unpinned_start_num_ctx`].
    fn resolve_num_ctx(base_url: &str, model: &str, explicit: Option<u32>) -> u32 {
        Self::resolve_num_ctx_with(base_url, model, explicit, |model| {
            Self::fit_context_to_free_vram(base_url, model)
        })
    }

    /// [`Self::resolve_num_ctx`] with the fit to this machine's GPU passed in,
    /// so the latch can be exercised without a card.
    fn resolve_num_ctx_with(
        base_url: &str,
        model: &str,
        explicit: Option<u32>,
        fit_to_this_gpu: impl FnOnce(&str) -> Option<u32>,
    ) -> u32 {
        if let Some(explicit) = explicit {
            return explicit;
        }
        // The latch is this server's: a size fitted for another server says
        // nothing about this one's VRAM (see `ctx_latch`).
        if let Some(latched) = Self::latched_num_ctx(base_url, model) {
            return latched;
        }
        // Explicit beats computed — in BOTH directions. `NANNA_OLLAMA_NUM_CTX`
        // is set by someone who measured THIS machine with THIS model;
        // `fit_context_to_free_vram` has only inferred a size from a SNAPSHOT
        // of free bytes. Both override directions have now been paid for:
        // the heuristic promoting past the operator's number cost a mission
        // (31/42 at the pinned 16384 vs 8/42 at the promoted 32768 — a 9B
        // model given twice the window loses the thread), and the heuristic
        // DEMOTING below it cost another (2026-08-09: a summarization burst
        // held the card while the model's first request sized, the min(env,
        // sized) start latched 4096 for the daemon's lifetime, and every
        // later mission request inherited the poisoned latch on an
        // otherwise-idle card).
        //
        // A transient probe must not override a durable measurement in
        // either direction. A pin that is genuinely too large for the card
        // still faults under load and gets walked down by `demote_context`
        // (the caller demotes on the second fault of a run): the safety net
        // for an over-pin is the fault ladder, not the snapshot.
        let start = Self::env_num_ctx().map_or_else(
            || Self::unpinned_start_num_ctx(base_url, model, fit_to_this_gpu),
            |pinned| {
                tracing::info!(
                    model = %model,
                    pinned,
                    source = "env",
                    "NANNA_OLLAMA_NUM_CTX pins the starting num_ctx latch; \
                     demote_context can still walk it lower on GPU faults"
                );
                pinned
            },
        );
        Self::latch_num_ctx_for(base_url, model, start)
    }

    /// The starting `num_ctx` when neither the caller nor the operator chose
    /// one: `fit_to_this_gpu`'s fit to this machine's free VRAM, or 16384 when
    /// the card cannot be read.
    ///
    /// Only a server on THIS machine is sized from this machine's card. A
    /// remote server loads the model onto its own GPU, whose free memory
    /// nothing here can see, and this machine's figure is not an estimate of
    /// it but an unrelated number: a desktop card busy with a game would pin
    /// a remote server with room to spare at the floor. So a remote server
    /// starts where an unreadable card starts, and a GPU-memory fault from it
    /// still walks the latch down through [`Self::demote_context`].
    fn unpinned_start_num_ctx(
        base_url: &str,
        model: &str,
        fit_to_this_gpu: impl FnOnce(&str) -> Option<u32>,
    ) -> u32 {
        if !ollama_server_is_local(base_url) {
            tracing::info!(
                model,
                server = base_url,
                num_ctx = Self::UNKNOWN_VRAM_NUM_CTX,
                "not sizing num_ctx to this machine's GPU: the Ollama server is not on this \
                 machine, so the model starts at the size used when VRAM is unknown"
            );
            return Self::UNKNOWN_VRAM_NUM_CTX;
        }
        fit_to_this_gpu(model).unwrap_or(Self::UNKNOWN_VRAM_NUM_CTX)
    }

    /// The start when the VRAM the model will load into is unknown: this
    /// machine's card cannot be read, or the server is on another machine.
    const UNKNOWN_VRAM_NUM_CTX: u32 = 16_384;

    /// The start a model's first request to `base_url` will latch, when it is
    /// known without measuring anything: the operator's pin, else, for a
    /// server not on this machine, [`Self::UNKNOWN_VRAM_NUM_CTX`]. `None` for
    /// a server on this machine with nothing pinned, whose start is this
    /// card's fit. Measuring shells out to `nvidia-smi` and asks the server
    /// what it holds, which a reader polled on every agent iteration must not
    /// do, so it is left to the first request.
    ///
    /// The precedence is [`Self::resolve_num_ctx_with`]'s, without its
    /// announcements, so a window read before a request is the `num_ctx` that
    /// request carries.
    fn unmeasured_start_num_ctx(base_url: &str) -> Option<u32> {
        Self::env_num_ctx()
            .or_else(|| (!ollama_server_is_local(base_url)).then_some(Self::UNKNOWN_VRAM_NUM_CTX))
    }

    /// The operator's explicit context size, if they set one.
    fn env_num_ctx() -> Option<u32> {
        std::env::var("NANNA_OLLAMA_NUM_CTX")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .filter(|v| *v >= 2048)
    }

    /// The `num_ctx` latched for each model on each Ollama server, keyed by
    /// [`Self::ctx_latch_key`].
    ///
    /// Per server because what a size is fitted to is per server: one on this
    /// machine is fitted to this machine's free VRAM, one elsewhere starts at
    /// the size for unknown VRAM, and each is walked down by its own
    /// GPU-memory faults. Keyed by model alone, the latch served one server's
    /// size to another: a local fit taken while a game held this machine's
    /// card went on sizing the remote server Settings then pointed chat at,
    /// and a summarizer's chunks were cut for chat's server while its requests
    /// carried its own, smaller size. So every reader names its server, which
    /// a client does by being one. Bounded by the servers the configuration
    /// has named this process, times the models sent to them.
    fn ctx_latch() -> &'static std::sync::Mutex<std::collections::HashMap<(String, String), u32>> {
        static LATCH: std::sync::OnceLock<
            std::sync::Mutex<std::collections::HashMap<(String, String), u32>>,
        > = std::sync::OnceLock::new();
        LATCH.get_or_init(Default::default)
    }

    /// A latch key: the server's base URL and the bare model name. The request
    /// carries `gemma4:12b` while callers hold the routed `ollama/gemma4:12b`,
    /// and two spellings of one model means the demotion writes an entry the
    /// request path never reads — a self-correcting loop that silently
    /// corrects nothing.
    fn ctx_latch_key(server: &str, model: &str) -> (String, String) {
        (server.to_string(), Self::bare_model(model).to_string())
    }

    fn bare_model(model: &str) -> &str {
        model.strip_prefix("ollama/").unwrap_or(model)
    }

    /// The size latched for `model` on `server`; `None` when the model was
    /// never sized there.
    fn latched_num_ctx(server: &str, model: &str) -> Option<u32> {
        Self::ctx_latch()
            .lock()
            .ok()?
            .get(&Self::ctx_latch_key(server, model))
            .copied()
    }

    /// The smallest size latched for `model` on any server.
    fn smallest_latched_num_ctx(model: &str) -> Option<u32> {
        let model = Self::bare_model(model);
        Self::ctx_latch()
            .lock()
            .ok()?
            .iter()
            .filter(|((_, latched_model), _)| latched_model == model)
            .map(|(_, &size)| size)
            .min()
    }

    /// The `num_ctx` a request for `model` to `server` will carry, as far as
    /// that is known without measuring: the size latched there, else the start
    /// [`Self::unmeasured_start_num_ctx`] can name.
    fn known_num_ctx(server: &str, model: &str) -> Option<u32> {
        Self::latched_num_ctx(server, model).or_else(|| Self::unmeasured_start_num_ctx(server))
    }

    /// Latch `ctx` as `model`'s size on `server`, and return the size latched
    /// there. A size another request latched for that server while this one
    /// was being measured is kept: that request already went out with it, and
    /// changing it would make Ollama reload the model.
    fn latch_num_ctx_for(server: &str, model: &str, ctx: u32) -> u32 {
        let Ok(mut latches) = Self::ctx_latch().lock() else {
            return ctx;
        };
        *latches
            .entry(Self::ctx_latch_key(server, model))
            .or_insert(ctx)
    }

    /// Set `model`'s size on `server`, as a sizing or a demotion would leave
    /// it.
    #[cfg(test)]
    fn latch_num_ctx(server: &str, model: &str, ctx: u32) {
        if let Ok(mut latches) = Self::ctx_latch().lock() {
            latches.insert(Self::ctx_latch_key(server, model), ctx);
        }
    }

    /// The effective `num_ctx` for `model` on this client's server: the size
    /// its next request there will carry, when that is known without
    /// measuring (the size latched there, else the operator's pin, else for a
    /// server on another machine the unknown-VRAM start). `None` means the
    /// provider-reported window stands: the client is not Ollama's (only an
    /// Ollama server has a `num_ctx`), or the server is on this machine and
    /// the model's first request has not measured the card yet.
    ///
    /// This is deliberately a poll-getter rather than a change notification:
    /// the latch is process-global state, and the agent loop already has a
    /// natural per-iteration point to re-read it. Callers budgeting prompts
    /// MUST treat the value as live — [`Self::demote_context`] rewrites it
    /// mid-run under VRAM pressure — and must read it through the client that
    /// sends the request: the same model on another server has a size of its
    /// own.
    #[must_use]
    pub fn effective_num_ctx(&self, model: &str) -> Option<u32> {
        if self.provider != Provider::Ollama {
            return None;
        }
        Self::known_num_ctx(&self.base_url, model)
    }

    /// The context window this client's server will honour for `model` right
    /// now: `reported` clamped by [`Self::effective_num_ctx`]. The form of
    /// [`effective_context_window`] to budget a request this client sends.
    #[must_use]
    pub fn effective_context_window(&self, model: &str, reported: usize) -> usize {
        clamp_window(reported, self.effective_num_ctx(model))
    }

    /// `info` with its window clamped to [`Self::effective_context_window`]
    /// and its output cap re-bounded to half of it, as
    /// [`clamp_model_info_to_effective_window`] does.
    #[must_use]
    pub fn clamp_model_info_to_effective_window(&self, model: &str, info: ModelInfo) -> ModelInfo {
        let effective = self.effective_context_window(model, info.context_window);
        clamp_model_info_to_window(model, info, effective)
    }

    /// [`model_info_from_cache_or_unknown`], clamped to the window of this
    /// client's server instead of the smallest window of any server.
    #[must_use]
    pub fn model_info_from_cache_or_unknown(&self, model: &str) -> ModelInfo {
        self.clamp_model_info_to_effective_window(model, cached_model_info_or_unknown(model, ""))
    }

    /// Drop `model` to the next smaller context rung on this client's server
    /// after the runner there failed in a way that means it ran out of GPU
    /// memory. Returns the new size, or `None` when the latch already sits at
    /// (or below) the minimum viable window — demoting past the floor
    /// manufactures a window the agent will loudly refuse, so the fault is
    /// surfaced instead of "healed" into an unusable size. `None` too, with
    /// nothing changed, for a client that is not Ollama's: only an Ollama
    /// server has a `num_ctx`.
    ///
    /// Only this server's size moves. Another server serving the same model
    /// did not fault, and walking its size instead left this one at the size
    /// that faults.
    ///
    /// This is the half of the sizing loop that measurement cannot supply.
    /// Sizing happens on the FIRST request, which is structurally the most
    /// optimistic moment there is — the desktop compositor has not grown, the
    /// embedder has not loaded, and this model's own weights are not yet
    /// resident. Re-measuring on later turns does not fix it: by then free VRAM
    /// is dominated by our own allocation, and acting on it would change
    /// `num_ctx`, which makes Ollama evict and reload the model, which frees
    /// the memory that justified the change. The only signal that is both late
    /// enough to be honest and unambiguous is the failure itself.
    ///
    /// The rung is **3/4 of the current size** (the latched one, else the
    /// start this server's first request would latch, else
    /// [`NUM_CTX_CEILING`]), rounded DOWN to [`NUM_CTX_QUANTUM`], never below
    /// `min_viable_ctx` (rounded UP to the same quantum; `None` falls back to
    /// [`DEFAULT_MIN_VIABLE_NUM_CTX`]).
    /// Derivation of the step: the KV cache scales linearly with `num_ctx`,
    /// so a 25% cut frees ~25% of the KV allocation — comfortably more than
    /// one retry's worth of KV growth, which is what the step must outpace —
    /// while halving (the old ladder) overshoots: it assumed faults repeat
    /// deterministically, but with a sane starting pin a single fault is
    /// usually a transient load blip, and 8192 → 4096 destroyed a viable
    /// window by landing below the agent's ~4.2k pressure-tier floor
    /// (observed live 2026-08-03: 38 below-floor stops, 8 resume cycles,
    /// run dead in 8 minutes). 3/4 is geometric, so it still reaches any
    /// floor in bounded rungs — ⌈log₄∕₃(start/floor)⌉, 7 from 32768 to 4608
    /// — and its 25% cuts can never skip PAST a viable window the way a 50%
    /// cut can.
    pub fn demote_context(&self, model: &str, min_viable_ctx: Option<u32>) -> Option<u32> {
        if self.provider != Provider::Ollama {
            return None;
        }
        let clamp = min_viable_ctx
            .unwrap_or(DEFAULT_MIN_VIABLE_NUM_CTX)
            .div_ceil(NUM_CTX_QUANTUM)
            .saturating_mul(NUM_CTX_QUANTUM);
        let key = Self::ctx_latch_key(&self.base_url, model);
        let mut guard = Self::ctx_latch().lock().ok()?;
        let current = guard
            .get(&key)
            .copied()
            .or_else(|| Self::unmeasured_start_num_ctx(&self.base_url))
            .unwrap_or(NUM_CTX_CEILING);
        if current <= clamp {
            tracing::warn!(
                model,
                server = %self.base_url,
                current,
                floor = clamp,
                "GPU memory failure at the context floor — NOT demoting further; \
                 a smaller window would be below the minimum viable request, so \
                 the fault stays loud (below-floor/fault failures are resumable) \
                 instead of being healed into an unusable window"
            );
            return None;
        }
        let stepped = (current / 4 * 3) / NUM_CTX_QUANTUM * NUM_CTX_QUANTUM;
        let clamped = stepped < clamp;
        let next = if clamped { clamp } else { stepped };
        guard.insert(key, next);
        drop(guard);
        tracing::warn!(
            model,
            server = %self.base_url,
            from = current,
            to = next,
            floor = clamp,
            clamped,
            "GPU memory failure — demoting context to 3/4 (quantized to 512, \
             clamped at the minimum viable window) and retrying; the effective \
             window shrank and downstream budgets re-derive from it (readable \
             via effective_num_ctx / effective_context_window)"
        );
        Some(next)
    }

    /// Floor of the VRAM-fitted context: below this a run is not worth
    /// starting; above it, diminishing returns for an agent whose steps
    /// rarely exceed a few thousand tokens. Shared between the fit
    /// arithmetic (its clamp floor) and the sizing probe's starved-card
    /// check, which keys on "the fit was forced all the way down here".
    const MIN_FIT_CTX: u32 = 4_096;

    /// Largest context that fits in the VRAM free *right now*, or `None` when
    /// the GPU cannot be queried (no NVIDIA tooling, or a non-CUDA backend) —
    /// in which case the caller falls back to a fixed size.
    ///
    /// `base_url` is the server the model runs on, and it must be on this
    /// machine: the card measured is this machine's.
    ///
    /// Deliberately empirical. The analytic route needs the model's
    /// sliding-window interleave and KV quantisation to be known, and getting
    /// those wrong is worse than measuring: for gemma4:12b the naive formula
    /// predicts 1.5 MB/token and the truth is ~10 KB/token, a 150x error. Two
    /// numbers we can actually read — free VRAM and the model's file size —
    /// plus a reserve for the prefill compute buffer, give a bound that adapts
    /// to whatever else is on the card.
    fn fit_context_to_free_vram(base_url: &str, model: &str) -> Option<u32> {
        const MIB: u64 = 1024 * 1024;
        debug_assert!(
            ollama_server_is_local(base_url),
            "only a server on this machine is sized from this machine's GPU"
        );
        let (free_bytes, weights_bytes, (ours_resident, others_resident)) =
            without_starving_the_runtime(|| {
                Some((
                    Self::nvidia_free_vram_bytes()?,
                    Self::ollama_model_size_bytes(base_url, model)?,
                    Self::ollama_resident_vram(base_url, model),
                ))
            })?;
        let fitted = Self::fit_context_for_budget(
            nanna_numeric::f64_from_u64(free_bytes),
            nanna_numeric::f64_from_u64(weights_bytes),
            nanna_numeric::f64_from_u64(ours_resident),
            nanna_numeric::f64_from_u64(others_resident),
        );
        // Floor-fit on an EMPTY server is the orphaned-runner signature, so
        // name it instead of latching quietly. `ollama ps` reports the
        // server's own children, not what is on the card: a runner orphaned
        // by a parent-only server kill (see the regression note on
        // `nanna-daemon`'s `restart_ollama_server`) keeps its ~weights-worth
        // of VRAM while being invisible here. Observed live 2026-08-09: four
        // orphans held 12.5 of 16 GB, `ollama ps` was empty, and this probe
        // honestly latched a below-floor 4096 that killed two endurance
        // attempts. A card that cannot fit more than the floor while Ollama
        // holds NOTHING is either starved by invisible consumers or too small
        // for the model — both are worth a loud line before the latch sticks.
        if fitted == Self::MIN_FIT_CTX && ours_resident + others_resident == 0 {
            tracing::warn!(
                model,
                // Integer division: identical to the float quotient's truncation
                // for every byte count below 2^53 (8 PiB).
                free_mib = free_bytes / MIB,
                weights_mib = weights_bytes / MIB,
                "num_ctx sized at the floor while `ollama ps` shows nothing \
                 resident — something invisible to Ollama is holding the \
                 card. Known cause: llama-server runner processes orphaned by \
                 a parent-only Ollama restart. Check `nvidia-smi` for \
                 llama-server/ollama processes and kill them; this size \
                 latches for the process lifetime"
            );
        }
        Some(fitted)
    }

    /// VRAM currently held by `model` and, separately, by every OTHER model the
    /// runner has resident — the embedder, chiefly.
    ///
    /// Both halves matter and they pull in opposite directions, so they are
    /// measured rather than assumed. Guessing the second one at 2.5 GB (a
    /// plausible embedder size) against an actual 0.32 GB was enough on its own
    /// to drive the budget negative and pin the context at the 4096 floor.
    fn ollama_resident_vram(base_url: &str, model: &str) -> (u64, u64) {
        let name = model.strip_prefix("ollama/").unwrap_or(model);
        let Ok(out) = std::process::Command::new("curl")
            .args(["-s", "--max-time", LOCAL_OLLAMA_QUERY_MAX_SECS, &format!("{base_url}/api/ps")])
            .output()
        else {
            return (0, 0);
        };
        let body = String::from_utf8_lossy(&out.stdout);
        let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&body) else {
            return (0, 0);
        };
        let Some(models) = parsed.get("models").and_then(serde_json::Value::as_array) else {
            return (0, 0);
        };
        let (mut ours, mut others) = (0u64, 0u64);
        for m in models {
            let vram = m
                .get("size_vram")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            if m.get("name").and_then(serde_json::Value::as_str) == Some(name) {
                ours += vram;
            } else {
                others += vram;
            }
        }
        (ours, others)
    }

    /// The arithmetic of [`Self::fit_context_to_free_vram`], split out from the
    /// measurements so it can be tested against the runs we actually observed
    /// rather than only against itself.
    fn fit_context_for_budget(
        free_bytes: f64,
        weights_bytes: f64,
        ours_resident: f64,
        others_resident: f64,
    ) -> u32 {
        /// Per-context-token VRAM cost: KV cache plus the attention scratch the
        /// prefill graph allocates.
        ///
        /// Measured, not derived, and the derivation is the reason. gemma4:12b
        /// reports `sliding_window = 1024` with `key_length_swa = 256`, so most
        /// of its 48 layers attend over a fixed 1024-token window and their KV
        /// does NOT grow with `n_ctx` at all — only the few full-attention
        /// layers do. Without knowing the exact interleave (Ollama does not
        /// report it) any closed form is a guess, and the naive one is off by
        /// 150x. So: KV measured directly at 10.5 KB/token
        /// ((8.4 GB - 8.1 GB) / (32768 - 4096)), plus ~32 KB/token for the
        /// full-attention scratch (`n_ubatch 512 x n_head 16 x 4 bytes`).
        ///
        /// Reading high is the safe direction — it costs context, not a run.
        const BYTES_PER_CTX_TOKEN: f64 = 43_000.0;
        /// Floor on what to set aside for an embedder that has not loaded yet.
        /// It arrives on the first `remember` and stays co-resident for the
        /// rest of the run, so budgeting as though it will never come is how a
        /// run that measured fine at minute one dies at minute ten.
        const EMBEDDER_RESERVE: f64 = 1.0 * 1024.0 * 1024.0 * 1024.0;
        /// Driver allocations, fragmentation, the fixed part of the compute
        /// graph, and the desktop's own growth after we measure.
        ///
        /// That last term is the reason this is not smaller. Sizing runs on the
        /// first request, which is the emptiest the card will be all run: the
        /// GUI's webview alone was measured taking a further 0.8 GB in the
        /// minute after the daemon came up. Budgeting against that instant
        /// without slack is how the first request picks a size the tenth
        /// request cannot honour.
        const SLACK: f64 = 1.2 * 1024.0 * 1024.0 * 1024.0;
        const MAX_CTX: u32 = 32_768;
        // Snap DOWN to a power-of-two bucket.
        //
        // Not cosmetic: Ollama keys a loaded model instance partly on its
        // options, so a changed `num_ctx` evicts and reloads the model — tens
        // of seconds of stall and a VRAM churn, mid-run. "Every turn" has to
        // mean the value is *rechecked* every turn, not that it is free to
        // wander: free VRAM drifting by a few hundred MB as the desktop
        // breathes must not reload a 7.5 GB model. Buckets make the answer
        // stable across ordinary drift while still moving when something real
        // changes (a game starts, the embedder loads).
        const BUCKETS: [u32; 4] = [32_768, 16_384, 8_192, 4_096];

        // Add back what THIS model already holds before subtracting what it
        // costs, so the answer does not depend on whether it happens to be
        // loaded yet.
        //
        // Without this the function contradicts itself: before the load it sees
        // a free card and picks a large context; after the load the same card
        // looks 8 GB poorer, it picks the 4096 floor, and because Ollama keys a
        // resident instance on its options, that new number evicts and reloads
        // the model — which frees the memory that justified shrinking, so the
        // next turn picks large again. Measured live: this path pinned
        // gemma4:12b to 4096 on a card with 10.4 GB genuinely available.
        let effective_free = free_bytes + ours_resident;
        let reserve = others_resident.max(EMBEDDER_RESERVE) + SLACK;
        let usable = effective_free - weights_bytes - reserve;
        if usable <= 0.0 {
            return Self::MIN_FIT_CTX;
        }
        // Tokens that fit, compared unrounded: for a whole bucket size `b`,
        // `b <= raw` is exactly `b <= floor(raw)`, the old truncating cast.
        let raw = usable / BYTES_PER_CTX_TOKEN;

        let fitted = BUCKETS
            .iter()
            .copied()
            .find(|b| f64::from(*b) <= raw)
            .unwrap_or(Self::MIN_FIT_CTX);
        fitted.clamp(Self::MIN_FIT_CTX, MAX_CTX)
    }

    /// Free VRAM in bytes, via `nvidia-smi`. `None` on any non-NVIDIA or
    /// tooling-absent system, which is a normal answer, not an error.
    fn nvidia_free_vram_bytes() -> Option<u64> {
        let out = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        let mib: u64 = String::from_utf8_lossy(&out.stdout)
            .lines()
            .next()?
            .trim()
            .parse()
            .ok()?;
        Some(mib * 1024 * 1024)
    }

    /// On-disk size of an Ollama model, a good proxy for its resident weights.
    ///
    /// Cached per server and model name for the life of the process: free
    /// VRAM is the term that has to be live, but a model's file size cannot
    /// change while we are mid-run against it, and this runs on every single
    /// request.
    ///
    /// Asked without the server's token: a credential never goes on a command
    /// line, where any local user's `ps` reads it. A local server that wants
    /// one answers 401, which reads as an unknown size, and sizing falls back.
    fn ollama_model_size_bytes(base_url: &str, model: &str) -> Option<u64> {
        /// Size by (server base URL, model name); `None` is a cached miss.
        type Sizes = std::collections::HashMap<(String, String), Option<u64>>;
        static CACHE: std::sync::OnceLock<std::sync::Mutex<Sizes>> = std::sync::OnceLock::new();
        let cache = CACHE.get_or_init(Default::default);
        let name = model.strip_prefix("ollama/").unwrap_or(model).to_string();
        let key = (base_url.to_string(), name.clone());

        if let Ok(guard) = cache.lock()
            && let Some(hit) = guard.get(&key) {
                return *hit;
            }

        let out = std::process::Command::new("curl")
            .args(["-s", "--max-time", LOCAL_OLLAMA_QUERY_MAX_SECS, &format!("{base_url}/api/tags")])
            .output()
            .ok()?;
        // No answer (refused, timed out) says nothing about the model, only
        // about the server right now, so it is not cached: a server that was
        // wedged for one request must not leave the model unsized for the
        // rest of the process.
        if !out.status.success() {
            return None;
        }
        let looked_up = (|| {
            let body = String::from_utf8_lossy(&out.stdout);
            let parsed: serde_json::Value = serde_json::from_str(&body).ok()?;
            parsed
                .get("models")?
                .as_array()?
                .iter()
                .find(|m| m.get("name").and_then(|n| n.as_str()) == Some(name.as_str()))
                .and_then(|m| m.get("size"))
                .and_then(serde_json::Value::as_u64)
        })();

        // An answer without the model is cached too. If the model is not in
        // `/api/tags` it will not appear later either, and retrying on every
        // request would spawn a process per turn to learn the same thing.
        if let Ok(mut guard) = cache.lock() {
            guard.insert(key, looked_up);
        }
        looked_up
    }

    fn build_ollama_http_client() -> Client {
        Client::builder()
            .read_timeout(std::time::Duration::from_secs(STREAM_READ_TIMEOUT_SECS))
            .connect_timeout(std::time::Duration::from_secs(10))
            .pool_max_idle_per_host(0)
            .build()
            .unwrap_or_else(|_| Client::new())
    }

    /// Pin an Ollama base URL to IPv4 loopback. "localhost" resolves to `::1`
    /// first on Windows, and the chat stream was the ONLY component talking
    /// to Ollama over v6 loopback (health probes and runner surgery already
    /// use 127.0.0.1) — the same v6 path where streams were cut
    /// mid-generation. Remote hosts pass through untouched.
    fn normalize_ollama_url(url: &str) -> String {
        // Trimmed of whitespace and trailing slashes because every call appends
        // `/api/...` itself: a pasted `https://host/ollama/` would otherwise
        // address `//api/chat`, which a path-routing proxy answers with 404.
        let trimmed = url.trim().trim_end_matches('/');
        // The parsed host, not a prefix of the text: `localhost.example.com`
        // is another machine, and rewriting it would connect somewhere other
        // than the address the bearer token was bound to.
        let Ok(mut parsed) = reqwest::Url::parse(trimmed) else {
            return trimmed.to_string();
        };
        if parsed.host_str() != Some("localhost") || parsed.set_host(Some("127.0.0.1")).is_err() {
            return trimmed.to_string();
        }
        parsed.as_str().trim_end_matches('/').to_string()
    }

    /// Create a new Anthropic client with API key
    pub fn anthropic(api_key: impl Into<String>) -> Self {
        Self {
            http: Self::build_http_client(),
            provider: Provider::Anthropic,
            api_key: api_key.into(),
            base_url: "https://api.anthropic.com".to_string(),
        }
    }

    /// Create a new Anthropic client with OAuth token
    /// Uses Authorization: Bearer header instead of x-api-key
    pub fn anthropic_oauth(oauth_token: impl Into<String>) -> Self {
        Self {
            http: Self::build_http_client(),
            provider: Provider::Anthropic,
            api_key: format!("oauth:{}", oauth_token.into()), // Prefix to indicate OAuth
            base_url: "https://api.anthropic.com".to_string(),
        }
    }

    /// Create a new `OpenAI` client
    pub fn openai(api_key: impl Into<String>) -> Self {
        Self {
            http: Self::build_http_client(),
            provider: Provider::OpenAI,
            api_key: api_key.into(),
            base_url: "https://api.openai.com".to_string(),
        }
    }

    /// Create a new `OpenRouter` client
    pub fn openrouter(api_key: impl Into<String>) -> Self {
        Self {
            http: Self::build_http_client(),
            provider: Provider::OpenRouter,
            api_key: api_key.into(),
            base_url: "https://openrouter.ai/api".to_string(),
        }
    }

    /// Create a new `GitHub Models` client (uses GitHub PAT)
    /// GitHub Models provides access to various LLMs including GPT-4o, Llama, Mistral, etc.
    pub fn github_models(api_key: impl Into<String>) -> Self {
        Self {
            http: Self::build_http_client(),
            provider: Provider::GitHubModels,
            api_key: api_key.into(),
            base_url: "https://models.inference.ai.azure.com".to_string(),
        }
    }

    /// Create a new `Claude Proxy` client for claude-max-api-proxy
    /// This proxies requests through a local server that uses Claude Code CLI credentials
    /// Default URL is <http://localhost:3456>
    pub fn claude_proxy(base_url: impl Into<String>) -> Self {
        Self {
            http: Self::build_http_client(),
            provider: Provider::ClaudeProxy,
            api_key: String::new(), // Proxy handles auth via Claude Code CLI
            base_url: base_url.into(),
        }
    }

    /// Create a Claude Proxy client with default localhost URL
    #[must_use]
    pub fn claude_proxy_default() -> Self {
        Self::claude_proxy("http://localhost:3456")
    }

    /// Create a new `Ollama` client (local, no API key needed)
    pub fn ollama(base_url: impl Into<String>) -> Self {
        Self {
            http: Self::build_ollama_http_client(),
            provider: Provider::Ollama,
            api_key: String::new(), // Ollama doesn't need auth
            base_url: Self::normalize_ollama_url(&Into::<String>::into(base_url)),
        }
    }

    /// Create an Ollama client with an API key (for remote/authenticated instances)
    pub fn ollama_with_key(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            http: Self::build_ollama_http_client(),
            provider: Provider::Ollama,
            // Trimmed: a blank key must mean no `Authorization` header at all
            // (`apply_ollama_auth` checks for empty), never `Bearer    `.
            api_key: api_key.into().trim().to_string(),
            base_url: Self::normalize_ollama_url(&Into::<String>::into(base_url)),
        }
    }

    /// Create an Ollama client with default localhost URL
    #[must_use]
    pub fn ollama_default() -> Self {
        Self::ollama("http://localhost:11434")
    }

    /// The base URL this client's requests go to — for Ollama, the configured
    /// server as the client addresses it (trimmed, `localhost` pinned to
    /// `127.0.0.1`).
    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Whether this Ollama client's server answers HTTP right now, with any
    /// status: one `GET /api/tags`, bounded by `timeout`.
    ///
    /// This is a liveness check, for waiting out a server that refused or
    /// dropped the connection, and any answer ends that condition. Whether
    /// the answer is a good one is the next request's to find out: a gateway
    /// that routes `/api/chat` alone answers this with 404 and serves chat
    /// fine, and asking for an Ollama model list here kept such a server
    /// "down" for as long as a caller would wait.
    ///
    /// Asked with the client's own token, as chat asks: a proxy that counts
    /// unauthenticated requests would otherwise see one every few seconds for
    /// the length of a wait. `false` for any other provider: its key is not
    /// Ollama's to send.
    pub async fn ollama_answers(&self, timeout: std::time::Duration) -> bool {
        if self.provider != Provider::Ollama {
            return false;
        }
        self.apply_ollama_auth(self.http.get(format!("{}/api/tags", self.base_url)))
            .timeout(timeout)
            .send()
            .await
            .is_ok()
    }

    /// Check if this client uses OAuth authentication
    fn is_oauth(&self) -> bool {
        is_oauth_token(&self.api_key)
    }

    /// Get the actual token (stripping oauth: prefix if present)
    fn get_token(&self) -> &str {
        get_raw_token(&self.api_key)
    }

    /// Apply Anthropic authentication headers to a request builder
    /// For OAuth, includes required beta headers and user-agent (matching Claude Code exactly)
    fn apply_anthropic_auth(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if self.is_oauth() {
            // Stealth mode: Mimic Claude Code's headers exactly
            // Beta features: claude-code identity, oauth, interleaved thinking, fine-grained tool streaming
            let beta = "claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14,fine-grained-tool-streaming-2025-05-14";
            let user_agent = format!("claude-cli/{CLAUDE_CODE_VERSION} (external, cli)");
            debug!(
                beta = %beta,
                user_agent = %user_agent,
                "OAuth stealth mode: applying Claude Code headers"
            );
            builder
                .header("Authorization", format!("Bearer {}", self.get_token()))
                .header("accept", "application/json")
                .header("anthropic-beta", beta)
                .header("user-agent", user_agent)
                .header("x-app", "cli")
                .header("anthropic-dangerous-direct-browser-access", "true")
        } else {
            builder.header("x-api-key", &self.api_key)
        }
    }

    /// Validate the API key by making a lightweight request.
    ///
    /// Apply Ollama auth header if an API key is configured.
    fn apply_ollama_auth(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if self.api_key.is_empty() {
            builder
        } else {
            builder.header("Authorization", format!("Bearer {}", self.api_key))
        }
    }

    /// # Errors
    ///
    /// Returns `LlmError::Api` with status 401 if the API key is invalid.
    /// Returns `LlmError::Network` if the request fails.
    pub async fn validate(&self) -> Result<(), LlmError> {
        match self.provider {
            Provider::Anthropic => {
                // Use count_tokens endpoint for validation - lightweight
                let request_builder = self
                    .http
                    .post(format!("{}/v1/messages/count_tokens", self.base_url));
                let response = self.apply_anthropic_auth(request_builder)
                    .header("anthropic-version", "2023-06-01")
                    .header("content-type", "application/json")
                    .json(&serde_json::json!({
                        "model": "claude-sonnet-4-20250514",
                        "messages": [{"role": "user", "content": "hi"}]
                    }))
                    .send()
                    .await?;

                if response.status().as_u16() == 401 {
                    return Err(LlmError::Api {
                        status: 401,
                        message: "Invalid API key or OAuth token".to_string(),
                    });
                }
                Ok(())
            }
            _ => Ok(()), // Skip validation for other providers for now
        }
    }

    /// Get the provider for this client
    #[must_use]
    pub const fn provider(&self) -> Provider {
        self.provider
    }

    /// Get model information with caching.
    ///
    /// Checks the file cache first (this client's server only), then asks the
    /// provider. When the provider gives no answer the unknown-model floor is
    /// returned for this lookup and nothing is cached, so the next lookup asks
    /// again (see [`ModelInfoCache`]). The window is clamped to the one this
    /// client's server honours for the model now
    /// ([`Self::effective_context_window`]), so a prompt sized from it fits the
    /// `num_ctx` this client's request carries.
    pub async fn get_model_info(&self, model: &str, cache: Option<&ModelInfoCache>) -> ModelInfo {
        let server = self.model_info_server();
        if let Some(cache) = cache
            && let Some(info) = cache.get_from(&server, model) {
                return self.clamp_model_info_to_effective_window(model, info);
            }

        let info = match self.fetch_model_info(model).await {
            Ok(info) => info,
            Err(e) => {
                warn!(
                    model = %model,
                    server = %server,
                    error = %e,
                    "no model info from the provider — budgets for this lookup use the \
                     unknown-model floor, and it is not cached, so the next lookup asks again"
                );
                if let Some(cache) = cache {
                    cache.forget_other_server(&server, model);
                }
                return self.clamp_model_info_to_effective_window(
                    model,
                    default_model_info(model, &format!("{:?}", self.provider)),
                );
            }
        };

        // Announce a model that the provider answered for but that ended up on
        // the unknown floor instead of real limits. This is the failure mode
        // that hid the Anthropic field-name bug: every Claude model silently
        // resolved to 32k/4k, was cached with a fresh timestamp, and nothing
        // anywhere said so. A model running on guessed limits is worth one line
        // per cache TTL. (A lookup with no answer at all is announced above,
        // every time, since it is never cached.)
        // Keyed on the window alone. Pairing it with the output floor made the
        // check unreachable for the two providers most likely to need it:
        // Ollama and OpenRouter derive the output cap as `context_window / 2`
        // when the provider answers but publishes no window, landing on
        // 32000/16000, which never matches 32000/4096. Ollama takes that path
        // routinely — GGUF keys the window by architecture
        // (`llama.context_length`, `qwen3.context_length`) while only
        // `general.context_length` is read — so the invented window was cached
        // for a week and every budget sized from it, in silence.
        //
        // A model whose real window happens to be exactly the floor trips this
        // too. That is unavoidable without recording provenance on ModelInfo,
        // and a false line once per cache TTL is the cheaper error: the floor
        // is a "we do not know" sentinel, and the failure this exists to catch
        // is total silence.
        if info.context_window == UNKNOWN_CONTEXT_WINDOW {
            warn!(
                model = %model,
                provider = ?self.provider,
                context_window = info.context_window,
                max_output_tokens = info.max_output_tokens,
                "model window matches the unknown-model floor — the provider likely published \
                 none, so budgets are sized for a far smaller model than this may be"
            );
        }

        // Cache the result — the UNCLAMPED provider claim. The cache stores
        // static facts about the model; the effective-window clamp below is
        // live per-process state (the Ollama num_ctx latch) and must never be
        // persisted as if the model itself shrank.
        if let Some(cache) = cache
            && let Err(e) = cache.set(&server, &info) {
                warn!(model = %model, error = %e, "Failed to cache model info");
            }

        self.clamp_model_info_to_effective_window(model, info)
    }

    /// Force refresh model info from API (bypasses cache).
    ///
    /// # Errors
    ///
    /// Returns `LlmError` if the provider gives no answer for `model`, and
    /// caches nothing then.
    pub async fn refresh_model_info(&self, model: &str, cache: Option<&ModelInfoCache>) -> Result<ModelInfo, LlmError> {
        let info = self.fetch_model_info(model).await?;

        if let Some(cache) = cache {
            cache.set(&self.model_info_server(), &info)?;
        }

        Ok(info)
    }

    /// Which server answers this client's model-info lookups: the provider
    /// and the base URL, stripped of any credentials in it (the name is
    /// written to disk with the answer, and `https://user:pw@host` must not
    /// put the password there).
    ///
    /// A model name alone names nothing across servers: `qwen3:8b` on a
    /// remote server can be built with another window than the local one, and
    /// a Claude proxy and Anthropic answer for the same Claude names.
    fn model_info_server(&self) -> String {
        let base = reqwest::Url::parse(&self.base_url).map_or_else(
            |_| self.base_url.clone(),
            |mut url| {
                let _ = url.set_username("");
                let _ = url.set_password(None);
                url.as_str().trim_end_matches('/').to_string()
            },
        );
        format!("{:?} {base}", self.provider)
    }

    /// Ask the provider about `model`.
    ///
    /// `Ok` only for the provider's answer, which [`Self::get_model_info`]
    /// caches. Anything that is not an answer is an `Err` and is never
    /// cached: a refusal or transport error obviously, and a 404 too — "not
    /// on this server" or "not in this catalogue" is a fact about the server
    /// right now (a pull or a corrected URL changes it), not about the
    /// model's window.
    async fn fetch_model_info(&self, model: &str) -> Result<ModelInfo, LlmError> {
        match self.provider {
            Provider::Anthropic => self.fetch_anthropic_model_info(model).await,
            Provider::OpenAI => self.fetch_openai_model_info(model).await,
            Provider::Ollama => self.fetch_ollama_model_info(model).await,
            Provider::OpenRouter => self.fetch_openrouter_model_info(model).await,
            // A Claude proxy serves Claude models, so ask the way Anthropic
            // answers. A proxy that does not implement the models endpoint
            // 404s or fails to deserialize, and `get_model_info` falls back to
            // defaults for that lookup. The failure is not cached, because a
            // proxy that is merely not up yet fails the same way; the cost is
            // one request per lookup to a proxy that never answers.
            Provider::ClaudeProxy => {
                self.fetch_anthropic_model_info(model)
                    .await
                    .map(|info| ModelInfo {
                        provider: "claude_proxy".to_string(),
                        ..info
                    })
            }
            // GitHub Models publishes no per-model limit endpoint, so the
            // floor is its standing answer — cached like any other, since no
            // later request could say more.
            Provider::GitHubModels => {
                Ok(default_model_info(model, &format!("{:?}", self.provider)))
            }
        }
    }

    /// Fetch model info from Anthropic API
    async fn fetch_anthropic_model_info(&self, model: &str) -> Result<ModelInfo, LlmError> {
        // Anthropic's /v1/models/{model_id} endpoint
        let url = format!("{}/v1/models/{}", self.base_url, model);

        let request_builder = self.http.get(&url);
        let response = self
            .apply_anthropic_auth(request_builder)
            .header("anthropic-version", "2023-06-01")
            .send()
            .await?;

        // A 404 included: see `fetch_model_info` for why it is not an answer.
        if !response.status().is_success() {
            let status = response.status().as_u16();
            let message = response.text().await.unwrap_or_default();
            return Err(LlmError::Api { status, message });
        }

        // The Models API spells these `max_input_tokens` (the context window)
        // and `max_tokens` (the output cap). There is no `context_window`
        // field — asking for one deserializes to `None`, silently falls back
        // to the unknown-model floor, and caches THAT, so every Anthropic
        // model resolved to 32k context / 4k output no matter what it really
        // was. The aliases keep the old spellings working if a proxy or
        // gateway answers with them.
        let api_response: AnthropicModelResponse = response.json().await?;

        // Use API values if available, otherwise use defaults
        let defaults = default_model_info(model, "anthropic");
        Ok(ModelInfo {
            id: api_response.id,
            context_window: api_response
                .max_input_tokens
                .unwrap_or(defaults.context_window),
            max_output_tokens: api_response.max_tokens.unwrap_or(defaults.max_output_tokens),
            supports_tools: defaults.supports_tools,
            supports_vision: defaults.supports_vision,
            embedding_dimension: None, // Anthropic doesn't provide embedding models via this API
            cached_at: current_timestamp(),
            provider: "anthropic".to_string(),
        })
    }

    /// Fetch model info from `OpenAI` API
    async fn fetch_openai_model_info(&self, model: &str) -> Result<ModelInfo, LlmError> {
        // OpenAI doesn't return context_window in their model endpoint
        // So we use defaults with the model ID from the API
        #[derive(Deserialize)]
        struct OpenAIModelResponse {
            id: String,
        }

        let url = format!("{}/v1/models/{}", self.base_url, model);

        let response = self
            .http
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await?;

        // A 404 included: see `fetch_model_info` for why it is not an answer.
        if !response.status().is_success() {
            let status = response.status().as_u16();
            let message = response.text().await.unwrap_or_default();
            return Err(LlmError::Api { status, message });
        }

        let api_response: OpenAIModelResponse = response.json().await?;
        Ok(default_model_info(&api_response.id, "openai").with_timestamp())
    }

    /// Fetch model info from Ollama API
    async fn fetch_ollama_model_info(&self, model: &str) -> Result<ModelInfo, LlmError> {
        let url = format!("{}/api/show", self.base_url);

        let response = self
            .apply_ollama_auth(self.http.post(&url))
            .json(&serde_json::json!({ "name": model }))
            .send()
            .await?;

        // A 404 included: Ollama says `model '…' not found` for a model that
        // is not pulled (yet), and a path-routing proxy says it for a base URL
        // missing its path. See `fetch_model_info`.
        if !response.status().is_success() {
            let status = response.status().as_u16();
            let message = response.text().await.unwrap_or_default();
            return Err(LlmError::Api { status, message });
        }

        let api_response: OllamaShowResponse = response.json().await?;
        let model_info = api_response.model_info.unwrap_or_default();

        // Use the model metadata verbatim. In particular, do not silently raise
        // a local model's configured context window above what Ollama reports.
        let defaults = default_model_info(model, "ollama");
        let context_window =
            gguf_metadata_usize(&model_info, "context_length").unwrap_or(defaults.context_window);
        let embedding_dimension = gguf_metadata_usize(&model_info, "embedding_length");

        Ok(ModelInfo {
            id: model.to_string(),
            context_window,
            // Ollama does not publish a distinct output cap. Derive the safe
            // generation budget from this model's reported total window.
            max_output_tokens: context_window / 2,
            supports_tools: true, // Ollama v0.5+ supports tool calling
            supports_vision: false,
            embedding_dimension,
            cached_at: current_timestamp(),
            provider: "ollama".to_string(),
        })
    }

    /// Fetch model info from `OpenRouter` API
    async fn fetch_openrouter_model_info(&self, model: &str) -> Result<ModelInfo, LlmError> {
        #[derive(Deserialize)]
        struct OpenRouterModelsResponse {
            data: Vec<OpenRouterModel>,
        }

        #[derive(Deserialize)]
        struct OpenRouterModel {
            id: String,
            #[serde(default)]
            context_length: Option<usize>,
            #[serde(default)]
            top_provider: Option<OpenRouterTopProvider>,
        }

        #[derive(Deserialize)]
        struct OpenRouterTopProvider {
            #[serde(default)]
            max_completion_tokens: Option<usize>,
        }

        // OpenRouter has a models endpoint that lists all models
        let url = format!("{}/v1/models", self.base_url);

        let response = self
            .http
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await?;

        // An error status is no answer. It used to come back as `Ok` with the
        // floor, which was then cached for a week like a real answer.
        if !response.status().is_success() {
            let status = response.status().as_u16();
            let message = response.text().await.unwrap_or_default();
            return Err(LlmError::Api { status, message });
        }

        let api_response: OpenRouterModelsResponse = response.json().await?;

        // Find our model in the list
        if let Some(m) = api_response.data.iter().find(|m| m.id == model) {
            return Ok(ModelInfo {
                id: m.id.clone(),
                context_window: m.context_length.unwrap_or(UNKNOWN_CONTEXT_WINDOW),
                max_output_tokens: m
                    .top_provider
                    .as_ref()
                    .and_then(|p| p.max_completion_tokens)
                    .unwrap_or_else(|| m.context_length.unwrap_or(UNKNOWN_CONTEXT_WINDOW) / 2),
                supports_tools: true, // OpenRouter usually routes to capable models
                supports_vision: false,
                embedding_dimension: None,
                cached_at: current_timestamp(),
                provider: "openrouter".to_string(),
            });
        }

        // Not listed is the catalogue's 404: true today, and changed by the
        // catalogue tomorrow without anything the cache can see.
        Err(LlmError::Api {
            status: 404,
            message: format!("{model} is not in OpenRouter's model list"),
        })
    }

    /// Send a completion request (simple, no tools).
    ///
    /// # Errors
    ///
    /// Returns `LlmError::Api` if the API returns an error.
    /// Returns `LlmError::Network` if the request fails.
    pub async fn complete(&self, request: &CompletionRequest) -> Result<String, LlmError> {
        match self.provider {
            Provider::Anthropic => self.complete_anthropic_simple(request).await,
            Provider::OpenAI | Provider::OpenRouter | Provider::GitHubModels | Provider::ClaudeProxy => self.complete_openai(request).await,
            Provider::Ollama => self.complete_ollama(request).await,
        }
    }

    /// Send a full Anthropic-format request with tools.
    ///
    /// Dispatches to the appropriate provider backend:
    /// - Anthropic: Native Anthropic API (`/v1/messages`)
    /// - OpenAI/OpenRouter/GitHub/ClaudeProxy: Converts to `OpenAI` chat completions format
    /// - Ollama: Converts to Ollama `/api/chat` format
    ///
    /// # Errors
    ///
    /// Returns `LlmError::Api` if the API returns an error.
    /// Returns `LlmError::Network` if the request fails.
    pub async fn complete_anthropic(&self, request: &AnthropicRequest) -> Result<AnthropicResponse, LlmError> {
        // Last stop before the wire: correct anything the caller's model
        // no longer accepts. See `conform_to_anthropic_contract`.
        let request = &*conform_to_anthropic_contract(request);
        match self.provider {
            Provider::Anthropic => self.complete_anthropic_native(request).await,
            Provider::OpenAI | Provider::OpenRouter | Provider::GitHubModels | Provider::ClaudeProxy =>
                self.complete_anthropic_via_openai(request).await,
            Provider::Ollama =>
                self.complete_anthropic_via_ollama(request).await,
        }
    }

    /// Native Anthropic API implementation for `complete_anthropic`
    async fn complete_anthropic_native(&self, request: &AnthropicRequest) -> Result<AnthropicResponse, LlmError> {
        pace_provider_requests(self.provider).await;
        let is_oauth = self.is_oauth();

        // Store original tools for reverse mapping
        let original_tools = request.tools.clone().unwrap_or_default();

        let request_builder = self
            .http
            .post(format!("{}/v1/messages", self.base_url));

        let response = if is_oauth {
            // OAuth mode: Use array-based system prompt and remap tool names
            let mut oauth_request = OAuthAnthropicRequest::from_request(request, true);

            // Remap tool names to Claude Code format
            if let Some(ref mut tools) = oauth_request.tools {
                for tool in tools {
                    tool.name = to_claude_code_tool_name(&tool.name);
                }
            }

            debug!(
                system_blocks = oauth_request.system.len(),
                tool_names = ?oauth_request.tools.as_ref().map(|t| t.iter().map(|x| x.name.as_str()).collect::<Vec<_>>()),
                "OAuth request prepared with array-based system prompt"
            );

            self.apply_anthropic_auth(request_builder)
                .header("anthropic-version", "2023-06-01")
                .header("content-type", "application/json")
                .json(&oauth_request)
                .send()
                .await?
        } else {
            // Standard mode: Use string-based system prompt
            self.apply_anthropic_auth(request_builder)
                .header("anthropic-version", "2023-06-01")
                .header("content-type", "application/json")
                .json(request)
                .send()
                .await?
        };

        if !response.status().is_success() {
            return Err(LlmError::from_response(response).await);
        }

        let mut result: AnthropicResponse = response.json().await?;

        // Log prompt cache usage
        if result.usage.cache_read_input_tokens > 0 || result.usage.cache_creation_input_tokens > 0 {
            info!(
                cache_read = result.usage.cache_read_input_tokens,
                cache_write = result.usage.cache_creation_input_tokens,
                input = result.usage.input_tokens,
                "📦 Prompt cache: read {} tokens, wrote {} tokens",
                result.usage.cache_read_input_tokens,
                result.usage.cache_creation_input_tokens,
            );
        }

        // Remap tool names back in response
        if is_oauth {
            for block in &mut result.content {
                if let ContentBlock::ToolUse { name, .. } = block {
                    *name = from_claude_code_tool_name(name, &original_tools);
                }
            }
        }

        Ok(result)
    }

    /// Convert `AnthropicRequest` → `OpenAI` chat completions and execute
    async fn complete_anthropic_via_openai(&self, request: &AnthropicRequest) -> Result<AnthropicResponse, LlmError> {
        pace_provider_requests(self.provider).await;
        let (messages_json, tools_json) = anthropic_to_openai_request(request);

        let mut body = serde_json::json!({
            "model": request.model,
            "messages": messages_json,
            "max_tokens": request.max_tokens,
        });
        if let Some(temp) = request.temperature {
            body["temperature"] = serde_json::json!(temp);
        }
        if let Some(tools) = tools_json {
            body["tools"] = tools;
        }

        let response = self
            .http
            .post(format!("{}/v1/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await?;

        learn_rate_limit_headers(self.provider, response.headers());
        if !response.status().is_success() {
            return Err(LlmError::from_response(response).await);
        }

        let result: serde_json::Value = response.json().await?;
        let response = openai_response_to_anthropic(&request.model, &result)?;

        // Log OpenAI prompt cache usage (automatic for prompts >1024 tokens, 50% discount)
        if response.usage.cache_read_input_tokens > 0 {
            info!(
                cache_read = response.usage.cache_read_input_tokens,
                input = response.usage.input_tokens,
                "📦 OpenAI prompt cache: read {} cached tokens (50% discount)",
                response.usage.cache_read_input_tokens,
            );
        }

        Ok(response)
    }

    /// Convert `AnthropicRequest` → Ollama /api/chat and execute
    async fn complete_anthropic_via_ollama(
        &self,
        request: &AnthropicRequest,
    ) -> Result<AnthropicResponse, LlmError> {
        pace_provider_requests(self.provider).await;
        let (messages_json, tools_json) = anthropic_to_ollama_request(request);

        let mut body = serde_json::json!({
            "model": request.model,
            "messages": messages_json,
            "stream": false,
        });

        // Ollama uses "options" for parameters
        let mut options = serde_json::Map::new();
        if let Some(temp) = request.temperature {
            options.insert("temperature".to_string(), serde_json::json!(temp));
        } else {
            // No caller override: Modelfile defaults for thinking models run
            // temp 1.0, which is hot for precise code emission. 0.6 is
            // Qwen's own non-thinking recommendation.
            options.insert("temperature".to_string(), serde_json::json!(0.6));
        }
        // Bound generation to the request's token budget. num_predict=-1
        // (unlimited) let a degraded runner emit degenerate tokens without
        // end; the caller's max_tokens already accounts for thinking room.
        options.insert("num_predict".to_string(), serde_json::json!(request.max_tokens));
        // Context size the CALLER asked for, falling back to 32K.
        //
        // This was hardcoded to 32768 for every Ollama model, which ignores the
        // `context_limit` the request already carries and, worse, ignores what
        // the GPU can actually hold. A 12B model's KV cache at 32k does not fit
        // beside its own 8.4 GB of weights on a 16 GB card that is already
        // giving 5 GB to the desktop — and the failure mode is not a clean
        // out-of-memory error, it is `CUDA error: an illegal memory access was
        // encountered` mid-generation, which surfaces here as a truncated
        // stream. Observed live 2026-07-28: gemma4:12b died this way three
        // times in three minutes while qwen3.5:9b (6.7 GB) never once did.
        //
        // Honouring the request means a model that does not fit at 32k can be
        // run at a size that does, instead of being unrunnable.
        //
        // Source order: what the caller asked for, then `NANNA_OLLAMA_NUM_CTX`,
        // then 32k. The env override exists because the right value is a
        // property of THIS MACHINE — how much VRAM is left once the model's
        // weights and the desktop have taken theirs — and nothing inside the
        // agent knows that. A per-model config field is the proper home for it;
        // until then this is the knob that makes an oversized model runnable
        // instead of unrunnable.
        // Size the context to the GPU as it is RIGHT NOW, every turn.
        //
        // A constant cannot be right here. 32k was unrunnable for a 12B model on
        // a 16 GB card sharing with the desktop; 16k would be leaving most of a
        // 24 GB card unused; and the amount of VRAM the desktop holds changes
        // when the user opens a browser or closes a game. The only honest input
        // is free memory measured at the moment of the request.
        //
        // What actually runs out is NOT the KV cache — measured on gemma4:12b,
        // that is ~10 KB/token, so even 32k costs under 0.4 GB. It is the
        // prefill compute buffer, which scales with how much prompt is pushed
        // through at once, and which overruns as
        // `CUDA error: an illegal memory access was encountered` rather than a
        // clean allocation failure. So the budget reserves a fraction of free
        // VRAM for it instead of spending everything on cache.
        let num_ctx = Self::resolve_num_ctx(&self.base_url, &request.model, request.context_limit);
        tracing::info!(
            model = %request.model,
            num_ctx,
            "sized ollama context to free VRAM"
        );
        options.insert("num_ctx".to_string(), serde_json::json!(num_ctx));
        // qwen3.5's Modelfile ships presence_penalty=1.5 (Qwen's thinking-mode
        // anti-repetition default). Code REQUIRES repetition, and at 1.5 the
        // model visibly degraded working multi-function files into semicolon
        // one-liner slop (observed live, round-5 drive). 1.0 keeps a mild
        // guard; the loop's repetition/spiral detectors cover the rest.
        options.insert("presence_penalty".to_string(), serde_json::json!(1.0));
        // repeat_penalty 1.1 (the Ollama default) punishes exactly the tokens
        // Python repeats by design — newlines and indentation runs — and the
        // model visibly compensates by cramming statements onto one line
        // (observed live across rounds even at presence 1.0). Code presets
        // use 1.0: no repetition penalty at all.
        options.insert("repeat_penalty".to_string(), serde_json::json!(1.0));
        if !options.is_empty() {
            body["options"] = serde_json::Value::Object(options);
        }

        // Ollama supports tools since v0.5
        if let Some(tools) = tools_json {
            body["tools"] = tools;
        }

        // Enable thinking separation for models that support it (qwen3, deepseek-r1, etc.)
        // This makes Ollama return thinking in a separate `thinking` field instead of
        // embedding <think>...</think> tags inside `content`.
        if ollama_separates_thinking(&request.model) {
            body["think"] = serde_json::json!(true);
        }

        // The same single-slot discipline as the streaming path and
        // `complete_ollama`. Every in-loop summarizer (context ladder,
        // tool-output compression and summaries, distillation, memory
        // extraction) calls this path on chat's own server, so without the
        // permit one session's summary cancels another session's live stream.
        let _slot = ollama_generation_slot(&self.base_url).acquire_owned().await;

        let response = self.apply_ollama_auth(self
            .http
            .post(format!("{}/api/chat", self.base_url))
            .header("content-type", "application/json"))
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(LlmError::from_response(response).await);
        }

        let result: serde_json::Value = response.json().await?;
        ollama_response_to_anthropic(&request.model, &result)
    }

    async fn complete_anthropic_simple(&self, request: &CompletionRequest) -> Result<String, LlmError> {
        // Extract system message
        let system_msg = request
            .messages
            .iter()
            .find(|m| m.role == Role::System)
            .map(|m| m.content.clone());

        // Convert to Anthropic format
        let messages: Vec<AnthropicMessage> = request
            .messages
            .iter()
            .filter(|m| m.role != Role::System)
            .map(|m| AnthropicMessage {
                role: match m.role {
                    Role::User | Role::System => "user", // System filtered above
                    Role::Assistant => "assistant",
                }.to_string(),
                content: vec![ContentBlock::Text { text: m.content.clone() }],
            })
            .collect();

        let anthropic_request = AnthropicRequest {
            context_limit: None,
            model: request.model.clone(),
            messages,
            max_tokens: request.max_tokens.unwrap_or(4096),
            temperature: request.temperature,
            system: system_msg,
            tools: None,
            stream: None,
            thinking: None,
            cache_control: None,
        };

        let result = self.complete_anthropic(&anthropic_request).await?;

        // Extract text from response
        let text = result
            .content
            .iter()
            .filter_map(|c| match c {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");

        Ok(text)
    }

    async fn complete_openai(&self, request: &CompletionRequest) -> Result<String, LlmError> {
        #[derive(Serialize)]
        struct OpenAIRequest<'a> {
            model: &'a str,
            messages: &'a [Message],
            #[serde(skip_serializing_if = "Option::is_none")]
            max_tokens: Option<u32>,
            #[serde(skip_serializing_if = "Option::is_none")]
            temperature: Option<f32>,
        }

        pace_provider_requests(self.provider).await;

        let body = OpenAIRequest {
            model: &request.model,
            messages: &request.messages,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
        };

        let response = self
            .http
            .post(format!("{}/v1/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await?;

        learn_rate_limit_headers(self.provider, response.headers());
        let status = response.status().as_u16();
        let text = response.text().await?;
        if !(200..300).contains(&status) {
            return Err(LlmError::from_api_response(status, text));
        }
        Self::parse_openai_completion(status, &text)
    }

    /// Decode an OpenAI-compatible completion body without assuming the happy
    /// shape.
    ///
    /// A blind `response.json::<T>()` here reported every mismatch as reqwest's
    /// opaque "error decoding response body", and `OpenRouter` produces three
    /// legitimate mismatches: whitespace keep-alive padding ahead of the JSON
    /// while a reasoning model thinks, `content: null` with the text in a
    /// `reasoning` field, and — the expensive one — **HTTP 200 whose body is an
    /// `{"error": ...}` envelope** (free-tier rate limits, moderation, upstream
    /// failures). That last one made a stored, working API key look absent:
    /// every dream-summarization call failed with the same opaque error (386
    /// times on 2026-07-30 alone, zero successes ever), the owner kept
    /// re-entering the key, and nothing changed because the key was never the
    /// problem.
    ///
    /// Rules: real errors surface with the API's own message (so failover and
    /// backoff can react to rate limits), and an undecodable body is reported
    /// WITH a snippet of itself — this failure class must never be opaque
    /// again.
    fn parse_openai_completion(status: u16, text: &str) -> Result<String, LlmError> {
        #[derive(Deserialize)]
        struct OpenAIResponse {
            choices: Vec<Choice>,
        }
        #[derive(Deserialize)]
        struct Choice {
            message: ResponseMessage,
        }
        /// Response-side message: unlike the request struct, `content` may be
        /// null (reasoning models), and the usable text may live in `reasoning`.
        #[derive(Deserialize)]
        struct ResponseMessage {
            #[serde(default)]
            content: Option<String>,
            #[serde(default)]
            reasoning: Option<String>,
        }
        #[derive(Deserialize)]
        struct ErrorEnvelope {
            error: ErrorBody,
        }
        #[derive(Deserialize)]
        struct ErrorBody {
            #[serde(default)]
            message: String,
            #[serde(default)]
            code: Option<serde_json::Value>,
        }

        // serde tolerates leading whitespace, but trim anyway so the snippet in
        // the undecodable-body error starts at the content, not the padding.
        let trimmed = text.trim_start();

        if let Ok(ok) = serde_json::from_str::<OpenAIResponse>(trimmed) {
            if let Some(choice) = ok.choices.first() {
                let content = choice.message.content.as_deref().unwrap_or("");
                if !content.trim().is_empty() {
                    return Ok(content.to_string());
                }
                // Reasoning models sometimes put everything in `reasoning` and
                // leave `content` empty; a summary built from the reasoning is
                // better than reporting an empty success.
                if let Some(reasoning) = &choice.message.reasoning
                    && !reasoning.trim().is_empty() {
                        return Ok(reasoning.clone());
                    }
            }
            return Err(LlmError::from_api_response(
                status,
                "completion succeeded but contained no text".to_string(),
            ));
        }

        if let Ok(err) = serde_json::from_str::<ErrorEnvelope>(trimmed) {
            // An error body inside a 200: classify by the code the ENVELOPE
            // carries, not the transport status. OpenRouter wraps its 429s in
            // an HTTP 200, and classifying that as status 200 turns a rate
            // limit into a generic failure — which the failover walk then
            // retries at full speed, the exact opposite of what a rate limit
            // asks for.
            let embedded_code = err.error.code.as_ref().and_then(|c| {
                c.as_u64()
                    .and_then(|v| u16::try_from(v).ok())
                    .or_else(|| c.as_str().and_then(|s| s.parse().ok()))
            });
            let code_note = err
                .error
                .code
                .as_ref()
                .map_or_default(|c| format!(" (code {c})"));
            return Err(LlmError::from_api_response(
                embedded_code.unwrap_or(status),
                format!("{}{code_note}", err.error.message),
            ));
        }

        let snippet: String = trimmed.chars().take(300).collect();
        Err(LlmError::from_api_response(
            status,
            format!("undecodable completion body: {snippet:?}"),
        ))
    }

    async fn complete_ollama(&self, request: &CompletionRequest) -> Result<String, LlmError> {
        // Ollama uses a slightly different format from OpenAI
        #[derive(Serialize)]
        struct OllamaMessage<'a> {
            role: &'a str,
            content: &'a str,
        }

        #[derive(Serialize)]
        struct OllamaRequest<'a> {
            model: &'a str,
            messages: Vec<OllamaMessage<'a>>,
            stream: bool,
            #[serde(skip_serializing_if = "Option::is_none")]
            options: Option<OllamaOptions>,
        }

        #[derive(Serialize)]
        struct OllamaOptions {
            #[serde(skip_serializing_if = "Option::is_none")]
            temperature: Option<f32>,
            #[serde(skip_serializing_if = "Option::is_none")]
            num_predict: Option<u32>,
            /// Context window size - limits KV cache to reduce VRAM usage
            #[serde(skip_serializing_if = "Option::is_none")]
            num_ctx: Option<u32>,
        }

        pace_provider_requests(self.provider).await;

        let messages: Vec<OllamaMessage> = request
            .messages
            .iter()
            .map(|m| OllamaMessage {
                role: match m.role {
                    Role::System => "system",
                    Role::User => "user",
                    Role::Assistant => "assistant",
                },
                content: &m.content,
            })
            .collect();

        // Size the window the same way the Anthropic-shaped path does. Left
        // unset, Ollama used the Modelfile default — commonly 2k-8k — and
        // silently truncated the prompt server-side, which is every secondary
        // caller on this path: dream summaries, multi-agent decomposition and
        // aggregation, and context compression. They were built and budgeted
        // against the model's real window and then quietly cut down to a
        // fraction of it.
        let num_ctx = Self::resolve_num_ctx(&self.base_url, &request.model, request.context_limit);
        let options = Some(OllamaOptions {
            temperature: request.temperature,
            num_predict: request.max_tokens,
            num_ctx: Some(num_ctx),
        });

        let body = OllamaRequest {
            model: &request.model,
            messages,
            stream: false,
            options,
        };

        // Same single-slot discipline as the streaming path: a non-streaming
        // completion still occupies the server's one generation slot, and
        // firing it into a live stream cancels that stream.
        let _slot = ollama_generation_slot(&self.base_url).acquire_owned().await;

        let response = self.apply_ollama_auth(self
            .http
            .post(format!("{}/api/chat", self.base_url))
            .header("content-type", "application/json"))
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(LlmError::from_response(response).await);
        }

        let result: serde_json::Value = response.json().await?;
        ollama_completion_text(&result)
    }
}

/// The text of a non-streaming Ollama `/api/chat` completion, held to what
/// [`ollama_response_to_anthropic`] holds the agent's own path to.
///
/// `done: false` is an aborted generation, not a short answer, and inline
/// `<think>…</think>` is reasoning, not reply. This path — dream summaries,
/// context compression, multi-agent decomposition — used to take `content`
/// as-is, so an abort became an empty or truncated "summary" and a reasoning
/// model's think block was stored as memory text.
fn ollama_completion_text(response: &serde_json::Value) -> Result<String, LlmError> {
    if response.get("done").and_then(serde_json::Value::as_bool) == Some(false) {
        return Err(LlmError::from_api_response(
            502,
            "Ollama aborted generation mid-response (done=false)".to_string(),
        ));
    }
    let content = response["message"]["content"].as_str().ok_or_else(|| {
        LlmError::Json("ollama completion: no message.content in the response".to_string())
    })?;
    Ok(if content.contains("<think>") {
        strip_think_tags(content)
    } else {
        content.to_string()
    })
}

// ============================================================================
// Token Estimation
// ============================================================================

/// Per-message framing overhead (role markers + delimiters), in tokens.
///
/// Both Anthropic and `OpenAI` wrap each message in a few control tokens
/// (`<|im_start|>role … <|im_end|>` and equivalents). Ignoring it under-counts
/// long conversations by several hundred tokens.
pub const MESSAGE_FRAMING_TOKENS: usize = 4;

/// Approximate char/token ratios used by the heuristic estimator.
///
/// These replace the old blanket `len()/4`:
/// - English/prose ASCII: ~4 chars/token
/// - Source code / JSON / identifiers: denser ~3.5 chars/token
/// - CJK / wide scripts: ~1 token per character
pub const CHARS_PER_TOKEN_ENGLISH: f32 = 4.0;
pub const CHARS_PER_TOKEN_CODE: f32 = 3.5;
/// Wide-script (CJK etc.) density, tokens per character (not chars/token).
pub const TOKENS_PER_WIDE_CHAR: f32 = 1.0;

/// Content family used to pick a char/token multiplier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenContentFamily {
    /// Natural-language English / Latin prose.
    English,
    /// Source code, JSON, XML, identifiers, punctuation-heavy text.
    Code,
    /// Auto-detect from content (default).
    Auto,
}

/// Estimate token count for a string using the family-aware heuristic.
///
/// Character-class aware: ASCII/Latin text tokenizes at roughly 4 chars per
/// token (English) or ~3.5 for code-like content, while CJK and other non-ASCII
/// scripts are far denser. The old `bytes / 4` heuristic badly under-counted CJK
/// — a single CJK char is 3 UTF-8 bytes yet ~1 token — which caused real context
/// overflows. Auto mode blends English/code ratios based on a simple
/// punctuation/identifier density scan.
#[must_use]
pub fn estimate_tokens(text: &str) -> usize {
    estimate_tokens_for_family(text, TokenContentFamily::Auto)
}

/// Estimate tokens with an explicit content family.
#[must_use]
pub fn estimate_tokens_for_family(text: &str, family: TokenContentFamily) -> usize {
    if text.is_empty() {
        return 0;
    }

    let mut ascii_chars = 0usize;
    let mut wide_chars = 0usize;
    let mut code_markers = 0usize;

    for ch in text.chars() {
        if ch.is_ascii() {
            ascii_chars += 1;
            // Common code/json/identifier density signals.
            if matches!(
                ch,
                '{' | '}' | '[' | ']' | '(' | ')' | ';' | ':' | '_' | '<' | '>' | '=' | '|'
                    | '&' | '*' | '/' | '\\' | '#' | '@' | '$' | '`'
            ) {
                code_markers += 1;
            }
        } else {
            wide_chars += 1;
        }
    }

    let ratio = match family {
        TokenContentFamily::English => CHARS_PER_TOKEN_ENGLISH,
        TokenContentFamily::Code => CHARS_PER_TOKEN_CODE,
        TokenContentFamily::Auto => {
            // If ≥8% of ASCII chars are code-ish punctuation, treat as code.
            if ascii_chars > 0 && (code_markers.saturating_mul(100) / ascii_chars) >= 8 {
                CHARS_PER_TOKEN_CODE
            } else {
                CHARS_PER_TOKEN_ENGLISH
            }
        }
    };

    // Round the ASCII share up so short strings never estimate to zero.
    let ascii_tokens = if ascii_chars == 0 {
        0
    } else {
        // ceil(ascii_chars / ratio) without float hacks in the hot edge cases.
        nanna_numeric::usize_from_f32((nanna_numeric::f32_from_usize(ascii_chars) / ratio).ceil())
    };
    let wide_tokens = nanna_numeric::usize_from_f32((nanna_numeric::f32_from_usize(wide_chars) * TOKENS_PER_WIDE_CHAR).ceil());
    // At least 1 token for non-empty input.
    (ascii_tokens + wide_tokens).max(1)
}

/// Exact `OpenAI` BPE tokenization via `tiktoken-rs` when the `tiktoken` feature
/// is enabled. Falls back to the heuristic when tokenization fails or the
/// feature is off.
///
/// - prefers `o200k_base` (GPT-4o / GPT-5 / o-series) because it is the modern
///   default encoding;
/// - falls back to `cl100k_base` (GPT-4 / GPT-3.5) if o200k cannot init.
#[must_use]
pub fn estimate_tokens_exact(text: &str) -> usize {
    #[cfg(feature = "tiktoken")]
    {
        if let Some(n) = tiktoken_count(text) {
            return n;
        }
    }
    estimate_tokens(text)
}

/// Exact count with an explicit model hint (e.g. `"gpt-4o"`, `"gpt-4"`,
/// `"o3-mini"`). When the feature is off, or the model is unknown, falls back
/// to [`estimate_tokens`].
#[must_use]
pub fn estimate_tokens_for_model(text: &str, model: &str) -> usize {
    #[cfg(feature = "tiktoken")]
    {
        if let Some(n) = tiktoken_count_for_model(text, model) {
            return n;
        }
    }
    let _ = model;
    estimate_tokens(text)
}

#[cfg(feature = "tiktoken")]
fn tiktoken_count(text: &str) -> Option<usize> {
    // Prefer modern o200k (GPT-4o / GPT-5 / o*), then cl100k.
    if let Ok(bpe) = tiktoken_rs::o200k_base() {
        return Some(bpe.encode_with_special_tokens(text).len());
    }
    if let Ok(bpe) = tiktoken_rs::cl100k_base() {
        return Some(bpe.encode_with_special_tokens(text).len());
    }
    None
}

#[cfg(feature = "tiktoken")]
fn tiktoken_count_for_model(text: &str, model: &str) -> Option<usize> {
    // bpe_for_model is the preferred API in 0.12; falls through on unknown names.
    if let Ok(bpe) = tiktoken_rs::bpe_for_model(model) {
        return Some(bpe.encode_with_special_tokens(text).len());
    }
    // Heuristic family pick from the model name string.
    let lower = model.to_ascii_lowercase();
    let o200k = lower.contains("gpt-4o")
        || lower.contains("gpt-5")
        || lower.contains("gpt-4.1")
        || lower.starts_with('o')
        || lower.contains("codex")
        || lower.contains("chatgpt");
    if o200k
        && let Ok(bpe) = tiktoken_rs::o200k_base() {
            return Some(bpe.encode_with_special_tokens(text).len());
        }
    if let Ok(bpe) = tiktoken_rs::cl100k_base() {
        return Some(bpe.encode_with_special_tokens(text).len());
    }
    None
}

/// Estimate total tokens for a completion request
#[must_use]
pub fn estimate_request_tokens(request: &CompletionRequest) -> usize {
    let mut total = 0;

    // Simple messages (each carries role + delimiter framing).
    for msg in &request.messages {
        total += estimate_tokens(&msg.content) + MESSAGE_FRAMING_TOKENS;
    }

    // Anthropic messages (can have multiple content blocks)
    for msg in &request.anthropic_messages {
        total += MESSAGE_FRAMING_TOKENS;
        for block in &msg.content {
            match block {
                ContentBlock::Text { text } => total += estimate_tokens(text),
                ContentBlock::ToolUse { input, .. } => {
                    // Tool inputs are code/JSON-shaped.
                    total += estimate_tokens_for_family(
                        &input.to_string(),
                        TokenContentFamily::Code,
                    );
                }
                ContentBlock::ToolResult { content, .. } => {
                    total += estimate_tokens(content);
                }
                ContentBlock::Image { .. } => {
                    // Images are ~1000 tokens for small, more for large
                    total += 1000;
                }
                ContentBlock::Thinking { thinking, .. } => {
                    total += estimate_tokens(thinking);
                }
            }
        }
    }

    // Tools definitions
    for tool in &request.tools {
        total += estimate_tokens_for_family(&tool.to_string(), TokenContentFamily::Code);
    }

    total
}

/// Model rate limit info (tokens per minute)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ModelLimits {
    pub input_tokens_per_minute: u32,
    pub output_tokens_per_minute: u32,
    /// Requests per minute (if known)
    pub requests_per_minute: Option<u32>,
    /// Retry-after seconds from last 429 (if any)
    pub retry_after_secs: Option<u64>,
}

impl Default for ModelLimits {
    fn default() -> Self {
        Self {
            input_tokens_per_minute: 20_000,
            output_tokens_per_minute: 8_000,
            requests_per_minute: None,
            retry_after_secs: None,
        }
    }
}

/// Rate limit info parsed from API response headers
#[derive(Debug, Clone)]
pub struct RateLimitHeaders {
    /// Tokens per minute limit
    pub limit_tokens: Option<u32>,
    /// Remaining tokens this minute
    pub remaining_tokens: Option<u32>,
    /// Requests per minute limit
    pub limit_requests: Option<u32>,
    /// Remaining requests this minute
    pub remaining_requests: Option<u32>,
    /// Seconds until limit resets
    pub reset_tokens_secs: Option<u64>,
    pub reset_requests_secs: Option<u64>,
    /// Retry-after (from 429 response)
    pub retry_after: Option<u64>,
}

impl RateLimitHeaders {
    /// The `RateLimitInfo` event a stream reports for these headers, when they
    /// carry a token limit or a remaining-token count.
    const fn stream_event(&self) -> Option<StreamEvent> {
        if self.limit_tokens.is_some() || self.remaining_tokens.is_some() {
            Some(StreamEvent::RateLimitInfo {
                limit_tokens: self.limit_tokens,
                remaining_tokens: self.remaining_tokens,
                reset_secs: self.reset_tokens_secs,
            })
        } else {
            None
        }
    }

    /// Parse rate limit headers from an HTTP response.
    /// Handles both Anthropic and `OpenAI` header formats.
    #[must_use]
    pub fn from_headers(headers: &reqwest::header::HeaderMap) -> Self {
        let get_u32 = |name: &str| -> Option<u32> {
            headers.get(name)
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse().ok())
        };
        let get_u64 = |name: &str| -> Option<u64> {
            headers.get(name)
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse().ok())
        };

        Self {
            // Anthropic uses x-ratelimit-limit-tokens, x-ratelimit-remaining-tokens
            // OpenAI uses x-ratelimit-limit-tokens, x-ratelimit-remaining-tokens (same!)
            limit_tokens: get_u32("x-ratelimit-limit-tokens"),
            remaining_tokens: get_u32("x-ratelimit-remaining-tokens"),
            limit_requests: get_u32("x-ratelimit-limit-requests"),
            remaining_requests: get_u32("x-ratelimit-remaining-requests"),
            reset_tokens_secs: get_u64("x-ratelimit-reset-tokens"),
            reset_requests_secs: get_u64("x-ratelimit-reset-requests"),
            retry_after: get_u64("retry-after"),
        }
    }

    /// Convert to `ModelLimits`, using defaults where headers are missing
    #[must_use]
    pub fn to_model_limits(&self, defaults: &ModelLimits) -> ModelLimits {
        ModelLimits {
            input_tokens_per_minute: self.limit_tokens.unwrap_or(defaults.input_tokens_per_minute),
            output_tokens_per_minute: defaults.output_tokens_per_minute, // Usually not in headers
            requests_per_minute: self.limit_requests.or(defaults.requests_per_minute),
            retry_after_secs: self.retry_after.or(defaults.retry_after_secs),
        }
    }
}

impl ModelLimits {
    /// Get default limits for common models.
    /// These are conservative fallbacks — actual limits come from API headers.
    ///
    /// **Note:** These are baseline Tier 1 limits. Higher tiers get much more.
    /// The system will update these from response headers at runtime.
    #[must_use]
    pub fn for_model(model: &str) -> Self {
        match model {
            // Anthropic - Tier 1 baseline (very conservative)
            // See https://docs.anthropic.com/en/api/rate-limits
            m if m.contains("opus") => Self {
                input_tokens_per_minute: 20_000,   // Tier 1 baseline
                output_tokens_per_minute: 4_000,
                requests_per_minute: Some(50),
                retry_after_secs: None,
            },
            m if m.contains("sonnet") => Self {
                input_tokens_per_minute: 40_000,   // Tier 1 baseline
                output_tokens_per_minute: 8_000,
                requests_per_minute: Some(50),
                retry_after_secs: None,
            },
            m if m.contains("haiku") => Self {
                input_tokens_per_minute: 50_000,   // Tier 1 baseline
                output_tokens_per_minute: 10_000,
                requests_per_minute: Some(50),
                retry_after_secs: None,
            },
            // OpenAI - Tier 1 baseline
            m if m.contains("gpt-4o") => Self {
                input_tokens_per_minute: 30_000,
                output_tokens_per_minute: 10_000,
                requests_per_minute: Some(500),
                retry_after_secs: None,
            },
            m if m.contains("gpt-4") => Self {
                input_tokens_per_minute: 10_000,
                output_tokens_per_minute: 10_000,
                requests_per_minute: Some(500),
                retry_after_secs: None,
            },
            m if m.contains("gpt-3.5") => Self {
                input_tokens_per_minute: 200_000,
                output_tokens_per_minute: 50_000,
                requests_per_minute: Some(3500),
                retry_after_secs: None,
            },
            // Ollama - unlimited (local)
            m if m.contains("ollama") || m.contains("llama") || m.contains("mistral") || m.contains("qwen") => Self {
                input_tokens_per_minute: u32::MAX,
                output_tokens_per_minute: u32::MAX,
                requests_per_minute: None,
                retry_after_secs: None,
            },
            // Default conservative estimate
            _ => Self::default(),
        }
    }

    /// Check if a request would likely exceed rate limits
    #[must_use]
    pub const fn would_exceed(&self, estimated_input_tokens: usize) -> bool {
        // Compared widened rather than truncating the estimate: identical below
        // four billion tokens, and an estimate above that now reads as exceeding.
        estimated_input_tokens > self.input_tokens_per_minute as usize
    }

    /// Update limits from API response headers
    pub const fn update_from_headers(&mut self, headers: &RateLimitHeaders) {
        if let Some(limit) = headers.limit_tokens {
            self.input_tokens_per_minute = limit;
        }
        if let Some(requests) = headers.limit_requests {
            self.requests_per_minute = Some(requests);
        }
        if let Some(retry) = headers.retry_after {
            self.retry_after_secs = Some(retry);
        }
    }
}

/// Helper trait for building requests fluently.
pub trait RequestBuilder {
    /// Set the model.
    #[must_use]
    fn with_model(self, model: impl Into<String>) -> Self;
    /// Add a message.
    #[must_use]
    fn with_message(self, message: Message) -> Self;
    /// Set max tokens.
    #[must_use]
    fn with_max_tokens(self, max_tokens: u32) -> Self;
    /// Set temperature.
    #[must_use]
    fn with_temperature(self, temperature: f32) -> Self;
}

impl RequestBuilder for CompletionRequest {
    fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    fn with_message(mut self, message: Message) -> Self {
        self.messages.push(message);
        self
    }

    fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }
}

impl CompletionRequest {
    /// Set context window limit (for Ollama `num_ctx`).
    /// This reduces VRAM usage by limiting the KV cache size,
    /// allowing more model layers to fit on GPU.
    #[must_use]
    pub const fn with_context_limit(mut self, limit: u32) -> Self {
        self.context_limit = Some(limit);
        self
    }
}

// ============================================================================
// Embeddings Support
// ============================================================================

/// Embedding provider type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingProvider {
    OpenAI,
    Ollama,
}

/// A bearer token every clone shares, read when each request is built — so it
/// can be replaced while the clients holding it keep running.
///
/// An embedding client is built once, at daemon boot, and lives inside the
/// embedding router for the life of the process. Holding its token by value
/// meant a token saved afterwards never reached it: against a server that
/// wants the token, every memory embed kept getting 401 until a restart, while
/// chat — whose clients ARE rebuilt on a config change — used the new token at
/// once. Replacing the token changes nothing else about a client: not its
/// server, not its model, so not the vectors it produces.
///
/// Stored trimmed; empty means no token.
#[derive(Clone, Default)]
pub struct SharedBearer(std::sync::Arc<std::sync::RwLock<BearerState>>);

#[derive(Default)]
struct BearerState {
    token: String,
    /// Moves on every replacement that changed the token and at no other
    /// time, so a holder can tell "the credential changed since then" without
    /// keeping a copy of it.
    generation: u64,
}

impl SharedBearer {
    /// A handle holding `token`, trimmed.
    #[must_use]
    pub fn new(token: &str) -> Self {
        Self(std::sync::Arc::new(std::sync::RwLock::new(BearerState {
            token: token.trim().to_string(),
            generation: 0,
        })))
    }

    /// Make `token` (trimmed; blank clears it) the one every holder sends from
    /// its next request on. Returns whether that changed anything.
    pub fn replace(&self, token: &str) -> bool {
        let token = token.trim();
        let mut state = self
            .0
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.token == token {
            return false;
        }
        token.clone_into(&mut state.token);
        state.generation += 1;
        true
    }

    /// The token as of now; empty when there is none.
    #[must_use]
    pub fn token(&self) -> String {
        self.state().token.clone()
    }

    /// How many times the token has changed since this handle was made.
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.state().generation
    }

    fn state(&self) -> std::sync::RwLockReadGuard<'_, BearerState> {
        self.0
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl std::fmt::Debug for SharedBearer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never the token itself: a Debug print can land in a log.
        let (set, generation) = {
            let state = self.state();
            (!state.token.is_empty(), state.generation)
        };
        f.debug_struct("SharedBearer")
            .field("set", &set)
            .field("generation", &generation)
            .finish()
    }
}

/// Embedding client for generating vector embeddings
#[derive(Clone)]
pub struct EmbeddingClient {
    http: Client,
    provider: EmbeddingProvider,
    /// Read as each request is built, never cached — see [`SharedBearer`].
    api_key: SharedBearer,
    base_url: String,
    model: String,
}

impl EmbeddingClient {
    /// Create `OpenAI` embedding client
    pub fn openai(api_key: impl Into<String>) -> Self {
        Self {
            http: Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .unwrap_or_else(|_| Client::new()),
            provider: EmbeddingProvider::OpenAI,
            api_key: SharedBearer::new(&api_key.into()),
            base_url: "https://api.openai.com".to_string(),
            model: "text-embedding-3-small".to_string(),
        }
    }

    /// Create `Ollama` embedding client (local, no API key needed)
    pub fn ollama(base_url: impl Into<String>) -> Self {
        Self {
            // The exact client the 2026-08-02 chat-stream fix built — SHARED,
            // not mirrored, so the two Ollama paths cannot drift apart again:
            // a read stall deadline instead of a whole-request one (a cold
            // embedding model can take most of a minute to load, but "no
            // bytes for 120s" is a dead connection), a connect timeout, and
            // no idle pooling — a pooled loopback socket Windows quietly
            // reset is a hang on the next request, and reconnecting to
            // loopback costs nothing.
            http: LlmClient::build_ollama_http_client(),
            provider: EmbeddingProvider::Ollama,
            api_key: SharedBearer::default(),
            // The same IPv4 pin as the chat path: "localhost" resolves to
            // ::1 first on Windows, and v6 loopback is where streams were
            // observed cut mid-transfer. The 2026-08-10 embed storm ran over
            // ::1 while every healthy component used 127.0.0.1.
            base_url: LlmClient::normalize_ollama_url(&Into::<String>::into(base_url)),
            model: "nomic-embed-text".to_string(), // Good default for Ollama
        }
    }

    /// Send whatever token `bearer` holds when each request is built, instead
    /// of the one this client was made with — for a server that wants a bearer
    /// token (a remote or proxied Ollama-compatible one) and a token that can
    /// change while this client lives. A blank token sends no `Authorization`
    /// header on the Ollama paths.
    #[must_use]
    pub fn with_bearer(mut self, bearer: SharedBearer) -> Self {
        self.api_key = bearer;
        self
    }

    /// The [`SharedBearer::generation`] of this client's token: it moves
    /// exactly when the credential the next request will carry has changed.
    #[must_use]
    pub fn credential_generation(&self) -> u64 {
        self.api_key.generation()
    }

    /// `request` with this client's token as of now, or untouched when there is
    /// none — a blank token must mean no header, never `Bearer `.
    fn with_ollama_auth(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        let token = self.api_key.token();
        if token.is_empty() {
            request
        } else {
            request.header("Authorization", format!("Bearer {token}"))
        }
    }

    /// Create Ollama embedding client with default localhost URL
    #[must_use]
    pub fn ollama_default() -> Self {
        Self::ollama("http://localhost:11434")
    }

    /// Create client with custom model.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Override the base URL (e.g. for OpenRouter-compatible embedding endpoints).
    #[must_use]
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        let url: String = url.into();
        self.base_url = match self.provider {
            // Keep the v4 loopback pin even when the URL arrives after
            // construction — same rationale as [`Self::ollama`].
            EmbeddingProvider::Ollama => LlmClient::normalize_ollama_url(&url),
            EmbeddingProvider::OpenAI => url,
        };
        self
    }

    /// Get embeddings for a batch of texts.
    ///
    /// # Errors
    ///
    /// Returns `LlmError::Api` if the API returns an error.
    /// Returns `LlmError::Network` if the request fails.
    pub async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, LlmError> {
        match self.provider {
            EmbeddingProvider::OpenAI => self.embed_openai(texts).await,
            EmbeddingProvider::Ollama => self.embed_ollama(texts).await,
        }
    }

    async fn embed_openai(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, LlmError> {
        #[derive(Serialize)]
        struct EmbedRequest<'a> {
            model: &'a str,
            input: &'a [&'a str],
        }

        #[derive(Deserialize)]
        struct EmbedResponse {
            data: Vec<EmbedData>,
        }

        #[derive(Deserialize)]
        struct EmbedData {
            embedding: Vec<f32>,
        }

        let response = self
            .http
            .post(format!("{}/v1/embeddings", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key.token()))
            .header("Content-Type", "application/json")
            .json(&EmbedRequest {
                model: &self.model,
                input: texts,
            })
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(LlmError::from_response(response).await);
        }

        // Heal malformed embedding JSON bodies (some proxies/wrappers garble arrays).
        let raw = response.text().await?;
        let result: EmbedResponse = match serde_json::from_str(&raw) {
            Ok(r) => r,
            Err(e) => {
                let healed = heal::heal_json(&raw)
                    .ok_or_else(|| LlmError::Json(format!("embedding response: {e}")))?;
                serde_json::from_value(healed)
                    .map_err(|e2| LlmError::Json(format!("embedding response after heal: {e2}")))?
            }
        };
        Ok(result.data.into_iter().map(|d| d.embedding).collect())
    }

    async fn embed_ollama(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, LlmError> {
        let mut results = Vec::with_capacity(texts.len());

        for text in texts {
            let embedding = self.embed_ollama_one(text).await?;
            if embedding.is_empty() {
                return Err(LlmError::Api {
                    status: 500,
                    message: format!(
                        "Ollama returned empty embedding for model '{}'. Is the model pulled?",
                        self.model
                    ),
                });
            }
            results.push(embedding);
        }

        Ok(results)
    }

    /// Embed a single text via Ollama, trying new API then legacy.
    async fn embed_ollama_one(&self, text: &str) -> Result<Vec<f32>, LlmError> {
        #[derive(Serialize)]
        struct NewReq<'a> { model: &'a str, input: &'a str }
        #[derive(Serialize)]
        struct LegacyReq<'a> { model: &'a str, prompt: &'a str }
        #[derive(Deserialize)]
        struct NewResp { #[serde(default)] embeddings: Vec<Vec<f32>> }
        #[derive(Deserialize)]
        struct LegacyResp { #[serde(default)] embedding: Vec<f32> }

        // Try new API: POST /api/embed { model, input }
        let req = self.with_ollama_auth(
            self.http
                .post(format!("{}/api/embed", self.base_url))
                .header("Content-Type", "application/json"),
        );
        let response = req
            .json(&NewReq { model: &self.model, input: text })
            .send()
            .await?;

        if response.status().is_success() {
            let raw = response.text().await.unwrap_or_default();
            let resp: Option<NewResp> = serde_json::from_str(&raw)
                .ok()
                .or_else(|| heal::heal_json(&raw).and_then(|v| serde_json::from_value(v).ok()));
            if let Some(resp) = resp
                && let Some(emb) = resp.embeddings.into_iter().next()
                    && !emb.is_empty() {
                        return Ok(emb);
                    }
        } else {
            let status = response.status().as_u16();
            let message = response.text().await.unwrap_or_default();
            if !legacy_embed_may_answer(status, &message) {
                return Err(LlmError::from_api_response(status, message));
            }
        }

        // Fall back to legacy API: POST /api/embeddings { model, prompt }
        let req = self.with_ollama_auth(
            self.http
                .post(format!("{}/api/embeddings", self.base_url))
                .header("Content-Type", "application/json"),
        );
        let response = req
            .json(&LegacyReq { model: &self.model, prompt: text })
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(LlmError::from_response(response).await);
        }

        let raw = response.text().await?;
        let result: LegacyResp = match serde_json::from_str(&raw) {
            Ok(r) => r,
            Err(e) => {
                let healed = heal::heal_json(&raw)
                    .ok_or_else(|| LlmError::Json(format!("ollama embedding: {e}")))?;
                serde_json::from_value(healed)
                    .map_err(|e2| LlmError::Json(format!("ollama embedding after heal: {e2}")))?
            }
        };
        // `embedding` defaults to empty when absent: an empty vector is no
        // embedding, and storing it would poison every similarity it meets.
        if result.embedding.is_empty() {
            return Err(LlmError::Api {
                status: 502,
                message: format!("ollama returned no embedding for model {}", self.model),
            });
        }
        Ok(result.embedding)
    }

    /// Get embedding for a single text.
    ///
    /// A deterministic input-overflow rejection ("input length N exceeds
    /// model maximum M") is healed HERE, not surfaced: the text is cut to the
    /// proportional prefix that fits M and retried. This embeds the longest
    /// prefix the model accepts — the same prefix semantics every backend
    /// applies silently when it truncates (see [`Self::context_window`]) —
    /// where returning the error let a fallback router bench a perfectly
    /// healthy provider for 240s over the caller's own oversized input
    /// (observed 2026-08-10). The stored memory keeps its full text either
    /// way; only the vector is built from the prefix.
    ///
    /// # Errors
    ///
    /// Returns `LlmError::Api` if the API returns an error or no embedding is returned.
    /// Returns `LlmError::Network` if the request fails.
    pub async fn embed_one(&self, text: &str) -> Result<Vec<f32>, LlmError> {
        let mut shrunk: Option<String> = None;
        loop {
            let attempt = shrunk.as_deref().unwrap_or(text);
            let err = match self.embed(&[attempt]).await {
                Ok(mut results) => {
                    return results.pop().ok_or_else(|| LlmError::Api {
                        status: 500,
                        message: "No embedding returned".to_string(),
                    });
                }
                Err(e) => e,
            };
            // No retry cap, and none needed: each pass keeps strictly fewer
            // chars than the last (limit < sent), so the walk terminates —
            // geometrically, in practice in one or two passes.
            let Some(shorter) = shrink_to_input_limit(&err, attempt) else {
                return Err(err);
            };
            debug!(
                from_chars = attempt.chars().count(),
                to_chars = shorter.chars().count(),
                "embedding input exceeded the model's window — retrying with the fitting prefix"
            );
            shrunk = Some(shorter);
        }
    }

    /// The maximum sequence length this embedding model accepts, in tokens.
    ///
    /// `None` means "not discoverable", which is different from "unbounded" —
    /// the caller must treat it as *no ceiling learned* and fall back to its own
    /// default, never as permission to send arbitrarily long text.
    ///
    /// This bound matters because exceeding it does not fail: every embedding
    /// backend silently truncates to the sequence limit and returns a
    /// well-formed vector for the prefix. The tail is not embedded, is not
    /// retrievable, and nothing reports it.
    pub async fn context_window(&self) -> Option<usize> {
        match self.provider {
            EmbeddingProvider::Ollama => self.ollama_context_window().await,
            // No OpenAI-compatible embeddings endpoint publishes a sequence
            // limit, and `/v1/models` carries no field for one. Guessing from
            // the model name would bake a table that silently rots as models
            // are added; an honest `None` sends the caller to its own default.
            EmbeddingProvider::OpenAI => None,
        }
    }

    async fn ollama_context_window(&self) -> Option<usize> {
        let response = self
            .with_ollama_auth(self.http.post(format!("{}/api/show", self.base_url)))
            .json(&serde_json::json!({ "name": self.model }))
            .send()
            .await
            .ok()?;
        if !response.status().is_success() {
            return None;
        }
        let show: OllamaShowResponse = response.json().await.ok()?;
        gguf_metadata_usize(&show.model_info?, "context_length")
    }
}

/// The strictly-shorter prefix of `text` that should fit the limit named by an
/// input-overflow error, or `None` when `err` is not that fault or no shorter
/// prefix exists.
///
/// The provider reports its unit (tokens); the cut happens in chars. The
/// proportional cut `limit/sent` maps one to the other well enough because the
/// caller LOOPS: a residual overflow produces a fresh, smaller report, the
/// char count strictly decreases every pass, and the walk terminates without
/// needing a retry cap.
fn shrink_to_input_limit(err: &LlmError, text: &str) -> Option<String> {
    if !err.is_input_overflow() {
        return None;
    }
    let LlmError::Api { message, .. } = err else {
        return None;
    };
    let (sent, limit) = input_overflow_limits(message)?;
    debug_assert!(sent > limit, "input_overflow_limits guarantees sent > limit");
    let total_chars = text.chars().count();
    let target_chars = total_chars.saturating_mul(limit) / sent;
    if target_chars == 0 || target_chars >= total_chars {
        // Nothing left to cut (or the ratio math degenerated) — this fault
        // cannot be healed by shrinking, so let the caller surface the error.
        return None;
    }
    Some(text.chars().take(target_chars).collect())
}

// ============================================================================
// Streaming Support
// ============================================================================

/// A streaming event from the Anthropic API
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// Start of a new message. Anthropic reports prompt-side token usage here
    /// (including prompt-cache hits/writes); other providers report zeros.
    MessageStart {
        id: String,
        model: String,
        /// Non-cached prompt tokens billed for this message.
        input_tokens: u32,
        /// Prompt tokens served from the cache (a read, cheaper).
        cache_read_tokens: u32,
        /// Prompt tokens written into the cache (a creation, one-time cost).
        cache_creation_tokens: u32,
        /// The 1-hour share of `cache_creation_tokens` (billed at 2x input, not 1.25x).
        cache_creation_1h_tokens: u32,
    },
    /// Start of a content block
    ContentBlockStart {
        index: usize,
        content_type: String,
        /// Tool use ID (only for `tool_use` blocks)
        tool_id: Option<String>,
        /// Tool name (only for `tool_use` blocks)
        tool_name: Option<String>,
    },
    /// Text delta within a content block
    TextDelta { index: usize, text: String },
    /// Thinking/reasoning delta (extended thinking mode)
    ThinkingDelta { index: usize, thinking: String },
    /// Signature delta for a thinking block (must be accumulated and sent back to Anthropic)
    SignatureDelta { index: usize, signature: String },
    /// Tool use delta (JSON fragment)
    ToolUseDelta { index: usize, partial_json: String },
    /// Content block finished
    ContentBlockStop { index: usize },
    /// Message finished
    MessageStop { stop_reason: String },
    /// Usage statistics
    MessageDelta {
        stop_reason: Option<String>,
        output_tokens: u32,
    },
    /// Ping (keepalive)
    Ping,
    /// Error event (non-recoverable)
    Error { message: String },
    /// Recoverable error mid-stream (e.g., 429 rate limit).
    /// Contains accumulated partial content that can be used to continue.
    RecoverableError {
        error: LlmError,
        /// Text accumulated before the error
        partial_text: String,
        /// In-progress tool calls (id, name, `partial_json`)
        partial_tool_calls: Vec<(String, String, String)>,
    },
    /// Rate limit headers received (allows updating limits cache)
    RateLimitInfo {
        limit_tokens: Option<u32>,
        remaining_tokens: Option<u32>,
        reset_secs: Option<u64>,
    },
}

/// Accumulated state during streaming (for recovery)
#[derive(Debug, Clone, Default)]
pub struct StreamAccumulator {
    /// Text content accumulated so far
    pub text: String,
    /// Thinking/reasoning content accumulated so far
    pub thinking: String,
    /// Accumulated signature for the thinking block (required by Anthropic in multi-turn)
    pub thinking_signature: String,
    /// Tool calls in progress: (index, id, name, `partial_json`)
    pub tool_calls: Vec<(usize, String, String, String)>,
}

impl StreamAccumulator {
    /// Create a new empty accumulator
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Process an event and accumulate content
    pub fn process(&mut self, event: &StreamEvent) {
        match event {
            StreamEvent::TextDelta { text, .. } => {
                self.text.push_str(text);
            }
            StreamEvent::ThinkingDelta { thinking, .. } => {
                self.thinking.push_str(thinking);
            }
            StreamEvent::SignatureDelta { signature, .. } => {
                self.thinking_signature.push_str(signature);
            }
            StreamEvent::ContentBlockStart { index, content_type, tool_id, tool_name } => {
                if content_type == "tool_use"
                    && let (Some(id), Some(name)) = (tool_id, tool_name) {
                        self.tool_calls.push((*index, id.clone(), name.clone(), String::new()));
                    }
            }
            StreamEvent::ToolUseDelta { index, partial_json } => {
                if let Some((_, _, _, json)) = self.tool_calls.iter_mut().find(|(i, _, _, _)| i == index) {
                    json.push_str(partial_json);
                }
            }
            _ => {}
        }
    }

    /// Get partial tool calls as (id, name, json) tuples
    #[must_use]
    pub fn partial_tool_calls(&self) -> Vec<(String, String, String)> {
        self.tool_calls.iter()
            .map(|(_, id, name, json)| (id.clone(), name.clone(), json.clone()))
            .collect()
    }

    /// Check if there's any accumulated content
    #[must_use]
    pub const fn has_content(&self) -> bool {
        !self.text.is_empty() || !self.thinking.is_empty() || !self.tool_calls.is_empty()
    }

    /// The item to yield when a stream breaks mid-flight: a recoverable error
    /// carrying the partial output when there is any, otherwise the bare error.
    fn interrupted(&self, error: LlmError) -> Result<StreamEvent, LlmError> {
        if self.has_content() {
            Ok(StreamEvent::RecoverableError {
                error,
                partial_text: self.text.clone(),
                partial_tool_calls: self.partial_tool_calls(),
            })
        } else {
            Err(error)
        }
    }
}

/// The OAuth ("Claude Code") form of a streaming request: an array-based system
/// prompt, and tool names remapped to Claude Code's canonical ones.
fn oauth_stream_request(request: &AnthropicRequest) -> OAuthAnthropicRequest {
    info!("Claude Code stealth mode: applying OAuth headers and array-based system prompt");

    let mut oauth_req = OAuthAnthropicRequest::from_request(request, true);
    oauth_req.stream = Some(true);

    // Remap tool names to Claude Code format
    if let Some(ref mut tools) = oauth_req.tools {
        for tool in tools {
            let old_name = tool.name.clone();
            tool.name = to_claude_code_tool_name(&tool.name);
            if old_name != tool.name {
                debug!(old = %old_name, new = %tool.name, "Remapped tool name");
            }
        }
    }

    debug!(
        system_blocks = oauth_req.system.len(),
        tool_names = ?oauth_req.tools.as_ref().map(|t| t.iter().map(|x| x.name.as_str()).collect::<Vec<_>>()),
        "OAuth streaming request prepared with array-based system prompt"
    );

    oauth_req
}

/// Authentication headers for a native Anthropic streaming request: Claude
/// Code's exact header set for an OAuth token, `x-api-key` otherwise.
fn with_stream_anthropic_auth(
    request_builder: reqwest::RequestBuilder,
    api_key: &str,
    is_oauth: bool,
) -> reqwest::RequestBuilder {
    if is_oauth {
        let token = get_raw_token(api_key);
        request_builder
            .header("Authorization", format!("Bearer {token}"))
            .header("accept", "application/json")
            .header("anthropic-beta", "claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14,fine-grained-tool-streaming-2025-05-14")
            .header("user-agent", format!("claude-cli/{CLAUDE_CODE_VERSION} (external, cli)"))
            .header("x-app", "cli")
            .header("anthropic-dangerous-direct-browser-access", "true")
    } else {
        request_builder.header("x-api-key", api_key)
    }
}

/// Map a Claude Code tool name in an OAuth stream's `ContentBlockStart` back to
/// the name the caller registered.
fn restore_oauth_tool_name(event: &mut StreamEvent, original_tools: &[ToolDefinition]) {
    if let StreamEvent::ContentBlockStart { tool_name: Some(ref mut name), .. } = *event {
        *name = from_claude_code_tool_name(name, original_tools);
    }
}

/// Anthropic SSE event types
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicSSE {
    MessageStart { message: MessageStartData },
    ContentBlockStart { index: usize, content_block: ContentBlockData },
    ContentBlockDelta { index: usize, delta: DeltaData },
    ContentBlockStop { index: usize },
    MessageDelta { delta: MessageDeltaData, usage: Option<UsageDelta> },
    MessageStop,
    Ping,
    Error { error: ErrorData },
}

#[derive(Debug, Deserialize)]
struct MessageStartData {
    id: String,
    model: String,
    #[serde(default)]
    usage: Option<MessageStartUsage>,
}

/// Prompt-side token usage reported in the Anthropic `message_start` event.
/// Field names mirror the wire JSON keys.
#[derive(Debug, Deserialize, Default)]
struct MessageStartUsage {
    #[serde(default)]
    input_tokens: u32,
    #[serde(default)]
    cache_read_input_tokens: u32,
    #[serde(default)]
    cache_creation_input_tokens: u32,
    #[serde(default)]
    cache_creation: Option<CacheCreation>,
}

#[derive(Debug, Deserialize)]
struct ContentBlockData {
    #[serde(rename = "type")]
    block_type: String,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    name: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum DeltaData {
    TextDelta { text: String },
    ThinkingDelta { thinking: String },
    InputJsonDelta { partial_json: String },
    /// Signature delta for thinking blocks (must be accumulated and sent back)
    SignatureDelta { signature: String },
    /// Catch-all for unknown delta types so the
    /// surrounding `ContentBlockDelta` event still deserializes successfully.
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
struct MessageDeltaData {
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct UsageDelta {
    output_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct ErrorData {
    /// Anthropic's error type (`overloaded_error`, `rate_limit_error`, …). An
    /// SSE `error` event carries no HTTP status, so this is the only thing
    /// that says whether another model is worth trying.
    #[serde(rename = "type", default)]
    error_type: String,
    message: String,
}

/// The HTTP status Anthropic uses for each documented error type, so an error
/// that arrives inside a 200 stream is judged by the same
/// [`LlmError::should_fallback`] rule as one that arrives as a status.
/// Unknown types map to 500, the status of `api_error`.
fn anthropic_error_status(error_type: &str) -> u16 {
    match error_type {
        "invalid_request_error" => 400,
        "authentication_error" => 401,
        "permission_error" => 403,
        "not_found_error" => 404,
        "request_too_large" => 413,
        "rate_limit_error" => 429,
        "overloaded_error" => 529,
        _ => 500,
    }
}

impl LlmClient {
    /// Stream an Anthropic-format request, dispatching to the appropriate provider.
    ///
    /// - Anthropic: Native SSE streaming via `/v1/messages`
    /// - OpenAI/OpenRouter/GitHub/ClaudeProxy: Converts to `OpenAI` SSE streaming
    /// - Ollama: Converts to Ollama streaming via `/api/chat`
    pub fn stream_anthropic(
        &self,
        request: &AnthropicRequest,
    ) -> impl Stream<Item = Result<StreamEvent, LlmError>> + '_ {
        let provider = self.provider;
        // Same boundary correction as the non-streaming path — a streamed
        // request carries the identical body and 400s the same way.
        let request = conform_to_anthropic_contract(request).into_owned();

        stream! {
            // One gate for every streamed request regardless of which backend
            // serves it — a stream spends the same quota as a completion.
            pace_provider_requests(provider).await;
            match provider {
                Provider::Anthropic => {
                    let raw = self.stream_anthropic_native(&request);
                    tokio::pin!(raw);
                    while let Some(item) = raw.next().await {
                        yield item;
                    }
                }
                Provider::OpenAI | Provider::OpenRouter | Provider::GitHubModels | Provider::ClaudeProxy => {
                    let raw = self.stream_anthropic_via_openai(&request);
                    tokio::pin!(raw);
                    while let Some(item) = raw.next().await {
                        yield item;
                    }
                }
                Provider::Ollama => {
                    let raw = self.stream_anthropic_via_ollama(&request);
                    tokio::pin!(raw);
                    while let Some(item) = raw.next().await {
                        yield item;
                    }
                }
            }
        }
    }

    /// Native Anthropic SSE streaming implementation
    fn stream_anthropic_native(
        &self,
        request: &AnthropicRequest,
    ) -> impl Stream<Item = Result<StreamEvent, LlmError>> + '_ {
        let mut request = request.clone();
        request.stream = Some(true);

        let http = self.http.clone();
        let api_key = self.api_key.clone();
        let base_url = self.base_url.clone();
        let is_oauth = is_oauth_token(&api_key);

        // Store original tools for reverse mapping
        let original_tools = request.tools.clone().unwrap_or_default();

        // Prepare OAuth request with array-based system prompt
        let oauth_request = is_oauth.then(|| oauth_stream_request(&request));

        stream! {
            let request_builder = with_stream_anthropic_auth(
                http.post(format!("{base_url}/v1/messages")),
                &api_key,
                is_oauth,
            );

            debug!(
                is_oauth = is_oauth,
                model = %request.model,
                has_system = request.system.is_some(),
                tools_count = request.tools.as_ref().map_or(0, std::vec::Vec::len),
                "Sending Anthropic streaming request"
            );

            let request_builder = request_builder
                .header("anthropic-version", "2023-06-01")
                .header("content-type", "application/json");
            let response = if let Some(ref oauth_req) = oauth_request {
                request_builder.json(oauth_req).send().await
            } else {
                request_builder.json(&request).send().await
            };

            let response = match response {
                Ok(r) => {
                    debug!(status = %r.status(), "Anthropic API response received");
                    r
                },
                Err(e) => {
                    warn!(error = %e, "Anthropic API request failed");
                    yield Err(LlmError::Http(e.to_string()));
                    return;
                }
            };

            if let Some(info) = RateLimitHeaders::from_headers(response.headers()).stream_event() {
                yield Ok(info);
            }

            if !response.status().is_success() {
                yield Err(LlmError::from_response(response).await);
                return;
            }

            let mut byte_stream = response.bytes_stream();
            let mut buffer = String::new();
            let mut accumulator = StreamAccumulator::new();

            let mut utf8 = Utf8StreamDecoder::default();
            while let Some(chunk_result) = byte_stream.next().await {
                let chunk = match chunk_result {
                    Ok(c) => c,
                    Err(e) => {
                        yield accumulator.interrupted(LlmError::Http(e.to_string()));
                        return;
                    }
                };

                match utf8.push(&chunk) {
                    Ok(s) => buffer.push_str(&s),
                    Err(e) => {
                        yield accumulator.interrupted(LlmError::Stream(format!("Invalid UTF-8 in stream: {e}")));
                        return;
                    }
                }

                while let Some(event_str) = extract_sse_event(&mut buffer) {
                    if let Some(mut stream_event) = parse_sse_event(&event_str) {
                        if is_oauth {
                            restore_oauth_tool_name(&mut stream_event, &original_tools);
                        }
                        accumulator.process(&stream_event);
                        yield Ok(stream_event);
                    }
                }
            }

            if !buffer.trim().is_empty()
                && let Some(mut stream_event) = parse_sse_event(&buffer) {
                    if is_oauth {
                        restore_oauth_tool_name(&mut stream_event, &original_tools);
                    }
                    accumulator.process(&stream_event);
                    yield Ok(stream_event);
                }
        }
    }

    /// Stream an `AnthropicRequest` via OpenAI-compatible SSE streaming.
    /// Converts the request to `OpenAI` format, streams via /v1/chat/completions,
    /// and converts events back to Anthropic `StreamEvent` format.
    fn stream_anthropic_via_openai(
        &self,
        request: &AnthropicRequest,
    ) -> impl Stream<Item = Result<StreamEvent, LlmError>> + '_ {
        let http = self.http.clone();
        let provider = self.provider;
        let api_key = self.api_key.clone();
        let base_url = self.base_url.clone();
        let request = request.clone();

        stream! {
            let (messages_json, tools_json) = anthropic_to_openai_request(&request);

            let mut body = serde_json::json!({
                "model": request.model,
                "messages": messages_json,
                "max_tokens": request.max_tokens,
                "stream": true,
            });
            if let Some(temp) = request.temperature {
                body["temperature"] = serde_json::json!(temp);
            }
            if let Some(tools) = tools_json {
                body["tools"] = tools;
            }

            let response = match http
                .post(format!("{base_url}/v1/chat/completions"))
                .header("Authorization", format!("Bearer {api_key}"))
                .header("content-type", "application/json")
                .json(&body)
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    yield Err(LlmError::Http(e.to_string()));
                    return;
                }
            };
            learn_rate_limit_headers(provider, response.headers());

            if !response.status().is_success() {
                yield Err(LlmError::from_response(response).await);
                return;
            }

            // Reuse OpenAI SSE parsing logic (same format as stream_openai)
            let mut byte_stream = response.bytes_stream();
            let mut buffer = String::new();
            let mut state = OpenAiStreamState::default();

            let mut utf8 = Utf8StreamDecoder::default();
            while let Some(chunk_result) = byte_stream.next().await {
                let chunk = match chunk_result {
                    Ok(c) => c,
                    Err(e) => {
                        yield Err(LlmError::Stream(e.to_string()));
                        return;
                    }
                };

                match utf8.push(&chunk) {
                    Ok(s) => buffer.push_str(&s),
                    Err(e) => {
                        yield Err(LlmError::Stream(format!("Invalid UTF-8: {e}")));
                        return;
                    }
                }

                while let Some(line_end) = buffer.find('\n') {
                    let line = buffer[..line_end].trim().to_string();
                    buffer = buffer[line_end + 1..].to_string();

                    let (events, finished) = match state.on_line::<OpenAiChunkDelta>(&line) {
                        Ok(translated) => translated,
                        Err(e) => {
                            yield Err(e);
                            return;
                        }
                    };
                    for event in events {
                        yield Ok(event);
                    }
                    if finished {
                        return;
                    }
                }
            }

            // Emit close events if stream ended without finish_reason
            for event in state.close_blocks() {
                yield Ok(event);
            }
            yield Ok(StreamEvent::MessageStop { stop_reason: "end_turn".to_string() });
        }
    }

    /// Stream an `AnthropicRequest` via Ollama `/api/chat` with `stream: true`.
    /// Ollama streams NDJSON (one JSON object per line, not SSE).
    fn stream_anthropic_via_ollama(
        &self,
        request: &AnthropicRequest,
    ) -> impl Stream<Item = Result<StreamEvent, LlmError>> + '_ {
        let http = self.http.clone();
        let base_url = self.base_url.clone();
        let api_key = self.api_key.clone();
        let request = request.clone();

        stream! {
            // Hold the server's single generation slot for the WHOLE stream.
            // The permit lives as long as this generator, so it is released
            // when the stream finishes, errors, or is dropped mid-flight.
            // Acquire before building the body: the wait is the point.
            let _slot = ollama_generation_slot(&base_url).acquire_owned().await;

            let body = Self::ollama_stream_body(&base_url, &request);

            let mut req = http
                .post(format!("{base_url}/api/chat"))
                .header("content-type", "application/json");
            if !api_key.is_empty() {
                req = req.header("Authorization", format!("Bearer {api_key}"));
            }
            let response = match req
                .json(&body)
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    yield Err(LlmError::Http(e.to_string()));
                    return;
                }
            };

            if !response.status().is_success() {
                yield Err(LlmError::from_response(response).await);
                return;
            }

            // Ollama streams NDJSON: one JSON per line
            let mut byte_stream = response.bytes_stream();
            let mut buffer = String::new();
            let mut bytes_received = 0usize;
            let mut lines_parsed = 0usize;
            let mut state = OllamaStreamState::default();
            let mut last_line: Option<String> = None;

            let mut utf8 = Utf8StreamDecoder::default();
            while let Some(chunk_result) = byte_stream.next().await {
                let chunk = match chunk_result {
                    Ok(c) => c,
                    Err(e) => {
                        yield Err(LlmError::Stream(e.to_string()));
                        return;
                    }
                };

                bytes_received += chunk.len();
                match utf8.push(&chunk) {
                    Ok(s) => buffer.push_str(&s),
                    Err(e) => {
                        yield Err(LlmError::Stream(format!("Invalid UTF-8: {e}")));
                        return;
                    }
                }

                // Process complete lines (NDJSON)
                while let Some(line_end) = buffer.find('\n') {
                    let line = buffer[..line_end].trim().to_string();
                    buffer = buffer[line_end + 1..].to_string();

                    if line.is_empty() {
                        continue;
                    }

                    let Ok(obj) = serde_json::from_str::<serde_json::Value>(&line) else {
                        continue;
                    };
                    lines_parsed += 1;
                    // Keep the last line so an abort can show what the model
                    // was actually emitting when the body stopped. Without it
                    // a truncation is indistinguishable from a wedged decoder,
                    // which is exactly the ambiguity that has cost two runs.
                    last_line = Some(line.clone());

                    let (items, finished) = state.on_object(&obj, bytes_received);
                    for item in items {
                        yield item;
                    }
                    if finished {
                        return;
                    }
                }
            }

            for item in state.finish(&buffer, last_line.as_deref(), bytes_received, lines_parsed) {
                yield item;
            }
        }
    }

    /// The `/api/chat` body for a streaming Ollama request: messages and tools
    /// in Ollama's native shape, plus the sampling options and the `num_ctx`
    /// sized for this request.
    fn ollama_stream_body(base_url: &str, request: &AnthropicRequest) -> serde_json::Value {
        let (messages_json, tools_json) = anthropic_to_ollama_request(request);

        let mut body = serde_json::json!({
            "model": request.model,
            "messages": messages_json,
            "stream": true,
        });

        let mut options = serde_json::Map::new();
        if let Some(temp) = request.temperature {
            options.insert("temperature".to_string(), serde_json::json!(temp));
        } else {
            // See the non-streaming site: Modelfile thinking defaults run
            // hot (1.0); 0.6 is Qwen's non-thinking recommendation and
            // measurably reduces one-line code cramming.
            options.insert("temperature".to_string(), serde_json::json!(0.6));
        }
        // Bound generation to the request's token budget. num_predict=-1
        // (unlimited) let a degraded runner emit degenerate tokens without
        // end; the caller's max_tokens already accounts for thinking room.
        options.insert("num_predict".to_string(), serde_json::json!(request.max_tokens));
        // Size the context to the GPU as it is RIGHT NOW, every turn — see
        // `resolve_num_ctx`. This is the path the agent loop takes, so a
        // constant here is the one that actually decides whether a run
        // survives: 32k was unrunnable for a 12B model on a 16 GB card
        // sharing with the desktop, and it failed as
        // `CUDA error: an illegal memory access was encountered` rather
        // than as a clean allocation failure.
        let num_ctx = Self::resolve_num_ctx(base_url, &request.model, request.context_limit);
        tracing::info!(
            model = %request.model,
            num_ctx,
            "sized ollama context to free VRAM (stream)"
        );
        options.insert("num_ctx".to_string(), serde_json::json!(num_ctx));
        // qwen3.5's Modelfile ships presence_penalty=1.5 (Qwen's thinking-
        // mode anti-repetition default). Code REQUIRES repetition, and at
        // 1.5 the model visibly degraded working multi-function files into
        // semicolon one-liner slop (observed live, round-5 drive). 1.0
        // keeps a mild guard; the loop's repetition/spiral detectors
        // cover the rest.
        options.insert("presence_penalty".to_string(), serde_json::json!(1.0));
        // repeat_penalty 1.1 (the Ollama default) punishes exactly the tokens
        // Python repeats by design — newlines and indentation runs — and the
        // model visibly compensates by cramming statements onto one line
        // (observed live across rounds even at presence 1.0). Code presets
        // use 1.0: no repetition penalty at all.
        options.insert("repeat_penalty".to_string(), serde_json::json!(1.0));
        if !options.is_empty() {
            body["options"] = serde_json::Value::Object(options);
        }
        if let Some(tools) = tools_json {
            body["tools"] = tools;
        }
        // The same request the non-streaming path makes. Without it the agent's
        // own (streaming) path got reasoning inline in `content`, leaving the
        // tag splitter as the only thing keeping it out of the reply.
        if ollama_separates_thinking(&request.model) {
            body["think"] = serde_json::json!(true);
        }
        body
    }

    /// Stream using OpenAI-compatible API (for `OpenAI`, `OpenRouter`, GitHub Models, Claude Proxy)
    /// Supports both text responses and tool calls.
    pub fn stream_openai(
        &self,
        request: &CompletionRequest,
    ) -> impl Stream<Item = Result<StreamEvent, LlmError>> + '_ {
        let http = self.http.clone();
        let api_key = self.api_key.clone();
        let base_url = self.base_url.clone();
        let request = request.clone();
        let provider = self.provider;

        stream! {
            // Same pacing gate as the non-streaming paths — a streamed chat
            // request spends the same quota as any other.
            pace_provider_requests(provider).await;

            let body = openai_stream_body(&request);

            let response = match http
                .post(format!("{base_url}/v1/chat/completions"))
                .header("Authorization", format!("Bearer {api_key}"))
                .header("content-type", "application/json")
                .json(&body)
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    yield Err(LlmError::Http(e.to_string()));
                    return;
                }
            };
            learn_rate_limit_headers(provider, response.headers());

            if !response.status().is_success() {
                yield Err(LlmError::from_response(response).await);
                return;
            }

            let mut byte_stream = response.bytes_stream();
            let mut buffer = String::new();

            // Track content blocks (text at index 0, tool calls at higher indices)
            let mut state = OpenAiStreamState::default();

            let mut utf8 = Utf8StreamDecoder::default();
            while let Some(chunk_result) = byte_stream.next().await {
                let chunk = match chunk_result {
                    Ok(c) => c,
                    Err(e) => {
                        yield Err(LlmError::Stream(e.to_string()));
                        return;
                    }
                };

                match utf8.push(&chunk) {
                    Ok(s) => buffer.push_str(&s),
                    Err(e) => {
                        yield Err(LlmError::Stream(format!("Invalid UTF-8: {e}")));
                        return;
                    }
                }

                // Parse SSE events from buffer
                while let Some(line_end) = buffer.find('\n') {
                    let line = buffer[..line_end].trim().to_string();
                    buffer = buffer[line_end + 1..].to_string();

                    let (events, finished) = match state.on_line::<OpenAiRoleCheckedDelta>(&line) {
                        Ok(translated) => translated,
                        Err(e) => {
                            yield Err(e);
                            return;
                        }
                    };
                    for event in events {
                        yield Ok(event);
                    }
                    if finished {
                        return;
                    }
                }
            }

            // If we got here without MessageStop, emit it
            for event in state.close_blocks() {
                yield Ok(event);
            }
            yield Ok(StreamEvent::MessageStop { stop_reason: "end_turn".to_string() });
        }
    }

    /// Convenience method to stream a `CompletionRequest`.
    /// Converts to provider-specific format automatically.
    ///
    /// Returns `StreamEvent` directly (errors converted to `Error` or `RecoverableError` events).
    pub fn stream(
        &self,
        request: &CompletionRequest,
    ) -> impl Stream<Item = StreamEvent> + '_ {
        let provider = self.provider;
        let request = request.clone();

        stream! {
            // Route to appropriate streaming method based on provider
            match provider {
                // Ollama streaming builds the same AnthropicRequest and delegates to
                // stream_anthropic, which already handles Ollama NDJSON streaming via
                // stream_anthropic_via_ollama.
                Provider::Anthropic | Provider::Ollama => {
                    // Convert to Anthropic format
                    let anthropic_request = streaming_anthropic_request(&request);
                    let raw_stream = self.stream_anthropic(&anthropic_request);
                    tokio::pin!(raw_stream);

                    while let Some(result) = raw_stream.next().await {
                        match result {
                            Ok(event) => yield event,
                            Err(e) => {
                                yield stream_error_event(e);
                                return;
                            }
                        }
                    }
                }
                Provider::OpenAI | Provider::OpenRouter | Provider::GitHubModels | Provider::ClaudeProxy => {
                    // Use OpenAI-compatible streaming
                    let raw_stream = self.stream_openai(&request);
                    tokio::pin!(raw_stream);

                    while let Some(result) = raw_stream.next().await {
                        match result {
                            Ok(event) => yield event,
                            Err(e) => {
                                yield stream_error_event(e);
                                return;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Stream with explicit accumulator for advanced recovery scenarios.
    ///
    /// Returns both events and the final accumulator state (even on error).
    pub fn stream_with_recovery(
        &self,
        request: &CompletionRequest,
    ) -> impl Stream<Item = (StreamEvent, StreamAccumulator)> + '_ {
        let base_stream = self.stream(request);

        stream! {
            let mut accumulator = StreamAccumulator::new();
            tokio::pin!(base_stream);

            while let Some(event) = base_stream.next().await {
                accumulator.process(&event);
                yield (event, accumulator.clone());
            }
        }
    }
}

/// The streaming `AnthropicRequest` that [`LlmClient::stream`] sends for a
/// `CompletionRequest` on the Anthropic and Ollama paths.
fn streaming_anthropic_request(request: &CompletionRequest) -> AnthropicRequest {
    let system_msg = request
        .messages
        .iter()
        .find(|m| m.role == Role::System)
        .map(|m| m.content.clone());

    let messages: Vec<AnthropicMessage> = request
        .messages
        .iter()
        .filter(|m| m.role != Role::System)
        .map(|m| AnthropicMessage {
            role: match m.role {
                Role::User | Role::System => "user",
                Role::Assistant => "assistant",
            }.to_string(),
            content: vec![ContentBlock::Text { text: m.content.clone() }],
        })
        .collect();

    let mut all_messages = messages;
    all_messages.extend(request.anthropic_messages.clone());

    let tools = if request.tools.is_empty() {
        None
    } else {
        Some(request.tools.iter().filter_map(|v| {
            serde_json::from_value::<ToolDefinition>(v.clone()).ok()
        }).collect())
    };

    AnthropicRequest {
        context_limit: None,
        model: request.model.clone(),
        messages: all_messages,
        max_tokens: request.max_tokens.unwrap_or(4096),
        // Model-family correction happens at the dispatcher
        // (`conform_to_anthropic_contract`), which sees the
        // resolved model; pass the caller's value through.
        temperature: request.temperature,
        system: system_msg,
        tools,
        stream: Some(true),
        thinking: request.thinking.clone(),
        cache_control: None,
    }
}

/// The event [`LlmClient::stream`] ends with when the provider stream fails:
/// recoverable when the error warrants a fallback to another model.
fn stream_error_event(e: LlmError) -> StreamEvent {
    if e.should_fallback() {
        StreamEvent::RecoverableError {
            error: e,
            partial_text: String::new(),
            partial_tool_calls: Vec::new(),
        }
    } else {
        StreamEvent::Error { message: e.to_string() }
    }
}

// ============================================================================
// Ollama streaming: NDJSON translation
// ============================================================================

/// A tool-use id for a provider that sends none (Ollama).
///
/// Unique for the life of the process. These ids used to be a per-response
/// counter (`toolu_00000001` for the first call of EVERY response), so one
/// step's context held several different calls under one id: consumers keyed
/// by id had to learn to tolerate collisions (the timeline journal and the
/// GUI both carry workarounds), and a request replaying that context to a
/// provider that requires unique ids — Anthropic, on a fallback mid-step —
/// would be refused. Seeded from the clock so a restarted daemon does not
/// begin again at the ids its persisted timelines already hold.
fn synthesized_tool_use_id() -> String {
    use std::sync::LazyLock;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: LazyLock<AtomicU64> = LazyLock::new(|| {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |since| since.as_secs() << 20);
        AtomicU64::new(seed)
    });
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    debug_assert!(
        n < u64::MAX,
        "the counter cannot wrap within a process lifetime"
    );
    let id = format!("toolu_{n:016x}");
    debug_assert_eq!(id.len(), "toolu_".len() + 16, "fixed-width ids");
    id
}

/// Incremental UTF-8 decoding for a provider byte stream.
///
/// Network chunks do not respect character boundaries. Every stream reader
/// here decoded each chunk on its own, so a multibyte character split across
/// two chunks — an emoji, an accent, any CJK — failed the whole stream with
/// `Invalid UTF-8: incomplete utf-8 byte sequence`, and the retries hit the
/// same wall: a long non-ASCII reply came back as "could not run". This
/// carries the incomplete tail (at most 3 bytes: the longest UTF-8 sequence
/// minus one) into the next chunk, and still rejects bytes that are invalid
/// rather than merely incomplete.
#[derive(Default)]
struct Utf8StreamDecoder {
    carry: Vec<u8>,
}

impl Utf8StreamDecoder {
    /// Decode `chunk` after whatever the previous chunk left incomplete.
    fn push(&mut self, chunk: &[u8]) -> Result<String, std::str::Utf8Error> {
        let mut bytes = std::mem::take(&mut self.carry);
        bytes.extend_from_slice(chunk);
        match std::str::from_utf8(&bytes) {
            Ok(text) => Ok(text.to_owned()),
            // `error_len() == None`: the input ENDS mid-character — hold
            // the tail for the next chunk.
            Err(error) if error.error_len().is_none() => {
                let complete = error.valid_up_to();
                let text = std::str::from_utf8(&bytes[..complete])?.to_owned();
                self.carry = bytes.split_off(complete);
                debug_assert!(
                    !self.carry.is_empty() && self.carry.len() < 4,
                    "an incomplete tail is 1 to 3 bytes"
                );
                Ok(text)
            }
            Err(error) => Err(error),
        }
    }
}

/// One routed piece of a content stream: reasoning or reply.
#[derive(Debug, PartialEq, Eq)]
enum ContentSegment {
    Thinking(String),
    Text(String),
}

/// Streaming twin of [`strip_think_tags`]: routes inline `<think>…</think>`
/// in `message.content` to thinking instead of the reply.
///
/// Models served without Ollama's thinking separation (no `think: true`, an
/// older Ollama, a family the request does not flag) put their reasoning in
/// `content`, tags and all. The non-streaming path already stripped it; the
/// streaming path — the one chat uses — passed it through, so the tags and
/// the whole chain of thought landed in the user's reply and in the history
/// replayed to the model on later turns.
///
/// A tag can be split across fragments (`<thi` + `nk>`), so the splitter holds
/// back the longest suffix that could still begin the tag it is waiting for.
/// That hold-back is bounded by the tag's length minus one, and released at
/// the next fragment or at [`Self::finish`].
#[derive(Default)]
struct InlineThinkSplitter {
    inside: bool,
    /// Text not yet routed: at most a partial tag.
    pending: String,
    /// A think block just closed: the reply's leading whitespace after it is
    /// the model's separator, not content (matches `strip_think_tags`' trim).
    trim_next_text: bool,
}

impl InlineThinkSplitter {
    const OPEN: &'static str = "<think>";
    const CLOSE: &'static str = "</think>";

    /// Route `fragment`, appending complete segments to `out`.
    fn push(&mut self, fragment: &str, out: &mut Vec<ContentSegment>) {
        self.pending.push_str(fragment);
        loop {
            let tag = if self.inside { Self::CLOSE } else { Self::OPEN };
            if let Some(at) = self.pending.find(tag) {
                let before: String = self.pending.drain(..at).collect();
                self.route(before, out);
                self.pending.drain(..tag.len());
                self.inside = !self.inside;
                self.trim_next_text = !self.inside;
                continue;
            }
            let held = partial_tag_suffix_len(&self.pending, tag);
            let ready: String = self.pending.drain(..self.pending.len() - held).collect();
            self.route(ready, out);
            break;
        }
        debug_assert!(
            self.pending.len() < Self::CLOSE.len(),
            "only a partial tag may be held back"
        );
    }

    /// Release whatever is held back, in the mode it was held in. An
    /// unclosed think block stays thinking — the reply never sees it.
    fn finish(&mut self, out: &mut Vec<ContentSegment>) {
        let rest = std::mem::take(&mut self.pending);
        self.route(rest, out);
        debug_assert!(self.pending.is_empty(), "finish releases everything");
    }

    fn route(&mut self, text: String, out: &mut Vec<ContentSegment>) {
        if self.inside {
            if !text.is_empty() {
                out.push(ContentSegment::Thinking(text));
            }
            return;
        }
        let text = if self.trim_next_text {
            let trimmed = text.trim_start();
            if trimmed.is_empty() {
                return;
            }
            self.trim_next_text = false;
            trimmed.to_string()
        } else {
            text
        };
        if !text.is_empty() {
            out.push(ContentSegment::Text(text));
        }
    }
}

/// Length of the longest proper suffix of `text` that is a prefix of `tag`.
/// `tag` is ASCII, so every prefix slice is a char boundary.
fn partial_tag_suffix_len(text: &str, tag: &str) -> usize {
    debug_assert!(tag.is_ascii(), "tags are ASCII so prefix slices are valid");
    (1..tag.len())
        .rev()
        .find(|&k| text.ends_with(&tag[..k]))
        .unwrap_or(0)
}

/// Block and stop bookkeeping while translating an Ollama `/api/chat` NDJSON
/// stream into Anthropic stream events.
#[derive(Default)]
struct OllamaStreamState {
    /// Inline `<think>` routing for `message.content` (see the type).
    think_splitter: InlineThinkSplitter,
    thinking_block_started: bool,
    saw_stop_sentinel: bool,
    text_block_started: bool,
    tool_block_count: usize,
    /// Ollama thinking uses block index 0 (same as Anthropic);
    /// text starts at index 1 when thinking is present, or 0 otherwise.
    next_block_index: usize,
    /// Degenerate-repetition watch (see `RepeatWatch`).
    repeat_watch: RepeatWatch,
}

impl OllamaStreamState {
    const fn stop_reason(&self) -> &'static str {
        if self.tool_block_count > 0 { "tool_use" } else { "end_turn" }
    }

    /// Translate one parsed NDJSON object into the items to yield, in order,
    /// and whether the stream is over (a wedged runner, or `done: true`).
    fn on_object(
        &mut self,
        obj: &serde_json::Value,
        bytes_received: usize,
    ) -> (Vec<Result<StreamEvent, LlmError>>, bool) {
        let mut items = Vec::new();

        // Check for done
        let done = obj["done"].as_bool().unwrap_or(false);

        // Stream thinking content (qwen3 and other thinking models)
        // Ollama returns thinking tokens in message.thinking.
        // Emit proper ContentBlockStart/Stop to match Anthropic's block structure.
        // Cut a wedged runner short instead of waiting out its loop.
        if let Some(frag) = obj["message"]["thinking"]
            .as_str()
            .or_else(|| obj["message"]["content"].as_str())
            && let Some(run) = self.repeat_watch.observe(frag) {
                if self.thinking_block_started || self.text_block_started {
                    items.push(Ok(StreamEvent::ContentBlockStop { index: self.next_block_index }));
                }
                items.push(Err(LlmError::WedgedRunner {
                    token: frag.to_string(),
                    run,
                    bytes_received,
                }));
                return (items, true);
            }

        if let Some(thinking) = obj["message"]["thinking"].as_str()
            && !thinking.is_empty() {
                self.emit_thinking(thinking, &mut items);
            }

        // Stream text content.
        // When text starts arriving and thinking was active, close the thinking block first.
        if let Some(content) = obj["message"]["content"].as_str() {
            self.on_content(content, &mut items);
        }

        // Tool calls: Ollama emits these in a message with done=false,
        // followed by a separate done=true message with empty content.
        // Process tool_calls from ANY message, not just the done message.
        if let Some(tool_calls) = obj["message"]["tool_calls"].as_array()
            && !tool_calls.is_empty() {
                self.on_tool_calls(tool_calls, &mut items);
            }

        if done {
            self.flush_content(&mut items);
            // Close any remaining open blocks
            if self.thinking_block_started {
                items.push(Ok(StreamEvent::ContentBlockStop { index: self.next_block_index }));
                self.next_block_index += 1;
            }
            if self.text_block_started {
                items.push(Ok(StreamEvent::ContentBlockStop { index: self.next_block_index }));
                // next_block_index not needed after this
            }

            // Emit usage. Ollama reports BOTH sides only in the
            // final done message: prompt_eval_count (input) and
            // eval_count (output). The prompt side rides a late
            // MessageStart — the stream consumer records it by
            // assignment wherever it appears, and Ollama streams
            // never emit an earlier one. Omitting it zeroed
            // every per-request input count downstream (run
            // benchmarks and the context-usage indicator).
            if let Some(prompt_eval) = obj["prompt_eval_count"].as_u64() {
                items.push(Ok(StreamEvent::MessageStart {
                    id: String::new(),
                    model: obj["model"].as_str().unwrap_or("").to_string(),
                    input_tokens: token_count_u32(prompt_eval),
                    cache_read_tokens: 0,
                    cache_creation_tokens: 0,
                    cache_creation_1h_tokens: 0,
                }));
            }
            // Emit output token count
            if let Some(eval_count) = obj["eval_count"].as_u64() {
                items.push(Ok(StreamEvent::MessageDelta {
                    stop_reason: None,
                    output_tokens: token_count_u32(eval_count),
                }));
            }

            items.push(Ok(StreamEvent::MessageStop { stop_reason: self.stop_reason().to_string() }));
            return (items, true);
        }
        (items, false)
    }

    /// Append the events for a `message.content` fragment, dropping (and
    /// remembering) a Gemma stop sentinel.
    fn on_content(&mut self, content: &str, items: &mut Vec<Result<StreamEvent, LlmError>>) {
        // Gemma's reserved sentinels (`<unused0>`..`<unused98>`)
        // are stop markers the chat template is supposed to
        // consume. Ollama passes them through as ordinary
        // content and then simply stops the stream, with no
        // done=true — so the terminator we need arrives
        // disguised as text. Note it, drop it from the visible
        // output, and let the end-of-stream handler close the
        // turn cleanly instead of reporting an aborted
        // generation. Observed live 2026-07-28: 57 such 502s in
        // one gemma4:12b run, each discarding a complete reply.
        let content = if LlmClient::is_gemma_stop_sentinel(content) {
            self.saw_stop_sentinel = true;
            tracing::info!("ollama: model stop sentinel seen (exact)");
            ""
        } else if let Some(prefix) = LlmClient::strip_trailing_stop_sentinel(content) {
            // The sentinel can also arrive glued to the end of
            // real text ("...all tests pass.<unused50>"), where
            // an exact-match test never fires. Keep the words,
            // drop the marker, remember that the turn ended.
            self.saw_stop_sentinel = true;
            tracing::info!("ollama: model stop sentinel seen (trailing)");
            prefix
        } else {
            content
        };
        let mut segments = Vec::new();
        self.think_splitter.push(content, &mut segments);
        self.emit_segments(segments, items);
    }

    /// Release the splitter's hold-back at the end of the stream.
    fn flush_content(&mut self, items: &mut Vec<Result<StreamEvent, LlmError>>) {
        let mut segments = Vec::new();
        self.think_splitter.finish(&mut segments);
        self.emit_segments(segments, items);
    }

    fn emit_segments(
        &mut self,
        segments: Vec<ContentSegment>,
        items: &mut Vec<Result<StreamEvent, LlmError>>,
    ) {
        for segment in segments {
            match segment {
                ContentSegment::Thinking(thinking) => self.emit_thinking(&thinking, items),
                ContentSegment::Text(text) => self.emit_text(&text, items),
            }
        }
    }

    /// A thinking delta, opening a thinking block (and closing an open text
    /// block first — reasoning can resume after reply text) as needed.
    fn emit_thinking(&mut self, thinking: &str, items: &mut Vec<Result<StreamEvent, LlmError>>) {
        debug_assert!(!thinking.is_empty(), "empty deltas are never emitted");
        if self.text_block_started {
            items.push(Ok(StreamEvent::ContentBlockStop {
                index: self.next_block_index,
            }));
            self.text_block_started = false;
            self.next_block_index += 1;
        }
        if !self.thinking_block_started {
            self.thinking_block_started = true;
            items.push(Ok(StreamEvent::ContentBlockStart {
                index: self.next_block_index,
                content_type: "thinking".to_string(),
                tool_id: None,
                tool_name: None,
            }));
        }
        debug_assert!(!self.text_block_started, "one open block at a time");
        items.push(Ok(StreamEvent::ThinkingDelta {
            index: self.next_block_index,
            thinking: thinking.to_string(),
        }));
    }

    /// A text delta, opening a text block (and closing an open thinking block
    /// first) as needed.
    fn emit_text(&mut self, text: &str, items: &mut Vec<Result<StreamEvent, LlmError>>) {
        debug_assert!(!text.is_empty(), "empty deltas are never emitted");
        if self.thinking_block_started {
            items.push(Ok(StreamEvent::ContentBlockStop {
                index: self.next_block_index,
            }));
            self.thinking_block_started = false;
            self.next_block_index += 1;
        }
        if !self.text_block_started {
            self.text_block_started = true;
            items.push(Ok(StreamEvent::ContentBlockStart {
                index: self.next_block_index,
                content_type: "text".to_string(),
                tool_id: None,
                tool_name: None,
            }));
        }
        debug_assert!(!self.thinking_block_started, "one open block at a time");
        items.push(Ok(StreamEvent::TextDelta {
            index: self.next_block_index,
            text: text.to_string(),
        }));
    }

    /// Append complete tool-use blocks for a message's `tool_calls`, closing any
    /// open thinking or text block first.
    fn on_tool_calls(&mut self, tool_calls: &[serde_json::Value], items: &mut Vec<Result<StreamEvent, LlmError>>) {
        // Close any open thinking/text blocks before tool calls
        if self.thinking_block_started {
            items.push(Ok(StreamEvent::ContentBlockStop { index: self.next_block_index }));
            self.thinking_block_started = false;
            self.next_block_index += 1;
        }
        if self.text_block_started {
            items.push(Ok(StreamEvent::ContentBlockStop { index: self.next_block_index }));
            self.text_block_started = false;
            self.next_block_index += 1;
        }

        for tc in tool_calls {
            self.tool_block_count += 1;
            let block_idx = self.next_block_index;
            self.next_block_index += 1;
            let name = tc["function"]["name"].as_str().unwrap_or("").to_string();
            let args = tc["function"]["arguments"].to_string();
            let id = tc["id"].as_str().map_or_else(synthesized_tool_use_id, String::from);
            items.push(Ok(StreamEvent::ContentBlockStart {
                index: block_idx,
                content_type: "tool_use".to_string(),
                tool_id: Some(id),
                tool_name: Some(name),
            }));
            items.push(Ok(StreamEvent::ToolUseDelta {
                index: block_idx,
                partial_json: args,
            }));
            items.push(Ok(StreamEvent::ContentBlockStop { index: block_idx }));
        }
    }

    /// The items that end a stream whose body closed without a `done: true`
    /// line: an unterminated final `done` object or a stop sentinel still close
    /// the turn cleanly; anything else is reported as an aborted generation.
    fn finish(
        &mut self,
        buffer: &str,
        last_line: Option<&str>,
        bytes_received: usize,
        lines_parsed: usize,
    ) -> Vec<Result<StreamEvent, LlmError>> {
        let mut items = Vec::new();

        // A final NDJSON object that arrived without its trailing newline
        // would otherwise sit in `buffer` unparsed and be reported as an
        // aborted generation. Cheap to honour, and it can only ever turn
        // a false 502 into the success it actually was. It is translated like
        // any other line: this used to only close the blocks, so whatever the
        // object itself carried — its last text, a tool call, its
        // `done_reason` — was dropped from a reply reported as complete.
        if let Ok(obj) = serde_json::from_str::<serde_json::Value>(buffer.trim())
            && obj["done"].as_bool().unwrap_or(false)
        {
            let (items, finished) = self.on_object(&obj, bytes_received);
            debug_assert!(finished, "a done object ends the stream");
            return items;
        }

        // A sentinel-terminated stream is a COMPLETE turn.
        //
        // The check below exists for genuinely aborted generations, and it
        // must stay — a degraded runner streams degenerate tokens and drops
        // the connection, and delivering that as a successful reply is how
        // garbage reached the GUI. But gemma ends its turn by EMITTING a
        // reserved `<unusedNN>` token and then closing, which is a normal
        // stop wearing an abort's clothing. The content before it is a
        // finished reply; discarding it cost one run 57 complete responses
        // and ultimately the whole benchmark.
        if self.saw_stop_sentinel {
            self.flush_content(&mut items);
            if self.thinking_block_started {
                items.push(Ok(StreamEvent::ContentBlockStop { index: self.next_block_index }));
            }
            if self.text_block_started {
                items.push(Ok(StreamEvent::ContentBlockStop { index: self.next_block_index }));
            }
            tracing::info!(
                bytes_received,
                lines_parsed,
                "ollama stream closed on a model stop sentinel rather than done=true"
            );
            items.push(Ok(StreamEvent::MessageStop { stop_reason: self.stop_reason().to_string() }));
            return items;
        }

        // The stream ended with no done=true terminator: an ABORTED
        // generation. A degraded Ollama runner streams degenerate tokens
        // and drops the stream with no final message (observed live) —
        // closing "gracefully" here delivered that garbage to the GUI as
        // a successful reply. Surface a retryable server error instead,
        // mirroring the non-streaming done=false check.
        if self.thinking_block_started {
            items.push(Ok(StreamEvent::ContentBlockStop { index: self.next_block_index }));
        }
        if self.text_block_started {
            items.push(Ok(StreamEvent::ContentBlockStop { index: self.next_block_index }));
        }
        // Say what was left unparsed. A bare byte/line count cannot
        // distinguish "the provider hung up mid-token" from "we dropped a
        // terminator", and 68 of these in one night taught us nothing
        // about which. A bounded tail of the residual buffer makes the
        // next occurrence self-diagnosing.
        let tail_note = last_line.map_or_else(
            || " No NDJSON line was ever parsed.".to_string(),
            |line| {
                let shown: String = line.chars().take(200).collect();
                format!(" Last line before the cut: {shown}")
            },
        );
        let residue = buffer.trim();
        let residue_note = if residue.is_empty() {
            format!("nothing was left unparsed (the stream simply stopped).{tail_note}")
        } else {
            let tail: String = residue.chars().rev().take(160).collect::<Vec<_>>()
                .into_iter().rev().collect();
            format!("{} unparsed bytes remained, ending: {tail}", residue.len())
        };
        items.push(Err(LlmError::from_api_response(
            502,
            format!(
                "Ollama stream ended without completion (no done=true) after \
                 {bytes_received} bytes / {lines_parsed} NDJSON lines — {residue_note}"
            ),
        )));
        items
    }
}

// ============================================================================
// OpenAI-compatible streaming: request body and SSE translation
// ============================================================================

/// A message in a streaming OpenAI-compatible chat request.
#[derive(Serialize)]
struct OpenAiStreamMessage {
    role: String,
    content: String,
}

/// A tool in a streaming OpenAI-compatible chat request.
#[derive(Serialize)]
struct OpenAiStreamTool {
    #[serde(rename = "type")]
    tool_type: String,
    function: OpenAiStreamFunction,
}

/// The function half of an [`OpenAiStreamTool`].
#[derive(Serialize)]
struct OpenAiStreamFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

/// The body [`LlmClient::stream_openai`] posts to `/v1/chat/completions`.
#[derive(Serialize)]
struct OpenAiStreamRequest {
    model: String,
    messages: Vec<OpenAiStreamMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAiStreamTool>>,
}

/// Build the streaming chat body for `request`. Tool values that do not decode
/// as a [`ToolDefinition`] are dropped.
fn openai_stream_body(request: &CompletionRequest) -> OpenAiStreamRequest {
    // Convert messages to OpenAI format
    let messages: Vec<OpenAiStreamMessage> = request
        .messages
        .iter()
        .map(|m| OpenAiStreamMessage {
            role: match m.role {
                Role::System => "system".to_string(),
                Role::User => "user".to_string(),
                Role::Assistant => "assistant".to_string(),
            },
            content: m.content.clone(),
        })
        .collect();

    // Convert tools to OpenAI format
    let tools: Option<Vec<OpenAiStreamTool>> = if request.tools.is_empty() {
        None
    } else {
        Some(request.tools.iter().filter_map(|v| {
            // Try to parse as ToolDefinition
            serde_json::from_value::<ToolDefinition>(v.clone()).ok().map(|tool_def| OpenAiStreamTool {
                tool_type: "function".to_string(),
                function: OpenAiStreamFunction {
                    name: tool_def.name,
                    description: tool_def.description,
                    parameters: tool_def.input_schema,
                },
            })
        }).collect())
    };

    OpenAiStreamRequest {
        model: request.model.clone(),
        messages,
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        stream: true,
        tools,
    }
}

/// One `data:` chunk of an OpenAI-compatible chat-completions stream.
#[derive(Deserialize, Debug)]
struct OpenAiStreamChunk<D> {
    choices: Vec<OpenAiChunkChoice<D>>,
}

#[derive(Deserialize, Debug)]
struct OpenAiChunkChoice<D> {
    delta: Option<D>,
    finish_reason: Option<String>,
}

/// The delta of a stream chunk choice.
#[derive(Deserialize, Debug)]
struct OpenAiChunkDelta {
    content: Option<String>,
    tool_calls: Option<Vec<OpenAiToolCallDelta>>,
}

/// [`OpenAiChunkDelta`] as [`LlmClient::stream_openai`] decodes it: a present
/// `role` must be a string, or the whole chunk is skipped.
#[derive(Deserialize, Debug)]
struct OpenAiRoleCheckedDelta {
    content: Option<String>,
    #[serde(rename = "role")]
    _role: Option<String>,
    tool_calls: Option<Vec<OpenAiToolCallDelta>>,
}

impl From<OpenAiRoleCheckedDelta> for OpenAiChunkDelta {
    fn from(delta: OpenAiRoleCheckedDelta) -> Self {
        Self {
            content: delta.content,
            tool_calls: delta.tool_calls,
        }
    }
}

#[derive(Deserialize, Debug)]
struct OpenAiToolCallDelta {
    index: usize,
    id: Option<String>,
    function: Option<OpenAiFunctionDelta>,
}

#[derive(Deserialize, Debug)]
struct OpenAiFunctionDelta {
    name: Option<String>,
    arguments: Option<String>,
}

/// Whether a failed `/api/embed` may be retried on the legacy `/api/embeddings`.
///
/// Only when the endpoint itself is missing — an Ollama older than `/api/embed`
/// answers `404` with a plain-text body. Any error Ollama itself reports is a
/// JSON `{"error": …}` (a missing model is a `404` too) and is the answer: the
/// fallback used to discard it and report the legacy call's error instead, so
/// the caller saw the wrong reason — and `embed_one`'s input-overflow heal,
/// which reads that message, never saw the overflow it exists to heal.
fn legacy_embed_may_answer(status: u16, body: &str) -> bool {
    status == 404
        && serde_json::from_str::<serde_json::Value>(body)
            .map_or(true, |value| value.get("error").is_none())
}

/// Whether to ask Ollama for `think: true` — reasoning in a separate
/// `message.thinking` field rather than inline `<think>` tags in `content` —
/// for `model`: the families known to support it (qwen3, deepseek-r1, qwq).
/// Asking a model without thinking support is a 400, so this stays a known list.
fn ollama_separates_thinking(model: &str) -> bool {
    let model = model.to_lowercase();
    ["qwen3", "deepseek-r1", "qwq"]
        .iter()
        .any(|family| model.contains(family))
}

/// Seconds to wait before retrying, as a rate-limited response's headers say.
///
/// `retry-after` wins when present (Anthropic sends it on every 429; plain
/// seconds). Otherwise the reset of each exhausted bucket — `OpenAI`'s
/// `x-ratelimit-remaining-{requests,tokens}` at `0` — and the longest of those,
/// since the request needs both. A bucket not known to be exhausted says
/// nothing about this refusal.
fn retry_after_from_headers(headers: &reqwest::header::HeaderMap) -> Option<u64> {
    let header = |name: &str| headers.get(name).and_then(|v| v.to_str().ok());
    if let Some(wait) = header("retry-after").and_then(parse_reset_secs) {
        return Some(wait);
    }
    ["requests", "tokens"]
        .iter()
        .filter(|bucket| {
            header(&format!("x-ratelimit-remaining-{bucket}")).is_some_and(|v| v.trim() == "0")
        })
        .filter_map(|bucket| {
            header(&format!("x-ratelimit-reset-{bucket}")).and_then(parse_reset_secs)
        })
        .max()
}

/// Whole seconds (rounded up) in a reset value: plain seconds (`"20"`), or a
/// Go-style duration as `OpenAI` sends it (`"6m0s"`, `"1.5s"`, `"250ms"`).
/// These used to be parsed as bare integers, so every duration form was `None`.
fn parse_reset_secs(value: &str) -> Option<u64> {
    let value = value.trim();
    if let Ok(secs) = value.parse::<u64>() {
        return Some(secs);
    }
    let mut millis: u64 = 0;
    let mut rest = value;
    while !rest.is_empty() {
        let number_len = rest.find(|c: char| !(c.is_ascii_digit() || c == '.'))?;
        let unit_len = rest[number_len..]
            .find(|c: char| c.is_ascii_digit())
            .unwrap_or(rest.len() - number_len);
        let (number, unit) = (
            &rest[..number_len],
            &rest[number_len..number_len + unit_len],
        );
        let (whole, fraction) = number.split_once('.').unwrap_or((number, ""));
        let whole: u64 = if whole.is_empty() {
            0
        } else {
            whole.parse().ok()?
        };
        // Milliseconds of the fraction: its first three digits, zero-padded.
        let fraction_millis: u64 = format!("{fraction:0<3}").get(..3)?.parse().ok()?;
        let unit_millis: u64 = match unit {
            "h" => 3_600_000,
            "m" => 60_000,
            "s" => 1_000,
            "ms" => 1,
            "us" | "µs" | "ns" => 0,
            _ => return None,
        };
        millis = millis
            .saturating_add(whole.saturating_mul(unit_millis))
            .saturating_add(fraction_millis.saturating_mul(unit_millis) / 1_000);
        rest = &rest[number_len + unit_len..];
    }
    Some(millis.div_ceil(1_000))
}

/// The error an OpenAI-compatible stream sends in place of a chunk, if `data`
/// is one: `{"error": {"message": …, "code": 429 | "…", "type": …}}`.
///
/// `OpenRouter` sends exactly this mid-stream when an upstream rate-limits or
/// fails after the `200` is already out; `OpenAI` does for server errors. It
/// decodes as no chunk, so it used to be skipped as noise, the stream ran to its
/// end, and the caller got an empty reply ending `end_turn` — a "successful"
/// silence, never retried, never classified as the 429 it was. The status is the
/// numeric `code` when it is an HTTP status, else 429 for a rate-limit `type`,
/// else 500; the message then classifies exactly as a response status would.
fn stream_error_envelope(data: &str) -> Option<LlmError> {
    let value: serde_json::Value = serde_json::from_str(data).ok()?;
    let error = value.get("error")?;
    let message = error
        .get("message")
        .and_then(serde_json::Value::as_str)
        .map_or_else(|| error.to_string(), str::to_string);
    let kind = error
        .get("type")
        .or_else(|| error.get("code"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    let status = error
        .get("code")
        .and_then(serde_json::Value::as_u64)
        .and_then(|code| u16::try_from(code).ok())
        .filter(|code| (400..=599).contains(code))
        .unwrap_or_else(|| {
            if kind.contains("rate_limit") {
                429
            } else {
                500
            }
        });
    Some(LlmError::from_api_response(status, message))
}

/// Block bookkeeping while translating an OpenAI-compatible SSE stream into
/// Anthropic stream events: text is block 0, tool call `i` is block `i + 1`.
#[derive(Default)]
struct OpenAiStreamState {
    text_block_started: bool,
    /// Block index -> (id, name) of every tool call started so far.
    tool_calls_started: std::collections::HashMap<usize, (String, String)>,
}

impl OpenAiStreamState {
    /// `ContentBlockStop` for an open text block, then for every started tool
    /// block in index order.
    fn close_blocks(&self) -> Vec<StreamEvent> {
        let mut events = Vec::new();
        if self.text_block_started {
            events.push(StreamEvent::ContentBlockStop { index: 0 });
        }
        let mut stop_indices: Vec<usize> = self.tool_calls_started.keys().copied().collect();
        stop_indices.sort_unstable();
        for idx in stop_indices {
            events.push(StreamEvent::ContentBlockStop { index: idx });
        }
        events
    }

    /// Translate one SSE line into the events to yield, in order, and whether
    /// the message ended (`[DONE]` or a `finish_reason`). Lines that are not
    /// `data:` lines, and chunks that do not decode as `D`, yield nothing.
    ///
    /// # Errors
    ///
    /// An error envelope in place of a chunk (see [`stream_error_envelope`]),
    /// classified as the same error in a response status would be.
    fn on_line<D>(&mut self, line: &str) -> Result<(Vec<StreamEvent>, bool), LlmError>
    where
        D: serde::de::DeserializeOwned + Into<OpenAiChunkDelta>,
    {
        let Some(data) = line.strip_prefix("data: ") else {
            return Ok((Vec::new(), false));
        };
        if let Some(error) = stream_error_envelope(data) {
            return Err(error);
        }

        if data == "[DONE]" {
            // Close any open blocks
            let mut events = self.close_blocks();
            events.push(StreamEvent::MessageStop { stop_reason: "end_turn".to_string() });
            return Ok((events, true));
        }

        let Ok(chunk) = serde_json::from_str::<OpenAiStreamChunk<D>>(data) else {
            return Ok((Vec::new(), false));
        };
        let mut events = Vec::new();
        for choice in chunk.choices {
            if let Some(delta) = choice.delta {
                self.on_delta(delta.into(), &mut events);
            }

            // Check for stop
            if let Some(reason) = choice.finish_reason {
                // Close any open blocks
                events.extend(self.close_blocks());
                let stop_reason = match reason.as_str() {
                    "tool_calls" => "tool_use",
                    "stop" => "end_turn",
                    other => other,
                };
                events.push(StreamEvent::MessageStop { stop_reason: stop_reason.to_string() });
                return Ok((events, true));
            }
        }
        Ok((events, false))
    }

    /// Append the events for one choice delta: text, then tool-call starts and
    /// argument fragments.
    fn on_delta(&mut self, delta: OpenAiChunkDelta, events: &mut Vec<StreamEvent>) {
        // Handle text content
        if let Some(text) = delta.content
            && !text.is_empty() {
                if !self.text_block_started {
                    self.text_block_started = true;
                    events.push(StreamEvent::ContentBlockStart {
                        index: 0,
                        content_type: "text".to_string(),
                        tool_id: None,
                        tool_name: None,
                    });
                }
                events.push(StreamEvent::TextDelta { index: 0, text });
            }

        // Handle tool calls
        if let Some(tool_calls) = delta.tool_calls {
            for tc in tool_calls {
                // Tool index offset by 1 (text is at 0)
                let block_index = tc.index + 1;

                // Check if this is a new tool call
                if let (Some(id), Some(func)) = (&tc.id, &tc.function)
                    && let Some(name) = &func.name {
                        // Close text block if open (tool calls come after text)
                        if self.text_block_started {
                            events.push(StreamEvent::ContentBlockStop { index: 0 });
                            self.text_block_started = false;
                        }

                        // New tool call
                        self.tool_calls_started.insert(block_index, (id.clone(), name.clone()));
                        events.push(StreamEvent::ContentBlockStart {
                            index: block_index,
                            content_type: "tool_use".to_string(),
                            tool_id: Some(id.clone()),
                            tool_name: Some(name.clone()),
                        });
                    }

                // Stream tool arguments
                if let Some(func) = &tc.function
                    && let Some(args) = &func.arguments
                        && !args.is_empty() {
                            events.push(StreamEvent::ToolUseDelta {
                                index: block_index,
                                partial_json: args.clone(),
                            });
                        }
            }
        }
    }
}

// ============================================================================
// Anthropic ↔ OpenAI/Ollama Conversion Layer
// ============================================================================

/// How to encode `tool_calls[].function.arguments` on the wire.
///
/// OpenAI-compatible endpoints (/v1/chat/completions) expect a JSON-encoded
/// STRING. Ollama's native /api/chat expects a JSON OBJECT and rejects the
/// string form at request-decode time with HTTP 400
/// `Value looks like object, but can't find closing '}' symbol` — even when
/// the string contains valid JSON (verified live against Ollama 0.31.2).
#[derive(Clone, Copy, PartialEq, Eq)]
enum ToolArgsWire {
    /// `"arguments": "{\"path\": ...}"` — `OpenAI` convention
    JsonString,
    /// `"arguments": {"path": ...}` — Ollama native convention
    JsonObject,
}

/// Convert an `AnthropicRequest` into OpenAI-compatible messages and tools JSON.
/// Returns (`messages_json_array`, `tools_json_array_or_none`).
fn anthropic_to_openai_request(request: &AnthropicRequest) -> (serde_json::Value, Option<serde_json::Value>) {
    anthropic_to_wire_request(request, ToolArgsWire::JsonString)
}

/// Convert an `AnthropicRequest` for Ollama's NATIVE /api/chat endpoint.
/// Identical to the `OpenAI` conversion except tool-call arguments stay a JSON
/// object — the string form 400s every request that echoes tool history.
fn anthropic_to_ollama_request(request: &AnthropicRequest) -> (serde_json::Value, Option<serde_json::Value>) {
    anthropic_to_wire_request(request, ToolArgsWire::JsonObject)
}

/// A tool result's text for a wire with no `is_error` flag (OpenAI-shaped,
/// Ollama): the flag travels as an `Error: ` prefix — unless the content
/// already announces itself as one. The agent loop writes failed results as
/// `Error: <message>`, so the unconditional prefix made every failure the
/// model read say `Error: Error: …`.
fn wire_tool_result_content(content: &str, is_error: bool) -> String {
    let announced = content
        .get(..ERROR_PREFIX.len())
        .is_some_and(|head| head.eq_ignore_ascii_case(ERROR_PREFIX));
    let text = if is_error && !announced {
        format!("{ERROR_PREFIX}{content}")
    } else {
        content.to_string()
    };
    debug_assert!(
        !is_error
            || text
                .get(..ERROR_PREFIX.len())
                .is_some_and(|h| h.eq_ignore_ascii_case(ERROR_PREFIX)),
        "an error result always reads as one"
    );
    text
}

/// How an error tool result announces itself on a flagless wire.
const ERROR_PREFIX: &str = "Error: ";

/// Append the wire messages for an Anthropic user message: its text blocks as
/// one user message, then each tool result as its own `tool` message.
fn push_wire_user_messages(
    messages: &mut Vec<serde_json::Value>,
    msg: &AnthropicMessage,
    wire: ToolArgsWire,
) {
    // User messages: may contain text, images and/or tool_result blocks
    let mut text_parts = Vec::new();
    let mut images: Vec<&ImageSource> = Vec::new();
    let mut tool_results = Vec::new();

    for block in &msg.content {
        match block {
            ContentBlock::Text { text } => text_parts.push(text.clone()),
            // Images used to fall into the catch-all below and vanish: an
            // image the user attached never reached an Ollama or
            // OpenAI-compatible model, only Anthropic's native wire.
            ContentBlock::Image { source } => images.push(source),
            ContentBlock::ToolResult { tool_use_id, content, is_error } => {
                tool_results.push(serde_json::json!({
                    "role": "tool",
                    "tool_call_id": tool_use_id,
                    "content": wire_tool_result_content(content, is_error.unwrap_or(false)),
                }));
            }
            _ => {}
        }
    }

    // Emit text (and any images) as a user message
    if !text_parts.is_empty() || !images.is_empty() {
        messages.push(wire_user_message(&text_parts.join("\n"), &images, wire));
    }
    // Emit tool results as separate "tool" role messages
    messages.extend(tool_results);
}

/// A user message with images in the wire's own shape.
///
/// - Ollama: `images` is a list of bare base64 strings beside `content`. It
///   takes no URLs, so a URL image is named in the text instead of sent.
/// - OpenAI-compatible: `content` becomes a list of parts, text first, each
///   image an `image_url` (a `data:` URL for inline base64).
fn wire_user_message(text: &str, images: &[&ImageSource], wire: ToolArgsWire) -> serde_json::Value {
    if images.is_empty() {
        return serde_json::json!({ "role": "user", "content": text });
    }
    match wire {
        ToolArgsWire::JsonObject => {
            let mut content = text.to_string();
            let mut inline: Vec<&str> = Vec::new();
            for image in images {
                match image {
                    ImageSource::Base64 { data, .. } => inline.push(data),
                    ImageSource::Url { url } => {
                        use std::fmt::Write as _;
                        let _ = write!(
                            content,
                            "\n[image at {url} — not sent: this model takes inline images only]"
                        );
                    }
                }
            }
            debug_assert!(inline.len() <= images.len());
            serde_json::json!({ "role": "user", "content": content, "images": inline })
        }
        ToolArgsWire::JsonString => {
            let mut parts = vec![serde_json::json!({ "type": "text", "text": text })];
            parts.extend(images.iter().map(|image| {
                let url = match image {
                    ImageSource::Base64 { media_type, data } => {
                        format!("data:{media_type};base64,{data}")
                    }
                    ImageSource::Url { url } => url.clone(),
                };
                serde_json::json!({ "type": "image_url", "image_url": { "url": url } })
            }));
            debug_assert_eq!(
                parts.len(),
                images.len() + 1,
                "text first, then one part per image"
            );
            serde_json::json!({ "role": "user", "content": parts })
        }
    }
}

/// The wire message for an Anthropic assistant message: its text blocks joined
/// as `content` (always present, `null` when empty) and its tool uses as
/// `tool_calls`, with arguments encoded per `args_wire`.
fn wire_assistant_message(msg: &AnthropicMessage, args_wire: ToolArgsWire) -> serde_json::Value {
    // Assistant messages: may contain text and/or tool_use blocks
    let mut text_parts = Vec::new();
    let mut tool_calls = Vec::new();

    for block in &msg.content {
        match block {
            ContentBlock::Text { text } => text_parts.push(text.clone()),
            ContentBlock::ToolUse { id, name, input } => {
                let arguments = match args_wire {
                    ToolArgsWire::JsonString => {
                        serde_json::Value::String(input.to_string())
                    }
                    // Ollama native: pass the object through. A
                    // non-object input would 400 the whole request,
                    // so degrade it to {} rather than lose the turn.
                    ToolArgsWire::JsonObject if input.is_object() => input.clone(),
                    ToolArgsWire::JsonObject => {
                        warn!(
                            tool = %name,
                            "tool_use input is not a JSON object; sending {{}} to Ollama"
                        );
                        serde_json::json!({})
                    }
                };
                tool_calls.push(serde_json::json!({
                    "id": id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments,
                    }
                }));
            }
            _ => {}
        }
    }

    let mut assistant_msg = serde_json::json!({ "role": "assistant" });
    // Always include `content` — some OpenAI-compatible providers
    // (e.g. Arcee AI via OpenRouter) reject messages where it's missing,
    // even when `tool_calls` is present.
    if text_parts.is_empty() {
        assistant_msg["content"] = serde_json::Value::Null;
    } else {
        assistant_msg["content"] = serde_json::json!(text_parts.join("\n"));
    }
    if !tool_calls.is_empty() {
        assistant_msg["tool_calls"] = serde_json::json!(tool_calls);
    }
    assistant_msg
}

fn anthropic_to_wire_request(
    request: &AnthropicRequest,
    args_wire: ToolArgsWire,
) -> (serde_json::Value, Option<serde_json::Value>) {
    let mut messages = Vec::new();

    // System prompt → system message
    if let Some(ref system) = request.system
        && !system.is_empty() {
            messages.push(serde_json::json!({
                "role": "system",
                "content": system,
            }));
        }

    // Convert Anthropic messages to OpenAI messages
    for msg in &request.messages {
        let role = msg.role.as_str();
        match role {
            "user" => push_wire_user_messages(&mut messages, msg, args_wire),
            "assistant" => messages.push(wire_assistant_message(msg, args_wire)),
            _ => {}
        }
    }

    // Convert tools to OpenAI format
    let tools_json = request.tools.as_ref().and_then(|tools| {
        if tools.is_empty() {
            None
        } else {
            let converted: Vec<serde_json::Value> = tools.iter().map(|t| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    }
                })
            }).collect();
            Some(serde_json::json!(converted))
        }
    });

    (serde_json::json!(messages), tools_json)
}

/// Convert an `OpenAI` chat completion response JSON to `AnthropicResponse`
fn openai_response_to_anthropic(model: &str, response: &serde_json::Value) -> Result<AnthropicResponse, LlmError> {
    let choice = response["choices"]
        .as_array()
        .and_then(|c| c.first())
        .ok_or_else(|| LlmError::Api { status: 500, message: "No choices in response".to_string() })?;

    let message = &choice["message"];
    let mut content = Vec::new();

    // Text content
    if let Some(text) = message["content"].as_str()
        && !text.is_empty() {
            content.push(ContentBlock::Text { text: text.to_string() });
        }

    // Tool calls
    if let Some(tool_calls) = message["tool_calls"].as_array() {
        for tc in tool_calls {
            let id = tc["id"].as_str().unwrap_or("").to_string();
            let name = tc["function"]["name"].as_str().unwrap_or("").to_string();
            let args_str = tc["function"]["arguments"].as_str().unwrap_or("{}");
            let input = heal::heal_tool_args(args_str);
            content.push(ContentBlock::ToolUse { id, name, input });
        }
    }

    // Map finish_reason
    let finish_reason = choice["finish_reason"].as_str().unwrap_or("end_turn");
    let stop_reason = match finish_reason {
        "tool_calls" => "tool_use",
        "stop" => "end_turn",
        "length" => "max_tokens",
        other => other,
    };

    let input_tokens = token_count_u32(response["usage"]["prompt_tokens"].as_u64().unwrap_or(0));
    let output_tokens = token_count_u32(response["usage"]["completion_tokens"].as_u64().unwrap_or(0));
    // OpenAI returns cached tokens in usage.prompt_tokens_details.cached_tokens
    let cache_read_input_tokens = token_count_u32(response["usage"]["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap_or(0));

    Ok(AnthropicResponse {
        id: response["id"].as_str().unwrap_or("").to_string(),
        response_type: "message".to_string(),
        role: "assistant".to_string(),
        content,
        model: model.to_string(),
        stop_reason: Some(stop_reason.to_string()),
        usage: Usage {
            input_tokens,
            output_tokens,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens,
            cache_creation: None,
        },
    })
}

/// Convert an Ollama /api/chat response JSON to `AnthropicResponse`
/// Extract content inside `<think>...</think>` tags.
fn extract_think_content(text: &str) -> Option<String> {
    let start = text.find("<think>")?;
    let end = text.find("</think>")?;
    if end > start + 7 {
        Some(text[start + 7..end].trim().to_string())
    } else {
        None
    }
}

/// Strip all `<think>...</think>` blocks from text, returning the remaining content.
fn strip_think_tags(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut remaining = text;
    while let Some(start) = remaining.find("<think>") {
        result.push_str(&remaining[..start]);
        if let Some(end) = remaining[start..].find("</think>") {
            remaining = &remaining[start + end + 8..];
        } else {
            // Unclosed <think> tag — discard the rest (model is still thinking)
            break;
        }
    }
    result.push_str(remaining);
    result.trim().to_string()
}

fn ollama_response_to_anthropic(model: &str, response: &serde_json::Value) -> Result<AnthropicResponse, LlmError> {
    // A non-streaming response with done=false is an ABORTED generation, not
    // a completion (observed live: a degraded Ollama runner emits ~31 tokens
    // of degenerate output and gives up, with no eval stats and no
    // done_reason). Parsing it as an empty success poisons every layer above;
    // surface it as a retryable server error instead.
    if response.get("done").and_then(serde_json::Value::as_bool) == Some(false) {
        return Err(LlmError::from_api_response(
            502,
            "Ollama aborted generation mid-response (done=false)".to_string(),
        ));
    }
    let message = &response["message"];
    let mut content = Vec::new();

    // Thinking content (qwen3 and other thinking models — Ollama separates this
    // into a `thinking` field when `think: true` is set in the request)
    if let Some(thinking) = message["thinking"].as_str()
        && !thinking.is_empty() {
            content.push(ContentBlock::Thinking { thinking: thinking.to_string(), signature: None });
        }

    // Text content — strip embedded <think>...</think> tags (fallback for when
    // Ollama doesn't separate thinking, e.g. older versions or models that embed tags)
    if let Some(raw_text) = message["content"].as_str() {
        // Extract thinking from tags if we didn't get it from the dedicated field
        if !content.iter().any(|b| matches!(b, ContentBlock::Thinking { .. }))
            && let Some(thinking) = extract_think_content(raw_text)
                && !thinking.is_empty() {
                    content.push(ContentBlock::Thinking { thinking, signature: None });
                }
        let text = strip_think_tags(raw_text);
        if !text.is_empty() {
            content.push(ContentBlock::Text { text });
        }
    }

    // Tool calls (Ollama v0.5+ format)
    if let Some(tool_calls) = message["tool_calls"].as_array() {
        for tc in tool_calls {
            let name = tc["function"]["name"].as_str().unwrap_or("").to_string();
            let input = tc["function"]["arguments"].clone();
            // Ollama doesn't provide tool call IDs, generate one
            let id = synthesized_tool_use_id();
            content.push(ContentBlock::ToolUse { id, name, input });
        }
    }

    // Determine stop reason
    let done_reason = response["done_reason"].as_str().unwrap_or("stop");
    let stop_reason = if content.iter().any(|b| matches!(b, ContentBlock::ToolUse { .. })) {
        "tool_use"
    } else {
        match done_reason {
            "stop" => "end_turn",
            "length" => "max_tokens",
            other => other,
        }
    };

    // Ollama provides eval_count, prompt_eval_count
    let input_tokens = token_count_u32(response["prompt_eval_count"].as_u64().unwrap_or(0));
    let output_tokens = token_count_u32(response["eval_count"].as_u64().unwrap_or(0));

    Ok(AnthropicResponse {
        id: format!("ollama-{}", current_timestamp()),
        response_type: "message".to_string(),
        role: "assistant".to_string(),
        content,
        model: model.to_string(),
        stop_reason: Some(stop_reason.to_string()),
        usage: Usage {
            input_tokens,
            output_tokens,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
            cache_creation: None,
        },
    })
}

/// Extract a complete SSE event from the buffer.
fn extract_sse_event(buffer: &mut String) -> Option<String> {
    // SSE events are separated by double newlines
    buffer.find("\n\n").map(|pos| {
        let event = buffer[..pos].to_string();
        *buffer = buffer[pos + 2..].to_string();
        event
    })
}

/// Parse an SSE event string into a `StreamEvent`
fn parse_sse_event(event: &str) -> Option<StreamEvent> {
    // Extract data field from SSE event (we determine event type from JSON structure)
    let data = event
        .lines()
        .find_map(|line| line.strip_prefix("data: ").map(str::trim))?;

    // Parse the JSON data (skip malformed events)
    let sse = match serde_json::from_str::<AnthropicSSE>(data) {
        Ok(sse) => sse,
        Err(e) => {
            let end = data.floor_char_boundary(data.len().min(120));
            debug!(error = %e, data_prefix = &data[..end], "Skipping unrecognised SSE event");
            return None;
        }
    };
    match sse {
        AnthropicSSE::MessageStart { message } => {
            let usage = message.usage.unwrap_or_default();
            Some(StreamEvent::MessageStart {
                id: message.id,
                model: message.model,
                input_tokens: usage.input_tokens,
                cache_read_tokens: usage.cache_read_input_tokens,
                cache_creation_tokens: usage.cache_creation_input_tokens,
                cache_creation_1h_tokens: hour_share(
                    usage.cache_creation.as_ref(),
                    usage.cache_creation_input_tokens,
                ),
            })
        }
        AnthropicSSE::ContentBlockStart { index, content_block } => {
            Some(StreamEvent::ContentBlockStart {
                index,
                content_type: content_block.block_type,
                tool_id: content_block.id,
                tool_name: content_block.name,
            })
        }
        AnthropicSSE::ContentBlockDelta { index, delta } => match delta {
            DeltaData::TextDelta { text } => Some(StreamEvent::TextDelta { index, text }),
            DeltaData::ThinkingDelta { thinking } => Some(StreamEvent::ThinkingDelta { index, thinking }),
            DeltaData::InputJsonDelta { partial_json } => {
                Some(StreamEvent::ToolUseDelta { index, partial_json })
            }
            DeltaData::SignatureDelta { signature } => {
                Some(StreamEvent::SignatureDelta { index, signature })
            }
            DeltaData::Unknown => {
                debug!(index, "Ignoring unknown content_block_delta type");
                None
            }
        },
        AnthropicSSE::ContentBlockStop { index } => Some(StreamEvent::ContentBlockStop { index }),
        AnthropicSSE::MessageDelta { delta, usage } => Some(StreamEvent::MessageDelta {
            stop_reason: delta.stop_reason,
            output_tokens: usage.map_or(0, |u| u.output_tokens),
        }),
        AnthropicSSE::MessageStop => Some(StreamEvent::MessageStop {
            stop_reason: "end_turn".to_string(),
        }),
        AnthropicSSE::Ping => Some(StreamEvent::Ping),
        // Typed, not flattened to a message: an `overloaded_error` mid-stream
        // is exactly the failure a fallback model exists for, and as a plain
        // `Error { message }` nothing downstream could tell it from a bad request.
        AnthropicSSE::Error { error } => {
            let status = anthropic_error_status(&error.error_type);
            Some(stream_error_event(LlmError::Api {
                status,
                message: if error.error_type.is_empty() {
                    error.message
                } else {
                    format!("{}: {}", error.error_type, error.message)
                },
            }))
        }
    }
}

#[cfg(test)]
mod tests {

    /// Both Ollama paths ask a thinking model for separated reasoning; the
    /// streaming one — the agent's own — used to leave it inline.
    #[test]
    fn the_stream_body_asks_a_thinking_model_to_separate_its_reasoning() {
        let request = |model: &str| AnthropicRequest {
            context_limit: None,
            model: model.to_string(),
            messages: vec![],
            max_tokens: 512,
            temperature: None,
            system: None,
            tools: None,
            stream: None,
            thinking: None,
            cache_control: None,
        };
        let body = LlmClient::ollama_stream_body("http://127.0.0.1:9", &request("qwen3.5:9b"));
        assert_eq!(body["think"], serde_json::json!(true));
        let plain = LlmClient::ollama_stream_body("http://127.0.0.1:9", &request("gemma3:12b"));
        assert!(
            plain.get("think").is_none(),
            "a model without thinking is not asked"
        );
    }

    /// Reset values in every form providers send, rounded up to whole seconds.
    #[test]
    fn reset_values_parse_in_every_form() {
        assert_eq!(parse_reset_secs("20"), Some(20));
        assert_eq!(parse_reset_secs("6m0s"), Some(360));
        assert_eq!(parse_reset_secs("1.5s"), Some(2));
        assert_eq!(parse_reset_secs("250ms"), Some(1));
        assert_eq!(parse_reset_secs("1h2m3s"), Some(3723));
        assert_eq!(parse_reset_secs("0s"), Some(0));
        assert_eq!(parse_reset_secs("soon"), None);
        assert_eq!(parse_reset_secs("5x"), None);
    }

    /// `retry-after` wins; otherwise only an exhausted bucket's reset counts.
    #[test]
    fn a_rate_limit_waits_as_the_headers_say() {
        use reqwest::header::{HeaderMap, HeaderValue};
        let mut openai = HeaderMap::new();
        openai.insert(
            "x-ratelimit-remaining-requests",
            HeaderValue::from_static("12"),
        );
        openai.insert("x-ratelimit-reset-requests", HeaderValue::from_static("1s"));
        openai.insert(
            "x-ratelimit-remaining-tokens",
            HeaderValue::from_static("0"),
        );
        openai.insert("x-ratelimit-reset-tokens", HeaderValue::from_static("6m0s"));
        assert_eq!(retry_after_from_headers(&openai), Some(360));

        openai.insert("retry-after", HeaderValue::from_static("7"));
        assert_eq!(
            retry_after_from_headers(&openai),
            Some(7),
            "retry-after wins"
        );

        let mut unknown = HeaderMap::new();
        unknown.insert("x-ratelimit-reset-tokens", HeaderValue::from_static("6m0s"));
        assert_eq!(
            retry_after_from_headers(&unknown),
            None,
            "not known to be exhausted"
        );
    }

    /// A non-streaming completion is held to the agent path's rules: an
    /// abort is an error, and a think block is not part of the reply.
    #[test]
    fn a_plain_ollama_completion_drops_think_and_refuses_an_abort() {
        let thinking = serde_json::json!({
            "message": { "role": "assistant", "content": "<think>plan it</think>\nThe summary." },
            "done": true
        });
        assert_eq!(
            ollama_completion_text(&thinking).expect("text"),
            "The summary."
        );
        let plain = serde_json::json!({ "message": { "content": "  as is  " }, "done": true });
        assert_eq!(ollama_completion_text(&plain).expect("text"), "  as is  ");
        let aborted = serde_json::json!({ "message": { "content": "half a sen" }, "done": false });
        assert!(matches!(
            ollama_completion_text(&aborted),
            Err(LlmError::Api { status: 502, .. })
        ));
        assert!(ollama_completion_text(&serde_json::json!({ "done": true })).is_err());
    }

/// Only a missing endpoint falls back to the legacy embeddings API; an
    /// error Ollama reports is the answer.
    #[test]
    fn only_a_missing_embed_endpoint_falls_back() {
        assert!(legacy_embed_may_answer(404, "404 page not found"));
        assert!(!legacy_embed_may_answer(
            404,
            r#"{"error":"model \"nomic\" not found"}"#
        ));
        assert!(!legacy_embed_may_answer(
            400,
            r#"{"error":"input length 9000 exceeds maximum context length 8192"}"#
        ));
        assert!(!legacy_embed_may_answer(500, "internal error"));
    }

    /// A mid-stream error envelope is the error it names — classified, so a
    /// 429 is a rate limit — not a chunk to skip on the way to an empty reply.
    #[test]
    fn a_stream_error_envelope_is_an_error_not_an_empty_reply() {
        let mut state = OpenAiStreamState::default();
        let rate = state.on_line::<OpenAiChunkDelta>(
            r#"data: {"error":{"code":429,"message":"Provider returned error: rate-limited upstream"}}"#,
        );
        assert!(matches!(rate, Err(LlmError::RateLimit { .. })), "{rate:?}");

        let typed = stream_error_envelope(
            r#"{"error":{"message":"slow down","type":"rate_limit_exceeded","code":null}}"#,
        );
        assert!(
            matches!(typed, Some(LlmError::RateLimit { .. })),
            "{typed:?}"
        );
        let server = stream_error_envelope(r#"{"error":{"message":"boom","type":"server_error"}}"#);
        assert!(
            matches!(server, Some(LlmError::Api { status: 500, ref message }) if message == "boom"),
            "{server:?}"
        );

        // Ordinary chunks and the terminator are untouched.
        let text = state
            .on_line::<OpenAiChunkDelta>(r#"data: {"choices":[{"delta":{"content":"hi"}}]}"#)
            .expect("a chunk");
        assert!(!text.1);
        assert!(
            text.0
                .iter()
                .any(|e| matches!(e, StreamEvent::TextDelta { text, .. } if text == "hi"))
        );
        assert!(
            state
                .on_line::<OpenAiChunkDelta>("data: [DONE]")
                .expect("done")
                .1
        );
    }

    // -----------------------------------------------------------------
    // GGUF metadata, which is namespaced by architecture
    // -----------------------------------------------------------------

    fn gguf(json: &str) -> serde_json::Map<String, serde_json::Value> {
        serde_json::from_str(json).expect("test fixture is valid JSON")
    }

    /// Verbatim from a live `POST /api/show` for `nomic-embed-text`, which is
    /// the default embedding model. There is no `general.*` key at all — this
    /// is the shape that made every local model resolve to the unknown floor.
    #[test]
    fn an_embedding_model_publishes_its_window_under_its_architecture() {
        let info = gguf(
            r#"{"nomic-bert.context_length": 2048, "nomic-bert.embedding_length": 768}"#,
        );
        assert_eq!(super::gguf_metadata_usize(&info, "context_length"), Some(2048));
        assert_eq!(super::gguf_metadata_usize(&info, "embedding_length"), Some(768));
    }

    #[test]
    fn a_general_key_wins_over_an_architecture_key() {
        let info = gguf(r#"{"general.context_length": 8192, "llama.context_length": 4096}"#);
        assert_eq!(super::gguf_metadata_usize(&info, "context_length"), Some(8192));
    }

    /// Two architectures both claiming a window is ambiguous, and this value
    /// bounds what we are allowed to send. The smaller claim is the only one
    /// that cannot overrun.
    #[test]
    fn ambiguous_architecture_keys_resolve_to_the_bound_that_cannot_overrun() {
        let info = gguf(r#"{"bert.context_length": 512, "llama.context_length": 32768}"#);
        assert_eq!(super::gguf_metadata_usize(&info, "context_length"), Some(512));
    }

    /// A suffix must not match a key that merely contains it, or
    /// `rope.scaling.original_context_length` would answer for the real window.
    #[test]
    fn a_suffix_matches_only_on_a_dotted_boundary() {
        let info = gguf(
            r#"{"llama.rope.scaling.original_context_length": 8192, "llama.context_length": 4096}"#,
        );
        assert_eq!(super::gguf_metadata_usize(&info, "context_length"), Some(4096));
    }

    /// Absent, zero, and non-numeric all mean "we learned nothing". Zero in
    /// particular must not pass through as a bound, or it would clamp every
    /// chunk to nothing.
    #[test]
    fn an_unlearnable_window_is_none_not_zero() {
        assert_eq!(super::gguf_metadata_usize(&gguf("{}"), "context_length"), None);
        let zeroed = gguf(r#"{"bert.context_length": 0}"#);
        assert_eq!(super::gguf_metadata_usize(&zeroed, "context_length"), None);
        let textual = gguf(r#"{"bert.context_length": "2048"}"#);
        assert_eq!(super::gguf_metadata_usize(&textual, "context_length"), None);
    }

    // -----------------------------------------------------------------
    // Context sizing
    // -----------------------------------------------------------------

    const GB: f64 = 1024.0 * 1024.0 * 1024.0;

    /// The two runs we actually have evidence from, on the same 16 GB card
    /// with ~11.8 GB free at rest:
    ///
    ///   qwen3.5:9b  (6.59 GB) ran 4h30m clean at 32768.
    ///   gemma4:12b  (7.56 GB) died at 32768 with
    ///               `CUDA error: an illegal memory access was encountered`,
    ///               three times in three minutes.
    ///
    /// A sizing rule that hands 32768 to both is the rule we already know is
    /// broken, so it is the thing worth pinning: the ~1 GB of extra weights
    /// has to be enough to move gemma down a bucket.
    /// Measured on the bench machine 2026-07-28: 16 GB card, ~5.1 GB held by
    /// the desktop and the GUI's webview, leaving ~10.45 GB actually available
    /// to the runner.
    const FREE_UNLOADED: f64 = 10.45 * GB;
    const EMBEDDER_VRAM: f64 = 0.32 * GB;

    #[test]
    fn sizing_separates_the_model_that_ran_from_the_one_that_crashed() {
        let qwen =
            LlmClient::fit_context_for_budget(FREE_UNLOADED, 6.59 * GB, 0.0, EMBEDDER_VRAM);
        let gemma =
            LlmClient::fit_context_for_budget(FREE_UNLOADED, 7.56 * GB, 0.0, EMBEDDER_VRAM);

        assert_eq!(qwen, 32_768, "qwen3.5:9b demonstrably ran 4h30m at 32k");
        assert!(
            gemma < 32_768,
            "gemma4:12b crashed at 32k three times; sizing must not return it, got {gemma}"
        );
        assert!(
            gemma >= 16_384,
            "gemma still needs a workable window, got {gemma}"
        );
    }

    /// The bug this function shipped with once: sizing changed the moment the
    /// model finished loading, and since Ollama keys a resident instance on its
    /// options, that change evicts and reloads it — which restores the memory
    /// that justified the change.
    #[test]
    fn sizing_does_not_change_when_the_model_finishes_loading() {
        let weights = 7.56 * GB;
        let before =
            LlmClient::fit_context_for_budget(FREE_UNLOADED, weights, 0.0, EMBEDDER_VRAM);
        // Once resident it holds weights + KV, and free VRAM drops by that much.
        let held = 8.06 * GB;
        let after = LlmClient::fit_context_for_budget(
            FREE_UNLOADED - held,
            weights,
            held,
            EMBEDDER_VRAM,
        );
        assert_eq!(
            before, after,
            "loading the model must not resize its own context"
        );
    }

    #[test]
    fn sizes_are_always_buckets() {
        for free in [2.0, 6.0, 9.0, 10.45, 14.0, 40.0] {
            let got =
                LlmClient::fit_context_for_budget(free * GB, 7.56 * GB, 0.0, EMBEDDER_VRAM);
            assert!(
                [4_096, 8_192, 16_384, 32_768].contains(&got),
                "free={free}GB gave a non-bucket size {got}"
            );
        }
    }

    /// Stability across turns comes from the latch, not from bucketing — a
    /// budget that lands near a bucket edge WILL cross it on ordinary drift.
    /// Since a changed `num_ctx` makes Ollama evict and reload the model, the
    /// size has to be decided once and then only ever walked down deliberately.
    #[test]
    fn a_latched_size_survives_re_measurement_and_only_walks_down() {
        let local = LlmClient::ollama(LOCAL_OLLAMA);
        let model = "test-latch-model:1b";
        LlmClient::latch_num_ctx(LOCAL_OLLAMA, model, 32_768);
        assert_eq!(
            LlmClient::latched_num_ctx(LOCAL_OLLAMA, model),
            Some(32_768)
        );

        // Demotion steps 3/4 at a time on the 512 quantum — the full default
        // ladder, documented end to end: geometric (bounded rungs) but never
        // the 50% cliff that skips past viable windows.
        for expected in [24_576, 18_432, 13_824, 10_240, 7_680, 5_632] {
            assert_eq!(local.demote_context(model, None), Some(expected));
        }
        // The last rung clamps AT the default minimum viable window (4608 —
        // the quantized pressure-tier floor), not below it...
        assert_eq!(local.demote_context(model, None), Some(4_608));
        // ...and once there, a repeat fault does NOT demote further: the
        // loud fault path stands rather than a manufactured unusable window.
        assert_eq!(local.demote_context(model, None), None);
        assert_eq!(LlmClient::latched_num_ctx(LOCAL_OLLAMA, model), Some(4_608));
    }

    /// The rung arithmetic itself: 3/4 of the current latch, rounded DOWN to
    /// the 512 quantum, strictly decreasing on every step.
    #[test]
    fn demotion_steps_to_three_quarters_on_the_quantum() {
        let local = LlmClient::ollama(LOCAL_OLLAMA);
        let model = "test-three-quarter-step:1b";
        LlmClient::latch_num_ctx(LOCAL_OLLAMA, model, 8_192);
        // Floor far below: pure stepping is visible. 8192·¾ = 6144 (on the
        // quantum), 6144·¾ = 4608 (on the quantum), 4608·¾ = 3456 → 3072
        // (rounded DOWN to the quantum), 3072·¾ = 2304 → 2048.
        let floor = Some(2_048);
        assert_eq!(local.demote_context(model, floor), Some(6_144));
        assert_eq!(local.demote_context(model, floor), Some(4_608));
        assert_eq!(local.demote_context(model, floor), Some(3_072));
        assert_eq!(local.demote_context(model, floor), Some(2_048));
        assert_eq!(local.demote_context(model, floor), None);
        for size in [6_144u32, 4_608, 3_072, 2_048] {
            assert_eq!(size % NUM_CTX_QUANTUM, 0, "every rung sits on the quantum");
        }
    }

    /// The live 2026-08-03 failure, fixed: ONE fault at a pinned 8192 must
    /// not land below the agent's floor. With the floor supplied, the rung
    /// that would undershoot clamps TO the floor's quantum instead.
    #[test]
    fn demotion_clamps_to_the_floor_quantum_instead_of_skipping_past_it() {
        let local = LlmClient::ollama(LOCAL_OLLAMA);
        let model = "test-floor-clamp:1b";
        LlmClient::latch_num_ctx(LOCAL_OLLAMA, model, 8_192);
        // An agent floor of 4300 tokens (≈ the live pressure-tier floor)
        // quantizes UP to 4608. 8192·¾ = 6144 clears it...
        assert_eq!(local.demote_context(model, Some(4_300)), Some(6_144));
        // ...but 6144·¾ = 4608 lands exactly on it, and the next rung
        // (4608·¾ = 3456) would undershoot — so it clamps and then stops.
        assert_eq!(local.demote_context(model, Some(4_300)), Some(4_608));
        assert_eq!(local.demote_context(model, Some(4_300)), None);
        assert_eq!(LlmClient::latched_num_ctx(LOCAL_OLLAMA, model), Some(4_608));

        // A rung strictly between two quanta clamps UP, never down: from
        // 8192 with a 5000-token floor (quantum 5120), ¾ gives 6144, then
        // 4608 < 5120 → the demotion lands AT 5120.
        let model2 = "test-floor-clamp-up:1b";
        LlmClient::latch_num_ctx(LOCAL_OLLAMA, model2, 8_192);
        assert_eq!(local.demote_context(model2, Some(5_000)), Some(6_144));
        assert_eq!(local.demote_context(model2, Some(5_000)), Some(5_120));
        assert_eq!(local.demote_context(model2, Some(5_000)), None);
    }

    /// No demotion may EVER produce a window below the supplied floor — even
    /// when the latch already sits under it (a floor that grew mid-run, e.g.
    /// a fatter step frame). The latch is left alone and the fault surfaces.
    #[test]
    fn no_demotion_ever_lands_below_the_floor() {
        let local = LlmClient::ollama(LOCAL_OLLAMA);
        let model = "test-never-below-floor:1b";
        LlmClient::latch_num_ctx(LOCAL_OLLAMA, model, 4_096);
        assert_eq!(
            local.demote_context(model, Some(8_192)),
            None,
            "a latch already below the floor must not be walked further"
        );
        assert_eq!(
            LlmClient::latched_num_ctx(LOCAL_OLLAMA, model),
            Some(4_096),
            "refusing to demote must not touch the latch"
        );
        // And with no floor supplied, the conservative default holds the line.
        LlmClient::latch_num_ctx(LOCAL_OLLAMA, model, DEFAULT_MIN_VIABLE_NUM_CTX);
        assert_eq!(local.demote_context(model, None), None);
        assert_eq!(
            LlmClient::latched_num_ctx(LOCAL_OLLAMA, model),
            Some(DEFAULT_MIN_VIABLE_NUM_CTX)
        );
    }

    /// Every demotion decision announces itself — old → new, the floor it
    /// respected, and whether it clamped — and the refusal at the floor is
    /// just as loud, so an operator can read the ladder off the logs.
    #[test]
    fn a_demotion_announces_old_new_floor_and_clamp() {
        let local = LlmClient::ollama(LOCAL_OLLAMA);
        let model = "test-demotion-announces:1b";
        LlmClient::latch_num_ctx(LOCAL_OLLAMA, model, 6_144);
        let out = capture_info_logs(|| {
            assert_eq!(local.demote_context(model, Some(5_000)), Some(5_120));
        });
        assert!(out.contains("demoting context"), "the decision: {out}");
        assert!(out.contains("test-demotion-announces:1b"), "the model: {out}");
        assert!(out.contains("6144"), "old size: {out}");
        assert!(out.contains("5120"), "new size AND the floor quantum: {out}");
        assert!(out.contains("clamped=true"), "clamped-or-not: {out}");

        let refused = capture_info_logs(|| {
            assert_eq!(local.demote_context(model, Some(5_000)), None);
        });
        assert!(
            refused.contains("NOT demoting further"),
            "the refusal at the floor must be announced too: {refused}"
        );
    }

    /// The routed name and the request name are different spellings of one
    /// model; if they key different latch entries, demotion corrects nothing.
    #[test]
    fn the_latch_ignores_the_provider_prefix() {
        let local = LlmClient::ollama(LOCAL_OLLAMA);
        LlmClient::latch_num_ctx(LOCAL_OLLAMA, "ollama/test-prefix-model:1b", 16_384);
        assert_eq!(
            LlmClient::latched_num_ctx(LOCAL_OLLAMA, "test-prefix-model:1b"),
            Some(16_384),
            "demotion keyed on the routed name must reach the request path"
        );
        assert_eq!(
            local.demote_context("ollama/test-prefix-model:1b", None),
            Some(12_288)
        );
        assert_eq!(
            LlmClient::latched_num_ctx(LOCAL_OLLAMA, "test-prefix-model:1b"),
            Some(12_288)
        );
    }

    /// The demotion must be OBSERVABLE, not just latched: a model that was
    /// never sized passes the provider claim through, and after a demotion
    /// the effective window is the latch, not the claim. This getter is what
    /// the agent loop polls to re-derive its budgets mid-run — the 2026-08-02
    /// failure was exactly this value existing but being unreadable.
    #[test]
    fn a_demotion_is_visible_through_the_effective_window() {
        let _guard = env_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        // SAFETY: see `an_explicit_override_beats_the_vram_heuristic`.
        unsafe { std::env::remove_var("NANNA_OLLAMA_NUM_CTX") };

        let model = "test-effective-window-model:9b";
        let local = LlmClient::ollama(LOCAL_OLLAMA);
        // Never sized: the provider claim stands, for local and cloud alike.
        assert_eq!(local.effective_num_ctx(model), None);
        assert_eq!(local.effective_context_window(model, 32_000), 32_000);
        assert_eq!(effective_context_window(model, 32_000), 32_000);

        // Sized at 16384, then demoted: the effective window follows the
        // latch, and a claim SMALLER than the latch is never inflated.
        LlmClient::latch_num_ctx(LOCAL_OLLAMA, model, 16_384);
        assert_eq!(local.effective_context_window(model, 32_000), 16_384);
        assert_eq!(local.demote_context(model, None), Some(12_288));
        assert_eq!(local.effective_num_ctx(model), Some(12_288));
        assert_eq!(local.effective_context_window(model, 32_000), 12_288);
        assert_eq!(effective_context_window(model, 32_000), 12_288);
        assert_eq!(
            local.effective_context_window(model, 4_000),
            4_000,
            "the latch is a ceiling, not a floor"
        );
    }

    /// Model info served to budget derivation must carry the LIVE window:
    /// context clamps to the latch and `max_output` re-bounds to half of it, so
    /// `hard_input_limit_for` / `effective_output_budget` stay consistent with
    /// what the runner will actually honour.
    #[test]
    fn model_info_clamps_to_the_demoted_window() {
        let model = "test-clamped-info-model:9b";
        let claim = ModelInfo {
            id: model.to_string(),
            context_window: 16_384,
            max_output_tokens: 8_192,
            supports_tools: true,
            supports_vision: false,
            embedding_dimension: None,
            cached_at: current_timestamp(),
            provider: "ollama".to_string(),
        };

        // Unlatched: the claim passes through untouched.
        let untouched = clamp_model_info_to_effective_window(model, claim.clone());
        assert_eq!(untouched.context_window, 16_384);
        assert_eq!(untouched.max_output_tokens, 8_192);

        LlmClient::latch_num_ctx(LOCAL_OLLAMA, model, 4_096);
        let through_the_client = LlmClient::ollama(LOCAL_OLLAMA)
            .clamp_model_info_to_effective_window(model, claim.clone());
        let clamped = clamp_model_info_to_effective_window(model, claim);
        assert_eq!(
            (clamped.context_window, clamped.max_output_tokens),
            (
                through_the_client.context_window,
                through_the_client.max_output_tokens
            ),
            "one server: the client's window is the smallest there is"
        );
        assert_eq!(clamped.context_window, 4_096);
        assert_eq!(
            clamped.max_output_tokens, 2_048,
            "output cap mirrors the Ollama half-window derivation"
        );
        // The derived budgets never over-commit the demoted window.
        let reserve = clamped.effective_output_budget(4_096);
        assert!(clamped.hard_input_limit_for(reserve) + reserve <= 4_096);
    }

    /// The regression this guards: the VRAM heuristic used to run FIRST and the
    /// env var was only consulted when measuring failed, so an operator's
    /// deliberate 16384 was silently promoted to a computed 32768. Same model,
    /// same mission, 31/42 became 8/42. (The override is a ceiling — it beats
    /// any LARGER measurement; min semantics are covered separately below.)
    ///
    /// Serialised against the other env-touching tests via `env_lock` because
    /// the environment is process-global.
    #[test]
    fn an_explicit_override_beats_the_vram_heuristic() {
        let _guard = env_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        // SAFETY: edition 2024 requires this; the lock above keeps the two
        // env-touching tests from overlapping, and nothing else reads this key.
        unsafe { std::env::set_var("NANNA_OLLAMA_NUM_CTX", "16384") };

        let mut request = request_with_tool_history(serde_json::json!({}));
        request.model = "test-env-precedence:1b".to_string();
        request.context_limit = None;

        assert_eq!(
            LlmClient::resolve_num_ctx(LOCAL_OLLAMA, &request.model, request.context_limit),
            16_384,
            "an explicit override must win over whatever the card measures"
        );

        unsafe { std::env::remove_var("NANNA_OLLAMA_NUM_CTX") };
    }

    /// ...but the override must not become a silent floor of its own: absent or
    /// unusable values fall through to sizing rather than to a wrong constant.
    #[test]
    fn an_absent_or_junk_override_falls_through_to_sizing() {
        let _guard = env_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        // SAFETY: see above.
        unsafe { std::env::remove_var("NANNA_OLLAMA_NUM_CTX") };
        assert_eq!(LlmClient::env_num_ctx(), None);

        unsafe { std::env::set_var("NANNA_OLLAMA_NUM_CTX", "not-a-number") };
        assert_eq!(LlmClient::env_num_ctx(), None, "garbage must not be honoured");

        // Too small to hold a system prompt plus one tool round.
        unsafe { std::env::set_var("NANNA_OLLAMA_NUM_CTX", "512") };
        assert_eq!(LlmClient::env_num_ctx(), None, "an unusable size is not a setting");

        unsafe { std::env::remove_var("NANNA_OLLAMA_NUM_CTX") };
    }

    /// The pin IS the starting latch, in both directions. The probe is a
    /// snapshot of free bytes and must not override a durable operator
    /// measurement either way: promotion past the pin cost a mission
    /// (31/42→8/42 at 32768), and demotion below it cost another
    /// (2026-08-09: a summarization burst held the card while the first
    /// request sized, min(env, sized) latched 4096 for the daemon's
    /// lifetime, and every later request inherited the poisoned latch on an
    /// idle card). A genuinely oversized pin is walked down by the fault
    /// ladder, not by the snapshot. Fake model names are absent from
    /// Ollama's tags, so the VRAM probe abstains deterministically here.
    #[test]
    fn the_env_pin_is_the_starting_latch_in_both_directions() {
        let _guard = env_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut request = request_with_tool_history(serde_json::json!({}));
        request.context_limit = None;

        // Below nominal: the pin is the starting latch.
        // SAFETY: edition 2024 requires this; env_lock serialises the
        // env-touching tests, and nothing else reads this key.
        unsafe { std::env::set_var("NANNA_OLLAMA_NUM_CTX", "8192") };
        request.model = "test-env-pin-shrinks:1b".to_string();
        assert_eq!(
            LlmClient::resolve_num_ctx(LOCAL_OLLAMA, &request.model, request.context_limit),
            8_192
        );
        assert_eq!(
            LlmClient::latched_num_ctx(LOCAL_OLLAMA, "test-env-pin-shrinks:1b"),
            Some(8_192),
            "the pin must be LATCHED as the start, not re-derived per request"
        );

        // Above nominal: the pin still wins — the snapshot cannot demote a
        // durable measurement; only the GPU fault ladder can.
        unsafe { std::env::set_var("NANNA_OLLAMA_NUM_CTX", "32768") };
        request.model = "test-env-pin-wins-upward:1b".to_string();
        assert_eq!(
            LlmClient::resolve_num_ctx(LOCAL_OLLAMA, &request.model, request.context_limit),
            32_768
        );

        unsafe { std::env::remove_var("NANNA_OLLAMA_NUM_CTX") };
    }

    /// The pin is a ceiling, not a disable: a real GPU fault still walks the
    /// latch below the pinned start, the walked-down value survives later
    /// requests (the latch, not the env, is consulted once it exists), and an
    /// explicit per-request `context_limit` keeps outranking both.
    #[test]
    fn demotion_still_walks_below_the_env_pin() {
        let _guard = env_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        // SAFETY: see above.
        unsafe { std::env::set_var("NANNA_OLLAMA_NUM_CTX", "8192") };

        let mut request = request_with_tool_history(serde_json::json!({}));
        request.model = "test-env-pin-demotes:1b".to_string();
        request.context_limit = None;
        assert_eq!(
            LlmClient::resolve_num_ctx(LOCAL_OLLAMA, &request.model, request.context_limit),
            8_192
        );

        assert_eq!(
            LlmClient::ollama(LOCAL_OLLAMA).demote_context("test-env-pin-demotes:1b", None),
            Some(6_144)
        );
        assert_eq!(
            LlmClient::resolve_num_ctx(LOCAL_OLLAMA, &request.model, request.context_limit),
            6_144,
            "a demotion below the pin must stick — the pin must not re-inflate it"
        );

        request.context_limit = Some(2_048);
        assert_eq!(
            LlmClient::resolve_num_ctx(LOCAL_OLLAMA, &request.model, request.context_limit),
            2_048,
            "an explicit request limit keeps its precedence over pin and latch"
        );

        unsafe { std::env::remove_var("NANNA_OLLAMA_NUM_CTX") };
    }

    /// A pin that changes the starting latch must say so at INFO — model,
    /// pinned value, and source — so an operator reading the logs knows the
    /// window was deliberate, not measured.
    #[test]
    fn the_env_pin_announces_itself_at_info() {
        let _guard = env_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        // SAFETY: see above.
        unsafe { std::env::set_var("NANNA_OLLAMA_NUM_CTX", "8192") };

        let mut request = request_with_tool_history(serde_json::json!({}));
        request.model = "test-env-pin-announces:1b".to_string();
        request.context_limit = None;

        let out = capture_info_logs(|| {
            assert_eq!(
                LlmClient::resolve_num_ctx(LOCAL_OLLAMA, &request.model, request.context_limit),
                8_192
            );
        });
        assert!(
            out.contains("NANNA_OLLAMA_NUM_CTX pins"),
            "the pin must announce itself, got: {out}"
        );
        assert!(
            out.contains("test-env-pin-announces:1b"),
            "the announcement must name the model, got: {out}"
        );
        assert!(
            out.contains("8192"),
            "the announcement must carry the pinned value, got: {out}"
        );
        assert!(
            out.contains("env"),
            "the announcement must name its source, got: {out}"
        );

        unsafe { std::env::remove_var("NANNA_OLLAMA_NUM_CTX") };
    }

    /// With the variable unset the sizing path is today's behavior exactly:
    /// the sized start, and no pin announcement to mislead an operator into
    /// hunting for a setting nobody made.
    #[test]
    fn an_unset_env_neither_pins_nor_announces() {
        let _guard = env_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        // SAFETY: see above.
        unsafe { std::env::remove_var("NANNA_OLLAMA_NUM_CTX") };

        let mut request = request_with_tool_history(serde_json::json!({}));
        request.model = "test-env-pin-unset:1b".to_string();
        request.context_limit = None;

        let out = capture_info_logs(|| {
            assert_eq!(
                LlmClient::resolve_num_ctx(LOCAL_OLLAMA, &request.model, request.context_limit),
                16_384
            );
        });
        assert!(
            !out.contains("NANNA_OLLAMA_NUM_CTX pins"),
            "no pin means no pin announcement, got: {out}"
        );
    }

    /// A remote server's window is not this machine's GPU to size. The fit is
    /// handed a card starved to the floor: a server on this machine gets that
    /// fit, and a server elsewhere gets the start used when VRAM is unknown,
    /// without this machine's card being measured at all.
    #[test]
    fn a_remote_servers_window_is_not_sized_from_this_machines_gpu() {
        let starved = |_: &str| Some(4_096);
        for local in [LOCAL_OLLAMA, "http://localhost:11500", "http://[::1]:11434"] {
            assert_eq!(
                LlmClient::unpinned_start_num_ctx(local, "qwen3:8b", starved),
                4_096,
                "{local} runs on this machine, so this machine's card sizes it"
            );
        }
        for remote in ["https://gpubox.example/ollama", "http://192.168.1.20:11434"] {
            let start = LlmClient::unpinned_start_num_ctx(remote, "qwen3:8b", |_: &str| {
                panic!("measured this machine's GPU to size {remote}")
            });
            assert_eq!(start, 16_384, "{remote} must start at the unknown-VRAM size");
        }
    }

    /// The same through the request path: with nothing pinned, a remote
    /// server's model latches the unknown-VRAM start, and the log says why it
    /// was not sized to the card.
    #[test]
    fn a_remote_server_latches_the_unsized_start_and_says_why() {
        let _guard = env_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        // SAFETY: see above.
        unsafe { std::env::remove_var("NANNA_OLLAMA_NUM_CTX") };

        let model = "test-remote-unsized:1b";
        let out = capture_info_logs(|| {
            assert_eq!(
                LlmClient::resolve_num_ctx("https://gpubox.example/ollama", model, None),
                16_384
            );
        });
        assert_eq!(
            LlmClient::latched_num_ctx(REMOTE_OLLAMA, model),
            Some(16_384)
        );
        assert!(
            out.contains("not on this machine") && out.contains("gpubox.example"),
            "the skipped sizing must name the server and the reason, got: {out}"
        );
    }

    /// A server on another machine: this machine's card must never size it.
    const REMOTE_OLLAMA: &str = "https://gpubox.example/ollama";

    fn never_fit(model: &str) -> Option<u32> {
        panic!("measured this machine's GPU to size {model} on {REMOTE_OLLAMA}")
    }

    /// The switch the latch outlived: a local server's model fitted while a
    /// game held this machine's card, then Settings pointed chat at a remote
    /// server (a config reload, no restart). The remote server gets the size
    /// for a server whose VRAM is unknown, not this card's fit, and that is
    /// the window a client on the remote server reads.
    #[test]
    fn a_size_latched_for_one_server_is_not_served_to_another() {
        let _guard = env_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        // SAFETY: see above.
        unsafe { std::env::remove_var("NANNA_OLLAMA_NUM_CTX") };

        let model = "test-latch-server-switch:8b";
        let starved = |_: &str| Some(4_096);
        assert_eq!(LlmClient::resolve_num_ctx_with(LOCAL_OLLAMA, model, None, starved), 4_096);
        assert_eq!(
            LlmClient::resolve_num_ctx_with(REMOTE_OLLAMA, model, None, never_fit),
            16_384,
            "the local card's fit must not size {REMOTE_OLLAMA}"
        );
        assert_eq!(
            LlmClient::ollama(REMOTE_OLLAMA).effective_num_ctx(model),
            Some(16_384)
        );
    }

    /// Two servers in turn under one model name — a client on this machine and
    /// one on a remote server (as when `[memory].ollama_host` moves between
    /// them, or, before 2026-09-18, when the summarizers had an Ollama address
    /// of their own). Each keeps its own latch: the local server is
    /// not re-fitted on every return (a changed `num_ctx` makes Ollama evict
    /// and reload the model), and the remote server's demotion is not lost to
    /// a fresh start that faults the same way again.
    #[test]
    fn each_server_keeps_its_own_latch_when_one_model_alternates_between_them() {
        let _guard = env_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        // SAFETY: see above.
        unsafe { std::env::remove_var("NANNA_OLLAMA_NUM_CTX") };

        let model = "test-latch-alternating:8b";
        let (local, remote) = (
            LlmClient::ollama(LOCAL_OLLAMA),
            LlmClient::ollama(REMOTE_OLLAMA),
        );
        let fits = std::cell::Cell::new(0_u32);
        let fit = |_: &str| {
            fits.set(fits.get() + 1);
            Some(8_192)
        };
        assert_eq!(LlmClient::resolve_num_ctx_with(LOCAL_OLLAMA, model, None, fit), 8_192);
        assert_eq!(
            LlmClient::resolve_num_ctx_with(REMOTE_OLLAMA, model, None, never_fit),
            16_384
        );
        // The remote server runs out of memory: the ladder walks down its size.
        assert_eq!(remote.demote_context(model, None), Some(12_288));

        assert_eq!(LlmClient::resolve_num_ctx_with(LOCAL_OLLAMA, model, None, fit), 8_192);
        assert_eq!(
            fits.get(),
            1,
            "returning to the local server must resume its size, not re-fit it"
        );
        assert_eq!(local.effective_num_ctx(model), Some(8_192));
        assert_eq!(
            LlmClient::resolve_num_ctx_with(REMOTE_OLLAMA, model, None, never_fit),
            12_288,
            "the remote server's demotion must survive the local server's turn"
        );
        assert_eq!(remote.effective_num_ctx(model), Some(12_288));
    }

    /// A fault before any request sized the model starts from what that
    /// server's first request would carry: the unknown-VRAM start for a server
    /// elsewhere, the ladder's ceiling for one on this machine whose card
    /// nothing has measured. The size is that server's alone.
    #[test]
    fn a_demotion_before_any_request_starts_from_that_servers_own_start() {
        let _guard = env_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        // SAFETY: see above.
        unsafe { std::env::remove_var("NANNA_OLLAMA_NUM_CTX") };

        let model = "test-demotion-before-sizing:8b";
        assert_eq!(
            LlmClient::ollama(REMOTE_OLLAMA).demote_context(model, None),
            Some(12_288)
        );
        assert_eq!(
            LlmClient::ollama(LOCAL_OLLAMA).demote_context(model, None),
            Some(24_576)
        );
        assert_eq!(
            LlmClient::resolve_num_ctx_with(REMOTE_OLLAMA, model, None, never_fit),
            12_288
        );
        let latched = |_: &str| -> Option<u32> { unreachable!("the demoted size is latched") };
        assert_eq!(
            LlmClient::resolve_num_ctx_with(LOCAL_OLLAMA, model, None, latched),
            24_576
        );
    }

    /// A model info cache of its own, in a fresh directory, holding one
    /// provider claim, so `get_model_info` answers without a server. The
    /// directory goes with it.
    struct ClaimCache {
        cache: ModelInfoCache,
        claim: ModelInfo,
        dir: std::path::PathBuf,
    }

    impl ClaimCache {
        fn claiming(model: &str, window: usize) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "nanna-llm-claims-{}-{}",
                std::process::id(),
                model.replace([':', '/'], "-")
            ));
            let claim = ModelInfo {
                id: model.to_string(),
                context_window: window,
                max_output_tokens: window / 2,
                supports_tools: true,
                supports_vision: false,
                embedding_dimension: None,
                cached_at: current_timestamp(),
                provider: "ollama".to_string(),
            };
            Self { cache: ModelInfoCache::new(&dir), claim, dir }
        }

        /// `client`'s model info for `model`, from this cache.
        ///
        /// The cache answers only for the server that gave an entry, and holds
        /// one entry per model, so the claim is filed under `client`'s server
        /// right before the read: every server claims the same window, and
        /// what differs between clients is only the `num_ctx` latch the
        /// window is clamped to. Without this the lookup would go to the
        /// network.
        fn info(&self, client: &LlmClient, model: &str) -> ModelInfo {
            self.cache
                .set(&client.model_info_server(), &self.claim)
                .expect("write the claim");
            futures::executor::block_on(client.get_model_info(model, Some(&self.cache)))
        }
    }

    impl Drop for ClaimCache {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    /// One model, two servers: one client on this machine (where the
    /// summarizers' own `[llm].ollama_url` pointed until 2026-09-18), another
    /// on a remote `[memory].ollama_host`. The local
    /// server fitted 4096 while a game held the card; the remote one started
    /// at 16384, and chat's request was the latest. The summarizer's chunks
    /// are cut to the window its client reports, and its requests carry 4096:
    /// a window read from chat's server cut them for 16384, and Ollama
    /// silently dropped three quarters of each.
    #[test]
    fn model_info_is_sized_for_the_server_the_client_sends_to() {
        let _guard = env_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        // SAFETY: see above.
        unsafe { std::env::remove_var("NANNA_OLLAMA_NUM_CTX") };

        let model = "test-info-per-server:8b";
        let claims = ClaimCache::claiming(model, 32_768);
        let starved = |_: &str| Some(4_096);
        assert_eq!(
            LlmClient::resolve_num_ctx_with(LOCAL_OLLAMA, model, None, starved),
            4_096
        );
        assert_eq!(
            LlmClient::resolve_num_ctx_with(REMOTE_OLLAMA, model, None, never_fit),
            16_384
        );

        let summarizer = LlmClient::ollama(LOCAL_OLLAMA);
        assert_eq!(
            claims.info(&summarizer, model).context_window,
            4_096,
            "the summarizer's requests carry 4096, whatever chat's server was sized to"
        );
        let chat = LlmClient::ollama(REMOTE_OLLAMA);
        assert_eq!(claims.info(&chat, model).context_window, 16_384);
    }

    /// Chat ran on this machine's server at a 24576 fit; Settings then point it
    /// at a remote server that has never served the model. That server's first
    /// request carries 16384, so the prompt must be sized for 16384 before it
    /// goes: sized for the old server's 24576, the first turn after the switch
    /// was truncated silently.
    #[test]
    fn after_a_server_switch_the_prompt_is_sized_for_the_new_servers_start() {
        let _guard = env_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        // SAFETY: see above.
        unsafe { std::env::remove_var("NANNA_OLLAMA_NUM_CTX") };

        let model = "test-info-server-switch:8b";
        let claims = ClaimCache::claiming(model, 32_768);
        let roomy = |_: &str| Some(24_576);
        assert_eq!(
            LlmClient::resolve_num_ctx_with(LOCAL_OLLAMA, model, None, roomy),
            24_576
        );

        let chat = LlmClient::ollama(REMOTE_OLLAMA);
        assert_eq!(
            claims.info(&chat, model).context_window,
            16_384,
            "the budget for the new server's first request"
        );
        assert_eq!(
            LlmClient::resolve_num_ctx_with(REMOTE_OLLAMA, model, None, never_fit),
            16_384,
            "the size that request carries"
        );
    }

    /// The window a client reports before a model's first request is the
    /// `num_ctx` that request carries, whenever it can be known without
    /// measuring: the operator's pin, or for a server elsewhere the
    /// unknown-VRAM start. A server on this machine with nothing pinned is
    /// sized by measuring this machine's card, which only the first request
    /// does, so until then the provider's claim stands.
    #[test]
    fn the_window_read_before_a_request_is_the_num_ctx_it_carries() {
        let _guard = env_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let fit = |_: &str| Some(8_192);

        // SAFETY: see above.
        unsafe { std::env::remove_var("NANNA_OLLAMA_NUM_CTX") };
        let model = "test-read-before-request-remote:8b";
        let remote = LlmClient::ollama(REMOTE_OLLAMA);
        assert_eq!(remote.effective_num_ctx(model), Some(16_384));
        assert_eq!(
            LlmClient::resolve_num_ctx_with(REMOTE_OLLAMA, model, None, never_fit),
            16_384
        );

        let model = "test-read-before-request-local:8b";
        let local = LlmClient::ollama(LOCAL_OLLAMA);
        assert_eq!(
            local.effective_num_ctx(model),
            None,
            "unmeasured, so unknown"
        );
        assert_eq!(
            LlmClient::resolve_num_ctx_with(LOCAL_OLLAMA, model, None, fit),
            8_192
        );
        assert_eq!(local.effective_num_ctx(model), Some(8_192));

        // SAFETY: see above.
        unsafe { std::env::set_var("NANNA_OLLAMA_NUM_CTX", "12288") };
        for (server, model) in [
            (LOCAL_OLLAMA, "test-read-before-request-pinned-local:8b"),
            (REMOTE_OLLAMA, "test-read-before-request-pinned-remote:8b"),
        ] {
            assert_eq!(
                LlmClient::ollama(server).effective_num_ctx(model),
                Some(12_288)
            );
            assert_eq!(
                LlmClient::resolve_num_ctx_with(server, model, None, fit),
                12_288
            );
        }
        unsafe { std::env::remove_var("NANNA_OLLAMA_NUM_CTX") };
    }

    /// Only an Ollama server has a `num_ctx`. A cloud provider's address is
    /// not on this machine, and it must not read as a remote Ollama starting
    /// at 16384, nor inherit a window an Ollama server latched for a model
    /// of the same name.
    #[test]
    fn a_cloud_client_is_never_clamped_by_an_ollama_window() {
        let _guard = env_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        // SAFETY: see above.
        unsafe { std::env::remove_var("NANNA_OLLAMA_NUM_CTX") };

        let model = "test-cloud-same-name:8b";
        let fit = |_: &str| Some(4_096);
        assert_eq!(
            LlmClient::resolve_num_ctx_with(LOCAL_OLLAMA, model, None, fit),
            4_096
        );
        for cloud in [
            LlmClient::openrouter("key"),
            LlmClient::anthropic("key"),
            LlmClient::openai("key"),
        ] {
            assert_eq!(
                cloud.effective_num_ctx(model),
                None,
                "{:?}",
                cloud.provider()
            );
            assert_eq!(cloud.effective_context_window(model, 32_000), 32_000);
            assert_eq!(cloud.demote_context(model, None), None, "nothing to demote");
        }
    }

    /// A GPU-memory fault is walked down on the server that faulted. Another
    /// server's size for the same model is untouched, whichever server the
    /// model's latest request went to: walking that one instead left the
    /// faulting server at the size that faults.
    #[test]
    fn a_fault_walks_down_only_the_window_of_the_server_that_faulted() {
        let _guard = env_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        // SAFETY: see above.
        unsafe { std::env::remove_var("NANNA_OLLAMA_NUM_CTX") };

        let model = "test-fault-per-server:8b";
        let fit = |_: &str| Some(8_192);
        assert_eq!(
            LlmClient::resolve_num_ctx_with(LOCAL_OLLAMA, model, None, fit),
            8_192
        );
        assert_eq!(
            LlmClient::resolve_num_ctx_with(REMOTE_OLLAMA, model, None, never_fit),
            16_384
        );

        let local = LlmClient::ollama(LOCAL_OLLAMA);
        assert_eq!(local.demote_context(model, None), Some(6_144));
        assert_eq!(local.effective_num_ctx(model), Some(6_144));
        assert_eq!(
            LlmClient::ollama(REMOTE_OLLAMA).effective_num_ctx(model),
            Some(16_384)
        );
        assert_eq!(
            LlmClient::resolve_num_ctx_with(REMOTE_OLLAMA, model, None, never_fit),
            16_384
        );
        assert_eq!(
            LlmClient::resolve_num_ctx_with(LOCAL_OLLAMA, model, None, fit),
            6_144
        );
    }

    #[test]
    fn only_a_server_on_this_machine_reads_as_local() {
        for local in [
            "http://127.0.0.1:11434",
            "http://localhost:11434",
            " http://LocalHost:11434/ ",
            "http://127.0.0.2:8080",
            "http://[::1]:11434",
            "http://[::ffff:127.0.0.1]:11434",
            // Unspecified: dialled, it reaches this machine.
            "http://0.0.0.0:11434",
            "http://[::]:11434",
        ] {
            assert!(ollama_server_is_local(local), "{local} is on this machine");
        }
        for remote in [
            "https://gpubox.example/ollama",
            "http://192.168.1.20:11434",
            "https://localhost.example.com",
            "https://host/127.0.0.1",
            "http://[2001:db8::1]:11434",
            // Unparseable fails safe to remote.
            "",
            "127.0.0.1:11434",
            "not a url",
        ] {
            assert!(!ollama_server_is_local(remote), "{remote:?} must read as remote");
        }
    }

    /// Ollama's default local address: the server the sizing tests size for.
    const LOCAL_OLLAMA: &str = "http://127.0.0.1:11434";

    fn env_lock() -> &'static std::sync::Mutex<()> {
        static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
        LOCK.get_or_init(Default::default)
    }

    /// Run `f` with a thread-local INFO subscriber and return everything it
    /// logged. Thread-local (`with_default`, not the global default) so
    /// parallel tests on other threads neither pollute nor race this capture.
    /// Serializes the log-capture tests.
    ///
    /// `with_default` installs a subscriber per thread, but tracing keeps a
    /// PROCESS-wide max-level hint recomputed as subscribers come and go, and a
    /// callsite evaluated during another thread's install/uninstall window can
    /// be cached as disabled. The capture then comes back empty and the test
    /// fails on its first `contains` assertion — intermittently, and only when
    /// something else in the crate happens to be running alongside it.
    static LOG_CAPTURE: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Pin tracing's PROCESS-wide max-level hint at INFO for the whole test
    /// binary.
    ///
    /// This is the other half of the race the comment above describes, and the
    /// half `LOG_CAPTURE` cannot fix. The mutex serializes the capture tests
    /// against *each other*, but the hint is recomputed every time a subscriber
    /// is installed or dropped, and it is global. With no global default
    /// installed, the hint falls back to OFF in the window between one
    /// `with_default` guard dropping and the next installing — and any of the
    /// crate's ~119 other tests running concurrently can evaluate a callsite in
    /// that window and have it cached as disabled. The next capture then comes
    /// back EMPTY and the test fails on its first `contains`.
    ///
    /// Observed 2026-09-09 under `cargo test --workspace -j 4`:
    /// `a_demotion_announces_old_new_floor_and_clamp` failed with
    /// `the decision: ` (an empty capture) while passing alone and 3/3 for the
    /// crate on its own — i.e. exactly when the crate is run alongside enough
    /// other work.
    ///
    /// Installing a permanent no-op INFO subscriber holds the hint at INFO for
    /// the process's whole life, so a callsite is never cached as disabled and
    /// the thread-local capture always sees its events. The global sink
    /// discards; only the thread-local `with_default` subscriber records.
    fn pin_global_log_level() {
        static PINNED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        PINNED.get_or_init(|| {
            struct Discard;
            impl std::io::Write for Discard {
                fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                    Ok(buf.len())
                }
                fn flush(&mut self) -> std::io::Result<()> {
                    Ok(())
                }
            }
            let subscriber = tracing_subscriber::fmt()
                .with_max_level(tracing::Level::INFO)
                .with_writer(|| Discard)
                .with_ansi(false)
                .finish();
            // Another test harness may legitimately have set one already; the
            // hint is what matters, not whose subscriber won.
            let _ = tracing::subscriber::set_global_default(subscriber);
        });
    }

    fn capture_info_logs(f: impl FnOnce()) -> String {
        #[derive(Clone, Default)]
        struct Sink(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);
        impl std::io::Write for Sink {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                self.0
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for Sink {
            type Writer = Self;
            fn make_writer(&'a self) -> Self::Writer {
                self.clone()
            }
        }

        pin_global_log_level();
        let _serialized = LOG_CAPTURE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);

        let sink = Sink::default();
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .with_writer(sink.clone())
            .with_ansi(false)
            .finish();
        tracing::subscriber::with_default(subscriber, f);
        let bytes = sink.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// A big co-resident model is real pressure and must actually shrink the
    /// window, unlike the assumed-constant it replaced.
    #[test]
    fn a_large_co_resident_model_shrinks_the_window() {
        let tight =
            LlmClient::fit_context_for_budget(FREE_UNLOADED, 7.56 * GB, 0.0, 2.5 * GB);
        assert!(tight < 16_384, "2.5 GB of neighbour must cost context, got {tight}");
    }

    #[test]
    fn a_card_with_nothing_left_still_returns_a_runnable_floor() {
        assert_eq!(
            LlmClient::fit_context_for_budget(4.0 * GB, 7.56 * GB, 0.0, 0.0),
            4_096
        );
        assert_eq!(LlmClient::fit_context_for_budget(0.0, 0.0, 0.0, 0.0), 4_096);
    }

    // -----------------------------------------------------------------
    // Degenerate-repetition watch
    // -----------------------------------------------------------------

    #[test]
    fn a_wedged_runner_is_caught() {
        // The live shape: the same one-character token forever.
        let mut w = RepeatWatch::default();
        let mut tripped_at = None;
        for i in 1..=60 {
            if let Some(run) = w.observe("0") {
                tripped_at = Some((i, run));
                break;
            }
        }
        let (i, run) = tripped_at.expect("a stuck token must be caught");
        assert_eq!(i, RepeatWatch::MAX_RUN, "must trip exactly at the bound");
        assert_eq!(run, RepeatWatch::MAX_RUN);
    }

    #[test]
    fn ordinary_repetition_is_left_alone() {
        // Indentation and ellipses repeat legitimately -- just not 40x.
        let mut w = RepeatWatch::default();
        for _ in 0..(RepeatWatch::MAX_RUN - 1) {
            assert!(w.observe("  ").is_none(), "whitespace is not a candidate");
        }
        // Real content RESETS the run, so separated bursts never accumulate
        // into a false positive — this is what keeps the guard off normal
        // output that happens to repeat a short token in places. Counts are
        // expressed against the bound so lowering it can't silently make
        // this test vacuous (it did exactly that at 40 → 20).
        let burst = RepeatWatch::MAX_RUN - 1;
        let mut w2 = RepeatWatch::default();
        for _ in 0..burst {
            assert!(w2.observe(".").is_none());
        }
        assert!(w2.observe(" and then some prose").is_none());
        for _ in 0..burst {
            assert!(
                w2.observe(".").is_none(),
                "the run must restart after intervening content"
            );
        }
    }

    /// The wedge carries its own fingerprint, and still renders as the 502 it
    /// used to be.
    ///
    /// Both halves are contracts. The fields exist so the daemon's retry
    /// ladder can compare one wedge against the next without parsing prose;
    /// the rendered text is what `is_transient_llm_error` ("API error: 5") and
    /// `wedged_runner_error` ("same token") match on, so a reword here would
    /// stop wedges being retried at all.
    #[test]
    fn a_wedge_carries_structure_and_keeps_its_rendered_form() {
        let err = LlmError::WedgedRunner {
            token: "0".to_string(),
            run: RepeatWatch::MAX_RUN,
            bytes_received: 2780,
        };
        let LlmError::WedgedRunner { token, run, bytes_received } = &err else {
            panic!("must stay a distinct variant the caller can match on");
        };
        assert_eq!((token.as_str(), *run, *bytes_received), ("0", 20, 2780));

        let rendered = err.to_string();
        assert!(rendered.contains("API error: 5"), "must stay transient: {rendered}");
        assert!(rendered.contains("same token"), "must stay a wedge: {rendered}");
        assert!(rendered.contains("2780 bytes"), "must still say where it gave up: {rendered}");
        // And it must keep taking the 502 fallback path it took as an Api error.
        assert!(err.should_fallback());
    }

    #[test]
    fn a_repeated_long_fragment_is_content_not_a_fault() {
        let mut w = RepeatWatch::default();
        for _ in 0..200 {
            assert!(
                w.observe("SELECT * FROM t;").is_none(),
                "long repeated fragments are legitimate output"
            );
        }
    }
    use super::*;

    #[test]
    fn test_message_construction() {
        let msg = Message::user("Hello");
        assert!(matches!(msg.role, Role::User));
        assert_eq!(msg.content, "Hello");
    }

    /// Request with an echoed tool call in history — the shape that follows
    /// every tool execution in the agent loop.
    fn request_with_tool_history(input: serde_json::Value) -> AnthropicRequest {
        AnthropicRequest {
            model: "ornith:latest".to_string(),
            messages: vec![
                AnthropicMessage::user_text("check the lock file"),
                AnthropicMessage::assistant(vec![ContentBlock::ToolUse {
                    id: "call_1".to_string(),
                    name: "read_file".to_string(),
                    input,
                }]),
                AnthropicMessage::user(vec![ContentBlock::ToolResult {
                    tool_use_id: "call_1".to_string(),
                    content: "NO_LOCK".to_string(),
                    is_error: None,
                }]),
            ],
            max_tokens: 128,
            temperature: None,
            system: None,
            tools: None,
            stream: None,
            thinking: None,
            cache_control: None,
            context_limit: None,
        }
    }

    fn first_tool_arguments(messages: &serde_json::Value) -> serde_json::Value {
        messages
            .as_array()
            .unwrap()
            .iter()
            .find_map(|m| m.get("tool_calls"))
            .and_then(|tc| tc.as_array())
            .and_then(|tc| tc.first())
            .map(|tc| tc["function"]["arguments"].clone())
            .expect("converted request must contain a tool_call")
    }

    #[test]
    fn openai_conversion_keeps_tool_arguments_as_json_string() {
        // OpenAI-compatible /v1/chat/completions requires the string form.
        let request = request_with_tool_history(serde_json::json!({"path": "D:/x"}));
        let (messages, _) = anthropic_to_openai_request(&request);
        let args = first_tool_arguments(&messages);
        assert_eq!(args, serde_json::json!("{\"path\":\"D:/x\"}"));
    }

    #[test]
    fn ollama_conversion_keeps_tool_arguments_as_object() {
        // Ollama's native /api/chat 400s on string-form arguments — even valid
        // JSON inside a string ("Value looks like object, but can't find
        // closing '}' symbol", verified live against Ollama 0.31.2).
        let request = request_with_tool_history(serde_json::json!({"path": "D:/x"}));
        let (messages, _) = anthropic_to_ollama_request(&request);
        let args = first_tool_arguments(&messages);
        assert_eq!(args, serde_json::json!({"path": "D:/x"}));
    }

    #[test]
    fn ollama_conversion_degrades_non_object_input_to_empty_object() {
        // A non-object input would 400 the whole request on the native
        // endpoint; losing the arguments beats losing the turn.
        let request = request_with_tool_history(serde_json::json!("not an object"));
        let (messages, _) = anthropic_to_ollama_request(&request);
        let args = first_tool_arguments(&messages);
        assert_eq!(args, serde_json::json!({}));
    }

    #[test]
    fn test_estimate_tokens_ascii_and_cjk() {
        assert_eq!(estimate_tokens(""), 0);
        // ASCII English: ~4 chars/token, rounded up so short strings aren't zero.
        assert_eq!(estimate_tokens("hi"), 1);
        assert_eq!(estimate_tokens(&"a".repeat(8)), 2);
        // CJK is denser: 4 chars ~ 4 tokens (byte-len/4 would wrongly give 3).
        let cjk = "你好世界"; // 4 chars, 12 UTF-8 bytes
        assert_eq!(cjk.len(), 12);
        assert_eq!(estimate_tokens(cjk), 4);
        // Mixed: 2 ASCII (→1) + 2 wide (→2) = 3.
        assert_eq!(estimate_tokens("hi你好"), 3);
    }

    #[test]
    fn test_estimate_tokens_code_vs_english_ratio() {
        // Pure prose ASCII letter run → English ratio (4 chars/token).
        let prose = "abcdefgh"; // 8 chars
        assert_eq!(
            estimate_tokens_for_family(prose, TokenContentFamily::English),
            2
        );
        // Same length treated as code → denser 3.5 chars/token → ceil(8/3.5)=3.
        assert_eq!(
            estimate_tokens_for_family(prose, TokenContentFamily::Code),
            3
        );
        // Auto-detect: punctuation-heavy JSON should pick code path.
        let jsonish = r#"{"a":1,"b":2,"c":3,"d":4}"#; // lots of braces/colons
        let auto = estimate_tokens_for_family(jsonish, TokenContentFamily::Auto);
        let as_code = estimate_tokens_for_family(jsonish, TokenContentFamily::Code);
        assert_eq!(auto, as_code);
    }

    #[test]
    fn test_estimate_request_tokens_adds_message_framing() {
        // Two 8-char ASCII messages: 2 tokens content each + framing per message.
        let request = CompletionRequest::default()
            .with_message(Message::user("aaaaaaaa"))
            .with_message(Message::user("bbbbbbbb"));
        let expected = 2 * (2 + MESSAGE_FRAMING_TOKENS);
        assert_eq!(estimate_request_tokens(&request), expected);
    }

    #[test]
    fn test_estimate_tokens_exact_smoke() {
        // Empty always 0; non-empty path must agree with some positive count
        // (exact under the tiktoken feature, heuristic otherwise).
        assert_eq!(estimate_tokens_exact(""), 0);
        let n = estimate_tokens_exact("hello world");
        assert!(n >= 1, "expected at least 1 token, got {n}");
        // Model-hinted path should also return a positive count.
        let m = estimate_tokens_for_model("hello world", "gpt-4o");
        assert!(m >= 1, "expected at least 1 token for gpt-4o, got {m}");
    }

    #[test]
    fn test_request_builder() {
        let request = CompletionRequest::default()
            .with_model("gpt-4")
            .with_message(Message::user("Hi"))
            .with_max_tokens(1000)
            .with_temperature(0.5);

        assert_eq!(request.model, "gpt-4");
        assert_eq!(request.messages.len(), 1);
        assert_eq!(request.max_tokens, Some(1000));
        assert_eq!(request.temperature, Some(0.5));
    }

    #[test]
    fn test_delta_data_unknown_variant() {
        // Truly unknown delta types should deserialize to Unknown
        let json = r#"{"type": "some_future_delta", "data": "xyz"}"#;
        let delta: DeltaData = serde_json::from_str(json).unwrap();
        assert!(matches!(delta, DeltaData::Unknown));
    }

    #[test]
    fn test_delta_data_signature_variant() {
        // signature_delta should now be recognized (required for multi-turn thinking)
        let json = r#"{"type": "signature_delta", "signature": "abc123"}"#;
        let delta: DeltaData = serde_json::from_str(json).unwrap();
        assert!(matches!(delta, DeltaData::SignatureDelta { .. }));
        if let DeltaData::SignatureDelta { signature } = delta {
            assert_eq!(signature, "abc123");
        }
    }

    #[test]
    fn test_delta_data_known_variants() {
        let text = r#"{"type": "text_delta", "text": "hello"}"#;
        let delta: DeltaData = serde_json::from_str(text).unwrap();
        assert!(matches!(delta, DeltaData::TextDelta { .. }));

        let json_delta = r#"{"type": "input_json_delta", "partial_json": "{\"key\":"}"#;
        let delta: DeltaData = serde_json::from_str(json_delta).unwrap();
        assert!(matches!(delta, DeltaData::InputJsonDelta { .. }));
    }

    #[test]
    fn test_content_block_delta_with_signature_delta() {
        // signature_delta should parse as SignatureDelta now
        let json = r#"{"type": "content_block_delta", "index": 1, "delta": {"type": "signature_delta", "signature": "sig"}}"#;
        let sse: AnthropicSSE = serde_json::from_str(json).unwrap();
        match sse {
            AnthropicSSE::ContentBlockDelta { index, delta } => {
                assert_eq!(index, 1);
                assert!(matches!(delta, DeltaData::SignatureDelta { .. }));
            }
            other => panic!("Expected ContentBlockDelta, got {other:?}"),
        }
    }

    #[test]
    fn test_content_block_delta_with_unknown_delta() {
        // Truly unknown delta types should still parse via Unknown fallback
        let json = r#"{"type": "content_block_delta", "index": 2, "delta": {"type": "future_delta_type"}}"#;
        let sse: AnthropicSSE = serde_json::from_str(json).unwrap();
        match sse {
            AnthropicSSE::ContentBlockDelta { index, delta } => {
                assert_eq!(index, 2);
                assert!(matches!(delta, DeltaData::Unknown));
            }
            other => panic!("Expected ContentBlockDelta, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_sse_event_signature_delta() {
        let event = "data: {\"type\": \"content_block_delta\", \"index\": 0, \"delta\": {\"type\": \"signature_delta\", \"signature\": \"abc\"}}";
        // Should now return a SignatureDelta event
        let result = parse_sse_event(event);
        assert!(result.is_some());
        assert!(matches!(result.unwrap(), StreamEvent::SignatureDelta { index: 0, .. }));
    }

    /// An overload inside a 200 stream is what a fallback model is for. It
    /// used to parse to a flat `Error { message }` that no consumer could tell
    /// from a malformed request — and that the agent loop ignored outright.
    #[test]
    fn an_sse_error_is_typed_so_an_overload_falls_back() {
        let overloaded = "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}";
        match parse_sse_event(overloaded).expect("an error event parses") {
            StreamEvent::RecoverableError { error, .. } => {
                assert!(error.should_fallback(), "{error}");
                assert!(error.to_string().contains("overloaded_error"), "{error}");
            }
            other => panic!("an overload must be recoverable, got {other:?}"),
        }
        let limited = "data: {\"type\":\"error\",\"error\":{\"type\":\"rate_limit_error\",\"message\":\"slow down\"}}";
        assert!(matches!(
            parse_sse_event(limited),
            Some(StreamEvent::RecoverableError { .. })
        ));
        let invalid = "data: {\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":\"bad\"}}";
        assert!(
            matches!(parse_sse_event(invalid), Some(StreamEvent::Error { .. })),
            "a bad request is final: another model would get the same request"
        );
        let untyped = "data: {\"type\":\"error\",\"error\":{\"message\":\"???\"}}";
        assert!(
            parse_sse_event(untyped).is_some(),
            "an untyped error still parses"
        );
    }

    #[test]
    fn test_parse_sse_event_message_start_carries_usage() {
        // message_start reports prompt-side usage incl. cache read/write — the
        // streaming path must surface these for accurate cache-cost tracking.
        let event = "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-sonnet-4\",\"usage\":{\"input_tokens\":12,\"cache_read_input_tokens\":900,\"cache_creation_input_tokens\":34}}}";
        let result = parse_sse_event(event).expect("message_start should parse");
        match result {
            StreamEvent::MessageStart {
                input_tokens,
                cache_read_tokens,
                cache_creation_tokens,
                ..
            } => {
                assert_eq!(input_tokens, 12);
                assert_eq!(cache_read_tokens, 900);
                assert_eq!(cache_creation_tokens, 34);
            }
            other => panic!("expected MessageStart, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_sse_event_message_start_without_usage_is_zero() {
        // Missing usage (or a non-Anthropic shape) must default to zeros, not fail.
        let event = "data: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"model\":\"x\"}}";
        let result = parse_sse_event(event).expect("message_start should parse");
        assert!(matches!(
            result,
            StreamEvent::MessageStart {
                input_tokens: 0,
                cache_read_tokens: 0,
                cache_creation_tokens: 0,
                ..
            }
        ));
    }

    fn model_info(context_window: usize, max_output_tokens: usize) -> ModelInfo {
        ModelInfo {
            id: "test".into(),
            context_window,
            max_output_tokens,
            supports_tools: false,
            supports_vision: false,
            embedding_dimension: None,
            cached_at: 0,
            provider: "test".into(),
        }
    }

    #[test]
    fn compression_threshold_stays_below_hard_limit_for_small_models() {
        // Small model, large output budget: 80%-of-context (6400) would exceed
        // the hard input cap (6000), so proactive compression must be pulled
        // below it instead of never firing.
        let small = model_info(8_000, 4_096);
        assert!(
            small.compression_threshold() < small.hard_input_limit(),
            "threshold {} must be below hard limit {}",
            small.compression_threshold(),
            small.hard_input_limit()
        );
    }

    #[test]
    fn compression_threshold_unchanged_for_large_models() {
        // Large context model: 80%-of-context is already the smaller value, so
        // the fix must not perturb it (no regression for cloud models).
        let large = model_info(200_000, 8_192);
        assert_eq!(large.compression_threshold(), 160_000);
        assert!(large.compression_threshold() < large.hard_input_limit());
    }

    #[test]
    fn compression_threshold_holds_when_output_exceeds_context() {
        // Degenerate provider report (max_output >= context): hard_input_limit
        // floors at 50% of context; the threshold must still sit at/below it.
        let degenerate = model_info(4_000, 8_000);
        assert!(degenerate.compression_threshold() <= degenerate.hard_input_limit());
    }

    #[test]
    fn hard_input_limit_caps_the_output_reservation() {
        // The live regression: Ollama reports max_output >= context for
        // qwen3.5:9b (32k window), and subtracting the claim verbatim halved
        // the input side to 16k — the user's prompt got emergency-truncated.
        // Reservation is capped at 25% of the window, so 24k stays usable.
        let qwen_like = model_info(32_000, 32_000);
        assert_eq!(qwen_like.hard_input_limit(), 24_000);
        // A modest, honest output claim is reserved in full.
        let honest = model_info(32_000, 4_096);
        assert_eq!(honest.hard_input_limit(), 32_000 - 4_096);
    }

    #[test]
    fn dynamic_output_reserve_frees_input_for_small_budgets() {
        // The reserve tracks the actual request budget, not the provider's
        // max_output claim: a 2k-output agent on a 32k window keeps ~94%
        // for input, and the pair never over-commits.
        let m = model_info(32_000, 32_000);
        assert_eq!(m.effective_output_budget(2_048), 2_048);
        assert_eq!(m.hard_input_limit_for(2_048), 29_952);
        // Degenerate request budgets are bounded by half the window.
        assert_eq!(m.effective_output_budget(30_000), 16_000);
        assert_eq!(m.hard_input_limit_for(16_000), 16_000);
        for requested in [1_usize, 2_048, 8_192, 30_000, 64_000] {
            let out = m.effective_output_budget(requested);
            assert!(
                m.hard_input_limit_for(out) + out <= 32_000,
                "over-commit at requested={requested}"
            );
            assert!(m.compression_threshold_for(out) <= m.hard_input_limit_for(out));
        }
    }

    #[test]
    fn input_and_output_budgets_never_over_commit_the_window() {
        // The paired-budget invariant: a request whose input respects
        // hard_input_limit and whose max_tokens respects output_token_budget
        // can never exceed the model's context window.
        for (ctx, out) in [
            (32_000, 32_000),
            (32_000, 4_096),
            (8_000, 4_096),
            (4_000, 8_000),
            (2_000, 1_000),
        ] {
            let m = model_info(ctx, out);
            assert!(
                m.hard_input_limit() + m.output_token_budget() <= ctx,
                "over-commit at ctx={ctx} out={out}: {} + {} > {ctx}",
                m.hard_input_limit(),
                m.output_token_budget()
            );
        }
    }

    #[test]
    fn model_context_window_uses_universal_floor_without_cache() {
        // No per-model table: uncached names all inherit the conservative floor.
        assert_eq!(model_context_window("claude-opus-4-8"), UNKNOWN_CONTEXT_WINDOW);
        assert_eq!(model_context_window("gpt-4"), UNKNOWN_CONTEXT_WINDOW);
        assert_eq!(model_context_window("some-unknown-model"), UNKNOWN_CONTEXT_WINDOW);
        let info = unknown_model_info("x", "test");
        assert_eq!(info.context_window, UNKNOWN_CONTEXT_WINDOW);
        assert_eq!(info.max_output_tokens, UNKNOWN_MAX_OUTPUT_TOKENS);
    }

    #[test]
    fn conversation_history_budget_respects_hard_limit() {
        let small = model_info(8_000, 4_096);
        let budget = small.conversation_history_budget(2_000, 2_000, 1_000);
        assert!(budget <= small.hard_input_limit());
        assert!(budget >= 1_000);
        let tiny = model_info(2_000, 1_000);
        // min_history must not exceed hard limit
        assert_eq!(
            tiny.conversation_history_budget(10_000, 10_000, 5_000),
            tiny.hard_input_limit()
        );
    }
}

#[cfg(test)]
mod anthropic_model_contract_tests {
    use super::{
        anthropic_model_contract, conform_to_anthropic_contract, AnthropicRequest, ThinkingConfig,
        ThinkingDisplay,
    };

    #[test]
    fn the_current_generation_dropped_budgets_and_sampling() {
        for model in [
            "claude-opus-5",
            "claude-opus-4-8",
            "claude-opus-4-7",
            "claude-sonnet-5",
        ] {
            let c = anthropic_model_contract(model);
            assert!(c.adaptive_thinking(), "{model} takes adaptive thinking");
            assert!(c.sampling_removed, "{model} rejects temperature");
            assert!(
                c.display_defaults_omitted(),
                "{model} hides reasoning text unless display is asked for"
            );
            assert!(!c.thinking_always_on(), "{model} accepts an explicit disable");
        }
    }

    #[test]
    fn fable_and_mythos_cannot_be_told_not_to_think() {
        for model in ["claude-fable-5", "claude-mythos-5"] {
            let c = anthropic_model_contract(model);
            assert!(c.thinking_always_on(), "{model} rejects an explicit disable");
            assert!(c.adaptive_thinking());
            assert!(c.sampling_removed);
        }
    }

    /// Mythos *Preview* is the model those two were migrated away FROM — the
    /// published "before" example configures it with `budget_tokens`, so it
    /// must not be swept into the always-on row by the `mythos` substring.
    #[test]
    fn mythos_preview_is_a_migration_source_not_a_mythos_five() {
        let c = anthropic_model_contract("claude-mythos-preview");
        assert!(!c.adaptive_thinking(), "it still takes budget_tokens");
        assert!(!c.thinking_always_on(), "nothing documents a rejected disable");
        assert!(!c.sampling_removed, "it still takes temperature");
    }

    #[test]
    fn the_four_six_family_is_adaptive_but_kept_sampling() {
        for model in ["claude-opus-4-6", "claude-sonnet-4-6"] {
            let c = anthropic_model_contract(model);
            assert!(c.adaptive_thinking(), "{model} takes adaptive thinking");
            assert!(!c.sampling_removed, "{model} kept temperature");
            assert!(
                !c.display_defaults_omitted(),
                "{model} already defaults to summarized reasoning"
            );
        }
    }

    #[test]
    fn pre_four_six_models_keep_budgets_and_sampling() {
        for model in [
            "claude-sonnet-4-20250514",
            "claude-opus-4-20250514",
            "claude-opus-4-5",
            "claude-opus-4-5-20251101",
            "claude-opus-4-1-20250805",
            "claude-sonnet-4-5-20250929",
            "claude-haiku-4-5-20251001",
            "claude-3-5-sonnet-20241022",
            "claude-3-opus-20240229",
        ] {
            let c = anthropic_model_contract(model);
            assert!(!c.adaptive_thinking(), "{model} still takes budget_tokens");
            assert!(!c.sampling_removed, "{model} still takes temperature");
        }
    }

    /// The family patterns are substrings, so the generations they must not
    /// confuse are worth pinning: `sonnet-5` does not occur inside
    /// `sonnet-4-5`, and `opus-5` does not occur inside `opus-4-5`. Getting
    /// this wrong sends `budget_tokens` to a model that 400s on it, or strips
    /// `temperature` from one that wanted it.
    #[test]
    fn adjacent_generations_do_not_collide() {
        assert!(anthropic_model_contract("claude-sonnet-5").adaptive_thinking());
        assert!(!anthropic_model_contract("claude-sonnet-4-5").adaptive_thinking());
        assert!(anthropic_model_contract("claude-opus-5").adaptive_thinking());
        assert!(!anthropic_model_contract("claude-opus-4-5").adaptive_thinking());
    }

    #[test]
    fn provider_prefixes_and_dotted_aliases_still_classify() {
        // Bedrock prefixes the first-party id; some configs write `4.6`.
        assert!(anthropic_model_contract("anthropic.claude-opus-5").sampling_removed);
        assert!(anthropic_model_contract("anthropic/claude-opus-4-6").adaptive_thinking());
        assert!(!anthropic_model_contract("claude-opus-4.6").sampling_removed);
        assert!(anthropic_model_contract("CLAUDE-OPUS-5").adaptive_thinking());
    }

    /// An unrecognized name is assumed to be newer than this table, not older:
    /// every generation since 4.6 removed `budget_tokens`, so guessing legacy
    /// would 400 on the first request against a model we simply had not heard
    /// of yet.
    #[test]
    fn unknown_models_get_the_current_contract() {
        let c = anthropic_model_contract("claude-something-unreleased");
        assert!(c.adaptive_thinking());
        assert!(c.sampling_removed);
        assert!(c.display_defaults_omitted());
    }

    /// Every row of the table, with all four facts pinned at once. The tests
    /// above each check the facts their family is known for; this one holds the
    /// complete answer per row, so a change to how the contract is represented
    /// cannot move a fact nobody happened to assert.
    #[test]
    fn every_row_keeps_all_four_facts() {
        // (model, adaptive_thinking, display_defaults_omitted, sampling_removed, thinking_always_on)
        let rows = [
            ("claude-fable-5", true, true, true, true),
            ("claude-mythos-5", true, true, true, true),
            ("claude-opus-5", true, true, true, false),
            ("claude-sonnet-5", true, true, true, false),
            ("claude-opus-4-8", true, true, true, false),
            ("claude-opus-4-7", true, true, true, false),
            ("claude-something-unreleased", true, true, true, false),
            ("claude-opus-4-6", true, false, false, false),
            ("claude-sonnet-4-6", true, false, false, false),
            ("claude-mythos-preview", false, false, false, false),
            ("claude-opus-4-5", false, false, false, false),
            ("claude-sonnet-4-20250514", false, false, false, false),
            ("claude-haiku-4-5-20251001", false, false, false, false),
            ("claude-3-opus-20240229", false, false, false, false),
        ];
        for (model, adaptive, omitted, sampling_removed, always_on) in rows {
            let c = anthropic_model_contract(model);
            assert_eq!(
                (
                    c.adaptive_thinking(),
                    c.display_defaults_omitted(),
                    c.sampling_removed,
                    c.thinking_always_on(),
                ),
                (adaptive, omitted, sampling_removed, always_on),
                "{model}"
            );
        }
    }

    fn probe_request(model: &str) -> AnthropicRequest {
        AnthropicRequest {
            context_limit: None,
            model: model.to_string(),
            messages: vec![],
            max_tokens: 512,
            temperature: Some(0.7),
            system: None,
            tools: None,
            stream: None,
            thinking: None,
            cache_control: None,
        }
    }

    /// The defect the boundary check exists for: `CompletionRequest::default()`
    /// carries `temperature: Some(0.7)`, so sub-agent questions, multi-agent
    /// decomposition, dream summaries and `AgentContext::compress` all inherit
    /// a temperature nobody chose — a 400 on the current generation.
    #[test]
    fn the_boundary_strips_an_inherited_temperature() {
        let req = probe_request("claude-opus-5");
        let conformed = conform_to_anthropic_contract(&req);
        assert_eq!(conformed.temperature, None);
    }

    /// Omitting `thinking` used to mean "no thinking" everywhere, which is
    /// what every `thinking: None` in this codebase was written to mean. On
    /// Opus 5 it now means adaptive, so a 512-token auxiliary request would
    /// spend its whole budget reasoning and return empty text.
    #[test]
    fn the_boundary_makes_an_implied_mute_explicit() {
        let req = probe_request("claude-opus-5");
        let conformed = conform_to_anthropic_contract(&req);
        assert_eq!(conformed.thinking, Some(ThinkingConfig::Disabled));
    }

    #[test]
    fn the_boundary_leaves_always_on_models_muted_by_omission() {
        // Fable rejects an explicit disable, so `None` has to stay `None`.
        let req = probe_request("claude-fable-5");
        let conformed = conform_to_anthropic_contract(&req);
        assert_eq!(conformed.thinking, None);
        assert_eq!(conformed.temperature, None);
    }

    #[test]
    fn the_boundary_leaves_legacy_and_non_claude_requests_alone() {
        // Legacy: omission already means off, and sampling still exists.
        let legacy_req = probe_request("claude-sonnet-4-20250514");
        let legacy = conform_to_anthropic_contract(&legacy_req);
        assert_eq!(legacy.thinking, None);
        assert_eq!(legacy.temperature, Some(0.7));
        assert!(
            matches!(legacy, std::borrow::Cow::Borrowed(_)),
            "a conforming request must not be cloned"
        );

        // A local model must keep its temperature — Ollama substitutes its
        // own default when the field is absent.
        let ollama_req = probe_request("qwen3.5:9b");
        let ollama = conform_to_anthropic_contract(&ollama_req);
        assert_eq!(ollama.temperature, Some(0.7));
        assert_eq!(ollama.thinking, None);
        assert!(matches!(ollama, std::borrow::Cow::Borrowed(_)));
    }

    #[test]
    fn the_boundary_does_not_disturb_a_request_that_already_thinks() {
        let mut req = probe_request("claude-opus-5");
        req.thinking = Some(ThinkingConfig::adaptive_summarized());
        let conformed = conform_to_anthropic_contract(&req);
        assert_eq!(
            conformed.thinking,
            Some(ThinkingConfig::adaptive_summarized())
        );
        assert_eq!(conformed.temperature, None);
    }

    /// The routing case: a request is built against the configured model, then
    /// `route_model` replaces `request.model` with a different generation. The
    /// boundary must rewrite a now-illegal shape, not just fill in a missing
    /// one, or a `budget_tokens` body reaches a model that rejects it.
    #[test]
    fn the_boundary_rewrites_a_shape_left_over_from_another_model() {
        let mut legacy_shape_on_current = probe_request("claude-opus-5");
        legacy_shape_on_current.thinking = Some(ThinkingConfig::enabled(4096));
        let conformed = conform_to_anthropic_contract(&legacy_shape_on_current);
        assert_eq!(
            conformed.thinking,
            Some(ThinkingConfig::adaptive_summarized()),
            "budget_tokens is a 400 on Opus 5 — the shape must be replaced"
        );
        assert_eq!(conformed.temperature, None);

        // And the mirror direction: adaptive left over on a pre-4.6 model.
        // This layer cannot invent an honest budget, so it drops the field —
        // valid, since omission means no thinking there.
        let mut current_shape_on_legacy = probe_request("claude-sonnet-4-20250514");
        current_shape_on_legacy.thinking = Some(ThinkingConfig::adaptive_summarized());
        let conformed = conform_to_anthropic_contract(&current_shape_on_legacy);
        assert_eq!(conformed.thinking, None);
    }

    #[test]
    fn the_boundary_drops_an_explicit_disable_that_the_model_rejects() {
        let mut req = probe_request("claude-fable-5");
        req.thinking = Some(ThinkingConfig::Disabled);
        let conformed = conform_to_anthropic_contract(&req);
        assert_eq!(
            conformed.thinking, None,
            "Fable rejects an explicit disable; omission is the only legal form"
        );
    }

    /// The Models API payload, spelled the way the API actually spells it.
    /// Reading the wrong field names is silent — it defaults to the
    /// unknown-model floor and caches that — so this pins the mapping.
    #[test]
    fn the_models_api_payload_is_read_with_the_right_field_names() {
        let payload = serde_json::json!({
            "id": "claude-opus-5",
            "type": "model",
            "display_name": "Claude Opus 5",
            "created_at": "2026-01-01T00:00:00Z",
            "max_input_tokens": 1_000_000,
            "max_tokens": 128_000,
        });
        let parsed: super::AnthropicModelResponse =
            serde_json::from_value(payload).expect("deserializes");
        assert_eq!(parsed.id, "claude-opus-5");
        assert_eq!(
            parsed.max_input_tokens,
            Some(1_000_000),
            "context window must come from max_input_tokens; there is no \
             context_window field on the Models API"
        );
        assert_eq!(parsed.max_tokens, Some(128_000));
    }

    #[test]
    fn the_older_field_spellings_still_parse() {
        let payload = serde_json::json!({
            "id": "claude-opus-5",
            "context_window": 200_000,
            "max_output_tokens": 64_000,
        });
        let parsed: super::AnthropicModelResponse =
            serde_json::from_value(payload).expect("deserializes");
        assert_eq!(parsed.max_input_tokens, Some(200_000));
        assert_eq!(parsed.max_tokens, Some(64_000));
    }

    #[test]
    fn the_wire_shapes_serialize_as_the_api_spells_them() {
        let adaptive = serde_json::to_value(ThinkingConfig::Adaptive {
            display: Some(ThinkingDisplay::Summarized),
        })
        .expect("serializes");
        assert_eq!(
            adaptive,
            serde_json::json!({"type": "adaptive", "display": "summarized"})
        );

        let bare = serde_json::to_value(ThinkingConfig::adaptive()).expect("serializes");
        assert_eq!(
            bare,
            serde_json::json!({"type": "adaptive"}),
            "an unset display must be absent, not null"
        );

        let budgeted = serde_json::to_value(ThinkingConfig::enabled(4096)).expect("serializes");
        assert_eq!(
            budgeted,
            serde_json::json!({"type": "enabled", "budget_tokens": 4096})
        );

        let off = serde_json::to_value(ThinkingConfig::Disabled).expect("serializes");
        assert_eq!(off, serde_json::json!({"type": "disabled"}));
    }
}

#[cfg(test)]
mod utf8_stream_decoder_tests {
    use super::Utf8StreamDecoder;

    /// Every split point of a multibyte string decodes to the same text.
    #[test]
    fn a_character_split_across_chunks_is_reassembled() {
        let text = "Ünïcödé 🌙 月";
        let bytes = text.as_bytes();
        for split in 0..=bytes.len() {
            let mut decoder = Utf8StreamDecoder::default();
            let mut out = decoder
                .push(&bytes[..split])
                .expect("a prefix is incomplete, not invalid");
            out.push_str(
                &decoder
                    .push(&bytes[split..])
                    .expect("the rest completes it"),
            );
            assert_eq!(out, text, "split at byte {split}");
        }
    }

    #[test]
    fn invalid_bytes_are_still_rejected() {
        let mut decoder = Utf8StreamDecoder::default();
        assert!(
            decoder.push(&[b'o', b'k', 0xFF, b'!']).is_err(),
            "0xFF is never UTF-8"
        );
        let mut decoder = Utf8StreamDecoder::default();
        assert_eq!(decoder.push(&[0xF0, 0x9F]).expect("incomplete"), "");
        assert!(
            decoder.push(b"x").is_err(),
            "a sequence broken by an ASCII byte is invalid, not incomplete"
        );
    }
}

#[cfg(test)]
mod wire_image_tests {
    use super::{ImageSource, ToolArgsWire, wire_user_message};

    fn png() -> ImageSource {
        ImageSource::Base64 {
            media_type: "image/png".to_string(),
            data: "AAAA".to_string(),
        }
    }

    #[test]
    fn ollama_gets_bare_base64_beside_the_text() {
        let message = wire_user_message("look", &[&png()], ToolArgsWire::JsonObject);
        assert_eq!(message["content"], "look");
        assert_eq!(message["images"], serde_json::json!(["AAAA"]));
        let url = ImageSource::Url {
            url: "https://x/y.png".to_string(),
        };
        let with_url = wire_user_message("look", &[&url], ToolArgsWire::JsonObject);
        assert_eq!(
            with_url["images"],
            serde_json::json!([]),
            "Ollama takes no URLs"
        );
        assert!(
            with_url["content"]
                .as_str()
                .is_some_and(|c| c.contains("https://x/y.png")),
            "a URL image is named, not silently dropped: {with_url}"
        );
    }

    #[test]
    fn openai_gets_text_then_image_parts() {
        let message = wire_user_message("look", &[&png()], ToolArgsWire::JsonString);
        let parts = message["content"].as_array().expect("content parts");
        assert_eq!(
            parts[0],
            serde_json::json!({ "type": "text", "text": "look" })
        );
        assert_eq!(parts[1]["image_url"]["url"], "data:image/png;base64,AAAA");
    }

    #[test]
    fn a_message_without_images_is_unchanged() {
        let message = wire_user_message("hi", &[], ToolArgsWire::JsonString);
        assert_eq!(
            message,
            serde_json::json!({ "role": "user", "content": "hi" })
        );
    }
}

#[cfg(test)]
mod wire_tool_result_tests {
    use super::wire_tool_result_content;

    #[test]
    fn an_error_result_is_announced_exactly_once() {
        assert_eq!(wire_tool_result_content("boom", true), "Error: boom");
        assert_eq!(
            wire_tool_result_content("Error: Tool not found: x", true),
            "Error: Tool not found: x",
            "already announced: no second prefix"
        );
        assert_eq!(
            wire_tool_result_content("error: lower", true),
            "error: lower"
        );
        assert_eq!(
            wire_tool_result_content("Err", true),
            "Error: Err",
            "too short to announce"
        );
        assert_eq!(
            wire_tool_result_content("Error: fine", false),
            "Error: fine"
        );
        assert_eq!(wire_tool_result_content("ok", false), "ok");
    }
}

#[cfg(test)]
mod inline_think_tests {
    use super::{ContentSegment, InlineThinkSplitter, OllamaStreamState, StreamEvent};

    fn split(fragments: &[&str]) -> Vec<ContentSegment> {
        let mut splitter = InlineThinkSplitter::default();
        let mut out = Vec::new();
        for fragment in fragments {
            splitter.push(fragment, &mut out);
        }
        splitter.finish(&mut out);
        out
    }

    /// Concatenate adjacent segments of the same kind — fragmentation is
    /// an artifact of the stream, not of the routing.
    fn joined(segments: Vec<ContentSegment>) -> Vec<ContentSegment> {
        let mut out: Vec<ContentSegment> = Vec::new();
        for segment in segments {
            match (out.last_mut(), segment) {
                (Some(ContentSegment::Text(a)), ContentSegment::Text(b))
                | (Some(ContentSegment::Thinking(a)), ContentSegment::Thinking(b)) => {
                    a.push_str(&b);
                }
                (_, segment) => out.push(segment),
            }
        }
        out
    }

    #[test]
    fn text_without_tags_passes_through_unchanged() {
        assert_eq!(
            joined(split(&["Hello ", "there, 1 < 2."])),
            vec![ContentSegment::Text("Hello there, 1 < 2.".to_string())]
        );
    }

    #[test]
    fn a_think_block_split_across_fragments_is_routed_to_thinking() {
        assert_eq!(
            joined(split(&[
                "<thi",
                "nk>let me ",
                "think</th",
                "ink>\n\nHello",
                " there!"
            ])),
            vec![
                ContentSegment::Thinking("let me think".to_string()),
                ContentSegment::Text("Hello there!".to_string()),
            ],
            "tags split mid-token are still tags, and the separator after one is not reply"
        );
    }

    #[test]
    fn a_held_back_prefix_that_is_not_a_tag_is_released() {
        assert_eq!(
            joined(split(&["a <th", "ing> b <"])),
            vec![ContentSegment::Text("a <thing> b <".to_string())]
        );
    }

    #[test]
    fn an_unclosed_think_block_never_reaches_the_reply() {
        assert_eq!(
            joined(split(&["<think>still going", " and going"])),
            vec![ContentSegment::Thinking(
                "still going and going".to_string()
            )]
        );
    }

    /// Synthesized tool-use ids never repeat: not within one response, and
    /// not across responses — the per-response counter they replaced gave the
    /// first call of every response the same id.
    #[test]
    fn synthesized_tool_use_ids_are_unique_across_responses() {
        let call = serde_json::json!({
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    { "function": { "name": "a", "arguments": {} } },
                    { "function": { "name": "b", "arguments": {} } },
                ],
            },
            "done": true,
        });
        let mut ids = Vec::new();
        for _response in 0..3 {
            let mut state = OllamaStreamState::default();
            let (items, _) = state.on_object(&call, 0);
            ids.extend(items.into_iter().filter_map(|item| match item {
                Ok(StreamEvent::ContentBlockStart {
                    tool_id: Some(id), ..
                }) => Some(id),
                _ => None,
            }));
        }
        assert_eq!(ids.len(), 6, "two calls per response, three responses");
        let unique: std::collections::HashSet<&String> = ids.iter().collect();
        assert_eq!(unique.len(), ids.len(), "ids repeat: {ids:?}");
    }

    /// A last line without its newline is translated in full: its text and
    /// tool call reach the reply, and the turn stops as a tool turn.
    #[test]
    fn an_unterminated_final_line_keeps_what_it_carries() {
        let mut state = OllamaStreamState::default();
        let first = serde_json::json!({ "message": { "content": "Hello" }, "done": false });
        let (mut items, _) = state.on_object(&first, 0);
        let last = r#"{"message":{"content":" world","tool_calls":[{"function":{"name":"read","arguments":{}}}]},"done":true,"done_reason":"stop"}"#;
        items.extend(state.finish(last, None, 0, 1));
        let events: Vec<StreamEvent> = items.into_iter().map(|i| i.expect("no error")).collect();
        let text: String = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::TextDelta { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(text, "Hello world", "{events:?}");
        assert!(
            events.iter().any(|e| matches!(e, StreamEvent::ContentBlockStart { tool_name: Some(n), .. } if n == "read")),
            "{events:?}"
        );
        assert!(
            matches!(events.last(), Some(StreamEvent::MessageStop { stop_reason }) if stop_reason == "tool_use"),
            "{events:?}"
        );
    }

/// Through the NDJSON translator: the reply's text deltas carry no tag
    /// and no reasoning, and the reasoning arrives as thinking.
    #[test]
    fn the_stream_translator_separates_inline_reasoning() {
        let mut state = OllamaStreamState::default();
        let mut events = Vec::new();
        for (content, done) in [
            ("<think>weigh it", false),
            ("</think>Hi!", false),
            ("", true),
        ] {
            let obj = serde_json::json!({
                "message": { "role": "assistant", "content": content },
                "done": done,
            });
            let (items, _) = state.on_object(&obj, 0);
            events.extend(items.into_iter().map(|item| item.expect("no stream error")));
        }
        let text: String = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::TextDelta { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        let thinking: String = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ThinkingDelta { thinking, .. } => Some(thinking.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(text, "Hi!");
        assert_eq!(thinking, "weigh it");
    }
}

#[cfg(test)]
mod gemma_sentinel_tests {
    use super::{pace_provider_requests, LlmClient, Provider};

    /// The exact payload that cost a benchmark run: gemma4:12b emitted
    /// `<unused50>` as assistant content and Ollama closed the stream with no
    /// `done=true`, 57 times, each discarding a complete reply.
    #[test]
    fn recognises_gemma_stop_sentinels() {
        for sentinel in ["<unused50>", "<unused0>", "<unused98>", " <unused7> "] {
            assert!(
                LlmClient::is_gemma_stop_sentinel(sentinel),
                "must recognise {sentinel} as a stop marker"
            );
        }
    }

    /// The sentinel glued to the end of real text — the form an exact-match
    /// test silently misses, which is why the first version of this fix did
    /// not stop the 502s.
    #[test]
    fn strips_a_trailing_sentinel_and_keeps_the_words() {
        assert_eq!(
            LlmClient::strip_trailing_stop_sentinel("all tests pass.<unused50>"),
            Some("all tests pass.")
        );
        assert_eq!(LlmClient::strip_trailing_stop_sentinel("<unused7>"), Some(""));
        assert_eq!(LlmClient::strip_trailing_stop_sentinel("no marker here"), None);
        assert_eq!(LlmClient::strip_trailing_stop_sentinel("<unused50> leads"), None);
        assert_eq!(LlmClient::strip_trailing_stop_sentinel("<unusedfoo>"), None);
    }

    /// Must stay exact. A reply that DISCUSSES the token is prose, and
    /// swallowing it would silently truncate the model's answer — the same
    /// class of bug as the 502, pointing the other way.
    #[test]
    fn leaves_ordinary_text_alone() {
        for text in [
            "<unused50> is a reserved token",
            "the answer is 42",
            "<unusedfoo>",
            "<unused>",
            "<unused123>",
            "",
            "<think>",
        ] {
            assert!(
                !LlmClient::is_gemma_stop_sentinel(text),
                "must NOT treat {text:?} as a stop marker"
            );
        }
    }

    // =========================================================================
    // OpenAI-compatible completion body decoding (OpenRouter reality)
    // =========================================================================
    // Shapes below are taken from live OpenRouter responses captured 2026-07-31
    // while diagnosing why a stored, WORKING key produced 386 straight
    // "error decoding response body" failures and zero successful dreams.

    /// `OpenRouter` pads a slow reasoning model's response with whitespace
    /// keep-alive before the JSON — observed live on nemotron-3-ultra.
    #[test]
    fn decodes_whitespace_padded_completion() {
        let body = format!(
            "{}{}",
            "\n         \n\n         \n\n         \n",
            r#"{"id":"gen-1","object":"chat.completion","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"OK","refusal":null,"reasoning":"The user wants OK."}}],"usage":{"prompt_tokens":18,"completion_tokens":20}}"#
        );
        assert_eq!(
            LlmClient::parse_openai_completion(200, &body).expect("padded body must decode"),
            "OK"
        );
    }

    /// Reasoning models may return `content: null` with the text in
    /// `reasoning`; an empty-success is worse than using what was sent.
    #[test]
    fn falls_back_to_reasoning_when_content_is_null() {
        let body = r#"{"choices":[{"message":{"role":"assistant","content":null,"reasoning":"Summary: the notes describe minidb progress."}}]}"#;
        assert_eq!(
            LlmClient::parse_openai_completion(200, body).expect("reasoning fallback"),
            "Summary: the notes describe minidb progress."
        );
    }

    /// The expensive one: `OpenRouter` returns HTTP 200 whose body is an error
    /// envelope (free-tier rate limits, moderation). This must surface as an
    /// error carrying the API's message — swallowing it as a decode failure is
    /// what made a working key look absent for two days.
    #[test]
    fn surfaces_error_envelope_inside_http_200() {
        let body = r#"{"error":{"message":"Rate limit exceeded: free-models-per-day","code":429}}"#;
        let err = LlmClient::parse_openai_completion(200, body).expect_err("error body is an error");
        let msg = err.to_string();
        assert!(
            msg.contains("Rate limit exceeded"),
            "the API's own message must survive: {msg}"
        );
    }

    /// A 200-wrapped 429 must CLASSIFY as a rate limit, not merely mention one:
    /// only `LlmError::RateLimit` engages the summarizer's backoff walk, and a
    /// rate limit that classifies as a generic failure gets retried at full
    /// speed — the one response it must never get.
    #[test]
    fn wrapped_rate_limit_classifies_as_rate_limit() {
        // Numeric code, as OpenRouter sends it.
        let body = r#"{"error":{"message":"Rate limit exceeded: free-models-per-day","code":429}}"#;
        let err = LlmClient::parse_openai_completion(200, body).expect_err("must fail");
        assert!(err.is_rate_limit(), "numeric embedded 429 must classify: {err}");

        // String code — some providers quote it.
        let body = r#"{"error":{"message":"slow down","code":"429"}}"#;
        let err = LlmClient::parse_openai_completion(200, body).expect_err("must fail");
        assert!(err.is_rate_limit(), "string embedded 429 must classify: {err}");

        // No code at all, but the human-form message — OpenRouter's spelling
        // has a space, which the old matcher (underscore only) missed.
        let body = r#"{"error":{"message":"Rate limit exceeded, retry later"}}"#;
        let err = LlmClient::parse_openai_completion(200, body).expect_err("must fail");
        assert!(err.is_rate_limit(), "message text must classify: {err}");

        // Negative space: an unrelated error must NOT classify as a rate limit.
        let body = r#"{"error":{"message":"model not found","code":404}}"#;
        let err = LlmClient::parse_openai_completion(200, body).expect_err("must fail");
        assert!(!err.is_rate_limit(), "404 is not a rate limit: {err}");
    }

    /// A body that is neither shape must be reported WITH a snippet of itself.
    /// "error decoding response body" with no body was the whole two-day bug.
    #[test]
    fn undecodable_body_includes_a_snippet() {
        let err = LlmClient::parse_openai_completion(200, ": OPENROUTER PROCESSING")
            .expect_err("garbage is an error");
        assert!(
            err.to_string().contains("OPENROUTER PROCESSING"),
            "snippet missing from: {err}"
        );
    }

    /// A valid completion with no usable text anywhere is a failure, not an
    /// empty-string success that would be stored as a real summary.
    #[test]
    fn empty_completion_is_an_error_not_empty_success() {
        let body = r#"{"choices":[{"message":{"role":"assistant","content":"","reasoning":""}}]}"#;
        assert!(LlmClient::parse_openai_completion(200, body).is_err());
    }

    // =========================================================================
    // Provider request pacing
    // =========================================================================

    /// Back-to-back requests to a paced provider must be held to its derived
    /// spacing — the first working dream cycle fired one request per cluster
    /// in the same second and lost 20 of 22 clusters to 429s.
    #[tokio::test(start_paused = true)]
    async fn paced_provider_requests_are_spaced() {
        let spacing = Provider::OpenRouter
            .min_request_spacing()
            .expect("OpenRouter is paced");
        let t0 = tokio::time::Instant::now();
        pace_provider_requests(Provider::OpenRouter).await;
        pace_provider_requests(Provider::OpenRouter).await;
        pace_provider_requests(Provider::OpenRouter).await;
        assert!(
            t0.elapsed() >= spacing * 2,
            "three consecutive calls must span two intervals, got {:?}",
            t0.elapsed()
        );
    }

    /// Each paced provider has its OWN clock: pacing one must not delay the
    /// other, or the gate reintroduces the cross-caller contention it exists
    /// to remove.
    ///
    /// Measured on `GitHubModels`, which no OTHER test touches — the clocks
    /// are process-global and the paced-spacing test stamps `OpenRouter`'s, so
    /// measuring `OpenRouter` here would depend on test ordering.
    #[tokio::test(start_paused = true)]
    async fn provider_clocks_are_independent() {
        pace_provider_requests(Provider::OpenRouter).await;
        let t0 = tokio::time::Instant::now();
        // GitHub's first-ever call: its own (empty) clock is what matters;
        // OpenRouter's just-stamped clock must not be consulted.
        pace_provider_requests(Provider::GitHubModels).await;
        assert_eq!(
            t0.elapsed(),
            tokio::time::Duration::ZERO,
            "a different provider's clock must not impose a wait"
        );
    }

    /// Providers without a published quota are never stalled: local Ollama's
    /// only quota is the GPU, and tier-dependent clouds are handled reactively.
    #[tokio::test(start_paused = true)]
    async fn unpaced_providers_are_never_stalled() {
        let t0 = tokio::time::Instant::now();
        for _ in 0..5 {
            pace_provider_requests(Provider::Ollama).await;
            pace_provider_requests(Provider::Anthropic).await;
            pace_provider_requests(Provider::ClaudeProxy).await;
        }
        assert_eq!(t0.elapsed(), tokio::time::Duration::ZERO);
    }

    // =========================================================================
    // Dynamic pacing policy (PaceState::required_wait is pure — no clocks)
    // =========================================================================

    use super::PaceState;
    use tokio::time::{Duration, Instant};

    /// A generous provider-reported budget must run FASTER than the static
    /// prior — dynamic pacing is not just a safety brake.
    #[tokio::test(start_paused = true)]
    async fn ample_reported_budget_beats_the_static_prior() {
        let now = Instant::now();
        let state = PaceState {
            last_request: Some(now),
            remaining: Some(100),
            reset_at: Some(now + Duration::from_secs(60)),
        };
        let wait = state.required_wait(now, Duration::from_secs(3));
        assert!(
            wait <= Duration::from_millis(600),
            "100 requests over 60s is 600ms spacing, got {wait:?}"
        );
    }

    /// A scarce budget spreads evenly across the window instead of burning out
    /// early: 5 left with 50s to go means one every 10s, prior be damned.
    #[tokio::test(start_paused = true)]
    async fn scarce_reported_budget_stretches_beyond_the_prior() {
        let now = Instant::now();
        let state = PaceState {
            last_request: Some(now),
            remaining: Some(5),
            reset_at: Some(now + Duration::from_secs(50)),
        };
        let wait = state.required_wait(now, Duration::from_secs(3));
        assert_eq!(wait, Duration::from_secs(10));
    }

    /// An exhausted budget waits out the window — the provider said no.
    #[tokio::test(start_paused = true)]
    async fn exhausted_budget_waits_for_the_reset() {
        let now = Instant::now();
        let state = PaceState {
            last_request: Some(now),
            remaining: Some(0),
            reset_at: Some(now + Duration::from_secs(17)),
        };
        assert_eq!(
            state.required_wait(now, Duration::from_secs(3)),
            Duration::from_secs(17)
        );
    }

    /// A lapsed window means a refilled budget: no wait, even though the stale
    /// `remaining` still reads zero.
    #[tokio::test(start_paused = true)]
    async fn lapsed_window_means_refilled_budget() {
        let now = Instant::now();
        let state = PaceState {
            last_request: Some(now),
            remaining: Some(0),
            reset_at: Some(now - Duration::from_secs(1)),
        };
        assert_eq!(state.required_wait(now, Duration::from_secs(3)), Duration::ZERO);
    }

    /// Before any response has taught us anything, the published prior rules.
    #[tokio::test(start_paused = true)]
    async fn unlearned_state_uses_the_prior() {
        let now = Instant::now();
        let state = PaceState {
            last_request: Some(now),
            remaining: None,
            reset_at: None,
        };
        assert_eq!(
            state.required_wait(now, Duration::from_secs(3)),
            Duration::from_secs(3)
        );
        // …and a cold start (no previous request at all) never waits.
        assert_eq!(
            PaceState::default().required_wait(now, Duration::from_secs(3)),
            Duration::ZERO
        );
    }

    /// Elapsed time since the previous request counts against the spacing —
    /// the gate enforces spacing, it does not add latency on top of it.
    #[tokio::test(start_paused = true)]
    async fn elapsed_time_counts_toward_the_spacing() {
        let now = Instant::now();
        let state = PaceState {
            last_request: Some(now - Duration::from_secs(2)),
            remaining: None,
            reset_at: None,
        };
        assert_eq!(
            state.required_wait(now, Duration::from_secs(3)),
            Duration::from_secs(1)
        );
    }
}

#[cfg(test)]
mod input_overflow_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    // ------------------------------------------------------------------
    // Classification: only the overflow shape, on input-fault statuses
    // ------------------------------------------------------------------

    #[test]
    fn overflow_limits_parse_the_ollama_message() {
        let parsed =
            input_overflow_limits("input length 1234 exceeds maximum context length 512");
        assert_eq!(parsed, Some((1234, 512)), "sent and limit, in that order");
    }

    /// Version digits inside a model name must not be mistaken for the pair —
    /// the pair is the first adjacent DESCENDING pair, and "v1.5" ascends.
    #[test]
    fn overflow_limits_step_over_model_version_digits() {
        let parsed = input_overflow_limits(
            "model 'nomic-embed-text-v1.5': input length 8000 exceeds the maximum of 2048 tokens",
        );
        assert_eq!(parsed, Some((8000, 2048)));
    }

    #[test]
    fn overflow_limits_reject_other_vocabulary_and_shapes() {
        // Right words, no numbers.
        assert_eq!(
            input_overflow_limits("input length exceeds maximum context length"),
            None
        );
        // Numbers, wrong vocabulary — a quota message is not an input fault.
        assert_eq!(
            input_overflow_limits("quota of 200 requests used, 50 remaining"),
            None
        );
        // Ascending numbers only — nothing states sent > limit.
        assert_eq!(
            input_overflow_limits("length 10 exceeds nothing; maximum is 512"),
            None
        );
    }

    #[test]
    fn input_overflow_is_scoped_to_input_fault_statuses() {
        let message = "input length 1000 exceeds maximum context length 500".to_string();
        for status in [400_u16, 413, 422] {
            let e = LlmError::Api {
                status,
                message: message.clone(),
            };
            assert!(e.is_input_overflow(), "{status} with the shape classifies");
        }
        // A 500 with the same words is a provider fault; benching stays correct.
        assert!(
            !LlmError::Api {
                status: 500,
                message: message.clone()
            }
            .is_input_overflow()
        );
        // Rate limits never classify as input overflow, whatever they say.
        assert!(
            !LlmError::RateLimit {
                message,
                retry_after: None
            }
            .is_input_overflow()
        );
    }

    // ------------------------------------------------------------------
    // Shrinking: strictly shorter, proportional, refuses the degenerate
    // ------------------------------------------------------------------

    #[test]
    fn shrink_cuts_proportionally_and_strictly_shorter() {
        let err = LlmError::Api {
            status: 422,
            message: "input length 100 exceeds maximum context length 50".into(),
        };
        let text = "x".repeat(200);
        let cut = shrink_to_input_limit(&err, &text).expect("a healable overflow shrinks");
        assert_eq!(cut.chars().count(), 100, "200 chars x 50/100 = 100");
        assert!(
            cut.chars().count() < text.chars().count(),
            "strictly shorter - termination"
        );
    }

    #[test]
    fn shrink_refuses_when_nothing_is_left_to_cut() {
        let err = LlmError::Api {
            status: 422,
            message: "input length 1000 exceeds maximum context length 1".into(),
        };
        // 1 char x 1/1000 floors to zero — unhealable, surface the error.
        assert_eq!(shrink_to_input_limit(&err, "x"), None);
        // Not the fault at all → untouched.
        let other = LlmError::Api {
            status: 500,
            message: "boom".into(),
        };
        assert_eq!(shrink_to_input_limit(&other, "some text"), None);
    }

    #[test]
    fn shrink_respects_char_boundaries() {
        let err = LlmError::Api {
            status: 422,
            message: "input length 4 exceeds maximum context length 2".into(),
        };
        // Multi-byte chars: a byte-indexed cut would panic or split a char.
        let cut = shrink_to_input_limit(&err, "éééé").expect("shrinks");
        assert_eq!(cut, "éé");
    }

    // ------------------------------------------------------------------
    // The healing loop, end to end against a live socket
    // ------------------------------------------------------------------

    /// Serve scripted (status, body) responses in order, then repeat the last
    /// one; `Connection: close` makes the accept count the request count.
    async fn spawn_scripted_server(
        script: Vec<(u16, String)>,
    ) -> (String, Arc<AtomicUsize>, Arc<Mutex<Vec<usize>>>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind an ephemeral port");
        let addr = listener.local_addr().expect("read back the bound addr");
        let hits = Arc::new(AtomicUsize::new(0));
        let body_lens = Arc::new(Mutex::new(Vec::new()));
        let counter = hits.clone();
        let lens = body_lens.clone();
        tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    break;
                };
                let n = counter.fetch_add(1, Ordering::SeqCst);
                let mut buf = vec![0_u8; 65536];
                let read = stream.read(&mut buf).await.unwrap_or(0);
                lens.lock().expect("lens lock").push(read);
                let (status, body) = script
                    .get(n)
                    .unwrap_or_else(|| script.last().expect("script is non-empty"));
                let response = format!(
                    "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len(),
                );
                let _ = stream.write_all(response.as_bytes()).await;
                let _ = stream.flush().await;
            }
        });
        (format!("http://{addr}"), hits, body_lens)
    }

    /// The incident's fix: one overflow report, one shrink, one clean retry —
    /// the caller gets a vector and the provider never looks unhealthy.
    #[tokio::test]
    async fn embed_one_shrinks_and_retries_after_an_overflow_report() {
        let (url, hits, body_lens) = spawn_scripted_server(vec![
            (
                422,
                r#"{"error":{"message":"input length 100 exceeds maximum context length 50"}}"#
                    .to_string(),
            ),
            (200, r#"{"data":[{"embedding":[0.1,0.2,0.3]}]}"#.to_string()),
        ])
        .await;
        let client = EmbeddingClient::openai("test-key")
            .with_model("test-embed")
            .with_base_url(&url);

        let text = "word ".repeat(100);
        let vector = client.embed_one(&text).await.expect("healed by shrinking");
        assert_eq!(vector.len(), 3, "the retry's vector comes through");
        assert_eq!(hits.load(Ordering::SeqCst), 2, "exactly one shrink retry");
        let lens = body_lens.lock().expect("lens lock");
        assert!(
            lens[1] < lens[0],
            "the retry carried a strictly smaller request ({} then {})",
            lens[0],
            lens[1]
        );
        drop(lens);
    }

    /// A provider that keeps reporting overflow whatever we send: the strict
    /// shrink guarantees termination — the loop bottoms out and surfaces the
    /// error instead of spinning forever.
    #[tokio::test]
    async fn embed_one_terminates_against_a_provider_that_always_overflows() {
        let (url, hits, _lens) = spawn_scripted_server(vec![(
            422,
            r#"{"error":{"message":"input length 100 exceeds maximum context length 50"}}"#
                .to_string(),
        )])
        .await;
        let client = EmbeddingClient::openai("test-key")
            .with_model("test-embed")
            .with_base_url(&url);

        let text = "x".repeat(64);
        let err = client
            .embed_one(&text)
            .await
            .expect_err("an unhealable overflow surfaces");
        assert!(
            err.is_input_overflow(),
            "the surfaced error keeps its class: {err}"
        );
        // Halving 64 chars bottoms out in ~7 passes; the point is that the
        // count is small and finite, not any particular number.
        let n = hits.load(Ordering::SeqCst);
        assert!((2..=8).contains(&n), "strictly-shrinking retries, got {n}");
    }

    // ------------------------------------------------------------------
    // The 2026-08-02 treatment applied to the embedding client
    // ------------------------------------------------------------------

    #[test]
    fn ollama_embedding_client_pins_v4_loopback() {
        let client = EmbeddingClient::ollama("http://localhost:11434");
        assert_eq!(client.base_url, "http://127.0.0.1:11434");
        // The pin survives a post-construction URL override…
        let client = client.with_base_url("http://localhost:9999");
        assert_eq!(client.base_url, "http://127.0.0.1:9999");
        // …and remote hosts pass through untouched.
        let client = EmbeddingClient::ollama("http://gpu-box:11434");
        assert_eq!(client.base_url, "http://gpu-box:11434");
    }

    #[test]
    fn only_the_host_localhost_is_pinned() {
        // A host that merely starts with "localhost" is another machine: the
        // client must connect to the name the address gives (and the bearer
        // token was bound to), not to `127.0.0.1.example.com`.
        for url in [
            "http://localhost.example.com:11434",
            "http://localhostfoo:11434/ollama",
        ] {
            assert_eq!(EmbeddingClient::ollama(url).base_url, url);
            assert_eq!(LlmClient::ollama(url).base_url, url);
        }
        // The name itself still is, however it is written.
        assert_eq!(
            LlmClient::ollama("http://LOCALHOST:11434/ollama/").base_url,
            "http://127.0.0.1:11434/ollama"
        );
    }

    #[test]
    fn openai_embedding_client_urls_are_not_rewritten() {
        // "localhost" in an OpenAI-compatible URL is someone's proxy, not our
        // Ollama — the pin is an Ollama-path decision only.
        let client = EmbeddingClient::openai("k").with_base_url("http://localhost:8080");
        assert_eq!(client.base_url, "http://localhost:8080");
    }
}

#[cfg(test)]
mod embedding_bearer_tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// Answer every request with one embedding, recording each request's
    /// `Authorization` header (`None` when it carried none). `Connection:
    /// close` makes every request its own connection.
    async fn spawn_recording_server() -> (String, Arc<Mutex<Vec<Option<String>>>>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind an ephemeral port");
        let addr = listener.local_addr().expect("read back the bound addr");
        let seen = Arc::new(Mutex::new(Vec::new()));
        let record = seen.clone();
        tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    break;
                };
                // The whole head, however many reads it takes to arrive.
                let mut head = Vec::new();
                let mut chunk = [0_u8; 4096];
                while !head.windows(4).any(|w| w == b"\r\n\r\n") {
                    match stream.read(&mut chunk).await {
                        Ok(0) | Err(_) => break,
                        Ok(n) => head.extend_from_slice(&chunk[..n]),
                    }
                }
                let head = String::from_utf8_lossy(&head).to_string();
                let authorization = head.lines().find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("authorization")
                        .then(|| value.trim().to_string())
                });
                record.lock().expect("record lock").push(authorization);
                let body = r#"{"embeddings":[[0.1,0.2,0.3]]}"#;
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len(),
                );
                let _ = stream.write_all(response.as_bytes()).await;
                let _ = stream.flush().await;
            }
        });
        (format!("http://{addr}"), seen)
    }

    /// An embedding client lives as long as the daemon; the token it sends is
    /// whatever its [`SharedBearer`] holds when each request is built — so a
    /// token saved after boot reaches the very next request, and clearing it
    /// stops the header rather than sending `Bearer `.
    #[tokio::test]
    async fn an_ollama_embedder_sends_the_token_its_bearer_holds_now() {
        let (url, seen) = spawn_recording_server().await;
        let bearer = SharedBearer::new("old-token");
        let client = EmbeddingClient::ollama(&url)
            .with_bearer(bearer.clone())
            .with_model("test-embed");

        client.embed_one("first").await.expect("embeds");
        assert!(
            bearer.replace(" new-token "),
            "a different token is a change"
        );
        assert!(
            !bearer.replace("new-token"),
            "the same token, trimmed, is not"
        );
        client.embed_one("second").await.expect("embeds");
        assert!(bearer.replace("   "), "clearing the token is a change");
        client.embed_one("third").await.expect("embeds");

        assert_eq!(
            *seen.lock().expect("record lock"),
            vec![
                Some("Bearer old-token".to_string()),
                Some("Bearer new-token".to_string()),
                None,
            ]
        );
    }

    /// The generation is what lets a holder notice "the credential changed"
    /// without keeping a copy of the token: it moves on every change and only
    /// then.
    #[test]
    fn the_generation_moves_only_when_the_token_changes() {
        let bearer = SharedBearer::new("a");
        let client = EmbeddingClient::ollama("http://127.0.0.1:1").with_bearer(bearer.clone());
        let start = client.credential_generation();
        assert!(!bearer.replace(" a "));
        assert_eq!(client.credential_generation(), start, "an unchanged token");
        assert!(bearer.replace("b"));
        assert_eq!(
            client.credential_generation(),
            start + 1,
            "one change, one step"
        );
        assert_eq!(bearer.token(), "b");
    }
}

#[cfg(test)]
mod ollama_generation_slot_tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// Answer `/api/chat` with a short assistant reply and count the chats;
    /// every other path (the model-info lookups) is a 404.
    async fn spawn_counting_ollama() -> (String, Arc<AtomicUsize>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind an ephemeral port");
        let addr = listener.local_addr().expect("read back the bound addr");
        let chats = Arc::new(AtomicUsize::new(0));
        let counter = chats.clone();
        tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    break;
                };
                let mut head = Vec::new();
                let mut chunk = [0_u8; 4096];
                while !head.windows(4).any(|w| w == b"\r\n\r\n") {
                    match stream.read(&mut chunk).await {
                        Ok(0) | Err(_) => break,
                        Ok(n) => head.extend_from_slice(&chunk[..n]),
                    }
                }
                let is_chat = String::from_utf8_lossy(&head).starts_with("POST /api/chat ");
                let (status, body) = if is_chat {
                    counter.fetch_add(1, Ordering::SeqCst);
                    (
                        "200 OK",
                        r#"{"message":{"role":"assistant","content":"done"},"done":true,"done_reason":"stop"}"#,
                    )
                } else {
                    ("404 Not Found", r#"{"error":"not found"}"#)
                };
                let response = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len(),
                );
                let _ = stream.write_all(response.as_bytes()).await;
                let _ = stream.flush().await;
            }
        });
        (format!("http://{addr}"), chats)
    }

    /// The Anthropic-shaped completion — every in-loop summarizer's call —
    /// takes the server's one generation slot like chat's stream and the
    /// plain completion do. Without it, a summary for one session fired into
    /// another session's live stream on the same server, and the server
    /// cancelled the stream: the self-inflicted failure the slot exists for.
    #[tokio::test(flavor = "multi_thread")]
    async fn an_anthropic_shaped_completion_waits_for_the_generation_slot() {
        let (url, chats) = spawn_counting_ollama().await;
        let client = LlmClient::ollama(&url);
        let request = AnthropicRequest {
            context_limit: Some(4_096),
            model: "slot-test:1b".to_string(),
            messages: vec![AnthropicMessage::user_text("summarize")],
            max_tokens: 16,
            temperature: None,
            system: None,
            tools: None,
            stream: None,
            thinking: None,
            cache_control: None,
        };

        // Another generation on this server holds its slot.
        let held = ollama_generation_slot(&client.base_url)
            .acquire_owned()
            .await
            .expect("the slot is open");
        let call = tokio::spawn(async move { client.complete_anthropic(&request).await });
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        assert_eq!(
            chats.load(Ordering::SeqCst),
            0,
            "no second generation reaches the server while one holds its slot"
        );

        drop(held);
        let response = tokio::time::timeout(std::time::Duration::from_secs(10), call)
            .await
            .expect("the call proceeds once the slot is free")
            .expect("the call task does not panic")
            .expect("the server answers");
        assert_eq!(chats.load(Ordering::SeqCst), 1);
        assert!(!response.content.is_empty(), "{response:?}");
    }
}

#[cfg(test)]
mod cache_ttl_tests {
    use super::*;

    fn cached_request(ttl: CacheTtl) -> AnthropicRequest {
        AnthropicRequest {
            model: "claude-opus-5".to_string(),
            messages: vec![AnthropicMessage::user_text("hello")],
            max_tokens: 128,
            context_limit: None,
            temperature: None,
            system: Some("You are Nanna.".to_string()),
            tools: None,
            stream: None,
            thinking: None,
            cache_control: Some(CacheControl::ephemeral_for(ttl)),
        }
    }

    #[test]
    fn the_default_marker_is_byte_identical_to_the_pre_ttl_wire_format() {
        let json = serde_json::to_string(&CacheControl::ephemeral()).expect("serializes");
        assert_eq!(json, r#"{"type":"ephemeral"}"#);
        assert_eq!(
            CacheControl::ephemeral_for(CacheTtl::FiveMinutes),
            CacheControl::ephemeral()
        );
    }

    #[test]
    fn a_one_hour_marker_sends_the_ttl_and_reads_back() {
        let hour = CacheControl::ephemeral_for(CacheTtl::OneHour);
        let json = serde_json::to_string(&hour).expect("serializes");
        assert_eq!(json, r#"{"type":"ephemeral","ttl":"1h"}"#);

        let parsed: CacheControl = serde_json::from_str(&json).expect("round-trips");
        assert_eq!(parsed.effective_ttl(), CacheTtl::OneHour);
        let bare: CacheControl = serde_json::from_str(r#"{"type":"ephemeral"}"#).expect("parses");
        assert_eq!(
            bare.effective_ttl(),
            CacheTtl::FiveMinutes,
            "absent means the API default"
        );
    }

    #[test]
    fn the_oauth_system_breakpoint_carries_the_request_ttl() {
        let request = cached_request(CacheTtl::OneHour);
        let oauth = OAuthAnthropicRequest::from_request(&request, true);
        assert_eq!(oauth.system.len(), 2, "identity block + the system prompt");
        assert_eq!(
            oauth.system[0].cache_control, None,
            "only the last block is a breakpoint"
        );
        assert_eq!(oauth.system[1].cache_control, request.cache_control);

        // On the wire: both breakpoints say 1h, and no marker falls back to the 5-minute
        // default — a 5m breakpoint ahead of a 1h one is the ordering the API rejects.
        let json = serde_json::to_string(&oauth).expect("serializes");
        assert_eq!(json.matches(r#""cache_control":"#).count(), 2);
        assert_eq!(json.matches(r#""ttl":"1h""#).count(), 2);
    }

    #[test]
    fn a_default_ttl_request_sends_no_ttl_key_anywhere() {
        let oauth =
            OAuthAnthropicRequest::from_request(&cached_request(CacheTtl::FiveMinutes), true);
        let json = serde_json::to_string(&oauth).expect("serializes");
        assert_eq!(json.matches(r#""cache_control":"#).count(), 2);
        assert!(
            !json.contains(r#""ttl""#),
            "the default must not change a byte: {json}"
        );
    }
}

#[cfg(test)]
mod cache_creation_split_tests {
    use super::*;

    #[test]
    fn message_start_surfaces_the_one_hour_share_of_cache_writes() {
        let event = "data: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"model\":\"claude-opus-5\",\"usage\":{\"input_tokens\":5,\"cache_creation_input_tokens\":34,\"cache_creation\":{\"ephemeral_5m_input_tokens\":10,\"ephemeral_1h_input_tokens\":24}}}}";
        match parse_sse_event(event).expect("message_start should parse") {
            StreamEvent::MessageStart {
                cache_creation_tokens,
                cache_creation_1h_tokens,
                ..
            } => {
                assert_eq!(cache_creation_tokens, 34);
                assert_eq!(cache_creation_1h_tokens, 24);
            }
            other => panic!("expected MessageStart, got {other:?}"),
        }
    }

    #[test]
    fn a_usage_without_the_split_counts_as_all_five_minute_writes() {
        let bare = r#"{"input_tokens":1,"output_tokens":2,"cache_creation_input_tokens":40}"#;
        let usage: Usage = serde_json::from_str(bare).expect("parses");
        assert_eq!(usage.cache_creation_1h_tokens(), 0);

        let split = r#"{"input_tokens":1,"output_tokens":2,"cache_creation_input_tokens":40,
            "cache_creation":{"ephemeral_5m_input_tokens":0,"ephemeral_1h_input_tokens":40}}"#;
        let usage: Usage = serde_json::from_str(split).expect("parses");
        assert_eq!(usage.cache_creation_1h_tokens(), 40);
    }

    #[test]
    fn an_inconsistent_split_is_clamped_to_its_total() {
        let json = r#"{"input_tokens":1,"output_tokens":2,"cache_creation_input_tokens":10,
            "cache_creation":{"ephemeral_1h_input_tokens":99}}"#;
        let usage: Usage = serde_json::from_str(json).expect("parses");
        assert_eq!(
            usage.cache_creation_1h_tokens(),
            10,
            "a share never exceeds its total"
        );
    }
}

#[cfg(test)]
mod model_info_cache_tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// `POST /api/show` for a model that publishes its window, and for one
    /// with a smaller window on a different server.
    const SHOW_40K: &str = r#"{"model_info":{"llama.context_length":40960}}"#;
    const SHOW_8K: &str = r#"{"model_info":{"llama.context_length":8192}}"#;
    const UNAUTHORIZED: &str = r#"{"error":"unauthorized"}"#;

    /// A cache in a directory of its own, removed when the test ends.
    struct ScratchCache {
        dir: std::path::PathBuf,
        cache: ModelInfoCache,
    }

    impl ScratchCache {
        fn new(tag: &str) -> Self {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos());
            let dir = std::env::temp_dir().join(format!(
                "nanna-model-info-{tag}-{}-{nanos}",
                std::process::id()
            ));
            Self {
                cache: ModelInfoCache::new(&dir),
                dir,
            }
        }
    }

    impl Drop for ScratchCache {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    /// Answer each connection with the next scripted `(status, body)`, then
    /// repeat the last; `Connection: close` makes the hit count the request
    /// count.
    async fn serve(script: Vec<(u16, &'static str)>) -> (String, Arc<AtomicUsize>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind an ephemeral port");
        let addr = listener.local_addr().expect("read back the bound addr");
        let hits = Arc::new(AtomicUsize::new(0));
        let counter = hits.clone();
        tokio::spawn(async move {
            while let Ok((mut stream, _)) = listener.accept().await {
                let n = counter.fetch_add(1, Ordering::SeqCst);
                let mut request = vec![0_u8; 65536];
                let _ = stream.read(&mut request).await;
                let (status, body) = script
                    .get(n)
                    .unwrap_or_else(|| script.last().expect("script is non-empty"));
                let response = format!(
                    "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\n\
                     Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len(),
                );
                let _ = stream.write_all(response.as_bytes()).await;
                let _ = stream.flush().await;
            }
        });
        (format!("http://{addr}"), hits)
    }

    /// The upgrade story: the previous build never sent the bearer token on
    /// `/api/show`, so a proxied server answered 401 and the floor was cached
    /// for a week. The refusal is not an answer, so it must not outlive the
    /// lookup: the next one (token saved, or the fixed build) asks again.
    #[tokio::test]
    async fn a_refused_lookup_is_not_cached_and_the_next_one_asks_again() {
        let (url, hits) = serve(vec![(401, UNAUTHORIZED), (200, SHOW_40K)]).await;
        let scratch = ScratchCache::new("refused");
        let model = "model-info-cache-refused:7b";
        let client = LlmClient::ollama(&url);

        let refused = client.get_model_info(model, Some(&scratch.cache)).await;
        assert_eq!(refused.context_window, UNKNOWN_CONTEXT_WINDOW);
        assert!(
            scratch.cache.get(model).is_none(),
            "a 401 fallback must not be written to the cache"
        );

        let answered = client.get_model_info(model, Some(&scratch.cache)).await;
        assert_eq!(
            answered.context_window, 40_960,
            "the next lookup must ask the server"
        );
        assert_eq!(hits.load(Ordering::SeqCst), 2);
    }

    /// A transport failure is the same case with no status at all.
    #[tokio::test]
    async fn an_unreachable_server_is_not_cached() {
        let port = {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
            listener.local_addr().expect("addr").port()
        };
        let scratch = ScratchCache::new("unreachable");
        let model = "model-info-cache-unreachable:7b";
        let client = LlmClient::ollama(format!("http://127.0.0.1:{port}"));

        let info = client.get_model_info(model, Some(&scratch.cache)).await;
        assert_eq!(info.context_window, UNKNOWN_CONTEXT_WINDOW);
        assert!(
            scratch.cache.get(model).is_none(),
            "a connect failure must not be cached"
        );
    }

    /// Ollama's 404 says the model is not on that server NOW — `ollama pull`
    /// changes that without anything the cache can see — or, for a proxied
    /// server, that the base URL is missing its path. Neither is a fact about
    /// the model's window.
    #[tokio::test]
    async fn a_model_the_server_does_not_have_yet_is_not_cached() {
        let (url, hits) = serve(vec![
            (
                404,
                r#"{"error":"model 'model-info-cache-unpulled:7b' not found"}"#,
            ),
            (200, SHOW_40K),
        ])
        .await;
        let scratch = ScratchCache::new("unpulled");
        let model = "model-info-cache-unpulled:7b";
        let client = LlmClient::ollama(&url);

        let missing = client.get_model_info(model, Some(&scratch.cache)).await;
        assert_eq!(missing.context_window, UNKNOWN_CONTEXT_WINDOW);
        assert!(
            scratch.cache.get(model).is_none(),
            "a 404 fallback must not be cached"
        );

        let pulled = client.get_model_info(model, Some(&scratch.cache)).await;
        assert_eq!(pulled.context_window, 40_960);
        assert_eq!(hits.load(Ordering::SeqCst), 2);
    }

    /// The cache still does its job for a real answer: one request, then
    /// served from disk — to the client and to the name-only readers.
    #[tokio::test]
    async fn an_answer_is_cached_and_served_without_asking_again() {
        let (url, hits) = serve(vec![(200, SHOW_40K)]).await;
        let scratch = ScratchCache::new("answered");
        let model = "model-info-cache-answered:7b";
        let client = LlmClient::ollama(&url);

        let first = client.get_model_info(model, Some(&scratch.cache)).await;
        let second = client.get_model_info(model, Some(&scratch.cache)).await;
        assert_eq!(
            (first.context_window, second.context_window),
            (40_960, 40_960)
        );
        assert_eq!(
            hits.load(Ordering::SeqCst),
            1,
            "the second lookup is a cache hit"
        );
        assert_eq!(
            scratch.cache.get(model).map(|info| info.context_window),
            Some(40_960),
            "name-only readers see the answer"
        );
    }

    /// Generation 3 is what every build before this fix wrote, poisoned
    /// fallbacks included, each with a fresh timestamp and nothing to tell it
    /// from a real answer. Such a file must never be served.
    #[tokio::test]
    async fn an_entry_from_the_poisoned_generation_is_not_served() {
        let (url, hits) = serve(vec![(200, SHOW_40K)]).await;
        let scratch = ScratchCache::new("generation");
        let model = "model-info-cache-generation:7b";
        std::fs::create_dir_all(&scratch.dir).expect("create the cache dir");
        let poisoned = unknown_model_info(model, "Ollama");
        std::fs::write(
            scratch.dir.join("model-info-cache-generation_7b.v3.json"),
            serde_json::to_string_pretty(&poisoned).expect("serializes"),
        )
        .expect("write the old entry");

        assert!(
            scratch.cache.get(model).is_none(),
            "a generation-3 file is not read"
        );
        let info = LlmClient::ollama(&url)
            .get_model_info(model, Some(&scratch.cache))
            .await;
        assert_eq!(info.context_window, 40_960);
        assert_eq!(
            hits.load(Ordering::SeqCst),
            1,
            "the lookup asked the server"
        );
    }

    /// A model name is not an identity across servers: `qwen3:8b` on a remote
    /// server can be built with a different window than the local one. A
    /// client must never be served another server's answer, and the
    /// name-only readers follow the server that was asked last.
    #[tokio::test]
    async fn an_answer_from_one_server_is_not_served_for_another() {
        let (local, local_hits) = serve(vec![(200, SHOW_40K)]).await;
        let (remote, remote_hits) = serve(vec![(200, SHOW_8K)]).await;
        let scratch = ScratchCache::new("two-servers");
        let model = "model-info-cache-two-servers:8b";

        let from_local = LlmClient::ollama(&local)
            .get_model_info(model, Some(&scratch.cache))
            .await;
        let from_remote = LlmClient::ollama_with_key(&remote, "tok")
            .get_model_info(model, Some(&scratch.cache))
            .await;
        assert_eq!(from_local.context_window, 40_960);
        assert_eq!(
            from_remote.context_window, 8_192,
            "the remote server must be asked"
        );
        assert_eq!(local_hits.load(Ordering::SeqCst), 1);
        assert_eq!(remote_hits.load(Ordering::SeqCst), 1);
        assert_eq!(
            scratch.cache.get(model).map(|info| info.context_window),
            Some(8_192),
            "name-only readers see the server asked last"
        );
    }

    /// When the server now in use cannot answer, the other server's entry
    /// must not stay behind for the name-only readers: they would size budgets
    /// from a window the lookup itself just declined to trust.
    #[tokio::test]
    async fn a_failed_lookup_vacates_another_servers_entry() {
        let (local, _) = serve(vec![(200, SHOW_40K)]).await;
        let (remote, remote_hits) = serve(vec![(401, UNAUTHORIZED)]).await;
        let scratch = ScratchCache::new("vacate");
        let model = "model-info-cache-vacate:8b";

        LlmClient::ollama(&local)
            .get_model_info(model, Some(&scratch.cache))
            .await;
        let refused = LlmClient::ollama(&remote)
            .get_model_info(model, Some(&scratch.cache))
            .await;
        assert_eq!(
            remote_hits.load(Ordering::SeqCst),
            1,
            "the remote server must be asked"
        );
        assert_eq!(refused.context_window, UNKNOWN_CONTEXT_WINDOW);
        assert!(
            scratch.cache.get(model).is_none(),
            "the local answer must not stand in"
        );
    }

    /// The same rule on the other providers' error paths, which used to turn
    /// a failure into an `Ok` floor and cache it.
    #[tokio::test]
    async fn openrouter_and_proxy_failures_are_not_cached() {
        let (url, _) = serve(vec![(500, r#"{"error":"upstream"}"#)]).await;
        let scratch = ScratchCache::new("openrouter");
        let model = "model-info-cache/openrouter";
        let mut openrouter = LlmClient::openrouter("key");
        openrouter.base_url.clone_from(&url);
        let info = openrouter.get_model_info(model, Some(&scratch.cache)).await;
        assert_eq!(info.context_window, UNKNOWN_CONTEXT_WINDOW);
        assert!(
            scratch.cache.get(model).is_none(),
            "an OpenRouter 500 must not be cached"
        );

        let (url, _) = serve(vec![(503, r#"{"error":"starting"}"#)]).await;
        let scratch = ScratchCache::new("proxy");
        let model = "model-info-cache-proxy";
        let info = LlmClient::claude_proxy(&url)
            .get_model_info(model, Some(&scratch.cache))
            .await;
        assert_eq!(info.context_window, UNKNOWN_CONTEXT_WINDOW);
        assert!(
            scratch.cache.get(model).is_none(),
            "a proxy failure must not be cached"
        );
    }

    /// The other two "not found" answers that used to be cached as the floor:
    /// Anthropic's 404, and a model missing from `OpenRouter`'s list.
    #[tokio::test]
    async fn not_found_is_not_cached_on_any_provider() {
        let (url, _) = serve(vec![(404, r#"{"type":"error"}"#)]).await;
        let scratch = ScratchCache::new("anthropic-404");
        let model = "model-info-cache-anthropic-404";
        let mut anthropic = LlmClient::anthropic("key");
        anthropic.base_url.clone_from(&url);
        let info = anthropic.get_model_info(model, Some(&scratch.cache)).await;
        assert_eq!(info.context_window, UNKNOWN_CONTEXT_WINDOW);
        assert!(
            scratch.cache.get(model).is_none(),
            "an Anthropic 404 must not be cached"
        );

        let (url, _) = serve(vec![(200, r#"{"data":[{"id":"someone/else"}]}"#)]).await;
        let scratch = ScratchCache::new("openrouter-unlisted");
        let model = "model-info-cache/unlisted";
        let mut openrouter = LlmClient::openrouter("key");
        openrouter.base_url.clone_from(&url);
        let info = openrouter.get_model_info(model, Some(&scratch.cache)).await;
        assert_eq!(info.context_window, UNKNOWN_CONTEXT_WINDOW);
        assert!(
            scratch.cache.get(model).is_none(),
            "an unlisted model must not be cached"
        );
    }

    /// One server spelled two ways (`localhost` and a trailing slash, as a
    /// pasted URL has them) is one server: the second spelling is a hit.
    #[tokio::test]
    async fn one_server_spelled_two_ways_shares_its_entry() {
        let (url, hits) = serve(vec![(200, SHOW_40K)]).await;
        let port = url.rsplit(':').next().expect("the url has a port");
        let scratch = ScratchCache::new("spellings");
        let model = "model-info-cache-spellings:7b";

        LlmClient::ollama(format!("http://localhost:{port}/"))
            .get_model_info(model, Some(&scratch.cache))
            .await;
        let again = LlmClient::ollama(&url)
            .get_model_info(model, Some(&scratch.cache))
            .await;
        assert_eq!(again.context_window, 40_960);
        assert_eq!(
            hits.load(Ordering::SeqCst),
            1,
            "the second spelling is a cache hit"
        );
    }

    /// The answering server is written to disk with the answer, so a
    /// credential in the base URL must not go with it.
    #[tokio::test]
    async fn the_cached_server_name_carries_no_credentials() {
        let (url, _) = serve(vec![(200, SHOW_40K)]).await;
        let with_credentials = url.replacen("http://", "http://someone:pa55word@", 1);
        let scratch = ScratchCache::new("credentials");
        let model = "model-info-cache-credentials:7b";

        let info = LlmClient::ollama(&with_credentials)
            .get_model_info(model, Some(&scratch.cache))
            .await;
        assert_eq!(info.context_window, 40_960);
        let written = std::fs::read_dir(&scratch.dir)
            .expect("the entry was written")
            .map(|entry| std::fs::read_to_string(entry.expect("dir entry").path()).expect("read"))
            .collect::<String>();
        assert!(written.contains(&url), "the server is recorded: {written}");
        assert!(
            !written.contains("pa55word") && !written.contains("someone"),
            "credentials must not be written: {written}"
        );
    }
}
