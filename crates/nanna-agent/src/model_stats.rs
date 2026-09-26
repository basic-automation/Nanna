//! Model performance statistics tracking.
//!
//! Tracks per-model metrics across requests to inform routing decisions
//! and provide visibility into model behavior:
//! - Response latency (p50, p95, p99)
//! - Token throughput (tokens/sec)
//! - Error rates and downtime detection
//! - Cost tracking (input/output/cache tokens)
//! - Success/failure rates per complexity tier

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, info};
use nanna_storage::StoredModelStats;

use crate::numeric::{millis_u64, f64_from_u64, f64_from_usize};
/// Called with every recorded request.
///
/// It is the per-request history the in-memory aggregates cannot give back (a
/// day's spend is not derivable from lifetime totals). Must not block: the
/// daemon's sink hands the write to a task.
pub type RequestSink = Arc<dyn Fn(&RequestObservation) + Send + Sync>;

/// Global model statistics tracker. Thread-safe, designed for concurrent access.
#[derive(Clone)]
pub struct ModelStatsTracker {
    inner: Arc<RwLock<StatsInner>>,
    /// Set once by the host; shared by every clone.
    sink: Arc<std::sync::OnceLock<RequestSink>>,
}

impl std::fmt::Debug for ModelStatsTracker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelStatsTracker")
            .field("sink_set", &self.sink.get().is_some())
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Default)]
struct StatsInner {
    /// Per-model statistics
    models: HashMap<String, ModelStats>,
}

/// Accumulated statistics for a single model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStats {
    /// Model identifier
    pub model: String,
    /// Total requests made
    pub total_requests: u64,
    /// Successful requests
    pub successful_requests: u64,
    /// Failed requests (errors, timeouts, malformed responses)
    pub failed_requests: u64,
    /// Total input tokens consumed
    pub total_input_tokens: u64,
    /// Total output tokens generated
    pub total_output_tokens: u64,
    /// Total tokens served from cache
    pub total_cache_read_tokens: u64,
    /// Total tokens written to cache
    pub total_cache_creation_tokens: u64,
    /// The 1-hour share of `total_cache_creation_tokens` (billed at 2x input, not 1.25x).
    #[serde(default)]
    pub total_cache_creation_1h_tokens: u64,
    /// Response latencies in milliseconds (ring buffer, last N)
    pub latencies_ms: Vec<u64>,
    /// Tokens per second measurements (ring buffer, last N)
    pub throughput_tps: Vec<f64>,
    /// Consecutive failures (reset on success) - for downtime detection
    pub consecutive_failures: u32,
    /// Timestamp of last successful request
    pub last_success_epoch_ms: u64,
    /// Timestamp of last failure
    pub last_failure_epoch_ms: u64,
    /// Per-tier success counts (for routing quality feedback)
    pub tier_successes: TierCounts,
    /// Per-tier failure counts
    pub tier_failures: TierCounts,
    /// Number of escalations (cheap model failed, had to use more expensive one)
    pub escalations: u64,
}

/// Per-complexity-tier counters.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TierCounts {
    pub simple: u64,
    pub medium: u64,
    pub complex: u64,
}

/// A completed request observation to record.
#[derive(Debug, Clone)]
pub struct RequestObservation {
    pub model: String,
    pub success: bool,
    pub latency: Duration,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cache_read_tokens: u32,
    pub cache_creation_tokens: u32,
    /// The 1-hour share of `cache_creation_tokens`.
    pub cache_creation_1h_tokens: u32,
    /// What complexity tier was this request classified as?
    pub tier: Option<super::loop_runner::TaskComplexity>,
    /// Was this an escalation from a cheaper model?
    pub escalated: bool,
}

/// Summary stats for a model, suitable for UI display.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStatsSummary {
    pub model: String,
    pub total_requests: u64,
    pub success_rate: f64,
    pub avg_latency_ms: u64,
    pub p95_latency_ms: u64,
    pub avg_throughput_tps: f64,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub total_cache_read_tokens: u64,
    /// Tokens written to the prompt cache (billed above input; needed for
    /// accurate cost display — was previously dropped from the summary).
    pub total_cache_creation_tokens: u64,
    /// The 1-hour share of `total_cache_creation_tokens`.
    #[serde(default)]
    pub total_cache_creation_1h_tokens: u64,
    pub cache_hit_rate: f64,
    pub consecutive_failures: u32,
    /// When the most recent failure was recorded (epoch ms; 0 = never). A
    /// cooldown is measured from here — measured from "now" instead, its
    /// deadline moved with the clock and never arrived.
    #[serde(default)]
    pub last_failure_epoch_ms: u64,
    pub is_healthy: bool,
    pub escalation_count: u64,
}

/// Estimated USD spend for one model over its tracked lifetime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCost {
    /// Model identifier.
    pub model: String,
    /// Estimated total spend in USD from accumulated token counts.
    pub estimated_cost_usd: f64,
    /// Whether pricing was known. `false` = local/unknown model (cost is 0,
    /// not billed) — distinguishes "free" from "priced at exactly $0".
    pub priced: bool,
}

/// Per-request stats to attach to an `AgentResponse` for UI display.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestModelStats {
    /// Which model actually handled this request
    pub model: String,
    /// Was this a routed request (vs primary model)?
    pub was_routed: bool,
    /// Complexity tier classification
    pub tier: String,
    /// Latency for this specific request (ms)
    pub latency_ms: u64,
    /// Tokens per second for this request
    pub throughput_tps: f64,
    /// Cache tokens read (saved money)
    pub cache_read_tokens: u32,
    /// Cache tokens written
    pub cache_creation_tokens: u32,
    /// The 1-hour share of `cache_creation_tokens`.
    #[serde(default)]
    pub cache_creation_1h_tokens: u32,
    /// Input tokens
    pub input_tokens: u32,
    /// Output tokens
    pub output_tokens: u32,
    /// The LIVE effective context window (the Ollama `num_ctx` latch of the
    /// server the request went to) this request ran under, when the sizing
    /// path has one. `None` for cloud models and unlatched local models.
    /// Records mid-run demotions in the per-request trail so a shrunken window
    /// is visible in the stats, not only in logs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effective_context_window: Option<usize>,
}

const MAX_LATENCY_SAMPLES: usize = 100;
const MAX_THROUGHPUT_SAMPLES: usize = 100;
/// A model with this many consecutive failures is considered unhealthy
const UNHEALTHY_THRESHOLD: u32 = 3;

impl ModelStatsTracker {
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(StatsInner::default())),
            sink: Arc::new(std::sync::OnceLock::new()),
        }
    }

    /// Send every future observation to `sink` as well. Returns `false` when a
    /// sink was already set (the first one stays).
    pub fn set_request_sink(&self, sink: RequestSink) -> bool {
        self.sink.set(sink).is_ok()
    }

    /// Record a completed request observation.
    pub async fn record(&self, obs: RequestObservation) {
        if let Some(sink) = self.sink.get() {
            sink(&obs);
        }
        let mut inner = self.inner.write().await;
        let stats = inner.models.entry(obs.model.clone()).or_insert_with(|| ModelStats::new(&obs.model));

        stats.total_requests += 1;
        let latency_ms = millis_u64(obs.latency);

        if obs.success {
            stats.successful_requests += 1;
            stats.consecutive_failures = 0;
            stats.last_success_epoch_ms = now_epoch_ms();

            // Record latency
            if stats.latencies_ms.len() >= MAX_LATENCY_SAMPLES {
                stats.latencies_ms.remove(0);
            }
            stats.latencies_ms.push(latency_ms);

            // Record throughput
            if obs.latency.as_millis() > 0 {
                let tps = f64::from(obs.output_tokens) / obs.latency.as_secs_f64();
                if stats.throughput_tps.len() >= MAX_THROUGHPUT_SAMPLES {
                    stats.throughput_tps.remove(0);
                }
                stats.throughput_tps.push(tps);
            }

            // Tier tracking
            if let Some(tier) = obs.tier {
                match tier {
                    super::loop_runner::TaskComplexity::Simple => stats.tier_successes.simple += 1,
                    super::loop_runner::TaskComplexity::Medium => stats.tier_successes.medium += 1,
                    super::loop_runner::TaskComplexity::Complex => stats.tier_successes.complex += 1,
                }
            }
        } else {
            stats.failed_requests += 1;
            stats.consecutive_failures += 1;
            stats.last_failure_epoch_ms = now_epoch_ms();

            if let Some(tier) = obs.tier {
                match tier {
                    super::loop_runner::TaskComplexity::Simple => stats.tier_failures.simple += 1,
                    super::loop_runner::TaskComplexity::Medium => stats.tier_failures.medium += 1,
                    super::loop_runner::TaskComplexity::Complex => stats.tier_failures.complex += 1,
                }
            }
        }

        // Token accounting
        stats.total_input_tokens += u64::from(obs.input_tokens);
        stats.total_output_tokens += u64::from(obs.output_tokens);
        stats.total_cache_read_tokens += u64::from(obs.cache_read_tokens);
        stats.total_cache_creation_tokens += u64::from(obs.cache_creation_tokens);
        debug_assert!(
            obs.cache_creation_1h_tokens <= obs.cache_creation_tokens,
            "the 1-hour share is a subset of the write total"
        );
        stats.total_cache_creation_1h_tokens += u64::from(obs.cache_creation_1h_tokens);

        if obs.escalated {
            stats.escalations += 1;
        }
        drop(inner);

        debug!(model = %obs.model, success = obs.success, latency_ms = latency_ms, input = obs.input_tokens, output = obs.output_tokens, cache_read = obs.cache_read_tokens, "📊 Model stats recorded");
    }

    /// Check if a model is considered healthy (not in a failure streak).
    pub async fn is_healthy(&self, model: &str) -> bool {
        let inner = self.inner.read().await;
        inner.models.get(model)
            .is_none_or(|s| s.consecutive_failures < UNHEALTHY_THRESHOLD)
    }

    /// Get summary statistics for all tracked models.
    pub async fn summaries(&self) -> Vec<ModelStatsSummary> {
        let inner = self.inner.read().await;
        inner.models.values().map(ModelStats::summary).collect()
    }

    /// Get summary for a specific model.
    pub async fn summary(&self, model: &str) -> Option<ModelStatsSummary> {
        let inner = self.inner.read().await;
        inner.models.get(model).map(ModelStats::summary)
    }

    /// Get all raw stats (for persistence/export).
    pub async fn all_stats(&self) -> HashMap<String, ModelStats> {
        let inner = self.inner.read().await;
        inner.models.clone()
    }

    /// Estimate accumulated USD spend per model from the tracked token counts.
    ///
    /// Uses [`crate::cost::default_pricing`] (reference list prices). Models
    /// with no known pricing (local/Ollama/unknown) are reported with
    /// `priced == false` and `estimated_cost_usd == 0.0` so callers never
    /// silently fold an unknown cost into a total.
    pub async fn cost_report(&self) -> Vec<ModelCost> {
        // Snapshot (model, token counts) under the read lock, then release it
        // before pricing lookups/arithmetic so the lock isn't held over work
        // that doesn't touch shared state.
        let usage: Vec<(String, u64, u64, u64, u64, u64)> = {
            let inner = self.inner.read().await;
            inner
                .models
                .values()
                .map(|s| {
                    (
                        s.model.clone(),
                        s.total_input_tokens,
                        s.total_output_tokens,
                        s.total_cache_read_tokens,
                        s.total_cache_creation_tokens,
                        s.total_cache_creation_1h_tokens,
                    )
                })
                .collect()
        };

        let mut report: Vec<ModelCost> = usage
            .into_iter()
            .map(|(model, input, output, cache_read, cache_write, cache_write_1h)| {
                crate::cost::default_pricing(&model).map_or_else(
                    || ModelCost {
                        model: model.clone(),
                        estimated_cost_usd: 0.0,
                        priced: false,
                    },
                    |pricing| ModelCost {
                        model: model.clone(),
                        estimated_cost_usd: crate::cost::estimate_cost_usd_with_hour_writes(
                            input,
                            output,
                            cache_read,
                            cache_write,
                            cache_write_1h,
                            &pricing,
                        ),
                        priced: true,
                    },
                )
            })
            .collect();
        // Deterministic order for UI/tests: priciest first, then by name.
        report.sort_by(|a, b| {
            b.estimated_cost_usd
                .total_cmp(&a.estimated_cost_usd)
                .then_with(|| a.model.cmp(&b.model))
        });
        report
    }

    /// Grand-total estimated USD spend across all **priced** models.
    ///
    /// Unpriced (local/unknown) models contribute nothing, matching
    /// [`Self::cost_report`] — the total is "known cloud spend", not a claim
    /// that local runs are free-and-therefore-$0-in-the-same-bucket.
    pub async fn total_cost_usd(&self) -> f64 {
        self.cost_report()
            .await
            .iter()
            .filter(|c| c.priced)
            .map(|c| c.estimated_cost_usd)
            .sum()
    }

    /// Load stats from a previous session (e.g., from disk).
    pub async fn load(&self, stats: HashMap<String, ModelStats>) {
        let mut inner = self.inner.write().await;
        inner.models = stats;
        info!(models = inner.models.len(), "Loaded model stats from persistence");
    }
}

impl Default for ModelStatsTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelStats {
    fn new(model: &str) -> Self {
        Self {
            model: model.to_string(),
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            total_input_tokens: 0,
            total_output_tokens: 0,
            total_cache_read_tokens: 0,
            total_cache_creation_tokens: 0,
            total_cache_creation_1h_tokens: 0,
            latencies_ms: Vec::with_capacity(MAX_LATENCY_SAMPLES),
            throughput_tps: Vec::with_capacity(MAX_THROUGHPUT_SAMPLES),
            consecutive_failures: 0,
            last_success_epoch_ms: 0,
            last_failure_epoch_ms: 0,
            tier_successes: TierCounts::default(),
            tier_failures: TierCounts::default(),
            escalations: 0,
        }
    }

    fn summary(&self) -> ModelStatsSummary {
        let success_rate = if self.total_requests > 0 {
            f64_from_u64(self.successful_requests) / f64_from_u64(self.total_requests)
        } else {
            1.0
        };

        let avg_latency_ms = if self.latencies_ms.is_empty() {
            0
        } else {
            self.latencies_ms.iter().sum::<u64>() / self.latencies_ms.len() as u64
        };

        let p95_latency_ms = percentile(&self.latencies_ms, 95);

        let avg_throughput_tps = if self.throughput_tps.is_empty() {
            0.0
        } else {
            self.throughput_tps.iter().sum::<f64>() / f64_from_usize(self.throughput_tps.len())
        };

        let total_cacheable = self.total_input_tokens + self.total_cache_read_tokens;
        let cache_hit_rate = if total_cacheable > 0 {
            f64_from_u64(self.total_cache_read_tokens) / f64_from_u64(total_cacheable)
        } else {
            0.0
        };

        ModelStatsSummary {
            model: self.model.clone(),
            total_requests: self.total_requests,
            success_rate,
            avg_latency_ms,
            p95_latency_ms,
            avg_throughput_tps,
            total_input_tokens: self.total_input_tokens,
            total_output_tokens: self.total_output_tokens,
            total_cache_read_tokens: self.total_cache_read_tokens,
            total_cache_creation_tokens: self.total_cache_creation_tokens,
            total_cache_creation_1h_tokens: self.total_cache_creation_1h_tokens,
            cache_hit_rate,
            consecutive_failures: self.consecutive_failures,
            last_failure_epoch_ms: self.last_failure_epoch_ms,
            is_healthy: self.consecutive_failures < UNHEALTHY_THRESHOLD,
            escalation_count: self.escalations,
        }
    }
}

fn percentile(sorted_data: &[u64], pct: usize) -> u64 {
    if sorted_data.is_empty() {
        return 0;
    }
    let mut sorted = sorted_data.to_vec();
    sorted.sort_unstable();
    let idx = (pct * sorted.len() / 100).min(sorted.len() - 1);
    sorted[idx]
}

fn now_epoch_ms() -> u64 {
    millis_u64(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default(),
    )
}

// =============================================================================
// Storage bridge (converts to/from nanna-storage::StoredModelStats)
// =============================================================================

impl ModelStatsTracker {
    /// Export all stats as storable format (for nanna-storage persistence).
    pub async fn export_for_storage(&self) -> Vec<StorableModelStats> {
        let inner = self.inner.read().await;
        inner.models.values().map(|s| StorableModelStats {
            model: s.model.clone(),
            total_requests: s.total_requests,
            successful_requests: s.successful_requests,
            failed_requests: s.failed_requests,
            total_input_tokens: s.total_input_tokens,
            total_output_tokens: s.total_output_tokens,
            total_cache_read_tokens: s.total_cache_read_tokens,
            total_cache_creation_tokens: s.total_cache_creation_tokens,
            total_cache_creation_1h_tokens: s.total_cache_creation_1h_tokens,
            consecutive_failures: s.consecutive_failures,
            last_success_epoch_ms: s.last_success_epoch_ms,
            last_failure_epoch_ms: s.last_failure_epoch_ms,
            tier_successes_simple: s.tier_successes.simple,
            tier_successes_medium: s.tier_successes.medium,
            tier_successes_complex: s.tier_successes.complex,
            tier_failures_simple: s.tier_failures.simple,
            tier_failures_medium: s.tier_failures.medium,
            tier_failures_complex: s.tier_failures.complex,
            escalations: s.escalations,
            latencies_ms: s.latencies_ms.clone(),
            throughput_tps: s.throughput_tps.clone(),
        }).collect()
    }

    /// Import stats from storage format.
    pub async fn import_from_storage(&self, stored: Vec<StorableModelStats>) {
        let mut inner = self.inner.write().await;
        for s in stored {
            let stats = ModelStats {
                model: s.model.clone(),
                total_requests: s.total_requests,
                successful_requests: s.successful_requests,
                failed_requests: s.failed_requests,
                total_input_tokens: s.total_input_tokens,
                total_output_tokens: s.total_output_tokens,
                total_cache_read_tokens: s.total_cache_read_tokens,
                total_cache_creation_tokens: s.total_cache_creation_tokens,
                total_cache_creation_1h_tokens: s.total_cache_creation_1h_tokens,
                consecutive_failures: s.consecutive_failures,
                last_success_epoch_ms: s.last_success_epoch_ms,
                last_failure_epoch_ms: s.last_failure_epoch_ms,
                tier_successes: TierCounts {
                    simple: s.tier_successes_simple,
                    medium: s.tier_successes_medium,
                    complex: s.tier_successes_complex,
                },
                tier_failures: TierCounts {
                    simple: s.tier_failures_simple,
                    medium: s.tier_failures_medium,
                    complex: s.tier_failures_complex,
                },
                escalations: s.escalations,
                latencies_ms: s.latencies_ms,
                throughput_tps: s.throughput_tps,
            };
            inner.models.insert(s.model, stats);
        }
        info!(models = inner.models.len(), "Imported model stats from storage");
    }
}

/// Flat struct matching `nanna-storage::StoredModelStats` layout.
/// This avoids a cross-crate dependency while keeping the types aligned.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorableModelStats {
    pub model: String,
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub total_cache_read_tokens: u64,
    pub total_cache_creation_tokens: u64,
    #[serde(default)]
    pub total_cache_creation_1h_tokens: u64,
    pub consecutive_failures: u32,
    pub last_success_epoch_ms: u64,
    pub last_failure_epoch_ms: u64,
    pub tier_successes_simple: u64,
    pub tier_successes_medium: u64,
    pub tier_successes_complex: u64,
    pub tier_failures_simple: u64,
    pub tier_failures_medium: u64,
    pub tier_failures_complex: u64,
    pub escalations: u64,
    pub latencies_ms: Vec<u64>,
    pub throughput_tps: Vec<f64>,
}

// =============================================================================
// From impls between StorableModelStats <-> StoredModelStats
// =============================================================================

impl From<StoredModelStats> for StorableModelStats {
    fn from(s: StoredModelStats) -> Self {
        Self {
            model: s.model,
            total_requests: s.total_requests,
            successful_requests: s.successful_requests,
            failed_requests: s.failed_requests,
            total_input_tokens: s.total_input_tokens,
            total_output_tokens: s.total_output_tokens,
            total_cache_read_tokens: s.total_cache_read_tokens,
            total_cache_creation_tokens: s.total_cache_creation_tokens,
            total_cache_creation_1h_tokens: s.total_cache_creation_1h_tokens,
            consecutive_failures: s.consecutive_failures,
            last_success_epoch_ms: s.last_success_epoch_ms,
            last_failure_epoch_ms: s.last_failure_epoch_ms,
            tier_successes_simple: s.tier_successes_simple,
            tier_successes_medium: s.tier_successes_medium,
            tier_successes_complex: s.tier_successes_complex,
            tier_failures_simple: s.tier_failures_simple,
            tier_failures_medium: s.tier_failures_medium,
            tier_failures_complex: s.tier_failures_complex,
            escalations: s.escalations,
            latencies_ms: s.latencies_ms,
            throughput_tps: s.throughput_tps,
        }
    }
}

impl From<StorableModelStats> for StoredModelStats {
    fn from(s: StorableModelStats) -> Self {
        Self {
            model: s.model,
            total_requests: s.total_requests,
            successful_requests: s.successful_requests,
            failed_requests: s.failed_requests,
            total_input_tokens: s.total_input_tokens,
            total_output_tokens: s.total_output_tokens,
            total_cache_read_tokens: s.total_cache_read_tokens,
            total_cache_creation_tokens: s.total_cache_creation_tokens,
            total_cache_creation_1h_tokens: s.total_cache_creation_1h_tokens,
            consecutive_failures: s.consecutive_failures,
            last_success_epoch_ms: s.last_success_epoch_ms,
            last_failure_epoch_ms: s.last_failure_epoch_ms,
            tier_successes_simple: s.tier_successes_simple,
            tier_successes_medium: s.tier_successes_medium,
            tier_successes_complex: s.tier_successes_complex,
            tier_failures_simple: s.tier_failures_simple,
            tier_failures_medium: s.tier_failures_medium,
            tier_failures_complex: s.tier_failures_complex,
            escalations: s.escalations,
            latencies_ms: s.latencies_ms,
            throughput_tps: s.throughput_tps,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn every_recorded_request_reaches_the_sink_once_set() {
        let tracker = ModelStatsTracker::new();
        tracker.record(observation("before-sink", true)).await;
        let seen: Arc<std::sync::Mutex<Vec<String>>> = Arc::default();
        let seen_in_sink = Arc::clone(&seen);
        assert!(
            tracker.set_request_sink(Arc::new(move |obs: &RequestObservation| {
                seen_in_sink.lock().expect("seen").push(obs.model.clone());
            }))
        );
        assert!(
            !tracker.set_request_sink(Arc::new(|_: &RequestObservation| {})),
            "set once"
        );
        // A clone shares the sink, as it shares the stats.
        tracker
            .clone()
            .record(observation("claude-opus-5", true))
            .await;
        tracker
            .record(observation("ollama/qwen3.5:9b", false))
            .await;
        assert_eq!(
            *seen.lock().expect("seen"),
            vec!["claude-opus-5".to_string(), "ollama/qwen3.5:9b".to_string()]
        );
        assert_eq!(
            tracker.summaries().await.len(),
            3,
            "the aggregates still record everything"
        );
    }

    fn observation(model: &str, success: bool) -> RequestObservation {
        RequestObservation {
            model: model.to_string(),
            success,
            latency: Duration::from_millis(10),
            input_tokens: 100,
            output_tokens: 20,
            cache_read_tokens: 0,
            cache_creation_tokens: 0,
            cache_creation_1h_tokens: 0,
            tier: None,
            escalated: false,
        }
    }

    // The daemon shares ONE tracker between the main agent, every sub-agent,
    // and the control plane by cloning it. This is only correct if clones
    // share underlying state — a record via one clone must be visible via
    // another. Guards the P11 "wire shared stats tracker" fix.
    #[tokio::test]
    async fn clone_shares_underlying_state() {
        let main = ModelStatsTracker::new();
        let sub_agent = main.clone();
        let control_plane = main.clone();

        // A sub-agent records a request...
        sub_agent.record(observation("claude-opus-4-8", true)).await;
        // ...the main agent records another for the same model...
        main.record(observation("claude-opus-4-8", true)).await;

        // ...and the control-plane clone sees BOTH (shared Arc<RwLock<_>>).
        let summaries = control_plane.summaries().await;
        assert_eq!(summaries.len(), 1, "one model tracked across all clones");
        let s = &summaries[0];
        assert_eq!(s.model, "claude-opus-4-8");
        assert_eq!(
            s.total_requests, 2,
            "both clones' records accumulate in the shared tracker"
        );
        assert_eq!(s.total_input_tokens, 200);
        assert_eq!(s.total_output_tokens, 40);
    }

    // A fresh (unshared) tracker must NOT see another tracker's records —
    // proves the sharing above comes from cloning, not global state.
    #[tokio::test]
    async fn independent_trackers_do_not_share() {
        let a = ModelStatsTracker::new();
        let b = ModelStatsTracker::new();
        a.record(observation("gpt-5", true)).await;
        assert_eq!(a.summaries().await.len(), 1);
        assert!(
            b.summaries().await.is_empty(),
            "a separately constructed tracker stays empty"
        );
    }

    // The summary must surface cache-creation tokens (record() accumulates
    // them, but the summary previously dropped the field, hiding cache-write
    // volume and breaking accurate cost display).
    #[tokio::test]
    async fn summary_surfaces_cache_creation_tokens() {
        let tracker = ModelStatsTracker::new();
        tracker
            .record(RequestObservation {
                model: "claude-sonnet-5".to_string(),
                success: true,
                latency: Duration::from_millis(10),
                input_tokens: 1000,
                output_tokens: 200,
                cache_read_tokens: 4000,
                cache_creation_tokens: 800,
                cache_creation_1h_tokens: 0,
                tier: None,
                escalated: false,
            })
            .await;
        let s = tracker.summary("claude-sonnet-5").await.expect("summary exists");
        assert_eq!(s.total_cache_creation_tokens, 800);
        assert_eq!(s.total_cache_read_tokens, 4000);
    }

    fn observation_with_tokens(model: &str, input: u32, output: u32) -> RequestObservation {
        RequestObservation {
            model: model.to_string(),
            success: true,
            latency: Duration::from_millis(10),
            input_tokens: input,
            output_tokens: output,
            cache_read_tokens: 0,
            cache_creation_tokens: 0,
            cache_creation_1h_tokens: 0,
            tier: None,
            escalated: false,
        }
    }

    // cost_report turns tracked tokens into USD for priced models and flags
    // unpriced (local) models instead of folding a bogus $0 into a total.
    #[tokio::test]
    async fn cost_report_prices_known_models_and_flags_unknown() {
        let tracker = ModelStatsTracker::new();
        // 1M input + 1M output on Sonnet-class ($3 in / $15 out) = $18.
        tracker
            .record(observation_with_tokens("claude-sonnet-5", 1_000_000, 1_000_000))
            .await;
        // A local model with tokens but no pricing.
        tracker
            .record(observation_with_tokens("ollama/llama3.2", 500_000, 500_000))
            .await;

        let report = tracker.cost_report().await;
        assert_eq!(report.len(), 2);
        // Sorted priciest-first, so the priced model leads.
        let claude = &report[0];
        assert_eq!(claude.model, "claude-sonnet-5");
        assert!(claude.priced);
        // Sonnet 5 is $2 in / $10 out per 1M (not Sonnet 4's $3/$15).
        assert!(
            (claude.estimated_cost_usd - 12.0).abs() < 1e-6,
            "got {}",
            claude.estimated_cost_usd
        );

        let local = &report[1];
        assert_eq!(local.model, "ollama/llama3.2");
        assert!(!local.priced, "local model must be flagged unpriced");
        assert!(local.estimated_cost_usd.abs() < f64::EPSILON);
    }

    // total_cost_usd sums only priced models: Sonnet 5 $12 + GPT-5 (1M in @ $1.25
    // + 1M out @ $10 = $11.25) = $23.25; the local model contributes nothing.
    #[tokio::test]
    async fn total_cost_sums_priced_models_only() {
        let tracker = ModelStatsTracker::new();
        tracker
            .record(observation_with_tokens("claude-sonnet-5", 1_000_000, 1_000_000))
            .await;
        tracker
            .record(observation_with_tokens("gpt-5", 1_000_000, 1_000_000))
            .await;
        tracker
            .record(observation_with_tokens("ollama/llama3.2", 9_000_000, 9_000_000))
            .await;
        let total = tracker.total_cost_usd().await;
        assert!((total - 23.25).abs() < 1e-6, "got {total}");
    }
}

#[cfg(test)]
mod hour_cache_write_tests {
    use super::*;

    fn write_observation(cache_writes: u32, cache_writes_1h: u32) -> RequestObservation {
        RequestObservation {
            model: "claude-sonnet-5".to_string(),
            success: true,
            latency: Duration::from_millis(10),
            input_tokens: 0,
            output_tokens: 0,
            cache_read_tokens: 0,
            cache_creation_tokens: cache_writes,
            cache_creation_1h_tokens: cache_writes_1h,
            tier: None,
            escalated: false,
        }
    }

    // A 1-hour write is billed at 2x input; before the split existed every write
    // was priced at the 5-minute 1.25x, under-reporting a 1h user's spend.
    #[tokio::test]
    async fn cost_report_prices_one_hour_writes_at_twice_input() {
        let pricing = crate::cost::default_pricing("claude-sonnet-5").expect("priced model");
        let tracker = ModelStatsTracker::new();
        tracker
            .record(write_observation(1_000_000, 1_000_000))
            .await;

        let report = tracker.cost_report().await;
        let expected = pricing.input_usd_per_mtok * 2.0;
        assert!(
            (report[0].estimated_cost_usd - expected).abs() < 1e-9,
            "got {}, expected {expected}",
            report[0].estimated_cost_usd
        );
        let summary = tracker
            .summary("claude-sonnet-5")
            .await
            .expect("summary exists");
        assert_eq!(summary.total_cache_creation_1h_tokens, 1_000_000);
    }

    #[tokio::test]
    async fn the_one_hour_share_survives_the_storage_bridge() {
        let tracker = ModelStatsTracker::new();
        tracker.record(write_observation(900, 600)).await;

        let exported = tracker.export_for_storage().await;
        assert_eq!(exported[0].total_cache_creation_1h_tokens, 600);
        let restored = ModelStatsTracker::new();
        restored.import_from_storage(exported).await;
        let summary = restored.summary("claude-sonnet-5").await.expect("imported");
        assert_eq!(summary.total_cache_creation_1h_tokens, 600);
    }
}
