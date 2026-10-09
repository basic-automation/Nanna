//! Tool performance statistics tracking.
//!
//! Tracks per-tool metrics across invocations to provide visibility
//! into tool behavior:
//! - Call count, success/failure rates
//! - Latency percentiles (P50, P95, P99)
//! - Output sizes
//! - Common errors
//! - Per-session aggregates (tool time vs LLM time)

use std::collections::HashMap;
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::numeric::{millis_u64, f64_from_u64};

/// Maximum number of latency samples to keep per tool (ring buffer).
const MAX_LATENCY_SAMPLES: usize = 200;
/// Maximum number of output-size samples to keep per tool (ring buffer).
const MAX_OUTPUT_SAMPLES: usize = 200;
/// Maximum number of distinct error messages to track per tool.
const MAX_ERROR_ENTRIES: usize = 50;

// =============================================================================
// Core tracker
// =============================================================================

/// Global tool statistics tracker. Thread-safe, designed for concurrent access.
#[derive(Debug, Clone)]
pub struct ToolStatsTracker {
    inner: Arc<RwLock<ToolStatsInner>>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct ToolStatsInner {
    /// Per-tool statistics keyed by tool name.
    tools: HashMap<String, ToolStats>,
    /// Per-session statistics keyed by session ID.
    sessions: HashMap<String, SessionStats>,
}

/// Accumulated statistics for a single tool.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ToolStats {
    /// Tool name
    pub name: String,
    /// Total invocations
    pub call_count: u64,
    /// Successful invocations
    pub success_count: u64,
    /// Failed invocations
    pub failure_count: u64,
    /// Invocations the harness answered with a breaker replay instead of
    /// dispatching (repeat-failure / zero-information / discovery-pause
    /// short-circuits). Their own outcome, NOT failures: the tool never ran,
    /// so counting them as `failure_count` made healthy tools look broken
    /// (P22 Tier 4 — an lfm leg's stats showed a wall of `success=0` rows
    /// that were all harness replays, not tool faults).
    pub short_circuit_count: u64,
    /// Recent latencies in milliseconds (ring buffer, last N)
    pub latencies_ms: Vec<u64>,
    /// Recent output sizes in bytes/chars (ring buffer, last N)
    pub output_sizes: Vec<usize>,
    /// Epoch-ms timestamp of the most recent invocation
    pub last_called: Option<u64>,
    /// Common error messages with occurrence counts
    pub errors: Vec<(String, u64)>,
    /// Every executed call's latency since the daemon started, bucketed for
    /// `/metrics` (not persisted; see [`crate::histogram`]).
    #[serde(skip)]
    pub latency_histogram: crate::histogram::LatencyHistogram,
}

/// Per-session aggregate statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SessionStats {
    /// Session identifier
    pub session_id: String,
    /// Number of agent-loop iterations
    pub iterations: u64,
    /// Total tool invocations in this session
    pub tool_calls: u64,
    /// Cumulative tool execution time (ms)
    pub tool_time_ms: u64,
    /// Cumulative LLM request time (ms)
    pub llm_time_ms: u64,
    /// Cumulative input tokens
    pub input_tokens: u64,
    /// Cumulative output tokens
    pub output_tokens: u64,
}

/// A single tool invocation observation to record.
#[derive(Debug)]
pub struct ToolObservation {
    pub tool_name: String,
    pub success: bool,
    /// The harness short-circuited this call (breaker replay) — it was never
    /// dispatched. Recorded as its own outcome so success/failure rates keep
    /// describing the tool, not the harness. When set, `success`, `error`,
    /// `duration_ms`, and `output_size` describe the replay, not an
    /// execution, and are not folded into the execution statistics.
    pub short_circuited: bool,
    pub duration_ms: u64,
    pub output_size: usize,
    /// Error message (if any)
    pub error: Option<String>,
    /// Optional session ID to attribute this call to
    pub session_id: Option<String>,
}

/// Summary stats for a single tool, suitable for UI display.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolStatsSummary {
    pub name: String,
    pub call_count: u64,
    pub success_count: u64,
    pub failure_count: u64,
    /// Harness short-circuits (breaker replays) — see [`ToolStats::short_circuit_count`].
    #[serde(default)]
    pub short_circuit_count: u64,
    /// Success rate over EXECUTED calls (`success + failure`) — short-circuits
    /// are excluded because the tool never ran.
    pub success_rate: f64,
    pub avg_latency_ms: u64,
    pub p50_latency_ms: u64,
    pub p95_latency_ms: u64,
    pub p99_latency_ms: u64,
    pub avg_output_size: usize,
    pub last_called: Option<u64>,
    pub top_errors: Vec<(String, u64)>,
    /// See [`ToolStats::latency_histogram`]; read by `/metrics`, not sent to
    /// clients.
    #[serde(skip)]
    pub latency_histogram: crate::histogram::LatencyHistogram,
}

/// Global dashboard summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalToolStats {
    /// Total tool calls across all tools
    pub total_calls: u64,
    /// Overall average latency (ms)
    pub avg_latency_ms: u64,
    /// Overall success rate
    pub success_rate: f64,
    /// Top 5 slowest tools by P95
    pub slowest_tools: Vec<ToolStatsSummary>,
    /// Top 10 most-used tools
    pub most_used_tools: Vec<ToolStatsSummary>,
    /// Top 5 most-failed tools (by error rate)
    pub most_failed_tools: Vec<ToolStatsSummary>,
    /// Aggregate session stats
    pub session_totals: SessionTotals,
}

/// Aggregated session totals across all sessions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionTotals {
    pub total_iterations: u64,
    pub total_tool_calls: u64,
    pub total_tool_time_ms: u64,
    pub total_llm_time_ms: u64,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
}

// =============================================================================
// Implementation
// =============================================================================

impl ToolStatsTracker {
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(ToolStatsInner::default())),
        }
    }

    /// Record a completed tool invocation.
    pub async fn record(&self, obs: ToolObservation) {
        let mut inner = self.inner.write().await;
        let stats = inner.tools.entry(obs.tool_name.clone())
            .or_insert_with(|| ToolStats::new(&obs.tool_name));

        stats.call_count += 1;
        stats.last_called = Some(now_epoch_ms());

        // A short-circuited call never dispatched: it gets its own outcome
        // counter and contributes NO execution samples — a 0 ms replay in the
        // latency ring or a replay notice counted as a failure would describe
        // the harness, not the tool.
        if obs.short_circuited {
            stats.short_circuit_count += 1;
            if let Some(ref sid) = obs.session_id {
                let session = inner.sessions.entry(sid.clone())
                    .or_insert_with(|| SessionStats::new(sid));
                session.tool_calls += 1;
            }
            drop(inner);
            debug!(
                tool = %obs.tool_name,
                "📊 Tool stats recorded (short-circuited — breaker replay, not executed)"
            );
            return;
        }

        if obs.success {
            stats.success_count += 1;
        } else {
            stats.failure_count += 1;
        }

        // Ring-buffer latency
        if stats.latencies_ms.len() >= MAX_LATENCY_SAMPLES {
            stats.latencies_ms.remove(0);
        }
        stats.latencies_ms.push(obs.duration_ms);
        stats
            .latency_histogram
            .observe(&crate::histogram::TOOL_LATENCY_BOUNDS_MS, obs.duration_ms);

        // Ring-buffer output size
        if stats.output_sizes.len() >= MAX_OUTPUT_SAMPLES {
            stats.output_sizes.remove(0);
        }
        stats.output_sizes.push(obs.output_size);

        // Track errors
        if let Some(ref err_msg) = obs.error {
            // Truncate error to first 200 chars for dedup
            let key = if err_msg.len() > 200 {
                format!("{}...", &err_msg[..err_msg.floor_char_boundary(200)])
            } else {
                err_msg.clone()
            };
            if let Some(entry) = stats.errors.iter_mut().find(|(e, _)| e == &key) {
                entry.1 += 1;
            } else if stats.errors.len() < MAX_ERROR_ENTRIES {
                stats.errors.push((key, 1));
            }
        }

        // Attribute to session
        if let Some(ref sid) = obs.session_id {
            let session = inner.sessions.entry(sid.clone())
                .or_insert_with(|| SessionStats::new(sid));
            session.tool_calls += 1;
            session.tool_time_ms += obs.duration_ms;
        }
        drop(inner);

        if obs.duration_ms > 5_000 {
            warn!(
                tool = %obs.tool_name,
                duration_ms = obs.duration_ms,
                "⚠️ Slow tool execution recorded in stats (>5s)"
            );
        }

        debug!(
            tool = %obs.tool_name,
            success = obs.success,
            duration_ms = obs.duration_ms,
            output_size = obs.output_size,
            "📊 Tool stats recorded"
        );
    }

    /// Record LLM time for a session.
    pub async fn record_llm_time(&self, session_id: &str, llm_time_ms: u64, input_tokens: u64, output_tokens: u64) {
        let mut inner = self.inner.write().await;
        let session = inner.sessions.entry(session_id.to_string())
            .or_insert_with(|| SessionStats::new(session_id));
        session.llm_time_ms += llm_time_ms;
        session.input_tokens += input_tokens;
        session.output_tokens += output_tokens;
        drop(inner);
    }

    /// Record an iteration for a session.
    pub async fn record_iteration(&self, session_id: &str) {
        let mut inner = self.inner.write().await;
        let session = inner.sessions.entry(session_id.to_string())
            .or_insert_with(|| SessionStats::new(session_id));
        session.iterations += 1;
        drop(inner);
    }

    /// Get summary statistics for all tracked tools.
    pub async fn summaries(&self) -> Vec<ToolStatsSummary> {
        let inner = self.inner.read().await;
        inner.tools.values().map(ToolStats::summary).collect()
    }

    /// Get summary for a specific tool.
    pub async fn summary(&self, tool_name: &str) -> Option<ToolStatsSummary> {
        let inner = self.inner.read().await;
        inner.tools.get(tool_name).map(ToolStats::summary)
    }

    /// Get the global dashboard stats.
    pub async fn global_stats(&self) -> GlobalToolStats {
        let inner = self.inner.read().await;
        let mut all_summaries: Vec<ToolStatsSummary> = inner.tools.values()
            .map(ToolStats::summary)
            .collect();

        let total_calls: u64 = all_summaries.iter().map(|s| s.call_count).sum();
        let total_success: u64 = all_summaries.iter().map(|s| s.success_count).sum();
        // Executed = success + failure; short-circuits never dispatched and
        // must not dilute the global success rate.
        let total_executed: u64 = all_summaries
            .iter()
            .map(|s| s.success_count + s.failure_count)
            .sum();

        // Overall average latency (weighted by call count)
        let weighted_latency: u64 = all_summaries.iter()
            .map(|s| s.avg_latency_ms * s.call_count)
            .sum();
        let avg_latency_ms = weighted_latency.checked_div(total_calls).unwrap_or(0);
        let success_rate = if total_executed > 0 { f64_from_u64(total_success) / f64_from_u64(total_executed) } else { 1.0 };

        // Top 5 slowest by P95
        let mut slowest = all_summaries.clone();
        slowest.sort_by_key(|s| std::cmp::Reverse(s.p95_latency_ms));
        slowest.truncate(5);

        // Top 10 most-used
        all_summaries.sort_by_key(|s| std::cmp::Reverse(s.call_count));
        let most_used: Vec<_> = all_summaries.iter().take(10).cloned().collect();

        // Top 5 most-failed (by error rate, min 2 calls)
        let mut by_error: Vec<_> = all_summaries.iter()
            .filter(|s| s.call_count >= 2)
            .cloned()
            .collect();
        by_error.sort_by(|a, b| {
            let rate_a = if a.call_count > 0 { f64_from_u64(a.failure_count) / f64_from_u64(a.call_count) } else { 0.0 };
            let rate_b = if b.call_count > 0 { f64_from_u64(b.failure_count) / f64_from_u64(b.call_count) } else { 0.0 };
            rate_b.partial_cmp(&rate_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        by_error.truncate(5);

        // Session totals
        let session_totals = SessionTotals {
            total_iterations: inner.sessions.values().map(|s| s.iterations).sum(),
            total_tool_calls: inner.sessions.values().map(|s| s.tool_calls).sum(),
            total_tool_time_ms: inner.sessions.values().map(|s| s.tool_time_ms).sum(),
            total_llm_time_ms: inner.sessions.values().map(|s| s.llm_time_ms).sum(),
            total_input_tokens: inner.sessions.values().map(|s| s.input_tokens).sum(),
            total_output_tokens: inner.sessions.values().map(|s| s.output_tokens).sum(),
        };
        drop(inner);

        GlobalToolStats {
            total_calls,
            avg_latency_ms,
            success_rate,
            slowest_tools: slowest,
            most_used_tools: most_used,
            most_failed_tools: by_error,
            session_totals,
        }
    }

    /// Get session stats for a specific session.
    pub async fn session_stats(&self, session_id: &str) -> Option<SessionStats> {
        let inner = self.inner.read().await;
        inner.sessions.get(session_id).cloned()
    }

    /// Get all raw tool stats (for persistence/export).
    pub async fn all_stats(&self) -> HashMap<String, ToolStats> {
        let inner = self.inner.read().await;
        inner.tools.clone()
    }

    /// Export the full inner state as JSON for persistence.
    pub async fn export_json(&self) -> serde_json::Value {
        let inner = self.inner.read().await;
        serde_json::to_value(&*inner).unwrap_or(serde_json::Value::Null)
    }

    /// Import stats from a previously persisted JSON blob.
    ///
    /// Tolerant by design: iterates the `tools`/`sessions` maps and deserializes
    /// each value individually, so a single drifted/legacy entry cannot nuke the
    /// whole import (the old whole-blob `from_value::<ToolStatsInner>` aborted on
    /// the first bad field — a boot logged `invalid type: integer 202, expected a
    /// map` because the DB aggregate emits `sessions` as a scalar count). A
    /// non-map `sessions` value is skipped rather than treated as fatal, and the
    /// map key is authoritative for the tool/session name (DB-shaped entries omit
    /// it).
    pub async fn import_json(&self, data: &serde_json::Value) {
        let mut inner = self.inner.write().await;

        let mut imported_tools = 0usize;
        let mut skipped_tools = 0usize;
        match data.get("tools") {
            Some(serde_json::Value::Object(tools)) => {
                for (name, value) in tools {
                    match serde_json::from_value::<ToolStats>(value.clone()) {
                        Ok(mut stats) => {
                            // Map key is authoritative (DB entries omit `name`).
                            stats.name.clone_from(name);
                            inner.tools.entry(name.clone()).or_insert(stats);
                            imported_tools += 1;
                        }
                        Err(e) => {
                            skipped_tools += 1;
                            warn!(tool = %name, "Skipping drifted tool-stats entry: {e}");
                        }
                    }
                }
            }
            Some(other) if !other.is_null() => {
                warn!("Tool stats 'tools' field is not a map; skipping tool import");
            }
            _ => {}
        }

        let mut imported_sessions = 0usize;
        let mut skipped_sessions = 0usize;
        match data.get("sessions") {
            Some(serde_json::Value::Object(sessions)) => {
                for (sid, value) in sessions {
                    match serde_json::from_value::<SessionStats>(value.clone()) {
                        Ok(mut session) => {
                            session.session_id.clone_from(sid);
                            inner.sessions.entry(sid.clone()).or_insert(session);
                            imported_sessions += 1;
                        }
                        Err(e) => {
                            skipped_sessions += 1;
                            warn!(session = %sid, "Skipping drifted session-stats entry: {e}");
                        }
                    }
                }
            }
            // Legacy/aggregate shape: `sessions` was an integer count — tolerate it.
            Some(other) if !other.is_null() => {
                debug!("Tool stats 'sessions' is not a map (legacy scalar); skipping session import");
            }
            _ => {}
        }

        info!(
            imported_tools,
            skipped_tools,
            imported_sessions,
            skipped_sessions,
            total_tools = inner.tools.len(),
            total_sessions = inner.sessions.len(),
            "Imported tool stats from persistence (tolerant)"
        );
    }
}

impl Default for ToolStatsTracker {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// ToolStats helpers
// =============================================================================

impl ToolStats {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            call_count: 0,
            success_count: 0,
            failure_count: 0,
            short_circuit_count: 0,
            latencies_ms: Vec::with_capacity(MAX_LATENCY_SAMPLES),
            output_sizes: Vec::with_capacity(MAX_OUTPUT_SAMPLES),
            last_called: None,
            errors: Vec::new(),
            latency_histogram: crate::histogram::LatencyHistogram::default(),
        }
    }

    fn summary(&self) -> ToolStatsSummary {
        // Rate over EXECUTED calls only: short-circuits never dispatched, so
        // they can neither succeed nor fail.
        let executed = self.success_count + self.failure_count;
        let success_rate = if executed > 0 {
            f64_from_u64(self.success_count) / f64_from_u64(executed)
        } else {
            1.0
        };

        let avg_latency_ms = if self.latencies_ms.is_empty() {
            0
        } else {
            self.latencies_ms.iter().sum::<u64>() / self.latencies_ms.len() as u64
        };

        let avg_output_size = if self.output_sizes.is_empty() {
            0
        } else {
            self.output_sizes.iter().sum::<usize>() / self.output_sizes.len()
        };

        // Sort top errors by count (descending), take top 5
        let mut top_errors = self.errors.clone();
        top_errors.sort_by_key(|e| std::cmp::Reverse(e.1));
        top_errors.truncate(5);

        ToolStatsSummary {
            name: self.name.clone(),
            call_count: self.call_count,
            success_count: self.success_count,
            failure_count: self.failure_count,
            short_circuit_count: self.short_circuit_count,
            success_rate,
            avg_latency_ms,
            p50_latency_ms: percentile(&self.latencies_ms, 50),
            p95_latency_ms: percentile(&self.latencies_ms, 95),
            p99_latency_ms: percentile(&self.latencies_ms, 99),
            avg_output_size,
            last_called: self.last_called,
            top_errors,
            latency_histogram: self.latency_histogram,
        }
    }
}

impl SessionStats {
    fn new(session_id: &str) -> Self {
        Self {
            session_id: session_id.to_string(),
            iterations: 0,
            tool_calls: 0,
            tool_time_ms: 0,
            llm_time_ms: 0,
            input_tokens: 0,
            output_tokens: 0,
        }
    }
}

// =============================================================================
// Helpers
// =============================================================================

fn percentile(data: &[u64], pct: usize) -> u64 {
    if data.is_empty() {
        return 0;
    }
    let mut sorted = data.to_vec();
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Executed calls reach the `/metrics` histogram, failures included; a
    /// short-circuit (never dispatched) does not, and the histogram is not
    /// part of what is exported or sent to clients.
    #[tokio::test]
    async fn executed_calls_feed_the_latency_histogram_and_replays_do_not() {
        let t = ToolStatsTracker::new();
        for (success, short_circuited, duration_ms) in [(true, false, 40), (false, false, 900), (false, true, 0)] {
            t.record(ToolObservation {
                tool_name: "exec".into(),
                success,
                short_circuited,
                duration_ms,
                output_size: 10,
                error: None,
                session_id: None,
            })
            .await;
        }
        let summary = t.summary("exec").await.expect("recorded");
        assert_eq!(summary.latency_histogram.count(), 2, "the replay is not an execution");
        assert_eq!(summary.latency_histogram.sum_ms(), 940);
        let wire = serde_json::to_value(&summary).expect("summary serializes");
        assert!(wire.get("latency_histogram").is_none(), "not sent to clients: {wire}");
    }

    #[tokio::test]
    async fn import_tolerates_bad_tool_entry() {
        // One bad entry must not drop the good ones.
        let t = ToolStatsTracker::new();
        t.import_json(&json!({
            "tools": {
                "good": {"name":"good","call_count":5,"success_count":5,"failure_count":0,
                         "latencies_ms":[10],"output_sizes":[100],"last_called":null,"errors":[]},
                "bad": 202
            },
            "sessions": {}
        }))
        .await;
        let good = t.summary("good").await;
        assert!(good.is_some());
        assert_eq!(good.unwrap().call_count, 5);
        assert!(t.summary("bad").await.is_none());
    }

    #[tokio::test]
    async fn import_tolerates_integer_sessions() {
        // The exact reported shape: DB aggregate emits `sessions` as a scalar
        // count and tool entries omit `name`. Must not abort the whole import.
        let t = ToolStatsTracker::new();
        t.import_json(&json!({
            "tools": {
                "read_file": {"call_count":3,"success_count":3,"failure_count":0,
                              "total_duration_ms":30,"last_called_epoch_ms":1,
                              "latencies_ms":[10],"output_sizes":[50],"errors":[]}
            },
            "sessions": 202
        }))
        .await;
        let s = t.summary("read_file").await;
        assert!(s.is_some(), "integer sessions must not abort the tool import");
        let s = s.unwrap();
        assert_eq!(s.call_count, 3);
        assert_eq!(s.name, "read_file", "name is backfilled from the map key");
    }

    #[tokio::test]
    async fn export_import_round_trips() {
        let t1 = ToolStatsTracker::new();
        t1.record(ToolObservation {
            tool_name: "read_file".into(),
            success: true,
            short_circuited: false,
            duration_ms: 12,
            output_size: 100,
            error: None,
            session_id: Some("s1".into()),
        })
        .await;
        t1.record(ToolObservation {
            tool_name: "exec".into(),
            success: false,
            short_circuited: false,
            duration_ms: 5,
            output_size: 0,
            error: Some("boom".into()),
            session_id: Some("s1".into()),
        })
        .await;
        t1.record_iteration("s1").await;
        t1.record_llm_time("s1", 100, 10, 20).await;

        let blob = t1.export_json().await;
        let t2 = ToolStatsTracker::new();
        t2.import_json(&blob).await;
        // Value equality on objects is order-independent, so this proves a
        // fully-valid blob round-trips losslessly (tools + sessions map intact).
        assert_eq!(t2.export_json().await, blob);
    }

    #[tokio::test]
    async fn import_skips_non_map_tools() {
        let t = ToolStatsTracker::new();
        t.import_json(&json!({"tools": 5})).await;
        assert!(t.summaries().await.is_empty());
    }

    /// P22 Tier 4: a breaker replay is its own outcome — not a failure, and
    /// not an execution sample. An lfm leg whose identical calls were
    /// short-circuited for hours must not read as a 0% success rate.
    #[tokio::test]
    async fn short_circuits_are_their_own_outcome_not_failures() {
        let t = ToolStatsTracker::new();
        t.record(ToolObservation {
            tool_name: "list_dir".into(),
            success: true,
            short_circuited: false,
            duration_ms: 40,
            output_size: 100,
            error: None,
            session_id: None,
        })
        .await;
        for _ in 0..3 {
            t.record(ToolObservation {
                tool_name: "list_dir".into(),
                success: false, // the replay notice is a `success: false` result…
                short_circuited: true, // …but the outcome is short_circuited
                duration_ms: 0,
                output_size: 500,
                error: Some("[ZERO-INFORMATION BREAKER] …".into()),
                session_id: None,
            })
            .await;
        }

        let s = t.summary("list_dir").await.expect("stats recorded");
        assert_eq!(s.call_count, 4, "every call the model made is counted");
        assert_eq!(s.short_circuit_count, 3);
        assert_eq!(s.failure_count, 0, "replays are not tool failures");
        assert_eq!(s.success_count, 1);
        assert!(
            (s.success_rate - 1.0).abs() < f64::EPSILON,
            "success rate covers executed calls only, got {}",
            s.success_rate
        );
        assert!(
            s.top_errors.is_empty(),
            "a replay notice is not a tool error: {:?}",
            s.top_errors
        );
        // Latency percentiles describe executions — the single real 40 ms
        // call, not three 0 ms replays.
        assert_eq!(s.p50_latency_ms, 40);

        let g = t.global_stats().await;
        assert!(
            (g.success_rate - 1.0).abs() < f64::EPSILON,
            "global success rate must not be diluted by replays"
        );
    }
}
