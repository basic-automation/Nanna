//! `GET /metrics` — the daemon's counters in Prometheus text format.
//!
//! The numbers already exist: the tool and model stats trackers behind the
//! GUI dashboards, the session store, the run registry, the MCP and scheduler
//! state. What an always-on daemon lacked was a way for anything *but* a
//! connected client to read them — a Prometheus/Grafana/uptime-kuma scrape, or
//! a `curl` from a shell. This renders them, and adds no dependency: the text
//! exposition format (0.0.4) is a line format, not a protocol.
//!
//! **Bounded by configuration, not by traffic.** Every label set comes from a
//! registered tool, a configured model or a configured MCP server — never from
//! a session id or a message — so the series count cannot grow with use.

use std::fmt::Write as _;

use nanna_agent::histogram::{
    LATENCY_BUCKETS, LatencyHistogram, MODEL_LATENCY_BOUNDS_MS, TOOL_LATENCY_BOUNDS_MS,
};
use nanna_agent::model_stats::ModelStatsSummary;
use nanna_agent::tool_stats::ToolStatsSummary;

use crate::mcp_startup::McpServerState;

/// Content type Prometheus expects for the text format.
pub const METRICS_CONTENT_TYPE: &str = "text/plain; version=0.0.4; charset=utf-8";

/// Everything `/metrics` reports, gathered in one pass.
#[derive(Debug, Clone, Default)]
pub struct MetricsSnapshot {
    pub uptime_secs: u64,
    pub sessions: usize,
    pub chat_runs_active: usize,
    pub memory_entries: Option<usize>,
    pub reminders_pending: Option<usize>,
    pub tools: Vec<ToolStatsSummary>,
    pub models: Vec<ModelStatsSummary>,
    pub mcp_servers: Vec<McpServerState>,
    pub channels: Vec<(String, crate::channel_counters::ChannelCount)>,
}

/// Escape a label value per the exposition format: backslash, quote, newline.
/// Pure.
fn label(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for c in value.chars() {
        match c {
            '\\' => escaped.push_str("\\\\"),
            '"' => escaped.push_str("\\\""),
            '\n' => escaped.push_str("\\n"),
            c => escaped.push(c),
        }
    }
    escaped
}

/// Append one metric family header.
fn family(out: &mut String, name: &str, kind: &str, help: &str) {
    debug_assert!(name.starts_with("nanna_"), "one namespace: {name}");
    debug_assert!(matches!(kind, "gauge" | "counter" | "histogram"));
    let _ = writeln!(out, "# HELP {name} {help}");
    let _ = writeln!(out, "# TYPE {name} {kind}");
}

fn render_daemon(out: &mut String, s: &MetricsSnapshot) {
    family(out, "nanna_up", "gauge", "1 while the daemon is serving.");
    out.push_str("nanna_up 1\n");
    family(
        out,
        "nanna_uptime_seconds",
        "gauge",
        "Seconds since the daemon started.",
    );
    let _ = writeln!(out, "nanna_uptime_seconds {}", s.uptime_secs);
    family(
        out,
        "nanna_sessions",
        "gauge",
        "Conversations held by the daemon.",
    );
    let _ = writeln!(out, "nanna_sessions {}", s.sessions);
    family(
        out,
        "nanna_chat_runs_active",
        "gauge",
        "Conversations with a turn in progress.",
    );
    let _ = writeln!(out, "nanna_chat_runs_active {}", s.chat_runs_active);
    if let Some(entries) = s.memory_entries {
        family(
            out,
            "nanna_memory_entries",
            "gauge",
            "Memories in the store.",
        );
        let _ = writeln!(out, "nanna_memory_entries {entries}");
    }
    if let Some(pending) = s.reminders_pending {
        family(
            out,
            "nanna_reminders_pending",
            "gauge",
            "Reminders not yet delivered.",
        );
        let _ = writeln!(out, "nanna_reminders_pending {pending}");
    }
}

fn render_tools(out: &mut String, tools: &[ToolStatsSummary]) {
    if tools.is_empty() {
        return;
    }
    family(
        out,
        "nanna_tool_calls_total",
        "counter",
        "Tool calls by outcome.",
    );
    for t in tools {
        let name = label(&t.name);
        for (outcome, count) in [
            ("success", t.success_count),
            ("failure", t.failure_count),
            ("short_circuit", t.short_circuit_count),
        ] {
            let _ = writeln!(
                out,
                "nanna_tool_calls_total{{tool=\"{name}\",outcome=\"{outcome}\"}} {count}"
            );
        }
    }
    family(
        out,
        "nanna_tool_latency_p95_milliseconds",
        "gauge",
        "95th percentile tool latency.",
    );
    for t in tools {
        let _ = writeln!(
            out,
            "nanna_tool_latency_p95_milliseconds{{tool=\"{}\"}} {}",
            label(&t.name),
            t.p95_latency_ms
        );
    }
    render_histogram(
        out,
        (
            "nanna_tool_duration_milliseconds",
            "Executed tool call latency since the daemon started.",
            "tool",
        ),
        &TOOL_LATENCY_BOUNDS_MS,
        tools.iter().map(|t| (t.name.as_str(), &t.latency_histogram)),
    );
}

/// One histogram family over `series` (label value, histogram): a cumulative
/// `_bucket` per bound plus `+Inf`, then `_sum` and `_count`, per series.
/// `_count` is the `+Inf` bucket, so the three always agree.
fn render_histogram<'a>(
    out: &mut String,
    (name, help, label_key): (&str, &str, &str),
    bounds: &[u64; LATENCY_BUCKETS],
    series: impl Iterator<Item = (&'a str, &'a LatencyHistogram)>,
) {
    family(out, name, "histogram", help);
    for (value, histogram) in series {
        let value = label(value);
        let cumulative = histogram.cumulative();
        for (bound, count) in bounds.iter().zip(cumulative) {
            let _ = writeln!(out, "{name}_bucket{{{label_key}=\"{value}\",le=\"{bound}\"}} {count}");
        }
        let total = cumulative[LATENCY_BUCKETS];
        debug_assert_eq!(total, histogram.count(), "+Inf is the count");
        let _ = writeln!(out, "{name}_bucket{{{label_key}=\"{value}\",le=\"+Inf\"}} {total}");
        let _ = writeln!(out, "{name}_sum{{{label_key}=\"{value}\"}} {}", histogram.sum_ms());
        let _ = writeln!(out, "{name}_count{{{label_key}=\"{value}\"}} {total}");
    }
}

fn render_models(out: &mut String, models: &[ModelStatsSummary]) {
    if models.is_empty() {
        return;
    }
    family(
        out,
        "nanna_model_requests_total",
        "counter",
        "Model requests made.",
    );
    for m in models {
        let _ = writeln!(
            out,
            "nanna_model_requests_total{{model=\"{}\"}} {}",
            label(&m.model),
            m.total_requests
        );
    }
    family(
        out,
        "nanna_model_tokens_total",
        "counter",
        "Tokens by direction.",
    );
    for m in models {
        let model = label(&m.model);
        for (kind, count) in [
            ("input", m.total_input_tokens),
            ("output", m.total_output_tokens),
            ("cache_read", m.total_cache_read_tokens),
            ("cache_write", m.total_cache_creation_tokens),
        ] {
            let _ = writeln!(
                out,
                "nanna_model_tokens_total{{model=\"{model}\",kind=\"{kind}\"}} {count}"
            );
        }
    }
    render_histogram(
        out,
        (
            "nanna_model_request_duration_milliseconds",
            "Model request latency since the daemon started, failures included.",
            "model",
        ),
        &MODEL_LATENCY_BOUNDS_MS,
        models.iter().map(|m| (m.model.as_str(), &m.latency_histogram)),
    );
    family(
        out,
        "nanna_model_healthy",
        "gauge",
        "1 when the model's recent requests are succeeding.",
    );
    for m in models {
        let _ = writeln!(
            out,
            "nanna_model_healthy{{model=\"{}\"}} {}",
            label(&m.model),
            u8::from(m.is_healthy)
        );
    }
}

fn render_mcp(out: &mut String, servers: &[McpServerState]) {
    let named: Vec<&McpServerState> = servers.iter().filter(|s| !s.name.is_empty()).collect();
    if named.is_empty() {
        return;
    }
    family(
        out,
        "nanna_mcp_server_up",
        "gauge",
        "1 when a configured MCP server started.",
    );
    for s in &named {
        let _ = writeln!(
            out,
            "nanna_mcp_server_up{{server=\"{}\",state=\"{}\"}} {}",
            label(&s.name),
            s.state,
            u8::from(s.state == "started")
        );
    }
    family(
        out,
        "nanna_mcp_server_tools",
        "gauge",
        "Tools registered from a configured MCP server.",
    );
    for s in &named {
        let _ = writeln!(
            out,
            "nanna_mcp_server_tools{{server=\"{}\"}} {}",
            label(&s.name),
            s.tools
        );
    }
}

fn render_channels(out: &mut String, channels: &[(String, crate::channel_counters::ChannelCount)]) {
    if channels.is_empty() {
        return;
    }
    family(
        out,
        "nanna_channel_messages_total",
        "counter",
        "Channel messages received and replies sent.",
    );
    for (name, count) in channels {
        let name = label(name);
        let _ = writeln!(
            out,
            "nanna_channel_messages_total{{channel=\"{name}\",direction=\"received\"}} {}",
            count.received
        );
        let _ = writeln!(
            out,
            "nanna_channel_messages_total{{channel=\"{name}\",direction=\"sent\"}} {}",
            count.sent
        );
    }
    family(
        out,
        "nanna_channel_send_failures_total",
        "counter",
        "Replies that could not be sent to a channel.",
    );
    for (name, count) in channels {
        let _ = writeln!(
            out,
            "nanna_channel_send_failures_total{{channel=\"{}\"}} {}",
            label(name),
            count.send_failures
        );
    }
}

/// Render a snapshot. Pure.
#[must_use]
pub fn render_metrics(snapshot: &MetricsSnapshot) -> String {
    let mut out = String::with_capacity(4096);
    render_daemon(&mut out, snapshot);
    render_tools(&mut out, &snapshot.tools);
    render_models(&mut out, &snapshot.models);
    render_mcp(&mut out, &snapshot.mcp_servers);
    render_channels(&mut out, &snapshot.channels);
    debug_assert!(out.ends_with('\n'), "the format is newline-terminated");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool(name: &str, ok: u64, failed: u64) -> ToolStatsSummary {
        serde_json::from_value(serde_json::json!({
            "name": name, "call_count": ok + failed, "success_count": ok, "failure_count": failed,
            "success_rate": 1.0, "avg_latency_ms": 5, "p50_latency_ms": 4, "p95_latency_ms": 9,
            "p99_latency_ms": 12, "avg_output_size": 100, "last_called": null, "top_errors": []
        }))
        .expect("summary shape")
    }

    #[test]
    fn a_bare_daemon_reports_only_what_it_has() {
        let text = render_metrics(&MetricsSnapshot {
            uptime_secs: 42,
            sessions: 3,
            ..Default::default()
        });
        assert!(
            text.contains("# TYPE nanna_up gauge\nnanna_up 1\n"),
            "{text}"
        );
        assert!(text.contains("nanna_uptime_seconds 42\n"));
        assert!(text.contains("nanna_sessions 3\n"));
        assert!(
            !text.contains("nanna_memory_entries"),
            "no memory, no series: {text}"
        );
        assert!(!text.contains("nanna_tool_calls_total"), "{text}");
    }

    #[test]
    fn tools_and_mcp_render_as_labelled_series() {
        let text = render_metrics(&MetricsSnapshot {
            tools: vec![tool("read_file", 7, 2)],
            mcp_servers: vec![
                McpServerState {
                    name: "files".into(),
                    state: "started",
                    tools: 4,
                    detail: None,
                    link: None,
                },
                McpServerState {
                    name: "git".into(),
                    state: "failed",
                    tools: 0,
                    detail: Some("x".into()),
                    link: None,
                },
                McpServerState {
                    name: String::new(),
                    state: "not_started",
                    tools: 0,
                    detail: Some("dup".into()),
                    link: None,
                },
            ],
            memory_entries: Some(10),
            channels: vec![(
                "telegram".into(),
                crate::channel_counters::ChannelCount {
                    received: 4,
                    sent: 3,
                    send_failures: 1,
                },
            )],
            ..Default::default()
        });
        assert!(
            text.contains(
                "nanna_channel_messages_total{channel=\"telegram\",direction=\"received\"} 4\n"
            ),
            "{text}"
        );
        assert!(
            text.contains("nanna_channel_send_failures_total{channel=\"telegram\"} 1\n"),
            "{text}"
        );
        assert!(
            text.contains("nanna_tool_calls_total{tool=\"read_file\",outcome=\"success\"} 7\n"),
            "{text}"
        );
        assert!(
            text.contains("nanna_tool_calls_total{tool=\"read_file\",outcome=\"failure\"} 2\n")
        );
        assert!(text.contains("nanna_tool_latency_p95_milliseconds{tool=\"read_file\"} 9\n"));
        assert!(text.contains("nanna_mcp_server_up{server=\"files\",state=\"started\"} 1\n"));
        assert!(text.contains("nanna_mcp_server_up{server=\"git\",state=\"failed\"} 0\n"));
        assert!(
            !text.contains("server=\"\""),
            "an unnamed skipped entry is not a series: {text}"
        );
        assert!(text.contains("nanna_memory_entries 10\n"));
        // Every sample line belongs to a declared family; a histogram's
        // `_bucket`/`_sum`/`_count` samples belong to its base name.
        for line in text.lines().filter(|l| !l.starts_with('#')) {
            let name = line.split(['{', ' ']).next().expect("name");
            let histogram_base = ["_bucket", "_sum", "_count"]
                .iter()
                .find_map(|suffix| name.strip_suffix(suffix))
                .filter(|base| text.contains(&format!("# TYPE {base} histogram\n")));
            assert!(
                histogram_base.is_some() || text.contains(&format!("# TYPE {name} ")),
                "undeclared series {name}"
            );
        }
    }

    #[test]
    fn label_values_are_escaped() {
        assert_eq!(label(r#"a"b\c"#), r#"a\"b\\c"#);
        assert_eq!(label("line\nbreak"), "line\\nbreak");
    }

    /// The histogram family is what Prometheus needs to compute quantiles
    /// across scrapes: cumulative buckets ending in `+Inf`, then `_sum` and a
    /// `_count` equal to the `+Inf` bucket.
    #[test]
    fn latencies_render_as_a_cumulative_histogram() {
        let mut read_file = tool("read_file", 3, 0);
        for ms in [8, 40, 400] {
            read_file
                .latency_histogram
                .observe(&TOOL_LATENCY_BOUNDS_MS, ms);
        }
        let text = render_metrics(&MetricsSnapshot {
            tools: vec![read_file],
            ..MetricsSnapshot::default()
        });
        assert!(text.contains("# TYPE nanna_tool_duration_milliseconds histogram\n"));
        for line in [
            "nanna_tool_duration_milliseconds_bucket{tool=\"read_file\",le=\"10\"} 1\n",
            "nanna_tool_duration_milliseconds_bucket{tool=\"read_file\",le=\"50\"} 2\n",
            "nanna_tool_duration_milliseconds_bucket{tool=\"read_file\",le=\"250\"} 2\n",
            "nanna_tool_duration_milliseconds_bucket{tool=\"read_file\",le=\"500\"} 3\n",
            "nanna_tool_duration_milliseconds_bucket{tool=\"read_file\",le=\"+Inf\"} 3\n",
            "nanna_tool_duration_milliseconds_sum{tool=\"read_file\"} 448\n",
            "nanna_tool_duration_milliseconds_count{tool=\"read_file\"} 3\n",
        ] {
            assert!(text.contains(line), "missing {line:?} in:\n{text}");
        }
        let buckets = text
            .lines()
            .filter(|l| l.starts_with("nanna_tool_duration_milliseconds_bucket"))
            .count();
        assert_eq!(buckets, LATENCY_BUCKETS + 1, "every bound plus +Inf, once");
    }
}
