//! Window, notification, stats, and lifecycle commands.

use crate::backend::{BackendMode, BackendStatus};
use crate::daemon_manager::BootLine;
use crate::state::{backend_handle, AppState, CloseMode, ModelStatusEvent};
use nanna_core::log_buffer::LogEntry;
use std::sync::Arc;
use tauri::{AppHandle, Manager, State};
use tokio::sync::RwLock;
use tracing::{info, warn};

/// Hide the main window to tray
///
/// # Errors
///
/// Returns the window error's text when hiding the main window fails. A missing
/// main window is not an error.
#[tauri::command]
pub async fn hide_to_tray(app: AppHandle) -> Result<(), String> {
    if let Some(window) = app.get_webview_window("main") {
        window.hide().map_err(|e| e.to_string())?;
    }
    Ok(())
}

// =============================================================================
// Notification Commands
// =============================================================================

// =============================================================================
// Model Status Commands
// =============================================================================

/// Get current model status (active model, rate-limited models)
///
/// # Errors
///
/// Never returns `Err`; the `Result` is what Tauri requires of an async command
/// that borrows `State`.
#[tauri::command]
pub async fn get_model_status(
    state: State<'_, Arc<RwLock<AppState>>>,
) -> Result<ModelStatusEvent, String> {
    let state_guard = state.read().await;
    let active = state_guard.active_model.read().await.clone();
    let rate_limited = state_guard.rate_limited_models.read().await;

    // Filter to only models that are still rate limited
    let now = chrono::Utc::now().timestamp();
    let still_limited: Vec<String> = rate_limited
        .iter()
        .filter(|(_, until)| now < **until)
        .map(|(model, _)| model.clone())
        .collect();
    drop(rate_limited);
    drop(state_guard);

    Ok(ModelStatusEvent {
        active_model: active,
        fallback_reason: None,
        rate_limited_models: still_limited,
    })
}

/// Get detailed model performance statistics
///
/// # Errors
///
/// Fails when the daemon passes the connection check but the
/// `system.model_stats` request then fails in transport (the connection
/// dropped, or the request timed out). Without a connected daemon it returns an
/// empty placeholder, not an error.
#[tauri::command]
pub async fn get_model_stats(
    state: State<'_, Arc<RwLock<AppState>>>,
) -> Result<serde_json::Value, String> {
    // Try daemon mode first
    let backend = backend_handle(&state).await;
    let status = backend.status().await;
    if status.connected {
        return backend.daemon_request(serde_json::json!({
            "type": "system",
            "action": "model_stats"
        })).await;
    }

    // Embedded mode: no model stats tracker available
    Ok(serde_json::json!({
        "models": [],
        "note": "Model stats are only available in daemon mode"
    }))
}

/// Estimated spend per day over the last `days` days (`system.cost_rollup`).
///
/// # Errors
///
/// Fails when there is no daemon connection or the `system.cost_rollup`
/// request is dropped or times out, and with the daemon's own `message` when
/// it refuses — no request log in storage, or a rollup that failed to read it.
#[tauri::command]
pub async fn get_cost_rollup(
    state: State<'_, Arc<RwLock<AppState>>>,
    days: Option<u32>,
) -> Result<serde_json::Value, String> {
    let state_guard = state.read().await;
    let result = state_guard
        .backend
        .daemon_request(serde_json::json!({
            "type": "system",
            "action": "cost_rollup",
            "days": days,
            "by": "day"
        }))
        .await?;
    drop(state_guard);
    if result.get("error").is_some() {
        return Err(result["message"]
            .as_str()
            .unwrap_or("Unknown error")
            .to_string());
    }
    Ok(result)
}

/// Get per-tool performance statistics
///
/// # Errors
///
/// Fails when the daemon passes the connection check but the
/// `system.tool_stats` request then fails in transport (the connection dropped,
/// or the request timed out). Without a connected daemon it returns an empty
/// placeholder, not an error.
#[tauri::command]
pub async fn get_tool_stats(
    state: State<'_, Arc<RwLock<AppState>>>,
) -> Result<serde_json::Value, String> {
    let backend = backend_handle(&state).await;
    let status = backend.status().await;
    if status.connected {
        let result = backend.daemon_request(serde_json::json!({
            "type": "system",
            "action": "tool_stats"
        })).await;
        info!("📊 get_tool_stats: daemon responded with {} tools",
            result.as_ref().ok()
                .and_then(|v| v.get("tools"))
                .and_then(|v| v.as_array())
                .map_or(0, std::vec::Vec::len));
        return result;
    }

    warn!("📊 get_tool_stats: NOT CONNECTED (mode={:?}, daemon_state={})",
        status.mode, status.daemon_state);
    Ok(serde_json::json!({
        "tools": [],
        "note": "Tool stats are only available in daemon mode"
    }))
}

/// Get global tool + session dashboard stats
///
/// # Errors
///
/// Fails when the daemon passes the connection check but the
/// `system.global_stats` request then fails in transport (the connection
/// dropped, or the request timed out). Without a connected daemon it returns
/// zeroed placeholder totals, not an error.
#[tauri::command]
pub async fn get_global_stats(
    state: State<'_, Arc<RwLock<AppState>>>,
) -> Result<serde_json::Value, String> {
    let backend = backend_handle(&state).await;
    let status = backend.status().await;
    if status.connected {
        let result = backend.daemon_request(serde_json::json!({
            "type": "system",
            "action": "global_stats"
        })).await;
        info!("📊 get_global_stats: daemon responded, total_calls={}",
            result.as_ref().ok()
                .and_then(|v| v.get("total_calls"))
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0));
        return result;
    }

    warn!("📊 get_global_stats: NOT CONNECTED (mode={:?}, daemon_state={})",
        status.mode, status.daemon_state);
    Ok(serde_json::json!({
        "total_calls": 0,
        "avg_latency_ms": 0,
        "success_rate": 1.0,
        "slowest_tools": [],
        "most_used_tools": [],
        "most_failed_tools": [],
        "session_totals": {
            "total_iterations": 0,
            "total_tool_calls": 0,
            "total_tool_time_ms": 0,
            "total_llm_time_ms": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0
        },
        "note": "Global stats are only available in daemon mode"
    }))
}

/// Get hourly tool stats time-series for graphs
///
/// # Errors
///
/// Fails when the daemon passes the connection check but the
/// `system.tool_stats_hourly` request then fails in transport (the connection
/// dropped, or the request timed out). Without a connected daemon it returns no
/// buckets, not an error.
#[tauri::command]
pub async fn get_tool_stats_hourly(
    state: State<'_, Arc<RwLock<AppState>>>,
    tool_name: Option<String>,
    hours: Option<u32>,
) -> Result<serde_json::Value, String> {
    let backend = backend_handle(&state).await;
    let status = backend.status().await;
    if status.connected {
        return backend.daemon_request(serde_json::json!({
            "type": "system",
            "action": "tool_stats_hourly",
            "tool_name": tool_name,
            "hours": hours.unwrap_or(24)
        })).await;
    }
    warn!("📊 get_tool_stats_hourly: NOT CONNECTED (mode={:?}, daemon_state={})",
        status.mode, status.daemon_state);
    Ok(serde_json::json!({ "buckets": [] }))
}

/// Get daily tool stats time-series for graphs
///
/// # Errors
///
/// Fails when the daemon passes the connection check but the
/// `system.tool_stats_daily` request then fails in transport (the connection
/// dropped, or the request timed out). Without a connected daemon it returns no
/// buckets, not an error.
#[tauri::command]
pub async fn get_tool_stats_daily(
    state: State<'_, Arc<RwLock<AppState>>>,
    tool_name: Option<String>,
    days: Option<u32>,
) -> Result<serde_json::Value, String> {
    let backend = backend_handle(&state).await;
    let status = backend.status().await;
    if status.connected {
        return backend.daemon_request(serde_json::json!({
            "type": "system",
            "action": "tool_stats_daily",
            "tool_name": tool_name,
            "days": days.unwrap_or(30)
        })).await;
    }
    warn!("📊 get_tool_stats_daily: NOT CONNECTED (mode={:?}, daemon_state={})",
        status.mode, status.daemon_state);
    Ok(serde_json::json!({ "buckets": [] }))
}

/// Get recent tool call log entries
///
/// # Errors
///
/// Fails when the daemon passes the connection check but the
/// `system.tool_call_log` request then fails in transport (the connection
/// dropped, or the request timed out). Without a connected daemon it returns no
/// entries, not an error.
#[tauri::command]
pub async fn get_tool_call_log(
    state: State<'_, Arc<RwLock<AppState>>>,
    tool_name: Option<String>,
    limit: Option<u32>,
) -> Result<serde_json::Value, String> {
    let backend = backend_handle(&state).await;
    let status = backend.status().await;
    if status.connected {
        return backend.daemon_request(serde_json::json!({
            "type": "system",
            "action": "tool_call_log",
            "tool_name": tool_name,
            "limit": limit.unwrap_or(50)
        })).await;
    }
    Ok(serde_json::json!({ "entries": [] }))
}

// =============================================================================
// Backend Mode Commands
// =============================================================================

/// Get current backend status (daemon or embedded mode)
///
/// # Errors
///
/// Never returns `Err`; the `Result` is what Tauri requires of an async command
/// that borrows `State`.
#[tauri::command]
pub async fn get_backend_status(
    state: State<'_, Arc<RwLock<AppState>>>,
) -> Result<BackendStatus, String> {
    Ok(backend_handle(&state).await.status().await)
}

/// The version the RUNNING daemon reports about itself.
///
/// Deliberately not `BackendStatus::version`, which is this GUI binary's own
/// `CARGO_PKG_VERSION` — it describes the process asking, not the process
/// answering. The two agree only when both were built from the same tree, and
/// the moment they diverge is exactly when the daemon's version is worth
/// showing: a packaged GUI talking to a separately-built daemon reports the
/// GUI's number for both and looks consistent while being wrong.
///
/// `None` means no daemon is connected to ask — a distinct state from "asked
/// and it did not say", which surfaces as an error.
///
/// # Errors
///
/// Fails when the daemon passes the connection check but the `system.version`
/// request then fails in transport — which is also how a daemon too old to know
/// that request answers, after the full request timeout, because its
/// parse-error reply carries no request id. A reply without a `version` string
/// is `Ok(None)`.
#[tauri::command]
pub async fn get_daemon_version(
    state: State<'_, Arc<RwLock<AppState>>>,
) -> Result<Option<String>, String> {
    let backend = backend_handle(&state).await;
    if !backend.status().await.connected {
        return Ok(None);
    }
    let response = backend
        .daemon_request(serde_json::json!({
            "type": "system",
            "action": "version"
        }))
        .await?;
    Ok(response
        .get("version")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string))
}

/// Initialize the backend - starts the daemon sidecar and connects.
///
/// # Errors
///
/// Never returns `Err`: a daemon that cannot be started or reached is reported
/// as `"disconnected"`.
#[tauri::command]
pub async fn init_backend(
    app: AppHandle,
    state: State<'_, Arc<RwLock<AppState>>>,
) -> Result<String, String> {
    let backend = backend_handle(&state).await;
    if backend.is_connected().await {
        return Ok("daemon".to_string());
    }
    let mode = backend.init(&app).await;
    Ok(match mode {
        BackendMode::Daemon => "daemon".to_string(),
        BackendMode::Disconnected => "disconnected".to_string(),
    })
}

/// Restart the daemon, a live but hung boot included, and return once a
/// fresh start has begun; poll `get_backend_status` for its progress (see
/// `Backend::restart`).
///
/// # Errors
///
/// Never returns `Err`; the `Result` is what Tauri requires of an async command
/// that borrows `State`. A start that fails shows in the status's
/// `last_error`.
#[tauri::command]
pub async fn restart_daemon(
    app: AppHandle,
    state: State<'_, Arc<RwLock<AppState>>>,
) -> Result<(), String> {
    backend_handle(&state).await.restart(app).await;
    Ok(())
}

/// The output lines of the current (or most recent) daemon sidecar, oldest
/// first: what the daemon printed while it booted, up to its exit or its
/// newest line. Works before any daemon answers.
///
/// # Errors
///
/// Never returns `Err`; the `Result` is what Tauri requires of an async command
/// that borrows `State`.
#[tauri::command]
pub async fn get_boot_log(
    state: State<'_, Arc<RwLock<AppState>>>,
) -> Result<Vec<BootLine>, String> {
    Ok(backend_handle(&state).await.boot_log().await)
}

/// How many lines a single `get_daemon_logs` call may return.
///
/// Bounds the response regardless of what the caller asks for: the page requests
/// 1000, and each buffer holds at most 5000, so a merged view can never exceed
/// this. Prevents an unbounded `limit` from serialising both whole buffers.
const MAX_LOG_LINES: usize = 2000;

/// Default when the caller does not specify a limit (what the Logs page asks for).
const DEFAULT_LOG_LINES: usize = 1000;

/// Merge log lines from every source that currently has any, newest last.
///
/// Both origins can be live at once: in daemon mode the GUI still emits its own
/// in-process lines while the daemon emits its own, so this returns the union with
/// each entry tagged (`source: "embedded" | "daemon"`) rather than picking one.
/// In embedded mode there is no daemon to ask, and the answer is just our buffer —
/// which is the whole reason this page used to be empty.
///
/// # Errors
///
/// Fails when a daemon is connected but the `system.logs` request fails in
/// transport (the connection dropped, or the request timed out); this process's
/// own lines are not returned then either.
#[tauri::command]
pub async fn get_daemon_logs(
    state: State<'_, Arc<RwLock<AppState>>>,
    limit: Option<usize>,
) -> Result<Vec<serde_json::Value>, String> {
    let limit = limit.unwrap_or(DEFAULT_LOG_LINES).min(MAX_LOG_LINES);
    let (backend, log_buffer) = {
        let state_guard = state.read().await;
        (Arc::clone(&state_guard.backend), state_guard.log_buffer.clone())
    };

    let mut entries: Vec<LogEntry> = Vec::new();

    // Daemon lines, if a daemon is attached. A daemon that predates the `source`
    // field sends untagged entries; serde defaults those to `daemon`, which is
    // where they came from. A malformed entry is skipped rather than failing the
    // whole page.
    if backend.is_connected().await {
        let raw = backend.get_logs(Some(limit)).await?;
        entries.extend(
            raw.into_iter()
                .filter_map(|value| serde_json::from_value::<LogEntry>(value).ok()),
        );
    }

    // This process's own lines — present in both modes.
    entries.extend(log_buffer.get_recent(limit));

    Ok(merge_log_entries(entries, limit))
}

/// Order merged entries oldest-first and keep the newest `limit`.
///
/// Timestamps are fixed-width local time (`%Y-%m-%d %H:%M:%S%.3f`) written by the
/// same `LogBufferLayer` on both sides, so they sort lexicographically. The daemon
/// is a process on this same machine, hence the same clock and timezone; this would
/// need real timestamps if logs ever arrived from another host.
fn merge_log_entries(mut entries: Vec<LogEntry>, limit: usize) -> Vec<serde_json::Value> {
    entries.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

    let start = entries.len().saturating_sub(limit);
    let merged: Vec<serde_json::Value> = entries[start..]
        .iter()
        .filter_map(|entry| serde_json::to_value(entry).ok())
        .collect();

    debug_assert!(merged.len() <= limit, "merged log view must respect the limit");
    merged
}

// =============================================================================
// Window Close Behavior
// =============================================================================

/// Get current close mode preference
///
/// # Errors
///
/// Never returns `Err`; the `Result` is what Tauri requires of an async command
/// that borrows `State`.
#[tauri::command]
pub async fn get_close_mode(
    state: State<'_, Arc<RwLock<AppState>>>,
) -> Result<String, String> {
    let mode = *state.read().await.close_mode.read().await;
    Ok(match mode {
        CloseMode::Ask => "ask".to_string(),
        CloseMode::MinimizeToTray => "minimize_to_tray".to_string(),
        CloseMode::QuitCompletely => "quit_completely".to_string(),
    })
}

/// Set close mode preference
///
/// # Errors
///
/// Returns `Unknown close mode: …` for anything but `ask`, `minimize_to_tray`
/// or `quit_completely`.
#[tauri::command]
pub async fn set_close_mode(
    state: State<'_, Arc<RwLock<AppState>>>,
    mode: String,
) -> Result<(), String> {
    let close_mode = match mode.as_str() {
        "ask" => CloseMode::Ask,
        "minimize_to_tray" => CloseMode::MinimizeToTray,
        "quit_completely" => CloseMode::QuitCompletely,
        _ => return Err(format!("Unknown close mode: {mode}")),
    };

    let state = state.read().await;
    *state.close_mode.write().await = close_mode;
    info!("Close mode set to: {:?}", close_mode);
    Ok(())
}

/// Handle window close - returns what action to take
/// Called from frontend before actual close
///
/// # Errors
///
/// Never returns `Err`; the `Result` is what Tauri requires of an async command
/// that borrows `State`. Hiding the window to the tray is best-effort.
#[tauri::command]
pub async fn handle_window_close(
    app: AppHandle,
    state: State<'_, Arc<RwLock<AppState>>>,
) -> Result<String, String> {
    let mode = *state.read().await.close_mode.read().await;

    match mode {
        CloseMode::Ask => {
            // Frontend should show dialog
            Ok("ask".to_string())
        }
        CloseMode::MinimizeToTray => {
            // Hide window to tray
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.hide();
            }
            Ok("minimized".to_string())
        }
        CloseMode::QuitCompletely => {
            // Will trigger actual exit
            Ok("quit".to_string())
        }
    }
}

/// Stop the daemon before an update is installed.
///
/// Installing ends this process without an exit event: Windows' installer
/// exits the app itself, and elsewhere the frontend calls `relaunch()`, which
/// restarts without running the `ExitRequested` hook that stops the sidecar.
/// A daemon left running keeps the port and the database lock, so the updated
/// app attached to the old server instead of starting its own (2026-09-17).
/// Called after the download succeeds, so a failed download never costs the
/// user their daemon; a failed install is recovered with `init_backend`.
///
/// # Errors
///
/// Never returns `Err`; the `Result` is what Tauri requires of an async command
/// that borrows `State`. A sidecar daemon that does not stop gracefully is
/// killed, and failures along the way are only logged.
#[tauri::command]
pub async fn stop_backend_for_update(
    state: State<'_, Arc<RwLock<AppState>>>,
) -> Result<(), String> {
    info!("Stopping the daemon before installing an update...");
    backend_handle(&state).await.shutdown().await;
    Ok(())
}

/// Perform actual quit (called after user confirms or preference is quit)
///
/// # Errors
///
/// Never returns `Err`; the `Result` is what Tauri requires of an async command
/// that borrows `State`.
#[tauri::command]
pub async fn perform_quit(
    app: AppHandle,
    state: State<'_, Arc<RwLock<AppState>>>,
) -> Result<(), String> {
    let backend = backend_handle(&state).await;

    // Stop the daemon sidecar. Memory/session persistence is the daemon's job.
    info!("Performing quit - shutting down backend...");
    backend.shutdown().await;

    app.exit(0);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nanna_core::log_buffer::LogSource;

    fn entry(timestamp: &str, source: LogSource, message: &str) -> LogEntry {
        LogEntry {
            timestamp: timestamp.to_string(),
            level: "info".to_string(),
            target: "test".to_string(),
            message: message.to_string(),
            source,
            scope: String::new(),
        }
    }

    fn messages(values: &[serde_json::Value]) -> Vec<String> {
        values
            .iter()
            .filter_map(|v| v.get("message")?.as_str().map(String::from))
            .collect()
    }

    /// The point of the merge: both origins interleave by time, so a daemon line
    /// emitted between two GUI lines lands between them rather than in a block.
    #[test]
    fn merges_both_sources_in_timestamp_order() {
        let merged = merge_log_entries(
            vec![
                entry("2024-01-01 12:00:03.000", LogSource::Embedded, "gui-late"),
                entry("2024-01-01 12:00:01.000", LogSource::Embedded, "gui-early"),
                entry("2024-01-01 12:00:02.000", LogSource::Daemon, "daemon-mid"),
            ],
            10,
        );

        assert_eq!(messages(&merged), ["gui-early", "daemon-mid", "gui-late"]);
    }

    /// Each line must carry its own origin — that is what makes a merged view
    /// readable when both are live.
    #[test]
    fn each_entry_keeps_its_source_tag() {
        let merged = merge_log_entries(
            vec![
                entry("2024-01-01 12:00:01.000", LogSource::Embedded, "a"),
                entry("2024-01-01 12:00:02.000", LogSource::Daemon, "b"),
            ],
            10,
        );

        assert_eq!(merged[0].get("source").and_then(|v| v.as_str()), Some("embedded"));
        assert_eq!(merged[1].get("source").and_then(|v| v.as_str()), Some("daemon"));
    }

    /// Over-limit merges keep the NEWEST lines, not the first N — a log tail is
    /// useless if it drops the most recent events.
    #[test]
    fn over_limit_keeps_the_newest_lines() {
        let entries = (0..10)
            .map(|i| {
                entry(
                    &format!("2024-01-01 12:00:{i:02}.000"),
                    LogSource::Daemon,
                    &format!("m{i}"),
                )
            })
            .collect();

        let merged = merge_log_entries(entries, 3);

        assert_eq!(messages(&merged), ["m7", "m8", "m9"]);
    }

    #[test]
    fn under_limit_returns_everything_and_empty_stays_empty() {
        let merged = merge_log_entries(
            vec![entry("2024-01-01 12:00:01.000", LogSource::Embedded, "only")],
            50,
        );
        assert_eq!(messages(&merged), ["only"]);

        assert_eq!(merge_log_entries(Vec::new(), 50), [] as [serde_json::Value; 0]);
    }

    /// A zero limit must not panic on the slice arithmetic.
    #[test]
    fn zero_limit_yields_no_lines() {
        let merged = merge_log_entries(
            vec![entry("2024-01-01 12:00:01.000", LogSource::Daemon, "x")],
            0,
        );
        assert_eq!(merged, [] as [serde_json::Value; 0]);
    }
}
