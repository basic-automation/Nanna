//! Scheduler and cron-job commands. The daemon runs heartbeat + cron and owns
//! the job store; these forward to it.
//!
//! The three whole-scheduler toggles persist to `[scheduler]` in the shared
//! `config.toml` and then ask the daemon to reload, which re-applies them to
//! the running scheduler loop — no daemon restart. They were no-ops before,
//! which left the daemon's heartbeat unconditional: a heartbeat is a full agent
//! turn on the same local model chat uses, so it would steal the single Ollama
//! slot mid-conversation and cancel the in-flight generation.

use crate::state::{backend_handle, AppState};
use std::sync::Arc;
use tauri::State;
use tokio::sync::RwLock;
use tracing::{info, warn};

/// Apply `change` to the `[scheduler]` section, persist it, and make the
/// daemon adopt it live.
///
/// The lock covers the change and the save — the part that must be ordered —
/// and is released BEFORE the daemon round trip. Held across it, every other
/// command waited on `AppState` for as long as the daemon took to answer.
async fn update_scheduler_config(
    state: &RwLock<AppState>,
    change: impl FnOnce(&mut nanna_config::SchedulerConfig),
) -> Result<(), String> {
    let backend = {
        let mut guard = state.write().await;
        change(&mut guard.config.scheduler);
        guard.config.save().map_err(|e| {
            warn!("Failed to save scheduler settings to config: {e}");
            format!("Failed to save scheduler settings: {e}")
        })?;
        Arc::clone(&guard.backend)
    };
    // The daemon re-reads the file and pushes the section onto its live
    // scheduler. A failure here means the setting is saved but not yet in
    // effect, so surface it rather than reporting success.
    backend
        .config_reload()
        .await
        .map(|_| ())
        .map_err(|e| format!("Saved, but the daemon did not pick it up: {e}"))
}

/// Enable/disable the whole scheduler (heartbeat, cron, consolidation, sweeps).
///
/// # Errors
///
/// Returns `Failed to save scheduler settings: …` when `config.toml` cannot be
/// written, and `Saved, but the daemon did not pick it up: …` when the daemon
/// cannot be reached or the `config.reload` request is dropped or times out —
/// the setting is then on disk but not yet live.
#[tauri::command]
pub async fn set_scheduler_enabled(
    state: State<'_, Arc<RwLock<AppState>>>,
    enabled: bool,
) -> Result<(), String> {
    update_scheduler_config(&state, |scheduler| scheduler.enabled = enabled).await?;
    info!("Scheduler enabled: {enabled}");
    Ok(())
}

/// Enable/disable the periodic heartbeat.
///
/// # Errors
///
/// Returns `Failed to save scheduler settings: …` when `config.toml` cannot be
/// written, and `Saved, but the daemon did not pick it up: …` when the daemon
/// cannot be reached or the `config.reload` request is dropped or times out —
/// the setting is then on disk but not yet live.
#[tauri::command]
pub async fn set_heartbeat_enabled(
    state: State<'_, Arc<RwLock<AppState>>>,
    enabled: bool,
) -> Result<(), String> {
    update_scheduler_config(&state, |scheduler| scheduler.heartbeat_enabled = enabled).await?;
    info!("Heartbeat enabled: {enabled}");
    Ok(())
}

/// Set the heartbeat interval, in seconds.
///
/// # Errors
///
/// Rejects an interval below `nanna_core::MIN_HEARTBEAT_INTERVAL_SECS` without
/// saving anything. Otherwise fails as [`set_scheduler_enabled`] does.
#[tauri::command]
pub async fn set_heartbeat_interval(
    state: State<'_, Arc<RwLock<AppState>>>,
    seconds: u64,
) -> Result<(), String> {
    // Same floor the scheduler itself clamps to — reject here so the user gets
    // a message instead of a value that is silently raised.
    if seconds < nanna_core::MIN_HEARTBEAT_INTERVAL_SECS {
        return Err(format!(
            "Heartbeat interval must be at least {} seconds",
            nanna_core::MIN_HEARTBEAT_INTERVAL_SECS
        ));
    }
    update_scheduler_config(&state, |scheduler| {
        scheduler.heartbeat_interval_secs = seconds;
    })
    .await?;
    info!("Heartbeat interval: {seconds}s");
    Ok(())
}

// =============================================================================
// Scheduler / Cron Job Commands
// =============================================================================

/// Cron job info for the GUI
#[derive(Debug, Clone, serde::Serialize)]
pub struct CronJobInfo {
    pub id: String,
    pub name: String,
    pub schedule: String,
    pub schedule_description: String,
    pub payload: String,
    pub enabled: bool,
    pub last_run: Option<String>,
    pub next_run: Option<String>,
    pub run_count: u64,
    pub timezone: String,
}

/// Job run info for the GUI
#[derive(Debug, Clone, serde::Serialize)]
pub struct JobRunInfo {
    pub id: i64,
    pub job_id: String,
    pub started_at: String,
    pub finished_at: Option<String>,
    pub success: bool,
    pub output: Option<String>,
    pub error: Option<String>,
    pub duration_ms: Option<i64>,
}

/// Build a `CronJobInfo` from the daemon's `scheduler.list` job JSON.
pub(crate) fn cron_job_info_from_daemon(job: &serde_json::Value) -> Option<CronJobInfo> {
    let schedule = job.get("schedule").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let schedule_description = if schedule == "heartbeat" {
        "Periodic heartbeat".to_string()
    } else if let Ok(parsed) = nanna_core::CronExpr::parse(&schedule) {
        parsed.describe()
    } else {
        schedule.clone()
    };
    Some(CronJobInfo {
        id: job.get("id")?.as_str()?.to_string(),
        name: job.get("name").and_then(|v| v.as_str()).unwrap_or("unnamed").to_string(),
        schedule,
        schedule_description,
        payload: job.get("payload").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        enabled: job.get("enabled").and_then(serde_json::Value::as_bool).unwrap_or(true),
        last_run: job.get("last_run").and_then(|v| v.as_str()).map(str::to_string),
        next_run: job.get("next_run").and_then(|v| v.as_str()).map(str::to_string),
        run_count: job.get("run_count").and_then(serde_json::Value::as_u64).unwrap_or(0),
        timezone: job.get("timezone").and_then(|v| v.as_str()).unwrap_or("UTC").to_string(),
    })
}

/// Get all scheduled jobs.
///
/// # Errors
///
/// Fails when the daemon cannot be reached or the `scheduler.list` request is
/// dropped or times out. A reply without a `jobs` array lists nothing.
#[tauri::command]
pub async fn list_cron_jobs(
    state: State<'_, Arc<RwLock<AppState>>>,
) -> Result<Vec<CronJobInfo>, String> {
    let result = backend_handle(&state).await.scheduler_list().await?;
    let jobs = result
        .get("jobs")
        .and_then(|v| v.as_array())
        .map_or_default(|arr| arr.iter().filter_map(cron_job_info_from_daemon).collect());
    Ok(jobs)
}

/// Create a new cron job.
///
/// # Errors
///
/// Fails with the parser's message when `schedule` is not a valid cron
/// expression, before the daemon is contacted. Fails when the daemon cannot be
/// reached or the `scheduler.add` request is dropped or times out. Fails with
/// `Daemon failed to create cron job: …` when the reply reports an `error`, and
/// with `Daemon returned no job id` when it has no `id`.
#[tauri::command]
pub async fn create_cron_job(
    state: State<'_, Arc<RwLock<AppState>>>,
    name: String,
    schedule: String,
    payload: String,
    timezone: Option<String>,
) -> Result<CronJobInfo, String> {
    use nanna_core::CronExpr;

    // Validate the cron expression up-front so we fail fast with a good message.
    let parsed = CronExpr::parse(&schedule).map_err(|e| e.to_string())?;
    let next_run = parsed.next_from_now();

    let result = backend_handle(&state)
        .await
        .scheduler_add(&schedule, &payload, Some(&name))
        .await?;
    if result.get("error").is_some() {
        let msg = result.get("message").and_then(|v| v.as_str()).unwrap_or("unknown error");
        return Err(format!("Daemon failed to create cron job: {msg}"));
    }
    let id = result
        .get("id")
        .and_then(|v| v.as_str())
        .ok_or("Daemon returned no job id")?
        .to_string();

    Ok(CronJobInfo {
        id,
        name,
        schedule: schedule.clone(),
        schedule_description: parsed.describe(),
        payload,
        enabled: true,
        last_run: None,
        next_run: next_run.map(|dt| dt.to_rfc3339()),
        run_count: 0,
        // The daemon's add defaults to UTC; a custom timezone is not forwarded yet.
        timezone: timezone.unwrap_or_else(|| "UTC".to_string()),
    })
}

/// Update a cron job's schedule.
///
/// # Errors
///
/// Fails when the daemon cannot be reached or the `scheduler.update` request is
/// dropped or times out. Also fails with the daemon's `message` when its reply
/// reports an error other than `not_found`; an unknown job is `Ok(false)`.
#[tauri::command]
pub async fn update_cron_job(
    state: State<'_, Arc<RwLock<AppState>>>,
    job_id: String,
    schedule: String,
) -> Result<bool, String> {
    let result = backend_handle(&state)
        .await
        .scheduler_update(&job_id, Some(&schedule), None, None)
        .await?;
    match result.get("error").and_then(|v| v.as_str()) {
        None => Ok(true),
        Some("not_found") => Ok(false),
        Some(_) => Err(result
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown error")
            .to_string()),
    }
}

/// Enable or disable a cron job.
///
/// # Errors
///
/// Fails when the daemon cannot be reached or the `scheduler.update` request is
/// dropped or times out. A refusal the daemon reports in its reply (an unknown
/// id, say) is not checked and still returns `Ok`.
#[tauri::command]
pub async fn set_cron_job_enabled(
    state: State<'_, Arc<RwLock<AppState>>>,
    job_id: String,
    enabled: bool,
) -> Result<(), String> {
    backend_handle(&state)
        .await
        .scheduler_update(&job_id, None, None, Some(enabled))
        .await?;
    Ok(())
}

/// Delete a cron job.
///
/// # Errors
///
/// Fails when the daemon cannot be reached or the `scheduler.remove` request is
/// dropped or times out. A job the daemon does not report `deleted` is
/// `Ok(false)`.
#[tauri::command]
pub async fn delete_cron_job(
    state: State<'_, Arc<RwLock<AppState>>>,
    job_id: String,
) -> Result<bool, String> {
    let result = backend_handle(&state).await.scheduler_remove(&job_id).await?;
    Ok(result.get("status").and_then(|v| v.as_str()) == Some("deleted"))
}

/// Delete all cron jobs with a given name (useful for cleanup).
///
/// # Errors
///
/// Fails when the daemon cannot be reached or the `scheduler.list` request, or
/// any `scheduler.remove` that follows it, is dropped or times out. Removals
/// made before such a failure stand.
#[tauri::command]
pub async fn delete_cron_jobs_by_name(
    state: State<'_, Arc<RwLock<AppState>>>,
    name: String,
) -> Result<usize, String> {
    let backend = backend_handle(&state).await;
    // No by-name removal over IPC — list, filter, remove each.
    let result = backend.scheduler_list().await?;
    let ids: Vec<String> = result
        .get("jobs")
        .and_then(|v| v.as_array())
        .map_or_default(|arr| {
            arr.iter()
                .filter(|j| j.get("name").and_then(|v| v.as_str()) == Some(name.as_str()))
                .filter_map(|j| j.get("id").and_then(|v| v.as_str()).map(str::to_string))
                .collect()
        });
    let mut removed = 0;
    for id in &ids {
        let result = backend.scheduler_remove(id).await?;
        if result.get("status").and_then(|v| v.as_str()) == Some("deleted") {
            removed += 1;
        }
    }
    Ok(removed)
}

/// Run a cron job immediately.
///
/// # Errors
///
/// Fails when the daemon cannot be reached or the `scheduler.run_now` request
/// is dropped or times out. A refusal reported in the reply is `Ok(None)`.
#[tauri::command]
pub async fn run_cron_job_now(
    state: State<'_, Arc<RwLock<AppState>>>,
    job_id: String,
) -> Result<Option<JobRunInfo>, String> {
    let result = backend_handle(&state).await.scheduler_run_now(&job_id).await?;
    if result.get("error").is_some() {
        return Ok(None);
    }
    let now = chrono::Utc::now().to_rfc3339();
    Ok(Some(JobRunInfo {
        id: 0,
        job_id,
        started_at: now.clone(),
        finished_at: Some(now),
        success: result.get("status").and_then(|v| v.as_str()) == Some("success"),
        output: result.get("output").and_then(|v| v.as_str()).map(str::to_string),
        error: result.get("error").and_then(|v| v.as_str()).map(str::to_string),
        duration_ms: result.get("duration_ms").and_then(serde_json::Value::as_i64),
    }))
}

/// Get job run history.
///
/// # Errors
///
/// Fails when the daemon cannot be reached or the `scheduler.history` request
/// is dropped or times out. A reply without a `history` array is an empty
/// history.
#[tauri::command]
pub async fn get_cron_job_history(
    state: State<'_, Arc<RwLock<AppState>>>,
    job_id: String,
    limit: Option<usize>,
) -> Result<Vec<JobRunInfo>, String> {
    let result = backend_handle(&state).await.scheduler_history(&job_id, limit).await?;
    let runs = result
        .get("history")
        .and_then(|v| v.as_array())
        .map_or_default(|arr| {
            arr.iter()
                .map(|r| JobRunInfo {
                    id: r.get("run_id").and_then(serde_json::Value::as_i64).unwrap_or(0),
                    job_id: r.get("job_id").and_then(|v| v.as_str()).unwrap_or_default().to_string(),
                    started_at: r.get("started_at").and_then(|v| v.as_str()).unwrap_or_default().to_string(),
                    finished_at: r.get("finished_at").and_then(|v| v.as_str()).map(str::to_string),
                    success: r.get("success").and_then(serde_json::Value::as_bool).unwrap_or(false),
                    output: r.get("output").and_then(|v| v.as_str()).map(str::to_string),
                    error: r.get("error").and_then(|v| v.as_str()).map(str::to_string),
                    duration_ms: None,
                })
                .collect()
        });
    Ok(runs)
}

/// Validate a cron expression.
///
/// # Errors
///
/// Never returns `Err`: an invalid expression is `Ok((false, reason))`.
#[tauri::command]
pub async fn validate_cron_expression(expression: String) -> Result<(bool, String), String> {
    use nanna_core::CronExpr;

    match CronExpr::parse(&expression) {
        Ok(parsed) => {
            let description = parsed.describe();
            let next = parsed
                .next_from_now().map_or_else(|| "N/A".to_string(), |dt| dt.format("%Y-%m-%d %H:%M").to_string());
            Ok((true, format!("{description} (next: {next})")))
        }
        Err(e) => Ok((false, e.to_string())),
    }
}
