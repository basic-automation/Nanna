//! User-tool and skill authoring commands.
//!
//! User tools live in the daemon (it owns the registry + `user_tools` dir), so
//! CRUD forwards over IPC. Skills are files under the active workspace's
//! `skills/` directory and are edited directly on disk here; the daemon loads
//! them from its `tools_dir` at startup.

use crate::commands::settings::ToolInfo;
use crate::state::{backend_handle, AppState};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tauri::State;
use tokio::sync::RwLock;
use tracing::info;

// =============================================================================
// User tool metadata (self-contained; mirrors the daemon's user-tool JSON)
// =============================================================================

const fn default_true() -> bool {
    true
}
fn default_language() -> String {
    "typescript".to_string()
}

/// Permissions block for a user tool (matches the daemon's serialized shape).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UserToolPermissions {
    #[serde(default)]
    pub net: Vec<String>,
    #[serde(default)]
    pub read: Vec<String>,
    #[serde(default)]
    pub write: Vec<String>,
    #[serde(default)]
    pub env: bool,
    #[serde(default)]
    pub run: bool,
}

/// User-created tool metadata as surfaced to the frontend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserToolMeta {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub source: String,
    #[serde(default = "default_language")]
    pub language: String,
    #[serde(default)]
    pub parameters: Option<serde_json::Value>,
    #[serde(default)]
    pub permissions: UserToolPermissions,
    #[serde(default)]
    pub created_at: i64,
    #[serde(default)]
    pub updated_at: i64,
    #[serde(default = "default_true")]
    pub enabled: bool,
}

/// Parse the daemon's user-tool list (`{tools: [...]}`) into `UserToolMeta`.
fn parse_user_tools(result: &serde_json::Value) -> Result<Vec<UserToolMeta>, String> {
    serde_json::from_value(result.get("tools").cloned().unwrap_or(serde_json::json!([])))
        .map_err(|e| format!("Failed to parse daemon response: {e}"))
}

/// Look up one user-authored tool by name.
///
/// # Errors
///
/// Fails when the daemon cannot be reached or the `tool.list_user` request is
/// dropped or times out. Fails with `Failed to parse daemon response: …` when
/// the reply's `tools` entries do not match [`UserToolMeta`]; a reply without
/// `tools` lists none. An unknown name is `Ok(None)`.
#[tauri::command]
pub async fn get_user_tool(
    state: State<'_, Arc<RwLock<AppState>>>,
    name: String,
) -> Result<Option<UserToolMeta>, String> {
    let result = backend_handle(&state).await.tool_list_user().await?;
    let tools = parse_user_tools(&result)?;
    Ok(tools.into_iter().find(|t| t.name == name))
}

/// Fetch a tool's source code from the daemon.
///
/// # Errors
///
/// Fails when the daemon cannot be reached or the `tool.get_source` request is
/// dropped or times out. A refusal the daemon reports in its reply is passed
/// through inside `Ok`.
#[tauri::command]
pub async fn get_tool_source(
    state: State<'_, Arc<RwLock<AppState>>>,
    name: String,
) -> Result<serde_json::Value, String> {
    backend_handle(&state).await.tool_get_source(&name).await
}

/// Create a user-authored tool from its source.
///
/// # Errors
///
/// Fails when the daemon cannot be reached or the `tool.create` request is
/// dropped or times out. Fails with `Failed to parse daemon response: …` when
/// neither the reply's `tool` object nor the reply itself parses as
/// [`UserToolMeta`] — which is how a refusal reported in the reply surfaces.
#[tauri::command]
pub async fn create_user_tool(
    state: State<'_, Arc<RwLock<AppState>>>,
    name: String,
    description: String,
    source: String,
    language: Option<String>,
    parameters: Option<serde_json::Value>,
) -> Result<UserToolMeta, String> {
    let _ = (language, parameters); // daemon derives language/params from source
    // Daemon tool_create uses (name, description, code, needs_shell).
    let result = backend_handle(&state).await.tool_create(&name, &description, &source, None).await?;
    let tool = result.get("tool").cloned().unwrap_or(result);
    serde_json::from_value(tool).map_err(|e| format!("Failed to parse daemon response: {e}"))
}

/// Update a user-authored tool's description and/or source.
///
/// # Errors
///
/// Fails when the daemon cannot be reached or the `tool.update` request is
/// dropped or times out. Fails with `Failed to parse daemon response: …` when
/// neither the reply's `tool` object nor the reply itself parses as
/// [`UserToolMeta`] — which is how a refusal reported in the reply surfaces.
#[tauri::command]
pub async fn update_user_tool(
    state: State<'_, Arc<RwLock<AppState>>>,
    name: String,
    description: Option<String>,
    source: Option<String>,
    parameters: Option<serde_json::Value>,
    enabled: Option<bool>,
) -> Result<UserToolMeta, String> {
    let _ = (parameters, enabled); // not exposed over the daemon tool_update action
    let result = backend_handle(&state)
        .await
        .tool_update(&name, description.as_deref(), source.as_deref(), None)
        .await?;
    let tool = result.get("tool").cloned().unwrap_or(result);
    serde_json::from_value(tool).map_err(|e| format!("Failed to parse daemon response: {e}"))
}

/// Delete a user-authored tool.
///
/// # Errors
///
/// Fails when the daemon cannot be reached or the `tool.delete` request is
/// dropped or times out. A refusal the daemon reports in its reply (an unknown
/// name, say) is not checked and still returns `Ok`.
#[tauri::command]
pub async fn delete_user_tool(
    state: State<'_, Arc<RwLock<AppState>>>,
    name: String,
) -> Result<(), String> {
    backend_handle(&state).await.tool_delete(&name).await?;
    Ok(())
}

// =============================================================================
// Tool Listing Commands (all registered tools)
// =============================================================================

/// List all registered tools.
///
/// # Errors
///
/// Fails when the daemon cannot be reached or the `tool.list` request is
/// dropped or times out. Fails with `Failed to fetch tools from daemon` when
/// the reply has no `tools` array.
#[tauri::command]
pub async fn list_tools(
    state: State<'_, Arc<RwLock<AppState>>>,
) -> Result<Vec<ToolInfo>, String> {
    let result = backend_handle(&state).await.tool_list().await?;
    let tools = crate::commands::settings::tool_infos(&result)
        .ok_or("Failed to fetch tools from daemon")?;
    Ok(tools)
}

/// Enable or disable a tool.
///
/// Covers bundled skills and user tools alike: the daemon dispatches on which
/// store owns the name, and canonicalizes aliases before touching the policy,
/// so `Bash` toggles `exec` rather than writing a denylist entry that gates
/// nothing.
///
/// The daemon answers with a JSON body rather than an HTTP-style status, so a
/// refusal arrives as `{"error": ...}` with a 200-equivalent envelope. Surface
/// it as `Err` — a toggle that silently fails to move is the failure mode this
/// whole path exists to avoid, and the switch must snap back rather than lie.
///
/// # Errors
///
/// Fails when the daemon cannot be reached or the `tool.enable` /
/// `tool.disable` request is dropped or times out. Fails with `Failed to enable
/// '<name>': …` (or `disable`) when the daemon refuses the toggle.
#[tauri::command]
pub async fn set_tool_enabled(
    state: State<'_, Arc<RwLock<AppState>>>,
    name: String,
    enabled: bool,
) -> Result<(), String> {
    let result = backend_handle(&state).await.tool_set_enabled(&name, enabled).await?;

    if let Some(err) = result.get("error").and_then(|v| v.as_str()) {
        let detail = result
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or(err);
        return Err(format!("Failed to {} '{name}': {detail}", if enabled { "enable" } else { "disable" }));
    }

    info!("Tool '{name}' {}", if enabled { "enabled" } else { "disabled" });
    Ok(())
}

/// Get details of a specific tool.
///
/// # Errors
///
/// Fails when the daemon cannot be reached or the `tool.get` request is dropped
/// or times out. A refusal the daemon reports in its reply is passed through
/// inside `Ok`.
#[tauri::command]
pub async fn get_tool(
    state: State<'_, Arc<RwLock<AppState>>>,
    name: String,
) -> Result<serde_json::Value, String> {
    backend_handle(&state).await.daemon_request(serde_json::json!({
        "type": "tool",
        "action": "get",
        "name": name,
    })).await
}

/// Read the per-call tool audit trail, newest first.
///
/// The trail is a file in the daemon's data directory and the GUI is a pure
/// daemon client (P16), so this goes over IPC rather than reading the disk —
/// the GUI has no idea where `--data-dir` put it, and guessing would show an
/// empty history on every isolated run.
///
/// Returns the daemon's envelope whole, including its account of itself
/// (`unparseable`, `reached_oldest`, …). A viewer that renders only the records
/// cannot tell a complete history from one screenful, so those fields are the
/// difference between a log and an audit.
///
/// # Errors
///
/// Fails when the daemon cannot be reached or the `tool.audit` request is
/// dropped or times out. A refusal the daemon reports in its reply is passed
/// through inside `Ok`.
#[tauri::command]
pub async fn get_tool_audit(
    state: State<'_, Arc<RwLock<AppState>>>,
    limit: Option<usize>,
) -> Result<serde_json::Value, String> {
    backend_handle(&state)
        .await
        .daemon_request(serde_json::json!({
            "type": "tool",
            "action": "audit",
            "limit": limit,
        }))
        .await
}

// =============================================================================
// Skill check (validates a script or manifest before it is saved)
// =============================================================================

/// Test a skill with sample input.
///
/// # Errors
///
/// For a `script`: fails when the daemon cannot be reached or the `tool.test`
/// request is dropped or times out. It also fails with `Invalid response from
/// daemon` when the reply has no `output` string. For a `manifest`: fails with
/// `Invalid YAML: …` when `code` does not parse. Any other `skill_type` fails
/// with `Unknown skill type: …`.
#[tauri::command]
pub async fn test_skill(
    state: State<'_, Arc<RwLock<AppState>>>,
    code: String,
    skill_type: String,
    input: serde_json::Map<String, serde_json::Value>,
) -> Result<String, String> {
    match skill_type.as_str() {
        "script" => {
            // Run the script through the daemon's tool sandbox.
            let result = backend_handle(&state)
                .await
                .tool_test(&code, serde_json::Value::Object(input))
                .await?;
            result
                .get("output")
                .and_then(|v| v.as_str())
                .map(str::to_string)
                .ok_or_else(|| "Invalid response from daemon".to_string())
        }
        "manifest" => match serde_yaml::from_str::<serde_json::Value>(&code) {
            Ok(_) => Ok("Manifest YAML is valid".to_string()),
            Err(e) => Err(format!("Invalid YAML: {e}")),
        },
        _ => Err(format!("Unknown skill type: {skill_type}")),
    }
}
