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
use tracing::{info, warn};

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

/// List the user-authored tools the daemon knows.
///
/// # Errors
///
/// Fails when the daemon cannot be reached or the `tool.list_user` request is
/// dropped or times out. Fails with `Failed to parse daemon response: …` when
/// the reply's `tools` entries do not match [`UserToolMeta`]; a reply without
/// `tools` lists none.
#[tauri::command]
pub async fn list_user_tools_cmd(
    state: State<'_, Arc<RwLock<AppState>>>,
) -> Result<Vec<UserToolMeta>, String> {
    let result = backend_handle(&state).await.tool_list_user().await?;
    parse_user_tools(&result)
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

/// Run tool source against sample input without saving it.
///
/// # Errors
///
/// Fails when the daemon cannot be reached or the `tool.test` request is
/// dropped or times out. Fails with `Invalid response from daemon` when the
/// reply has no `output` string, a refusal reported in the reply included.
#[tauri::command]
pub async fn test_user_tool(
    state: State<'_, Arc<RwLock<AppState>>>,
    source: String,
    input: serde_json::Map<String, serde_json::Value>,
) -> Result<String, String> {
    let result = backend_handle(&state)
        .await
        .tool_test(&source, serde_json::Value::Object(input))
        .await?;
    result
        .get("output")
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .ok_or_else(|| "Invalid response from daemon".to_string())
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
// Skill Directory Commands (workspace-based tools, edited on disk)
// =============================================================================

#[derive(Debug, Clone, serde::Serialize)]
pub struct SkillInfo {
    name: String,
    #[serde(rename = "type")]
    skill_type: String,
    language: Option<String>,
    path: String,
    code: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SkillListResult {
    skills: Vec<SkillInfo>,
    path: String,
}

/// Resolve the skills directory: the active workspace's `skills/`, else a
/// per-user data directory fallback.
pub(crate) async fn get_skills_path(state: &AppState) -> std::path::PathBuf {
    {
        let registry = state.workspaces.read().await;
        if let Some(ws) = registry.active() {
            return ws.path.join("skills");
        }
    }
    nanna_config::project_dirs().map_or_else(|| std::path::PathBuf::from("skills"), |p| p.data_dir().join("skills"))
}

/// List all skills in the workspace `skills/` directory.
///
/// # Errors
///
/// Never returns `Err`: a skills directory that cannot be created is logged and
/// lists no skills, and a skill whose file cannot be read lists without code.
#[tauri::command]
pub async fn list_skills(
    state: State<'_, Arc<RwLock<AppState>>>,
) -> Result<SkillListResult, String> {
    let skills_path = get_skills_path(&*state.read().await).await;

    if !skills_path.exists()
        && let Err(e) = std::fs::create_dir_all(&skills_path) {
            warn!("Failed to create skills directory: {e}");
        }

    let discovered = nanna_tools::skills::discover_skills(&skills_path);

    let mut skills = Vec::new();
    for skill in discovered {
        let (skill_type, language) = match &skill.source {
            nanna_tools::skills::SkillSource::Script(p) => {
                let lang = p
                    .extension()
                    .and_then(|e| e.to_str())
                    .map_or("javascript", |e| if e == "ts" { "typescript" } else { "javascript" });
                ("script".to_string(), Some(lang.to_string()))
            }
            nanna_tools::skills::SkillSource::Manifest(_) => ("manifest".to_string(), None),
        };

        let code_path = match &skill.source {
            nanna_tools::skills::SkillSource::Script(p)
            | nanna_tools::skills::SkillSource::Manifest(p) => p.clone(),
        };
        let code = std::fs::read_to_string(&code_path).ok();

        skills.push(SkillInfo {
            name: skill.name,
            skill_type,
            language,
            path: skill.path.display().to_string(),
            code,
        });
    }

    Ok(SkillListResult {
        skills,
        path: skills_path.display().to_string(),
    })
}

/// Create a new skill in the workspace.
///
/// # Errors
///
/// Fails when `name` has characters other than lowercase ASCII letters, digits,
/// `_` and `-`; when a skill of that name already exists; when the skill
/// directory cannot be created; when `skill_type` is neither `manifest` nor
/// `script` (checked after the directory is created, which is then left empty);
/// and when the skill file cannot be written.
#[tauri::command]
pub async fn create_skill(
    state: State<'_, Arc<RwLock<AppState>>>,
    name: String,
    skill_type: String,
    code: String,
) -> Result<SkillInfo, String> {
    if !name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-') {
        return Err("Skill name must be lowercase alphanumeric with underscores or hyphens".to_string());
    }

    let skills_path = get_skills_path(&*state.read().await).await;
    let skill_dir = skills_path.join(&name);
    if skill_dir.exists() {
        return Err(format!("Skill '{name}' already exists"));
    }
    std::fs::create_dir_all(&skill_dir).map_err(|e| format!("Failed to create skill directory: {e}"))?;

    let (filename, language) = match skill_type.as_str() {
        "manifest" => ("tool.yaml", None),
        "script" => ("tool.ts", Some("typescript".to_string())),
        _ => return Err(format!("Unknown skill type: {skill_type}")),
    };

    let code_path = skill_dir.join(filename);
    std::fs::write(&code_path, &code).map_err(|e| format!("Failed to write skill code: {e}"))?;

    info!("Created new skill: {name} at {}", skill_dir.display());

    Ok(SkillInfo {
        name,
        skill_type,
        language,
        path: skill_dir.display().to_string(),
        code: Some(code),
    })
}

/// The name of an existing skill, trimmed, or why it cannot name one.
///
/// One rule for every command that addresses an existing skill directory by
/// name: non-empty, no path separator, no `..`, and only ASCII letters,
/// digits, `-`, `_` and `.`. `Path::join` does the rest of the damage without
/// it — `../x` walks out of the skills directory, and an absolute name
/// REPLACES the base path outright.
///
/// # Errors
///
/// Returns the reason `name` is refused.
fn validate_existing_skill_name(name: &str) -> Result<&str, String> {
    let name = name.trim();
    if name.is_empty() {
        return Err("Skill name must be non-empty".into());
    }
    if name.contains('/') || name.contains('\\') || name.contains("..") {
        return Err(format!(
            "Invalid skill name '{name}': path separators and '..' are not allowed"
        ));
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
    {
        return Err(format!(
            "Invalid skill name '{name}': only alphanumeric, '-', '_', '.' are allowed"
        ));
    }
    debug_assert!(!name.starts_with('/'), "a separator-free name is relative");
    Ok(name)
}

/// Update an existing skill's code.
///
/// # Errors
///
/// Fails when `name` is refused by [`validate_existing_skill_name`], when no
/// skill directory named `name` exists or it resolves outside the skills
/// directory (a symlink), when it holds none of `tool.ts`, `tool.js`,
/// `tool.yaml` and `tool.yml`, and when that file cannot be written.
#[tauri::command]
pub async fn update_skill(
    state: State<'_, Arc<RwLock<AppState>>>,
    name: String,
    code: String,
) -> Result<SkillInfo, String> {
    let name = validate_existing_skill_name(&name)?.to_string();
    let skills_path = get_skills_path(&*state.read().await).await;

    let skill_dir = skills_path.join(&name);
    if !skill_dir.exists() {
        return Err(format!("Skill '{name}' not found"));
    }
    // A skill directory that is a symlink resolves wherever it points; the
    // write below must still land inside the skills directory.
    let root = std::fs::canonicalize(&skills_path)
        .map_err(|e| format!("Failed to resolve skills directory: {e}"))?;
    let resolved = std::fs::canonicalize(&skill_dir)
        .map_err(|e| format!("Failed to resolve skill '{name}': {e}"))?;
    if !resolved.starts_with(&root) {
        return Err(format!(
            "Skill '{name}' resolves outside the skills directory"
        ));
    }

    let code_files = ["tool.ts", "tool.js", "tool.yaml", "tool.yml"];
    let code_path = code_files
        .iter()
        .map(|f| skill_dir.join(f))
        .find(|p| p.exists())
        .ok_or_else(|| format!("No tool file found in skill '{name}'"))?;

    std::fs::write(&code_path, &code).map_err(|e| format!("Failed to update skill code: {e}"))?;

    let (skill_type, language) = match code_path.extension().and_then(|e| e.to_str()) {
        Some("yaml" | "yml") => ("manifest".to_string(), None),
        Some("ts") => ("script".to_string(), Some("typescript".to_string())),
        Some("js") => ("script".to_string(), Some("javascript".to_string())),
        _ => ("unknown".to_string(), None),
    };

    info!("Updated skill: {name}");

    Ok(SkillInfo {
        name,
        skill_type,
        language,
        path: skill_dir.display().to_string(),
        code: Some(code),
    })
}

/// Delete a skill.
///
/// Hardens the delete path against symlink escapes: the skill name is sanitized
/// so `$skills_path/<name>` cannot resolve outside the skills root, and
/// symlinked skill directories (or symlink children) are refused.
///
/// # Errors
///
/// Fails when `name` is empty after trimming, contains `/`, `\` or `..`, or has
/// characters other than ASCII letters, digits, `-`, `_` and `.`; when the
/// skills directory does not exist or cannot be resolved; when no such skill
/// exists or it is not a directory; when it or any direct child is a symlink,
/// or it resolves outside the skills directory; and when reading or removing it
/// fails.
#[tauri::command]
pub async fn delete_skill(
    state: State<'_, Arc<RwLock<AppState>>>,
    name: String,
) -> Result<(), String> {
    let name = validate_existing_skill_name(&name)?;

    let skills_path = get_skills_path(&*state.read().await).await;
    let skills_root = if skills_path.exists() {
        std::fs::canonicalize(&skills_path)
            .map_err(|e| format!("Failed to resolve skills directory: {e}"))?
    } else {
        return Err(format!("Skills directory \"{}\" does not exist", skills_path.display()));
    };

    let skill_dir = skills_root.join(name);
    if !skill_dir.exists() {
        return Err(format!("Skill '{name}' not found"));
    }

    let meta = std::fs::symlink_metadata(&skill_dir)
        .map_err(|e| format!("Failed to stat skill '{name}': {e}"))?;
    if meta.file_type().is_symlink() {
        return Err(format!("Refusing to delete skill '{name}': path is a symlink (escape risk)"));
    }
    if !meta.is_dir() {
        return Err(format!("Skill path '{name}' is not a directory"));
    }

    let canonical = std::fs::canonicalize(&skill_dir)
        .map_err(|e| format!("Failed to resolve skill '{name}': {e}"))?;
    if !canonical.starts_with(&skills_root) {
        return Err(format!(
            "Refusing to delete skill '{name}': resolved path escapes skills directory"
        ));
    }

    for entry in std::fs::read_dir(&canonical).map_err(|e| format!("Failed to read skill '{name}': {e}"))? {
        let entry = entry.map_err(|e| format!("Failed to read skill entry: {e}"))?;
        let ft = entry.file_type().map_err(|e| format!("Failed to stat skill entry: {e}"))?;
        if ft.is_symlink() {
            return Err(format!(
                "Refusing to delete skill '{name}': contains symlink child '{}'",
                entry.file_name().display()
            ));
        }
    }

    std::fs::remove_dir_all(&canonical).map_err(|e| format!("Failed to delete skill: {e}"))?;
    info!("Deleted skill: {name}");
    Ok(())
}

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

#[cfg(test)]
mod skill_name_tests {
    /// The name reaches `Path::join`: `../x` walks out of the skills
    /// directory and an absolute name replaces it. `update_skill` joined
    /// whatever it was given.
    #[test]
    fn a_skill_name_can_only_name_a_directory_inside_the_skills_dir() {
        for bad in [
            "",
            "   ",
            "../escape",
            "a/b",
            "a\\b",
            "/etc/passwd",
            "..",
            "x y",
        ] {
            assert!(
                super::validate_existing_skill_name(bad).is_err(),
                "{bad:?} accepted"
            );
        }
        assert_eq!(
            super::validate_existing_skill_name(" web_search ").unwrap(),
            "web_search"
        );
        assert_eq!(
            super::validate_existing_skill_name("My-Tool.v2").unwrap(),
            "My-Tool.v2"
        );
    }
}
