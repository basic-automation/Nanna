//! Workspace management commands.

use crate::backend::Backend;
use crate::state::AppState;
use nanna_core::{Workspace, WorkspaceRegistry};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tauri::State;
use tokio::sync::RwLock;
use tracing::{info, warn};

// =============================================================================
// Workspace Commands
// =============================================================================

/// Workspace info for frontend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceInfo {
    pub id: String,
    pub name: String,
    pub path: String,
    pub active: bool,
    /// Flattened, so the wire keys stay top-level (`has_readme`, …).
    #[serde(flatten)]
    pub orientation: OrientationFiles,
    #[serde(flatten)]
    pub process: ProcessFiles,
    pub context_chars: usize,
}

/// Presence of the orientation files a reader opens first: `README.md` (for
/// people) and `AGENTS.md` (for agents).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct OrientationFiles {
    pub has_readme: bool,
    pub has_agents: bool,
}

/// Presence of the working-process files: `CONTRIBUTING.md` (how to work in
/// the repo) and `ROADMAP.md` (what to work on next).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessFiles {
    pub has_contributing: bool,
    pub has_roadmap: bool,
}

impl From<&Workspace> for WorkspaceInfo {
    fn from(ws: &Workspace) -> Self {
        Self {
            id: ws.id.clone(),
            name: ws.name.clone(),
            path: ws.path.to_string_lossy().to_string(),
            active: ws.active,
            orientation: OrientationFiles {
                has_readme: ws.context.readme.is_some(),
                has_agents: ws.context.agents.is_some(),
            },
            process: ProcessFiles {
                has_contributing: ws.context.contributing.is_some(),
                has_roadmap: ws.context.roadmap.is_some(),
            },
            context_chars: ws.context.total_chars(),
        }
    }
}

/// The daemon handle and the workspace-registry cache, cloned out of the
/// shared state together (see [`backend_handle`](crate::state::backend_handle)).
/// [`AppState::workspaces`] is set once and never replaced, so the state lock
/// is released before the registry is touched; the registry's own lock does
/// the serializing.
async fn backend_and_registry(
    state: &RwLock<AppState>,
) -> (Arc<Backend>, Arc<RwLock<WorkspaceRegistry>>) {
    let state_guard = state.read().await;
    (
        Arc::clone(&state_guard.backend),
        Arc::clone(&state_guard.workspaces),
    )
}

/// List all registered workspaces, read through to the daemon.
///
/// The local registry is a CACHE, not the truth: the daemon owns registration,
/// and anything registered after this client connected — by a script, another
/// client, or a benchmark harness — would otherwise stay invisible until the
/// app restarted. Observed 2026-07-29: a workspace created two minutes before
/// the picker was opened simply wasn't in it.
///
/// Only the workspace SET is refreshed. Which workspace this client considers
/// active is its own view state and is deliberately left alone, so a second
/// client opening something never yanks this one's selection.
///
/// Best-effort: if the daemon is unreachable, serve the cache rather than
/// failing the picker.
///
/// # Errors
///
/// Never returns `Err`: when the daemon cannot be listed, the cached registry
/// is served as is.
#[tauri::command]
pub async fn list_workspaces(
    state: State<'_, Arc<RwLock<AppState>>>,
) -> Result<Vec<WorkspaceInfo>, String> {
    let (backend, workspaces) = backend_and_registry(&state).await;

    if let Ok(result) = backend.workspace_list().await {
        let records = result
            .get("workspaces")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        let mut registry = workspaces.write().await;
        for record in &records {
            let (Some(id), Some(path)) = (
                record.get("id").and_then(|v| v.as_str()),
                record.get("path").and_then(|v| v.as_str()),
            ) else {
                continue;
            };
            let path = std::path::PathBuf::from(path);
            if registry.get(id).is_some() || !path.exists() {
                continue;
            }
            let mut ws = Workspace::new(&path);
            ws.id = id.to_string();
            // Context load is for the workspace-file editing commands; a
            // failure must not keep the workspace out of the picker.
            if let Err(e) = ws.load_context().await {
                warn!("Failed to load context for {path:?}: {e}");
            }
            registry.register(ws);
        }
    }

    let registry = workspaces.read().await;
    Ok(registry
        .list()
        .iter()
        .map(|ws| WorkspaceInfo::from(*ws))
        .collect())
}

/// Open a workspace by path
///
/// # Errors
///
/// Fails with `Failed to load workspace: …` when the directory's context files
/// cannot be read. Fails when the daemon cannot be reached or the
/// `workspace.open` request is dropped or times out. Fails with `Daemon failed
/// to open workspace: …` when the daemon refuses. Nothing is registered locally
/// in any of these cases. A path already in the cache returns its entry without
/// contacting the daemon, and a failure to activate the workspace on the daemon
/// afterwards is only logged.
#[tauri::command]
pub async fn open_workspace(
    state: State<'_, Arc<RwLock<AppState>>>,
    path: String,
) -> Result<WorkspaceInfo, String> {
    let (backend, workspaces) = backend_and_registry(&state).await;
    // Held across the context load and the daemon open, so the "already
    // registered?" check and the registration below stay one step.
    let mut registry = workspaces.write().await;

    let path = std::path::PathBuf::from(&path);

    // Check if already registered
    if let Some(ws) = registry.get_by_path(&path) {
        return Ok(WorkspaceInfo::from(ws));
    }

    // Create and load new workspace
    let mut workspace = Workspace::new(&path);
    workspace
        .load_context()
        .await
        .map_err(|e| format!("Failed to load workspace: {e}"))?;

    // The daemon owns persistence (nanna.db): open the workspace there first and
    // adopt ITS id locally, so both registries agree and it survives a restart.
    let result = backend.workspace_open(&path.to_string_lossy()).await?;
    if result.get("error").is_some() {
        let msg = result
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown error");
        return Err(format!("Daemon failed to open workspace: {msg}"));
    }
    if let Some(daemon_id) = result.get("id").and_then(|v| v.as_str()) {
        workspace.id = daemon_id.to_string();
    }

    let id = registry.register(workspace);
    registry.set_active(&id);

    let ws = registry.get(&id).ok_or_else(|| {
        format!("Workspace {id} missing from the registry right after registering it")
    })?;
    let info = WorkspaceInfo::from(ws);
    info!("Opened workspace: {} at {:?}", ws.name, path);

    drop(registry);
    // The daemon's open does not activate; sync it (drives tool cwd too).
    if let Err(e) = backend.workspace_set_active(&id).await {
        warn!("Failed to activate workspace on daemon: {}", e);
    }

    Ok(info)
}

/// Set active workspace
///
/// # Errors
///
/// Returns `Workspace not found: …` when `id` is not in the cached registry,
/// or the daemon's own message when it refused the activation. A daemon that
/// cannot be reached is only logged.
#[tauri::command]
pub async fn set_active_workspace(
    state: State<'_, Arc<RwLock<AppState>>>,
    id: String,
) -> Result<(), String> {
    let (backend, workspaces) = backend_and_registry(&state).await;
    if workspaces.read().await.get(&id).is_none() {
        return Err(format!("Workspace not found: {id}"));
    }
    // The daemon decides — it owns the tool working directory. A refusal (a
    // workspace another client closed) used to be ignored, and the GUI showed
    // it Active while the daemon's cwd never moved. An unreachable daemon is
    // still only logged: the local view is all there is then.
    match backend.workspace_set_active(&id).await {
        Ok(reply) => super::daemon_refusal(&reply, "Could not activate the workspace")?,
        Err(e) => warn!("Failed to notify daemon of workspace activation: {}", e),
    }
    let activated = workspaces.write().await.set_active(&id);
    debug_assert!(activated, "the workspace was found above");
    info!("Activated workspace: {}", id);
    Ok(())
}

/// Clear active workspace (go back to global)
///
/// # Errors
///
/// Never returns `Err`: a failure to tell the daemon is only logged.
#[tauri::command]
pub async fn clear_active_workspace(state: State<'_, Arc<RwLock<AppState>>>) -> Result<(), String> {
    let (backend, workspaces) = backend_and_registry(&state).await;
    workspaces.write().await.clear_active();
    info!("Cleared active workspace, now in global mode");
    // Notify the daemon so it clears its working directory.
    if let Err(e) = backend.workspace_clear_active().await {
        warn!("Failed to notify daemon of workspace deactivation: {}", e);
    }
    Ok(())
}

/// Reload workspace context from disk
///
/// # Errors
///
/// Returns `Workspace not found: …` when `id` is not in the cached registry,
/// and `Failed to reload workspace: …` when its context files cannot be read. A
/// failure to tell the daemon is only logged.
#[tauri::command]
pub async fn reload_workspace(
    state: State<'_, Arc<RwLock<AppState>>>,
    id: String,
) -> Result<WorkspaceInfo, String> {
    let (backend, workspaces) = backend_and_registry(&state).await;

    // Reload the local cache from disk, then best-effort notify the daemon so
    // its own context copy refreshes too. The registry stays write-locked for
    // the whole reload, as the workspace is mutated in place.
    let info = reload_cached_context(&mut *workspaces.write().await, &id).await?;
    if let Err(e) = backend.workspace_reload(&id).await {
        warn!("Failed to notify daemon of workspace reload: {}", e);
    }
    Ok(info)
}

/// Re-read one cached workspace's context files from disk.
async fn reload_cached_context(
    registry: &mut WorkspaceRegistry,
    id: &str,
) -> Result<WorkspaceInfo, String> {
    let ws = registry
        .get_mut(id)
        .ok_or_else(|| format!("Workspace not found: {id}"))?;
    ws.load_context()
        .await
        .map_err(|e| format!("Failed to reload workspace: {e}"))?;
    info!("Reloaded workspace: {}", ws.name);
    Ok(WorkspaceInfo::from(&*ws))
}

/// Close a workspace
///
/// # Errors
///
/// Fails when the daemon cannot be reached or the `workspace.close` request is
/// dropped or times out. The cached entry is then kept. A refusal the daemon
/// reports in its reply is not checked.
#[tauri::command]
pub async fn close_workspace(
    state: State<'_, Arc<RwLock<AppState>>>,
    id: String,
) -> Result<(), String> {
    let (backend, workspaces) = backend_and_registry(&state).await;

    // The daemon owns persistence; close there, then drop the local cache entry.
    backend.workspace_close(&id).await?;

    workspaces.write().await.remove(&id);
    info!("Closed workspace: {}", id);
    Ok(())
}

/// Initialize a minimal workspace at path (root AGENTS.md + optional ROADMAP.md).
///
/// Persona/user/memory are NOT scaffolded — they live in global config + the DB store.
///
/// # Errors
///
/// Fails when the directory cannot be created, when `AGENTS.md`/`ROADMAP.md` or
/// a requested `README.md`/`CONTRIBUTING.md` cannot be written, or when the new
/// context cannot be loaded. Fails when the daemon cannot be reached or the
/// `workspace.open` request is dropped or times out. Files already written stay
/// on disk.
#[tauri::command]
pub async fn init_workspace(
    state: State<'_, Arc<RwLock<AppState>>>,
    path: String,
    files: Vec<String>,
) -> Result<WorkspaceInfo, String> {
    let path = std::path::PathBuf::from(&path);
    if !path.exists() {
        tokio::fs::create_dir_all(&path)
            .await
            .map_err(|e| format!("Failed to create directory: {e}"))?;
    }

    let mut workspace = Workspace::new(&path);
    let with_roadmap = files.iter().any(|f| f == "ROADMAP.md" || f == "roadmap");
    // Always create AGENTS.md; ROADMAP only if requested
    workspace
        .initialize_minimal(with_roadmap)
        .await
        .map_err(|e| format!("Failed to initialize workspace: {e}"))?;

    // Honour any requested standard context files beyond the defaults
    for file in &files {
        if file == "AGENTS.md" || file == "ROADMAP.md" {
            continue; // handled by initialize_minimal
        }
        if nanna_core::STANDARD_CONTEXT_FILES.contains(&file.as_str()) {
            let fp = path.join(file);
            if !fp.exists() {
                let content = match file.as_str() {
                    "README.md" => format!("# {}\n", workspace.name),
                    "CONTRIBUTING.md" => {
                        "# Contributing\n\n(How to work in this repo.)\n".to_string()
                    }
                    _ => continue,
                };
                tokio::fs::write(&fp, content)
                    .await
                    .map_err(|e| format!("Failed to write {file}: {e}"))?;
            }
        }
    }

    workspace
        .load_context()
        .await
        .map_err(|e| format!("Failed to load workspace: {e}"))?;

    // Open on the daemon so persistence agrees
    let (backend, workspaces) = backend_and_registry(&state).await;
    let result = backend.workspace_open(&path.to_string_lossy()).await?;
    if let Some(id) = result.get("id").and_then(|v| v.as_str()) {
        workspace.id = id.to_string();
    }

    let info = WorkspaceInfo::from(&workspace);
    workspaces.write().await.register(workspace);
    Ok(info)
}

/// Check if a path looks like a valid workspace (standard project signals)
///
/// # Errors
///
/// Never returns `Err`: a missing path is reported with `exists: false`.
#[tauri::command]
pub async fn check_workspace_validity(path: String) -> Result<WorkspaceValidityCheck, String> {
    use nanna_core::{
        AGENTS_FILE, CONTRIBUTING_FILE, README_FILE, ROADMAP_FILE, WORKSPACE_MARKERS,
    };

    let path = std::path::PathBuf::from(&path);

    if !path.exists() {
        return Ok(WorkspaceValidityCheck {
            exists: false,
            is_valid: false,
            orientation: OrientationFiles {
                has_readme: false,
                has_agents: false,
            },
            process: ProcessFiles {
                has_contributing: false,
                has_roadmap: false,
            },
            project: ProjectSignals {
                has_git: false,
                has_manifest: false,
            },
        });
    }

    let has_readme = path.join(README_FILE).exists();
    let has_agents = path.join(AGENTS_FILE).exists();
    let has_contributing = path.join(CONTRIBUTING_FILE).exists();
    let has_roadmap = path.join(ROADMAP_FILE).exists();
    let has_git = path.join(".git").exists();
    let has_manifest = ["Cargo.toml", "package.json", "pyproject.toml", "go.mod"]
        .iter()
        .any(|m| path.join(m).exists());

    // Valid if any standard project signal is present
    let is_valid = WORKSPACE_MARKERS.iter().any(|m| path.join(m).exists());

    Ok(WorkspaceValidityCheck {
        exists: true,
        is_valid,
        orientation: OrientationFiles {
            has_readme,
            has_agents,
        },
        process: ProcessFiles {
            has_contributing,
            has_roadmap,
        },
        project: ProjectSignals {
            has_git,
            has_manifest,
        },
    })
}

#[derive(Debug, Clone, Serialize)]
pub struct WorkspaceValidityCheck {
    exists: bool,
    is_valid: bool,
    /// Flattened, so the wire keys stay top-level (`has_readme`, …).
    #[serde(flatten)]
    orientation: OrientationFiles,
    #[serde(flatten)]
    process: ProcessFiles,
    #[serde(flatten)]
    project: ProjectSignals,
}

/// Project markers that are not documents: a `.git` directory, and a build
/// manifest (`Cargo.toml`, `package.json`, `pyproject.toml` or `go.mod`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ProjectSignals {
    has_git: bool,
    has_manifest: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The flattened groups must serialize to exactly the bytes the flat
    /// struct did — same keys, same order — and read back the same.
    #[test]
    fn workspace_info_wire_shape_is_unchanged() {
        let flat = r#"{"id":"w1","name":"repo","path":"/src/repo","active":true,"has_readme":true,"has_agents":false,"has_contributing":true,"has_roadmap":false,"context_chars":42}"#;
        let info = WorkspaceInfo {
            id: "w1".to_string(),
            name: "repo".to_string(),
            path: "/src/repo".to_string(),
            active: true,
            orientation: OrientationFiles {
                has_readme: true,
                has_agents: false,
            },
            process: ProcessFiles {
                has_contributing: true,
                has_roadmap: false,
            },
            context_chars: 42,
        };
        assert_eq!(serde_json::to_string(&info).expect("serializes"), flat);

        let read_back: WorkspaceInfo = serde_json::from_str(flat).expect("deserializes");
        assert_eq!(read_back.orientation, info.orientation);
        assert_eq!(read_back.process, info.process);
        assert_eq!(serde_json::to_string(&read_back).expect("serializes"), flat);
    }

    #[test]
    fn validity_check_wire_shape_is_unchanged() {
        let check = WorkspaceValidityCheck {
            exists: true,
            is_valid: true,
            orientation: OrientationFiles {
                has_readme: false,
                has_agents: true,
            },
            process: ProcessFiles {
                has_contributing: false,
                has_roadmap: true,
            },
            project: ProjectSignals {
                has_git: true,
                has_manifest: false,
            },
        };
        assert_eq!(
            serde_json::to_string(&check).expect("serializes"),
            r#"{"exists":true,"is_valid":true,"has_readme":false,"has_agents":true,"has_contributing":false,"has_roadmap":true,"has_git":true,"has_manifest":false}"#
        );
    }
}
