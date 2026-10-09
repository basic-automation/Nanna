//! Workspace handlers for the [`ControlPlane`].

use super::{error, json, info, warn, ControlPlane, Event, WorkspaceAction, Value, PathBuf, Workspace};

impl ControlPlane {
    // =========================================================================
    // Workspace Handlers
    // =========================================================================
    
    /// Persist current workspace registry to the database
    async fn save_workspaces(&self) {
        let Some(ref storage) = self.storage else { return };
        let registry = self.workspaces.read().await;
        let repo = storage.workspaces();
        for ws in registry.list() {
            let record = nanna_storage::WorkspaceRecord {
                id: ws.id.clone(),
                name: ws.name.clone(),
                path: ws.path.display().to_string(),
                active: ws.active,
                created_at: String::new(), // DB handles default
                last_accessed: String::new(), // DB handles default
            };
            if let Err(e) = repo.upsert(&record).await {
                error!("Failed to save workspace {}: {}", ws.name, e);
            }
        }
    }

    /// Tell every connected client the known-workspace set changed.
    ///
    /// Registration is global state; activation is not, so this deliberately
    /// carries no active id — see [`Event::WorkspacesChanged`]. Fire-and-forget:
    /// a send error only means nobody is subscribed.
    fn notify_workspaces_changed(&self) {
        if let Some(ref tx) = self.event_tx {
            let _ = tx.send(Event::WorkspacesChanged);
        }
    }

    pub(super) async fn handle_workspace(&self, _client_id: &str, action: WorkspaceAction) -> Value {
        match action {
            WorkspaceAction::List => {
                let registry = self.workspaces.read().await;
                let workspaces: Vec<_> = registry.list().iter()
                    .map(|ws| json!({
                        "id": ws.id,
                        "name": ws.name,
                        "path": ws.path,
                        "active": ws.active,
                        "last_accessed": ws.last_accessed,
                    }))
                    .collect();
                let active_id = registry.active().map(|ws| ws.id.clone());
                drop(registry);
                json!({ "workspaces": workspaces, "active_id": active_id })
            }
            WorkspaceAction::Get { id } => {
                let registry = self.workspaces.read().await;
                registry.get(&id).map_or_else(
                    || json!({ "error": "not_found", "id": id }),
                    |ws| {
                        json!({
                            "workspace": {
                                "id": ws.id,
                                "name": ws.name,
                                "path": ws.path,
                                "active": ws.active,
                                "last_accessed": ws.last_accessed,
                                "metadata": ws.metadata,
                                "context_loaded": !ws.context.is_empty(),
                            }
                        })
                    },
                )
            }
            WorkspaceAction::Open { path } => self.workspace_open(path).await,
            WorkspaceAction::Close { id } => {
                let mut registry = self.workspaces.write().await;
                let was_active = registry.active().is_some_and(|ws| ws.id == id);
                if let Some(ws) = registry.remove(&id) {
                    info!("Closed workspace: {} ({})", ws.name, id);
                    drop(registry);
                    // Closing the active workspace is global mode, like
                    // ClearActive: tools stop resolving paths in a project
                    // that is no longer open (only a registered, active
                    // workspace sets the tool cwd).
                    if was_active && let Some(ref tools) = self.tools {
                        tools.set_default_workdir(None).await;
                    }
                    // Remove from database — and say so when that failed:
                    // the row would bring the workspace back on restart.
                    let deleted = match self.storage {
                        Some(ref storage) => storage.workspaces().delete(&id).await.err(),
                        None => None,
                    };
                    self.notify_workspaces_changed();
                    deleted.map_or_else(
                        || json!({ "status": "closed", "id": id }),
                        |e| {
                            json!({
                                "status": "closed", "id": id, "persisted": false,
                                "message": format!("closed for now, but it will reopen on restart: {e}"),
                            })
                        },
                    )
                } else {
                    json!({ "error": "not_found", "id": id })
                }
            }
            WorkspaceAction::SetActive { id } => self.workspace_set_active(id).await,
            WorkspaceAction::ClearActive => {
                let mut registry = self.workspaces.write().await;
                registry.clear_active();
                drop(registry);
                // Clear tool registry's default working directory
                if let Some(ref tools) = self.tools {
                    tools.set_default_workdir(None).await;
                }
                info!("Cleared active workspace (global mode)");
                self.save_workspaces().await;
                json!({ "status": "cleared" })
            }
            WorkspaceAction::Reload { id } => self.workspace_reload(id).await,
            WorkspaceAction::GetContext { id } => {
                let registry = self.workspaces.read().await;
                registry.get(&id).map_or_else(
                    || json!({ "error": "not_found", "id": id }),
                    |ws| {
                        json!({
                            "context": {
                                "readme": ws.context.readme,
                                "agents": ws.context.agents,
                                "contributing": ws.context.contributing,
                                "roadmap": ws.context.roadmap,
                            },
                            "total_chars": ws.context.total_chars(),
                            "system_prompt_injection": ws.context.build_system_prompt_injection(),
                        })
                    },
                )
            }
            WorkspaceAction::UpdateContext { id, file, content } => self.workspace_update_context(id, file, content).await,
        }
    }

    /// `WorkspaceAction::Open`: register a directory as a workspace, creating its `.nanna` folder if needed.
    async fn workspace_open(&self, path: String) -> Value {
        let path = PathBuf::from(&path);

        // Check if workspace already registered
        {
            let registry = self.workspaces.read().await;
            if let Some(existing) = registry.get_by_path(&path) {
                return json!({ 
                    "status": "already_registered", 
                    "id": existing.id,
                    "name": existing.name,
                });
            }
        }

        // Check if path is a valid workspace
        if !nanna_core::is_workspace_root(&path).await {
            // Create .nanna folder to make it a workspace
            let nanna_folder = path.join(nanna_core::NANNA_FOLDER);
            if let Err(e) = tokio::fs::create_dir_all(&nanna_folder).await {
                return json!({ "error": "create_failed", "message": e.to_string() });
            }
            info!("Created workspace at {:?}", path);
        }

        // Create and register workspace
        let mut ws = Workspace::new(&path);
        if let Err(e) = ws.load_context().await {
            warn!("Failed to load workspace context: {}", e);
        }

        let id = ws.id.clone();
        let name = ws.name.clone();

        let mut registry = self.workspaces.write().await;
        registry.register(ws);
        // Release before save_workspaces: it takes a READ on this same
        // non-reentrant RwLock. Holding the write guard across that
        // call was a guaranteed self-deadlock — observed live, it
        // wedged every workspace action until a daemon restart AND
        // silently skipped persistence (the upsert was never reached).
        // SetActive/ClearActive already drop their guards; Open was
        // the one caller that didn't.
        drop(registry);

        info!("Registered workspace: {} ({})", name, id);
        self.save_workspaces().await;
        self.notify_workspaces_changed();
        json!({ "status": "opened", "id": id, "name": name })
    }

    /// `WorkspaceAction::UpdateContext`: write one standard context file into a workspace.
    async fn workspace_update_context(&self, id: String, file: String, content: String) -> Value {
        // Validate against standard project context files
        let valid_files = nanna_core::STANDARD_CONTEXT_FILES;

        if !valid_files.contains(&file.as_str()) {
            return json!({ 
                "error": "invalid_file", 
                "file": file,
                "valid_files": valid_files,
            });
        }

        let registry = self.workspaces.read().await;
        if let Some(ws) = registry.get(&id) {
            match ws.save_context_file(&file, &content).await {
                Ok(()) => {
                    info!("Updated workspace file: {} in {}", file, ws.name);
                    json!({ "status": "updated", "id": id, "file": file })
                }
                Err(e) => json!({ "error": "save_failed", "message": e.to_string() })
            }
        } else {
            json!({ "error": "not_found", "id": id })
        }
    }

    /// `WorkspaceAction::SetActive`: activate a workspace and point tool paths at it.
    async fn workspace_set_active(&self, id: String) -> Value {
        let mut registry = self.workspaces.write().await;
        if registry.set_active(&id) {
            let ws_path = registry.get(&id).map(|ws| ws.path.clone());
            let name = registry.get(&id).map(|ws| ws.name.clone());
            drop(registry);
            // Update tool registry's default working directory to workspace path
            if let (Some(tools), Some(path)) = (&self.tools, &ws_path) {
                tools.set_default_workdir(Some(path.clone())).await;
                info!("Set tool working directory to {:?}", path);
            }
            info!("Set active workspace: {:?}", name);
            self.save_workspaces().await;
            json!({ "status": "activated", "id": id, "name": name })
        } else {
            json!({ "error": "not_found", "id": id })
        }
    }

    /// `WorkspaceAction::Reload`: re-read one workspace's context, git snapshot included.
    async fn workspace_reload(&self, id: String) -> Value {
        let mut registry = self.workspaces.write().await;
        if let Some(ws) = registry.get_mut(&id) {
            // An explicit user-driven reload of ONE workspace, so it
            // takes the git snapshot too — that is what the caller is
            // asking to refresh.
            match ws.load_context_with_git().await {
                Ok(()) => {
                    info!("Reloaded workspace context: {}", ws.name);
                    json!({ 
                        "status": "reloaded", 
                        "id": id,
                        "context_chars": ws.context.total_chars(),
                    })
                }
                Err(e) => json!({ "error": "reload_failed", "message": e.to_string() })
            }
        } else {
            json!({ "error": "not_found", "id": id })
        }
    }
}
