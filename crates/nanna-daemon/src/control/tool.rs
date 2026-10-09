//! Tool handlers for the [`ControlPlane`].

use super::{json, info, ControlPlane, ToolAction, ToolRegistry, Value};

impl ControlPlane {
    // =========================================================================
    // Tool Handlers
    // =========================================================================
    
    pub(super) async fn handle_tool(&self, _client_id: &str, action: ToolAction) -> Value {
        let Some(ref tools) = self.tools else {
            return json!({ "error": "tools_unavailable", "message": "Tool registry not configured" });
        };
        
        match action {
            ToolAction::List => self.tool_list(tools).await,
            ToolAction::Get { name } => {
                // `get` rather than `definitions()` for the same reason as
                // `tool_list`: a disabled tool must still be inspectable, or
                // its detail panel reports "not found" the moment it is switched
                // off.
                if let Some(tool) = tools.get(&name).await {
                    let definition = tool.definition();
                    let canonical = tools.canonical_name(&name).await;
                    json!({ "tool": {
                        "name": definition.name,
                        "description": definition.description,
                        "parameters": definition.parameters,
                        "enabled": tools.policy().await.permits(&canonical),
                    }})
                } else {
                    json!({ "error": "not_found", "name": name })
                }
            }
            ToolAction::Enable { name } => self.set_tool_enabled(&name, true).await,
            ToolAction::Disable { name } => self.set_tool_enabled(&name, false).await,
            ToolAction::Execute {
                name,
                input,
                session_id,
            } => Self::tool_execute(tools, name, input, session_id).await,
            ToolAction::Create { name, description, code, needs_shell } => self.tool_create(name, description, code, needs_shell).await,
            ToolAction::Update { name, description, code, needs_shell } => self.tool_update(name, description, code, needs_shell).await,
            ToolAction::Delete { name } => {
                let Some(ref user_tools) = self.user_tools else {
                    return json!({ "error": "user_tools_unavailable", "message": "User tool manager not configured" });
                };
                
                match user_tools.delete_tool(&name).await {
                    Ok(()) => {
                        // Make the deletion take effect live: a tool that's gone
                        // from disk must also stop being callable without a daemon
                        // restart (previously it lingered in the registry).
                        if let Some(ref tools) = self.tools {
                            tools.unregister(&name).await;
                        }
                        info!("Deleted user tool: {}", name);
                        json!({ "status": "deleted", "name": name })
                    }
                    Err(e) => json!({ "error": "delete_failed", "message": e })
                }
            }
            ToolAction::Test { code, input } => {
                let Some(ref user_tools) = self.user_tools else {
                    return json!({ "error": "user_tools_unavailable", "message": "User tool manager not configured" });
                };
                
                let input_map: std::collections::HashMap<String, Value> = match input {
                    Value::Object(map) => map.into_iter().collect(),
                    _ => std::collections::HashMap::new(),
                };
                
                match user_tools.test_tool(&code, input_map).await {
                    Ok(output) => json!({ "status": "success", "output": output }),
                    Err(e) => json!({ "status": "error", "error": e })
                }
            }
            ToolAction::GetSource { name } => self.tool_get_source(name).await,
            ToolAction::ListUser => {
                let Some(ref user_tools) = self.user_tools else {
                    return json!({ "error": "user_tools_unavailable", "message": "User tool manager not configured" });
                };

                let tools = user_tools.list_tools().await;
                let tool_list: Vec<_> = tools.into_iter()
                    .map(|t| json!({
                        "name": t.name,
                        "description": t.description,
                        "source": t.source,
                        "language": t.language,
                        "enabled": t.enabled,
                        "created_at": t.created_at,
                        "updated_at": t.updated_at,
                    }))
                    .collect();
                json!({ "tools": tool_list })
            }
            ToolAction::Audit { limit } => self.tool_audit(limit),
        }
    }

    /// `ToolAction::List`: every registered tool plus disabled user tools.
    async fn tool_list(&self, tools: &ToolRegistry) -> Value {
        // `inventory()`, NOT `definitions()`. The latter hides
        // policy-denied tools — correct for the model, which must not be
        // offered a tool the gate would refuse, but fatal for a
        // management surface: a disabled tool would vanish from the only
        // list the GUI can see, and disabling would be a one-way door.
        let entries = tools.inventory().await;

        // A disabled user tool is unregistered from the live registry, so
        // it is absent from the inventory above and has to be merged back
        // in from its own store, or it would be missing for the same
        // reason. Its store carries the authoritative flag.
        let user_tools = match self.user_tools {
            Some(ref ut) => ut.list_tools().await,
            None => Vec::new(),
        };
        let user_names: std::collections::HashSet<&str> =
            user_tools.iter().map(|t| t.name.as_str()).collect();

        let mut tool_list: Vec<_> = entries
            .iter()
            .map(|t| {
                json!({
                    "name": t.name,
                    "description": t.description,
                    "enabled": t.enabled,
                    "is_user_tool": user_names.contains(t.name.as_str()),
                })
            })
            .collect();

        let listed: std::collections::HashSet<&str> =
            entries.iter().map(|t| t.name.as_str()).collect();
        tool_list.extend(
            user_tools
                .iter()
                .filter(|t| !listed.contains(t.name.as_str()))
                .map(|t| {
                    json!({
                        "name": t.name,
                        "description": t.description,
                        "enabled": t.enabled,
                        "is_user_tool": true,
                    })
                }),
        );

        json!({ "tools": tool_list })
    }

    /// `ToolAction::Execute`: run one tool directly, outside an agent turn.
    ///
    /// `session_id` binds the call to that session for the tools that read it
    /// (file history, workspace cwd); without one the call runs unattributed.
    async fn tool_execute(
        tools: &ToolRegistry,
        name: String,
        input: Value,
        session_id: Option<String>,
    ) -> Value {
        use nanna_tools::ToolCall;

        let params: std::collections::HashMap<String, Value> = match input {
            Value::Object(map) => map.into_iter().collect(),
            _ => std::collections::HashMap::new(),
        };

        let call = ToolCall {
            id: uuid::Uuid::new_v4().to_string(),
            name: name.clone(),
            parameters: params,
        };

        let result = match session_id {
            Some(session_id) => {
                nanna_tools::ToolRegistry::with_run_session(session_id, tools.execute(call)).await
            }
            None => tools.execute(call).await,
        };

        json!({
            "name": name,
            "success": result.result.success,
            "output": result.result.content,
            // A failed tool's explanation lives here, not in `output`;
            // without it a direct call that failed answered only
            // `success: false` with an empty string.
            "error": result.result.error,
        })
    }

    /// `ToolAction::Create`: write a user tool and register it live.
    async fn tool_create(&self, name: String, description: String, code: String, needs_shell: Option<bool>) -> Value {
        let Some(ref user_tools) = self.user_tools else {
            return json!({ "error": "user_tools_unavailable", "message": "User tool manager not configured" });
        };

        // Build permissions
        let permissions = if needs_shell.unwrap_or(false) {
            Some(crate::user_tools::UserToolPermissions {
                run: true,
                ..Default::default()
            })
        } else {
            None
        };

        match user_tools.create_tool(name.clone(), description, code, None, None, permissions).await {
            Ok(meta) => {
                // Register with tool registry immediately
                if let Some(ref tools) = self.tools
                    && let Ok(tool_impl) = user_tools.create_tool_impl(&meta)
                {
                    tools.register_boxed(tool_impl).await;
                }

                info!("Created user tool: {}", name);
                json!({
                    "status": "created",
                    "tool": {
                        "name": meta.name,
                        "description": meta.description,
                        "language": meta.language,
                        "enabled": meta.enabled,
                        "created_at": meta.created_at,
                    }
                })
            }
            Err(e) => json!({ "error": "create_failed", "message": e })
        }
    }

    /// `ToolAction::GetSource`: a tool's source, from the tools directory first.
    async fn tool_get_source(&self, name: String) -> Value {
        // Try tools directory first, then user tools
        if let Some(ref dir) = self.tools_dir {
            let path = dir.join(&name).join("tool.ts");
            if let Ok(source) = std::fs::read_to_string(&path) {
                return json!({
                    "name": name,
                    "source": source,
                    "language": "typescript",
                    "path": path.to_string_lossy(),
                });
            }
        }
        // Fall back to user tools
        if let Some(ref user_tools) = self.user_tools
            && let Some(meta) = user_tools.get_tool(&name).await
        {
            return json!({
                "name": meta.name,
                "source": meta.source,
                "language": meta.language,
            });
        }
        json!({ "error": "not_found", "name": name })
    }

    /// `ToolAction::Update`: rewrite a user tool and re-register it live.
    async fn tool_update(&self, name: String, description: Option<String>, code: Option<String>, needs_shell: Option<bool>) -> Value {
        let Some(ref user_tools) = self.user_tools else {
            return json!({ "error": "user_tools_unavailable", "message": "User tool manager not configured" });
        };

        // `needs_shell` sets the shell grant on the tool's CURRENT permissions.
        // `false` used to map to "no change" (so shell access could not be
        // revoked here while the reply said `updated`), and `true` replaced
        // the whole set with a default one, dropping its read/write scopes.
        let permissions = match needs_shell {
            Some(run) => user_tools.get_tool(&name).await.map(|meta| {
                let mut permissions = meta.permissions;
                permissions.run = run;
                permissions
            }),
            None => None,
        };
        debug_assert!(
            permissions
                .as_ref()
                .is_none_or(|p| Some(p.run) == needs_shell),
            "the grant asked for is the grant written"
        );

        match user_tools.update_tool(&name, description, code, None, permissions, None).await {
            Ok(meta) => {
                // Make the edit take effect live: drop the old registration
                // and re-register the new source (if still enabled).
                self.reconcile_tool_registration(&meta).await;
                info!("Updated user tool: {}", name);
                json!({
                    "status": "updated",
                    "tool": {
                        "name": meta.name,
                        "description": meta.description,
                        "language": meta.language,
                        "enabled": meta.enabled,
                        "updated_at": meta.updated_at,
                    }
                })
            }
            Err(e) => json!({ "error": "update_failed", "message": e })
        }
    }

    /// `ToolAction::Audit`: the newest page of the per-call audit trail.
    fn tool_audit(&self, limit: Option<usize>) -> Value {
        // No trail configured is not an empty trail. Saying "0 records"
        // here would answer "what has Nanna been doing?" with silence
        // that reads as "nothing", when the truth is that nobody was
        // writing it down.
        let Some(ref path) = self.audit_log_path else {
            return json!({
                "enabled": false,
                "records": [],
                "message": "The tool audit trail is off. Set `[tools] audit_log = true` \
                            and restart the daemon to begin recording tool calls.",
            });
        };

        let page =
            nanna_tools::read_recent_audit(path, limit.unwrap_or(nanna_tools::AUDIT_PAGE_DEFAULT));
        json!({
            "enabled": true,
            "path": path.display().to_string(),
            "records": page.records,
            // Everything below is the reader's account of itself. A
            // viewer that shows records without them cannot tell a
            // complete history from a screenful, or a clean file from
            // one it partly failed to read.
            "unparseable": page.unparseable,
            "scanned": page.scanned,
            "generations_read": page.generations_read,
            "reached_oldest": page.reached_oldest,
        })
    }
}
