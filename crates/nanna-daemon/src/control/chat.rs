//! Chat handlers for the [`ControlPlane`].

use std::fmt::Write as _;

use super::{debug, json, warn, info, PathBuf, ControlPlane, Arc, ChatAction, Value, MessageRole};

/// Everything a chat turn needs that is NOT required for the delivery ack:
/// the assembled system prompt (persona, recall, conversation), the rendered
/// conversation for the planner, and the resolved workspace. Produced by
/// [`ControlPlane::prepare_chat_turn`] INSIDE the spawned turn (P22) — recall
/// rides on embedding providers that can stall for minutes when benched, and
/// paying that before the ack is how `chat.send` twice timed out at 120s
/// while the daemon had in fact accepted the mission (2026-08-10 forensics).
pub(super) struct ChatTurnPrep {
    pub system_prompt: String,
    pub conversation: Option<String>,
    pub workspace_root: Option<PathBuf>,
    pub workspace_context: Option<String>,
    /// This chat's pinned model, read off the session that was already loaded
    /// to build the prompt above. `None` = follow the global `[llm]` default,
    /// which is the whole of the precedence rule.
    ///
    /// It rides in the prep rather than being re-read at the point of use
    /// because the turn resolves its model exactly ONCE, from this value, into
    /// the per-turn `AgentConfig` clone — a second read is a second place the
    /// answer could differ, and the planner and the step runner must never
    /// disagree about which model this chat runs on.
    pub chat_model: Option<String>,
    /// Tools the user selected for THIS chat (empty = no restriction).
    pub chat_tools: Vec<String>,
}

impl ControlPlane {
    // =========================================================================
    // Chat Handlers
    // =========================================================================

    pub(super) async fn handle_chat(self: &Arc<Self>, client_id: &str, action: ChatAction) -> Value {
        match action {
            ChatAction::Send { session_id, content, attachments } => {
                debug!("Chat send from {} to session {}", client_id, session_id);

                // Nothing to answer: a blank message used to be persisted and
                // run as a whole planned turn, the model guessing at a
                // request that was never made.
                if content.trim().is_empty() && attachments.is_empty() {
                    return json!({
                        "error": "empty_message",
                        "message": "The message is empty; there is nothing to answer."
                    });
                }

                // Add user message to session — persisting it is the fact the
                // delivery ack below certifies.
                let Some(_msg_id) = self.sessions.add_message(&session_id, MessageRole::User, &content).await else {
                    return json!({
                        "error": "session_not_found",
                        "message": format!("Session {} not found", session_id)
                    });
                };

                // Check if agent is available (a turn without one is born dead)
                if self.agent.is_none() {
                    return json!({
                        "error": "agent_unavailable",
                        "message": "Agent service not configured"
                    });
                }

                // Images ride every step of the turn; anything the model
                // cannot read is named in the goal so it can say so, instead
                // of being dropped with only a daemon-log warning (P19).
                let (images, unreadable) = crate::tasks::split_attachments(&attachments);
                let goal = unreadable
                    .map_or_else(|| content.clone(), |note| format!("{content}\n\n{note}"));

                // ── Long-horizon chat (P19): the only path ──
                // Every turn is a harness run: the message is planned, the
                // plan is driven with re-anchored steps, and the work streams
                // into this transcript as it happens. A message sent while a
                // run is live joins that run at the next step boundary.
                //
                // P22: only the claim-or-interject decision happens before
                // the ack. Everything heavier — recall, workspace context,
                // memory writes, planning — runs inside the spawned turn
                // (`prepare_chat_turn`), so this response is a genuine
                // DELIVERY ack in milliseconds, not a progress report.
                match self.run_chat_turn_with(&session_id, &goal, images).await {
                    // The run proceeds in a spawned task; ACK immediately so
                    // the IPC request never outlives the client's patience —
                    // a run can last hours, and the transcript is driven by
                    // events, not by this response.
                    Ok(Some(message_id)) => {
                        super::chat_harness::started_response(&message_id)
                    }
                    // The message joined the run already in flight.
                    Ok(None) => {
                        let depth = self
                            .chat_runs
                            .pending_for(&session_id)
                            .await
                            .len()
                            .await;
                        super::chat_harness::interjected_response(&session_id, depth)
                    }
                    // Only reachable when the daemon is degraded (agent,
                    // router, tools or storage missing) — report it honestly
                    // rather than pretending a second chat path exists.
                    Err(message) => {
                        warn!(%message, "long-horizon chat could not start");
                        json!({
                            "error": "chat_failed",
                            "message": message,
                        })
                    }
                }
            }
            ChatAction::Cancel { session_id } => {
                info!("Chat cancel for session {}", session_id);
                if let Some(ref agent) = self.agent {
                    let cancelled = agent.cancel(&session_id).await;
                    json!({ "status": if cancelled { "cancelled" } else { "not_active" }, "session_id": session_id })
                } else {
                    json!({ "error": "agent_unavailable" })
                }
            }
            ChatAction::Regenerate { session_id } => {
                info!("Chat regenerate for session {}", session_id);
                // Drop the stale assistant reply, recover the user message that
                // produced it, and replay the turn through the normal send path
                // (which re-adds the user message, rebuilds context, and runs
                // the agent) — so regeneration reuses all of Send's logic.
                let Some(mut session) = self.sessions.get(&session_id).await else {
                    return json!({
                        "error": "session_not_found",
                        "message": format!("Session {session_id} not found")
                    });
                };
                let Some(content) = session.take_last_user_turn() else {
                    return json!({
                        "status": "nothing_to_regenerate",
                        "session_id": session_id,
                        "message": "No user message to regenerate from"
                    });
                };
                // `replace`, not `update`: the dropped turn must leave the
                // store too, or it returns on the next restart.
                self.sessions.replace(session).await;
                Box::pin(self.handle_chat(
                    client_id,
                    ChatAction::Send {
                        session_id,
                        content,
                        attachments: Vec::new(),
                    },
                ))
                .await
            }
        }
    }

    /// Cancel the turn in `session_id` on behalf of a channel command, through
    /// the same arm the IPC `chat.cancel` verb uses.
    pub(crate) async fn handle_chat_cancel_for_channel(
        self: &Arc<Self>,
        session_id: &str,
    ) -> Value {
        self.handle_chat(
            "channel:command",
            ChatAction::Cancel {
                session_id: session_id.to_string(),
            },
        )
        .await
    }

    /// Assemble everything a turn needs beyond the delivery ack: system
    /// prompt (persona + recall + conversation), planner conversation,
    /// resolved workspace. Runs INSIDE the spawned turn, after the ack is on
    /// the wire — see [`ChatTurnPrep`] for why.
    pub(super) async fn prepare_chat_turn(
        &self,
        session_id: &str,
        content: &str,
    ) -> Result<ChatTurnPrep, String> {
        let Some(ref agent) = self.agent else {
            return Err("Agent service not configured".to_string());
        };

        // Auto-remember user message only when the user has opted in
        // (`[memory] auto_remember_messages = true`). Default is off so a
        // first-run install does not silently store conversation content.
        let auto_remember = self.config.read().await.memory.auto_remember_messages;
        if auto_remember
            && let Some(ref memory) = self.memory
            && content.split_whitespace().count() >= 3
        {
            let meta = std::collections::HashMap::new();
            if let Err(e) = memory.remember_with_importance(content, meta, 1.0).await {
                debug!("Failed to auto-remember user message: {}", e);
            }
        }

        // Get session history (all messages *before* the one just added)
        let Some(session) = self.sessions.get(session_id).await else {
            return Err(format!("Session {session_id} not found"));
        };

        // Prior messages = everything except the last one (the user message we just added)
        let prior_messages: Vec<_> = if session.messages.len() > 1 {
            session.messages[..session.messages.len() - 1].to_vec()
        } else {
            Vec::new()
        };

        // Build system prompt with persona + memory + workspace context
        let mut system_prompt = self.system_prompt.read().await.clone();

        // Global persona / user profile (config — independent of workspace)
        {
            let cfg = self.config.read().await;
            let persona = nanna_core::GlobalPersona {
                persona: cfg.agent.persona.clone(),
                user_profile: cfg.agent.user_profile.clone(),
            };
            drop(cfg);
            let inj = persona.build_system_prompt_injection();
            if !inj.is_empty() {
                system_prompt.push_str("

");
                system_prompt.push_str(&inj);
            }
        }

        let (effective_ws_id, ws_source) = self.resolve_turn_workspace(session_id, &session).await;

        // The shared slot must name the session being prepared before anything
        // derived from it is written. Nothing in a run reads it any more — the
        // turn carries its own binding (`ToolRegistry::with_run_session`) — but
        // the control-plane callers of `set_default_workdir` still do.
        agent.tools().set_session_id(Some(session_id.to_string())).await;

        let workspace_root = if let Some(ref ws_id) = effective_ws_id {
            let registry = self.workspaces.read().await;
            registry.get(ws_id).map(|ws| ws.path.clone())
        } else {
            None
        };

        // Whatever we resolved — including NOTHING — is recorded against THIS
        // session. Keyed here, not inferred from the shared slot, so an incoming
        // turn cannot file its root under the outgoing session's key: that is
        // what turned "this session has no workspace" into "this session writes
        // into the last session's directory", and then into a turn already
        // streaming resolving its relative paths inside another project.
        //
        // This binding is the ONLY workdir write a chat turn makes, and it is
        // turn-scoped: `run_chat_turn`'s release tail clears it again. The two
        // `set_default_workdir` calls that used to live here (one for the
        // resolved root, one clearing it when nothing resolved) are deliberately
        // gone, and must not come back:
        //
        //   - `set_default_workdir` writes the PROCESS-WIDE slot. Writing it per
        //     chat turn made a global mean "the last chat turn's workspace",
        //     which is the cross-session re-rooting this per-session map exists
        //     to end.
        //   - It is documented and asserted control-plane-only:
        //     `debug_assert!(RUN_SESSION_ID is unset)`. `prepare_chat_turn` runs
        //     inside the turn's `with_run_session` scope, so calling it here
        //     would now trip that assert rather than merely being unwise.
        //
        // Nothing is left unmaintained by their removal. The global keeps the
        // meaning its own doc gives it — the ACTIVE workspace — and keeps the
        // writers that give it that meaning: `WorkspaceAction::SetActive` /
        // `ClearActive`, and the boot seeding from the persisted active
        // workspace. Readers that want THIS session's root read it under this
        // session's scope and get the binding above; readers with no session in
        // hand fall through to the active workspace, which is the honest answer
        // when nobody knows who is asking.
        agent.tools().bind_session_workdir(session_id, workspace_root.clone()).await;

        self.log_turn_workspace(session_id, effective_ws_id.as_ref(), ws_source).await;

        // Inject workspace context (reload from disk so edits within the session are picked up)
        let workspace_context = match effective_ws_id {
            Some(ref ws_id) => self.inject_workspace_context(ws_id, &mut system_prompt).await,
            None => None,
        };

        // Add memory context if available (gate on message complexity)
        Self::append_recalled_memories(
            agent,
            content,
            effective_ws_id.as_deref(),
            &prior_messages,
            &mut system_prompt,
        )
        .await;

        // Update workspace ID for script services (memory scoping)
        if let Some(ref ws_arc) = self.services_workspace_id {
            *ws_arc.write().await = effective_ws_id.clone();
        }

        // Each harness step re-anchors from the task store, so the
        // conversation must ride in the system prompt (the retired
        // direct path passed it as a message array). The same bounded
        // rendering is handed to the planner as context.
        let conversation = super::chat_harness::conversation_context(&prior_messages);
        if let Some(ref conversation) = conversation {
            system_prompt.push_str("\n\n## Conversation so far\n");
            system_prompt.push_str(conversation);
        }

        Ok(ChatTurnPrep {
            system_prompt,
            conversation,
            workspace_root,
            workspace_context,
            // Read from the session snapshot taken at the top of this function,
            // so the pin the turn honours is the one that was set when the
            // message was accepted — not one a Settings click may land halfway
            // through a run that has already started streaming.
            chat_model: session.chat_model().map(str::to_string),
            chat_tools: session.chat_tools(),
        })
    }

    /// Resolve the turn's workspace: the session's own workspace when the
    /// registry still has it, otherwise the globally active one. Returns the
    /// id and which of the two supplied it (`"session"` or `"active"`).
    ///
    /// The session's id is VALIDATED against the registry first. An
    /// id the registry does not contain used to suppress the
    /// fallback and then fail every lookup silently: the turn ran
    /// with no workspace context, `workspace_root = None`, and
    /// whatever `default_workdir` some earlier code path had left on
    /// the tool registry — i.e. it executed in a different
    /// directory than anyone believed, with no error anywhere. A
    /// session pointing at a workspace that no longer exists must
    /// degrade to the active one loudly, not run somewhere random.
    async fn resolve_turn_workspace(
        &self,
        session_id: &str,
        session: &crate::session::Session,
    ) -> (Option<String>, &'static str) {
        let registry = self.workspaces.read().await;
        let session_ws = session
            .workspace_id
            .clone()
            .filter(|id| registry.get(id).is_some());
        if session.workspace_id.is_some() && session_ws.is_none() {
            warn!(
                session_id = %session_id,
                missing = ?session.workspace_id,
                "session names a workspace the registry does not have — \
                 falling back to the active workspace"
            );
        }
        session_ws.map_or_else(
            || (registry.active().map(|ws| ws.id.clone()), "active"),
            |id| (Some(id), "session"),
        )
    }

    /// SAY which workspace won. The precedence itself is right, but it was
    /// silent, and a silent override is indistinguishable from a
    /// bug: activating a workspace over IPC appeared to work while
    /// every turn quietly used the session's instead. Three
    /// consecutive "fresh workspace" benchmark runs wrote into the
    /// FIRST workspace's directory before anyone noticed (one of
    /// them scored 0/42 while its artifact grew next door).
    async fn log_turn_workspace(
        &self,
        session_id: &str,
        effective_ws_id: Option<&String>,
        ws_source: &str,
    ) {
        let registry = self.workspaces.read().await;
        let active_id = registry.active().map(|ws| ws.id.clone());
        let path = effective_ws_id
            .and_then(|id| registry.get(id))
            .map(|ws| ws.path.display().to_string());
        drop(registry);
        let overridden = ws_source == "session"
            && active_id.is_some()
            && active_id.as_ref() != effective_ws_id;
        if overridden {
            info!(
                session_id = %session_id,
                using = ?effective_ws_id,
                path = ?path,
                active_workspace = ?active_id,
                "chat turn uses the SESSION's workspace, overriding the active one"
            );
        } else {
            info!(
                session_id = %session_id,
                source = ws_source,
                using = ?effective_ws_id,
                path = ?path,
                "chat turn workspace resolved"
            );
        }
    }

    /// Reload `ws_id`'s context from disk (so edits within the session are
    /// picked up), name its root in `system_prompt`, and hand back the bounded
    /// workspace files that ride beside the prompt.
    async fn inject_workspace_context(
        &self,
        ws_id: &str,
        system_prompt: &mut String,
    ) -> Option<String> {
        {
            let mut registry = self.workspaces.write().await;
            if let Some(ws) = registry.get_mut(ws_id) {
                // The git-aware variant, and only here: this is the one
                // workspace the turn is bound to, so the two `git` calls it
                // costs are paid once per turn rather than once per
                // registered workspace at boot.
                if let Err(e) = ws.load_context_with_git().await {
                    warn!("Failed to reload workspace context: {}", e);
                }
            }
        }
        let registry = self.workspaces.read().await;
        let ws = registry.get(ws_id)?;
        // Add workspace root path prominently so model knows where to look
        let ws_path = ws.path.display();
        let _ = write!(
            system_prompt,
            "\n\n## Active Workspace\n\
            **Root directory: {ws_path}**\n\
            All file operations and commands MUST use this directory as the base.\n\
            Use relative paths (resolved against {ws_path}) or absolute paths within it.\n\
            Do NOT search in home directory or other locations unless explicitly asked.\n"
        );

        // Workspace files (README.md, AGENTS.md, ROADMAP.md, …) ride
        // OUTSIDE the system prompt: as `AgentContext::workspace_context`
        // each step bounds them at the model-window-derived cap
        // (`workspace_context_cap_chars`) with a marker that announces
        // the cut. Appended here they were unbounded — a long
        // ROADMAP.md dominated a 16k window and read as a work order
        // (observed live 2026-08-02: "mutex vs semaphore?" answered
        // with a roadmap status report, next turn created roadmap
        // todos unasked).
        let ws_context = ws.context.build_system_prompt_injection();
        drop(registry);
        if ws_context.is_empty() {
            None
        } else {
            Some(ws_context)
        }
    }

    /// Recall memories relevant to `content` (gated on message complexity)
    /// and append the ones recent history does not already show to
    /// `system_prompt`, each bounded.
    async fn append_recalled_memories(
        agent: &crate::agent_service::AgentService,
        content: &str,
        workspace_id: Option<&str>,
        prior_messages: &[crate::session::SessionMessage],
        system_prompt: &mut String,
    ) {
        // Bounded per memory, and it must be.
        //
        // These go in VERBATIM. A merged entry may now hold
        // four maximum-size observations, so five recalled
        // memories could push ~136 KB into the system prompt
        // of a model with a 32 k window — the recall meant to
        // orient the turn would instead evict the plan it was
        // recalled to serve.
        //
        // The owner's rule already says what to do here:
        // context gets a summary, and recall is how you get
        // the whole thing. So cut at a char boundary and say
        // so, rather than silently truncating or silently
        // flooding.
        const RECALL_INJECT_MAX_CHARS: usize = 1_200;

        let should_recall = content.split_whitespace().count() > 5
            || content.contains('?')
            || content.len() > 80;
        if !should_recall {
            return;
        }

        // Scoped recall: workspace sessions see global + workspace memories
        let memories = agent.recall_memories_scoped(
            content, 5, workspace_id
        ).await;
        if memories.is_empty() {
            return;
        }
        // Dedup: skip memories whose content already appears in recent history
        let recent_text: String = prior_messages.iter()
            .rev().take(4)
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        let fresh_memories: Vec<_> = memories.into_iter()
            .filter(|m| {
                // Find a safe char boundary for the snippet (max 100 bytes)
                let max = m.content.len().min(100);
                let end = m.content.floor_char_boundary(max);
                let snippet = &m.content[..end];
                !recent_text.contains(snippet)
            })
            .collect();

        if fresh_memories.is_empty() {
            return;
        }
        system_prompt.push_str("\n\n## Remembered Context\n");
        for mem in fresh_memories {
            if mem.content.chars().count() <= RECALL_INJECT_MAX_CHARS {
                let _ = writeln!(system_prompt, "- {}", mem.content);
                continue;
            }
            let head: String = mem
                .content
                .chars()
                .take(RECALL_INJECT_MAX_CHARS)
                .collect();
            let _ = write!(
                system_prompt,
                "- {head}…\n  [OPENING {RECALL_INJECT_MAX_CHARS} CHARS of a \
                 longer memory, shown so it cannot crowd out your context. \
                 Nothing was lost — recall(\"{}\") returns it whole.]\n",
                mem.id
            );
        }
    }
}
