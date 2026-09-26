//! Task control actions (P15 store + P14 long-horizon runs).

use super::{json, Arc, ControlPlane, PathBuf, TaskAction, Value};
use crate::tasks::{AgentStepRunner, TursoTaskSource};
use nanna_agent::harness::LongHorizonConfig;
use nanna_storage::{NewTask, TaskPatch, TaskRepository};

impl ControlPlane {
    /// Resolve `(scope, scope_id)` for a task action. Workspace scope binds
    /// to the active workspace; session scope requires an explicit id.
    ///
    /// Control-plane clients have no session of their own to fall back on
    /// (unlike a tool call, where the bridge supplies the running session), so
    /// the error names the field to send and the scopes that need no id.
    async fn resolve_task_scope(
        &self,
        scope: Option<&str>,
        session_id: Option<&str>,
    ) -> Result<(String, Option<String>), String> {
        let scope = scope.unwrap_or("session").to_lowercase();
        match scope.as_str() {
            "session" => {
                let session_id = session_id.filter(|s| !s.is_empty()).ok_or_else(|| {
                    "session scope requires a session id — send \"session_id\": \"<id>\" with \
                     this task action, or use \"scope\": \"workspace\" (the active workspace) \
                     or \"scope\": \"global\""
                        .to_string()
                })?;
                Ok(("session".to_string(), Some(session_id.to_string())))
            }
            "workspace" => {
                // The guard only resolves the active id; nothing after it
                // needs the registry locked.
                let active_id = self
                    .workspaces
                    .read()
                    .await
                    .active()
                    .map(|ws| ws.id.clone())
                    .ok_or_else(|| "workspace scope requires an active workspace".to_string())?;
                Ok(("workspace".to_string(), Some(active_id)))
            }
            "global" => Ok(("global".to_string(), None)),
            other => Err(format!("unknown scope '{other}'")),
        }
    }

    pub(super) async fn handle_task(&self, _client_id: &str, action: TaskAction) -> Value {
        let Some(ref storage) = self.storage else {
            return json!({"error": "storage_unavailable", "message": "task store requires storage"});
        };
        let repo = storage.tasks();

        match action {
            TaskAction::List { scope, session_id, include_closed } => {
                self.task_list(&repo, scope, session_id, include_closed).await
            }

            TaskAction::Get { id } => {
                let task = match repo.get(id).await {
                    Ok(task) => task,
                    Err(e) => return json!({"error": "task_not_found", "message": e.to_string()}),
                };
                let notes = repo.notes(id, 50).await.unwrap_or_default();
                let activity = repo.activity(id, 50).await.unwrap_or_default();
                json!({"task": task, "notes": notes, "activity": activity})
            }

            TaskAction::Next { scope, session_id } => self.task_next(&repo, scope, session_id).await,

            TaskAction::Create {
                title, scope, session_id, parent_id, description, priority, labels, tools,
                due_at, deadline_at, recurrence, depends_on, acceptance, project, assignee,
            } => {
                let request = CreateTask {
                    title, scope, session_id, parent_id, description, priority, labels, tools,
                    due_at, deadline_at, recurrence, depends_on, acceptance, project, assignee,
                };
                self.task_create(&repo, request).await
            }

            TaskAction::Update { id, patch } => Self::task_update(&repo, id, &patch).await,

            TaskAction::Done { id, workdir } => self.task_done(&repo, id, workdir).await,

            TaskAction::Delete { id } => match repo.delete(id, Some("gui")).await {
                Ok(removed) => json!({"removed": removed}),
                Err(e) => json!({"error": "task_delete_failed", "message": e.to_string()}),
            },

            TaskAction::Note { id, content } => {
                // A note from the GUI is the human posting on the card's
                // thread, so it names the human member (P25 decision 2). The
                // legacy `author` string stays "gui" until Stage 4 drops it.
                match repo
                    .post(
                        id,
                        Some("gui"),
                        Some(nanna_storage::HUMAN_MEMBER_ID),
                        nanna_storage::TaskNoteKind::Comment,
                        &content,
                    )
                    .await
                {
                    Ok(note) => json!({"note": note}),
                    Err(e) => json!({"error": "task_note_failed", "message": e.to_string()}),
                }
            }

            TaskAction::Query { filter, scope, session_id } => {
                self.task_query(&repo, &filter, scope, session_id).await
            }

            TaskAction::StartRun {
                goal,
                scope,
                session_id,
                workdir,
                max_wall_clock_secs,
                max_total_tokens,
            } => {
                let limits = (max_wall_clock_secs, max_total_tokens);
                self.start_task_run(storage, goal, (scope, session_id), workdir, limits).await
            }

            TaskAction::RunStatus { scope, session_id } => self.task_run_status(scope, session_id).await,

            TaskAction::CancelRun { scope, session_id } => self.cancel_task_run(scope, session_id).await,

            TaskAction::Verdicts { window } => Self::task_verdicts(&repo, window).await,
        }
    }

    /// `TaskAction::Verdicts`: the per-member, per-label verdict rollup.
    async fn task_verdicts(repo: &TaskRepository, window: Option<usize>) -> Value {
        let window = window.unwrap_or(crate::protocol::TASK_VERDICT_WINDOW_DEFAULT);
        if window == 0 || window > nanna_storage::VERDICT_WINDOW_MAX {
            return json!({
                "error": "bad_window",
                "message": format!(
                    "window must be in 1..={}, got {window}",
                    nanna_storage::VERDICT_WINDOW_MAX
                ),
            });
        }
        match repo.verdict_rollup(window).await {
            Ok(tallies) => json!({ "window": window, "verdicts": tallies }),
            Err(e) => json!({"error": "task_verdicts_failed", "message": e.to_string()}),
        }
    }

    /// `(scope, scope_id)` for a task action, or the `bad_scope` reply.
    async fn task_scope_or_reply(
        &self,
        scope: Option<String>,
        session_id: Option<String>,
    ) -> Result<(String, Option<String>), Value> {
        self.resolve_task_scope(scope.as_deref(), session_id.as_deref())
            .await
            .map_err(|message| json!({"error": "bad_scope", "message": message}))
    }

    /// `TaskAction::List`.
    async fn task_list(
        &self,
        repo: &TaskRepository,
        scope: Option<String>,
        session_id: Option<String>,
        include_closed: Option<bool>,
    ) -> Value {
        let (scope, scope_id) = match self.task_scope_or_reply(scope, session_id).await {
            Ok(resolved) => resolved,
            Err(reply) => return reply,
        };
        match repo
            .list(&scope, scope_id.as_deref(), include_closed.unwrap_or(true))
            .await
        {
            Ok(tasks) => json!({"tasks": tasks}),
            Err(e) => json!({"error": "task_list_failed", "message": e.to_string()}),
        }
    }

    /// `TaskAction::Next`.
    async fn task_next(
        &self,
        repo: &TaskRepository,
        scope: Option<String>,
        session_id: Option<String>,
    ) -> Value {
        let (scope, scope_id) = match self.task_scope_or_reply(scope, session_id).await {
            Ok(resolved) => resolved,
            Err(reply) => return reply,
        };
        match repo.next(&scope, scope_id.as_deref()).await {
            Ok(task) => json!({"task": task}),
            Err(e) => json!({"error": "task_next_failed", "message": e.to_string()}),
        }
    }

    /// `TaskAction::Create`: place the task (a subtask in its parent's scope),
    /// canonicalize its acceptance, and create it.
    async fn task_create(&self, repo: &TaskRepository, request: CreateTask) -> Value {
        let CreateTask {
            title, scope, session_id, parent_id, description, priority, labels, tools,
            due_at, deadline_at, recurrence, depends_on, acceptance, project, assignee,
        } = request;
        // A subtask always lives in its parent's scope and inherits
        // its ladder position; a new root task appends after
        // everything (sort_order 0 would jump the whole queue).
        let (scope, scope_id, sort_order) = if let Some(parent_id) = parent_id {
            match repo.get(parent_id).await {
                Ok(parent) => (parent.scope, parent.scope_id, parent.sort_order),
                Err(e) => {
                    return json!({"error": "task_not_found", "message": e.to_string()});
                }
            }
        } else {
            let (scope, scope_id) = match self
                .resolve_task_scope(scope.as_deref(), session_id.as_deref())
                .await
            {
                Ok(resolved) => resolved,
                Err(message) => return json!({"error": "bad_scope", "message": message}),
            };
            let append_after = repo
                .list(&scope, scope_id.as_deref(), true)
                .await
                .map_or(0, |tasks| {
                    tasks
                        .iter()
                        .map(|t| t.sort_order)
                        .max()
                        .unwrap_or(0)
                        .saturating_add(1)
                });
            (scope, scope_id, append_after)
        };
        // Canonicalize through the harness parser: a shape that would
        // fail at run time is rejected here, not mid-run.
        let acceptance = match acceptance {
            Some(raw) => {
                match nanna_agent::harness::AcceptanceCheck::from_json(&raw)
                    .and_then(|c| serde_json::to_value(&c).map_err(|e| e.to_string()))
                {
                    Ok(canonical) => Some(canonical),
                    Err(message) => {
                        return json!({"error": "bad_acceptance", "message": message});
                    }
                }
            }
            None => None,
        };
        let new = NewTask {
            parent_id,
            scope,
            scope_id,
            project,
            title,
            description,
            priority: priority.unwrap_or(3),
            labels: labels.unwrap_or_default(),
            tool_scope: tools.unwrap_or_default(),
            due_at,
            deadline_at,
            recurrence,
            depends_on: depends_on.unwrap_or_default(),
            acceptance,
            assignee,
            sort_order,
        };
        // No `created` event is emitted here. `TaskRepository::create` emits
        // `Event::TaskEvent{ kind: Created }` for every writer, not just this
        // one; announcing it again from the IPC path would double-report a
        // card created through the GUI and report nothing for a card created
        // by a tool. (It also used to ride on `TaskRunProgress`, which is the
        // harness *run* family — a store mutation wearing run clothes.)
        match repo.create(new).await {
            Ok(task) => json!({"task": task}),
            Err(e) => json!({"error": "task_create_failed", "message": e.to_string()}),
        }
    }

    /// `TaskAction::Update`: apply a partial patch.
    async fn task_update(repo: &TaskRepository, id: i64, patch: &Value) -> Value {
        let string_vec = |v: &Value| -> Vec<String> {
            v.as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default()
        };
        // Null/absent/mistyped values SKIP a field, never wipe it —
        // partial patches must not clear what they did not mention.
        let acceptance = match patch.get("acceptance").filter(|v| !v.is_null()) {
            Some(raw) => {
                match nanna_agent::harness::AcceptanceCheck::from_json(raw)
                    .and_then(|c| serde_json::to_value(&c).map_err(|e| e.to_string()))
                {
                    Ok(canonical) => Some(Some(canonical)),
                    Err(message) => {
                        return json!({"error": "bad_acceptance", "message": message});
                    }
                }
            }
            None => None,
        };
        let task_patch = TaskPatch {
            title: patch
                .get("title")
                .and_then(Value::as_str)
                .map(str::to_string),
            description: patch
                .get("description")
                .and_then(Value::as_str)
                .map(|s| Some(s.to_string())),
            status: patch
                .get("status")
                .and_then(Value::as_str)
                .map(str::to_string),
            priority: patch.get("priority").and_then(Value::as_i64),
            labels: patch
                .get("labels")
                .filter(|v| v.is_array())
                .map(&string_vec),
            tool_scope: patch.get("tools").filter(|v| v.is_array()).map(&string_vec),
            due_at: patch
                .get("due_at")
                .and_then(Value::as_str)
                .map(|s| Some(s.to_string())),
            deadline_at: patch
                .get("deadline_at")
                .and_then(Value::as_str)
                .map(|s| Some(s.to_string())),
            recurrence: patch
                .get("recurrence")
                .and_then(Value::as_str)
                .map(|s| Some(s.to_string())),
            depends_on: patch.get("depends_on").filter(|v| v.is_array()).map(|v| {
                v.as_array()
                    .map(|arr| arr.iter().filter_map(Value::as_i64).collect())
                    .unwrap_or_default()
            }),
            acceptance,
            assignee: patch
                .get("assignee")
                .and_then(Value::as_str)
                .map(|s| Some(s.to_string())),
            parent_id: patch.get("parent_id").and_then(Value::as_i64).map(Some),
            project: patch
                .get("project")
                .and_then(Value::as_str)
                .map(|s| Some(s.to_string())),
            sort_order: patch.get("sort_order").and_then(Value::as_i64),
        };
        match repo.update(id, task_patch, Some("gui")).await {
            Ok(task) => json!({"task": task}),
            Err(e) => json!({"error": "task_update_failed", "message": e.to_string()}),
        }
    }

    /// `TaskAction::Done`: run the acceptance check, then record completion.
    async fn task_done(&self, repo: &TaskRepository, id: i64, workdir: Option<String>) -> Value {
        // Same verdict-first flow as the tasks.done service: run the
        // acceptance check before completion is recorded.
        let task = match repo.get(id).await {
            Ok(task) => task,
            Err(e) => return json!({"error": "task_not_found", "message": e.to_string()}),
        };
        if let Some(acceptance) = &task.acceptance {
            let check = match nanna_agent::harness::AcceptanceCheck::from_json(acceptance) {
                Ok(check) => check,
                Err(message) => {
                    return json!({"error": "bad_acceptance", "message": message});
                }
            };
            // Default to the active workspace root — the daemon's own
            // cwd is meaningless for workspace artifacts.
            let dir = match workdir {
                Some(dir) => PathBuf::from(dir),
                None => self
                    .workspaces
                    .read()
                    .await
                    .active()
                    .map_or_else(|| PathBuf::from("."), |w| w.path.clone()),
            };
            let verdict = check.run(&dir).await;
            let _ = repo
                .log_activity(
                    id,
                    Some("gui"),
                    "acceptance_checked",
                    Some(json!({"passed": verdict.passed, "detail": verdict.detail})),
                )
                .await;
            if !verdict.passed {
                return json!({"done": false, "verdict": verdict.detail});
            }
        }
        match repo.complete(id, Some("gui"), None).await {
            Ok(outcome) => json!({
                "done": true,
                "already_done": outcome.already_done,
                "auto_completed": outcome.auto_completed,
            }),
            Err(e) => json!({"error": "task_done_failed", "message": e.to_string()}),
        }
    }

    /// `TaskAction::Query`.
    async fn task_query(
        &self,
        repo: &TaskRepository,
        filter: &str,
        scope: Option<String>,
        session_id: Option<String>,
    ) -> Value {
        let (scope, scope_id) = match self.task_scope_or_reply(scope, session_id).await {
            Ok(resolved) => resolved,
            Err(reply) => return reply,
        };
        match repo.query(&scope, scope_id.as_deref(), filter).await {
            Ok(tasks) => json!({"tasks": tasks}),
            Err(e) => json!({"error": "task_query_failed", "message": e.to_string()}),
        }
    }

    /// `TaskAction::StartRun`: start a background long-horizon run.
    /// `scope_request` is `(scope, session_id)`; `limits` is
    /// `(max_wall_clock_secs, max_total_tokens)`.
    async fn start_task_run(
        &self,
        storage: &Arc<nanna_storage::Storage>,
        goal: String,
        (scope, session_id): (Option<String>, Option<String>),
        workdir: Option<String>,
        (max_wall_clock_secs, max_total_tokens): (Option<u64>, Option<u64>),
    ) -> Value {
        let Some(ref task_runs) = self.task_runs else {
            return json!({"error": "task_runs_unavailable", "message": "run manager not attached"});
        };
        let (Some(agent), Some(router), Some(tools)) = (
            self.agent.as_ref(),
            self.router.as_ref(),
            self.tools.as_ref(),
        ) else {
            return json!({"error": "agent_unavailable", "message": "agent service required"});
        };
        let Some(event_tx) = self.event_tx.clone() else {
            return json!({"error": "events_unavailable", "message": "event bus required"});
        };
        let (scope, scope_id) = match self
            .resolve_task_scope(scope.as_deref(), session_id.as_deref())
            .await
        {
            Ok(resolved) => resolved,
            Err(message) => return json!({"error": "bad_scope", "message": message}),
        };

        // Workdir: explicit > active workspace root > current dir.
        let workspace_root = self
            .workspaces
            .read()
            .await
            .active()
            .map(|w| w.path.clone());
        let dir = workdir
            .map(PathBuf::from)
            .or_else(|| workspace_root.clone())
            .unwrap_or_else(|| PathBuf::from("."));

        let source = TursoTaskSource::new(
            storage.clone(),
            scope.clone(),
            scope_id.clone(),
            "harness".to_string(),
            Some(event_tx.clone()),
        );
        let runner = AgentStepRunner {
            discovered_tools: Arc::new(tokio::sync::RwLock::new(std::collections::HashSet::new())),
            // Background runs have no chat, so no user tool picks.
            user_selected_tools: Vec::new(),
            // One ledger for the whole background run — a long-horizon
            // run is exactly where a per-step reset hurts most.
            repeat_ledger: Arc::new(nanna_agent::RepeatLedger::new()),
            router: router.clone(),
            tools: tools.clone(),
            agent_config: agent.agent_config().await,
            system_prompt: self.system_prompt.read().await.clone(),
            workspace_root,
            // Background runs never carried the workspace files; the
            // goal names its own work.
            workspace_context: None,
            stats: Some(self.model_stats.clone()),
            // Background run: no transcript to stream into.
            chat_sink: None,
            memory: None,
            workspace_id: None,
            gpu_fault_count: Arc::new(std::sync::atomic::AtomicU32::new(0)),
            degradations: self.degradations.clone(),
            // A task run is not answering a message: nothing was attached.
            attachments: Arc::default(),
            // Background runs fall back down the priority list like chat.
            model_chain: {
                let models = agent.chat_model_chain().await;
                (!models.is_empty())
                    .then(|| Arc::new(crate::tasks::ModelChain::new(models, router)))
            },
        };
        let mut config = LongHorizonConfig::default();
        if let Some(secs) = max_wall_clock_secs {
            config.max_wall_clock = std::time::Duration::from_secs(secs);
        }
        config.max_total_tokens = max_total_tokens;

        match task_runs
            .start(goal, source, runner, config, dir, event_tx)
            .await
        {
            Ok(()) => json!({"started": true, "scope": scope, "scope_id": scope_id}),
            Err(message) => json!({"error": "run_start_failed", "message": message}),
        }
    }

    /// `TaskAction::RunStatus`.
    async fn task_run_status(&self, scope: Option<String>, session_id: Option<String>) -> Value {
        let Some(ref task_runs) = self.task_runs else {
            return json!({"error": "task_runs_unavailable", "message": "run manager not attached"});
        };
        let (scope, scope_id) = match self
            .resolve_task_scope(scope.as_deref(), session_id.as_deref())
            .await
        {
            Ok(resolved) => resolved,
            Err(message) => return json!({"error": "bad_scope", "message": message}),
        };
        let status = task_runs.status(&scope, scope_id.as_deref()).await;
        serde_json::to_value(&status).unwrap_or_else(|_| json!({"running": false}))
    }

    /// `TaskAction::CancelRun`.
    async fn cancel_task_run(&self, scope: Option<String>, session_id: Option<String>) -> Value {
        let Some(ref task_runs) = self.task_runs else {
            return json!({"error": "task_runs_unavailable", "message": "run manager not attached"});
        };
        let (scope, scope_id) = match self
            .resolve_task_scope(scope.as_deref(), session_id.as_deref())
            .await
        {
            Ok(resolved) => resolved,
            Err(message) => return json!({"error": "bad_scope", "message": message}),
        };
        let cancelled = task_runs.cancel(&scope, scope_id.as_deref()).await;
        json!({"cancelled": cancelled})
    }
}

/// The fields of a `TaskAction::Create`.
struct CreateTask {
    title: String,
    scope: Option<String>,
    session_id: Option<String>,
    parent_id: Option<i64>,
    description: Option<String>,
    priority: Option<i64>,
    labels: Option<Vec<String>>,
    tools: Option<Vec<String>>,
    due_at: Option<String>,
    deadline_at: Option<String>,
    recurrence: Option<String>,
    depends_on: Option<Vec<i64>>,
    /// Boxed to match `TaskAction::Create`, whose field it is moved from; see
    /// the note there. Unboxed again by `task_create` before it reaches
    /// `NewTask`, which stores the canonicalized value inline.
    acceptance: Option<Box<Value>>,
    project: Option<String>,
    assignee: Option<String>,
}
