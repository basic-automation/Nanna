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
    pub(super) async fn resolve_task_scope(
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

            TaskAction::QuickAdd { text, scope, parent_id } => {
                self.task_quick_add(&repo, &text, scope, parent_id).await
            }

            TaskAction::Assigned { member_id, limit } => {
                Self::task_assigned(&repo, member_id, limit).await
            }

            TaskAction::Update { id, patch } => Self::task_update(&repo, id, &patch).await,

            TaskAction::Done { id, workdir } => self.task_done(&repo, id, workdir).await,

            TaskAction::Delete { id } => match repo.delete(id, Some("gui")).await {
                Ok(removed) => json!({"removed": removed}),
                Err(e) => json!({"error": "task_delete_failed", "message": e.to_string()}),
            },

            TaskAction::Note { id, content } => Self::task_note(&repo, id, &content).await,

            TaskAction::Query { filter, scope, session_id } => {
                self.task_query(&repo, &filter, scope, session_id).await
            }

            TaskAction::StartRun {
                card_id: Some(card_id),
                workdir,
                max_wall_clock_secs,
                max_total_tokens,
                ..
            } => {
                let limits = (max_wall_clock_secs, max_total_tokens);
                self.start_card_run(storage, card_id, workdir, limits).await
            }

            TaskAction::StartRun {
                card_id: None,
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

            TaskAction::RunStatus {
                card_id: Some(card_id),
                ..
            } => self.card_run_status(card_id).await,

            TaskAction::RunStatus {
                card_id: None,
                scope,
                session_id,
            } => self.task_run_status(scope, session_id).await,

            TaskAction::CancelRun {
                card_id: Some(card_id),
                ..
            } => self.cancel_card_run(card_id).await,

            TaskAction::CancelRun {
                card_id: None,
                scope,
                session_id,
            } => self.cancel_task_run(scope, session_id).await,

            TaskAction::Verdicts { window } => Self::task_verdicts(&repo, window).await,
        }
    }

    /// `TaskAction::Note`: a note from the GUI is the human posting on the
    /// card's thread, so it names the human member (P25 decision 2). The
    /// legacy `author` string stays "gui" until Stage 4 drops it.
    async fn task_note(repo: &TaskRepository, id: i64, content: &str) -> Value {
        match repo
            .post(
                id,
                Some("gui"),
                Some(nanna_storage::HUMAN_MEMBER_ID),
                nanna_storage::TaskNoteKind::Comment,
                content,
            )
            .await
        {
            Ok(note) => json!({"note": note}),
            Err(e) => json!({"error": "task_note_failed", "message": e.to_string()}),
        }
    }

    /// `TaskAction::Assigned`: a member's open cards on every board.
    async fn task_assigned(
        repo: &TaskRepository,
        member_id: Option<String>,
        limit: Option<usize>,
    ) -> Value {
        let limit = limit.unwrap_or(nanna_storage::ASSIGNED_CARDS_MAX);
        if limit == 0 || limit > nanna_storage::ASSIGNED_CARDS_MAX {
            return json!({
                "error": "bad_limit",
                "message": format!(
                    "limit must be in 1..={}, got {limit}",
                    nanna_storage::ASSIGNED_CARDS_MAX
                ),
            });
        }
        let member_id =
            member_id.unwrap_or_else(|| nanna_storage::HUMAN_MEMBER_ID.to_string());
        let today = chrono::Utc::now().format("%Y-%m-%d").to_string();
        match repo.assigned_open(&member_id, limit).await {
            Ok(cards) => json!({ "member_id": member_id, "today": today, "cards": cards }),
            Err(e) => json!({"error": "task_assigned_failed", "message": e.to_string()}),
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
    pub(super) async fn task_create(&self, repo: &TaskRepository, request: CreateTask) -> Value {
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
            // IPC writers are recorded as `gui`, as its update and complete are.
            created_by: Some("gui".to_string()),
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
                .map(clearable),
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
            // `null` skips a field like everywhere else in a patch, so an
            // empty string is how a client clears a date or a description
            // (the board's inputs send "" when emptied).
            due_at: patch
                .get("due_at")
                .and_then(Value::as_str)
                .map(clearable),
            deadline_at: patch
                .get("deadline_at")
                .and_then(Value::as_str)
                .map(clearable),
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
        let models = agent.chat_model_chain().await;
        let runner = self
            .background_runner((agent, router, tools), workspace_root, models)
            .await;
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

    /// The step runner a background run uses: no chat, no transcript, no
    /// attachments, one repeat ledger for the whole run, and `models` walked
    /// with failover (empty = the router's own default).
    async fn background_runner(
        &self,
        (agent, router, tools): (
            &Arc<crate::agent_service::AgentService>,
            &Arc<crate::llm_router::LlmRouter>,
            &Arc<nanna_tools::ToolRegistry>,
        ),
        workspace_root: Option<PathBuf>,
        models: Vec<String>,
    ) -> AgentStepRunner {
        AgentStepRunner {
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
                (!models.is_empty())
                    .then(|| Arc::new(crate::tasks::ModelChain::new(models, router)))
            },
        }
    }

    /// What a card run needs to know before it may start: the card and the
    /// member it is assigned to. Every refusal is the IPC reply.
    pub(super) async fn card_claim(
        storage: &nanna_storage::Storage,
        card_id: i64,
    ) -> Result<
        (
            nanna_storage::Task,
            nanna_storage::Member,
            crate::tasks::CardClaim,
        ),
        Value,
    > {
        let repo = storage.tasks();
        let card = repo
            .get(card_id)
            .await
            .map_err(|e| json!({"error": "task_not_found", "message": e.to_string()}))?;
        if card.scope == "session" {
            return Err(json!({"error": "not_a_board_card", "message": format!(
                "card #{card_id} is session-scoped; only board cards (workspace or global) are worked by members"
            )}));
        }
        if card.status != "pending" && card.status != "in_progress" {
            return Err(json!({"error": "card_closed", "message": format!(
                "card #{card_id} is {}", card.status
            )}));
        }
        let Some(assignee) = card.assignee.clone() else {
            return Err(json!({"error": "card_unassigned", "message": format!(
                "card #{card_id} has no assignee; assign it to an agent first"
            )}));
        };
        let member = storage
            .members()
            .get(&assignee)
            .await
            .map_err(|e| json!({"error": "member_unreadable", "message": e.to_string()}))?;
        if member.kind != nanna_storage::MemberKind::Agent
            || member.id.starts_with(nanna_storage::ROUTER_MEMBER_PREFIX)
        {
            return Err(json!({"error": "not_an_agent", "message": format!(
                "card #{card_id} is assigned to {}, who is not an agent that takes work", member.id
            )}));
        }
        let claim = crate::tasks::CardClaim {
            card_id,
            member_id: member.id.clone(),
        };
        Ok((card, member, claim))
    }

    /// Start card `card_id`'s run as its assignment does (P25 Stage 3, see
    /// [`crate::card_run_trigger`]): no explicit workdir, the default bounds.
    pub(crate) async fn start_assigned_card(&self, card_id: i64) -> Value {
        let Some(storage) = self.storage.clone() else {
            return json!({"error": "storage_unavailable", "message": "no task store"});
        };
        self.start_card_run(&storage, card_id, None, (None, None))
            .await
    }

    /// `TaskAction::StartRun {card_id}`: the card's assignee works the card's
    /// subtree in the background (P25 Stage 3). Its notes and closings become
    /// posts on the card threads, and it shows as busy until the run ends.
    async fn start_card_run(
        &self,
        storage: &Arc<nanna_storage::Storage>,
        card_id: i64,
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
        let (card, member, claim) = match Self::card_claim(storage, card_id).await {
            Ok(found) => found,
            Err(reply) => return reply,
        };
        // Workdir: explicit > the card's own board > the active workspace.
        let workspace_root = {
            let registry = self.workspaces.read().await;
            card.scope_id
                .as_deref()
                .filter(|_| card.scope == "workspace")
                .and_then(|id| registry.get(id))
                .or_else(|| registry.active())
                .map(|w| w.path.clone())
        };
        let dir = workdir
            .map(PathBuf::from)
            .or_else(|| workspace_root.clone())
            .unwrap_or_else(|| PathBuf::from("."));
        let models = crate::board_router_trigger::router_models(
            &member.profile,
            &agent.chat_model_chain().await,
        );
        let runner = self
            .background_runner((agent, router, tools), workspace_root, models)
            .await;
        let source = TursoTaskSource::new(
            storage.clone(),
            card.scope.clone(),
            card.scope_id.clone(),
            member.id.clone(),
            Some(event_tx.clone()),
        )
        .within_subtree(card_id)
        .posting_as(member.id.clone());
        let mut config = LongHorizonConfig::default();
        if let Some(secs) = max_wall_clock_secs {
            config.max_wall_clock = std::time::Duration::from_secs(secs);
        }
        config.max_total_tokens = max_total_tokens;
        let goal = card.description.as_deref().map_or_else(
            || card.title.clone(),
            |description| format!("{}\n\n{description}", card.title),
        );
        let spec = crate::tasks::TaskRunSpec {
            goal,
            source,
            runner,
            config,
            workdir: dir,
        };
        match task_runs.start_card(spec, claim, event_tx).await {
            Ok(()) => json!({"started": true, "card_id": card_id, "member_id": member.id}),
            Err(message) => json!({"error": "run_start_failed", "message": message}),
        }
    }
    /// `TaskAction::RunStatus {card_id}`.
    async fn card_run_status(&self, card_id: i64) -> Value {
        let Some(ref task_runs) = self.task_runs else {
            return json!({"error": "task_runs_unavailable", "message": "run manager not attached"});
        };
        serde_json::to_value(task_runs.card_status(card_id).await)
            .unwrap_or_else(|_| json!({"running": false}))
    }

    /// `TaskAction::CancelRun {card_id}`.
    async fn cancel_card_run(&self, card_id: i64) -> Value {
        let Some(ref task_runs) = self.task_runs else {
            return json!({"error": "task_runs_unavailable", "message": "run manager not attached"});
        };
        json!({"cancelled": task_runs.cancel_card(card_id).await})
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

/// A patch's string for an optional field: `""` clears it, anything else sets it.
fn clearable(value: &str) -> Option<String> {
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

/// The fields of a `TaskAction::Create` (and of the card a
/// `TaskAction::QuickAdd` line becomes).
pub(super) struct CreateTask {
    pub(super) title: String,
    pub(super) scope: Option<String>,
    pub(super) session_id: Option<String>,
    pub(super) parent_id: Option<i64>,
    pub(super) description: Option<String>,
    pub(super) priority: Option<i64>,
    pub(super) labels: Option<Vec<String>>,
    pub(super) tools: Option<Vec<String>>,
    pub(super) due_at: Option<String>,
    pub(super) deadline_at: Option<String>,
    pub(super) recurrence: Option<String>,
    pub(super) depends_on: Option<Vec<i64>>,
    /// Boxed to match `TaskAction::Create`, whose field it is moved from; see
    /// the note there. Unboxed again by `task_create` before it reaches
    /// `NewTask`, which stores the canonicalized value inline.
    pub(super) acceptance: Option<Box<Value>>,
    pub(super) project: Option<String>,
    pub(super) assignee: Option<String>,
}
