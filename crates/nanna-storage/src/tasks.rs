//! Agent-grade task repository (P15).
//!
//! The task store is the control structure a long-horizon run is driven from
//! (P14): hierarchy is decomposition, dependencies make `next()` derivable
//! instead of guessed, `blocked` is derived state the model can never write,
//! and `done` on a parent is a harness-enforced invariant, not a prompt
//! convention.
//!
//! No SQL triggers or transactions are used (the migration runner splits on
//! `;` and the codebase serializes all access through one connection mutex);
//! multi-statement invariants hold because every method acquires the single
//! connection lock for the duration of its writes.

use crate::{
    NewTask, StorageError, Task, TaskActivityEntry, TaskEvent, TaskEventKind, TaskEventSink,
    TaskNote, TaskNoteKind, TaskPatch, VerdictTally, task_filter,
};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::Arc;
use tokio::sync::Mutex;
use turso::Connection;

/// Maximum title length in bytes.
///
/// Bound justification: the title is re-injected into the O(1) per-step
/// prompt on every harness iteration; 500 bytes (~125 tokens) keeps a single
/// title from ever dominating a small model's window.
pub const TASK_TITLE_MAX_BYTES: usize = 500;

/// Maximum note length in bytes.
///
/// Bound justification: notes carry sub-agent findings and are read back as a
/// bounded tail into prompts; 16 KiB (~4k tokens) is the most a note could
/// ever usefully contribute to a context injection.
pub const TASK_NOTE_MAX_BYTES: usize = 16 * 1024;

/// Maximum direct dependencies per task.
///
/// Bound justification: `next()` and the cycle check walk dependency edges;
/// an item depending on more than 100 siblings is degenerate decomposition
/// (that is what a parent is for), and the bound caps graph-walk work.
pub const TASK_DEPS_MAX: usize = 100;

/// Maximum hierarchy depth.
///
/// Bound justification: parent-chain walks (cycle check, auto-complete)
/// recurse once per level; 32 levels is far beyond meaningful decomposition
/// and keeps every walk trivially bounded.
pub const TASK_DEPTH_MAX: usize = 32;

/// Most verdicts [`TaskRepository::verdict_rollup`] will scan.
///
/// Bound justification: a verdict row is ~200 B decoded, so the cap holds a
/// scan under ~20 MiB and a few hundred ms on this store, while being two
/// orders of magnitude past the few hundred recent outcomes the router needs
/// to tell members apart.
pub const VERDICT_WINDOW_MAX: usize = 100_000;

/// Maximum tasks per scope.
///
/// Bound justification: `next()`, filters, and cycle checks load a scope into
/// memory (~1 KiB/task ⇒ ≤10 MiB); the bound also brakes a runaway agent
/// stuck in a task-creation loop long before the store degrades.
pub const TASKS_PER_SCOPE_MAX: usize = 10_000;

const TASK_COLUMNS: &str = "id, parent_id, scope, scope_id, project, title, description, status, \
     priority, labels, tool_scope, due_at, recurrence, depends_on, acceptance, assignee, \
     sort_order, created_at, updated_at, completed_at, deadline_at, due_announced_at, \
     overdue_announced_at";

/// Outcome of completing a task.
#[derive(Debug, Clone)]
pub struct CompleteOutcome {
    pub task: Task,
    /// Ancestor ids auto-completed because all their children finished.
    pub auto_completed: Vec<i64>,
    /// True when the task was already done (idempotent no-op).
    pub already_done: bool,
}

/// Task repository over the shared Turso connection.
pub struct TaskRepository {
    conn: Arc<Mutex<Connection>>,
    /// Where lifecycle events go; `None` when nobody is listening, which is
    /// every test and every caller that predates the daemon's event bus.
    events: Option<Arc<dyn TaskEventSink>>,
}

impl TaskRepository {
    #[must_use]
    pub const fn new(conn: Arc<Mutex<Connection>>) -> Self {
        Self { conn, events: None }
    }

    /// Attach the lifecycle event sink (see [`crate::Storage::set_task_events`]).
    #[must_use]
    pub fn with_events(mut self, events: Option<Arc<dyn TaskEventSink>>) -> Self {
        self.events = events;
        self
    }

    /// Emit `Blocked`/`Unblocked` for every task whose derived `blocked` flag
    /// differs between two snapshots of the same scope.
    ///
    /// `blocked` is derived on read and never stored, so a transition is a
    /// *difference between two computations*, not a column that changed. Taking
    /// the snapshots around the whole mutation — rather than reasoning about
    /// which dependents a particular write should have touched — is what makes
    /// the cascades free: `complete` auto-closes ancestors and `update` cancels
    /// a subtree, and both are already reflected in the "after" snapshot.
    ///
    /// Only tasks present in **both** snapshots transition: a task that was
    /// deleted did not become unblocked, it stopped existing.
    fn emit_block_transitions(&self, before: &[Task], after: &[Task], actor: Option<&str>) {
        if self.events.is_none() {
            return;
        }
        let was_blocked = blocked_ids(before);
        let now_blocked = blocked_ids(after);
        let existed: HashSet<i64> = before.iter().map(|t| t.id).collect();
        for task in after {
            if !existed.contains(&task.id) {
                continue;
            }
            let before_blocked = was_blocked.contains(&task.id);
            let after_blocked = now_blocked.contains(&task.id);
            if before_blocked == after_blocked {
                continue;
            }
            let kind = if after_blocked {
                TaskEventKind::Blocked
            } else {
                TaskEventKind::Unblocked
            };
            self.emit(
                kind,
                task,
                actor,
                serde_json::json!({ "depends_on": task.depends_on }),
            );
        }
    }

    /// Publish one lifecycle event, if anyone is listening.
    ///
    /// # Invariant
    ///
    /// **Never called while the connection guard is held.** The sink is
    /// foreign code, and running it under the mutex that serializes every
    /// write in this process would let one consumer stall the whole store.
    /// Every call site sits after its writes have completed and the guard has
    /// been dropped — which also means an event is published only for a
    /// mutation that actually reached the database.
    fn emit(
        &self,
        kind: TaskEventKind,
        task: &Task,
        actor: Option<&str>,
        detail: serde_json::Value,
    ) {
        let Some(sink) = self.events.as_ref() else {
            return;
        };
        sink.publish(TaskEvent {
            kind,
            task_id: task.id,
            scope: task.scope.clone(),
            scope_id: task.scope_id.clone(),
            actor: actor.map(str::to_string),
            detail,
        });
    }

    /// Create a task. Validates scope, title, priority, parent, and
    /// dependencies (existence + bounds); a new task cannot introduce a
    /// dependency cycle because nothing depends on it yet.
    ///
    /// # Errors
    /// Returns [`StorageError::Invalid`] if the scope/`scope_id` pairing is
    /// invalid, the title is empty or longer than [`TASK_TITLE_MAX_BYTES`],
    /// the priority is outside `1..=4`, there are more than [`TASK_DEPS_MAX`]
    /// dependencies, the acceptance check is malformed, the scope already
    /// holds [`TASKS_PER_SCOPE_MAX`] tasks, the parent is missing from the
    /// scope or closed, the parent is at the depth limit, or a dependency is
    /// missing from the scope or is an ancestor. Returns
    /// [`StorageError::Database`] if a read or write fails, or
    /// [`StorageError::NotFound`] if the inserted row cannot be read back. A
    /// failure writing the `created` activity entry leaves the task created.
    pub async fn create(&self, mut new: NewTask) -> Result<Task, StorageError> {
        validate_new_task_fields(&new)?;
        // Canonicalize BEFORE storing: the row must hold the object shape the
        // harness reads back, whatever dialect the caller handed us.
        if let Some(acceptance) = &new.acceptance {
            new.acceptance = Some(admit_acceptance(acceptance)?);
        }

        let scope_tasks = self.load_scope(&new.scope, new.scope_id.as_deref()).await?;
        if scope_tasks.len() >= TASKS_PER_SCOPE_MAX {
            return Err(StorageError::Invalid(format!(
                "scope already holds {TASKS_PER_SCOPE_MAX} tasks"
            )));
        }
        let by_id: HashMap<i64, &Task> = scope_tasks.iter().map(|t| (t.id, t)).collect();

        if let Some(parent_id) = new.parent_id {
            let parent = by_id.get(&parent_id).ok_or_else(|| {
                StorageError::Invalid(format!("parent task #{parent_id} not found in scope"))
            })?;
            if is_closed_status(&parent.status) {
                return Err(StorageError::Invalid(format!(
                    "cannot add a child to {} task #{parent_id}",
                    parent.status
                )));
            }
            let depth = parent_depth(&by_id, parent_id)?;
            if depth + 1 >= TASK_DEPTH_MAX {
                return Err(StorageError::Invalid(format!(
                    "hierarchy depth limit {TASK_DEPTH_MAX} reached"
                )));
            }
        }
        for dep in &new.depends_on {
            if !by_id.contains_key(dep) {
                return Err(StorageError::Invalid(format!(
                    "dependency task #{dep} not found in scope"
                )));
            }
        }
        // A pure depends_on cycle is impossible for a new task (nothing
        // depends on it yet) — but the parent-completion invariant makes an
        // ancestor an implicit dependent, so depending on an ancestor
        // (directly or transitively) would wedge both sides forever.
        if new.parent_id.is_some() && !new.depends_on.is_empty() {
            let ancestors = ancestor_chain(&by_id, new.parent_id);
            check_ancestor_dependency(&by_id, &ancestors, &new.depends_on)?;
        }

        let labels_json = serde_json::to_string(&new.labels)?;
        let tool_scope_json = serde_json::to_string(&new.tool_scope)?;
        let depends_json = serde_json::to_string(&new.depends_on)?;
        let acceptance_json = new
            .acceptance
            .as_ref()
            .map(std::string::ToString::to_string);

        let conn = self.conn.lock().await;
        if let Err(err) = ensure_member_exists(&conn, new.assignee.as_deref(), "assignee").await {
            drop(conn);
            return Err(err);
        }
        conn.execute(
            "INSERT INTO tasks (parent_id, scope, scope_id, project, title, description, status, \
             priority, labels, tool_scope, due_at, recurrence, depends_on, acceptance, assignee, \
             sort_order, deadline_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'pending', ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)",
            turso::params![
                new.parent_id,
                new.scope.as_str(),
                new.scope_id.as_deref(),
                new.project.as_deref(),
                new.title.as_str(),
                new.description.as_deref(),
                new.priority,
                labels_json.as_str(),
                tool_scope_json.as_str(),
                new.due_at.as_deref(),
                new.recurrence.as_deref(),
                depends_json.as_str(),
                acceptance_json.as_deref(),
                new.assignee.as_deref(),
                new.sort_order,
                new.deadline_at.as_deref(),
            ],
        )
        .await?;

        let mut rows = conn
            .query(
                &format!("SELECT {TASK_COLUMNS} FROM tasks ORDER BY id DESC LIMIT 1"),
                (),
            )
            .await?;
        let task = match rows.next().await? {
            Some(row) => decode_task_row(&row)?,
            None => return Err(StorageError::NotFound("task just created".to_string())),
        };
        // Drop the open cursor before any further statement: an unfinished
        // Rows on the shared turso connection silently swallows later writes.
        drop(rows);
        drop(conn);

        self.log_activity(task.id, new.assignee.as_deref(), "created", None)
            .await?;
        self.emit(TaskEventKind::Created, &task, new.assignee.as_deref(), created_detail(&task));
        Ok(task)
    }

    /// Get one task with its derived `blocked` flag.
    ///
    /// # Errors
    /// Returns [`StorageError::NotFound`] if no task has `id`, or
    /// [`StorageError::Database`] if a read fails.
    pub async fn get(&self, id: i64) -> Result<Task, StorageError> {
        let mut task = self.get_raw(id).await?;
        if !task.depends_on.is_empty() {
            let scope_tasks = self
                .load_scope(&task.scope, task.scope_id.as_deref())
                .await?;
            let by_id: HashMap<i64, &Task> = scope_tasks.iter().map(|t| (t.id, t)).collect();
            task.blocked = is_blocked(&task, &by_id);
        }
        Ok(task)
    }

    /// List tasks in a scope with derived `blocked` flags, ordered by
    /// hierarchy-friendly (`sort_order`, `id`).
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the scope query fails or a row
    /// does not decode.
    pub async fn list(
        &self,
        scope: &str,
        scope_id: Option<&str>,
        include_closed: bool,
    ) -> Result<Vec<Task>, StorageError> {
        let mut tasks = self.load_scope(&normalize_scope(scope), scope_id).await?;
        let statuses: HashMap<i64, String> =
            tasks.iter().map(|t| (t.id, t.status.clone())).collect();
        for task in &mut tasks {
            task.blocked = task.depends_on.iter().any(|dep| {
                statuses
                    .get(dep)
                    .is_some_and(|s| !is_closed_status(s))
            });
        }
        if !include_closed {
            tasks.retain(|t| !is_closed_status(&t.status));
        }
        tasks.sort_by_key(|t| (t.sort_order, t.id));
        Ok(tasks)
    }

    /// The single most valuable call: return the one actionable item —
    /// open, unblocked, a leaf (no open children) — ordered by
    /// `in_progress` first (resume what you started), then priority, due
    /// date, explicit order, and id.
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the scope query fails or a row
    /// does not decode.
    pub async fn next(
        &self,
        scope: &str,
        scope_id: Option<&str>,
    ) -> Result<Option<Task>, StorageError> {
        self.next_admitted(scope, scope_id, |_| true).await
    }

    /// [`next`](Self::next), restricted to the tasks `admit` accepts.
    ///
    /// The ordering and the open-children rule are computed over the WHOLE
    /// open scope exactly as `next` does — an inadmissible task still blocks
    /// its parent — and only the final choice is filtered. A chat turn uses
    /// this to run the work it planned without also running every item an
    /// earlier turn left open (see the daemon's `TurnAdmission`).
    ///
    /// # Errors
    /// Returns an error if the scope's tasks cannot be listed.
    pub async fn next_admitted(
        &self,
        scope: &str,
        scope_id: Option<&str>,
        admit: impl Fn(&Task) -> bool,
    ) -> Result<Option<Task>, StorageError> {
        let tasks = self.list(scope, scope_id, false).await?;
        let mut open_children: HashSet<i64> = HashSet::new();
        for task in &tasks {
            if let Some(parent) = task.parent_id {
                open_children.insert(parent);
            }
        }
        // Ladder depth per task, from the same in-memory scope list. Depth
        // 0 is a root (a seeded feature), 1 its direct subtask, 2+ the
        // model's recursive scaffolding. Cycle-safe by construction: the walk
        // is bounded by the map size.
        let parent_of: HashMap<i64, i64> =
            tasks.iter().filter_map(|t| t.parent_id.map(|p| (t.id, p))).collect();
        let depth_of = |id: i64| -> usize {
            let mut depth = 0usize;
            let mut cursor = id;
            while let Some(&p) = parent_of.get(&cursor) {
                depth += 1;
                cursor = p;
                if depth >= tasks.len() {
                    break;
                }
            }
            depth
        };
        let mut candidates: Vec<&Task> = tasks
            .iter()
            .filter(|t| !t.blocked && !open_children.contains(&t.id) && admit(t))
            .collect();
        candidates.sort_by(|a, b| {
            let a_progress = i32::from(a.status != "in_progress");
            let b_progress = i32::from(b.status != "in_progress");
            // Depth ranks after resume-what-you-started and BEFORE priority:
            // the seeded ladder and its direct subtasks always outrank the
            // model's recursive scaffolding. This is the structural half of
            // decomposition damping — the advisory notes on `tasks.add` fired
            // 69 times in one endurance run while the model split anyway, so
            // the schedule, not the model, has to hold the line. Depth 0 and 1
            // rank EQUALLY (a feature and its direct subtasks are both real
            // work); beyond that the rank is GRADED by actual ladder depth,
            // not a binary flag. The binary form went vacuous on a strict
            // dependency chain (observed 2026-08-08, ministral: with every
            // shallow leaf blocked, all schedulable work was depth 2-6 and
            // tied — the model recursed to depth 6 with nothing pulling it
            // back). Grading derives the rank from the structure itself:
            // each level of scaffolding is one planning step farther from
            // executable work, so within an all-deep candidate set the
            // scheduler still pulls toward the shallowest.
            let a_deep = depth_of(a.id).saturating_sub(1);
            let b_deep = depth_of(b.id).saturating_sub(1);
            a_progress
                .cmp(&b_progress)
                .then(a_deep.cmp(&b_deep))
                .then(a.priority.cmp(&b.priority))
                .then_with(|| match (&a.due_at, &b.due_at) {
                    (Some(x), Some(y)) => x.cmp(y),
                    (Some(_), None) => std::cmp::Ordering::Less,
                    (None, Some(_)) => std::cmp::Ordering::Greater,
                    (None, None) => std::cmp::Ordering::Equal,
                })
                .then(a.sort_order.cmp(&b.sort_order))
                .then(a.id.cmp(&b.id))
        });
        Ok(candidates.first().map(|t| (*t).clone()))
    }

    /// Apply a partial update. `status` accepts only
    /// `pending | in_progress | cancelled`: `done` must go through
    /// [`complete`](Self::complete) and `blocked` is derived, never written.
    ///
    /// # Errors
    /// Returns [`StorageError::NotFound`] if task `id` does not exist, and
    /// [`StorageError::Invalid`] if the patch is rejected: status `done`,
    /// `blocked`, or unknown; an invalid title, priority, or acceptance check;
    /// more than [`TASK_DEPS_MAX`] dependencies; or a dependency or parent
    /// change that names a task outside the scope, forms a cycle, moves the
    /// task under a closed parent, exceeds [`TASK_DEPTH_MAX`], or makes a task
    /// depend on its own ancestor. Returns [`StorageError::Database`] if a
    /// read or write fails. The task row is written before the activity entry
    /// and any cancel cascade, so a failure there leaves the update applied.
    pub async fn update(
        &self,
        id: i64,
        patch: TaskPatch,
        actor: Option<&str>,
    ) -> Result<Task, StorageError> {
        let mut task = self.get_raw(id).await?;
        let needs_graph_check = patch.depends_on.is_some() || patch.parent_id.is_some();
        let parent_changed = patch.parent_id.is_some();
        let changed = apply_patch(&mut task, patch)?;
        validate_dates(task.due_at.as_deref(), task.deadline_at.as_deref())?;
        // Snapshot before the write when this update can change what blocks
        // what: a status transition opens or closes a dependency, and a
        // `depends_on` change moves the edges themselves. `task` is already
        // patched in memory, but the rows still hold the pre-update state.
        let blocking_before = if self.events.is_some()
            && (changed.contains(&"status") || changed.contains(&"depends_on"))
        {
            Some(
                self.load_scope(&task.scope, task.scope_id.as_deref())
                    .await?,
            )
        } else {
            None
        };

        {
            // Validate and write under ONE connection guard: the mutex is the
            // transaction, so a concurrent writer cannot slip a conflicting
            // graph change between the checks and the write.
            let conn = self.conn.lock().await;
            if changed.contains(&"assignee")
                && let Err(err) =
                    ensure_member_exists(&conn, task.assignee.as_deref(), "assignee").await
            {
                drop(conn);
                return Err(err);
            }
            if needs_graph_check {
                let scope_tasks =
                    load_scope_with(&conn, &task.scope, task.scope_id.as_deref()).await?;
                let mut by_id: HashMap<i64, Task> =
                    scope_tasks.into_iter().map(|t| (t.id, t)).collect();
                // Evaluate the graph as it would be after this update.
                by_id.insert(task.id, task.clone());
                let borrow: HashMap<i64, &Task> = by_id.iter().map(|(k, v)| (*k, v)).collect();
                check_graph_update(&borrow, &task, parent_changed)?;
            }
            write_task_with(&conn, &task).await?;
            // Held from the graph check through the write (see above).
            drop(conn);
        }
        if !changed.is_empty() {
            self.log_activity(
                task.id,
                actor,
                "updated",
                Some(serde_json::json!({ "fields": changed })),
            )
            .await?;
        }

        // Cancelling a branch closes its whole open subtree — children of a
        // cancelled plan must not surface from next() — EXCEPT children that
        // carry their OWN acceptance contract (one differing from their
        // parent's): those are independently verifiable work, not shards of
        // the dead branch's contract, and they re-parent to the cancelled
        // branch's own parent instead of dying with it. Observed 2026-08-08
        // (qwen endurance): the model correctly diagnosed the run's two root
        // causes and created repair tasks with their own checks — both were
        // cascade-cancelled when the parent's budget ran out, and the
        // cascade deleted the exact work that would have unstuck the run.
        // Children whose acceptance is absent or inherited (byte-equal to
        // the parent's) are scaffolding of the dead contract and still
        // cancel; their subtrees are walked, so a grandchild with its own
        // check survives the same way.
        if changed.contains(&"status") && task.status == "cancelled" {
            self.cascade_cancel(&task, actor).await?;
        }
        // Emitted from `changed`, so each event describes a real transition:
        // `apply_patch` compares both of these fields before recording them.
        if changed.contains(&"status") {
            self.emit(
                TaskEventKind::StatusChanged,
                &task,
                actor,
                serde_json::json!({ "status": task.status }),
            );
        }
        if changed.contains(&"assignee") {
            self.emit(
                TaskEventKind::Assigned,
                &task,
                actor,
                serde_json::json!({ "assignee": task.assignee }),
            );
        }
        // After the cascade, so a subtree cancelled above is already reflected
        // and its dependents' releases are part of this one diff.
        if let Some(before) = blocking_before {
            let after = self
                .load_scope(&task.scope, task.scope_id.as_deref())
                .await?;
            self.emit_block_transitions(&before, &after, actor);
        }
        self.get(id).await
    }

    /// Complete a task. The harness (or service layer) runs the acceptance
    /// check *before* calling this — completion here is the recorded verdict.
    ///
    /// Enforced invariants: a parent cannot complete while a child is open,
    /// and ancestors without acceptance checks auto-complete when their last
    /// open child finishes (a parent *with* an acceptance check must be
    /// completed explicitly so its own check runs).
    ///
    /// # Errors
    /// Returns [`StorageError::NotFound`] if no task has `id`,
    /// [`StorageError::Invalid`] if the task still has open children, or
    /// [`StorageError::Database`] if a read or write fails. Once the task is
    /// marked done it stays done: a later failure (activity log, ancestor
    /// auto-completion) returns the error without undoing it.
    pub async fn complete(
        &self,
        id: i64,
        actor: Option<&str>,
        detail: Option<serde_json::Value>,
    ) -> Result<CompleteOutcome, StorageError> {
        let task = self.get_raw(id).await?;
        if task.status == "done" {
            return Ok(CompleteOutcome {
                task,
                auto_completed: Vec::new(),
                already_done: true,
            });
        }

        // Check-then-mark under ONE connection guard so a child created
        // concurrently cannot slip in between the open-children check and the
        // completion write (the mutex is the transaction).
        let scope_tasks;
        {
            let conn = self.conn.lock().await;
            scope_tasks = load_scope_with(&conn, &task.scope, task.scope_id.as_deref()).await?;
            let open_children: Vec<i64> = scope_tasks
                .iter()
                .filter(|t| {
                    t.parent_id == Some(id) && !is_closed_status(&t.status)
                })
                .map(|t| t.id)
                .collect();
            if !open_children.is_empty() {
                return Err(StorageError::Invalid(format!(
                    "task #{id} has open children {open_children:?}; complete them first"
                )));
            }
            mark_done_with(&conn, id).await?;
        }
        self.log_activity(id, actor, "completed", detail).await?;

        // Auto-complete ancestors whose children are now all closed.
        let by_id: HashMap<i64, &Task> = scope_tasks.iter().map(|t| (t.id, t)).collect();
        let mut closed: HashSet<i64> = scope_tasks
            .iter()
            .filter(|t| is_closed_status(&t.status))
            .map(|t| t.id)
            .collect();
        closed.insert(id);

        let mut auto_completed = Vec::new();
        let mut current = task.parent_id;
        // Bounded by the hierarchy depth limit enforced at write time.
        for _ in 0..TASK_DEPTH_MAX {
            let Some(parent_id) = current else { break };
            let Some(parent) = by_id.get(&parent_id) else {
                break;
            };
            if closed.contains(&parent_id) || parent.acceptance.is_some() {
                break;
            }
            let all_children_closed = scope_tasks
                .iter()
                .filter(|t| t.parent_id == Some(parent_id))
                .all(|t| closed.contains(&t.id));
            if !all_children_closed {
                break;
            }
            self.mark_done(parent_id).await?;
            self.log_activity(
                parent_id,
                actor,
                "auto_completed",
                Some(serde_json::json!({ "trigger": id })),
            )
            .await?;
            closed.insert(parent_id);
            auto_completed.push(parent_id);
            current = parent.parent_id;
        }

        let task = self.get_raw(id).await?;
        debug_assert!(task.status == "done", "complete() must leave status=done");
        self.emit(
            TaskEventKind::Verdict,
            &task,
            actor,
            serde_json::json!({ "auto": false, "auto_completed": auto_completed }),
        );
        // Ancestors closed by this completion get their own verdict. They are
        // the reason these events are emitted from the repository at all: no
        // caller ever named them, so a consumer listening one layer up would
        // watch a parent silently become done.
        for parent_id in &auto_completed {
            let Some(parent) = by_id.get(parent_id) else {
                continue;
            };
            self.emit(
                TaskEventKind::Verdict,
                parent,
                actor,
                serde_json::json!({ "auto": true, "trigger": id }),
            );
        }
        // Closing a task releases whatever depended on it. The reload also
        // picks up the ancestors auto-closed above, so one diff covers both.
        if self.events.is_some() {
            let after = self
                .load_scope(&task.scope, task.scope_id.as_deref())
                .await?;
            self.emit_block_transitions(&scope_tasks, &after, actor);
        }
        Ok(CompleteOutcome {
            task,
            auto_completed,
            already_done: false,
        })
    }

    /// Reopen a closed task (recurrence firing, or replan of a wrong verdict).
    ///
    /// # Errors
    /// Returns [`StorageError::NotFound`] if no task has `id`, or
    /// [`StorageError::Database`] if a read or write fails; a failed activity
    /// entry leaves the task reopened.
    pub async fn reopen(&self, id: i64, actor: Option<&str>) -> Result<Task, StorageError> {
        let task = self.get_raw(id).await?;
        if !is_closed_status(&task.status) {
            return Ok(task);
        }
        // Reopening runs the transition the other way: a dependency that was
        // closed is open again, so its dependents become blocked.
        let blocking_before = if self.events.is_some() {
            Some(
                self.load_scope(&task.scope, task.scope_id.as_deref())
                    .await?,
            )
        } else {
            None
        };
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE tasks SET status = 'pending', completed_at = NULL, \
             due_announced_at = NULL, overdue_announced_at = NULL, \
             updated_at = datetime('now') WHERE id = ?1",
            turso::params![id],
        )
        .await?;
        drop(conn);
        self.log_activity(id, actor, "reopened", None).await?;
        let reopened = self.get_raw(id).await?;
        self.emit(
            TaskEventKind::StatusChanged,
            &reopened,
            actor,
            serde_json::json!({ "status": reopened.status, "reopened": true }),
        );
        if let Some(before) = blocking_before {
            let after = self
                .load_scope(&task.scope, task.scope_id.as_deref())
                .await?;
            self.emit_block_transitions(&before, &after, actor);
        }
        Ok(reopened)
    }

    /// Append a working note (the durable scratchpad sub-agents leave
    /// findings in).
    ///
    /// # Errors
    /// Returns [`StorageError::Invalid`] if the trimmed content is empty or
    /// longer than [`TASK_NOTE_MAX_BYTES`], [`StorageError::NotFound`] if no
    /// task has `task_id` or the new note cannot be read back, or
    /// [`StorageError::Database`] if a read or write fails.
    pub async fn add_note(
        &self,
        task_id: i64,
        author: Option<&str>,
        content: &str,
    ) -> Result<TaskNote, StorageError> {
        self.post(task_id, author, None, TaskNoteKind::Comment, content)
            .await
    }

    /// Append a thread post (P25 decision 2): the same append-only write as
    /// [`Self::add_note`], but naming the board member who posted and what the
    /// post is doing.
    ///
    /// `author_member_id` is checked against `members` when present — the same
    /// enforcement `tasks.assignee` gets, for the same reason. `author` stays
    /// as the pre-members actor string so a post from the harness still says
    /// where it came from.
    ///
    /// There is no counterpart that edits a post: the thread is the permanent
    /// record, so append is the only write shape.
    ///
    /// # Errors
    /// Returns [`StorageError::Invalid`] if the trimmed content is empty or
    /// longer than [`TASK_NOTE_MAX_BYTES`] or if `author_member_id` is not a
    /// board member, [`StorageError::NotFound`] if no task has `task_id` or
    /// the new post cannot be read back, or [`StorageError::Database`] if a
    /// read or write fails.
    pub async fn post(
        &self,
        task_id: i64,
        author: Option<&str>,
        author_member_id: Option<&str>,
        kind: TaskNoteKind,
        content: &str,
    ) -> Result<TaskNote, StorageError> {
        let trimmed = content.trim();
        if trimmed.is_empty() {
            return Err(StorageError::Invalid("note content is empty".to_string()));
        }
        if trimmed.len() > TASK_NOTE_MAX_BYTES {
            return Err(StorageError::Invalid(format!(
                "note exceeds {TASK_NOTE_MAX_BYTES} bytes (got {})",
                trimmed.len()
            )));
        }
        // Ensure the task exists before writing (FKs are unenforced). The row
        // is kept rather than discarded: the lifecycle event below names the
        // card's scope, and re-reading it after the write would be a second
        // round-trip for data already in hand.
        let task = self.get_raw(task_id).await?;

        let conn = self.conn.lock().await;
        if let Err(err) = ensure_member_exists(&conn, author_member_id, "note author").await {
            drop(conn);
            return Err(err);
        }
        let kind_token = kind.as_str();
        conn.execute(
            "INSERT INTO task_notes (task_id, author, author_member_id, kind, content) \
             VALUES (?1, ?2, ?3, ?4, ?5)",
            turso::params![task_id, author, author_member_id, kind_token, trimmed],
        )
        .await?;
        let mut rows = conn
            .query(
                "SELECT id, task_id, author, content, created_at, author_member_id, kind \
                 FROM task_notes ORDER BY id DESC LIMIT 1",
                (),
            )
            .await?;
        let note = rows.next().await?.map_or_else(
            || Err(StorageError::NotFound("note just created".to_string())),
            |row| decode_task_note(&row),
        );
        // Held from the insert through the read-back: "newest row" is only
        // this note while no other writer can interleave, and the cursor must
        // be gone before the guard is.
        drop(rows);
        drop(conn);
        // Only a post that was actually written is announced — the thread is
        // the permanent record, so an event for a failed append would describe
        // a post nobody can ever read.
        if let Ok(ref note) = note {
            self.emit(
                TaskEventKind::Posted,
                &task,
                author,
                serde_json::json!({
                    "note_id": note.id,
                    "kind": kind_token,
                    "author_member_id": author_member_id,
                }),
            );
        }
        note
    }

    /// Last `limit` notes for a task, oldest first.
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the query fails or a row does not
    /// decode.
    pub async fn notes(&self, task_id: i64, limit: i64) -> Result<Vec<TaskNote>, StorageError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT id, task_id, author, content, created_at, author_member_id, kind \
                 FROM task_notes WHERE task_id = ?1 ORDER BY id DESC LIMIT ?2",
                turso::params![task_id, limit],
            )
            .await?;
        let mut notes = Vec::new();
        while let Some(row) = rows.next().await? {
            notes.push(decode_task_note(&row)?);
        }
        // Held until the cursor is gone: an open `Rows` on the shared
        // connection swallows later writes.
        drop(rows);
        drop(conn);
        notes.reverse();
        Ok(notes)
    }

    /// The `limit` most recently closed board cards (done or cancelled, any
    /// scope but `session`), newest first — the dream fold's candidates.
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the query fails or a row does not
    /// decode.
    ///
    /// # Panics
    /// Panics if `limit` is 0.
    pub async fn closed_board_cards(&self, limit: usize) -> Result<Vec<Task>, StorageError> {
        assert!(
            limit > 0,
            "a scan for closed cards is bounded and non-empty"
        );
        let limit = i64::try_from(limit).unwrap_or(i64::MAX);
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                &format!(
                    "SELECT {TASK_COLUMNS} FROM tasks \
                     WHERE status IN ('done', 'cancelled') AND scope != 'session' \
                     ORDER BY COALESCE(completed_at, updated_at) DESC, id DESC LIMIT ?1"
                ),
                turso::params![limit],
            )
            .await?;
        let mut cards = Vec::new();
        while let Some(row) = rows.next().await? {
            cards.push(decode_task_row(&row)?);
        }
        drop(rows);
        drop(conn);
        debug_assert!(cards.iter().all(|t| is_closed_status(&t.status)));
        Ok(cards)
    }

    /// One thread post by its id.
    ///
    /// Task events name a post by id rather than carrying it (the bus is not a
    /// replication channel), so a consumer that needs the text reads it here.
    ///
    /// # Errors
    /// Returns [`StorageError::NotFound`] if no post has `note_id`, or
    /// [`StorageError::Database`] if the query fails or the row does not decode.
    pub async fn note(&self, note_id: i64) -> Result<TaskNote, StorageError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT id, task_id, author, content, created_at, author_member_id, kind \
                 FROM task_notes WHERE id = ?1",
                turso::params![note_id],
            )
            .await?;
        let note = match rows.next().await? {
            Some(row) => Some(decode_task_note(&row)?),
            None => None,
        };
        drop(rows);
        drop(conn);
        let note = note.ok_or_else(|| StorageError::NotFound(format!("task note #{note_id}")))?;
        debug_assert_eq!(note.id, note_id, "looked up by primary key");
        Ok(note)
    }

    /// Last `limit` activity entries for a task, oldest first.
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the query fails or a row does not
    /// decode. Unparseable detail JSON loads as `None`.
    pub async fn activity(
        &self,
        task_id: i64,
        limit: i64,
    ) -> Result<Vec<TaskActivityEntry>, StorageError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT id, task_id, actor, action, detail, created_at, assignee \
                 FROM task_activity WHERE task_id = ?1 ORDER BY id DESC LIMIT ?2",
                turso::params![task_id, limit],
            )
            .await?;
        let mut entries = Vec::new();
        while let Some(row) = rows.next().await? {
            let detail_str: Option<String> = row.get(4)?;
            entries.push(TaskActivityEntry {
                id: row.get(0)?,
                task_id: row.get(1)?,
                actor: row.get(2)?,
                action: row.get(3)?,
                detail: detail_str.and_then(|s| serde_json::from_str(&s).ok()),
                created_at: row.get(5)?,
                assignee: row.get(6)?,
            });
        }
        // Held until the cursor is gone: an open `Rows` on the shared
        // connection swallows later writes.
        drop(rows);
        drop(conn);
        entries.reverse();
        Ok(entries)
    }

    /// Record an activity entry (public so the harness can log run events
    /// like acceptance verdicts against the task).
    ///
    /// The task is not checked for existence.
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the insert fails.
    pub async fn log_activity(
        &self,
        task_id: i64,
        actor: Option<&str>,
        action: &str,
        detail: Option<serde_json::Value>,
    ) -> Result<(), StorageError> {
        let detail_json = detail.as_ref().map(std::string::ToString::to_string);
        let conn = self.conn.lock().await;
        // The assignee is read inside the INSERT, so the row names whoever held
        // the card at the instant it was written — no caller supplies it and
        // no concurrent re-assignment can land between a read and the write.
        conn.execute(
            "INSERT INTO task_activity (task_id, actor, action, detail, assignee) \
             VALUES (?1, ?2, ?3, ?4, (SELECT assignee FROM tasks WHERE id = ?1))",
            turso::params![task_id, actor, action, detail_json.as_deref()],
        )
        .await?;
        drop(conn);
        Ok(())
    }

    /// Each member's acceptance verdicts over the last `window` of them, per
    /// label and overall — the router's evidence of what each member actually
    /// passes (P25 decisions 4 and 15).
    ///
    /// A verdict is an `acceptance_checked` row, the one both the `tasks.complete`
    /// tool and the harness write. It counts toward the member stamped on that
    /// row — assigned when it was judged, not now (migration 021). Rows with no
    /// stamped member (pre-021, or an unassigned card) and verdicts the check
    /// itself marked `unknown` are left out: neither says anything about a
    /// member. Labels are the card's current ones.
    ///
    /// `window` bounds the scan to the most recent verdicts: the history is
    /// unbounded, and recent outcomes are the ones that predict.
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the query fails or a row does not
    /// decode.
    ///
    /// # Panics
    /// Panics if `window` is 0 or above [`VERDICT_WINDOW_MAX`].
    pub async fn verdict_rollup(&self, window: usize) -> Result<Vec<VerdictTally>, StorageError> {
        assert!(
            window > 0 && window <= VERDICT_WINDOW_MAX,
            "verdict window must be in 1..={VERDICT_WINDOW_MAX}, got {window}"
        );
        let window_i64 = i64::try_from(window).unwrap_or(i64::MAX);
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT a.assignee, t.labels, a.detail FROM task_activity a \
                 JOIN tasks t ON t.id = a.task_id \
                 WHERE a.action = 'acceptance_checked' AND a.assignee IS NOT NULL \
                 ORDER BY a.id DESC LIMIT ?1",
                turso::params![window_i64],
            )
            .await?;
        let mut tallies: BTreeMap<(String, Option<String>), (u64, u64)> = BTreeMap::new();
        let mut scanned = 0usize;
        while let Some(row) = rows.next().await? {
            scanned += 1;
            let member: String = row.get(0)?;
            let labels: String = row.get(1)?;
            let detail: Option<String> = row.get(2)?;
            let Some(passed) = verdict_outcome(detail.as_deref()) else {
                continue;
            };
            let labels: Vec<String> = serde_json::from_str(&labels)?;
            let keys = std::iter::once(None).chain(labels.into_iter().map(Some));
            for label in keys {
                let tally = tallies.entry((member.clone(), label)).or_default();
                if passed {
                    tally.0 += 1;
                } else {
                    tally.1 += 1;
                }
            }
        }
        drop(rows);
        drop(conn);
        debug_assert!(scanned <= window, "the scan is bounded by the window");
        Ok(tallies
            .into_iter()
            .map(|((member_id, label), (passed, failed))| VerdictTally {
                member_id,
                label,
                passed,
                failed,
            })
            .collect())
    }

    /// Query tasks in a scope with the filter language
    /// (see [`task_filter`]). Evaluated in memory; `blocked` is derived
    /// before evaluation so `blocked` atoms see real state.
    ///
    /// # Errors
    /// Returns [`StorageError::Invalid`] if `filter` does not parse (the
    /// message carries the parser's reason), or [`StorageError::Database`] if
    /// the scope query fails or a row does not decode.
    pub async fn query(
        &self,
        scope: &str,
        scope_id: Option<&str>,
        filter: &str,
    ) -> Result<Vec<Task>, StorageError> {
        let expr = task_filter::parse(filter)
            .map_err(|e| StorageError::Invalid(format!("filter: {e}")))?;
        let tasks = self.list(scope, scope_id, true).await?;
        let today = chrono::Utc::now().format("%Y-%m-%d").to_string();
        Ok(tasks
            .into_iter()
            .filter(|t| expr.matches(t, &today))
            .collect())
    }

    /// Delete a task and its whole subtree (notes + activity included), and
    /// strip dangling references from other tasks' `depends_on`.
    ///
    /// # Errors
    /// Returns [`StorageError::NotFound`] if no task has `id`, or
    /// [`StorageError::Database`] if a read or write fails. Nothing is rolled
    /// back: rows deleted before a failure stay deleted, and a failure while
    /// stripping references leaves some `depends_on` lists naming deleted ids.
    pub async fn delete(&self, id: i64, actor: Option<&str>) -> Result<u64, StorageError> {
        let task = self.get_raw(id).await?;
        let scope_tasks = self
            .load_scope(&task.scope, task.scope_id.as_deref())
            .await?;

        // Collect the subtree, bounded by scope size.
        let mut doomed: Vec<i64> = vec![id];
        let mut queue: VecDeque<i64> = VecDeque::from([id]);
        while let Some(current) = queue.pop_front() {
            for t in &scope_tasks {
                if t.parent_id == Some(current) && !doomed.contains(&t.id) {
                    doomed.push(t.id);
                    queue.push_back(t.id);
                }
            }
        }

        let conn = self.conn.lock().await;
        for task_id in &doomed {
            conn.execute(
                "DELETE FROM task_notes WHERE task_id = ?1",
                turso::params![*task_id],
            )
            .await?;
            conn.execute(
                "DELETE FROM task_activity WHERE task_id = ?1",
                turso::params![*task_id],
            )
            .await?;
            conn.execute("DELETE FROM tasks WHERE id = ?1", turso::params![*task_id])
                .await?;
        }
        drop(conn);

        // Strip dangling dependency references. This is itself an unblocking
        // mechanism — a dependent whose only dependency was deleted is free —
        // so the diff below is taken after the stripping, not before it.
        let doomed_set: HashSet<i64> = doomed.iter().copied().collect();
        for t in &scope_tasks {
            if doomed_set.contains(&t.id) {
                continue;
            }
            if t.depends_on.iter().any(|d| doomed_set.contains(d)) {
                let mut kept = t.clone();
                kept.depends_on.retain(|d| !doomed_set.contains(d));
                self.write_task(&kept).await?;
            }
        }

        if self.events.is_some() {
            let after = self
                .load_scope(&task.scope, task.scope_id.as_deref())
                .await?;
            self.emit_block_transitions(&scope_tasks, &after, actor);
        }

        let count = doomed.len() as u64;
        if let Some(actor) = actor {
            tracing::debug!("{actor} deleted task #{id} subtree ({count} tasks)");
        }
        Ok(count)
    }

    /// Clear closed tasks (or every task) in a scope. Returns deleted count.
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if a read or delete fails, or
    /// [`StorageError::NotFound`] if a target is deleted concurrently between
    /// its existence check and its delete. Subtrees cleared before a failure
    /// stay cleared.
    pub async fn clear(
        &self,
        scope: &str,
        scope_id: Option<&str>,
        closed_only: bool,
    ) -> Result<u64, StorageError> {
        let tasks = self.load_scope(&normalize_scope(scope), scope_id).await?;
        let by_id: HashMap<i64, &Task> = tasks.iter().map(|t| (t.id, t)).collect();
        let mut deleted = 0u64;
        // Delete children before parents so subtree deletes never double-count.
        let mut targets: Vec<&Task> = tasks
            .iter()
            .filter(|t| !closed_only || is_closed_status(&t.status))
            .collect();
        targets.sort_by_key(|t| std::cmp::Reverse(parent_depth_lenient(&tasks, t.id)));
        for target in targets {
            // delete() removes whole subtrees — clearing closed tasks must
            // never take a still-open descendant down with a closed parent.
            if closed_only {
                let has_open_descendant = subtree_ids_from(&by_id, target.id)
                    .into_iter()
                    .filter(|member_id| *member_id != target.id)
                    .any(|member_id| {
                        by_id
                            .get(&member_id)
                            .is_some_and(|t| !is_closed_status(&t.status))
                    });
                if has_open_descendant {
                    continue;
                }
            }
            // The subtree may already be gone via an earlier parent delete —
            // that, and only that, is a skip. A read that FAILED is not
            // evidence the row is gone; treating it as "already deleted" made
            // a database error look like a smaller, successful clear.
            match self.get_raw(target.id).await {
                Ok(_) => deleted += self.delete(target.id, None).await?,
                Err(StorageError::NotFound(_)) => {}
                Err(e) => return Err(e),
            }
        }
        Ok(deleted)
    }

    /// Import v0.1 JSON todo items (session-scoped flat list). `blocked`
    /// items become `pending` — blocked is derived now — with an activity
    /// entry preserving the old label. Returns imported count.
    ///
    /// # Errors
    /// Returns [`StorageError::Invalid`] if an item is imported into an empty
    /// `session_id` (a session scope needs an id) or a full scope, or
    /// [`StorageError::Database`] if a read or write fails. Items imported
    /// before a failure stay imported — and since the import skips any
    /// non-empty scope, a retry will not finish it.
    pub async fn import_v01(
        &self,
        session_id: &str,
        items: &[(String, String)],
    ) -> Result<usize, StorageError> {
        // Idempotence: a session being migrated has no store tasks yet. If
        // any exist, a previous (possibly partial) import or v0.2 usage
        // already populated the scope — never duplicate.
        let existing = self.load_scope("session", Some(session_id)).await?;
        if !existing.is_empty() {
            return Ok(0);
        }
        let mut imported = 0usize;
        for (index, (text, status)) in items.iter().enumerate() {
            if text.trim().is_empty() {
                continue;
            }
            let mut title = text.trim().to_string();
            if title.len() > TASK_TITLE_MAX_BYTES {
                // Cut on a char boundary: String::truncate panics mid-char.
                let mut cut = TASK_TITLE_MAX_BYTES;
                while cut > 0 && !title.is_char_boundary(cut) {
                    cut -= 1;
                }
                title.truncate(cut);
            }
            let task = self
                .create(NewTask {
                    scope: "session".to_string(),
                    scope_id: Some(session_id.to_string()),
                    title,
                    priority: 3,
                    // An enumerate index is below `items.len()`, which never
                    // exceeds `isize::MAX`, so the fallback is unreachable.
                    sort_order: i64::try_from(index).unwrap_or(i64::MAX),
                    ..NewTask::default()
                })
                .await?;
            match status.as_str() {
                "done" => {
                    let _ = self
                        .complete(task.id, Some("todo-v0.1-migration"), None)
                        .await?;
                }
                "in_progress" => {
                    let _ = self
                        .update(
                            task.id,
                            TaskPatch {
                                status: Some("in_progress".to_string()),
                                ..TaskPatch::default()
                            },
                            Some("todo-v0.1-migration"),
                        )
                        .await?;
                }
                "blocked" => {
                    self.log_activity(
                        task.id,
                        Some("todo-v0.1-migration"),
                        "imported_blocked",
                        Some(serde_json::json!({
                            "note": "v0.1 'blocked' label dropped: blocked is derived from depends_on now"
                        })),
                    )
                    .await?;
                }
                _ => {}
            }
            imported += 1;
        }
        Ok(imported)
    }

    /// Announce the time-driven lifecycle events: cards whose defer date has
    /// arrived (`due`) and open cards whose deadline has passed (`overdue`).
    ///
    /// Returns `(due, overdue)` counts. `now` is passed in rather than read
    /// here so the caller owns the clock and a test can cross a boundary
    /// without sleeping.
    ///
    /// Comparison is at **day** granularity, matching `validate_dates` and the
    /// filter language: these fields hold ISO strings that may be a bare day
    /// (`2026-07-20`) or a full timestamp, and a raw string compare would make
    /// a day-only deadline overdue from midnight of the day it is due.
    ///
    /// **Each card is announced once per crossing, not once per sweep.** The
    /// marker columns are what make that true, and they are re-armed when the
    /// date moves or the card is reopened — see migration `020`. Without them a
    /// sweep running every five minutes would re-announce the same overdue card
    /// forever, which is a notification bug rather than an event stream.
    ///
    /// Closed cards are skipped: a card that is done is not late, and P25
    /// decision 10 measures `overdue` against `deadline_at` alone — `due_at` is
    /// the defer date and says only when the card may start.
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the scan or a marker write fails.
    /// A failure partway leaves the cards already marked announced, which is
    /// the safe direction: the alternative is announcing them twice.
    pub async fn announce_due(
        &self,
        now: &str,
    ) -> Result<(usize, usize), StorageError> {
        let today = day_of(now);
        let pending = self.list_time_announcable(&today).await?;
        let mut due_count = 0usize;
        let mut overdue_count = 0usize;
        for task in pending {
            let due_crossed = task.due_announced_at.is_none()
                && task.due_at.as_deref().is_some_and(|at| day_of(at) <= today);
            let overdue_crossed = task.overdue_announced_at.is_none()
                && task
                    .deadline_at
                    .as_deref()
                    .is_some_and(|at| day_of(at) < today);
            if due_crossed {
                self.mark_announced(task.id, "due_announced_at", now).await?;
                self.emit(
                    TaskEventKind::Due,
                    &task,
                    None,
                    serde_json::json!({ "due_at": task.due_at }),
                );
                due_count += 1;
            }
            if overdue_crossed {
                self.mark_announced(task.id, "overdue_announced_at", now)
                    .await?;
                self.emit(
                    TaskEventKind::Overdue,
                    &task,
                    None,
                    serde_json::json!({ "deadline_at": task.deadline_at }),
                );
                overdue_count += 1;
            }
        }
        Ok((due_count, overdue_count))
    }

    /// Open cards with an un-announced date or deadline that `now` has reached.
    ///
    /// The filtering is done in SQL so a board with thousands of settled cards
    /// does not decode all of them every five minutes.
    async fn list_time_announcable(&self, today: &str) -> Result<Vec<Task>, StorageError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                &format!(
                    "SELECT {TASK_COLUMNS} FROM tasks \
                     WHERE status NOT IN ('done', 'cancelled') AND ( \
                       (due_announced_at IS NULL AND due_at IS NOT NULL \
                        AND substr(due_at, 1, 10) <= ?1) \
                       OR (overdue_announced_at IS NULL AND deadline_at IS NOT NULL \
                           AND substr(deadline_at, 1, 10) < ?1))"
                ),
                turso::params![today],
            )
            .await?;
        let mut tasks = Vec::new();
        while let Some(row) = rows.next().await? {
            tasks.push(decode_task_row(&row)?);
        }
        // Held until the cursor is gone: an open `Rows` on the shared
        // connection swallows later writes.
        drop(rows);
        drop(conn);
        Ok(tasks)
    }

    /// Stamp one announcement marker. The column name is a caller-supplied
    /// literal, never user input — the two call sites above are its only
    /// callers and both pass a constant.
    async fn mark_announced(
        &self,
        id: i64,
        column: &str,
        now: &str,
    ) -> Result<(), StorageError> {
        debug_assert!(
            column == "due_announced_at" || column == "overdue_announced_at",
            "marker column must be one of the two announcement markers"
        );
        let conn = self.conn.lock().await;
        conn.execute(
            &format!("UPDATE tasks SET {column} = ?1 WHERE id = ?2"),
            turso::params![now, id],
        )
        .await?;
        drop(conn);
        Ok(())
    }

    /// All completed tasks that carry a recurrence expression, across every
    /// scope. The daemon's recurrence sweep (one recurrence engine — the P8
    /// scheduler) computes the next occurrence and reopens due ones.
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the query fails or a row does not
    /// decode.
    pub async fn list_recurring_closed(&self) -> Result<Vec<Task>, StorageError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                &format!(
                    "SELECT {TASK_COLUMNS} FROM tasks \
                     WHERE recurrence IS NOT NULL AND status = 'done'"
                ),
                (),
            )
            .await?;
        let mut tasks = Vec::new();
        while let Some(row) = rows.next().await? {
            tasks.push(decode_task_row(&row)?);
        }
        // Held until the cursor is gone: an open `Rows` on the shared
        // connection swallows later writes.
        drop(rows);
        drop(conn);
        Ok(tasks)
    }

    /// (open, closed) counts for a scope — the progress line.
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the scope query fails or a row
    /// does not decode.
    pub async fn counts(
        &self,
        scope: &str,
        scope_id: Option<&str>,
    ) -> Result<(u64, u64), StorageError> {
        let tasks = self.load_scope(&normalize_scope(scope), scope_id).await?;
        let closed = tasks
            .iter()
            .filter(|t| is_closed_status(&t.status))
            .count() as u64;
        let open = tasks.len() as u64 - closed;
        Ok((open, closed))
    }

    // ------------------------------------------------------------------
    // internals
    // ------------------------------------------------------------------

    /// Close the open subtree of a just-cancelled `task` — see the comment at
    /// the call site in [`Self::update`] for which children re-parent instead.
    async fn cascade_cancel(&self, task: &Task, actor: Option<&str>) -> Result<(), StorageError> {
        let scope_tasks = self
            .load_scope(&task.scope, task.scope_id.as_deref())
            .await?;
        let acceptance_of: HashMap<i64, Option<serde_json::Value>> = scope_tasks
            .iter()
            .map(|t| (t.id, t.acceptance.clone()))
            .chain(std::iter::once((task.id, task.acceptance.clone())))
            .collect();
        let mut queue: VecDeque<i64> = VecDeque::from([task.id]);
        // Bounded by scope size: each task enters the queue at most once.
        let mut seen: HashSet<i64> = HashSet::from([task.id]);
        while let Some(current) = queue.pop_front() {
            for t in &scope_tasks {
                if t.parent_id == Some(current) && seen.insert(t.id) {
                    let own_contract = t.status != "done"
                        && t.status != "cancelled"
                        && t.acceptance.is_some()
                        && acceptance_of
                            .get(&current)
                            .is_none_or(|parent_acc| parent_acc != &t.acceptance);
                    if own_contract {
                        let mut child = t.clone();
                        child.parent_id = task.parent_id;
                        self.write_task(&child).await?;
                        self.log_activity(
                            t.id,
                            actor,
                            "reparented",
                            Some(serde_json::json!({
                                "cascade_from": task.id,
                                "old_parent": current,
                                "reason": "carries its own acceptance contract",
                            })),
                        )
                        .await?;
                        // Its subtree moves with it — do not descend.
                        continue;
                    }
                    if !is_closed_status(&t.status) {
                        let mut child = t.clone();
                        child.status = "cancelled".to_string();
                        self.write_task(&child).await?;
                        self.log_activity(
                            t.id,
                            actor,
                            "cancelled",
                            Some(serde_json::json!({ "cascade_from": task.id })),
                        )
                        .await?;
                    }
                    queue.push_back(t.id);
                }
            }
        }
        Ok(())
    }

    async fn get_raw(&self, id: i64) -> Result<Task, StorageError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                &format!("SELECT {TASK_COLUMNS} FROM tasks WHERE id = ?1"),
                turso::params![id],
            )
            .await?;
        let task = rows.next().await?.map_or_else(
            || Err(StorageError::NotFound(format!("Task: #{id}"))),
            |row| decode_task_row(&row),
        );
        // Held until the cursor is gone: an open `Rows` on the shared
        // connection swallows later writes.
        drop(rows);
        drop(conn);
        task
    }

    async fn load_scope(
        &self,
        scope: &str,
        scope_id: Option<&str>,
    ) -> Result<Vec<Task>, StorageError> {
        let conn = self.conn.lock().await;
        load_scope_with(&conn, scope, scope_id).await
    }

    async fn mark_done(&self, id: i64) -> Result<(), StorageError> {
        let conn = self.conn.lock().await;
        mark_done_with(&conn, id).await
    }

    async fn write_task(&self, task: &Task) -> Result<(), StorageError> {
        let conn = self.conn.lock().await;
        write_task_with(&conn, task).await
    }
}

/// Load a scope's tasks using an already-held connection guard (the mutex is
/// the transaction: validate-then-write sequences hold one guard throughout).
async fn load_scope_with(
    conn: &Connection,
    scope: &str,
    scope_id: Option<&str>,
) -> Result<Vec<Task>, StorageError> {
    let mut rows = match scope_id {
        Some(sid) => {
            conn.query(
                &format!("SELECT {TASK_COLUMNS} FROM tasks WHERE scope = ?1 AND scope_id = ?2"),
                turso::params![scope, sid],
            )
            .await?
        }
        None => {
            conn.query(
                &format!("SELECT {TASK_COLUMNS} FROM tasks WHERE scope = ?1 AND scope_id IS NULL"),
                turso::params![scope],
            )
            .await?
        }
    };
    let mut tasks = Vec::new();
    while let Some(row) = rows.next().await? {
        tasks.push(decode_task_row(&row)?);
    }
    Ok(tasks)
}

async fn mark_done_with(conn: &Connection, id: i64) -> Result<(), StorageError> {
    conn.execute(
        "UPDATE tasks SET status = 'done', completed_at = datetime('now'), \
         updated_at = datetime('now') WHERE id = ?1",
        turso::params![id],
    )
    .await?;
    Ok(())
}

async fn write_task_with(conn: &Connection, task: &Task) -> Result<(), StorageError> {
    let labels_json = serde_json::to_string(&task.labels)?;
    let tool_scope_json = serde_json::to_string(&task.tool_scope)?;
    let depends_json = serde_json::to_string(&task.depends_on)?;
    let acceptance_json = task
        .acceptance
        .as_ref()
        .map(std::string::ToString::to_string);
    conn.execute(
        "UPDATE tasks SET parent_id = ?1, project = ?2, title = ?3, description = ?4, \
         status = ?5, priority = ?6, labels = ?7, tool_scope = ?8, due_at = ?9, \
         recurrence = ?10, depends_on = ?11, acceptance = ?12, assignee = ?13, \
         sort_order = ?14, completed_at = ?15, deadline_at = ?16, \
         due_announced_at = ?17, overdue_announced_at = ?18, \
         updated_at = datetime('now') WHERE id = ?19",
        turso::params![
            task.parent_id,
            task.project.as_deref(),
            task.title.as_str(),
            task.description.as_deref(),
            task.status.as_str(),
            task.priority,
            labels_json.as_str(),
            tool_scope_json.as_str(),
            task.due_at.as_deref(),
            task.recurrence.as_deref(),
            depends_json.as_str(),
            acceptance_json.as_deref(),
            task.assignee.as_deref(),
            task.sort_order,
            task.completed_at.as_deref(),
            task.deadline_at.as_deref(),
            task.due_announced_at.as_deref(),
            task.overdue_announced_at.as_deref(),
            task.id,
        ],
    )
    .await?;
    Ok(())
}

/// Apply `patch.status`, if present — the first field [`apply_patch`] checks,
/// so a rejected status fails the patch before any other field is examined.
fn apply_status_patch(
    task: &mut Task,
    patch: &TaskPatch,
    changed: &mut Vec<&'static str>,
) -> Result<(), StorageError> {
    if let Some(status) = &patch.status {
        match status.as_str() {
            "pending" | "in_progress" | "cancelled" => {
                if task.status != *status {
                    changed.push("status");
                }
                task.status.clone_from(status);
                // Reopening a closed task must clear its completion stamp.
                if status != "cancelled" && task.completed_at.is_some() {
                    task.completed_at = None;
                }
            }
            "done" => {
                return Err(StorageError::Invalid(
                    "status 'done' is a verdict: use the complete/done action so the \
                     acceptance check can run"
                        .to_string(),
                ));
            }
            "blocked" => {
                return Err(StorageError::Invalid(
                    "'blocked' is derived from depends_on and cannot be set directly; \
                     add a dependency instead"
                        .to_string(),
                ));
            }
            other => {
                return Err(StorageError::Invalid(format!("unknown status '{other}'")));
            }
        }
    }
    Ok(())
}

/// What a `Created` event carries: enough for the router to place a new card
/// without re-reading it, and nothing more — the event says what happened, and
/// a full row on every mutation would make the bus a replication channel.
/// What an `acceptance_checked` row says about the member: `Some(passed)`, or
/// `None` when it says nothing — no detail, no boolean `passed`, or a check
/// that marked its own verdict `unknown` (it could not tell).
fn verdict_outcome(detail: Option<&str>) -> Option<bool> {
    let detail: serde_json::Value = serde_json::from_str(detail?).ok()?;
    if detail.get("unknown").and_then(serde_json::Value::as_bool) == Some(true) {
        return None;
    }
    detail.get("passed").and_then(serde_json::Value::as_bool)
}

fn created_detail(task: &Task) -> serde_json::Value {
    serde_json::json!({
        "title": task.title,
        "assignee": task.assignee,
        "parent_id": task.parent_id,
    })
}

/// Apply every field of `patch` to `task` in memory, validating each, and
/// return the names of the fields that changed. Nothing is written here.
fn apply_patch(task: &mut Task, patch: TaskPatch) -> Result<Vec<&'static str>, StorageError> {
    let mut changed: Vec<&'static str> = Vec::new();
    apply_status_patch(task, &patch, &mut changed)?;
    if let Some(title) = patch.title {
        validate_title(&title)?;
        if task.title != title {
            changed.push("title");
        }
        task.title = title;
    }
    if let Some(description) = patch.description {
        changed.push("description");
        task.description = description;
    }
    if let Some(priority) = patch.priority {
        validate_priority(priority)?;
        if task.priority != priority {
            changed.push("priority");
        }
        task.priority = priority;
    }
    if let Some(labels) = patch.labels {
        changed.push("labels");
        task.labels = labels;
    }
    if let Some(tool_scope) = patch.tool_scope {
        changed.push("tool_scope");
        task.tool_scope = tool_scope;
    }
    if let Some(due_at) = patch.due_at {
        changed.push("due_at");
        // Moving the date re-arms its announcement: a card pushed to next week
        // must be able to fall due again, and a marker left set would mean it
        // silently never does.
        if task.due_at != due_at {
            task.due_announced_at = None;
        }
        task.due_at = due_at;
    }
    if let Some(recurrence) = patch.recurrence {
        changed.push("recurrence");
        task.recurrence = recurrence;
    }
    if let Some(acceptance) = patch.acceptance {
        // Same write-boundary rule as `create`: canonicalize, then store
        // the canonical object (never the caller's raw value).
        let acceptance = match &acceptance {
            Some(check) => Some(admit_acceptance(check)?),
            None => None,
        };
        changed.push("acceptance");
        task.acceptance = acceptance;
    }
    if let Some(deadline_at) = patch.deadline_at {
        changed.push("deadline_at");
        // Same re-arm as `due_at`: extending a deadline must let the card go
        // overdue against the new one.
        if task.deadline_at != deadline_at {
            task.overdue_announced_at = None;
        }
        task.deadline_at = deadline_at;
    }
    if let Some(assignee) = patch.assignee {
        // Compared, not assumed. Every other field here may over-report a
        // change harmlessly into the activity log, but `assignee` now drives
        // an `assigned` lifecycle event that wakes the router: re-writing the
        // same member must not look like a hand-off.
        if task.assignee != assignee {
            changed.push("assignee");
        }
        task.assignee = assignee;
    }
    if let Some(project) = patch.project {
        changed.push("project");
        task.project = project;
    }
    if let Some(sort_order) = patch.sort_order {
        if task.sort_order != sort_order {
            changed.push("sort_order");
        }
        task.sort_order = sort_order;
    }

    if let Some(depends_on) = patch.depends_on {
        if depends_on.len() > TASK_DEPS_MAX {
            return Err(StorageError::Invalid(format!(
                "too many dependencies: {} (max {TASK_DEPS_MAX})",
                depends_on.len()
            )));
        }
        changed.push("depends_on");
        task.depends_on = depends_on;
    }
    if let Some(parent_id) = patch.parent_id {
        changed.push("parent_id");
        task.parent_id = parent_id;
    }
    Ok(changed)
}

/// Validate a dependency/parent change against the scope as it would be after
/// the update (`by_id` already holds the updated `task`). Pure: the caller
/// holds the connection guard across this and the write.
fn check_graph_update(
    by_id: &HashMap<i64, &Task>,
    task: &Task,
    parent_changed: bool,
) -> Result<(), StorageError> {
    for dep in &task.depends_on {
        if *dep == task.id {
            return Err(StorageError::Invalid(format!(
                "task #{} cannot depend on itself",
                task.id
            )));
        }
        if !by_id.contains_key(dep) {
            return Err(StorageError::Invalid(format!(
                "dependency task #{dep} not found in scope"
            )));
        }
    }
    check_dependency_cycle(by_id, task.id)?;

    if let Some(parent_id) = task.parent_id {
        if parent_id == task.id {
            return Err(StorageError::Invalid(format!(
                "task #{} cannot be its own parent",
                task.id
            )));
        }
        let Some(parent) = by_id.get(&parent_id) else {
            return Err(StorageError::Invalid(format!(
                "parent task #{parent_id} not found in scope"
            )));
        };
        if parent_changed && is_closed_status(&parent.status) {
            return Err(StorageError::Invalid(format!(
                "cannot move a task under {} task #{parent_id}",
                parent.status
            )));
        }
        check_parent_cycle(by_id, task.id)?;

        if parent_changed {
            // Re-parenting must not push the moved subtree past
            // the depth bound.
            let new_depth = parent_depth(by_id, parent_id)? + 1;
            let height = subtree_height(by_id, task.id);
            if new_depth + height >= TASK_DEPTH_MAX {
                return Err(StorageError::Invalid(format!(
                    "hierarchy depth limit {TASK_DEPTH_MAX} exceeded by re-parenting"
                )));
            }
        }
    }

    // The parent-completion invariant makes ancestors implicit
    // dependents: no task in the (possibly re-parented) subtree
    // may depend on one of its ancestors, or both wedge forever.
    let ancestors = ancestor_chain(by_id, task.parent_id);
    check_ancestor_dependency(by_id, &ancestors, &task.depends_on)?;
    if parent_changed {
        for member_id in subtree_ids_from(by_id, task.id) {
            if member_id == task.id {
                continue;
            }
            if let Some(member) = by_id.get(&member_id) {
                check_ancestor_dependency(by_id, &ancestors, &member.depends_on)?;
            }
        }
    }
    Ok(())
}

fn normalize_scope(scope: &str) -> String {
    let lower = scope.trim().to_lowercase();
    if lower.is_empty() {
        "session".to_string()
    } else {
        lower
    }
}

fn validate_scope(scope: &str, scope_id: Option<&str>) -> Result<(), StorageError> {
    match scope {
        "session" | "workspace" => {
            if scope_id.is_none_or(str::is_empty) {
                return Err(StorageError::Invalid(format!(
                    "scope '{scope}' requires a scope_id"
                )));
            }
            Ok(())
        }
        "global" => {
            if scope_id.is_some() {
                return Err(StorageError::Invalid(
                    "scope 'global' must not carry a scope_id".to_string(),
                ));
            }
            Ok(())
        }
        other => Err(StorageError::Invalid(format!(
            "unknown scope '{other}' (expected session|workspace|global)"
        ))),
    }
}

fn validate_title(title: &str) -> Result<(), StorageError> {
    let trimmed = title.trim();
    if trimmed.is_empty() {
        return Err(StorageError::Invalid("title is empty".to_string()));
    }
    if trimmed.len() > TASK_TITLE_MAX_BYTES {
        return Err(StorageError::Invalid(format!(
            "title exceeds {TASK_TITLE_MAX_BYTES} bytes (got {})",
            trimmed.len()
        )));
    }
    Ok(())
}

fn validate_priority(priority: i64) -> Result<(), StorageError> {
    if (1..=4).contains(&priority) {
        Ok(())
    } else {
        Err(StorageError::Invalid(format!(
            "priority {priority} out of range 1..=4 (1 is highest)"
        )))
    }
}

/// Every canonical acceptance shape, quoted verbatim in every parse error.
///
/// serde names the Rust type it wanted ("expected internally tagged enum
/// `AcceptanceCheck`") and never the shape, so a model has nothing to copy: the
/// daemon logs hold 121 identical malformed acceptance calls. An error that
/// carries the target shapes is the only version of this message that can end
/// the loop on the next attempt.
pub const ACCEPTANCE_SHAPES: &str = concat!(
    "`acceptance` must be a JSON OBJECT (not a string), exactly one of:\n",
    "  {\"kind\":\"command\",\"command\":\"cargo test\"}\n",
    "  {\"kind\":\"file_exists\",\"path\":\"docs/plan.md\"}\n",
    "  {\"kind\":\"regex\",\"pattern\":\"0 failed\",\"path\":\"build.log\"}\n",
    "  {\"kind\":\"regex\",\"pattern\":\"0 failed\",\"command\":\"cargo test\"}\n",
    "Optional on command and regex: \"timeout_secs\": <integer seconds>."
);

/// Normalize an acceptance payload to the ONE representation the store holds:
/// a JSON object.
///
/// The store is the single source of truth for what an acceptance check looks
/// like, so the tolerance for the dialect models actually emit lives here, at
/// the write boundary — not only in the reader. A reader that accepted more
/// than the writer stored is the exact disagreement that let a stringified
/// check pass the planner's admission and then be rejected by `create`,
/// silently dropping the task.
///
/// The one tolerated dialect is the object handed over as a JSON *string*. It
/// is unwrapped ONCE: a value still stringy after that is double-encoded
/// noise, and peeling deeper would accept payloads nobody wrote.
///
/// # Errors
/// Returns a shape-carrying message when the payload is neither an object nor
/// a string that decodes to one.
pub fn canonicalize_acceptance(
    value: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    if value.is_object() {
        let mut obj = value.clone();
        normalize_integral_timeout(&mut obj);
        return Ok(obj);
    }
    let Some(text) = value.as_str() else {
        return Err(format!("invalid acceptance check: {ACCEPTANCE_SHAPES}"));
    };
    let mut decoded = parse_stringified_acceptance(text)?;
    if !decoded.is_object() {
        return Err(format!(
            "invalid acceptance check: it arrived as a string, and the text inside is not an \
             object. {ACCEPTANCE_SHAPES}"
        ));
    }
    normalize_integral_timeout(&mut decoded);
    Ok(decoded)
}

/// Coerce an integral float in `timeout_secs` to the integer it denotes.
///
/// Every number that crosses the JS tool bridge is an f64 — a model writing
/// `timeout_secs: 60` hands the store `60.0` — so rejecting integral floats
/// refuses calls the model wrote correctly (observed 2026-08-08: one run had
/// 50 valid task-adds bounced on this, each answered with a rephrased
/// duplicate batch). Only exact non-negative integers coerce; a fractional
/// or negative value is left alone for [`validate_acceptance`] to reject —
/// the leniency is lossless by construction: [`exact_u64_below_2_pow_53`]
/// admits only integral, non-negative values below 2^53 (far beyond any
/// plausible timeout), and every one of those converts exactly.
fn normalize_integral_timeout(value: &mut serde_json::Value) {
    if let Some(obj) = value.as_object_mut()
        && let Some(timeout) = obj.get_mut("timeout_secs")
        && !timeout.is_u64()
        && let Some(f) = timeout.as_f64()
        && let Some(whole) = exact_u64_below_2_pow_53(f)
    {
        *timeout = serde_json::Value::from(whole);
    }
}

/// The integer `f` denotes when it is integral, non-negative, and below 2^53;
/// `None` for anything else (a fraction, a negative, NaN, an infinity, or an
/// integer too large for every neighbour to be exact). `-0.0` denotes 0.
///
/// Read off the IEEE-754 bits instead of cast: no `TryFrom<f64>` exists for
/// any integer type. A finite `f64` is `significand * 2^(exponent - 52)` with
/// the significand's top bit at position 52, so a value in `[1, 2^53)` has an
/// unbiased exponent in `0..=52`, is integral exactly when the
/// `52 - exponent` bits below its binary point are zero, and then denotes the
/// significand shifted right by that many bits — no step rounds. This admits
/// exactly the values `f.fract() == 0.0 && (0.0..2^53).contains(&f)` admits,
/// and returns the same integer `f as u64` would.
fn exact_u64_below_2_pow_53(f: f64) -> Option<u64> {
    const FRACTION_BITS: u64 = 52;
    const EXPONENT_BIAS: u64 = 1023;
    if f == 0.0 {
        return Some(0); // +0.0 and -0.0 alike
    }
    if f.is_sign_negative() {
        return None;
    }
    let bits = f.to_bits();
    // The sign bit is clear, so the biased exponent is everything above the
    // fraction. Below the bias is a fraction of one (or a subnormal); above
    // bias + 52 is 2^53 or more, an infinity, or NaN.
    let exponent = (bits >> FRACTION_BITS)
        .checked_sub(EXPONENT_BIAS)
        .filter(|&exponent| exponent <= FRACTION_BITS)?;
    let significand = (bits & ((1 << FRACTION_BITS) - 1)) | (1 << FRACTION_BITS);
    let fraction_shift = FRACTION_BITS - exponent;
    (significand & ((1 << fraction_shift) - 1) == 0).then_some(significand >> fraction_shift)
}

/// Decode an acceptance object that arrived as a string: strict JSON first,
/// then the JS-object-literal flavor via [`repair_js_object_literal`].
fn parse_stringified_acceptance(text: &str) -> Result<serde_json::Value, String> {
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(text) {
        return Ok(value);
    }
    repair_js_object_literal(text)
        .and_then(|repaired| serde_json::from_str::<serde_json::Value>(&repaired).ok())
        .ok_or_else(|| {
            format!(
                "invalid acceptance check: it arrived as a string, and the text inside is \
                 not JSON. {ACCEPTANCE_SHAPES}"
            )
        })
}

/// Rewrite the JS-object-literal flavor — `{kind: 'command', command: 'cargo
/// test'}` — into strict JSON, or `None` when that cannot be done without
/// guessing.
///
/// Only two rewrites are unambiguous: an unquoted KEY (a bare identifier
/// immediately followed by `:`) gains quotes, and a single-quoted string
/// becomes double-quoted. A bare token in VALUE position (`command: cargo
/// test --all`) is not repairable: where the value ends is a guess, and a
/// wrong guess writes an acceptance check the model never asked for. Those
/// fall through to the shape-carrying error, which is what teaches the fix.
fn repair_js_object_literal(text: &str) -> Option<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len() + text.len() / 4);
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            // Already-valid JSON string: copy through, escapes intact.
            '"' => {
                out.push('"');
                i += 1;
                loop {
                    let c = *chars.get(i)?;
                    i += 1;
                    out.push(c);
                    if c == '\\' {
                        out.push(*chars.get(i)?);
                        i += 1;
                    } else if c == '"' {
                        break;
                    }
                }
            }
            // Single-quoted string → double-quoted.
            '\'' => {
                out.push('"');
                i += 1;
                loop {
                    let c = *chars.get(i)?;
                    i += 1;
                    match c {
                        '\\' => {
                            let escaped = *chars.get(i)?;
                            i += 1;
                            // \' is not a JSON escape; every other escape is
                            // passed through unchanged.
                            if escaped == '\'' {
                                out.push('\'');
                            } else {
                                out.push('\\');
                                out.push(escaped);
                            }
                        }
                        '\'' => break,
                        '"' => out.push_str("\\\""),
                        other => out.push(other),
                    }
                }
                out.push('"');
            }
            c if c.is_alphabetic() || c == '_' => {
                let start = i;
                while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let token: String = chars[start..i].iter().collect();
                let colon_follows = chars[i..]
                    .iter()
                    .find(|c| !c.is_whitespace())
                    .is_some_and(|c| *c == ':');
                if colon_follows {
                    out.push('"');
                    out.push_str(&token);
                    out.push('"');
                } else if matches!(token.as_str(), "true" | "false" | "null") {
                    out.push_str(&token);
                } else {
                    return None;
                }
            }
            other => {
                out.push(other);
                i += 1;
            }
        }
    }
    Some(out)
}

/// Canonicalize + validate an acceptance payload on its way into the store.
///
/// Every write path funnels through here, so what `create`/`update` persist
/// is exactly what [`canonicalize_acceptance`] admits — never the raw value a
/// caller happened to hold.
fn admit_acceptance(value: &serde_json::Value) -> Result<serde_json::Value, StorageError> {
    let canonical = canonicalize_acceptance(value).map_err(StorageError::Invalid)?;
    validate_acceptance(&canonical)?;
    Ok(canonical)
}

/// Validate the acceptance-check JSON shape at write time so the harness
/// never meets a malformed check at run time.
fn validate_acceptance(value: &serde_json::Value) -> Result<(), StorageError> {
    let kind = value
        .get("kind")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            StorageError::Invalid(
                "acceptance requires a 'kind' of command|file_exists|regex".to_string(),
            )
        })?;
    let require_str = |field: &str| -> Result<(), StorageError> {
        if value
            .get(field)
            .and_then(serde_json::Value::as_str)
            .is_none_or(str::is_empty)
        {
            return Err(StorageError::Invalid(format!(
                "acceptance kind '{kind}' requires a non-empty '{field}'"
            )));
        }
        Ok(())
    };
    // timeout_secs must be an unsigned integer when present — the harness
    // deserializes it strictly, and a shape that validates here but fails to
    // parse there would wedge every run in the scope.
    if let Some(timeout) = value.get("timeout_secs")
        && !timeout.is_u64() {
            return Err(StorageError::Invalid(
                "acceptance 'timeout_secs' must be an unsigned integer (seconds)".to_string(),
            ));
        }
    match kind {
        "command" => require_str("command"),
        "file_exists" => require_str("path"),
        "regex" => {
            require_str("pattern")?;
            // regex needs a target: a file path or a command whose output is matched
            if value
                .get("path")
                .and_then(serde_json::Value::as_str)
                .is_none()
                && value
                    .get("command")
                    .and_then(serde_json::Value::as_str)
                    .is_none()
            {
                // Both forms are spelled out: the bare "requires path or
                // command" wording left the model to invent the rest, and it
                // re-sent the same targetless check 11 times.
                return Err(StorageError::Invalid(
                    "acceptance kind 'regex' needs something to match against — either a \
                     file: {\"kind\":\"regex\",\"pattern\":\"0 failed\",\"path\":\"build.log\"} \
                     or a command's output: \
                     {\"kind\":\"regex\",\"pattern\":\"0 failed\",\"command\":\"cargo test\"}"
                        .to_string(),
                ));
            }
            Ok(())
        }
        other => Err(StorageError::Invalid(format!(
            "unknown acceptance kind '{other}' (expected command|file_exists|regex)"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Acceptance evidence + user-declared file invariants (P23)
// ---------------------------------------------------------------------------

/// Every workspace path a canonical acceptance check NAMES — the subject it
/// observes plus anything its command reads.
///
/// Lives beside [`canonicalize_acceptance`] for the same reason the shapes do:
/// the write boundary owns what an acceptance check *is*, so a reader that
/// disagreed with it about which files a check names would be the exact drift
/// that let a stringified check pass admission and then be rejected at run
/// time. Callers hold a typed check; they serialize it back to the canonical
/// object and ask here, so there is one extraction, not two.
///
/// Deliberately conservative: only tokens that LOOK like a path are returned
/// (see [`looks_like_path`]). A caller resolves them against its workdir and
/// keeps the ones that are really files — the acceptance text is the bound on
/// the set, and a token that resolves to nothing costs one failed stat.
#[must_use]
pub fn acceptance_referenced_paths(acceptance: &serde_json::Value) -> Vec<String> {
    let mut paths: Vec<String> = Vec::new();
    // `path` is a path by contract (file_exists, path-flavored regex).
    if let Some(path) = acceptance.get("path").and_then(serde_json::Value::as_str) {
        push_path_token(&mut paths, path);
    }
    for evidence in acceptance_evidence_paths(acceptance) {
        push_path_token(&mut paths, &evidence);
    }
    paths
}

/// The subset of [`acceptance_referenced_paths`] that is the check's
/// INSTRUMENT rather than its subject: the files its command reads.
///
/// The distinction is load-bearing, and it is a distinction the check kinds
/// already draw. `file_exists` and a path-flavored `regex` observe their
/// subject DIRECTLY — the named file is the deliverable, and producing it is
/// the work, so there is no separable evidence to tamper with. A command
/// check runs a program: `sh tests/test_01.sh` renders its verdict THROUGH
/// that script, and a session that rewrites the script has changed what the
/// verdict means without changing the work at all.
///
/// Callers that must not confuse "the model built the artifact" with "the
/// model edited the judge" ask for this list, never the full one.
#[must_use]
pub fn acceptance_evidence_paths(acceptance: &serde_json::Value) -> Vec<String> {
    let mut paths: Vec<String> = Vec::new();
    let Some(command) = acceptance
        .get("command")
        .and_then(serde_json::Value::as_str)
    else {
        return paths;
    };
    // Split on shell word boundaries only — this is identification, not shell
    // parsing.
    for token in command.split(|c: char| c.is_whitespace() || matches!(c, '|' | ';' | '&')) {
        push_path_token(&mut paths, token);
    }
    paths
}

/// Append `candidate` to `paths` when it is an unambiguous path referent and
/// not already present.
fn push_path_token(paths: &mut Vec<String>, candidate: &str) {
    let cleaned = trim_path_token(candidate);
    if !cleaned.is_empty() && looks_like_path(cleaned) && !paths.iter().any(|p| p == cleaned) {
        paths.push(cleaned.to_string());
    }
}

/// Strip the punctuation a path token is wrapped in when it appears inside
/// prose or a shell line (quotes, backticks, brackets, a trailing comma or
/// sentence period). A trailing directory slash is kept — it is what makes
/// "tests/" a path referent at all.
fn trim_path_token(token: &str) -> &str {
    let trimmed = token.trim_matches(|c: char| {
        matches!(c, '"' | '\'' | '`' | '(' | ')' | '[' | ']' | '{' | '}' | ',' | ':')
            || c.is_whitespace()
    });
    // A sentence period must go; an extension's period must stay. Only a
    // TRAILING period directly after another period-free tail is punctuation.
    trimmed.strip_suffix('.').unwrap_or(trimmed)
}

/// Does this token name a file or directory *explicitly*?
///
/// Three unambiguous shapes, and nothing else: a trailing separator
/// ("tests/"), a glob ("tests/**"), or a dotted extension of at least two
/// characters containing a letter ("run.sh", "src/main.rs"). Everything
/// looser was rejected on purpose — a bare word ("tests") is a topic, not a
/// path, and "e.g." / "1.2.3" are not files. Over-matching here would let
/// prose register a durable prohibition the user never stated.
#[must_use]
pub fn looks_like_path(token: &str) -> bool {
    if token.is_empty() || token.contains("://") || token.starts_with('-') {
        return false;
    }
    let normalized = token.replace('\\', "/");
    if normalized.ends_with('/') {
        return true;
    }
    if normalized.contains('*') {
        return true;
    }
    let last = normalized.rsplit('/').next().unwrap_or("");
    let Some((stem, ext)) = last.rsplit_once('.') else {
        return false;
    };
    !stem.is_empty()
        && ext.len() >= 2
        && ext.chars().all(|c| c.is_ascii_alphanumeric())
        && ext.chars().any(|c| c.is_ascii_alphabetic())
}

/// The workspace file every write tool reads user-declared file prohibitions
/// from.
///
/// Same directory as the write ratchet's ledger — one place holds the
/// per-workspace state the tool layer consults before mutating.
pub const DECLARED_INVARIANTS_FILE: &str = ".nanna/declared_invariants.json";

/// Schema version of [`DECLARED_INVARIANTS_FILE`]. The tool side fails open on
/// anything it cannot read, so the version is a compatibility marker, never a
/// gate.
pub const DECLARED_INVARIANTS_VERSION: u64 = 1;

/// One durable file prohibition the USER stated in chat.
///
/// `source` is the user's own sentence, verbatim: a refusal that paraphrases
/// the constraint reads as a tool malfunction and gets routed around, while
/// one that quotes the user is recognizably the user's own instruction.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DeclaredInvariant {
    /// `read_only` | `no_delete` | `no_create_under`.
    pub kind: String,
    /// The path or glob, in the canonical spelling the write ratchet uses.
    pub glob: String,
    /// The user's verbatim sentence that declared it.
    pub source: String,
    /// The scope that declared it (a task scope id, or "session").
    pub scope: String,
}

/// Words that make a sentence a PROHIBITION rather than a description.
const INVARIANT_NEGATIONS: &[&str] = &[
    "never",
    "do not",
    "don't",
    "dont",
    "must not",
    "mustn't",
    "cannot",
    "can not",
    "shall not",
    "should not",
    "shouldn't",
    "hands off",
];

/// Standing declarations that ARE the prohibition (no verb needed).
const INVARIANT_DECLARATIONS: &[&str] = &[
    "read-only",
    "readonly",
    "is read only",
    "are read only",
    "as read only",
    "off-limits",
    "off limits",
];

/// Anything that makes the sentence conditional, hedged, or permissive. A
/// durable constraint is unconditional by definition, so any of these means
/// the sentence registers NOTHING.
const INVARIANT_AMBIGUITY: &[&str] = &[
    "if ",
    "unless",
    "except",
    "when ",
    "while ",
    "maybe",
    "might",
    "probably",
    "not sure",
    "unsure",
    "you can ",
    "you may ",
    "feel free",
    "ok to ",
    "okay to ",
    "fine to ",
    "allowed to",
    "prefer not",
    "try not",
];

/// Extract durable file prohibitions from the user's own text.
///
/// The contract is deliberately lopsided: the tool layer FAILS OPEN when the
/// registry is absent, so a missed constraint costs exactly today's behavior
/// while an invented one blocks work the user asked for. Everything below is
/// therefore a reason to register nothing — only an imperative sentence with
/// an explicit path referent and an unambiguous verb ever produces an entry.
///
/// Bound: the registry is what the user actually said. One entry per (kind,
/// path) named in the text; no caps, because there is no growth term that is
/// not the user typing.
#[must_use]
pub fn extract_declared_invariants(text: &str, scope: &str) -> Vec<DeclaredInvariant> {
    let mut out: Vec<DeclaredInvariant> = Vec::new();
    for sentence in split_sentences(text) {
        let source = sentence.trim().trim_end_matches(['.', '!', ';']).trim();
        if source.is_empty() {
            continue;
        }
        let lower = source.to_lowercase();
        // A question asks; it does not declare.
        if sentence.trim_end().ends_with('?') || lower.contains(" or should ") {
            continue;
        }
        if INVARIANT_AMBIGUITY.iter().any(|m| lower.contains(m)) {
            continue;
        }
        let negated = INVARIANT_NEGATIONS.iter().any(|m| lower.contains(m));
        let declared = INVARIANT_DECLARATIONS.iter().any(|m| lower.contains(m));
        if !negated && !declared {
            continue;
        }
        let kinds = prohibited_kinds(&lower, negated, declared);
        if kinds.is_empty() {
            continue;
        }
        for token in source.split_whitespace() {
            let cleaned = trim_path_token(token);
            if cleaned.is_empty() || !looks_like_path(cleaned) {
                continue;
            }
            let glob = normalize_invariant_glob(cleaned);
            if glob.is_empty() {
                continue;
            }
            for kind in &kinds {
                if out.iter().any(|e| e.kind == *kind && e.glob == glob) {
                    continue;
                }
                out.push(DeclaredInvariant {
                    kind: (*kind).to_string(),
                    glob: glob.clone(),
                    source: source.to_string(),
                    scope: scope.to_string(),
                });
            }
        }
    }
    out
}

/// Which prohibition kinds does this (already-negated or already-declarative)
/// sentence state? A negation with no recognized verb states nothing — "never
/// again with tests/run.sh" is not a file rule.
fn prohibited_kinds(lower: &str, negated: bool, declared: bool) -> Vec<&'static str> {
    let mut kinds: Vec<&'static str> = Vec::new();
    let mut add = |kind: &'static str| {
        if !kinds.contains(&kind) {
            kinds.push(kind);
        }
    };
    if declared {
        add("read_only");
    }
    if negated {
        const MODIFY: &[&str] = &[
            "edit", "modify", "change", "rewrite", "overwrite", "alter", "update", "touch",
            "write", "patch",
        ];
        // " rm " keeps its spaces on purpose: the bare substring lives inside
        // "perform", and a prohibition invented out of "do not perform" is
        // exactly the over-registration this extractor exists to avoid.
        const DELETE: &[&str] = &["delete", "remove", "erase", "unlink", "wipe", " rm "];
        const CREATE: &[&str] = &["create", "add ", "new file", "generate", "scaffold"];
        if MODIFY.iter().any(|v| lower.contains(v)) {
            add("read_only");
        }
        if DELETE.iter().any(|v| lower.contains(v)) {
            add("no_delete");
        }
        if CREATE.iter().any(|v| lower.contains(v)) {
            add("no_create_under");
        }
    }
    kinds
}

/// Sentence split that a path can survive: only a terminator FOLLOWED BY
/// whitespace (or end of text) ends a sentence, so "config.json" stays whole
/// while "…config.json. Fix the code" splits. Newlines always split — a list
/// of rules is a list of sentences.
fn split_sentences(text: &str) -> Vec<&str> {
    let bytes = text.as_bytes();
    let mut out: Vec<&str> = Vec::new();
    let mut start = 0usize;
    for (i, c) in text.char_indices() {
        let terminator = matches!(c, '.' | '!' | '?' | ';' | '\n');
        if !terminator {
            continue;
        }
        let next = bytes.get(i + c.len_utf8());
        let ends_here = c == '\n'
            || next.is_none()
            || next.is_some_and(u8::is_ascii_whitespace);
        if !ends_here {
            continue;
        }
        let end = i + c.len_utf8();
        out.push(&text[start..end]);
        start = end;
    }
    if start < text.len() {
        out.push(&text[start..]);
    }
    out
}

/// The canonical spelling of a declared glob.
///
/// It is the SAME normalization the write ratchet's ledger key uses
/// (backslashes to slashes, lowercase, no `./` prefix, no doubled or trailing
/// separators), so one file has one identity on both sides of the contract.
#[must_use]
pub fn normalize_invariant_glob(glob: &str) -> String {
    let mut key = glob.replace('\\', "/").to_lowercase();
    while let Some(rest) = key.strip_prefix("./") {
        key = rest.to_string();
    }
    while key.contains("//") {
        key = key.replace("//", "/");
    }
    while key.len() > 1 && key.ends_with('/') {
        key.pop();
    }
    key
}

/// Fold freshly-extracted invariants into the registry document already on
/// disk, returning the new document — or `None` when nothing changed, so a
/// turn that declares nothing new never rewrites the file.
///
/// Invariants ACCUMULATE: a constraint stated three turns ago is still the
/// user's instruction this turn, and only the user lifts one. An unreadable or
/// malformed existing document is treated as empty (the same fail-open rule
/// the tool side applies), so a corrupted registry heals on the next
/// declaration instead of wedging.
#[must_use]
pub fn merge_declared_invariants(
    existing_json: &str,
    fresh: &[DeclaredInvariant],
) -> Option<String> {
    let mut merged: Vec<DeclaredInvariant> = parse_declared_invariants(existing_json);
    let mut added = 0usize;
    for invariant in fresh {
        if merged
            .iter()
            .any(|e| e.kind == invariant.kind && e.glob == invariant.glob)
        {
            continue;
        }
        merged.push(invariant.clone());
        added += 1;
    }
    if added == 0 {
        return None;
    }
    serde_json::to_string_pretty(&serde_json::json!({
        "version": DECLARED_INVARIANTS_VERSION,
        "invariants": merged,
    }))
    .ok()
}

/// The registry's invariants whose glob is `glob` (trailing `/` ignored). Pure.
#[must_use]
pub fn declared_invariants_for(existing_json: &str, glob: &str) -> Vec<DeclaredInvariant> {
    let wanted = glob.trim().trim_end_matches('/');
    parse_declared_invariants(existing_json)
        .into_iter()
        .filter(|inv| !wanted.is_empty() && inv.glob.trim_end_matches('/') == wanted)
        .collect()
}

fn parse_declared_invariants(existing_json: &str) -> Vec<DeclaredInvariant> {
    serde_json::from_str::<serde_json::Value>(existing_json)
        .ok()
        .and_then(|v| v.get("invariants").cloned())
        .and_then(|v| serde_json::from_value::<Vec<DeclaredInvariant>>(v).ok())
        .unwrap_or_default()
}

/// The registry with every invariant on `glob` removed, or `None` when none
/// matched. Pure. Lifting is the user's act, so the caller must already hold
/// their explicit yes (see [`is_explicit_yes`]).
#[must_use]
pub fn lift_declared_invariants(existing_json: &str, glob: &str) -> Option<String> {
    let wanted = glob.trim().trim_end_matches('/');
    let all = parse_declared_invariants(existing_json);
    let kept: Vec<&DeclaredInvariant> = all
        .iter()
        .filter(|inv| wanted.is_empty() || inv.glob.trim_end_matches('/') != wanted)
        .collect();
    if kept.len() == all.len() {
        return None;
    }
    debug_assert!(kept.len() < all.len());
    serde_json::to_string_pretty(&serde_json::json!({
        "version": DECLARED_INVARIANTS_VERSION,
        "invariants": kept,
    }))
    .ok()
}

/// Whether a reply is an unambiguous yes. Pure.
///
/// Deliberately narrow — the same conservatism the extractor applies to
/// declaring an invariant applies to lifting one. The reply must START with a
/// yes word and carry no negation anywhere: "yes" and "sure, go ahead" lift;
/// "yes but not tests/unit", "no", "not yet" and silence do not.
#[must_use]
pub fn is_explicit_yes(reply: &str) -> bool {
    const YES: &[&str] = &[
        "yes", "y", "yep", "yeah", "sure", "ok", "okay", "allow", "lift", "go",
    ];
    const NEGATIONS: &[&str] = &[
        "no", "not", "don't", "dont", "never", "wait", "stop", "keep",
    ];
    let lowered = reply.trim().to_lowercase();
    let words: Vec<&str> = lowered
        .split(|c: char| !(c.is_alphanumeric() || c == '\''))
        .filter(|w| !w.is_empty())
        .collect();
    let Some(first) = words.first() else {
        return false;
    };
    YES.contains(first) && !words.iter().any(|w| NEGATIONS.contains(w))
}

/// Which tasks in a scope snapshot are blocked.
///
/// Mirrors [`is_blocked`] over a whole snapshot: a dependency blocks only if it
/// is both **present** in the scope and **open**, so a dangling reference does
/// not block (deleting a dependency releases its dependents, which is what
/// `delete`'s dep-stripping relies on).
fn blocked_ids(tasks: &[Task]) -> HashSet<i64> {
    let open: HashSet<i64> = tasks
        .iter()
        .filter(|t| !is_closed_status(&t.status))
        .map(|t| t.id)
        .collect();
    tasks
        .iter()
        .filter(|t| t.depends_on.iter().any(|dep| open.contains(dep)))
        .map(|t| t.id)
        .collect()
}

/// The day part of an ISO date or timestamp, which is the granularity the task
/// store compares dates at (see `validate_dates` and the filter language). A
/// bare `2026-07-20` and a full `2026-07-20T14:03:00Z` both yield `2026-07-20`.
fn day_of(at: &str) -> String {
    at.chars().take(10).collect()
}

/// Whether a status means the task is closed.
///
/// The single definition of "closed" in the store. It was written out by hand
/// in twelve places, which matters more than it looks: `blocked` is *derived*
/// from this predicate on every read, so a copy that drifts does not produce a
/// visibly wrong field — it produces a task that is blocked to one reader and
/// actionable to another. Adding a third closed status is now one edit.
fn is_closed_status(status: &str) -> bool {
    matches!(status, "done" | "cancelled")
}

fn is_blocked(task: &Task, by_id: &HashMap<i64, &Task>) -> bool {
    task.depends_on.iter().any(|dep| {
        by_id
            .get(dep)
            .is_some_and(|t| !is_closed_status(&t.status))
    })
}

/// Depth of a task counted from the root (root = 0). Errors when the parent
/// chain exceeds the depth bound (which would indicate a corrupted store).
fn parent_depth(by_id: &HashMap<i64, &Task>, id: i64) -> Result<usize, StorageError> {
    let mut depth = 0usize;
    let mut current = Some(id);
    while let Some(task_id) = current {
        if depth >= TASK_DEPTH_MAX {
            return Err(StorageError::Invalid(format!(
                "hierarchy depth limit {TASK_DEPTH_MAX} exceeded at task #{task_id}"
            )));
        }
        current = by_id.get(&task_id).and_then(|t| t.parent_id);
        if current.is_some() {
            depth += 1;
        }
    }
    Ok(depth)
}

/// Lenient depth for delete ordering (never errors; saturates at the bound).
fn parent_depth_lenient(tasks: &[Task], id: i64) -> usize {
    let by_id: HashMap<i64, &Task> = tasks.iter().map(|t| (t.id, t)).collect();
    let mut depth = 0usize;
    let mut current = by_id.get(&id).and_then(|t| t.parent_id);
    while let Some(parent_id) = current {
        depth += 1;
        if depth >= TASK_DEPTH_MAX {
            break;
        }
        current = by_id.get(&parent_id).and_then(|t| t.parent_id);
    }
    depth
}

/// The set of a task's ancestors: the parent chain starting at
/// `start_parent`, bounded by the depth limit.
fn ancestor_chain(by_id: &HashMap<i64, &Task>, start_parent: Option<i64>) -> HashSet<i64> {
    let mut ancestors = HashSet::new();
    let mut current = start_parent;
    while let Some(id) = current {
        if !ancestors.insert(id) || ancestors.len() >= TASK_DEPTH_MAX {
            break;
        }
        current = by_id.get(&id).and_then(|t| t.parent_id);
    }
    ancestors
}

/// Reject dependencies that (transitively) reach one of the task's ancestors.
/// The parent-completion invariant makes an ancestor an implicit dependent of
/// this task — depending on it creates an unresolvable mutual wait that
/// `next()` would report as an (untrue) empty plan.
fn check_ancestor_dependency(
    by_id: &HashMap<i64, &Task>,
    ancestors: &HashSet<i64>,
    deps: &[i64],
) -> Result<(), StorageError> {
    if ancestors.is_empty() || deps.is_empty() {
        return Ok(());
    }
    let mut visited: HashSet<i64> = HashSet::new();
    let mut queue: VecDeque<i64> = deps.iter().copied().collect();
    while let Some(current) = queue.pop_front() {
        if ancestors.contains(&current) {
            return Err(StorageError::Invalid(format!(
                "task #{current} is an ancestor of this task; depending on it (directly or \
                 transitively) would wedge both — an ancestor cannot complete while its \
                 children are open"
            )));
        }
        if !visited.insert(current) {
            continue;
        }
        if let Some(task) = by_id.get(&current) {
            queue.extend(task.depends_on.iter().copied());
        }
    }
    Ok(())
}

/// Every task in the subtree rooted at `root` (including `root`), bounded by
/// the number of tasks in the map.
fn subtree_ids_from(by_id: &HashMap<i64, &Task>, root: i64) -> Vec<i64> {
    let mut ids = vec![root];
    let mut queue: VecDeque<i64> = VecDeque::from([root]);
    let mut seen: HashSet<i64> = HashSet::from([root]);
    while let Some(current) = queue.pop_front() {
        for task in by_id.values() {
            if task.parent_id == Some(current) && seen.insert(task.id) {
                ids.push(task.id);
                queue.push_back(task.id);
            }
        }
    }
    ids
}

/// Height of the subtree below `root` (0 = leaf), saturating at the depth
/// bound.
fn subtree_height(by_id: &HashMap<i64, &Task>, root: i64) -> usize {
    let mut height = 0usize;
    let mut frontier = vec![root];
    let mut seen: HashSet<i64> = HashSet::from([root]);
    for _ in 0..TASK_DEPTH_MAX {
        let mut next_frontier = Vec::new();
        for current in &frontier {
            for task in by_id.values() {
                if task.parent_id == Some(*current) && seen.insert(task.id) {
                    next_frontier.push(task.id);
                }
            }
        }
        if next_frontier.is_empty() {
            break;
        }
        height += 1;
        frontier = next_frontier;
    }
    height
}

/// Reject dependency graphs where following `depends_on` edges from `start`
/// can reach `start` again (cycle check on write — reject, don't detect
/// later). Bounded by the number of tasks in scope.
fn check_dependency_cycle(by_id: &HashMap<i64, &Task>, start: i64) -> Result<(), StorageError> {
    let Some(start_task) = by_id.get(&start) else {
        return Ok(());
    };
    let mut visited: HashSet<i64> = HashSet::new();
    let mut queue: VecDeque<i64> = start_task.depends_on.iter().copied().collect();
    while let Some(current) = queue.pop_front() {
        if current == start {
            return Err(StorageError::Invalid(format!(
                "dependency cycle: task #{start} would (transitively) depend on itself"
            )));
        }
        if !visited.insert(current) {
            continue;
        }
        if let Some(task) = by_id.get(&current) {
            queue.extend(task.depends_on.iter().copied());
        }
    }
    Ok(())
}

/// Reject parent chains that loop back to `start`.
fn check_parent_cycle(by_id: &HashMap<i64, &Task>, start: i64) -> Result<(), StorageError> {
    let mut current = by_id.get(&start).and_then(|t| t.parent_id);
    let mut steps = 0usize;
    while let Some(parent_id) = current {
        if parent_id == start {
            return Err(StorageError::Invalid(format!(
                "hierarchy cycle: task #{start} would be its own ancestor"
            )));
        }
        steps += 1;
        if steps >= TASK_DEPTH_MAX {
            return Err(StorageError::Invalid(format!(
                "hierarchy depth limit {TASK_DEPTH_MAX} exceeded"
            )));
        }
        current = by_id.get(&parent_id).and_then(|t| t.parent_id);
    }
    Ok(())
}

/// A card deferred past its own deadline can never be worked (P25 decision 10):
/// it stays out of the inbox until `due_at`, by which time `deadline_at` has
/// already passed. Same-day is fine — the comparison is on the date part, so an
/// afternoon deadline on a morning's defer date is admitted.
///
/// Both values are ISO (`YYYY-MM-DD` or a full timestamp), which the store has
/// required since P15, so the first ten characters order correctly as text.
fn validate_dates(due_at: Option<&str>, deadline_at: Option<&str>) -> Result<(), StorageError> {
    let (Some(defer), Some(deadline)) = (due_at, deadline_at) else {
        return Ok(());
    };
    let defer_day: String = defer.chars().take(10).collect();
    let deadline_day: String = deadline.chars().take(10).collect();
    if deadline_day < defer_day {
        return Err(StorageError::Invalid(format!(
            "deadline {deadline} is before the defer date {defer}: the card would never be \
             workable"
        )));
    }
    Ok(())
}

/// `tasks.assignee` holds a `members.id` (P25 Stage 1). SQLite cannot add a
/// foreign key to an existing column and `PRAGMA foreign_keys` is off on this
/// connection, so the reference is enforced here — on write, under the same
/// guard as the write, which is stronger than the declarative constraint would
/// have been anyway.
///
/// `None` is unassigned and always admitted: that is the state a card sits in
/// between creation and the router placing it.
/// Every field-level check a new task must pass, in one place: they are
/// independent of the store, so running them before the scope is loaded keeps
/// a malformed request from costing a read.
fn validate_new_task_fields(new: &NewTask) -> Result<(), StorageError> {
    validate_scope(&new.scope, new.scope_id.as_deref())?;
    validate_title(&new.title)?;
    validate_priority(new.priority)?;
    validate_dates(new.due_at.as_deref(), new.deadline_at.as_deref())?;
    if new.depends_on.len() > TASK_DEPS_MAX {
        return Err(StorageError::Invalid(format!(
            "too many dependencies: {} (max {TASK_DEPS_MAX})",
            new.depends_on.len()
        )));
    }
    Ok(())
}

/// The shared half of the reference check above: `None` is admitted, anything
/// else must name a row in `members`. `role` names the field in the error so a
/// bad assignee and a bad note author do not read identically.
async fn ensure_member_exists(
    conn: &Connection,
    member_id: Option<&str>,
    role: &str,
) -> Result<(), StorageError> {
    let Some(member_id) = member_id else {
        return Ok(());
    };
    let owned = member_id.to_string();
    let mut rows = conn
        .query("SELECT 1 FROM members WHERE id = ?1", turso::params![owned])
        .await?;
    let found = rows.next().await?.is_some();
    // Drop the open cursor before the caller's write: an unfinished Rows on
    // the shared turso connection silently swallows later writes.
    drop(rows);
    if found {
        return Ok(());
    }
    Err(StorageError::Invalid(format!(
        "{role} '{member_id}' is not a board member; create the member first"
    )))
}

/// Decode one `task_notes` row in `TASK_NOTE_COLUMNS` order.
fn decode_task_note(row: &turso::Row) -> Result<TaskNote, StorageError> {
    let id: i64 = row.get(0)?;
    let kind_token: String = row.get(6)?;
    // An unknown kind is a post from a newer schema. Flattening it to a
    // comment would silently demote a question or a verdict.
    let kind = TaskNoteKind::parse(&kind_token).ok_or_else(|| {
        StorageError::Invalid(format!("note #{id} has unknown kind '{kind_token}'"))
    })?;
    Ok(TaskNote {
        id,
        task_id: row.get(1)?,
        author: row.get(2)?,
        content: row.get(3)?,
        created_at: row.get(4)?,
        author_member_id: row.get(5)?,
        kind,
    })
}

fn decode_task_row(row: &turso::Row) -> Result<Task, StorageError> {
    let labels_str: String = row.get(9)?;
    let tool_scope_str: String = row.get(10)?;
    let depends_str: String = row.get(13)?;
    let acceptance_str: Option<String> = row.get(14)?;
    Ok(Task {
        id: row.get(0)?,
        parent_id: row.get(1)?,
        scope: row.get(2)?,
        scope_id: row.get(3)?,
        project: row.get(4)?,
        title: row.get(5)?,
        description: row.get(6)?,
        status: row.get(7)?,
        priority: row.get(8)?,
        labels: serde_json::from_str(&labels_str).unwrap_or_default(),
        tool_scope: serde_json::from_str(&tool_scope_str).unwrap_or_default(),
        due_at: row.get(11)?,
        recurrence: row.get(12)?,
        depends_on: serde_json::from_str(&depends_str).unwrap_or_default(),
        acceptance: acceptance_str.and_then(|s| serde_json::from_str(&s).ok()),
        assignee: row.get(15)?,
        sort_order: row.get(16)?,
        created_at: row.get(17)?,
        updated_at: row.get(18)?,
        completed_at: row.get(19)?,
        deadline_at: row.get(20)?,
        due_announced_at: row.get(21)?,
        overdue_announced_at: row.get(22)?,
        blocked: false,
    })
}

#[cfg(test)]
mod tests {

    #[test]
    fn only_an_unambiguous_yes_lifts_a_declared_rule() {
        for yes in ["yes", "Yes!", "  y", "sure, go ahead", "ok lift it", "yeah"] {
            assert!(is_explicit_yes(yes), "{yes:?}");
        }
        for not_yes in [
            "",
            "no",
            "not yet",
            "yes but not tests/unit",
            "keep it",
            "maybe",
            "wait, yes",
            "don't",
        ] {
            assert!(!is_explicit_yes(not_yes), "{not_yes:?}");
        }
    }

    #[test]
    fn lifting_removes_exactly_the_rules_on_that_glob() {
        let registry = serde_json::json!({
            "version": 1,
            "invariants": [
                { "kind": "read_only", "glob": "tests/", "source": "don't touch tests/", "scope": "session" },
                { "kind": "no_delete", "glob": "tests", "source": "never delete tests", "scope": "session" },
                { "kind": "read_only", "glob": "spec.md", "source": "leave spec.md alone", "scope": "session" },
            ]
        })
        .to_string();
        assert_eq!(
            declared_invariants_for(&registry, "tests").len(),
            2,
            "trailing slash is the same rule"
        );
        let lifted = lift_declared_invariants(&registry, "tests/").expect("something lifted");
        let remaining = declared_invariants_for(&lifted, "spec.md");
        assert_eq!(remaining.len(), 1);
        assert_eq!(declared_invariants_for(&lifted, "tests").len(), 0);
        assert_eq!(
            lift_declared_invariants(&lifted, "tests"),
            None,
            "nothing left to lift"
        );
        assert_eq!(lift_declared_invariants("not json", "tests"), None);
        assert_eq!(
            lift_declared_invariants(&registry, "  "),
            None,
            "an empty glob lifts nothing"
        );
    }

    use super::*;
    use crate::Storage;

    async fn repo() -> (Storage, TaskRepository) {
        let storage = Storage::in_memory().await.unwrap();
        let tasks = storage.tasks();
        (storage, tasks)
    }

    fn new_task(title: &str) -> NewTask {
        NewTask {
            scope: "session".to_string(),
            scope_id: Some("s1".to_string()),
            title: title.to_string(),
            priority: 3,
            ..NewTask::default()
        }
    }

    #[tokio::test]
    async fn create_assigns_id_and_logs_activity() {
        let (_s, repo) = repo().await;
        let task = repo.create(new_task("first")).await.unwrap();
        assert!(task.id >= 1, "expected positive id, got {}", task.id);
        assert_eq!(task.status, "pending");
        let activity = repo.activity(task.id, 10).await.unwrap();
        assert_eq!(activity.len(), 1);
        assert_eq!(activity[0].action, "created");
    }

    #[tokio::test]
    async fn create_rejects_empty_title_and_bad_priority_and_bad_scope() {
        let (_s, repo) = repo().await;
        let mut t = new_task("  ");
        assert!(matches!(
            repo.create(t).await,
            Err(StorageError::Invalid(_))
        ));
        t = new_task("ok");
        t.priority = 0;
        assert!(matches!(
            repo.create(t).await,
            Err(StorageError::Invalid(_))
        ));
        t = new_task("ok");
        t.priority = 5;
        assert!(matches!(
            repo.create(t).await,
            Err(StorageError::Invalid(_))
        ));
        t = new_task("ok");
        t.scope = "galaxy".to_string();
        assert!(matches!(
            repo.create(t).await,
            Err(StorageError::Invalid(_))
        ));
        t = new_task("ok");
        t.scope_id = None;
        assert!(
            matches!(repo.create(t).await, Err(StorageError::Invalid(_))),
            "session scope requires scope_id"
        );
    }

    #[tokio::test]
    async fn scopes_are_disjoint() {
        let (_s, repo) = repo().await;
        repo.create(new_task("session task")).await.unwrap();
        let mut g = new_task("global task");
        g.scope = "global".to_string();
        g.scope_id = None;
        repo.create(g).await.unwrap();

        let session = repo.list("session", Some("s1"), true).await.unwrap();
        let global = repo.list("global", None, true).await.unwrap();
        assert_eq!(session.len(), 1);
        assert_eq!(global.len(), 1);
        assert_eq!(session[0].title, "session task");
        assert_eq!(global[0].title, "global task");
    }

    #[tokio::test]
    async fn blocked_is_derived_from_open_dependencies() {
        let (_s, repo) = repo().await;
        let dep = repo.create(new_task("dep")).await.unwrap();
        let mut nt = new_task("dependent");
        nt.depends_on = vec![dep.id];
        let dependent = repo.create(nt).await.unwrap();

        let fetched = repo.get(dependent.id).await.unwrap();
        assert!(fetched.blocked, "open dependency must derive blocked=true");

        repo.complete(dep.id, None, None).await.unwrap();
        let fetched = repo.get(dependent.id).await.unwrap();
        assert!(!fetched.blocked, "completed dependency must unblock");
    }

    #[tokio::test]
    async fn blocked_cannot_be_written_directly() {
        let (_s, repo) = repo().await;
        let task = repo.create(new_task("t")).await.unwrap();
        let err = repo
            .update(
                task.id,
                TaskPatch {
                    status: Some("blocked".to_string()),
                    ..TaskPatch::default()
                },
                None,
            )
            .await;
        assert!(matches!(err, Err(StorageError::Invalid(_))));
    }

    #[tokio::test]
    async fn done_cannot_be_written_via_update() {
        let (_s, repo) = repo().await;
        let task = repo.create(new_task("t")).await.unwrap();
        let err = repo
            .update(
                task.id,
                TaskPatch {
                    status: Some("done".to_string()),
                    ..TaskPatch::default()
                },
                None,
            )
            .await;
        assert!(
            matches!(err, Err(StorageError::Invalid(_))),
            "done must go through complete()"
        );
    }

    #[tokio::test]
    async fn dependency_cycle_is_rejected_on_write() {
        let (_s, repo) = repo().await;
        let a = repo.create(new_task("a")).await.unwrap();
        let mut nt = new_task("b");
        nt.depends_on = vec![a.id];
        let b = repo.create(nt).await.unwrap();

        // a -> b would close the loop a -> b -> a
        let err = repo
            .update(
                a.id,
                TaskPatch {
                    depends_on: Some(vec![b.id]),
                    ..TaskPatch::default()
                },
                None,
            )
            .await;
        assert!(
            matches!(err, Err(StorageError::Invalid(_))),
            "cycle must be rejected"
        );

        // self-dependency is the smallest cycle
        let err = repo
            .update(
                a.id,
                TaskPatch {
                    depends_on: Some(vec![a.id]),
                    ..TaskPatch::default()
                },
                None,
            )
            .await;
        assert!(matches!(err, Err(StorageError::Invalid(_))));
    }

    #[tokio::test]
    async fn parent_cycle_is_rejected_on_write() {
        let (_s, repo) = repo().await;
        let a = repo.create(new_task("a")).await.unwrap();
        let mut nt = new_task("b");
        nt.parent_id = Some(a.id);
        let b = repo.create(nt).await.unwrap();
        let err = repo
            .update(
                a.id,
                TaskPatch {
                    parent_id: Some(Some(b.id)),
                    ..TaskPatch::default()
                },
                None,
            )
            .await;
        assert!(matches!(err, Err(StorageError::Invalid(_))));
    }

    #[tokio::test]
    async fn parent_cannot_complete_with_open_children() {
        let (_s, repo) = repo().await;
        let parent = repo.create(new_task("parent")).await.unwrap();
        let mut nt = new_task("child");
        nt.parent_id = Some(parent.id);
        let child = repo.create(nt).await.unwrap();

        let err = repo.complete(parent.id, None, None).await;
        assert!(
            matches!(err, Err(StorageError::Invalid(_))),
            "open child must block parent completion"
        );

        repo.complete(child.id, None, None).await.unwrap();
        // Parent without acceptance auto-completed with its last child.
        let parent = repo.get(parent.id).await.unwrap();
        assert_eq!(parent.status, "done", "parent should auto-complete");
    }

    #[tokio::test]
    async fn parent_with_acceptance_does_not_auto_complete() {
        let (_s, repo) = repo().await;
        let mut np = new_task("parent");
        np.acceptance = Some(serde_json::json!({"kind": "file_exists", "path": "out.txt"}));
        let parent = repo.create(np).await.unwrap();
        let mut nc = new_task("child");
        nc.parent_id = Some(parent.id);
        let child = repo.create(nc).await.unwrap();

        repo.complete(child.id, None, None).await.unwrap();
        let parent = repo.get(parent.id).await.unwrap();
        assert_eq!(
            parent.status, "pending",
            "a parent with an acceptance check must be completed explicitly"
        );
    }

    #[tokio::test]
    async fn complete_is_idempotent() {
        let (_s, repo) = repo().await;
        let task = repo.create(new_task("t")).await.unwrap();
        let first = repo.complete(task.id, Some("agent"), None).await.unwrap();
        assert!(!first.already_done);
        let second = repo.complete(task.id, Some("agent"), None).await.unwrap();
        assert!(second.already_done);
    }

    #[tokio::test]
    async fn next_prefers_in_progress_then_priority_then_due() {
        let (_s, repo) = repo().await;
        let mut a = new_task("low prio");
        a.priority = 4;
        let a = repo.create(a).await.unwrap();
        let mut b = new_task("high prio");
        b.priority = 1;
        let b = repo.create(b).await.unwrap();

        let next = repo.next("session", Some("s1")).await.unwrap().unwrap();
        assert_eq!(next.id, b.id, "p1 beats p4");

        // An in_progress item beats a higher-priority pending one.
        repo.update(
            a.id,
            TaskPatch {
                status: Some("in_progress".to_string()),
                ..TaskPatch::default()
            },
            None,
        )
        .await
        .unwrap();
        let next = repo.next("session", Some("s1")).await.unwrap().unwrap();
        assert_eq!(next.id, a.id, "in_progress resumes before pending");
    }

    #[tokio::test]
    async fn next_skips_blocked_and_parents_with_open_children() {
        let (_s, repo) = repo().await;
        let parent = repo.create(new_task("parent")).await.unwrap();
        let mut nc = new_task("child");
        nc.parent_id = Some(parent.id);
        nc.priority = 2;
        let child = repo.create(nc).await.unwrap();
        let mut nb = new_task("blocked one");
        nb.priority = 1;
        nb.depends_on = vec![child.id];
        repo.create(nb).await.unwrap();

        let next = repo.next("session", Some("s1")).await.unwrap().unwrap();
        assert_eq!(
            next.id, child.id,
            "parent has open children and p1 item is blocked; the child is the one actionable item"
        );
    }

    /// `next_admitted` filters only the final choice: the order is `next`'s,
    /// an inadmissible item is skipped for the next admissible one, and an
    /// inadmissible child still holds its parent back.
    #[tokio::test]
    async fn next_admitted_skips_inadmissible_items_but_keeps_the_rules() {
        let (_s, repo) = repo().await;
        let mut first = new_task("leftover");
        first.priority = 1;
        let leftover = repo.create(first).await.unwrap();
        let fresh = repo.create(new_task("fresh")).await.unwrap();
        assert_eq!(
            repo.next("session", Some("s1")).await.unwrap().unwrap().id,
            leftover.id,
            "unfiltered, the higher-priority leftover wins"
        );
        let admitted = repo
            .next_admitted("session", Some("s1"), |t| t.id != leftover.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(admitted.id, fresh.id, "filtered, the next admissible item");

        let mut nc = new_task("child of fresh");
        nc.parent_id = Some(fresh.id);
        let child = repo.create(nc).await.unwrap();
        let none = repo
            .next_admitted("session", Some("s1"), |t| t.id == fresh.id)
            .await
            .unwrap();
        assert!(
            none.is_none(),
            "an inadmissible open child ({}) still holds its parent back",
            child.id
        );
    }

    #[tokio::test]
    async fn next_returns_none_when_everything_is_closed() {
        let (_s, repo) = repo().await;
        let t = repo.create(new_task("only")).await.unwrap();
        repo.complete(t.id, None, None).await.unwrap();
        assert!(repo.next("session", Some("s1")).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn cancelling_a_parent_cascades_to_open_descendants() {
        let (_s, repo) = repo().await;
        let parent = repo.create(new_task("parent")).await.unwrap();
        let mut nc = new_task("child");
        nc.parent_id = Some(parent.id);
        let child = repo.create(nc).await.unwrap();
        let mut ng = new_task("grandchild");
        ng.parent_id = Some(child.id);
        let grandchild = repo.create(ng).await.unwrap();

        repo.update(
            parent.id,
            TaskPatch {
                status: Some("cancelled".to_string()),
                ..TaskPatch::default()
            },
            None,
        )
        .await
        .unwrap();

        assert_eq!(repo.get(child.id).await.unwrap().status, "cancelled");
        assert_eq!(repo.get(grandchild.id).await.unwrap().status, "cancelled");
        assert!(repo.next("session", Some("s1")).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn cancelled_dependency_does_not_block() {
        let (_s, repo) = repo().await;
        let dep = repo.create(new_task("dep")).await.unwrap();
        let mut nt = new_task("dependent");
        nt.depends_on = vec![dep.id];
        let dependent = repo.create(nt).await.unwrap();
        repo.update(
            dep.id,
            TaskPatch {
                status: Some("cancelled".to_string()),
                ..TaskPatch::default()
            },
            None,
        )
        .await
        .unwrap();
        assert!(!repo.get(dependent.id).await.unwrap().blocked);
    }

    #[tokio::test]
    async fn notes_are_append_only_and_bounded() {
        let (_s, repo) = repo().await;
        let task = repo.create(new_task("t")).await.unwrap();
        repo.add_note(task.id, Some("agent-a"), "found the bug")
            .await
            .unwrap();
        repo.add_note(task.id, Some("agent-b"), "fixed in commit")
            .await
            .unwrap();
        let notes = repo.notes(task.id, 10).await.unwrap();
        assert_eq!(notes.len(), 2);
        assert_eq!(notes[0].content, "found the bug", "oldest first");

        assert!(matches!(
            repo.add_note(task.id, None, "  ").await,
            Err(StorageError::Invalid(_))
        ));
        let oversized = "x".repeat(TASK_NOTE_MAX_BYTES + 1);
        assert!(matches!(
            repo.add_note(task.id, None, &oversized).await,
            Err(StorageError::Invalid(_))
        ));
    }

    #[tokio::test]
    async fn query_filters_with_derived_blocked() {
        let (_s, repo) = repo().await;
        let dep = repo.create(new_task("dep")).await.unwrap();
        let mut nt = new_task("dependent");
        nt.depends_on = vec![dep.id];
        nt.priority = 1;
        repo.create(nt).await.unwrap();

        let blocked = repo.query("session", Some("s1"), "blocked").await.unwrap();
        assert_eq!(blocked.len(), 1);
        assert_eq!(blocked[0].title, "dependent");

        let actionable = repo
            .query("session", Some("s1"), "!blocked & !done")
            .await
            .unwrap();
        assert_eq!(actionable.len(), 1);
        assert_eq!(actionable[0].title, "dep");
    }

    #[tokio::test]
    async fn query_rejects_malformed_filters() {
        let (_s, repo) = repo().await;
        assert!(matches!(
            repo.query("session", Some("s1"), "banana &").await,
            Err(StorageError::Invalid(_))
        ));
    }

    #[tokio::test]
    async fn delete_removes_subtree_and_dangling_deps() {
        let (_s, repo) = repo().await;
        let parent = repo.create(new_task("parent")).await.unwrap();
        let mut nc = new_task("child");
        nc.parent_id = Some(parent.id);
        let child = repo.create(nc).await.unwrap();
        let mut no = new_task("outsider");
        no.depends_on = vec![child.id];
        let outsider = repo.create(no).await.unwrap();

        let deleted = repo.delete(parent.id, None).await.unwrap();
        assert_eq!(deleted, 2, "parent + child");
        assert!(repo.get(parent.id).await.is_err());
        assert!(repo.get(child.id).await.is_err());
        let outsider = repo.get(outsider.id).await.unwrap();
        assert!(
            outsider.depends_on.is_empty(),
            "dangling dependency must be stripped, got {:?}",
            outsider.depends_on
        );
        assert!(!outsider.blocked);
    }

    #[tokio::test]
    async fn clear_closed_keeps_open_tasks() {
        let (_s, repo) = repo().await;
        let done = repo.create(new_task("done one")).await.unwrap();
        repo.complete(done.id, None, None).await.unwrap();
        repo.create(new_task("open one")).await.unwrap();

        let removed = repo.clear("session", Some("s1"), true).await.unwrap();
        assert_eq!(removed, 1);
        let remaining = repo.list("session", Some("s1"), true).await.unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].title, "open one");
    }

    #[tokio::test]
    async fn import_v01_preserves_order_and_maps_statuses() {
        let (_s, repo) = repo().await;
        let items = vec![
            ("first".to_string(), "done".to_string()),
            ("second".to_string(), "in_progress".to_string()),
            ("third".to_string(), "blocked".to_string()),
            (String::new(), "pending".to_string()),
        ];
        let imported = repo.import_v01("s1", &items).await.unwrap();
        assert_eq!(imported, 3, "empty titles are skipped");

        let tasks = repo.list("session", Some("s1"), true).await.unwrap();
        assert_eq!(tasks.len(), 3);
        assert_eq!(tasks[0].title, "first");
        assert_eq!(tasks[0].status, "done");
        assert_eq!(tasks[1].status, "in_progress");
        assert_eq!(
            tasks[2].status, "pending",
            "v0.1 'blocked' becomes pending (blocked is derived now)"
        );
    }

    #[tokio::test]
    async fn acceptance_shape_is_validated_on_write() {
        let (_s, repo) = repo().await;
        let mut nt = new_task("bad acceptance");
        nt.acceptance = Some(serde_json::json!({"kind": "vibes"}));
        assert!(matches!(
            repo.create(nt).await,
            Err(StorageError::Invalid(_))
        ));

        let mut nt = new_task("regex needs target");
        nt.acceptance = Some(serde_json::json!({"kind": "regex", "pattern": "ok"}));
        assert!(matches!(
            repo.create(nt).await,
            Err(StorageError::Invalid(_))
        ));

        let mut nt = new_task("good");
        nt.acceptance = Some(serde_json::json!({
            "kind": "command", "command": "cargo test -p nanna-storage"
        }));
        assert!(repo.create(nt).await.is_ok());
    }

    /// The write boundary is where the acceptance representation is decided:
    /// whatever dialect a caller holds, the ROW holds an object. A store that
    /// merely validated (and rejected) the stringified form dropped the task
    /// entirely, because the reader had already admitted it upstream.
    #[tokio::test]
    async fn acceptance_is_canonicalized_to_an_object_at_write() {
        let (_s, repo) = repo().await;

        // The object handed over as a JSON string.
        let mut nt = new_task("stringified");
        nt.acceptance = Some(serde_json::json!(
            "{\"kind\":\"command\",\"command\":\"cargo test\"}"
        ));
        let created = repo.create(nt).await.expect("a stringified check is admitted");
        assert_eq!(
            created.acceptance.expect("stored"),
            serde_json::json!({"kind": "command", "command": "cargo test"}),
            "the row must hold the object, never the string"
        );

        // The JS-object-literal flavor, same rule.
        let mut nt = new_task("js literal");
        nt.acceptance = Some(serde_json::json!("{kind: 'file_exists', path: 'out.txt'}"));
        let created = repo.create(nt).await.expect("the literal flavor is repaired");
        assert_eq!(
            created.acceptance.expect("stored"),
            serde_json::json!({"kind": "file_exists", "path": "out.txt"})
        );

        // `update` obeys the same boundary.
        let updated = repo
            .update(
                created.id,
                TaskPatch {
                    acceptance: Some(Some(serde_json::json!(
                        "{\"kind\":\"regex\",\"pattern\":\"0 failed\",\"path\":\"build.log\"}"
                    ))),
                    ..Default::default()
                },
                None,
            )
            .await
            .expect("update canonicalizes too");
        let stored = updated.acceptance.expect("stored");
        assert!(stored.is_object(), "{stored}");
        assert_eq!(stored["kind"], serde_json::json!("regex"));
    }

    /// One unwrap, no more: a double-encoded payload is noise nobody wrote,
    /// and peeling deeper would accept it.
    #[tokio::test]
    async fn double_encoded_acceptance_is_still_rejected() {
        let (_s, repo) = repo().await;
        let mut nt = new_task("double encoded");
        nt.acceptance = Some(serde_json::json!(
            "\"{\\\"kind\\\":\\\"command\\\",\\\"command\\\":\\\"cargo test\\\"}\""
        ));
        let Err(StorageError::Invalid(message)) = repo.create(nt).await else {
            panic!("double-encoded acceptance must be rejected");
        };
        assert!(
            message.contains(r#"{"kind":"command","command":"cargo test"}"#),
            "the rejection must carry the shapes: {message}"
        );
    }

    /// The rejection has to carry the fix: a regex check missing its target
    /// was re-sent 11 times against the old "requires 'path' or 'command'"
    /// wording, which named the fields but never the shape.
    #[tokio::test]
    async fn targetless_regex_acceptance_names_both_valid_forms() {
        let (_s, repo) = repo().await;
        let mut nt = new_task("regex needs target");
        nt.acceptance = Some(serde_json::json!({"kind": "regex", "pattern": "0 failed"}));
        let Err(StorageError::Invalid(message)) = repo.create(nt).await else {
            panic!("a regex check with no target must be rejected");
        };
        assert!(
            message.contains(r#"{"kind":"regex","pattern":"0 failed","path":"build.log"}"#),
            "{message}"
        );
        assert!(
            message.contains(r#"{"kind":"regex","pattern":"0 failed","command":"cargo test"}"#),
            "{message}"
        );
    }

    #[tokio::test]
    async fn dependency_on_ancestor_is_rejected_on_create_and_update() {
        let (_s, repo) = repo().await;
        let grandparent = repo.create(new_task("grandparent")).await.unwrap();
        let mut np = new_task("parent");
        np.parent_id = Some(grandparent.id);
        let parent = repo.create(np).await.unwrap();

        // create: child depending on its direct parent wedges both forever
        let mut nc = new_task("child");
        nc.parent_id = Some(parent.id);
        nc.depends_on = vec![parent.id];
        assert!(
            matches!(repo.create(nc).await, Err(StorageError::Invalid(_))),
            "dep on direct parent must be rejected"
        );

        // create: dep on a grandparent is the same wedge, transitively
        let mut nc = new_task("child");
        nc.parent_id = Some(parent.id);
        nc.depends_on = vec![grandparent.id];
        assert!(matches!(
            repo.create(nc).await,
            Err(StorageError::Invalid(_))
        ));

        // update: adding the ancestor dep later must be rejected too
        let mut nc = new_task("child");
        nc.parent_id = Some(parent.id);
        let child = repo.create(nc).await.unwrap();
        let err = repo
            .update(
                child.id,
                TaskPatch {
                    depends_on: Some(vec![grandparent.id]),
                    ..TaskPatch::default()
                },
                None,
            )
            .await;
        assert!(matches!(err, Err(StorageError::Invalid(_))));
    }

    #[tokio::test]
    async fn reparent_under_closed_parent_is_rejected() {
        let (_s, repo) = repo().await;
        let done_parent = repo.create(new_task("done parent")).await.unwrap();
        repo.complete(done_parent.id, None, None).await.unwrap();
        let orphan = repo.create(new_task("orphan")).await.unwrap();
        let err = repo
            .update(
                orphan.id,
                TaskPatch {
                    parent_id: Some(Some(done_parent.id)),
                    ..TaskPatch::default()
                },
                None,
            )
            .await;
        assert!(
            matches!(err, Err(StorageError::Invalid(_))),
            "re-parenting under a done parent breaks the parent-done invariant"
        );
    }

    #[tokio::test]
    async fn reopen_via_update_clears_completed_at() {
        let (_s, repo) = repo().await;
        let task = repo.create(new_task("t")).await.unwrap();
        repo.complete(task.id, None, None).await.unwrap();
        assert!(repo.get(task.id).await.unwrap().completed_at.is_some());
        let reopened = repo
            .update(
                task.id,
                TaskPatch {
                    status: Some("pending".to_string()),
                    ..TaskPatch::default()
                },
                None,
            )
            .await
            .unwrap();
        assert_eq!(reopened.status, "pending");
        assert!(
            reopened.completed_at.is_none(),
            "a reopened task must not keep a completion stamp"
        );
    }

    #[tokio::test]
    async fn clear_closed_never_deletes_open_descendants() {
        let (_s, repo) = repo().await;
        let parent = repo.create(new_task("parent")).await.unwrap();
        let mut nc = new_task("child");
        nc.parent_id = Some(parent.id);
        let child = repo.create(nc).await.unwrap();
        // Close the subtree (parent auto-completes), then reopen the child.
        repo.complete(child.id, None, None).await.unwrap();
        repo.update(
            child.id,
            TaskPatch {
                status: Some("pending".to_string()),
                ..TaskPatch::default()
            },
            None,
        )
        .await
        .unwrap();

        let removed = repo.clear("session", Some("s1"), true).await.unwrap();
        assert_eq!(
            removed, 0,
            "closed parent with an open child must be skipped"
        );
        assert!(
            repo.get(child.id).await.is_ok(),
            "open child must survive clear"
        );
    }

    #[tokio::test]
    async fn import_v01_truncates_multibyte_titles_on_char_boundary() {
        let (_s, repo) = repo().await;
        // 200 x 4-byte chars = 800 bytes; byte 500 is mid-char.
        let long_title = "🌀".repeat(200);
        let items = vec![(long_title, "pending".to_string())];
        let imported = repo.import_v01("s1", &items).await.unwrap();
        assert_eq!(imported, 1);
        let tasks = repo.list("session", Some("s1"), true).await.unwrap();
        assert!(tasks[0].title.len() <= TASK_TITLE_MAX_BYTES);
        assert!(tasks[0].title.chars().all(|c| c == '🌀'));
    }

    #[tokio::test]
    async fn import_v01_is_idempotent() {
        let (_s, repo) = repo().await;
        let items = vec![("only".to_string(), "pending".to_string())];
        assert_eq!(repo.import_v01("s1", &items).await.unwrap(), 1);
        assert_eq!(
            repo.import_v01("s1", &items).await.unwrap(),
            0,
            "a second import must not duplicate"
        );
        assert_eq!(
            repo.list("session", Some("s1"), true).await.unwrap().len(),
            1
        );
    }

    #[tokio::test]
    async fn acceptance_timeout_type_is_validated() {
        let (_s, repo) = repo().await;
        let mut nt = new_task("bad timeout");
        nt.acceptance = Some(serde_json::json!({
            "kind": "command", "command": "exit 0", "timeout_secs": "five"
        }));
        assert!(matches!(
            repo.create(nt).await,
            Err(StorageError::Invalid(_))
        ));
    }

    #[tokio::test]
    async fn acceptance_timeout_integral_float_is_coerced() {
        // Every number through the JS tool bridge is an f64: `60` arrives as
        // `60.0`. The write boundary admits it as the integer it denotes —
        // rejecting it bounced 50 valid adds in one observed run.
        let (_s, repo) = repo().await;
        let mut nt = new_task("float timeout");
        nt.acceptance = Some(serde_json::json!({
            "kind": "command", "command": "exit 0", "timeout_secs": 60.0
        }));
        let created = repo.create(nt).await.unwrap();
        let stored = created.acceptance.expect("acceptance persisted");
        assert_eq!(
            stored.get("timeout_secs").and_then(serde_json::Value::as_u64),
            Some(60),
            "integral float canonicalizes to the integer it denotes: {stored}"
        );

        // A fractional timeout is NOT integral — no guessing, still rejected.
        let mut frac = new_task("fractional timeout");
        frac.acceptance = Some(serde_json::json!({
            "kind": "command", "command": "exit 0", "timeout_secs": 60.5
        }));
        assert!(matches!(
            repo.create(frac).await,
            Err(StorageError::Invalid(_))
        ));

        // Negative stays rejected too: -1.0 denotes no valid duration.
        let mut neg = new_task("negative timeout");
        neg.acceptance = Some(serde_json::json!({
            "kind": "command", "command": "exit 0", "timeout_secs": -1.0
        }));
        assert!(matches!(
            repo.create(neg).await,
            Err(StorageError::Invalid(_))
        ));
    }

    #[test]
    fn exact_u64_admits_only_exact_non_negative_integers_below_2_pow_53() {
        // Admitted: every whole number below 2^53, including both zeros and
        // the neighbours of the powers of two where the exponent changes.
        for (f, whole) in [
            (0.0, 0),
            (-0.0, 0),
            (1.0, 1),
            (60.0, 60),
            (1_023.0, 1_023),
            (1_024.0, 1_024),
            (4_503_599_627_370_495.0, 4_503_599_627_370_495),
            (4_503_599_627_370_496.0, 4_503_599_627_370_496),
            (4_503_599_627_370_497.0, 4_503_599_627_370_497),
            (9_007_199_254_740_991.0, 9_007_199_254_740_991),
        ] {
            assert_eq!(exact_u64_below_2_pow_53(f), Some(whole), "{f:?}");
        }
        for n in [2, 3, 255, 65_535, 1 << 31, u32::MAX] {
            assert_eq!(exact_u64_below_2_pow_53(f64::from(n)), Some(u64::from(n)), "{n}");
        }

        // Refused: fractions, negatives, non-finite values, and 2^53 upward.
        for f in [
            0.5,
            1.5,
            60.5,
            4_503_599_627_370_495.5,
            5e-324,
            f64::MIN_POSITIVE,
            -1.0,
            -60.0,
            f64::NAN,
            -f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            9_007_199_254_740_992.0,
            9_007_199_254_740_994.0,
            1e300,
            f64::MAX,
        ] {
            assert_eq!(exact_u64_below_2_pow_53(f), None, "{f:?}");
        }
    }

    #[tokio::test]
    async fn counts_reports_open_and_closed() {
        let (_s, repo) = repo().await;
        let a = repo.create(new_task("a")).await.unwrap();
        repo.create(new_task("b")).await.unwrap();
        repo.complete(a.id, None, None).await.unwrap();
        let (open, closed) = repo.counts("session", Some("s1")).await.unwrap();
        assert_eq!((open, closed), (1, 1));
    }

    // -----------------------------------------------------------------
    // Acceptance evidence inputs (P23)
    // -----------------------------------------------------------------

    #[test]
    fn referenced_paths_finds_the_files_a_check_names() {
        let check = serde_json::json!({"kind": "command", "command": "sh tests/test_01.sh"});
        assert_eq!(
            acceptance_referenced_paths(&check),
            vec!["tests/test_01.sh".to_string()]
        );
        let file = serde_json::json!({"kind": "file_exists", "path": "build/minidb.exe"});
        assert_eq!(
            acceptance_referenced_paths(&file),
            vec!["build/minidb.exe".to_string()]
        );
        // Flags and bare words are not path referents — a false referent
        // costs a stat, but a false PROHIBITION would block real work.
        let noisy =
            serde_json::json!({"kind": "command", "command": "cargo test --all -p nanna"});
        assert_eq!(acceptance_referenced_paths(&noisy), [] as [std::string::String; 0]);
    }

    /// The instrument/subject split. A `file_exists` (or path-`regex`) check
    /// observes its subject directly: the named file is the DELIVERABLE, and
    /// producing it is the work — treating that as evidence tampering would
    /// make every first completion suspicious. Only a command check has an
    /// instrument that can be edited out from under the verdict.
    #[test]
    fn only_a_command_check_has_evidence_inputs() {
        let file = serde_json::json!({"kind": "file_exists", "path": "artifact.txt"});
        assert_eq!(acceptance_evidence_paths(&file), [] as [std::string::String; 0]);
        let path_regex =
            serde_json::json!({"kind": "regex", "pattern": "PASS", "path": "report.txt"});
        assert_eq!(acceptance_evidence_paths(&path_regex), [] as [std::string::String; 0]);
        assert_eq!(
            acceptance_referenced_paths(&path_regex),
            vec!["report.txt".to_string()],
            "the subject still carries artifact identity onto the completion record"
        );

        let command = serde_json::json!({"kind": "command", "command": "sh tests/test_01.sh"});
        assert_eq!(
            acceptance_evidence_paths(&command),
            vec!["tests/test_01.sh".to_string()]
        );
        let command_regex = serde_json::json!({
            "kind": "regex", "pattern": "0 failed", "command": "sh tests/run_all.sh"
        });
        assert_eq!(
            acceptance_evidence_paths(&command_regex),
            vec!["tests/run_all.sh".to_string()]
        );
    }

    #[test]
    fn path_referents_reject_prose_that_merely_contains_a_dot() {
        assert!(!looks_like_path("e.g."));
        assert!(!looks_like_path("1.2.3"));
        assert!(!looks_like_path("tests"));
        assert!(!looks_like_path("https://example.com/x.sh"));
        assert!(looks_like_path("tests/"));
        assert!(looks_like_path("tests/**"));
        assert!(looks_like_path("src/main.rs"));
    }

    // -----------------------------------------------------------------
    // User-declared file invariants (P23)
    // -----------------------------------------------------------------

    #[test]
    fn declared_invariants_register_the_imperative_with_a_path() {
        let found = extract_declared_invariants(
            "Fix the failing behaviour in ./minidb. Never create, edit or delete \
             anything under tests/.",
            "session",
        );
        let kinds: Vec<&str> = found.iter().map(|i| i.kind.as_str()).collect();
        assert!(kinds.contains(&"read_only"), "{found:?}");
        assert!(kinds.contains(&"no_delete"), "{found:?}");
        assert!(kinds.contains(&"no_create_under"), "{found:?}");
        assert!(found.iter().all(|i| i.glob == "tests"), "{found:?}");
        assert!(
            found
                .iter()
                .all(|i| i.source == "Never create, edit or delete anything under tests/"),
            "the user's own sentence must be quotable verbatim: {found:?}"
        );
        // The other sentence names a path but states no prohibition.
        assert!(found.iter().all(|i| i.glob != "minidb"), "{found:?}");
    }

    #[test]
    fn declared_invariants_register_nothing_when_the_phrasing_is_ambiguous() {
        // Conditional, hedged, permissive, interrogative, and path-free —
        // every one registers nothing, because the tool layer fails open and a
        // missed constraint costs only today's behavior.
        for text in [
            "If the test is wrong, do not edit tests/run.sh",
            "Try not to edit tests/run.sh",
            "You can edit tests/run.sh, but do not delete it",
            "Should I avoid editing tests/run.sh?",
            "Never touch the tests",
            "Never again with this task",
        ] {
            assert!(
                extract_declared_invariants(text, "session").is_empty(),
                "must register nothing: {text}"
            );
        }
    }

    #[test]
    fn declared_invariants_accumulate_and_stay_idempotent() {
        let first = extract_declared_invariants("tests/ is read-only.", "session");
        assert_eq!(first.len(), 1, "{first:?}");
        let doc = merge_declared_invariants("", &first).expect("first declaration writes");
        assert!(doc.contains("\"version\": 1"), "{doc}");
        assert!(doc.contains("read_only"), "{doc}");
        // Re-declaring the same constraint changes nothing — a turn that adds
        // no new rule must not rewrite the registry.
        assert!(merge_declared_invariants(&doc, &first).is_none());
        // A NEW constraint folds in beside the old one; nothing is dropped.
        let more = extract_declared_invariants("Never delete docs/plan.md.", "session");
        let merged = merge_declared_invariants(&doc, &more).expect("a new rule writes");
        assert!(merged.contains("tests"), "{merged}");
        assert!(merged.contains("docs/plan.md"), "{merged}");
        // A corrupted registry is treated as empty, never as a wedge.
        let healed = merge_declared_invariants("{not json", &first).expect("heals");
        assert!(healed.contains("read_only"), "{healed}");
    }
}
