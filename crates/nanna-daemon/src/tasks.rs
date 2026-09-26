//! Daemon-side task-store integration (P15) and long-horizon run manager (P14).
//!
//! Three pieces live here:
//! - `build_task_services` — the `tasks.*` script services the `todo` skill
//!   calls via `Nanna.service(...)`. The store is the daemon's Turso DB; the
//!   JS tool never touches the filesystem for task state again.
//! - `TursoTaskSource` / `AgentStepRunner` — the production implementations of
//!   the harness traits: the P15 repository as [`nanna_agent::harness::TaskSource`],
//!   and a fresh `Agent` + empty context per step as
//!   [`nanna_agent::harness::StepRunner`] (the re-anchor: parent state never
//!   accumulates in a transcript).
//! - `TaskRunManager` — starts/cancels/reports background long-horizon runs
//!   and broadcasts their lifecycle as events. The task store itself is the
//!   checkpoint: resuming after a crash is just starting a run in the same
//!   scope again.

use crate::llm_router::LlmRouter;
use crate::protocol::Event;
use nanna_agent::harness::{
    AcceptanceCheck, Interjector, LongHorizonConfig, LongHorizonReport, LongHorizonRunner,
    StepOutcome, StepRequest, StepRunner, StepToolCall, StopReason, TaskSource, TaskStep,
    touched_path_of,
};
use nanna_agent::planner::{Plan, build_plan_prompt, plan_or_fallback};
use nanna_agent::CancelToken;
use nanna_scripting::ServiceFn;
use nanna_storage::{NewTask, Storage, StorageError, Task, TaskPatch};
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

/// Notes injected into a step prompt.
///
/// Bound justification: each note is already capped at 16 KiB by the store,
/// but a prompt tail only has room for a handful of findings — 5 recent notes
/// keeps the O(1) prompt O(1).
const STEP_NOTES_TAIL: i64 = 5;

/// How many malformed-acceptance tasks `next()` will cancel before giving up.
///
/// Bound justification: each skip cancels one task (the open set strictly
/// shrinks), so this only trips when a scope is saturated with corrupt rows —
/// at that point stopping loudly beats grinding through thousands.
const TASK_NEXT_SKIP_MAX: usize = 100;

// ---------------------------------------------------------------------------
// Scope resolution
// ---------------------------------------------------------------------------

/// Resolve `(scope, scope_id)` from service params + the active workspace.
///
/// A call from inside a session never reaches the missing-id error: the
/// scripting bridge fills `session_id` from the session the tool is running
/// in. Only a caller with no session context at all gets here, so the message
/// names both ways out instead of just restating the requirement.
async fn resolve_scope(
    params: &Value,
    workspace_id: &Arc<RwLock<Option<String>>>,
) -> Result<(String, Option<String>), String> {
    let scope = params
        .get("scope")
        .and_then(Value::as_str)
        .unwrap_or("session")
        .to_lowercase();
    match scope.as_str() {
        "session" => {
            let session_id = params
                .get("session_id")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    "session scope needs a session id and this caller has none — pass \
                     session_id explicitly, or use scope 'workspace' (the active workspace) \
                     or scope 'global'"
                        .to_string()
                })?;
            Ok(("session".to_string(), Some(session_id.to_string())))
        }
        "workspace" => {
            let ws = workspace_id.read().await.clone();
            let ws =
                ws.ok_or_else(|| "workspace scope requires an active workspace".to_string())?;
            Ok(("workspace".to_string(), Some(ws)))
        }
        "global" => Ok(("global".to_string(), None)),
        other => Err(format!("unknown scope '{other}'")),
    }
}

fn task_to_json(task: &Task) -> Value {
    json!({
        "id": task.id,
        "parent_id": task.parent_id,
        "scope": task.scope,
        "project": task.project,
        "title": task.title,
        "description": task.description,
        "status": task.status,
        "blocked": task.blocked,
        "priority": task.priority,
        "labels": task.labels,
        "tools": task.tool_scope,
        "due_at": task.due_at,
        "deadline_at": task.deadline_at,
        "recurrence": task.recurrence,
        "depends_on": task.depends_on,
        "acceptance": task.acceptance,
        "assignee": task.assignee,
        "created_at": task.created_at,
        "completed_at": task.completed_at,
    })
}

// ---------------------------------------------------------------------------
// Param dialect readers
// ---------------------------------------------------------------------------
//
// Every param here arrives as JSON written by a model and carried across the
// Boa bridge. Models routinely stringify scalars and nested JSON — `"2109"`
// for an id, `"[2108]"` for a dependency list — which is the same dialect
// PR #174 taught `acceptance` to read. Two rules hold for every reader below:
//
// 1. A value that can be read LOSSLESSLY is read, whatever shape it wore.
//    Anything lossy or ambiguous is refused rather than guessed at: guessing
//    writes a row nobody asked for.
// 2. A value that cannot be read is an ERROR that names what arrived. It is
//    never dropped. Silently discarding `depends_on` (observed live
//    2026-08-04: `"[2108]"` vanished, and `"2109"` was answered with "id is
//    required") left the model believing it had declared a dependency the
//    store never recorded — the failure mode that makes a dialect bug
//    invisible.
//
// Absent and JSON null both mean "not supplied": the Boa bridge serializes
// `undefined` object members as null, so null must keep meaning "not
// mentioned" or every partial update would wipe the fields it omitted.

/// How much of an offending value an error message quotes back.
///
/// Bound justification: the message exists to identify WHICH value was
/// rejected, not to reproduce it — a tool result is read back into the
/// model's context, so echoing an arbitrarily large payload into an error
/// would make the error the thing that floods the window. 48 chars quotes any
/// plausible id, label, status, or short filter in full.
const PARAM_PREVIEW_CHARS: usize = 48;

/// Largest magnitude an `f64` represents with integer precision (2^53).
///
/// Bound justification: this is a property of the representation, not a
/// policy. At or below 2^53 every integer has a distinct `f64`; above it,
/// adjacent integers share a bit pattern, so "the fraction is zero" stops
/// proving the value is the integer the caller meant.
const F64_EXACT_INT_MAX: f64 = 9_007_199_254_740_992.0;

/// Name the type (and, for scalars, the value) that actually arrived.
///
/// Shared with the `memory.*` services in `server.rs` so both script-service
/// surfaces answer a mistyped param in one vocabulary.
pub(crate) fn describe_value(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(b) => format!("boolean {b}"),
        Value::Number(n) => format!("number {n}"),
        Value::String(s) => {
            let truncated = s.chars().count() > PARAM_PREVIEW_CHARS;
            let mut preview: String = s.chars().take(PARAM_PREVIEW_CHARS).collect();
            if truncated {
                preview.push('…');
            }
            format!("string \"{preview}\"")
        }
        Value::Array(items) => format!("an array of {} item(s)", items.len()),
        Value::Object(_) => "an object".to_string(),
    }
}

/// Convert a whole `f64` to `i64`, or `None` when the conversion would be a
/// guess. See [`F64_EXACT_INT_MAX`] for why the magnitude is bounded.
// The conversion is exact by construction: fract() == 0.0 and |f| <= 2^53
// put the value well inside i64.
fn whole_f64_to_i64(f: f64) -> Option<i64> {
    (f.fract() == 0.0 && f.abs() <= F64_EXACT_INT_MAX).then_some(crate::numeric::i64_from_f64(f))
}

/// Read an i64 from the dialects that actually reach this service.
///
/// Accepted, each because it is unambiguous:
/// - a JSON integer — the canonical shape;
/// - a JSON number with no fractional part, within [`F64_EXACT_INT_MAX`] —
///   Boa numbers cross the bridge as `f64` once arithmetic touches them, so
///   `3` can arrive as `3.0`;
/// - a JSON string whose TRIMMED text parses as an integer — models stringify
///   scalars constantly (`"2109"`). Parsed as `i64` first so the full i64
///   range survives without an `f64` round trip; `"2109.0"` then falls back to
///   the whole-number rule above.
///
/// Refused (`None`) because reading them would be a guess, not a coercion:
/// empty or whitespace-only text, non-numeric text, any fractional value
/// (`1.5`, `"1.5"`), a magnitude no longer exactly representable, and
/// booleans — JS `true` is not 1 on this surface.
fn as_i64_lenient(value: &Value) -> Option<i64> {
    match value {
        Value::Number(_) => value
            .as_i64()
            .or_else(|| value.as_f64().and_then(whole_f64_to_i64)),
        Value::String(text) => {
            let text = text.trim();
            if text.is_empty() {
                return None;
            }
            text.parse::<i64>()
                .ok()
                .or_else(|| text.parse::<f64>().ok().and_then(whole_f64_to_i64))
        }
        _ => None,
    }
}

/// Read a bool from the dialects that actually reach this service: the JSON
/// literal, or the literal spelled out as text (`"true"`, `" False "`).
///
/// `1`/`0`/`"yes"`/`""` are refused on purpose — mapping them is a guess about
/// intent, and this surface would rather say what it saw than pick a side.
fn as_bool_lenient(value: &Value) -> Option<bool> {
    match value {
        Value::Bool(b) => Some(*b),
        Value::String(text) => match text.trim().to_ascii_lowercase().as_str() {
            "true" => Some(true),
            "false" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

/// Unwrap a list that may have arrived as a JSON array or as a STRING holding
/// one (`"[2108]"`).
///
/// Exactly ONE unwrap, mirroring PR #174's rule for stringified acceptance: a
/// doubly-encoded payload (`"\"[1]\""`) is a bug worth surfacing, not one to
/// peel until something array-shaped falls out. Coercing the ELEMENTS is a
/// separate step — that is what turns `["2108"]` into `[2108]`.
fn unwrap_array(value: &Value) -> Option<Vec<Value>> {
    match value {
        Value::Array(items) => Some(items.clone()),
        Value::String(text) => match serde_json::from_str::<Value>(text.trim()) {
            Ok(Value::Array(items)) => Some(items),
            _ => None,
        },
        _ => None,
    }
}

/// `Ok(None)` = not supplied (absent or null). `Ok(Some(_))` = read.
/// `Err(_)` = supplied in a shape this service cannot read.
type ParamResult<T> = Result<Option<T>, String>;

/// Read an optional integer param, keeping MISSING and UNINTERPRETABLE apart.
pub(crate) fn opt_i64(params: &Value, key: &str) -> ParamResult<i64> {
    match params.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => as_i64_lenient(value).map(Some).ok_or_else(|| {
            format!(
                "{key} must be an integer (got {}). Pass a number, e.g. {{\"{key}\": 42}} — the \
                 digits as a string (\"42\") are accepted too.",
                describe_value(value)
            )
        }),
    }
}

/// Read a required integer param.
///
/// Missing and uninterpretable get DIFFERENT answers on purpose: replying
/// "id is required" to a supplied `"2109"` is a false statement, and the model
/// that read it re-sent the same call three times before giving up (observed
/// live 2026-08-04). `id` of `0` is supplied, not missing.
fn req_i64(params: &Value, key: &str) -> Result<i64, String> {
    opt_i64(params, key)?.ok_or_else(|| format!("{key} is required"))
}

/// Read an optional bool param, keeping MISSING and UNINTERPRETABLE apart.
///
/// An unreadable value is an error rather than a fall-through to the default:
/// a default silently substituted for `"include_done": "maybe"` answers a
/// question the caller did not ask.
fn opt_bool(params: &Value, key: &str) -> ParamResult<bool> {
    match params.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => as_bool_lenient(value).map(Some).ok_or_else(|| {
            format!(
                "{key} must be true or false (got {}).",
                describe_value(value)
            )
        }),
    }
}

/// Read an optional list-of-ids param (`depends_on`).
///
/// An unreadable list ERRORS instead of resolving to an empty vec: an empty
/// `depends_on` is a real, meaningful value ("nothing blocks this"), so
/// quietly substituting it for a list the caller did state is how a declared
/// dependency disappears without a trace.
fn opt_i64_vec(params: &Value, key: &str) -> ParamResult<Vec<i64>> {
    let Some(value) = params.get(key).filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    let items = unwrap_array(value).ok_or_else(|| {
        format!(
            "{key} must be a list of task ids (got {}). Pass an array, e.g. \
             {{\"{key}\": [12, 34]}} — the array as a JSON string (\"[12, 34]\") is accepted too.",
            describe_value(value)
        )
    })?;
    items
        .iter()
        .enumerate()
        .map(|(index, item)| {
            as_i64_lenient(item).ok_or_else(|| {
                format!(
                    "{key}[{index}] must be a task id (got {}). Every entry has to be an \
                     integer, e.g. {{\"{key}\": [12, 34]}}.",
                    describe_value(item)
                )
            })
        })
        .collect::<Result<Vec<i64>, String>>()
        .map(Some)
}

/// Read an optional list-of-strings param (`labels`, `tools`).
///
/// Scalar elements are rendered as their text (`123` -> `"123"`) because that
/// is lossless; a nested array or object is not a label by any reading and
/// errors. Same no-silent-empty rule as [`opt_i64_vec`].
fn opt_string_vec(params: &Value, key: &str) -> ParamResult<Vec<String>> {
    let Some(value) = params.get(key).filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    let items = unwrap_array(value).ok_or_else(|| {
        format!(
            "{key} must be a list of strings (got {}). Pass an array, e.g. \
             {{\"{key}\": [\"one\", \"two\"]}} — the array as a JSON string \
             (\"[\\\"one\\\"]\") is accepted too.",
            describe_value(value)
        )
    })?;
    items
        .iter()
        .enumerate()
        .map(|(index, item)| match item {
            Value::String(s) => Ok(s.clone()),
            Value::Number(n) => Ok(n.to_string()),
            Value::Bool(b) => Ok(b.to_string()),
            other => Err(format!(
                "{key}[{index}] must be text (got {}). Every entry has to be a string, e.g. \
                 {{\"{key}\": [\"one\", \"two\"]}}.",
                describe_value(other)
            )),
        })
        .collect::<Result<Vec<String>, String>>()
        .map(Some)
}

fn opt_string(params: &Value, key: &str) -> Option<String> {
    params
        .get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

/// Canonicalize an acceptance payload through the shared entry point so a
/// shape that would fail at run time is rejected at write time instead of
/// wedging every future run in the scope — and so what is persisted is the
/// canonical object, identical to what the planner admits and what `create`
/// stores.
fn canonical_acceptance(params: &Value) -> Result<Option<Value>, String> {
    params
        .get("acceptance")
        .filter(|v| !v.is_null())
        .map_or(Ok(None), |raw| AcceptanceCheck::canonicalize(raw).map(Some))
}

// ---------------------------------------------------------------------------
// tasks.* script services
// ---------------------------------------------------------------------------

/// Planning depth at which creating a task earns a do-the-work note.
///
/// Depth 0 is a root item, 1 a subtask, 2 a subtask of a subtask. The two
/// endurance datapoints split cleanly on this line: qwen3.5:9b's baseline runs
/// created only depth-1 subtasks (9 over six hours, 14/42 → 32/42 verified),
/// while gemma4:e4b-it-qat recursed to depth 3 ("Subtask 1 of #115", #115
/// itself a subtask), created 103 items against 42 seeded, and verified 7 —
/// it never advanced past feature ~10 because the window went to planning.
/// A NOTE, never a refusal: the owner's directive is no hard caps, and the
/// escalating-soft-nudge family is the loop's containment idiom.
const LADDER_NUDGE_DEPTH: usize = 2;

/// Open-subtask count under one parent at which the (N+1)th add earns a
/// finish-what-you-started note. qwen's whole six-hour run created nine
/// subtasks TOTAL; a parent holding four open ones at once is planning
/// outrunning work.
const SIBLING_NUDGE_AT: usize = 4;

/// How deep this new task would sit in the ladder: 0 for a root item, 1 for
/// a child of a root, and so on.
///
/// Bounded and cycle-safe: the walk stops at 6 hops (deeper than anything a
/// sane plan holds, and the nudge wording maxes out long before) and on any
/// id it has already seen — a corrupted parent cycle must degrade to a capped
/// depth, not an infinite loop on the model's write path.
async fn ladder_depth(storage: &Arc<Storage>, parent_id: Option<i64>) -> usize {
    let mut depth = 0usize;
    let mut cursor = parent_id;
    let mut seen = std::collections::HashSet::new();
    while let Some(id) = cursor {
        if depth >= 6 || !seen.insert(id) {
            break;
        }
        depth += 1;
        cursor = match storage.tasks().get(id).await {
            Ok(parent) => parent.parent_id,
            // An unreadable parent ends the walk at the depth proven so far.
            Err(_) => None,
        };
    }
    depth
}

/// The decomposition-damping note for a task created at `depth` whose parent
/// already had `open_siblings` open children, or `None` when neither signal
/// fires. Pure so the wording and thresholds are testable without storage.
fn decomposition_note(depth: usize, open_siblings: usize, parent_id: Option<i64>) -> Option<String> {
    let mut parts: Vec<String> = Vec::new();
    if depth > LADDER_NUDGE_DEPTH {
        parts.push(format!(
            "STOP SPLITTING: this item sits at planning depth {depth} (a subtask of a subtask              of a subtask). Every level of decomposition here has produced more planning, not              more working code. Execute this item's work in your very next step."
        ));
    } else if depth >= LADDER_NUDGE_DEPTH {
        parts.push(format!(
            "Note: this is a subtask of a subtask (planning depth {depth}). Planning is not              progress — in your next step, DO the work for this item directly instead of              splitting it again."
        ));
    }
    if let (Some(parent), true) = (parent_id, open_siblings >= SIBLING_NUDGE_AT) {
        parts.push(format!(
            "Parent task #{parent} now has {} open subtasks. Finish and verify some of them              before adding more.",
            open_siblings + 1
        ));
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" "))
    }
}

/// Build the `tasks.*` services the todo skill calls. Registered in
/// `build_script_services` when storage is available.
pub fn build_task_services(
    storage: Arc<Storage>,
    workspace_id: Arc<RwLock<Option<String>>>,
    turn_baselines: Arc<TurnBaselines>,
) -> HashMap<String, ServiceFn> {
    let mut services: HashMap<String, ServiceFn> = HashMap::new();

    services.insert("tasks.next".to_string(), task_next_service(&storage, &workspace_id));
    services.insert(
        "tasks.add".to_string(),
        task_add_service(&storage, &workspace_id, turn_baselines),
    );
    services.insert("tasks.update".to_string(), task_update_service(&storage));
    services.insert("tasks.done".to_string(), task_done_service(&storage));
    services.insert("tasks.list".to_string(), task_list_service(&storage, &workspace_id));
    services.insert("tasks.query".to_string(), task_query_service(&storage, &workspace_id));
    services.insert("tasks.note".to_string(), task_note_service(&storage));
    services.insert("tasks.remove".to_string(), task_remove_service(&storage));
    services.insert("tasks.clear".to_string(), task_clear_service(&storage, &workspace_id));
    services.insert("tasks.counts".to_string(), task_counts_service(&storage, workspace_id));
    services.insert("tasks.import".to_string(), task_import_service(storage));

    services
}

// tasks.next {scope?, session_id?}
fn task_next_service(
    storage: &Arc<Storage>,
    workspace_id: &Arc<RwLock<Option<String>>>,
) -> ServiceFn {
    let err_str = |e: StorageError| e.to_string();
    let storage = storage.clone();
    let workspace_id = workspace_id.clone();
    Arc::new(move |params: Value| {
        let storage = storage.clone();
        let workspace_id = workspace_id.clone();
        Box::pin(async move {
            let (scope, scope_id) = resolve_scope(&params, &workspace_id).await?;
            let next = storage
                .tasks()
                .next(&scope, scope_id.as_deref())
                .await
                .map_err(err_str)?;
            match next {
                Some(task) => {
                    let notes = storage
                        .tasks()
                        .notes(task.id, STEP_NOTES_TAIL)
                        .await
                        .map_err(err_str)?;
                    let mut value = task_to_json(&task);
                    value["notes"] =
                        json!(notes.iter().map(|n| n.content.clone()).collect::<Vec<_>>());
                    Ok(json!({ "task": value }))
                }
                None => Ok(json!({ "task": Value::Null })),
            }
        })
    })
}

// tasks.add {title, scope?, session_id?, parent_id?, priority?, labels?,
//            tools?, due_at?, deadline_at?, recurrence?, depends_on?,
//            acceptance?, project?, assignee?, description?}
//
// `due_at` defers the card, `deadline_at` bounds it (P25 decision 10).
fn task_add_service(
    storage: &Arc<Storage>,
    workspace_id: &Arc<RwLock<Option<String>>>,
    turn_baselines: Arc<TurnBaselines>,
) -> ServiceFn {
    let err_str = |e: StorageError| e.to_string();
    let storage = storage.clone();
    let workspace_id = workspace_id.clone();
    Arc::new(move |params: Value| {
        let storage = storage.clone();
        let workspace_id = workspace_id.clone();
        let turn_baselines = turn_baselines.clone();
        Box::pin(async move {
            // A subtask always lives in its parent's scope — replan
            // steps only know the parent id, not the run's scope.
            let parent_id = opt_i64(&params, "parent_id")?;
            let (scope, scope_id, parent_sort, parent_acceptance) =
                if let Some(parent_id) = parent_id {
                    let parent = storage.tasks().get(parent_id).await.map_err(err_str)?;
                    (
                        parent.scope,
                        parent.scope_id,
                        Some(parent.sort_order),
                        parent.acceptance,
                    )
                } else {
                    let (scope, scope_id) = resolve_scope(&params, &workspace_id).await?;
                    (scope, scope_id, None, None)
                };
            let title = opt_string(&params, "title")
                .or_else(|| opt_string(&params, "text"))
                .ok_or_else(|| "title is required".to_string())?;

            // Idempotent add: re-adding a title that is ALREADY open
            // in this scope returns the existing item instead of a
            // second copy. Observed live (lfm2.5 smoke): 5 seeded
            // tasks became ~50 as the model re-decomposed the same
            // work every step — "Write data file with header and 3
            // rows" was created ten times — so the plan grew faster
            // than it was worked and the real items drowned. Only
            // OPEN items dedupe: a genuinely recurring chore
            // ("run the tests") must still be addable after the
            // previous one is closed, and reopening history would be
            // the worse failure.
            let open_items = storage
                .tasks()
                .list(&scope, scope_id.as_deref(), false)
                .await
                .map_err(err_str)?;
            if let Some(existing) = open_items
                .iter()
                .find(|t| t.parent_id == parent_id && same_title(&t.title, &title))
            {
                // The model asked for this work: if it is an item an earlier
                // turn left open, the live turn must now admit it, or the
                // model is told its task exists while nothing schedules it.
                if let Some(admission) = turn_baselines.admission(&scope, scope_id.as_deref()).await
                {
                    admission.adopt_id(existing.id);
                }
                return Ok(json!({
                    "task": task_to_json(existing),
                    "deduplicated": true,
                    // Consistent with the closed-this-turn guard so
                    // callers have ONE flag for "nothing was created"
                    // — the todo skill suppresses its "Added task"
                    // confirmation on exactly this field.
                    "created": false,
                    "note": format!(
                        "Task #{} \"{}\" already exists and is still open — reusing it \
                         instead of creating a duplicate. Work on it rather than \
                         planning it again.",
                        existing.id, existing.title
                    ),
                }));
            }

            // Closed-since-turn-start guard. The open-title reuse
            // above deliberately lets a CLOSED title be added again —
            // a recurring chore must stay addable tomorrow — but
            // "tomorrow" is the point: within ONE turn, a title the
            // run already finished or abandoned is settled work, and
            // re-creating it is how a turn spins. Observed live
            // 2026-08-02 (session 05775d1d): the harness's replan step
            // drives this very service through the todo tool, and
            // task #2059 (abandoned) came straight back as #2060 with
            // the identical title, immediately re-seeded.
            //
            // The baseline is the same one `seed_continuation` uses
            // and the comparison is the same `same_title` rule, so
            // both doors now answer "is this the same task?"
            // identically. No live turn → no baseline → unchanged
            // behavior for adds from outside a run.
            if let Some(reply) = closed_this_turn_reply(
                &storage,
                &turn_baselines,
                &scope,
                scope_id.as_deref(),
                &title,
            )
            .await?
            {
                return Ok(reply);
            }

            // Ordering: a subtask inherits its parent's ladder
            // position; a new root task appends AFTER everything
            // (defaulting to 0 would jump the whole queue — observed
            // live as a task explosion drowning the seeded plan).
            let sort_order =
                add_sort_order(&storage, &params, parent_sort, &scope, scope_id.as_deref())
                    .await?;
            let new = NewTask {
                parent_id,
                scope,
                scope_id,
                project: opt_string(&params, "project"),
                title,
                description: opt_string(&params, "description"),
                // Floored at 2: priority 1 is the user speaking (an
                // interjection jumps the queue at p1), and this
                // service is the MODEL's write surface — a
                // self-created todo must never outrank user-origin
                // work (observed live 2026-08-02: a self-created p1
                // side-task preempted the user's p2 request).
                priority: opt_i64(&params, "priority")?.unwrap_or(3).max(2),
                labels: opt_string_vec(&params, "labels")?.unwrap_or_default(),
                tool_scope: opt_string_vec(&params, "tools")?.unwrap_or_default(),
                due_at: opt_string(&params, "due_at"),
                deadline_at: opt_string(&params, "deadline_at"),
                recurrence: opt_string(&params, "recurrence"),
                depends_on: opt_i64_vec(&params, "depends_on")?.unwrap_or_default(),
                // Acceptance inheritance: a subtask that declares no
                // check of its own answers to its parent's — the same
                // write-time principle as scope inheritance. Without
                // it, self-created children are self-graded: closing
                // them looks like progress to model and scheduler
                // alike while the parent's real check fails
                // byte-identically (observed 2026-08-07/08: 97 items
                // marked done against 6 features actually verified —
                // the whole gap was acceptance-less scaffolding).
                // "Done is a verdict" only binds where a verdict
                // exists; this gives every child one by default.
                acceptance: canonical_acceptance(&params)?.or(parent_acceptance),
                assignee: opt_string(&params, "assignee"),
                sort_order,
            };
            // Decomposition damping — a returned note, never a
            // refusal. The item is created regardless; the note rides
            // the tool result, which is exactly where the model is
            // looking at the moment it chose planning over working.
            let open_siblings = parent_id
                .map_or(0, |pid| {
                    open_items
                        .iter()
                        .filter(|t| t.parent_id == Some(pid))
                        .count()
                });
            let depth = ladder_depth(&storage, parent_id).await;
            let note = decomposition_note(depth, open_siblings, parent_id);

            let task = storage.tasks().create(new).await.map_err(err_str)?;
            Ok(note.map_or_else(
                || json!({ "task": task_to_json(&task) }),
                |note| json!({ "task": task_to_json(&task), "note": note }),
            ))
        })
    })
}

/// The reply `tasks.add` returns instead of re-creating `title`, when the live
/// turn in this scope already closed a task with the same title. `None` when
/// there is no live turn or no closed title matches.
async fn closed_this_turn_reply(
    storage: &Arc<Storage>,
    turn_baselines: &TurnBaselines,
    scope: &str,
    scope_id: Option<&str>,
    title: &str,
) -> Result<Option<Value>, String> {
    let Some(baseline) = turn_baselines.baseline(scope, scope_id).await else {
        return Ok(None);
    };
    let closed = tasks_closed_since(storage, scope, scope_id, &baseline)
        .await?
        .into_iter()
        .find(|t| same_title(&t.title, title));
    let Some(closed) = closed else {
        return Ok(None);
    };
    tracing::info!(
        task_id = closed.id,
        status = %closed.status,
        title = %title,
        "refusing to re-create a title this turn already closed"
    );
    // A returned notice, never an error: the model
    // reads it as a normal result and moves on.
    Ok(Some(json!({
        "task": task_to_json(&closed),
        "created": false,
        "closed_this_turn": true,
        "note": closed_this_turn_notice(&closed.title, &closed.status),
    })))
}

/// The ladder position for a task `tasks.add` creates: the explicit
/// `sort_order` parameter, else the parent's position, else one past every
/// task in the scope.
async fn add_sort_order(
    storage: &Arc<Storage>,
    params: &Value,
    parent_sort: Option<i64>,
    scope: &str,
    scope_id: Option<&str>,
) -> Result<i64, String> {
    Ok(match opt_i64(params, "sort_order")? {
        Some(explicit) => explicit,
        None => match parent_sort {
            Some(parent_sort) => parent_sort,
            None => storage
                .tasks()
                .list(scope, scope_id, true)
                .await
                .map_err(|e| e.to_string())?
                .iter()
                .map(|t| t.sort_order)
                .max()
                .unwrap_or(0)
                .saturating_add(1),
        },
    })
}

// tasks.update {id, ...patch}
fn task_update_service(storage: &Arc<Storage>) -> ServiceFn {
    let err_str = |e: StorageError| e.to_string();
    let storage = storage.clone();
    Arc::new(move |params: Value| {
        let storage = storage.clone();
        Box::pin(async move {
            let id = req_i64(&params, "id")?;
            // Null/absent values SKIP a field, never wipe it: the Boa
            // bridge serializes `undefined` object members as null, so
            // a partial update from the tool must not clear every
            // field it did not mention. (Clearing a field is a
            // deliberate op this service intentionally does not
            // expose.) A value that is PRESENT but unreadable is a
            // different thing entirely and errors — skipping it left
            // the model believing it had set a field the store never
            // saw.
            let patch = TaskPatch {
                title: opt_string(&params, "title").or_else(|| opt_string(&params, "text")),
                description: params
                    .get("description")
                    .and_then(Value::as_str)
                    .map(|s| Some(s.to_string())),
                status: opt_string(&params, "status"),
                // Same floor as tasks.add: the model cannot promote
                // its own work to p1 after the fact either.
                priority: opt_i64(&params, "priority")?.map(|p| p.max(2)),
                labels: opt_string_vec(&params, "labels")?,
                tool_scope: opt_string_vec(&params, "tools")?,
                due_at: params
                    .get("due_at")
                    .and_then(Value::as_str)
                    .map(|s| Some(s.to_string())),
                deadline_at: params
                    .get("deadline_at")
                    .and_then(Value::as_str)
                    .map(|s| Some(s.to_string())),
                recurrence: params
                    .get("recurrence")
                    .and_then(Value::as_str)
                    .map(|s| Some(s.to_string())),
                depends_on: opt_i64_vec(&params, "depends_on")?,
                acceptance: canonical_acceptance(&params)?.map(Some),
                assignee: params
                    .get("assignee")
                    .and_then(Value::as_str)
                    .map(|s| Some(s.to_string())),
                parent_id: opt_i64(&params, "parent_id")?.map(Some),
                project: params
                    .get("project")
                    .and_then(Value::as_str)
                    .map(|s| Some(s.to_string())),
                sort_order: opt_i64(&params, "sort_order")?,
            };
            let actor = opt_string(&params, "actor");
            let task = storage
                .tasks()
                .update(id, patch, actor.as_deref())
                .await
                .map_err(err_str)?;
            Ok(json!({ "task": task_to_json(&task) }))
        })
    })
}

// tasks.done {id, actor?, workdir?} — runs the acceptance check first:
// done is a verdict, not an assertion (the P14 anti-drift keystone).
fn task_done_service(storage: &Arc<Storage>) -> ServiceFn {
    let err_str = |e: StorageError| e.to_string();
    let storage = storage.clone();
    Arc::new(move |params: Value| {
        let storage = storage.clone();
        Box::pin(async move {
            let id = req_i64(&params, "id")?;
            let actor = opt_string(&params, "actor");
            let task = storage.tasks().get(id).await.map_err(err_str)?;

            let (verified, verdict_detail) = if let Some(acceptance) = &task.acceptance {
                let check = AcceptanceCheck::from_json(acceptance)?;
                let workdir = opt_string(&params, "workdir")
                    .map_or_else(|| PathBuf::from("."), PathBuf::from);
                let verdict = check.run(&workdir).await;
                storage
                    .tasks()
                    .log_activity(
                        id,
                        actor.as_deref(),
                        "acceptance_checked",
                        Some(json!({
                            "passed": verdict.passed,
                            "detail": verdict.detail,
                        })),
                    )
                    .await
                    .map_err(err_str)?;
                if !verdict.passed {
                    return Ok(json!({
                        "done": false,
                        "verdict": verdict.detail,
                        "message": format!(
                            "Acceptance check failed — task #{id} is NOT done: {}",
                            verdict.detail
                        ),
                    }));
                }
                (true, json!(verdict.detail))
            } else {
                (false, Value::Null)
            };

            let outcome = storage
                .tasks()
                .complete(
                    id,
                    actor.as_deref(),
                    Some(json!({ "verified": verified, "verdict": verdict_detail })),
                )
                .await
                .map_err(err_str)?;
            // Parent refocus — the generalizable form of "notice when
            // the goal is already satisfied". The model finishes a
            // subtask and is pointed back at the goal it serves; if it
            // believes the parent is done it CLAIMS so, and the
            // ordinary verify-on-claim path runs the check. The model
            // stays in the loop; nothing is auto-scored. In chat the
            // same reflex is returning to the user's request after a
            // sub-errand.
            let refocus = match task.parent_id {
                Some(pid) => match storage.tasks().get(pid).await {
                    Ok(parent)
                        if parent.status != "done" && parent.status != "cancelled" =>
                    {
                        Some(format!(
                            "This was a subtask of #{} '{}'. If that goal is now \
                             satisfied, mark it done — its acceptance check will \
                             verify. Do not split it further.",
                            parent.id, parent.title
                        ))
                    }
                    _ => None,
                },
                None => None,
            };
            Ok(json!({
                "done": true,
                "verified": verified,
                "refocus": refocus,
                "already_done": outcome.already_done,
                "auto_completed": outcome.auto_completed,
            }))
        })
    })
}

// tasks.list {scope?, session_id?, include_done?}
fn task_list_service(
    storage: &Arc<Storage>,
    workspace_id: &Arc<RwLock<Option<String>>>,
) -> ServiceFn {
    let err_str = |e: StorageError| e.to_string();
    let storage = storage.clone();
    let workspace_id = workspace_id.clone();
    Arc::new(move |params: Value| {
        let storage = storage.clone();
        let workspace_id = workspace_id.clone();
        Box::pin(async move {
            let (scope, scope_id) = resolve_scope(&params, &workspace_id).await?;
            let include_done = opt_bool(&params, "include_done")?.unwrap_or(true);
            let tasks = storage
                .tasks()
                .list(&scope, scope_id.as_deref(), include_done)
                .await
                .map_err(err_str)?;
            Ok(json!({ "tasks": tasks.iter().map(task_to_json).collect::<Vec<_>>() }))
        })
    })
}

// tasks.query {filter, scope?, session_id?}
fn task_query_service(
    storage: &Arc<Storage>,
    workspace_id: &Arc<RwLock<Option<String>>>,
) -> ServiceFn {
    let err_str = |e: StorageError| e.to_string();
    let storage = storage.clone();
    let workspace_id = workspace_id.clone();
    Arc::new(move |params: Value| {
        let storage = storage.clone();
        let workspace_id = workspace_id.clone();
        Box::pin(async move {
            let (scope, scope_id) = resolve_scope(&params, &workspace_id).await?;
            let filter = opt_string(&params, "filter")
                .ok_or_else(|| "filter is required".to_string())?;
            let tasks = storage
                .tasks()
                .query(&scope, scope_id.as_deref(), &filter)
                .await
                .map_err(err_str)?;
            Ok(json!({ "tasks": tasks.iter().map(task_to_json).collect::<Vec<_>>() }))
        })
    })
}

// tasks.note {id, content, author?}
fn task_note_service(storage: &Arc<Storage>) -> ServiceFn {
    let err_str = |e: StorageError| e.to_string();
    let storage = storage.clone();
    Arc::new(move |params: Value| {
        let storage = storage.clone();
        Box::pin(async move {
            let id = req_i64(&params, "id")?;
            let content = opt_string(&params, "content")
                .or_else(|| opt_string(&params, "text"))
                .ok_or_else(|| "content is required".to_string())?;
            let author = opt_string(&params, "author");
            let author_member_id = opt_string(&params, "author_member_id");
            // An unknown kind is refused rather than silently posted as a
            // comment: a caller that meant `verdict` must not have it read as
            // prose.
            let kind = match opt_string(&params, "kind") {
                Some(token) => nanna_storage::TaskNoteKind::parse(&token).ok_or_else(|| {
                    format!("unknown note kind '{token}' (comment|progress|question|verdict)")
                })?,
                None => nanna_storage::TaskNoteKind::Comment,
            };
            let note = storage
                .tasks()
                .post(
                    id,
                    author.as_deref(),
                    author_member_id.as_deref(),
                    kind,
                    &content,
                )
                .await
                .map_err(err_str)?;
            Ok(json!({ "note_id": note.id }))
        })
    })
}

// tasks.remove {id}
fn task_remove_service(storage: &Arc<Storage>) -> ServiceFn {
    let err_str = |e: StorageError| e.to_string();
    let storage = storage.clone();
    Arc::new(move |params: Value| {
        let storage = storage.clone();
        Box::pin(async move {
            let id = req_i64(&params, "id")?;
            let actor = opt_string(&params, "actor");

            // A task carrying a machine-checkable acceptance check is
            // a CONTRACT — someone (the planner, the eval, the user)
            // defined what "done" means for it, and deleting it
            // destroys the only objective record of the goal. Erasing
            // it is never the right move: the honest outcomes are
            // finish it or cancel it, both of which keep the item and
            // its history. Observed live (lfm2.5 smoke): blocked on a
            // refusing write_file and holding only `write_file` and
            // `todo`, the model deleted a SEEDED plan item — the run
            // then died on a task the harness still expected to
            // verify. Scratch items the model invents carry no
            // acceptance and stay freely removable.
            let existing = storage.tasks().get(id).await.map_err(err_str)?;
            if existing.acceptance.is_some() {
                return Ok(json!({
                    "removed": false,
                    "refused": true,
                    "note": format!(
                        "Task #{} \"{}\" has a machine-checkable acceptance check, so it is \
                         a commitment rather than scratch work and was NOT deleted — it is \
                         fully intact. If the work is finished, complete it (the check runs \
                         automatically). If it should not be done at all, set its status to \
                         cancelled. Either way the record survives.",
                        existing.id, existing.title
                    ),
                }));
            }

            let removed = storage
                .tasks()
                .delete(id, actor.as_deref())
                .await
                .map_err(err_str)?;
            Ok(json!({ "removed": removed }))
        })
    })
}

// tasks.clear {scope?, session_id?, closed_only?}
fn task_clear_service(
    storage: &Arc<Storage>,
    workspace_id: &Arc<RwLock<Option<String>>>,
) -> ServiceFn {
    let err_str = |e: StorageError| e.to_string();
    let storage = storage.clone();
    let workspace_id = workspace_id.clone();
    Arc::new(move |params: Value| {
        let storage = storage.clone();
        let workspace_id = workspace_id.clone();
        Box::pin(async move {
            let (scope, scope_id) = resolve_scope(&params, &workspace_id).await?;
            let closed_only = opt_bool(&params, "closed_only")?.unwrap_or(true);

            // Same contract rule as tasks.remove, applied in bulk —
            // and this is the path that actually bites. Observed live
            // (lfm2.5 endurance, 2026-07-25): guarding only the
            // per-id remove left `clear` wide open, and one call took
            // the scope from 42 tasks to 6 mid-run, destroying 36
            // seeded features the harness was still driving.
            //
            // Ancestors of a contract are protected too: `delete`
            // removes whole SUBTREES, so clearing a scratch parent
            // would take a contract-bearing child down with it.
            let all = storage
                .tasks()
                .list(&scope, scope_id.as_deref(), true)
                .await
                .map_err(err_str)?;
            let parents: HashMap<i64, Option<i64>> =
                all.iter().map(|t| (t.id, t.parent_id)).collect();
            let mut protected: std::collections::HashSet<i64> =
                std::collections::HashSet::new();
            for task in all.iter().filter(|t| t.acceptance.is_some()) {
                let mut cursor = Some(task.id);
                while let Some(id) = cursor {
                    if !protected.insert(id) {
                        break; // this ancestor chain is already marked
                    }
                    cursor = parents.get(&id).copied().flatten();
                }
            }

            let mut removed = 0u64;
            for task in &all {
                if protected.contains(&task.id) {
                    continue;
                }
                if closed_only && task.status != "done" && task.status != "cancelled" {
                    continue;
                }
                // A subtree delete may already have taken this id;
                // that is success, not an error.
                if let Ok(count) = storage.tasks().delete(task.id, None).await {
                    removed += count;
                }
            }

            let kept = protected.len();
            Ok(json!({
                "removed": removed,
                "protected": kept,
                "note": if kept > 0 {
                    format!(
                        "Cleared {removed} scratch task(s). {kept} task(s) carrying an \
                         acceptance contract were KEPT — they define what \"done\" means \
                         and are still intact. Complete or cancel those instead."
                    )
                } else {
                    format!("Cleared {removed} task(s).")
                },
            }))
        })
    })
}

// tasks.counts {scope?, session_id?}
fn task_counts_service(
    storage: &Arc<Storage>,
    workspace_id: Arc<RwLock<Option<String>>>,
) -> ServiceFn {
    let err_str = |e: StorageError| e.to_string();
    let storage = storage.clone();
    Arc::new(move |params: Value| {
        let storage = storage.clone();
        let workspace_id = workspace_id.clone();
        Box::pin(async move {
            let (scope, scope_id) = resolve_scope(&params, &workspace_id).await?;
            let (open, closed) = storage
                .tasks()
                .counts(&scope, scope_id.as_deref())
                .await
                .map_err(err_str)?;
            Ok(json!({ "open": open, "closed": closed }))
        })
    })
}

// tasks.import {session_id, items: [{text, status}]} — v0.1 JSON migration
fn task_import_service(storage: Arc<Storage>) -> ServiceFn {
    let err_str = |e: StorageError| e.to_string();
    Arc::new(move |params: Value| {
        let storage = storage.clone();
        Box::pin(async move {
            let session_id = opt_string(&params, "session_id")
                .ok_or_else(|| "session_id is required".to_string())?;
            let items: Vec<(String, String)> = params
                .get("items")
                .and_then(Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .map(|item| {
                            (
                                item.get("text")
                                    .and_then(Value::as_str)
                                    .unwrap_or("")
                                    .to_string(),
                                item.get("status")
                                    .and_then(Value::as_str)
                                    .unwrap_or("pending")
                                    .to_string(),
                            )
                        })
                        .collect()
                })
                .unwrap_or_default();
            let imported = storage
                .tasks()
                .import_v01(&session_id, &items)
                .await
                .map_err(err_str)?;
            info!(session_id = %session_id, imported, "Migrated v0.1 todo JSON into task store");
            Ok(json!({ "imported": imported }))
        })
    })
}

// ---------------------------------------------------------------------------
// Harness trait implementations
// ---------------------------------------------------------------------------

/// The P15 store as the harness's task source, scoped to one run.
///
/// It carries no `workdir`. It used to, for "where ancestor acceptance checks
/// run during promotion" — but the ancestor-promotion experiment was cut as
/// benchmark-shaped (see the note on [`Self::complete`]: it converted the eval
/// metric directly and would almost never fire in a chat workflow). The field
/// outlived the feature, was never read, and was always `None`, so it was a
/// standing invitation to "finish wiring it" — which the no-benchmark-shaped-
/// levers rule forbids. Removed 2026-08-26; per-call acceptance checks still
/// take their own `workdir` from `tasks.done {workdir?}`, which is a different
/// and live mechanism.
pub struct TursoTaskSource {
    storage: Arc<Storage>,
    scope: String,
    scope_id: Option<String>,
    actor: String,
    event_tx: Option<tokio::sync::broadcast::Sender<Event>>,
    /// Which open items this source may hand out (see [`TurnAdmission`]).
    /// `None` serves the whole scope — the task-run manager, and a chat turn
    /// that IS a resume.
    admission: Option<Arc<TurnAdmission>>,
}

/// Which open items a chat turn may work.
///
/// Owner directive (2026-07-25, quoted at `run_mission`): *"the model should
/// decide to resume or answer another question by the user … i don't think
/// we should assume that the user wants to resume."* The planner is shown
/// outstanding work so it can choose — but the task source used to serve
/// every open item in the scope, so an item an earlier turn left open (a
/// question the user then STOPPED) was worked anyway on their next, unrelated
/// message. Measured: `first` → Stop → `second` answered both.
///
/// So a turn admits everything created during it (its seeds, interjections,
/// replan subtasks, continuation rounds — none of which exist yet when this
/// is built) and, of the items already open when it began, only those the
/// planner re-adopted by proposing the same work again ([`same_title`], the
/// store's one definition of "the same task").
#[derive(Debug, Default)]
pub struct TurnAdmission {
    /// Open when the turn began: id → title.
    leftovers: HashMap<i64, String>,
    adopted: std::sync::Mutex<HashSet<i64>>,
}

impl TurnAdmission {
    /// Snapshot the scope's open items as this turn's leftovers.
    ///
    /// # Errors
    /// Returns an error if the scope's tasks cannot be listed.
    pub async fn at_turn_start(
        storage: &Storage,
        scope: &str,
        scope_id: Option<&str>,
    ) -> Result<Self, String> {
        let open = storage
            .tasks()
            .list(scope, scope_id, false)
            .await
            .map_err(|e| e.to_string())?;
        let leftovers: HashMap<i64, String> =
            open.into_iter().map(|task| (task.id, task.title)).collect();
        Ok(Self {
            leftovers,
            adopted: std::sync::Mutex::new(HashSet::new()),
        })
    }

    /// Whether this turn may work task `id`.
    #[must_use]
    pub fn admits(&self, id: i64) -> bool {
        !self.leftovers.contains_key(&id)
            || self
                .adopted
                .lock()
                .is_ok_and(|adopted| adopted.contains(&id))
    }

    /// Admit leftover `id` into this turn — the model asked for that work
    /// again through `tasks.add`, which reused the open item instead of
    /// creating a duplicate. A no-op for an id that is not a leftover.
    pub fn adopt_id(&self, id: i64) {
        if self.leftovers.contains_key(&id)
            && let Ok(mut adopted) = self.adopted.lock()
        {
            adopted.insert(id);
        }
        debug_assert!(self.admits(id), "an adopted or new id is admitted");
    }

    /// Re-adopt the leftovers the plan proposes again, removing those tasks
    /// from the plan (they already exist) and returning the adopted ids in
    /// plan order. A leftover is adopted at most once.
    pub fn adopt_from(&self, plan: &mut Plan) -> Vec<i64> {
        let mut ids = Vec::new();
        plan.tasks.retain(|task| {
            let matched = self
                .leftovers
                .iter()
                .find(|(id, title)| !ids.contains(*id) && same_title(title, &task.title));
            match matched {
                Some((id, _)) => {
                    ids.push(*id);
                    false
                }
                None => true,
            }
        });
        if let Ok(mut adopted) = self.adopted.lock() {
            adopted.extend(ids.iter().copied());
        }
        debug_assert!(
            ids.iter().all(|id| self.admits(*id)),
            "adopted means admitted"
        );
        ids
    }
}

impl TursoTaskSource {
    #[must_use]
    pub const fn new(
        storage: Arc<Storage>,
        scope: String,
        scope_id: Option<String>,
        actor: String,
        event_tx: Option<tokio::sync::broadcast::Sender<Event>>,
    ) -> Self {
        Self {
            storage,
            scope,
            scope_id,
            actor,
            event_tx,
            admission: None,
        }
    }

    /// Serve only what `admission` admits (see [`TurnAdmission`]).
    #[must_use]
    pub fn with_admission(mut self, admission: Option<Arc<TurnAdmission>>) -> Self {
        self.admission = admission;
        self
    }

    fn emit(&self, task_id: i64, kind: &str, detail: Value) {
        if let Some(tx) = &self.event_tx {
            let _ = tx.send(Event::TaskRunProgress {
                scope: self.scope.clone(),
                scope_id: self.scope_id.clone(),
                task_id: Some(task_id),
                kind: kind.to_string(),
                detail,
            });
        }
    }
}

impl TursoTaskSource {
    /// Cancel every open descendant of a task proven done.
    ///
    /// Fired only from VERIFIED completions: the parent's acceptance check ran
    /// and passed, so its open subtasks describe work toward a goal that is
    /// already satisfied. Leaving them open sends later steps back into the
    /// tree (observed across three gemma4:e4b endurance runs, where subtask
    /// scaffolding outlived the code it was planning).
    async fn clear_open_descendants(&self, parent_id: i64) {
        let repo = self.storage.tasks();
        let Ok(open_items) = repo.list(&self.scope, self.scope_id.as_deref(), false).await else {
            return;
        };
        let mut frontier = vec![parent_id];
        let mut cleared = 0usize;
        while let Some(pid) = frontier.pop() {
            for child in open_items.iter().filter(|t| t.parent_id == Some(pid)) {
                frontier.push(child.id);
                let _ = repo
                    .update(
                        child.id,
                        TaskPatch { status: Some("cancelled".to_string()), ..TaskPatch::default() },
                        Some(&self.actor),
                    )
                    .await;
                let _ = repo
                    .log_activity(
                        child.id,
                        Some(&self.actor),
                        "scaffolding_cleared",
                        Some(json!({ "parent": parent_id, "reason": "parent verified done" })),
                    )
                    .await;
                cleared += 1;
            }
        }
        if cleared > 0 {
            tracing::info!(parent = parent_id, cleared, "cleared open scaffolding under a verified-done task");
        }
    }
}

#[async_trait::async_trait]
impl TaskSource for TursoTaskSource {
    async fn next(&self) -> Result<Option<TaskStep>, String> {
        let repo = self.storage.tasks();
        // Write-time canonicalization should make malformed acceptance JSON
        // impossible, but legacy or hand-edited rows must not wedge the run:
        // close them visibly and move on. Bounded: every malformed item is
        // cancelled, strictly shrinking the open set.
        for _ in 0..TASK_NEXT_SKIP_MAX {
            let admission = self.admission.as_deref();
            let task = repo
                .next_admitted(&self.scope, self.scope_id.as_deref(), |task| {
                    admission.is_none_or(|admission| admission.admits(task.id))
                })
                .await
                .map_err(|e| e.to_string())?;
            let Some(task) = task else { return Ok(None) };
            let acceptance = match &task.acceptance {
                Some(value) => match AcceptanceCheck::from_json(value) {
                    Ok(check) => Some(check),
                    Err(e) => {
                        let _ = repo
                            .log_activity(
                                task.id,
                                Some(&self.actor),
                                "acceptance_invalid",
                                Some(json!({ "error": e })),
                            )
                            .await;
                        let _ = repo
                            .update(
                                task.id,
                                TaskPatch {
                                    status: Some("cancelled".to_string()),
                                    ..TaskPatch::default()
                                },
                                Some(&self.actor),
                            )
                            .await;
                        self.emit(
                            task.id,
                            "abandoned",
                            json!({ "reason": format!("invalid acceptance check: {e}") }),
                        );
                        continue;
                    }
                },
                None => None,
            };
            let notes = repo
                .notes(task.id, STEP_NOTES_TAIL)
                .await
                .map_err(|e| e.to_string())?;
            return Ok(Some(TaskStep {
                id: task.id,
                title: task.title,
                description: task.description,
                acceptance,
                tool_scope: task.tool_scope,
                notes_tail: notes.into_iter().map(|n| n.content).collect(),
            }));
        }
        Err(format!(
            "gave up after cancelling {TASK_NEXT_SKIP_MAX} tasks with malformed acceptance checks"
        ))
    }

    async fn start(&self, id: i64) -> Result<(), String> {
        let repo = self.storage.tasks();
        let task = repo.get(id).await.map_err(|e| e.to_string())?;
        if task.status == "pending" {
            repo.update(
                id,
                TaskPatch {
                    status: Some("in_progress".to_string()),
                    ..TaskPatch::default()
                },
                Some(&self.actor),
            )
            .await
            .map_err(|e| e.to_string())?;
            self.emit(id, "started", Value::Null);
        }
        Ok(())
    }

    async fn complete(&self, id: i64, detail: Value) -> Result<(), String> {
        self.storage
            .tasks()
            .complete(id, Some(&self.actor), Some(detail.clone()))
            .await
            .map_err(|e| e.to_string())?;
        self.emit(id, "completed", detail.clone());
        // A VERIFIED-done task cancels its own open children. Not a scoring
        // mechanism — the ancestor-promotion experiment that auto-probed and
        // auto-scored parent checks was cut as benchmark-shaped (it converted
        // the eval metric directly and would almost never fire in a chat
        // workflow, where tasks rarely carry machine checks). Its last residue
        // was a `workdir: Option<PathBuf>` on this struct, hard-coded to `None`
        // by the only constructor and read by nothing — a dead-code warning
        // whose doc comment nonetheless read as though promotion existed and
        // was merely switched off. Removed; if promotion is ever wanted back it
        // needs a deliberate design, not a field waiting to be filled in. This
        // is pure
        // coherence: a parent proven done with scaffolding still open is a
        // contradictory state, and working those children is spending steps on
        // a goal that no longer exists.
        if detail.get("verified").and_then(Value::as_bool) == Some(true) {
            self.clear_open_descendants(id).await;
        }
        Ok(())
    }

    async fn add_note(&self, id: i64, content: &str) -> Result<(), String> {
        self.storage
            .tasks()
            .add_note(id, Some(&self.actor), content)
            .await
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    async fn log(&self, id: i64, action: &str, detail: Value) -> Result<(), String> {
        self.storage
            .tasks()
            .log_activity(id, Some(&self.actor), action, Some(detail.clone()))
            .await
            .map_err(|e| e.to_string())?;
        self.emit(id, action, detail);
        Ok(())
    }

    async fn abandon(&self, id: i64, reason: &str) -> Result<(), String> {
        let repo = self.storage.tasks();
        repo.update(
            id,
            TaskPatch {
                status: Some("cancelled".to_string()),
                ..TaskPatch::default()
            },
            Some(&self.actor),
        )
        .await
        .map_err(|e| e.to_string())?;
        repo.log_activity(
            id,
            Some(&self.actor),
            "abandoned",
            Some(json!({ "reason": reason })),
        )
        .await
        .map_err(|e| e.to_string())?;
        self.emit(id, "abandoned", json!({ "reason": reason }));
        Ok(())
    }

    async fn reopen(&self, id: i64, reason: &str) -> Result<(), String> {
        // The environment's verdict changed after the item closed (a
        // verified completion regressed, or an abandoned item's check now
        // passes) — the store records what is true NOW. Same patch door as
        // every other status change, so scope/cascade invariants hold.
        let repo = self.storage.tasks();
        repo.update(
            id,
            TaskPatch {
                status: Some("pending".to_string()),
                ..TaskPatch::default()
            },
            Some(&self.actor),
        )
        .await
        .map_err(|e| e.to_string())?;
        repo.log_activity(
            id,
            Some(&self.actor),
            "reopened",
            Some(json!({ "reason": reason })),
        )
        .await
        .map_err(|e| e.to_string())?;
        self.emit(id, "reopened", json!({ "reason": reason }));
        Ok(())
    }

    /// Open children of `id` in this scope — the runner's evidence that a
    /// replan step actually decomposed something. Counted from the scope's
    /// open list rather than a dedicated query because that is the same view
    /// `next()` selects from: work that exists but can never be surfaced is
    /// not work the replan produced.
    async fn open_subtasks(&self, id: i64) -> Result<Option<usize>, String> {
        Ok(Some(
            self.storage
                .tasks()
                .list(&self.scope, self.scope_id.as_deref(), false)
                .await
                .map_err(|e| e.to_string())?
                .into_iter()
                .filter(|t| t.parent_id == Some(id))
                .count(),
        ))
    }
}

/// Runs one harness step as a fresh `Agent` with an isolated context — the
/// re-anchor.
#[derive(Clone)]
pub struct AgentStepRunner {
    /// Tools the model has discovered so far in THIS harness run.
    ///
    /// Shared across steps so `discover_tools` is paid once per tool rather
    /// than once per step — the runner is one object for the whole run, while
    /// each step gets a fresh `RunState`.
    pub discovered_tools: Arc<tokio::sync::RwLock<std::collections::HashSet<String>>>,
    /// Tools the user manually added to this chat's context (per-session
    /// selection, stored on the session and threaded through `ChatTurnPrep`).
    ///
    /// Strictly additive: unioned into every step's active set AFTER the core
    /// set and BEFORE run-discovered tools, deduplicated. Empty = no manual
    /// selection, which leaves the active set byte-identical to the
    /// pre-feature behavior (core + discovered only). Unknown names cost
    /// nothing — the registry simply has nothing to activate for them.
    pub user_selected_tools: Vec<String>,
    /// The sibling breakers' repeat-call ledger for THIS run.
    ///
    /// Shared across steps for exactly the reason `discovered_tools` above
    /// is: the runner is one object for the whole run, while each step gets a
    /// fresh `RunState`. A per-step ledger reset the streak counters 22 times
    /// in the 20-minute wedged turn of 2026-08-02 (session 05775d1d), so the
    /// K = 3 thresholds were never reachable — the model made 2-3 identical
    /// calls per step, forever.
    pub repeat_ledger: nanna_agent::SharedRepeatLedger,
    pub router: Arc<LlmRouter>,
    pub tools: Arc<nanna_tools::ToolRegistry>,
    pub agent_config: nanna_agent::AgentConfig,
    pub system_prompt: String,
    pub workspace_root: Option<PathBuf>,
    /// Workspace files rendered as background reference (README/AGENTS/…).
    ///
    /// Carried SEPARATELY from `system_prompt` so each step routes it through
    /// `AgentContext::workspace_context`, where the model-window-derived cap
    /// (`workspace_context_cap_chars`) and its truncation marker apply.
    /// Inlined into `system_prompt` it was unbounded: a long ROADMAP.md
    /// dominated a 16k window and read as a work order (observed live
    /// 2026-08-02 — a factual question answered with a roadmap status report).
    pub workspace_context: Option<String>,
    pub stats: Option<nanna_agent::ModelStatsTracker>,
    /// Memory sink for tool results.
    ///
    /// The design is "a tool result goes to MEMORY, and only a stub goes to
    /// context — recall is how you get the full thing back". `loop_runner`
    /// implements exactly that, but only when `RunOptions::on_memory` is set,
    /// and it was set on the OLD direct-chat path only. When the harness
    /// became the single chat path this runner replaced that one without
    /// carrying the sink across, so the branch silently no-opped: a 4-hour
    /// run made 1315 exec + 501 `read_file` + 87 `edit_file` calls and stored
    /// NOTHING (observed 2026-07-27). Wiring it back restores the intended
    /// behaviour for every turn, since every turn is now a harness run.
    pub memory: Option<Arc<nanna_memory::MemoryService>>,
    /// Workspace scope for stored memories, so a run's observations belong to
    /// the workspace they happened in rather than leaking global.
    pub workspace_id: Option<String>,
    /// When set, every step streams its text and tool activity into the
    /// session's chat transcript ("show your work as you go"). None for
    /// background runs that have no transcript to show.
    pub chat_sink: Option<ChatSink>,
    /// GPU-memory faults observed in THIS harness run, across all its steps.
    ///
    /// Drives the blip-tolerant demotion ladder (`gpu_fault_action`): the
    /// FIRST fault of a run gets a runner reset alone (reload at the same
    /// size), the SECOND and every later one demote the context as well.
    /// Runners serving the same run (the chat harness's planner and step
    /// runners) share one counter, so the two faults that prove a repeat do
    /// not have to land on the same runner object.
    pub gpu_fault_count: Arc<std::sync::atomic::AtomicU32>,
    /// Capability-transition notices (P22 Tier 4), shared with the daemon's
    /// provider plumbing. Each pending transition reaches the model once, in
    /// the next tool result — see [`nanna_agent::DegradationLedger`].
    pub degradations: Option<Arc<nanna_agent::DegradationLedger>>,
    /// Images the user attached to the message this run answers, as
    /// `(base64_data, media_type)`. Sent with EVERY step: each step starts
    /// from a fresh context, so a later step working on the image would
    /// otherwise not see it. Empty for runs no message started.
    pub attachments: Arc<Vec<(String, String)>>,
    /// The models this run may fall back to, in `[llm] model_priority`
    /// order, shared by every runner of the run (planner and steps).
    ///
    /// `None` = no fallback: the run lives and dies on
    /// `agent_config.model` (a chat pinned to its own model — substituting
    /// another one silently is exactly what a pin forbids — and tests).
    pub model_chain: Option<Arc<ModelChain>>,
}

/// The fallback walk down `[llm] model_priority` for ONE run.
///
/// The harness is the only chat path, and it used to run every step on the
/// head of the priority list alone: when that model's provider failed, the
/// step ladder retried the same model, the continuation loop retried the
/// round, and the turn ended in "could not run" while the rest of the list
/// sat unused (observed 2026-09-21: a remote Ollama answering every request
/// with "error decoding response body", Claude models configured below it,
/// three turns in a row with no reply). The retired direct-chat path walked
/// the list; this restores that for every run.
///
/// Sticky for the run: once a model has failed past its retries, the rest of
/// the run stays on the model that replaced it rather than paying the dead
/// model's retry ladder again on every step. The NEXT run builds a fresh
/// chain from the top, so a recovered primary is used again at once.
pub struct ModelChain {
    models: Vec<String>,
    active: std::sync::atomic::AtomicUsize,
    /// Where a switch is announced. On the chain rather than the runner
    /// because the planner — whose runner deliberately has no transcript
    /// sink — is usually the first to meet a dead model.
    announce: Option<ChatSink>,
}

impl ModelChain {
    /// A chain over `models` (in priority order), starting at the first one
    /// a registered provider serves — or at the head when none is, so the
    /// caller's unserved-model notice names the model the user put first.
    ///
    /// # Panics
    ///
    /// When `models` is empty: a run always has a model to start on.
    #[must_use]
    pub fn new(models: Vec<String>, router: &LlmRouter) -> Self {
        assert!(!models.is_empty(), "a model chain needs at least one model");
        let start = models
            .iter()
            .position(|m| router.client_for_model(m).is_some())
            .unwrap_or(0);
        Self {
            models,
            active: std::sync::atomic::AtomicUsize::new(start),
            announce: None,
        }
    }

    /// Announce every switch in `sink`'s transcript (see
    /// [`ChatSink::model_fallback`]).
    #[must_use]
    pub fn announcing_to(mut self, sink: ChatSink) -> Self {
        self.announce = Some(sink);
        self
    }

    /// The model the run is on now.
    #[must_use]
    pub fn active(&self) -> String {
        let index = self.active.load(std::sync::atomic::Ordering::SeqCst);
        debug_assert!(index < self.models.len());
        self.models[index].clone()
    }

    /// `failed` gave up: move to the next model a registered provider serves
    /// and return it, or `None` when the list is exhausted.
    ///
    /// Only advances when `failed` is still the active model, so two runners
    /// reporting the same failure move the chain once, not twice.
    pub fn advance_past(&self, failed: &str, router: &LlmRouter) -> Option<String> {
        use std::sync::atomic::Ordering;
        let current = self.active.load(Ordering::SeqCst);
        if self.models[current] != failed {
            // Someone already moved on; follow them.
            return Some(self.models[current].clone());
        }
        let next = (current + 1..self.models.len())
            .find(|&i| router.client_for_model(&self.models[i]).is_some())?;
        match self
            .active
            .compare_exchange(current, next, Ordering::SeqCst, Ordering::SeqCst)
        {
            Ok(_) => Some(self.models[next].clone()),
            Err(now) => Some(self.models[now].clone()),
        }
    }
}

/// A step failure that says the MODEL (or its provider) cannot serve this
/// run, so the next model in the priority list should get it: transient
/// provider faults that outlived the step's retry ladder, any provider 4xx
/// (auth, billing, rate limit, unknown model, context too long for this
/// window), an unregistered provider, and a model that kept answering with
/// nothing. Everything else — a Stop, a harness guard — is not the model's
/// fault, and another model would fail it the same way.
#[must_use]
pub(crate) fn warrants_model_fallback(message: &str) -> bool {
    is_transient_llm_error(message)
        || message.contains("API error: 4")
        || message.contains("No provider available for model")
        || message.contains("empty completion")
}

/// Split a message's attachments into the images a step can send to the
/// model, `(base64_data, media_type)`, and a note naming the ones it cannot.
///
/// The harness path used to warn in the daemon log and DROP every attachment
/// — an image sent with "what is in this picture?" reached the model as the
/// bare question. Images (inline base64 of an `image/*` type) now ride each
/// step; anything else — a PDF, a URL, a text file — is named in the note so
/// the model can say it could not read it, rather than answering as if
/// nothing had been attached.
#[must_use]
pub fn split_attachments(
    attachments: &[crate::protocol::Attachment],
) -> (Vec<(String, String)>, Option<String>) {
    let mut images = Vec::new();
    let mut unreadable = Vec::new();
    for attachment in attachments {
        let inline =
            !attachment.data.starts_with("http://") && !attachment.data.starts_with("https://");
        if attachment.content_type.starts_with("image/") && inline {
            images.push((attachment.data.clone(), attachment.content_type.clone()));
        } else {
            unreadable.push(format!(
                "{} ({})",
                attachment.filename, attachment.content_type
            ));
        }
    }
    debug_assert_eq!(images.len() + unreadable.len(), attachments.len());
    let note = (!unreadable.is_empty()).then(|| {
        format!(
            "[The user attached {} that cannot be read in this chat — only inline images \
             can: {}. Say so if the request depends on it.]",
            if unreadable.len() == 1 {
                "a file"
            } else {
                "files"
            },
            unreadable.join(", ")
        )
    });
    (images, note)
}

/// Streams a harness step into a chat session using the *existing* chat event
/// contract (`MessageDelta` / `ToolStart` / `ToolEnd`).
///
/// A long-horizon run therefore renders in the transcript with no protocol
/// change on the GUI side.
///
/// When `run` is set (chat-backed runs), every callback ALSO fills the
/// registered [`crate::agent_service::ExternalRunHandle`] buffers, which is
/// what makes navigation recovery (`get_run_state`), Stop, and end-of-run
/// timeline persistence work — events alone vanish the moment the page
/// unmounts.
#[derive(Clone)]
pub struct ChatSink {
    pub session_id: String,
    pub message_id: String,
    pub event_tx: tokio::sync::broadcast::Sender<Event>,
    pub run: Option<crate::agent_service::ExternalRunHandle>,
    /// When set, every completed tool call is recorded to the shared stats
    /// tracker (and the Turso time-series when `storage` is also set) —
    /// parity with the retired direct chat path, which recorded after the
    /// run from the flat result.
    pub tool_stats: Option<nanna_agent::ToolStatsTracker>,
    pub storage: Option<Arc<Storage>>,
    /// The single item of a one-task (conversation-shaped) plan. Steps for
    /// this item render with no `**[working]**` banner so a plain question
    /// reads as a plain reply; items added later (interjections, replans)
    /// are announced — by then there IS a run to attribute work to.
    /// Shared (not per-clone) state: the finalizer sets it after seeding,
    /// and the step runner's clone must see it.
    pub quiet_item: Arc<std::sync::Mutex<Option<i64>>>,
    /// Session liveness ledger (P22): every sink event also stamps the
    /// per-session ledger the liveness beat and `session.liveness` verb read,
    /// so "working, wedged, or finished" is answerable without log greps.
    /// `None` on paths that have no beat (task-run manager, tests).
    pub liveness: Option<Arc<crate::liveness::SessionLiveness>>,
    /// Where the reply's text stands, so each harness step's text starts a
    /// new paragraph instead of running on from the last step's final word.
    /// Shared (not per-clone) for the same reason as `quiet_item`.
    pub text_join: Arc<std::sync::Mutex<StepTextJoin>>,
}

/// How consecutive harness steps' text joins into one reply.
///
/// Steps are separate agent runs streaming into the same message. The loop
/// separates its own iterations (`loop_runner` emits a space between them),
/// but nothing separated steps, and once the step banner moved out of the
/// text (run mechanics are not message content) the steps simply abutted:
/// `…read the file.The file says…`, in the stream AND in the persisted reply.
#[derive(Debug, Default)]
pub struct StepTextJoin {
    /// The last emitted character was whitespace (or nothing was emitted).
    ends_with_whitespace: bool,
    /// A step began and has not emitted text yet.
    step_break_pending: bool,
    /// The text of the last step that emitted any, and of the current one —
    /// what [`Self::repeated_last_step`] compares. Bounded by two steps'
    /// output, which the reply already holds.
    previous_step_text: String,
    current_step_text: String,
    /// The harness has finished its steps: later text (the run summary, stop
    /// notices) is not step text and must not extend the last step's.
    steps_ended: bool,
    /// Bytes emitted into the reply so far, separators included.
    emitted_bytes: usize,
    /// Where the current step's text sits in the reply: from the start of
    /// the separator before it to the end of its last delta.
    current_step_span: Option<(usize, usize)>,
}

/// What goes between two steps' text: a paragraph break, since each step is
/// a separate action, not a continuation of the previous sentence.
const STEP_TEXT_SEPARATOR: &str = "\n\n";

impl StepTextJoin {
    /// A step is starting: its first text, if any, opens a new paragraph.
    fn begin_step(&mut self) {
        self.step_break_pending = true;
        // A step that said nothing does not displace the last one that did.
        if !self.current_step_text.trim().is_empty() {
            self.previous_step_text = std::mem::take(&mut self.current_step_text);
        }
        self.current_step_text.clear();
        self.current_step_span = None;
        debug_assert!(self.step_break_pending, "a started step must arm the break");
    }

    /// No more steps: later text is the harness's summary, not a step's.
    pub fn end_steps(&mut self) {
        self.steps_ended = true;
        debug_assert!(self.steps_ended);
    }

    /// The last step's text when it repeats the step before it verbatim
    /// (whitespace-collapsed) — the converging answer of a model that never
    /// said `TASK COMPLETE` (see `answer_converged` in the harness). The live
    /// stream has already shown it; the persisted reply keeps one copy.
    ///
    /// Returns the repeat's text and its byte span in the reply (separator
    /// included), so the caller can cut exactly it even when harness text
    /// (the run summary) follows.
    #[must_use]
    pub fn repeated_last_step(&self) -> Option<(String, std::ops::Range<usize>)> {
        let current: Vec<&str> = self.current_step_text.split_whitespace().collect();
        let previous: Vec<&str> = self.previous_step_text.split_whitespace().collect();
        let (start, end) = self.current_step_span?;
        debug_assert!(start <= end, "a span runs forward");
        (!current.is_empty() && current == previous)
            .then(|| (self.current_step_text.clone(), start..end))
    }

    /// Account for `text` about to be emitted and return the separator that
    /// must precede it (empty when none is needed).
    ///
    /// No separator before the reply's first text, and none when either side
    /// of the seam already carries whitespace — the model's own `\n` must not
    /// grow into three.
    fn separator_before(&mut self, text: &str) -> &'static str {
        debug_assert!(!text.is_empty(), "empty deltas are dropped before joining");
        let needs_break = self.step_break_pending
            && self.emitted_bytes > 0
            && !self.ends_with_whitespace
            && !text.starts_with(char::is_whitespace);
        self.step_break_pending = false;
        self.ends_with_whitespace = text.ends_with(char::is_whitespace);
        let separator = if needs_break { STEP_TEXT_SEPARATOR } else { "" };
        let start = self.emitted_bytes;
        self.emitted_bytes += separator.len() + text.len();
        if !self.steps_ended {
            self.current_step_text.push_str(text);
            let span_start = self.current_step_span.map_or(start, |(from, _)| from);
            self.current_step_span = Some((span_start, self.emitted_bytes));
        }
        debug_assert!(
            !self.step_break_pending,
            "the break is spent by the first text"
        );
        separator
    }
}

impl ChatSink {
    pub(crate) fn delta(&self, text: &str) {
        if text.is_empty() {
            return;
        }
        let separator = self
            .text_join
            .lock()
            .map_or("", |mut join| join.separator_before(text));
        if separator.is_empty() {
            self.emit_delta(text);
        } else {
            let mut joined = String::with_capacity(separator.len() + text.len());
            joined.push_str(separator);
            joined.push_str(text);
            self.emit_delta(&joined);
        }
    }

    /// Stream `text` and record it in the run's buffers, exactly as given.
    fn emit_delta(&self, text: &str) {
        if let Some(live) = &self.liveness {
            live.on_stream_delta(text.len());
        }
        let _ = self.event_tx.send(Event::MessageDelta {
            session_id: self.session_id.clone(),
            message_id: self.message_id.clone(),
            delta: text.to_string(),
        });
        if let Some(run) = &self.run {
            // Infallible, as in the service path (`agent_service::lock_buf`): a
            // snapshot clone holds the lock only for the clone.
            {
                let mut acc = crate::agent_service::lock_buf(&run.accumulated_text);
                acc.push_str(text);
            }
            // The journal lock is std::sync and infallible by design (see
            // `agent_service::timeline_lock`, which both writers share) — merge
            // into the trailing Text item so a token stream stays one item, not
            // thousands.
            let mut journal = crate::agent_service::timeline_lock(&run.timeline);
            if let Some(crate::session::TimelineItem::Text { content, .. }) = journal.last_mut() {
                content.push_str(text);
            } else {
                journal.push(crate::session::TimelineItem::Text {
                    content: text.to_string(),
                    at: chrono::Utc::now().to_rfc3339(),
                });
            }
        }
    }

    fn thinking(&self, text: &str) {
        if text.is_empty() {
            return;
        }
        if let Some(live) = &self.liveness {
            live.on_thinking(text.len());
        }
        let _ = self.event_tx.send(Event::ThinkingDelta {
            session_id: self.session_id.clone(),
            delta: text.to_string(),
        });
        if let Some(run) = &self.run {
            {
                let mut acc = crate::agent_service::lock_buf(&run.accumulated_thinking);
                acc.push_str(text);
            }
            let mut journal = crate::agent_service::timeline_lock(&run.timeline);
            if let Some(crate::session::TimelineItem::Thinking { content, .. }) = journal.last_mut()
            {
                content.push_str(text);
            } else {
                journal.push(crate::session::TimelineItem::Thinking {
                    content: text.to_string(),
                    at: chrono::Utc::now().to_rfc3339(),
                });
            }
        }
    }

    fn tool_start(&self, call_id: &str, name: &str, input: &Value, model: Option<&str>) {
        if let Some(live) = &self.liveness {
            live.on_tool_start(name);
        }
        let _ = self.event_tx.send(Event::ToolStart {
            session_id: self.session_id.clone(),
            call_id: call_id.to_string(),
            name: name.to_string(),
            input: input.clone(),
            model: model.map(String::from),
            tokens: None,
            total_tokens: None,
        });
        if let Some(run) = &self.run {
            {
                let mut active = crate::agent_service::lock_buf(&run.active_tool_calls);
                active.push(crate::agent_service::ActiveToolCallInfo {
                    call_id: call_id.to_string(),
                    name: name.to_string(),
                    started_at: chrono::Utc::now(),
                });
            }
            // The chat path's own journal writer, not a copy of it: this sink
            // writes to the same journal, and the two used to disagree (see
            // `tool_end`). 0/0 tokens = unknown, stored as None as before.
            crate::agent_service::timeline_tool_start(&run.timeline, call_id, name, input, 0, 0);
        }
    }

    fn tool_end(
        &self,
        call_id: &str,
        name: &str,
        output: &str,
        success: bool,
        duration_ms: u64,
        data: Option<&Value>,
    ) {
        // P22 Tier 4: a breaker replay carries a machine-readable marker in
        // its structured data — recorded as its own outcome so a wall of
        // short-circuits never reads as tool failures in the stats.
        let short_circuited = data
            .and_then(|d| d.get("short_circuited"))
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if let Some(live) = &self.liveness {
            // A short-circuited replay did NOT run the tool: it must not
            // count as side-effect evidence in the liveness ledger, or a
            // wall of replays would read as "the world changed" to the
            // repeat-completion escalation.
            live.on_tool_end(name, success && !short_circuited);
        }
        let _ = self.event_tx.send(Event::ToolEnd {
            session_id: self.session_id.clone(),
            call_id: call_id.to_string(),
            output: output.to_string(),
            success,
            duration_ms,
            data: data.cloned(),
        });
        if let Some(run) = &self.run {
            {
                let mut active = crate::agent_service::lock_buf(&run.active_tool_calls);
                active.retain(|t| t.call_id != call_id);
            }
            {
                let mut done = crate::agent_service::lock_buf(&run.completed_tool_calls);
                done.push(crate::agent_service::CompletedToolCallInfo {
                    call_id: call_id.to_string(),
                    name: name.to_string(),
                    output: output.to_string(),
                    success,
                    duration_ms,
                });
            }
            // The chat path's writer, not a second opinion. This sink used to
            // hand-roll the back-fill and got three things wrong that the
            // shared one gets right: it stored the output UNCAPPED (the
            // journal's 4000-byte bound exists so a thousand-call mission's
            // record stays shippable), it matched the newest item with this
            // call id even when that call had already completed — Ollama
            // reuses ids per response, so it could overwrite an EARLIER call's
            // outcome — and an end with no journaled start vanished instead of
            // being recorded. The replay marker and the edit diff ride along.
            crate::agent_service::timeline_tool_end(
                &run.timeline,
                call_id,
                name,
                crate::agent_service::ToolEndRecord::from_result(
                    output,
                    success,
                    duration_ms,
                    data,
                ),
            );
        }
        if let Some(stats) = &self.tool_stats {
            let observation = nanna_agent::ToolObservation {
                tool_name: name.to_string(),
                success,
                short_circuited,
                duration_ms,
                output_size: output.len(),
                // A replay notice is harness steering, not a tool error.
                error: (!success && !short_circuited).then(|| output.to_string()),
                session_id: Some(self.session_id.clone()),
            };
            let stats = stats.clone();
            let storage = self.storage.clone();
            let session_id = self.session_id.clone();
            // Recording is async and this callback is sync — hand it off.
            tokio::spawn(async move {
                if let Some(storage) = storage
                    && let Err(e) = storage
                        .log_tool_call(&nanna_storage::NewToolCall {
                            tool_name: &observation.tool_name,
                            success: observation.success,
                            short_circuited: observation.short_circuited,
                            duration_ms: observation.duration_ms,
                            output_size: observation.output_size,
                            error_message: observation.error.as_deref(),
                            session_id: Some(&session_id),
                        })
                        .await
                {
                    tracing::warn!("Failed to log tool call to DB: {e}");
                }
                stats.record(observation).await;
            });
        }
    }

    /// The per-request usage callback — the daemon's only live context-usage
    /// signal, and the reason a client can show "23k of 27k" instead of a bare
    /// spinner.
    ///
    /// `window` is the agent loop's OWN enforced input bound at the moment of
    /// the request (`AgentContext::hard_limit`), so used-vs-window is measured
    /// against the limit that request was actually judged against — a mid-run
    /// `num_ctx` demotion moves the denominator with it.
    ///
    /// Only the event is filled here. The run-scoped token totals the retired
    /// in-service path also accumulated (`run_input_tokens` /
    /// `run_output_tokens`, served in `RunStateSnapshot`) live on
    /// `ActiveChat`, and `ExternalRunHandle` — this path's only handle on that
    /// state — does not expose them; when it does, they belong here beside the
    /// event.
    /// Wire a harness step's text, thinking, tool and usage streams into this
    /// sink.
    ///
    /// The live context gauge rides along: the step path never set
    /// `on_usage`, so the ONLY emitter of `ContextUsage` sat in the retired
    /// in-service chat path: every client's meter read 0 of 0 for the whole
    /// run while the daemon recomputed the real figures on every request (59
    /// of 59 captured run-state snapshots, 56 of them mid-run).
    fn attach_stream_callbacks(&self, options: &mut nanna_agent::RunOptions) {
        let text_sink = self.clone();
        let think_sink = self.clone();
        let start_sink = self.clone();
        let end_sink = self.clone();
        options.on_text = Some(Box::new(move |chunk: &str| text_sink.delta(chunk)));
        options.on_thinking = Some(Box::new(move |chunk: &str| think_sink.thinking(chunk)));
        options.on_tool_start = Some(Box::new(
            move |call_id: &str, name: &str, input: &Value, model: Option<&str>| {
                start_sink.tool_start(call_id, name, input, model);
            },
        ));
        options.on_tool_end = Some(Box::new(
            move |call_id: &str,
                  name: &str,
                  output: &str,
                  success: bool,
                  duration_ms: u64,
                  data: Option<&Value>| {
                end_sink.tool_end(call_id, name, output, success, duration_ms, data);
            },
        ));
        options.on_usage = Some(self.usage_callback());
    }

    fn usage_callback(&self) -> Box<dyn Fn(u32, u32, u64) + Send + Sync> {
        let event_tx = self.event_tx.clone();
        let session_id = self.session_id.clone();
        Box::new(move |input: u32, _output: u32, window: u64| {
            let _ = event_tx.send(Event::ContextUsage {
                session_id: session_id.clone(),
                used: u64::from(input),
                window,
            });
        })
    }

    /// Announce which item the run is starting, so the transcript reads as
    /// work-in-progress rather than a wall of unattributed text.
    /// The label is read back out of the step prompt (`build_step_prompt`
    /// writes `Task #id: title`); if that line is ever absent the header
    /// degrades to the bare item id rather than failing.
    fn step_header(&self, request: &StepRequest) {
        // Every step — quiet or announced — is a new action, so its text
        // starts a new paragraph (see `StepTextJoin`).
        if let Ok(mut join) = self.text_join.lock() {
            join.begin_step();
        }
        // The liveness ledger tracks EVERY step, including banner-free quiet
        // items — a wedge inside a conversation-shaped turn must still be
        // visible to the beat and the `session.liveness` verb.
        if let Some(live) = &self.liveness {
            let kind = match request.step_kind {
                nanna_agent::harness::StepKind::Plan => "planning",
                nanna_agent::harness::StepKind::Verify => "verifying",
                nanna_agent::harness::StepKind::Execute => "working",
            };
            live.on_step(request.step_index, kind, &request.item_title);
        }
        // Conversation-shaped turns (one-task plans) stay banner-free so the
        // transcript feels like chat — see the `quiet_item` field.
        if self
            .quiet_item
            .lock()
            .is_ok_and(|quiet| *quiet == Some(request.item_id))
        {
            return;
        }
        let kind = match request.step_kind {
            nanna_agent::harness::StepKind::Plan => "planning",
            nanna_agent::harness::StepKind::Verify => "verifying",
            nanna_agent::harness::StepKind::Execute => "working",
        };
        let label = if request.item_title.trim().is_empty() {
            format!("Task #{}", request.item_id)
        } else {
            request.item_title.clone()
        };

        // A step banner is run mechanics — it belongs in the journal the GUI
        // renders as status, NOT in the message body. Written as text it also
        // ended up in conversation history and was replayed to the model on
        // later turns as if it had said it.
        let _ = self.event_tx.send(Event::StepStarted {
            session_id: self.session_id.clone(),
            kind: kind.to_string(),
            label: label.clone(),
            item_id: request.item_id,
        });
        if let Some(run) = &self.run {
            crate::agent_service::timeline_lock(&run.timeline)
                .push(crate::session::TimelineItem::Step {
                    phase: kind.to_string(),
                    label,
                    item_id: request.item_id,
                    at: chrono::Utc::now().to_rfc3339(),
                });
        }
    }

    /// Say — where the user is looking — that `failed` gave up and the run
    /// continues on `next`: a fault in the run journal (what the chat shows
    /// inline), the model badge's switch event, and the liveness ledger's
    /// model. A silent substitution would read as the configured model
    /// answering.
    pub fn model_fallback(&self, failed: &str, next: &str, error: &str) {
        let message = format!(
            "{failed} failed: {} — continuing on {next}",
            crate::agent_service::truncate(error, 200)
        );
        if let Some(live) = &self.liveness {
            live.set_model(next);
        }
        let _ = self.event_tx.send(Event::ModelSwitch {
            model: next.to_string(),
            reason: Some(message.clone()),
        });
        let _ = self.event_tx.send(Event::Error {
            code: "model_error".to_string(),
            message: message.clone(),
            session_id: Some(self.session_id.clone()),
        });
        if let Some(run) = &self.run {
            crate::agent_service::timeline_lock(&run.timeline).push(crate::session::TimelineItem::Fault {
                message,
                at: chrono::Utc::now().to_rfc3339(),
            });
        }
    }
}

/// In-step retries for transient provider errors.
///
/// Bound justification: local models corrupt their own tool-call template
/// mid-generation (observed: Ollama 500s from qwen3.5), and Ollama's runner
/// intermittently enters a degraded state where a stale KV "context
/// checkpoint" restore sends generation straight to a stop token (observed:
/// 200s with ~33 generated tokens and empty output, recurring ~1h into a
/// sustained run). Three retries with escalating backoff — and a runner
/// reset before the last — absorb both without masking a dead endpoint; the
/// harness circuit breaker still sees persistent failure.
const STEP_LLM_RETRIES: usize = 3;

/// Backoff before retry attempts 1..=3.
const STEP_RETRY_BACKOFF_SECS: [u64; 3] = [2, 5, 10];

/// Transient provider faults worth retrying in place: 5xx (including the
/// synthesized 502 for aborted Ollama generations), client timeouts and
/// connection failures ("error sending request" is reqwest's send-phase
/// failure), mid-stream drops ("Stream error:"), and stream-watchdog trips
/// (`AgentError::StreamWatchdog` — the stream FUTURE wedged while the
/// transport thought it healthy; a fresh request builds a fresh future, so
/// the ladder's retry-plus-runner-reset is exactly the right medicine, and a
/// persistent wedge still surfaces through the circuit breaker). Shared by
/// the step runner and the chat path — both heal with the same ladder.
pub(crate) fn is_transient_llm_error(message: &str) -> bool {
    message.contains("API error: 5")
        || message.contains("timed out")
        || message.contains("connection")
        || message.contains("error sending request")
        || message.contains("Stream error:")
        || message.contains("stream watchdog:")
}

/// Forensics: append the exact prompt of an empty-completion step to a temp
/// file so the deterministic trigger can be replayed and minimized offline.
fn dump_empty_step(request: &StepRequest, attempt: usize) {
    use std::io::Write;
    let path = std::env::temp_dir().join("nanna_empty_step_prompts.log");
    let entry = format!(
        "==== {} item#{} kind={:?} attempt={attempt} ====\n{}\n\n",
        chrono::Utc::now().to_rfc3339(),
        request.item_id,
        request.step_kind,
        request.prompt,
    );
    let _ = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .and_then(|mut f| f.write_all(entry.as_bytes()));
}

/// An "empty completion": HTTP success but no text, no tool calls, and ~no
/// generated tokens. Observed live from Ollama (a whole 42-item plan was
/// burned by 462 such no-op steps in 9 minutes — each one "succeeded", made
/// no progress, and marched every item to abandonment). Treat as a transient
/// provider failure, never as a step result. The token bound distinguishes
/// this from a legitimate thinking-only step, whose reasoning tokens count.
fn is_empty_completion(outcome: &StepOutcome) -> bool {
    outcome.tool_calls.is_empty() && outcome.text.trim().is_empty() && outcome.output_tokens <= 8
}

/// A step that neither acted nor said anything: reasoning tokens were spent,
/// but no tool ran and no text came back.
///
/// This is NOT a step result. A harness step exists to move an item, and
/// thinking alone moves nothing — yet it used to be accepted as an outcome
/// because the reasoning tokens put it over [`is_empty_completion`]'s bound.
/// Observed live 2026-07-27: the same item was re-entered SEVEN times, each
/// visit producing a "Thinking …" block and nothing else, and the run ended
/// "8 steps · 0 items completed" with the artifact untouched for 19 minutes.
///
/// Distinct from an empty completion (a degraded provider emitting ~nothing),
/// which is already handled: here the model is generating fine, it just did
/// not commit to an action, so the retry re-anchors it with an explicit nudge.
fn made_no_progress(outcome: &StepOutcome) -> bool {
    outcome.tool_calls.is_empty() && outcome.text.trim().is_empty()
}

/// Appended to a step prompt after a no-progress attempt.
///
/// Says what was missing and what the only two acceptable shapes are. Kept
/// short: the step prompt is already long, and a small model reads the tail.
const NO_PROGRESS_NUDGE: &str = "\n\n[SYSTEM: your previous attempt at this step produced only \
     reasoning — no tool call and no answer, so nothing changed. Do not think further about it. \
     Either CALL A TOOL now (that is how work happens), or, if the step is genuinely already \
     satisfied, reply with one short sentence saying so.]";

/// A short, constant-size label for the fault a retry is healing.
///
/// The classified KIND, never the raw message: an error body carries provider
/// JSON, stream tails and byte counts, and pasting that into a prompt spends
/// window on noise the model cannot act on. Every arm reuses a classifier the
/// ladder already consults, so the label and the healing action can never
/// disagree.
pub(crate) fn transient_fault_kind(message: &str) -> &'static str {
    if gpu_memory_error(message) {
        "the model ran out of GPU memory"
    } else if ollama_generation_parse_error(message).is_some() {
        "the provider could not parse the tool call"
    } else if message.contains("No NDJSON line was ever parsed") {
        "the provider's stream died before any output"
    } else if wedged_runner_error(message) {
        "the provider's runner wedged mid-generation"
    } else if message.contains("stream watchdog:") {
        "the stream stalled and the watchdog cut it"
    } else if message.contains("timed out") {
        "the request timed out"
    } else {
        "a provider transport fault"
    }
}

/// Appended to a step prompt when the SAME step is retried after a transient
/// provider fault interrupted it mid-flight.
///
/// A retry re-enters the step with a FRESH context — that is the re-anchor,
/// and it is what makes retries cheap. But the fresh context is also a lie by
/// omission: the interrupted attempt may already have called tools, and those
/// calls RAN. Their effects are on disk while the conversation the model can
/// see shows none of them, so a model that reads its own context as ground
/// truth concludes the file is untouched and rewrites it from scratch —
/// destroying the work the interrupted attempt did. Observed as the
/// abandonment/truncation chain: peak content early, a shrunken file at the
/// end.
///
/// Replaying the partial transcript is not the fix and is not attempted:
/// [`StepAttemptError`] is lossy by design (the ladder keeps the classified
/// message and the wedge fingerprint, nothing else), and a half transcript
/// re-fed as history is exactly the corrupted state the re-anchor exists to
/// escape. Naming the situation costs a fixed handful of sentences instead.
///
/// What the retry carries BESIDE this warning is [`carried_side_effect_note`]:
/// the calls that actually landed, surfaced from the liveness ledger. That is
/// a record of what changed on disk, not a half transcript posing as history,
/// so it does not reopen the state this note exists to escape.
pub(crate) fn transient_retry_note(attempt: usize, fault: &str) -> String {
    format!(
        "\n\n[SYSTEM: this same step was interrupted mid-flight by a provider fault (attempt \
         {attempt}; fault: {fault}) and is being retried. Tool calls the interrupted attempt had \
         already made DID execute — their effects are on disk even though this conversation does \
         not show them. Continue the step, do not restart it: re-read the working artifact before \
         any whole-file write; a rewrite that drops existing functions is destruction, not \
         progress.]"
    )
}

/// Byte budget for the carried side-effect list, as a share of the model's
/// LIVE hard input limit.
///
/// Derived, not chosen. The harness has already decided how many bytes of a
/// step prompt may be spent re-stating work the model cannot see: one
/// screenful — [`nanna_agent::harness::STEP_RESULT_TAIL_MAX_BYTES`], the slice
/// of the previous step's output it feeds forward — and that number was sized
/// against the universal window floor ([`nanna_llm::UNKNOWN_CONTEXT_WINDOW`]).
/// Read as a SHARE rather than as a constant, it is 2000 bytes of that floor's
/// hard input limit, ~1.8% of the input budget once bytes are converted at the
/// usual ~4 chars/token. Scaling by the live hard limit holds the share fixed
/// while the window moves: a `num_ctx`-demoted model pays the demoted window's
/// worth, a large window may afford more.
///
/// A share is the bound precisely because "however many calls the attempt
/// made" is not one: 104 side-effecting calls were observed in a single
/// aborted attempt, which rendered in full is ~13% of everything a small model
/// can read — a re-anchor large enough to displace the work it re-anchors.
pub(crate) fn carried_side_effect_budget_bytes(hard_input_limit_tokens: usize) -> usize {
    let floor_hard_limit = nanna_llm::unknown_model_info("", "").hard_input_limit();
    nanna_agent::harness::STEP_RESULT_TAIL_MAX_BYTES.saturating_mul(hard_input_limit_tokens)
        / floor_hard_limit.max(1)
}

/// The interrupted attempt's side-effecting calls, newest first, plus the
/// pointer saying where their full inputs and outputs still are.
///
/// This is the concrete half of the re-anchor. [`transient_retry_note`] can
/// only say that tool calls "DID execute"; a warning with no subject is not
/// something a model can act on, and the model's next move after a blind
/// re-entry is a whole-file rewrite. The calls are SURFACED from the P22
/// Tier 4 [`crate::liveness::SessionLiveness`] ledger, which recorded them as
/// they ran — nothing here is reconstructed, and no partial transcript is
/// replayed.
///
/// The shape follows the harness's own bounded synthesis
/// (`loop_runner::step_activity_digest`): newest calls first, cut at the
/// budget, and the calls that did not fit announced as a count rather than
/// dropped in silence. The announcement itself always ships — a budget too
/// small for even one line must still leave the model knowing the attempt
/// acted, which is the whole fact it is otherwise missing.
///
/// It names only `discover_tools`, the one tool every request ships. A prompt
/// that names a tool whose schema was never sent is how malformed acceptance
/// checks went from 3.1% to 53-66% (see [`CORE_TOOLS`]).
///
/// `outputs_recallable` is whether this run HAS a memory service writing tool
/// results (`memory_sink`). Without one, `loop_runner` keeps results in
/// context instead of storing them, so pointing the model at a store that was
/// never written is a promise the run cannot keep — the note then says only
/// what remains true.
pub(crate) fn carried_side_effect_note(
    marks: &[crate::liveness::ToolMarkSnapshot],
    budget_bytes: usize,
    outputs_recallable: bool,
) -> String {
    if marks.is_empty() {
        return String::new();
    }
    let head = format!(
        "\n[SYSTEM: that attempt made {} side-effecting tool call{} before it was cut, newest \
         first — each one ran, and whatever it changed is already on disk:",
        marks.len(),
        if marks.len() == 1 { "" } else { "s" }
    );
    let tail = if outputs_recallable {
        "\nMost of what those calls produced was stored as it ran and may be readable out of \
         your memory — reach the memory tools with discover_tools. Do not rely on it: read the \
         artifact back off disk rather than assuming what a file now holds.]"
    } else {
        "\nWhat those calls produced is not in this conversation any more — read the artifact \
         back off disk before you write over it rather than assuming what it holds.]"
    };
    let mut lines: Vec<String> = Vec::new();
    let mut used = head.len() + tail.len();
    for (idx, mark) in marks.iter().enumerate() {
        let line = format!("\n- {} ({}s ago)", mark.name, mark.secs_ago);
        if used + line.len() > budget_bytes {
            let omitted = marks.len() - idx;
            lines.push(format!(
                "\n- …and {omitted} earlier call{}",
                if omitted == 1 { "" } else { "s" }
            ));
            break;
        }
        used += line.len();
        lines.push(line);
    }
    format!("{head}{}{tail}", lines.concat())
}

/// The only tool definitions shipped with a step. Everything else is reached
/// through `discover_tools`.
///
/// Two rules, and they are not in tension:
///
/// 1. **A plan never restricts capability.** Its `tool_scope` is a guess made
///    before the work starts, and a step sealed inside that guess cannot do
///    the job the step turned out to need. Observed 2026-07-27: a plan that
///    scoped steps to `exec` produced 141 exec calls and ZERO `write_file` /
///    `read_file` / `edit_file` — every file operation went through shell
///    heredocs, sailing past the anti-erosion ratchet, the syntax gate and
///    the fork refusals, which is why those guards never fired during the
///    `minidb.sh` forks we spent hours chasing.
///
/// 2. **Context is not free.** Shipping ~30 schemas on every request costs
///    thousands of tokens per step on a 32k window and measurably degrades
///    small-model tool selection. So the request carries the minimum, and the
///    model pulls in what it needs: `discover_tools` to reach any tool, and
///    `recall` because memory is how a step recovers what it already knows
///    (including the `[memory:…]` handles in its own context).
///
/// Anything activated stays active for the rest of the run, so discovery is
/// paid once per tool, not once per step.
/// ONE tool ships: the one that reaches every other tool.
///
/// Schemas are context, and on a 32k window the full set costs thousands of
/// tokens per step and measurably degrades small-model tool selection. So the
/// request carries `discover_tools` and nothing else, and the model pulls in
/// what the task actually needs.
///
/// The hazard this creates is real and worth naming: a model that cannot see a
/// file tool will write files by shell redirection instead, which bypasses the
/// anti-erosion ratchet, the suffixed-fork guard and the path repair — every
/// healing feature lives in the tools, so improvising around them silently
/// loses work. Two things prevent it. `stable_prefix` tells the model plainly
/// that the other tools exist behind discovery and that improvising around them
/// is how work gets lost; and activation now persists for the whole run, so
/// discovery is paid once rather than re-paid out of every step's iteration
/// budget.
///
/// What must NOT come back is the previous compromise, where the prompt ordered
/// the model to call `todo(action='note', …)` while `todo`'s schema was never
/// sent. That is not gating, it is the prompt and the request disagreeing about
/// what exists — and the model obeys blind. Measured 2026-07-28: malformed
/// acceptance checks went from 3.1% of `todo` calls to 53-66%, and `exec` took
/// 50 calls passing `query` (the only parameter name left in context) instead
/// of `command`. If a prompt ever needs to name a tool, the answer is to fix
/// the prompt, not to widen this list.
const CORE_TOOLS: &[&str] = &["discover_tools"];

/// The one place a step's starting active-tool set is assembled: the default
/// set (core + tools discovered so far this run, exactly as before), with the
/// user's manual per-chat selections APPENDED.
///
/// The contract this function exists to make testable:
/// - **Empty selection = unchanged.** With no user selections the result is
///   byte-identical to the pre-feature computation — core first, then
///   discovered — so default behavior (core tools, `discover_tools` paging,
///   per-task scopes) cannot drift.
/// - **Selections only ever ADD.** A user pick already present in the default
///   set is not duplicated; nothing is ever removed or reordered by a pick.
fn merge_active_tools(
    core: &[&str],
    discovered: &std::collections::HashSet<String>,
    user_selected: &[String],
) -> Vec<String> {
    let mut active: Vec<String> = core
        .iter()
        .map(|t| (*t).to_string())
        .chain(discovered.iter().cloned())
        .collect();
    for tool in user_selected {
        if !active.iter().any(|t| t == tool) {
            active.push(tool.clone());
        }
    }
    active
}

/// Faults that mean the local runner's STATE is bad, not that the request
/// was unlucky. These are healed by unloading/reloading the model; another
/// identical request just re-hits the wedge.
///
/// - "empty completion": 200s with ~no generated tokens (stale KV checkpoint
///   restore sending generation straight to a stop token).
/// - "same token": the repetition watch caught a decoder emitting one token
///   forever. Both were surfacing as "no done=true" stream aborts before the
///   watch existed, and both come from the same degraded-runner condition.
/// - "aborted generation mid-response": the server closed a generation with
///   done=false. Observed live 2026-08-08 (ministral-3:8b endurance): five
///   abort clusters ended all three run segments while this classifier
///   stayed silent — and both segment-boundary heals (the same runner reset
///   this gate arms) restored service instantly, one segment then running
///   104 productive minutes. The retry backoff ladder alone can never heal
///   it: only the model unload clears the runner state.
/// - Ollama-side tool-call parse 500s ("invalid character ... in string
///   literal" and Go-JSON kin): each is an aborted generation server-side,
///   and in the same run parse-500 storms directly preceded every done=false
///   cluster — accumulated aborts degrade the runner until no stream
///   finishes. A single one is often sampling luck (the plain retry stays
///   first); the reset arms only under the existing attempt>=2 repeat rule.
/// - "No NDJSON line was ever parsed": the stream hit EOF having produced
///   ZERO bytes of body and ZERO parsed lines, with nothing left in the
///   residual buffer — the dead-stream shape, emitted by `nanna-llm` only in
///   exactly that case. Observed 2026-08-15: 116 of 160 in-window step
///   retries carried it while this classifier stayed silent, so the ladder
///   re-asked a dead runner three times per step; the reset armed only when
///   the sibling "aborted generation mid-response" shape finally arrived at
///   11:48:43, after 9 blind retries, and healed it at once.
///
/// The bytes>0 variants of that same stream-end error ("Last line before the
/// cut: …", "N unparsed bytes remained") deliberately stay OUT. A stream that
/// produced output before dying is ambiguous with a generation-slot
/// contention cancel (`agent_service` drops the loser's stream), and a reset
/// there unloads a runner that was working on someone else's turn.
fn wedged_runner_error(message: &str) -> bool {
    message.contains("empty completion")
        || message.contains("same token")
        || message.contains("aborted generation mid-response")
        || message.contains("No NDJSON line was ever parsed")
        || (message.contains("invalid character ") && message.contains(" in string literal"))
        || message.contains(" in string escape code")
        || message.contains("after object key:value pair")
}

/// The character an Ollama tool-call parse abort names, when the body is one.
///
/// Same last-line-fingerprint style as `provider_unknown_tool_name`
/// (nanna-agent's `loop_runner)`: the provider's own error body is the signal,
/// read out of whatever transport prefixes wrap it (`API error: 500 - {…}`).
/// Ollama's Go JSON decoder rejects the model's generated tool call before it
/// ever reaches the registry and names the offending byte — `invalid
/// character '\t' in string literal`, `… in string escape code`, `… after
/// object key:value pair`. The model emitted a RAW control character inside a
/// tool-call string argument, which no JSON parser accepts; retrying the same
/// request unchanged reproduces it, because the fault is in how the model
/// encodes, not in the transport.
///
/// Deliberately narrow in both directions. It requires the `invalid character
/// '<c>'` prefix AND one of the three parse sites, so an arbitrary 500 body
/// cannot arm a correction. And it explicitly refuses the dead-stream /
/// aborted-generation classes, whose bodies can quote the MODEL's own text
/// back in a tail: those faults belong to the runner reset, and a correction
/// aimed at the model would be advice about someone else's failure.
fn ollama_generation_parse_error(message: &str) -> Option<String> {
    if message.contains("Ollama stream ended without completion")
        || message.contains("aborted generation mid-response")
        || message.contains("empty completion")
    {
        return None;
    }
    let (_, rest) = message.split_once("invalid character '")?;
    let (character, tail) = rest.split_once('\'')?;
    if character.is_empty() {
        return None;
    }
    let names_a_parse_site = tail.starts_with(" in string literal")
        || tail.starts_with(" in string escape code")
        || tail.starts_with(" after object key:value pair");
    names_a_parse_site.then(|| character.to_string())
}

/// Appended to a step prompt after the provider refused to parse the model's
/// tool call because of a raw control character in a string argument.
///
/// The transport wall this heals is invisible from the model's side: the call
/// never reached the registry, so no tool result comes back to learn from,
/// and the plain retry re-generates the same encoding. So the note says what
/// was rejected, names the character the provider named, and gives the two
/// routes that actually work — JSON escapes for the argument, or `exec` when
/// the FILE genuinely needs literal control bytes. It rides the existing
/// `STEP_LLM_RETRIES` ladder: no extra attempt is bought, the attempt that
/// was already going to happen just carries the correction.
fn control_char_transport_note(character: &str) -> String {
    format!(
        "\n\n[SYSTEM: the provider REJECTED the tool call you were generating — its JSON \
         parser hit an invalid character ({character}) inside a string argument, so that call \
         never reached the tools and nothing from it ran. A raw control character cannot \
         travel inside a tool-call argument. Two legal routes: (1) ESCAPE it in the JSON — \
         write \\t for a tab and \\n for a newline inside the string argument, and the tool \
         receives the real character; or (2) when the file content genuinely needs literal \
         control bytes, call exec and let the shell produce them (printf 'a\\tb' > file, or a \
         quoted heredoc: cat <<'EOF' > file). Re-issue the call one of those two ways.]"
    )
}

/// Record whether a control-character correction healed the attempt it rode
/// on, so the correction rate is auditable from the log — the same
/// accountability the unserved-tool heal keeps for its own retries.
///
/// `still_failing` carries the message when the attempt failed again, and is
/// `None` when it succeeded.
fn log_parse_heal_outcome(attempt: usize, character: &str, still_failing: Option<&str>) {
    match still_failing {
        None => tracing::info!(
            attempt,
            character,
            "control-character encoding correction healed the step"
        ),
        Some(error) => tracing::warn!(
            attempt,
            character,
            error,
            "control-character encoding correction did not heal the step"
        ),
    }
}

/// What one repetition abort looked like, kept so the next one can be
/// recognised as the same fault rather than a fresh unlucky generation.
#[derive(Debug, Clone, PartialEq, Eq)]
struct WedgeFingerprint {
    /// The fragment the decoder was stuck on.
    token: String,
    /// Stream bytes received before the watch gave up.
    bytes: usize,
}

impl WedgeFingerprint {
    /// How far apart two abort lengths may sit and still be the same wedge,
    /// as a divisor of the longer one (1/64 ≈ 1.6%).
    ///
    /// Derived from the two distances it has to separate, not picked for
    /// feel. Same-wedge noise: a re-entered wedge reproduces its prefix to
    /// within a couple of bytes (≤3 on ~2780, ≈0.1%). Different-wedge
    /// distance: a different prompt's prefix has no relation to this one's,
    /// so it differs by hundreds or thousands of bytes. Any bound between
    /// those works; 1/64 sits ~15x above the noise and far below a real
    /// difference, which is what makes the result insensitive to the exact
    /// divisor rather than tuned to it.
    const LENGTH_TOLERANCE_DIVISOR: usize = 64;

    /// Whether `self` is the same wedge as `previous` — same stuck token,
    /// same place it gave up.
    ///
    /// Compared as a band, not an equality, and that is the whole point.
    /// Live evidence (2026-07-28, qwen3.5:9b, 12 aborts across one day):
    /// every abort was stuck on `"0"` and every one landed between 2777 and
    /// 2780 bytes, but consecutive attempts of a single step went 2780→2779,
    /// 2779→2778, 2780→2779, 2780→2780 and 2777→2779. Sampling leaves the
    /// prefix a byte or two different before it collapses, so requiring
    /// equal byte counts would have recognised ONE of those five pairs and
    /// left the other four to pay a full generation apiece.
    ///
    /// The token is the real fingerprint — the watch trips precisely because
    /// the decoder is stuck on one specific token — and the length only has
    /// to tell "the same collapse again" from "a different generation that
    /// also collapsed".
    fn is_repeat_of(&self, previous: &Self) -> bool {
        if self.token != previous.token {
            return false;
        }
        let tolerance = self.bytes.max(previous.bytes) / Self::LENGTH_TOLERANCE_DIVISOR;
        self.bytes.abs_diff(previous.bytes) <= tolerance
    }
}

/// Why the ladder is clearing the runner before the next attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WedgeReset {
    /// This abort is the one we just saw. Nothing is learned by generating
    /// against it again.
    Confirmed,
    /// The fault repeated, but not identically — the original ladder.
    Repeated,
}

/// Whether to clear the runner before the next attempt, and why.
///
/// `current` is the wedge the last attempt aborted on; `previous` is the one
/// the runner produced before it (from an earlier attempt, or an earlier
/// step — see [`record_wedge`]). Both are `None` unless the last failure was
/// a repetition abort carrying a fingerprint.
fn wedge_reset_due(
    attempt: usize,
    last_err: &str,
    current: Option<&WedgeFingerprint>,
    previous: Option<&WedgeFingerprint>,
) -> Option<WedgeReset> {
    if !wedged_runner_error(last_err) {
        return None;
    }
    // Same token, same abort point: the retry re-entered the same wedge, so
    // the free first retry is not buying evidence — it already arrived.
    if let (Some(cur), Some(prev)) = (current, previous)
        && cur.is_repeat_of(prev)
    {
        return Some(WedgeReset::Confirmed);
    }
    // Otherwise the original rule: reset once the fault has repeated at all.
    // This still covers "empty completion", which has no fingerprint to
    // compare.
    (attempt >= 2).then_some(WedgeReset::Repeated)
}

/// The last wedge seen from a given model's runner.
///
/// Global, and keyed by model, because a wedge is a property of the RUNNER
/// rather than of the caller: one Ollama server serves every step and every
/// chat turn, and [`reset_ollama_runner_for`] clears it for all of them. Same
/// reasoning as the generation slot in `nanna-llm`, which is keyed by base URL.
///
/// Surviving across steps is what makes the comparison worth anything. Within
/// one step the second wedge does not exist until `attempt` reaches 2 —
/// exactly where the old ladder already reset — so a same-step-only check
/// could never fire earlier than the rule it exists to pre-empt. Across steps
/// it can, and that is the observed shape: 2026-07-28 18:53:53 aborted on one
/// step and 18:55:23 aborted identically on the FIRST try of the next, which
/// then spent a second full generation before the ladder acted at 18:55:32.
fn wedge_store() -> &'static std::sync::Mutex<HashMap<String, WedgeFingerprint>> {
    static WEDGES: std::sync::OnceLock<std::sync::Mutex<HashMap<String, WedgeFingerprint>>> =
        std::sync::OnceLock::new();
    WEDGES.get_or_init(|| std::sync::Mutex::new(HashMap::new()))
}

/// Record `wedge` as the newest one for `model`, returning the one it
/// replaced — i.e. the wedge to judge it against.
///
/// A poisoned lock carries no invalid state here (the map is only inserted
/// into and removed from), so recover rather than propagate a panic into the
/// retry path.
fn record_wedge(model: &str, wedge: WedgeFingerprint) -> Option<WedgeFingerprint> {
    wedge_store()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(model.to_string(), wedge)
}

/// Forget a model's last wedge, because its runner was just cleared.
///
/// Without this a reset would leave its own fingerprint behind, and the first
/// wedge on the FRESH runner would match it and reset again immediately — the
/// free retry has to come back when the state it was spent on is gone.
fn forget_wedge(model: &str) {
    wedge_store()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .remove(model);
}

/// Forget every recorded wedge: a server restart clears every model's runner,
/// not just the one that faulted.
fn forget_all_wedges() {
    wedge_store()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clear();
}

/// A failed step attempt: the message the ladder logs and classifies, plus
/// the wedge fingerprint when the failure was one.
///
/// [`AgentStepRunner::try_run_step`] used to return a bare `String`, and that
/// was the one lossy boundary between the repetition watch — which knows the
/// stuck token and the abort length exactly — and the retry ladder that has
/// to act on them. Everything in between (`LlmError` → `AgentError`) already
/// carried the structure.
struct StepAttemptError {
    message: String,
    wedge: Option<WedgeFingerprint>,
}

impl From<String> for StepAttemptError {
    fn from(message: String) -> Self {
        Self { message, wedge: None }
    }
}

impl StepAttemptError {
    /// Flatten an agent error to its message, keeping the wedge fingerprint
    /// when there is one.
    fn from_agent(e: &nanna_agent::AgentError) -> Self {
        let wedge = match e {
            nanna_agent::AgentError::Llm(nanna_llm::LlmError::WedgedRunner {
                token,
                bytes_received,
                ..
            }) => Some(WedgeFingerprint { token: token.clone(), bytes: *bytes_received }),
            _ => None,
        };
        Self { message: format!("step error: {e}"), wedge }
    }
}

/// The runner ran out of GPU memory.
///
/// On CUDA this does not arrive as a clean allocation failure — the prefill
/// compute buffer overruns and the driver reports an illegal memory access, so
/// matching only on "out of memory" misses the form it actually takes here.
fn gpu_memory_error(message: &str) -> bool {
    let m = message.to_ascii_lowercase();
    m.contains("illegal memory access")
        || m.contains("out of memory")
        || m.contains("cuda error")
        || m.contains("failed to allocate")
}

/// What to do about the `fault_ordinal`-th GPU-memory fault of a run (1-based).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GpuFaultAction {
    /// Unload the runner (`keep_alive: 0`) and retry at the SAME size.
    ResetOnly,
    /// Reset AND walk the `num_ctx` latch down a rung (floor-clamped).
    ResetAndDemote,
}

/// Blip tolerance for the demotion ladder.
///
/// Every fault gets the runner reset — reloading at the same size IS the
/// blip fix. Shrinking is the repeat fix, so demotion engages only from the
/// SECOND fault of a run. The old code demoted on the first: three runs had
/// died to the fault repeating until the run ended, but that evidence
/// predates the starting pin (PR #158, `NANNA_OLLAMA_NUM_CTX`). With a sane
/// pinned start a single fault is usually a transient load blip, and demoting
/// on it destroys a viable window — observed live 2026-08-03: exactly ONE
/// fault all run, 8192 halved to 4096 under a ~4.2k pressure-tier floor,
/// 38 below-floor stops, 8 resume cycles, run dead in 8 minutes. The second
/// fault of a run is the evidence the first one lacked, so it (and every
/// later one) acts immediately.
const fn gpu_fault_action(fault_ordinal: u32) -> GpuFaultAction {
    if fault_ordinal <= 1 {
        GpuFaultAction::ResetOnly
    } else {
        GpuFaultAction::ResetAndDemote
    }
}

#[async_trait::async_trait]
impl StepRunner for AgentStepRunner {
    /// The step's retry ladder on the run's active model, then — when that
    /// model gives up with a fault another model could get past — the same
    /// step on the next model in the chain (see [`ModelChain`]).
    async fn run_step(&self, request: StepRequest) -> Result<StepOutcome, String> {
        let Some(chain) = self.model_chain.as_ref() else {
            return self.run_step_ladder(request).await;
        };
        let mut request = request;
        loop {
            let model = chain.active();
            let result = if model == self.agent_config.model {
                self.run_step_ladder(request.clone()).await
            } else {
                self.on_model(&model).run_step_ladder(request.clone()).await
            };
            let err = match result {
                Ok(outcome) => return Ok(outcome),
                Err(err) => err,
            };
            let cancelled = request.cancel.as_ref().is_some_and(CancelToken::is_cancelled);
            if cancelled || !warrants_model_fallback(&err) {
                return Err(err);
            }
            let Some(next) = chain.advance_past(&model, &self.router) else {
                return Err(err);
            };
            tracing::warn!(
                failed = %model,
                next = %next,
                error = %err,
                "model gave up on this step — falling back to the next model in the priority list"
            );
            if let Some(sink) = &chain.announce {
                sink.model_fallback(&model, &next, &err);
            }
            // The failed attempt may have run tools the next model's fresh
            // context cannot show: the same re-anchor the ladder uses.
            if is_transient_llm_error(&err) {
                request
                    .prompt
                    .push_str(&transient_retry_note(1, transient_fault_kind(&err)));
                if let Some(line) = self.carried_side_effects_line() {
                    request.prompt.push_str(&line);
                }
            }
        }
    }
}

impl AgentStepRunner {
    /// The model this run is on now: the chain's active model, or the
    /// configured one when the run has no fallback chain.
    #[must_use]
    pub fn active_model(&self) -> String {
        self.model_chain
            .as_ref()
            .map_or_else(|| self.agent_config.model.clone(), |chain| chain.active())
    }

    /// This runner moved onto `model`: same run state, the model swapped the
    /// way a chat pin swaps it (routing tiers cleared, or iteration 2 would
    /// route straight back to the model that just failed).
    fn on_model(&self, model: &str) -> Self {
        let mut runner = self.clone();
        crate::agent_service::apply_chat_model_override(&mut runner.agent_config, model);
        runner
    }

    /// One step on `agent_config.model`: attempts plus the in-place retry
    /// ladder for transient faults.
    async fn run_step_ladder(&self, request: StepRequest) -> Result<StepOutcome, String> {
        let mut last_err = String::new();
        let mut nudge_pending = false;
        // Set once a transient fault has interrupted THIS step, and holds the
        // classified kind of the newest one. Every later attempt of the step
        // carries the re-anchor: the effects of the interrupted attempt stay
        // on disk for the rest of the step, so the warning stays true.
        let mut transient_fault: Option<&'static str> = None;
        // The character the provider named when it refused to parse a tool
        // call, carried into the next attempt's prompt as a correction.
        let mut parse_fault_char: Option<String> = None;
        // The wedge the last attempt aborted on, and the one the runner
        // produced before it — the pair `wedge_reset_due` judges.
        let mut cur_wedge: Option<WedgeFingerprint> = None;
        let mut prev_wedge: Option<WedgeFingerprint> = None;
        for attempt in 0..=STEP_LLM_RETRIES {
            if attempt > 0
                && !self
                    .prepare_retry(&request, attempt, &last_err, cur_wedge.as_ref(), prev_wedge.as_ref())
                    .await
            {
                return Err(last_err);
            }
            // A fresh context per attempt: the re-anchor makes retries free —
            // there is no partial transcript worth salvaging. After a
            // no-progress attempt the prompt carries a nudge, so the retry is
            // never a verbatim repeat of the request that just stalled.
            //
            // The same site carries the other two corrections, for the same
            // reason and at the same cost: a re-anchor when a provider fault
            // interrupted this step mid-flight (its tool effects are on disk
            // and the fresh context cannot show them), and the encoding
            // correction when the provider refused to parse a tool call.
            let mut attempt_request = request.clone();
            if nudge_pending {
                attempt_request.prompt.push_str(NO_PROGRESS_NUDGE);
            }
            if let Some(fault) = transient_fault {
                attempt_request
                    .prompt
                    .push_str(&transient_retry_note(attempt + 1, fault));
                if let Some(line) = self.carried_side_effects_line() {
                    attempt_request.prompt.push_str(&line);
                }
            }
            // The correction riding on THIS attempt, so its outcome can be
            // logged the way the unserved-tool heal logs its own.
            let heal_in_flight = parse_fault_char.clone();
            if let Some(character) = heal_in_flight.as_deref() {
                attempt_request
                    .prompt
                    .push_str(&control_char_transport_note(character));
            }
            // Each attempt is judged on its own: only a repetition abort
            // refills these, so an interleaved network blip cannot leave a
            // stale pair behind for the next wedge to match against.
            cur_wedge = None;
            prev_wedge = None;
            match self.try_run_step(&attempt_request).await {
                Ok(outcome) if is_empty_completion(&outcome) => {
                    dump_empty_step(&request, attempt);
                    last_err =
                        "empty completion (no text, no tool calls, ~0 tokens) from provider"
                            .to_string();
                }
                // Thought but did not act: not a result. Retry with the nudge
                // rather than letting the item be marched forward on nothing.
                Ok(outcome) if made_no_progress(&outcome) => {
                    tracing::warn!(
                        attempt,
                        output_tokens = outcome.output_tokens,
                        "step produced reasoning but no tool call and no text — retrying with a nudge"
                    );
                    nudge_pending = true;
                    last_err =
                        "no-progress step (reasoning only: no tool call, no answer)".to_string();
                }
                Ok(outcome) => {
                    if let Some(character) = heal_in_flight.as_deref() {
                        log_parse_heal_outcome(attempt, character, None);
                    }
                    return Ok(outcome);
                }
                Err(e) if is_transient_llm_error(&e.message) => {
                    // Remember this wedge so the NEXT one can be recognised
                    // as the same fault. `record_wedge` hands back what it
                    // replaced, which is the previous attempt's wedge within
                    // a step and the previous step's across one.
                    if let Some(wedge) = e.wedge {
                        prev_wedge = record_wedge(&self.agent_config.model, wedge.clone());
                        cur_wedge = Some(wedge);
                    }
                    // The step is being retried after a fault that may have
                    // executed tool calls this attempt's context will never
                    // show — the next attempt says so out loud.
                    transient_fault = Some(transient_fault_kind(&e.message));
                    // A named-character parse refusal is a fault in how the
                    // model ENCODED the call, so the next attempt carries the
                    // correction as well as the re-anchor.
                    if let Some(character) = ollama_generation_parse_error(&e.message) {
                        tracing::warn!(
                            attempt,
                            character = %character,
                            error = %e.message,
                            "provider refused to parse a tool call over a raw control \
                             character — retrying with the encoding correction"
                        );
                        parse_fault_char = Some(character);
                    }
                    last_err = e.message;
                }
                Err(e) => {
                    if let Some(character) = heal_in_flight.as_deref() {
                        log_parse_heal_outcome(attempt, character, Some(&e.message));
                    }
                    return Err(e.message);
                }
            }
            if let Some(character) = heal_in_flight.as_deref() {
                log_parse_heal_outcome(attempt, character, Some(&last_err));
            }
        }
        Err(last_err)
    }

    /// Get ready for retry `attempt` of a step: honour a Stop, back off, and
    /// clear a wedged or out-of-memory runner. Returns `false` when the user
    /// cancelled, in which case the step ends with the last error.
    async fn prepare_retry(
        &self,
        request: &StepRequest,
        attempt: usize,
        last_err: &str,
        cur_wedge: Option<&WedgeFingerprint>,
        prev_wedge: Option<&WedgeFingerprint>,
    ) -> bool {
        // Stop pressed while the failed attempt ran: don't sleep out
        // a backoff and burn a whole fresh step after the user asked
        // to abort.
        if request
            .cancel
            .as_ref()
            .is_some_and(CancelToken::is_cancelled)
        {
            return false;
        }
        tracing::warn!(attempt, error = %last_err, "retrying step after transient LLM error");
        let backoff = STEP_RETRY_BACKOFF_SECS[attempt - 1];
        let sleep = tokio::time::sleep(std::time::Duration::from_secs(backoff));
        // A Stop arriving mid-backoff aborts the sleep too — the
        // user is not waiting on a retry they just cancelled.
        if let Some(token) = request.cancel.as_ref() {
            tokio::select! {
                biased;
                () = token.cancelled() => return false,
                () = sleep => {}
            }
        } else {
            sleep.await;
        }
        // A WEDGED runner does not recover by being asked again: its
        // state has to be cleared. Retrying one three times against
        // the same wedge is how a run dies — observed 2026-07-27,
        // 17 repetition aborts collapsing into back-to-back
        // attempt=1,2,3 failures until the run ended.
        //
        // Two ways in. The fingerprint says this abort IS the last
        // one (same stuck token, same abort point), which is already
        // the evidence a retry would have bought — so act now.
        // Failing that, the original ladder: reset from the SECOND
        // retry, by which point the fault has repeated once, leaving
        // the first retry free for a genuinely transient drop.
        if let Some(reason) = wedge_reset_due(attempt, last_err, cur_wedge, prev_wedge) {
            // The reset logs whether it went ahead: it declines a server that
            // is not on this machine.
            tracing::warn!(
                attempt,
                ?reason,
                "wedged runner — asking for a runner reset before the next attempt"
            );
            self.reset_ollama_runner().await;
        }
        // Out of VRAM: reset on EVERY fault (reloading at the same
        // size is the blip fix), demote only once the fault REPEATS
        // within the run (shrinking is the repeat fix) — see
        // `gpu_fault_action` for why the first fault gets tolerance
        // the old code denied it. The demotion is clamped to the
        // run's measured minimum viable window, so a fault can never
        // be "healed" into a size the floor check refuses.
        if gpu_memory_error(last_err) {
            self.handle_gpu_fault(request).await;
        }
        true
    }

    /// One GPU memory fault: reset the runner, and demote the context window
    /// once the fault repeats within the run.
    async fn handle_gpu_fault(&self, request: &StepRequest) {
        let fault = self
            .gpu_fault_count
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            + 1;
        match gpu_fault_action(fault) {
            GpuFaultAction::ResetOnly => {
                tracing::warn!(
                    gpu_faults = fault,
                    "first GPU memory fault of this run — resetting \
                     the runner at the SAME size; a single fault at \
                     a sane starting pin is a load blip until a \
                     second fault proves otherwise"
                );
            }
            GpuFaultAction::ResetAndDemote => {
                // The clamp: the smallest window THIS run's steps
                // stay viable in (pressure-tier floor solved for
                // the window).
                let floor = nanna_agent::min_viable_num_ctx(
                    &self.tools,
                    &self.system_prompt,
                    self.workspace_context.as_deref(),
                    self.workspace_root.as_deref(),
                    &request.prompt,
                    self.agent_config.max_tokens as usize,
                )
                .await;
                // demote_context announces old → new and
                // clamped-or-not; `None` means the latch already
                // sits at the floor — the retry then reloads at
                // the same size and, if the fault persists, the
                // existing loud below-floor/fault failure
                // surfaces (resumable) instead of a manufactured
                // unusable window.
                let demoted =
                    demote_context_on_router(&self.router, &self.agent_config.model, Some(floor));
                tracing::warn!(
                    gpu_faults = fault,
                    min_viable_ctx = floor,
                    demoted_to = ?demoted,
                    "repeat GPU memory fault — demotion ladder engaged"
                );
            }
        }
        // The reset rides along on every fault, demoted or not.
        self.reset_ollama_runner().await;
    }

    /// The re-anchor's concrete half: the side-effecting calls the
    /// INTERRUPTED ATTEMPT made, read from the P22 Tier 4 liveness ledger and
    /// bounded by a share of this model's live input budget.
    ///
    /// No counter gate is needed to keep an earlier turn's history from being
    /// claimed as this attempt's work: the ledger clears the list at every
    /// step boundary, and a step boundary is stamped on entry to every
    /// attempt, so an empty list means this attempt changed nothing. `None`
    /// on runs with no liveness handle (task-run manager, tests), where the
    /// general re-anchor stands alone.
    fn carried_side_effects_line(&self) -> Option<String> {
        let live = self.chat_sink.as_ref()?.liveness.as_ref()?;
        let marks = live.attempt_side_effects();
        if marks.is_empty() {
            return None;
        }
        let budget = carried_side_effect_budget_bytes(self.live_hard_input_limit());
        // The pointer at the stored outputs is only honest when this run has a
        // memory service writing them — `memory_sink` is the same gate.
        Some(carried_side_effect_note(&marks, budget, self.memory.is_some()))
    }

    /// This runner's live hard input limit, in tokens.
    ///
    /// Computed exactly as [`nanna_agent::AgentContext`] computes the limit
    /// the request is actually judged against (`configure_for_model_with_output`):
    /// model info from the cache-or-universal-floor — clamped to the `num_ctx`
    /// of the server the router sends the model to, so a mid-run VRAM
    /// demotion is reflected — paired with this runner's own output
    /// reservation. Deriving it a second way would let the budget and the
    /// window it is a share of disagree.
    fn live_hard_input_limit(&self) -> usize {
        let info = cached_model_info_on_router(&self.router, &self.agent_config.model);
        let reserve = info.effective_output_budget(self.agent_config.max_tokens as usize);
        info.hard_input_limit_for(reserve)
    }

    /// Seed the step's FRESH context with what this scope has already proven —
    /// the do-not-regress digest, in the never-compressed slot.
    ///
    /// Why here and not only in the planner: verified state must reach the
    /// model that EDITS. Each step gets a brand-new `AgentContext`, so
    /// everything the previous step proved is gone by construction; and inside
    /// a step, every summarization/distillation pass can collapse ordinary
    /// history. `verified_outcomes` is the one slot proven to survive every
    /// compression path (P22 Tier 2), so the digest is seeded there and carry
    /// is session-scoped by PERSISTENCE — the store — rather than by keeping a
    /// long-lived context alive.
    ///
    /// Source is the same store read the turn-start `established_work_context`
    /// block uses (`established_rows`), which is newest-completion-first and
    /// already bounded by that module's `ESTABLISHED_MAX` — so the bound is
    /// reused, not re-declared. Collapse-to-×N and the monotone timestamp rule
    /// are the slot's own (`record_verified_outcome_at`).
    ///
    /// Chat scope is (`session`, `session_id`), which is exactly the scope the
    /// chat harness seeds and reads. Background task runs have no chat sink and
    /// therefore no session to read — they keep today's behavior.
    async fn seed_verified_outcomes(&self, context: &mut nanna_agent::AgentContext) {
        let Some(sink) = &self.chat_sink else {
            return;
        };
        let Some(storage) = &sink.storage else {
            return;
        };
        let rows = crate::control::chat_harness::established_rows(
            storage,
            "session",
            Some(&sink.session_id),
        )
        .await;
        for row in &rows {
            // Only VERIFIED completions are facts proven by execution; an
            // unverified close is a claim, and the slot's whole contract is
            // that everything in it was run.
            let Some(verdict) = &row.verdict else {
                continue;
            };
            // The subject a model can act on: the command the check runs when
            // there is one, the item otherwise.
            let subject = row
                .acceptance
                .as_ref()
                .and_then(|value| {
                    nanna_agent::harness::AcceptanceCheck::from_json(value).ok()
                })
                .and_then(|check| check.self_check_command().map(str::to_string))
                .unwrap_or_else(|| format!("#{} {}", row.id, row.title));
            // The store's own completion time, never "now": a fact verified in
            // an earlier turn must not claim to have been verified this step.
            let verified_at = chrono::DateTime::parse_from_rfc3339(&row.when)
                .map(|t| t.timestamp())
                .unwrap_or_default();
            context.record_verified_outcome_at(subject, verdict.clone(), verified_at);
        }
        if !rows.is_empty() {
            tracing::debug!(
                seeded = context.verified_outcomes.len(),
                rows = rows.len(),
                "seeded the step context's never-compressed verified-outcome slot from \
                 the scope's stored verdicts"
            );
        }
    }

    /// Put the step's own output — its answer, and each reasoning block — into
    /// memory, in the same `[kind → subject]` shape the episodic tool writer
    /// uses, so answers, thoughts and actions all read alike on recall.
    ///
    /// Reasoning is stored per BLOCK rather than as one lump: the blocks are
    /// already interleaved between tool calls, so one block is one thought
    /// about one action — which is the unit a later step actually wants back.
    async fn remember_step_narration(
        &self,
        request: &StepRequest,
        result: &nanna_agent::AgentResponse,
    ) {
        let Some(sink) = self.memory_sink() else {
            return;
        };
        let label = &request.item_title;

        let text = result.text.trim();
        if !text.is_empty() {
            let mut tags = HashMap::new();
            tags.insert("kind".to_string(), "assistant_text".to_string());
            tags.insert("step".to_string(), label.clone());
            sink(nanna_agent::ExtractedMemory {
                content: format!("[said → {label}] {text}"),
                category: "assistant_text".to_string(),
                provenance: nanna_agent::MemoryProvenance::Observed,
                tags: Some(tags),
            })
            .await;
        }

        let Some(reasoning) = result.reasoning.as_ref() else {
            return;
        };
        // Providers that do not split reasoning still fill `content`; treat
        // that as a single block rather than storing nothing.
        let blocks: Vec<&str> = if reasoning.blocks.is_empty() {
            vec![reasoning.content.as_str()]
        } else {
            reasoning.blocks.iter().map(|b| b.content.as_str()).collect()
        };
        let total = blocks.len();
        for (idx, block) in blocks.into_iter().enumerate() {
            let thought = block.trim();
            if thought.is_empty() {
                continue;
            }
            let mut tags = HashMap::new();
            tags.insert("kind".to_string(), "thinking".to_string());
            tags.insert("step".to_string(), label.clone());
            tags.insert("block".to_string(), format!("{}/{total}", idx + 1));
            sink(nanna_agent::ExtractedMemory {
                content: format!("[thought → {label}] {thought}"),
                category: "thinking".to_string(),
                provenance: nanna_agent::MemoryProvenance::Observed,
                tags: Some(tags),
            })
            .await;
        }
    }

    /// The sink that puts tool results into memory, or `None` when no memory
    /// service exists (then `loop_runner` keeps results in context as before,
    /// which is the correct degradation — never silently lose the output).
    fn memory_sink(&self) -> Option<nanna_agent::MemoryCallback> {
        let service = self.memory.clone()?;
        let workspace_id = self.workspace_id.clone();
        Some(Box::new(move |memory: nanna_agent::ExtractedMemory| {
            let service = service.clone();
            let workspace_id = workspace_id.clone();
            Box::pin(async move {
                // Filter, importance and route all live in `memory_adapter` —
                // ONE copy, shared with the interactive-chat sink in
                // `agent_service.rs`. This path only decides what to log.
                let category = memory.category.clone();
                let bytes = memory.content.len();
                match crate::memory_adapter::store_extracted_memory(
                    &service,
                    memory,
                    workspace_id,
                )
                .await
                {
                    // INFO, not DEBUG. The daemon runs at INFO, so the old
                    // debug! meant 704 discarded writes left no operator-visible
                    // trace at all in a run whose store held 90 rows — the
                    // shortfall was invisible until someone counted by hand.
                    // A dropped memory is a fact about the run, not a detail.
                    Ok(None) => tracing::info!(
                        category = %category,
                        bytes,
                        "dropping low-signal memory (machine noise)"
                    ),
                    Ok(Some(_)) => {}
                    Err(e) => {
                        tracing::warn!(error = %e, "failed to store tool result in memory");
                    }
                }
            })
        }))
    }

    /// Whether this runner's model is served by Ollama. Healing must be
    /// provider-aware: a `:free` suffix on an `OpenRouter` model id must never
    /// trigger local-server surgery. (Where that server runs is the healers'
    /// own check: they act on processes only for a server on this machine.)
    #[must_use]
    pub fn is_ollama_model(&self) -> bool {
        crate::llm_router::ProviderId::from_model(&self.agent_config.model)
            == crate::llm_router::ProviderId::Ollama
    }

    /// Force Ollama to unload the model (`keep_alive: 0`), clearing runner
    /// state — the observed degraded mode restores a stale KV context
    /// checkpoint that sends every generation straight to a stop token, and
    /// only a fresh runner clears it. No-op for non-Ollama models, and for an
    /// Ollama server not on this machine.
    pub async fn reset_ollama_runner(&self) {
        reset_ollama_runner_for(&self.router, &self.agent_config.model).await;
    }

    /// The agent for one step: the runner's config with a step-appropriate
    /// temperature, bound to the provider that serves the model.
    fn step_agent(
        &self,
        step_kind: nanna_agent::harness::StepKind,
        context: nanna_agent::AgentContext,
    ) -> Result<nanna_agent::Agent, StepAttemptError> {
        let mut config = self.agent_config.clone();
        // Execution/verification wants determinism (pass^k reliability on a
        // small model); planning keeps the configured creative temperature.
        if matches!(
            step_kind,
            nanna_agent::harness::StepKind::Execute | nanna_agent::harness::StepKind::Verify
        ) {
            config.temperature = config.temperature.min(0.3);
        }
        let model_display = config.model.clone();
        let llm_client = self
            .router
            .client_for_model(&config.model)
            .ok_or_else(|| format!("No provider available for model '{model_display}'"))?;
        config.model = LlmRouter::strip_model_prefix(&config.model);

        // Summarizers resolve through the chat router, per call: the same
        // grammar and credentials as chat, and a reload reaches a running step.
        let mut agent = nanna_agent::Agent::new(config, llm_client, self.tools.clone())
            .with_context(context)
            .with_summarizer_clients(crate::llm_router::summarizer_clients(&self.router));
        if let Some(tracker) = &self.stats {
            agent = agent.with_stats(tracker.clone());
        }
        Ok(agent)
    }

    async fn try_run_step(
        &self,
        request: &StepRequest,
    ) -> Result<StepOutcome, StepAttemptError> {
        use nanna_agent::{AgentContext, RunOptions};

        // Name the item before its output starts arriving, so a multi-step
        // run reads as a sequence of labelled pieces of work rather than one
        // undifferentiated stream.
        if let Some(sink) = &self.chat_sink {
            sink.step_header(request);
        }

        let mut context = AgentContext::new(uuid::Uuid::new_v4().to_string())
            .with_system_prompt(&self.system_prompt);
        if let Some(ws_root) = &self.workspace_root {
            context.workspace_root = Some(ws_root.clone());
        }
        if let Some(ws_ctx) = &self.workspace_context {
            context.workspace_context = Some(ws_ctx.clone());
        }
        self.seed_verified_outcomes(&mut context).await;

        let agent = self.step_agent(request.step_kind, context)?;

        // The plan's `tool_scope` does NOT gate anything. It is a guess made
        // before the work started, and every time it has been enforced it has
        // cost a run: scoped to a hallucinated "file" the model had nothing to
        // work with (7/42), scoped to "exec" it shell-heredoc'd around every
        // write guard (141 exec / 0 writes). A step reaches whatever the work
        // turns out to need, through `discover_tools`.
        if !request.tool_scope.is_empty() {
            tracing::debug!(
                scope = %request.tool_scope.join(", "),
                "plan named tools for this step — treated as a hint, not a restriction"
            );
        }

        // Context is the scarce resource, so the REQUEST carries only the core
        // pair; the model activates the rest by discovering them, and what it
        // activates persists for the run.
        // Carry forward everything discovered earlier in THIS run. The doc
        // above promised activation persists; `RunState::new()` per step meant
        // it never did, so each step re-paid discovery out of its 8-iteration
        // budget and then ran out — 40 of 45 steps in one run died at the cap
        // with no text at all. Now discovery is paid once per tool per run,
        // which is what the design said all along.
        let active: Vec<String> = {
            let discovered = self.discovered_tools.read().await;
            merge_active_tools(CORE_TOOLS, &discovered, &self.user_selected_tools)
        };
        let restrict_to_active = true;

        let mut options = RunOptions {
            max_iterations: request.max_iterations,
            token_budget: request.token_budget,
            max_wall_clock: request.max_wall_clock,
            // The Stop button's token. Without this the agent loop streams
            // and executes tools until the step ends on its own — observed
            // live (2026-07-30): Stop had no effect for minutes because the
            // flag was only polled between harness steps, and a chat turn is
            // usually ONE step.
            cancel: request.cancel.clone(),
            step_kind: Some(request.step_kind),
            // Task anchor for injected steering texts (breaker notices,
            // nudges): the live item title keeps a mid-run meta-instruction
            // reading as an aside within THIS task instead of a conversation
            // reset. An empty title falls back (inside the loop) to the
            // step prompt's goal line rather than fabricating a header.
            task_anchor: Some(request.item_title.clone()).filter(|t| !t.trim().is_empty()),
            // The RUN's breaker ledger, shared for the same structural reason
            // `discovered_tools` is: this runner is one object for the whole
            // run, while each step gets a fresh `RunState`. Per step, a model
            // repeating itself 2-3 times never reached the breaker threshold
            // and the counter reset at every boundary (observed live
            // 2026-08-02, session 05775d1d: 22 steps, 79 identical
            // `explore {}` calls, 3 short-circuits).
            repeat_ledger: Some(Arc::clone(&self.repeat_ledger)),
            attachments: self.attachments.as_ref().clone(),
            initial_active_tools: active,
            tool_activation: nanna_agent::ToolActivation {
                restrict_to_active_tools: restrict_to_active,
                ..nanna_agent::ToolActivation::default()
            },
            is_sub_agent: true,
            // Tool results land in memory; context keeps only the stub.
            on_memory: self.memory_sink(),
            // Capability transitions (provider benched, writes queued) reach
            // the model once, in its next tool result.
            degradations: self.degradations.clone(),
            ..Default::default()
        };
        // Show the work as it happens: the step's text, thinking, and tool
        // activity stream through the same events a plain chat turn uses,
        // and fill the registered run buffers for recovery/persistence.
        if let Some(sink) = &self.chat_sink {
            sink.attach_stream_callbacks(&mut options);
        }

        let result = agent
            .run(&request.prompt, options)
            .await
            // The one place the structured LLM error used to be flattened to
            // prose. The wedge fingerprint is lifted out here instead.
            .map_err(|e| StepAttemptError::from_agent(&e))?;

        // The step's OWN words belong in memory too. Tool results were being
        // captured; the model's answer and its reasoning were not captured
        // anywhere. Assistant text reached memory at most once per user `Send`,
        // from a tail block that runs only after the whole harness run ends —
        // which a multi-hour run never reaches — and thinking blocks had no
        // path at all: ~2500 produced, 0 stored, in one measured run.
        //
        // Captured verbatim and directly rather than by flipping
        // `auto_extract_memories`, which spends an extra LLM call per step to
        // produce a lossy semantic digest. What is wanted here is the record,
        // not a précis of it — compressing is dreaming's job, done later with
        // the whole corpus in view rather than one step at a time.
        // Fold this step's discoveries into the run so the next step starts
        // with them already in hand.
        if !result.active_tools.is_empty() {
            let mut discovered = self.discovered_tools.write().await;
            for tool in &result.active_tools {
                discovered.insert(tool.clone());
            }
        }

        self.remember_step_narration(request, &result).await;

        let tool_calls = result
            .tool_calls
            .iter()
            .map(|record| StepToolCall {
                name: record.name.clone(),
                input_digest: digest(&record.input.to_string()),
                output_digest: digest(&record.output),
                success: record.success,
            })
            .collect();

        // Paths the step's write/edit calls targeted, for the harness's
        // mid-run sweep: a write that lands on a file some verified item's
        // acceptance references triggers an immediate re-check of that item.
        let mut touched_paths: Vec<String> = Vec::new();
        for record in &result.tool_calls {
            if let Some(path) = touched_path_of(&record.name, &record.input)
                && !touched_paths.contains(&path)
            {
                touched_paths.push(path);
            }
        }

        Ok(StepOutcome {
            text: result.text,
            input_tokens: u64::from(result.input_tokens),
            output_tokens: u64::from(result.output_tokens),
            tool_calls,
            touched_paths,
            degenerate_loop: result.degenerate_loop,
        })
    }
}

// ---------------------------------------------------------------------------
// Planning (P19): every chat turn becomes a plan the harness can execute
// ---------------------------------------------------------------------------

/// Wall-clock ceiling for one planning call.
///
/// Bound justification: planning sits in front of *every* chat turn, so it is
/// on the latency path of "hi". The reference local tier answers a short
/// structured prompt in single-digit seconds; 30s covers a cold model load
/// without letting a wedged provider hold a turn hostage — on timeout the
/// caller falls back to the single-task plan and the turn proceeds.
const PLAN_TIMEOUT_SECS: u64 = 30;

/// Iterations allowed inside a planning step.
///
/// Bound justification: planning emits one JSON array and calls no tools.
/// More than one iteration means the model is looping, not planning.
const PLAN_ITERATIONS: usize = 1;

/// Turns a request into a plan using the configured model.
///
/// Wraps [`AgentStepRunner`] rather than duplicating its provider handling:
/// planning is just a step whose prompt asks for JSON and whose tool scope is
/// empty.
pub struct AgentPlanner {
    pub runner: Arc<dyn StepRunner>,
}

impl AgentPlanner {
    #[must_use]
    pub const fn new(runner: Arc<dyn StepRunner>) -> Self {
        Self { runner }
    }

    /// Plan `goal`, degrading to the single-task plan on any failure.
    ///
    /// Never returns an error: a planner problem must not cost the user a
    /// turn. The returned `Plan::origin` records which path was taken.
    ///
    /// `cancel` is the run's Stop token: planning sits in front of every
    /// turn, so a Stop during it must abort the planning call immediately
    /// rather than wait out `PLAN_TIMEOUT_SECS` (observed live 2026-07-31:
    /// Stop during planning hung the turn for the full 30s window). The
    /// caller decides what a cancelled plan means — chat skips seeding and
    /// running it.
    pub async fn plan(&self, goal: &str, context: Option<&str>, cancel: Option<&CancelToken>) -> Plan {
        if cancel.is_some_and(CancelToken::is_cancelled) {
            return Plan::single(goal);
        }
        // Two attempts, not one: even with tool definitions stripped from
        // Plan steps, a local model can emit an empty completion on the
        // single planning iteration, and one silent fallback here cost a
        // whole mission (observed 2026-08-08: every plan degraded to the
        // fallback monolith and the turn died dry at 13/42). The retry is
        // bounded to one — a planner that answers empty twice is not going
        // to speak on the third ask, and the fallback keeps the turn alive.
        for attempt in 0..2 {
            if cancel.is_some_and(CancelToken::is_cancelled) {
                return Plan::single(goal);
            }
            let request = StepRequest {
                // Planning is not attached to an item yet; the store assigns
                // ids when the plan is seeded.
                item_id: 0,
                step_index: 0,
                step_kind: nanna_agent::harness::StepKind::Plan,
                item_title: "Planning".to_string(),
                prompt: build_plan_prompt(goal, context),
                tool_scope: Vec::new(),
                token_budget: None,
                max_iterations: Some(PLAN_ITERATIONS),
                max_wall_clock: Some(std::time::Duration::from_secs(PLAN_TIMEOUT_SECS)),
                // Threaded so the in-step LLM call aborts on Stop too.
                cancel: cancel.cloned(),
            };

            let planning = tokio::time::timeout(
                std::time::Duration::from_secs(PLAN_TIMEOUT_SECS),
                self.runner.run_step(request),
            );
            let outcome = if let Some(token) = cancel {
                tokio::select! {
                    biased;
                    outcome = planning => outcome,
                    () = token.cancelled() => {
                        tracing::info!("planning cancelled — degrading to the single-task plan");
                        return Plan::single(goal);
                    }
                }
            } else {
                planning.await
            };

            match outcome {
                Ok(Ok(step)) => {
                    if step.text.trim().is_empty() && attempt == 0 {
                        tracing::warn!(
                            "planner answered with empty text — retrying the planning call once"
                        );
                        continue;
                    }
                    return plan_or_fallback(goal, &step.text);
                }
                Ok(Err(message)) => {
                    tracing::warn!(%message, "planner failed - falling back to a single-task plan");
                    return Plan::single(goal);
                }
                Err(_) => {
                    tracing::warn!(
                        timeout_secs = PLAN_TIMEOUT_SECS,
                        "planner timed out - falling back to a single-task plan"
                    );
                    return Plan::single(goal);
                }
            }
        }
        // Unreachable in practice (attempt 1 always returns), but the loop
        // shape cannot prove it: both attempts empty degrades to fallback.
        Plan::single(goal)
    }
}

/// Seed a plan into the store, returning the created task ids in plan order.
///
/// Every seed sorts strictly below the scope's current minimum `sort_order`:
/// the plan being seeded is the user's CURRENT request, and leftovers from
/// earlier turns are information for the planner, not work that outranks it
/// (observed live 2026-08-02: a fresh question seeded at sort 0 tied a
/// cancelled turn's leftovers and lost on id, so the stale work ran first).
///
/// `jump_queue` is how an interjected message reaches the front of a LIVE
/// run: the store orders `next()` by `in_progress` first, then priority,
/// then `sort_order`, so an interjection additionally (a) takes priority 1 -
/// the user speaking mid-run outranks anything already planned - and (b)
/// must not be outranked by an item the harness already marked `in_progress`
/// - see `SessionInterjector::yield_current_item`.
///
/// # Errors
///
/// Returns an error when the scope's existing tasks cannot be listed, or when
/// not one planned task could be created (an empty plan included). Individual
/// rejected tasks are logged and skipped while at least one lands.
pub async fn seed_plan(
    storage: &Arc<Storage>,
    scope: &str,
    scope_id: Option<&str>,
    plan: &Plan,
    jump_queue: bool,
) -> Result<Vec<i64>, String> {
    let repo = storage.tasks();
    // Strictly below every existing item in the scope so this plan is
    // selected ahead of leftovers. Explicitly computed rather than hardcoded
    // to 0 - 0 collides with the default and merely ties.
    let existing = repo
        .list(scope, scope_id, false)
        .await
        .map_err(|e| e.to_string())?;
    let base_sort = existing.iter().map(|t| t.sort_order).min().unwrap_or(0)
        - crate::numeric::i64_saturating(plan.tasks.len())
        - 1;

    let mut ids = Vec::with_capacity(plan.tasks.len());
    for (index, task) in plan.tasks.iter().enumerate() {
        let new = NewTask {
            parent_id: None,
            scope: scope.to_string(),
            scope_id: scope_id.map(String::from),
            project: None,
            title: task.title.clone(),
            description: task.description.clone(),
            // 1 is the highest the store accepts; an interjection is the
            // user speaking, which outranks anything already planned.
            priority: if jump_queue { 1 } else { 2 },
            labels: vec!["chat".to_string()],
            tool_scope: task.tool_scope.clone(),
            due_at: None,
            deadline_at: None,
            recurrence: None,
            depends_on: Vec::new(),
            acceptance: task.acceptance.clone(),
            assignee: None,
            sort_order: base_sort + crate::numeric::i64_saturating(index),
        };
        match repo.create(new).await {
            Ok(created) => ids.push(created.id),
            // One rejected task must not sink the plan - the rest still runs.
            Err(e) => {
                tracing::warn!(title = %task.title, error = %e, "failed to seed planned task");
            }
        }
    }
    if ids.is_empty() {
        return Err("no planned task could be created".to_string());
    }
    Ok(ids)
}

/// Crude, deliberately SYMMETRIC plural fold: drop a trailing `s` from tokens
/// of 3+ characters that do not already end in `ss`.
///
/// Not a stemmer and not trying to be — the only plural that matters here is a
/// planner writing "PRs" one round and "PR" the next. Because the fold is
/// applied to both sides, an over-fold ("status" → "statu", "tests" → "test")
/// cannot make two DIFFERENT titles match unless they differ only in that
/// trailing `s`, which is precisely the case being folded. The failure it can
/// produce is collapsing a genuine singular/plural distinction ("close the
/// issue" vs "close the issues"); that is accepted, because a planner does not
/// use the plural `s` to name a different job.
fn fold_plural(word: &str) -> &str {
    if word.len() >= 3 && word.ends_with('s') && !word.ends_with("ss") {
        &word[..word.len() - 1]
    } else {
        word
    }
}

/// A title's tokens in ORDER: lowercased, split on non-alphanumerics, plurals
/// folded. Nothing is dropped — every word the planner wrote still has to be
/// there, in the same place.
fn title_tokens(title: &str) -> Vec<String> {
    title
        .split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(|word| fold_plural(&word.to_lowercase()).to_string())
        .collect()
}

/// Title identity — the same rule the `tasks.add` idempotent-reuse check
/// applies, so "is this the same task?" has exactly one definition.
///
/// **This predicate can DELETE WORK.** [`seed_continuation`] drops a proposed
/// task when its title matches one already closed this turn, and `tasks.add`
/// hands back an existing id instead of creating a new task. A missed
/// duplicate costs one extra continuation round; a WRONG match silently
/// discards a job nobody will ever do. The two errors are not comparable, so
/// the rule is deliberately conservative: it normalizes only what cannot
/// change meaning and matches nothing else.
///
/// Normalized away (none of it can name different work):
/// - surrounding whitespace and internal punctuation/separator runs,
/// - ASCII case,
/// - trailing-`s` plurals (`fold_plural`) — "merge the PRs" and "merge the
///   PR" are one job.
///
/// Everything else is load-bearing and compared verbatim, IN ORDER. Two titles
/// are the same work only when their normalized token sequences are equal.
/// Sequence rather than set because word order carries the subject: "merge
/// master into the PR" and "merge the PR into master" are the same multiset
/// and opposite jobs. A sequence comparison is strictly more conservative than
/// a set comparison, and conservative is the safe direction here.
///
/// **Known consequence, accepted on purpose.** An earlier revision of this
/// function dropped a filler list ("check", "verify", "list", "current",
/// "state", …) and matched on subsets, to catch the live 2026-08-03 mission
/// ("fix conflicts and merge all open prs") whose planner re-proposed finished
/// work in a new spelling every round — "check the current PR state", then
/// "enumerate the currently open PRs", then "check which PRs are still open".
/// That rule reduced those titles to `{pr}` and `{pr, open}`, which are
/// subsets of "merge the PRs" and "merge all open PRs": the filter would have
/// dropped the planner's proposal to actually MERGE as a duplicate of a
/// VERIFICATION task, and the mission would have converged reporting success
/// with the merges never done. Strictly worse than the bug it fixed.
///
/// So those three titles no longer match each other — they use different verbs
/// and name different work as far as any string rule can tell. Terminating
/// that mission is not this function's job: the acceptance PRE-CHECK
/// (`LongHorizonConfig::precheck_acceptance_items`) asks the environment
/// instead, and the environment is the only judge that cannot be fooled by a
/// synonym.
/// This predicate is the narrow safety net underneath it, not the terminator.
#[must_use]
pub fn same_title(a: &str, b: &str) -> bool {
    let left = title_tokens(a);
    let right = title_tokens(b);
    if left.is_empty() || right.is_empty() {
        // A title with no alphanumerics at all ("???") normalizes to nothing,
        // and "nothing == nothing" would match every such title with every
        // other. Fall back to exact comparison rather than invent identity.
        return a.trim().eq_ignore_ascii_case(b.trim());
    }
    left == right
}

/// Ids of every closed (`done` or `cancelled`) task in a scope.
///
/// Captured by the chat harness at TURN START as the baseline for
/// [`seed_continuation`]: only tasks that close after this snapshot count as
/// "work this turn already did". An id snapshot rather than a timestamp bound
/// because a cancelled task carries no completion stamp and the store's
/// datetime strings would need parsing — set difference is exact.
///
/// # Errors
///
/// Returns the store's error, as text, when the scope's tasks cannot be listed.
pub async fn closed_task_ids(
    storage: &Arc<Storage>,
    scope: &str,
    scope_id: Option<&str>,
) -> Result<HashSet<i64>, String> {
    Ok(storage
        .tasks()
        .list(scope, scope_id, true)
        .await
        .map_err(|e| e.to_string())?
        .into_iter()
        .filter(|t| t.status == "done" || t.status == "cancelled")
        .map(|t| t.id)
        .collect())
}

/// Tasks closed in a scope SINCE `baseline` was captured — the work this turn
/// has already finished or given up on.
///
/// The one definition of "this turn already worked that", shared by every
/// in-run item-creation path: the continuation planner
/// ([`seed_continuation`]) and the model's own write surface (`tasks.add`,
/// which the harness's replan step drives through the todo tool). They used
/// to disagree, and that disagreement WAS the hole — see [`TurnBaselines`].
///
/// Returns whole tasks rather than titles so the one caller that must NAME
/// what it refused (`tasks.add`, whose notice cites the settled task's id and
/// verdict) does not need a second query with a second chance to drift.
///
/// # Errors
///
/// Returns the store's error, as text, when the scope's tasks cannot be listed.
pub async fn tasks_closed_since<S: std::hash::BuildHasher + Sync>(
    storage: &Arc<Storage>,
    scope: &str,
    scope_id: Option<&str>,
    baseline: &HashSet<i64, S>,
) -> Result<Vec<Task>, String> {
    Ok(storage
        .tasks()
        .list(scope, scope_id, true)
        .await
        .map_err(|e| e.to_string())?
        .into_iter()
        .filter(|t| (t.status == "done" || t.status == "cancelled") && !baseline.contains(&t.id))
        .collect())
}

/// Scope identity for [`TurnBaselines`]. The unit separator (U+001F) cannot
/// appear in a scope name, so no (scope, `scope_id`) pair collides with a
/// differently-split one — the same idiom the loop runner's repeat key uses.
fn scope_key(scope: &str, scope_id: Option<&str>) -> String {
    format!("{scope}\u{1f}{}", scope_id.unwrap_or(""))
}

/// Turn-start closed-task baselines, keyed by scope — the shared boundary
/// that makes "closed since this turn started" answerable from anywhere.
///
/// Why it exists (observed live 2026-08-02, ollama/gemma4:12b, session
/// 05775d1d): [`seed_continuation`] filters continuation-planner titles that
/// closed this turn, but it is not the only path that creates items mid-run.
/// The harness's replan step tells the MODEL to decompose a stalled item into
/// subtasks through the todo tool, which lands in `tasks.add` — and that
/// dedups only against OPEN titles, exactly so a genuinely recurring chore
/// stays addable tomorrow. So a title the run had just ABANDONED was created
/// again, unfiltered, and immediately re-seeded: task #2059 (abandoned) →
/// #2060 (identical title). One guard covered one door.
///
/// The baseline is registered by the chat harness for the life of a turn and
/// removed on the same exit path that releases the run claim, so at most one
/// entry exists per session with a live run. Absent entry = no live turn =
/// no filtering: a scope with no registered baseline behaves exactly as
/// before, which is what a plain `tasks.add` from outside a run should do.
///
/// **What is guarded, and what deliberately is not.** The guard covers the
/// paths where the RUN proposes its own next work: the continuation planner
/// ([`seed_continuation`]) and the model's own writes (`tasks.add`, which the
/// replan step drives). It does NOT cover interjections — a mid-run user
/// message plans through [`seed_plan`] — because the user re-asking for
/// something the run just finished is new intent, not a run spinning on
/// itself, and silently dropping it would be the worse failure.
///
/// **The trade.** Matching is scope-wide, exactly as [`seed_continuation`]
/// matches, rather than parent-scoped like the OPEN-title reuse above. Two
/// genuinely different subtasks under different parents sharing one generic
/// title ("run the tests"), where one closed this turn, are refused — the
/// notice tells the model to retitle, which costs one call. The alternative
/// leaves the resurrection door open for exactly the shape that wedged the
/// run, since a replan re-parents freely.
#[derive(Debug, Default)]
pub struct TurnBaselines {
    inner: RwLock<HashMap<String, HashSet<i64>>>,
    /// The live turn's [`TurnAdmission`] per scope, so `tasks.add`'s reuse of
    /// an open item can admit it into the turn that asked for it.
    admissions: RwLock<HashMap<String, Arc<TurnAdmission>>>,
}

impl TurnBaselines {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record the turn-start baseline for a scope, replacing any stale entry
    /// (a previous turn in the same scope that never got to close cleanly).
    pub async fn open_turn(&self, scope: &str, scope_id: Option<&str>, closed: HashSet<i64>) {
        self.inner
            .write()
            .await
            .insert(scope_key(scope, scope_id), closed);
    }

    /// Drop a scope's baseline at turn end. Called from the same exit tail
    /// that releases the run claim, so a leaked entry would need the whole
    /// tail to be skipped — and the next turn's `open_turn` replaces it.
    pub async fn close_turn(&self, scope: &str, scope_id: Option<&str>) {
        let key = scope_key(scope, scope_id);
        self.inner.write().await.remove(&key);
        self.admissions.write().await.remove(&key);
    }

    /// Publish the live turn's admission for a scope (replacing a stale one).
    pub async fn register_admission(
        &self,
        scope: &str,
        scope_id: Option<&str>,
        admission: Arc<TurnAdmission>,
    ) {
        self.admissions
            .write()
            .await
            .insert(scope_key(scope, scope_id), admission);
    }

    /// The live turn's admission for a scope, or `None` when no turn is live
    /// or the turn admits the whole scope.
    pub async fn admission(
        &self,
        scope: &str,
        scope_id: Option<&str>,
    ) -> Option<Arc<TurnAdmission>> {
        self.admissions
            .read()
            .await
            .get(&scope_key(scope, scope_id))
            .cloned()
    }

    /// The turn-start baseline for a scope, or `None` when no turn is live.
    pub async fn baseline(&self, scope: &str, scope_id: Option<&str>) -> Option<HashSet<i64>> {
        self.inner
            .read()
            .await
            .get(&scope_key(scope, scope_id))
            .cloned()
    }
}

/// The notice `tasks.add` returns when a title closed earlier THIS TURN is
/// proposed again.
///
/// A returned notice, never an error (the repo rule: tool failures are
/// returned, not thrown), and it announces itself: WHAT happened (nothing was
/// created), WHY (this exact title was already finished or abandoned in this
/// same turn), and WHAT TO DO instead (move on, or add genuinely different
/// work). Naming the prior task id makes the claim checkable rather than
/// something the model has to take on faith.
#[must_use]
pub fn closed_this_turn_notice(title: &str, status: &str) -> String {
    let verdict = if status == "cancelled" {
        "abandoned"
    } else {
        "completed"
    };
    format!(
        "NOT created: \"{title}\" was already {verdict} earlier in this same turn. \
         Re-creating it would restart work this run has already settled, which is how a \
         turn spins instead of finishing. Nothing was written and nothing was lost — the \
         earlier task's outcome stands. Move on to the next piece of the goal, or add a \
         task whose title describes genuinely different work. If the earlier attempt was \
         wrong, say so in your answer rather than silently redoing it."
    )
}

/// Seed a CONTINUATION round's plan, dropping tasks this turn already closed.
///
/// A continuation round re-plans the same goal, and a small planner routinely
/// re-emits the very task the run just finished. [`seed_plan`] would create it
/// again — the earlier copy is CLOSED, so the open-title reuse in `tasks.add`
/// never applies — and the loop treadmills: observed live 2026-08-02 (session
/// 7ccc455a), one conversational question was re-planned ELEVEN times
/// ("Explain the difference between a mutex and a semaphore", tasks
/// #2033..#2044) and re-answered for 20+ minutes until `ROUNDS_MAX`.
///
/// The convergence criterion is principled, not a cap: work the run has
/// ALREADY COMPLETED this turn is by definition not new work. A proposed task
/// whose title matches ([`same_title`]) a title closed since
/// `closed_before_turn` was captured is dropped; a plan that is empty after
/// the drop seeds nothing and the caller counts a dry round. Titles closed
/// BEFORE the turn never filter — a user may legitimately re-ask yesterday's
/// question — and genuinely new titles (decomposed subtasks, next features)
/// seed exactly as before.
///
/// # Errors
///
/// Returns an error when the scope's tasks cannot be listed, or — once at
/// least one proposed task survives the filter — when [`seed_plan`] fails.
/// A plan whose every task was already closed this turn is `Ok` and empty.
pub async fn seed_continuation<S: std::hash::BuildHasher + Sync>(
    storage: &Arc<Storage>,
    scope: &str,
    scope_id: Option<&str>,
    plan: &Plan,
    closed_before_turn: &HashSet<i64, S>,
) -> Result<Vec<i64>, String> {
    let closed_this_turn = tasks_closed_since(storage, scope, scope_id, closed_before_turn).await?;
    let fresh: Vec<_> = plan
        .tasks
        .iter()
        .filter(|task| {
            !closed_this_turn
                .iter()
                .any(|closed| same_title(&closed.title, &task.title))
        })
        .cloned()
        .collect();
    if fresh.is_empty() {
        // Everything proposed was already worked this turn: a dry round, not
        // an error — the caller's dry counter is the mission's exit.
        return Ok(Vec::new());
    }
    if fresh.len() < plan.tasks.len() {
        tracing::info!(
            proposed = plan.tasks.len(),
            fresh = fresh.len(),
            "continuation re-proposed finished work — seeding only the new tasks"
        );
    }
    seed_plan(
        storage,
        scope,
        scope_id,
        &Plan {
            tasks: fresh,
            origin: plan.origin,
        },
        false,
    )
    .await
}

/// Put every `in_progress` item in a scope back to `pending`.
///
/// One invariant behind every caller: `next()` sorts `in_progress` first
/// ("resume what you started"), which is right only while the run that started
/// the item is alive. An interjection yields the in-flight item so the user's
/// message wins the next selection, and **every** chat turn releases its
/// leftovers on exit — cancelled or not — so a finished turn cannot
/// auto-resume ahead of the user's next request. Unfinished work is
/// information for the model, not an instruction to the harness.
///
/// Releasing only on cancellation left normally-finished turns holding their
/// items forever, which both hijacked the next message and accumulated: 191
/// items were found stuck `in_progress` on a live database, the oldest chats
/// carrying over a hundred each.
///
/// Returns how many items were demoted.
///
/// # Errors
///
/// Returns the store's error, as text, when the scope's tasks cannot be listed
/// or an item cannot be updated; items demoted before the failure stay
/// demoted.
pub async fn demote_in_progress(
    storage: &Arc<Storage>,
    scope: &str,
    scope_id: Option<&str>,
    actor: &str,
) -> Result<usize, String> {
    let repo = storage.tasks();
    let tasks = repo
        .list(scope, scope_id, false)
        .await
        .map_err(|e| e.to_string())?;
    let mut demoted = 0usize;
    for task in tasks.iter().filter(|t| t.status == "in_progress") {
        repo.update(
            task.id,
            TaskPatch {
                status: Some("pending".to_string()),
                ..TaskPatch::default()
            },
            Some(actor),
        )
        .await
        .map_err(|e| e.to_string())?;
        demoted += 1;
    }
    Ok(demoted)
}

// ---------------------------------------------------------------------------
// Interjection: mid-run messages join the plan at the next step boundary
// ---------------------------------------------------------------------------

/// Messages held for one session before admission.
///
/// Bound justification: these are messages a human typed while watching a run —
/// an unbounded queue here is a memory leak fed by a stuck run. 64 is far
/// past any realistic burst of human typing between two step boundaries, and
/// overflow drops the OLDEST so the most recent intent always survives.
pub const PENDING_MESSAGES_MAX: usize = 64;

/// Pending user messages for one session, drained by `SessionInterjector`.
///
/// This replaces "queue behind the mutex until the run ends" for the chat
/// path. A run can be hours long; blocking a follow-up message for that long
/// is the behaviour this exists to remove.
#[derive(Debug, Default)]
pub struct PendingMessages {
    inner: RwLock<Vec<String>>,
}

impl PendingMessages {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Queue a message for admission at the next step boundary.
    /// Returns the queue depth after the push.
    pub async fn push(&self, message: String) -> usize {
        let mut queue = self.inner.write().await;
        if queue.len() >= PENDING_MESSAGES_MAX {
            let dropped = queue.remove(0);
            // A dropped message is lost user input; it must not go silently
            // (summaries-must-announce-themselves).
            tracing::warn!(
                held = PENDING_MESSAGES_MAX,
                dropped_chars = dropped.chars().count(),
                "pending-message queue full — the OLDEST message sent during this run was dropped"
            );
        }
        queue.push(message);
        debug_assert!(queue.len() <= PENDING_MESSAGES_MAX, "the queue is bounded");
        queue.len()
    }

    /// Take everything waiting.
    pub async fn drain(&self) -> Vec<String> {
        let mut queue = self.inner.write().await;
        std::mem::take(&mut *queue)
    }

    /// Cheap check used on the hot path - the overwhelmingly common answer
    /// is "nothing waiting", and that must not cost a write lock.
    pub async fn is_empty(&self) -> bool {
        self.inner.read().await.is_empty()
    }

    /// Current depth.
    pub async fn len(&self) -> usize {
        self.inner.read().await.len()
    }
}

/// Admits pending chat messages into a live run at step boundaries.
pub struct SessionInterjector {
    pub storage: Arc<Storage>,
    pub scope: String,
    pub scope_id: Option<String>,
    pub pending: Arc<PendingMessages>,
    pub planner: Arc<AgentPlanner>,
    pub actor: String,
    pub event_tx: Option<tokio::sync::broadcast::Sender<Event>>,
    /// The run's Stop token, so a cancel arriving while an interjection is
    /// being planned aborts that planning call instead of waiting it out.
    pub cancel: Option<CancelToken>,
}

impl SessionInterjector {
    /// Put the in-flight item back to `pending` so the interjected task wins
    /// the next selection.
    ///
    /// `TaskRepository::next` sorts `in_progress` ahead of everything else
    /// ("resume what you started"), which would otherwise starve an
    /// interjection behind a long multi-step item. Yielding is safe: the
    /// harness re-marks an item `in_progress` via `start()` whenever it
    /// selects it, notes are untouched, and the harness's in-memory progress
    /// counters are keyed by item id, so the original resumes exactly where
    /// it stood once the user's request is done.
    async fn yield_current_item(&self) -> Result<(), String> {
        demote_in_progress(
            &self.storage,
            &self.scope,
            self.scope_id.as_deref(),
            &self.actor,
        )
        .await
        .map(|_| ())
    }
}

#[async_trait::async_trait]
impl Interjector for SessionInterjector {
    async fn interject(&self) -> Result<usize, String> {
        // Hot path: a read lock and an is_empty, nothing more.
        if self.pending.is_empty().await {
            return Ok(0);
        }
        let messages = self.pending.drain().await;
        if messages.is_empty() {
            return Ok(0);
        }

        let mut admitted = 0usize;
        let mut queue = messages.into_iter();
        while let Some(message) = queue.next() {
            let plan = self
                .planner
                .plan(&message, None, self.cancel.as_ref())
                .await;
            // Stop pressed while this interjection was being planned: do not
            // seed work nobody will run. The drained messages go back to
            // `pending` so the next turn claims them instead of losing them.
            if self.cancel.as_ref().is_some_and(CancelToken::is_cancelled) {
                self.pending.push(message).await;
                for rest in queue {
                    self.pending.push(rest).await;
                }
                return Ok(admitted);
            }
            // Yield before seeding so the new tasks are the only candidates
            // not outranked by an in-progress item.
            if let Err(e) = self.yield_current_item().await {
                tracing::warn!(error = %e, "could not yield the in-flight item for an interjection");
            }
            let ids = seed_plan(
                &self.storage,
                &self.scope,
                self.scope_id.as_deref(),
                &plan,
                true,
            )
            .await?;
            admitted += ids.len();
            if let Some(tx) = &self.event_tx {
                let _ = tx.send(Event::TaskRunProgress {
                    scope: self.scope.clone(),
                    scope_id: self.scope_id.clone(),
                    task_id: ids.first().copied(),
                    kind: "interjected".to_string(),
                    detail: json!({
                        "message": message,
                        "tasks": ids,
                        "plan_origin": plan.origin,
                    }),
                });
            }
        }
        Ok(admitted)
    }
}

// ---------------------------------------------------------------------------
// Recurrence sweep
// ---------------------------------------------------------------------------

/// Reopen completed recurring tasks whose next cron occurrence has arrived.
///
/// Driven by the daemon scheduler (P8) — one recurrence engine, not two: the
/// task stores the cron expression, the scheduler provides the clock.
/// Returns the number of tasks reopened.
pub async fn sweep_recurrences(storage: &Arc<Storage>) -> usize {
    let repo = storage.tasks();
    let tasks = match repo.list_recurring_closed().await {
        Ok(tasks) => tasks,
        Err(e) => {
            tracing::warn!("recurrence sweep failed to list tasks: {e}");
            return 0;
        }
    };
    let now = chrono::Utc::now();
    let mut reopened = 0usize;
    for task in tasks {
        let Some(expr_str) = &task.recurrence else {
            continue;
        };
        let expr = match nanna_core::CronExpr::parse(expr_str) {
            Ok(expr) => expr,
            Err(e) => {
                tracing::warn!(task_id = task.id, "invalid recurrence '{expr_str}': {e}");
                continue;
            }
        };
        let completed = task
            .completed_at
            .as_deref()
            .and_then(parse_db_time)
            .unwrap_or(now);
        if expr.next(&completed).is_some_and(|next| next <= now)
            && repo.reopen(task.id, Some("recurrence")).await.is_ok()
        {
            info!(task_id = task.id, "recurring task reopened");
            reopened += 1;
        }
    }
    reopened
}

/// Parse a stored timestamp: RFC3339 first, then turso's
/// `datetime('now')` format (`YYYY-MM-DD HH:MM:SS`, UTC).
fn parse_db_time(s: &str) -> Option<chrono::DateTime<chrono::Utc>> {
    chrono::DateTime::parse_from_rfc3339(s)
        .map(|dt| dt.with_timezone(&chrono::Utc))
        .ok()
        .or_else(|| {
            chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S")
                .ok()
                .map(|naive| naive.and_utc())
        })
}

/// Cheap stable digest for repetition comparison (not cryptographic).
fn digest(content: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    content.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

// ---------------------------------------------------------------------------
// Run manager
// ---------------------------------------------------------------------------

/// Minimum spacing between Ollama server restarts across ALL callers —
/// chat sessions, heartbeats, and task runs share one local server. Bounds
/// the blast radius when a model is persistently broken (e.g. OOM on every
/// load): without it, every failing chat re-kills the server out from under
/// every other client.
const OLLAMA_RESTART_COOLDOWN_SECS: u64 = 600;

/// Unix-epoch seconds of the last restart this process performed.
static LAST_OLLAMA_RESTART_EPOCH_SECS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Where the healing ladder acts: the Ollama server chat is talking to, and
/// whether its processes are this machine's.
///
/// The address is the configured one (`[memory].ollama_host`, Settings →
/// Models → Server URL), taken from the router's Ollama client at the moment
/// of healing; a config reload rebuilds that client, so a server changed
/// mid-run is the server healed. `OLLAMA_HOST` is not read: it is Ollama's own
/// bind variable, chat never read it, and letting it steer only the ladder is
/// how the ladder came to wait on, unload from and kill a local server while
/// chat was failing against a remote one. A local server on a non-default port
/// is configured like any other (`http://127.0.0.1:11500`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OllamaHealTarget {
    /// The server's base URL, as chat addresses it.
    pub(crate) base_url: String,
    /// Whether the server runs on this machine. Only then may the ladder act
    /// on processes: unload the model's runner, or kill and respawn the server
    /// (`pkill` reaches this machine's processes, whichever server they
    /// serve). A server elsewhere is only waited on.
    pub(crate) on_this_machine: bool,
}

/// Walk `model`'s context window down one rung on the server the router
/// sends it to, which is where its next attempt goes. See
/// [`nanna_llm::LlmClient::demote_context`] for the rung and the floor.
fn demote_context_on_router(
    router: &LlmRouter,
    model: &str,
    min_viable_ctx: Option<u32>,
) -> Option<u32> {
    router
        .client_for_model(model)?
        .demote_context(&LlmRouter::strip_model_prefix(model), min_viable_ctx)
}

/// `model`'s info from the cache or the universal floor, clamped to the window
/// the server the router sends it to honours now.
///
/// The lookup takes the prefix-stripped name: the model info cache is written
/// under the id the provider was asked about, so a `provider/model` spelling
/// misses every entry and silently falls back to the universal floor.
fn cached_model_info_on_router(router: &LlmRouter, model: &str) -> nanna_llm::ModelInfo {
    let bare = LlmRouter::strip_model_prefix(model);
    router.client_for_model(model).map_or_else(
        || nanna_llm::model_info_from_cache_or_unknown(&bare, ""),
        |client| client.model_info_from_cache_or_unknown(&bare),
    )
}

/// The healing target for the Ollama server at `base_url`.
pub(crate) fn ollama_heal_target(base_url: &str) -> OllamaHealTarget {
    OllamaHealTarget {
        base_url: base_url.to_string(),
        on_this_machine: nanna_llm::ollama_server_is_local(base_url),
    }
}

/// The router's Ollama server as a healing target, read now. `None` when the
/// router serves no Ollama, so there is nothing to heal.
fn current_ollama_target(router: &LlmRouter) -> Option<OllamaHealTarget> {
    router
        .ollama_client()
        .map(|client| ollama_heal_target(client.base_url()))
}

/// The router's Ollama server, when a process-level cure may act on it: one on
/// this machine. Both such cures — the runner unload and the server restart —
/// take their server from here, so neither reaches this machine's processes
/// for a server they do not serve. For a server elsewhere, `declined` is logged
/// with its address and `None` returned; `None` too when no Ollama is
/// configured.
fn server_for_process_cure(router: &LlmRouter, declined: &str) -> Option<OllamaHealTarget> {
    let Some(target) = current_ollama_target(router) else {
        tracing::warn!("no Ollama server is configured, so there is no Ollama process to cure");
        return None;
    };
    if !target.on_this_machine {
        tracing::warn!(server = %target.base_url, "{declined}");
        return None;
    }
    Some(target)
}

/// Whether the router's Ollama server answers at all right now, asked with
/// chat's token: one request, bounded to 3s. Any status is an answer, because
/// what ends a server-down wait is the server taking connections again (see
/// [`nanna_llm::LlmClient::ollama_answers`]). The client is read per poll, so
/// a wait that spans a config reload polls the new server.
async fn ollama_answers(router: &LlmRouter) -> bool {
    match router.ollama_client() {
        Some(client) => client.ollama_answers(std::time::Duration::from_secs(3)).await,
        None => false,
    }
}

/// Restart the local Ollama server: tree-kill the server process AND its
/// per-model runner children (the tray supervisor respawns the server), then
/// wait for the API to come back.
///
/// This is the cure for the sticky degraded-runner state (every generation
/// aborted with `done:false`; model unloads do not clear it — verified live).
/// Callers gate it: bouncing a shared local service is an operator decision.
/// Refuses to act when the configured server (`[memory].ollama_host`) is not
/// on this machine, and at most once per `OLLAMA_RESTART_COOLDOWN_SECS`
/// process-wide.
///
/// REGRESSION (2026-08-09): the kill must take the whole runner tree, not
/// just the server. This function originally ran `taskkill /F /IM
/// ollama.exe`, which on Windows kills only the parent — each loaded model's
/// `llama-server.exe` runner child (~6 GB VRAM for an 8B model) survived as
/// an orphan that the respawned server cannot see, because `ollama ps`
/// reports the new server's own runners, not what is actually on the card.
/// Verified live: four orphans (parents dead) held ~12.5 GB of a 16 GB card,
/// the `num_ctx` sizing probe honestly latched 4096 from the 1.5 GB that
/// remained — below the ~4.2k min-viable floor — and two endurance attempts
/// died in minutes on below-floor stops. Each restart heal leaks one runner
/// per loaded model, so repeated heals (e.g. the 2026-08-08 ministral leg's
/// two) starve the card cumulatively while looking like successful cures.
pub async fn restart_ollama_server(router: &LlmRouter) -> bool {
    let Some(target) = server_for_process_cure(
        router,
        "not restarting the Ollama server: it is not on this machine, and killing this \
         machine's ollama/llama-server processes cannot cure a server elsewhere",
    ) else {
        return false;
    };

    let now_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let last = LAST_OLLAMA_RESTART_EPOCH_SECS.load(std::sync::atomic::Ordering::Relaxed);
    if now_secs.saturating_sub(last) < OLLAMA_RESTART_COOLDOWN_SECS
        || LAST_OLLAMA_RESTART_EPOCH_SECS
            .compare_exchange(
                last,
                now_secs,
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
            )
            .is_err()
    {
        tracing::warn!(
            "skipping Ollama restart — one already ran {}s ago (cooldown {}s)",
            now_secs.saturating_sub(last),
            OLLAMA_RESTART_COOLDOWN_SECS
        );
        return false;
    }

    tracing::warn!(
        server = %target.base_url,
        "restarting the Ollama server (degraded runner state)"
    );
    // Every model's runner dies with the server, so every fingerprint
    // describing one is now stale.
    forget_all_wedges();
    // `/T` takes runners still parented to a live server; the follow-up
    // image-name sweep takes orphans from EARLIER parent-only kills (this
    // function pre-fix, or an external restart). The sweep cannot hit a live
    // server's runner: with every `ollama.exe` dead, any surviving
    // `llama-server.exe` is by definition an orphan, and the tray's respawned
    // server loads models lazily so it has none of its own yet.
    #[cfg(windows)]
    {
        let _ = std::process::Command::new("taskkill")
            .args(["/F", "/T", "/IM", "ollama.exe"])
            .output();
        let _ = std::process::Command::new("taskkill")
            .args(["/F", "/IM", "llama-server.exe"])
            .output();
    }
    #[cfg(not(windows))]
    {
        let _ = std::process::Command::new("pkill")
            .args(["-x", "ollama"])
            .output();
        let _ = std::process::Command::new("pkill")
            .args(["-x", "llama-server"])
            .output();
    }
    for _ in 0..20 {
        tokio::time::sleep(std::time::Duration::from_secs(3)).await;
        if ollama_answers(router).await {
            info!("Ollama is back after restart");
            return true;
        }
    }
    tracing::warn!("Ollama did not come back within 60s of restart");
    false
}

/// Whether runs may restart the Ollama server as last-resort healing.
/// ON by default (owner decision): the degraded state bricks every client of
/// the server anyway, and a restart is the only known cure. Set
/// `NANNA_OLLAMA_RESTART_ON_DEGRADED=0` to opt out on setups where bouncing
/// the shared server is unacceptable.
pub(crate) fn ollama_restart_allowed() -> bool {
    std::env::var("NANNA_OLLAMA_RESTART_ON_DEGRADED").as_deref() != Ok("0")
}

/// Poll the configured Ollama server — on this machine or not — until it
/// answers, up to `max_secs`. The chat healer uses this to WAIT OUT a
/// server-down window (our own runner-surgery restart, or any external
/// restart) instead of burning retry budget: attempts against a down server
/// complete zero tool calls, so the progress-based replenishment can never
/// refill the budget, and the 2/5/10s backoffs burn out inside a ~20-60s
/// restart — observed live killing a 4h55m mission two minutes after the
/// surgery that cured its fault storm.
pub(crate) async fn wait_for_ollama_ready(router: &LlmRouter, max_secs: u64) -> bool {
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(max_secs);
    loop {
        if ollama_answers(router).await {
            return true;
        }
        if tokio::time::Instant::now() >= deadline {
            return false;
        }
        tokio::time::sleep(std::time::Duration::from_secs(3)).await;
    }
}

/// Unload an Ollama-served model (`keep_alive: 0`) to clear runner state —
/// the observed degraded mode restores a stale KV context checkpoint that
/// sends every generation straight to a stop token, and only a fresh runner
/// clears it. Free-function variant so the chat path can heal with the same
/// ladder as the step runner. No-op for non-Ollama models: healing is
/// provider-gated, a `:free` `OpenRouter` suffix must never trigger it. No-op
/// too for a server not on this machine: its runners are its own operator's,
/// and an unload there pulls the model from every other client it serves.
pub(crate) async fn reset_ollama_runner_for(router: &LlmRouter, model: &str) {
    if crate::llm_router::ProviderId::from_model(model) != crate::llm_router::ProviderId::Ollama {
        return;
    }
    let Some(target) = server_for_process_cure(
        router,
        &format!(
            "not resetting the Ollama runner for {model}: the server is not on this machine, and \
             unloading the model there would pull it from every other client that server has"
        ),
    ) else {
        return;
    };
    tracing::warn!(model = %model, "resetting Ollama runner (keep_alive=0) after transient failures");
    // The runner this fingerprint described is about to stop existing, so the
    // next wedge must be judged fresh rather than matched against it.
    forget_wedge(model);
    let client = reqwest::Client::new();
    let _ = client
        .post(format!("{}/api/generate", target.base_url))
        .json(&serde_json::json!({
            "model": LlmRouter::strip_model_prefix(model),
            "keep_alive": 0
        }))
        .timeout(std::time::Duration::from_secs(20))
        .send()
        .await;
    // Give the runner a moment to tear down before the reload request.
    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
}

/// Fold per-segment reports into one run-level report: counters sum, the
/// stop reason is the final segment's, and tokens-per-item is recomputed
/// over the aggregate.
fn fold_reports(segments: &[LongHorizonReport]) -> LongHorizonReport {
    let mut folded = segments.last().cloned().unwrap_or(LongHorizonReport {
        tool_calls: 0,
        side_effect_tool_calls: 0,
        stop: StopReason::AllTasksDone,
        steps_taken: 0,
        items_completed: 0,
        items_completed_unverified: 0,
        items_revived: 0,
        abandoned_unmet: Vec::new(),
        abandoned_unverifiable: Vec::new(),
        items_regressed_reopened: 0,
        items_already_satisfied: 0,
        items_abandoned: 0,
        last_runner_error: None,
        replans: 0,
        false_success_claims: 0,
        input_tokens: 0,
        output_tokens: 0,
        wall_clock_secs: 0,
        tokens_per_completed_item: None,
        interjected_items: 0,
        verified_outcomes: Vec::new(),
        acceptance_unknown: 0,
    });
    folded.steps_taken = segments.iter().map(|r| r.steps_taken).sum();
    folded.tool_calls = segments.iter().map(|r| r.tool_calls).sum();
    folded.side_effect_tool_calls = segments.iter().map(|r| r.side_effect_tool_calls).sum();
    folded.items_completed = segments.iter().map(|r| r.items_completed).sum();
    folded.items_completed_unverified =
        segments.iter().map(|r| r.items_completed_unverified).sum();
    folded.items_revived = segments.iter().map(|r| r.items_revived).sum();
    folded.items_regressed_reopened =
        segments.iter().map(|r| r.items_regressed_reopened).sum();
    folded.items_already_satisfied = segments.iter().map(|r| r.items_already_satisfied).sum();
    folded.items_abandoned = segments.iter().map(|r| r.items_abandoned).sum();
    folded.replans = segments.iter().map(|r| r.replans).sum();
    folded.false_success_claims = segments.iter().map(|r| r.false_success_claims).sum();
    folded.input_tokens = segments.iter().map(|r| r.input_tokens).sum();
    folded.output_tokens = segments.iter().map(|r| r.output_tokens).sum();
    folded.wall_clock_secs = segments.iter().map(|r| r.wall_clock_secs).sum();
    folded.interjected_items = segments.iter().map(|r| r.interjected_items).sum();
    folded.acceptance_unknown = segments.iter().map(|r| r.acceptance_unknown).sum();
    // The naming lists concat too, or a run that abandoned work in an earlier
    // segment reports a count with nothing behind it — the same "a number with
    // no names" gap this list was added to close.
    folded.abandoned_unmet = segments
        .iter()
        .flat_map(|r| r.abandoned_unmet.iter().cloned())
        .collect();
    folded.abandoned_unverifiable = segments
        .iter()
        .flat_map(|r| r.abandoned_unverifiable.iter().cloned())
        .collect();
    // Union by id, newest verdict wins — same rule as the chat harness fold.
    folded.verified_outcomes = Vec::new();
    for segment in segments {
        for v in &segment.verified_outcomes {
            folded.verified_outcomes.retain(|p: &nanna_agent::harness::VerifiedOutcome| p.id != v.id);
            folded.verified_outcomes.push(v.clone());
        }
    }
    // Like the stop reason, the newest recorded error is the one that
    // describes where the run ended up.
    folded.last_runner_error = segments
        .iter()
        .rev()
        .find_map(|r| r.last_runner_error.clone());
    folded.tokens_per_completed_item = if folded.items_completed > 0 {
        Some((folded.input_tokens + folded.output_tokens) / folded.items_completed as u64)
    } else {
        None
    };
    folded
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------
    // Step outcomes: what counts as progress
    // -----------------------------------------------------------------

    fn outcome(text: &str, tools: usize, output_tokens: u64) -> StepOutcome {
        StepOutcome {
            text: text.to_string(),
            input_tokens: 100,
            output_tokens,
            tool_calls: (0..tools)
                .map(|i| StepToolCall {
                    name: format!("tool_{i}"),
                    input_digest: String::new(),
                    output_digest: String::new(),
                    success: true,
                })
                .collect(),
            touched_paths: Vec::new(),
            degenerate_loop: false,
        }
    }

    // -----------------------------------------------------------------
    // GPU-fault recovery ladder: blip tolerance before demotion
    // -----------------------------------------------------------------

    /// The first fault of a run is a blip until proven otherwise: reset the
    /// runner at the SAME size, never shrink a viable window on one data
    /// point (the 2026-08-03 run died to exactly that). The second and every
    /// later fault is the proof — demote as well.
    #[test]
    fn the_first_gpu_fault_resets_only_and_the_second_demotes() {
        assert_eq!(gpu_fault_action(1), GpuFaultAction::ResetOnly);
        assert_eq!(gpu_fault_action(2), GpuFaultAction::ResetAndDemote);
        assert_eq!(gpu_fault_action(3), GpuFaultAction::ResetAndDemote);
        assert_eq!(gpu_fault_action(u32::MAX), GpuFaultAction::ResetAndDemote);
    }

    /// The run-scoped counter drives the ladder: two faults recorded through
    /// the shared tally cross the demotion threshold exactly once, at the
    /// second fault — regardless of which runner clone observed each fault.
    #[test]
    fn the_shared_fault_tally_crosses_into_demotion_at_the_second_fault() {
        let tally = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let record = |t: &Arc<std::sync::atomic::AtomicU32>| {
            t.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1
        };
        assert_eq!(gpu_fault_action(record(&tally)), GpuFaultAction::ResetOnly);
        // A second runner sharing the same run shares the tally.
        let other_runner_view = tally.clone();
        assert_eq!(
            gpu_fault_action(record(&other_runner_view)),
            GpuFaultAction::ResetAndDemote
        );
        assert_eq!(gpu_fault_action(record(&tally)), GpuFaultAction::ResetAndDemote);
    }

    #[test]
    fn reasoning_without_action_is_not_progress() {
        // The live failure: 50 words of thinking, no tool call, no answer.
        // Reasoning tokens put it over the empty-completion bound, so it was
        // accepted as a step result and the item was re-entered seven times.
        let thinking_only = outcome("", 0, 240);
        assert!(made_no_progress(&thinking_only));
        assert!(
            !is_empty_completion(&thinking_only),
            "still not an empty completion — the provider is generating fine"
        );
    }

    #[test]
    fn acting_or_answering_counts_as_progress() {
        assert!(!made_no_progress(&outcome("", 1, 300)), "a tool call is progress");
        assert!(!made_no_progress(&outcome("done: 7/42 pass", 0, 300)), "an answer is progress");
        assert!(
            !made_no_progress(&outcome("  \n ", 1, 300)),
            "blank text with a tool call is still progress"
        );
    }

    #[test]
    fn the_no_progress_nudge_demands_an_action() {
        // The retry must not be a verbatim repeat of what just stalled.
        assert!(NO_PROGRESS_NUDGE.contains("CALL A TOOL"));
        assert!(NO_PROGRESS_NUDGE.contains("nothing changed"));
    }

    // -----------------------------------------------------------------
    // Pending messages (interjection intake)
    // -----------------------------------------------------------------

    #[tokio::test]
    async fn pending_messages_drain_in_arrival_order_and_empty_the_queue() {
        let pending = PendingMessages::new();
        assert!(pending.is_empty().await);
        pending.push("first".to_string()).await;
        pending.push("second".to_string()).await;
        assert_eq!(pending.len().await, 2);
        assert!(!pending.is_empty().await);

        assert_eq!(pending.drain().await, vec!["first", "second"]);
        assert!(pending.is_empty().await, "drain must empty the queue");
        assert!(pending.drain().await.is_empty(), "second drain is a no-op");
    }

    #[tokio::test]
    async fn pending_messages_overflow_drops_the_oldest_not_the_newest() {
        // The bound exists so a stuck run cannot leak memory; when it bites,
        // the user's most recent intent is what must survive.
        let pending = PendingMessages::new();
        for i in 0..(PENDING_MESSAGES_MAX + 5) {
            pending.push(format!("m{i}")).await;
        }
        assert_eq!(pending.len().await, PENDING_MESSAGES_MAX);
        let drained = pending.drain().await;
        assert_eq!(drained.last().unwrap(), &format!("m{}", PENDING_MESSAGES_MAX + 4));
        assert_eq!(drained.first().unwrap(), "m5");
    }

    // -----------------------------------------------------------------
    // Plan seeding + queue jumping
    // -----------------------------------------------------------------

    /// A turn admits what it creates, not what it found open — unless the
    /// plan proposes that work again, which adopts it (once), by the store's
    /// own title identity.
    #[test]
    fn a_turn_admits_its_own_work_and_what_its_plan_readopts() {
        let admission = TurnAdmission {
            leftovers: HashMap::from([
                (1, "Answer: what is the capital of France?".to_string()),
                (2, "Fix the build".to_string()),
            ]),
            adopted: std::sync::Mutex::new(HashSet::new()),
        };
        assert!(admission.admits(99), "created during the turn");
        assert!(!admission.admits(1), "left open by an earlier turn");

        let mut plan = plan_of(&["Answer: What is the capital of France?", "Write the notes"]);
        let adopted = admission.adopt_from(&mut plan);
        assert_eq!(adopted, [1], "same work, re-proposed: adopted");
        assert!(admission.admits(1));
        assert!(!admission.admits(2), "never re-proposed: stays out");
        assert_eq!(plan.tasks.len(), 1, "the adopted task is not created again");
        assert_eq!(plan.tasks[0].title, "Write the notes");

        let mut twice = plan_of(&["Fix the builds", "fix the build"]);
        assert_eq!(
            admission.adopt_from(&mut twice),
            [2],
            "plural-folded; adopted once"
        );
        assert_eq!(twice.tasks.len(), 1, "the second copy is new work");
    }

    fn plan_of(titles: &[&str]) -> Plan {
        Plan {
            tasks: titles
                .iter()
                .map(|t| nanna_agent::planner::PlannedTask {
                    title: (*t).to_string(),
                    description: None,
                    acceptance: None,
                    tool_scope: Vec::new(),
                })
                .collect(),
            origin: nanna_agent::planner::PlanOrigin::Model,
        }
    }

    #[tokio::test]
    async fn seeded_plan_is_executed_in_plan_order() {
        let storage = Arc::new(Storage::in_memory().await.expect("storage"));
        let ids = seed_plan(&storage, "session", Some("s1"), &plan_of(&["a", "b", "c"]), false)
            .await
            .expect("seeded");
        assert_eq!(ids.len(), 3);

        let repo = storage.tasks();
        let next = repo.next("session", Some("s1")).await.unwrap().unwrap();
        assert_eq!(next.title, "a", "plan order decides what runs first");
    }

    /// The leak this release rule closes: an item left `in_progress` by a
    /// finished turn outranks whatever the user says next, because `next()`
    /// sorts `in_progress` first. Releasing on cancellation alone left every
    /// normally-finished turn holding its items — 191 were found stuck on a
    /// live database, the busiest chat carrying 155 open.
    #[tokio::test]
    async fn a_released_item_stops_outranking_the_users_next_message() {
        let storage = Arc::new(Storage::in_memory().await.expect("storage"));
        seed_plan(&storage, "session", Some("s1"), &plan_of(&["stale work"]), false)
            .await
            .expect("seeded");
        let repo = storage.tasks();

        // The turn picks it up and ends without finishing it.
        let started = repo.next("session", Some("s1")).await.unwrap().unwrap();
        repo.update(
            started.id,
            TaskPatch {
                status: Some("in_progress".to_string()),
                ..TaskPatch::default()
            },
            Some("harness"),
        )
        .await
        .expect("started");

        // The user asks for something else.
        seed_plan(&storage, "session", Some("s1"), &plan_of(&["the new question"]), true)
            .await
            .expect("interjected");
        assert_eq!(
            repo.next("session", Some("s1")).await.unwrap().unwrap().title,
            "stale work",
            "an unreleased item wins — this is the hijack"
        );

        // Turn end releases it.
        let demoted = demote_in_progress(&storage, "session", Some("s1"), "chat")
            .await
            .expect("released");
        assert_eq!(demoted, 1);
        assert_eq!(
            repo.next("session", Some("s1")).await.unwrap().unwrap().title,
            "the new question",
            "after release the user's message goes first"
        );

        // Released, not closed: it is still open work the planner can see.
        let open = repo.list("session", Some("s1"), false).await.unwrap();
        assert!(
            open.iter().any(|t| t.title == "stale work" && t.status == "pending"),
            "release must return the item to pending, not close it"
        );
    }

    #[tokio::test]
    async fn an_interjected_plan_preempts_pending_work() {
        let storage = Arc::new(Storage::in_memory().await.expect("storage"));
        seed_plan(&storage, "session", Some("s1"), &plan_of(&["original"]), false)
            .await
            .expect("seeded");
        seed_plan(&storage, "session", Some("s1"), &plan_of(&["urgent"]), true)
            .await
            .expect("interjected");

        let next = storage
            .tasks()
            .next("session", Some("s1"))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(next.title, "urgent", "the user's new message goes first");
    }

    #[tokio::test]
    async fn an_in_progress_item_outranks_an_interjection_until_it_yields() {
        // This is the reason SessionInterjector::yield_current_item exists:
        // TaskRepository::next sorts in_progress ahead of priority, so a
        // mid-run interjection would otherwise starve behind a long item.
        let storage = Arc::new(Storage::in_memory().await.expect("storage"));
        let original = seed_plan(&storage, "session", Some("s1"), &plan_of(&["original"]), false)
            .await
            .expect("seeded")[0];
        let repo = storage.tasks();
        repo.update(
            original,
            TaskPatch {
                status: Some("in_progress".to_string()),
                ..TaskPatch::default()
            },
            Some("harness"),
        )
        .await
        .expect("marked in progress");

        seed_plan(&storage, "session", Some("s1"), &plan_of(&["urgent"]), true)
            .await
            .expect("interjected");

        // Without yielding, the in-flight item still wins.
        let blocked = repo.next("session", Some("s1")).await.unwrap().unwrap();
        assert_eq!(blocked.title, "original");

        // Yielding it back to pending is what lets the interjection through.
        repo.update(
            original,
            TaskPatch {
                status: Some("pending".to_string()),
                ..TaskPatch::default()
            },
            Some("harness"),
        )
        .await
        .expect("yielded");
        let freed = repo.next("session", Some("s1")).await.unwrap().unwrap();
        assert_eq!(freed.title, "urgent");
    }

    #[tokio::test]
    async fn the_yielded_item_resumes_once_the_interjection_completes() {
        let storage = Arc::new(Storage::in_memory().await.expect("storage"));
        seed_plan(&storage, "session", Some("s1"), &plan_of(&["original"]), false)
            .await
            .expect("seeded");
        let urgent = seed_plan(&storage, "session", Some("s1"), &plan_of(&["urgent"]), true)
            .await
            .expect("interjected")[0];

        let repo = storage.tasks();
        repo.complete(urgent, Some("harness"), Some(json!({"verified": false})))
            .await
            .expect("completed");

        let resumed = repo.next("session", Some("s1")).await.unwrap().unwrap();
        assert_eq!(
            resumed.title, "original",
            "the interrupted plan continues where it stood"
        );
    }

    #[tokio::test]
    async fn repeated_interjections_stay_in_arrival_order() {
        let storage = Arc::new(Storage::in_memory().await.expect("storage"));
        seed_plan(&storage, "session", Some("s1"), &plan_of(&["original"]), false)
            .await
            .expect("seeded");
        seed_plan(&storage, "session", Some("s1"), &plan_of(&["first ask"]), true)
            .await
            .expect("first");
        seed_plan(&storage, "session", Some("s1"), &plan_of(&["second ask"]), true)
            .await
            .expect("second");

        // The later interjection sorts ahead — a user who speaks twice means
        // the most recent thing most.
        let next = storage
            .tasks()
            .next("session", Some("s1"))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(next.title, "second ask");
    }

    /// REGRESSION (live 2026-08-02): turn 1 was cancelled with its item still
    /// `in_progress`; turn 2 asked something unrelated, and `next()`'s
    /// in_progress-first ordering ran the stale leftover ahead of the user's
    /// new request. The cancel path demotes, the new seed sorts ahead.
    #[tokio::test]
    async fn a_cancelled_turns_leftover_does_not_preempt_the_next_turns_plan() {
        let storage = Arc::new(Storage::in_memory().await.expect("storage"));
        let turn1 = seed_plan(&storage, "session", Some("s1"), &plan_of(&["turn one"]), false)
            .await
            .expect("seeded")[0];
        let repo = storage.tasks();
        repo.update(
            turn1,
            TaskPatch {
                status: Some("in_progress".to_string()),
                ..TaskPatch::default()
            },
            Some("harness"),
        )
        .await
        .expect("harness started it");

        // Stop pressed: the chat run's exit path demotes its in-flight items.
        let demoted = demote_in_progress(&storage, "session", Some("s1"), "chat")
            .await
            .expect("demote");
        assert_eq!(demoted, 1);

        seed_plan(&storage, "session", Some("s1"), &plan_of(&["turn two"]), false)
            .await
            .expect("seeded");

        let next = repo.next("session", Some("s1")).await.unwrap().unwrap();
        assert_eq!(
            next.title, "turn two",
            "the user's current turn outranks a cancelled turn's leftover"
        );
        assert_eq!(
            repo.get(turn1).await.unwrap().status,
            "pending",
            "the leftover stays open as information for the planner"
        );
    }

    /// Even without a cancel, a fresh turn's plan seeds ahead of pending
    /// leftovers: each seed sorts strictly below the scope's current minimum.
    #[tokio::test]
    async fn a_fresh_turns_plan_outranks_pending_leftovers() {
        let storage = Arc::new(Storage::in_memory().await.expect("storage"));
        seed_plan(
            &storage,
            "session",
            Some("s1"),
            &plan_of(&["leftover a", "leftover b"]),
            false,
        )
        .await
        .expect("seeded");
        seed_plan(&storage, "session", Some("s1"), &plan_of(&["fresh ask"]), false)
            .await
            .expect("seeded");

        let next = storage
            .tasks()
            .next("session", Some("s1"))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(next.title, "fresh ask", "each turn seeds ahead of prior leftovers");
    }

    /// REGRESSION (live 2026-08-02): the model self-created a p1 todo
    /// ("Migrate Lucide icons") that outranked the user's p2 request — the
    /// agent was about to start an unrequested repo-wide migration. The
    /// tasks.* services floor model priority at 2 on add AND update.
    #[tokio::test]
    async fn a_tool_created_todo_never_outranks_the_turns_plan() {
        let storage = Arc::new(Storage::in_memory().await.expect("storage"));
        let workspace_id = Arc::new(RwLock::new(None));
        let services =
            build_task_services(storage.clone(), workspace_id, Arc::new(TurnBaselines::new()));

        seed_plan(&storage, "session", Some("s1"), &plan_of(&["user request"]), false)
            .await
            .expect("seeded");

        let created = add_task(
            &services,
            json!({
                "title": "Migrate Lucide icons to @lucide/vue",
                "scope": "session",
                "session_id": "s1",
                "priority": 1,
            }),
        )
        .await;
        assert_eq!(
            created["task"]["priority"],
            json!(2),
            "model-asked p1 floors at 2"
        );

        // Nor can the model promote its own item to p1 after the fact.
        let id = created["task"]["id"].as_i64().expect("id");
        let patched = services
            .get("tasks.update")
            .expect("tasks.update service")(json!({ "id": id, "priority": 1 }))
        .await
        .expect("update succeeds");
        assert_eq!(patched["task"]["priority"], json!(2));

        let next = storage
            .tasks()
            .next("session", Some("s1"))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            next.title, "user request",
            "self-created side-work cannot jump the user's plan"
        );
    }

    /// An interjection (the user speaking mid-run) still jumps everything:
    /// leftovers, the current turn's plan, and any self-created todo.
    #[tokio::test]
    async fn an_interjection_still_jumps_leftovers_and_the_current_plan() {
        let storage = Arc::new(Storage::in_memory().await.expect("storage"));
        seed_plan(&storage, "session", Some("s1"), &plan_of(&["leftover"]), false)
            .await
            .expect("seeded");
        seed_plan(&storage, "session", Some("s1"), &plan_of(&["current turn"]), false)
            .await
            .expect("seeded");
        seed_plan(&storage, "session", Some("s1"), &plan_of(&["interjection"]), true)
            .await
            .expect("interjected");

        let next = storage
            .tasks()
            .next("session", Some("s1"))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            next.title, "interjection",
            "the user speaking mid-run still goes first"
        );
    }

    /// The live hazard: `[memory].ollama_host` names a remote server, that
    /// server's proxy answers 502, and the ladder unloaded a model on THIS
    /// machine's Ollama and then `pkill`ed its `ollama` and `llama-server`. A
    /// remote server is the one the ladder acts on, and only by waiting.
    #[test]
    fn a_remote_configured_server_is_never_operated_on_locally() {
        for remote in ["https://gpubox.example/ollama", "http://192.168.1.20:11434"] {
            let target = ollama_heal_target(remote);
            assert_eq!(target.base_url, remote, "the ladder must act on the configured server");
            assert!(!target.on_this_machine, "{remote}'s processes are not this machine's");
        }
    }

    /// A server on this machine is healed where it listens — a non-default
    /// port included, which only the configured address knows.
    #[test]
    fn a_local_configured_server_is_healed_where_it_listens() {
        for local in ["http://127.0.0.1:11500", "http://localhost:11434", "http://0.0.0.0:11434"] {
            let target = ollama_heal_target(local);
            assert_eq!(target.base_url, local, "the ladder must act on the configured server");
            assert!(target.on_this_machine, "{local} is on this machine");
        }
    }

    /// An Ollama-compatible server on a free loopback port that lists a model
    /// to a request carrying `Authorization: Bearer <token>` and answers 401 to
    /// any other — the shape of a server behind an authenticating proxy. The
    /// count is of the requests that came without the token.
    async fn serve_tags_requiring(
        token: &'static str,
    ) -> (String, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        let unauthenticated = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counted = std::sync::Arc::clone(&unauthenticated);
        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                let mut request = [0_u8; 4096];
                let n = socket.read(&mut request).await.unwrap_or(0);
                let head = String::from_utf8_lossy(&request[..n]).to_lowercase();
                let (status, body) = if head.contains(&format!("authorization: bearer {token}")) {
                    ("200 OK", r#"{"models":[{"name":"qwen3:8b","size":7}]}"#)
                } else {
                    counted.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    ("401 Unauthorized", r#"{"error":"unauthorized"}"#)
                };
                let response = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\n\
                     Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = socket.write_all(response.as_bytes()).await;
            }
        });
        (format!("http://{addr}"), unauthenticated)
    }

    /// The wait polls the server chat talks to, with chat's token: that server
    /// being up ends it at once, and a configured address with nothing
    /// listening is not "reachable again" because some other Ollama is.
    #[tokio::test]
    async fn the_readiness_wait_polls_the_configured_server() {
        let (url, unauthenticated) = serve_tags_requiring("s3cret").await;
        let router = LlmRouter::new().with_ollama_authenticated(&url, "s3cret");
        assert!(
            wait_for_ollama_ready(&router, 5).await,
            "the configured server at {url} is up"
        );
        assert_eq!(
            unauthenticated.load(std::sync::atomic::Ordering::Relaxed),
            0,
            "the poll must carry chat's token"
        );

        // Port 9: nothing listens there. A port freed by a dropped listener
        // could be handed to a server another test binds at the same moment.
        let router = LlmRouter::new().with_ollama("http://127.0.0.1:9");
        assert!(
            !wait_for_ollama_ready(&router, 0).await,
            "nothing listens at the configured 127.0.0.1:9"
        );
    }

    /// A server on a free loopback port that answers every request with
    /// `status` and an empty body.
    async fn serve_every_request_with(status: &'static str) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                let mut request = [0_u8; 4096];
                let _ = socket.read(&mut request).await;
                let response =
                    format!("HTTP/1.1 {status}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
                let _ = socket.write_all(response.as_bytes()).await;
            }
        });
        format!("http://{addr}")
    }

    /// The wait starts on a connection-level failure (refused, reset) and is
    /// over when the configured server answers at all. A gateway that routes
    /// `/api/chat` and answers 404 to everything else is back, as is a proxy
    /// answering 401 to a token it no longer takes; what they answer chat is
    /// the retry ladder's to judge. Holding out for an Ollama model list kept
    /// such a server "down" for every wait, six minutes before the retries it
    /// needed.
    #[tokio::test]
    async fn any_answer_from_the_configured_server_ends_the_wait() {
        for status in ["404 Not Found", "401 Unauthorized"] {
            let url = serve_every_request_with(status).await;
            let router = LlmRouter::new().with_ollama(&url);
            assert!(
                wait_for_ollama_ready(&router, 0).await,
                "{url} answered {status}, so it is up"
            );
        }
    }

    /// A repeat GPU fault walks down the window of the server the router
    /// sends the model to, which is where the next attempt goes, and the
    /// runner's budgets are read from that same server. The model's size on
    /// another server (here one on this machine) is not the one that faulted
    /// and stays as it was.
    #[test]
    fn a_gpu_fault_walks_the_window_of_the_server_the_router_sends_to() {
        let model = "ollama/test-router-demotion:8b";
        let bare = "test-router-demotion:8b";
        let other_server = nanna_llm::LlmClient::ollama("http://127.0.0.1:11434");
        assert_eq!(other_server.demote_context(bare, None), Some(24_576));

        let router = LlmRouter::new().with_ollama("https://gpubox.example/ollama");
        assert_eq!(
            demote_context_on_router(&router, model, None),
            Some(12_288),
            "one rung down from the 16384 the remote server's requests carry"
        );
        assert_eq!(
            cached_model_info_on_router(&router, model).context_window,
            12_288
        );
        assert_eq!(other_server.effective_num_ctx(bare), Some(24_576));
    }

    /// Both process-level cures take their server from this guard, through the
    /// router as a heal does: the configured server as its client addresses
    /// it, and only when it is on this machine.
    #[test]
    fn only_a_server_on_this_machine_is_handed_to_the_process_cures() {
        for remote in ["https://gpubox.example/ollama", "http://192.168.1.20:11434"] {
            let router = LlmRouter::new().with_ollama(remote);
            assert_eq!(
                server_for_process_cure(&router, "declined"),
                None,
                "{remote}'s processes are not this machine's"
            );
        }
        let router = LlmRouter::new().with_ollama(" http://localhost:11500/ ");
        assert_eq!(
            server_for_process_cure(&router, "declined").map(|target| target.base_url),
            Some("http://127.0.0.1:11500".to_string()),
            "a local server is cured where chat reaches it"
        );
        assert_eq!(
            server_for_process_cure(&LlmRouter::new(), "declined"),
            None,
            "with no Ollama configured there is nothing to cure"
        );
    }

    /// The runner reset declines a server on another machine end to end: it
    /// returns without the unload POST and the 5s teardown pause that follow
    /// for a local server. 192.0.2.1 is TEST-NET-1 (RFC 5737), routed nowhere,
    /// so a reset that went ahead would fail this on time — the pause alone is
    /// 5s — and reach no server.
    #[tokio::test]
    async fn the_runner_reset_declines_a_server_on_another_machine() {
        let router = LlmRouter::new().with_ollama("http://192.0.2.1:9");
        let started = std::time::Instant::now();
        reset_ollama_runner_for(&router, "qwen3:8b").await;
        assert!(
            started.elapsed() < std::time::Duration::from_secs(5),
            "the reset went ahead against a server elsewhere ({:?})",
            started.elapsed()
        );
    }

    #[test]
    fn transient_classifier_matches_the_live_failure_strings() {
        // The exact strings observed in the daemon log the chat-path healing
        // was built from — each one must be retried, not fatal.
        assert!(is_transient_llm_error(
            "LLM error: HTTP error: error sending request for url (http://localhost:11434/api/chat)"
        ));
        assert!(is_transient_llm_error(
            "LLM error: Stream error: error decoding response body"
        ));
        assert!(is_transient_llm_error(
            "API error: 502 - Ollama aborted generation mid-response (done=false)"
        ));
        assert!(is_transient_llm_error(
            "API error: 502 - Ollama stream ended without completion (no done=true)"
        ));
        assert!(is_transient_llm_error("request timed out"));
        // P22 stream watchdog: a wedged stream future gets a fresh request
        // through the same ladder (persistent wedges still hit the breaker).
        assert!(is_transient_llm_error(
            "step error: stream watchdog: no stream event from ollama/x for 240s"
        ));
        // Non-transient failures must fall through to the next model.
        assert!(!is_transient_llm_error("API error: 401 - invalid api key"));
        assert!(!is_transient_llm_error("API error: 400 - context length exceeded"));
    }

    fn seg(
        steps: usize,
        completed: usize,
        tokens_in: u64,
        tokens_out: u64,
        stop: StopReason,
    ) -> LongHorizonReport {
        LongHorizonReport {
            verified_outcomes: Vec::new(),
            acceptance_unknown: 0,
            tool_calls: 0,
            side_effect_tool_calls: 0,
            stop,
            steps_taken: steps,
            items_completed: completed,
            items_completed_unverified: 0,
            items_revived: 0,
            abandoned_unmet: Vec::new(),
            abandoned_unverifiable: Vec::new(),
            items_regressed_reopened: 0,
            items_already_satisfied: 0,
            items_abandoned: 0,
            last_runner_error: None,
            replans: 0,
            false_success_claims: 0,
            input_tokens: tokens_in,
            output_tokens: tokens_out,
            wall_clock_secs: 60,
            tokens_per_completed_item: None,
            interjected_items: 0,
        }
    }

    #[test]
    fn fold_sums_counters_and_keeps_the_final_stop() {
        let folded = fold_reports(&[
            seg(
                10,
                3,
                1000,
                100,
                StopReason::RunnerErrors {
                    message: "x".to_string(),
                },
            ),
            seg(5, 2, 500, 50, StopReason::AllTasksDone),
        ]);
        assert_eq!(folded.steps_taken, 15);
        assert_eq!(folded.items_completed, 5);
        assert_eq!(folded.input_tokens, 1500);
        assert_eq!(folded.wall_clock_secs, 120);
        assert_eq!(folded.stop, StopReason::AllTasksDone, "final segment's stop wins");
        assert_eq!(
            folded.tokens_per_completed_item,
            Some(330),
            "recomputed over the aggregate (1650/5)"
        );
    }

    #[test]
    fn fold_with_zero_completions_has_no_per_item_rate() {
        let folded = fold_reports(&[seg(4, 0, 100, 10, StopReason::WallClockExhausted)]);
        assert_eq!(folded.tokens_per_completed_item, None);
    }

    #[test]
    fn fold_of_nothing_is_a_benign_empty_report() {
        let folded = fold_reports(&[]);
        assert_eq!(folded.steps_taken, 0);
        assert_eq!(folded.stop, StopReason::AllTasksDone);
    }

    // -----------------------------------------------------------------
    // todo-tool re-decomposition guard
    // -----------------------------------------------------------------

    async fn add_task(services: &HashMap<String, ServiceFn>, params: Value) -> Value {
        services
            .get("tasks.add")
            .expect("tasks.add service")(params)
        .await
        .expect("add succeeds")
    }

    // -----------------------------------------------------------------
    // Decomposition damping — notes, never refusals
    //
    // REGRESSION (gemma4:e4b-it-qat endurance, 2026-08-06): the model
    // recursed features into subtasks of subtasks (103 self-created items
    // against 42 seeded, three levels deep), closed 108 items in 4.5h, and
    // verified 7/42 — the window went to planning. qwen3.5:9b on the same
    // harness created 9 depth-1 subtasks over six hours.
    // -----------------------------------------------------------------

    /// The thresholds separate the two live datapoints exactly: everything
    /// qwen's baseline run did is note-free, and everything in gemma's
    /// signature earns one.
    #[test]
    fn qwen_shaped_planning_is_note_free_and_gemma_shaped_planning_is_not() {
        // Root items and first-level subtasks with few siblings: silent.
        assert_eq!(decomposition_note(0, 0, None), None);
        assert_eq!(decomposition_note(1, 0, Some(1)), None);
        assert_eq!(decomposition_note(1, SIBLING_NUDGE_AT - 1, Some(1)), None);

        // A subtask of a subtask: the do-the-work note.
        let depth2 = decomposition_note(2, 0, Some(9)).expect("depth-2 note");
        assert!(depth2.contains("DO the work"), "{depth2}");

        // Depth three and beyond escalates.
        let depth3 = decomposition_note(3, 0, Some(9)).expect("depth-3 note");
        assert!(depth3.contains("STOP SPLITTING"), "{depth3}");

        // A parent accumulating open subtasks: finish-first note, with the
        // count the model is about to be responsible for.
        let crowded = decomposition_note(1, SIBLING_NUDGE_AT, Some(7)).expect("sibling note");
        assert!(crowded.contains("#7") && crowded.contains("5 open subtasks"), "{crowded}");

        // Both signals join into one note.
        let both = decomposition_note(3, SIBLING_NUDGE_AT, Some(7)).expect("both");
        assert!(both.contains("STOP SPLITTING") && both.contains("open subtasks"), "{both}");
    }

    /// The note rides the created task — creation is NEVER refused. A hard
    /// cap is the owner-rejected design; the damping must be advisory even at
    /// absurd depth.
    #[tokio::test]
    async fn a_sub_subtask_is_created_with_a_note_not_refused() {
        let storage = Arc::new(Storage::in_memory().await.expect("storage"));
        let services = build_task_services(
            storage.clone(),
            Arc::new(RwLock::new(None)),
            Arc::new(TurnBaselines::new()),
        );

        let feature = add_task(
            &services,
            json!({"title": "Feature: exists command", "scope": "session", "session_id": "s1"}),
        )
        .await;
        assert!(feature.get("note").is_none(), "a root item earns no note");

        let sub = add_task(
            &services,
            json!({"title": "parse arguments", "parent_id": feature["task"]["id"]}),
        )
        .await;
        assert!(sub.get("note").is_none(), "a first-level subtask earns no note");

        let subsub = add_task(
            &services,
            json!({"title": "split on tabs", "parent_id": sub["task"]["id"]}),
        )
        .await;
        assert!(subsub["task"]["id"].is_i64(), "creation must never be refused");
        let note = subsub["note"].as_str().expect("depth-2 add carries the note");
        assert!(note.contains("DO the work"), "{note}");

        let subsubsub = add_task(
            &services,
            json!({"title": "handle empty field", "parent_id": subsub["task"]["id"]}),
        )
        .await;
        assert!(subsubsub["task"]["id"].is_i64());
        assert!(
            subsubsub["note"].as_str().expect("note").contains("STOP SPLITTING"),
            "depth 3 escalates"
        );
    }

    /// The sibling nudge counts OPEN children only — a parent whose subtasks
    /// are getting finished can keep planning, because that is the loop
    /// working as intended.
    #[tokio::test]
    async fn the_sibling_nudge_fires_on_open_children_and_clears_as_they_close() {
        let storage = Arc::new(Storage::in_memory().await.expect("storage"));
        let services = build_task_services(
            storage.clone(),
            Arc::new(RwLock::new(None)),
            Arc::new(TurnBaselines::new()),
        );

        let parent = add_task(
            &services,
            json!({"title": "Feature: put command", "scope": "session", "session_id": "s1"}),
        )
        .await;
        let pid = parent["task"]["id"].as_i64().expect("id");

        let mut child_ids = Vec::new();
        for i in 0..SIBLING_NUDGE_AT {
            let child = add_task(
                &services,
                json!({"title": format!("step {i}"), "parent_id": pid}),
            )
            .await;
            child_ids.push(child["task"]["id"].as_i64().expect("id"));
            if i < SIBLING_NUDGE_AT - 1 {
                assert!(
                    child.get("note").is_none(),
                    "under the threshold planning stays silent (child {i})"
                );
            }
        }

        // The add that crossed the line carried the finish-first note.
        let crowded = add_task(&services, json!({"title": "one more", "parent_id": pid})).await;
        assert!(
            crowded["note"].as_str().expect("note").contains("open subtasks"),
            "the parent has {SIBLING_NUDGE_AT}+ open children"
        );

        // Close enough children and the same shape of add is silent again.
        for id in &child_ids {
            storage
                .tasks()
                .update(
                    *id,
                    TaskPatch { status: Some("cancelled".to_string()), ..TaskPatch::default() },
                    Some("test"),
                )
                .await
                .expect("close");
        }
        let after = add_task(&services, json!({"title": "post-close step", "parent_id": pid})).await;
        assert!(
            after.get("note").is_none(),
            "closed children do not count against the parent"
        );
    }

    /// REGRESSION (lfm2.5 smoke, 2026-07-25): the model re-planned the same
    /// work every step, turning 5 seeded tasks into ~50 — "Write data file
    /// with header and 3 rows" was created ten times — so the plan grew
    /// faster than it was worked. Re-adding an OPEN title reuses that item.
    /// When `tasks.add` reuses an item an earlier turn left open, the live
    /// turn admits it — otherwise the model is told "work on it rather than
    /// planning it again" about an item the harness will never schedule.
    #[tokio::test]
    async fn reusing_a_leftover_admits_it_into_the_live_turn() {
        let storage = Arc::new(Storage::in_memory().await.expect("storage"));
        let baselines = Arc::new(TurnBaselines::new());
        let services = build_task_services(
            storage.clone(),
            Arc::new(RwLock::new(None)),
            baselines.clone(),
        );
        let leftover = add_task(
            &services,
            json!({"title": "Write data file", "scope": "session", "session_id": "s1"}),
        )
        .await;
        let leftover_id = leftover["task"]["id"].as_i64().expect("an id");

        // A new turn starts: the item is now a leftover it does not admit.
        let admission = Arc::new(
            TurnAdmission::at_turn_start(&storage, "session", Some("s1"))
                .await
                .expect("snapshot"),
        );
        baselines
            .register_admission("session", Some("s1"), admission.clone())
            .await;
        assert!(!admission.admits(leftover_id));

        let again = add_task(
            &services,
            json!({"title": "write data file", "scope": "session", "session_id": "s1"}),
        )
        .await;
        assert_eq!(again["deduplicated"], json!(true));
        assert!(
            admission.admits(leftover_id),
            "the reused leftover joins the turn that asked for it"
        );

        baselines.close_turn("session", Some("s1")).await;
        assert!(
            baselines.admission("session", Some("s1")).await.is_none(),
            "the admission is dropped with the turn"
        );
    }

    #[tokio::test]
    async fn re_adding_an_open_title_reuses_the_existing_task() {
        let storage = Arc::new(Storage::in_memory().await.expect("storage"));
        let workspace_id = Arc::new(RwLock::new(None));
        let services =
            build_task_services(storage.clone(), workspace_id, Arc::new(TurnBaselines::new()));

        let first = add_task(
            &services,
            json!({"title": "Write data file", "scope": "session", "session_id": "s1"}),
        )
        .await;
        let again = add_task(
            &services,
            json!({"title": "  write DATA file  ", "scope": "session", "session_id": "s1"}),
        )
        .await;

        assert_eq!(
            first["task"]["id"], again["task"]["id"],
            "a re-add of an open title must reuse the item, not clone it"
        );
        assert_eq!(again["deduplicated"], json!(true));
        let open = storage
            .tasks()
            .list("session", Some("s1"), false)
            .await
            .expect("list");
        assert_eq!(open.len(), 1, "exactly one task exists");
    }

    /// The dedupe must not swallow a genuinely recurring chore: once the
    /// previous one is closed, the same title is addable again.
    #[tokio::test]
    async fn the_same_title_is_addable_again_once_the_previous_one_is_closed() {
        let storage = Arc::new(Storage::in_memory().await.expect("storage"));
        let workspace_id = Arc::new(RwLock::new(None));
        let services =
            build_task_services(storage.clone(), workspace_id, Arc::new(TurnBaselines::new()));

        let first = add_task(
            &services,
            json!({"title": "run the tests", "scope": "session", "session_id": "s1"}),
        )
        .await;
        let id = first["task"]["id"].as_i64().expect("id");
        storage
            .tasks()
            .complete(id, Some("test"), None)
            .await
            .expect("complete");

        let second = add_task(
            &services,
            json!({"title": "run the tests", "scope": "session", "session_id": "s1"}),
        )
        .await;
        assert_ne!(
            second["task"]["id"].as_i64(),
            Some(id),
            "a closed chore may be scheduled again"
        );
    }

    /// Sibling scoping: the same subtask title under DIFFERENT parents is
    /// legitimate ("write the header" for two different files).
    #[tokio::test]
    async fn the_same_title_under_a_different_parent_is_not_a_duplicate() {
        let storage = Arc::new(Storage::in_memory().await.expect("storage"));
        let workspace_id = Arc::new(RwLock::new(None));
        let services =
            build_task_services(storage.clone(), workspace_id, Arc::new(TurnBaselines::new()));

        let a = add_task(
            &services,
            json!({"title": "feature A", "scope": "session", "session_id": "s1"}),
        )
        .await;
        let b = add_task(
            &services,
            json!({"title": "feature B", "scope": "session", "session_id": "s1"}),
        )
        .await;
        let sub_a = add_task(
            &services,
            json!({"title": "write the header", "parent_id": a["task"]["id"]}),
        )
        .await;
        let sub_b = add_task(
            &services,
            json!({"title": "write the header", "parent_id": b["task"]["id"]}),
        )
        .await;

        assert_ne!(
            sub_a["task"]["id"], sub_b["task"]["id"],
            "same title under different parents is different work"
        );
    }

    // -----------------------------------------------------------------
    // Closed-since-turn-start guard on the model's own write surface.
    //
    // REGRESSION (live 2026-08-02, ollama/gemma4:12b, session 05775d1d):
    // `seed_continuation` filtered continuation-planner titles closed this
    // turn, but the HARNESS replan path creates items by telling the model to
    // add subtasks through the todo tool — landing in `tasks.add`, which
    // dedups only against OPEN titles. Task #2059 was abandoned and came
    // straight back as #2060 with the identical title, immediately re-seeded.
    // -----------------------------------------------------------------

    /// Mark a task abandoned exactly as `TursoTaskSource::abandon` does.
    async fn abandon(storage: &Arc<Storage>, id: i64) {
        storage
            .tasks()
            .update(
                id,
                TaskPatch {
                    status: Some("cancelled".to_string()),
                    ..TaskPatch::default()
                },
                Some("test"),
            )
            .await
            .expect("abandon");
    }

    #[tokio::test]
    async fn a_title_abandoned_this_turn_is_not_re_created() {
        let storage = Arc::new(Storage::in_memory().await.expect("storage"));
        let baselines = Arc::new(TurnBaselines::new());
        let services = build_task_services(
            storage.clone(),
            Arc::new(RwLock::new(None)),
            baselines.clone(),
        );

        // The turn opens with nothing closed yet — the harness's snapshot.
        baselines
            .open_turn(
                "session",
                Some("s1"),
                closed_task_ids(&storage, "session", Some("s1"))
                    .await
                    .expect("baseline"),
            )
            .await;

        let doomed = add_task(
            &services,
            json!({"title": "Implement the parser", "scope": "session", "session_id": "s1"}),
        )
        .await;
        let doomed_id = doomed["task"]["id"].as_i64().expect("id");
        abandon(&storage, doomed_id).await;

        // The replan re-proposes the very title the run just gave up on.
        let again = add_task(
            &services,
            json!({"title": "  implement THE parser ", "scope": "session", "session_id": "s1"}),
        )
        .await;

        assert_eq!(again["created"], json!(false), "nothing was created");
        assert_eq!(again["closed_this_turn"], json!(true));
        assert_eq!(
            again["task"]["id"].as_i64(),
            Some(doomed_id),
            "the notice points at the settled task, so the claim is checkable"
        );
        let note = again["note"].as_str().expect("a note explains the refusal");
        assert!(note.contains("NOT created"), "{note}");
        assert!(note.contains("abandoned"), "names the verdict: {note}");
        assert!(note.contains("Nothing was written"), "{note}");

        let all = storage
            .tasks()
            .list("session", Some("s1"), true)
            .await
            .expect("list");
        assert_eq!(all.len(), 1, "no #2060 clone exists");
    }

    #[tokio::test]
    async fn a_completed_title_is_refused_the_same_way_within_the_turn() {
        let storage = Arc::new(Storage::in_memory().await.expect("storage"));
        let baselines = Arc::new(TurnBaselines::new());
        let services = build_task_services(
            storage.clone(),
            Arc::new(RwLock::new(None)),
            baselines.clone(),
        );
        baselines
            .open_turn("session", Some("s1"), HashSet::new())
            .await;

        let done = add_task(
            &services,
            json!({"title": "answer the question", "scope": "session", "session_id": "s1"}),
        )
        .await;
        storage
            .tasks()
            .complete(done["task"]["id"].as_i64().expect("id"), Some("test"), None)
            .await
            .expect("complete");

        let again = add_task(
            &services,
            json!({"title": "answer the question", "scope": "session", "session_id": "s1"}),
        )
        .await;
        assert_eq!(again["created"], json!(false));
        assert!(
            again["note"].as_str().expect("note").contains("completed"),
            "a finished title names the finished verdict"
        );
    }

    /// The trade the OPEN-title dedupe was deliberately built around must
    /// survive: a chore closed BEFORE this turn is legitimate work to schedule
    /// again. Only titles closed since the turn's own snapshot are refused.
    #[tokio::test]
    async fn a_title_closed_before_the_turn_started_is_still_addable() {
        let storage = Arc::new(Storage::in_memory().await.expect("storage"));
        let baselines = Arc::new(TurnBaselines::new());
        let services = build_task_services(
            storage.clone(),
            Arc::new(RwLock::new(None)),
            baselines.clone(),
        );

        // Yesterday's chore, closed before the turn opens.
        let yesterday = add_task(
            &services,
            json!({"title": "run the tests", "scope": "session", "session_id": "s1"}),
        )
        .await;
        let yesterday_id = yesterday["task"]["id"].as_i64().expect("id");
        storage
            .tasks()
            .complete(yesterday_id, Some("test"), None)
            .await
            .expect("complete");

        // NOW the turn starts — the snapshot includes yesterday's chore.
        baselines
            .open_turn(
                "session",
                Some("s1"),
                closed_task_ids(&storage, "session", Some("s1"))
                    .await
                    .expect("baseline"),
            )
            .await;

        let today = add_task(
            &services,
            json!({"title": "run the tests", "scope": "session", "session_id": "s1"}),
        )
        .await;
        assert_ne!(
            today["task"]["id"].as_i64(),
            Some(yesterday_id),
            "a recurring chore closed in HISTORY is still schedulable"
        );
        assert!(today["closed_this_turn"].is_null());
    }

    /// Ordering: an OPEN title still reuses its item rather than tripping the
    /// closed guard — reuse points the model at live work, which is strictly
    /// more useful than a refusal.
    #[tokio::test]
    async fn an_open_title_still_reuses_while_a_turn_is_live() {
        let storage = Arc::new(Storage::in_memory().await.expect("storage"));
        let baselines = Arc::new(TurnBaselines::new());
        let services = build_task_services(
            storage.clone(),
            Arc::new(RwLock::new(None)),
            baselines.clone(),
        );
        baselines
            .open_turn("session", Some("s1"), HashSet::new())
            .await;

        let first = add_task(
            &services,
            json!({"title": "build the thing", "scope": "session", "session_id": "s1"}),
        )
        .await;
        let again = add_task(
            &services,
            json!({"title": "build the thing", "scope": "session", "session_id": "s1"}),
        )
        .await;
        assert_eq!(again["deduplicated"], json!(true));
        assert_eq!(first["task"]["id"], again["task"]["id"]);
    }

    /// The guard is scoped to the TURN, not to history: once the turn's exit
    /// tail drops the baseline, the title is addable again.
    #[tokio::test]
    async fn the_guard_lifts_when_the_turn_closes() {
        let storage = Arc::new(Storage::in_memory().await.expect("storage"));
        let baselines = Arc::new(TurnBaselines::new());
        let services = build_task_services(
            storage.clone(),
            Arc::new(RwLock::new(None)),
            baselines.clone(),
        );
        baselines
            .open_turn("session", Some("s1"), HashSet::new())
            .await;

        let done = add_task(
            &services,
            json!({"title": "summarize the log", "scope": "session", "session_id": "s1"}),
        )
        .await;
        storage
            .tasks()
            .complete(done["task"]["id"].as_i64().expect("id"), Some("test"), None)
            .await
            .expect("complete");
        assert_eq!(
            add_task(
                &services,
                json!({"title": "summarize the log", "scope": "session", "session_id": "s1"}),
            )
            .await["created"],
            json!(false)
        );

        // Turn over — the user may legitimately ask for it again.
        baselines.close_turn("session", Some("s1")).await;
        let next_turn = add_task(
            &services,
            json!({"title": "summarize the log", "scope": "session", "session_id": "s1"}),
        )
        .await;
        assert!(
            next_turn["closed_this_turn"].is_null(),
            "a new turn may re-ask: {next_turn}"
        );
    }

    /// Both in-run creation doors answer "did this turn already settle that?"
    /// with the SAME rule — that disagreement was the hole.
    #[tokio::test]
    async fn the_planner_door_and_the_model_door_share_one_rule() {
        let storage = Arc::new(Storage::in_memory().await.expect("storage"));
        let baselines = Arc::new(TurnBaselines::new());
        let services = build_task_services(
            storage.clone(),
            Arc::new(RwLock::new(None)),
            baselines.clone(),
        );
        let baseline = closed_task_ids(&storage, "session", Some("s1"))
            .await
            .expect("baseline");
        baselines
            .open_turn("session", Some("s1"), baseline.clone())
            .await;

        let ids = seed_plan(
            &storage,
            "session",
            Some("s1"),
            &plan_of(&["explain mutexes"]),
            false,
        )
        .await
        .expect("seeded");
        storage
            .tasks()
            .complete(ids[0], Some("test"), None)
            .await
            .expect("complete");

        // Door 1: the continuation planner re-proposes it — filtered.
        let reseeded = seed_continuation(
            &storage,
            "session",
            Some("s1"),
            &plan_of(&["explain mutexes"]),
            &baseline,
        )
        .await
        .expect("continuation");
        assert!(reseeded.is_empty(), "the planner door filters it");

        // Door 2: the model re-adds it through the todo tool — refused, and
        // by the same `titles_closed_since` view.
        let re_added = add_task(
            &services,
            json!({"title": "explain mutexes", "scope": "session", "session_id": "s1"}),
        )
        .await;
        assert_eq!(
            re_added["created"],
            json!(false),
            "the model door refuses it too"
        );
        let settled: Vec<String> = tasks_closed_since(&storage, "session", Some("s1"), &baseline)
            .await
            .expect("settled")
            .into_iter()
            .map(|t| t.title)
            .collect();
        assert_eq!(
            settled,
            vec!["explain mutexes".to_string()],
            "one shared view of what this turn settled"
        );
    }

    // -----------------------------------------------------------------
    // mission-continuation seeding: closed-title dedup
    // -----------------------------------------------------------------

    /// REGRESSION (live 2026-08-02, session 7ccc455a): the continuation loop
    /// re-planned an already-answered question because the finished copy was
    /// CLOSED — only open titles dedupe at add — so every round seeded a
    /// fresh clone and the dry detector never tripped. A continuation plan
    /// made ONLY of titles this turn closed (done OR cancelled, in any case /
    /// whitespace dress) must seed nothing: that is the dry-round signal.
    #[tokio::test]
    async fn continuation_plan_of_only_closed_titles_is_dry() {
        let storage = Arc::new(Storage::in_memory().await.expect("storage"));
        let baseline = closed_task_ids(&storage, "session", Some("s1"))
            .await
            .expect("baseline");

        let ids = seed_plan(
            &storage,
            "session",
            Some("s1"),
            &plan_of(&["Explain mutexes", "Compare semaphores"]),
            false,
        )
        .await
        .expect("initial seed");
        // The run closes one by completion and one by cancellation (poison
        // containment cancels) — both are "closed", both must dedup.
        storage
            .tasks()
            .complete(ids[0], Some("test"), None)
            .await
            .expect("complete");
        storage
            .tasks()
            .update(
                ids[1],
                TaskPatch {
                    status: Some("cancelled".to_string()),
                    ..TaskPatch::default()
                },
                Some("test"),
            )
            .await
            .expect("cancel");

        let seeded = seed_continuation(
            &storage,
            "session",
            Some("s1"),
            &plan_of(&["  explain MUTEXES ", "compare semaphores"]),
            &baseline,
        )
        .await
        .expect("continuation");
        assert!(seeded.is_empty(), "an all-duplicate round is dry");
        let (open, closed) = storage
            .tasks()
            .counts("session", Some("s1"))
            .await
            .expect("counts");
        assert_eq!(open, 0, "no clone was seeded");
        assert_eq!(closed, 2, "the store still holds only the original work");
    }

    /// A genuine mission keeps continuing: a mixed continuation plan seeds
    /// exactly the titles the turn has NOT already closed.
    #[tokio::test]
    async fn continuation_seeds_only_the_new_titles() {
        let storage = Arc::new(Storage::in_memory().await.expect("storage"));
        let baseline = closed_task_ids(&storage, "session", Some("s1"))
            .await
            .expect("baseline");

        let ids = seed_plan(&storage, "session", Some("s1"), &plan_of(&["step 1"]), false)
            .await
            .expect("initial seed");
        storage
            .tasks()
            .complete(ids[0], Some("test"), None)
            .await
            .expect("complete");

        let seeded = seed_continuation(
            &storage,
            "session",
            Some("s1"),
            &plan_of(&["step 1", "step 2"]),
            &baseline,
        )
        .await
        .expect("continuation");
        assert_eq!(seeded.len(), 1, "only the new title seeds");
        let open = storage
            .tasks()
            .list("session", Some("s1"), false)
            .await
            .expect("list");
        assert_eq!(open.len(), 1);
        assert_eq!(open[0].title, "step 2");
    }

    /// The turn-start boundary matters: a title closed BEFORE the turn began
    /// is history, not this turn's work — the planner may legitimately
    /// schedule it again mid-mission.
    #[tokio::test]
    async fn continuation_does_not_filter_titles_closed_before_the_turn() {
        let storage = Arc::new(Storage::in_memory().await.expect("storage"));
        let old = seed_plan(&storage, "session", Some("s1"), &plan_of(&["old chore"]), false)
            .await
            .expect("yesterday's seed");
        storage
            .tasks()
            .complete(old[0], Some("test"), None)
            .await
            .expect("complete");

        // Turn start: the chore is already closed, so it is in the baseline.
        let baseline = closed_task_ids(&storage, "session", Some("s1"))
            .await
            .expect("baseline");
        let seeded = seed_continuation(
            &storage,
            "session",
            Some("s1"),
            &plan_of(&["old chore"]),
            &baseline,
        )
        .await
        .expect("continuation");
        assert_eq!(seeded.len(), 1, "history never dedups a continuation");
    }

    /// Initial turn seeding must NOT dedup against closed history at all: a
    /// user may re-ask today what was answered yesterday, and `seed_plan` is
    /// the initial path. Only continuation rounds carry the closed-title
    /// guard.
    #[tokio::test]
    async fn initial_seeding_is_unaffected_by_closed_history() {
        let storage = Arc::new(Storage::in_memory().await.expect("storage"));
        let first = seed_plan(&storage, "session", Some("s1"), &plan_of(&["run the tests"]), false)
            .await
            .expect("first turn");
        storage
            .tasks()
            .complete(first[0], Some("test"), None)
            .await
            .expect("complete");

        let second = seed_plan(&storage, "session", Some("s1"), &plan_of(&["run the tests"]), false)
            .await
            .expect("the same request on a later turn seeds again");
        assert_eq!(second.len(), 1);
        assert_ne!(second[0], first[0], "a fresh task, not the closed one");
    }

    // -----------------------------------------------------------------
    // Title identity: normalization only, because a match DELETES work
    // -----------------------------------------------------------------

    /// What the rule is allowed to normalize away — case, surrounding
    /// whitespace, punctuation, and trailing-`s` plurals. None of it can name
    /// different work.
    #[test]
    fn normalization_only_ignores_what_cannot_change_the_work() {
        assert!(same_title("Write data file", "  write DATA file  "));
        assert!(same_title("merge the open PRs", "Merge the open PR!"));
        assert!(same_title("fix conflicts", "fix   conflicts."));
        assert!(same_title("run tests: nanna-daemon", "run tests nanna daemon"));
        assert!(same_title("merge PR 163", "  MERGE pr 163 "));
    }

    /// THE correctness bar, and the reason this rule stays this narrow.
    ///
    /// A `same_title` match DELETES work: [`seed_continuation`] drops the
    /// matched proposal and `tasks.add` returns an existing id instead of
    /// creating a task. An earlier revision dropped an inspection-verb filler
    /// list ("check", "enumerate", "verify", "list", "state", "current", …)
    /// and matched token SUBSETS, which reduced "check the current PR state"
    /// to `{pr}` and "check which PRs are still open" to `{pr, open}` — both
    /// subsets of the ACTION titles below, at a share the majority rule
    /// accepted. The filter would then have dropped the planner's proposal to
    /// actually MERGE as a duplicate of a finished VERIFICATION task, and the
    /// mission would have converged reporting success with the merges never
    /// done.
    ///
    /// Every pair here is subset-shaped — the exact region the removed rule
    /// decided — and every one of them must refuse.
    #[test]
    fn a_verification_title_never_swallows_the_action_it_verifies() {
        // Inspection vs action: the live 2026-08-03 titles against the work
        // the mission actually existed to do.
        assert!(!same_title(
            "check which PRs are still open",
            "merge all open PRs"
        ));
        assert!(!same_title("check the current PR state", "merge the PRs"));
        // Compound work: the added half is REAL work, not a qualifier.
        assert!(!same_title(
            "run the tests",
            "run the tests and fix the failures"
        ));
        assert!(!same_title("fix conflicts", "fix conflicts in nanna-daemon"));
        assert!(!same_title(
            "merge the PRs",
            "merge the PRs after resolving the nanna-daemon conflicts"
        ));
        // A qualifier that narrows the target is still a different target.
        assert!(!same_title("merge PR", "merge PR 163"));
        assert!(!same_title("run the tests", "run the tests again"));
    }

    /// Different targets, different work — the identifier bar the earlier
    /// revision handled with a dedicated numeric rule and sequence equality
    /// gets for free.
    #[test]
    fn titles_naming_different_targets_are_not_duplicates() {
        assert!(!same_title("merge PR 163", "merge PR 161"));
        assert!(!same_title("step 1", "step 2"));
        assert!(!same_title(
            "fix conflicts in nanna-daemon",
            "fix conflicts in nanna-gui"
        ));
        assert!(!same_title("merge PR 163", "close PR 163"));
        assert!(!same_title("feature A", "feature B"));
        // Same words, opposite jobs: why identity is a SEQUENCE, not a set.
        assert!(!same_title(
            "merge master into the PR",
            "merge the PR into master"
        ));
    }

    /// ACCEPTED CONSEQUENCE, asserted so it cannot be lost by accident: the
    /// live transcript's rephrasings used DIFFERENT VERBS, and no honest
    /// string rule can call them the same work. Title matching therefore does
    /// NOT converge that mission — the acceptance pre-check does (see
    /// `LongHorizonConfig::precheck_acceptance_items`). The safe failure
    /// direction is one extra round, never deleted work.
    #[test]
    fn rephrasing_with_a_different_verb_is_not_matched_by_title() {
        let live = [
            "check the current PR state",
            "enumerate the currently open PRs",
            "check which PRs are still open",
        ];
        for (i, a) in live.iter().enumerate() {
            for b in &live[i + 1..] {
                assert!(
                    !same_title(a, b),
                    "{a:?} and {b:?} differ in words that could name real work — \
                     convergence is the acceptance pre-check's job, not this rule's"
                );
            }
        }
    }

    /// A title with no alphanumerics normalizes to nothing, and "nothing ==
    /// nothing" would match every such title with every other. Fall back to
    /// exact identity instead.
    #[test]
    fn a_contentless_title_falls_back_to_exact_identity() {
        assert!(same_title("???", " ??? "));
        assert!(!same_title("???", "!!!"));
        assert!(!same_title("???", "merge PR 163"));
    }

    /// End to end through the seeding path: a re-proposal that differs only
    /// in the ways normalization ignores seeds NOTHING once the original is
    /// closed — a dry round, which is what the mission loop needs.
    #[tokio::test]
    async fn continuation_filters_the_renormalized_duplicates() {
        let storage = Arc::new(Storage::in_memory().await.expect("storage"));
        let baseline = closed_task_ids(&storage, "session", Some("s1"))
            .await
            .expect("baseline");

        let ids = seed_plan(
            &storage,
            "session",
            Some("s1"),
            &plan_of(&["merge the open PRs"]),
            false,
        )
        .await
        .expect("initial seed");
        storage
            .tasks()
            .complete(ids[0], Some("test"), None)
            .await
            .expect("complete");

        let seeded = seed_continuation(
            &storage,
            "session",
            Some("s1"),
            &plan_of(&["Merge the open PR.", "merge   the OPEN prs"]),
            &baseline,
        )
        .await
        .expect("continuation");
        assert!(
            seeded.is_empty(),
            "the same sentence respelled is not new work"
        );
    }

    /// …and the filter never eats the mission's actual job: the ACTION title
    /// seeds even though a VERIFICATION title about the same subject already
    /// closed this turn. This is the case the blocked revision got wrong.
    #[tokio::test]
    async fn continuation_seeds_the_action_after_a_verification_closed() {
        let storage = Arc::new(Storage::in_memory().await.expect("storage"));
        let baseline = closed_task_ids(&storage, "session", Some("s1"))
            .await
            .expect("baseline");

        let ids = seed_plan(
            &storage,
            "session",
            Some("s1"),
            &plan_of(&["check which PRs are still open"]),
            false,
        )
        .await
        .expect("initial seed");
        storage
            .tasks()
            .complete(ids[0], Some("test"), None)
            .await
            .expect("complete");

        let seeded = seed_continuation(
            &storage,
            "session",
            Some("s1"),
            &plan_of(&["merge all open PRs"]),
            &baseline,
        )
        .await
        .expect("continuation");
        assert_eq!(seeded.len(), 1, "merging is the work; it must survive");
        let open = storage
            .tasks()
            .list("session", Some("s1"), false)
            .await
            .expect("list");
        assert_eq!(open[0].title, "merge all open PRs");
    }

    /// …and the mission still moves: a different PR number is different work
    /// and seeds, even though every other word matches a closed title.
    #[tokio::test]
    async fn continuation_seeds_a_different_identifier() {
        let storage = Arc::new(Storage::in_memory().await.expect("storage"));
        let baseline = closed_task_ids(&storage, "session", Some("s1"))
            .await
            .expect("baseline");

        let ids = seed_plan(
            &storage,
            "session",
            Some("s1"),
            &plan_of(&["merge PR 161"]),
            false,
        )
        .await
        .expect("initial seed");
        storage
            .tasks()
            .complete(ids[0], Some("test"), None)
            .await
            .expect("complete");

        let seeded = seed_continuation(
            &storage,
            "session",
            Some("s1"),
            &plan_of(&["merge PR 163"]),
            &baseline,
        )
        .await
        .expect("continuation");
        assert_eq!(seeded.len(), 1, "the next PR is real work");
        let open = storage
            .tasks()
            .list("session", Some("s1"), false)
            .await
            .expect("list");
        assert_eq!(open[0].title, "merge PR 163");
    }

    /// REGRESSION (lfm2.5 smoke, 2026-07-25): the model deleted a SEEDED
    /// plan item it could not finish, and the run then panicked on a task the
    /// harness still expected to verify. A task with an acceptance contract
    /// is never the model's to erase.
    #[tokio::test]
    async fn a_task_with_an_acceptance_contract_cannot_be_deleted() {
        let storage = Arc::new(Storage::in_memory().await.expect("storage"));
        let workspace_id = Arc::new(RwLock::new(None));
        let services =
            build_task_services(storage.clone(), workspace_id, Arc::new(TurnBaselines::new()));

        let seeded = storage
            .tasks()
            .create(NewTask {
                scope: "session".to_string(),
                scope_id: Some("s1".to_string()),
                title: "Create the greeting file".to_string(),
                priority: 3,
                acceptance: Some(json!({
                    "kind": "regex", "path": "greeting.txt", "pattern": "hello"
                })),
                ..NewTask::default()
            })
            .await
            .expect("seed");

        let out = services
            .get("tasks.remove")
            .expect("tasks.remove service")(json!({ "id": seeded.id }))
        .await
        .expect("call succeeds");

        assert_eq!(out["removed"], json!(false));
        assert_eq!(out["refused"], json!(true));
        assert!(
            storage.tasks().get(seeded.id).await.is_ok(),
            "the seeded task must still exist"
        );
    }

    /// REGRESSION (lfm2.5 endurance, 2026-07-25): guarding only the per-id
    /// remove left `clear` open, and one call took the scope from 42 tasks to
    /// 6 mid-run — destroying 36 seeded features the harness was still
    /// driving. Bulk clear obeys the same contract rule, including for
    /// ancestors (delete removes whole subtrees).
    #[tokio::test]
    async fn clear_keeps_every_task_carrying_a_contract() {
        let storage = Arc::new(Storage::in_memory().await.expect("storage"));
        let workspace_id = Arc::new(RwLock::new(None));
        let services =
            build_task_services(storage.clone(), workspace_id, Arc::new(TurnBaselines::new()));

        let seeded = |title: &str| NewTask {
            scope: "session".to_string(),
            scope_id: Some("s1".to_string()),
            title: title.to_string(),
            priority: 3,
            acceptance: Some(json!({"kind": "regex", "path": "f", "pattern": "x"})),
            ..NewTask::default()
        };
        let a = storage.tasks().create(seeded("feature A")).await.expect("a");
        let b = storage.tasks().create(seeded("feature B")).await.expect("b");

        // A scratch parent whose CHILD holds a contract: clearing the parent
        // would take the child with it, so the parent is protected too.
        let scratch_parent = add_task(
            &services,
            json!({"title": "scratch parent", "scope": "session", "session_id": "s1"}),
        )
        .await;
        let parent_id = scratch_parent["task"]["id"].as_i64().expect("id");
        let mut child = seeded("contract child");
        child.parent_id = Some(parent_id);
        let child = storage.tasks().create(child).await.expect("child");

        let free = add_task(
            &services,
            json!({"title": "pure scratch", "scope": "session", "session_id": "s1"}),
        )
        .await;
        let free_id = free["task"]["id"].as_i64().expect("id");

        let out = services
            .get("tasks.clear")
            .expect("tasks.clear service")(
            json!({"scope": "session", "session_id": "s1", "closed_only": false}),
        )
        .await
        .expect("clear succeeds");

        assert_eq!(
            out["protected"],
            json!(4),
            "A, B, the contract child, and the scratch parent that would take it down"
        );
        for id in [a.id, b.id, child.id, parent_id] {
            assert!(
                storage.tasks().get(id).await.is_ok(),
                "task #{id} must survive a bulk clear"
            );
        }
        assert!(
            storage.tasks().get(free_id).await.is_err(),
            "contract-free scratch is still cleared"
        );
    }

    /// The guard must not make the store un-tidyable: scratch items the model
    /// invents for itself carry no acceptance check and stay removable.
    #[tokio::test]
    async fn a_scratch_task_without_acceptance_is_still_removable() {
        let storage = Arc::new(Storage::in_memory().await.expect("storage"));
        let workspace_id = Arc::new(RwLock::new(None));
        let services =
            build_task_services(storage.clone(), workspace_id, Arc::new(TurnBaselines::new()));

        let scratch = add_task(
            &services,
            json!({"title": "think about it", "scope": "session", "session_id": "s1"}),
        )
        .await;
        let id = scratch["task"]["id"].as_i64().expect("id");

        let out = services
            .get("tasks.remove")
            .expect("tasks.remove service")(json!({ "id": id }))
        .await
        .expect("call succeeds");

        assert_ne!(
            out["removed"],
            json!(false),
            "a scratch item must actually be removed (the store reports a count)"
        );
        assert!(out.get("refused").is_none(), "no contract, so no refusal");
        assert!(storage.tasks().get(id).await.is_err(), "scratch item is gone");
    }

    // -----------------------------------------------------------------
    // Planner cancellation (abortive Stop)
    // -----------------------------------------------------------------

    /// A runner that never returns — the stand-in for a wedged or slow
    /// provider mid-planning.
    struct NeverReturns;

    #[async_trait::async_trait]
    impl StepRunner for NeverReturns {
        async fn run_step(&self, _request: StepRequest) -> Result<StepOutcome, String> {
            std::future::pending().await
        }
    }

    /// Stop during planning must abort the planning call immediately, not
    /// wait out the 30s planner timeout (observed live 2026-07-31: Stop in
    /// the planning phase hung the turn for the full window).
    #[tokio::test]
    async fn a_cancel_during_planning_returns_the_fallback_plan_promptly() {
        let planner = AgentPlanner::new(Arc::new(NeverReturns));
        let cancel = CancelToken::new();
        let stop = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            stop.cancel();
        });
        let plan = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            planner.plan("build the thing", None, Some(&cancel)),
        )
        .await
        .expect("cancel must abort planning promptly, not wait out the timeout");
        assert_eq!(plan.origin, nanna_agent::PlanOrigin::Fallback);
    }

    /// Counts how often it is asked to plan.
    struct CountsCalls(Arc<std::sync::atomic::AtomicUsize>);

    #[async_trait::async_trait]
    impl StepRunner for CountsCalls {
        async fn run_step(&self, _request: StepRequest) -> Result<StepOutcome, String> {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(StepOutcome {
                text: String::new(),
                input_tokens: 0,
                output_tokens: 0,
                tool_calls: Vec::new(),
                touched_paths: Vec::new(),
                degenerate_loop: false,
            })
        }
    }

    /// A turn cancelled before planning starts must not invoke the model at
    /// all — the caller then skips seeding, so no run ever starts.
    #[tokio::test]
    async fn an_already_cancelled_plan_never_invokes_the_runner() {
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let planner = AgentPlanner::new(Arc::new(CountsCalls(calls.clone())));
        let cancel = CancelToken::new();
        cancel.cancel();
        let plan = planner.plan("hello", None, Some(&cancel)).await;
        assert_eq!(plan.origin, nanna_agent::PlanOrigin::Fallback);
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    // -----------------------------------------------------------------
    // ChatSink → ExternalRunHandle (P19 navigation recovery)
    // -----------------------------------------------------------------

    fn external_run_handle() -> crate::agent_service::ExternalRunHandle {
        crate::agent_service::ExternalRunHandle {
            cancel: nanna_agent::CancelToken::new(),
            accumulated_text: Arc::new(std::sync::Mutex::new(String::new())),
            accumulated_thinking: Arc::new(std::sync::Mutex::new(String::new())),
            active_tool_calls: Arc::new(std::sync::Mutex::new(Vec::new())),
            completed_tool_calls: Arc::new(std::sync::Mutex::new(Vec::new())),
            timeline: Arc::new(std::sync::Mutex::new(Vec::new())),
        }
    }

    fn chat_sink(run: crate::agent_service::ExternalRunHandle) -> ChatSink {
        let (event_tx, _) = tokio::sync::broadcast::channel(64);
        ChatSink {
            session_id: "s1".to_string(),
            message_id: "m1".to_string(),
            event_tx,
            run: Some(run),
            tool_stats: None,
            storage: None,
            liveness: None,
            quiet_item: Arc::new(std::sync::Mutex::new(None)),
            text_join: Arc::default(),
        }
    }

    /// REGRESSION (P19 live drive, 2026-07-24): tool calls vanished when the
    /// user navigated away mid-run, because the harness streamed events only —
    /// `get_run_state` recovery reads the `ActiveChat` buffers, and nothing was
    /// filling them. Every sink callback must land in the shared run handle,
    /// not just on the event bus.
    #[tokio::test]
    async fn tool_calls_survive_navigation_via_run_buffers() {
        let run = external_run_handle();
        let sink = chat_sink(run.clone());

        sink.delta("running ls ");
        sink.thinking("what files exist?");
        sink.tool_start("c1", "exec", &json!({"cmd": "ls"}), None);
        sink.tool_end("c1", "exec", "file.txt", true, 12, None);
        sink.delta("done");

        // What get_run_state serves after navigating away and back:
        assert_eq!(
            crate::agent_service::lock_buf(&run.accumulated_text).as_str(),
            "running ls done"
        );
        assert_eq!(
            crate::agent_service::lock_buf(&run.accumulated_thinking).as_str(),
            "what files exist?"
        );
        assert!(
            crate::agent_service::lock_buf(&run.active_tool_calls).is_empty(),
            "the call completed — it must not linger as active"
        );
        {
            let done = crate::agent_service::lock_buf(&run.completed_tool_calls);
            assert_eq!(done.len(), 1);
            assert_eq!(done[0].name, "exec");
            assert_eq!(done[0].output, "file.txt");
            assert!(done[0].success);
            drop(done);
        }

        // And the journal persisted with the final message: text merged per
        // burst, the tool call back-filled in place with its result.
        let journal = run.timeline.lock().unwrap().clone();
        assert_eq!(journal.len(), 4, "text, thinking, tool, text — in order");
        assert!(matches!(
            &journal[0],
            crate::session::TimelineItem::Text { content, .. } if content == "running ls "
        ));
        assert!(matches!(
            &journal[1],
            crate::session::TimelineItem::Thinking { content, .. } if content == "what files exist?"
        ));
        assert!(matches!(
            &journal[2],
            crate::session::TimelineItem::Tool {
                call_id,
                output: Some(output),
                success: Some(true),
                duration_ms: Some(12),
                ..
            } if call_id == "c1" && output == "file.txt"
        ));
        assert!(matches!(
            &journal[3],
            crate::session::TimelineItem::Text { content, .. } if content == "done"
        ));
    }

    /// The context gauge, end to end from the sink's side: every LLM request
    /// reports its prompt size against the window the loop enforced, and that
    /// pair leaves the daemon as a `ContextUsage` event. Without this callback
    /// the harness chat path emitted none at all, so every client's meter read
    /// 0 of 0 for the whole run while the daemon knew the real numbers.
    #[tokio::test]
    async fn every_request_reports_its_context_usage() {
        let sink = chat_sink(external_run_handle());
        let mut events = sink.event_tx.subscribe();

        let on_usage = sink.usage_callback();
        on_usage(12_400, 640, 27_904);

        match events.try_recv().expect("a usage event was broadcast") {
            Event::ContextUsage { session_id, used, window } => {
                assert_eq!(session_id, "s1");
                assert_eq!(used, 12_400, "used is the request's prompt size");
                assert_eq!(window, 27_904, "window is the loop's enforced input bound");
            }
            other => panic!("expected ContextUsage, got {other:?}"),
        }

        // The denominator follows a mid-run demotion rather than latching:
        // a meter measured against a window the model no longer has is a
        // wrong reading, not a stale one.
        on_usage(3_900, 120, 4_608);
        match events.try_recv().expect("the second request reports too") {
            Event::ContextUsage { used, window, .. } => {
                assert_eq!(used, 3_900);
                assert_eq!(window, 4_608);
            }
            other => panic!("expected ContextUsage, got {other:?}"),
        }
    }

    /// "It should feel like the chat did before": a one-task plan is a
    /// conversation-shaped turn and is announced not at all. Items that join
    /// later (interjection, replan) ARE announced — as a timeline Step, the
    /// status row the GUI renders, never as text in the reply. Asserting on
    /// `accumulated_text` here would pass again the moment a banner leaked
    /// back into the message, so the test pins both halves: nothing in the
    /// text, an entry in the journal.
    #[tokio::test]
    async fn a_one_task_plan_streams_with_no_step_banner() {
        let run = external_run_handle();
        let sink = chat_sink(run.clone());
        *sink.quiet_item.lock().unwrap() = Some(7);

        let request = |item_id: i64| StepRequest {
            item_title: format!("test item {item_id}"),
            item_id,
            step_index: 0,
            step_kind: nanna_agent::harness::StepKind::Execute,
            prompt: format!("Task #{item_id}: reply to the user"),
            tool_scope: Vec::new(),
            token_budget: None,
            max_iterations: None,
            max_wall_clock: None,
            cancel: None,
        };

        sink.step_header(&request(7));
        assert!(
            crate::agent_service::lock_buf(&run.accumulated_text).is_empty(),
            "conversational turn: no banner"
        );

        assert!(
            run.timeline.lock().unwrap().is_empty(),
            "the quiet item must not even reach the journal"
        );

        sink.step_header(&request(9));
        let streamed = crate::agent_service::lock_buf(&run.accumulated_text).clone();
        assert!(
            streamed.is_empty(),
            "run mechanics must NEVER be message text (got: {streamed:?})"
        );
        let journal = run.timeline.lock().unwrap().clone();
        match journal.as_slice() {
            [crate::session::TimelineItem::Step { phase, label, item_id, .. }] => {
                assert_eq!(phase, "working");
                assert_eq!(label, "test item 9", "the label is the item TITLE");
                assert_eq!(*item_id, 9);
            }
            other => panic!("expected one Step entry, got: {other:?}"),
        }
    }

    fn execute_request(item_id: i64, step_index: usize) -> StepRequest {
        StepRequest {
            item_title: format!("test item {item_id}"),
            item_id,
            step_index,
            step_kind: nanna_agent::harness::StepKind::Execute,
            prompt: format!("Task #{item_id}: reply to the user"),
            tool_scope: Vec::new(),
            token_budget: None,
            max_iterations: None,
            max_wall_clock: None,
            cancel: None,
        }
    }

    /// Consecutive steps are separate actions streaming into one reply, and
    /// used to abut: `…the file.The file says…` in the stream and in the
    /// persisted message. Each step's text now opens a new paragraph — for
    /// quiet (conversation-shaped) items too, which are exactly the ones with
    /// no banner or journal entry to mark the seam.
    #[tokio::test]
    async fn each_step_opens_a_new_paragraph_in_the_reply() {
        let run = external_run_handle();
        let sink = chat_sink(run.clone());
        *sink.quiet_item.lock().unwrap() = Some(7);

        sink.step_header(&execute_request(7, 0));
        sink.delta("Let me check");
        sink.delta(" the file.");
        sink.step_header(&execute_request(7, 1));
        sink.delta("It says hello.");

        assert_eq!(
            crate::agent_service::lock_buf(&run.accumulated_text).as_str(),
            "Let me check the file.\n\nIt says hello.",
            "one break at the step seam, none inside a step, none before the first text"
        );
    }

    /// Two steps that say the same thing are the converging repeat: the join
    /// reports the second copy and exactly where it sits in the reply — even
    /// with the run summary after it, which is not step text.
    #[tokio::test]
    async fn a_repeated_step_is_reported_with_its_span() {
        let run = external_run_handle();
        let sink = chat_sink(run.clone());
        sink.step_header(&execute_request(1, 0));
        sink.delta("Paris.");
        sink.step_header(&execute_request(1, 1));
        sink.delta("Paris.");
        sink.text_join.lock().unwrap().end_steps();
        sink.delta("\n\n_2 steps_");

        let reply = crate::agent_service::lock_buf(&run.accumulated_text).clone();
        let (repeat, span) = sink
            .text_join
            .lock()
            .unwrap()
            .repeated_last_step()
            .expect("the second step repeats the first");
        assert_eq!(repeat, "Paris.");
        assert_eq!(
            &reply[span], "\n\nParis.",
            "the separator and the copy, nothing else"
        );

        let other = chat_sink(external_run_handle());
        other.step_header(&execute_request(1, 0));
        other.delta("Paris.");
        other.step_header(&execute_request(1, 1));
        other.delta("Lyon.");
        assert!(
            other
                .text_join
                .lock()
                .unwrap()
                .repeated_last_step()
                .is_none()
        );
    }

    /// The break is only added where the seam has no whitespace of its own:
    /// a model that already ended on a newline, or starts with one, keeps its
    /// own spacing, and a step that emits nothing leaves no stray break.
    #[tokio::test]
    async fn a_step_seam_that_already_has_whitespace_is_left_alone() {
        let run = external_run_handle();
        let sink = chat_sink(run.clone());

        sink.step_header(&execute_request(1, 0));
        sink.delta("First.\n");
        sink.step_header(&execute_request(1, 1));
        sink.delta("Second.");
        sink.step_header(&execute_request(1, 2));
        sink.delta("\nThird.");
        // A step with no text, then one with text: exactly one break.
        sink.step_header(&execute_request(1, 3));
        sink.step_header(&execute_request(1, 4));
        sink.delta("Fourth.");

        assert_eq!(
            crate::agent_service::lock_buf(&run.accumulated_text).as_str(),
            "First.\nSecond.\nThird.\n\nFourth."
        );
    }

    /// Parity with the retired direct chat path: completed tool calls feed
    /// the shared stats tracker (the recording is handed off to a task, so
    /// poll briefly).
    #[tokio::test]
    async fn chat_tool_calls_are_recorded_to_tool_stats() {
        let stats = nanna_agent::ToolStatsTracker::new();
        let mut sink = chat_sink(external_run_handle());
        sink.tool_stats = Some(stats.clone());

        sink.tool_end("c1", "exec", "ok", true, 5, None);

        for _ in 0..200 {
            if stats.export_json().await.to_string().contains("exec") {
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        panic!("tool call was never recorded to stats");
    }

    /// P22 Tier 4: a breaker replay (marked `short_circuited` in the tool
    /// result's structured data) is recorded as its own outcome, never as a
    /// tool failure — hours of replays must not read as a broken tool.
    #[tokio::test]
    async fn short_circuited_replays_are_not_recorded_as_failures() {
        let stats = nanna_agent::ToolStatsTracker::new();
        let mut sink = chat_sink(external_run_handle());
        sink.tool_stats = Some(stats.clone());

        let marker = serde_json::json!({ "short_circuited": true });
        sink.tool_end(
            "c1",
            "list_dir",
            "[ZERO-INFORMATION BREAKER] This call was NOT executed. …",
            false,
            0,
            Some(&marker),
        );

        for _ in 0..200 {
            if let Some(s) = stats.summary("list_dir").await {
                assert_eq!(s.short_circuit_count, 1);
                assert_eq!(s.failure_count, 0, "a replay is not a tool failure");
                assert!(s.top_errors.is_empty(), "the notice is not a tool error");
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        panic!("short-circuited call was never recorded to stats");
    }

    /// The task-run sink journals through the chat path's writer, so its
    /// record obeys the same bound: a huge tool output is capped with a
    /// self-describing marker instead of being stored whole.
    #[test]
    fn a_task_runs_journal_caps_tool_output() {
        let run = external_run_handle();
        let sink = chat_sink(run.clone());
        let huge = "x".repeat(10_000);
        sink.tool_start("c1", "exec", &json!({"cmd": "cat big"}), None);
        sink.tool_end("c1", "exec", &huge, true, 3, None);

        let journal = run.timeline.lock().unwrap().clone();
        match journal.as_slice() {
            [
                crate::session::TimelineItem::Tool {
                    output: Some(out), ..
                },
            ] => {
                assert!(
                    out.len() < huge.len(),
                    "the journal must not store it whole"
                );
                assert!(
                    out.contains("journal shows the first"),
                    "…and says so: {}",
                    out.get(4000..).unwrap_or_default()
                );
            }
            other => panic!("expected one tool item, got {other:?}"),
        }
    }

    /// Ollama reuses call ids per response. An end whose start was never
    /// journaled must be RECORDED, not written over an earlier completed call
    /// that happens to share the id — the old hand-rolled back-fill did the
    /// latter, erasing the earlier call's outcome.
    #[test]
    fn a_reused_call_id_never_overwrites_an_earlier_outcome() {
        let run = external_run_handle();
        let sink = chat_sink(run.clone());
        sink.tool_start("toolu_00000000", "exec", &json!({"cmd": "ls"}), None);
        sink.tool_end("toolu_00000000", "exec", "first", true, 1, None);
        sink.tool_end("toolu_00000000", "exec", "second", false, 2, None);

        let journal = run.timeline.lock().unwrap().clone();
        assert_eq!(journal.len(), 2, "the orphan end is its own record");
        assert!(
            matches!(
                &journal[0],
                crate::session::TimelineItem::Tool { output: Some(out), success: Some(true), .. } if out == "first"
            ),
            "the earlier outcome survives"
        );
        assert!(matches!(
            &journal[1],
            crate::session::TimelineItem::Tool { output: Some(out), success: Some(false), .. } if out == "second"
        ));
    }

    /// P18: an unattended task run is where a user most needs to see what
    /// each edit did, so the sink journals the diff like the chat path.
    #[test]
    fn a_task_runs_journal_keeps_an_edits_diff() {
        let run = external_run_handle();
        let sink = chat_sink(run.clone());
        let data = json!({ "diff": {
            "start_line": 4, "removed": ["old"], "added": ["new"], "truncated": false
        }});
        sink.tool_start("c1", "edit_file", &json!({}), None);
        sink.tool_end("c1", "edit_file", "Edited a.txt", true, 2, Some(&data));

        let journal = run.timeline.lock().unwrap().clone();
        match journal.as_slice() {
            [
                crate::session::TimelineItem::Tool {
                    diff: Some(diff), ..
                },
            ] => {
                assert_eq!(diff.start_line, 4);
                assert_eq!(diff.removed, vec!["old"]);
            }
            other => panic!("expected the edit with its diff, got {other:?}"),
        }
    }
}

/// A live long-horizon run.
struct ActiveRun {
    cancel: CancelToken,
    goal: String,
    started_at: chrono::DateTime<chrono::Utc>,
}

/// Status snapshot returned over IPC.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RunStatus {
    pub running: bool,
    pub goal: Option<String>,
    pub started_at: Option<String>,
    pub last_report: Option<LongHorizonReport>,
    /// Provider incidents healed by resuming the plan (0 when none).
    pub resumes: usize,
}

/// What a long-horizon run works on: the goal, the plan it drains, the runner
/// that executes each step, the run's bounds, and the directory it runs in.
pub struct TaskRunSpec {
    pub goal: String,
    pub source: TursoTaskSource,
    pub runner: AgentStepRunner,
    pub config: LongHorizonConfig,
    pub workdir: PathBuf,
}

/// Starts, cancels, and reports background long-horizon runs — one per scope
/// key at a time (the store serializes the plan; two runners over one plan
/// would race `next()`).
#[derive(Default)]
pub struct TaskRunManager {
    runs: RwLock<HashMap<String, ActiveRun>>,
    reports: RwLock<HashMap<String, (LongHorizonReport, usize)>>,
}

impl TaskRunManager {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn scope_key(scope: &str, scope_id: Option<&str>) -> String {
        format!("{scope}:{}", scope_id.unwrap_or(""))
    }

    /// Spawn a background run. Errors if one is already active for the scope.
    ///
    /// # Errors
    ///
    /// Returns an error when a run is already active for the source's scope.
    pub async fn start(
        self: &Arc<Self>,
        goal: String,
        source: TursoTaskSource,
        runner: AgentStepRunner,
        config: LongHorizonConfig,
        workdir: PathBuf,
        event_tx: tokio::sync::broadcast::Sender<Event>,
    ) -> Result<(), String> {
        let spec = TaskRunSpec { goal, source, runner, config, workdir };
        self.start_with_interjector(spec, event_tx, None).await
    }

    /// [`Self::start`], plus the hook that lets user messages join this run
    /// at a step boundary instead of queueing behind it. Chat-backed runs
    /// always pass one; background runs do not.
    ///
    /// # Errors
    ///
    /// Returns an error when a run is already active for the source's scope, or
    /// when the run registered here is gone again before its cancel token can
    /// be read back.
    pub async fn start_with_interjector(
        self: &Arc<Self>,
        spec: TaskRunSpec,
        event_tx: tokio::sync::broadcast::Sender<Event>,
        interjector: Option<Arc<SessionInterjector>>,
    ) -> Result<(), String> {
        let TaskRunSpec { goal, source, runner, config, workdir } = spec;
        let key = Self::scope_key(&source.scope, source.scope_id.as_deref());
        {
            let mut runs = self.runs.write().await;
            if runs.contains_key(&key) {
                return Err(format!("a task run is already active for scope {key}"));
            }
            runs.insert(
                key.clone(),
                ActiveRun {
                    cancel: CancelToken::new(),
                    goal: goal.clone(),
                    started_at: chrono::Utc::now(),
                },
            );
        }
        let cancel = {
            let runs = self.runs.read().await;
            runs.get(&key).map(|r| r.cancel.clone())
        }
        .ok_or_else(|| "run vanished during start".to_string())?;

        let scope = source.scope.clone();
        let scope_id = source.scope_id.clone();
        let _ = event_tx.send(Event::TaskRunStarted {
            scope: scope.clone(),
            scope_id: scope_id.clone(),
            goal: goal.clone(),
        });

        let manager = self.clone();
        tokio::spawn(async move {
            // Bounded auto-resume (standard for every model): the task store
            // IS the checkpoint, so a run stopped by a provider incident is
            // resumed by simply starting again — next() picks up exactly
            // where the plan stands. Bound: 8 resumes tolerates an incident
            // every ~40 minutes of a long run without letting a permanently
            // dead provider spin forever.
            const RUN_RESUMES_MAX: usize = 8;
            let started = std::time::Instant::now();
            let mut segments: Vec<LongHorizonReport> = Vec::new();
            let mut resumes = 0usize;
            loop {
                let remaining = config.max_wall_clock.saturating_sub(started.elapsed());
                let segment_config = LongHorizonConfig {
                    max_wall_clock: remaining,
                    ..config.clone()
                };
                let report = LongHorizonRunner::new(segment_config)
                    .run_with_interjector(
                        &goal,
                        &source,
                        &runner,
                        &workdir,
                        Some(cancel.clone()),
                        interjector.as_deref().map(|i| i as &dyn Interjector),
                    )
                    .await;
                let provider_died = matches!(report.stop, StopReason::RunnerErrors { .. });
                segments.push(report);
                if !provider_died
                    || resumes >= RUN_RESUMES_MAX
                    || started.elapsed() >= config.max_wall_clock
                {
                    break;
                }
                resumes += 1;
                tracing::warn!(
                    scope = %scope,
                    resumes,
                    "provider incident — healing and resuming the plan"
                );
                let _ = event_tx.send(Event::TaskRunProgress {
                    scope: scope.clone(),
                    scope_id: scope_id.clone(),
                    task_id: None,
                    kind: "resumed".to_string(),
                    detail: serde_json::json!({ "resumes": resumes }),
                });
                // Healing ladder — provider-aware. Local Ollama: server
                // restart (unless opted out), else runner reset. Cloud
                // providers (incl. the openrouter/free auto-router, where the
                // serving model varies per request) and an Ollama server on
                // another machine (both calls decline it): nothing local to
                // heal — the pause + resume + in-step retries ARE the healing.
                if runner.is_ollama_model()
                    && !(ollama_restart_allowed() && restart_ollama_server(&runner.router).await)
                {
                    runner.reset_ollama_runner().await;
                }
                tokio::time::sleep(std::time::Duration::from_secs(15)).await;
            }
            let report = fold_reports(&segments);
            info!(
                scope = %scope,
                stop = ?report.stop,
                items_completed = report.items_completed,
                tokens_per_item = ?report.tokens_per_completed_item,
                resumes,
                "Long-horizon run finished"
            );
            let _ = event_tx.send(Event::TaskRunCompleted {
                scope: scope.clone(),
                scope_id: scope_id.clone(),
                report: serde_json::to_value(&report).unwrap_or(Value::Null),
            });
            manager.runs.write().await.remove(&key);
            manager.reports.write().await.insert(key, (report, resumes));
        });
        Ok(())
    }

    /// Request cancellation. Returns false when no run is active.
    pub async fn cancel(&self, scope: &str, scope_id: Option<&str>) -> bool {
        let key = Self::scope_key(scope, scope_id);
        let runs = self.runs.read().await;
        runs.get(&key).is_some_and(|run| {
            run.cancel.cancel();
            true
        })
    }

    /// Current status for a scope.
    pub async fn status(&self, scope: &str, scope_id: Option<&str>) -> RunStatus {
        let key = Self::scope_key(scope, scope_id);
        let runs = self.runs.read().await;
        // `runs` is held first and throughout: a finishing run removes itself
        // from `runs` before recording its report, so while this guard is held
        // the report for this key cannot change and needs no guard of its own.
        let (last_report, resumes) = self
            .reports
            .read()
            .await
            .get(&key)
            .map_or((None, 0), |(report, resumes)| (Some(report.clone()), *resumes));
        runs.get(&key).map_or_else(
            || RunStatus {
                running: false,
                goal: None,
                started_at: None,
                last_report: last_report.clone(),
                resumes,
            },
            |run| RunStatus {
                running: true,
                goal: Some(run.goal.clone()),
                started_at: Some(run.started_at.to_rfc3339()),
                last_report: last_report.clone(),
                resumes,
            },
        )
    }
}

#[cfg(test)]
mod acceptance_canonicalization_tests {
    use super::{canonical_acceptance, seed_plan};
    use nanna_agent::harness::AcceptanceCheck;
    use nanna_agent::{Plan, PlanOrigin, parse_plan};
    use nanna_storage::Storage;
    use serde_json::json;
    use std::sync::Arc;

    /// REGRESSION: an acceptance check handed over as a JSON *string* — the
    /// dialect models actually emit — must reach the harness as a real check.
    ///
    /// Tolerating the string only in the READER admitted it at the planner and
    /// then had `create` reject it (`value.get("kind")` is None for a string),
    /// so the task was silently skipped; if every planned item was stringified
    /// the whole turn died with "no planned task could be created". The write
    /// boundary canonicalizes, so the store only ever holds the object.
    #[tokio::test]
    async fn a_stringified_acceptance_survives_planner_seed_store_and_harness_read() {
        let raw = r#"[{"title":"make it build",
                       "acceptance":"{\"kind\":\"command\",\"command\":\"cargo build\"}"}]"#;
        let tasks = parse_plan(raw).expect("the plan parses");
        assert_eq!(tasks.len(), 1);

        // 1. Planner admission holds the canonical OBJECT, not the raw string.
        let admitted = tasks[0]
            .acceptance
            .as_ref()
            .expect("a stringified check must still be admitted");
        assert!(
            admitted.is_object(),
            "the planner must admit the canonical object: {admitted}"
        );

        // 2. Seeding creates the task instead of dropping it.
        let storage = Arc::new(Storage::in_memory().await.expect("storage"));
        let plan = Plan {
            tasks,
            origin: PlanOrigin::Model,
        };
        let ids = seed_plan(&storage, "session", Some("sess-plan"), &plan, false)
            .await
            .expect("a stringified acceptance must not sink the plan");
        assert_eq!(ids.len(), 1, "the planned task was created");

        // 3. The STORED row is an object.
        let stored = storage.tasks().get(ids[0]).await.expect("stored task");
        let acceptance = stored.acceptance.expect("acceptance persisted");
        assert!(
            acceptance.is_object(),
            "the store must hold an object: {acceptance}"
        );
        assert_eq!(acceptance["kind"], json!("command"));

        // 4. The harness reads it back as a real check.
        assert_eq!(
            AcceptanceCheck::from_json(&acceptance).expect("harness read"),
            AcceptanceCheck::Command {
                command: "cargo build".to_string(),
                timeout_secs: None,
            }
        );
    }

    /// The `tasks.add`/`tasks.update` script services canonicalize through the
    /// same entry point, so a stringified check written by the todo tool is
    /// stored identically to one written by the planner.
    #[tokio::test]
    async fn the_task_services_canonicalize_the_same_way() {
        let canonical = canonical_acceptance(&json!({
            "acceptance": "{kind: 'regex', pattern: '0 failed', command: 'cargo test'}"
        }))
        .expect("the JS-object-literal flavor is repaired")
        .expect("an acceptance was present");
        assert!(canonical.is_object(), "{canonical}");
        assert_eq!(canonical["kind"], json!("regex"));

        // And the store accepts exactly what was canonicalized.
        let storage = Arc::new(Storage::in_memory().await.expect("storage"));
        let mut new = nanna_storage::NewTask {
            parent_id: None,
            scope: "global".to_string(),
            scope_id: None,
            project: None,
            title: "canonical".to_string(),
            description: None,
            priority: 2,
            labels: Vec::new(),
            tool_scope: Vec::new(),
            due_at: None,
            deadline_at: None,
            recurrence: None,
            depends_on: Vec::new(),
            acceptance: Some(canonical),
            assignee: None,
            sort_order: 0,
        };
        let created = storage.tasks().create(new.clone()).await.expect("create");
        assert!(created.acceptance.expect("stored").is_object());

        // The raw string form goes in too — canonicalized at the boundary.
        new.title = "raw string".to_string();
        new.acceptance = Some(json!("{\"kind\":\"file_exists\",\"path\":\"out.txt\"}"));
        let created = storage.tasks().create(new).await.expect("create");
        let stored = created.acceptance.expect("stored");
        assert_eq!(stored, json!({"kind": "file_exists", "path": "out.txt"}));
    }
}

#[cfg(test)]
mod session_scope_tests {
    use super::{build_task_services, TurnBaselines};
    use nanna_storage::Storage;
    use serde_json::json;
    use std::sync::Arc;
    use tokio::sync::RwLock;

    // -----------------------------------------------------------------
    // Session scope, against the params the todo tool ACTUALLY emits
    //
    // `todo/tool.ts` calls `Nanna.sessionId()` and forwards the result on
    // every scope-carrying call. That produces exactly two shapes, and both
    // are exercised below:
    //   bound run    -> {"scope":"session","session_id":"<id>", ...}
    //   session-less -> `Nanna.sessionId()` is JS null, so `base` carries
    //                   `"session_id": null` and `compact()` (which drops
    //                   null) omits the key entirely on add/update.
    // -----------------------------------------------------------------

    fn services_over(storage: &Arc<Storage>) -> nanna_scripting::NannaBridge {
        let services = build_task_services(
            storage.clone(),
            Arc::new(RwLock::new(None)),
            Arc::new(TurnBaselines::new()),
        );
        nanna_scripting::NannaBridge::new(nanna_scripting::ToolPermissions::default())
            .with_services(services)
    }

    /// The shape a run WITH a bound session emits: scope and `session_id` both
    /// present, straight from `Nanna.sessionId()`. It lands in that session.
    #[tokio::test]
    async fn the_todo_tools_bound_session_params_land_in_that_session() {
        let storage = Arc::new(Storage::in_memory().await.expect("storage"));
        let bridge = services_over(&storage);

        let added = bridge
            .call_service(
                "tasks.add",
                json!({
                    "scope": "session",
                    "session_id": "scheduled-heartbeat-1",
                    "title": "write the parser"
                }),
            )
            .await
            .expect("the tool's own params shape must be accepted");
        assert_eq!(added["task"]["scope"], json!("session"));

        let listed = storage
            .tasks()
            .list("session", Some("scheduled-heartbeat-1"), true)
            .await
            .expect("list");
        assert_eq!(listed.len(), 1, "the task landed in the calling session");
        assert_eq!(listed[0].title, "write the parser");
    }

    /// The shape a SESSION-LESS run emits on `next`/`list`: `session_id` is
    /// present but null, because `Nanna.sessionId()` returned null and `base`
    /// is not compacted. It must fail with a message that names every way out
    /// — restating "requires `session_id`" is precisely what the model could not
    /// act on across 35 logged failures.
    #[tokio::test]
    async fn a_null_session_id_is_told_how_to_supply_a_scope() {
        let storage = Arc::new(Storage::in_memory().await.expect("storage"));
        let bridge = services_over(&storage);

        let err = bridge
            .call_service("tasks.next", json!({"scope": "session", "session_id": null}))
            .await
            .expect_err("no session anywhere: the caller must be told");
        assert!(err.contains("session_id"), "{err}");
        assert!(err.contains("workspace"), "names the id-free scopes: {err}");
        assert!(err.contains("global"), "{err}");
    }

    /// The same session-less run on `add`, where `compact()` drops the null
    /// and the key is absent altogether. Same teaching error.
    #[tokio::test]
    async fn an_absent_session_id_is_told_how_to_supply_a_scope() {
        let storage = Arc::new(Storage::in_memory().await.expect("storage"));
        let bridge = services_over(&storage);

        let err = bridge
            .call_service("tasks.add", json!({"scope": "session", "title": "x"}))
            .await
            .expect_err("compact() drops the null id, so the key is missing");
        assert!(err.contains("session_id"), "{err}");
        assert!(err.contains("workspace"), "{err}");
        assert!(err.contains("global"), "{err}");
    }

    /// The id-free escapes the error names have to actually work, or the
    /// message is teaching a dead end.
    #[tokio::test]
    async fn the_scopes_the_error_names_are_usable_without_an_id() {
        let storage = Arc::new(Storage::in_memory().await.expect("storage"));
        let bridge = services_over(&storage);

        bridge
            .call_service("tasks.add", json!({"scope": "global", "title": "global work"}))
            .await
            .expect("global scope needs no id");
        let listed = storage.tasks().list("global", None, true).await.expect("list");
        assert_eq!(listed.len(), 1);
    }
}

#[cfg(test)]
mod wedged_runner_tests {
    use super::wedged_runner_error;

    /// REGRESSION (2026-07-27): the repetition watch started aborting wedged
    /// streams, but the retry ladder only reset the runner when the message
    /// said "empty completion" — so the new abort never triggered a reset and
    /// three retries hit the same wedge. 17 aborts ended the run.
    #[test]
    fn both_wedge_shapes_trigger_a_reset() {
        assert!(wedged_runner_error(
            "empty completion (no text, no tool calls, ~0 tokens) from provider"
        ));
        assert!(wedged_runner_error(
            "Ollama emitted the same token 20x in a row (\"0\") — a wedged runner, not a generation."
        ));
    }

    /// REGRESSION (2026-08-08, ministral-3:8b endurance): five clusters of
    /// done=false stream aborts ended all three run segments without one
    /// mid-step reset — the classifier did not know the message, while the
    /// segment-boundary resets healed the runner instantly each time. The
    /// tool-call parse 500s that preceded every cluster are the same
    /// degraded-runner evidence, one aborted generation at a time.
    #[test]
    fn abort_and_parse_wedge_shapes_trigger_a_reset() {
        assert!(wedged_runner_error(
            "step error: LLM error: API error: 502 - Ollama aborted generation mid-response (done=false)"
        ));
        assert!(wedged_runner_error(
            "API error: 500 - {\"error\":\"invalid character '\\t' in string literal\"}"
        ));
        assert!(wedged_runner_error(
            "API error: 500 - {\"error\":\"invalid character '$' in string escape code\"}"
        ));
        assert!(wedged_runner_error(
            "API error: 500 - {\"error\":\"invalid character '.' after object key:value pair\"}"
        ));
        // Ordinary provider errors stay out of the wedge path: a 404, a
        // refused connection, or a plain timeout heals by retry/backoff,
        // not by unloading the model.
        assert!(!wedged_runner_error("API error: 404 - model not found"));
        assert!(!wedged_runner_error("connection refused"));
        assert!(!wedged_runner_error("request timed out after 120s"));
    }

    /// The dead-stream message as it actually arrives: EOF at 0 bytes and 0
    /// NDJSON lines, nothing left in the residual buffer.
    const DEAD_STREAM: &str = "step error: LLM error: API error: 502 - Ollama stream ended \
         without completion (no done=true) after 0 bytes / 0 NDJSON lines — nothing was left \
         unparsed (the stream simply stopped). No NDJSON line was ever parsed.";

    /// REGRESSION (2026-08-15 leg): 116 of 160 in-window step retries died on
    /// a stream that produced NOTHING, and the classifier stayed silent — so
    /// the ladder re-asked a dead runner three times per step. The reset
    /// armed only when the sibling "aborted generation mid-response" shape
    /// finally arrived at 11:48:43, after 9 blind retries, and healed it at
    /// once. A stream that never emitted a line is the runner's state, not
    /// the request's luck.
    #[test]
    fn a_zero_byte_dead_stream_triggers_a_reset() {
        assert!(wedged_runner_error(DEAD_STREAM));
    }

    /// The counter-case that keeps the reset honest. A stream that DID
    /// produce output before dying is ambiguous with a generation-slot
    /// contention cancel (`agent_service` drops the loser's stream), and
    /// unloading the model there punishes a runner that was working fine.
    /// Both bytes>0 variants of the same error must stay out.
    #[test]
    fn a_stream_that_produced_bytes_never_resets() {
        assert!(!wedged_runner_error(
            "step error: LLM error: API error: 502 - Ollama stream ended without completion \
             (no done=true) after 2780 bytes / 14 NDJSON lines — nothing was left unparsed \
             (the stream simply stopped). Last line before the cut: {\"message\":{\"content\":\"x\"}}"
        ));
        assert!(!wedged_runner_error(
            "step error: LLM error: API error: 502 - Ollama stream ended without completion \
             (no done=true) after 4096 bytes / 9 NDJSON lines — 412 unparsed bytes remained, \
             ending: {\"message\":{\"content\":\"the"
        ));
    }

    /// The dead stream carries no fingerprint, so it rides the existing
    /// repeat rule exactly as "empty completion" does: the first retry stays
    /// free, the reset arms from the second.
    #[test]
    fn the_dead_stream_rides_the_existing_repeat_ladder() {
        assert_eq!(wedge_reset_due(1, DEAD_STREAM, None, None), None);
        assert_eq!(
            wedge_reset_due(2, DEAD_STREAM, None, None),
            Some(WedgeReset::Repeated)
        );
    }

    use super::{wedge_reset_due, WedgeFingerprint, WedgeReset};

    /// The wedge message as it actually arrives, so the tests below exercise
    /// the same classification path the ladder does.
    fn wedge_msg(bytes: usize) -> String {
        format!(
            "step error: LLM error: API error: 502 - Ollama emitted the same token 20x in a \
             row (\"0\") — a wedged runner, not a generation. Aborted after {bytes} bytes so \
             the step can be retried instead of waiting out the loop."
        )
    }

    fn wedge(bytes: usize) -> WedgeFingerprint {
        WedgeFingerprint { token: "0".to_string(), bytes }
    }

    /// The pair the change exists for: the retry re-entered the SAME wedge,
    /// so the reset happens on the first retry instead of costing another
    /// generation to re-learn it.
    #[test]
    fn an_identical_wedge_is_confirmed_and_reset_immediately() {
        assert_eq!(
            wedge_reset_due(1, &wedge_msg(2780), Some(&wedge(2780)), Some(&wedge(2780))),
            Some(WedgeReset::Confirmed),
            "equal abort lengths on the same token must not wait for attempt 2"
        );
    }

    /// And the counter-case: a wedge that gave up somewhere else is a
    /// different fault, so the free first retry stands.
    #[test]
    fn a_differently_sized_wedge_is_not_confirmed() {
        assert_eq!(
            wedge_reset_due(1, &wedge_msg(2780), Some(&wedge(2780)), Some(&wedge(400))),
            None,
            "unrelated abort lengths must leave the first retry free"
        );
        // A different stuck token is a different wedge even at the same length.
        let other_token = WedgeFingerprint { token: "\n".to_string(), bytes: 2780 };
        assert_eq!(
            wedge_reset_due(1, &wedge_msg(2780), Some(&wedge(2780)), Some(&other_token)),
            None,
            "the stuck token is the fingerprint — a different one is a different fault"
        );
    }

    /// REGRESSION (2026-07-28): the byte counts do NOT repeat exactly.
    ///
    /// Twelve aborts in one day, every one stuck on "0" and every one landing
    /// in 2777-2780 bytes, but consecutive attempts of a single step went
    /// 2780→2779, 2779→2778, 2780→2779, 2780→2780 and 2777→2779 — sampling
    /// leaves the prefix a byte or two different before it collapses. An
    /// equality test would have recognised one of these five and left the
    /// other four to pay a full generation each, which is the entire cost the
    /// confirmation exists to avoid.
    #[test]
    fn the_observed_near_miss_pairs_are_all_the_same_wedge() {
        for (prev, cur) in [(2780, 2779), (2779, 2778), (2780, 2779), (2780, 2780), (2777, 2779)] {
            assert_eq!(
                wedge_reset_due(1, &wedge_msg(cur), Some(&wedge(cur)), Some(&wedge(prev))),
                Some(WedgeReset::Confirmed),
                "live pair {prev}→{cur} is the same wedge re-entered"
            );
        }
    }

    /// The original ladder still covers everything the fingerprint cannot:
    /// a first wedge with nothing to compare against, and "empty completion",
    /// which carries no fingerprint at all.
    #[test]
    fn the_repeat_ladder_still_backs_the_fingerprint() {
        // First wedge of a run: no predecessor, so wait as before.
        assert_eq!(wedge_reset_due(1, &wedge_msg(2780), Some(&wedge(2780)), None), None);
        assert_eq!(
            wedge_reset_due(2, &wedge_msg(2780), Some(&wedge(2780)), None),
            Some(WedgeReset::Repeated)
        );
        // The other wedge shape has no fingerprint and is judged on repeats.
        let empty = "empty completion (no text, no tool calls, ~0 tokens) from provider";
        assert_eq!(wedge_reset_due(1, empty, None, None), None);
        assert_eq!(wedge_reset_due(2, empty, None, None), Some(WedgeReset::Repeated));
    }

    /// A reset is an unload/reload. It must never fire for a fault that is
    /// not the runner's state, however the fingerprints happen to line up.
    #[test]
    fn a_non_wedge_failure_never_resets() {
        assert_eq!(
            wedge_reset_due(3, "Stream error: connection reset", Some(&wedge(2780)), Some(&wedge(2780))),
            None
        );
    }

    #[test]
    fn ordinary_transient_faults_do_not_reset_the_runner() {
        // Resetting unloads/reloads the model — far too expensive for a
        // network blip, and it would add seconds to every recoverable drop.
        for msg in [
            "error sending request for url (http://127.0.0.1:11434/api/chat)",
            "Stream error: connection reset",
            "API error: 500 - internal",
            "step error: LLM error: API error: 429 - rate limited",
        ] {
            assert!(!wedged_runner_error(msg), "must not reset for: {msg}");
        }
    }
}

#[cfg(test)]
mod retry_note_tests {
    use super::{
        carried_side_effect_budget_bytes, carried_side_effect_note, control_char_transport_note,
        ollama_generation_parse_error, transient_fault_kind, transient_retry_note,
    };
    use crate::liveness::ToolMarkSnapshot;

    /// The marks the ledger hands over, newest first.
    fn marks(count: usize) -> Vec<ToolMarkSnapshot> {
        (0..count)
            .map(|i| ToolMarkSnapshot {
                name: if i % 2 == 0 { "write_file" } else { "exec" }.to_string(),
                at: "2026-08-20T10:00:00.000Z".to_string(),
                secs_ago: i as u64,
            })
            .collect()
    }

    /// The three Go-JSON parse sites Ollama reports when the model emits a raw
    /// control character inside a tool-call string argument, each wrapped in
    /// the transport prefixes the message actually arrives with.
    #[test]
    fn each_parse_site_yields_the_offending_character() {
        assert_eq!(
            ollama_generation_parse_error(
                "step error: LLM error: API error: 500 - {\"error\":\"invalid character \
                 '\\t' in string literal\"}"
            )
            .as_deref(),
            Some("\\t")
        );
        assert_eq!(
            ollama_generation_parse_error(
                "API error: 500 - {\"error\":\"invalid character '$' in string escape code\"}"
            )
            .as_deref(),
            Some("$")
        );
        assert_eq!(
            ollama_generation_parse_error(
                "API error: 500 - {\"error\":\"invalid character '.' after object key:value \
                 pair\"}"
            )
            .as_deref(),
            Some(".")
        );
    }

    /// The correction is aimed at the MODEL's encoding, so it must never fire
    /// for the dead-stream / aborted-generation classes, which are the
    /// runner's state and heal by reset. The last case is the one that needs
    /// the explicit exclusion: a stream-end body can quote the model's own
    /// text back in its tail.
    #[test]
    fn the_reset_owned_classes_never_arm_a_correction() {
        for msg in [
            "step error: LLM error: API error: 502 - Ollama stream ended without completion \
             (no done=true) after 0 bytes / 0 NDJSON lines — nothing was left unparsed (the \
             stream simply stopped). No NDJSON line was ever parsed.",
            "step error: LLM error: API error: 502 - Ollama aborted generation mid-response \
             (done=false)",
            "empty completion (no text, no tool calls, ~0 tokens) from provider",
            "step error: LLM error: API error: 502 - Ollama stream ended without completion \
             (no done=true) after 900 bytes / 3 NDJSON lines — nothing was left unparsed (the \
             stream simply stopped). Last line before the cut: invalid character '\\t' in \
             string literal",
        ] {
            assert!(
                ollama_generation_parse_error(msg).is_none(),
                "must not arm an encoding correction for: {msg}"
            );
        }
    }

    /// A body that names no character, or names one at no parse site, is a
    /// different fault — the correction would be advice about nothing.
    #[test]
    fn unrelated_errors_arm_nothing() {
        assert!(ollama_generation_parse_error("API error: 500 - internal").is_none());
        assert!(ollama_generation_parse_error("API error: 404 - model not found").is_none());
        assert!(ollama_generation_parse_error(
            "API error: 500 - {\"error\":\"invalid character 'x' looking for beginning of value\"}"
        )
        .is_none());
    }

    /// The correction has to be actionable: name what was rejected, the
    /// character the provider named, and both legal routes.
    #[test]
    fn the_correction_names_the_character_and_both_routes() {
        let note = control_char_transport_note("\\t");
        assert!(note.contains("REJECTED"));
        assert!(
            note.contains("(\\t)"),
            "the character the provider named must appear, not just the generic advice: {note}"
        );
        assert!(note.contains("ESCAPE"));
        assert!(note.contains("printf"));
        assert!(note.contains("heredoc"));
    }

    /// The re-anchor's whole job: say that the interrupted attempt's tool
    /// effects are real and on disk, and forbid the rewrite-from-scratch that
    /// a fresh context invites.
    #[test]
    fn the_re_anchor_states_the_effects_and_forbids_the_restart() {
        let note = transient_retry_note(2, "the request timed out");
        assert!(note.contains("attempt 2"));
        assert!(note.contains("the request timed out"));
        assert!(note.contains("DID execute"));
        assert!(note.contains("on disk"));
        assert!(note.contains("do not restart it"));
        assert!(note.contains("re-read"));
    }

    /// Every fault label comes from a classifier the ladder already consults,
    /// so the sentence the model reads and the action the ladder takes cannot
    /// disagree.
    #[test]
    fn fault_kinds_come_from_the_ladders_own_classifiers() {
        assert_eq!(
            transient_fault_kind("CUDA error: an illegal memory access was encountered"),
            "the model ran out of GPU memory"
        );
        assert_eq!(
            transient_fault_kind(
                "API error: 500 - {\"error\":\"invalid character '\\t' in string literal\"}"
            ),
            "the provider could not parse the tool call"
        );
        assert_eq!(
            transient_fault_kind(
                "API error: 502 - Ollama stream ended without completion (no done=true) after \
                 0 bytes / 0 NDJSON lines — nothing was left unparsed (the stream simply \
                 stopped). No NDJSON line was ever parsed."
            ),
            "the provider's stream died before any output"
        );
        assert_eq!(
            transient_fault_kind("API error: 502 - Ollama aborted generation mid-response"),
            "the provider's runner wedged mid-generation"
        );
        assert_eq!(
            transient_fault_kind("stream watchdog: no event for 240s"),
            "the stream stalled and the watchdog cut it"
        );
        assert_eq!(transient_fault_kind("request timed out"), "the request timed out");
        assert_eq!(
            transient_fault_kind("error sending request for url (http://127.0.0.1:11434)"),
            "a provider transport fault"
        );
    }

    /// The carried list points the re-anchor at what the lost attempt did —
    /// every call, newest first — and at where the full record still lives.
    /// A general warning is not actionable; "you already wrote three files,
    /// and their outputs are in memory" is.
    #[test]
    fn the_carried_list_names_the_calls_and_where_their_outputs_are() {
        let note = carried_side_effect_note(&marks(3), carried_side_effect_budget_bytes(27_904), true);
        assert!(note.contains("3 side-effecting tool calls"), "{note}");
        assert!(note.contains("- write_file (0s ago)"), "{note}");
        assert!(note.contains("- exec (1s ago)"), "{note}");
        assert!(note.contains("- write_file (2s ago)"), "{note}");
        assert!(note.contains("on disk"), "{note}");
        assert!(
            note.contains("readable out of"),
            "the pointer to the stored record is the half that makes re-reading \
             cheaper than rewriting: {note}"
        );
        // Only the tool the request actually ships may be named.
        assert!(note.contains("discover_tools"), "{note}");
        assert!(!note.contains("recall("), "{note}");
        assert!(!note.contains("read_file("), "{note}");

        // Singular reads as singular — a model that is told "1 calls" is
        // being told the note was generated, not observed.
        let one = carried_side_effect_note(&marks(1), carried_side_effect_budget_bytes(27_904), true);
        assert!(one.contains("1 side-effecting tool call before"), "{one}");

        assert_eq!(carried_side_effect_note(&[], 2_000, true), "");

        // A run with no memory service never stored those outputs, so the
        // note must not send the model looking for them — it keeps the half
        // that is still true (the artifact on disk).
        let no_store =
            carried_side_effect_note(&marks(3), carried_side_effect_budget_bytes(27_904), false);
        assert!(no_store.contains("- write_file (0s ago)"), "{no_store}");
        assert!(!no_store.contains("discover_tools"), "{no_store}");
        assert!(!no_store.contains("your memory"), "{no_store}");
        assert!(no_store.contains("read the artifact back off disk"), "{no_store}");
    }

    /// The bound is the window's share, not the attempt's call count. The
    /// aborted attempt that motivated this made 104 side-effecting calls;
    /// rendered in full that is ~13% of a small model's input budget, so the
    /// re-anchor would displace the work it exists to protect.
    #[test]
    fn the_carried_list_is_bounded_by_the_window_not_by_the_call_count() {
        let small_window = 27_904; // the universal floor's hard input limit
        let note = carried_side_effect_note(&marks(104), carried_side_effect_budget_bytes(small_window), true);

        // Every call is still ACCOUNTED for, in the count and in the tail
        // line — bounded is not the same as silently truncated.
        assert!(note.contains("104 side-effecting tool calls"), "{note}");
        assert!(note.contains("earlier call"), "{note}");
        assert!(
            note.matches("\n- ").count() < 104,
            "the whole list must not be rendered: {note}"
        );

        // The measured share: at ~4 chars per token, the note is a couple of
        // percent of what this model can read, against the 13% observed.
        let input_budget_chars = small_window * 4;
        assert!(
            note.len() * 100 / input_budget_chars <= 3,
            "note is {} bytes of a {input_budget_chars}-char input budget",
            note.len()
        );

        // And the share is a share: a wider window buys more of the list, a
        // demoted one buys less. Same input, different budgets.
        let wide = carried_side_effect_note(&marks(104), carried_side_effect_budget_bytes(small_window * 8), true);
        let demoted = carried_side_effect_note(&marks(104), carried_side_effect_budget_bytes(2_304), true);
        assert!(wide.matches("\n- ").count() > note.matches("\n- ").count());
        assert!(demoted.matches("\n- ").count() < note.matches("\n- ").count());
        // Even at the smallest window the announcement survives: the model
        // must never be left thinking the attempt did nothing.
        assert!(demoted.contains("104 side-effecting tool calls"), "{demoted}");
        assert!(demoted.contains("discover_tools"), "{demoted}");
    }

    /// The budget is the harness's own one-screenful carry-forward, expressed
    /// as a share of the window it was sized against — so it tracks the live
    /// window instead of going stale beside it.
    #[test]
    fn the_budget_is_the_step_tails_share_of_the_live_window() {
        let floor = nanna_llm::unknown_model_info("", "").hard_input_limit();
        assert_eq!(
            carried_side_effect_budget_bytes(floor),
            nanna_agent::harness::STEP_RESULT_TAIL_MAX_BYTES,
            "at the window the step tail was sized against, the budget IS the step tail"
        );
        assert_eq!(
            carried_side_effect_budget_bytes(floor * 2),
            nanna_agent::harness::STEP_RESULT_TAIL_MAX_BYTES * 2,
            "the share is constant, so the budget scales with the window"
        );
        assert!(carried_side_effect_budget_bytes(floor / 4) < carried_side_effect_budget_bytes(floor));
        // A degenerate window must not panic or hand out an unbounded budget.
        assert_eq!(carried_side_effect_budget_bytes(0), 0);
    }
}

#[cfg(test)]
mod core_tool_gating_tests {
    use super::CORE_TOOLS;

    /// The contract, in three directions.
    ///
    /// A plan must never restrict capability (enforcing its guess cost us two
    /// runs: 7/42 when it scoped to a nonexistent "file", and 141-exec/0-write
    /// when it scoped to "exec" and the model shell-heredoc'd around every
    /// write guard).
    ///
    /// Exactly ONE tool ships, because schemas are context.
    ///
    /// And the prompt must not name tools it does not ship. `stable_prefix`
    /// used to order a `todo(action='note', …)` call every step while `todo`'s
    /// schema was never sent; the model obeyed blind and malformed acceptance
    /// checks went from 3.1% to 53-66%. The fix was to stop the prompt naming
    /// tools — not to widen this list. If this list grows again, check whether
    /// a prompt is making promises the request does not keep.
    #[test]
    fn only_discovery_ships_by_default() {
        assert_eq!(CORE_TOOLS, &["discover_tools"]);
        assert!(
            !CORE_TOOLS.contains(&"exec"),
            "exec must be discovered, not shipped — schemas are context"
        );
        assert!(
            !CORE_TOOLS.contains(&"write_file"),
            "write_file must be discovered, not shipped"
        );
        assert!(
            CORE_TOOLS.contains(&"discover_tools"),
            "without discovery a step cannot reach anything at all"
        );
        assert!(
            !CORE_TOOLS.contains(&"recall"),
            "even memory search is discovered — one tool ships, and it is the one \
             that reaches the others"
        );
        assert_eq!(
            CORE_TOOLS.len(),
            1,
            "the list is one tool. Every past widening started as a good local \
             reason and ended as the prompt and the request disagreeing about \
             what the model can call"
        );
    }
}

#[cfg(test)]
mod merge_active_tools_tests {
    use super::{merge_active_tools, CORE_TOOLS};
    use std::collections::HashSet;

    fn discovered(names: &[&str]) -> HashSet<String> {
        names.iter().map(|n| (*n).to_string()).collect()
    }

    /// THE compatibility assertion: with no user selection the merge is
    /// byte-identical to the pre-feature computation (core first, then
    /// discovered) — default tool behavior cannot drift.
    #[test]
    fn empty_selection_is_unchanged() {
        let disc = discovered(&["exec", "read_file"]);
        let legacy: Vec<String> = CORE_TOOLS
            .iter()
            .map(|t| (*t).to_string())
            .chain(disc.iter().cloned())
            .collect();
        assert_eq!(
            merge_active_tools(CORE_TOOLS, &disc, &[]),
            legacy,
            "no picks must mean the exact same active set as before the feature"
        );
    }

    /// User picks are unioned in — appended after the default set.
    #[test]
    fn selection_adds_tools() {
        let disc = discovered(&["exec"]);
        let picks = vec!["web_search".to_string(), "read_pdf".to_string()];
        let active = merge_active_tools(CORE_TOOLS, &disc, &picks);
        assert!(active.contains(&"web_search".to_string()));
        assert!(active.contains(&"read_pdf".to_string()));
        // And the defaults are all still there — picks only ever ADD.
        assert!(active.contains(&"discover_tools".to_string()));
        assert!(active.contains(&"exec".to_string()));
    }

    /// A pick already in the default set is not duplicated — selecting a core
    /// or already-discovered tool is a no-op, never a reorder or removal.
    #[test]
    fn duplicate_picks_are_not_doubled() {
        let disc = discovered(&["exec"]);
        let picks = vec![
            "discover_tools".to_string(), // core
            "exec".to_string(),           // discovered
            "exec".to_string(),           // repeated pick
            "todo".to_string(),           // genuinely new
        ];
        let active = merge_active_tools(CORE_TOOLS, &disc, &picks);
        assert_eq!(
            active.iter().filter(|t| *t == "discover_tools").count(),
            1,
            "picking a core tool must not duplicate it"
        );
        assert_eq!(
            active.iter().filter(|t| *t == "exec").count(),
            1,
            "picking a discovered tool must not duplicate it"
        );
        assert_eq!(
            active.iter().filter(|t| *t == "todo").count(),
            1,
            "a repeated pick lands once"
        );
    }
}

#[cfg(test)]
mod workspace_reference_tests {
    use nanna_agent::harness::{build_step_prompt, TaskStep};
    use nanna_agent::AgentContext;

    /// The chat-turn prompt, assembled the way `try_run_step` assembles it.
    ///
    /// Live 2026-08-02 (gemma4:12b, workspace with a long ROADMAP.md):
    /// "difference between a mutex and a semaphore?" was answered with a
    /// roadmap status report first, and the NEXT turn ignored the user and
    /// created roadmap todos. Three properties prevent that, and this test
    /// pins all three on the assembled prompt:
    /// - the reference block is BOUNDED by the model-window-derived cap and
    ///   the cut announces itself;
    /// - the block leads with the non-instruction header (and keeps it even
    ///   when truncated, since truncation keeps the head);
    /// - the user's message lands AFTER the reference material — it is the
    ///   step prompt, a user-role message every provider places after the
    ///   system prompt, and recency wins for small models.
    #[test]
    fn workspace_reference_is_bounded_framed_and_precedes_the_user_message() {
        let ws = nanna_core::WorkspaceContext {
            readme: Some("A small desktop agent.".into()),
            agents: None,
            contributing: None,
            // A 1000-line roadmap — far past what a 16k window can spare.
            roadmap: Some("- [ ] migrate a file off lucide-vue-next\n".repeat(1_000)),
            git: None,
        };

        let mut ctx = AgentContext::new("t");
        ctx.system_prompt = "BASE".to_string();
        ctx.workspace_context = Some(ws.build_system_prompt_injection());
        // 16k window with a 4k output reserve — the model class this
        // regression was observed on. In production `configure_for_model*`
        // sets this from the live window at run start.
        ctx.hard_limit = 12_288;

        let system = ctx.effective_system_prompt();

        // Framed: the non-instruction header survives into the prompt.
        assert!(
            system.contains("# Project Context (background reference)"),
            "the reference header must be present"
        );
        assert!(system.contains("NOT instructions"));

        // Bounded: the slice is cut at the window-derived cap and says so.
        assert!(
            system.len() <= ctx.system_prompt.len() + ctx.workspace_context_cap_chars() + 256,
            "a 1000-line ROADMAP must not exceed the cap plus the marker \
             (got {} chars, cap {})",
            system.len(),
            ctx.workspace_context_cap_chars()
        );
        assert!(system.contains("[workspace context truncated"));

        // Ordered: the user's message is the step prompt, which follows the
        // system prompt in every provider request — flattened in that order,
        // the goal must appear after the reference block.
        let goal = "difference between a mutex and a semaphore?";
        let step = TaskStep {
            id: 1,
            title: "Answer the question".into(),
            description: None,
            acceptance: None,
            tool_scope: Vec::new(),
            notes_tail: Vec::new(),
        };
        let user_message = build_step_prompt(goal, &step, None, None, "budget: fresh");
        let assembled = format!("{system}\n\n{user_message}");
        let reference_at = assembled
            .find("# Project Context (background reference)")
            .expect("reference block present");
        let goal_at = assembled.rfind(goal).expect("goal present");
        assert!(
            goal_at > reference_at,
            "the user's message must come after the reference material"
        );
    }
}

/// What the `tasks.*` services accept from a model, and what they refuse.
///
/// The dialect is the contract: every param here is JSON a model wrote, so a
/// reader that only admits the canonical shape is a reader that fails on real
/// traffic — and one that drops what it cannot read fails invisibly.
#[cfg(test)]
mod param_dialect_tests {
    use super::*;

    async fn add_task(services: &HashMap<String, ServiceFn>, params: Value) -> Value {
        services
            .get("tasks.add")
            .expect("tasks.add service")(params)
        .await
        .expect("add succeeds")
    }

    /// REGRESSION (live 2026-08-04, claude-opus-5 driving `todo`): the model
    /// sent `{"action":"update","id":"2109"}` three times running and was told
    /// "id is required" each time. The id was right there — only quoted.
    #[test]
    fn a_stringified_id_is_an_id() {
        assert_eq!(req_i64(&json!({ "id": "2109" }), "id"), Ok(2109));
        assert_eq!(req_i64(&json!({ "id": 2109 }), "id"), Ok(2109));
        // Boa hands whole numbers back as floats once arithmetic touches them.
        assert_eq!(req_i64(&json!({ "id": 2109.0 }), "id"), Ok(2109));
        assert_eq!(req_i64(&json!({ "id": "2109.0" }), "id"), Ok(2109));
        assert_eq!(req_i64(&json!({ "id": " 2109 " }), "id"), Ok(2109));
        assert_eq!(req_i64(&json!({ "id": "-7" }), "id"), Ok(-7));
    }

    /// Zero is a value, not an absence — the JS `!input.id` guard this fix
    /// also tightened made exactly that mistake.
    #[test]
    fn an_id_of_zero_is_supplied() {
        assert_eq!(req_i64(&json!({ "id": 0 }), "id"), Ok(0));
        assert_eq!(req_i64(&json!({ "id": "0" }), "id"), Ok(0));
        assert_eq!(opt_i64(&json!({ "id": 0 }), "id"), Ok(Some(0)));
    }

    /// MISSING and UNINTERPRETABLE are different failures and must read
    /// differently: "id is required" said about a supplied id is a false
    /// statement, and a model that believes it loops on the same call.
    #[test]
    fn a_missing_id_and_an_unreadable_id_say_different_things() {
        assert_eq!(req_i64(&json!({}), "id"), Err("id is required".to_string()));
        // The bridge writes `undefined` members as null — still "not supplied".
        assert_eq!(
            req_i64(&json!({ "id": Value::Null }), "id"),
            Err("id is required".to_string())
        );

        let unreadable = req_i64(&json!({ "id": "abc" }), "id").expect_err("not an integer");
        assert!(
            unreadable.contains("id must be an integer"),
            "must name the requirement, got: {unreadable}"
        );
        assert!(
            unreadable.contains("got string \"abc\""),
            "must name what ARRIVED, got: {unreadable}"
        );
        assert!(
            !unreadable.contains("is required"),
            "an id WAS supplied — saying otherwise is what caused the loop"
        );
    }

    /// Lossy or ambiguous readings are refused rather than guessed at: a wrong
    /// guess writes a row nobody asked for.
    #[test]
    fn a_lossy_or_ambiguous_id_is_refused() {
        for bad in [
            json!(1.5),
            json!("1.5"),
            json!(""),
            json!("   "),
            json!(true),
            json!([1]),
            json!({ "id": 1 }),
            // Past 2^53 an f64 cannot tell adjacent integers apart, so
            // "the fraction is zero" stops proving anything.
            json!(9_007_199_254_740_994.0_f64),
        ] {
            assert!(
                as_i64_lenient(&bad).is_none(),
                "{bad} must not be read as an integer"
            );
        }
        // A string, though, is parsed as i64 directly — no f64 round trip, so
        // the full i64 range survives.
        assert_eq!(
            as_i64_lenient(&json!("9007199254740993")),
            Some(9_007_199_254_740_993)
        );
    }

    /// The silent half of the same bug: `depends_on: "[2108]"` hit
    /// `Value::as_array`, returned None, and was dropped with no error at all.
    /// The model believed it had declared a dependency the store never saw.
    #[test]
    fn a_stringified_dependency_list_survives() {
        assert_eq!(
            opt_i64_vec(&json!({ "depends_on": "[2108]" }), "depends_on"),
            Ok(Some(vec![2108]))
        );
        assert_eq!(
            opt_i64_vec(&json!({ "depends_on": ["2108"] }), "depends_on"),
            Ok(Some(vec![2108]))
        );
        assert_eq!(
            opt_i64_vec(&json!({ "depends_on": [2108, "2109"] }), "depends_on"),
            Ok(Some(vec![2108, 2109]))
        );
        assert_eq!(
            opt_i64_vec(&json!({ "depends_on": [] }), "depends_on"),
            Ok(Some(vec![]))
        );
        assert_eq!(opt_i64_vec(&json!({}), "depends_on"), Ok(None));
    }

    /// One unwrap only, mirroring PR #174: a doubly-encoded list is a bug to
    /// surface, not one to peel.
    #[test]
    fn a_dependency_list_is_unwrapped_exactly_once() {
        assert!(
            opt_i64_vec(&json!({ "depends_on": "\"[2108]\"" }), "depends_on").is_err(),
            "a doubly-encoded list must be reported, not peeled twice"
        );
    }

    /// An unreadable list must never resolve to an empty vec: empty is a real,
    /// meaningful value ("nothing blocks this"), so substituting it silently
    /// erases a dependency the caller did state.
    #[test]
    fn an_unreadable_dependency_list_errors_instead_of_emptying() {
        let err = opt_i64_vec(&json!({ "depends_on": "not json" }), "depends_on")
            .expect_err("must not silently empty");
        assert!(err.contains("depends_on must be a list of task ids"), "{err}");
        assert!(err.contains("got string \"not json\""), "{err}");

        let element = opt_i64_vec(&json!({ "depends_on": ["abc"] }), "depends_on")
            .expect_err("an unreadable element must not vanish either");
        assert!(element.contains("depends_on[0]"), "{element}");
        assert!(element.contains("got string \"abc\""), "{element}");
    }

    /// Same rule for the string lists. A scalar element renders losslessly;
    /// a nested array is not a label under any reading.
    #[test]
    fn label_lists_read_the_same_dialects() {
        assert_eq!(
            opt_string_vec(&json!({ "labels": "[\"a\",\"b\"]" }), "labels"),
            Ok(Some(vec!["a".to_string(), "b".to_string()]))
        );
        assert_eq!(
            opt_string_vec(&json!({ "labels": [1, true] }), "labels"),
            Ok(Some(vec!["1".to_string(), "true".to_string()]))
        );
        assert!(opt_string_vec(&json!({ "labels": "urgent" }), "labels").is_err());
        assert!(opt_string_vec(&json!({ "labels": [["a"]] }), "labels").is_err());
    }

    /// Booleans arrive spelled out too. Only the two literals are read —
    /// `1`/`"yes"` would be a guess about intent, and a default quietly
    /// substituted for an unreadable flag answers a question nobody asked.
    #[test]
    fn booleans_read_the_literal_spelled_out_and_nothing_else() {
        assert_eq!(opt_bool(&json!({ "b": "true" }), "b"), Ok(Some(true)));
        assert_eq!(opt_bool(&json!({ "b": " False " }), "b"), Ok(Some(false)));
        assert_eq!(opt_bool(&json!({ "b": false }), "b"), Ok(Some(false)));
        assert_eq!(opt_bool(&json!({}), "b"), Ok(None));
        assert!(opt_bool(&json!({ "b": 1 }), "b").is_err());
        assert!(opt_bool(&json!({ "b": "yes" }), "b").is_err());
    }

    /// End to end through the live services: the exact three calls from the
    /// 2026-08-04 transcript now land.
    #[tokio::test]
    async fn the_service_surface_accepts_the_dialect_the_model_sends() {
        let storage = Arc::new(Storage::in_memory().await.expect("storage"));
        let workspace_id = Arc::new(RwLock::new(None));
        let services =
            build_task_services(storage.clone(), workspace_id, Arc::new(TurnBaselines::new()));

        let blocker = add_task(
            &services,
            json!({ "title": "blocker", "scope": "session", "session_id": "s1" }),
        )
        .await;
        let blocker_id = blocker["task"]["id"].as_i64().expect("id");
        let target = add_task(
            &services,
            json!({ "title": "target", "scope": "session", "session_id": "s1" }),
        )
        .await;
        let target_id = target["task"]["id"].as_i64().expect("id");

        // The failing call, verbatim in shape: id and depends_on both quoted.
        let updated = services.get("tasks.update").expect("tasks.update")(json!({
            "id": target_id.to_string(),
            "depends_on": format!("[{blocker_id}]"),
            "title": "Move the four dialect readers",
        }))
        .await
        .expect("a quoted id is an id");
        assert_eq!(updated["task"]["id"], json!(target_id));
        assert_eq!(
            updated["task"]["depends_on"],
            json!([blocker_id]),
            "the dependency must be RECORDED, not silently dropped"
        );
        assert_eq!(updated["task"]["title"], json!("Move the four dialect readers"));

        // A note and a completion against the same quoted id.
        services.get("tasks.note").expect("tasks.note")(
            json!({ "id": target_id.to_string(), "content": "found it" }),
        )
        .await
        .expect("a quoted id notes");
        services.get("tasks.done").expect("tasks.done")(
            json!({ "id": blocker_id.to_string() }),
        )
        .await
        .expect("a quoted id completes");

        // And the remove that closed the transcript.
        let removed = services.get("tasks.remove").expect("tasks.remove")(
            json!({ "id": target_id.to_string() }),
        )
        .await
        .expect("a quoted id removes");
        assert_eq!(removed["removed"], json!(1));

        // A genuinely unreadable id still fails — loudly, and about itself.
        let err = services.get("tasks.update").expect("tasks.update")(
            json!({ "id": "abc" }),
        )
        .await
        .expect_err("garbage is still garbage");
        assert!(err.contains("got string \"abc\""), "{err}");
    }

    /// A dependency the store cannot read stops the write instead of landing a
    /// task whose plan is quietly missing an edge.
    #[tokio::test]
    async fn an_unreadable_dependency_refuses_the_add() {
        let storage = Arc::new(Storage::in_memory().await.expect("storage"));
        let workspace_id = Arc::new(RwLock::new(None));
        let services =
            build_task_services(storage.clone(), workspace_id, Arc::new(TurnBaselines::new()));

        let err = services.get("tasks.add").expect("tasks.add")(json!({
            "title": "depends on something",
            "scope": "session",
            "session_id": "s1",
            "depends_on": "not json",
        }))
        .await
        .expect_err("a dropped dependency must not pass silently");
        assert!(err.contains("depends_on must be a list of task ids"), "{err}");

        let open = storage
            .tasks()
            .list("session", Some("s1"), true)
            .await
            .expect("list");
        assert!(
            open.is_empty(),
            "the refused add must not have written a half-formed task"
        );
    }
}

/// Every chat turn, sub-agent turn and background run builds its agents in
/// [`AgentStepRunner::step_agent`], so this is where the summarizers must be
/// given the router.
#[cfg(test)]
mod summarizer_wiring_tests {
    use super::AgentStepRunner;
    use crate::embedding_reload::test_ollama::{CHAT_MEMORY, ollama_credentials, spawn_token_gated};
    use crate::llm_router::LlmRouter;
    use nanna_llm::AnthropicMessage;
    use std::sync::Arc;

    fn runner(router: Arc<LlmRouter>) -> AgentStepRunner {
        AgentStepRunner {
            discovered_tools: Arc::new(tokio::sync::RwLock::new(std::collections::HashSet::new())),
            user_selected_tools: Vec::new(),
            repeat_ledger: Arc::new(nanna_agent::RepeatLedger::new()),
            router,
            tools: Arc::new(nanna_tools::ToolRegistry::new()),
            agent_config: nanna_agent::AgentConfig {
                model: "ollama/nanna-test-chat:1b".to_string(),
                summarization_priority: vec!["ollama/nanna-test-summarizer:1b".to_string()],
                ..nanna_agent::AgentConfig::default()
            },
            system_prompt: String::new(),
            workspace_root: None,
            workspace_context: None,
            stats: None,
            memory: None,
            workspace_id: None,
            chat_sink: None,
            gpu_fault_count: Arc::new(std::sync::atomic::AtomicU32::new(0)),
            degradations: None,
            attachments: Arc::default(),
            model_chain: None,
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_steps_summarizer_goes_where_chat_goes_and_follows_a_move() {
        let (first_host, first_seen) = spawn_token_gated("bound-token").await;
        let (moved_host, moved_seen) = spawn_token_gated("bound-token").await;
        let router = Arc::new(LlmRouter::new());
        router.rebuild(&ollama_credentials(
            &first_host,
            "bound-token",
        ));

        let mut context = nanna_agent::AgentContext::new("s");
        context.messages.push(AnthropicMessage::user_text(
            "Please keep the build green on every commit; that is the one rule here.",
        ));
        context.messages.push(AnthropicMessage::assistant_text(
            "Understood: every commit keeps the build green.",
        ));
        let Ok(agent) = runner(Arc::clone(&router))
            .step_agent(nanna_agent::harness::StepKind::Execute, context)
        else {
            panic!("Ollama always registers, so the chat model has a provider");
        };

        let memories = agent.extract_memories().await.expect("the summarizer answers");
        assert_eq!(
            memories.iter().map(|m| m.content.as_str()).collect::<Vec<_>>(),
            vec![CHAT_MEMORY]
        );
        let reached = first_seen.lock().expect("record lock").clone();
        assert!(
            reached.iter().any(|r| r.path == "/api/chat"
                && r.authorization.as_deref() == Some("Bearer bound-token")
                && r.model.as_deref() == Some("nanna-test-summarizer:1b")),
            "{reached:?}"
        );

        router.rebuild(&ollama_credentials(
            &moved_host,
            "bound-token",
        ));
        agent.extract_memories().await.expect("the moved server answers");
        let reached = moved_seen.lock().expect("record lock").clone();
        assert!(
            reached.iter().any(|r| r.path == "/api/chat"
                && r.authorization.as_deref() == Some("Bearer bound-token")),
            "{reached:?}"
        );
    }
}

#[cfg(test)]
mod model_fallback_tests {
    use super::{AgentStepRunner, ModelChain, warrants_model_fallback};
    use crate::embedding_reload::test_ollama::spawn_answering;
    use crate::llm_router::LlmRouter;
    use nanna_agent::harness::{StepKind, StepRequest, StepRunner};
    use std::sync::Arc;

    fn runner(router: Arc<LlmRouter>, model: &str, chain: Option<Arc<ModelChain>>) -> AgentStepRunner {
        AgentStepRunner {
            discovered_tools: Arc::new(tokio::sync::RwLock::new(std::collections::HashSet::new())),
            user_selected_tools: Vec::new(),
            repeat_ledger: Arc::new(nanna_agent::RepeatLedger::new()),
            router,
            tools: Arc::new(nanna_tools::ToolRegistry::new()),
            agent_config: nanna_agent::AgentConfig {
                model: model.to_string(),
                ..nanna_agent::AgentConfig::default()
            },
            system_prompt: String::new(),
            workspace_root: None,
            workspace_context: None,
            stats: None,
            memory: None,
            workspace_id: None,
            chat_sink: None,
            gpu_fault_count: Arc::new(std::sync::atomic::AtomicU32::new(0)),
            degradations: None,
            attachments: Arc::default(),
            model_chain: chain,
        }
    }

    fn request() -> StepRequest {
        StepRequest {
            item_title: "say hello".to_string(),
            item_id: 1,
            step_index: 0,
            step_kind: StepKind::Execute,
            prompt: "Task #1: say hello".to_string(),
            tool_scope: Vec::new(),
            token_budget: None,
            max_iterations: Some(1),
            max_wall_clock: None,
            cancel: None,
        }
    }

    fn models(names: &[&str]) -> Vec<String> {
        names.iter().map(|m| (*m).to_string()).collect()
    }

    #[test]
    fn the_live_failures_fall_back_and_harness_guards_do_not() {
        // The exact string the 2026-09-21 turns died on, three times running.
        assert!(warrants_model_fallback(
            "step error: LLM error: Stream error: error decoding response body"
        ));
        assert!(warrants_model_fallback("step error: LLM error: API error: 502 - bad gateway"));
        assert!(warrants_model_fallback("step error: LLM error: API error: 401 - invalid api key"));
        assert!(warrants_model_fallback("step error: LLM error: API error: 404 - model not found"));
        assert!(warrants_model_fallback("No provider available for model 'claude-opus-5'"));
        assert!(warrants_model_fallback(
            "empty completion (no text, no tool calls, ~0 tokens) from provider"
        ));
        assert!(!warrants_model_fallback("step cancelled by the user"));
        assert!(!warrants_model_fallback(
            "step error: effective context window (8192 tokens) is below the floor"
        ));
    }

    #[test]
    fn a_chain_starts_at_the_first_served_model_and_skips_unserved_ones() {
        let router = LlmRouter::new().with_ollama("http://127.0.0.1:9");
        // No Anthropic credential: the head is unserved, so the run starts
        // on the first model a provider can actually reach.
        let chain = ModelChain::new(
            models(&["claude-opus-5", "ollama/a:1b", "claude-sonnet-5", "ollama/b:1b"]),
            &router,
        );
        assert_eq!(chain.active(), "ollama/a:1b");
        assert_eq!(chain.advance_past("ollama/a:1b", &router).as_deref(), Some("ollama/b:1b"));
        assert_eq!(chain.active(), "ollama/b:1b");
        assert_eq!(chain.advance_past("ollama/b:1b", &router), None, "exhausted");
        assert_eq!(chain.active(), "ollama/b:1b", "an exhausted chain stays put");
    }

    #[test]
    fn two_reports_of_one_failure_advance_the_chain_once() {
        let router = LlmRouter::new().with_ollama("http://127.0.0.1:9");
        let chain = ModelChain::new(models(&["ollama/a:1b", "ollama/b:1b", "ollama/c:1b"]), &router);
        assert_eq!(chain.advance_past("ollama/a:1b", &router).as_deref(), Some("ollama/b:1b"));
        // The planner and a step both saw `a` fail: the second report
        // follows the chain rather than skipping `b`.
        assert_eq!(chain.advance_past("ollama/a:1b", &router).as_deref(), Some("ollama/b:1b"));
        assert_eq!(chain.active(), "ollama/b:1b");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_step_the_head_model_fails_is_answered_by_the_next_model() {
        // `dead:1b` is not served (404), `alive:1b` answers.
        let (host, seen) = spawn_answering(&[("alive:1b", "hello from the fallback")]).await;
        let router = Arc::new(LlmRouter::new().with_ollama(&host));
        let chain = Arc::new(ModelChain::new(models(&["ollama/dead:1b", "ollama/alive:1b"]), &router));
        let step = runner(Arc::clone(&router), "ollama/dead:1b", Some(Arc::clone(&chain)));

        let outcome = step.run_step(request()).await.expect("the fallback model answers");

        assert!(outcome.text.contains("hello from the fallback"), "{}", outcome.text);
        assert_eq!(chain.active(), "ollama/alive:1b", "the run stays on the model that worked");
        let asked: Vec<Option<String>> = seen.lock().expect("seen").iter().map(|r| r.model.clone()).collect();
        assert!(asked.contains(&Some("dead:1b".to_string())), "the head was tried first: {asked:?}");
        assert_eq!(asked.last(), Some(&Some("alive:1b".to_string())), "{asked:?}");

        // The next step starts on the model that worked, not the dead head.
        let before = seen.lock().expect("seen").len();
        step.run_step(request()).await.expect("second step");
        // Chat requests only: a model-info lookup rides along without a
        // `model` field.
        let later: Vec<String> = seen.lock().expect("seen")[before..]
            .iter()
            .filter(|r| r.path == "/api/chat")
            .filter_map(|r| r.model.clone())
            .collect();
        assert!(!later.is_empty() && later.iter().all(|m| m == "alive:1b"), "{later:?}");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn without_a_chain_a_failing_model_still_fails_the_step() {
        // A pinned chat: no substitution, the failure is reported as is.
        let (host, seen) = spawn_answering(&[("alive:1b", "should never be asked")]).await;
        let router = Arc::new(LlmRouter::new().with_ollama(&host));
        let step = runner(router, "ollama/dead:1b", None);

        let err = step.run_step(request()).await.expect_err("the pinned model fails");

        assert!(err.contains("404"), "{err}");
        assert!(
            seen.lock().expect("seen").iter().all(|r| r.model.as_deref() != Some("alive:1b")),
            "no other model was asked"
        );
    }
}
