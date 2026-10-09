//! The board router's decisions and what each one does to the store (P25
//! Stage 2).
//!
//! The Task Management Agent reads a card and answers with exactly one of four
//! decisions — assign, split, clarify, park (P25 decision 4). This module is
//! the half of the router that needs no model: the decision's wire shape, a
//! bounded parser for the model's structured output, and [`apply_decision`],
//! which turns a decision into store writes plus the thread post that makes it
//! visible and reversible. The daemon's `RouterService` (the half that wakes on
//! bus events and asks the router member's model chain) calls into this, so
//! every rule below holds whichever model produced the decision.
//!
//! The rules enforced here, each from a settled P25 decision:
//! - The router takes no work (4): it can never assign a card to a router.
//! - Every decision is a thread post by the router (4), so nothing it does is
//!   silent.
//! - A clarification is a card assigned to the human that the work card
//!   depends on (6), so the work is *blocked* — derived, no new state.
//! - A card with a live run is never reassigned (7); the caller says whether a
//!   run is live, because only the daemon's run registry knows.

use std::fmt::Write as _;

use serde::{Deserialize, Serialize};

use crate::{
    HUMAN_MEMBER_ID, MemberRepository, NewTask, ROUTER_MEMBER_PREFIX, StorageError, TASK_DEPS_MAX,
    TASK_NOTE_MAX_BYTES, TASK_TITLE_MAX_BYTES, Task, TaskNoteKind, TaskPatch, TaskRepository,
};

/// Most sub-tasks one split may create.
///
/// The split is announced in ONE thread post that names every child, so the
/// bound is how many maximal titles (plus a short `#id → member` prefix) fit
/// in one post: `TASK_NOTE_MAX_BYTES / (TASK_TITLE_MAX_BYTES + 32)` = 30.
pub const SPLIT_SUBTASKS_MAX: usize = TASK_NOTE_MAX_BYTES / (TASK_TITLE_MAX_BYTES + 32);

/// Most bytes of model output [`parse_decision`] will look at.
///
/// A decision is a small object; its largest legal form is a maximal split,
/// whose titles and reasons are each bounded by one post. Four posts' worth is
/// room for that with its JSON punctuation and any prose around it.
pub const DECISION_TEXT_BYTES_MAX: usize = 4 * TASK_NOTE_MAX_BYTES;

/// Most labels one `assign` may add to a card.
///
/// Labels are how the router files a card for the members who match it
/// (capability tags, P25 decision 15); a handful is a filing, more is noise
/// the board's filter row would have to render.
pub const ASSIGN_LABELS_MAX: usize = 8;

/// Longest label the router may add, in bytes: the store's own bound. A
/// label is a `#token` in the quick-add and filter language, so it is short
/// and has no whitespace.
pub const LABEL_BYTES_MAX: usize = crate::TASK_LABEL_MAX_BYTES;

/// Label stamped on a clarification card, so the board can filter them.
pub const CLARIFICATION_LABEL: &str = "clarification";

/// Label stamped on a parked card.
pub const PARKED_LABEL: &str = "parked";

/// One sub-task in a [`RouterDecision::Split`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubtaskSpec {
    pub title: String,
    #[serde(default)]
    pub description: Option<String>,
    /// A member id, or `None` to leave it for the router's next pass.
    #[serde(default)]
    pub assignee: Option<String>,
}

/// What the router decided for one card — the structured output its model is
/// asked for, tagged by `decision`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "decision", rename_all = "snake_case")]
pub enum RouterDecision {
    /// Give the card to one member, filling in what the human left blank
    /// (P25 decision 5): labels are added to the card's own, and an
    /// acceptance check is written only when the card has none — the human's
    /// check always stands.
    Assign {
        member: String,
        reason: String,
        #[serde(default)]
        labels: Vec<String>,
        #[serde(default)]
        acceptance: Option<serde_json::Value>,
    },
    /// Break the card into sub-tasks under it.
    Split {
        subtasks: Vec<SubtaskSpec>,
        reason: String,
    },
    /// Ask the human; the card waits on the answer.
    Clarify { question: String, reason: String },
    /// Leave the card where it is, and say why.
    Park { reason: String },
}

impl RouterDecision {
    /// The decision's name, as it appears on the wire.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Assign { .. } => "assign",
            Self::Split { .. } => "split",
            Self::Clarify { .. } => "clarify",
            Self::Park { .. } => "park",
        }
    }

    const fn reason(&self) -> &String {
        match self {
            Self::Assign { reason, .. }
            | Self::Split { reason, .. }
            | Self::Clarify { reason, .. }
            | Self::Park { reason } => reason,
        }
    }
}

/// What [`apply_decision`] did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppliedDecision {
    /// The router's thread post on the card.
    pub note_id: i64,
    /// Cards the decision created (sub-tasks, or the clarification card).
    pub created: Vec<i64>,
    /// The card's assignee after the decision, when the decision set one.
    pub assigned_to: Option<String>,
}

/// Read a decision out of the router model's reply.
///
/// Tolerates the wrappers models put around JSON — a fenced block, a sentence
/// before or after — by taking the outermost `{…}`; everything else is strict.
/// The result is validated, so a decision that parses but could never be
/// applied (a blank reason, an empty split) is refused here, with a message
/// the router can be re-asked with.
///
/// # Errors
/// A description of what is wrong with the reply, fit to send back to the
/// model: too long, no JSON object, not one of the four decisions, or a field
/// that fails validation.
pub fn parse_decision(text: &str) -> Result<RouterDecision, String> {
    if text.len() > DECISION_TEXT_BYTES_MAX {
        return Err(format!(
            "the reply is {} bytes; a decision fits in {DECISION_TEXT_BYTES_MAX}",
            text.len()
        ));
    }
    let (Some(start), Some(end)) = (text.find('{'), text.rfind('}')) else {
        return Err("no JSON object in the reply; answer with one decision object".to_string());
    };
    if end < start {
        return Err("no JSON object in the reply; answer with one decision object".to_string());
    }
    let decision: RouterDecision = serde_json::from_str(&text[start..=end]).map_err(|e| {
        format!(
            "not a decision ({e}); use {{\"decision\": \"assign\"|\"split\"|\"clarify\"|\"park\", …}}"
        )
    })?;
    validate(&decision)?;
    Ok(decision)
}

/// The checks every decision passes before it may touch the store.
fn validate(decision: &RouterDecision) -> Result<(), String> {
    let reason = decision.reason().trim();
    if reason.is_empty() {
        return Err(format!("a {} decision needs a reason", decision.name()));
    }
    if reason.len() > TASK_NOTE_MAX_BYTES / 2 {
        return Err(format!(
            "the reason is {} bytes; keep it under {}",
            reason.len(),
            TASK_NOTE_MAX_BYTES / 2
        ));
    }
    match decision {
        RouterDecision::Assign {
            member,
            labels,
            acceptance,
            ..
        } => {
            if member.trim().is_empty() {
                return Err("an assign decision names no member".to_string());
            }
            if member.starts_with(ROUTER_MEMBER_PREFIX) {
                return Err("the router takes no work; assign a person or an agent".to_string());
            }
            validate_labels(labels)?;
            if let Some(check) = acceptance.as_ref().filter(|c| !c.is_null()) {
                crate::tasks::admit_acceptance(check)
                    .map_err(|e| format!("the acceptance check is unusable: {e}"))?;
            }
        }
        RouterDecision::Split { subtasks, .. } => {
            if subtasks.is_empty() {
                return Err("a split with no sub-tasks; assign or park instead".to_string());
            }
            if subtasks.len() > SPLIT_SUBTASKS_MAX {
                return Err(format!(
                    "{} sub-tasks; one split creates at most {SPLIT_SUBTASKS_MAX}",
                    subtasks.len()
                ));
            }
            for (index, sub) in subtasks.iter().enumerate() {
                let title = sub.title.trim();
                if title.is_empty() || title.len() > TASK_TITLE_MAX_BYTES {
                    return Err(format!(
                        "sub-task {} needs a title of 1..={TASK_TITLE_MAX_BYTES} bytes",
                        index + 1
                    ));
                }
                if sub
                    .assignee
                    .as_deref()
                    .is_some_and(|m| m.starts_with(ROUTER_MEMBER_PREFIX))
                {
                    return Err(format!(
                        "sub-task {} is assigned to the router, which takes no work",
                        index + 1
                    ));
                }
            }
        }
        RouterDecision::Clarify { question, .. } => {
            let question = question.trim();
            if question.is_empty() {
                return Err("a clarify decision asks no question".to_string());
            }
            if question.len() > TASK_NOTE_MAX_BYTES / 2 {
                return Err(format!(
                    "the question is {} bytes; keep it under {}",
                    question.len(),
                    TASK_NOTE_MAX_BYTES / 2
                ));
            }
        }
        RouterDecision::Park { .. } => {}
    }
    Ok(())
}

/// Labels an `assign` adds: at most [`ASSIGN_LABELS_MAX`], each one token of
/// 1..=[`LABEL_BYTES_MAX`] bytes.
fn validate_labels(labels: &[String]) -> Result<(), String> {
    if labels.len() > ASSIGN_LABELS_MAX {
        return Err(format!(
            "{} labels; an assign adds at most {ASSIGN_LABELS_MAX}",
            labels.len()
        ));
    }
    for label in labels {
        let label = label.trim().trim_start_matches('#');
        if label.is_empty() || label.len() > LABEL_BYTES_MAX {
            return Err(format!("a label must be 1..={LABEL_BYTES_MAX} bytes"));
        }
        if label.chars().any(char::is_whitespace) {
            return Err(format!(
                "the label '{label}' has a space; a label is one word"
            ));
        }
    }
    Ok(())
}

/// Carry out `decision` on card `task_id` as the router member `router_id`.
///
/// `run_is_live` is whether a harness run is working the card right now; only
/// the daemon's run registry knows, so the caller says. With a live run the
/// only decision allowed is `park` — the others reassign the card or change
/// what it waits on under the run's feet (P25 decision 7).
///
/// Writes happen in the order that leaves the board readable if one fails:
/// the cards a decision creates first, then the change to this card, then the
/// post announcing it — so a post never names a card that does not exist.
///
/// # Errors
/// [`StorageError::Invalid`] when the decision fails validation, the card is
/// closed, a run is live and the decision is not `park`, the assignee is a
/// router or not a board member, or a store bound is hit (a clarification
/// would take the card past [`TASK_DEPS_MAX`] dependencies); plus whatever
/// the underlying repository calls return.
///
/// # Panics
/// Panics if `router_id` is not a router member id — a programming error in
/// the caller, never model output.
pub async fn apply_decision(
    tasks: &TaskRepository,
    members: &MemberRepository,
    router_id: &str,
    task_id: i64,
    decision: &RouterDecision,
    run_is_live: bool,
) -> Result<AppliedDecision, StorageError> {
    assert!(
        router_id.starts_with(ROUTER_MEMBER_PREFIX),
        "decisions are applied as a router member, not {router_id}"
    );
    validate(decision).map_err(StorageError::Invalid)?;
    let task = tasks.get(task_id).await?;
    if matches!(task.status.as_str(), "done" | "cancelled") {
        return Err(StorageError::Invalid(format!(
            "card #{task_id} is {}; the router routes open cards only",
            task.status
        )));
    }
    // The card is read here, AFTER the router's model call, and a card picked
    // up meanwhile is `in_progress`: the caller's `run_is_live` was judged
    // before that call (which can take minutes), so on its own it let an
    // `assign`/`split` land on a card a run had started working.
    let worked = run_is_live || task.status == "in_progress";
    if worked && !matches!(decision, RouterDecision::Park { .. }) {
        return Err(StorageError::Invalid(format!(
            "card #{task_id} has a live run; the router may only park it (never reassign a card \
             that is being worked)"
        )));
    }

    let applied = match decision {
        RouterDecision::Assign {
            member,
            reason,
            labels,
            acceptance,
        } => {
            let fill = Fill {
                labels,
                acceptance: acceptance.as_ref().filter(|c| !c.is_null()),
            };
            apply_assign(tasks, members, router_id, &task, member, reason, &fill).await?
        }
        RouterDecision::Split { subtasks, reason } => {
            apply_split(tasks, members, router_id, &task, subtasks, reason).await?
        }
        RouterDecision::Clarify { question, reason } => {
            apply_clarify(tasks, router_id, &task, question, reason).await?
        }
        RouterDecision::Park { reason } => apply_park(tasks, router_id, &task, reason).await?,
    };
    debug_assert!(applied.note_id > 0, "every decision is posted");
    Ok(applied)
}

/// What an `assign` fills in besides the assignee.
struct Fill<'a> {
    labels: &'a [String],
    acceptance: Option<&'a serde_json::Value>,
}

/// The card's labels plus the new ones it lacks (trimmed, `#` dropped),
/// or `None` when nothing is new.
fn labels_after(card: &[String], added: &[String]) -> Option<Vec<String>> {
    let mut labels = card.to_vec();
    for label in added {
        let label = label.trim().trim_start_matches('#');
        if !labels.iter().any(|l| l == label) {
            labels.push(label.to_string());
        }
    }
    debug_assert!(labels.len() <= card.len() + added.len(), "only additions");
    (labels.len() > card.len()).then_some(labels)
}

async fn apply_assign(
    tasks: &TaskRepository,
    members: &MemberRepository,
    router_id: &str,
    task: &Task,
    member_id: &str,
    reason: &str,
    fill: &Fill<'_>,
) -> Result<AppliedDecision, StorageError> {
    let member = members.get(member_id).await.map_err(|e| match e {
        StorageError::NotFound(_) => StorageError::Invalid(format!(
            "'{member_id}' is not a board member; assign someone on the board"
        )),
        other => other,
    })?;
    let labels = labels_after(&task.labels, fill.labels);
    // Decision 5: the router writes the check only when the human left it
    // blank; one the human set always stands.
    let acceptance = fill.acceptance.filter(|_| task.acceptance.is_none());
    let reassigned = task.assignee.as_deref() != Some(member_id);
    if reassigned || labels.is_some() || acceptance.is_some() {
        // One patch, so the store admits all of it or none of it.
        let patch = TaskPatch {
            assignee: reassigned.then(|| Some(member_id.to_string())),
            labels: labels.clone(),
            acceptance: acceptance.map(|check| Some(check.clone())),
            ..TaskPatch::default()
        };
        tasks.update(task.id, patch, Some(router_id)).await?;
    }
    let mut text = format!(
        "Assigned to {} ({member_id}). {}",
        member.name,
        reason.trim()
    );
    if let Some(labels) = &labels {
        let added = &labels[task.labels.len()..];
        let _ = write!(text, "\nLabelled: {}", added.join(", "));
    }
    if acceptance.is_some() {
        let written = tasks.get(task.id).await?.acceptance;
        let shown = written.map_or_else(String::new, |check| check.to_string());
        let _ = write!(
            text,
            "\nDone when: {}",
            cut_to(&shown, TASK_NOTE_MAX_BYTES / 4)
        );
    }
    let note = post(tasks, router_id, task.id, TaskNoteKind::Comment, &text).await?;
    Ok(AppliedDecision {
        note_id: note,
        created: Vec::new(),
        assigned_to: Some(member_id.to_string()),
    })
}

async fn apply_split(
    tasks: &TaskRepository,
    members: &MemberRepository,
    router_id: &str,
    task: &Task,
    subtasks: &[SubtaskSpec],
    reason: &str,
) -> Result<AppliedDecision, StorageError> {
    // Every named assignee is checked BEFORE the first child is written. The
    // children are created one at a time, so a typo in sub-task 2 used to
    // leave sub-task 1 on the board — unannounced (the post comes last) and,
    // with an agent assignee, already starting a run — under a card that was
    // never routed.
    for (index, sub) in subtasks.iter().enumerate() {
        let Some(assignee) = sub.assignee.as_deref() else {
            continue;
        };
        members.get(assignee).await.map_err(|e| match e {
            StorageError::NotFound(_) => StorageError::Invalid(format!(
                "sub-task {} names '{assignee}', who is not a board member; nothing was split",
                index + 1
            )),
            other => other,
        })?;
    }
    let mut created = Vec::with_capacity(subtasks.len());
    let mut lines = String::new();
    for sub in subtasks {
        let child = tasks
            .create(NewTask {
                parent_id: Some(task.id),
                scope: task.scope.clone(),
                scope_id: task.scope_id.clone(),
                project: task.project.clone(),
                title: sub.title.trim().to_string(),
                description: sub.description.clone(),
                priority: task.priority,
                assignee: sub.assignee.clone(),
                created_by: Some(router_id.to_string()),
                ..NewTask::default()
            })
            .await?;
        created.push(child.id);
        let owner = child.assignee.as_deref().unwrap_or("unassigned");
        let _ = write!(lines, "\n- #{} → {owner}: {}", child.id, child.title);
    }
    let text = format!(
        "Split into {} sub-task(s). {}{lines}",
        created.len(),
        reason.trim()
    );
    let note = post(tasks, router_id, task.id, TaskNoteKind::Comment, &text).await?;
    debug_assert_eq!(created.len(), subtasks.len(), "one card per sub-task");
    Ok(AppliedDecision {
        note_id: note,
        created,
        assigned_to: None,
    })
}

async fn apply_clarify(
    tasks: &TaskRepository,
    router_id: &str,
    task: &Task,
    question: &str,
    reason: &str,
) -> Result<AppliedDecision, StorageError> {
    ask_on_card(tasks, router_id, task, question, reason).await
}

/// Put `question` to the human as a clarification card that `task` waits on
/// (P25 decision 6).
///
/// Asked by member `asker_id` — the router deciding, or the
/// member working the card. The card is derived-`blocked` until the human
/// completes the clarification; the asker's `question` post on the card says
/// which card to answer.
///
/// # Errors
/// [`StorageError::Invalid`] when `task` already waits on
/// [`TASK_DEPS_MAX`] cards, else the store failure creating the
/// clarification, updating `task` or posting.
pub async fn ask_on_card(
    tasks: &TaskRepository,
    asker_id: &str,
    task: &Task,
    question: &str,
    reason: &str,
) -> Result<AppliedDecision, StorageError> {
    let router_id = asker_id;
    if task.depends_on.len() >= TASK_DEPS_MAX {
        return Err(StorageError::Invalid(format!(
            "card #{} already waits on {TASK_DEPS_MAX} cards; it cannot wait on a clarification too",
            task.id
        )));
    }
    let question = question.trim();
    let clarification = tasks
        .create(NewTask {
            scope: task.scope.clone(),
            scope_id: task.scope_id.clone(),
            project: task.project.clone(),
            title: clarification_title(task.id, question),
            description: Some(question.to_string()),
            priority: task.priority,
            labels: vec![CLARIFICATION_LABEL.to_string()],
            assignee: Some(HUMAN_MEMBER_ID.to_string()),
            created_by: Some(router_id.to_string()),
            ..NewTask::default()
        })
        .await?;
    let mut depends_on = task.depends_on.clone();
    depends_on.push(clarification.id);
    let patch = TaskPatch {
        depends_on: Some(depends_on),
        ..TaskPatch::default()
    };
    tasks.update(task.id, patch, Some(router_id)).await?;
    let text = format!(
        "Waiting on #{} for you: {question}\n{}",
        clarification.id,
        reason.trim()
    );
    let note = post(tasks, router_id, task.id, TaskNoteKind::Question, &text).await?;
    Ok(AppliedDecision {
        note_id: note,
        created: vec![clarification.id],
        assigned_to: None,
    })
}

async fn apply_park(
    tasks: &TaskRepository,
    router_id: &str,
    task: &Task,
    reason: &str,
) -> Result<AppliedDecision, StorageError> {
    if !task.labels.iter().any(|l| l == PARKED_LABEL) {
        let mut labels = task.labels.clone();
        labels.push(PARKED_LABEL.to_string());
        let patch = TaskPatch {
            labels: Some(labels),
            ..TaskPatch::default()
        };
        tasks.update(task.id, patch, Some(router_id)).await?;
    }
    let text = format!("Parked. {}", reason.trim());
    let note = post(tasks, router_id, task.id, TaskNoteKind::Comment, &text).await?;
    Ok(AppliedDecision {
        note_id: note,
        created: Vec::new(),
        assigned_to: None,
    })
}

/// The router's thread post, cut to one post's bound on a char boundary.
async fn post(
    tasks: &TaskRepository,
    router_id: &str,
    task_id: i64,
    kind: TaskNoteKind,
    text: &str,
) -> Result<i64, StorageError> {
    let text = cut_to(text, TASK_NOTE_MAX_BYTES);
    let note = tasks
        .post(task_id, Some(router_id), Some(router_id), kind, text)
        .await?;
    Ok(note.id)
}

/// The clarification card's title: its first line, within the title bound.
fn clarification_title(task_id: i64, question: &str) -> String {
    let prefix = format!("Clarify #{task_id}: ");
    let first_line = question.lines().next().unwrap_or(question);
    let room = TASK_TITLE_MAX_BYTES.saturating_sub(prefix.len());
    let title = format!("{prefix}{}", cut_to(first_line, room));
    debug_assert!(
        title.len() <= TASK_TITLE_MAX_BYTES,
        "within the title bound"
    );
    title
}

/// `text` cut to at most `max_bytes`, on a char boundary.
fn cut_to(text: &str, max_bytes: usize) -> &str {
    if text.len() <= max_bytes {
        return text;
    }
    let mut end = max_bytes;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MemberKind, MemberOwner, MemberStatus, NewMember, Storage};

    const ROUTER: &str = "router:global";

    async fn board() -> (Storage, TaskRepository, MemberRepository) {
        let storage = Storage::in_memory().await.unwrap();
        let tasks = storage.tasks();
        let members = storage.members();
        members
            .create(NewMember {
                id: "agent:coder".to_string(),
                name: "Coder".to_string(),
                avatar: None,
                kind: MemberKind::Agent,
                owner_kind: MemberOwner::Workspace,
                owner_id: None,
                status: MemberStatus::Idle,
                profile: serde_json::json!({}),
            })
            .await
            .unwrap();
        (storage, tasks, members)
    }

    async fn card(tasks: &TaskRepository, title: &str) -> Task {
        tasks
            .create(NewTask {
                scope: "global".to_string(),
                title: title.to_string(),
                priority: 2,
                ..NewTask::default()
            })
            .await
            .unwrap()
    }

    #[test]
    fn a_decision_is_read_out_of_the_wrappers_models_put_around_it() {
        let reply = "Here is my decision:\n```json\n{\"decision\": \"assign\", \"member\": \"agent:coder\", \"reason\": \"it writes Rust\"}\n```";
        assert_eq!(
            parse_decision(reply).unwrap(),
            RouterDecision::Assign {
                member: "agent:coder".to_string(),
                reason: "it writes Rust".to_string(),
                labels: Vec::new(),
                acceptance: None,
            }
        );
        let park = r#"{"decision":"park","reason":"waiting on the release"}"#;
        assert_eq!(parse_decision(park).unwrap().name(), "park");
    }

    #[test]
    fn a_decision_that_could_never_apply_is_refused_with_a_reason_to_resend() {
        let cases = [
            ("no json here", "no JSON object"),
            (r#"{"decision":"delegate","reason":"x"}"#, "not a decision"),
            (r#"{"decision":"park","reason":"  "}"#, "needs a reason"),
            (
                r#"{"decision":"assign","member":"router:global","reason":"x"}"#,
                "takes no work",
            ),
            (
                r#"{"decision":"split","subtasks":[],"reason":"x"}"#,
                "no sub-tasks",
            ),
            (
                r#"{"decision":"split","subtasks":[{"title":" "}],"reason":"x"}"#,
                "needs a title",
            ),
            (
                r#"{"decision":"clarify","question":"","reason":"x"}"#,
                "asks no question",
            ),
        ];
        for (reply, want) in cases {
            let err = parse_decision(reply).unwrap_err();
            assert!(err.contains(want), "{reply}: {err}");
        }
        let too_many = serde_json::json!({
            "decision": "split",
            "reason": "x",
            "subtasks": (0..=SPLIT_SUBTASKS_MAX).map(|i| serde_json::json!({"title": format!("t{i}")})).collect::<Vec<_>>(),
        });
        assert!(
            parse_decision(&too_many.to_string())
                .unwrap_err()
                .contains("at most")
        );
        assert!(parse_decision(&"x".repeat(DECISION_TEXT_BYTES_MAX + 1)).is_err());
    }

    #[test]
    fn the_split_bound_is_what_one_post_can_announce() {
        assert_eq!(SPLIT_SUBTASKS_MAX, 30);
        let line = 32 + TASK_TITLE_MAX_BYTES;
        assert!(SPLIT_SUBTASKS_MAX * line <= TASK_NOTE_MAX_BYTES);
    }

    #[tokio::test]
    async fn assign_sets_the_assignee_and_says_so_on_the_thread() {
        let (_s, tasks, members) = board().await;
        let task = card(&tasks, "port the parser").await;
        let decision = RouterDecision::Assign {
            member: "agent:coder".to_string(),
            reason: "it has the Rust tools".to_string(),
            labels: Vec::new(),
            acceptance: None,
        };
        let applied = apply_decision(&tasks, &members, ROUTER, task.id, &decision, false)
            .await
            .unwrap();
        assert_eq!(applied.assigned_to.as_deref(), Some("agent:coder"));
        assert_eq!(
            tasks.get(task.id).await.unwrap().assignee.as_deref(),
            Some("agent:coder")
        );
        let notes = tasks.notes(task.id, 10).await.unwrap();
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].id, applied.note_id);
        assert_eq!(notes[0].author_member_id.as_deref(), Some(ROUTER));
        assert!(
            notes[0].content.contains("Assigned to Coder"),
            "{}",
            notes[0].content
        );

        let ghost = RouterDecision::Assign {
            member: "agent:ghost".to_string(),
            reason: "x".to_string(),
            labels: Vec::new(),
            acceptance: None,
        };
        let err = apply_decision(&tasks, &members, ROUTER, task.id, &ghost, false)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("not a board member"), "{err}");
    }

    /// Decision 5: the human fills in what they like and the router completes
    /// the rest — it adds labels and writes the check a blank card lacks, and
    /// never replaces a check the human wrote.
    #[tokio::test]
    async fn assign_fills_what_the_human_left_blank_and_keeps_what_they_set() {
        let (_s, tasks, members) = board().await;
        let check = serde_json::json!({"kind": "file_exists", "path": "out/parser.rs"});
        let fill = |labels: &[&str]| RouterDecision::Assign {
            member: "agent:coder".to_string(),
            reason: "it writes Rust".to_string(),
            labels: labels.iter().map(|l| (*l).to_string()).collect(),
            acceptance: Some(check.clone()),
        };

        let blank = card(&tasks, "port the parser").await;
        apply_decision(
            &tasks,
            &members,
            ROUTER,
            blank.id,
            &fill(&["#rust", "parser"]),
            false,
        )
        .await
        .unwrap();
        let filled = tasks.get(blank.id).await.unwrap();
        assert_eq!(
            filled.labels,
            vec!["rust".to_string(), "parser".to_string()]
        );
        assert_eq!(
            filled.acceptance.as_ref().and_then(|c| c.get("path")),
            Some(&serde_json::json!("out/parser.rs")),
            "the blank check is written"
        );
        let post = &tasks.notes(blank.id, 1).await.unwrap()[0].content;
        assert!(post.contains("Labelled: rust, parser"), "{post}");
        assert!(post.contains("Done when:"), "{post}");

        let mut set = NewTask {
            scope: "global".to_string(),
            title: "the human's own check".to_string(),
            priority: 3,
            labels: vec!["rust".to_string()],
            ..NewTask::default()
        };
        set.acceptance = Some(serde_json::json!({"kind": "command", "command": "cargo test"}));
        let set = tasks.create(set).await.unwrap();
        apply_decision(&tasks, &members, ROUTER, set.id, &fill(&["rust"]), false)
            .await
            .unwrap();
        let kept = tasks.get(set.id).await.unwrap();
        assert_eq!(kept.labels, vec!["rust".to_string()], "no duplicate label");
        assert_eq!(
            kept.acceptance.as_ref().and_then(|c| c.get("command")),
            Some(&serde_json::json!("cargo test")),
            "the human's check stands"
        );
        let post = &tasks.notes(set.id, 1).await.unwrap()[0].content;
        assert!(
            !post.contains("Labelled") && !post.contains("Done when"),
            "{post}"
        );
    }

    #[test]
    fn an_assign_whose_fill_could_not_be_stored_is_sent_back() {
        let too_many = serde_json::json!({
            "decision": "assign", "member": "agent:coder", "reason": "x",
            "labels": ["a", "b", "c", "d", "e", "f", "g", "h", "i"],
        });
        let cases = [
            (too_many.to_string(), "at most"),
            (
                r#"{"decision":"assign","member":"agent:coder","reason":"x","labels":["two words"]}"#
                    .to_string(),
                "one word",
            ),
            (
                r#"{"decision":"assign","member":"agent:coder","reason":"x","acceptance":{"kind":"vibes"}}"#
                    .to_string(),
                "acceptance check is unusable",
            ),
        ];
        for (reply, expected) in cases {
            let err = parse_decision(&reply).unwrap_err();
            assert!(err.contains(expected), "{reply}: {err}");
        }
        // Absent and null fill are both "nothing to fill".
        let bare = r#"{"decision":"assign","member":"agent:coder","reason":"x","acceptance":null}"#;
        assert!(parse_decision(bare).is_ok());
    }

    #[tokio::test]
    async fn split_creates_children_under_the_card_and_names_them() {
        let (_s, tasks, members) = board().await;
        let task = card(&tasks, "ship the board").await;
        let decision = RouterDecision::Split {
            reason: "two independent halves".to_string(),
            subtasks: vec![
                SubtaskSpec {
                    title: "store".to_string(),
                    description: None,
                    assignee: Some("agent:coder".to_string()),
                },
                SubtaskSpec {
                    title: "client".to_string(),
                    description: Some("the kanban view".to_string()),
                    assignee: None,
                },
            ],
        };
        let applied = apply_decision(&tasks, &members, ROUTER, task.id, &decision, false)
            .await
            .unwrap();
        assert_eq!(applied.created.len(), 2);
        for id in &applied.created {
            let child = tasks.get(*id).await.unwrap();
            assert_eq!(child.parent_id, Some(task.id));
            assert_eq!(child.scope, task.scope);
            assert_eq!(
                child.priority, task.priority,
                "children inherit the priority"
            );
        }
        let post = &tasks.notes(task.id, 10).await.unwrap()[0].content;
        assert!(
            post.contains(&format!("#{} → agent:coder: store", applied.created[0])),
            "{post}"
        );
        assert!(
            post.contains(&format!("#{} → unassigned: client", applied.created[1])),
            "{post}"
        );
    }

    #[tokio::test]
    async fn a_card_picked_up_while_the_router_thought_can_only_be_parked() {
        let (_s, tasks, members) = board().await;
        let task = card(&tasks, "ship the board").await;
        // A run started on it after the router's wake judged it free.
        tasks
            .update(
                task.id,
                TaskPatch {
                    status: Some("in_progress".to_string()),
                    ..TaskPatch::default()
                },
                None,
            )
            .await
            .unwrap();
        let assign = RouterDecision::Assign {
            member: "agent:coder".to_string(),
            reason: "it writes Rust".to_string(),
            labels: Vec::new(),
            acceptance: None,
        };
        let refused = apply_decision(&tasks, &members, ROUTER, task.id, &assign, false)
            .await
            .unwrap_err();
        assert!(refused.to_string().contains("may only park"), "{refused}");
        assert_eq!(tasks.get(task.id).await.unwrap().assignee, None);
        let park = RouterDecision::Park {
            reason: "it is being worked".to_string(),
        };
        apply_decision(&tasks, &members, ROUTER, task.id, &park, false)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn a_split_naming_a_stranger_creates_nothing() {
        let (_s, tasks, members) = board().await;
        let task = card(&tasks, "ship the board").await;
        let decision = RouterDecision::Split {
            reason: "two halves".to_string(),
            subtasks: vec![
                SubtaskSpec {
                    title: "store".to_string(),
                    description: None,
                    assignee: Some("agent:coder".to_string()),
                },
                SubtaskSpec {
                    title: "client".to_string(),
                    description: None,
                    assignee: Some("agent:typo".to_string()),
                },
            ],
        };
        let refused = apply_decision(&tasks, &members, ROUTER, task.id, &decision, false)
            .await
            .unwrap_err();
        assert!(
            refused
                .to_string()
                .contains("sub-task 2 names 'agent:typo'"),
            "{refused}"
        );
        let board = tasks.list("global", None, true).await.unwrap();
        assert_eq!(
            board.iter().map(|t| t.id).collect::<Vec<_>>(),
            [task.id],
            "no child of a refused split is left behind"
        );
        assert!(tasks.notes(task.id, 10).await.unwrap().is_empty());
    }

    /// Decision 6: the clarification is a card for the human, and the work
    /// card depends on it — so the work is blocked, derived, until answered.
    #[tokio::test]
    async fn clarify_blocks_the_work_on_a_card_for_the_human() {
        let (_s, tasks, members) = board().await;
        let task = card(&tasks, "email the client").await;
        let decision = RouterDecision::Clarify {
            question: "Which client — Acme or Globex?\nBoth have open threads.".to_string(),
            reason: "the card names no client".to_string(),
        };
        let applied = apply_decision(&tasks, &members, ROUTER, task.id, &decision, false)
            .await
            .unwrap();
        let clarification = tasks.get(applied.created[0]).await.unwrap();
        assert_eq!(clarification.assignee.as_deref(), Some(HUMAN_MEMBER_ID));
        assert_eq!(clarification.labels, vec![CLARIFICATION_LABEL.to_string()]);
        assert_eq!(
            clarification.title,
            format!("Clarify #{}: Which client — Acme or Globex?", task.id)
        );
        let work = tasks.get(task.id).await.unwrap();
        assert_eq!(work.depends_on, vec![clarification.id]);
        assert!(work.blocked, "the work waits on the answer");
        let notes = tasks.notes(task.id, 10).await.unwrap();
        assert_eq!(notes[0].kind, TaskNoteKind::Question);
    }

    /// Decision 7: a card being worked is never reassigned — parking is the
    /// only decision that may land while its run is live.
    #[tokio::test]
    async fn a_card_with_a_live_run_can_only_be_parked() {
        let (_s, tasks, members) = board().await;
        let task = card(&tasks, "long job").await;
        let assign = RouterDecision::Assign {
            member: "agent:coder".to_string(),
            reason: "x".to_string(),
            labels: Vec::new(),
            acceptance: None,
        };
        let err = apply_decision(&tasks, &members, ROUTER, task.id, &assign, true)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("live run"), "{err}");
        assert_eq!(
            tasks.get(task.id).await.unwrap().assignee,
            None,
            "nothing changed"
        );
        assert!(
            tasks.notes(task.id, 10).await.unwrap().is_empty(),
            "nothing posted"
        );

        let park = RouterDecision::Park {
            reason: "the run owns it".to_string(),
        };
        apply_decision(&tasks, &members, ROUTER, task.id, &park, true)
            .await
            .unwrap();
        apply_decision(&tasks, &members, ROUTER, task.id, &park, true)
            .await
            .unwrap();
        let parked = tasks.get(task.id).await.unwrap();
        assert_eq!(
            parked.labels,
            vec![PARKED_LABEL.to_string()],
            "labelled once"
        );
        assert_eq!(
            tasks.notes(task.id, 10).await.unwrap().len(),
            2,
            "each decision is posted"
        );
    }

    #[tokio::test]
    async fn a_closed_card_is_not_routed() {
        let (_s, tasks, members) = board().await;
        let task = card(&tasks, "gone").await;
        tasks
            .update(
                task.id,
                TaskPatch {
                    status: Some("cancelled".to_string()),
                    ..TaskPatch::default()
                },
                None,
            )
            .await
            .unwrap();
        let park = RouterDecision::Park {
            reason: "x".to_string(),
        };
        let err = apply_decision(&tasks, &members, ROUTER, task.id, &park, false)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("open cards only"), "{err}");
    }
}
