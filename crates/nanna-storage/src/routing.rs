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
    /// Give the card to one member.
    Assign { member: String, reason: String },
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
        RouterDecision::Assign { member, .. } => {
            if member.trim().is_empty() {
                return Err("an assign decision names no member".to_string());
            }
            if member.starts_with(ROUTER_MEMBER_PREFIX) {
                return Err("the router takes no work; assign a person or an agent".to_string());
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
    if run_is_live && !matches!(decision, RouterDecision::Park { .. }) {
        return Err(StorageError::Invalid(format!(
            "card #{task_id} has a live run; the router may only park it (never reassign a card \
             that is being worked)"
        )));
    }

    let applied = match decision {
        RouterDecision::Assign { member, reason } => {
            apply_assign(tasks, members, router_id, &task, member, reason).await?
        }
        RouterDecision::Split { subtasks, reason } => {
            apply_split(tasks, router_id, &task, subtasks, reason).await?
        }
        RouterDecision::Clarify { question, reason } => {
            apply_clarify(tasks, router_id, &task, question, reason).await?
        }
        RouterDecision::Park { reason } => apply_park(tasks, router_id, &task, reason).await?,
    };
    debug_assert!(applied.note_id > 0, "every decision is posted");
    Ok(applied)
}

async fn apply_assign(
    tasks: &TaskRepository,
    members: &MemberRepository,
    router_id: &str,
    task: &Task,
    member_id: &str,
    reason: &str,
) -> Result<AppliedDecision, StorageError> {
    let member = members.get(member_id).await.map_err(|e| match e {
        StorageError::NotFound(_) => StorageError::Invalid(format!(
            "'{member_id}' is not a board member; assign someone on the board"
        )),
        other => other,
    })?;
    if task.assignee.as_deref() != Some(member_id) {
        let patch = TaskPatch {
            assignee: Some(Some(member_id.to_string())),
            ..TaskPatch::default()
        };
        tasks.update(task.id, patch, Some(router_id)).await?;
    }
    let text = format!(
        "Assigned to {} ({member_id}). {}",
        member.name,
        reason.trim()
    );
    let note = post(tasks, router_id, task.id, TaskNoteKind::Comment, &text).await?;
    Ok(AppliedDecision {
        note_id: note,
        created: Vec::new(),
        assigned_to: Some(member_id.to_string()),
    })
}

async fn apply_split(
    tasks: &TaskRepository,
    router_id: &str,
    task: &Task,
    subtasks: &[SubtaskSpec],
    reason: &str,
) -> Result<AppliedDecision, StorageError> {
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
                reason: "it writes Rust".to_string()
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
        };
        let err = apply_decision(&tasks, &members, ROUTER, task.id, &ghost, false)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("not a board member"), "{err}");
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
