//! The board router's model half (P25 Stage 2): read a card, ask the router
//! member's model for one decision, apply it.
//!
//! [`nanna_storage::routing`] owns what a decision *is* and what it does to the
//! store; this module owns how one is *obtained*: the context the router reads
//! (the card, its recent thread, the board's roster with profiles, and each
//! member's verdict history — P25 decision 4), the prompt built from it within
//! a fixed byte budget, the model call, and one re-ask when the reply is not a
//! usable decision. The model is injected as a completion function, the same
//! shape `dream_summarizer::summarize_with_failover` returns, so the router
//! walks a model list with failover exactly as every other one-shot caller
//! does, and tests drive it with a scripted reply.
//!
//! Woken by [`crate::board_router_trigger`] — only for cards the board
//! client creates, while the chat harness lives: its own workspace-scoped
//! todos are board cards too, and routing them (a `clarify` blocks the
//! harness's work on a human card) would fight it.

use std::fmt::Write as _;
use std::future::Future;
use std::pin::Pin;

use nanna_storage::routing::{
    AppliedDecision, CLARIFICATION_LABEL, apply_decision, parse_decision,
};
use nanna_storage::{
    Member, ROUTER_MEMBER_PREFIX, Storage, Task, TaskNote, TaskRepository, VerdictTally,
    router_member_id,
};

/// A one-shot completion: prompt in, reply text out, or why not.
pub type RouterComplete =
    dyn Fn(String) -> Pin<Box<dyn Future<Output = Result<String, String>> + Send>> + Send + Sync;

/// Bytes the router prompt may occupy.
///
/// The router runs on the same model list as any agent, so its prompt must fit
/// the smallest window an agent runs in — `nanna_llm::DEFAULT_MIN_VIABLE_NUM_CTX`
/// (4 608 tokens) — with room left for the reply. Half of it, at ~4 bytes a
/// token, is 9 216 bytes.
pub const ROUTER_PROMPT_BYTES_MAX: usize = 9_216;

/// Thread posts the router reads, newest last.
pub const ROUTER_THREAD_POSTS_MAX: usize = 8;

/// Bytes of one member's profile shown to the router.
pub const ROUTER_PROFILE_PREVIEW_BYTES: usize = 400;

/// Bytes of the card description (and of its labels line) shown to the router.
pub const ROUTER_TEXT_PREVIEW_BYTES: usize = 800;

/// Bytes of one thread post shown to the router.
///
/// Sized so the fixed sections cannot crowd out the roster: the instructions
/// (~1.6 KB), a maximal title (500 B), the description and labels (800 B
/// each), the acceptance check (240 B), eight posts at this size (3.2 KB) and
/// two answered clarifications (~0.7 KB) total about 7.9 KB, leaving the
/// roster at least 1.3 KB of [`ROUTER_PROMPT_BYTES_MAX`] — room for the human
/// and a few agents even in the worst case (pinned by
/// `the_roster_keeps_room_in_the_worst_case_prompt`).
pub const ROUTER_POST_PREVIEW_BYTES: usize = 400;

/// Bytes of the card's acceptance check shown to the router.
pub const ROUTER_ACCEPTANCE_PREVIEW_BYTES: usize = 240;

/// How many recent verdicts the router's outcome history is drawn from.
pub const ROUTER_VERDICT_WINDOW: usize = 500;

/// Answered clarifications the router reads, newest first.
///
/// The question is already on the card's thread (the router's own `clarify`
/// post), so only the answer is shown. Two answers at
/// [`ROUTER_ANSWER_PREVIEW_BYTES`] cost ~700 B of [`ROUTER_PROMPT_BYTES_MAX`],
/// which still leaves the roster ~1.1 KB in the worst case the post-preview
/// bound was sized for.
pub const ROUTER_ANSWERS_MAX: usize = 2;

/// Bytes of one clarification answer shown to the router.
pub const ROUTER_ANSWER_PREVIEW_BYTES: usize = 320;

/// A clarification card the human has completed, and what they said on it.
#[derive(Debug, Clone)]
pub struct ClarificationAnswer {
    pub card_id: i64,
    /// The human's newest post on the clarification card, or `None` when they
    /// completed it without posting — which the router is told, not left to
    /// infer from silence.
    pub answer: Option<String>,
}

/// What the router reads before it decides.
#[derive(Debug, Clone)]
pub struct RouterContext {
    pub card: Task,
    /// Oldest first, at most [`ROUTER_THREAD_POSTS_MAX`].
    pub thread: Vec<TaskNote>,
    /// Members who can take the card — never a router.
    pub roster: Vec<Member>,
    pub verdicts: Vec<VerdictTally>,
    /// Completed clarifications the card depends on, at most
    /// [`ROUTER_ANSWERS_MAX`].
    pub answers: Vec<ClarificationAnswer>,
}

/// The router member for `card`'s board, or `None` for a card that is not on
/// a board (session scope — the chat harness's own list).
#[must_use]
pub fn router_for(card: &Task) -> Option<String> {
    match card.scope.as_str() {
        "global" => Some(router_member_id(None)),
        "workspace" => card
            .scope_id
            .as_deref()
            .map(|id| router_member_id(Some(id))),
        _ => None,
    }
}

/// Gather what the router reads about card `task_id`.
///
/// # Errors
/// A description of the store failure.
pub async fn gather_context(storage: &Storage, task_id: i64) -> Result<RouterContext, String> {
    let tasks = storage.tasks();
    let card = tasks.get(task_id).await.map_err(|e| e.to_string())?;
    let workspace = (card.scope == "workspace")
        .then(|| card.scope_id.clone())
        .flatten();
    let mut thread = tasks
        .notes(
            task_id,
            i64::try_from(ROUTER_THREAD_POSTS_MAX).unwrap_or(i64::MAX),
        )
        .await
        .map_err(|e| e.to_string())?;
    // `notes` answers newest first; the router reads a thread top to bottom.
    thread.sort_by_key(|n| n.id);
    let roster = storage
        .members()
        .list_for_workspace(workspace.as_deref())
        .await
        .map_err(|e| e.to_string())?
        .into_iter()
        .filter(|m| !m.id.starts_with(ROUTER_MEMBER_PREFIX))
        .collect();
    let verdicts = tasks
        .verdict_rollup(ROUTER_VERDICT_WINDOW)
        .await
        .map_err(|e| e.to_string())?;
    let answers = answered_clarifications(&tasks, &card).await?;
    debug_assert!(thread.len() <= ROUTER_THREAD_POSTS_MAX, "thread bounded");
    debug_assert!(answers.len() <= ROUTER_ANSWERS_MAX, "answers bounded");
    Ok(RouterContext {
        card,
        thread,
        roster,
        verdicts,
        answers,
    })
}

/// The completed clarifications `card` depends on, newest first, at most
/// [`ROUTER_ANSWERS_MAX`]. A dependency that no longer exists is skipped: it
/// answers nothing.
///
/// # Errors
/// A description of the store failure.
pub async fn answered_clarifications(
    tasks: &TaskRepository,
    card: &Task,
) -> Result<Vec<ClarificationAnswer>, String> {
    let mut answers = Vec::new();
    // Newest dependency last: a later clarification supersedes an earlier one.
    for dep_id in card.depends_on.iter().rev() {
        if answers.len() == ROUTER_ANSWERS_MAX {
            break;
        }
        let Ok(dep) = tasks.get(*dep_id).await else {
            continue;
        };
        if dep.status != "done" || !dep.labels.iter().any(|l| l == CLARIFICATION_LABEL) {
            continue;
        }
        let answer = tasks
            .notes(
                dep.id,
                i64::try_from(ROUTER_THREAD_POSTS_MAX).unwrap_or(i64::MAX),
            )
            .await
            .map_err(|e| e.to_string())?
            .into_iter()
            .find(|n| {
                !n.author_member_id
                    .as_deref()
                    .is_some_and(|who| who.starts_with(ROUTER_MEMBER_PREFIX))
            })
            .map(|n| n.content);
        answers.push(ClarificationAnswer {
            card_id: dep.id,
            answer,
        });
    }
    debug_assert!(answers.len() <= ROUTER_ANSWERS_MAX, "bounded");
    Ok(answers)
}

/// The router prompt for `context`, at most [`ROUTER_PROMPT_BYTES_MAX`] bytes.
///
/// Fixed parts first (the job, the answer format, the card), then the roster
/// until the budget is spent — a member that does not fit is counted, never
/// silently absent, so the router knows the list is partial.
#[must_use]
pub fn router_prompt(context: &RouterContext) -> String {
    let mut prompt = String::from(
        "You are the board's Task Management Agent. You take no work yourself: you read one \
         card and decide who does it. Answer with exactly ONE JSON object and nothing else:\n\
         {\"decision\":\"assign\",\"member\":\"<member id>\",\"labels\":[\"…\"],\"acceptance\":<check or null>,\"reason\":\"…\"}\n\
         {\"decision\":\"split\",\"subtasks\":[{\"title\":\"…\",\"assignee\":\"<member id or null>\"}],\"reason\":\"…\"}\n\
         {\"decision\":\"clarify\",\"question\":\"<what the human must answer>\",\"reason\":\"…\"}\n\
         {\"decision\":\"park\",\"reason\":\"…\"}\n\
         Assign when one member fits; split when the card is several independent pieces of \
         work; clarify only when the card cannot be done without an answer from the human; \
         park when it should wait. Prefer members whose profile and past verdicts fit the \
         card. Every decision is posted on the card's thread with your reason. When you assign \
         a card that has no acceptance check, write the one it is done by — \
         {\"kind\":\"command\",\"command\":\"…\"}, {\"kind\":\"file_exists\",\"path\":\"…\"} or \
         {\"kind\":\"regex\",\"path\":\"…\",\"pattern\":\"…\"} — or null if no machine check \
         fits; labels (one word each, at most 8) file the card for the members who match it.\
         \n\n== CARD ==\n",
    );
    write_card(&mut prompt, &context.card);
    write_thread(&mut prompt, &context.thread);
    write_answers(&mut prompt, &context.answers);

    prompt.push_str("\n== MEMBERS ==\n");
    let mut omitted = 0usize;
    for member in &context.roster {
        let line = member_line(member, &context.verdicts);
        // Keep room for the omission line itself.
        if prompt.len() + line.len() + 96 > ROUTER_PROMPT_BYTES_MAX {
            omitted += 1;
            continue;
        }
        prompt.push_str(&line);
    }
    if omitted > 0 {
        let _ = writeln!(
            prompt,
            "(…and {omitted} more member(s) not listed, to fit the prompt)"
        );
    }
    debug_assert!(
        omitted > 0 || prompt.len() <= ROUTER_PROMPT_BYTES_MAX,
        "within budget unless the fixed part alone overflows"
    );
    prompt
}

/// The card's own lines: title, description, labels, and the fields set.
fn write_card(prompt: &mut String, card: &Task) {
    let _ = writeln!(prompt, "#{} {} (p{})", card.id, card.title, card.priority);
    if let Some(description) = card.description.as_deref().filter(|d| !d.trim().is_empty()) {
        let _ = writeln!(
            prompt,
            "{}",
            preview(description, ROUTER_TEXT_PREVIEW_BYTES)
        );
    }
    if !card.labels.is_empty() {
        let _ = writeln!(
            prompt,
            "Labels: {}",
            preview(&card.labels.join(", "), ROUTER_TEXT_PREVIEW_BYTES)
        );
    }
    if let Some(assignee) = &card.assignee {
        let _ = writeln!(prompt, "Currently assigned to: {assignee}");
    }
    if let Some(deadline) = &card.deadline_at {
        let _ = writeln!(prompt, "Deadline: {deadline}");
    }
    if let Some(check) = &card.acceptance {
        let _ = writeln!(
            prompt,
            "Acceptance (set; keep it): {}",
            preview(&check.to_string(), ROUTER_ACCEPTANCE_PREVIEW_BYTES)
        );
    }
    if let Some(recurrence) = &card.recurrence {
        let _ = writeln!(
            prompt,
            "Recurs (cron): {}",
            preview(recurrence, ROUTER_POST_PREVIEW_BYTES)
        );
    }
}

/// The card's recent thread, oldest first.
fn write_thread(prompt: &mut String, thread: &[TaskNote]) {
    if !thread.is_empty() {
        prompt.push_str("\n== THREAD (oldest first) ==\n");
        for note in thread {
            let who = note.author_member_id.as_deref().unwrap_or("unknown");
            let _ = writeln!(
                prompt,
                "[{}] {who}: {}",
                note.kind.as_str(),
                preview(&note.content, ROUTER_POST_PREVIEW_BYTES)
            );
        }
    }
}

/// The human's answers to the clarifications the card waited on.
fn write_answers(prompt: &mut String, answers: &[ClarificationAnswer]) {
    if !answers.is_empty() {
        prompt.push_str("\n== ANSWERED CLARIFICATIONS (newest first) ==\n");
        for answered in answers {
            match answered.answer.as_deref() {
                Some(text) => {
                    let _ = writeln!(
                        prompt,
                        "#{} answered: {}",
                        answered.card_id,
                        preview(text, ROUTER_ANSWER_PREVIEW_BYTES)
                    );
                }
                None => {
                    let _ = writeln!(
                        prompt,
                        "#{} was completed with no answer posted",
                        answered.card_id
                    );
                }
            }
        }
    }
}

/// One roster line: id, name, kind, status, profile preview, verdict record.
fn member_line(member: &Member, verdicts: &[VerdictTally]) -> String {
    let (passed, failed) = verdicts
        .iter()
        .filter(|v| v.member_id == member.id && v.label.is_none())
        .fold((0u64, 0u64), |(p, f), v| (p + v.passed, f + v.failed));
    let profile = if member.profile.is_null()
        || member
            .profile
            .as_object()
            .is_some_and(serde_json::Map::is_empty)
    {
        "no profile".to_string()
    } else {
        preview(&member.profile.to_string(), ROUTER_PROFILE_PREVIEW_BYTES)
    };
    format!(
        "- {} \"{}\" ({}, {}) — {profile} — verdicts: {passed} passed, {failed} failed\n",
        member.id,
        member.name,
        member.kind.as_str(),
        member.status.as_str(),
    )
}

/// `text` cut to at most `max_bytes` on a char boundary, newlines flattened.
fn preview(text: &str, max_bytes: usize) -> String {
    let mut end = text.len().min(max_bytes);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    let mut shown = text[..end].replace('\n', " ");
    if end < text.len() {
        shown.push('…');
    }
    shown
}

/// Route card `task_id`: gather, ask, and apply.
///
/// The reply is parsed with [`parse_decision`]; if it is not a usable
/// decision the model is asked once more with the reason appended, and a
/// second unusable reply is an error — the card is left untouched, so a
/// confused model can never half-apply a decision.
///
/// # Errors
/// Why the card was not routed: not on a board, a store failure, the model
/// failing, two unusable replies, or `apply_decision` refusing the decision.
pub async fn route_card(
    storage: &Storage,
    complete: &RouterComplete,
    task_id: i64,
    run_is_live: bool,
) -> Result<AppliedDecision, String> {
    let context = gather_context(storage, task_id).await?;
    let router_id = router_for(&context.card)
        .ok_or_else(|| format!("card #{task_id} is not on a board (session scope)"))?;
    let prompt = router_prompt(&context);

    let reply = complete(prompt.clone()).await?;
    let decision = match parse_decision(&reply) {
        Ok(decision) => decision,
        Err(why) => {
            tracing::info!(task_id, %why, "router reply unusable; asking once more");
            let retry = format!(
                "{prompt}\n\nYour previous reply could not be used: {why}\n\
                 Answer again with exactly one JSON decision object."
            );
            let reply = complete(retry).await?;
            parse_decision(&reply)
                .map_err(|why| format!("the router model gave no usable decision twice: {why}"))?
        }
    };

    apply_decision(
        &storage.tasks(),
        &storage.members(),
        &router_id,
        task_id,
        &decision,
        run_is_live,
    )
    .await
    .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nanna_storage::{MemberKind, MemberOwner, MemberStatus, NewMember, NewTask};
    use std::sync::{Arc, Mutex};

    /// A scripted model: answers each call with the next reply, and records
    /// every prompt it was given.
    fn scripted(replies: Vec<&str>) -> (Box<RouterComplete>, Arc<Mutex<Vec<String>>>) {
        let replies: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(
            replies.into_iter().rev().map(String::from).collect(),
        ));
        let prompts = Arc::new(Mutex::new(Vec::new()));
        let seen = Arc::clone(&prompts);
        let complete: Box<RouterComplete> = Box::new(move |prompt: String| {
            seen.lock().unwrap().push(prompt);
            let next = replies.lock().unwrap().pop();
            Box::pin(async move { next.ok_or_else(|| "script exhausted".to_string()) })
        });
        (complete, prompts)
    }

    async fn board() -> Storage {
        let storage = Storage::in_memory().await.unwrap();
        storage
            .members()
            .create(NewMember {
                id: "agent:coder".to_string(),
                name: "Coder".to_string(),
                avatar: None,
                kind: MemberKind::Agent,
                owner_kind: MemberOwner::Workspace,
                owner_id: None,
                status: MemberStatus::Idle,
                profile: serde_json::json!({"capabilities": ["rust", "tests"]}),
            })
            .await
            .unwrap();
        storage
    }

    async fn card(storage: &Storage, scope: &str) -> Task {
        storage
            .tasks()
            .create(NewTask {
                scope: scope.to_string(),
                scope_id: (scope == "session").then(|| "s1".to_string()),
                title: "port the parser".to_string(),
                description: Some("Move it to the new AST.\nKeep the tests green.".to_string()),
                priority: 2,
                ..NewTask::default()
            })
            .await
            .unwrap()
    }

    /// P25 decision 6: once the human completes a clarification, the router
    /// decides again — and must see what the human said, not only that the
    /// card is unblocked.
    #[tokio::test]
    async fn the_prompt_carries_the_humans_answer_to_a_clarification() {
        let storage = board().await;
        let task = card(&storage, "global").await;
        let (complete, _) = scripted(vec![
            r#"{"decision":"clarify","question":"Which AST crate?","reason":"two exist"}"#,
        ]);
        let applied = route_card(&storage, &complete, task.id, false)
            .await
            .unwrap();
        let clarification = applied.created[0];
        let tasks = storage.tasks();
        tasks
            .post(
                clarification,
                Some("gui"),
                Some(nanna_storage::HUMAN_MEMBER_ID),
                nanna_storage::TaskNoteKind::Comment,
                "The new one\nin crates/ast.",
            )
            .await
            .unwrap();
        let before = router_prompt(&gather_context(&storage, task.id).await.unwrap());
        assert!(
            !before.contains("ANSWERED CLARIFICATIONS"),
            "an open clarification answers nothing yet: {before}"
        );

        tasks
            .complete(clarification, Some("gui"), None)
            .await
            .unwrap();
        let prompt = router_prompt(&gather_context(&storage, task.id).await.unwrap());
        assert!(
            prompt.contains(&format!(
                "#{clarification} answered: The new one in crates/ast."
            )),
            "{prompt}"
        );
        assert!(
            prompt.contains("Which AST crate?"),
            "the question is on the thread already: {prompt}"
        );
        assert!(prompt.len() <= ROUTER_PROMPT_BYTES_MAX);
    }

    /// The budget arithmetic on [`ROUTER_POST_PREVIEW_BYTES`], run: every
    /// section at its maximum still leaves the human and two agents listed.
    #[tokio::test]
    async fn the_roster_keeps_room_in_the_worst_case_prompt() {
        let storage = board().await;
        let task = card(&storage, "global").await;
        let mut context = gather_context(&storage, task.id).await.unwrap();
        let long = "x".repeat(4 * ROUTER_TEXT_PREVIEW_BYTES);
        context.card.title = "t".repeat(nanna_storage::TASK_TITLE_MAX_BYTES);
        context.card.description = Some(long.clone());
        context.card.labels = vec![long.clone()];
        context.card.acceptance = Some(serde_json::json!({"kind": "command", "command": long}));
        context.card.assignee = Some("agent:coder".to_string());
        context.card.recurrence = Some("0 9 * * 1".to_string());
        let post = TaskNote {
            id: 1,
            task_id: task.id,
            author: None,
            author_member_id: Some("human".to_string()),
            kind: nanna_storage::TaskNoteKind::Comment,
            content: long.clone(),
            created_at: String::new(),
        };
        context.thread = vec![post; ROUTER_THREAD_POSTS_MAX];
        context.answers = (0..ROUTER_ANSWERS_MAX)
            .map(|i| ClarificationAnswer {
                card_id: i64::try_from(i).unwrap() + 100,
                answer: Some(long.clone()),
            })
            .collect();
        assert_eq!(context.roster.len(), 2, "the human and the coder");
        let mut second = context.roster[1].clone();
        second.id = "agent:reviewer".to_string();
        context.roster.push(second);

        let prompt = router_prompt(&context);
        assert!(prompt.len() <= ROUTER_PROMPT_BYTES_MAX, "{}", prompt.len());
        assert!(
            !prompt.contains("not listed"),
            "every member fits: {prompt}"
        );
        assert!(prompt.contains("- agent:reviewer "), "{prompt}");
    }

    #[tokio::test]
    async fn the_prompt_shows_the_card_and_every_member_but_no_router() {
        let storage = board().await;
        let task = card(&storage, "global").await;
        let context = gather_context(&storage, task.id).await.unwrap();
        let prompt = router_prompt(&context);
        assert!(
            prompt.contains(&format!("#{} port the parser (p2)", task.id)),
            "{prompt}"
        );
        assert!(
            prompt.contains("Move it to the new AST. Keep the tests green."),
            "{prompt}"
        );
        assert!(
            prompt.contains("- agent:coder \"Coder\" (agent, idle)"),
            "{prompt}"
        );
        assert!(
            prompt.contains("\"rust\""),
            "the profile is shown: {prompt}"
        );
        assert!(
            prompt.contains("- human "),
            "the human is a member: {prompt}"
        );
        assert!(
            !prompt.contains("- router:"),
            "the router takes no work: {prompt}"
        );
        assert!(prompt.len() <= ROUTER_PROMPT_BYTES_MAX);
    }

    #[tokio::test]
    async fn a_big_roster_is_cut_to_the_budget_and_says_so() {
        let storage = board().await;
        for i in 0..200 {
            storage
                .members()
                .create(NewMember {
                    id: format!("agent:a{i}"),
                    name: format!("Agent {i}"),
                    avatar: None,
                    kind: MemberKind::Agent,
                    owner_kind: MemberOwner::Workspace,
                    owner_id: None,
                    status: MemberStatus::Idle,
                    profile: serde_json::json!({"notes": "x".repeat(1_000)}),
                })
                .await
                .unwrap();
        }
        let task = card(&storage, "global").await;
        let prompt = router_prompt(&gather_context(&storage, task.id).await.unwrap());
        assert!(
            prompt.len() <= ROUTER_PROMPT_BYTES_MAX,
            "{} bytes",
            prompt.len()
        );
        assert!(
            prompt.contains("more member(s) not listed"),
            "the cut is announced"
        );
    }

    #[tokio::test]
    async fn a_decision_is_asked_for_and_applied() {
        let storage = board().await;
        let task = card(&storage, "global").await;
        let (complete, prompts) = scripted(vec![
            r#"{"decision":"assign","member":"agent:coder","reason":"Rust work"}"#,
        ]);
        let applied = route_card(&storage, complete.as_ref(), task.id, false)
            .await
            .unwrap();
        assert_eq!(applied.assigned_to.as_deref(), Some("agent:coder"));
        assert_eq!(
            prompts.lock().unwrap().len(),
            1,
            "one call when the first reply is usable"
        );
        let card = storage.tasks().get(task.id).await.unwrap();
        assert_eq!(card.assignee.as_deref(), Some("agent:coder"));
    }

    /// An unusable reply is re-asked once, with the reason; a second one
    /// leaves the card untouched.
    #[tokio::test]
    async fn an_unusable_reply_is_re_asked_once_then_refused() {
        let storage = board().await;
        let task = card(&storage, "global").await;
        let (complete, prompts) = scripted(vec![
            "I think the coder should do it.",
            r#"{"decision":"park","reason":"needs the new AST first"}"#,
        ]);
        let applied = route_card(&storage, complete.as_ref(), task.id, false)
            .await
            .unwrap();
        assert_eq!(applied.created, Vec::<i64>::new(), "a park creates nothing");
        let prompts = prompts.lock().unwrap().clone();
        assert_eq!(prompts.len(), 2);
        assert!(
            prompts[1].contains("could not be used: no JSON object"),
            "{}",
            prompts[1]
        );

        let other = card(&storage, "global").await;
        let (complete, _) = scripted(vec!["nope", "still nope"]);
        let err = route_card(&storage, complete.as_ref(), other.id, false)
            .await
            .unwrap_err();
        assert!(err.contains("no usable decision twice"), "{err}");
        assert!(
            storage
                .tasks()
                .notes(other.id, 10)
                .await
                .unwrap()
                .is_empty(),
            "nothing was applied or posted"
        );
    }

    #[tokio::test]
    async fn a_card_off_the_board_is_not_routed() {
        let storage = board().await;
        let task = card(&storage, "session").await;
        let (complete, prompts) = scripted(vec![]);
        let err = route_card(&storage, complete.as_ref(), task.id, false)
            .await
            .unwrap_err();
        assert!(err.contains("not on a board"), "{err}");
        assert!(prompts.lock().unwrap().is_empty(), "no model call for it");
    }
}
