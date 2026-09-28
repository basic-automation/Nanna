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

use nanna_storage::routing::{AppliedDecision, apply_decision, parse_decision};
use nanna_storage::{
    Member, ROUTER_MEMBER_PREFIX, Storage, Task, TaskNote, VerdictTally, router_member_id,
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
/// (~1.1 KB), a maximal title (500 B), the description and labels (800 B
/// each) and eight posts at this size total about 7.4 KB, leaving the roster
/// at least 1.8 KB of [`ROUTER_PROMPT_BYTES_MAX`] — room for the human and a
/// handful of agents even in the worst case.
pub const ROUTER_POST_PREVIEW_BYTES: usize = 480;

/// How many recent verdicts the router's outcome history is drawn from.
pub const ROUTER_VERDICT_WINDOW: usize = 500;

/// What the router reads before it decides.
#[derive(Debug, Clone)]
pub struct RouterContext {
    pub card: Task,
    /// Oldest first, at most [`ROUTER_THREAD_POSTS_MAX`].
    pub thread: Vec<TaskNote>,
    /// Members who can take the card — never a router.
    pub roster: Vec<Member>,
    pub verdicts: Vec<VerdictTally>,
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
    debug_assert!(thread.len() <= ROUTER_THREAD_POSTS_MAX, "thread bounded");
    Ok(RouterContext {
        card,
        thread,
        roster,
        verdicts,
    })
}

/// The router prompt for `context`, at most [`ROUTER_PROMPT_BYTES_MAX`] bytes.
///
/// Fixed parts first (the job, the answer format, the card), then the roster
/// until the budget is spent — a member that does not fit is counted, never
/// silently absent, so the router knows the list is partial.
#[must_use]
pub fn router_prompt(context: &RouterContext) -> String {
    let card = &context.card;
    let mut prompt = String::from(
        "You are the board's Task Management Agent. You take no work yourself: you read one \
         card and decide who does it. Answer with exactly ONE JSON object and nothing else:\n\
         {\"decision\":\"assign\",\"member\":\"<member id>\",\"reason\":\"…\"}\n\
         {\"decision\":\"split\",\"subtasks\":[{\"title\":\"…\",\"assignee\":\"<member id or null>\"}],\"reason\":\"…\"}\n\
         {\"decision\":\"clarify\",\"question\":\"<what the human must answer>\",\"reason\":\"…\"}\n\
         {\"decision\":\"park\",\"reason\":\"…\"}\n\
         Assign when one member fits; split when the card is several independent pieces of \
         work; clarify only when the card cannot be done without an answer from the human; \
         park when it should wait. Prefer members whose profile and past verdicts fit the \
         card. Every decision is posted on the card's thread with your reason.\n\n== CARD ==\n",
    );
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
    if !context.thread.is_empty() {
        prompt.push_str("\n== THREAD (oldest first) ==\n");
        for note in &context.thread {
            let who = note.author_member_id.as_deref().unwrap_or("unknown");
            let _ = writeln!(
                prompt,
                "[{}] {who}: {}",
                note.kind.as_str(),
                preview(&note.content, ROUTER_POST_PREVIEW_BYTES)
            );
        }
    }

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
