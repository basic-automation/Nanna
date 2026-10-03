//! `task.quick_add`: one line of text becomes one board card (P25 decision 1).
//!
//! The tokens are parsed by [`nanna_storage::quick_add`]; what is left here is
//! the part only the daemon can do — choosing the board and resolving the
//! `@member` handle against that board's roster — before the card goes
//! through the same `task.create` path as every other IPC card (so it is
//! recorded as `gui`, and the board's router completes what the line left out).

use super::member::agent_member_id;
use super::task::CreateTask;
use super::{ControlPlane, Value, json};
use nanna_storage::{HUMAN_MEMBER_ID, Member, ROUTER_MEMBER_PREFIX, TaskRepository};

/// How many handles a "no such member" reply lists.
///
/// Bound justification: the reply is read by a human in one toast; a board
/// with more members than this is better served by the roster page, and the
/// count of the rest is still given.
const UNKNOWN_MEMBER_HANDLES_MAX: usize = 12;

impl ControlPlane {
    /// `TaskAction::QuickAdd`. `scope` defaults to the active workspace's
    /// board, else the global board; session scope is refused — quick-add is
    /// the board's entry, and a session card is not on any board.
    pub(super) async fn task_quick_add(
        &self,
        repo: &TaskRepository,
        text: &str,
        scope: Option<String>,
        parent_id: Option<i64>,
        workspace_id: Option<String>,
    ) -> Value {
        let today = chrono::Utc::now().date_naive();
        let parsed = match nanna_storage::quick_add::parse(text, today) {
            Ok(parsed) => parsed,
            Err(e) => return json!({"error": "bad_quick_add", "message": e.to_string()}),
        };
        let (scope, scope_id) = match self
            .quick_add_board(repo, scope, parent_id, workspace_id.as_deref())
            .await
        {
            Ok(board) => board,
            Err(reply) => return reply,
        };
        debug_assert!(scope != "session", "refused above");
        let assignee = match parsed.assignee.as_deref() {
            Some(handle) => match self.board_member(scope_id.as_deref(), handle).await {
                Ok(id) => Some(id),
                Err(reply) => return reply,
            },
            None => None,
        };
        // The board resolved here is the one the card goes on: hand its id
        // through, or `task_create` would resolve "workspace" again — to the
        // daemon's active workspace, which need not be this one.
        let board_workspace = (scope == "workspace").then(|| scope_id.clone()).flatten();
        let request = CreateTask {
            workspace_id: board_workspace,
            title: parsed.title.clone(),
            scope: Some(scope),
            session_id: None,
            parent_id,
            description: None,
            priority: parsed.priority,
            labels: Some(parsed.labels.clone()),
            tools: None,
            due_at: parsed.due_at.clone(),
            deadline_at: parsed.deadline_at.clone(),
            recurrence: None,
            depends_on: None,
            acceptance: None,
            project: None,
            assignee,
        };
        let mut reply = self.task_create(repo, request).await;
        if let Some(object) = reply.as_object_mut().filter(|o| o.contains_key("task")) {
            object.insert("parsed".to_string(), json!(parsed));
        }
        reply
    }

    /// The board a quick-add line lands on: the parent card's (a sub-card),
    /// else `scope` — default the open workspace's board, else global. A
    /// session card is on no board, so neither it nor a session parent is
    /// accepted; every refusal is the IPC reply.
    async fn quick_add_board(
        &self,
        repo: &TaskRepository,
        scope: Option<String>,
        parent_id: Option<i64>,
        workspace_id: Option<&str>,
    ) -> Result<(String, Option<String>), Value> {
        const NOT_A_BOARD: &str =
            "quick-add creates board cards — use scope \"workspace\" or \"global\"";
        if let Some(parent_id) = parent_id {
            let parent = repo
                .get(parent_id)
                .await
                .map_err(|e| json!({"error": "task_not_found", "message": e.to_string()}))?;
            if parent.scope == "session" {
                return Err(json!({"error": "bad_scope", "message": NOT_A_BOARD}));
            }
            return Ok((parent.scope, parent.scope_id));
        }
        let has_active = self.workspaces.read().await.active().is_some();
        let named = workspace_id.is_some_and(|id| !id.trim().is_empty());
        let scope = scope.unwrap_or_else(|| {
            if has_active || named {
                "workspace"
            } else {
                "global"
            }
            .to_string()
        });
        if scope.eq_ignore_ascii_case("session") {
            return Err(json!({"error": "bad_scope", "message": NOT_A_BOARD}));
        }
        let board = self
            .resolve_board_scope(Some(&scope), None, workspace_id)
            .await
            .map_err(|message| json!({"error": "bad_scope", "message": message}))?;
        debug_assert!(board.0 != "session", "refused above");
        Ok(board)
    }

    /// The id of the member `@handle` names on the board `workspace_id`
    /// (`None` = the global board), or the reply that says why it names none.
    async fn board_member(
        &self,
        workspace_id: Option<&str>,
        handle: &str,
    ) -> Result<String, Value> {
        let Some(ref storage) = self.storage else {
            return Err(json!({"error": "storage_unavailable", "message": "no roster"}));
        };
        let roster = storage
            .members()
            .list_for_workspace(workspace_id)
            .await
            .map_err(|e| json!({"error": "member_list_failed", "message": e.to_string()}))?;
        resolve_member_handle(handle, &roster)
            .map_err(|message| json!({"error": "unknown_member", "message": message}))
    }
}

/// Which member `@handle` names on a board with `roster`.
///
/// A handle matches a member's id (`agent:ada`), the id without its `agent:`
/// prefix (`ada`), the slug of the member's name (`ada-lovelace` for "Ada
/// Lovelace", the same slug its id was derived from), or the name itself
/// ignoring case; `me` is the human. A router is never a match target —
/// it takes no work (P25 decision 4) — and a handle naming one says so.
///
/// # Errors
/// The message to show: no member, a router, or more than one member.
pub(super) fn resolve_member_handle(handle: &str, roster: &[Member]) -> Result<String, String> {
    let wanted = handle.to_lowercase();
    if wanted == "me" {
        return Ok(HUMAN_MEMBER_ID.to_string());
    }
    let wanted_id = agent_member_id(handle);
    let matches = |member: &&Member| {
        member.id.eq_ignore_ascii_case(&wanted)
            || member.id == wanted_id
            || agent_member_id(&member.name) == wanted_id
            || member.name.to_lowercase() == wanted
    };
    let found: Vec<&Member> = roster.iter().filter(matches).collect();
    if found.iter().any(|m| m.id.starts_with(ROUTER_MEMBER_PREFIX)) {
        return Err(format!(
            "@{handle} is the board's router, which assigns work but takes none"
        ));
    }
    match found.as_slice() {
        [member] => Ok(member.id.clone()),
        [] => Err(unknown_handle_message(handle, roster)),
        many => Err(format!(
            "@{handle} names {} members ({}) — use the member's id",
            many.len(),
            many.iter()
                .map(|m| m.id.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )),
    }
}

/// "No member @x on this board — try @a, @b, …" with at most
/// [`UNKNOWN_MEMBER_HANDLES_MAX`] handles.
fn unknown_handle_message(handle: &str, roster: &[Member]) -> String {
    let takers: Vec<&Member> = roster
        .iter()
        .filter(|m| !m.id.starts_with(ROUTER_MEMBER_PREFIX))
        .collect();
    let shown: Vec<String> = takers
        .iter()
        .take(UNKNOWN_MEMBER_HANDLES_MAX)
        .map(|m| format!("@{}", m.id.strip_prefix("agent:").unwrap_or(&m.id)))
        .collect();
    debug_assert!(shown.len() <= UNKNOWN_MEMBER_HANDLES_MAX, "bounded list");
    let rest = takers.len().saturating_sub(shown.len());
    let more = if rest > 0 {
        format!(" and {rest} more")
    } else {
        String::new()
    };
    format!(
        "no member @{handle} on this board — members: {}{more}",
        shown.join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use nanna_storage::{MemberKind, MemberOwner, MemberStatus};

    fn member(id: &str, name: &str, kind: MemberKind) -> Member {
        Member {
            id: id.to_string(),
            name: name.to_string(),
            avatar: None,
            kind,
            owner_kind: MemberOwner::Workspace,
            owner_id: None,
            status: MemberStatus::Idle,
            profile: json!({}),
            created_at: String::new(),
            updated_at: String::new(),
        }
    }

    fn roster() -> Vec<Member> {
        vec![
            member("human", "You", MemberKind::Human),
            member("router:global", "Task Manager", MemberKind::Agent),
            member("agent:ada-lovelace", "Ada Lovelace", MemberKind::Agent),
            member("agent:builder", "Builder", MemberKind::Agent),
            member("agent:x1f2a3b4", "ミク", MemberKind::Agent),
        ]
    }

    #[test]
    fn a_handle_resolves_by_id_slug_or_name() {
        let roster = roster();
        for handle in ["agent:builder", "builder", "Builder", "BUILDER"] {
            assert_eq!(
                resolve_member_handle(handle, &roster).unwrap(),
                "agent:builder",
                "{handle}"
            );
        }
        for handle in ["ada-lovelace", "Ada-Lovelace", "ada_lovelace"] {
            assert_eq!(
                resolve_member_handle(handle, &roster).unwrap(),
                "agent:ada-lovelace",
                "{handle}"
            );
        }
        assert_eq!(
            resolve_member_handle("ミク", &roster).unwrap(),
            "agent:x1f2a3b4"
        );
    }

    #[test]
    fn me_and_human_are_the_human() {
        assert_eq!(resolve_member_handle("me", &roster()).unwrap(), "human");
        assert_eq!(resolve_member_handle("Human", &roster()).unwrap(), "human");
    }

    #[test]
    fn the_router_is_never_assigned() {
        let err = resolve_member_handle("router:global", &roster()).unwrap_err();
        assert!(err.contains("router"), "{err}");
        let err = resolve_member_handle("task-manager", &roster()).unwrap_err();
        assert!(err.contains("takes none"), "{err}");
    }

    #[test]
    fn an_unknown_handle_lists_the_board_without_its_router() {
        let err = resolve_member_handle("nobody", &roster()).unwrap_err();
        assert!(
            err.contains("@builder") && err.contains("@ada-lovelace") && err.contains("@human"),
            "{err}"
        );
        assert!(!err.contains("router"), "{err}");
    }

    #[test]
    fn the_unknown_handle_list_is_bounded() {
        let roster: Vec<Member> = (0..30)
            .map(|i| member(&format!("agent:a{i}"), &format!("A{i}"), MemberKind::Agent))
            .collect();
        let err = resolve_member_handle("nobody", &roster).unwrap_err();
        assert_eq!(
            err.matches('@').count(),
            UNKNOWN_MEMBER_HANDLES_MAX + 1,
            "{err}"
        );
        assert!(err.ends_with("and 18 more"), "{err}");
    }

    #[test]
    fn two_members_answering_one_handle_are_refused() {
        let mut roster = roster();
        roster.push(member("agent:builder-2", "builder", MemberKind::Agent));
        let err = resolve_member_handle("builder", &roster).unwrap_err();
        assert!(err.contains("2 members"), "{err}");
    }
}
