//! Which workspace a run's memory and task services scope to.
//!
//! The services (`memory.store`, `memory.search`, `memory.embed`, the
//! `tasks.*` workspace scope) used to read ONE process-wide slot that every
//! chat turn's preparation overwrote and nothing reset. A card run on
//! workspace B's board therefore recalled and wrote workspace A's memories
//! after a chat in A — or every workspace's, before any chat had run — and two
//! sessions running turns in different workspaces at once read each other's.
//! Memory belongs to the workspace (P25 decision 13); this is the same
//! process-wide-slot bug the tool working directory had, fixed the same way:
//! each run binds its own workspace under its run session, and a service call
//! resolves through the caller's run session first.
//!
//! The binding is keyed by the run session (`ToolRegistry::with_run_session`),
//! so a lookup must happen on the run's own task — a task-local does not cross
//! `tokio::spawn`. A caller with no run session, or a run that never bound,
//! falls back to the shared slot exactly as before.

use std::collections::HashMap;
use std::sync::{LazyLock, PoisonError, RwLock};

/// Most run sessions bound at once.
///
/// Bound justification: a binding lives exactly as long as its run — a chat
/// turn or a card run — and runs are bounded by the sessions and cards the
/// daemon works concurrently (tens). 4096 is far past that, so reaching it
/// means a release was missed; the binding is then refused (logged) and the
/// run falls back to the shared slot rather than the map growing forever.
pub const RUN_WORKSPACE_BINDINGS_MAX: usize = 4096;

static BINDINGS: LazyLock<RwLock<HashMap<String, Option<String>>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

/// Bind run session `session` to `workspace` (`None`: no workspace — global
/// and unscoped, which is a binding too, not an absence).
///
/// # Panics
/// When `session` is empty: every run session has an id.
pub fn bind(session: &str, workspace: Option<String>) {
    assert!(!session.is_empty(), "a run session has an id");
    let mut bindings = BINDINGS.write().unwrap_or_else(PoisonError::into_inner);
    let full = bindings.len() >= RUN_WORKSPACE_BINDINGS_MAX && !bindings.contains_key(session);
    debug_assert!(bindings.len() <= RUN_WORKSPACE_BINDINGS_MAX);
    if !full {
        bindings.insert(session.to_string(), workspace);
    }
    drop(bindings);
    if full {
        tracing::warn!(
            session,
            "run workspace bindings are full (a release was missed); this run uses the shared slot"
        );
    }
}

/// Drop run session `session`'s binding (call when its run ends).
pub fn release(session: &str) {
    BINDINGS
        .write()
        .unwrap_or_else(PoisonError::into_inner)
        .remove(session);
}

/// Run session `session`'s binding: `Some(workspace)` when bound (the
/// workspace itself may be `None`), `None` when it never bound.
#[must_use]
pub fn bound(session: &str) -> Option<Option<String>> {
    BINDINGS
        .read()
        .unwrap_or_else(PoisonError::into_inner)
        .get(session)
        .cloned()
}

/// The workspace in effect for the caller: its run session's binding, else
/// the shared slot.
pub async fn resolve(shared: &tokio::sync::RwLock<Option<String>>) -> Option<String> {
    if let Some(session) = nanna_tools::ToolRegistry::run_session_id()
        && let Some(workspace) = bound(&session)
    {
        return workspace;
    }
    shared.read().await.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A run resolves to its own binding whatever the shared slot holds; a
    /// caller outside any run, or in a run that never bound, gets the slot.
    #[tokio::test]
    async fn a_run_resolves_its_own_workspace_not_the_shared_slot() {
        let shared = tokio::sync::RwLock::new(Some("ws-a".to_string()));
        bind("card:9001", Some("ws-b".to_string()));
        bind("card:9002", None);

        let in_b =
            nanna_tools::ToolRegistry::with_run_session("card:9001".to_string(), resolve(&shared))
                .await;
        assert_eq!(
            in_b.as_deref(),
            Some("ws-b"),
            "the card's board, not the last chat's"
        );
        let unscoped =
            nanna_tools::ToolRegistry::with_run_session("card:9002".to_string(), resolve(&shared))
                .await;
        assert_eq!(unscoped, None, "a bound None is a binding, not a fallback");
        let unbound =
            nanna_tools::ToolRegistry::with_run_session("card:9003".to_string(), resolve(&shared))
                .await;
        assert_eq!(
            unbound.as_deref(),
            Some("ws-a"),
            "never bound: the shared slot"
        );
        assert_eq!(
            resolve(&shared).await.as_deref(),
            Some("ws-a"),
            "no run: the shared slot"
        );

        release("card:9001");
        release("card:9002");
        assert_eq!(bound("card:9001"), None, "released");
    }
}
