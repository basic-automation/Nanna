//! Board commands (P25 Stage 4): the board client's reads and writes.
//!
//! The board lists cards through `list_tasks` (scope `workspace` or `global`)
//! and edits them through `update_task`/`complete_task`; these are the parts
//! that had no command yet — the quick-add line, a card's thread, posting on
//! it, and the roster the assignee field and avatars come from.

use crate::state::{AppState, backend_handle};
use std::sync::Arc;
use tauri::State;
use tokio::sync::RwLock;

/// Turn one quick-add line into a card (`#label p1 @member friday {deadline}`);
/// with `parent_id`, a sub-card of that card; with `workspace_id`, on that
/// workspace's board rather than the daemon's active one.
///
/// # Errors
///
/// Fails when the daemon cannot be reached or the request is dropped or times
/// out. A line the daemon refuses comes back inside `Ok` with an `error`.
#[tauri::command]
pub async fn quick_add_card(
    state: State<'_, Arc<RwLock<AppState>>>,
    text: String,
    scope: Option<String>,
    parent_id: Option<i64>,
    workspace_id: Option<String>,
) -> Result<serde_json::Value, String> {
    backend_handle(&state)
        .await
        .task_quick_add(&text, scope.as_deref(), parent_id, workspace_id.as_deref())
        .await
}

/// One card with its thread (`notes`) and activity.
///
/// # Errors
///
/// Fails as [`quick_add_card`] does.
#[tauri::command]
pub async fn get_card(
    state: State<'_, Arc<RwLock<AppState>>>,
    id: i64,
) -> Result<serde_json::Value, String> {
    backend_handle(&state).await.task_get(id).await
}

/// The human posts on a card's thread.
///
/// # Errors
///
/// Fails as [`quick_add_card`] does.
#[tauri::command]
pub async fn post_on_card(
    state: State<'_, Arc<RwLock<AppState>>>,
    id: i64,
    content: String,
) -> Result<serde_json::Value, String> {
    backend_handle(&state).await.task_note(id, &content).await
}

/// One board's roster; `workspace_id` `None` is the global board.
///
/// # Errors
///
/// Fails as [`quick_add_card`] does.
#[tauri::command]
pub async fn list_members(
    state: State<'_, Arc<RwLock<AppState>>>,
    workspace_id: Option<String>,
) -> Result<serde_json::Value, String> {
    backend_handle(&state)
        .await
        .member_list(workspace_id.as_deref())
        .await
}

/// Every open card assigned to `member_id` (default: the human) across every
/// board, with the store's `today` — the Inbox and Upcoming lists.
///
/// # Errors
///
/// Fails as [`quick_add_card`] does.
#[tauri::command]
pub async fn list_assigned_cards(
    state: State<'_, Arc<RwLock<AppState>>>,
    member_id: Option<String>,
) -> Result<serde_json::Value, String> {
    backend_handle(&state)
        .await
        .task_assigned(member_id.as_deref())
        .await
}

/// Add an agent to a board's roster: the open workspace's (`workspace_id`),
/// the global board's (`None`), or the human's own (`personal`).
///
/// # Errors
///
/// Fails as [`quick_add_card`] does.
#[tauri::command]
pub async fn create_member(
    state: State<'_, Arc<RwLock<AppState>>>,
    name: String,
    workspace_id: Option<String>,
    personal: Option<bool>,
    profile: Option<serde_json::Value>,
) -> Result<serde_json::Value, String> {
    backend_handle(&state)
        .await
        .member_create(
            &name,
            workspace_id.as_deref(),
            personal.unwrap_or(false),
            profile,
        )
        .await
}

/// Change a member's name or profile.
///
/// # Errors
///
/// Fails as [`quick_add_card`] does.
#[tauri::command]
pub async fn update_member(
    state: State<'_, Arc<RwLock<AppState>>>,
    id: String,
    name: Option<String>,
    profile: Option<serde_json::Value>,
) -> Result<serde_json::Value, String> {
    backend_handle(&state)
        .await
        .member_update(&id, name.as_deref(), profile)
        .await
}

/// Remove an agent from the roster.
///
/// # Errors
///
/// Fails as [`quick_add_card`] does.
#[tauri::command]
pub async fn delete_member(
    state: State<'_, Arc<RwLock<AppState>>>,
    id: String,
) -> Result<serde_json::Value, String> {
    backend_handle(&state).await.member_delete(&id).await
}

/// Start — or resume, for a paused card — the assignee's run on a card.
///
/// # Errors
///
/// Fails as [`quick_add_card`] does.
#[tauri::command]
pub async fn start_card_run(
    state: State<'_, Arc<RwLock<AppState>>>,
    card_id: i64,
) -> Result<serde_json::Value, String> {
    backend_handle(&state).await.card_run_start(card_id).await
}

/// Whether a run works a card now (`{running, …}`).
///
/// # Errors
///
/// Fails as [`quick_add_card`] does.
#[tauri::command]
pub async fn card_run_status(
    state: State<'_, Arc<RwLock<AppState>>>,
    card_id: i64,
) -> Result<serde_json::Value, String> {
    backend_handle(&state).await.card_run_status(card_id).await
}

/// Stop the run working a card; the card stays with its member, paused.
///
/// # Errors
///
/// Fails as [`quick_add_card`] does.
#[tauri::command]
pub async fn stop_card_run(
    state: State<'_, Arc<RwLock<AppState>>>,
    card_id: i64,
) -> Result<serde_json::Value, String> {
    backend_handle(&state).await.card_run_cancel(card_id).await
}
