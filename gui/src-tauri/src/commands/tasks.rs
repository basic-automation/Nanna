//! Task-store commands for the GUI task management UI.
//!
//! The task store is the chat engine (P19): the planner seeds it, the harness
//! drains it, and the agent's `todo` skill writes to it. The GUI provides
//! a task checklist sidebar on the chat page, plus full CRUD operations
//! for manual task management.

use crate::state::{backend_handle, AppState};
use std::sync::Arc;
use tauri::State;
use tokio::sync::RwLock;

/// List tasks in a scope ("session" | "workspace" | "global"). With
/// `workspace_id`, a `workspace` scope is that workspace's board rather than
/// the daemon's active one.
///
/// # Errors
///
/// Fails when the daemon cannot be reached or the `task.list` request is
/// dropped or times out. A refusal the daemon reports in its reply is passed
/// through inside `Ok`.
#[tauri::command]
pub async fn list_tasks(
    state: State<'_, Arc<RwLock<AppState>>>,
    scope: String,
    session_id: Option<String>,
    include_closed: Option<bool>,
    workspace_id: Option<String>,
) -> Result<serde_json::Value, String> {
    backend_handle(&state)
        .await
        .task_list(
            &scope,
            session_id.as_deref(),
            include_closed,
            workspace_id.as_deref(),
        )
        .await
}

/// Create a new task.
///
/// # Errors
///
/// Fails when the daemon cannot be reached or the `task.create` request is
/// dropped or times out. A refusal the daemon reports in its reply is passed
/// through inside `Ok`.
#[tauri::command]
pub async fn create_task(
    state: State<'_, Arc<RwLock<AppState>>>,
    title: String,
    scope: String,
    session_id: Option<String>,
    parent_id: Option<i64>,
    description: Option<String>,
    priority: Option<i64>,
) -> Result<serde_json::Value, String> {
    backend_handle(&state)
        .await
        .task_create(
            &title,
            &scope,
            session_id.as_deref(),
            parent_id,
            description.as_deref(),
            priority,
        )
        .await
}

/// Update a task with a partial patch.
///
/// # Errors
///
/// Fails when the daemon cannot be reached or the `task.update` request is
/// dropped or times out. A refusal the daemon reports in its reply is passed
/// through inside `Ok`.
#[tauri::command]
pub async fn update_task(
    state: State<'_, Arc<RwLock<AppState>>>,
    id: i64,
    patch: serde_json::Value,
) -> Result<serde_json::Value, String> {
    backend_handle(&state).await.task_update(id, patch).await
}

/// Mark a task as done (with optional acceptance check).
///
/// # Errors
///
/// Fails when the daemon cannot be reached or the `task.done` request is
/// dropped or times out. A refusal the daemon reports in its reply is passed
/// through inside `Ok`.
#[tauri::command]
pub async fn complete_task(
    state: State<'_, Arc<RwLock<AppState>>>,
    id: i64,
    workdir: Option<String>,
) -> Result<serde_json::Value, String> {
    backend_handle(&state).await.task_done(id, workdir.as_deref()).await
}

/// Delete a task and its subtree.
///
/// # Errors
///
/// Fails when the daemon cannot be reached or the `task.delete` request is
/// dropped or times out. A refusal the daemon reports in its reply is passed
/// through inside `Ok`.
#[tauri::command]
pub async fn delete_task(
    state: State<'_, Arc<RwLock<AppState>>>,
    id: i64,
) -> Result<serde_json::Value, String> {
    backend_handle(&state).await.task_delete(id).await
}

/// Reorder a task by updating its priority.
///
/// # Errors
///
/// Fails when the daemon cannot be reached or the `task.update` request is
/// dropped or times out. A refusal the daemon reports in its reply is passed
/// through inside `Ok`.
#[tauri::command]
pub async fn reorder_task(
    state: State<'_, Arc<RwLock<AppState>>>,
    id: i64,
    new_priority: i64,
) -> Result<serde_json::Value, String> {
    backend_handle(&state).await.task_reorder(id, new_priority).await
}
