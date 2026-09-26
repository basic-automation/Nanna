//! Memory management commands. The daemon owns the memory store; these forward
//! to it.
//!
//! Tuning knobs that have a config home are persisted to `config.toml` and
//! pushed to the daemon; knobs the daemon manages internally are no-ops.

use crate::state::{backend_handle, AppState};
use serde::Serialize;
use std::sync::Arc;
use tauri::State;
use tokio::sync::RwLock;
use tracing::info;

/// Memory search result
#[derive(Debug, Clone, Serialize)]
pub struct MemorySearchResult {
    pub session_id: String,
    pub session_name: String,
    pub message_id: String,
    pub role: String,
    pub content: String,
    pub timestamp: String,
    pub snippet: String,
    pub relevance: f32,
}

/// A count from a daemon reply; 0 when the key is absent or not an unsigned
/// integer. Lossless on the 64-bit targets this ships for; it saturates where
/// the former `as usize` would have wrapped.
/// Characters of context shown on each side of a search match.
const SNIPPET_CONTEXT_CHARS: usize = 50;

/// The first case-insensitive match of `query_lower` in `content`, with up to
/// `context_chars` characters either side and `...` where the text continues, or
/// `None` when there is no match.
///
/// Char-safe in both directions. The old code sliced `content` at byte offsets
/// ±50 from a match, which panics inside a multi-byte character (an em dash
/// within 50 bytes of a match was enough to abort the GUI), and it found the
/// match in the LOWERCASED copy and applied that offset to the original —
/// but lowercasing can change a character's byte length (`İ` is 2 bytes, its
/// lowercase 3), so the offset could point anywhere. Here every byte of the
/// lowered copy remembers the original character it came from.
fn match_snippet(content: &str, query_lower: &str, context_chars: usize) -> Option<String> {
    if query_lower.is_empty() {
        return None;
    }
    let mut lowered = String::with_capacity(content.len());
    let mut origin: Vec<usize> = Vec::with_capacity(content.len());
    for (byte, ch) in content.char_indices() {
        let before = lowered.len();
        lowered.extend(ch.to_lowercase());
        origin.resize(lowered.len(), byte);
        debug_assert!(
            lowered.len() > before,
            "every char lowers to at least one char"
        );
    }
    let found = lowered.find(query_lower)?;
    let match_start = origin[found];
    let match_end = origin
        .get(found + query_lower.len())
        .copied()
        .unwrap_or(content.len());
    let start = content[..match_start]
        .char_indices()
        .rev()
        .nth(context_chars.saturating_sub(1))
        .map_or(0, |(i, _)| i);
    let end = content[match_end..]
        .char_indices()
        .nth(context_chars)
        .map_or(content.len(), |(i, _)| match_end + i);
    debug_assert!(content.is_char_boundary(start) && content.is_char_boundary(end));
    let prefix = if start > 0 { "..." } else { "" };
    let suffix = if end < content.len() { "..." } else { "" };
    Some(format!("{prefix}{}{suffix}", &content[start..end]))
}

fn count_field(reply: &serde_json::Value, key: &str) -> usize {
    reply
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .map_or(0, |n| usize::try_from(n).unwrap_or(usize::MAX))
}

/// Share of `content_len` taken up by `matches` query hits, capped at 1.
///
/// Relevance is `f32` on the wire; both counts are exact in `f32` up to 2^24
/// (16 MiB of message content), past which the score only loses precision it
/// cannot display.
fn match_density(matches: usize, content_len: usize) -> f32 {
    (nanna_numeric::f32_from_usize(matches) / nanna_numeric::f32_from_usize(content_len.max(1))).min(1.0)
}

/// Narrow a daemon-reported `f64` score to the `f32` the memory page's wire
/// type carries. The daemon's scores are `f32` widened to `f64` in its JSON,
/// so narrowing them back restores the exact value.
fn score_to_f32(score: f64) -> f32 {
    nanna_numeric::f32_from_f64(score)
}

/// Search across all sessions (substring match over daemon-stored history).
///
/// # Errors
///
/// Returns `Failed to list sessions: …` when the daemon cannot be reached or
/// the `session.list` request is dropped or times out. A session whose
/// `session.history` request fails is skipped rather than failing the search.
#[tauri::command]
pub async fn search_memory(
    state: State<'_, Arc<RwLock<AppState>>>,
    query: String,
    limit: Option<u32>,
) -> Result<Vec<MemorySearchResult>, String> {
    let backend = backend_handle(&state).await;
    let max_results = limit.unwrap_or(50) as usize;
    let query_lower = query.to_lowercase();

    // Sessions come from the daemon (it owns nanna.db).
    let sessions: Vec<(String, String)> = {
        let result = backend
            .sessions_list()
            .await
            .map_err(|e| format!("Failed to list sessions: {e}"))?;
        result
            .get("sessions")
            .and_then(|v| v.as_array())
            .map_or_default(|arr| {
                arr.iter()
                    .filter_map(|s| {
                        let id = s.get("id")?.as_str()?.to_string();
                        let name = s
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("Untitled")
                            .to_string();
                        Some((id, name))
                    })
                    .collect()
            })
    };

    let mut results = Vec::new();

    for (session_id, session_name) in &sessions {
        let messages: Vec<(String, String, String, String)> = backend
            .session_history(session_id, Some(1000))
            .await
            .map_or_else(
                |_| vec![],
                |result| {
                    result
                        .get("messages")
                        .and_then(|v| v.as_array())
                        .map_or_default(|msgs| {
                            msgs.iter()
                                .filter_map(|m| {
                                    Some((
                                        m.get("id")?.as_str()?.to_string(),
                                        m.get("role")?.as_str()?.to_string(),
                                        m.get("content")?.as_str()?.to_string(),
                                        m.get("timestamp")?.as_str()?.to_string(),
                                    ))
                                })
                                .collect()
                        })
                },
            );

        for (msg_id, role, content, timestamp) in messages {
            let content_lower = content.to_lowercase();
            if let Some(snippet) = match_snippet(&content, &query_lower, SNIPPET_CONTEXT_CHARS) {
                let matches = content_lower.matches(&query_lower).count();
                let relevance = match_density(matches, content.len());

                results.push(MemorySearchResult {
                    session_id: session_id.clone(),
                    session_name: session_name.clone(),
                    message_id: msg_id,
                    role,
                    content,
                    timestamp,
                    snippet,
                    relevance,
                });
            }
        }
    }

    results.sort_by(|a, b| b.relevance.partial_cmp(&a.relevance).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(max_results);

    Ok(results)
}

/// Statistics for the memory browser
#[derive(Debug, Clone, Serialize)]
pub struct MemoryStats {
    pub total_sessions: u32,
    pub total_messages: u32,
    pub oldest_session: Option<String>,
    pub newest_session: Option<String>,
}

/// Session and message totals for the memory browser.
///
/// # Errors
///
/// Returns `Failed to list sessions: …` when the daemon cannot be reached or
/// the `session.list` request is dropped or times out.
#[tauri::command]
pub async fn get_memory_stats(
    state: State<'_, Arc<RwLock<AppState>>>,
) -> Result<MemoryStats, String> {
    let result = backend_handle(&state)
        .await
        .sessions_list()
        .await
        .map_err(|e| format!("Failed to list sessions: {e}"))?;
    let sessions = result
        .get("sessions")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let mut total_messages = 0u32;
    let mut timestamps: Vec<String> = Vec::new();
    for session in &sessions {
        // Saturates where `as` wrapped; a count past u32::MAX is unreachable.
        total_messages += session
            .get("message_count")
            .and_then(serde_json::Value::as_u64)
            .map_or(0, |n| u32::try_from(n).unwrap_or(u32::MAX));
        if let Some(created) = session.get("created_at").and_then(|v| v.as_str()) {
            timestamps.push(created.to_string());
        }
    }
    timestamps.sort();

    Ok(MemoryStats {
        total_sessions: u32::try_from(sessions.len()).unwrap_or(u32::MAX),
        total_messages,
        oldest_session: timestamps.first().cloned(),
        newest_session: timestamps.last().cloned(),
    })
}

/// Set whether messages are automatically remembered (persisted to config +
/// pushed to the daemon).
///
/// # Errors
///
/// Returns `Failed to save config: …` when `config.toml` cannot be written; the
/// daemon is then not told, though this client's cached value has already
/// changed. The push to the daemon (`config.set` of
/// `memory.auto_remember_messages`) is best-effort and never fails the command.
#[tauri::command]
pub async fn set_auto_remember_messages(
    state: State<'_, Arc<RwLock<AppState>>>,
    enabled: bool,
) -> Result<(), String> {
    let mut state_guard = state.write().await;
    state_guard.config.memory.auto_remember_messages = enabled;
    state_guard.config.save().map_err(|e| format!("Failed to save config: {e}"))?;
    let _ = state_guard
        .backend
        .config_set("memory.auto_remember_messages", serde_json::json!(enabled))
        .await;
    drop(state_guard);
    info!("Auto-remember messages set: {enabled}");
    Ok(())
}

/// Set max compression ratio for memory consolidation (persisted to config +
/// pushed to the daemon).
///
/// # Errors
///
/// Returns `Failed to save config: …` when `config.toml` cannot be written; the
/// daemon is then not told, though this client's cached value has already
/// changed. The push to the daemon (`config.set` of
/// `memory.max_compression_ratio`) is best-effort and never fails the command.
#[tauri::command]
pub async fn set_max_compression_ratio(
    state: State<'_, Arc<RwLock<AppState>>>,
    ratio: f32,
) -> Result<(), String> {
    let mut state_guard = state.write().await;
    let clamped = ratio.clamp(0.1, 0.9);
    state_guard.config.memory.max_compression_ratio = clamped;
    state_guard.config.save().map_err(|e| format!("Failed to save config: {e}"))?;
    let _ = state_guard
        .backend
        .config_set("memory.max_compression_ratio", serde_json::json!(clamped))
        .await;
    drop(state_guard);
    info!("Max compression ratio set: {clamped}");
    Ok(())
}

/// Set minimum remaining memories floor for consolidation (persisted to config +
/// pushed to the daemon).
///
/// # Errors
///
/// Returns `Failed to save config: …` when `config.toml` cannot be written; the
/// daemon is then not told, though this client's cached value has already
/// changed. The push to the daemon (`config.set` of
/// `memory.min_remaining_memories`) is best-effort and never fails the command.
#[tauri::command]
pub async fn set_min_remaining_memories(
    state: State<'_, Arc<RwLock<AppState>>>,
    count: usize,
) -> Result<(), String> {
    let mut state_guard = state.write().await;
    let clamped = count.max(5);
    state_guard.config.memory.min_remaining_memories = clamped;
    state_guard.config.save().map_err(|e| format!("Failed to save config: {e}"))?;
    let _ = state_guard
        .backend
        .config_set("memory.min_remaining_memories", serde_json::json!(clamped))
        .await;
    drop(state_guard);
    info!("Min remaining memories set: {clamped}");
    Ok(())
}

// =============================================================================
// Cognitive Memory Commands (FSRS-6 + Dreaming)
// =============================================================================

/// Cognitive memory statistics
#[derive(Debug, Clone, Serialize)]
pub struct CognitiveMemoryStats {
    pub total_memories: usize,
    pub active: usize,
    pub dormant: usize,
    pub silent: usize,
    pub unavailable: usize,
    pub consolidation_enabled: bool,
    pub last_consolidation: Option<String>,
}

/// Memory-state totals (FSRS active/dormant/silent/unavailable) from the
/// daemon.
///
/// # Errors
///
/// Returns `Failed to get memory stats: …` when the daemon cannot be reached or
/// the `memory.stats` request is dropped or times out.
#[tauri::command]
pub async fn get_cognitive_memory_stats(
    state: State<'_, Arc<RwLock<AppState>>>,
) -> Result<CognitiveMemoryStats, String> {
    let result = backend_handle(&state)
        .await
        .memory_stats()
        .await
        .map_err(|e| format!("Failed to get memory stats: {e}"))?;

    Ok(CognitiveMemoryStats {
        total_memories: count_field(&result, "total"),
        active: count_field(&result, "active"),
        dormant: count_field(&result, "dormant"),
        silent: count_field(&result, "silent"),
        unavailable: count_field(&result, "unavailable"),
        consolidation_enabled: result
            .get("consolidation_enabled")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true),
        last_consolidation: result
            .get("last_consolidation")
            .and_then(|v| v.as_str())
            .map(str::to_string),
    })
}

/// Consolidation result for frontend
#[derive(Debug, Clone, Serialize)]
pub struct ConsolidationResultInfo {
    pub memories_processed: usize,
    pub clusters_formed: usize,
    pub memories_merged: usize,
    pub memories_expanded: usize,
    pub errors: Vec<String>,
}

/// Manually trigger memory consolidation ("dream").
///
/// # Errors
///
/// Returns `Consolidation failed: …` when the daemon cannot be reached or the
/// `memory.consolidate` request is dropped or times out. Consolidation can run
/// long; the request is allowed the client's full request timeout.
#[tauri::command]
pub async fn trigger_consolidation(
    state: State<'_, Arc<RwLock<AppState>>>,
) -> Result<ConsolidationResultInfo, String> {
    let result = backend_handle(&state)
        .await
        .memory_consolidate()
        .await
        .map_err(|e| format!("Consolidation failed: {e}"))?;

    Ok(ConsolidationResultInfo {
        memories_processed: count_field(&result, "memories_processed"),
        clusters_formed: count_field(&result, "clusters_formed"),
        memories_merged: count_field(&result, "memories_merged"),
        memories_expanded: count_field(&result, "memories_expanded"),
        errors: result
            .get("errors")
            .and_then(|v| v.as_array())
            .map_or_default(|arr| arr.iter().filter_map(|v| v.as_str().map(str::to_string)).collect()),
    })
}

/// Apply pending FSRS updates. The daemon applies these itself during recall, so
/// this is a no-op accepted for UI compatibility.
///
/// # Errors
///
/// Never returns `Err`; the `Result` is what Tauri requires of an async command
/// that borrows `State`.
#[tauri::command]
pub async fn apply_memory_updates(
    _state: State<'_, Arc<RwLock<AppState>>>,
) -> Result<(), String> {
    Ok(())
}

/// Manually save memories. The daemon persists via Turso write-through on every
/// mutation, so there is nothing to flush from the client — a no-op.
///
/// # Errors
///
/// Never returns `Err`; the `Result` is what Tauri requires of an async command
/// that borrows `State`.
#[tauri::command]
pub async fn save_memories(
    _state: State<'_, Arc<RwLock<AppState>>>,
) -> Result<(), String> {
    Ok(())
}

// =============================================================================
// Memory Management Commands
// =============================================================================

/// Memory entry for frontend display
#[derive(Debug, Clone, Serialize)]
pub struct MemoryItem {
    pub id: String,
    pub content: String,
    pub fact_type: String,
    pub importance: f32,
    pub state: String,
    pub weight: f32,
    pub retrievability: f32,
    pub access_count: u32,
    pub created_at: String,
    pub session_id: Option<String>,
    pub workspace_id: Option<String>,
}

fn memory_item_from_json(m: &serde_json::Value) -> Option<MemoryItem> {
    Some(MemoryItem {
        id: m.get("id")?.as_str()?.to_string(),
        content: m.get("content")?.as_str()?.to_string(),
        fact_type: m.get("fact_type").and_then(|v| v.as_str()).unwrap_or("stated").to_string(),
        importance: score_to_f32(m.get("importance").and_then(serde_json::Value::as_f64).unwrap_or(3.0)),
        state: m.get("state").and_then(|v| v.as_str()).unwrap_or("active").to_string(),
        weight: score_to_f32(m.get("weight").and_then(serde_json::Value::as_f64).unwrap_or(1.0)),
        retrievability: score_to_f32(m.get("retrievability").and_then(serde_json::Value::as_f64).unwrap_or(1.0)),
        // Saturates where `as` wrapped; an access count past u32::MAX is unreachable.
        access_count: m
            .get("access_count")
            .and_then(serde_json::Value::as_u64)
            .map_or(0, |n| u32::try_from(n).unwrap_or(u32::MAX)),
        created_at: m.get("created_at").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        session_id: m.get("session_id").and_then(|v| v.as_str()).map(String::from),
        workspace_id: m.get("workspace_id").and_then(|v| v.as_str()).map(String::from),
    })
}

/// Resolve the page's tab keyword into what the daemon expects: the daemon
/// filter takes "global" or an actual workspace ID. The page sends the
/// literal "workspace" plus the active workspace's id — forwarding the
/// literal matched a workspace named "workspace" (nothing) and showed the
/// global set on both tabs (observed live).
fn resolve_memory_scope(scope: Option<String>, workspace_id: Option<String>) -> Option<String> {
    match scope.as_deref() {
        Some("workspace") => workspace_id.or(scope),
        _ => scope,
    }
}

/// List semantic memories. Scope semantics: "global" = global-only;
/// a workspace id = global + that workspace (what the agent sees there).
///
/// # Errors
///
/// Returns `Failed to list memories: …` when the daemon cannot be reached or
/// the `memory.list` request is dropped or times out.
#[tauri::command]
pub async fn list_memories(
    state: State<'_, Arc<RwLock<AppState>>>,
    scope: Option<String>,
    workspace_id: Option<String>,
) -> Result<Vec<MemoryItem>, String> {
    let effective = resolve_memory_scope(scope, workspace_id);
    let result = backend_handle(&state)
        .await
        .memory_list(effective.as_deref())
        .await
        .map_err(|e| format!("Failed to list memories: {e}"))?;

    let mut items: Vec<MemoryItem> = result
        .get("memories")
        .and_then(|v| v.as_array())
        .map_or_default(|arr| arr.iter().filter_map(memory_item_from_json).collect());
    items.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    Ok(items)
}

/// Get a single memory by ID.
///
/// # Errors
///
/// Returns `Failed to get memory: …` when the daemon cannot be reached or the
/// `memory.get` request is dropped or times out. An unknown id — a reply
/// without a usable `memory` object — is `Ok(None)`.
#[tauri::command]
pub async fn get_memory(
    state: State<'_, Arc<RwLock<AppState>>>,
    id: String,
) -> Result<Option<MemoryItem>, String> {
    let result = backend_handle(&state)
        .await
        .memory_get(&id)
        .await
        .map_err(|e| format!("Failed to get memory: {e}"))?;
    Ok(result.get("memory").and_then(memory_item_from_json))
}

/// Delete a memory by ID.
///
/// # Errors
///
/// Returns `Failed to delete memory: …` when the daemon cannot be reached or
/// the `memory.delete` request is dropped or times out. A refusal the daemon
/// reports in its reply is not checked.
#[tauri::command]
pub async fn delete_memory(
    state: State<'_, Arc<RwLock<AppState>>>,
    id: String,
) -> Result<(), String> {
    backend_handle(&state)
        .await
        .memory_delete(&id)
        .await
        .map_err(|e| format!("Failed to delete memory: {e}"))?;
    info!("Deleted memory: {id}");
    Ok(())
}

/// Update a memory's content.
///
/// # Errors
///
/// Returns `Failed to update memory: …` when the daemon cannot be reached or
/// the `memory.update` request is dropped or times out. A refusal the daemon
/// reports in its reply is not checked.
#[tauri::command]
pub async fn update_memory(
    state: State<'_, Arc<RwLock<AppState>>>,
    id: String,
    content: String,
) -> Result<(), String> {
    backend_handle(&state)
        .await
        .memory_update(&id, Some(&content), None)
        .await
        .map_err(|e| format!("Failed to update memory: {e}"))?;
    info!("Updated memory: {id}");
    Ok(())
}

/// Clear memories in a scope.
///
/// "global" clears global-only; a workspace id clears ONLY that workspace's
/// entries (never the globals its tab also displays — destructive ops stay
/// conservative); no scope clears all.
/// (This is the command the memory page invokes. The old `clear_all_memories`
/// name never existed as a command — and the note here claiming nothing called
/// it was wrong: Settings → Data still invoked it until 2026-07-24, so its
/// "Delete All Memories" button was dead. Both call sites now use this one.)
///
/// # Errors
///
/// Returns `Failed to clear memories: …` when the daemon cannot be reached or
/// the `memory.clear` request is dropped or times out, and the daemon's own
/// message when it kept memories it could not delete from storage.
#[tauri::command]
pub async fn clear_memories(
    state: State<'_, Arc<RwLock<AppState>>>,
    scope: Option<String>,
    workspace_id: Option<String>,
) -> Result<(), String> {
    let effective = resolve_memory_scope(scope, workspace_id);
    let reply = backend_handle(&state)
        .await
        .memory_clear(effective.as_deref())
        .await
        .map_err(|e| format!("Failed to clear memories: {e}"))?;
    // A refused or partial clear arrives inside the `Ok` reply. Dropping it
    // here toasted "All memories cleared" over memories that were kept.
    if reply.get("error").is_some() {
        return Err(reply["message"]
            .as_str()
            .unwrap_or("The daemon refused to clear memories")
            .to_string());
    }
    info!("Cleared memories (scope: {:?}, via daemon)", effective);
    Ok(())
}

// =============================================================================
// Similarity Threshold Configuration
// =============================================================================

#[cfg(test)]
mod snippet_tests {
    use super::match_snippet;

    /// Both ways the old byte slicing aborted the GUI: a multi-byte character
    /// inside the context window, and a lowercase whose byte length differs
    /// from the original's.
    #[test]
    fn a_snippet_never_splits_a_character() {
        let dashes = format!("{}needle{}", "—".repeat(60), "—".repeat(60));
        let snippet = match_snippet(&dashes, "needle", 5).expect("found");
        assert_eq!(snippet, "...—————needle—————...");

        // `İ` is 2 bytes; its lowercase is 3. Offsets from the lowered copy
        // would have pointed into the wrong place in the original.
        let turkish = format!("{}Needle tail", "İ".repeat(40));
        let snippet = match_snippet(&turkish, "needle", 3).expect("found");
        assert_eq!(snippet, "...İİİNeedle ta...");

        assert_eq!(
            match_snippet("short", "short", 50).as_deref(),
            Some("short")
        );
        assert!(match_snippet("nothing here", "needle", 50).is_none());
        assert!(match_snippet("anything", "", 50).is_none());
    }
}
