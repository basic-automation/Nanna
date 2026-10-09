//! HTTP route definitions

use crate::state::AppState;
use crate::webhooks;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Create the main router
pub fn create_router(state: AppState) -> Router {
    Router::new()
        // Health check and metrics
        .route("/health", get(health_check))
        .route("/metrics", get(metrics))
        // API routes
        .route("/api/v1/chat", post(chat))
        .route("/api/v1/sessions", post(create_session))
        .route("/api/v1/sessions/{session_id}", get(get_session))
        .route("/api/v1/sessions/{session_id}/messages", post(send_message))
        .route("/api/v1/sessions/{session_id}/messages", get(get_messages))
        // Webhook routes
        .route("/webhooks/telegram", post(webhooks::telegram::handle))
        .route("/webhooks/discord", post(webhooks::discord::handle))
        .route("/webhooks/slack", post(webhooks::slack::handle))
        .route("/webhooks/signal", post(webhooks::signal::handle))
        .route("/webhooks/signal/health", get(webhooks::signal::health))
        .route("/webhooks/generic", post(webhooks::generic::handle))
        .with_state(state)
}

/// Health check response
#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    version: &'static str,
    gpu: bool,
}

async fn health_check(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
        gpu: state.bot.has_gpu(),
    })
}

/// Metrics response
#[derive(Serialize)]
struct MetricsResponse {
    uptime_secs: u64,
    active_agents: usize,
    total_sessions: i64,
    total_messages: i64,
    gpu_available: bool,
    version: &'static str,
}

async fn metrics(State(state): State<AppState>) -> Json<MetricsResponse> {
    // Count active agents
    let active_agents = state.agents.read().await.len();

    // Get storage stats
    let total_sessions = state
        .storage
        .sessions()
        .list_recent(10000)
        .await
        // At most 10,000 sessions were requested, so the count always fits.
        .map_or(0, |s| i64::try_from(s.len()).unwrap_or(i64::MAX));

    Json(MetricsResponse {
        uptime_secs: 0, // Would need start time tracking
        active_agents,
        total_sessions,
        total_messages: 0, // Would need aggregate query
        gpu_available: state.bot.has_gpu(),
        version: env!("CARGO_PKG_VERSION"),
    })
}

/// Chat request
#[derive(Deserialize)]
pub struct ChatRequest {
    pub message: String,
    pub session_id: Option<String>,
    pub system_prompt: Option<String>,
}

/// Chat response
#[derive(Serialize)]
pub struct ChatResponse {
    pub session_id: String,
    pub message: String,
}

/// Simple chat endpoint - creates session if needed
async fn chat(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, (StatusCode, String)> {
    let session_id = req.session_id.unwrap_or_else(|| Uuid::new_v4().to_string());
    
    let response = state
        .process_message(&session_id, &req.message, req.system_prompt.as_deref())
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    Ok(Json(ChatResponse {
        session_id,
        message: response,
    }))
}

/// Create session request
#[derive(Deserialize)]
pub struct CreateSessionRequest {
    pub channel: Option<String>,
    pub user_id: Option<String>,
    pub system_prompt: Option<String>,
}

/// Session response
#[derive(Serialize)]
pub struct SessionResponse {
    pub session_id: String,
    pub channel: String,
    pub user_id: Option<String>,
    pub created_at: String,
}

async fn create_session(
    State(state): State<AppState>,
    Json(req): Json<CreateSessionRequest>,
) -> Result<Json<SessionResponse>, (StatusCode, String)> {
    let session_id = Uuid::new_v4().to_string();
    let channel = req.channel.unwrap_or_else(|| "api".to_string());
    
    // Create in storage
    let session = state
        .storage
        .sessions()
        .create(&session_id, &channel, req.user_id.as_deref())
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // Pre-create agent with system prompt if provided
    if req.system_prompt.is_some() {
        let _ = state.get_or_create_agent(&session_id, req.system_prompt.as_deref()).await;
    }
    
    Ok(Json(SessionResponse {
        session_id: session.session_id,
        channel: session.channel,
        user_id: session.user_id,
        created_at: session.created_at,
    }))
}

async fn get_session(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
) -> Result<Json<SessionResponse>, (StatusCode, String)> {
    let session = state
        .storage
        .sessions()
        .get(&session_id)
        .await
        .map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))?;
    
    Ok(Json(SessionResponse {
        session_id: session.session_id,
        channel: session.channel,
        user_id: session.user_id,
        created_at: session.created_at,
    }))
}

/// Send message request
#[derive(Deserialize)]
pub struct SendMessageRequest {
    pub content: String,
    pub system_prompt: Option<String>,
}

/// Message response
#[derive(Serialize)]
pub struct MessageResponse {
    pub id: i64,
    pub role: String,
    pub content: String,
    pub created_at: String,
    pub tokens_in: Option<i64>,
    pub tokens_out: Option<i64>,
}

async fn send_message(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
    Json(req): Json<SendMessageRequest>,
) -> Result<Json<MessageResponse>, (StatusCode, String)> {
    let reply = state
        .process_message_reply(&session_id, &req.content, req.system_prompt.as_deref())
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(MessageResponse::from_reply(reply)))
}

impl MessageResponse {
    /// The reply as the route answers it: the stored row when the turn's write
    /// landed (its id, timestamp and token counts), else the bare text with
    /// id 0 — never another message of the session.
    fn from_reply(reply: crate::ProcessedReply) -> Self {
        let Some(row) = reply.stored else {
            return Self {
                id: 0,
                role: "assistant".to_string(),
                content: reply.text,
                created_at: chrono::Utc::now().to_rfc3339(),
                tokens_in: None,
                tokens_out: None,
            };
        };
        debug_assert_eq!(row.content, reply.text, "the stored row is this reply");
        Self {
            id: row.id,
            role: row.role,
            content: row.content,
            created_at: row.created_at,
            tokens_in: row.tokens_in,
            tokens_out: row.tokens_out,
        }
    }
}

/// A session's newest 100 messages, oldest first.
async fn get_messages(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
) -> Result<Json<Vec<MessageResponse>>, (StatusCode, String)> {
    let messages = state
        .storage
        .messages()
        .get_recent_by_session(&session_id, 100)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    let responses: Vec<MessageResponse> = messages
        .into_iter()
        .map(|msg| MessageResponse {
            id: msg.id,
            role: msg.role,
            content: msg.content,
            created_at: msg.created_at,
            tokens_in: msg.tokens_in,
            tokens_out: msg.tokens_out,
        })
        .collect();
    
    Ok(Json(responses))
}

#[cfg(test)]
mod tests {
    use super::MessageResponse;
    use crate::ProcessedReply;

    fn stored_reply(id: i64, content: &str) -> nanna_storage::Message {
        nanna_storage::Message {
            id,
            session_id: "s".to_string(),
            role: "assistant".to_string(),
            content: content.to_string(),
            content_type: "text".to_string(),
            tool_use_id: None,
            created_at: "2026-10-09 12:00:00".to_string(),
            tokens_in: Some(11),
            tokens_out: Some(7),
            metadata: None,
        }
    }

    #[test]
    fn a_reply_answers_with_the_row_this_turn_stored() {
        let reply = ProcessedReply {
            text: "second answer".to_string(),
            stored: Some(stored_reply(42, "second answer")),
        };
        let response = MessageResponse::from_reply(reply);
        assert_eq!(response.id, 42);
        assert_eq!(response.content, "second answer");
        assert_eq!(response.created_at, "2026-10-09 12:00:00");
        assert_eq!((response.tokens_in, response.tokens_out), (Some(11), Some(7)));
    }

    #[test]
    fn a_reply_whose_write_failed_still_carries_the_text_and_no_id() {
        let reply = ProcessedReply {
            text: "unsaved".to_string(),
            stored: None,
        };
        let response = MessageResponse::from_reply(reply);
        assert_eq!(response.id, 0);
        assert_eq!(response.role, "assistant");
        assert_eq!(response.content, "unsaved");
        assert_eq!((response.tokens_in, response.tokens_out), (None, None));
    }
}
