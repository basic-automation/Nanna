//! HTTP route definitions

use crate::state::AppState;
use crate::webhooks;
use axum::{
    Json, Router,
    extract::{Path, Request, State},
    http::{HeaderMap, StatusCode, header},
    middleware::{self, Next},
    response::Response,
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Create the main router
pub fn create_router(state: AppState) -> Router {
    // The API drives the agent, tools included: only a local native caller.
    let api = Router::new()
        .route("/api/v1/chat", post(chat))
        .route("/api/v1/sessions", post(create_session))
        .route("/api/v1/sessions/{session_id}", get(get_session))
        .route("/api/v1/sessions/{session_id}/messages", post(send_message))
        .route("/api/v1/sessions/{session_id}/messages", get(get_messages))
        .route_layer(middleware::from_fn(local_callers_only));
    Router::new()
        // Health check and metrics
        .route("/health", get(health_check))
        .route("/metrics", get(metrics))
        .merge(api)
        // Webhook routes — reached through a tunnel under the provider's
        // host, and authenticated by their signatures instead.
        .route("/webhooks/telegram", post(webhooks::telegram::handle))
        .route("/webhooks/discord", post(webhooks::discord::handle))
        .route("/webhooks/slack", post(webhooks::slack::handle))
        .route("/webhooks/signal", post(webhooks::signal::handle))
        .route("/webhooks/signal/health", get(webhooks::signal::health))
        .route("/webhooks/generic", post(webhooks::generic::handle))
        .with_state(state)
}

/// Refuse an API request a web page could have made.
///
/// A browser marks a cross-site request with `Origin`; a DNS-rebinding page is
/// same-origin, so a `GET` from it carries none — but its `Host` is the
/// attacker's name, never a loopback one. Native clients send no `Origin` and
/// address the server as `127.0.0.1`, `localhost` or `[::1]`.
async fn local_callers_only(request: Request, next: Next) -> Result<Response, StatusCode> {
    if is_local_native_request(request.headers()) {
        return Ok(next.run(request).await);
    }
    tracing::warn!(
        host = ?request.headers().get(header::HOST),
        origin = ?request.headers().get(header::ORIGIN),
        "refused an API request from a web page or a foreign host"
    );
    Err(StatusCode::FORBIDDEN)
}

/// Whether `headers` describe a native caller on this machine.
fn is_local_native_request(headers: &HeaderMap) -> bool {
    if headers.contains_key(header::ORIGIN) {
        return false;
    }
    let Some(host) = headers.get(header::HOST).and_then(|h| h.to_str().ok()) else {
        // HTTP/1.1 requires Host; a request without one is not a browser's.
        return true;
    };
    let name = host.strip_prefix('[').map_or_else(
        || host.rsplit_once(':').map_or(host, |(name, _port)| name),
        |bracketed| {
            bracketed
                .split_once(']')
                .map_or(bracketed, |(name, _)| name)
        },
    );
    matches!(name, "127.0.0.1" | "localhost" | "::1")
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
    use super::{HeaderMap, MessageResponse, is_local_native_request};
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

    fn headers(pairs: &[(&'static str, &'static str)]) -> HeaderMap {
        let mut map = HeaderMap::new();
        for (name, value) in pairs {
            map.insert(*name, value.parse().expect("header value"));
        }
        map
    }

    /// A web page is refused whether it is cross-site (it sends `Origin`) or
    /// a DNS-rebinding page posing as same-origin (its `Host` is not loopback);
    /// a native client on this machine is let through.
    #[test]
    fn only_a_local_native_caller_reaches_the_api() {
        for local in [
            "127.0.0.1:3000",
            "localhost:3000",
            "[::1]:3000",
            "localhost",
        ] {
            assert!(
                is_local_native_request(&headers(&[("host", local)])),
                "{local}"
            );
        }
        assert!(
            is_local_native_request(&HeaderMap::new()),
            "no Host: not a browser"
        );
        for foreign in ["evil.example:3000", "127.0.0.1.evil.example", "[::2]:3000"] {
            assert!(
                !is_local_native_request(&headers(&[("host", foreign)])),
                "{foreign}"
            );
        }
        assert!(!is_local_native_request(&headers(&[
            ("host", "127.0.0.1:3000"),
            ("origin", "https://example.com"),
        ])));
    }
}
