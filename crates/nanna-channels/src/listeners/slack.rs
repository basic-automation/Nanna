//! Slack Socket Mode Listener
//!
//! Connects to Slack's Socket Mode WebSocket API for real-time events.
//! Requires an app token (xapp-*) in addition to the bot token.

use super::circuit_breaker::{BreakerAction, CircuitBreaker};
use super::{Listener, ListenerError, ListenerHandle};
use crate::status::StatusManager;
use crate::{ChannelId, IncomingMessage, MessageContent, Sender};
use async_trait::async_trait;
use futures_util::stream::SplitSink;
use futures_util::{SinkExt, StreamExt};
use reqwest::Client;
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, RwLock};
use tokio::net::TcpStream;
use tokio_tungstenite::{
    connect_async,
    tungstenite::Message as WsMessage,
    MaybeTlsStream, WebSocketStream,
};
use tracing::{debug, error, info, warn};

const SLACK_API_BASE: &str = "https://slack.com/api";

/// How long the Socket Mode connection may be silent before it is pinged,
/// and then again before it is given up as dead. Slack pings an open
/// connection far more often than this, so a healthy one never trips it; a
/// dead one is found within two intervals.
const SLACK_IDLE_PROBE: std::time::Duration = std::time::Duration::from_secs(60);

/// Slack Socket Mode listener
pub struct SlackListener {
    /// App-level token (xapp-*)
    app_token: String,
    /// Bot user token (xoxb-*) - for API calls
    bot_token: String,
    /// HTTP client for API calls
    client: Client,
    /// Bot's own user ID (to ignore own messages)
    self_id: Arc<RwLock<Option<String>>>,
    /// Allowed channel IDs (empty = allow all)
    allowed_channels: Vec<String>,
    /// Optional status manager for reporting connection state to the UI
    status_manager: Option<Arc<StatusManager>>,
}

impl SlackListener {
    /// Create a new Slack Socket Mode listener
    ///
    /// - `app_token`: App-level token starting with `xapp-`
    /// - `bot_token`: Bot user token starting with `xoxb-`
    pub fn new(app_token: impl Into<String>, bot_token: impl Into<String>) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap_or_else(|_| Client::new());

        Self {
            app_token: app_token.into(),
            bot_token: bot_token.into(),
            client,
            self_id: Arc::new(RwLock::new(None)),
            allowed_channels: Vec::new(),
            status_manager: None,
        }
    }

    /// Only process messages from specific channels
    #[must_use]
    pub fn with_allowed_channels(mut self, channels: Vec<String>) -> Self {
        self.allowed_channels = channels;
        self
    }

    /// Attach a status manager for reporting connection state to the UI
    #[must_use]
    pub fn with_status_manager(mut self, manager: Arc<StatusManager>) -> Self {
        self.status_manager = Some(manager);
        self
    }

    /// Get a WebSocket URL from Slack's apps.connections.open API
    async fn get_ws_url(&self) -> Result<String, ListenerError> {
        #[derive(Deserialize)]
        struct ConnectionsResponse {
            ok: bool,
            url: Option<String>,
            error: Option<String>,
        }

        let response = self
            .client
            .post(format!("{SLACK_API_BASE}/apps.connections.open"))
            .header("Authorization", format!("Bearer {}", self.app_token))
            .header("Content-Type", "application/x-www-form-urlencoded")
            .send()
            .await
            .map_err(|e| ListenerError::Connection(e.to_string()))?;

        let body: ConnectionsResponse = response
            .json()
            .await
            .map_err(|e| ListenerError::Api(format!("Failed to parse response: {e}")))?;

        if !body.ok {
            let err = body.error.unwrap_or_else(|| "Unknown error".to_string());
            // Slack returns specific error codes for auth issues
            if err == "invalid_auth" || err == "not_authed" || err == "account_inactive" || err == "token_revoked" {
                return Err(ListenerError::Auth(err));
            }
            return Err(ListenerError::Api(err));
        }

        body.url.ok_or_else(|| ListenerError::Api("No URL in response".to_string()))
    }

    /// Get bot's own user ID
    async fn get_self_id(&self) -> Option<String> {
        #[derive(Deserialize)]
        struct AuthTestResponse {
            ok: bool,
            user_id: Option<String>,
        }

        let response = self
            .client
            .post(format!("{SLACK_API_BASE}/auth.test"))
            .header("Authorization", format!("Bearer {}", self.bot_token))
            .header("Content-Type", "application/x-www-form-urlencoded")
            .send()
            .await
            .ok()?;

        let body: AuthTestResponse = response.json().await.ok()?;
        if body.ok {
            body.user_id
        } else {
            None
        }
    }

    /// Convert Slack event to `IncomingMessage`
    fn convert_event(&self, event: &Value, self_id: Option<&str>) -> Option<IncomingMessage> {
        let event_type = event.get("type")?.as_str()?;

        // Only handle messages
        if event_type != "message" {
            return None;
        }

        // Skip message subtypes (edits, deletes, etc.)
        if event.get("subtype").is_some() {
            return None;
        }

        // Skip messages from self
        let user_id = event.get("user")?.as_str()?;
        if self_id == Some(user_id) {
            return None;
        }

        // Skip bot messages
        if event.get("bot_id").is_some() {
            return None;
        }

        let channel_id = event.get("channel")?.as_str()?;

        // Check if channel is allowed
        if !self.allowed_channels.is_empty()
            && !self.allowed_channels.iter().any(|c| c == channel_id) {
                debug!("Ignoring message from non-allowed channel {}", channel_id);
                return None;
            }

        let text = event.get("text")?.as_str()?.to_string();
        if text.is_empty() {
            return None;
        }

        let ts = event.get("ts")?.as_str()?;
        let timestamp = ts.split('.').next()?.parse::<i64>().ok()?;

        let thread_ts = event.get("thread_ts").and_then(|v| v.as_str()).map(String::from);

        Some(IncomingMessage {
            id: format!("{channel_id}:{ts}"),
            channel: ChannelId::new("slack", channel_id.to_string()),
            sender: Sender {
                id: user_id.to_string(),
                name: None, // Would need users.info call
                username: None,
            },
            content: MessageContent::Text { text },
            timestamp,
            reply_to: thread_ts.map(|t| format!("{channel_id}:{t}")),
        })
    }

    /// Run the Socket Mode loop
    async fn socket_loop(
        self: Arc<Self>,
        sender: mpsc::Sender<IncomingMessage>,
        mut shutdown_rx: mpsc::Receiver<()>,
    ) {
        info!("Slack Socket Mode listener starting");

        // Get bot's user ID
        if let Some(id) = self.get_self_id().await {
            *self.self_id.write().await = Some(id.clone());
            debug!("Slack bot user ID: {}", id);
        }

        let mut cb = CircuitBreaker::new("slack");
        if let Some(sm) = &self.status_manager {
            cb = cb.with_status_manager(Arc::clone(sm));
        }

        loop {
            // Check for shutdown
            if shutdown_rx.try_recv().is_ok() {
                info!("Slack Socket Mode listener received shutdown signal");
                break;
            }

            cb.report_connecting().await;

            // Get WebSocket URL
            let ws_url = match self.get_ws_url().await {
                Ok(url) => url,
                Err(ListenerError::Auth(ref e)) => {
                    if cb.record_auth_failure(e).await == BreakerAction::Stop {
                        break;
                    }
                    cb.backoff().await;
                    continue;
                }
                Err(e) => {
                    let detail = e.to_string();
                    if cb.record_conn_failure(&detail).await == BreakerAction::Stop {
                        break;
                    }
                    cb.backoff().await;
                    continue;
                }
            };

            info!("Connecting to Slack Socket Mode");

            let (ws_stream, _): (WebSocketStream<MaybeTlsStream<TcpStream>>, _) = match connect_async(&ws_url).await {
                Ok(conn) => conn,
                Err(e) => {
                    let detail = format!("WebSocket connect failed: {e}");
                    if cb.record_conn_failure(&detail).await == BreakerAction::Stop {
                        break;
                    }
                    cb.backoff().await;
                    continue;
                }
            };

            cb.record_success().await;
            let (mut write, mut read) = ws_stream.split();

            info!("Slack Socket Mode connected");

            // Whether the socket went quiet for a whole probe interval and was
            // pinged; quiet for another one, it is dead. Without this a
            // half-open socket (sleep/resume, a NAT timeout) was awaited
            // forever while the channel still reported `Connected`.
            let mut probed = false;
            'connection: loop {
                tokio::select! {
                    _ = shutdown_rx.recv() => {
                        info!("Slack Socket Mode listener shutting down");
                        let _ = write.close().await;
                        break 'connection;
                    }
                    next = tokio::time::timeout(SLACK_IDLE_PROBE, read.next()) => {
                        let Ok(msg) = next else {
                            if probed {
                                warn!("Slack Socket Mode went silent, even to a ping; reconnecting");
                                break 'connection;
                            }
                            probed = true;
                            if write.send(WsMessage::Ping(Vec::new().into())).await.is_err() {
                                break 'connection;
                            }
                            continue;
                        };
                        probed = false;
                        let Some(msg) = msg else {
                            warn!("Slack Socket Mode connection closed");
                            break 'connection;
                        };

                        if self.on_frame(msg, &mut write, &sender).await == EnvelopeFlow::Reconnect {
                            break 'connection;
                        }
                    }
                }
            }

            // Backoff before reconnecting (no-op if counters are zero)
            cb.backoff().await;
        }

        info!("Slack Socket Mode listener stopped");
    }

    /// Act on one frame read from the Socket Mode connection.
    async fn on_frame(
        &self,
        msg: Result<WsMessage, tokio_tungstenite::tungstenite::Error>,
        write: &mut SocketSink,
        sender: &mpsc::Sender<IncomingMessage>,
    ) -> EnvelopeFlow {
        let text = match msg {
            Ok(WsMessage::Text(text)) => text,
            Ok(WsMessage::Close(_)) => {
                info!("Slack sent close frame");
                return EnvelopeFlow::Reconnect;
            }
            Ok(WsMessage::Ping(data)) => {
                let _ = write.send(WsMessage::Pong(data)).await;
                return EnvelopeFlow::Continue;
            }
            Ok(_) => return EnvelopeFlow::Continue,
            Err(e) => {
                warn!("Slack Socket Mode error: {}", e);
                return EnvelopeFlow::Reconnect;
            }
        };
        match serde_json::from_str::<SocketModePayload>(&text) {
            Ok(payload) => self.handle_envelope(&payload, write, sender).await,
            Err(e) => {
                warn!("Failed to parse Socket Mode payload: {}", e);
                EnvelopeFlow::Continue
            }
        }
    }

    /// Act on one decoded Socket Mode envelope: acknowledge it and forward any
    /// user message to the router.
    async fn handle_envelope(
        &self,
        payload: &SocketModePayload,
        write: &mut SocketSink,
        sender: &mpsc::Sender<IncomingMessage>,
    ) -> EnvelopeFlow {
        // Handle different envelope types
        match payload.envelope_type.as_str() {
            "hello" => {
                debug!("Slack Socket Mode hello received");
            }
            "disconnect" => {
                info!("Slack requested disconnect");
                return EnvelopeFlow::Reconnect;
            }
            "events_api" => {
                // Acknowledge the event immediately
                if let Some(envelope_id) = &payload.envelope_id {
                    let ack = json!({ "envelope_id": envelope_id });
                    if write.send(WsMessage::Text(ack.to_string().into())).await.is_err() {
                        warn!("Failed to send event acknowledgment");
                        return EnvelopeFlow::Reconnect;
                    }
                }

                // Process the event
                if let Some(event_payload) = &payload.payload
                    && let Some(event) = event_payload.get("event") {
                        let self_id = self.self_id.read().await.clone();
                        if let Some(message) = self.convert_event(event, self_id.as_deref()) {
                            debug!("Slack message: {:?}", message.id);
                            if sender.send(message).await.is_err() {
                                error!("Failed to send message to router");
                                return EnvelopeFlow::Reconnect;
                            }
                        }
                    }
            }
            "interactive" => {
                // Acknowledge interactive payloads
                if let Some(envelope_id) = &payload.envelope_id {
                    let ack = json!({ "envelope_id": envelope_id });
                    let _ = write.send(WsMessage::Text(ack.to_string().into())).await;
                }
                debug!("Slack interactive event received");
            }
            "slash_commands" => {
                // Acknowledge slash commands
                if let Some(envelope_id) = &payload.envelope_id {
                    let ack = json!({ "envelope_id": envelope_id });
                    let _ = write.send(WsMessage::Text(ack.to_string().into())).await;
                }
                debug!("Slack slash command received");
            }
            _ => {
                debug!("Unknown Slack envelope type: {}", payload.envelope_type);
            }
        }
        EnvelopeFlow::Continue
    }
}

/// Write half of the Socket Mode WebSocket.
type SocketSink = SplitSink<WebSocketStream<MaybeTlsStream<TcpStream>>, WsMessage>;

/// What the connection loop does after handling one Socket Mode envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EnvelopeFlow {
    /// Keep reading this connection.
    Continue,
    /// Leave this connection; the outer loop backs off and reconnects.
    Reconnect,
}

#[async_trait]
impl Listener for SlackListener {
    fn provider(&self) -> &'static str {
        "slack"
    }

    async fn start(
        self: Arc<Self>,
        sender: mpsc::Sender<IncomingMessage>,
    ) -> Result<ListenerHandle, ListenerError> {
        let (shutdown_tx, shutdown_rx) = mpsc::channel(1);

        let listener = Arc::clone(&self);
        let handle = tokio::spawn(async move {
            listener.socket_loop(sender, shutdown_rx).await;
        });

        Ok(ListenerHandle::new(shutdown_tx, handle))
    }
}

/// Slack Socket Mode envelope
#[derive(Debug, Deserialize)]
struct SocketModePayload {
    #[serde(rename = "type")]
    envelope_type: String,
    envelope_id: Option<String>,
    payload: Option<Value>,
    #[serde(rename = "accepts_response_payload")]
    _accepts_response_payload: Option<bool>,
    #[serde(rename = "retry_attempt")]
    _retry_attempt: Option<u32>,
    #[serde(rename = "retry_reason")]
    _retry_reason: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let listener = SlackListener::new("xapp-test", "xoxb-test");
        assert_eq!(listener.app_token, "xapp-test");
    }
}
