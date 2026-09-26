//! `WhatsApp` Web Bridge Listener
//!
//! Connects to a `WhatsApp` Web bridge server for personal/regular accounts.
//! 
//! Supports multiple bridge backends:
//! - whatsapp-web.js based servers (wwebjs, baileys-api, etc.)
//! - whatsmeow-based servers
//!
//! The bridge must expose:
//! - GET /status - connection status
//! - GET /qr - QR code for linking (if not authenticated)
//! - WebSocket or SSE endpoint for receiving messages
//! - POST /send - for sending messages (handled by `WhatsAppWebChannel`)

use super::circuit_breaker::{BreakerAction, CircuitBreaker};
use super::{Listener, ListenerError, ListenerHandle};
use crate::status::StatusManager;
use crate::{ChannelId, IncomingMessage, MessageContent, Sender};
use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::Client;
use serde::Deserialize;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

/// `WhatsApp` Web bridge listener
pub struct WhatsAppWebListener {
    client: Client,
    /// Base URL of the bridge server (e.g., "<http://localhost:3000>")
    api_url: String,
    /// Session/instance ID (some bridges support multiple sessions)
    session_id: String,
    /// Receive mode
    receive_mode: ReceiveMode,
    /// Optional status manager for reporting connection state to the UI
    status_manager: Option<Arc<StatusManager>>,
}

/// How to receive messages from the bridge
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReceiveMode {
    /// WebSocket connection (most bridges support this)
    #[default]
    WebSocket,
    /// Server-Sent Events
    Sse,
    /// Long-polling
    Poll,
}

/// Deadline for one poll request (poll mode). Generous next to a bridge's
/// normal sub-second answer; it exists so a hung bridge cannot stall polling.
const POLL_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

impl WhatsAppWebListener {
    /// Create a new `WhatsApp` Web listener
    ///
    /// # Arguments
    /// * `api_url` - Base URL of the bridge server
    /// * `session_id` - Session identifier (use "default" if bridge doesn't support multi-session)
    pub fn new(api_url: impl Into<String>, session_id: impl Into<String>) -> Self {
        let client = super::sse::long_lived_client();

        Self {
            client,
            api_url: api_url.into().trim_end_matches('/').to_string(),
            session_id: session_id.into(),
            receive_mode: ReceiveMode::default(),
            status_manager: None,
        }
    }

    /// Set receive mode
    #[must_use]
    pub const fn with_receive_mode(mut self, mode: ReceiveMode) -> Self {
        self.receive_mode = mode;
        self
    }

    /// Attach a status manager for reporting connection state to the UI
    #[must_use]
    pub fn with_status_manager(mut self, manager: Arc<StatusManager>) -> Self {
        self.status_manager = Some(manager);
        self
    }

    /// Build channel ID
    fn channel_id(&self) -> ChannelId {
        ChannelId::new("whatsapp-web", &self.session_id)
    }

    /// Check bridge status
    async fn check_status(&self) -> Result<BridgeStatus, ListenerError> {
        let url = format!("{}/api/sessions/{}/status", self.api_url, self.session_id);
        
        let response = self
            .client
            .get(&url)
            .timeout(Duration::from_secs(10))
            .send()
            .await
            .map_err(|e| ListenerError::Connection(format!("Cannot reach bridge: {e}")))?;

        if !response.status().is_success() {
            // Try alternative endpoint format
            let alt_url = format!("{}/status", self.api_url);
            let alt_response = self
                .client
                .get(&alt_url)
                .timeout(Duration::from_secs(10))
                .send()
                .await
                .map_err(|e| ListenerError::Connection(format!("Cannot reach bridge: {e}")))?;

            if !alt_response.status().is_success() {
                return Err(ListenerError::Connection(format!(
                    "Bridge returned {}",
                    response.status()
                )));
            }

            return alt_response
                .json()
                .await
                .map_err(|e| ListenerError::Api(format!("Invalid status response: {e}")));
        }

        response
            .json()
            .await
            .map_err(|e| ListenerError::Api(format!("Invalid status response: {e}")))
    }

    /// Parse incoming message from bridge
    fn parse_message(&self, msg: BridgeMessage) -> Option<IncomingMessage> {
        // Skip non-text messages for now (can extend later)
        let body = msg.body.or(msg.text)?;
        
        if body.trim().is_empty() {
            return None;
        }

        // Parse sender info
        let sender_id = msg.from.clone().or_else(|| msg.sender.clone())?;
        let sender_name = msg.sender_name.or(msg.push_name);
        
        // Check if it's a group message
        let is_group = msg.is_group.unwrap_or(false) || sender_id.contains("@g.us");
        
        // For groups, the chat ID is different from sender
        let _chat_id = if is_group {
            msg.chat_id.clone().unwrap_or_else(|| sender_id.clone())
        } else {
            sender_id.clone()
        };

        Some(IncomingMessage {
            id: msg.id.unwrap_or_else(|| msg.timestamp.unwrap_or(0).to_string()),
            channel: self.channel_id(),
            sender: Sender {
                id: sender_id.clone(),
                name: sender_name,
                username: Some(sender_id),
            },
            content: MessageContent::Text { text: body },
            timestamp: msg.timestamp.unwrap_or_else(|| chrono::Utc::now().timestamp()),
            reply_to: msg.quoted_msg_id,
        })
    }

    /// Create a circuit breaker pre-configured for this listener.
    fn make_circuit_breaker(&self) -> CircuitBreaker {
        let mut cb = CircuitBreaker::new("whatsapp-web");
        if let Some(sm) = &self.status_manager {
            cb = cb.with_status_manager(Arc::clone(sm));
        }
        cb
    }

    /// Run WebSocket receiver
    async fn run_websocket_receiver(
        self: Arc<Self>,
        sender: mpsc::Sender<IncomingMessage>,
        mut shutdown_rx: mpsc::Receiver<()>,
    ) {
        use tokio_tungstenite::connect_async;
        
        // Build WebSocket URL
        let ws_url = self.api_url
            .replace("http://", "ws://")
            .replace("https://", "wss://");
        let url = format!("{}/ws/{}", ws_url, self.session_id);

        let mut cb = self.make_circuit_breaker();

        loop {
            cb.report_connecting().await;
            info!(url = %url, "Connecting to WhatsApp Web bridge WebSocket");

            let ws_stream = match connect_async(&url).await {
                Ok((stream, _)) => stream,
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
            info!("Connected to WhatsApp Web bridge");

            let (_, mut read) = ws_stream.split();

            loop {
                tokio::select! {
                    _ = shutdown_rx.recv() => {
                        info!("WhatsApp WebSocket receiver shutting down");
                        return;
                    }
                    msg = read.next() => {
                        match msg {
                            Some(Ok(tokio_tungstenite::tungstenite::Message::Text(text))) => {
                                // Parse the message
                                if let Ok(bridge_msg) = serde_json::from_str::<BridgeEvent>(&text)
                                    && let Some(message) = bridge_msg.message
                                        && let Some(incoming) = self.parse_message(message) {
                                            debug!(msg_id = %incoming.id, "Received WhatsApp message");
                                            if sender.send(incoming).await.is_err() {
                                                error!("Failed to send message to router");
                                                return;
                                            }
                                        }
                            }
                            Some(Ok(tokio_tungstenite::tungstenite::Message::Ping(data))) => {
                                debug!("Received ping");
                                // Pong is handled automatically by tungstenite
                                let _ = data;
                            }
                            Some(Ok(tokio_tungstenite::tungstenite::Message::Close(_))) => {
                                warn!("WebSocket closed by server");
                                break;
                            }
                            Some(Err(e)) => {
                                warn!(error = %e, "WebSocket error");
                                break;
                            }
                            None => {
                                warn!("WebSocket stream ended");
                                break;
                            }
                            _ => {}
                        }
                    }
                }
            }

            // Connection lost — reconnect with backoff
            let detail = "WebSocket connection lost";
            if cb.record_conn_failure(detail).await == BreakerAction::Stop {
                break;
            }
            cb.backoff().await;
        }
    }

    /// Run SSE receiver
    async fn run_sse_receiver(
        self: Arc<Self>,
        sender: mpsc::Sender<IncomingMessage>,
        mut shutdown_rx: mpsc::Receiver<()>,
    ) {
        let url = format!("{}/api/sessions/{}/events", self.api_url, self.session_id);

        let mut cb = self.make_circuit_breaker();

        loop {
            cb.report_connecting().await;
            info!(url = %url, "Connecting to WhatsApp Web bridge SSE");

            let response = match self
                .client
                .get(&url)
                .header("Accept", "text/event-stream")
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    let detail = format!("SSE connect failed: {e}");
                    if cb.record_conn_failure(&detail).await == BreakerAction::Stop {
                        break;
                    }
                    cb.backoff().await;
                    continue;
                }
            };

            if !response.status().is_success() {
                let status = response.status();
                if status.as_u16() == 401 || status.as_u16() == 403 {
                    if cb.record_auth_failure(&format!("HTTP {status}")).await == BreakerAction::Stop {
                        break;
                    }
                } else {
                    let detail = format!("SSE connection returned HTTP {status}");
                    if cb.record_conn_failure(&detail).await == BreakerAction::Stop {
                        break;
                    }
                }
                cb.backoff().await;
                continue;
            }

            cb.record_success().await;
            info!("Connected to WhatsApp Web bridge SSE");

            let mut stream = response.bytes_stream();
            // Raw bytes, framed before decoding — see `sse::take_event`.
            let mut buffer: Vec<u8> = Vec::new();

            loop {
                tokio::select! {
                    _ = shutdown_rx.recv() => {
                        info!("WhatsApp SSE receiver shutting down");
                        return;
                    }
                    chunk = stream.next() => {
                        match chunk {
                            Some(Ok(bytes)) => {
                                buffer.extend_from_slice(&bytes);
                                while let Some(event) = super::sse::take_event(&mut buffer) {
                                    
                                    // Parse SSE event
                                    for line in event.lines() {
                                        if let Some(data) = line.strip_prefix("data: ")
                                            && let Ok(bridge_event) = serde_json::from_str::<BridgeEvent>(data)
                                                && let Some(message) = bridge_event.message
                                                    && let Some(incoming) = self.parse_message(message) {
                                                        debug!(msg_id = %incoming.id, "Received WhatsApp message via SSE");
                                                        if sender.send(incoming).await.is_err() {
                                                            error!("Failed to send message to router");
                                                            return;
                                                        }
                                                    }
                                    }
                                }
                            }
                            Some(Err(e)) => {
                                warn!(error = %e, "SSE stream error");
                                break;
                            }
                            None => {
                                warn!("SSE stream ended");
                                break;
                            }
                        }
                    }
                }
            }

            // Connection lost — reconnect with backoff
            let detail = "SSE connection lost";
            if cb.record_conn_failure(detail).await == BreakerAction::Stop {
                break;
            }
            cb.backoff().await;
        }
    }

    /// Run polling receiver
    async fn run_poll_receiver(
        self: Arc<Self>,
        sender: mpsc::Sender<IncomingMessage>,
        mut shutdown_rx: mpsc::Receiver<()>,
    ) {
        let url = format!("{}/api/sessions/{}/messages", self.api_url, self.session_id);
        let poll_interval = Duration::from_secs(2);
        let mut last_timestamp: i64 = chrono::Utc::now().timestamp();

        let mut cb = self.make_circuit_breaker();

        loop {
            tokio::select! {
                _ = shutdown_rx.recv() => {
                    info!("WhatsApp poll receiver shutting down");
                    return;
                }
                () = tokio::time::sleep(poll_interval) => {
                    let poll_url = format!("{url}?since={last_timestamp}");

                    // The client has no whole-request timeout (it also serves the
                    // SSE stream), so a poll bounds itself.
                    match self.client.get(&poll_url).timeout(POLL_REQUEST_TIMEOUT).send().await {
                        Ok(response) => {
                            if response.status().is_success() {
                                cb.record_success().await;
                                if let Ok(messages) = response.json::<Vec<BridgeMessage>>().await {
                                    for msg in messages {
                                        if let Some(ts) = msg.timestamp
                                            && ts > last_timestamp {
                                                last_timestamp = ts;
                                            }

                                        if let Some(incoming) = self.parse_message(msg) {
                                            debug!(msg_id = %incoming.id, "Received WhatsApp message");
                                            if sender.send(incoming).await.is_err() {
                                                error!("Failed to send message to router");
                                                return;
                                            }
                                        }
                                    }
                                }
                            } else {
                                let status = response.status();
                                if status.as_u16() == 401 || status.as_u16() == 403 {
                                    if cb.record_auth_failure(&format!("HTTP {status}")).await == BreakerAction::Stop {
                                        break;
                                    }
                                } else {
                                    let detail = format!("Poll returned HTTP {status}");
                                    if cb.record_conn_failure(&detail).await == BreakerAction::Stop {
                                        break;
                                    }
                                }
                                cb.backoff().await;
                            }
                        }
                        Err(e) => {
                            let detail = e.to_string();
                            if cb.record_conn_failure(&detail).await == BreakerAction::Stop {
                                break;
                            }
                            cb.backoff().await;
                        }
                    }
                }
            }
        }
    }
}

#[async_trait]
impl Listener for WhatsAppWebListener {
    fn provider(&self) -> &'static str {
        "whatsapp-web"
    }

    async fn start(
        self: Arc<Self>,
        sender: mpsc::Sender<IncomingMessage>,
    ) -> Result<ListenerHandle, ListenerError> {
        info!(
            api = %self.api_url,
            session = %self.session_id,
            mode = ?self.receive_mode,
            "Starting WhatsApp Web listener"
        );

        // Check bridge status
        let status = self.check_status().await?;
        
        if !status.connected.unwrap_or(false) && !status.authenticated.unwrap_or(false) {
            return Err(ListenerError::Auth(
                "WhatsApp not connected. Scan QR code at bridge server.".to_string()
            ));
        }

        info!("WhatsApp Web bridge is connected");

        let (shutdown_tx, shutdown_rx) = mpsc::channel(1);

        let listener = self.clone();
        let join_handle = match self.receive_mode {
            ReceiveMode::WebSocket => {
                tokio::spawn(async move {
                    listener.run_websocket_receiver(sender, shutdown_rx).await;
                })
            }
            ReceiveMode::Sse => {
                tokio::spawn(async move {
                    listener.run_sse_receiver(sender, shutdown_rx).await;
                })
            }
            ReceiveMode::Poll => {
                tokio::spawn(async move {
                    listener.run_poll_receiver(sender, shutdown_rx).await;
                })
            }
        };

        Ok(ListenerHandle::new(shutdown_tx, join_handle))
    }
}

// =============================================================================
// Bridge Response Types (flexible to support various bridge implementations)
// =============================================================================

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BridgeStatus {
    connected: Option<bool>,
    authenticated: Option<bool>,
    #[serde(alias = "ready")]
    _is_ready: Option<bool>,
    #[serde(alias = "state")]
    _status: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BridgeEvent {
    #[serde(alias = "type", alias = "event")]
    _event_type: Option<String>,
    message: Option<BridgeMessage>,
    #[serde(alias = "msg")]
    _data: Option<BridgeMessage>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BridgeMessage {
    // Message identification
    id: Option<String>,
    #[serde(alias = "messageId", alias = "key")]
    _message_id: Option<String>,
    
    // Sender info (different bridges use different field names)
    from: Option<String>,
    sender: Option<String>,
    #[serde(alias = "senderName", alias = "notifyName")]
    sender_name: Option<String>,
    #[serde(alias = "pushName")]
    push_name: Option<String>,
    
    // Chat info
    #[serde(alias = "chat", alias = "remoteJid")]
    chat_id: Option<String>,
    #[serde(alias = "isGroup", alias = "isGroupMsg")]
    is_group: Option<bool>,
    
    // Message content
    body: Option<String>,
    text: Option<String>,
    #[serde(alias = "content")]
    _message_content: Option<String>,
    
    // Metadata
    timestamp: Option<i64>,
    #[serde(alias = "t", alias = "messageTimestamp")]
    _unix_timestamp: Option<i64>,
    
    // Reply context
    #[serde(alias = "quotedMsgId", alias = "quotedStanzaID")]
    quoted_msg_id: Option<String>,
    
    // Message type
    #[serde(alias = "type", alias = "messageType")]
    _message_type: Option<String>,
}
