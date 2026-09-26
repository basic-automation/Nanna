//! Signal Listener
//!
//! Connects to signald (or signal-cli REST API) and pushes incoming messages
//! to the message router.
//!
//! Supports two backends:
//! - signald: Unix socket / TCP connection (default)
//! - signal-cli-rest-api: HTTP REST API (simpler setup)

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

/// Signal listener using signal-cli-rest-api
/// 
/// This uses the simpler HTTP REST API approach with SSE for receiving messages.
/// See: <https://github.com/bbernhard/signal-cli-rest-api>
pub struct SignalListener {
    client: Client,
    /// Base URL of the signal-cli-rest-api server
    api_url: String,
    /// Phone number of the registered account (e.g., "+1234567890")
    phone_number: String,
    /// Receive mode: "websocket" or "poll"
    receive_mode: ReceiveMode,
    /// Optional status manager for reporting connection state to the UI
    status_manager: Option<Arc<StatusManager>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
pub enum ReceiveMode {
    /// Use Server-Sent Events (recommended)
    #[default]
    Sse,
    /// Poll /v1/receive endpoint
    Poll,
}


impl SignalListener {
    /// Create a new Signal listener
    /// 
    /// # Arguments
    /// * `api_url` - Base URL of signal-cli-rest-api (e.g., "<http://localhost:8080>")
    /// * `phone_number` - Phone number of the registered account (e.g., "+1234567890")
    pub fn new(api_url: impl Into<String>, phone_number: impl Into<String>) -> Self {
        let client = super::sse::long_lived_client();

        Self {
            client,
            api_url: api_url.into().trim_end_matches('/').to_string(),
            phone_number: phone_number.into(),
            receive_mode: ReceiveMode::default(),
            status_manager: None,
        }
    }

    /// Set the receive mode
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

    /// Build the channel ID
    fn channel_id(&self) -> ChannelId {
        ChannelId::new("signal", &self.phone_number)
    }

    /// Poll for messages (fallback mode)
    async fn poll_messages(&self) -> Result<Vec<SignalEnvelope>, ListenerError> {
        let url = format!(
            "{}/v1/receive/{}",
            self.api_url,
            urlencoding::encode(&self.phone_number)
        );

        let response = self
            .client
            .get(&url)
            .timeout(Duration::from_secs(30))
            .send()
            .await
            .map_err(|e| ListenerError::Connection(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            if status.as_u16() == 401 || status.as_u16() == 403 {
                return Err(ListenerError::Auth(format!("HTTP {status}")));
            }
            let text = response.text().await.unwrap_or_default();
            return Err(ListenerError::Api(format!("HTTP {status}: {text}")));
        }

        let envelopes: Vec<SignalEnvelope> = response
            .json()
            .await
            .map_err(|e| ListenerError::Api(format!("Failed to parse response: {e}")))?;

        Ok(envelopes)
    }

    /// Parse a `SignalEnvelope` into `IncomingMessage`
    fn parse_envelope(&self, envelope: SignalEnvelope) -> Option<IncomingMessage> {
        // Skip messages without data
        let data_message = envelope.envelope.data_message?;
        let body = data_message.message?;
        
        // Skip empty messages
        if body.trim().is_empty() {
            return None;
        }

        let source = envelope.envelope.source_number
            .or_else(|| envelope.envelope.source_uuid.clone())?;

        let sender_name = envelope.envelope.source_name.clone();
        
        // Determine if this is a group message
        let (_chat_id, group_name) = data_message.group_info.as_ref().map_or_else(
            || (source.clone(), None),
            |group| (group.group_id.clone(), group.group_name.clone()),
        );

        Some(IncomingMessage {
            id: envelope.envelope.timestamp.to_string(),
            channel: self.channel_id(),
            sender: Sender {
                id: source.clone(),
                name: sender_name.or(group_name),
                username: Some(source),
            },
            content: MessageContent::Text { text: body },
            timestamp: envelope.envelope.timestamp,
            reply_to: data_message.quote.map(|q| q.id.to_string()),
        })
    }

    /// Create a circuit breaker pre-configured for this listener.
    fn make_circuit_breaker(&self) -> CircuitBreaker {
        let mut cb = CircuitBreaker::new("signal");
        if let Some(sm) = &self.status_manager {
            cb = cb.with_status_manager(Arc::clone(sm));
        }
        cb
    }

    /// Run the poll-based receiver
    async fn run_poll_receiver(
        self: Arc<Self>,
        sender: mpsc::Sender<IncomingMessage>,
        mut shutdown_rx: mpsc::Receiver<()>,
    ) {
        let mut cb = self.make_circuit_breaker();

        loop {
            tokio::select! {
                _ = shutdown_rx.recv() => {
                    info!("Signal poll receiver shutting down");
                    break;
                }
                result = self.poll_messages() => {
                    match result {
                        Ok(envelopes) => {
                            cb.record_success().await;
                            for envelope in envelopes {
                                if let Some(msg) = self.parse_envelope(envelope) {
                                    debug!(msg_id = %msg.id, "Received Signal message");
                                    if sender.send(msg).await.is_err() {
                                        error!("Failed to send message to router");
                                        return;
                                    }
                                }
                            }
                            // Small delay between polls even on success
                            tokio::time::sleep(Duration::from_secs(1)).await;
                        }
                        Err(ListenerError::Auth(ref e)) => {
                            if cb.record_auth_failure(e).await == BreakerAction::Stop {
                                break;
                            }
                            cb.backoff().await;
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

    /// Run the SSE-based receiver
    async fn run_sse_receiver(
        self: Arc<Self>,
        sender: mpsc::Sender<IncomingMessage>,
        mut shutdown_rx: mpsc::Receiver<()>,
    ) {
        let url = format!(
            "{}/v1/receive/{}/sse",
            self.api_url,
            urlencoding::encode(&self.phone_number)
        );

        let mut cb = self.make_circuit_breaker();

        loop {
            cb.report_connecting().await;
            info!(url = %url, "Connecting to Signal SSE stream");

            // Create SSE connection
            let response = match self.client
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
            info!("Connected to Signal SSE stream");

            // Read SSE stream
            let mut stream = response.bytes_stream();
            // Raw bytes, framed before decoding — see `sse::take_event`.
            let mut buffer: Vec<u8> = Vec::new();

            loop {
                tokio::select! {
                    _ = shutdown_rx.recv() => {
                        info!("Signal SSE receiver shutting down");
                        return;
                    }
                    chunk = stream.next() => {
                        match chunk {
                            Some(Ok(bytes)) => {
                                buffer.extend_from_slice(&bytes);
                                while let Some(event) = super::sse::take_event(&mut buffer) {
                                    
                                    // Parse SSE event
                                    if let Some(data) = event.strip_prefix("data: ")
                                        && let Ok(envelope) = serde_json::from_str::<SignalEnvelope>(data.trim())
                                            && let Some(msg) = self.parse_envelope(envelope) {
                                                debug!(msg_id = %msg.id, "Received Signal message via SSE");
                                                if sender.send(msg).await.is_err() {
                                                    error!("Failed to send message to router");
                                                    return;
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

            // SSE stream lost — reconnect with backoff
            let detail = "SSE connection lost";
            if cb.record_conn_failure(detail).await == BreakerAction::Stop {
                break;
            }
            cb.backoff().await;
        }
    }
}

#[async_trait]
impl Listener for SignalListener {
    fn provider(&self) -> &'static str {
        "signal"
    }

    async fn start(
        self: Arc<Self>,
        sender: mpsc::Sender<IncomingMessage>,
    ) -> Result<ListenerHandle, ListenerError> {
        info!(
            phone = %self.phone_number,
            api = %self.api_url,
            mode = ?self.receive_mode,
            "Starting Signal listener"
        );

        // Test connection
        let test_url = format!("{}/v1/about", self.api_url);
        let response = self
            .client
            .get(&test_url)
            .timeout(Duration::from_secs(5))
            .send()
            .await
            .map_err(|e| ListenerError::Connection(format!("Cannot reach signal-cli-rest-api: {e}")))?;

        if !response.status().is_success() {
            return Err(ListenerError::Connection(format!(
                "signal-cli-rest-api returned {}",
                response.status()
            )));
        }

        info!("Connected to signal-cli-rest-api");

        let (shutdown_tx, shutdown_rx) = mpsc::channel(1);

        let listener = self.clone();
        let join_handle = match self.receive_mode {
            ReceiveMode::Poll => {
                tokio::spawn(async move {
                    listener.run_poll_receiver(sender, shutdown_rx).await;
                })
            }
            ReceiveMode::Sse => {
                tokio::spawn(async move {
                    listener.run_sse_receiver(sender, shutdown_rx).await;
                })
            }
        };

        Ok(ListenerHandle::new(shutdown_tx, join_handle))
    }
}

// =============================================================================
// Signal API Response Types
// =============================================================================

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SignalEnvelope {
    envelope: SignalEnvelopeInner,
    #[serde(default, rename = "account")]
    _account: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SignalEnvelopeInner {
    source_number: Option<String>,
    source_uuid: Option<String>,
    source_name: Option<String>,
    timestamp: i64,
    #[serde(default)]
    data_message: Option<SignalDataMessage>,
    #[serde(default, rename = "sync_message")]
    _sync_message: Option<serde_json::Value>,
    #[serde(default, rename = "receipt_message")]
    _receipt_message: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SignalDataMessage {
    message: Option<String>,
    #[serde(rename = "timestamp")]
    _timestamp: Option<i64>,
    #[serde(default)]
    group_info: Option<SignalGroupInfo>,
    #[serde(default)]
    quote: Option<SignalQuote>,
    #[serde(default, rename = "attachments")]
    _attachments: Option<Vec<SignalAttachment>>,
    #[serde(default, rename = "reaction")]
    _reaction: Option<SignalReactionInfo>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SignalGroupInfo {
    group_id: String,
    #[serde(rename = "type")]
    _group_type: Option<String>,
    group_name: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SignalQuote {
    id: i64,
    #[serde(rename = "author")]
    _author: Option<String>,
    #[serde(rename = "text")]
    _text: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SignalAttachment {
    #[serde(rename = "content_type")]
    _content_type: Option<String>,
    #[serde(rename = "filename")]
    _filename: Option<String>,
    #[serde(rename = "id")]
    _id: Option<String>,
    #[serde(rename = "size")]
    _size: Option<i64>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SignalReactionInfo {
    #[serde(rename = "emoji")]
    _emoji: String,
    #[serde(rename = "target_author")]
    _target_author: Option<String>,
    #[serde(rename = "target_sent_timestamp")]
    _target_sent_timestamp: Option<i64>,
    #[serde(rename = "is_remove")]
    _is_remove: Option<bool>,
}
