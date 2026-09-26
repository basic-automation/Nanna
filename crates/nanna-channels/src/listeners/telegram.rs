//! Telegram Long-Polling Listener
//!
//! Connects to Telegram's getUpdates API and pushes incoming messages
//! to the message router.

use super::circuit_breaker::{BreakerAction, CircuitBreaker};
use super::{Listener, ListenerError, ListenerHandle};
use crate::status::StatusManager;
use crate::{ChannelId, IncomingMessage, MessageContent, Sender};
use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{debug, error, info};

const TELEGRAM_API_BASE: &str = "https://api.telegram.org";
const LONG_POLL_TIMEOUT: u64 = 30;

/// Telegram long-polling listener
pub struct TelegramListener {
    client: Client,
    bot_token: String,
    /// Last processed update ID
    offset: AtomicI64,
    /// Allowed chat IDs (empty = allow all)
    allowed_chats: Vec<i64>,
    /// Optional status manager for reporting connection state to the UI
    status_manager: Option<Arc<StatusManager>>,
}

impl TelegramListener {
    /// Create a new Telegram listener
    pub fn new(bot_token: impl Into<String>) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(LONG_POLL_TIMEOUT + 10))
            .build()
            .unwrap_or_else(|_| Client::new());

        Self {
            client,
            bot_token: bot_token.into(),
            offset: AtomicI64::new(0),
            allowed_chats: Vec::new(),
            status_manager: None,
        }
    }

    /// Only process messages from specific chats
    #[must_use]
    pub fn with_allowed_chats(mut self, chats: Vec<i64>) -> Self {
        self.allowed_chats = chats;
        self
    }

    /// Set the starting offset (for resuming after restart)
    #[must_use]
    pub fn with_offset(self, offset: i64) -> Self {
        self.offset.store(offset, Ordering::SeqCst);
        self
    }

    /// Attach a status manager for reporting connection state to the UI
    #[must_use]
    pub fn with_status_manager(mut self, manager: Arc<StatusManager>) -> Self {
        self.status_manager = Some(manager);
        self
    }

    /// Build API URL
    fn api_url(&self, method: &str) -> String {
        format!("{}/bot{}/{}", TELEGRAM_API_BASE, self.bot_token, method)
    }

    /// Fetch updates from Telegram
    async fn get_updates(&self) -> Result<Vec<TelegramUpdate>, ListenerError> {
        let offset = self.offset.load(Ordering::SeqCst);

        let response = self
            .client
            .get(self.api_url("getUpdates"))
            .query(&[
                ("offset", offset.to_string()),
                ("timeout", LONG_POLL_TIMEOUT.to_string()),
                (
                    "allowed_updates",
                    r#"["message","edited_message","stopped_message_generation"]"#.to_string(),
                ),
            ])
            .send()
            .await
            // The URL embeds the bot token; reqwest's Display would print it.
            .map_err(|e| ListenerError::Connection(e.without_url().to_string()))?;

        let status = response.status();
        if status == 401 {
            return Err(ListenerError::Auth("Invalid bot token".to_string()));
        }

        let body: TelegramApiResponse<Vec<TelegramUpdate>> = response
            .json()
            .await
            .map_err(|e| {
                ListenerError::Api(format!("Failed to parse response: {}", e.without_url()))
            })?;

        if !body.ok {
            return Err(ListenerError::Api(
                body.description.unwrap_or_else(|| format!("HTTP {status}")),
            ));
        }

        Ok(body.result.unwrap_or_default())
    }

    /// Convert a Telegram update to an `IncomingMessage`
    fn convert_update(&self, update: &TelegramUpdate) -> Option<IncomingMessage> {
        if let Some(stopped) = &update.stopped_message_generation {
            return self.stop_request(stopped);
        }
        let message = update.message.as_ref().or(update.edited_message.as_ref())?;

        // Check if chat is allowed
        if !self.allowed_chats.is_empty() && !self.allowed_chats.contains(&message.chat.id) {
            debug!(
                "Ignoring message from non-allowed chat {}",
                message.chat.id
            );
            return None;
        }

        let sender = message.from.as_ref()?;
        let content = Self::extract_content(message)?;

        Some(IncomingMessage {
            id: format!("{}:{}", message.chat.id, message.message_id),
            channel: ChannelId::new("telegram", message.chat.id.to_string()),
            sender: Sender {
                id: sender.id.to_string(),
                name: Some(format!(
                    "{}{}",
                    sender.first_name,
                    sender
                        .last_name
                        .as_ref()
                        .map_or_default(|l| format!(" {l}"))
                )),
                username: sender.username.clone(),
            },
            content,
            timestamp: message.date,
            reply_to: message
                .reply_to_message
                .as_ref()
                .map(|r| format!("{}:{}", message.chat.id, r.message_id)),
        })
    }

    /// A press of a draft's stop button, as the `/stop` its user would have
    /// typed — so it cancels the same turn through the same path. Drafts
    /// exist only in private chats, where the chat id IS the user's id, which
    /// makes the synthesized sender the one whose session is running.
    fn stop_request(&self, stopped: &MessageGenerationStopped) -> Option<IncomingMessage> {
        let chat_id = stopped.chat.id;
        if !self.allowed_chats.is_empty() && !self.allowed_chats.contains(&chat_id) {
            return None;
        }
        if chat_id <= 0 {
            debug!("Ignoring a generation stop from non-private chat {chat_id}");
            return None;
        }
        Some(IncomingMessage {
            id: format!("{chat_id}:stop:{}", stopped.draft_id),
            channel: ChannelId::new("telegram", chat_id.to_string()),
            sender: Sender {
                id: chat_id.to_string(),
                name: None,
                username: None,
            },
            content: MessageContent::Text {
                text: "/stop".to_string(),
            },
            timestamp: chrono::Utc::now().timestamp(),
            reply_to: None,
        })
    }

    /// Extract message content
    fn extract_content(message: &TelegramMessage) -> Option<MessageContent> {
        // Text message
        if let Some(text) = &message.text {
            return Some(MessageContent::Text { text: text.clone() });
        }

        // Photo
        if let Some(photos) = &message.photo
            && let Some(largest) = photos.last() {
                return Some(MessageContent::Image {
                    url: largest.file_id.clone(), // Will need to be resolved via getFile
                    caption: message.caption.clone(),
                });
            }

        // Document
        if let Some(doc) = &message.document {
            return Some(MessageContent::Document {
                url: doc.file_id.clone(),
                filename: doc.file_name.clone().unwrap_or_else(|| "document".to_string()),
            });
        }

        // Audio
        if let Some(audio) = &message.audio {
            return Some(MessageContent::Audio {
                url: audio.file_id.clone(),
                duration_secs: Some(duration_secs(audio.duration)),
            });
        }

        // Voice
        if let Some(voice) = &message.voice {
            return Some(MessageContent::Audio {
                url: voice.file_id.clone(),
                duration_secs: Some(duration_secs(voice.duration)),
            });
        }

        // Video
        if let Some(video) = &message.video {
            return Some(MessageContent::Video {
                url: video.file_id.clone(),
                duration_secs: Some(duration_secs(video.duration)),
                caption: message.caption.clone(),
            });
        }

        // Location
        if let Some(loc) = &message.location {
            return Some(MessageContent::Location {
                latitude: loc.latitude,
                longitude: loc.longitude,
            });
        }

        // Sticker
        if let Some(sticker) = &message.sticker {
            return Some(MessageContent::Sticker {
                id: sticker.file_id.clone(),
                emoji: sticker.emoji.clone(),
            });
        }

        // Caption-only (for media without specific handling)
        if let Some(caption) = &message.caption {
            return Some(MessageContent::Text {
                text: caption.clone(),
            });
        }

        None
    }

    /// Run the polling loop
    async fn poll_loop(
        self: Arc<Self>,
        sender: mpsc::Sender<IncomingMessage>,
        mut shutdown_rx: mpsc::Receiver<()>,
    ) {
        info!("Telegram listener started");

        let mut cb = CircuitBreaker::new("telegram");
        if let Some(sm) = &self.status_manager {
            cb = cb.with_status_manager(Arc::clone(sm));
        }

        loop {
            tokio::select! {
                _ = shutdown_rx.recv() => {
                    info!("Telegram listener received shutdown signal");
                    break;
                }
                result = self.get_updates() => {
                    match result {
                        Ok(updates) => {
                            cb.record_success().await;

                            for update in updates {
                                // Update offset to acknowledge this update
                                self.offset.store(update.update_id + 1, Ordering::SeqCst);

                                if let Some(message) = self.convert_update(&update) {
                                    debug!("Received message: {:?}", message.id);

                                    if sender.send(message).await.is_err() {
                                        error!("Failed to send message to router (receiver dropped)");
                                        return;
                                    }
                                }
                            }
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

        info!("Telegram listener stopped");
    }
}

/// A Telegram media duration (whole seconds, `i32` on the wire) as the `f32`
/// seconds the channel model carries: exact for every whole second below 2^24
/// (~194 days), and a longer duration rounding to a neighbouring second is
/// harmless.
fn duration_secs(secs: i32) -> f32 {
    nanna_numeric::f32_from_i32(secs)
}

#[async_trait]
impl Listener for TelegramListener {
    fn provider(&self) -> &'static str {
        "telegram"
    }

    async fn start(
        self: Arc<Self>,
        sender: mpsc::Sender<IncomingMessage>,
    ) -> Result<ListenerHandle, ListenerError> {
        // Delete any existing webhook to enable polling
        let _ = self
            .client
            .post(self.api_url("deleteWebhook"))
            .send()
            .await;

        let (shutdown_tx, shutdown_rx) = mpsc::channel(1);

        let listener = Arc::clone(&self);
        let handle = tokio::spawn(async move {
            listener.poll_loop(sender, shutdown_rx).await;
        });

        Ok(ListenerHandle::new(shutdown_tx, handle))
    }
}

// ============================================================================
// Telegram API Types
// ============================================================================

#[derive(Debug, Deserialize)]
struct TelegramApiResponse<T> {
    ok: bool,
    result: Option<T>,
    description: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TelegramUpdate {
    update_id: i64,
    message: Option<TelegramMessage>,
    edited_message: Option<TelegramMessage>,
    /// The user pressed the stop button on a streamed draft (Bot API 10.3).
    stopped_message_generation: Option<MessageGenerationStopped>,
}

/// `MessageGenerationStopped`: which chat's draft the user stopped.
#[derive(Debug, Deserialize)]
struct MessageGenerationStopped {
    chat: TelegramChat,
    draft_id: i64,
}

#[derive(Debug, Deserialize)]
struct TelegramMessage {
    message_id: i64,
    chat: TelegramChat,
    from: Option<TelegramUser>,
    date: i64,
    text: Option<String>,
    caption: Option<String>,
    photo: Option<Vec<TelegramPhotoSize>>,
    document: Option<TelegramDocument>,
    audio: Option<TelegramAudio>,
    voice: Option<TelegramVoice>,
    video: Option<TelegramVideo>,
    location: Option<TelegramLocation>,
    sticker: Option<TelegramSticker>,
    reply_to_message: Option<Box<Self>>,
}

#[derive(Debug, Deserialize)]
struct TelegramChat {
    id: i64,
    #[serde(rename = "type")]
    _chat_type: String,
    #[serde(rename = "title")]
    _title: Option<String>,
    #[serde(rename = "username")]
    _username: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TelegramUser {
    id: i64,
    first_name: String,
    last_name: Option<String>,
    username: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TelegramPhotoSize {
    file_id: String,
    #[serde(rename = "width")]
    _width: i32,
    #[serde(rename = "height")]
    _height: i32,
}

#[derive(Debug, Deserialize)]
struct TelegramDocument {
    file_id: String,
    file_name: Option<String>,
    #[serde(rename = "mime_type")]
    _mime_type: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TelegramAudio {
    file_id: String,
    duration: i32,
    #[serde(rename = "title")]
    _title: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TelegramVoice {
    file_id: String,
    duration: i32,
}

#[derive(Debug, Deserialize)]
struct TelegramVideo {
    file_id: String,
    duration: i32,
    #[serde(rename = "width")]
    _width: i32,
    #[serde(rename = "height")]
    _height: i32,
}

#[derive(Debug, Deserialize)]
struct TelegramLocation {
    latitude: f64,
    longitude: f64,
}

#[derive(Debug, Deserialize)]
struct TelegramSticker {
    file_id: String,
    emoji: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_url() {
        let listener = TelegramListener::new("123:ABC");
        assert_eq!(
            listener.api_url("getUpdates"),
            "https://api.telegram.org/bot123:ABC/getUpdates"
        );
    }

    #[test]
    fn a_pressed_stop_button_arrives_as_the_users_stop() {
        let listener = TelegramListener::new("123:ABC");
        let update: TelegramUpdate = serde_json::from_value(serde_json::json!({
            "update_id": 9,
            "stopped_message_generation": {
                "chat": { "id": 4242, "type": "private" },
                "draft_id": 7
            }
        }))
        .expect("the Bot API 10.3 shape parses");
        let stop = listener.convert_update(&update).expect("routed");
        assert_eq!(stop.channel, ChannelId::new("telegram", "4242"));
        assert_eq!(stop.sender.id, "4242", "the private chat's own user");
        assert!(matches!(stop.content, MessageContent::Text { ref text } if text == "/stop"));

        // Not from an allowed chat, or not a private chat: ignored.
        let fenced = TelegramListener::new("t").with_allowed_chats(vec![1]);
        assert!(fenced.convert_update(&update).is_none());
        let group: TelegramUpdate = serde_json::from_value(serde_json::json!({
            "update_id": 10,
            "stopped_message_generation": { "chat": { "id": -100, "type": "group" }, "draft_id": 1 }
        }))
        .expect("parses");
        assert!(listener.convert_update(&group).is_none());
    }

    #[test]
    fn test_with_allowed_chats() {
        let listener = TelegramListener::new("token").with_allowed_chats(vec![123, 456]);
        assert_eq!(listener.allowed_chats, vec![123, 456]);
    }
}
