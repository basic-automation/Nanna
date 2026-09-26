//! Telegram channel implementation
//!
//! Provides a full Telegram Bot API client for sending messages,
//! reactions, edits, and more.

use crate::{
    Channel, ChannelCapabilities, ChannelError, ChannelFeatures, ChannelId, MessageContent,
    OutgoingMessage,
};
use async_trait::async_trait;
use reqwest::Client;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::time::Duration;
use tracing::warn;

const TELEGRAM_API_BASE: &str = "https://api.telegram.org";

/// Telegram Bot API client
#[derive(Clone)]
pub struct TelegramChannel {
    client: Client,
    bot_token: String,
    default_parse_mode: Option<String>,
    /// `https://api.telegram.org`, or a self-hosted Bot API server.
    api_base: String,
}

/// Longest draft Telegram accepts, in characters (the message text limit).
pub const TELEGRAM_DRAFT_CHARS_MAX: usize = 4096;

/// The text a draft shows for `text`: all of it when it fits, else its tail
/// behind a marker, so the newest words stay visible as the reply grows. Pure.
#[must_use]
pub fn draft_text(text: &str) -> String {
    const MARKER: &str = "…";
    let count = text.chars().count();
    if count <= TELEGRAM_DRAFT_CHARS_MAX {
        return text.to_string();
    }
    let keep = TELEGRAM_DRAFT_CHARS_MAX - MARKER.chars().count();
    let tail: String = text.chars().skip(count - keep).collect();
    let shown = format!("{MARKER}{tail}");
    debug_assert_eq!(shown.chars().count(), TELEGRAM_DRAFT_CHARS_MAX);
    shown
}

impl TelegramChannel {
    /// Create a new Telegram channel with the given bot token.
    pub fn new(bot_token: impl Into<String>) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap_or_else(|_| Client::new());

        Self {
            client,
            bot_token: bot_token.into(),
            default_parse_mode: Some("Markdown".to_string()),
            api_base: TELEGRAM_API_BASE.to_string(),
        }
    }

    /// Talk to `api_base` instead of `https://api.telegram.org` — a
    /// self-hosted Bot API server (or a test double).
    #[must_use]
    pub fn with_api_base(mut self, api_base: impl Into<String>) -> Self {
        self.api_base = api_base.into().trim_end_matches('/').to_string();
        self
    }

    /// Stream `text` into a private chat as the reply's draft
    /// (`sendMessageDraft`, Bot API 9.x+; private chats only).
    ///
    /// # Errors
    ///
    /// As [`Self::send_typing`].
    ///
    /// # Panics
    ///
    /// On a zero `draft_id` — Telegram rejects it, so it is a caller bug.
    pub async fn send_message_draft(
        &self,
        chat_id: i64,
        draft_id: i64,
        text: &str,
    ) -> Result<bool, ChannelError> {
        #[derive(Serialize)]
        struct DraftParams {
            chat_id: i64,
            draft_id: i64,
            text: String,
            /// Show a stop button; a press arrives as `stopped_message_generation`
            /// (Bot API 10.3), which the listener turns into `/stop`.
            can_stop: bool,
            /// Keep what was written visible after a stop, until the final
            /// message (the partial answer) replaces it.
            keep_on_stop: bool,
        }
        assert!(draft_id != 0, "Telegram requires a non-zero draft_id");
        debug_assert!(chat_id > 0, "drafts exist only in private chats");
        self.request(
            "sendMessageDraft",
            DraftParams {
                chat_id,
                draft_id,
                text: draft_text(text),
                can_stop: true,
                keep_on_stop: true,
            },
        )
        .await
    }

    /// Set the default parse mode for messages.
    #[must_use]
    pub fn with_parse_mode(mut self, mode: impl Into<String>) -> Self {
        self.default_parse_mode = Some(mode.into());
        self
    }

    /// Build the API URL for a method.
    fn api_url(&self, method: &str) -> String {
        format!("{}/bot{}/{}", self.api_base, self.bot_token, method)
    }

    /// Make an API request.
    async fn request<T: DeserializeOwned>(
        &self,
        method: &str,
        params: impl Serialize,
    ) -> Result<T, ChannelError> {
        let response = self
            .client
            .post(self.api_url(method))
            .json(&params)
            .send()
            .await
            // `without_url`: the request URL embeds the bot token
            // (`/bot<token>/…`), and reqwest's Display prints it — every
            // connection error put the token in logs and channel status.
            .map_err(|e| ChannelError::Send(e.without_url().to_string()))?;

        let status = response.status();
        let body: TelegramApiResponse<T> = response
            .json()
            .await
            .map_err(|e| {
                ChannelError::Send(format!("Failed to parse response: {}", e.without_url()))
            })?;

        if body.ok {
            body.result
                .ok_or_else(|| ChannelError::Send("Empty result".to_string()))
        } else {
            let error_msg = body
                .description
                .unwrap_or_else(|| format!("HTTP {status}"));
            
            // Check for rate limiting
            if status.as_u16() == 429 {
                return Err(ChannelError::RateLimited);
            }
            
            Err(ChannelError::Send(error_msg))
        }
    }

    /// Send a text message.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelError::RateLimited`] if Telegram rejects the call with
    /// HTTP 429, and [`ChannelError::Send`] if the request cannot be sent, the
    /// response body is not a Telegram API envelope with the expected result, or
    /// Telegram rejects the call for any other reason (its `description` is
    /// carried in the error).
    pub async fn send_text(
        &self,
        chat_id: i64,
        text: &str,
        reply_to: Option<i64>,
    ) -> Result<TelegramMessageResult, ChannelError> {
        #[derive(Serialize)]
        struct SendMessageParams<'a> {
            chat_id: i64,
            text: &'a str,
            #[serde(skip_serializing_if = "Option::is_none")]
            parse_mode: Option<&'a str>,
            #[serde(skip_serializing_if = "Option::is_none")]
            reply_to_message_id: Option<i64>,
        }

        self.request(
            "sendMessage",
            SendMessageParams {
                chat_id,
                text,
                parse_mode: self.default_parse_mode.as_deref(),
                reply_to_message_id: reply_to,
            },
        )
        .await
    }

    /// Send a photo.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelError::RateLimited`] if Telegram rejects the call with
    /// HTTP 429, and [`ChannelError::Send`] if the request cannot be sent, the
    /// response body is not a Telegram API envelope with the expected result, or
    /// Telegram rejects the call for any other reason (its `description` is
    /// carried in the error).
    pub async fn send_photo(
        &self,
        chat_id: i64,
        photo_url: &str,
        caption: Option<&str>,
        reply_to: Option<i64>,
    ) -> Result<TelegramMessageResult, ChannelError> {
        #[derive(Serialize)]
        struct SendPhotoParams<'a> {
            chat_id: i64,
            photo: &'a str,
            #[serde(skip_serializing_if = "Option::is_none")]
            caption: Option<&'a str>,
            #[serde(skip_serializing_if = "Option::is_none")]
            parse_mode: Option<&'a str>,
            #[serde(skip_serializing_if = "Option::is_none")]
            reply_to_message_id: Option<i64>,
        }

        self.request(
            "sendPhoto",
            SendPhotoParams {
                chat_id,
                photo: photo_url,
                caption,
                parse_mode: self.default_parse_mode.as_deref(),
                reply_to_message_id: reply_to,
            },
        )
        .await
    }

    /// Send a document.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelError::RateLimited`] if Telegram rejects the call with
    /// HTTP 429, and [`ChannelError::Send`] if the request cannot be sent, the
    /// response body is not a Telegram API envelope with the expected result, or
    /// Telegram rejects the call for any other reason (its `description` is
    /// carried in the error).
    pub async fn send_document(
        &self,
        chat_id: i64,
        document_url: &str,
        caption: Option<&str>,
        reply_to: Option<i64>,
    ) -> Result<TelegramMessageResult, ChannelError> {
        #[derive(Serialize)]
        struct SendDocumentParams<'a> {
            chat_id: i64,
            document: &'a str,
            #[serde(skip_serializing_if = "Option::is_none")]
            caption: Option<&'a str>,
            #[serde(skip_serializing_if = "Option::is_none")]
            parse_mode: Option<&'a str>,
            #[serde(skip_serializing_if = "Option::is_none")]
            reply_to_message_id: Option<i64>,
        }

        self.request(
            "sendDocument",
            SendDocumentParams {
                chat_id,
                document: document_url,
                caption,
                parse_mode: self.default_parse_mode.as_deref(),
                reply_to_message_id: reply_to,
            },
        )
        .await
    }

    /// Send a location.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelError::RateLimited`] if Telegram rejects the call with
    /// HTTP 429, and [`ChannelError::Send`] if the request cannot be sent, the
    /// response body is not a Telegram API envelope with the expected result, or
    /// Telegram rejects the call for any other reason (its `description` is
    /// carried in the error).
    pub async fn send_location(
        &self,
        chat_id: i64,
        latitude: f64,
        longitude: f64,
        reply_to: Option<i64>,
    ) -> Result<TelegramMessageResult, ChannelError> {
        #[derive(Serialize)]
        struct SendLocationParams {
            chat_id: i64,
            latitude: f64,
            longitude: f64,
            #[serde(skip_serializing_if = "Option::is_none")]
            reply_to_message_id: Option<i64>,
        }

        self.request(
            "sendLocation",
            SendLocationParams {
                chat_id,
                latitude,
                longitude,
                reply_to_message_id: reply_to,
            },
        )
        .await
    }

    /// Send a poll.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelError::RateLimited`] if Telegram rejects the call with
    /// HTTP 429, and [`ChannelError::Send`] if the request cannot be sent, the
    /// response body is not a Telegram API envelope with the expected result, or
    /// Telegram rejects the call for any other reason (its `description` is
    /// carried in the error).
    pub async fn send_poll(
        &self,
        chat_id: i64,
        question: &str,
        options: &[String],
        allows_multiple: bool,
    ) -> Result<TelegramMessageResult, ChannelError> {
        #[derive(Serialize)]
        struct SendPollParams<'a> {
            chat_id: i64,
            question: &'a str,
            options: &'a [String],
            allows_multiple_answers: bool,
        }

        self.request(
            "sendPoll",
            SendPollParams {
                chat_id,
                question,
                options,
                allows_multiple_answers: allows_multiple,
            },
        )
        .await
    }

    /// Edit a message.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelError::RateLimited`] if Telegram rejects the call with
    /// HTTP 429, and [`ChannelError::Send`] if the request cannot be sent, the
    /// response body is not a Telegram API envelope with the expected result, or
    /// Telegram rejects the call for any other reason (its `description` is
    /// carried in the error).
    pub async fn edit_message(
        &self,
        chat_id: i64,
        message_id: i64,
        text: &str,
    ) -> Result<TelegramMessageResult, ChannelError> {
        #[derive(Serialize)]
        struct EditMessageParams<'a> {
            chat_id: i64,
            message_id: i64,
            text: &'a str,
            #[serde(skip_serializing_if = "Option::is_none")]
            parse_mode: Option<&'a str>,
        }

        self.request(
            "editMessageText",
            EditMessageParams {
                chat_id,
                message_id,
                text,
                parse_mode: self.default_parse_mode.as_deref(),
            },
        )
        .await
    }

    /// Delete a message.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelError::RateLimited`] if Telegram rejects the call with
    /// HTTP 429, and [`ChannelError::Send`] if the request cannot be sent, the
    /// response body is not a Telegram API envelope with the expected result, or
    /// Telegram rejects the call for any other reason (its `description` is
    /// carried in the error).
    pub async fn delete_message(&self, chat_id: i64, message_id: i64) -> Result<bool, ChannelError> {
        #[derive(Serialize)]
        struct DeleteMessageParams {
            chat_id: i64,
            message_id: i64,
        }

        self.request(
            "deleteMessage",
            DeleteMessageParams {
                chat_id,
                message_id,
            },
        )
        .await
    }

    /// Set a reaction on a message.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelError::RateLimited`] if Telegram rejects the call with
    /// HTTP 429, and [`ChannelError::Send`] if the request cannot be sent, the
    /// response body is not a Telegram API envelope with the expected result, or
    /// Telegram rejects the call for any other reason (its `description` is
    /// carried in the error).
    pub async fn set_reaction(
        &self,
        chat_id: i64,
        message_id: i64,
        emoji: &str,
    ) -> Result<bool, ChannelError> {
        #[derive(Serialize)]
        struct ReactionType<'a> {
            #[serde(rename = "type")]
            reaction_type: &'a str,
            emoji: &'a str,
        }

        #[derive(Serialize)]
        struct SetReactionParams<'a> {
            chat_id: i64,
            message_id: i64,
            reaction: Vec<ReactionType<'a>>,
        }

        self.request(
            "setMessageReaction",
            SetReactionParams {
                chat_id,
                message_id,
                reaction: vec![ReactionType {
                    reaction_type: "emoji",
                    emoji,
                }],
            },
        )
        .await
    }

    /// Remove reaction from a message.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelError::RateLimited`] if Telegram rejects the call with
    /// HTTP 429, and [`ChannelError::Send`] if the request cannot be sent, the
    /// response body is not a Telegram API envelope with the expected result, or
    /// Telegram rejects the call for any other reason (its `description` is
    /// carried in the error).
    pub async fn remove_reaction(&self, chat_id: i64, message_id: i64) -> Result<bool, ChannelError> {
        #[derive(Serialize)]
        struct SetReactionParams {
            chat_id: i64,
            message_id: i64,
            reaction: Vec<()>,
        }

        self.request(
            "setMessageReaction",
            SetReactionParams {
                chat_id,
                message_id,
                reaction: vec![],
            },
        )
        .await
    }

    /// Pin a message.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelError::RateLimited`] if Telegram rejects the call with
    /// HTTP 429, and [`ChannelError::Send`] if the request cannot be sent, the
    /// response body is not a Telegram API envelope with the expected result, or
    /// Telegram rejects the call for any other reason (its `description` is
    /// carried in the error).
    pub async fn pin_message(&self, chat_id: i64, message_id: i64) -> Result<bool, ChannelError> {
        #[derive(Serialize)]
        struct PinParams {
            chat_id: i64,
            message_id: i64,
            disable_notification: bool,
        }

        self.request(
            "pinChatMessage",
            PinParams {
                chat_id,
                message_id,
                disable_notification: true,
            },
        )
        .await
    }

    /// Unpin a message.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelError::RateLimited`] if Telegram rejects the call with
    /// HTTP 429, and [`ChannelError::Send`] if the request cannot be sent, the
    /// response body is not a Telegram API envelope with the expected result, or
    /// Telegram rejects the call for any other reason (its `description` is
    /// carried in the error).
    pub async fn unpin_message(&self, chat_id: i64, message_id: i64) -> Result<bool, ChannelError> {
        #[derive(Serialize)]
        struct UnpinParams {
            chat_id: i64,
            message_id: i64,
        }

        self.request("unpinChatMessage", UnpinParams { chat_id, message_id })
            .await
    }

    /// Send typing indicator.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelError::RateLimited`] if Telegram rejects the call with
    /// HTTP 429, and [`ChannelError::Send`] if the request cannot be sent, the
    /// response body is not a Telegram API envelope with the expected result, or
    /// Telegram rejects the call for any other reason (its `description` is
    /// carried in the error).
    pub async fn send_typing(&self, chat_id: i64) -> Result<bool, ChannelError> {
        #[derive(Serialize)]
        struct ChatActionParams {
            chat_id: i64,
            action: &'static str,
        }

        self.request(
            "sendChatAction",
            ChatActionParams {
                chat_id,
                action: "typing",
            },
        )
        .await
    }

    /// Get bot info.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelError::RateLimited`] if Telegram rejects the call with
    /// HTTP 429, and [`ChannelError::Send`] if the request cannot be sent, the
    /// response body is not a Telegram API envelope with the expected result, or
    /// Telegram rejects the call for any other reason (its `description` is
    /// carried in the error).
    pub async fn get_me(&self) -> Result<TelegramUser, ChannelError> {
        self.request("getMe", serde_json::json!({})).await
    }

    /// Set webhook URL.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelError::RateLimited`] if Telegram rejects the call with
    /// HTTP 429, and [`ChannelError::Send`] if the request cannot be sent, the
    /// response body is not a Telegram API envelope with the expected result, or
    /// Telegram rejects the call for any other reason (its `description` is
    /// carried in the error).
    pub async fn set_webhook(&self, url: &str, secret_token: Option<&str>) -> Result<bool, ChannelError> {
        #[derive(Serialize)]
        struct SetWebhookParams<'a> {
            url: &'a str,
            #[serde(skip_serializing_if = "Option::is_none")]
            secret_token: Option<&'a str>,
        }

        self.request(
            "setWebhook",
            SetWebhookParams { url, secret_token },
        )
        .await
    }

    /// Delete webhook (switch to polling mode).
    ///
    /// # Errors
    ///
    /// Returns [`ChannelError::RateLimited`] if Telegram rejects the call with
    /// HTTP 429, and [`ChannelError::Send`] if the request cannot be sent, the
    /// response body is not a Telegram API envelope with the expected result, or
    /// Telegram rejects the call for any other reason (its `description` is
    /// carried in the error).
    pub async fn delete_webhook(&self) -> Result<bool, ChannelError> {
        self.request("deleteWebhook", serde_json::json!({})).await
    }
}

// ============================================================================
// Telegram API Response Types
// ============================================================================

#[derive(Debug, Deserialize)]
struct TelegramApiResponse<T> {
    ok: bool,
    result: Option<T>,
    description: Option<String>,
}

/// Result of sending a message
#[derive(Debug, Clone, Deserialize)]
pub struct TelegramMessageResult {
    pub message_id: i64,
    pub chat: TelegramChat,
    pub date: i64,
    pub text: Option<String>,
}

/// Telegram chat info
#[derive(Debug, Clone, Deserialize)]
pub struct TelegramChat {
    pub id: i64,
    #[serde(rename = "type")]
    pub chat_type: String,
    pub title: Option<String>,
    pub username: Option<String>,
}

/// Telegram user info
#[derive(Debug, Clone, Deserialize)]
pub struct TelegramUser {
    pub id: i64,
    pub is_bot: bool,
    pub first_name: String,
    pub last_name: Option<String>,
    pub username: Option<String>,
}

// ============================================================================
// Channel Trait Implementation
// ============================================================================

#[async_trait]
impl Channel for TelegramChannel {
    fn provider(&self) -> String {
        "telegram".to_string()
    }

    fn capabilities(&self) -> ChannelCapabilities {
        ChannelCapabilities {
            features: ChannelFeatures::REACTIONS
                | ChannelFeatures::REPLIES
                | ChannelFeatures::EDITS
                | ChannelFeatures::IMAGES
                | ChannelFeatures::AUDIO
                | ChannelFeatures::DOCUMENTS
                | ChannelFeatures::POLLS
                | ChannelFeatures::PINS
                | ChannelFeatures::TYPING
                | ChannelFeatures::STICKERS
                | ChannelFeatures::MARKDOWN,
            max_message_length: Some(4096),
        }
    }

    async fn send(&self, message: OutgoingMessage) -> Result<String, ChannelError> {
        let chat_id: i64 = message
            .channel
            .id
            .parse()
            .map_err(|_| ChannelError::Send("Invalid chat ID".to_string()))?;

        let reply_to: Option<i64> = message
            .reply_to
            .as_deref()
            .and_then(|r| crate::native_reply_id(r).parse().ok());

        let result = match message.content {
            MessageContent::Text { text } => {
                self.send_text(chat_id, &text, reply_to).await?
            }
            MessageContent::Image { url, caption } => {
                self.send_photo(chat_id, &url, caption.as_deref(), reply_to).await?
            }
            MessageContent::Document { url, filename: _ } => {
                self.send_document(chat_id, &url, None, reply_to).await?
            }
            MessageContent::Location { latitude, longitude } => {
                self.send_location(chat_id, latitude, longitude, reply_to).await?
            }
            MessageContent::Poll { question, options, multiple } => {
                self.send_poll(chat_id, &question, &options, multiple).await?
            }
            MessageContent::Audio { url, .. } => {
                // Send as document for now
                self.send_document(chat_id, &url, None, reply_to).await?
            }
            MessageContent::Video { url, caption, .. } => {
                // Send as document for now
                self.send_document(chat_id, &url, caption.as_deref(), reply_to).await?
            }
            MessageContent::Sticker { .. } => {
                // Would need sendSticker API
                warn!("Sticker sending not yet implemented");
                return Err(ChannelError::Send("Stickers not yet supported".to_string()));
            }
        };

        Ok(result.message_id.to_string())
    }

    async fn react(&self, message_id: &str, emoji: &str) -> Result<(), ChannelError> {
        // Telegram reactions need chat_id too - we'd need to parse from message_id
        // Format: "chat_id:message_id"
        let parts: Vec<&str> = message_id.split(':').collect();
        if parts.len() != 2 {
            return Err(ChannelError::Send(
                "Invalid message ID format (expected chat_id:message_id)".to_string(),
            ));
        }

        let chat_id: i64 = parts[0]
            .parse()
            .map_err(|_| ChannelError::Send("Invalid chat ID".to_string()))?;
        let msg_id: i64 = parts[1]
            .parse()
            .map_err(|_| ChannelError::Send("Invalid message ID".to_string()))?;

        self.set_reaction(chat_id, msg_id, emoji).await?;
        Ok(())
    }

    async fn unreact(&self, message_id: &str, _emoji: &str) -> Result<(), ChannelError> {
        let parts: Vec<&str> = message_id.split(':').collect();
        if parts.len() != 2 {
            return Err(ChannelError::Send(
                "Invalid message ID format (expected chat_id:message_id)".to_string(),
            ));
        }

        let chat_id: i64 = parts[0]
            .parse()
            .map_err(|_| ChannelError::Send("Invalid chat ID".to_string()))?;
        let msg_id: i64 = parts[1]
            .parse()
            .map_err(|_| ChannelError::Send("Invalid message ID".to_string()))?;

        self.remove_reaction(chat_id, msg_id).await?;
        Ok(())
    }

    async fn edit(&self, message_id: &str, content: MessageContent) -> Result<(), ChannelError> {
        let parts: Vec<&str> = message_id.split(':').collect();
        if parts.len() != 2 {
            return Err(ChannelError::Send(
                "Invalid message ID format (expected chat_id:message_id)".to_string(),
            ));
        }

        let chat_id: i64 = parts[0]
            .parse()
            .map_err(|_| ChannelError::Send("Invalid chat ID".to_string()))?;
        let msg_id: i64 = parts[1]
            .parse()
            .map_err(|_| ChannelError::Send("Invalid message ID".to_string()))?;

        let MessageContent::Text { text } = content else {
            return Err(ChannelError::Send("Can only edit text messages".to_string()));
        };

        self.edit_message(chat_id, msg_id, &text).await?;
        Ok(())
    }

    async fn delete(&self, message_id: &str) -> Result<(), ChannelError> {
        let parts: Vec<&str> = message_id.split(':').collect();
        if parts.len() != 2 {
            return Err(ChannelError::Send(
                "Invalid message ID format (expected chat_id:message_id)".to_string(),
            ));
        }

        let chat_id: i64 = parts[0]
            .parse()
            .map_err(|_| ChannelError::Send("Invalid chat ID".to_string()))?;
        let msg_id: i64 = parts[1]
            .parse()
            .map_err(|_| ChannelError::Send("Invalid message ID".to_string()))?;

        self.delete_message(chat_id, msg_id).await?;
        Ok(())
    }

    async fn pin(&self, message_id: &str) -> Result<(), ChannelError> {
        let parts: Vec<&str> = message_id.split(':').collect();
        if parts.len() != 2 {
            return Err(ChannelError::Send(
                "Invalid message ID format (expected chat_id:message_id)".to_string(),
            ));
        }

        let chat_id: i64 = parts[0]
            .parse()
            .map_err(|_| ChannelError::Send("Invalid chat ID".to_string()))?;
        let msg_id: i64 = parts[1]
            .parse()
            .map_err(|_| ChannelError::Send("Invalid message ID".to_string()))?;

        self.pin_message(chat_id, msg_id).await?;
        Ok(())
    }

    async fn unpin(&self, message_id: &str) -> Result<(), ChannelError> {
        let parts: Vec<&str> = message_id.split(':').collect();
        if parts.len() != 2 {
            return Err(ChannelError::Send(
                "Invalid message ID format (expected chat_id:message_id)".to_string(),
            ));
        }

        let chat_id: i64 = parts[0]
            .parse()
            .map_err(|_| ChannelError::Send("Invalid chat ID".to_string()))?;
        let msg_id: i64 = parts[1]
            .parse()
            .map_err(|_| ChannelError::Send("Invalid message ID".to_string()))?;

        self.unpin_message(chat_id, msg_id).await?;
        Ok(())
    }

    /// A positive chat id is a private chat (groups and channels are
    /// negative), which is the only place Telegram shows drafts.
    fn supports_drafts(&self, channel: &ChannelId) -> bool {
        channel.id.parse::<i64>().is_ok_and(|chat_id| chat_id > 0)
    }

    async fn send_draft(
        &self,
        channel: &ChannelId,
        draft_id: i64,
        text: &str,
    ) -> Result<(), ChannelError> {
        let chat_id: i64 = channel
            .id
            .parse()
            .map_err(|_| ChannelError::Send("Invalid chat ID".to_string()))?;
        if chat_id <= 0 {
            return Ok(());
        }
        // Empty text is deliberate: Telegram shows its animated "Thinking…"
        // placeholder for it, which is what a turn shows before its words.
        self.send_message_draft(chat_id, draft_id, text).await?;
        Ok(())
    }

    async fn send_typing(&self, channel_id: &ChannelId) -> Result<(), ChannelError> {
        let chat_id: i64 = channel_id
            .id
            .parse()
            .map_err(|_| ChannelError::Send("Invalid chat ID".to_string()))?;

        self.send_typing(chat_id).await?;
        Ok(())
    }

    async fn send_poll(
        &self,
        channel: &ChannelId,
        question: &str,
        options: &[String],
        multiple: bool,
    ) -> Result<String, ChannelError> {
        let chat_id: i64 = channel
            .id
            .parse()
            .map_err(|_| ChannelError::Send("Invalid chat ID".to_string()))?;

        let result = Self::send_poll(self, chat_id, question, options, multiple).await?;
        Ok(result.message_id.to_string())
    }
}

#[cfg(test)]
mod tests {
    /// A failed request names what failed, never the token in its URL.
    #[tokio::test]
    async fn a_connection_error_does_not_carry_the_token() {
        let token = "123456:SECRET-token-value";
        let channel = super::TelegramChannel::new(token).with_api_base("http://127.0.0.1:9");
        let err = channel
            .send_message_draft(1, 1, "hi")
            .await
            .expect_err("nothing listens on port 9");
        let shown = format!("{err} {err:?}");
        assert!(!shown.contains("SECRET"), "{shown}");
    }

    use super::*;

    #[test]
    fn a_long_draft_shows_its_newest_words() {
        assert_eq!(draft_text("short"), "short");
        let long: String = "a".repeat(TELEGRAM_DRAFT_CHARS_MAX) + "END";
        let shown = draft_text(&long);
        assert_eq!(shown.chars().count(), TELEGRAM_DRAFT_CHARS_MAX);
        assert!(shown.starts_with('…'));
        assert!(shown.ends_with("aEND"), "the tail survives");
        let multibyte = "é".repeat(TELEGRAM_DRAFT_CHARS_MAX + 10);
        assert_eq!(
            draft_text(&multibyte).chars().count(),
            TELEGRAM_DRAFT_CHARS_MAX,
            "counted in chars, not bytes"
        );
    }

    #[test]
    fn drafts_are_offered_for_private_chats_only() {
        let telegram = TelegramChannel::new("123:ABC");
        assert!(telegram.supports_drafts(&ChannelId::new("telegram", "4242")));
        assert!(
            !telegram.supports_drafts(&ChannelId::new("telegram", "-1001234")),
            "a group"
        );
        assert!(!telegram.supports_drafts(&ChannelId::new("telegram", "not-a-number")));
    }

    #[tokio::test]
    async fn a_draft_is_posted_as_send_message_draft() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let mut buf = [0_u8; 4096];
            // Read until the JSON body is complete (headers + a closing brace).
            while !request.ends_with(b"}") {
                let n = socket.read(&mut buf).await.unwrap();
                if n == 0 {
                    break;
                }
                request.extend_from_slice(&buf[..n]);
            }
            let body = r#"{"ok":true,"result":true}"#;
            let reply = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
                body.len()
            );
            socket.write_all(reply.as_bytes()).await.unwrap();
            String::from_utf8(request).unwrap()
        });
        let telegram = TelegramChannel::new("123:ABC").with_api_base(base);
        telegram
            .send_draft(&ChannelId::new("telegram", "4242"), 7, "Thinking about it")
            .await
            .expect("sent");
        let request = server.await.unwrap();
        assert!(
            request.starts_with("POST /bot123:ABC/sendMessageDraft "),
            "{request}"
        );
        let body = &request[request.find('{').unwrap()..];
        let json: serde_json::Value = serde_json::from_str(body).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "chat_id": 4242, "draft_id": 7, "text": "Thinking about it",
                "can_stop": true, "keep_on_stop": true
            })
        );

        // A group chat is not asked at all (no server is listening now).
        telegram
            .send_draft(&ChannelId::new("telegram", "-1001"), 7, "x")
            .await
            .expect("a no-op, not an error");
    }

    #[test]
    fn test_api_url() {
        let channel = TelegramChannel::new("123456:ABC-DEF");
        assert_eq!(
            channel.api_url("sendMessage"),
            "https://api.telegram.org/bot123456:ABC-DEF/sendMessage"
        );
    }

    #[test]
    fn test_capabilities() {
        let channel = TelegramChannel::new("token");
        let caps = channel.capabilities();
        assert!(caps.supports_reactions());
        assert!(caps.supports_replies());
        assert!(caps.supports_edits());
        assert!(caps.supports_polls());
        assert_eq!(caps.max_message_length, Some(4096));
    }
}
