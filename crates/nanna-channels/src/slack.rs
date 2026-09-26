//! Slack channel implementation
//!
//! Provides Slack Web API client for sending messages, reactions, threads, and more.
//! Uses Bot tokens (xoxb-) for authentication.

use crate::{
    Channel, ChannelCapabilities, ChannelError, ChannelFeatures, MessageContent,
    OutgoingMessage, ThreadInfo,
};
use async_trait::async_trait;
use reqwest::Client;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::time::Duration;
use tracing::debug;

const SLACK_API_BASE: &str = "https://slack.com/api";

/// Slack Web API client
#[derive(Clone)]
pub struct SlackChannel {
    client: Client,
    bot_token: String,
    default_channel: Option<String>,
}

impl SlackChannel {
    /// Create a new Slack channel with the given bot token (xoxb-...).
    pub fn new(bot_token: impl Into<String>) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap_or_else(|_| Client::new());

        Self {
            client,
            bot_token: bot_token.into(),
            default_channel: None,
        }
    }

    /// Set a default channel for messages.
    #[must_use]
    pub fn with_default_channel(mut self, channel: impl Into<String>) -> Self {
        self.default_channel = Some(channel.into());
        self
    }

    /// Make a Slack API request.
    async fn api<T: DeserializeOwned>(
        &self,
        method: &str,
        params: impl Serialize,
    ) -> Result<T, ChannelError> {
        let url = format!("{SLACK_API_BASE}/{method}");
        
        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.bot_token))
            .header("Content-Type", "application/json; charset=utf-8")
            .json(&params)
            .send()
            .await
            .map_err(|e| ChannelError::Send(e.to_string()))?;

        let status = response.status();
        let body: SlackApiResponse<T> = response
            .json()
            .await
            .map_err(|e| ChannelError::Send(format!("Failed to parse response: {e}")))?;

        if body.ok {
            body.result
                .ok_or_else(|| ChannelError::Send("Empty result".to_string()))
        } else {
            let error = body.error.unwrap_or_else(|| "unknown_error".to_string());
            
            // Handle rate limiting
            if error == "ratelimited" || status.as_u16() == 429 {
                return Err(ChannelError::RateLimited);
            }
            
            // Handle common errors
            match error.as_str() {
                "channel_not_found" => Err(ChannelError::NotFound(error)),
                "not_in_channel" => Err(ChannelError::Send("Bot not in channel".to_string())),
                "invalid_auth" | "token_revoked" => Err(ChannelError::Auth(error)),
                _ => Err(ChannelError::Send(format!("Slack API error: {error}"))),
            }
        }
    }

    /// Make a Slack API request that returns just ok/error (no result field).
    async fn api_ok(&self, method: &str, params: impl Serialize) -> Result<(), ChannelError> {
        let url = format!("{SLACK_API_BASE}/{method}");
        
        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.bot_token))
            .header("Content-Type", "application/json; charset=utf-8")
            .json(&params)
            .send()
            .await
            .map_err(|e| ChannelError::Send(e.to_string()))?;

        let status = response.status();
        let body: SlackOkResponse = response
            .json()
            .await
            .map_err(|e| ChannelError::Send(format!("Failed to parse response: {e}")))?;

        if body.ok {
            Ok(())
        } else {
            let error = body.error.unwrap_or_else(|| "unknown_error".to_string());
            if error == "ratelimited" || status.as_u16() == 429 {
                Err(ChannelError::RateLimited)
            } else {
                Err(ChannelError::Send(format!("Slack API error: {error}")))
            }
        }
    }

    // ========================================================================
    // Message Operations
    // ========================================================================

    /// Send a text message to a channel.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelError::RateLimited`] if Slack reports `ratelimited` or
    /// answers HTTP 429, [`ChannelError::NotFound`] for `channel_not_found`,
    /// [`ChannelError::Auth`] for `invalid_auth` or `token_revoked`, and
    /// [`ChannelError::Send`] if the request cannot be sent, the response body
    /// is not a posted-message result, or Slack reports any other error.
    pub async fn send_text(
        &self,
        channel: &str,
        text: &str,
        thread_ts: Option<&str>,
    ) -> Result<SlackMessage, ChannelError> {
        #[derive(Serialize)]
        struct Params<'a> {
            channel: &'a str,
            text: &'a str,
            #[serde(skip_serializing_if = "Option::is_none")]
            thread_ts: Option<&'a str>,
            #[serde(skip_serializing_if = "Option::is_none")]
            mrkdwn: Option<bool>,
        }

        debug!(channel, "Sending Slack message");

        let result: PostMessageResponse = self
            .api(
                "chat.postMessage",
                Params {
                    channel,
                    text,
                    thread_ts,
                    mrkdwn: Some(true),
                },
            )
            .await?;

        Ok(SlackMessage {
            ts: result.ts,
            channel: result.channel,
            text: Some(text.to_string()),
        })
    }

    /// Update a message.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelError::RateLimited`] if Slack reports `ratelimited` or
    /// answers HTTP 429, [`ChannelError::NotFound`] for `channel_not_found`,
    /// [`ChannelError::Auth`] for `invalid_auth` or `token_revoked`, and
    /// [`ChannelError::Send`] if the request cannot be sent, the response body
    /// is not a posted-message result, or Slack reports any other error.
    pub async fn update_message(
        &self,
        channel: &str,
        ts: &str,
        text: &str,
    ) -> Result<SlackMessage, ChannelError> {
        #[derive(Serialize)]
        struct Params<'a> {
            channel: &'a str,
            ts: &'a str,
            text: &'a str,
        }

        debug!(channel, ts, "Updating Slack message");

        let result: PostMessageResponse = self
            .api("chat.update", Params { channel, ts, text })
            .await?;

        Ok(SlackMessage {
            ts: result.ts,
            channel: result.channel,
            text: Some(text.to_string()),
        })
    }

    /// Delete a message.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelError::RateLimited`] if Slack reports `ratelimited` or
    /// answers HTTP 429, and [`ChannelError::Send`] if the request cannot be
    /// sent, the response body is not a Slack `ok`/`error` object, or Slack
    /// reports any other error.
    pub async fn delete_message(&self, channel: &str, ts: &str) -> Result<(), ChannelError> {
        #[derive(Serialize)]
        struct Params<'a> {
            channel: &'a str,
            ts: &'a str,
        }

        debug!(channel, ts, "Deleting Slack message");

        self.api_ok("chat.delete", Params { channel, ts }).await
    }

    // ========================================================================
    // Reactions
    // ========================================================================

    /// Add a reaction to a message.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelError::RateLimited`] if Slack reports `ratelimited` or
    /// answers HTTP 429, and [`ChannelError::Send`] if the request cannot be
    /// sent, the response body is not a Slack `ok`/`error` object, or Slack
    /// reports any other error.
    pub async fn add_reaction(
        &self,
        channel: &str,
        ts: &str,
        emoji: &str,
    ) -> Result<(), ChannelError> {
        #[derive(Serialize)]
        struct Params<'a> {
            channel: &'a str,
            timestamp: &'a str,
            name: &'a str,
        }

        debug!(channel, ts, emoji, "Adding Slack reaction");

        // Remove colons if present (Slack API wants just the name)
        let emoji = emoji.trim_matches(':');

        self.api_ok(
            "reactions.add",
            Params {
                channel,
                timestamp: ts,
                name: emoji,
            },
        )
        .await
    }

    /// Remove a reaction from a message.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelError::RateLimited`] if Slack reports `ratelimited` or
    /// answers HTTP 429, and [`ChannelError::Send`] if the request cannot be
    /// sent, the response body is not a Slack `ok`/`error` object, or Slack
    /// reports any other error.
    pub async fn remove_reaction(
        &self,
        channel: &str,
        ts: &str,
        emoji: &str,
    ) -> Result<(), ChannelError> {
        #[derive(Serialize)]
        struct Params<'a> {
            channel: &'a str,
            timestamp: &'a str,
            name: &'a str,
        }

        let emoji = emoji.trim_matches(':');

        self.api_ok(
            "reactions.remove",
            Params {
                channel,
                timestamp: ts,
                name: emoji,
            },
        )
        .await
    }

    /// Get reactions on a message.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelError::RateLimited`] if Slack reports `ratelimited` or
    /// answers HTTP 429, [`ChannelError::NotFound`] for `channel_not_found`,
    /// [`ChannelError::Auth`] for `invalid_auth` or `token_revoked`, and
    /// [`ChannelError::Send`] if the request cannot be sent, the response body
    /// is not a `reactions.get` result, or Slack reports any other error.
    pub async fn get_reactions(
        &self,
        channel: &str,
        ts: &str,
    ) -> Result<Vec<SlackReaction>, ChannelError> {
        #[derive(Serialize)]
        struct Params<'a> {
            channel: &'a str,
            timestamp: &'a str,
            full: bool,
        }

        #[derive(Deserialize)]
        struct Response {
            message: Option<MessageWithReactions>,
        }

        #[derive(Deserialize)]
        struct MessageWithReactions {
            reactions: Option<Vec<SlackReaction>>,
        }

        let result: Response = self
            .api(
                "reactions.get",
                Params {
                    channel,
                    timestamp: ts,
                    full: true,
                },
            )
            .await?;

        Ok(result
            .message
            .and_then(|m| m.reactions)
            .unwrap_or_default())
    }

    // ========================================================================
    // Threads
    // ========================================================================

    /// Reply in a thread.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelError::RateLimited`] if Slack reports `ratelimited` or
    /// answers HTTP 429, [`ChannelError::NotFound`] for `channel_not_found`,
    /// [`ChannelError::Auth`] for `invalid_auth` or `token_revoked`, and
    /// [`ChannelError::Send`] if the request cannot be sent, the response body
    /// is not a posted-message result, or Slack reports any other error.
    pub async fn reply_thread(
        &self,
        channel: &str,
        thread_ts: &str,
        text: &str,
    ) -> Result<SlackMessage, ChannelError> {
        self.send_text(channel, text, Some(thread_ts)).await
    }

    /// Get thread replies.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelError::RateLimited`] if Slack reports `ratelimited` or
    /// answers HTTP 429, [`ChannelError::NotFound`] for `channel_not_found`,
    /// [`ChannelError::Auth`] for `invalid_auth` or `token_revoked`, and
    /// [`ChannelError::Send`] if the request cannot be sent, the response body
    /// is not a `conversations.replies` result, or Slack reports any other error.
    pub async fn get_thread_replies(
        &self,
        channel: &str,
        thread_ts: &str,
    ) -> Result<Vec<SlackMessage>, ChannelError> {
        #[derive(Serialize)]
        struct Params<'a> {
            channel: &'a str,
            ts: &'a str,
        }

        #[derive(Deserialize)]
        struct Response {
            messages: Vec<SlackMessage>,
        }

        let result: Response = self
            .api(
                "conversations.replies",
                Params {
                    channel,
                    ts: thread_ts,
                },
            )
            .await?;

        Ok(result.messages)
    }

    // ========================================================================
    // Pins
    // ========================================================================

    /// Pin a message.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelError::RateLimited`] if Slack reports `ratelimited` or
    /// answers HTTP 429, and [`ChannelError::Send`] if the request cannot be
    /// sent, the response body is not a Slack `ok`/`error` object, or Slack
    /// reports any other error.
    pub async fn pin_message(&self, channel: &str, ts: &str) -> Result<(), ChannelError> {
        #[derive(Serialize)]
        struct Params<'a> {
            channel: &'a str,
            timestamp: &'a str,
        }

        self.api_ok("pins.add", Params { channel, timestamp: ts })
            .await
    }

    /// Unpin a message.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelError::RateLimited`] if Slack reports `ratelimited` or
    /// answers HTTP 429, and [`ChannelError::Send`] if the request cannot be
    /// sent, the response body is not a Slack `ok`/`error` object, or Slack
    /// reports any other error.
    pub async fn unpin_message(&self, channel: &str, ts: &str) -> Result<(), ChannelError> {
        #[derive(Serialize)]
        struct Params<'a> {
            channel: &'a str,
            timestamp: &'a str,
        }

        self.api_ok("pins.remove", Params { channel, timestamp: ts })
            .await
    }

    // ========================================================================
    // Files
    // ========================================================================

    /// Upload a file to a channel.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelError::Send`] if the request cannot be sent, the response
    /// body is not valid JSON, Slack reports the upload failed (a rate limit
    /// included), or the response carries no file object.
    pub async fn upload_file(
        &self,
        channels: &[&str],
        content: &[u8],
        filename: &str,
        title: Option<&str>,
    ) -> Result<SlackFile, ChannelError> {
        #[derive(Deserialize)]
        struct Response {
            ok: bool,
            file: Option<SlackFile>,
            error: Option<String>,
        }

        // Use files.uploadV2 (newer API)
        let url = format!("{SLACK_API_BASE}/files.uploadV2");

        // Build multipart form
        let mut form = reqwest::multipart::Form::new()
            .text("channels", channels.join(","))
            .text("filename", filename.to_string());

        if let Some(t) = title {
            form = form.text("title", t.to_string());
        }

        let part = reqwest::multipart::Part::bytes(content.to_vec())
            .file_name(filename.to_string());
        form = form.part("file", part);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.bot_token))
            .multipart(form)
            .send()
            .await
            .map_err(|e| ChannelError::Send(e.to_string()))?;

        let result: Response = response
            .json()
            .await
            .map_err(|e| ChannelError::Send(e.to_string()))?;

        if result.ok {
            result
                .file
                .ok_or_else(|| ChannelError::Send("No file in response".to_string()))
        } else {
            Err(ChannelError::Send(format!(
                "Upload failed: {}",
                result.error.unwrap_or_default()
            )))
        }
    }

    // ========================================================================
    // Channel Info
    // ========================================================================

    /// Get channel info.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelError::RateLimited`] if Slack reports `ratelimited` or
    /// answers HTTP 429, [`ChannelError::NotFound`] for `channel_not_found`,
    /// [`ChannelError::Auth`] for `invalid_auth` or `token_revoked`, and
    /// [`ChannelError::Send`] if the request cannot be sent, the response body
    /// is not a `conversations.info` result, or Slack reports any other error.
    pub async fn get_channel_info(&self, channel: &str) -> Result<SlackChannelInfo, ChannelError> {
        #[derive(Serialize)]
        struct Params<'a> {
            channel: &'a str,
        }

        #[derive(Deserialize)]
        struct Response {
            channel: SlackChannelInfo,
        }

        let result: Response = self
            .api("conversations.info", Params { channel })
            .await?;

        Ok(result.channel)
    }

    /// List channels the bot is in.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelError::RateLimited`] if Slack reports `ratelimited` or
    /// answers HTTP 429, [`ChannelError::NotFound`] for `channel_not_found`,
    /// [`ChannelError::Auth`] for `invalid_auth` or `token_revoked`, and
    /// [`ChannelError::Send`] if the request cannot be sent, the response body
    /// is not a `conversations.list` result, or Slack reports any other error.
    pub async fn list_channels(&self) -> Result<Vec<SlackChannelInfo>, ChannelError> {
        #[derive(Serialize)]
        struct Params {
            types: &'static str,
            exclude_archived: bool,
        }

        #[derive(Deserialize)]
        struct Response {
            channels: Vec<SlackChannelInfo>,
        }

        let result: Response = self
            .api(
                "conversations.list",
                Params {
                    types: "public_channel,private_channel",
                    exclude_archived: true,
                },
            )
            .await?;

        Ok(result.channels)
    }

    /// Validate the bot token by calling auth.test.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelError::RateLimited`] if Slack reports `ratelimited` or
    /// answers HTTP 429, [`ChannelError::NotFound`] for `channel_not_found`,
    /// [`ChannelError::Auth`] for `invalid_auth` or `token_revoked`, and
    /// [`ChannelError::Send`] if the request cannot be sent, the response body
    /// is not an `auth.test` result, or Slack reports any other error.
    pub async fn validate(&self) -> Result<SlackAuthInfo, ChannelError> {
        #[derive(Serialize)]
        struct Empty {}

        let result: SlackAuthInfo = self.api("auth.test", Empty {}).await?;
        Ok(result)
    }
}

// ============================================================================
// Channel Trait Implementation
// ============================================================================

#[async_trait]
impl Channel for SlackChannel {
    fn provider(&self) -> String {
        "slack".to_string()
    }

    fn capabilities(&self) -> ChannelCapabilities {
        ChannelCapabilities {
            features: ChannelFeatures::REACTIONS
                | ChannelFeatures::REPLIES
                | ChannelFeatures::EDITS
                | ChannelFeatures::THREADS
                | ChannelFeatures::PINS
                | ChannelFeatures::UPLOADS
                | ChannelFeatures::DOCUMENTS
                | ChannelFeatures::MARKDOWN,
            max_message_length: Some(40000), // Slack limit
        }
    }

    async fn send(&self, message: OutgoingMessage) -> Result<String, ChannelError> {
        let channel = &message.channel.id;
        let thread_ts = message.reply_to.as_deref().map(crate::native_reply_id);

        let result = match message.content {
            MessageContent::Text { text } => {
                self.send_text(channel, &text, thread_ts).await?
            }
            MessageContent::Image { url, caption } => {
                let text = caption.unwrap_or_else(|| url.clone());
                self.send_text(channel, &format!("{text}\n{url}"), thread_ts).await?
            }
            _ => {
                // For other types, convert to text representation
                let text = format!("{:?}", message.content);
                self.send_text(channel, &text, thread_ts).await?
            }
        };

        Ok(result.ts)
    }

    async fn react(&self, message_id: &str, emoji: &str) -> Result<(), ChannelError> {
        // message_id format: "channel:ts"
        let (channel, ts) = parse_message_id(message_id)?;
        self.add_reaction(&channel, &ts, emoji).await
    }

    async fn unreact(&self, message_id: &str, emoji: &str) -> Result<(), ChannelError> {
        let (channel, ts) = parse_message_id(message_id)?;
        self.remove_reaction(&channel, &ts, emoji).await
    }

    async fn get_reactions(&self, message_id: &str) -> Result<Vec<crate::Reaction>, ChannelError> {
        let (channel, ts) = parse_message_id(message_id)?;
        let reactions = self.get_reactions(&channel, &ts).await?;
        
        Ok(reactions
            .into_iter()
            .flat_map(|r| {
                r.users.into_iter().map(move |user| crate::Reaction {
                    emoji: r.name.clone(),
                    user_id: user,
                    timestamp: None,
                })
            })
            .collect())
    }

    async fn edit(&self, message_id: &str, content: MessageContent) -> Result<(), ChannelError> {
        let (channel, ts) = parse_message_id(message_id)?;
        let MessageContent::Text { text } = content else {
            return Err(ChannelError::Send("Can only edit text messages".to_string()));
        };
        self.update_message(&channel, &ts, &text).await?;
        Ok(())
    }

    async fn delete(&self, message_id: &str) -> Result<(), ChannelError> {
        let (channel, ts) = parse_message_id(message_id)?;
        self.delete_message(&channel, &ts).await
    }

    async fn pin(&self, message_id: &str) -> Result<(), ChannelError> {
        let (channel, ts) = parse_message_id(message_id)?;
        self.pin_message(&channel, &ts).await
    }

    async fn unpin(&self, message_id: &str) -> Result<(), ChannelError> {
        let (channel, ts) = parse_message_id(message_id)?;
        self.unpin_message(&channel, &ts).await
    }

    async fn create_thread(&self, message_id: &str, name: &str) -> Result<ThreadInfo, ChannelError> {
        // Slack threads are automatic - just reply to create one
        let (channel, ts) = parse_message_id(message_id)?;
        let msg = self.reply_thread(&channel, &ts, name).await?;
        
        Ok(ThreadInfo {
            id: format!("{}:{}", msg.channel, msg.ts),
            name: Some(name.to_string()),
            message_count: Some(1),
        })
    }

    async fn reply_thread(&self, thread_id: &str, content: MessageContent) -> Result<String, ChannelError> {
        let (channel, thread_ts) = parse_message_id(thread_id)?;
        let MessageContent::Text { text } = content else {
            return Err(ChannelError::Send("Can only send text to threads".to_string()));
        };
        let msg = self.reply_thread(&channel, &thread_ts, &text).await?;
        Ok(format!("{}:{}", msg.channel, msg.ts))
    }

    async fn upload_file(
        &self,
        filename: &str,
        data: &[u8],
        _content_type: &str,
    ) -> Result<String, ChannelError> {
        let channel = self
            .default_channel
            .as_deref()
            .ok_or_else(|| ChannelError::Send("No default channel set".to_string()))?;
        
        let file = self.upload_file(&[channel], data, filename, None).await?;
        Ok(file.permalink.unwrap_or(file.id))
    }
}

/// Parse message ID in format "channel:ts" 
fn parse_message_id(id: &str) -> Result<(String, String), ChannelError> {
    let parts: Vec<&str> = id.splitn(2, ':').collect();
    if parts.len() != 2 {
        return Err(ChannelError::Send(format!(
            "Invalid message ID format: {id} (expected channel:ts)"
        )));
    }
    Ok((parts[0].to_string(), parts[1].to_string()))
}

// ============================================================================
// Response Types
// ============================================================================

/// Generic Slack API response wrapper.
#[derive(Deserialize)]
struct SlackApiResponse<T> {
    ok: bool,
    #[serde(flatten)]
    result: Option<T>,
    error: Option<String>,
}

/// Slack API response with just ok/error.
#[derive(Deserialize)]
struct SlackOkResponse {
    ok: bool,
    error: Option<String>,
}

/// Response from chat.postMessage.
#[derive(Deserialize)]
struct PostMessageResponse {
    ts: String,
    channel: String,
}

/// Slack message.
#[derive(Debug, Clone, Deserialize)]
pub struct SlackMessage {
    pub ts: String,
    pub channel: String,
    pub text: Option<String>,
}

/// Slack reaction.
#[derive(Debug, Clone, Deserialize)]
pub struct SlackReaction {
    pub name: String,
    pub count: u32,
    pub users: Vec<String>,
}

/// Slack file.
#[derive(Debug, Clone, Deserialize)]
pub struct SlackFile {
    pub id: String,
    pub name: String,
    pub permalink: Option<String>,
    pub url_private: Option<String>,
}

/// Slack channel info.
#[derive(Debug, Clone, Deserialize)]
pub struct SlackChannelInfo {
    pub id: String,
    pub name: String,
    pub is_channel: Option<bool>,
    pub is_private: Option<bool>,
    pub is_member: Option<bool>,
}

/// Auth info from auth.test.
#[derive(Debug, Clone, Deserialize)]
pub struct SlackAuthInfo {
    pub url: String,
    pub team: String,
    pub user: String,
    pub team_id: String,
    pub user_id: String,
    pub bot_id: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_message_id() {
        let (channel, ts) = parse_message_id("C1234567:1234567890.123456").unwrap();
        assert_eq!(channel, "C1234567");
        assert_eq!(ts, "1234567890.123456");
    }

    #[test]
    fn test_parse_invalid_message_id() {
        assert!(parse_message_id("invalid").is_err());
    }
}
