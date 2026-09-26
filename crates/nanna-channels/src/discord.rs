//! Discord channel implementation
//!
//! Provides Discord Bot API for sending messages, reactions, and more.
//! Gateway (WebSocket) connection is handled separately.

use crate::{
    Channel, ChannelCapabilities, ChannelError, ChannelFeatures, ChannelId, MessageContent,
    OutgoingMessage, ThreadInfo,
};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

const DISCORD_API_BASE: &str = "https://discord.com/api/v10";

/// Discord Bot API client
#[derive(Clone)]
pub struct DiscordChannel {
    client: Client,
    bot_token: String,
    application_id: String,
}

impl DiscordChannel {
    /// Create a new Discord channel with bot token and application ID.
    pub fn new(bot_token: impl Into<String>, application_id: impl Into<String>) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap_or_else(|_| Client::new());

        Self {
            client,
            bot_token: bot_token.into(),
            application_id: application_id.into(),
        }
    }

    /// Build authorization header value.
    fn auth_header(&self) -> String {
        format!("Bot {}", self.bot_token)
    }

    /// Make a GET request.
    async fn get<T: for<'de> Deserialize<'de>>(&self, path: &str) -> Result<T, ChannelError> {
        let response = self
            .client
            .get(format!("{DISCORD_API_BASE}{path}"))
            .header("Authorization", self.auth_header())
            .send()
            .await
            .map_err(|e| ChannelError::Send(e.to_string()))?;

        self.handle_response(response).await
    }

    /// Make a POST request with JSON body.
    async fn post<T: for<'de> Deserialize<'de>>(
        &self,
        path: &str,
        body: impl Serialize,
    ) -> Result<T, ChannelError> {
        let response = self
            .client
            .post(format!("{DISCORD_API_BASE}{path}"))
            .header("Authorization", self.auth_header())
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| ChannelError::Send(e.to_string()))?;

        self.handle_response(response).await
    }

    /// Make a POST request without expecting a response body.
    async fn post_empty(&self, path: &str, body: impl Serialize) -> Result<(), ChannelError> {
        let response = self
            .client
            .post(format!("{DISCORD_API_BASE}{path}"))
            .header("Authorization", self.auth_header())
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| ChannelError::Send(e.to_string()))?;

        if response.status().is_success() {
            Ok(())
        } else {
            let status = response.status().as_u16();
            let text = response.text().await.unwrap_or_default();
            Err(ChannelError::Send(format!("HTTP {status}: {text}")))
        }
    }

    /// Make a PATCH request.
    async fn patch<T: for<'de> Deserialize<'de>>(
        &self,
        path: &str,
        body: impl Serialize,
    ) -> Result<T, ChannelError> {
        let response = self
            .client
            .patch(format!("{DISCORD_API_BASE}{path}"))
            .header("Authorization", self.auth_header())
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| ChannelError::Send(e.to_string()))?;

        self.handle_response(response).await
    }

    /// Make a DELETE request.
    async fn delete(&self, path: &str) -> Result<(), ChannelError> {
        let response = self
            .client
            .delete(format!("{DISCORD_API_BASE}{path}"))
            .header("Authorization", self.auth_header())
            .send()
            .await
            .map_err(|e| ChannelError::Send(e.to_string()))?;

        if response.status().is_success() || response.status().as_u16() == 204 {
            Ok(())
        } else {
            let status = response.status().as_u16();
            let text = response.text().await.unwrap_or_default();
            Err(ChannelError::Send(format!("HTTP {status}: {text}")))
        }
    }

    /// Make a PUT request without body.
    async fn put_empty(&self, path: &str) -> Result<(), ChannelError> {
        let response = self
            .client
            .put(format!("{DISCORD_API_BASE}{path}"))
            .header("Authorization", self.auth_header())
            .header("Content-Length", "0")
            .send()
            .await
            .map_err(|e| ChannelError::Send(e.to_string()))?;

        if response.status().is_success() || response.status().as_u16() == 204 {
            Ok(())
        } else {
            let status = response.status().as_u16();
            let text = response.text().await.unwrap_or_default();
            Err(ChannelError::Send(format!("HTTP {status}: {text}")))
        }
    }

    /// Handle API response.
    async fn handle_response<T: for<'de> Deserialize<'de>>(
        &self,
        response: reqwest::Response,
    ) -> Result<T, ChannelError> {
        let status = response.status();

        if status.as_u16() == 429 {
            return Err(ChannelError::RateLimited);
        }

        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(ChannelError::Send(format!("HTTP {}: {}", status.as_u16(), text)));
        }

        response
            .json()
            .await
            .map_err(|e| ChannelError::Send(format!("JSON parse error: {e}")))
    }

    // ========================================================================
    // Message Operations
    // ========================================================================

    /// Send a message to a channel.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelError::RateLimited`] if Discord answers HTTP 429, and
    /// [`ChannelError::Send`] if the request cannot be sent, Discord answers any
    /// other non-success status, or the response body is not a valid Discord message object.
    pub async fn send_message(
        &self,
        channel_id: &str,
        content: &str,
        reply_to: Option<&str>,
    ) -> Result<DiscordMessage, ChannelError> {
        #[derive(Serialize)]
        struct CreateMessage<'a> {
            content: &'a str,
            #[serde(skip_serializing_if = "Option::is_none")]
            message_reference: Option<MessageReference<'a>>,
        }

        #[derive(Serialize)]
        struct MessageReference<'a> {
            message_id: &'a str,
        }

        let body = CreateMessage {
            content,
            message_reference: reply_to.map(|id| MessageReference { message_id: id }),
        };

        self.post(&format!("/channels/{channel_id}/messages"), body).await
    }

    /// Send an embed to a channel.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelError::RateLimited`] if Discord answers HTTP 429, and
    /// [`ChannelError::Send`] if the request cannot be sent, Discord answers any
    /// other non-success status, or the response body is not a valid Discord message object.
    pub async fn send_embed(
        &self,
        channel_id: &str,
        embed: DiscordEmbed,
    ) -> Result<DiscordMessage, ChannelError> {
        #[derive(Serialize)]
        struct CreateMessage {
            embeds: Vec<DiscordEmbed>,
        }

        self.post(
            &format!("/channels/{channel_id}/messages"),
            CreateMessage { embeds: vec![embed] },
        )
        .await
    }

    /// Edit a message.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelError::RateLimited`] if Discord answers HTTP 429, and
    /// [`ChannelError::Send`] if the request cannot be sent, Discord answers any
    /// other non-success status, or the response body is not a valid Discord message object.
    pub async fn edit_message(
        &self,
        channel_id: &str,
        message_id: &str,
        content: &str,
    ) -> Result<DiscordMessage, ChannelError> {
        #[derive(Serialize)]
        struct EditMessage<'a> {
            content: &'a str,
        }

        self.patch(
            &format!("/channels/{channel_id}/messages/{message_id}"),
            EditMessage { content },
        )
        .await
    }

    /// Delete a message.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelError::Send`] if the request cannot be sent or Discord
    /// answers a non-success status (a 429 rate limit included).
    pub async fn delete_message(&self, channel_id: &str, message_id: &str) -> Result<(), ChannelError> {
        self.delete(&format!("/channels/{channel_id}/messages/{message_id}"))
            .await
    }

    // ========================================================================
    // Reaction Operations
    // ========================================================================

    /// Add a reaction to a message.
    ///
    /// For custom emoji: `name:id` format
    /// For unicode emoji: URL-encoded emoji
    ///
    /// # Errors
    ///
    /// Returns [`ChannelError::Send`] if the request cannot be sent or Discord
    /// answers a non-success status (a 429 rate limit included).
    pub async fn add_reaction(
        &self,
        channel_id: &str,
        message_id: &str,
        emoji: &str,
    ) -> Result<(), ChannelError> {
        let encoded = urlencoding::encode(emoji);
        self.put_empty(&format!(
            "/channels/{channel_id}/messages/{message_id}/reactions/{encoded}/@me"
        ))
        .await
    }

    /// Remove own reaction from a message.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelError::Send`] if the request cannot be sent or Discord
    /// answers a non-success status (a 429 rate limit included).
    pub async fn remove_reaction(
        &self,
        channel_id: &str,
        message_id: &str,
        emoji: &str,
    ) -> Result<(), ChannelError> {
        let encoded = urlencoding::encode(emoji);
        self.delete(&format!(
            "/channels/{channel_id}/messages/{message_id}/reactions/{encoded}/@me"
        ))
        .await
    }

    // ========================================================================
    // Thread Operations
    // ========================================================================

    /// Create a thread from a message.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelError::RateLimited`] if Discord answers HTTP 429, and
    /// [`ChannelError::Send`] if the request cannot be sent, Discord answers any
    /// other non-success status, or the response body is not a valid Discord channel object.
    pub async fn create_message_thread(
        &self,
        channel_id: &str,
        message_id: &str,
        name: &str,
        auto_archive_duration: Option<u32>,
    ) -> Result<DiscordChannelInfo, ChannelError> {
        #[derive(Serialize)]
        struct CreateThread<'a> {
            name: &'a str,
            #[serde(skip_serializing_if = "Option::is_none")]
            auto_archive_duration: Option<u32>,
        }

        self.post(
            &format!("/channels/{channel_id}/messages/{message_id}/threads"),
            CreateThread {
                name,
                auto_archive_duration,
            },
        )
        .await
    }

    // ========================================================================
    // Channel Operations
    // ========================================================================

    /// Get channel info.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelError::RateLimited`] if Discord answers HTTP 429, and
    /// [`ChannelError::Send`] if the request cannot be sent, Discord answers any
    /// other non-success status, or the response body is not a valid Discord channel object.
    pub async fn get_channel(&self, channel_id: &str) -> Result<DiscordChannelInfo, ChannelError> {
        self.get(&format!("/channels/{channel_id}")).await
    }

    /// Trigger typing indicator.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelError::Send`] if the request cannot be sent or Discord
    /// answers a non-success status (a 429 rate limit included).
    pub async fn trigger_typing(&self, channel_id: &str) -> Result<(), ChannelError> {
        self.post_empty(&format!("/channels/{channel_id}/typing"), serde_json::json!({}))
            .await
    }

    /// Pin a message.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelError::Send`] if the request cannot be sent or Discord
    /// answers a non-success status (a 429 rate limit included).
    pub async fn pin_message(&self, channel_id: &str, message_id: &str) -> Result<(), ChannelError> {
        self.put_empty(&format!("/channels/{channel_id}/pins/{message_id}"))
            .await
    }

    /// Unpin a message.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelError::Send`] if the request cannot be sent or Discord
    /// answers a non-success status (a 429 rate limit included).
    pub async fn unpin_message(&self, channel_id: &str, message_id: &str) -> Result<(), ChannelError> {
        self.delete(&format!("/channels/{channel_id}/pins/{message_id}"))
            .await
    }

    // ========================================================================
    // Interaction Responses (for slash commands)
    // ========================================================================

    /// Send a followup message to an interaction.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelError::RateLimited`] if Discord answers HTTP 429, and
    /// [`ChannelError::Send`] if the request cannot be sent, Discord answers any
    /// other non-success status, or the response body is not a valid Discord message object.
    pub async fn send_followup(
        &self,
        interaction_token: &str,
        content: &str,
    ) -> Result<DiscordMessage, ChannelError> {
        #[derive(Serialize)]
        struct FollowupMessage<'a> {
            content: &'a str,
        }

        self.post(
            &format!("/webhooks/{}/{interaction_token}", self.application_id),
            FollowupMessage { content },
        )
        .await
    }

    /// Edit the original interaction response.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelError::RateLimited`] if Discord answers HTTP 429, and
    /// [`ChannelError::Send`] if the request cannot be sent, Discord answers any
    /// other non-success status, or the response body is not a valid Discord message object.
    pub async fn edit_original(
        &self,
        interaction_token: &str,
        content: &str,
    ) -> Result<DiscordMessage, ChannelError> {
        #[derive(Serialize)]
        struct EditMessage<'a> {
            content: &'a str,
        }

        self.patch(
            &format!(
                "/webhooks/{}/{interaction_token}/messages/@original",
                self.application_id
            ),
            EditMessage { content },
        )
        .await
    }
}

// ============================================================================
// Discord API Types
// ============================================================================

/// Discord message object
#[derive(Debug, Clone, Deserialize)]
pub struct DiscordMessage {
    pub id: String,
    pub channel_id: String,
    pub author: DiscordUser,
    pub content: String,
    pub timestamp: String,
    #[serde(default)]
    pub edited_timestamp: Option<String>,
    #[serde(default)]
    pub mention_everyone: bool,
    #[serde(default)]
    pub mentions: Vec<DiscordUser>,
    #[serde(default)]
    pub attachments: Vec<DiscordAttachment>,
    #[serde(default)]
    pub embeds: Vec<DiscordEmbed>,
    #[serde(default)]
    pub reactions: Vec<DiscordReaction>,
    #[serde(default)]
    pub pinned: bool,
    #[serde(rename = "type")]
    pub message_type: u8,
}

/// Discord user object
#[derive(Debug, Clone, Deserialize)]
pub struct DiscordUser {
    pub id: String,
    pub username: String,
    #[serde(default)]
    pub discriminator: Option<String>,
    #[serde(default)]
    pub global_name: Option<String>,
    #[serde(default)]
    pub avatar: Option<String>,
    #[serde(default)]
    pub bot: bool,
}

/// Discord attachment
#[derive(Debug, Clone, Deserialize)]
pub struct DiscordAttachment {
    pub id: String,
    pub filename: String,
    pub size: u64,
    pub url: String,
    #[serde(default)]
    pub content_type: Option<String>,
}

/// Discord embed
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DiscordEmbed {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub footer: Option<EmbedFooter>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<EmbedMedia>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thumbnail: Option<EmbedMedia>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub author: Option<EmbedAuthor>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fields: Vec<EmbedField>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedFooter {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub icon_url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedMedia {
    pub url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedAuthor {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub icon_url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedField {
    pub name: String,
    pub value: String,
    #[serde(default)]
    pub inline: bool,
}

/// Discord reaction
#[derive(Debug, Clone, Deserialize)]
pub struct DiscordReaction {
    pub count: u32,
    pub me: bool,
    pub emoji: DiscordEmoji,
}

/// Discord emoji
#[derive(Debug, Clone, Deserialize)]
pub struct DiscordEmoji {
    pub id: Option<String>,
    pub name: Option<String>,
}

/// Discord channel info
#[derive(Debug, Clone, Deserialize)]
pub struct DiscordChannelInfo {
    pub id: String,
    #[serde(rename = "type")]
    pub channel_type: u8,
    pub guild_id: Option<String>,
    pub name: Option<String>,
    pub topic: Option<String>,
}

// ============================================================================
// Channel Trait Implementation
// ============================================================================

#[async_trait]
impl Channel for DiscordChannel {
    fn provider(&self) -> String {
        "discord".to_string()
    }

    fn capabilities(&self) -> ChannelCapabilities {
        ChannelCapabilities {
            features: ChannelFeatures::REACTIONS
                | ChannelFeatures::REPLIES
                | ChannelFeatures::EDITS
                | ChannelFeatures::THREADS
                | ChannelFeatures::IMAGES
                | ChannelFeatures::DOCUMENTS
                | ChannelFeatures::PINS
                | ChannelFeatures::TYPING
                | ChannelFeatures::MARKDOWN,
            max_message_length: Some(2000),
        }
    }

    async fn send(&self, message: OutgoingMessage) -> Result<String, ChannelError> {
        let channel_id = &message.channel.id;

        let result = match message.content {
            MessageContent::Text { text } => {
                // Discord has 2000 char limit - split if needed
                if text.len() <= 2000 {
                    self.send_message(channel_id, &text, message.reply_to.as_deref().map(crate::native_reply_id))
                        .await?
                } else {
                    // Send first chunk, then rest as followups
                    let chunks: Vec<&str> = text
                        .as_bytes()
                        .chunks(2000)
                        .map(|c| std::str::from_utf8(c).unwrap_or(""))
                        .collect();

                    let mut last_msg = self
                        .send_message(channel_id, chunks[0], message.reply_to.as_deref().map(crate::native_reply_id))
                        .await?;

                    for chunk in &chunks[1..] {
                        last_msg = self.send_message(channel_id, chunk, None).await?;
                    }
                    last_msg
                }
            }
            MessageContent::Image { url, caption } => {
                let embed = DiscordEmbed {
                    image: Some(EmbedMedia { url }),
                    description: caption,
                    ..Default::default()
                };
                self.send_embed(channel_id, embed).await?
            }
            _ => {
                // For other content types, send as text description
                let text = "[Unsupported content type]".to_string();
                self.send_message(channel_id, &text, message.reply_to.as_deref().map(crate::native_reply_id))
                    .await?
            }
        };

        Ok(result.id)
    }

    async fn react(&self, message_id: &str, emoji: &str) -> Result<(), ChannelError> {
        // message_id format: "channel_id:message_id"
        let parts: Vec<&str> = message_id.split(':').collect();
        if parts.len() != 2 {
            return Err(ChannelError::Send(
                "Invalid message ID format (expected channel_id:message_id)".to_string(),
            ));
        }
        self.add_reaction(parts[0], parts[1], emoji).await
    }

    async fn unreact(&self, message_id: &str, emoji: &str) -> Result<(), ChannelError> {
        let parts: Vec<&str> = message_id.split(':').collect();
        if parts.len() != 2 {
            return Err(ChannelError::Send(
                "Invalid message ID format (expected channel_id:message_id)".to_string(),
            ));
        }
        self.remove_reaction(parts[0], parts[1], emoji).await
    }

    async fn edit(&self, message_id: &str, content: MessageContent) -> Result<(), ChannelError> {
        let parts: Vec<&str> = message_id.split(':').collect();
        if parts.len() != 2 {
            return Err(ChannelError::Send(
                "Invalid message ID format (expected channel_id:message_id)".to_string(),
            ));
        }

        let MessageContent::Text { text } = content else {
            return Err(ChannelError::Send("Can only edit text messages".to_string()));
        };

        self.edit_message(parts[0], parts[1], &text).await?;
        Ok(())
    }

    async fn delete(&self, message_id: &str) -> Result<(), ChannelError> {
        let parts: Vec<&str> = message_id.split(':').collect();
        if parts.len() != 2 {
            return Err(ChannelError::Send(
                "Invalid message ID format (expected channel_id:message_id)".to_string(),
            ));
        }
        self.delete_message(parts[0], parts[1]).await
    }

    async fn pin(&self, message_id: &str) -> Result<(), ChannelError> {
        let parts: Vec<&str> = message_id.split(':').collect();
        if parts.len() != 2 {
            return Err(ChannelError::Send(
                "Invalid message ID format (expected channel_id:message_id)".to_string(),
            ));
        }
        self.pin_message(parts[0], parts[1]).await
    }

    async fn unpin(&self, message_id: &str) -> Result<(), ChannelError> {
        let parts: Vec<&str> = message_id.split(':').collect();
        if parts.len() != 2 {
            return Err(ChannelError::Send(
                "Invalid message ID format (expected channel_id:message_id)".to_string(),
            ));
        }
        self.unpin_message(parts[0], parts[1]).await
    }

    async fn create_thread(&self, message_id: &str, name: &str) -> Result<ThreadInfo, ChannelError> {
        let parts: Vec<&str> = message_id.split(':').collect();
        if parts.len() != 2 {
            return Err(ChannelError::Send(
                "Invalid message ID format (expected channel_id:message_id)".to_string(),
            ));
        }

        let thread = self.create_message_thread(parts[0], parts[1], name, Some(1440)).await?;
        Ok(ThreadInfo {
            id: thread.id,
            name: thread.name,
            message_count: None,
        })
    }

    async fn send_typing(&self, channel_id: &ChannelId) -> Result<(), ChannelError> {
        self.trigger_typing(&channel_id.id).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discord_channel_creation() {
        let channel = DiscordChannel::new("token", "app_id");
        assert_eq!(channel.provider(), "discord");
    }

    #[test]
    fn test_capabilities() {
        let channel = DiscordChannel::new("token", "app_id");
        let caps = channel.capabilities();
        assert!(caps.supports_reactions());
        assert!(caps.supports_threads());
        assert!(caps.supports_edits());
        assert_eq!(caps.max_message_length, Some(2000));
    }
}
