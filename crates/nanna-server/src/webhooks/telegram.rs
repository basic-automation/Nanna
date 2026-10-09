//! Telegram webhook handler

use crate::state::AppState;
use crate::webhooks::auth;
use axum::{Json, body::Bytes, extract::State, http::HeaderMap, http::StatusCode};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

// Re-used for linking messages to memories
const PROVIDER: &str = "telegram";

// Telegram API types - fields are part of the API spec even if not all are currently used

/// Telegram Update object (simplified)
#[derive(Debug, Deserialize)]
pub struct TelegramUpdate {
    pub update_id: i64,
    pub message: Option<TelegramMessage>,
#[serde(rename = "callback_query")]
    pub _callback_query: Option<CallbackQuery>,
    pub message_reaction: Option<MessageReactionUpdate>,
}

/// Telegram message reaction update
#[derive(Debug, Deserialize)]
pub struct MessageReactionUpdate {
    pub message_id: i64,
    pub chat: TelegramChat,
#[serde(rename = "user")]
    pub _user: Option<TelegramUser>,
#[serde(rename = "date")]
    pub _date: i64,
#[serde(rename = "old_reaction")]
    pub _old_reaction: Vec<ReactionType>,
    pub new_reaction: Vec<ReactionType>,
}

/// Reaction type (emoji or custom)
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReactionType {
    Emoji { emoji: String },
    CustomEmoji { _custom_emoji_id: String },
}

#[derive(Debug, Deserialize)]
pub struct TelegramMessage {
    pub message_id: i64,
    pub from: Option<TelegramUser>,
    pub chat: TelegramChat,
#[serde(rename = "date")]
    pub _date: i64,
    pub text: Option<String>,
#[serde(rename = "reply_to_message")]
    pub _reply_to_message: Option<Box<Self>>,
}

#[derive(Debug, Deserialize)]
pub struct TelegramUser {
    pub id: i64,
#[serde(rename = "first_name")]
    pub _first_name: String,
#[serde(rename = "last_name")]
    pub _last_name: Option<String>,
    pub username: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct TelegramChat {
    pub id: i64,
    #[serde(rename = "type")]
    pub _chat_type: String,
#[serde(rename = "title")]
    pub _title: Option<String>,
#[serde(rename = "username")]
    pub _username: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct CallbackQuery {
#[serde(rename = "id")]
    pub _id: String,
#[serde(rename = "from")]
    pub _from: TelegramUser,
#[serde(rename = "message")]
    pub _message: Option<TelegramMessage>,
#[serde(rename = "data")]
    pub _data: Option<String>,
}

/// Telegram API response wrapper
#[derive(Serialize)]
pub struct TelegramResponse {
    method: &'static str,
    chat_id: i64,
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    reply_to_message_id: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parse_mode: Option<&'static str>,
}

/// Handle incoming Telegram webhook
///
/// Origin proof is the `secret_token` given to Telegram's `setWebhook`, which
/// it echoes as `X-Telegram-Bot-Api-Secret-Token` on every POST. This route is
/// a fixed path with no bot token in it, so that header is the ONLY thing
/// separating Telegram from any other caller who can reach the port — and the
/// payload drives the agent, which runs tools. No secret configured means no
/// service (see `auth::refuse_unconfigured`).
pub async fn handle(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Json<TelegramResponse>, StatusCode> {
    let Some(secret) = auth::configured(state.telegram_webhook_secret.as_ref()) else {
        return Err(auth::refuse_unconfigured(
            "Telegram",
            "channels.telegram.webhook_secret",
        ));
    };
    let presented = headers
        .get("X-Telegram-Bot-Api-Secret-Token")
        .and_then(|v| v.to_str().ok());
    if !auth::secret_matches(secret, presented) {
        warn!("Telegram webhook: invalid secret token");
        return Err(StatusCode::UNAUTHORIZED);
    }

    // Parse only now. `Bytes` rather than `Json<T>` so the credential check
    // above runs first — see `auth::parse_authenticated_body`.
    let update: TelegramUpdate = auth::parse_authenticated_body("Telegram", &body)?;

    debug!("Received Telegram update: {:?}", update.update_id);

    // Handle message reactions (for memory feedback)
    if let Some(reaction) = update.message_reaction {
        handle_reaction(&state, &reaction).await;
        return Err(StatusCode::OK); // Acknowledge
    }

    // Extract message
    // New messages only (an edit is acknowledged, not answered again).
    let message = update.message.ok_or(StatusCode::OK)?; // No message, just acknowledge

    let text = message.text.as_deref().unwrap_or("");
    if text.is_empty() {
        return Err(StatusCode::OK); // No text, acknowledge
    }

    let chat_id = message.chat.id;
    let user_id = message.from.as_ref().map_or(0, |u| u.id);
    let username = message.from.as_ref().and_then(|u| u.username.clone());
    let session_id = format!("telegram:{chat_id}:{user_id}");

    info!(
        "Telegram message from {} (@{}) in {}: {}",
        user_id,
        username.as_deref().unwrap_or("unknown"),
        chat_id,
        text.chars().take(50).collect::<String>()
    );

    // Build system prompt for this user
    let system_prompt = format!(
        "You are Nanna — moon god of the digital realm.\n\
         You're chatting on Telegram with user {} (@{}).\n\
         Be helpful, concise, and conversational.",
        user_id,
        username.as_deref().unwrap_or("unknown")
    );

    // Process message (with memory extraction if enabled)
    let response_text = match state.process_message(&session_id, text, Some(&system_prompt)).await {
        Ok(text) => text,
        Err(e) => {
            warn!("Error processing Telegram message: {}", e);
            "Sorry, I encountered an error processing your message.".to_string()
        }
    };

    // Link this message to session for reaction-based feedback
    // We use the reply message ID since that's what reactions will be on
    let message_key = format!("{}:{}:{}", PROVIDER, chat_id, message.message_id);
    state.link_message_to_session(&message_key, &session_id).await;

    // Reply using webhook response method
    Ok(Json(TelegramResponse {
        method: "sendMessage",
        chat_id,
        text: response_text,
        reply_to_message_id: Some(message.message_id),
        parse_mode: Some("Markdown"),
    }))
}

/// Handle Telegram message reaction (maps to memory feedback)
async fn handle_reaction(state: &AppState, reaction: &MessageReactionUpdate) {
    // Build message ID key
    let message_key = format!("telegram:{}:{}", reaction.chat.id, reaction.message_id);
    
    // Check if any positive/negative reactions were added
    for r in &reaction.new_reaction {
        let (emoji, positive) = match r {
            ReactionType::Emoji { emoji } => {
                // Map common reaction emojis to positive/negative feedback
                let positive = matches!(
                    emoji.as_str(),
                    "👍" | "❤️" | "🔥" | "👏" | "🎉" | "💯" | "✅" | "⭐" | "🙏" | "😍"
                );
                let negative = matches!(
                    emoji.as_str(),
                    "👎" | "😡" | "💩" | "❌" | "🤮" | "😤" | "🙄"
                );
                
                if positive {
                    (emoji.clone(), true)
                } else if negative {
                    (emoji.clone(), false)
                } else {
                    continue; // Neutral reaction, skip
                }
            }
            ReactionType::CustomEmoji { .. } => continue, // Skip custom emojis
        };
        
        info!(
            "Telegram reaction {} on message {} (positive: {})",
            emoji, message_key, positive
        );
        
        state.record_message_feedback(&message_key, positive).await;
    }
}
