//! Which messages a session read returns: its opening or its latest exchange.
//!
//! `get_by_session` is oldest-first with a `LIMIT`, so it returns a session's
//! FIRST messages. Two callers wanted the latest ones and silently got the
//! opening instead: the CLI's resume (a long session came back without its
//! recent context) and the REST history route.

#![warn(clippy::all)]
#![warn(clippy::pedantic, clippy::nursery)]

use nanna_storage::{NewMessage, Storage};

async fn session_with(count: usize) -> (Storage, Vec<i64>) {
    let storage = Storage::in_memory().await.expect("in-memory storage");
    storage
        .sessions()
        .create("s", "test", None)
        .await
        .expect("session");
    let mut ids = Vec::with_capacity(count);
    for index in 0..count {
        let role = if index % 2 == 0 { "user" } else { "assistant" };
        let row = storage
            .messages()
            .create(NewMessage {
                session_id: "s".to_string(),
                role: role.to_string(),
                content: format!("message {index}"),
                content_type: "text".to_string(),
                tool_use_id: None,
                tokens_in: None,
                tokens_out: None,
                metadata: None,
            })
            .await
            .expect("message");
        ids.push(row.id);
    }
    (storage, ids)
}

fn contents(messages: &[nanna_storage::Message]) -> Vec<&str> {
    messages.iter().map(|m| m.content.as_str()).collect()
}

/// All seven rows land inside one second, so `created_at` ties everywhere and
/// only the `id` tiebreak orders them.
#[tokio::test]
async fn the_recent_read_is_the_latest_messages_in_conversation_order() {
    let (storage, ids) = session_with(7).await;

    let recent = storage
        .messages()
        .get_recent_by_session("s", 3)
        .await
        .expect("recent");
    assert_eq!(contents(&recent), ["message 4", "message 5", "message 6"]);
    assert_eq!(recent.iter().map(|m| m.id).collect::<Vec<_>>(), ids[4..]);

    let opening = storage
        .messages()
        .get_by_session("s", 3)
        .await
        .expect("opening");
    assert_eq!(contents(&opening), ["message 0", "message 1", "message 2"]);
}

#[tokio::test]
async fn a_limit_past_the_session_returns_every_message_once_in_order() {
    let (storage, _) = session_with(4).await;

    let recent = storage
        .messages()
        .get_recent_by_session("s", 50)
        .await
        .expect("recent");
    let opening = storage
        .messages()
        .get_by_session("s", 50)
        .await
        .expect("opening");
    assert_eq!(
        contents(&recent),
        ["message 0", "message 1", "message 2", "message 3"]
    );
    assert_eq!(contents(&recent), contents(&opening));

    let other = storage
        .messages()
        .get_recent_by_session("no-such-session", 50)
        .await
        .expect("empty read");
    assert!(other.is_empty());
}
