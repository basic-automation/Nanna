//! Tauri command handlers, grouped by feature area.

pub mod board;
pub mod channels;
pub mod chat;
pub mod memory;
pub mod scheduler;
pub mod sessions;
pub mod settings;
pub mod system;
pub mod tasks;
pub mod tools;
pub mod workspaces;

/// `Err` with the daemon's own words when `reply` is a refusal.
///
/// The daemon answers every request it parsed with an `Ok` reply and reports a
/// refusal inside it (`{"error": …, "message": …}`), so a command that checks
/// only the transport `Result` tells the page "done" for something the daemon
/// declined (a dream with no model configured, an edit of a memory a dream had
/// merged, activating a workspace another client closed).
pub(crate) fn daemon_refusal(reply: &serde_json::Value, what: &str) -> Result<(), String> {
    let Some(code) = reply.get("error") else {
        return Ok(());
    };
    let message = reply
        .get("message")
        .and_then(serde_json::Value::as_str)
        .map_or_else(
            || format!("{what}: {code}"),
            |message| format!("{what}: {message}"),
        );
    Err(message)
}

#[cfg(test)]
mod tests {
    use super::daemon_refusal;
    use serde_json::json;

    #[test]
    fn a_refusal_inside_an_ok_reply_is_an_error() {
        assert_eq!(
            daemon_refusal(&json!({"status": "updated"}), "Update"),
            Ok(())
        );
        assert_eq!(
            daemon_refusal(
                &json!({"error": "no_models", "message": "no model"}),
                "Dream"
            ),
            Err("Dream: no model".to_string())
        );
        assert_eq!(
            daemon_refusal(&json!({"error": "not_found", "id": "x"}), "Activate"),
            Err("Activate: \"not_found\"".to_string())
        );
    }
}
