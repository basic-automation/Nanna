//! Serving a server's elicitation requests (MCP 2026-07-28, multi round-trip).
//!
//! A modern server that needs something from the user mid-call answers the
//! call with `resultType: "input_required"` and a map of `inputRequests`; the
//! client gathers the answers and retries the same call with
//! `inputResponses` (and the server's opaque `requestState`, echoed verbatim).
//!
//! This client serves **form-mode elicitation only** — the one kind that is a
//! question for the user — and only when an [`Elicitor`] is installed (the
//! daemon's puts the question into the conversation through `ask_user`).
//! Sampling and roots are never declared, so a conforming server never asks
//! for them. The user answers in free text; [`answer_to_content`] maps that
//! onto the requested schema, and a server that still lacks something asks
//! again (bounded by [`MRTR_ROUNDS_MAX`]).
//!
//! Source: <https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr>.

use crate::{McpError, Result};
use async_trait::async_trait;
use serde_json::{Map, Value, json};
use std::fmt::Write as _;

/// Most retries of one call. A server may re-ask when an answer did not fit;
/// three rounds covers a question, a correction and a confirmation, and stops
/// a server that asks forever.
pub const MRTR_ROUNDS_MAX: usize = 3;

/// Most `inputRequests` served in one round.
const INPUT_REQUESTS_MAX: usize = 8;

/// Longest server message put to the user, in characters.
const ELICIT_MESSAGE_CHARS_MAX: usize = 1000;

/// The capability a client with an elicitor declares.
#[must_use]
pub fn elicitation_capability() -> Value {
    json!({ "elicitation": { "form": {} } })
}

/// Puts a server's question to the user and returns their reply.
#[async_trait]
pub trait Elicitor: Send + Sync {
    /// Ask `question`; `None` when no answer came (no conversation to ask
    /// in, or the user did not reply in time).
    async fn ask(&self, question: &str) -> Option<String>;

    /// The longest one [`Self::ask`] waits for a reply. A tool call that
    /// may ask the user is given this much more time per question, so the
    /// user's think-time is not cut off by a deadline sized for the server.
    fn answer_wait_max(&self) -> std::time::Duration;
}

/// The longest one tool call may take, end to end.
///
/// Every request takes at most [`crate::MCP_REQUEST_TIMEOUT`], and — when an
/// elicitor answering within `answer_wait_max` is installed — there are up to
/// [`MRTR_ROUNDS_MAX`] retries, each after at most [`INPUT_REQUESTS_MAX`]
/// questions to the user. Without an
/// elicitor there are no retries, so it is one request. Pure.
///
/// The call's deadline used to be one request's (60 s + margin) whatever
/// happened inside it, so a question to the user that waited up to 600 s
/// was cut off at about 75 s and the answer never reached the server.
#[must_use]
pub fn tool_call_time_max(answer_wait_max: Option<std::time::Duration>) -> std::time::Duration {
    let Some(answer_wait_max) = answer_wait_max else {
        return crate::MCP_REQUEST_TIMEOUT;
    };
    let rounds = u32::try_from(MRTR_ROUNDS_MAX).unwrap_or(u32::MAX);
    let questions = u32::try_from(INPUT_REQUESTS_MAX).unwrap_or(u32::MAX);
    let requests = crate::MCP_REQUEST_TIMEOUT.saturating_mul(rounds.saturating_add(1));
    let asking = answer_wait_max.saturating_mul(rounds.saturating_mul(questions));
    let total = requests.saturating_add(asking);
    debug_assert!(total >= crate::MCP_REQUEST_TIMEOUT);
    total
}

/// The question the user sees for one form request. Pure.
#[must_use]
pub fn elicitation_question(server: &str, message: &str, schema: &Value) -> String {
    let message: String = message.chars().take(ELICIT_MESSAGE_CHARS_MAX).collect();
    let fields = schema_fields(schema);
    let mut question = format!("The MCP server `{server}` asks: {}", message.trim());
    if fields.len() > 1 {
        question.push_str("\nPlease answer each, one per line as `name: value`:");
        for field in &fields {
            let _ = write!(question, "\n- {}", field.describe());
        }
    } else if let Some(field) = fields.first()
        && let Some(options) = &field.options
    {
        let _ = write!(question, " (one of: {})", options.join(", "));
    }
    question
}

/// One property of a requested schema.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Field {
    name: String,
    kind: String,
    required: bool,
    description: Option<String>,
    options: Option<Vec<String>>,
}

impl Field {
    fn describe(&self) -> String {
        let mut text = format!("{} ({}", self.name, self.kind);
        if self.required {
            text.push_str(", required");
        }
        text.push(')');
        if let Some(options) = &self.options {
            let _ = write!(text, " one of: {}", options.join(", "));
        }
        if let Some(description) = &self.description {
            let _ = write!(text, " — {description}");
        }
        text
    }
}

/// The flat properties of a form schema (the spec restricts form schemas to
/// flat objects of primitives). Pure.
fn schema_fields(schema: &Value) -> Vec<Field> {
    let required: Vec<&str> = schema
        .get("required")
        .and_then(Value::as_array)
        .map_or_default(|r| r.iter().filter_map(Value::as_str).collect());
    let Some(properties) = schema.get("properties").and_then(Value::as_object) else {
        return Vec::new();
    };
    properties
        .iter()
        .take(INPUT_REQUESTS_MAX * 4)
        .map(|(name, property)| Field {
            name: name.clone(),
            kind: property
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("string")
                .to_string(),
            required: required.contains(&name.as_str()),
            description: property
                .get("description")
                .and_then(Value::as_str)
                .map(str::to_string),
            options: property
                .get("enum")
                .and_then(Value::as_array)
                .map(|values| {
                    values
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect()
                }),
        })
        .collect()
}

/// Map a free-text reply onto the requested schema's properties. Pure.
///
/// One property takes the whole reply. Several take `name: value` (or
/// `name = value`) lines, or a JSON object. Values are coerced to the
/// property's type; one that does not coerce, or an enum value that matches
/// no option, is left out — the server then asks again for what it lacks.
#[must_use]
pub fn answer_to_content(schema: &Value, answer: &str) -> Value {
    let fields = schema_fields(schema);
    let answer = answer.trim();
    let mut content = Map::new();
    if let [field] = fields.as_slice() {
        if let Some(value) = coerce(field, answer) {
            content.insert(field.name.clone(), value);
        }
        return Value::Object(content);
    }
    let pairs = parse_pairs(answer);
    for field in &fields {
        let raw = pairs
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(&field.name))
            .map(|(_, value)| value.as_str());
        if let Some(value) = raw.and_then(|raw| coerce(field, raw)) {
            content.insert(field.name.clone(), value);
        }
    }
    debug_assert!(content.len() <= fields.len());
    Value::Object(content)
}

/// `name: value` / `name = value` lines, or a JSON object's entries. Pure.
fn parse_pairs(answer: &str) -> Vec<(String, String)> {
    if let Ok(Value::Object(object)) = serde_json::from_str::<Value>(answer) {
        return object
            .into_iter()
            .map(|(k, v)| (k, v.as_str().map_or_else(|| v.to_string(), str::to_string)))
            .collect();
    }
    answer
        .lines()
        .filter_map(|line| {
            let line = line.trim().trim_start_matches(['-', '*']).trim();
            let (key, value) = line.split_once(':').or_else(|| line.split_once('='))?;
            Some((key.trim().to_string(), value.trim().to_string()))
        })
        .collect()
}

/// Coerce one raw reply to a field's type. Pure.
fn coerce(field: &Field, raw: &str) -> Option<Value> {
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }
    if let Some(options) = &field.options {
        return options
            .iter()
            .find(|option| option.eq_ignore_ascii_case(raw))
            .map(|option| Value::String(option.clone()));
    }
    match field.kind.as_str() {
        "integer" => raw.parse::<i64>().ok().map(Value::from),
        "number" => raw
            .parse::<f64>()
            .ok()
            .and_then(|n| serde_json::Number::from_f64(n).map(Value::Number)),
        "boolean" => match raw.to_ascii_lowercase().as_str() {
            "yes" | "y" | "true" | "ok" | "sure" => Some(Value::Bool(true)),
            "no" | "n" | "false" => Some(Value::Bool(false)),
            _ => None,
        },
        _ => Some(Value::String(raw.to_string())),
    }
}

/// One form request out of an `input_required` result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FormRequest {
    /// The server's key for it, echoed in `inputResponses`.
    pub key: String,
    pub message: String,
    pub schema: Value,
}

/// The form requests in an `input_required` result. Pure.
///
/// # Errors
///
/// Returns [`McpError::Protocol`] for a request this client cannot serve
/// (sampling, roots, URL-mode elicitation — none of which it declared, so a
/// conforming server does not send them) or for more requests than one
/// round serves.
pub fn form_requests(result: &Value) -> Result<Vec<FormRequest>> {
    let Some(requests) = result.get("inputRequests").and_then(Value::as_object) else {
        return Ok(Vec::new());
    };
    if requests.len() > INPUT_REQUESTS_MAX {
        return Err(McpError::Protocol(format!(
            "the MCP server asked {} questions at once; at most {INPUT_REQUESTS_MAX} are served",
            requests.len()
        )));
    }
    let mut forms = Vec::with_capacity(requests.len());
    for (key, request) in requests {
        let method = request.get("method").and_then(Value::as_str).unwrap_or("");
        let params = request.get("params").cloned().unwrap_or(Value::Null);
        let mode = params.get("mode").and_then(Value::as_str).unwrap_or("form");
        if method != "elicitation/create" || mode != "form" {
            return Err(McpError::Protocol(format!(
                "the MCP server asked for `{method}` ({mode}), which this client does not serve"
            )));
        }
        forms.push(FormRequest {
            key: key.clone(),
            message: params
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            schema: params
                .get("requestedSchema")
                .cloned()
                .unwrap_or_else(|| json!({})),
        });
    }
    debug_assert!(forms.len() <= INPUT_REQUESTS_MAX);
    Ok(forms)
}

/// The `ElicitResult` for one form: `accept` with the mapped content when
/// the user answered, `cancel` when no answer came. Pure.
#[must_use]
pub fn elicit_result(form: &FormRequest, answer: Option<&str>) -> Value {
    answer.map_or_else(
        || json!({ "action": "cancel" }),
        |answer| json!({ "action": "accept", "content": answer_to_content(&form.schema, answer) }),
    )
}

/// The retry's params: the original's, plus `inputResponses`, plus the
/// server's `requestState` echoed exactly — or absent when the server sent
/// none (the spec forbids inventing one). Pure.
#[must_use]
pub fn retry_params(
    original: &Value,
    responses: Map<String, Value>,
    request_state: Option<&Value>,
) -> Value {
    let mut params = original.as_object().cloned().unwrap_or_default();
    params.insert("inputResponses".to_string(), Value::Object(responses));
    params.remove("requestState");
    if let Some(state) = request_state {
        params.insert("requestState".to_string(), state.clone());
    }
    Value::Object(params)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Without an elicitor a call is one request; with one, every question
    /// the call may ask gets the user's whole wait, so an answer given after
    /// a minute still reaches the server.
    #[test]
    fn a_call_that_may_ask_the_user_gets_the_users_time() {
        use std::time::Duration;
        assert_eq!(tool_call_time_max(None), crate::MCP_REQUEST_TIMEOUT);
        let wait = Duration::from_secs(600);
        let with_asking = tool_call_time_max(Some(wait));
        // 4 requests of 60 s, plus 3 rounds of at most 8 questions of 10 min.
        assert_eq!(with_asking, Duration::from_mins(4 + 3 * 8 * 10));
        assert!(with_asking > wait, "one whole answer fits inside the call");
        assert_eq!(
            tool_call_time_max(Some(Duration::MAX)),
            Duration::MAX,
            "saturates rather than overflowing"
        );
    }

    fn color_schema() -> Value {
        json!({ "type": "object", "properties": {
            "color": { "type": "string", "description": "a color" } }, "required": ["color"] })
    }

    #[test]
    fn a_single_field_takes_the_whole_reply() {
        assert_eq!(
            answer_to_content(&color_schema(), "  teal \n"),
            json!({ "color": "teal" })
        );
        let question = elicitation_question("paint", "Favorite color?", &color_schema());
        assert_eq!(question, "The MCP server `paint` asks: Favorite color?");
    }

    #[test]
    fn several_fields_come_from_lines_or_json_and_are_coerced() {
        let schema = json!({ "type": "object", "properties": {
            "name": { "type": "string" },
            "age": { "type": "integer" },
            "subscribe": { "type": "boolean" },
            "tier": { "type": "string", "enum": ["free", "pro"] }
        }, "required": ["name"] });
        let question = elicitation_question("crm", "Sign-up details", &schema);
        assert!(question.contains("- name (string, required)"), "{question}");
        assert!(
            question.contains("tier (string) one of: free, pro"),
            "{question}"
        );

        let lines = "name: Ada\nAge = 36\n- subscribe: yes\ntier: PRO";
        assert_eq!(
            answer_to_content(&schema, lines),
            json!({ "name": "Ada", "age": 36, "subscribe": true, "tier": "pro" })
        );
        let json_reply = r#"{"name": "Grace", "age": "not a number", "tier": "gold"}"#;
        assert_eq!(
            answer_to_content(&schema, json_reply),
            json!({ "name": "Grace" }),
            "values that do not coerce are left for the server to re-ask"
        );
    }

    #[test]
    fn only_form_elicitation_is_served() {
        let result = json!({ "resultType": "input_required", "inputRequests": {
            "color": { "method": "elicitation/create",
                       "params": { "mode": "form", "message": "Color?", "requestedSchema": color_schema() } } } });
        let forms = form_requests(&result).unwrap();
        assert_eq!(forms.len(), 1);
        assert_eq!(forms[0].key, "color");

        for (method, mode) in [
            ("sampling/createMessage", "form"),
            ("roots/list", "form"),
            ("elicitation/create", "url"),
        ] {
            let bad = json!({ "inputRequests": { "k": { "method": method, "params": { "mode": mode } } } });
            assert!(form_requests(&bad).is_err(), "{method} {mode}");
        }
        assert_eq!(
            form_requests(&json!({ "requestState": "s" })).unwrap(),
            Vec::new()
        );
    }

    #[test]
    fn the_retry_echoes_state_exactly_and_never_invents_it() {
        let form = FormRequest {
            key: "color".into(),
            message: String::new(),
            schema: color_schema(),
        };
        let mut responses = Map::new();
        responses.insert("color".into(), elicit_result(&form, Some("teal")));
        let original = json!({ "name": "favorite", "arguments": {} });
        let with_state = retry_params(&original, responses.clone(), Some(&json!("opaque-1")));
        assert_eq!(with_state["requestState"], "opaque-1");
        assert_eq!(
            with_state["inputResponses"]["color"],
            json!({ "action": "accept", "content": { "color": "teal" } })
        );
        assert_eq!(with_state["name"], "favorite");

        let stale = json!({ "name": "favorite", "requestState": "old" });
        let without = retry_params(&stale, responses, None);
        assert!(
            without.get("requestState").is_none(),
            "no state from the server, none sent"
        );

        assert_eq!(elicit_result(&form, None), json!({ "action": "cancel" }));
    }
}
