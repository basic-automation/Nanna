//! Streamable HTTP transport, MCP revision `2026-07-28`, with the legacy
//! (2025-era, sessionful) shape as the fallback the dual-era client needs.
//!
//! Every JSON-RPC message is its own `POST` to one endpoint. The answer to a
//! request is either one JSON object or an SSE stream scoped to that request
//! (related notifications, then the response). The body's routing fields are
//! mirrored into headers — `MCP-Protocol-Version`, `Mcp-Method`, `Mcp-Name`,
//! and `Mcp-Param-*` for tool parameters annotated with `x-mcp-header` — so
//! intermediaries can route without parsing; a server rejects any mismatch
//! with `-32020`.
//!
//! Era detection is the client's (see [`crate::era`]); this transport only
//! reports what the server said, in a form the probe can classify:
//! a JSON-RPC error body is returned as a response whatever the HTTP status,
//! and any other non-success is [`McpError::HttpStatus`].
//!
//! Source: <https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http>.

use crate::era::META_PROTOCOL_VERSION;
use crate::transport::reply_to_server_request;
use crate::{
    JsonRpcNotification, JsonRpcRequest, JsonRpcResponse, ListChangedFlags, McpError, McpList,
    Result, Transport,
};
use async_trait::async_trait;
use base64::Engine as _;
use futures::StreamExt;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, warn};

/// How long one exchange may take, send to final response — the same bound
/// the stdio transport puts on a request ([`crate::MCP_REQUEST_TIMEOUT`]).
pub const HTTP_REQUEST_TIMEOUT: Duration = crate::MCP_REQUEST_TIMEOUT;

/// Upper bound on one SSE event's data, and on a plain JSON body.
///
/// A tool result is the largest thing a server sends; 16 MiB is far past any the
/// daemon would hand a model, and it stops a runaway stream from growing
/// memory without limit.
pub const HTTP_BODY_BYTES_MAX: usize = 16 * 1024 * 1024;

/// A `subscriptions/listen` stream that has been silent this long is
/// reopened.
///
/// Servers are encouraged to send keep-alive comments; one that
/// does not still gets its stream renewed, at the cost of one request.
pub const LISTEN_IDLE_TIMEOUT: Duration = Duration::from_secs(300);

/// First and last wait between listen reconnects (doubling in between).
const LISTEN_BACKOFF_MIN: Duration = Duration::from_secs(1);
const LISTEN_BACKOFF_MAX: Duration = Duration::from_secs(60);

/// Bound on how much of an error body is kept for the message.
const ERROR_BODY_BYTES_MAX: usize = 512;

/// Bound on the schema nodes the `x-mcp-header` walk visits. Tool schemas are
/// already gated by `schema_guard` (depth and size); this only keeps the walk
/// itself bounded if it is ever called on an ungated schema.
const SCHEMA_WALK_NODES_MAX: usize = 10_000;

/// The Base64 sentinel the spec wraps non-header-safe values in.
const SENTINEL_PREFIX: &str = "=?base64?";
const SENTINEL_SUFFIX: &str = "?=";

// ---------------------------------------------------------------------------
// Header values
// ---------------------------------------------------------------------------

/// Whether `value` can travel as a plain header value: visible ASCII, space
/// and tab only, no leading or trailing whitespace, and not itself shaped
/// like the sentinel (which would be ambiguous). Pure.
#[must_use]
pub fn is_header_safe(value: &str) -> bool {
    let visible = value
        .bytes()
        .all(|b| b == b'\t' || (0x20..=0x7E).contains(&b));
    let trimmed = value.trim_matches(|c| c == ' ' || c == '\t').len() == value.len();
    let sentinel_shaped = value.starts_with(SENTINEL_PREFIX) && value.ends_with(SENTINEL_SUFFIX);
    visible && trimmed && !sentinel_shaped
}

/// Encode a value for an `Mcp-Name` or `Mcp-Param-*` header: as-is when
/// header-safe, otherwise the Base64 sentinel over its UTF-8. Pure.
#[must_use]
pub fn encode_header_value(value: &str) -> String {
    if is_header_safe(value) {
        return value.to_string();
    }
    let encoded = base64::engine::general_purpose::STANDARD.encode(value.as_bytes());
    let wrapped = format!("{SENTINEL_PREFIX}{encoded}{SENTINEL_SUFFIX}");
    debug_assert!(wrapped.is_ascii());
    wrapped
}

/// The routing headers every request carries, from its body. Pure.
///
/// `MCP-Protocol-Version` comes from the body's `_meta` (the spec requires
/// the two to match), or from `legacy_version` — the revision a legacy
/// `initialize` negotiated — when the body has none. `initialize` itself
/// never carries one: a modern server rejects a handshake sent with a
/// version header.
#[must_use]
pub fn routing_headers(
    request: &JsonRpcRequest,
    legacy_version: Option<&str>,
) -> Vec<(String, String)> {
    let params = request.params.as_ref();
    let mut headers = vec![(
        "Mcp-Method".to_string(),
        encode_header_value(&request.method),
    )];
    let body_version = params
        .and_then(|p| p.get("_meta"))
        .and_then(|m| m.get(META_PROTOCOL_VERSION))
        .and_then(Value::as_str);
    let version = if request.method == "initialize" {
        None
    } else {
        body_version.or(legacy_version)
    };
    if let Some(version) = version {
        headers.push(("MCP-Protocol-Version".to_string(), version.to_string()));
    }
    let name_field = match request.method.as_str() {
        "tools/call" | "prompts/get" => Some("name"),
        "resources/read" => Some("uri"),
        _ => None,
    };
    if let Some(name) = name_field
        .and_then(|field| params.and_then(|p| p.get(field)))
        .and_then(Value::as_str)
    {
        headers.push(("Mcp-Name".to_string(), encode_header_value(name)));
    }
    debug_assert!(headers.iter().all(|(_, v)| v.is_ascii()));
    headers
}

// ---------------------------------------------------------------------------
// x-mcp-header
// ---------------------------------------------------------------------------

/// One tool parameter the server asked to have mirrored into a header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeaderParam {
    /// The `{name}` of `Mcp-Param-{name}`, as the schema spelled it.
    pub name: String,
    /// The chain of `properties` keys from the schema root to the parameter.
    pub path: Vec<String>,
}

/// Whether `name` is an RFC 9110 token (`1*tchar`). Pure.
fn is_token(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&b))
}

/// Collect a tool schema's `x-mcp-header` parameters, validating every rule
/// the spec lays down. Pure.
///
/// # Errors
///
/// Returns the reason the tool definition is invalid: an annotation that is
/// empty or not a token, a case-insensitive duplicate, one on a non-primitive
/// (or `number`) parameter, or one anywhere not reachable from the root
/// through `properties` keys alone (under `items`, `oneOf`, `$ref`, …).
pub fn header_params(schema: &Value) -> std::result::Result<Vec<HeaderParam>, String> {
    let mut found = Vec::new();
    let mut visited = 0_usize;
    walk_schema(schema, &mut Vec::new(), true, &mut found, &mut visited)?;
    let mut seen: Vec<String> = Vec::with_capacity(found.len());
    for param in &found {
        let lower = param.name.to_ascii_lowercase();
        if seen.contains(&lower) {
            return Err(format!("x-mcp-header `{}` is declared twice", param.name));
        }
        seen.push(lower);
    }
    Ok(found)
}

/// Visit every schema node. `reachable` is true only while the chain from the
/// root has been `properties` keys alone.
fn walk_schema(
    node: &Value,
    path: &mut Vec<String>,
    reachable: bool,
    found: &mut Vec<HeaderParam>,
    visited: &mut usize,
) -> std::result::Result<(), String> {
    *visited += 1;
    if *visited > SCHEMA_WALK_NODES_MAX {
        return Err(format!(
            "schema has more than {SCHEMA_WALK_NODES_MAX} nodes"
        ));
    }
    match node {
        Value::Object(object) => {
            if let Some(annotation) = object.get("x-mcp-header") {
                found.push(check_annotation(annotation, object, path, reachable)?);
            }
            for (key, child) in object {
                if key == "properties" {
                    if let Value::Object(properties) = child {
                        for (property, schema) in properties {
                            path.push(property.clone());
                            walk_schema(schema, path, reachable, found, visited)?;
                            path.pop();
                        }
                    }
                } else if key != "x-mcp-header" {
                    walk_schema(child, path, false, found, visited)?;
                }
            }
            Ok(())
        }
        Value::Array(items) => {
            for item in items {
                walk_schema(item, path, false, found, visited)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

/// Validate one `x-mcp-header` annotation found at `path`.
fn check_annotation(
    annotation: &Value,
    schema: &serde_json::Map<String, Value>,
    path: &[String],
    reachable: bool,
) -> std::result::Result<HeaderParam, String> {
    let Some(name) = annotation.as_str().filter(|n| is_token(n)) else {
        return Err(format!("x-mcp-header {annotation} is not an HTTP token"));
    };
    if !reachable || path.is_empty() {
        return Err(format!(
            "x-mcp-header `{name}` is not reachable from the schema root through properties alone"
        ));
    }
    match schema.get("type").and_then(Value::as_str) {
        Some("string" | "integer" | "boolean") => Ok(HeaderParam {
            name: name.to_string(),
            path: path.to_vec(),
        }),
        other => Err(format!(
            "x-mcp-header `{name}` is on a parameter of type {}; only string, integer and boolean are allowed",
            other.unwrap_or("(none)")
        )),
    }
}

/// The `Mcp-Param-*` headers for one call's arguments.
///
/// A missing or `null` value omits its header; so does a value whose type the annotation does
/// not allow (the server's own validation then names the problem). Pure.
#[must_use]
pub fn param_headers(params: &[HeaderParam], arguments: Option<&Value>) -> Vec<(String, String)> {
    /// JavaScript's safe-integer bound, which the spec imposes.
    const SAFE_INTEGER_MAX: i64 = (1 << 53) - 1;
    let mut headers = Vec::with_capacity(params.len());
    for param in params {
        let mut value = arguments;
        for key in &param.path {
            value = value.and_then(|v| v.get(key));
        }
        let text = match value {
            Some(Value::String(s)) => Some(s.clone()),
            Some(Value::Bool(b)) => Some(b.to_string()),
            Some(Value::Number(n)) => n
                .as_i64()
                .filter(|i| i.abs() <= SAFE_INTEGER_MAX)
                .map(|i| i.to_string()),
            _ => None,
        };
        if let Some(text) = text {
            headers.push((
                format!("Mcp-Param-{}", param.name),
                encode_header_value(&text),
            ));
        }
    }
    debug_assert!(headers.len() <= params.len());
    headers
}

/// Drop every tool whose `x-mcp-header` annotations are invalid from a
/// `tools/list` result (the spec: the client MUST exclude it), and return the
/// header parameters of the ones kept, by tool name.
pub fn gate_tools_list(result: &mut Value) -> HashMap<String, Vec<HeaderParam>> {
    let mut table = HashMap::new();
    let Some(tools) = result.get_mut("tools").and_then(Value::as_array_mut) else {
        return table;
    };
    let count_in = tools.len();
    tools.retain(|tool| {
        let name = tool
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("(unnamed)");
        let schema = tool.get("inputSchema").unwrap_or(&Value::Null);
        match header_params(schema) {
            Ok(params) => {
                if !params.is_empty() {
                    table.insert(name.to_string(), params);
                }
                true
            }
            Err(reason) => {
                warn!(
                    tool = name,
                    reason, "Dropping MCP tool with an invalid x-mcp-header"
                );
                false
            }
        }
    });
    debug_assert!(tools.len() <= count_in);
    table
}

// ---------------------------------------------------------------------------
// SSE
// ---------------------------------------------------------------------------

/// Incremental `text/event-stream` parser that yields each event's `data`.
///
/// Only `data:` matters to MCP; `event:`, `id:`, `retry:` and comment lines
/// (a leading `:`, used as keep-alives) are ignored.
#[derive(Debug, Default)]
pub struct SseParser {
    buffer: Vec<u8>,
    data: String,
    event: Option<String>,
}

/// One complete SSE event: its `event:` name (if any) and its data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
    pub event: Option<String>,
    pub data: String,
}

impl SseParser {
    /// Feed bytes; returns the data of every event they completed.
    ///
    /// # Errors
    ///
    /// Returns [`McpError::Protocol`] if one event outgrows
    /// [`HTTP_BODY_BYTES_MAX`] or a line is not UTF-8.
    pub fn push(&mut self, chunk: &[u8]) -> Result<Vec<String>> {
        Ok(self
            .push_events(chunk)?
            .into_iter()
            .map(|e| e.data)
            .collect())
    }

    /// As [`Self::push`], keeping each event's name (the 2024 HTTP+SSE
    /// transport announces its message endpoint in an `endpoint` event).
    ///
    /// # Errors
    ///
    /// As [`Self::push`].
    pub fn push_events(&mut self, chunk: &[u8]) -> Result<Vec<SseEvent>> {
        self.buffer.extend_from_slice(chunk);
        let mut events = Vec::new();
        while let Some(end) = self.buffer.iter().position(|&b| b == b'\n') {
            let mut line: Vec<u8> = self.buffer.drain(..=end).collect();
            line.pop();
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            let line = String::from_utf8(line)
                .map_err(|_| McpError::Protocol("SSE line is not UTF-8".into()))?;
            if line.is_empty() {
                if !self.data.is_empty() {
                    events.push(SseEvent {
                        event: self.event.take(),
                        data: std::mem::take(&mut self.data),
                    });
                }
                self.event = None;
                continue;
            }
            if let Some(name) = line.strip_prefix("event:") {
                self.event = Some(name.trim().to_string());
                continue;
            }
            if let Some(value) = line.strip_prefix("data:") {
                if !self.data.is_empty() {
                    self.data.push('\n');
                }
                self.data.push_str(value.strip_prefix(' ').unwrap_or(value));
            }
        }
        if self.data.len() + self.buffer.len() > HTTP_BODY_BYTES_MAX {
            return Err(McpError::Protocol(format!(
                "an SSE event exceeded {HTTP_BODY_BYTES_MAX} bytes"
            )));
        }
        Ok(events)
    }
}

// ---------------------------------------------------------------------------
// Transport
// ---------------------------------------------------------------------------

/// Streamable HTTP transport (see the module docs).
pub struct StreamableHttpTransport {
    client: reqwest::Client,
    endpoint: String,
    /// Sent as `Authorization: Bearer …`. Never logged.
    bearer_token: Option<String>,
    /// A legacy server's `Mcp-Session-Id`, echoed on every later request.
    session_id: Mutex<Option<String>>,
    /// The revision a legacy `initialize` negotiated, for the version header.
    legacy_version: Mutex<Option<String>>,
    /// `x-mcp-header` parameters by tool name, from the last `tools/list`.
    tool_headers: RwLock<HashMap<String, Vec<HeaderParam>>>,
    /// Marked by the listen task when the server announces a list change.
    list_changed: Arc<ListChangedFlags>,
    /// Signals the listen task to stop; dropping the transport does too.
    listen_stop: tokio::sync::watch::Sender<bool>,
}

impl StreamableHttpTransport {
    /// A transport for the MCP endpoint at `endpoint`.
    ///
    /// # Errors
    ///
    /// Returns [`McpError::Transport`] if the URL is not `http(s)` or the
    /// HTTP client cannot be built.
    pub fn new(endpoint: impl Into<String>, bearer_token: Option<String>) -> Result<Self> {
        let endpoint = endpoint.into();
        let parsed = reqwest::Url::parse(&endpoint)
            .map_err(|e| McpError::Transport(format!("invalid MCP URL `{endpoint}`: {e}")))?;
        if !matches!(parsed.scheme(), "http" | "https") {
            return Err(McpError::Transport(format!(
                "MCP URL `{endpoint}` must be http or https"
            )));
        }
        let client = reqwest::Client::builder()
            .timeout(HTTP_REQUEST_TIMEOUT)
            .build()
            .map_err(|e| McpError::Transport(e.to_string()))?;
        Ok(Self {
            client,
            endpoint,
            bearer_token: bearer_token.filter(|t| !t.is_empty()),
            session_id: Mutex::new(None),
            legacy_version: Mutex::new(None),
            tool_headers: RwLock::new(HashMap::new()),
            list_changed: Arc::new(ListChangedFlags::default()),
            listen_stop: tokio::sync::watch::channel(false).0,
        })
    }

    /// The POST every message goes out as, with every header but the
    /// per-method ones.
    async fn post(&self, body: String) -> reqwest::RequestBuilder {
        let mut builder = self
            .client
            .post(&self.endpoint)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .header(
                reqwest::header::ACCEPT,
                "application/json, text/event-stream",
            )
            .body(body);
        if let Some(token) = &self.bearer_token {
            builder = builder.bearer_auth(token);
        }
        if let Some(session) = self.session_id.lock().await.as_ref() {
            builder = builder.header("Mcp-Session-Id", session);
        }
        builder
    }

    /// Record a legacy server's `Mcp-Session-Id`, if the response minted one.
    async fn note_session(&self, response: &reqwest::Response) {
        if let Some(session) = response
            .headers()
            .get("mcp-session-id")
            .and_then(|v| v.to_str().ok())
        {
            *self.session_id.lock().await = Some(session.to_string());
        }
    }

    /// Read the body of an answer to `request` into its JSON-RPC response.
    async fn read_answer(
        &self,
        request: &JsonRpcRequest,
        response: reqwest::Response,
    ) -> Result<JsonRpcResponse> {
        let status = response.status();
        let is_sse = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|ct| ct.starts_with("text/event-stream"));
        if is_sse && status.is_success() {
            return self.read_sse(request, response).await;
        }
        let bytes = response
            .bytes()
            .await
            .map_err(|e| McpError::Transport(e.to_string()))?;
        if bytes.len() > HTTP_BODY_BYTES_MAX {
            return Err(McpError::Protocol(format!(
                "response body exceeded {HTTP_BODY_BYTES_MAX} bytes"
            )));
        }
        // A JSON-RPC body is the server's answer whatever the status: modern
        // servers put their era-identifying errors in 400 and 404 bodies.
        if let Some(answer) = json_rpc_answer(&bytes, &request.id) {
            return Ok(answer);
        }
        let text = String::from_utf8_lossy(&bytes);
        Err(McpError::HttpStatus {
            status: status.as_u16(),
            body: text.chars().take(ERROR_BODY_BYTES_MAX).collect(),
        })
    }

    /// Read an SSE answer until the response to `request` arrives.
    async fn read_sse(
        &self,
        request: &JsonRpcRequest,
        response: reqwest::Response,
    ) -> Result<JsonRpcResponse> {
        let mut parser = SseParser::default();
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| McpError::Transport(e.to_string()))?;
            for data in parser.push(&chunk)? {
                if let Some(answer) = self.route_event(&data, &request.id).await {
                    return Ok(answer);
                }
            }
        }
        Err(McpError::ConnectionClosed)
    }

    /// Handle one SSE event; `Some` when it is the awaited response.
    async fn route_event(&self, data: &str, awaited: &crate::RequestId) -> Option<JsonRpcResponse> {
        let Ok(message) = serde_json::from_str::<Value>(data) else {
            warn!("Unparseable MCP SSE event");
            return None;
        };
        let method = message.get("method").and_then(Value::as_str);
        let id = message.get("id").filter(|id| !id.is_null());
        match (method, id) {
            // A legacy server may ask something mid-stream; answer so it
            // does not wait forever (modern servers never do this).
            (Some(method), Some(id)) => {
                let reply = reply_to_server_request(id, method).to_string();
                if let Err(e) = self.post(reply).await.send().await {
                    warn!(error = %e, method, "Could not answer an MCP server request");
                }
                None
            }
            (Some(method), None) => {
                debug!(method, "MCP notification on a response stream");
                None
            }
            (None, _) => serde_json::from_value::<JsonRpcResponse>(message)
                .ok()
                .filter(|response| &response.id == awaited),
        }
    }
}

/// Everything the listen task needs, owned, so it outlives no borrow.
struct Listener {
    client: reqwest::Client,
    endpoint: String,
    bearer_token: Option<String>,
    request: JsonRpcRequest,
    flags: Arc<ListChangedFlags>,
    stop: tokio::sync::watch::Receiver<bool>,
}

/// How one listen stream ended.
enum ListenEnd {
    /// The stream dropped or went idle: reopen it.
    Reopen,
    /// The server does not serve `subscriptions/listen` (or refused it): stop.
    Unsupported(String),
}

impl Listener {
    /// Keep a listen stream open until told to stop, reconnecting with a
    /// bounded backoff. A stream that reconnects marks every list dirty: the
    /// gap may have hidden a change.
    async fn run(self) {
        // A separate handle: `select!` below borrows it mutably while
        // `listen_once` borrows the rest of `self`.
        let mut stop = self.stop.clone();
        let mut backoff = LISTEN_BACKOFF_MIN;
        let mut opened_before = false;
        loop {
            let started = tokio::time::Instant::now();
            let end = tokio::select! {
                _ = stop.changed() => return,
                end = self.listen_once(&mut opened_before) => end,
            };
            if let ListenEnd::Unsupported(reason) = end {
                debug!(reason, "MCP server has no change-notification stream");
                return;
            }
            if started.elapsed() > LISTEN_BACKOFF_MAX {
                backoff = LISTEN_BACKOFF_MIN;
            }
            tokio::select! {
                _ = stop.changed() => return,
                () = tokio::time::sleep(backoff) => {}
            }
            backoff = (backoff * 2).min(LISTEN_BACKOFF_MAX);
            debug_assert!(backoff <= LISTEN_BACKOFF_MAX);
        }
    }

    /// Open one stream and route its notifications until it ends.
    async fn listen_once(&self, opened_before: &mut bool) -> ListenEnd {
        let mut builder = self
            .client
            .post(&self.endpoint)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .header(
                reqwest::header::ACCEPT,
                "application/json, text/event-stream",
            );
        for (name, value) in routing_headers(&self.request, None) {
            builder = builder.header(name, value);
        }
        if let Some(token) = &self.bearer_token {
            builder = builder.bearer_auth(token);
        }
        let Ok(body) = serde_json::to_string(&self.request) else {
            return ListenEnd::Unsupported("request did not serialize".into());
        };
        let response = match builder.body(body).send().await {
            Ok(response) => response,
            Err(e) => {
                debug!(error = %e, "MCP listen request failed; will retry");
                return ListenEnd::Reopen;
            }
        };
        let is_sse = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|ct| ct.starts_with("text/event-stream"));
        if !(response.status().is_success() && is_sse) {
            return ListenEnd::Unsupported(format!("HTTP {} without a stream", response.status()));
        }
        let mut parser = SseParser::default();
        let mut stream = response.bytes_stream();
        loop {
            let Ok(Some(Ok(chunk))) =
                tokio::time::timeout(LISTEN_IDLE_TIMEOUT, stream.next()).await
            else {
                // Dropped, ended, or idle past the timeout: reopen.
                return ListenEnd::Reopen;
            };
            let Ok(events) = parser.push(&chunk) else {
                return ListenEnd::Reopen;
            };
            for data in events {
                self.route(&data, opened_before);
            }
        }
    }

    /// Route one event from the listen stream into the flags.
    fn route(&self, data: &str, opened_before: &mut bool) {
        let Ok(message) = serde_json::from_str::<Value>(data) else {
            return;
        };
        match message.get("method").and_then(Value::as_str) {
            Some("notifications/subscriptions/acknowledged") => {
                if *opened_before {
                    // A reopened stream: anything may have changed meanwhile.
                    for list in [McpList::Tools, McpList::Resources, McpList::Prompts] {
                        self.flags.mark(list);
                    }
                }
                *opened_before = true;
            }
            Some("notifications/tools/list_changed") => self.flags.mark(McpList::Tools),
            Some("notifications/resources/list_changed") => self.flags.mark(McpList::Resources),
            Some("notifications/prompts/list_changed") => self.flags.mark(McpList::Prompts),
            _ => {}
        }
    }
}

/// Parse a body as a JSON-RPC answer. An error with a missing or `null` id
/// (allowed when the server could not read ours) is attributed to `id`.
fn json_rpc_answer(bytes: &[u8], id: &crate::RequestId) -> Option<JsonRpcResponse> {
    let mut value: Value = serde_json::from_slice(bytes).ok()?;
    let object = value.as_object_mut()?;
    if !object.contains_key("result") && !object.contains_key("error") {
        return None;
    }
    if object.get("id").is_none_or(Value::is_null) {
        object.insert("id".to_string(), serde_json::to_value(id).ok()?);
    }
    serde_json::from_value(value).ok()
}

#[async_trait]
impl Transport for StreamableHttpTransport {
    async fn request(&self, request: JsonRpcRequest) -> Result<JsonRpcResponse> {
        let legacy_version = self.legacy_version.lock().await.clone();
        let mut headers = routing_headers(&request, legacy_version.as_deref());
        if request.method == "tools/call" {
            let params = request.params.as_ref();
            if let Some(tool) = params.and_then(|p| p.get("name")).and_then(Value::as_str) {
                let declared = self.tool_headers.read().await.get(tool).cloned();
                if let Some(declared) = declared {
                    let arguments = params.and_then(|p| p.get("arguments"));
                    headers.extend(param_headers(&declared, arguments));
                }
            }
        }
        let mut builder = self.post(serde_json::to_string(&request)?).await;
        for (name, value) in &headers {
            builder = builder.header(name.as_str(), value.as_str());
        }
        let response = builder.send().await.map_err(|e| {
            if e.is_timeout() {
                McpError::Timeout
            } else {
                McpError::Transport(e.to_string())
            }
        })?;
        self.note_session(&response).await;
        let mut answer = self.read_answer(&request, response).await?;
        if let Some(result) = answer.result.as_mut() {
            if request.method == "tools/list" {
                *self.tool_headers.write().await = gate_tools_list(result);
            }
            if request.method == "initialize"
                && let Some(version) = result.get("protocolVersion").and_then(Value::as_str)
            {
                *self.legacy_version.lock().await = Some(version.to_string());
            }
        }
        Ok(answer)
    }

    async fn notify(&self, notification: JsonRpcNotification) -> Result<()> {
        let legacy_version = self.legacy_version.lock().await.clone();
        let mut builder = self
            .post(serde_json::to_string(&notification)?)
            .await
            .header("Mcp-Method", encode_header_value(&notification.method));
        if let Some(version) = legacy_version {
            builder = builder.header("MCP-Protocol-Version", version);
        }
        let response = builder
            .send()
            .await
            .map_err(|e| McpError::Transport(e.to_string()))?;
        let status = response.status();
        if status.is_success() {
            return Ok(());
        }
        let body = response.text().await.unwrap_or_default();
        Err(McpError::HttpStatus {
            status: status.as_u16(),
            body: body.chars().take(ERROR_BODY_BYTES_MAX).collect(),
        })
    }

    /// Nothing to tear down in the modern revision (no sessions); a legacy
    /// session is ended with the `DELETE` its revision defines, best effort.
    async fn close(&self) -> Result<()> {
        let _ = self.listen_stop.send(true);
        let Some(session) = self.session_id.lock().await.take() else {
            return Ok(());
        };
        let mut builder = self
            .client
            .delete(&self.endpoint)
            .header("Mcp-Session-Id", session);
        if let Some(token) = &self.bearer_token {
            builder = builder.bearer_auth(token);
        }
        if let Err(e) = builder.send().await {
            debug!(error = %e, "Ending the legacy MCP session failed");
        }
        Ok(())
    }

    fn list_changed_flags(&self) -> Option<Arc<ListChangedFlags>> {
        Some(self.list_changed.clone())
    }

    /// Run the listen stream on its own task (with its own client: the
    /// request client's 30 s total timeout would cut a long-lived stream).
    async fn open_listen(&self, request: JsonRpcRequest) -> Result<()> {
        debug_assert_eq!(request.method, "subscriptions/listen");
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .build()
            .map_err(|e| McpError::Transport(e.to_string()))?;
        let listener = Listener {
            client,
            endpoint: self.endpoint.clone(),
            bearer_token: self.bearer_token.clone(),
            request,
            flags: self.list_changed.clone(),
            stop: self.listen_stop.subscribe(),
        };
        tokio::spawn(listener.run());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn request(method: &str, params: Value) -> JsonRpcRequest {
        JsonRpcRequest::new(1_i64, method, Some(params))
    }

    fn header<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
        headers
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_str())
    }

    #[test]
    fn header_values_follow_the_spec_examples() {
        // The encoding table in the 2026-07-28 Streamable HTTP binding.
        assert_eq!(encode_header_value("us-west1"), "us-west1");
        assert_eq!(
            encode_header_value("Hello, 世界"),
            "=?base64?SGVsbG8sIOS4lueVjA==?="
        );
        assert_eq!(encode_header_value(" padded "), "=?base64?IHBhZGRlZCA=?=");
        assert_eq!(
            encode_header_value("line1\nline2"),
            "=?base64?bGluZTEKbGluZTI=?="
        );
        assert_eq!(
            encode_header_value("=?base64?literal?="),
            "=?base64?PT9iYXNlNjQ/bGl0ZXJhbD89?="
        );
    }

    #[test]
    fn routing_headers_mirror_the_body() {
        let meta = json!({ META_PROTOCOL_VERSION: "2026-07-28" });
        let call = request(
            "tools/call",
            json!({ "name": "get_weather", "_meta": meta }),
        );
        let headers = routing_headers(&call, None);
        assert_eq!(header(&headers, "Mcp-Method"), Some("tools/call"));
        assert_eq!(header(&headers, "Mcp-Name"), Some("get_weather"));
        assert_eq!(header(&headers, "MCP-Protocol-Version"), Some("2026-07-28"));

        let read = request(
            "resources/read",
            json!({ "uri": "file:///a b.txt", "_meta": meta }),
        );
        assert_eq!(
            header(&routing_headers(&read, None), "Mcp-Name"),
            Some("file:///a b.txt")
        );

        let list = request("tools/list", json!({ "_meta": meta }));
        assert_eq!(header(&routing_headers(&list, None), "Mcp-Name"), None);
    }

    #[test]
    fn legacy_requests_carry_the_negotiated_version_but_initialize_carries_none() {
        let init = request("initialize", json!({ "protocolVersion": "2024-11-05" }));
        assert_eq!(
            header(
                &routing_headers(&init, Some("2025-06-18")),
                "MCP-Protocol-Version"
            ),
            None
        );
        let list = JsonRpcRequest::new(2_i64, "tools/list", None);
        assert_eq!(
            header(&routing_headers(&list, None), "MCP-Protocol-Version"),
            None
        );
        assert_eq!(
            header(
                &routing_headers(&list, Some("2024-11-05")),
                "MCP-Protocol-Version"
            ),
            Some("2024-11-05")
        );
    }

    #[test]
    fn x_mcp_header_is_collected_from_properties_chains() {
        let schema = json!({ "type": "object", "properties": {
            "region": { "type": "string", "x-mcp-header": "Region" },
            "query": { "type": "string" },
            "opts": { "type": "object", "properties": {
                "dry": { "type": "boolean", "x-mcp-header": "Dry-Run" } } }
        }});
        let params = header_params(&schema).unwrap();
        assert_eq!(params.len(), 2);
        assert!(params.contains(&HeaderParam {
            name: "Region".into(),
            path: vec!["region".into()]
        }));
        assert!(params.contains(&HeaderParam {
            name: "Dry-Run".into(),
            path: vec!["opts".into(), "dry".into()]
        }));

        let args = json!({ "region": "us-west1", "opts": { "dry": true } });
        let headers = param_headers(&params, Some(&args));
        assert_eq!(header(&headers, "Mcp-Param-Region"), Some("us-west1"));
        assert_eq!(header(&headers, "Mcp-Param-Dry-Run"), Some("true"));
        // Absent or null values omit the header.
        let sparse = param_headers(&params, Some(&json!({ "region": null })));
        assert!(sparse.is_empty(), "{sparse:?}");
    }

    #[test]
    fn invalid_x_mcp_header_annotations_are_rejected() {
        let cases = [
            json!({ "properties": { "a": { "type": "number", "x-mcp-header": "A" } } }),
            json!({ "properties": { "a": { "type": "object", "x-mcp-header": "A" } } }),
            json!({ "properties": { "a": { "type": "string", "x-mcp-header": "" } } }),
            json!({ "properties": { "a": { "type": "string", "x-mcp-header": "Bad Name" } } }),
            json!({ "properties": {
                "a": { "type": "string", "x-mcp-header": "Dup" },
                "b": { "type": "string", "x-mcp-header": "dup" } } }),
            json!({ "properties": { "list": { "type": "array",
                "items": { "type": "string", "x-mcp-header": "Item" } } } }),
            json!({ "oneOf": [ { "properties": { "a": { "type": "string", "x-mcp-header": "A" } } } ] }),
            json!({ "type": "string", "x-mcp-header": "Root" }),
        ];
        for schema in cases {
            assert!(header_params(&schema).is_err(), "must reject {schema}");
        }
    }

    #[test]
    fn a_tools_list_drops_only_the_invalid_tool() {
        let mut result = json!({ "tools": [
            { "name": "ok", "inputSchema": { "properties": {
                "region": { "type": "string", "x-mcp-header": "Region" } } } },
            { "name": "bad", "inputSchema": { "properties": {
                "n": { "type": "number", "x-mcp-header": "N" } } } },
            { "name": "plain", "inputSchema": { "type": "object" } }
        ]});
        let table = gate_tools_list(&mut result);
        let names: Vec<&str> = result["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, ["ok", "plain"]);
        assert_eq!(table.len(), 1);
        assert_eq!(table["ok"][0].name, "Region");
    }

    #[test]
    fn sse_events_are_reassembled_across_chunks() {
        let mut parser = SseParser::default();
        let none: Vec<String> = Vec::new();
        assert_eq!(
            parser.push(b"event: message\r\ndata: {\"a\":").unwrap(),
            none
        );
        assert_eq!(parser.push(b"1}\r\n").unwrap(), none);
        let events = parser
            .push(b"\r\n: keep-alive\n\ndata: x\ndata: y\n\n")
            .unwrap();
        assert_eq!(events, ["{\"a\":1}", "x\ny"]);
    }

    #[test]
    fn an_oversized_sse_event_is_an_error_not_unbounded_growth() {
        let mut parser = SseParser::default();
        let chunk = vec![b'a'; HTTP_BODY_BYTES_MAX + 1];
        assert!(parser.push(&chunk).is_err());
    }

    #[test]
    fn error_bodies_without_an_id_are_attributed_to_the_request() {
        let id = crate::RequestId::Number(7);
        let body = br#"{"jsonrpc":"2.0","error":{"code":-32000,"message":"Bad Request: Server not initialized"},"id":null}"#;
        let answer = json_rpc_answer(body, &id).unwrap();
        assert_eq!(answer.id, id);
        assert_eq!(answer.error.unwrap().code, -32000);
        assert!(json_rpc_answer(b"missing or wrong bearer token", &id).is_none());
        assert!(json_rpc_answer(br#"{"hello":1}"#, &id).is_none());
    }

    #[test]
    fn only_http_urls_are_accepted() {
        assert!(StreamableHttpTransport::new("http://127.0.0.1:1/mcp", None).is_ok());
        assert!(StreamableHttpTransport::new("file:///etc/passwd", None).is_err());
        assert!(StreamableHttpTransport::new("not a url", None).is_err());
    }
}
