//! MCP protocol types (JSON-RPC 2.0 based)
//!
//! Implements the wire format for Model Context Protocol messages.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// JSON-RPC 2.0 request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: RequestId,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

impl JsonRpcRequest {
    pub fn new(
        id: impl Into<RequestId>,
        method: impl Into<String>,
        params: Option<serde_json::Value>,
    ) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id: id.into(),
            method: method.into(),
            params,
        }
    }
}

/// JSON-RPC 2.0 response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: RequestId,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

/// JSON-RPC 2.0 notification (no id, no response expected)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcNotification {
    pub jsonrpc: String,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

impl JsonRpcNotification {
    pub fn new(method: impl Into<String>, params: Option<serde_json::Value>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            method: method.into(),
            params,
        }
    }
}

/// JSON-RPC error object
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

/// JSON-RPC / MCP error codes this client recognises.
///
/// The 2026-07-28 revision partitions the JSON-RPC server-error range:
/// `-32000..=-32019` stays implementation-defined (existing SDK usage is
/// grandfathered) and `-32020..=-32099` is reserved for the MCP specification.
/// Matching on named constants keeps that partition legible at the call sites.
pub mod error_codes {
    /// JSON-RPC standard "Invalid params". Since the 2026-07-28 revision this is
    /// also what a server returns for a **missing resource**, replacing the
    /// MCP-custom [`LEGACY_RESOURCE_NOT_FOUND`].
    pub const INVALID_PARAMS: i32 = -32602;

    /// Pre-2026-07-28 MCP-custom "resource not found". Still emitted by servers
    /// pinned to an older revision, so both codes must be accepted.
    pub const LEGACY_RESOURCE_NOT_FOUND: i32 = -32002;

    /// The server rejected a routable header (2026-07-28, MCP-reserved range).
    pub const HEADER_MISMATCH: i32 = -32020;

    /// The server requires a client capability this client did not advertise.
    pub const MISSING_REQUIRED_CLIENT_CAPABILITY: i32 = -32021;

    /// The server does not support the protocol revision the client offered.
    pub const UNSUPPORTED_PROTOCOL_VERSION: i32 = -32022;

    /// Does `code` mean "the resource you asked for does not exist"?
    ///
    /// Accepts both revisions' spellings so a `resources/read` against an old
    /// *or* a 2026-07-28 server maps to the same typed error. Pure and total —
    /// every other code (including the three MCP-reserved "modern server" codes,
    /// which are explicitly *not* legacy indicators) answers `false`.
    #[must_use]
    pub const fn is_resource_missing(code: i32) -> bool {
        matches!(code, INVALID_PARAMS | LEGACY_RESOURCE_NOT_FOUND)
    }
}

/// Request ID (can be string or number)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(untagged)]
pub enum RequestId {
    String(String),
    Number(i64),
}

impl From<String> for RequestId {
    fn from(s: String) -> Self {
        Self::String(s)
    }
}

impl From<&str> for RequestId {
    fn from(s: &str) -> Self {
        Self::String(s.to_string())
    }
}

impl From<i64> for RequestId {
    fn from(n: i64) -> Self {
        Self::Number(n)
    }
}

impl std::fmt::Display for RequestId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::String(s) => write!(f, "{s}"),
            Self::Number(n) => write!(f, "{n}"),
        }
    }
}

// ============================================================================
// MCP Protocol Messages
// ============================================================================

/// Client capabilities sent during initialization
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClientCapabilities {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub roots: Option<RootsCapability>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sampling: Option<SamplingCapability>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub experimental: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RootsCapability {
    #[serde(default)]
    pub list_changed: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SamplingCapability {}

/// Server capabilities returned during initialization
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ServerCapabilities {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<ToolsCapability>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resources: Option<ResourcesCapability>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompts: Option<PromptsCapability>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logging: Option<LoggingCapability>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub experimental: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolsCapability {
    #[serde(default)]
    pub list_changed: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourcesCapability {
    #[serde(default)]
    pub subscribe: bool,
    #[serde(default)]
    pub list_changed: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptsCapability {
    #[serde(default)]
    pub list_changed: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LoggingCapability {}

/// Initialize request params
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeParams {
    pub protocol_version: String,
    pub capabilities: ClientCapabilities,
    pub client_info: ClientInfo,
}

/// Client info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientInfo {
    pub name: String,
    pub version: String,
}

/// Initialize response
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeResult {
    pub protocol_version: String,
    pub capabilities: ServerCapabilities,
    pub server_info: ServerInfo,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
}

/// Server info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerInfo {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
}

// ============================================================================
// Tools
// ============================================================================

/// Tool definition from server
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Tool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
}

/// List tools response
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ListToolsResult {
    pub tools: Vec<Tool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<String>,
}

/// Call tool params
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallToolParams {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<serde_json::Value>,
}

/// Tool call result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CallToolResult {
    pub content: Vec<ToolContent>,
    #[serde(default)]
    pub is_error: bool,
    /// Machine-readable result payload, when the server provides one.
    ///
    /// The 2026-07-28 revision lifts the restriction that this be an *object* —
    /// it may be **any** JSON value (object, array, string, number, bool).
    /// Typing it as a bare [`serde_json::Value`] is therefore the spec-correct
    /// shape; narrowing it to a map would silently drop a conforming server's
    /// result.
    ///
    /// An explicit `null` deserializes to `None`, i.e. it is treated exactly
    /// like an omitted field. That collapse is deliberate: a `null` payload
    /// carries no information, and keeping the two apart would only let a
    /// server that always emits the key attach a meaningless `data: null` to
    /// every tool result downstream.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structured_content: Option<serde_json::Value>,
}

/// Content returned by tool
///
/// `rename_all` on a tagged enum renames the VARIANTS (the `type` tag) and
/// not their fields, so every camelCase field is renamed by hand. `Image`'s
/// `mimeType` was not: every image block failed to deserialize, and with it
/// the whole `tools/call` result — text blocks included.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolContent {    Text {
        text: String,
    },
    Image {
        data: String,
        #[serde(rename = "mimeType")]
        mime_type: String,
    },
    Audio {
        data: String,
        #[serde(rename = "mimeType")]
        mime_type: String,
    },
    Resource {
        resource: ResourceContents,
    },
    /// A link to a resource the server did not inline (MCP 2025-06-18).
    ResourceLink {
        uri: String,
        #[serde(default)]
        name: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        #[serde(default, rename = "mimeType", skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
    },
    /// A content type newer than this client. Kept as a marker rather than
    /// failing the result: one unknown block used to discard every other block
    /// the server sent with it.
    #[serde(other)]
    Unknown,
}

impl ToolContent {
    /// The block as the text an agent reads: text verbatim, binary content
    /// described by its type and size, a link by where it points.
    #[must_use]
    pub fn to_text(&self) -> Option<String> {
        match self {
            Self::Text { text } => Some(text.clone()),
            Self::Image { data, mime_type } => {
                Some(format!("[Image: {mime_type}, {} bytes]", data.len()))
            }
            Self::Audio { data, mime_type } => {
                Some(format!("[Audio: {mime_type}, {} bytes]", data.len()))
            }
            Self::Resource { resource } => resource.text.clone().or_else(|| {
                resource
                    .blob
                    .as_ref()
                    .map(|b| format!("[Blob: {} bytes]", b.len()))
            }),
            Self::ResourceLink {
                uri,
                name,
                description,
                ..
            } => {
                let label = name.as_deref().unwrap_or(uri);
                Some(description.as_deref().map_or_else(
                    || format!("[Resource link: {label} <{uri}>]"),
                    |d| format!("[Resource link: {label} <{uri}> — {d}]"),
                ))
            }
            Self::Unknown => Some("[Content of a type this client does not know]".to_string()),
        }
    }
}

// ============================================================================
// Resources
// ============================================================================

/// Resource definition from server
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Resource {
    pub uri: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

/// List resources response
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ListResourcesResult {
    pub resources: Vec<Resource>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<String>,
}

/// Read resource params
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadResourceParams {
    pub uri: String,
}

/// Read resource result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadResourceResult {
    pub contents: Vec<ResourceContents>,
}

/// Resource contents
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourceContents {
    pub uri: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blob: Option<String>,
}

// ============================================================================
// Prompts
// ============================================================================

/// Prompt definition from server
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prompt {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<Vec<PromptArgument>>,
}

/// Prompt argument definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptArgument {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default)]
    pub required: bool,
}

/// List prompts response
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ListPromptsResult {
    pub prompts: Vec<Prompt>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<String>,
}

/// Get prompt params
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetPromptParams {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<HashMap<String, String>>,
}

/// Get prompt result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetPromptResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub messages: Vec<PromptMessage>,
}

/// Prompt message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptMessage {
    pub role: PromptRole,
    pub content: PromptContent,
}

/// Prompt role
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PromptRole {
    User,
    Assistant,
}

/// Prompt content
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PromptContent {    Text {
        text: String,
    },
    Image {
        data: String,
        #[serde(rename = "mimeType")]
        mime_type: String,
    },
    Resource {
        resource: ResourceContents,
    },
}

// ============================================================================
// Logging
// ============================================================================

/// Log level
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    Debug,
    Info,
    Notice,
    Warning,
    Error,
    Critical,
    Alert,
    Emergency,
}

/// Log message notification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogMessageParams {
    pub level: LogLevel,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logger: Option<String>,
    pub data: serde_json::Value,
}

// ============================================================================
// Pagination
// ============================================================================

/// Pagination params for list operations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PaginationParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cursor: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::{CallToolResult, ToolContent};

    use super::error_codes::{
        HEADER_MISMATCH, INVALID_PARAMS, LEGACY_RESOURCE_NOT_FOUND,
        MISSING_REQUIRED_CLIENT_CAPABILITY, UNSUPPORTED_PROTOCOL_VERSION, is_resource_missing,
    };
    use serde_json::json;

    /// A `tools/call` result as MCP servers send it. Image blocks carried
    /// `mimeType` and failed to parse as `mime_type`, discarding the whole
    /// result; `audio`, `resource_link` and anything newer did the same.
    #[test]
    fn every_content_block_type_parses_and_reads_as_text() {
        let raw = serde_json::json!({
            "content": [
                { "type": "text", "text": "done" },
                { "type": "image", "data": "aGVsbG8=", "mimeType": "image/png" },
                { "type": "audio", "data": "AAAA", "mimeType": "audio/wav" },
                {
                    "type": "resource_link", "uri": "file:///r.md", "name": "r.md",
                    "description": "the report", "mimeType": "text/markdown"
                },
                { "type": "something_from_2027", "whatever": 1 }
            ],
            "isError": false
        });
        let result: CallToolResult = serde_json::from_value(raw).expect("a spec result parses");
        let text: Vec<String> = result
            .content
            .iter()
            .filter_map(ToolContent::to_text)
            .collect();
        assert_eq!(text[0], "done");
        assert_eq!(text[1], "[Image: image/png, 8 bytes]");
        assert_eq!(text[2], "[Audio: audio/wav, 4 bytes]");
        assert_eq!(text[3], "[Resource link: r.md <file:///r.md> — the report]");
        assert!(matches!(result.content[4], ToolContent::Unknown));

        // And it round-trips in the spec's field names.
        let image = serde_json::to_value(&result.content[1]).expect("serializes");
        assert_eq!(image["mimeType"], "image/png");
    }

    /// Deserialize a `tools/call` result body.
    fn parse(body: serde_json::Value) -> CallToolResult {
        serde_json::from_value(body).expect("a well-formed CallToolResult must parse")
    }

    #[test]
    fn structured_content_accepts_any_json_value() {
        // The 2026-07-28 revision allows ANY JSON value here, not only an object.
        // Each of these must round-trip verbatim rather than being rejected.
        // (`null` is covered separately — it collapses to `None` by design.)
        for payload in [
            json!({ "rows": 3 }),
            json!([1, 2, 3]),
            json!("a bare string"),
            json!(42),
            json!(true),
        ] {
            let result = parse(json!({
                "content": [{ "type": "text", "text": "ok" }],
                "structuredContent": payload,
            }));
            assert_eq!(
                result.structured_content.as_ref(),
                Some(&payload),
                "structuredContent must survive as {payload}"
            );
        }
    }

    #[test]
    fn absent_structured_content_is_none_and_is_not_reserialized() {
        let result = parse(json!({ "content": [{ "type": "text", "text": "ok" }] }));
        assert!(
            result.structured_content.is_none(),
            "absent payload must be None, not Some(null)"
        );
        // Absent must stay absent on the wire (skip_serializing_if), so a server
        // round-tripping our value does not see a spurious null field.
        let wire = serde_json::to_value(&result).expect("serialize");
        assert!(wire.get("structuredContent").is_none());
    }

    #[test]
    fn explicit_null_structured_content_collapses_to_absent() {
        // Documented behaviour: an explicit `null` is treated exactly like an
        // omitted field, so a server that always emits the key cannot attach a
        // meaningless `data: null` to every tool result. Pinned as a test so the
        // collapse stays a decision rather than an accident of serde defaults.
        let result = parse(json!({
            "content": [],
            "structuredContent": null,
        }));
        assert!(result.structured_content.is_none());
        // …and it round-trips as absent, not as a null field.
        let wire = serde_json::to_value(&result).expect("serialize");
        assert!(wire.get("structuredContent").is_none());
    }

    #[test]
    fn resource_missing_matches_both_spec_revisions() {
        // Pre-2026-07-28 servers send the MCP-custom code…
        assert!(is_resource_missing(LEGACY_RESOURCE_NOT_FOUND));
        // …2026-07-28 servers send the JSON-RPC standard one.
        assert!(is_resource_missing(INVALID_PARAMS));
    }

    #[test]
    fn resource_missing_rejects_unrelated_codes() {
        // Negative space: the three MCP-reserved "modern server" codes are
        // explicitly not missing-resource indicators, nor is method-not-found or
        // a generic implementation-defined code.
        for code in [
            HEADER_MISMATCH,
            MISSING_REQUIRED_CLIENT_CAPABILITY,
            UNSUPPORTED_PROTOCOL_VERSION,
            -32601, // method not found
            -32000, // implementation-defined
            0,
        ] {
            assert!(
                !is_resource_missing(code),
                "code {code} must not read as a missing resource"
            );
        }
    }

    #[test]
    fn mcp_reserved_codes_sit_inside_the_reserved_range() {
        // The revision partitions -32020..=-32099 for the spec; keep the named
        // constants inside it so a future addition cannot silently collide with
        // the grandfathered -32000..=-32019 implementation-defined band.
        for code in [
            HEADER_MISMATCH,
            MISSING_REQUIRED_CLIENT_CAPABILITY,
            UNSUPPORTED_PROTOCOL_VERSION,
        ] {
            assert!(code <= -32020, "{code} must be at or below -32020");
            assert!(code >= -32099, "{code} must be at or above -32099");
        }
    }
}
