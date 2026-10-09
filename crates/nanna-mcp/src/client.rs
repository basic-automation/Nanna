//! High-level MCP client implementation

use crate::{
    McpError, Result,
    era::{
        DiscoverResult, LEGACY_PROTOCOL_VERSION, MODERN_PROTOCOL_VERSIONS, ProbeVerdict,
        ProtocolEra, classify_probe, ensure_complete, with_modern_meta,
    },
    protocol::{
        CallToolParams, CallToolResult, ClientCapabilities, ClientInfo, GetPromptParams,
        GetPromptResult, InitializeParams, InitializeResult, JsonRpcNotification, JsonRpcRequest,
        ListPromptsResult, ListResourcesResult, ListToolsResult, Prompt, ReadResourceParams,
        ReadResourceResult, RequestId, Resource, ServerCapabilities, ServerInfo, Tool,
    },
    transport::{McpList, Transport},
};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::schema_guard::validate_tool_schema;

/// The legacy MCP revision the `initialize` handshake offers.
pub const PROTOCOL_VERSION: &str = LEGACY_PROTOCOL_VERSION;

/// Map a `resources/read` failure onto a typed error.
///
/// A missing resource arrives as a JSON-RPC error whose code depends on the
/// server's spec revision — `-32002` before 2026-07-28, the standard `-32602`
/// after it. Both mean the same thing to a caller, so both collapse to
/// [`McpError::ResourceNotFound`]; every other failure passes through unchanged
/// so a transport/protocol fault is never mistaken for a missing URI.
fn resource_error_for(uri: &str, error: McpError) -> McpError {
    match error {
        McpError::JsonRpc { code, .. }
            if crate::protocol::error_codes::is_resource_missing(code) =>
        {
            debug_assert!(!uri.is_empty(), "a resource read always carries a URI");
            McpError::ResourceNotFound(uri.to_string())
        }
        other => other,
    }
}

/// MCP client for connecting to tool servers
pub struct McpClient<T: Transport> {
    transport: Arc<T>,
    /// Request ID counter
    id_counter: AtomicI64,
    /// Server info (set after initialization)
    server_info: RwLock<Option<ServerInfo>>,
    /// Server capabilities
    capabilities: RwLock<Option<ServerCapabilities>>,
    /// Cached tools
    tools: RwLock<Vec<Tool>>,
    /// Cached resources
    resources: RwLock<Vec<Resource>>,
    /// Cached prompts
    prompts: RwLock<Vec<Prompt>>,
    /// Whether client is initialized
    initialized: RwLock<bool>,
    /// The server's protocol era, decided once by [`McpClient::initialize`]
    /// and cached for the server's lifetime. `Legacy` until then.
    era: RwLock<ProtocolEra>,
    /// Puts a modern server's elicitation to the user. `None`: the
    /// capability is not declared and an `input_required` answer is an error.
    elicitor: Option<Arc<dyn crate::elicit::Elicitor>>,
}

/// Most `tools/list` pages followed.
///
/// Bound justification: at the smallest page size a server would sensibly use
/// (about 10 tools) this is ~640 tools — far past what a model can be offered —
/// while a server that never stops returning cursors cannot hold the client.
const TOOL_LIST_PAGES_MAX: usize = 64;

impl<T: Transport> McpClient<T> {
    /// Create a new MCP client with the given transport
    pub fn new(transport: T) -> Self {
        Self {
            transport: Arc::new(transport),
            id_counter: AtomicI64::new(1),
            server_info: RwLock::new(None),
            capabilities: RwLock::new(None),
            tools: RwLock::new(Vec::new()),
            resources: RwLock::new(Vec::new()),
            prompts: RwLock::new(Vec::new()),
            initialized: RwLock::new(false),
            era: RwLock::new(ProtocolEra::Legacy),
            elicitor: None,
        }
    }

    /// Serve the server's form elicitations through `elicitor` (modern era).
    /// Set before [`Self::initialize`]: it decides what the client declares.
    #[must_use]
    pub fn with_elicitor(mut self, elicitor: Arc<dyn crate::elicit::Elicitor>) -> Self {
        self.elicitor = Some(elicitor);
        self
    }

    /// The client capabilities a modern request declares.
    fn declared_capabilities(&self) -> serde_json::Value {
        if self.elicitor.is_some() {
            crate::elicit::elicitation_capability()
        } else {
            serde_json::json!({})
        }
    }

    /// `params` with the modern `_meta` attached when the era is modern.
    async fn with_era_meta(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<Option<serde_json::Value>> {
        match &*self.era.read().await {
            ProtocolEra::Modern { version } => Ok(Some(crate::era::with_modern_meta_declaring(
                params,
                version,
                &self.declared_capabilities(),
            )?)),
            ProtocolEra::Legacy => Ok(params),
        }
    }

    /// Generate next request ID
    fn next_id(&self) -> RequestId {
        RequestId::Number(self.id_counter.fetch_add(1, Ordering::SeqCst))
    }

    /// Send a request and parse the result. Under the modern era every
    /// request carries the per-request `_meta`.
    async fn request<R>(&self, method: &str, params: Option<serde_json::Value>) -> Result<R>
    where
        R: serde::de::DeserializeOwned,
    {
        let params = self.with_era_meta(params).await?;
        let result = self.request_value(method, params).await?;
        serde_json::from_value(result).map_err(Into::into)
    }

    /// Send a request exactly as given and return its result, which must be
    /// `complete`.
    async fn request_value(
        &self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<serde_json::Value> {
        let result = self.request_value_raw(method, params).await?;
        ensure_complete(&result)?;
        Ok(result)
    }

    /// Send a request exactly as given and return its result whatever its
    /// `resultType`.
    async fn request_value_raw(
        &self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<serde_json::Value> {
        let request = JsonRpcRequest::new(self.next_id(), method, params);
        let response = self.transport.request(request).await?;

        if let Some(error) = response.error {
            return Err(McpError::JsonRpc {
                code: error.code,
                message: error.message,
                data: error.data,
            });
        }

        response
            .result
            .ok_or_else(|| McpError::Protocol("Missing result in response".into()))
    }

    /// Connect to the server in whichever protocol era it speaks.
    ///
    /// Must be called before using any other methods. Probes with
    /// `server/discover` (MCP 2026-07-28): a modern server is then spoken to
    /// statelessly with per-request `_meta`; any other answer, or none within
    /// the transport's request timeout, falls back to the legacy
    /// `initialize` handshake. The era is cached for the client's lifetime.
    ///
    /// # Errors
    ///
    /// Returns error if the server dies during the probe, is a modern server
    /// sharing no revision with this client, or rejects the connection.
    pub async fn initialize(&self) -> Result<InitializeResult> {
        let probe_version = MODERN_PROTOCOL_VERSIONS[0];
        let outcome = self
            .request_value(
                "server/discover",
                Some(with_modern_meta(None, probe_version)?),
            )
            .await;
        let answered = outcome.as_ref().ok().cloned();
        match classify_probe(outcome)? {
            ProbeVerdict::Modern { version } => {
                let discover = match answered {
                    Some(result) if version == probe_version => result,
                    // The server named another mutual revision: ask again in it.
                    _ => {
                        let params = with_modern_meta(None, &version)?;
                        self.request_value("server/discover", Some(params)).await?
                    }
                };
                self.finish_modern(version, discover).await
            }
            ProbeVerdict::Legacy => {
                debug!(
                    "MCP server did not answer server/discover as a modern server; using initialize"
                );
                self.initialize_legacy().await
            }
            ProbeVerdict::Incompatible { reason } => Err(McpError::Protocol(reason)),
        }
    }

    /// Adopt the modern era from a `server/discover` answer.
    async fn finish_modern(
        &self,
        version: String,
        discover: serde_json::Value,
    ) -> Result<InitializeResult> {
        assert!(!version.is_empty(), "a modern era names its revision");
        let discover: DiscoverResult = serde_json::from_value(discover)?;
        debug_assert!(discover.supported_versions.contains(&version));
        let server_info = discover.server_info().unwrap_or_else(|| ServerInfo {
            name: "unnamed MCP server".to_string(),
            version: None,
        });
        info!(
            server = %server_info.name,
            version = server_info.version.as_deref().unwrap_or("unknown"),
            protocol = %version,
            "Connected to MCP server (modern, no handshake)"
        );
        *self.era.write().await = ProtocolEra::Modern {
            version: version.clone(),
        };
        let result = InitializeResult {
            protocol_version: version,
            capabilities: discover.capabilities,
            server_info,
            instructions: discover.instructions,
        };
        self.adopt(&result).await;
        self.open_listen(&result.capabilities).await;
        Ok(result)
    }

    /// Connect with the legacy `initialize` handshake only, skipping the
    /// era probe. For transports whose era detection is not the stdio probe
    /// (the deprecated HTTP+SSE transport speaks only this).
    ///
    /// # Errors
    ///
    /// Returns error if initialization fails or server rejects the connection
    pub async fn initialize_legacy(&self) -> Result<InitializeResult> {
        let params = InitializeParams {
            protocol_version: PROTOCOL_VERSION.to_string(),
            // No `roots`: this client has none to list, and a declared
            // capability invites a `roots/list` request it would only refuse.
            capabilities: ClientCapabilities {
                roots: None,
                sampling: None,
                experimental: None,
            },
            client_info: ClientInfo {
                name: "nanna".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
        };

        let result: InitializeResult = self
            .request("initialize", Some(serde_json::to_value(&params)?))
            .await?;

        info!(
            server = %result.server_info.name,
            version = result.server_info.version.as_deref().unwrap_or("unknown"),
            "Connected to MCP server"
        );

        // Send initialized notification
        self.transport
            .notify(JsonRpcNotification::new("notifications/initialized", None))
            .await?;
        self.adopt(&result).await;
        Ok(result)
    }

    /// Record a connected server's identity and capabilities, mark the client
    /// initialized, and pre-fetch the lists the server says it has.
    async fn adopt(&self, result: &InitializeResult) {
        *self.server_info.write().await = Some(result.server_info.clone());
        *self.capabilities.write().await = Some(result.capabilities.clone());
        *self.initialized.write().await = true;

        // A failed pre-fetch leaves that list empty — for tools, a server that
        // connected but offers nothing. That used to be silent, so "my MCP tools
        // are missing" had no trace in the log; now it names the server and why.
        let server = &result.server_info.name;
        if result.capabilities.tools.is_some() {
            match self.list_tools_internal().await {
                Ok(tools_result) => {
                    *self.tools.write().await = Self::gate_tool_schemas(tools_result.tools);
                }
                Err(e) => warn!(%server, error = %e, "MCP tools/list failed; no tools registered"),
            }
        }
        if result.capabilities.resources.is_some() {
            match self.list_resources_internal().await {
                Ok(resources_result) => {
                    *self.resources.write().await = resources_result.resources;
                }
                Err(e) => warn!(%server, error = %e, "MCP resources/list failed"),
            }
        }
        if result.capabilities.prompts.is_some() {
            match self.list_prompts_internal().await {
                Ok(prompts_result) => *self.prompts.write().await = prompts_result.prompts,
                Err(e) => warn!(%server, error = %e, "MCP prompts/list failed"),
            }
        }
    }

    /// Ask a modern server for its change notifications. Revision
    /// 2026-07-28 sends `list_changed` only on a `subscriptions/listen`
    /// stream, so without one a cached list never learns it is stale.
    /// Best effort: a failure is logged and the lists stay as fetched.
    async fn open_listen(&self, capabilities: &ServerCapabilities) {
        let Some(filter) = crate::era::listen_filter(capabilities) else {
            return;
        };
        let version = match &*self.era.read().await {
            ProtocolEra::Modern { version } => version.clone(),
            ProtocolEra::Legacy => return,
        };
        let params = match with_modern_meta(Some(filter), &version) {
            Ok(params) => params,
            Err(e) => {
                warn!(error = %e, "Could not build subscriptions/listen");
                return;
            }
        };
        let request = JsonRpcRequest::new(self.next_id(), "subscriptions/listen", Some(params));
        if let Err(e) = self.transport.open_listen(request).await {
            warn!(error = %e, "Could not open the MCP change-notification stream");
        }
    }

    /// The transport's list-changed flags, if it tracks any.
    #[must_use]
    pub fn list_changed_flags(&self) -> Option<Arc<crate::transport::ListChangedFlags>> {
        self.transport.list_changed_flags()
    }

    /// Mark the client connected without a handshake — for tests that drive
    /// a scripted transport through the real code paths.
    #[cfg(all(test, feature = "tools-integration"))]
    pub(crate) async fn mark_initialized_for_test(&self) {
        *self.initialized.write().await = true;
    }

    /// The protocol era this client settled on (`Legacy` before `initialize`).
    pub async fn era(&self) -> ProtocolEra {
        self.era.read().await.clone()
    }

    /// Check if client is initialized
    async fn ensure_initialized(&self) -> Result<()> {
        if !*self.initialized.read().await {
            return Err(McpError::NotInitialized);
        }
        Ok(())
    }

    // ========================================================================
    // Tools
    // ========================================================================

    /// Read-and-clear the transport's "server changed this list" flag, if the
    /// transport surfaces one. `true` means the local cache for `list` is stale
    /// and the caller should refresh before serving it.
    fn take_list_changed(&self, list: McpList) -> bool {
        self.transport
            .list_changed_flags()
            .is_some_and(|flags| flags.take(list))
    }

    /// List available tools (internal, no init check)
    /// Every page of `tools/list`, following `nextCursor`.
    ///
    /// Only the first page used to be read, so a server that paginates had
    /// every tool past its first page silently unregistered. Bounded by
    /// [`TOOL_LIST_PAGES_MAX`], and a cursor that repeats ends the walk: a
    /// server returning the same cursor forever would otherwise hold boot.
    async fn list_tools_internal(&self) -> Result<ListToolsResult> {
        let mut tools = Vec::new();
        let mut cursor: Option<String> = None;
        for _ in 0..TOOL_LIST_PAGES_MAX {
            let params = cursor.as_ref().map(|c| serde_json::json!({ "cursor": c }));
            let page: ListToolsResult = self.request("tools/list", params).await?;
            tools.extend(page.tools);
            match page.next_cursor {
                Some(next) if !next.is_empty() && cursor.as_ref() != Some(&next) => {
                    cursor = Some(next);
                }
                _ => {
                    return Ok(ListToolsResult {
                        tools,
                        next_cursor: None,
                    });
                }
            }
        }
        warn!(
            "MCP server still paginating after {TOOL_LIST_PAGES_MAX} tools/list pages; \
             using the {} tools read so far",
            tools.len()
        );
        Ok(ListToolsResult {
            tools,
            next_cursor: cursor,
        })
    }

    /// List available tools
    ///
    /// # Errors
    ///
    /// Returns error if not initialized or request fails
    pub async fn list_tools(&self) -> Result<Vec<Tool>> {
        self.ensure_initialized().await?;
        // If the server pushed a tools/list_changed, the cache is stale — refresh.
        if self.take_list_changed(McpList::Tools) {
            debug!("MCP tools list_changed — refreshing cache");
            return self.refresh_tools().await;
        }
        let tools = self.tools.read().await;
        Ok(tools.clone())
    }

    /// Refresh the tools cache
    ///
    /// # Errors
    ///
    /// Returns error if not initialized or request fails
    pub async fn refresh_tools(&self) -> Result<Vec<Tool>> {
        self.ensure_initialized().await?;
        let result = self.list_tools_internal().await?;
        let safe = Self::gate_tool_schemas(result.tools);
        self.tools.write().await.clone_from(&safe);
        Ok(safe)
    }

    /// Drop any tool whose server-supplied `input_schema` breaches the untrusted-schema
    /// bounds (over-deep, over-large, or carrying an external `$ref` we refuse to fetch).
    ///
    /// A single hostile or malformed tool must not deny the whole server's toolset, so
    /// this filters rather than failing the refresh — the offender is logged and skipped
    /// while every safe tool still reaches the cache.
    fn gate_tool_schemas(tools: Vec<Tool>) -> Vec<Tool> {
        let count_in = tools.len();
        let safe: Vec<Tool> = tools
            .into_iter()
            .filter(|tool| match validate_tool_schema(&tool.input_schema) {
                Ok(()) => true,
                Err(violation) => {
                    warn!(
                        tool = %tool.name,
                        %violation,
                        "Dropping MCP tool with an unsafe input schema"
                    );
                    false
                }
            })
            .collect();
        debug_assert!(
            safe.len() <= count_in,
            "gating can only drop tools, never add: {} in, {} out",
            count_in,
            safe.len()
        );
        safe
    }

    /// Call a tool by name
    ///
    /// # Errors
    ///
    /// Returns error if tool not found, not initialized, or execution fails
    pub async fn call_tool(
        &self,
        name: &str,
        arguments: Option<serde_json::Value>,
    ) -> Result<CallToolResult> {
        self.ensure_initialized().await?;

        debug!(tool = name, "Calling MCP tool");

        let params = serde_json::to_value(&CallToolParams {
            name: name.to_string(),
            arguments,
        })?;
        let result = match self.request_with_input("tools/call", params.clone()).await {
            // The server's headers and our cached tool definition disagree —
            // typically a parameter that gained `x-mcp-header` since our last
            // `tools/list`. The binding says: re-list, then retry. Once: a
            // second mismatch is the server's problem, not a stale cache.
            Err(McpError::JsonRpc { code, .. })
                if code == crate::protocol::error_codes::HEADER_MISMATCH =>
            {
                debug!(
                    tool = name,
                    "MCP header mismatch; re-listing tools and retrying once"
                );
                self.refresh_tools().await?;
                self.request_with_input("tools/call", params).await?
            }
            other => other?,
        };
        serde_json::from_value(result).map_err(Into::into)
    }

    /// Send `method` and serve any multi-round-trip input it asks for: while
    /// the server answers `input_required`, put its form questions to the
    /// elicitor and retry with the answers (a new request id each time, the
    /// server's `requestState` echoed verbatim), at most
    /// [`crate::elicit::MRTR_ROUNDS_MAX`] times.
    async fn request_with_input(
        &self,
        method: &str,
        original: serde_json::Value,
    ) -> Result<serde_json::Value> {
        use crate::elicit::{
            MRTR_ROUNDS_MAX, elicit_result, elicitation_question, form_requests, retry_params,
        };
        let mut params = original.clone();
        // Whether the last round got no answer at all: the server was told
        // `cancel`, so asking again cannot get a different reply.
        let mut unanswered = false;
        for round in 0..=MRTR_ROUNDS_MAX {
            let wire = self.with_era_meta(Some(params)).await?;
            let result = self.request_value_raw(method, wire).await?;
            let asks = result.get("resultType").and_then(serde_json::Value::as_str)
                == Some("input_required");
            let Some(elicitor) = self.elicitor.as_ref().filter(|_| asks) else {
                ensure_complete(&result)?;
                return Ok(result);
            };
            if unanswered {
                return Err(McpError::Protocol(
                    "the MCP server needs an answer from the user, and none came \
                     (no conversation to ask in, or no reply in time)"
                        .into(),
                ));
            }
            if round == MRTR_ROUNDS_MAX {
                break;
            }
            let server = self
                .server_info
                .read()
                .await
                .as_ref()
                .map_or_else(|| "unnamed".to_string(), |info| info.name.clone());
            let mut responses = serde_json::Map::new();
            let mut answered = false;
            for form in form_requests(&result)? {
                let question = elicitation_question(&server, &form.message, &form.schema);
                let answer = elicitor.ask(&question).await;
                answered |= answer.is_some();
                responses.insert(form.key.clone(), elicit_result(&form, answer.as_deref()));
            }
            unanswered = !answered;
            params = retry_params(&original, responses, result.get("requestState"));
        }
        Err(McpError::Protocol(format!(
            "the MCP server was still asking for input after {MRTR_ROUNDS_MAX} rounds"
        )))
    }

    /// Get a tool by name
    pub async fn get_tool(&self, name: &str) -> Option<Tool> {
        let tools = self.tools.read().await;
        tools.iter().find(|t| t.name == name).cloned()
    }

    // ========================================================================
    // Resources
    // ========================================================================

    /// List available resources (internal)
    async fn list_resources_internal(&self) -> Result<ListResourcesResult> {
        self.request("resources/list", None).await
    }

    /// List available resources
    ///
    /// # Errors
    ///
    /// Returns error if not initialized or request fails
    pub async fn list_resources(&self) -> Result<Vec<Resource>> {
        self.ensure_initialized().await?;
        // If the server pushed a resources/list_changed, the cache is stale — refresh.
        if self.take_list_changed(McpList::Resources) {
            debug!("MCP resources list_changed — refreshing cache");
            return self.refresh_resources().await;
        }
        let resources = self.resources.read().await;
        Ok(resources.clone())
    }

    /// Refresh the resources cache
    ///
    /// # Errors
    ///
    /// Returns error if not initialized or request fails
    pub async fn refresh_resources(&self) -> Result<Vec<Resource>> {
        self.ensure_initialized().await?;
        let result = self.list_resources_internal().await?;
        self.resources.write().await.clone_from(&result.resources);
        Ok(result.resources)
    }

    /// Read a resource by URI
    ///
    /// # Errors
    ///
    /// Returns error if resource not found, not initialized, or read fails
    pub async fn read_resource(&self, uri: &str) -> Result<ReadResourceResult> {
        self.ensure_initialized().await?;

        debug!(uri, "Reading MCP resource");

        let params = ReadResourceParams {
            uri: uri.to_string(),
        };

        self.request("resources/read", Some(serde_json::to_value(&params)?))
            .await
            .map_err(|e| resource_error_for(uri, e))
    }

    // ========================================================================
    // Prompts
    // ========================================================================

    /// List available prompts (internal)
    async fn list_prompts_internal(&self) -> Result<ListPromptsResult> {
        self.request("prompts/list", None).await
    }

    /// List available prompts
    ///
    /// # Errors
    ///
    /// Returns error if not initialized or request fails
    pub async fn list_prompts(&self) -> Result<Vec<Prompt>> {
        self.ensure_initialized().await?;
        // If the server pushed a prompts/list_changed, the cache is stale — refresh.
        if self.take_list_changed(McpList::Prompts) {
            debug!("MCP prompts list_changed — refreshing cache");
            return self.refresh_prompts().await;
        }
        let prompts = self.prompts.read().await;
        Ok(prompts.clone())
    }

    /// Refresh the prompts cache
    ///
    /// # Errors
    ///
    /// Returns error if not initialized or request fails
    pub async fn refresh_prompts(&self) -> Result<Vec<Prompt>> {
        self.ensure_initialized().await?;
        let result = self.list_prompts_internal().await?;
        self.prompts.write().await.clone_from(&result.prompts);
        Ok(result.prompts)
    }

    /// Get a prompt by name with arguments
    ///
    /// # Errors
    ///
    /// Returns error if prompt not found, not initialized, or retrieval fails
    pub async fn get_prompt(
        &self,
        name: &str,
        arguments: Option<HashMap<String, String>>,
    ) -> Result<GetPromptResult> {
        self.ensure_initialized().await?;

        debug!(prompt = name, "Getting MCP prompt");

        let params = GetPromptParams {
            name: name.to_string(),
            arguments,
        };

        self.request("prompts/get", Some(serde_json::to_value(&params)?))
            .await
    }

    // ========================================================================
    // Server Info
    // ========================================================================

    /// Get server info
    pub async fn server_info(&self) -> Option<ServerInfo> {
        let info = self.server_info.read().await;
        info.clone()
    }

    /// Get server capabilities
    pub async fn capabilities(&self) -> Option<ServerCapabilities> {
        let caps = self.capabilities.read().await;
        caps.clone()
    }

    /// Check if server supports tools
    pub async fn supports_tools(&self) -> bool {
        let caps = self.capabilities.read().await;
        caps.as_ref().is_some_and(|c| c.tools.is_some())
    }

    /// Check if server supports resources
    pub async fn supports_resources(&self) -> bool {
        let caps = self.capabilities.read().await;
        caps.as_ref().is_some_and(|c| c.resources.is_some())
    }

    /// Check if server supports prompts
    pub async fn supports_prompts(&self) -> bool {
        let caps = self.capabilities.read().await;
        caps.as_ref().is_some_and(|c| c.prompts.is_some())
    }

    // ========================================================================
    // Lifecycle
    // ========================================================================

    /// Close the connection
    ///
    /// # Errors
    ///
    /// Returns error if close fails
    pub async fn close(&self) -> Result<()> {
        self.transport.close().await
    }
}

/// Builder for creating MCP clients with configuration
pub struct McpClientBuilder {
    /// Custom client name
    client_name: Option<String>,
    /// Custom client version
    client_version: Option<String>,
}

impl Default for McpClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl McpClientBuilder {
    /// Create a new builder
    #[must_use]
    pub const fn new() -> Self {
        Self {
            client_name: None,
            client_version: None,
        }
    }

    /// Set custom client name
    #[must_use]
    pub fn client_name(mut self, name: impl Into<String>) -> Self {
        self.client_name = Some(name.into());
        self
    }

    /// Set custom client version
    #[must_use]
    pub fn client_version(mut self, version: impl Into<String>) -> Self {
        self.client_version = Some(version.into());
        self
    }

    /// Build the client with the given transport
    pub fn build<T: Transport>(self, transport: T) -> McpClient<T> {
        McpClient::new(transport)
    }
}

// ============================================================================
// Convenience functions for spawning common servers
// ============================================================================

#[cfg(feature = "stdio")]
impl McpClient<crate::StdioTransport> {
    /// Spawn an MCP server and create a connected client
    ///
    /// # Errors
    ///
    /// Returns error if spawn or initialization fails
    pub async fn spawn(program: &str, args: &[&str]) -> Result<Self> {
        let transport = crate::StdioTransport::spawn(program, args)?;
        let client = Self::new(transport);
        client.initialize().await?;
        Ok(client)
    }

    /// Spawn with environment variables
    ///
    /// # Errors
    ///
    /// Returns error if spawn or initialization fails
    pub async fn spawn_with_env(
        program: &str,
        args: &[&str],
        env: &[(&str, &str)],
    ) -> Result<Self> {
        let transport = crate::StdioTransport::spawn_with_env(program, args, env)?;
        let client = Self::new(transport);
        client.initialize().await?;
        Ok(client)
    }
}

#[cfg(feature = "http")]
impl McpClient<crate::StreamableHttpTransport> {
    /// Connect to a Streamable HTTP MCP endpoint in whichever era it speaks:
    /// a modern (2026-07-28) server is used statelessly, a 2025-era one
    /// through its `initialize` handshake and session. `bearer_token` is sent
    /// as `Authorization: Bearer` on every request.
    ///
    /// # Errors
    ///
    /// Returns error if the URL is not http(s), the server refuses the
    /// credential, or the connection fails.
    pub async fn connect_streamable(
        url: impl Into<String>,
        bearer_token: Option<String>,
    ) -> Result<Self> {
        let transport = crate::StreamableHttpTransport::new(url, bearer_token)?;
        let client = Self::new(transport);
        client.initialize().await?;
        Ok(client)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::JsonRpcResponse;
    use crate::transport::ListChangedFlags;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn test_protocol_version() {
        assert_ne!(PROTOCOL_VERSION, "");
    }

    /// A transport whose `tools/list` reply encodes how many times it has been
    /// called (tool name `tool-N`), so a test can tell a cache hit (stale) from a
    /// refresh (a new request). Surfaces a shared `ListChangedFlags` like stdio.
    struct CountingTransport {
        flags: Arc<ListChangedFlags>,
        tools_list_calls: AtomicUsize,
    }

    #[async_trait::async_trait]
    impl Transport for CountingTransport {
        async fn request(&self, request: JsonRpcRequest) -> Result<JsonRpcResponse> {
            let n = self.tools_list_calls.fetch_add(1, Ordering::SeqCst);
            let result = serde_json::json!({
                "tools": [{ "name": format!("tool-{n}"), "inputSchema": {} }]
            });
            Ok(JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id: request.id,
                result: Some(result),
                error: None,
            })
        }
        async fn notify(&self, _n: JsonRpcNotification) -> Result<()> {
            Ok(())
        }
        async fn close(&self) -> Result<()> {
            Ok(())
        }
        fn list_changed_flags(&self) -> Option<Arc<ListChangedFlags>> {
            Some(self.flags.clone())
        }
    }

    #[tokio::test]
    async fn list_tools_refreshes_only_when_flag_is_dirty() {
        let flags = Arc::new(ListChangedFlags::default());
        let client = McpClient::new(CountingTransport {
            flags: flags.clone(),
            tools_list_calls: AtomicUsize::new(0),
        });
        *client.initialized.write().await = true;

        // Prime the cache with the server's first list (tool-0).
        let primed = client.refresh_tools().await.unwrap();
        assert_eq!(primed[0].name, "tool-0");

        // With no list_changed pending, list_tools serves the cache — no new request.
        let cached = client.list_tools().await.unwrap();
        assert_eq!(cached[0].name, "tool-0");

        // Server announces tools/list_changed → next list_tools refreshes. This is
        // the 2nd `tools/list` request (call index 1, since the cache hit above
        // issued none), so the server returns tool-1.
        flags.mark(McpList::Tools);
        let refreshed = client.list_tools().await.unwrap();
        assert_eq!(refreshed[0].name, "tool-1");

        // Flag was consumed: a following call serves the refreshed cache, no request.
        let after = client.list_tools().await.unwrap();
        assert_eq!(after[0].name, "tool-1");
    }

    /// A server that paginates `tools/list` two tools at a time over three
    /// pages, and one that never stops returning the same cursor.
    struct PagingTransport {
        stuck: bool,
        pages_served: AtomicUsize,
    }

    #[async_trait::async_trait]
    impl Transport for PagingTransport {
        async fn request(&self, request: JsonRpcRequest) -> Result<JsonRpcResponse> {
            self.pages_served.fetch_add(1, Ordering::SeqCst);
            let cursor = request
                .params
                .as_ref()
                .and_then(|p| p.get("cursor"))
                .and_then(serde_json::Value::as_str)
                .map(str::to_string);
            let (page, next) = match (self.stuck, cursor.as_deref()) {
                (true, _) => (0, Some("again")),
                (false, None) => (0, Some("p1")),
                (false, Some("p1")) => (1, Some("p2")),
                (false, _) => (2, None),
            };
            let result = serde_json::json!({
                "tools": [
                    { "name": format!("t{page}a"), "inputSchema": {} },
                    { "name": format!("t{page}b"), "inputSchema": {} },
                ],
                "nextCursor": next,
            });
            Ok(JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id: request.id,
                result: Some(result),
                error: None,
            })
        }
        async fn notify(&self, _n: JsonRpcNotification) -> Result<()> {
            Ok(())
        }
        async fn close(&self) -> Result<()> {
            Ok(())
        }
    }

    /// Only the first page was ever read: a paginating server had every tool
    /// past it silently unregistered.
    #[tokio::test]
    async fn every_page_of_tools_list_is_read_and_a_repeating_cursor_ends_it() {
        let client = McpClient::new(PagingTransport {
            stuck: false,
            pages_served: AtomicUsize::new(0),
        });
        *client.initialized.write().await = true;
        let names: Vec<String> = client
            .refresh_tools()
            .await
            .unwrap()
            .into_iter()
            .map(|t| t.name)
            .collect();
        assert_eq!(names, ["t0a", "t0b", "t1a", "t1b", "t2a", "t2b"]);

        let stuck = McpClient::new(PagingTransport {
            stuck: true,
            pages_served: AtomicUsize::new(0),
        });
        *stuck.initialized.write().await = true;
        let tools = stuck.refresh_tools().await.unwrap();
        assert_eq!(
            tools.len(),
            4,
            "the first page, then the repeated cursor stops the walk"
        );
    }

    /// A transport that returns a fixed mix of safe and unsafe tool schemas, so a test
    /// can assert the ingest gate drops only the unsafe ones.
    struct MixedSchemaTransport;

    #[async_trait::async_trait]
    impl Transport for MixedSchemaTransport {
        async fn request(&self, request: JsonRpcRequest) -> Result<JsonRpcResponse> {
            let result = serde_json::json!({
                "tools": [
                    { "name": "safe", "inputSchema": {
                        "type": "object",
                        "properties": { "path": { "type": "string" } }
                    }},
                    { "name": "external-ref", "inputSchema": {
                        "type": "object",
                        "properties": { "x": { "$ref": "https://evil.example/s.json" } }
                    }},
                    { "name": "internal-ref-ok", "inputSchema": {
                        "type": "object",
                        "properties": { "y": { "$ref": "#/$defs/y" } }
                    }},
                ]
            });
            Ok(JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id: request.id,
                result: Some(result),
                error: None,
            })
        }
        async fn notify(&self, _n: JsonRpcNotification) -> Result<()> {
            Ok(())
        }
        async fn close(&self) -> Result<()> {
            Ok(())
        }
        fn list_changed_flags(&self) -> Option<Arc<ListChangedFlags>> {
            None
        }
    }

    #[tokio::test]
    async fn refresh_tools_drops_tools_with_unsafe_schemas() {
        let client = McpClient::new(MixedSchemaTransport);
        *client.initialized.write().await = true;

        let tools = client.refresh_tools().await.unwrap();
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();

        // The external-$ref tool is gated out; the safe tool and the internal-fragment
        // ref survive (a `#/…` ref needs no fetch).
        assert!(names.contains(&"safe"), "safe tool must survive: {names:?}");
        assert!(
            names.contains(&"internal-ref-ok"),
            "internal fragment ref must survive: {names:?}"
        );
        assert!(
            !names.contains(&"external-ref"),
            "external $ref tool must be dropped: {names:?}"
        );
        assert_eq!(tools.len(), 2, "exactly one tool should be gated out");

        // The cache reflects the gated set too (not just the returned Vec).
        let cached = client.list_tools().await.unwrap();
        assert_eq!(cached.len(), 2, "cache must hold only the safe tools");
    }

    /// A transport that answers every request with a fixed JSON-RPC error code,
    /// so a test can pin how `read_resource` classifies each spec revision.
    struct ErrorCodeTransport(i32);

    #[async_trait::async_trait]
    impl Transport for ErrorCodeTransport {
        async fn request(&self, request: JsonRpcRequest) -> Result<JsonRpcResponse> {
            Ok(JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id: request.id,
                result: None,
                error: Some(crate::protocol::JsonRpcError {
                    code: self.0,
                    message: "nope".to_string(),
                    data: None,
                }),
            })
        }
        async fn notify(&self, _n: JsonRpcNotification) -> Result<()> {
            Ok(())
        }
        async fn close(&self) -> Result<()> {
            Ok(())
        }
        fn list_changed_flags(&self) -> Option<Arc<ListChangedFlags>> {
            None
        }
    }

    async fn read_missing(code: i32) -> McpError {
        let client = McpClient::new(ErrorCodeTransport(code));
        *client.initialized.write().await = true;
        client
            .read_resource("file:///gone.txt")
            .await
            .expect_err("an error response must surface as an error")
    }

    #[tokio::test]
    async fn read_resource_maps_both_missing_codes_to_resource_not_found() {
        // Legacy (-32002) and 2026-07-28 (-32602) must be indistinguishable to
        // the caller, and the URI must be carried through.
        for code in [
            crate::protocol::error_codes::LEGACY_RESOURCE_NOT_FOUND,
            crate::protocol::error_codes::INVALID_PARAMS,
        ] {
            match read_missing(code).await {
                McpError::ResourceNotFound(uri) => {
                    assert_eq!(uri, "file:///gone.txt", "code {code} must carry the URI");
                }
                other => panic!("code {code} should map to ResourceNotFound, got {other:?}"),
            }
        }
    }

    #[tokio::test]
    async fn read_resource_leaves_unrelated_errors_untouched() {
        // Negative space: a real protocol fault must NOT be laundered into
        // "resource not found" — that would hide a broken server as an empty read.
        match read_missing(-32601).await {
            McpError::JsonRpc { code, .. } => assert_eq!(code, -32601),
            other => panic!("method-not-found must pass through, got {other:?}"),
        }
    }

    /// A server that answers like either reference SDK: `modern` like
    /// `@modelcontextprotocol/server` 2.0 (discover answered, `initialize`
    /// rejected with -32022), otherwise like `@modelcontextprotocol/sdk` 1.30
    /// (discover is `-32601`, `initialize` answered). Records what it saw.
    struct EraTransport {
        modern: bool,
        requests: std::sync::Mutex<Vec<JsonRpcRequest>>,
        notifications: std::sync::Mutex<Vec<String>>,
    }

    impl EraTransport {
        fn new(modern: bool) -> Self {
            Self {
                modern,
                requests: std::sync::Mutex::new(Vec::new()),
                notifications: std::sync::Mutex::new(Vec::new()),
            }
        }

        fn answer(
            &self,
            method: &str,
        ) -> std::result::Result<serde_json::Value, (i32, serde_json::Value)> {
            let tools = serde_json::json!({ "tools": [{ "name": "shout", "inputSchema": { "type": "object" } }] });
            match (self.modern, method) {
                (true, "server/discover") => Ok(serde_json::json!({
                    "supportedVersions": ["2026-07-28"],
                    "capabilities": { "tools": { "listChanged": true } },
                    "resultType": "complete",
                    "_meta": { "io.modelcontextprotocol/serverInfo": { "name": "modern", "version": "2" } }
                })),
                (true, "initialize") => {
                    Err((-32022, serde_json::json!({ "supported": ["2026-07-28"] })))
                }
                (false, "initialize") => Ok(serde_json::json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": { "tools": {} },
                    "serverInfo": { "name": "legacy" }
                })),
                (_, "tools/list") => Ok(tools),
                _ => Err((-32601, serde_json::Value::Null)),
            }
        }
    }

    #[async_trait::async_trait]
    impl Transport for EraTransport {
        async fn request(&self, request: JsonRpcRequest) -> Result<JsonRpcResponse> {
            let answer = self.answer(&request.method);
            let id = request.id.clone();
            self.requests.lock().unwrap().push(request);
            let (result, error) = match answer {
                Ok(result) => (Some(result), None),
                Err((code, data)) => (
                    None,
                    Some(crate::protocol::JsonRpcError {
                        code,
                        message: "refused".to_string(),
                        data: Some(data),
                    }),
                ),
            };
            Ok(JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id,
                result,
                error,
            })
        }
        async fn notify(&self, n: JsonRpcNotification) -> Result<()> {
            self.notifications.lock().unwrap().push(n.method);
            Ok(())
        }
        async fn close(&self) -> Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn a_modern_server_is_spoken_to_without_a_handshake() {
        let client = McpClient::new(EraTransport::new(true));
        let connected = client
            .initialize()
            .await
            .expect("modern server must connect");
        assert_eq!(connected.protocol_version, "2026-07-28");
        assert_eq!(connected.server_info.name, "modern");
        assert_eq!(
            client.era().await,
            ProtocolEra::Modern {
                version: "2026-07-28".into()
            }
        );
        assert_eq!(client.list_tools().await.unwrap()[0].name, "shout");

        let transport = &client.transport;
        let methods: Vec<String> = transport
            .requests
            .lock()
            .unwrap()
            .iter()
            .map(|r| r.method.clone())
            .collect();
        assert_eq!(
            methods,
            ["server/discover", "tools/list"],
            "no initialize to a modern server"
        );
        assert!(
            transport.notifications.lock().unwrap().is_empty(),
            "no notifications/initialized"
        );
        for request in transport.requests.lock().unwrap().iter() {
            let meta = &request
                .params
                .as_ref()
                .expect("modern requests carry params")["_meta"];
            assert_eq!(
                meta["io.modelcontextprotocol/protocolVersion"], "2026-07-28",
                "{}",
                request.method
            );
            assert!(
                meta["io.modelcontextprotocol/clientCapabilities"].is_object(),
                "{}",
                request.method
            );
        }
    }

    #[tokio::test]
    async fn a_legacy_server_falls_back_to_initialize() {
        let client = McpClient::new(EraTransport::new(false));
        let connected = client
            .initialize()
            .await
            .expect("legacy server must connect");
        assert_eq!(connected.server_info.name, "legacy");
        assert_eq!(client.era().await, ProtocolEra::Legacy);
        assert_eq!(client.list_tools().await.unwrap()[0].name, "shout");

        let transport = &client.transport;
        let requests = transport.requests.lock().unwrap();
        let methods: Vec<&str> = requests.iter().map(|r| r.method.as_str()).collect();
        assert_eq!(methods, ["server/discover", "initialize", "tools/list"]);
        assert!(
            requests[2].params.is_none(),
            "legacy requests carry no modern _meta"
        );
        drop(requests);
        assert_eq!(
            *transport.notifications.lock().unwrap(),
            ["notifications/initialized"]
        );
    }

    /// A modern server whose `tools/call` needs client input (MRTR).
    struct InputRequiredTransport;

    #[async_trait::async_trait]
    impl Transport for InputRequiredTransport {
        async fn request(&self, request: JsonRpcRequest) -> Result<JsonRpcResponse> {
            Ok(JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id: request.id,
                result: Some(
                    serde_json::json!({ "resultType": "input_required", "inputRequests": {} }),
                ),
                error: None,
            })
        }
        async fn notify(&self, _n: JsonRpcNotification) -> Result<()> {
            Ok(())
        }
        async fn close(&self) -> Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn an_input_required_result_is_an_error_not_an_empty_success() {
        let client = McpClient::new(InputRequiredTransport);
        *client.initialized.write().await = true;
        match client.call_tool("ask", None).await {
            Err(McpError::Protocol(message)) => {
                assert!(message.contains("more input"), "{message}");
            }
            other => panic!("expected a protocol error, got {other:?}"),
        }
    }

    /// A modern server whose `tools/call` asks for a color until it gets one
    /// (or forever, with `always_ask`). Records every call's params and id.
    struct AskingServer {
        always_ask: bool,
        calls: std::sync::Mutex<Vec<(RequestId, serde_json::Value)>>,
    }

    #[async_trait::async_trait]
    impl Transport for AskingServer {
        async fn request(&self, request: JsonRpcRequest) -> Result<JsonRpcResponse> {
            let params = request.params.clone().unwrap_or_default();
            self.calls
                .lock()
                .unwrap()
                .push((request.id.clone(), params.clone()));
            let color = params["inputResponses"]["color"]["content"]["color"].as_str();
            let result = match color {
                Some(color) if !self.always_ask => serde_json::json!({
                    "resultType": "complete",
                    "content": [{ "type": "text", "text": format!("favorite={color}") }]
                }),
                _ => serde_json::json!({
                    "resultType": "input_required",
                    "inputRequests": { "color": { "method": "elicitation/create", "params": {
                        "mode": "form", "message": "Favorite color?",
                        "requestedSchema": { "type": "object",
                            "properties": { "color": { "type": "string" } }, "required": ["color"] }
                    } } },
                    "requestState": "state-1"
                }),
            };
            Ok(JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id: request.id,
                result: Some(result),
                error: None,
            })
        }
        async fn notify(&self, _n: JsonRpcNotification) -> Result<()> {
            Ok(())
        }
        async fn close(&self) -> Result<()> {
            Ok(())
        }
    }

    struct Replies(std::sync::Mutex<Vec<String>>, Option<&'static str>);

    #[async_trait::async_trait]
    impl crate::elicit::Elicitor for Replies {
        async fn ask(&self, question: &str) -> Option<String> {
            self.0.lock().unwrap().push(question.to_string());
            self.1.map(str::to_string)
        }
    }

    async fn modern_client(
        always_ask: bool,
        reply: Option<&'static str>,
    ) -> (McpClient<AskingServer>, Arc<Replies>) {
        let replies = Arc::new(Replies(std::sync::Mutex::new(Vec::new()), reply));
        let client = McpClient::new(AskingServer {
            always_ask,
            calls: std::sync::Mutex::new(Vec::new()),
        })
        .with_elicitor(replies.clone());
        *client.initialized.write().await = true;
        *client.era.write().await = ProtocolEra::Modern {
            version: "2026-07-28".into(),
        };
        (client, replies)
    }

    /// A server whose first `tools/call` is refused with `-32020`; records
    /// every method so the recovery order can be checked.
    struct MismatchOnce {
        methods: std::sync::Mutex<Vec<String>>,
        refusals_left: std::sync::atomic::AtomicUsize,
    }

    #[async_trait::async_trait]
    impl Transport for MismatchOnce {
        async fn request(&self, request: JsonRpcRequest) -> Result<JsonRpcResponse> {
            self.methods.lock().unwrap().push(request.method.clone());
            let (result, error) = match request.method.as_str() {
                "tools/call"
                    if self
                        .refusals_left
                        .try_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
                        .is_ok() =>
                {
                    (
                        None,
                        Some(crate::protocol::JsonRpcError {
                            code: -32020,
                            message: "Mcp-Param-Region header is absent".into(),
                            data: None,
                        }),
                    )
                }
                "tools/call" => (
                    Some(serde_json::json!({ "content": [{ "type": "text", "text": "ok" }] })),
                    None,
                ),
                _ => (Some(serde_json::json!({ "tools": [] })), None),
            };
            Ok(JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id: request.id,
                result,
                error,
            })
        }
        async fn notify(&self, _n: JsonRpcNotification) -> Result<()> {
            Ok(())
        }
        async fn close(&self) -> Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn a_header_mismatch_relists_tools_and_retries_once() {
        let client = McpClient::new(MismatchOnce {
            methods: std::sync::Mutex::new(Vec::new()),
            refusals_left: AtomicUsize::new(1),
        });
        *client.initialized.write().await = true;
        client.call_tool("regional", None).await.expect("recovered");
        let methods = client.transport.methods.lock().unwrap().clone();
        assert_eq!(methods, ["tools/call", "tools/list", "tools/call"]);

        // A server that keeps refusing gets one retry, not a loop.
        let stubborn = McpClient::new(MismatchOnce {
            methods: std::sync::Mutex::new(Vec::new()),
            refusals_left: AtomicUsize::new(5),
        });
        *stubborn.initialized.write().await = true;
        let error = stubborn
            .call_tool("regional", None)
            .await
            .expect_err("still refused");
        assert!(
            matches!(error, McpError::JsonRpc { code: -32020, .. }),
            "{error:?}"
        );
        assert_eq!(stubborn.transport.methods.lock().unwrap().len(), 3);
    }

    #[tokio::test]
    async fn an_elicitation_is_asked_and_the_call_retried_with_the_answer() {
        let (client, replies) = modern_client(false, Some("teal")).await;
        let result = client
            .call_tool("favorite", None)
            .await
            .expect("answered on the retry");
        let text = serde_json::to_value(&result).unwrap()["content"][0]["text"].clone();
        assert_eq!(text, "favorite=teal");

        let asked = replies.0.lock().unwrap().clone();
        assert_eq!(asked.len(), 1);
        assert!(asked[0].contains("Favorite color?"), "{}", asked[0]);

        let calls = client.transport.calls.lock().unwrap().clone();
        assert_eq!(calls.len(), 2);
        assert_ne!(calls[0].0, calls[1].0, "the retry is a new request id");
        let retry = &calls[1].1;
        assert_eq!(retry["requestState"], "state-1", "state echoed verbatim");
        assert_eq!(retry["name"], "favorite", "the original params are kept");
        assert_eq!(
            retry["_meta"]["io.modelcontextprotocol/clientCapabilities"]["elicitation"],
            serde_json::json!({ "form": {} }),
            "elicitation is declared when an elicitor is installed"
        );
        assert!(calls[0].1.get("inputResponses").is_none());
    }

    #[tokio::test]
    async fn no_answer_is_sent_as_cancel_once_then_the_call_ends() {
        let (client, _) = modern_client(false, None).await;
        let error = client
            .call_tool("favorite", None)
            .await
            .expect_err("unanswered");
        assert!(error.to_string().contains("none came"), "{error}");
        let calls = client.transport.calls.lock().unwrap().clone();
        assert_eq!(calls.len(), 2, "one cancel retry, not a cancel per round");
        assert_eq!(
            calls[1].1["inputResponses"]["color"],
            serde_json::json!({ "action": "cancel" })
        );
    }

    #[tokio::test]
    async fn a_server_that_never_stops_asking_is_bounded() {
        let (client, replies) = modern_client(true, Some("teal")).await;
        let error = client
            .call_tool("favorite", None)
            .await
            .expect_err("bounded");
        assert!(error.to_string().contains("still asking"), "{error}");
        let calls = client.transport.calls.lock().unwrap().clone();
        assert_eq!(calls.len(), crate::elicit::MRTR_ROUNDS_MAX + 1);
        assert_eq!(
            replies.0.lock().unwrap().len(),
            crate::elicit::MRTR_ROUNDS_MAX
        );
    }
}
