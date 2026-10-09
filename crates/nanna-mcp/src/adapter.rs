//! Adapter for integrating MCP tools with nanna-tools
//!
//! Provides a bridge between MCP tool servers and the nanna-tools registry,
//! allowing MCP tools to be used seamlessly in the agent loop.

use crate::transport::Transport;
use crate::{McpClient, McpError, Tool as McpTool, ToolContent};
use std::collections::HashMap;
use std::sync::Arc;
#[cfg(feature = "tools-integration")]
use tokio::sync::RwLock;
use tracing::{debug, warn};

/// Result from executing an MCP tool
#[derive(Debug, Clone)]
pub struct McpToolResult {
    /// Text content from the tool
    pub content: String,
    /// Whether the tool reported an error
    pub is_error: bool,
    /// Raw content blocks from MCP
    pub raw: Vec<ToolContent>,
    /// The server's machine-readable payload, if it sent one. Any JSON value
    /// (2026-07-28 lifted the object-only restriction), passed through verbatim.
    pub structured: Option<serde_json::Value>,
}

/// Longest tool name every provider Nanna routes to accepts.
///
/// Anthropic and `OpenAI` both validate tool names against
/// `^[a-zA-Z0-9_-]{1,64}$`; a name outside it rejects the WHOLE request, not
/// just the tool, so one badly named MCP tool would break every turn.
pub const WIRE_TOOL_NAME_CHARS_MAX: usize = 64;

/// Characters kept from a truncated name when a hash suffix is appended.
const WIRE_TOOL_NAME_STEM_CHARS: usize = WIRE_TOOL_NAME_CHARS_MAX - 1 - 8;

/// The model-facing name of MCP tool `tool` on server `server`. Pure.
///
/// `mcp__{server}__{tool}`, the convention MCP clients have settled on, with
/// every character outside `[A-Za-z0-9_-]` replaced by `_`. The previous
/// `server:tool` form was invalid for every provider (`:` is not allowed).
/// A name longer than 64 characters keeps its first 55 and gains `_` plus an
/// 8-hex FNV-1a hash of the full unsanitized name, so two long tools that share
/// a prefix still get distinct names, and the same tool always gets the same
/// one.
#[must_use]
pub fn wire_tool_name(server: &str, tool: &str) -> String {
    let joined = if server.is_empty() {
        tool.to_string()
    } else {
        format!("mcp__{server}__{tool}")
    };
    let sanitized: String = joined
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    debug_assert!(
        sanitized.is_ascii(),
        "sanitized names are ASCII by construction"
    );
    if sanitized.len() <= WIRE_TOOL_NAME_CHARS_MAX {
        return sanitized;
    }
    let hash = joined.bytes().fold(0x811c_9dc5_u32, |hash, byte| {
        (hash ^ u32::from(byte)).wrapping_mul(0x0100_0193)
    });
    let name = format!("{}_{hash:08x}", &sanitized[..WIRE_TOOL_NAME_STEM_CHARS]);
    debug_assert_eq!(name.len(), WIRE_TOOL_NAME_CHARS_MAX);
    name
}

// ============================================================================
// nanna-tools Integration
// ============================================================================

#[cfg(feature = "tools-integration")]
mod tools_impl {
    use super::{debug, warn, wire_tool_name, Transport, Arc, McpClient, McpTool, HashMap, ToolContent, RwLock, McpError};
    use async_trait::async_trait;
    use nanna_tools::{ParameterType, Tool, ToolDefinition, ToolError, ToolParameter, ToolResult};
    use serde_json::Value;
    use std::collections::HashSet;

    /// Least time between two registry resyncs in
    /// [`McpToolsManager::watch_list_changes`]: one second coalesces a burst
    /// of announcements into one `tools/list` without making a real change
    /// wait noticeably.
    pub const RESYNC_INTERVAL_MIN: std::time::Duration = std::time::Duration::from_secs(1);

    /// Wrapper that adapts an MCP tool to the nanna-tools `Tool` trait
    pub struct McpToolWrapper<T: Transport + 'static> {
        /// The MCP client
        client: Arc<McpClient<T>>,
        /// The tool definition from MCP
        tool: McpTool,
        /// Prefix for the tool name (usually server name)
        prefix: String,
    }

    impl<T: Transport + 'static> McpToolWrapper<T> {
        /// Create a new wrapper for an MCP tool
        pub fn new(client: Arc<McpClient<T>>, tool: McpTool, prefix: impl Into<String>) -> Self {
            Self {
                client,
                tool,
                prefix: prefix.into(),
            }
        }

        /// The model-facing tool name — see [`wire_tool_name`].
        #[must_use]
        pub fn full_name(&self) -> String {
            wire_tool_name(&self.prefix, &self.tool.name)
        }
    }

    /// Convert an MCP tool `input_schema` (JSON Schema 2020-12) into nanna-tools
    /// parameters.
    ///
    /// Reads the top-level `properties`/`required`, and — because the 2026-07-28 spec
    /// lifts input schemas to full 2020-12 composition — also folds in the properties of
    /// each `allOf`/`anyOf`/`oneOf` branch (one level deep). Without this, a tool that
    /// describes its inputs via composition would silently yield **zero** parameters, so
    /// the model would call it with no arguments.
    ///
    /// A property is marked required only when the root schema or an `allOf` branch (all
    /// of which must hold) requires it; `anyOf`/`oneOf` branches contribute *optional*
    /// properties, since only one branch applies to any given call. Property order is
    /// root-first, then branch order, first definition of a name winning — bounded by the
    /// finite schema (already depth/size-capped at ingest by `schema_guard`).
    fn schema_to_parameters(schema: &Value) -> Vec<ToolParameter> {
        // Ordered accumulation so parameter order is stable; `seen` enforces first-wins.
        let mut ordered_props: Vec<(String, Value)> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        let mut required: HashSet<String> = HashSet::new();

        // Root schema: its properties and its hard requireds.
        collect_schema_object(schema, &mut ordered_props, &mut seen, &mut required, true);

        // Composition branches, one level deep. `allOf` requireds are hard (every branch
        // must hold); `anyOf`/`oneOf` requireds are not, since only one branch applies.
        for (keyword, requireds_are_hard) in [("allOf", true), ("anyOf", false), ("oneOf", false)] {
            if let Some(branches) = schema.get(keyword).and_then(Value::as_array) {
                for branch in branches {
                    collect_schema_object(
                        branch,
                        &mut ordered_props,
                        &mut seen,
                        &mut required,
                        requireds_are_hard,
                    );
                }
            }
        }

        debug_assert_eq!(
            ordered_props.len(),
            seen.len(),
            "each accumulated property name must be unique"
        );
        ordered_props
            .into_iter()
            .map(|(name, prop)| {
                let is_required = required.contains(&name);
                property_to_parameter(name, &prop, is_required)
            })
            .collect()
    }

    /// Fold one schema object's `properties` (ordered, first-wins via `seen`) and, when
    /// `requireds_are_hard`, its `required` names into the accumulators. A non-object
    /// branch or one without `properties` contributes nothing.
    fn collect_schema_object(
        schema: &Value,
        ordered_props: &mut Vec<(String, Value)>,
        seen: &mut HashSet<String>,
        required: &mut HashSet<String>,
        requireds_are_hard: bool,
    ) {
        if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
            for (name, prop) in properties {
                if seen.insert(name.clone()) {
                    ordered_props.push((name.clone(), prop.clone()));
                }
            }
        }
        if requireds_are_hard && let Some(names) = schema.get("required").and_then(Value::as_array)
        {
            for name in names.iter().filter_map(Value::as_str) {
                required.insert(name.to_string());
            }
        }
        debug_assert_eq!(
            ordered_props.len(),
            seen.len(),
            "every pushed property is recorded in `seen`, so the two stay in lockstep"
        );
    }

    /// Build a `ToolParameter` from a single JSON Schema property node. Unknown/absent
    /// `type` falls back to `String` (the safest wire type for an unconstrained value).
    fn property_to_parameter(name: String, prop: &Value, required: bool) -> ToolParameter {
        let param_type =
            prop.get("type")
                .and_then(Value::as_str)
                .map_or(ParameterType::String, |t| match t {
                    "integer" => ParameterType::Integer,
                    "number" => ParameterType::Number,
                    "boolean" => ParameterType::Boolean,
                    "array" => ParameterType::Array,
                    "object" => ParameterType::Object,
                    _ => ParameterType::String,
                });

        let description = prop
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();

        let enum_values = prop.get("enum").and_then(Value::as_array).map(|arr| {
            arr.iter()
                .filter_map(Value::as_str)
                .map(String::from)
                .collect()
        });

        let default = prop.get("default").cloned();

        ToolParameter {
            name,
            description,
            param_type,
            required,
            default,
            enum_values,
        }
    }

    #[async_trait]
    impl<T: Transport + 'static> Tool for McpToolWrapper<T> {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: self.full_name(),
                description: self.tool.description.clone().unwrap_or_default(),
                parameters: schema_to_parameters(&self.tool.input_schema),
                output_schema: None,
            }
        }

        async fn execute(&self, params: HashMap<String, Value>) -> Result<ToolResult, ToolError> {
            debug!(tool = %self.full_name(), "Executing MCP tool");

            // Convert HashMap to Value for MCP
            let arguments = if params.is_empty() {
                None
            } else {
                Some(Value::Object(params.into_iter().collect()))
            };

            // Call the MCP tool
            let result = self
                .client
                .call_tool(&self.tool.name, arguments)
                .await
                .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

            // Convert MCP content to string
            let content = result
                .content
                .iter()
                .filter_map(ToolContent::to_text)
                .collect::<Vec<_>>()
                .join("\n");

            // A server's `structuredContent` is the machine-readable twin of the
            // text blocks; surface it as the tool result's `data` so downstream
            // consumers (agent loop, scripted tools) can use it without
            // re-parsing prose. Attached on the success path only — an errored
            // call's payload is not a result.
            let tool_result = if result.is_error {
                ToolResult::error(content)
            } else {
                match result.structured_content {
                    Some(data) => ToolResult::success(content).with_data(data),
                    None => ToolResult::success(content),
                }
            };
            Ok(tool_result)
        }

        /// The registry's deadline sits just past the transport's own, so the
        /// transport's timeout — which names the server — is the one a slow
        /// call reports, not the registry's generic one. The two used to
        /// disagree outright: 60 s advertised here, 30 s enforced below.
        fn timeout_secs(&self) -> Option<u64> {
            Some(crate::MCP_REQUEST_TIMEOUT.as_secs() + crate::MCP_DEADLINE_MARGIN_SECS)
        }
    }

    /// Manager for multiple MCP server connections with nanna-tools integration
    pub struct McpToolsManager<T: Transport + 'static> {
        /// Connected MCP clients by server name
        clients: RwLock<HashMap<String, Arc<McpClient<T>>>>,
        /// All available tool wrappers
        tool_wrappers: RwLock<Vec<Arc<McpToolWrapper<T>>>>,
    }

    impl<T: Transport + 'static> McpToolsManager<T> {
        /// Create a new MCP tools manager
        #[must_use]
        pub fn new() -> Self {
            Self {
                clients: RwLock::new(HashMap::new()),
                tool_wrappers: RwLock::new(Vec::new()),
            }
        }

        /// Register an MCP client and its tools
        ///
        /// # Errors
        ///
        /// Returns error if client is not initialized
        pub async fn register(
            &self,
            name: impl Into<String>,
            client: McpClient<T>,
        ) -> Result<Vec<Arc<McpToolWrapper<T>>>, McpError> {
            let name = name.into();
            let client = Arc::new(client);

            // Get tools from the server
            let tools = client.list_tools().await?;
            debug!(server = %name, count = tools.len(), "Registered MCP server tools");

            let mut wrappers = Vec::new();
            for tool in tools {
                let wrapper = Arc::new(McpToolWrapper::new(client.clone(), tool, &name));
                wrappers.push(wrapper);
            }

            // Store
            {
                let mut clients = self.clients.write().await;
                clients.insert(name, client);
            }
            {
                let mut tool_wrappers = self.tool_wrappers.write().await;
                tool_wrappers.extend(wrappers.iter().cloned());
            }

            Ok(wrappers)
        }

        /// Register all tools with a `ToolRegistry`
        ///
        /// # Errors
        ///
        /// Returns error if registration fails
        pub async fn register_with_registry(
            &self,
            registry: &nanna_tools::ToolRegistry,
        ) -> Result<usize, McpError> {
            let wrappers = self.tool_wrappers.read().await;
            let count = wrappers.len();

            for wrapper in wrappers.iter() {
                registry.register_boxed(wrapper.clone()).await;
            }
            drop(wrappers);

            Ok(count)
        }

        /// Get all tool wrappers
        pub async fn tools(&self) -> Vec<Arc<McpToolWrapper<T>>> {
            self.tool_wrappers.read().await.clone()
        }

        /// Refresh tools from all servers
        ///
        /// # Errors
        ///
        /// Returns error if refresh fails for any server
        pub async fn refresh(&self) -> Result<(), McpError> {
            let mut new_wrappers = Vec::new();

            let clients = self.clients.read().await;
            for (name, client) in clients.iter() {
                match client.refresh_tools().await {
                    Ok(tools) => {
                        for tool in tools {
                            let wrapper = Arc::new(McpToolWrapper::new(client.clone(), tool, name));
                            new_wrappers.push(wrapper);
                        }
                    }
                    Err(e) => {
                        warn!(server = %name, error = %e, "Failed to refresh tools");
                    }
                }
            }

            *self.tool_wrappers.write().await = new_wrappers;
            // `clients` is held until the refreshed wrappers are published, as it
            // always was: a concurrent `connect` waits for its client insert, so
            // the tools it appends land after this replacement, not under it.
            drop(clients);

            Ok(())
        }

        /// Re-read one server's tools and bring `registry` in line with them:
        /// tools it no longer offers are unregistered, the rest (new or
        /// changed) registered over their old entries. Returns
        /// `(registered, removed)`.
        ///
        /// # Errors
        ///
        /// Returns [`McpError::Protocol`] for an unknown server, or the
        /// refresh error; the registry is untouched on error.
        pub async fn resync_server(
            &self,
            server: &str,
            registry: &nanna_tools::ToolRegistry,
        ) -> Result<(usize, usize), McpError> {
            let client = self
                .clients
                .read()
                .await
                .get(server)
                .cloned()
                .ok_or_else(|| McpError::Protocol(format!("unknown MCP server '{server}'")))?;
            let fresh: Vec<Arc<McpToolWrapper<T>>> = client
                .refresh_tools()
                .await?
                .into_iter()
                .map(|tool| Arc::new(McpToolWrapper::new(client.clone(), tool, server)))
                .collect();
            let fresh_names: HashSet<String> = fresh.iter().map(|w| w.full_name()).collect();
            let mut gone: Vec<String> = Vec::new();
            {
                let mut wrappers = self.tool_wrappers.write().await;
                wrappers.retain(|wrapper| {
                    if wrapper.prefix != server {
                        return true;
                    }
                    let name = wrapper.full_name();
                    if !fresh_names.contains(&name) {
                        gone.push(name);
                    }
                    false
                });
                wrappers.extend(fresh.iter().cloned());
            }
            for name in &gone {
                registry.unregister(name).await;
            }
            for wrapper in &fresh {
                registry.register_boxed(wrapper.clone()).await;
            }
            debug_assert!(gone.iter().all(|name| !fresh_names.contains(name)));
            Ok((fresh.len(), gone.len()))
        }

        /// Keep `registry` in step with every server's tool list until `stop`
        /// resolves: wait for any server to announce a change, resync the
        /// servers whose tools changed, repeat. Resyncs are at least
        /// [`RESYNC_INTERVAL_MIN`] apart, so a server announcing changes in a
        /// burst (or in a loop) costs one `tools/list` per interval, not one
        /// per announcement.
        pub async fn watch_list_changes(
            &self,
            registry: &nanna_tools::ToolRegistry,
            stop: impl std::future::Future<Output = ()>,
        ) {
            let watched: Vec<(String, Arc<crate::transport::ListChangedFlags>)> = self
                .clients
                .read()
                .await
                .iter()
                .filter_map(|(name, client)| client.list_changed_flags().map(|f| (name.clone(), f)))
                .collect();
            tokio::pin!(stop);
            if watched.is_empty() {
                stop.await;
                return;
            }
            loop {
                let any_change = futures::future::select_all(
                    watched.iter().map(|(_, flags)| Box::pin(flags.changed())),
                );
                tokio::select! {
                    () = &mut stop => return,
                    _ = any_change => {}
                }
                for (server, flags) in &watched {
                    if !flags.take(crate::transport::McpList::Tools) {
                        continue;
                    }
                    match self.resync_server(server, registry).await {
                        Ok((registered, removed)) => tracing::info!(
                            server = %server, registered, removed,
                            "MCP tool list changed; registry resynced"
                        ),
                        Err(e) => warn!(server = %server, error = %e, "MCP tool resync failed"),
                    }
                }
                tokio::select! {
                    () = &mut stop => return,
                    () = tokio::time::sleep(RESYNC_INTERVAL_MIN) => {}
                }
            }
        }

        /// Close all connections, concurrently: each close may spend a grace
        /// period waiting for its server to exit, and one slow or failing
        /// server must not keep the others running.
        ///
        /// # Errors
        ///
        /// Returns the first close error, after every close has been attempted.
        pub async fn close_all(&self) -> Result<(), McpError> {
            let clients = self.clients.read().await;
            let outcomes =
                futures::future::join_all(clients.values().map(|client| client.close())).await;
            debug_assert_eq!(outcomes.len(), clients.len());
            drop(clients);
            outcomes.into_iter().collect()
        }
    }

    impl<T: Transport + 'static> Default for McpToolsManager<T> {
        fn default() -> Self {
            Self::new()
        }
    }

    #[cfg(test)]
    mod resync_tests {
        use super::{McpClient, McpToolsManager};
        use crate::protocol::{JsonRpcNotification, JsonRpcRequest, JsonRpcResponse};
        use crate::transport::{ListChangedFlags, McpList, Transport};
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        /// Serves `[a, b]` on the first `tools/list` and `[b, c]` after.
        struct ShiftingTools {
            calls: AtomicUsize,
            flags: Arc<ListChangedFlags>,
        }

        #[async_trait::async_trait]
        impl Transport for ShiftingTools {
            async fn request(&self, request: JsonRpcRequest) -> crate::Result<JsonRpcResponse> {
                let names = if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    ["a", "b"]
                } else {
                    ["b", "c"]
                };
                let tools: Vec<_> = names
                    .iter()
                    .map(|n| serde_json::json!({ "name": n, "inputSchema": { "type": "object" } }))
                    .collect();
                Ok(JsonRpcResponse {
                    jsonrpc: "2.0".into(),
                    id: request.id,
                    result: Some(serde_json::json!({ "tools": tools })),
                    error: None,
                })
            }
            async fn notify(&self, _n: JsonRpcNotification) -> crate::Result<()> {
                Ok(())
            }
            async fn close(&self) -> crate::Result<()> {
                Ok(())
            }
            fn list_changed_flags(&self) -> Option<Arc<ListChangedFlags>> {
                Some(self.flags.clone())
            }
        }

        async fn names(registry: &nanna_tools::ToolRegistry) -> Vec<String> {
            let mut names = Vec::new();
            for name in ["mcp__srv__a", "mcp__srv__b", "mcp__srv__c"] {
                if registry.get(name).await.is_some() {
                    names.push(name.trim_start_matches("mcp__srv__").to_string());
                }
            }
            names
        }

        #[tokio::test]
        async fn a_list_change_adds_and_removes_registry_tools() {
            let flags = Arc::new(ListChangedFlags::default());
            let client = McpClient::new(ShiftingTools {
                calls: AtomicUsize::new(0),
                flags: flags.clone(),
            });
            client.mark_initialized_for_test().await;
            client.refresh_tools().await.unwrap();
            let manager = McpToolsManager::new();
            manager.register("srv", client).await.unwrap();
            let registry = nanna_tools::ToolRegistry::new();
            manager.register_with_registry(&registry).await.unwrap();
            assert_eq!(names(&registry).await, ["a", "b"]);

            // The server announces a change; the watch loop resyncs and stops.
            flags.mark(McpList::Tools);
            let stop = tokio::time::sleep(std::time::Duration::from_millis(300));
            manager.watch_list_changes(&registry, stop).await;
            assert_eq!(
                names(&registry).await,
                ["b", "c"],
                "a removed, c added, b kept"
            );
            assert_eq!(
                manager.tools().await.len(),
                2,
                "the manager's own list matches"
            );
        }
    }

    #[cfg(test)]
    mod schema_param_tests {
        use super::{ParameterType, schema_to_parameters};
        use serde_json::json;

        fn names_required(schema: &serde_json::Value) -> Vec<(String, bool)> {
            schema_to_parameters(schema)
                .into_iter()
                .map(|p| (p.name, p.required))
                .collect()
        }

        #[test]
        fn reads_flat_object_properties_and_required() {
            let params = schema_to_parameters(&json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "the path" },
                    "count": { "type": "integer" }
                },
                "required": ["path"]
            }));
            let path = params.iter().find(|p| p.name == "path").unwrap();
            assert!(matches!(path.param_type, ParameterType::String));
            assert_eq!(path.description, "the path");
            assert!(path.required, "path is in `required`");
            let count = params.iter().find(|p| p.name == "count").unwrap();
            assert!(matches!(count.param_type, ParameterType::Integer));
            assert!(!count.required, "count is not in `required`");
        }

        #[test]
        fn folds_allof_branch_properties_with_hard_required() {
            // A composed schema: base props via allOf, each branch with its own required.
            let got = names_required(&json!({
                "type": "object",
                "allOf": [
                    { "properties": { "a": { "type": "string" } }, "required": ["a"] },
                    { "properties": { "b": { "type": "integer" } } }
                ]
            }));
            // Both branch properties surface; only the allOf-required `a` is required.
            assert!(got.contains(&("a".to_string(), true)), "got {got:?}");
            assert!(got.contains(&("b".to_string(), false)), "got {got:?}");
        }

        #[test]
        fn folds_oneof_and_anyof_branches_as_optional() {
            let got = names_required(&json!({
                "type": "object",
                "properties": { "mode": { "type": "string" } },
                "required": ["mode"],
                "oneOf": [
                    { "properties": { "x": { "type": "number" } }, "required": ["x"] },
                    { "properties": { "y": { "type": "number" } }, "required": ["y"] }
                ]
            }));
            // Root `mode` stays required; oneOf branch props appear but are NOT required
            // (only one branch applies to a given call).
            assert!(got.contains(&("mode".to_string(), true)), "got {got:?}");
            assert!(got.contains(&("x".to_string(), false)), "got {got:?}");
            assert!(got.contains(&("y".to_string(), false)), "got {got:?}");
        }

        #[test]
        fn first_definition_of_a_name_wins() {
            // Root defines `p` as string; an allOf branch redefines it as integer.
            let params = schema_to_parameters(&json!({
                "properties": { "p": { "type": "string" } },
                "allOf": [ { "properties": { "p": { "type": "integer" } } } ]
            }));
            let p: Vec<_> = params.iter().filter(|q| q.name == "p").collect();
            assert_eq!(p.len(), 1, "duplicate names collapse to one");
            assert!(
                matches!(p[0].param_type, ParameterType::String),
                "root wins over branch"
            );
        }

        #[test]
        fn empty_or_typeless_schema_yields_no_or_string_params() {
            assert!(schema_to_parameters(&json!({})).is_empty());
            // A property with no `type` falls back to String rather than being dropped.
            let params = schema_to_parameters(&json!({
                "properties": { "loose": { "description": "anything" } }
            }));
            assert_eq!(params.len(), 1);
            assert!(matches!(params[0].param_type, ParameterType::String));
        }
    }
}

#[cfg(feature = "tools-integration")]
pub use tools_impl::{McpToolWrapper, McpToolsManager, RESYNC_INTERVAL_MIN};

// ============================================================================
// Standalone adapter (no nanna-tools dependency)
// ============================================================================

/// Wrapper that adapts an MCP tool for standalone use
pub struct McpToolAdapter<T: Transport + 'static> {
    /// The MCP client
    client: Arc<McpClient<T>>,
    /// The tool definition from MCP
    tool: McpTool,
}

impl<T: Transport + 'static> McpToolAdapter<T> {
    /// Create a new adapter for an MCP tool
    pub const fn new(client: Arc<McpClient<T>>, tool: McpTool) -> Self {
        Self { client, tool }
    }

    /// Get the tool name
    #[must_use]
    pub fn name(&self) -> &str {
        &self.tool.name
    }

    /// Get the tool description
    #[must_use]
    pub fn description(&self) -> Option<&str> {
        self.tool.description.as_deref()
    }

    /// Get the input schema
    #[must_use]
    pub const fn input_schema(&self) -> &serde_json::Value {
        &self.tool.input_schema
    }

    /// Execute the tool
    ///
    /// # Errors
    ///
    /// Returns error if tool execution fails
    pub async fn execute(
        &self,
        arguments: Option<serde_json::Value>,
    ) -> Result<McpToolResult, McpError> {
        debug!(tool = %self.tool.name, "Executing MCP tool");

        let result = self.client.call_tool(&self.tool.name, arguments).await?;

        // Convert MCP content to string result
        let content = result
            .content
            .iter()
            .filter_map(ToolContent::to_text)
            .collect::<Vec<_>>()
            .join("\n");

        Ok(McpToolResult {
            content,
            is_error: result.is_error,
            raw: result.content,
            structured: result.structured_content,
        })
    }
}

/// Convert an MCP tool to Anthropic tool format for LLM function calling
#[must_use]
pub fn to_anthropic_format(tool: &McpTool) -> serde_json::Value {
    serde_json::json!({
        "name": tool.name,
        "description": tool.description.as_deref().unwrap_or(""),
        "input_schema": tool.input_schema
    })
}

/// Convert an MCP tool to `OpenAI` tool format
#[must_use]
pub fn to_openai_format(tool: &McpTool) -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description.as_deref().unwrap_or(""),
            "parameters": tool.input_schema
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_provider_valid(name: &str) -> bool {
        (1..=WIRE_TOOL_NAME_CHARS_MAX).contains(&name.len())
            && name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    }

    #[test]
    fn wire_names_are_valid_for_every_provider() {
        assert_eq!(
            wire_tool_name("files", "read_file"),
            "mcp__files__read_file"
        );
        assert_eq!(wire_tool_name("", "read_file"), "read_file");
        assert_eq!(
            wire_tool_name("my.server", "get:thing/x"),
            "mcp__my_server__get_thing_x"
        );
        // One `_` per non-ASCII character, not per byte.
        assert_eq!(wire_tool_name("ünï", "t"), "mcp___n___t");
        for (server, tool) in [("a b", "c d"), ("x", &"y".repeat(200)), ("日本", "ツール")] {
            let name = wire_tool_name(server, tool);
            assert!(is_provider_valid(&name), "{name}");
        }
    }

    #[test]
    fn long_names_stay_distinct_and_stable() {
        let stem = "t".repeat(80);
        let a = wire_tool_name("srv", &format!("{stem}_alpha"));
        let b = wire_tool_name("srv", &format!("{stem}_beta"));
        assert_eq!(a.len(), WIRE_TOOL_NAME_CHARS_MAX);
        assert_ne!(
            a, b,
            "a shared prefix must not collapse two tools into one name"
        );
        assert_eq!(a, wire_tool_name("srv", &format!("{stem}_alpha")));
        let exactly = "t".repeat(WIRE_TOOL_NAME_CHARS_MAX - "mcp__s__".len());
        assert_eq!(
            wire_tool_name("s", &exactly).len(),
            WIRE_TOOL_NAME_CHARS_MAX
        );
        assert!(
            wire_tool_name("s", &exactly).ends_with('t'),
            "at the limit nothing is hashed"
        );
    }

    #[test]
    fn test_tool_format_conversion() {
        let tool = McpTool {
            name: "read_file".to_string(),
            description: Some("Read a file from disk".to_string()),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file"
                    }
                },
                "required": ["path"]
            }),
        };

        let anthropic = to_anthropic_format(&tool);
        assert_eq!(anthropic["name"], "read_file");
        assert!(anthropic["input_schema"]["properties"]["path"].is_object());

        let openai = to_openai_format(&tool);
        assert_eq!(openai["type"], "function");
        assert_eq!(openai["function"]["name"], "read_file");
    }
}
