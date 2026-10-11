//! MCP server integration for the agent
//!
//! Provides utilities for spawning MCP servers and registering their tools
//! with the agent's tool registry.

#[cfg(feature = "mcp")]
use nanna_mcp::{
    AnyTransport, LegacySseTransport, McpClient, McpToolsManager, StdioTransport,
    StreamableHttpTransport,
};
use nanna_tools::ToolRegistry;
use tracing::{debug, error, info};

/// MCP server configuration
#[derive(Clone)]
pub struct McpServerConfig {
    /// Unique name for this server
    pub name: String,
    /// Command to run (e.g., "npx", "python", "node"); empty for a `url` server
    pub command: String,
    /// Arguments to pass to the command
    pub args: Vec<String>,
    /// Environment variables (values may be secrets — never logged)
    pub env: Vec<(String, String)>,
    /// Streamable HTTP endpoint; when set, `command`/`args`/`env` are unused
    pub url: Option<String>,
    /// Sent as `Authorization: Bearer` to a `url` server (never logged)
    pub bearer_token: Option<String>,
    /// Whether to auto-start on agent init
    pub auto_start: bool,
}

/// Redacts every value that may be a secret: `env` values and the token.
impl std::fmt::Debug for McpServerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let env_names: Vec<&str> = self.env.iter().map(|(name, _)| name.as_str()).collect();
        f.debug_struct("McpServerConfig")
            .field("name", &self.name)
            .field("command", &self.command)
            .field("args", &self.args)
            .field("env", &env_names)
            .field("url", &self.url)
            .field(
                "bearer_token",
                &self.bearer_token.as_ref().map(|_| "<redacted>"),
            )
            .field("auto_start", &self.auto_start)
            .finish()
    }
}

impl McpServerConfig {
    /// Create a new MCP server config
    pub fn new(name: impl Into<String>, command: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            command: command.into(),
            args: Vec::new(),
            env: Vec::new(),
            url: None,
            bearer_token: None,
            auto_start: true,
        }
    }

    /// A Streamable HTTP server at `url`.
    pub fn http(name: impl Into<String>, url: impl Into<String>) -> Self {
        let mut config = Self::new(name, "");
        config.url = Some(url.into());
        config
    }

    /// The bearer token sent to a `url` server.
    #[must_use]
    pub fn bearer_token(mut self, token: Option<String>) -> Self {
        self.bearer_token = token;
        self
    }

    /// Add arguments
    #[must_use]
    pub fn args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.args = args.into_iter().map(Into::into).collect();
        self
    }

    /// Add environment variables
    #[must_use]
    pub fn env<I, K, V>(mut self, env: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        self.env = env.into_iter().map(|(k, v)| (k.into(), v.into())).collect();
        self
    }

    /// Set auto-start
    #[must_use]
    pub const fn auto_start(mut self, auto_start: bool) -> Self {
        self.auto_start = auto_start;
        self
    }

    // Common MCP server configurations

    /// Filesystem server
    pub fn filesystem(name: impl Into<String>, paths: &[&str]) -> Self {
        Self::new(name, "npx")
            .args(
                ["-y", "@modelcontextprotocol/server-filesystem"]
                    .into_iter()
                    .chain(paths.iter().copied()),
            )
    }

    /// GitHub server
    pub fn github(name: impl Into<String>, token: impl Into<String>) -> Self {
        Self::new(name, "npx")
            .args(["-y", "@modelcontextprotocol/server-github"])
            .env([("GITHUB_PERSONAL_ACCESS_TOKEN", token.into())])
    }

    /// Brave Search server
    pub fn brave_search(name: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self::new(name, "npx")
            .args(["-y", "@modelcontextprotocol/server-brave-search"])
            .env([("BRAVE_API_KEY", api_key.into())])
    }

    /// Fetch server (web fetching)
    pub fn fetch(name: impl Into<String>) -> Self {
        Self::new(name, "npx").args(["-y", "@modelcontextprotocol/server-fetch"])
    }

    /// Memory server
    pub fn memory(name: impl Into<String>) -> Self {
        Self::new(name, "npx").args(["-y", "@modelcontextprotocol/server-memory"])
    }

    /// Puppeteer server
    pub fn puppeteer(name: impl Into<String>) -> Self {
        Self::new(name, "npx").args(["-y", "@modelcontextprotocol/server-puppeteer"])
    }

    /// Sequential thinking server
    pub fn sequential_thinking(name: impl Into<String>) -> Self {
        Self::new(name, "npx").args(["-y", "@modelcontextprotocol/server-sequential-thinking"])
    }

    /// Custom npx server
    pub fn npx(name: impl Into<String>, package: impl Into<String>) -> Self {
        Self::new(name, "npx").args(["-y", &package.into()])
    }
}

/// MCP integration manager for the agent
/// What starting one MCP server produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpStarted {
    /// Tools it registered.
    pub tools: usize,
    /// The protocol revision and transport it was reached over, e.g.
    /// `2026-07-28 over Streamable HTTP` — so a fallback (to the 2024
    /// handshake, or to HTTP+SSE) is visible rather than silent.
    pub link: String,
}

#[cfg(feature = "mcp")]
pub struct McpIntegration {
    /// Tool manager for MCP servers
    manager: McpToolsManager<AnyTransport>,
    /// Server configurations
    configs: Vec<McpServerConfig>,
    /// Serves servers' elicitation requests; `None` declares none.
    elicitor: Option<std::sync::Arc<dyn nanna_mcp::Elicitor>>,
}

#[cfg(feature = "mcp")]
impl McpIntegration {
    /// Create a new MCP integration
    #[must_use]
    pub fn new() -> Self {
        Self {
            manager: McpToolsManager::new(),
            configs: Vec::new(),
            elicitor: None,
        }
    }

    /// Put servers' questions (MCP elicitation) to the user through
    /// `elicitor`. Set before [`Self::start_all`].
    pub fn set_elicitor(&mut self, elicitor: std::sync::Arc<dyn nanna_mcp::Elicitor>) {
        self.elicitor = Some(elicitor);
    }

    /// Add a server configuration
    pub fn add_server(&mut self, config: McpServerConfig) {
        self.configs.push(config);
    }

    /// Spawn all configured servers, concurrently, registering each one's
    /// tools into `registry` the moment its handshake finishes.
    ///
    /// Servers used to start one after another and register only after the
    /// whole loop, so one slow `npx -y` server (a package download, then a
    /// 60 s probe and a 60 s `initialize`) held back every other server's
    /// tools and status. Now each is usable as soon as it is up, and
    /// `on_outcome` hears about each one as it settles — `Ok` or the reason
    /// it did not start — so a caller can report per-server state live.
    /// One server failing does not stop the others. Concurrency is bounded
    /// by the configured server count (`MCP_SERVERS_MAX` in the daemon).
    ///
    /// Returns every auto-started server's outcome, in configuration order.
    pub async fn start_all(
        &self,
        registry: &ToolRegistry,
        mut on_outcome: impl AsyncFnMut(&str, &Result<McpStarted, String>),
    ) -> Vec<(String, Result<McpStarted, String>)> {
        use futures::stream::{FuturesUnordered, StreamExt as _};
        let mut pending: FuturesUnordered<_> = self
            .configs
            .iter()
            .enumerate()
            .filter(|(_, config)| {
                if !config.auto_start {
                    debug!(server = %config.name, "Skipping MCP server (auto_start=false)");
                }
                config.auto_start
            })
            .map(|(index, config)| async move {
                let outcome = self.start_server(config, registry).await.map_err(|e| {
                    error!(server = %config.name, error = %e, "Failed to start MCP server");
                    e.to_string()
                });
                (index, outcome)
            })
            .collect();
        let mut outcomes: Vec<(usize, String, Result<McpStarted, String>)> =
            Vec::with_capacity(pending.len());
        while let Some((index, outcome)) = pending.next().await {
            let name = self.configs[index].name.clone();
            on_outcome(&name, &outcome).await;
            outcomes.push((index, name, outcome));
        }
        outcomes.sort_by_key(|(index, _, _)| *index);
        debug_assert!(outcomes.len() <= self.configs.len());
        let started = outcomes.iter().filter(|(_, _, o)| o.is_ok()).count();
        info!(servers = outcomes.len(), started, "MCP integration started");
        outcomes
            .into_iter()
            .map(|(_, name, outcome)| (name, outcome))
            .collect()
    }

    /// Spawn a `command` server or connect to a `url` one, and run the
    /// dual-era handshake either way.
    async fn connect(
        config: &McpServerConfig,
        elicitor: Option<std::sync::Arc<dyn nanna_mcp::Elicitor>>,
    ) -> Result<(McpClient<AnyTransport>, &'static str), nanna_mcp::McpError> {
        let with_elicitor = |client: McpClient<AnyTransport>| match &elicitor {
            Some(elicitor) => client.with_elicitor(std::sync::Arc::clone(elicitor)),
            None => client,
        };
        let Some(url) = &config.url else {
            let args: Vec<&str> = config.args.iter().map(String::as_str).collect();
            let env: Vec<(&str, &str)> = config
                .env
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_str()))
                .collect();
            let transport = AnyTransport::Stdio(StdioTransport::spawn_with_env(
                &config.command,
                &args,
                &env,
            )?);
            let client = with_elicitor(McpClient::new(transport));
            client.initialize().await?;
            return Ok((client, "stdio"));
        };
        let token = config.bearer_token.clone();
        let transport =
            AnyTransport::Http(Box::new(StreamableHttpTransport::new(url, token.clone())?));
        let client = with_elicitor(McpClient::new(transport));
        match client.initialize().await {
            Ok(_) => Ok((client, "Streamable HTTP")),
            // No Streamable HTTP endpoint here, and no modern error body
            // either: the binding's cue to try the deprecated HTTP+SSE
            // transport, which is legacy-only (so no era probe).
            Err(nanna_mcp::McpError::HttpStatus {
                status: 400 | 404 | 405,
                ..
            }) => {
                info!(%url, "No Streamable HTTP endpoint; trying the 2024 HTTP+SSE transport");
                let legacy = LegacySseTransport::connect(url, token).await?;
                let client = with_elicitor(McpClient::new(AnyTransport::Sse(Box::new(legacy))));
                client.initialize_legacy().await?;
                Ok((client, "HTTP+SSE"))
            }
            Err(e) => Err(e),
        }
    }

    /// Start a single MCP server and register its tools into `registry`.
    async fn start_server(
        &self,
        config: &McpServerConfig,
        registry: &ToolRegistry,
    ) -> Result<McpStarted, McpStartError> {
        if let Some(url) = &config.url {
            info!(server = %config.name, %url, "Connecting to MCP server");
        } else {
            info!(server = %config.name, command = %config.command, "Starting MCP server");
        }

        let (client, transport) = Self::connect(config, self.elicitor.clone())
            .await
            .map_err(|e| McpStartError::Spawn(config.name.clone(), e.to_string()))?;
        let link = match client.era().await {
            nanna_mcp::ProtocolEra::Modern { version } => format!("{version} over {transport}"),
            nanna_mcp::ProtocolEra::Legacy => {
                format!("{} over {transport}", nanna_mcp::PROTOCOL_VERSION)
            }
        };

        // Register with manager
        let tools = self
            .manager
            .register(&config.name, client)
            .await
            .map_err(|e| McpStartError::Registration(e.to_string()))?;
        for tool in &tools {
            registry.register_boxed(tool.clone()).await;
        }

        info!(
            server = %config.name,
            tools = tools.len(),
            link = %link,
            "MCP server started"
        );

        Ok(McpStarted {
            tools: tools.len(),
            link,
        })
    }

    /// Keep `registry` in step with every server's tool list until `stop`
    /// resolves (see [`McpToolsManager::watch_list_changes`]).
    pub async fn watch(
        &self,
        registry: &ToolRegistry,
        stop: impl std::future::Future<Output = ()>,
    ) {
        self.manager.watch_list_changes(registry, stop).await;
    }

    /// Get the tool manager
    #[must_use]
    pub const fn manager(&self) -> &McpToolsManager<AnyTransport> {
        &self.manager
    }

    /// Refresh tools from all servers
    ///
    /// # Errors
    ///
    /// Returns error if refresh fails
    pub async fn refresh(&self) -> Result<(), McpStartError> {
        self.manager
            .refresh()
            .await
            .map_err(|e| McpStartError::Refresh(e.to_string()))
    }

    /// Shutdown all MCP servers
    ///
    /// # Errors
    ///
    /// Returns error if shutdown fails
    pub async fn shutdown(&self) -> Result<(), McpStartError> {
        self.manager
            .close_all()
            .await
            .map_err(|e| McpStartError::Shutdown(e.to_string()))
    }
}

#[cfg(feature = "mcp")]
impl Default for McpIntegration {
    fn default() -> Self {
        Self::new()
    }
}

/// Error starting MCP servers
#[derive(Debug, thiserror::Error)]
pub enum McpStartError {
    #[error("Failed to spawn server '{0}': {1}")]
    Spawn(String, String),
    #[error("Failed to register tools: {0}")]
    Registration(String),
    #[error("Failed to refresh tools: {0}")]
    Refresh(String),
    #[error("Failed to shutdown: {0}")]
    Shutdown(String),
}

/// Builder for MCP integration
#[cfg(feature = "mcp")]
pub struct McpIntegrationBuilder {
    configs: Vec<McpServerConfig>,
}

#[cfg(feature = "mcp")]
impl McpIntegrationBuilder {
    /// Create a new builder
    #[must_use]
    pub const fn new() -> Self {
        Self {
            configs: Vec::new(),
        }
    }

    /// Add a server
    #[must_use]
    pub fn server(mut self, config: McpServerConfig) -> Self {
        self.configs.push(config);
        self
    }

    /// Add filesystem access
    #[must_use]
    pub fn filesystem(self, paths: &[&str]) -> Self {
        self.server(McpServerConfig::filesystem("fs", paths))
    }

    /// Add GitHub access
    #[must_use]
    pub fn github(self, token: impl Into<String>) -> Self {
        self.server(McpServerConfig::github("github", token))
    }

    /// Add web search
    #[must_use]
    pub fn brave_search(self, api_key: impl Into<String>) -> Self {
        self.server(McpServerConfig::brave_search("search", api_key))
    }

    /// Add web fetching
    #[must_use]
    pub fn fetch(self) -> Self {
        self.server(McpServerConfig::fetch("fetch"))
    }

    /// Build the integration
    #[must_use]
    pub fn build(self) -> McpIntegration {
        let mut integration = McpIntegration::new();
        integration.configs = self.configs;
        integration
    }
}

#[cfg(feature = "mcp")]
impl Default for McpIntegrationBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn debug_never_prints_a_secret() {
        let config = super::McpServerConfig::http("notion", "https://x/mcp")
            .bearer_token(Some("sk-live-123".into()));
        let stdio = super::McpServerConfig::new("gh", "npx").env([("GITHUB_TOKEN", "ghp_456")]);
        let printed = format!("{config:?} {stdio:?}");
        assert!(!printed.contains("sk-live-123"), "{printed}");
        assert!(!printed.contains("ghp_456"), "{printed}");
        assert!(
            printed.contains("GITHUB_TOKEN"),
            "names stay visible: {printed}"
        );
        assert!(printed.contains("<redacted>"), "{printed}");
    }

    use super::*;

    #[test]
    fn test_server_config() {
        let config = McpServerConfig::filesystem("fs", &["/tmp", "/home"]);
        assert_eq!(config.name, "fs");
        assert_eq!(config.command, "npx");
        assert!(config.args.contains(&"-y".to_string()));
    }

    #[test]
    fn test_builder() {
        let _integration = McpIntegrationBuilder::new()
            .filesystem(&["/tmp"])
            .fetch()
            .build();
    }
}

#[cfg(all(test, unix, feature = "mcp"))]
mod start_tests {
    use super::{McpIntegration, McpServerConfig};
    use nanna_tools::ToolRegistry;
    use std::sync::Arc;
    use std::time::Duration;

    /// A legacy-era stdio MCP server with one tool, `echo`.
    const FAST_SERVER: &str = r#"
import json, sys
for line in sys.stdin:
    msg = json.loads(line)
    if "id" not in msg:
        continue
    method = msg.get("method")
    if method == "initialize":
        result = {"protocolVersion": msg["params"]["protocolVersion"],
                  "capabilities": {"tools": {}}, "serverInfo": {"name": "fast"}}
        reply = {"jsonrpc": "2.0", "id": msg["id"], "result": result}
    elif method == "tools/list":
        tools = [{"name": "echo", "inputSchema": {"type": "object"}}]
        reply = {"jsonrpc": "2.0", "id": msg["id"], "result": {"tools": tools}}
    else:
        reply = {"jsonrpc": "2.0", "id": msg["id"],
                 "error": {"code": -32601, "message": "no such method"}}
    sys.stdout.write(json.dumps(reply) + "\n")
    sys.stdout.flush()
"#;

    /// One server that never answers must not hold back another's tools:
    /// `start_all` used to start servers one after another and register only
    /// after the loop, so the fast server's tools waited out the slow one's
    /// probe and handshake timeouts (two minutes).
    #[tokio::test]
    async fn a_slow_server_does_not_hold_back_another_servers_tools() {
        if std::process::Command::new("python3").arg("--version").output().is_err() {
            return;
        }
        let mut integration = McpIntegration::new();
        // Listed first, so a sequential start would reach it first.
        integration.add_server(McpServerConfig::new("slow", "sh").args(["-c", "sleep 600"]));
        integration.add_server(McpServerConfig::new("fast", "python3").args(["-c", FAST_SERVER]));
        let integration = Arc::new(integration);
        let registry = Arc::new(ToolRegistry::new());
        let starting = {
            let (integration, registry) = (Arc::clone(&integration), Arc::clone(&registry));
            tokio::spawn(async move {
                integration.start_all(&registry, async |_: &str, _: &_| {}).await
            })
        };
        let deadline = tokio::time::Instant::now() + Duration::from_secs(20);
        while registry.get("mcp__fast__echo").await.is_none() {
            assert!(
                tokio::time::Instant::now() < deadline,
                "the fast server's tool waited on the slow server"
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert!(!starting.is_finished(), "the slow server is still starting");
        starting.abort();
    }
}
