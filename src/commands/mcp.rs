//! `nanna mcp` — expose Nanna's tools to an external MCP client.
//!
//! Nanna already ships an MCP *server* (`nanna-mcp::McpServer`); this module
//! starts it over stdio JSON-RPC — the transport every MCP client (Claude
//! Code, Claude Desktop, editors) speaks.
//!
//! **With a daemon running, the served tools are the daemon's.** `tools/list`
//! is its live registry (so memory, sub-agent and the daemon's own MCP-server
//! tools are all there, under its `[tools]` policy) and every call is a
//! `tool.execute` over IPC — the daemon stays the only process that owns
//! `nanna.db`. With no daemon (or `--standalone`), the filesystem JS/TS skills
//! are loaded locally instead: filesystem, shell, web and code tools work,
//! anything needing memory or the agent does not.
//!
//! **stdout is the protocol.** The caller must send logs to stderr; a stray
//! `println!` or a stdout-writing tracing layer corrupts the JSON-RPC stream and
//! the client disconnects with a parse error. `main` handles this by installing
//! a stderr writer for this command.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use nanna_client::{Client, ClientConfig};
use nanna_config::Config;
use nanna_mcp::{CallToolResult, McpServer, McpServerConfig, Tool, ToolContent, tools_bridge};
use nanna_tools::{ToolDefinition, ToolPolicy, ToolRegistry};
use serde_json::Value;
use tracing::{info, warn};

/// How long `serve` waits for a daemon before deciding there is none.
const DAEMON_CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);

/// Most daemon tools served. The registry is a few dozen; the bound only
/// stops a pathological listing from costing unbounded `tool.get` calls.
const DAEMON_TOOLS_MAX: usize = 512;

/// Where `serve` looks for the daemon.
#[must_use]
pub fn default_daemon_url() -> String {
    format!(
        "ws://{}:{}",
        nanna_config::bind::LOOPBACK_HOST,
        nanna_daemon::DEFAULT_IPC_PORT
    )
}

/// Serve the MCP surface: the running daemon's tools when one answers at
/// `daemon_url`, the local skills otherwise (or with `standalone`).
///
/// # Errors
///
/// Returns an error if `daemon_url` was given explicitly and no daemon
/// answers there, or as [`serve_standalone`] / [`serve_via_daemon`].
pub async fn serve(
    config: &Config,
    tools_dir_override: Option<PathBuf>,
    standalone: bool,
    daemon_url: Option<String>,
) -> anyhow::Result<()> {
    if !standalone {
        let url = daemon_url.clone().unwrap_or_else(default_daemon_url);
        match connect_daemon(&url).await {
            Ok(client) => return serve_via_daemon(client).await,
            Err(e) if daemon_url.is_some() => return Err(e),
            Err(e) => warn!(
                "no daemon at {url} ({e:#}); serving the standalone surface — memory, \
                 sub-agent and MCP-server tools need the daemon"
            ),
        }
    }
    serve_standalone(config, tools_dir_override).await
}

async fn connect_daemon(url: &str) -> anyhow::Result<Client> {
    let client_config = ClientConfig::new(url);
    match tokio::time::timeout(DAEMON_CONNECT_TIMEOUT, Client::connect(client_config)).await {
        Ok(Ok(client)) => Ok(client),
        Ok(Err(e)) => Err(anyhow::anyhow!(e)).context("the daemon refused the connection"),
        Err(_) => anyhow::bail!("no answer within {DAEMON_CONNECT_TIMEOUT:?}"),
    }
}

/// Serve the daemon's live tool surface over stdio until the client leaves.
///
/// # Errors
///
/// Returns an error if the daemon's tool listing cannot be read, or the stdio
/// loop fails.
pub async fn serve_via_daemon(client: Client) -> anyhow::Result<()> {
    let client = Arc::new(client);
    let listing = client
        .tools()
        .list()
        .await
        .map_err(|e| anyhow::anyhow!(e))
        .context("the daemon did not list its tools")?;
    let names = enabled_tool_names(&listing);
    let server = Arc::new(McpServer::new(McpServerConfig {
        name: "nanna".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        instructions: Some(
            "Nanna's tools, executed by the running Nanna daemon: memory, files, shell, \
             web, code, and the MCP servers it is connected to."
                .to_string(),
        ),
    }));
    let mut exposed = 0_usize;
    for name in names.iter().take(DAEMON_TOOLS_MAX) {
        let Some(tool) = daemon_tool(&client, name).await else {
            continue;
        };
        let executor = Arc::clone(&client);
        let tool_name = tool.name.clone();
        server
            .register_tool(tool, move |input: Value| {
                let client = Arc::clone(&executor);
                let name = tool_name.clone();
                async move {
                    Ok(match client.tools().execute(&name, input).await {
                        Ok(reply) => daemon_reply_to_result(&reply),
                        Err(e) => error_result(format!("the daemon did not run {name}: {e}")),
                    })
                }
            })
            .await;
        exposed += 1;
    }
    debug_assert!(exposed <= names.len());
    info!("Exposing {exposed} daemon tools over MCP stdio");
    server.run_stdio().await.context("MCP stdio loop failed")?;
    info!("MCP client disconnected");
    Ok(())
}

/// The names a `tool.list` reply marks enabled, in its order. Pure.
fn enabled_tool_names(listing: &Value) -> Vec<String> {
    listing
        .get("tools")
        .and_then(Value::as_array)
        .map_or_default(|tools| {
            tools
                .iter()
                .filter(|t| t.get("enabled").and_then(Value::as_bool) != Some(false))
                .filter_map(|t| t.get("name").and_then(Value::as_str).map(str::to_string))
                .collect()
        })
}

/// One daemon tool as an MCP tool definition, from `tool.get`. `None` (and a
/// warning) when the daemon cannot describe it.
async fn daemon_tool(client: &Client, name: &str) -> Option<Tool> {
    let reply = match client.tools().get(name).await {
        Ok(reply) => reply,
        Err(e) => {
            warn!("skipping {name}: the daemon could not describe it ({e})");
            return None;
        }
    };
    let definition = reply
        .get("tool")
        .cloned()
        .and_then(|tool| serde_json::from_value::<ToolDefinition>(tool).ok());
    let Some(definition) = definition else {
        warn!("skipping {name}: its definition did not parse");
        return None;
    };
    Some(Tool {
        name: definition.name.clone(),
        description: Some(definition.description.clone()),
        input_schema: definition.to_anthropic_format()["input_schema"].clone(),
    })
}

/// Map a `tool.execute` reply (`{name, output, success, error}`) to an MCP
/// tool result. Pure.
fn daemon_reply_to_result(reply: &Value) -> CallToolResult {
    let success = reply
        .get("success")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if !success {
        let error = reply
            .get("error")
            .and_then(Value::as_str)
            .filter(|e| !e.is_empty())
            .or_else(|| reply.get("output").and_then(Value::as_str))
            .unwrap_or("the tool failed without saying why");
        return error_result(error.to_string());
    }
    let output = reply.get("output").map_or_else(String::new, |o| {
        o.as_str().map_or_else(|| o.to_string(), str::to_string)
    });
    CallToolResult {
        content: vec![ToolContent::Text { text: output }],
        is_error: false,
        structured_content: None,
    }
}

fn error_result(text: String) -> CallToolResult {
    CallToolResult {
        content: vec![ToolContent::Text { text }],
        is_error: true,
        structured_content: None,
    }
}

/// Serve the locally loaded skills over stdio JSON-RPC until the client
/// closes the stream.
///
/// `tools_dir_override` wins over the config's `[tools] tools_dir`, which in
/// turn is resolved by the shared `resolve_tools_dir` (so `NANNA_TOOLS_DIR` and
/// the dev-tree fallback keep working).
///
/// # Errors
///
/// Returns an error if no tools directory can be resolved, or if the stdio loop
/// fails to read or write.
pub async fn serve_standalone(
    config: &Config,
    tools_dir_override: Option<PathBuf>,
) -> anyhow::Result<()> {
    let tools_dir = tools_dir_override
        .or_else(|| {
            nanna_tools::skills::defaults::resolve_tools_dir(config.tools.tools_dir.as_deref())
        })
        .context(
            "no tools directory found — set [tools] tools_dir in config.toml or NANNA_TOOLS_DIR",
        )?;
    anyhow::ensure!(
        tools_dir.is_dir(),
        "tools directory {} does not exist",
        tools_dir.display()
    );

    let registry = Arc::new(ToolRegistry::new());
    let loaded = registry.load_skills(&tools_dir).await;
    info!("Loaded {loaded} tools from {}", tools_dir.display());
    if loaded == 0 {
        warn!("No tools loaded — the MCP client will see an empty tool list");
    }

    // Apply the user's [tools] enabled/disabled policy BEFORE advertising
    // anything. `definitions()` filters denied tools out of the listing and
    // `execute()` re-checks after alias/fuzzy resolution, so a disabled tool is
    // neither offered to nor invocable by the connecting client.
    let policy = ToolPolicy::from_config_lists(Some(&config.tools.enabled), &config.tools.disabled);
    if !policy.is_unrestricted() {
        info!("Applying [tools] policy to the MCP surface");
        registry.set_policy(policy).await;
    }

    let server = Arc::new(McpServer::new(McpServerConfig {
        name: "nanna".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        instructions: Some(
            "Nanna's local tool surface: filesystem, shell, web, and code tools \
             executed on this machine."
                .to_string(),
        ),
    }));

    let exposed = tools_bridge::register_tools_from_registry(&server, Arc::clone(&registry))
        .await
        .context("failed to register tools with the MCP server")?;
    info!("Exposing {exposed} tools over MCP stdio");
    debug_assert!(
        loaded > 0 || exposed == 0,
        "an empty registry must not advertise any tool"
    );

    server.run_stdio().await.context("MCP stdio loop failed")?;
    info!("MCP client disconnected");
    Ok(())
}

/// Why `server`/`var` cannot name a stored MCP secret, if they cannot. Pure.
fn secret_target_problem(server: &str, var: &str) -> Option<String> {
    if server.trim().is_empty() {
        return Some("the server name is empty".to_string());
    }
    if !nanna_config::mcp::is_env_var_name(var) {
        return Some(format!(
            "'{var}' is not an environment variable name (letters, digits and _, not starting with a digit)"
        ));
    }
    None
}

/// A note when the stored secret is not wired to a configured server yet. Pure.
fn secret_wiring_note(config: &Config, server: &str, var: &str) -> Option<String> {
    let server = server.trim();
    match config
        .mcp
        .servers
        .iter()
        .find(|entry| entry.name.trim() == server)
    {
        None => Some(format!(
            "note: no [[mcp.servers]] entry is named '{server}' yet; add one with secret_env = [\"{var}\"]"
        )),
        Some(entry)
            if !entry.secret_env.iter().any(|name| name == var)
                && entry.bearer_secret.as_deref() != Some(var) =>
        {
            Some(format!(
                "note: '{server}' names {var} in neither secret_env nor bearer_secret, so it \
                 will not receive it"
            ))
        }
        Some(_) => None,
    }
}

/// `nanna mcp secret set <server> <VAR>`: read the value without echoing it
/// (or from stdin when piped) and store it in the secure store.
///
/// # Errors
///
/// Returns an error for a bad name, an empty value, or a store failure.
pub fn secret_set(config: &Config, server: &str, var: &str) -> anyhow::Result<()> {
    use std::io::{BufRead, IsTerminal};
    if let Some(problem) = secret_target_problem(server, var) {
        anyhow::bail!("{problem}");
    }
    let value = if std::io::stdin().is_terminal() {
        dialoguer::Password::new()
            .with_prompt(format!("{var} for MCP server '{}'", server.trim()))
            .interact()?
    } else {
        let mut line = String::new();
        std::io::stdin().lock().read_line(&mut line)?;
        line.trim_end_matches(['\r', '\n']).to_string()
    };
    anyhow::ensure!(
        !value.trim().is_empty(),
        "the value is empty; nothing was stored"
    );
    nanna_config::credentials::SecureStore::new()
        .set(&nanna_config::mcp_secret_key(server, var), &value)
        .context("could not write the secure store")?;
    eprintln!("Stored {var} for MCP server '{}'.", server.trim());
    if let Some(note) = secret_wiring_note(config, server, var) {
        eprintln!("{note}");
    }
    eprintln!("The daemon reads it when it starts the server (restart the daemon to apply).");
    Ok(())
}

/// `nanna mcp secret delete <server> <VAR>`.
///
/// # Errors
///
/// Returns an error for a bad name or a store failure.
pub fn secret_delete(server: &str, var: &str) -> anyhow::Result<()> {
    if let Some(problem) = secret_target_problem(server, var) {
        anyhow::bail!("{problem}");
    }
    nanna_config::credentials::SecureStore::new()
        .delete(&nanna_config::mcp_secret_key(server, var))
        .context("could not remove it from the secure store")?;
    eprintln!("Removed {var} for MCP server '{}'.", server.trim());
    Ok(())
}

#[cfg(test)]
mod secret_tests {
    use super::*;

    #[test]
    fn secret_targets_are_checked_before_the_store_is_touched() {
        assert_eq!(secret_target_problem("github", "GITHUB_TOKEN"), None);
        assert!(secret_target_problem("  ", "GITHUB_TOKEN").is_some());
        assert!(secret_target_problem("github", "GITHUB-TOKEN").is_some());
    }

    #[test]
    fn a_secret_nobody_will_receive_is_pointed_out() {
        let mut config = Config::default();
        assert!(
            secret_wiring_note(&config, "github", "GITHUB_TOKEN")
                .is_some_and(|note| note.contains("no [[mcp.servers]] entry"))
        );
        config.mcp.servers.push(nanna_config::McpServerEntry {
            url: String::new(),
            bearer_secret: None,
            name: "github".into(),
            command: "npx".into(),
            args: Vec::new(),
            enabled: true,
            secret_env: Vec::new(),
        });
        assert!(
            secret_wiring_note(&config, " github ", "GITHUB_TOKEN")
                .is_some_and(|note| note.contains("names GITHUB_TOKEN in neither"))
        );
        config.mcp.servers[0].secret_env.push("GITHUB_TOKEN".into());
        assert_eq!(secret_wiring_note(&config, "github", "GITHUB_TOKEN"), None);
    }
}

#[cfg(test)]
mod daemon_proxy_tests {
    use super::*;

    #[test]
    fn only_enabled_daemon_tools_are_served() {
        let listing = serde_json::json!({ "tools": [
            { "name": "recall", "enabled": true },
            { "name": "exec", "enabled": false },
            { "name": "legacy_entry" },
            { "enabled": true }
        ]});
        assert_eq!(enabled_tool_names(&listing), ["recall", "legacy_entry"]);
        assert_eq!(
            enabled_tool_names(&serde_json::json!({})),
            Vec::<String>::new()
        );
    }

    #[test]
    fn a_daemon_reply_becomes_an_mcp_result() {
        let ok = daemon_reply_to_result(&serde_json::json!({
            "name": "recall", "output": "found 3 memories", "success": true, "error": null
        }));
        assert!(!ok.is_error);
        assert!(matches!(&ok.content[0], ToolContent::Text { text } if text == "found 3 memories"));

        let failed = daemon_reply_to_result(&serde_json::json!({
            "name": "nope", "output": "", "success": false, "error": "Tool not found: nope."
        }));
        assert!(failed.is_error);
        assert!(
            matches!(&failed.content[0], ToolContent::Text { text } if text == "Tool not found: nope.")
        );

        let silent = daemon_reply_to_result(&serde_json::json!({ "success": false }));
        assert!(
            silent.is_error,
            "a malformed reply is a failure, not an empty success"
        );
    }
}
