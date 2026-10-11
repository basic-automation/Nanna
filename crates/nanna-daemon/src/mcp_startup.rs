//! Starting the configured MCP servers (`[mcp]`) and registering their tools.
//!
//! `nanna-mcp` (client, transports, schema guard) and `nanna-agent`'s
//! `McpIntegration` were complete and constructed nowhere: no config section
//! existed and the daemon never called them, so Nanna could not use a single
//! MCP server. This is the missing boot step.
//!
//! **Started in the background.** A server launched through `npx -y` may
//! download a package on its first run; waiting for it would hold the daemon's
//! readiness hostage to the network. Servers start concurrently and tools
//! register as each server finishes its handshake, and a server that fails is
//! logged by name and skipped — the rest still start.

use std::sync::Arc;

use nanna_agent::mcp::{McpIntegration, McpServerConfig};
use nanna_config::McpConfig;
use nanna_tools::ToolRegistry;
use tokio::sync::broadcast;
use tracing::{info, warn};

/// What one configured MCP server is doing, for `system.status`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct McpServerState {
    pub name: String,
    /// `starting` | `started` | `failed` | `not_started`
    pub state: &'static str,
    /// Tools registered from it (started servers only).
    pub tools: usize,
    /// Why it failed or was not started.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    /// How a started server was reached, e.g. `2026-07-28 over stdio`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub link: Option<String>,
}

/// Per-server MCP state, shared between the boot task that writes it and the
/// control plane that reports it. Bounded by the configured server count.
pub type McpStatus = Arc<tokio::sync::RwLock<Vec<McpServerState>>>;

/// Record one server's outcome in place. Pure over the slice.
fn record_outcome(
    servers: &mut [McpServerState],
    name: &str,
    outcome: Result<nanna_agent::mcp::McpStarted, String>,
) {
    let Some(state) = servers.iter_mut().find(|s| s.name == name) else {
        debug_assert!(
            false,
            "an outcome for a server that was never recorded: {name}"
        );
        return;
    };
    match outcome {
        Ok(started) => {
            state.state = "started";
            state.tools = started.tools;
            state.link = Some(started.link);
        }
        Err(reason) => {
            state.state = "failed";
            state.detail = Some(reason);
        }
    }
}

/// What [`spawn_mcp_servers`] started.
#[derive(Debug)]
pub struct McpStartup {
    /// How many servers were handed to the background task.
    pub count: usize,
    /// The background task: it owns the clients, and finishes once it has
    /// closed them after `shutdown` fires. `None` when nothing was started.
    /// The daemon awaits it (bounded by [`MCP_SHUTDOWN_DEADLINE`]) so the
    /// close actually runs before the process exits.
    pub task: Option<tokio::task::JoinHandle<()>>,
}

/// How long the daemon's shutdown waits for the MCP task.
///
/// Every server's exit grace (they close concurrently) plus one second for
/// the kill and the reap. Past it the task is aborted, which drops the transports and so
/// kills any child still running (`kill_on_drop`).
pub const MCP_SHUTDOWN_DEADLINE: std::time::Duration =
    nanna_mcp::MCP_EXIT_GRACE.saturating_add(std::time::Duration::from_secs(1));

/// Spawn every startable server from `config` and register its tools into
/// `tools`.
///
/// The servers are shut down when `shutdown` fires, so their child processes
/// do not outlive a clean daemon exit — provided the caller awaits the
/// returned task (see [`McpStartup::task`]).
///
/// `secret` looks a `secret_env` value up by store key — the secure store in the
/// daemon. It is called only for servers that list secrets, so a config without
/// them never touches the keyring at boot. A server missing one is reported
/// `not_started` with the command that sets it, and the others still start.
///
/// # Panics
///
/// Never in practice: the assertion restates the bound
/// [`McpConfig::startable`] already enforces.
pub async fn spawn_mcp_servers(
    config: &McpConfig,
    tools: Arc<ToolRegistry>,
    status: McpStatus,
    mut shutdown: broadcast::Receiver<()>,
    secret: impl Fn(&str) -> Option<String>,
    elicitor: Option<Arc<dyn nanna_mcp::Elicitor>>,
) -> McpStartup {
    let (startable, skipped) = config.startable();
    let mut start = Vec::with_capacity(startable.len());
    // Unlike `skipped`, these have a well-formed, unique name to report under.
    let mut refused: Vec<(String, String)> = Vec::new();
    for entry in startable {
        let resolved = entry
            .resolve_secret_env(&secret)
            .and_then(|env| Ok((env, entry.resolve_bearer(&secret)?)));
        match resolved {
            Ok((env, bearer)) => start.push((entry, env, bearer)),
            Err(reason) => refused.push((entry.name.trim().to_string(), reason)),
        }
    }
    for (_, reason) in &refused {
        warn!("{reason}");
    }
    for reason in &skipped {
        warn!("{reason}");
    }
    let starting: Vec<&str> = start
        .iter()
        .map(|(entry, _, _)| entry.name.trim())
        .collect();
    publish_initial_states(&status, &starting, refused, &skipped).await;
    if start.is_empty() {
        return McpStartup {
            count: 0,
            task: None,
        };
    }
    let count = start.len();
    let mut integration = McpIntegration::new();
    if let Some(elicitor) = elicitor {
        integration.set_elicitor(elicitor);
    }
    for (entry, env, bearer) in start {
        let url = entry.url.trim();
        integration.add_server(if url.is_empty() {
            McpServerConfig::new(entry.name.trim(), entry.command.trim())
                .args(entry.args.clone())
                .env(env)
        } else {
            McpServerConfig::http(entry.name.trim(), url).bearer_token(bearer)
        });
    }
    assert!(
        count <= nanna_config::MCP_SERVERS_MAX,
        "startable() enforces the bound"
    );
    info!(servers = count, "Starting MCP servers in the background");
    let task = tokio::spawn(async move {
        // Each server's state lands the moment it settles, so a slow one no
        // longer shows every other server as `starting`.
        let status_for_outcomes = Arc::clone(&status);
        let outcomes = integration
            .start_all(&tools, async move |name: &str, outcome: &Result<_, String>| {
                let mut servers = status_for_outcomes.write().await;
                record_outcome(&mut servers, name, outcome.clone());
            })
            .await;
        let registered: usize = outcomes
            .iter()
            .filter_map(|(_, outcome)| outcome.as_ref().ok())
            .map(|started| started.tools)
            .sum();
        info!(
            servers = count,
            tools = registered,
            "MCP tools registered (named mcp__<server>__<tool>)"
        );
        // Keep the clients alive for the daemon's lifetime, following every
        // server's tool-list changes into the registry; close them on a clean
        // shutdown rather than leaving it to process exit.
        integration
            .watch(&tools, async {
                let _ = shutdown.recv().await;
            })
            .await;
        if let Err(e) = integration.shutdown().await {
            warn!("MCP servers did not shut down cleanly: {e}");
        }
        info!(servers = count, "MCP servers closed");
    });
    McpStartup {
        count,
        task: Some(task),
    }
}

/// Replace the status table with what boot decided: each server about to
/// start, each one refused for a missing secret, each entry skipped.
async fn publish_initial_states(
    status: &McpStatus,
    starting: &[&str],
    refused: Vec<(String, String)>,
    skipped: &[String],
) {
    let mut servers = status.write().await;
    servers.clear();
    servers.extend(starting.iter().map(|name| McpServerState {
        name: (*name).to_string(),
        state: "starting",
        tools: 0,
        detail: None,
        link: None,
    }));
    servers.extend(refused.into_iter().map(|(name, reason)| McpServerState {
        name,
        state: "not_started",
        tools: 0,
        detail: Some(reason),
        link: None,
    }));
    servers.extend(skipped.iter().map(|reason| McpServerState {
        name: String::new(),
        state: "not_started",
        tools: 0,
        detail: Some(reason.clone()),
        link: None,
    }));
    debug_assert!(servers.len() >= starting.len());
    drop(servers);
}

/// Wait for the MCP task to close its servers, at most
/// [`MCP_SHUTDOWN_DEADLINE`]; past it, abort the task so dropping the
/// transports kills whatever is still running.
pub async fn await_mcp_shutdown(task: tokio::task::JoinHandle<()>) {
    let abort = task.abort_handle();
    match tokio::time::timeout(MCP_SHUTDOWN_DEADLINE, task).await {
        Ok(Ok(())) => {}
        Ok(Err(e)) => warn!("MCP shutdown task failed: {e}"),
        Err(_) => {
            warn!(
                deadline_ms = MCP_SHUTDOWN_DEADLINE.as_millis(),
                "MCP servers did not close in time; aborting (children are killed on drop)"
            );
            abort.abort();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn starting(name: &str) -> McpServerState {
        McpServerState {
            name: name.into(),
            state: "starting",
            tools: 0,
            detail: None,
            link: None,
        }
    }

    #[test]
    fn outcomes_land_on_the_server_they_belong_to() {
        let mut servers = vec![starting("files"), starting("git")];
        record_outcome(&mut servers, "git", Err("No such file or directory".into()));
        record_outcome(
            &mut servers,
            "files",
            Ok(nanna_agent::mcp::McpStarted {
                tools: 3,
                link: "2026-07-28 over stdio".into(),
            }),
        );
        assert_eq!(servers[0].state, "started");
        assert_eq!(servers[0].tools, 3);
        assert_eq!(servers[0].link.as_deref(), Some("2026-07-28 over stdio"));
        assert_eq!(
            servers[1].link, None,
            "a failed server was reached over nothing"
        );
        assert_eq!(servers[1].state, "failed");
        assert_eq!(
            servers[1].detail.as_deref(),
            Some("No such file or directory")
        );
    }

    #[tokio::test]
    async fn skipped_entries_are_reported_and_nothing_starts_without_servers() {
        let status = McpStatus::default();
        let config = McpConfig {
            servers: vec![nanna_config::McpServerEntry {
                url: String::new(),
                bearer_secret: None,
                name: "broken".into(),
                command: " ".into(),
                args: Vec::new(),
                enabled: true,
                secret_env: Vec::new(),
            }],
        };
        let (_tx, rx) = broadcast::channel(1);
        let started = spawn_mcp_servers(
            &config,
            Arc::new(ToolRegistry::new()),
            status.clone(),
            rx,
            |_| panic!("no server lists a secret, so the store is not read"),
            None,
        )
        .await;
        assert_eq!(started.count, 0);
        assert!(started.task.is_none());
        let servers = status.read().await.clone();
        assert_eq!(servers.len(), 1);
        assert_eq!(servers[0].state, "not_started");
        assert!(
            servers[0]
                .detail
                .as_deref()
                .is_some_and(|d| d.contains("has no command"))
        );
    }

    #[tokio::test]
    async fn a_server_missing_a_secret_is_not_started_and_says_how_to_set_it() {
        let status = McpStatus::default();
        let config = McpConfig {
            servers: vec![nanna_config::McpServerEntry {
                url: String::new(),
                bearer_secret: None,
                name: "github".into(),
                command: "npx".into(),
                args: Vec::new(),
                enabled: true,
                secret_env: vec!["GITHUB_TOKEN".into()],
            }],
        };
        let (_tx, rx) = broadcast::channel(1);
        let started = spawn_mcp_servers(
            &config,
            Arc::new(ToolRegistry::new()),
            status.clone(),
            rx,
            |_| None,
            None,
        )
        .await;
        assert_eq!(started.count, 0, "nothing is spawned without its token");
        assert!(started.task.is_none());
        let servers = status.read().await.clone();
        assert_eq!(servers.len(), 1);
        assert_eq!(servers[0].name, "github", "reported under its own name");
        assert_eq!(servers[0].state, "not_started");
        assert!(
            servers[0]
                .detail
                .as_deref()
                .is_some_and(|d| d.contains("nanna mcp secret set github GITHUB_TOKEN")),
            "{servers:?}"
        );
    }
}
