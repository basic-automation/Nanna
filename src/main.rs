#![warn(clippy::all)]
#![warn(clippy::pedantic, clippy::nursery)]
// Raised for the same reason as `nanna-daemon`'s: proving a future or closure
// is `Send` walks into wgpu's `Global`/`Hub`/`Registry` graph by way of
// `CosineSimilaritySearch`, which is deeper than the default limit of 128.
// nightly-2026-08-25 turned that overflow into the future-incompatible
// `recursion_depth_exceeding_limit` warning (rust#159228), which is scheduled
// to become a hard error. Solver depth only — no behaviour, no codegen change.
#![recursion_limit = "256"]

//! Nanna - High-performance AI assistant in Rust.
//!
//! Moon god of the digital realm.
//! Built with SIMD and GPU acceleration for unrelenting performance.

mod commands;
mod onboarding;
mod setup;

use clap::{Parser, Subcommand};
use commands::cli::{list_sessions, run_cli, run_once};
use commands::credentials::handle_credentials_command;
use commands::daemon::handle_daemon_command;
use commands::serve::{run_daemon, run_server};
use commands::workspace::handle_workspace_command;
use nanna_config::Config;
use nanna_config::bind::LOOPBACK_HOST;
use nanna_daemon::DEFAULT_IPC_PORT;
use nanna_daemon::log_buffer::{LogBuffer, LogBufferLayer, LogSource};
use setup::ensure_api_key;
use std::path::PathBuf;
use tracing::{info, Level};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Parser)]
#[command(name = "nanna")]
#[command(author, version, about = "High-performance AI assistant", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Config file path
    #[arg(short, long)]
    config: Option<PathBuf>,

    /// Log level (trace, debug, info, warn, error)
    #[arg(short, long, default_value = "info")]
    log_level: String,

    /// Run in daemon mode (background service)
    ///
    /// Never with `--config`. The daemon reads its config file in several
    /// places, and each finds it through `NANNA_CONFIG_PATH`, which `nanna
    /// --config <file> daemon start` sets for the daemon it launches. A
    /// `--config` here would reach only some of them, and the rest would read
    /// and save the default file.
    #[arg(long, hide = true, conflicts_with = "config")]
    daemon_mode: bool,

    /// Daemon host
    #[arg(long, hide = true, default_value = LOOPBACK_HOST)]
    host: String,

    /// Daemon port
    #[arg(long, hide = true, default_value_t = DEFAULT_IPC_PORT)]
    port: u16,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize Nanna with setup wizard
    Init,

    /// Show configuration status
    Status,

    /// Diagnose configuration problems and say how to fix each one.
    ///
    /// Offline by default: no provider, network or keyring probe runs, so this
    /// is safe and fast anywhere. Exits non-zero when a check fails.
    Doctor {
        /// Also probe each Ollama server the configuration uses: is it
        /// answering, and does it have the configured models? Never reads the
        /// keyring or sends a provider key.
        #[arg(long)]
        online: bool,
    },

    /// Start the HTTP server
    Server {
        /// Host to bind to. Defaults to loopback; pass `0.0.0.0` to expose the
        /// server to other machines (it has no authentication of its own).
        #[arg(short = 'H', long, default_value = nanna_config::LOOPBACK_HOST)]
        host: String,

        /// Port to listen on. Defaults to `[server].port` in config.toml (or the
        /// `PORT` environment variable), which itself defaults to 3000.
        #[arg(short, long)]
        port: Option<u16>,
    },

    /// Daemon management (always-on background service)
    Daemon {
        #[command(subcommand)]
        action: DaemonAction,
    },

    /// Interactive CLI mode
    Chat {
        /// Session ID to resume
        #[arg(short, long)]
        session: Option<String>,

        /// Model to use
        #[arg(short, long)]
        model: Option<String>,

        /// Stream responses (print as they arrive)
        #[arg(long)]
        stream: bool,
    },

    /// List recent sessions
    Sessions {
        /// Number of sessions to show
        #[arg(short, long, default_value = "10")]
        limit: i64,
    },

    /// Export a session — or, with `--memories`, the memory store — as
    /// Markdown (readable) or JSON (lossless)
    //
    // Exactly one target, enforced by a group: `requires = "memories"` on
    // `--scope` was satisfied by the bool flag's implicit `false` default, so
    // `nanna export <id> --scope x` parsed and silently ignored the scope.
    #[command(group(clap::ArgGroup::new("target").required(true).args(["session", "memories"])))]
    Export {
        /// Session ID (see `nanna sessions`); omit it with --memories
        session: Option<String>,

        /// Export the memory store instead of a session
        #[arg(long)]
        memories: bool,

        /// With --memories: `global`, or a workspace id (global plus that
        /// workspace). Every memory when omitted.
        #[arg(long, conflicts_with = "session")]
        scope: Option<String>,

        /// Document format
        #[arg(short, long, value_enum, default_value_t = commands::export::ExportFormatArg::Markdown)]
        format: commands::export::ExportFormatArg,

        /// Write here — a file, or a directory that receives the suggested
        /// file name. Prints to stdout when omitted.
        #[arg(short, long)]
        output: Option<std::path::PathBuf>,

        /// The daemon's IPC address (default: the local daemon on the default
        /// port) — for a daemon started with `daemon start --port`
        #[arg(long)]
        daemon: Option<String>,
    },

    /// Run a single prompt and exit
    Run {
        /// The prompt to run
        prompt: String,

        /// Model to use
        #[arg(short, long)]
        model: Option<String>,
    },

    /// Show or generate configuration
    Config {
        /// Generate default config
        #[arg(long)]
        generate: bool,
    },

    /// Workspace management
    Workspace {
        #[command(subcommand)]
        action: WorkspaceAction,
    },

    /// Manage Claude CLI credentials (OAuth)
    Credentials {
        #[command(subcommand)]
        action: CredentialsAction,
    },

    /// Model Context Protocol server — expose Nanna's tools to an MCP client
    Mcp {
        #[command(subcommand)]
        action: McpAction,
    },
}

#[derive(Subcommand)]
enum McpAction {
    /// Serve Nanna's tools over stdio JSON-RPC (for Claude Code, editors, …).
    ///
    /// stdout carries the protocol, so all logging goes to stderr.
    Serve {
        /// Directory of JS/TS tool skills (default: `[tools] tools_dir`,
        /// `NANNA_TOOLS_DIR`, or the dev tree) — used by the standalone surface
        #[arg(long)]
        tools_dir: Option<std::path::PathBuf>,
        /// Serve the locally loaded skills even if a daemon is running
        #[arg(long)]
        standalone: bool,
        /// The daemon's IPC address (default: the local daemon); naming one
        /// makes its absence an error instead of a fallback
        #[arg(long)]
        daemon: Option<String>,
    },

    /// Store or remove a secret an MCP server gets as an environment variable
    /// (listed by name in its `secret_env`). Values go to the OS keyring,
    /// never to config.toml, and are read from a prompt or stdin — not argv.
    Secret {
        #[command(subcommand)]
        action: McpSecretAction,
    },
}

#[derive(Subcommand)]
enum McpSecretAction {
    /// Store a value: `nanna mcp secret set github GITHUB_PERSONAL_ACCESS_TOKEN`
    Set { server: String, var: String },
    /// Remove a stored value
    Delete { server: String, var: String },
}

#[derive(Subcommand)]
enum WorkspaceAction {
    /// Initialize a new workspace in the current directory
    Init {
        /// Template to use (minimal, project)
        #[arg(short, long, default_value = "project")]
        template: String,

        /// Path to initialize (defaults to current directory)
        path: Option<String>,
    },

    /// Show current workspace status
    Status,

    /// List available templates
    Templates,

    /// Reload workspace files
    Reload,
}

#[derive(Subcommand)]
enum CredentialsAction {
    /// Show current credential status
    Status,

    /// Import credentials from Claude Code CLI (~/.claude/.credentials.json)
    Import,

    /// Run `claude setup-token` to authenticate via Claude Code CLI
    Setup,

    /// Refresh the OAuth token (if expired or expiring soon)
    Refresh,

    /// Clear stored credentials
    Clear,
}

#[derive(Subcommand)]
enum DaemonAction {
    /// Start the daemon in the background
    Start {
        /// Host to bind to
        #[arg(short = 'H', long, default_value = LOOPBACK_HOST)]
        host: String,

        /// Port to listen on
        #[arg(short, long, default_value_t = DEFAULT_IPC_PORT)]
        port: u16,
    },

    /// Stop the running daemon
    Stop,

    /// Check daemon status
    Status {
        /// Host the daemon was started on
        #[arg(short = 'H', long, default_value = LOOPBACK_HOST)]
        host: String,

        /// Port the daemon was started on (`daemon start --port`)
        #[arg(short, long, default_value_t = DEFAULT_IPC_PORT)]
        port: u16,
    },

    /// Restart the daemon
    Restart {
        /// Host to bind to
        #[arg(short = 'H', long, default_value = LOOPBACK_HOST)]
        host: String,

        /// Port to listen on
        #[arg(short, long, default_value_t = DEFAULT_IPC_PORT)]
        port: u16,
    },
}

/// `nanna export`, out of `main`'s match.
async fn run_export(command: Commands) -> anyhow::Result<()> {
    let Commands::Export {
        session,
        memories,
        scope,
        format,
        output,
        daemon,
    } = command
    else {
        anyhow::bail!("not the export command");
    };
    let target = commands::export::ExportTarget::from_cli(session, memories, scope)?;
    commands::export::export(target, format, output, daemon).await
}

/// Parse the `--log-level` string into a level, defaulting to INFO for an
/// unrecognised value rather than failing the whole run over a typo.
fn parse_log_level(raw: &str) -> Level {
    match raw.to_lowercase().as_str() {
        "trace" => Level::TRACE,
        "debug" => Level::DEBUG,
        "warn" => Level::WARN,
        "error" => Level::ERROR,
        _ => Level::INFO,
    }
}

/// Install the global tracing subscriber.
///
/// `logs_to_stderr` exists for `nanna mcp serve`, which speaks JSON-RPC on
/// stdout: ANY log written there corrupts the stream and the client drops the
/// connection with a parse error. Every other command keeps writing to stdout,
/// so this is a per-command decision, not a global one.
///
/// `log_buffer`, when given, also receives every line: the daemon's in-memory
/// tail, served over `system.logs`.
fn init_logging(log_level: Level, logs_to_stderr: bool, log_buffer: Option<LogBuffer>) {
    let filter = tracing_subscriber::EnvFilter::builder()
        .with_default_directive(log_level.into())
        .from_env_lossy();
    // `None` is a no-op layer.
    let buffer_layer = log_buffer.map(LogBufferLayer::new);
    if logs_to_stderr {
        tracing_subscriber::registry()
            .with(tracing_subscriber::fmt::layer().with_writer(std::io::stderr))
            .with(buffer_layer)
            .with(filter)
            .init();
    } else {
        tracing_subscriber::registry()
            .with(tracing_subscriber::fmt::layer())
            .with(buffer_layer)
            .with(filter)
            .init();
    }
}

/// `nanna mcp …`.
async fn run_mcp(config: &Config, action: McpAction) -> anyhow::Result<()> {
    match action {
        McpAction::Serve {
            tools_dir,
            standalone,
            daemon,
        } => commands::mcp::serve(config, tools_dir, standalone, daemon).await,
        McpAction::Secret { action } => match action {
            McpSecretAction::Set { server, var } => commands::mcp::secret_set(config, &server, &var),
            McpSecretAction::Delete { server, var } => commands::mcp::secret_delete(&server, &var),
        },
    }
}


/// The default config file as loaded, or — when it exists but cannot be read
/// — an error, except for `doctor`, which runs on the defaults to report it.
///
/// A file that did not parse used to become the DEFAULTS, logged at info
/// level, and every command ran on them: the first one that saved (quick
/// setup asking for an API key, a credentials command) wrote the defaults
/// over the user's file. A missing file is not an error (`Config::load`
/// answers the defaults for it).
fn config_or_refusal<E: std::fmt::Display>(
    loaded: Result<Config, E>,
    diagnosing: bool,
) -> anyhow::Result<Config> {
    match loaded {
        Ok(config) => Ok(config),
        Err(e) if diagnosing => {
            info!("Using default config ({e}) — doctor reports why");
            Ok(Config::default())
        }
        Err(e) => anyhow::bail!(
            "config.toml could not be read: {e}. Fix it (`nanna doctor` checks it) — \
             running on the defaults instead could write them over your file"
        ),
    }
}

/// The file `nanna doctor` reports on: the one the config was loaded from
/// (`--config`), not always the default path.
fn doctor_config_path(given: Option<&PathBuf>) -> anyhow::Result<PathBuf> {
    Ok(match given {
        Some(path) => path.clone(),
        None => Config::default_config_path()?,
    })
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let logs_to_stderr = matches!(cli.command, Some(Commands::Mcp { .. }));
    // A daemon keeps its recent lines in memory for `system.logs` (the GUI's
    // Logs page), as `nanna-daemon` does. Other commands keep none.
    let daemon_log_buffer = cli
        .daemon_mode
        .then(|| LogBuffer::new(nanna_daemon::LOG_BUFFER_LINES, LogSource::Daemon));
    init_logging(
        parse_log_level(&cli.log_level),
        logs_to_stderr,
        daemon_log_buffer.clone(),
    );

    // The banner is a log line, so it follows the same writer — it must never
    // land on stdout ahead of the first JSON-RPC response.
    info!("🌙 Nanna v{} rising...", env!("CARGO_PKG_VERSION"));

    // Daemon mode - run background server. The daemon loads its own
    // configuration, through the builder `nanna-daemon` uses, so it is
    // dispatched before the CLI loads one.
    if let Some(log_buffer) = daemon_log_buffer {
        info!("Starting in daemon mode on {}:{}", cli.host, cli.port);
        return run_daemon(cli.host, cli.port, log_buffer).await;
    }

    // Load configuration
    let config = if let Some(path) = &cli.config {
        Config::load_from(path)?
    } else {
        config_or_refusal(Config::load(), matches!(cli.command, Some(Commands::Doctor { .. })))?
    }
    .with_env_overrides();

    // Handle commands
    match cli.command {
        Some(Commands::Init) => {
            let _config = onboarding::run_onboarding()?;
            return Ok(());
        }
        Some(Commands::Status) => {
            onboarding::show_status(&config)?;
            return Ok(());
        }
        Some(Commands::Doctor { online }) => {
            let path = doctor_config_path(cli.config.as_ref())?;
            let worst = commands::doctor::run(&config, &path, online).await;
            // Non-zero on a real fault so this is usable from a script or a
            // health probe, not just by eye.
            if worst == commands::doctor::Severity::Fail {
                std::process::exit(1);
            }
            return Ok(());
        }
        Some(Commands::Config { generate }) => {
            if generate {
                println!("{}", nanna_config::generate_default_config());
            } else {
                // The file this config came from: under `--config` the
                // default path labelled another file's settings.
                let path = doctor_config_path(cli.config.as_ref())?;
                println!("Config path: {}", path.display());
                println!("\n{}", toml::to_string_pretty(&config)?);
            }
            return Ok(());
        }
        Some(Commands::Workspace { action }) => {
            handle_workspace_command(action).await?;
            return Ok(());
        }
        Some(Commands::Credentials { action }) => {
            handle_credentials_command(action).await?;
            return Ok(());
        }
        Some(Commands::Daemon { action }) => {
            handle_daemon_command(action, &config, cli.config.as_deref()).await?;
            return Ok(());
        }
        Some(Commands::Mcp { action }) => {
            run_mcp(&config, action).await?;
            return Ok(());
        }
        Some(Commands::Server { host, port }) => {
            // Check for API key, offer quick setup if missing
            let config = ensure_api_key(config)?;
            let port = commands::serve::server_port(port, &config);
            run_server(&config, host, port).await?;
        }
        Some(Commands::Chat { session, model, stream }) => {
            let config = interactive_config(config)?;
            run_cli(&config, session, model, stream).await?;
        }
        Some(export @ Commands::Export { .. }) => return run_export(export).await,
        Some(Commands::Sessions { limit }) => {
            list_sessions(&config, limit).await?;
        }
        Some(Commands::Run { prompt, model }) => {
            let config = ensure_api_key(config)?;
            run_once(&config, &prompt, model).await?;
        }
        None => {
            // Default: interactive chat.
            let config = interactive_config(config)?;
            run_cli(&config, None, None, false).await?;
        }
    }

    Ok(())
}

/// The config an interactive chat starts with: onboarding on a first run,
/// otherwise the loaded config, with quick setup offered if no API key is set.
fn interactive_config(config: Config) -> anyhow::Result<Config> {
    if onboarding::is_first_run() {
        println!("Welcome! Let's get you set up first.\n");
        return onboarding::run_onboarding();
    }
    ensure_api_key(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `nanna server` with no `--port` must parse to `None` so the config
    /// decides the port: a clap `default_value` here is exactly what used to
    /// shadow `[server].port`. The subcommand's flag must also not be
    /// confused with the hidden global daemon `--port`.
    #[test]
    fn the_server_port_flag_is_optional_so_the_config_can_decide() {
        let bare = Cli::try_parse_from(["nanna", "server"]).expect("bare `nanna server` parses");
        match bare.command {
            Some(Commands::Server { port, .. }) => assert_eq!(port, None),
            _ => panic!("expected the server command"),
        }
        let flagged = Cli::try_parse_from(["nanna", "server", "--port", "8080"])
            .expect("`nanna server --port 8080` parses");
        match flagged.command {
            Some(Commands::Server { port, .. }) => assert_eq!(port, Some(8080)),
            _ => panic!("expected the server command"),
        }
        assert_eq!(
            flagged.port, DEFAULT_IPC_PORT,
            "the subcommand's --port does not leak into the global daemon port"
        );
    }

    /// `nanna daemon start` launches `nanna <DAEMON_MODE_FLAG>`, and the
    /// daemon's single-instance probe tells that legacy daemon from any other
    /// `nanna` command by the same flag on its command line — so the constant
    /// must be exactly the flag clap parses.
    #[test]
    fn the_daemon_mode_flag_the_probe_looks_for_is_the_one_clap_parses() {
        let cli = Cli::try_parse_from(["nanna", nanna_daemon::health::DAEMON_MODE_FLAG])
            .expect("the daemon-mode flag parses");
        assert!(cli.daemon_mode);
        assert!(!Cli::try_parse_from(["nanna"]).expect("bare `nanna` parses").daemon_mode);
    }

    /// The command line `nanna daemon start` gives its child parses back to
    /// exactly what was asked for: daemon mode, on the host and port the
    /// starting command used.
    #[test]
    fn the_daemon_command_line_parses_back_to_its_settings() {
        let args = commands::daemon::daemon_args("127.0.0.2", 6001);
        let cli = Cli::try_parse_from(std::iter::once("nanna".into()).chain(args))
            .expect("the daemon command line parses");
        assert!(cli.daemon_mode);
        assert!(cli.command.is_none());
        assert_eq!(cli.config, None);
        assert_eq!(cli.host, "127.0.0.2");
        assert_eq!(cli.port, 6001);
    }

    /// `daemon status` probes the port the daemon was started on: a daemon
    /// started with `--port` used to be reported "Not responding" because
    /// status always probed the default.
    #[test]
    fn daemon_status_takes_the_port_start_was_given() {
        let parse = |args: &[&str]| {
            let cli = Cli::try_parse_from(["nanna", "daemon", "status"].iter().chain(args))
                .expect("`nanna daemon status` parses");
            match cli.command {
                Some(Commands::Daemon {
                    action: DaemonAction::Status { host, port },
                }) => (host, port),
                _ => panic!("expected daemon status"),
            }
        };
        assert_eq!(parse(&[]), (LOOPBACK_HOST.to_string(), DEFAULT_IPC_PORT));
        assert_eq!(parse(&["--port", "6001"]).1, 6001);
        assert_eq!(parse(&["-H", "127.0.0.2"]).0, "127.0.0.2");
    }

    /// `export` reaches a daemon started on another port: it always dialled
    /// the default.
    #[test]
    fn export_takes_the_daemon_address() {
        let daemon_of = |args: &[&str]| {
            let cli = Cli::try_parse_from(["nanna", "export", "--memories"].iter().chain(args))
                .expect("`nanna export --memories` parses");
            match cli.command {
                Some(Commands::Export { daemon, .. }) => daemon,
                _ => panic!("expected export"),
            }
        };
        assert_eq!(daemon_of(&[]), None, "the default is resolved at run time");
        assert_eq!(
            daemon_of(&["--daemon", "ws://127.0.0.1:6001"]).as_deref(),
            Some("ws://127.0.0.1:6001")
        );
    }

    /// A config file that does not parse stops every command but `doctor`.
    #[test]
    fn a_config_that_does_not_parse_is_refused_not_defaulted() {
        let broken = || Err::<Config, &str>("TOML parse error at line 3");
        let refused = config_or_refusal(broken(), false).expect_err("refused");
        assert!(refused.to_string().contains("line 3"), "{refused}");
        assert!(config_or_refusal(broken(), true).is_ok(), "doctor runs to report it");
        let mut mine = Config::default();
        mine.llm.model = "mine".to_string();
        assert_eq!(config_or_refusal::<&str>(Ok(mine), false).expect("ok").llm.model, "mine");
    }

    /// A daemon is never given its config file as `--config`, which only part
    /// of it would read. `NANNA_CONFIG_PATH` reaches every reader.
    #[test]
    fn daemon_mode_refuses_a_config_argument() {
        let refused = Cli::try_parse_from([
            "nanna",
            nanna_daemon::health::DAEMON_MODE_FLAG,
            "--config",
            "nanna.toml",
        ]);
        assert!(refused.is_err(), "daemon mode accepted --config");
        assert!(
            Cli::try_parse_from(["nanna", "--config", "nanna.toml", "daemon", "status"]).is_ok(),
            "the daemon commands still take --config"
        );
    }

    /// `doctor` stays offline unless asked: the network leg is opt-in.
    #[test]
    fn doctor_is_offline_unless_asked() {
        let bare = Cli::try_parse_from(["nanna", "doctor"]).expect("`nanna doctor` parses");
        assert!(matches!(
            bare.command,
            Some(Commands::Doctor { online: false })
        ));
        let online =
            Cli::try_parse_from(["nanna", "doctor", "--online"]).expect("`--online` parses");
        assert!(matches!(
            online.command,
            Some(Commands::Doctor { online: true })
        ));
    }

    #[test]
    fn export_parses_its_session_format_and_output() {
        let cli = Cli::try_parse_from(["nanna", "export", "abc123", "-f", "md", "-o", "out.md"])
            .expect("`nanna export` parses");
        match cli.command {
            Some(Commands::Export {
                session,
                memories,
                format,
                output,
                ..
            }) => {
                assert_eq!(session.as_deref(), Some("abc123"));
                assert!(!memories, "a session export is not a memory export");
                assert_eq!(
                    format,
                    commands::export::ExportFormatArg::Markdown,
                    "`md` is an alias"
                );
                assert_eq!(output, Some(std::path::PathBuf::from("out.md")));
            }
            _ => panic!("expected the export command"),
        }
        let bare = Cli::try_parse_from(["nanna", "export", "abc123"]).expect("parses");
        match bare.command {
            Some(Commands::Export { format, output, .. }) => {
                assert_eq!(
                    format,
                    commands::export::ExportFormatArg::Markdown,
                    "markdown by default"
                );
                assert_eq!(output, None, "stdout by default");
            }
            _ => panic!("expected the export command"),
        }
    }

    #[test]
    fn export_memories_takes_a_scope_and_no_session() {
        let cli = Cli::try_parse_from(["nanna", "export", "--memories", "--scope", "ws1"])
            .expect("`nanna export --memories` parses");
        match cli.command {
            Some(Commands::Export {
                session,
                memories,
                scope,
                ..
            }) => {
                assert!(memories);
                assert_eq!(session, None);
                assert_eq!(scope.as_deref(), Some("ws1"));
            }
            _ => panic!("expected the export command"),
        }
        assert!(
            Cli::try_parse_from(["nanna", "export", "abc123", "--memories"]).is_err(),
            "a session id and --memories are one or the other"
        );
        assert!(
            Cli::try_parse_from(["nanna", "export", "abc123", "--scope", "ws1"]).is_err(),
            "--scope only means something with --memories"
        );
        assert!(
            Cli::try_parse_from(["nanna", "export"]).is_err(),
            "exporting nothing is refused"
        );
    }
}
