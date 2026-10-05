#![warn(clippy::pedantic, clippy::nursery, clippy::all)]
// The auto-trait solver proves `DaemonServer::run`'s scheduler closure is
// `Send` by walking the whole state tuple, and that walk descends through
// `MemoryService` -> `VectorStore` -> `CosineSimilaritySearch` -> wgpu's
// `Global`/`Hub`/`Registry` graph, which is deeper than the default limit of
// 128. nightly-2026-08-25 turned the resulting overflow into the
// future-incompatible `recursion_depth_exceeding_limit` warning (rust#159228)
// — it is scheduled to become a hard error, so the limit is raised here rather
// than left to break a future toolchain bump. This costs solver depth only;
// it changes no behaviour and no generated code.
#![recursion_limit = "256"]

//! Nanna Daemon - Main entry point
//!
//! Usage:
//!   nanna-daemon run              Run the daemon in foreground
//!   nanna-daemon start            Start as background service
//!   nanna-daemon stop             Stop the background service
//!   nanna-daemon status           Show daemon status
//!   nanna-daemon install          Install as system service
//!   nanna-daemon uninstall        Uninstall system service
//!   nanna-daemon service          (Windows only) Run as Windows Service

use clap::{Args, Parser, Subcommand};
use nanna_daemon::server::DaemonBuilder;
use nanna_daemon::service::{ServiceConfig, ServiceManager, ServiceStatus};
use std::path::PathBuf;
use std::sync::OnceLock;
use tracing::{error, info};
use tracing_subscriber::filter::filter_fn;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::Layer as _;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// Global log buffer initialized before the tracing subscriber
static LOG_BUFFER: OnceLock<nanna_daemon::log_buffer::LogBuffer> = OnceLock::new();

#[cfg(windows)]
use nanna_daemon::windows_service;

#[derive(Parser)]
#[command(name = "nanna-daemon")]
#[command(about = "Nanna AI assistant background daemon")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    
    /// WebSocket port to listen on
    #[arg(short, long, default_value_t = nanna_daemon::DEFAULT_IPC_PORT)]
    port: u16,
    
    /// Host to bind to
    #[arg(long, default_value = nanna_config::bind::LOOPBACK_HOST)]
    host: String,
    
    /// HTTP health port (default: 5148)
    #[arg(long, default_value_t = nanna_daemon::DEFAULT_HEALTH_PORT)]
    health_port: u16,
    
    // Flattened in place, so `--no-health-server`, `--no-pid-file` and
    // `--enable-webhooks` keep their names and their position in `--help`.
    #[command(flatten)]
    servers: ServerToggles,
    
    /// Webhook server port (default: 3000)
    #[arg(long, default_value = "3000")]
    webhook_port: u16,
    
    /// Data directory
    #[arg(long)]
    data_dir: Option<PathBuf>,
    
    /// Log level (trace, debug, info, warn, error)
    #[arg(long, default_value = "info")]
    log_level: String,

    /// Directory for rotating file logs (default: `{data_dir}/logs`).
    /// Logs roll daily; at most 7 files are kept.
    #[arg(long)]
    log_dir: Option<PathBuf>,

    /// Disable rotating file logs (console + in-memory buffer only).
    #[arg(long)]
    no_file_log: bool,

    /// Shut down (gracefully) when the process that started this daemon
    /// exits. The GUI passes it for its sidecar, so a killed GUI does not
    /// leave a daemon holding its port and store; a standalone daemon never
    /// gets it. Unix only — on Windows the GUI's Job Object already does this.
    #[arg(long)]
    exit_with_parent: bool,
}

/// The switches for the daemon's optional servers and its PID file.
#[derive(Args)]
struct ServerToggles {
    /// Disable HTTP health server
    #[arg(long)]
    no_health_server: bool,
    
    /// Disable PID file (allows multiple instances)
    #[arg(long)]
    no_pid_file: bool,
    
    /// Enable webhook server for inbound messages
    #[arg(long)]
    enable_webhooks: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Run the daemon in foreground
    Run,
    /// Start as background service
    Start,
    /// Stop the background service
    Stop,
    /// Restart the background service
    Restart,
    /// Show daemon status
    Status,
    /// Install as system service
    Install,
    /// Uninstall system service
    Uninstall,
    /// Run as Windows Service (called by SCM)
    #[cfg(windows)]
    Service,
}

/// Resolve the log directory and build a non-blocking, daily-rotating file
/// writer for the daemon. Returns `None` when file logging is disabled
/// (`--no-file-log`) or the appender can't be built — in both cases startup
/// falls back to console + in-memory logging rather than aborting.
fn file_log_writer(
    cli: &Cli,
) -> Option<(
    tracing_appender::non_blocking::NonBlocking,
    tracing_appender::non_blocking::WorkerGuard,
)> {
    if cli.no_file_log {
        return None;
    }

    // Logs belong beside the data they describe, so this must resolve the data
    // dir the same way the daemon itself will: `--data-dir` wins, then
    // `[general] data_dir` from the config file, then the platform default.
    //
    // Both of the first two were previously ignored here — `--data-dir /tmp/x`
    // put the store in /tmp/x and the logs in the default location, and a
    // relocated install split them the same way.
    //
    // Reading the config deliberately does NOT go through `Config::load`: that
    // hydrates secrets from the OS keyring, which can block on an unlock
    // prompt, and this runs *before* the tracing subscriber is installed — so a
    // daemon parked there would say nothing at all about why it never started.
    let data_dir = cli
        .data_dir
        .clone()
        .or_else(|| nanna_config::Config::data_dir_from_disk().ok())
        .unwrap_or_else(|| PathBuf::from("."));
    let log_dir = nanna_daemon::log_file::resolve_log_dir(cli.log_dir.as_deref(), &data_dir);
    debug_assert!(
        !log_dir.as_os_str().is_empty(),
        "resolved log dir must be non-empty"
    );

    // Forensic dumps (whole prompts) go beside the logs, never to the shared
    // temp directory.
    nanna_daemon::log_file::set_forensics_dir(&log_dir);

    match nanna_daemon::log_file::build_appender(&log_dir) {
        Ok(appender) => {
            let (writer, guard) = tracing_appender::non_blocking(appender);
            Some((writer, guard))
        }
        Err(e) => {
            // Non-fatal: console + in-memory logging still work.
            eprintln!("warning: file logging disabled: {e}");
            None
        }
    }
}

/// Per-layer filter for the fmt layers: every event, but only Nanna's own
/// spans. Span CLOSE lines are enabled for the agent loop's spans, and the
/// setting cannot be scoped by target — without this, a dependency that opens
/// INFO spans prints a close line for each (measured: zbus's keyring
/// handshake emitted a burst of them at boot). A dependency's events still
/// print; they only lose that dependency's own span prefix.
fn own_spans_only(metadata: &tracing::Metadata<'_>) -> bool {
    !metadata.is_span() || metadata.target().starts_with("nanna")
}

fn main() {
    let cli = Cli::parse();

    // Before anything starts a thread: a daemon inside the AppImage mount
    // re-execs from a copy outside it (see `nanna_daemon::appimage`).
    #[cfg(target_os = "linux")]
    let staging = matches!(cli.command, Commands::Run)
        .then(nanna_daemon::appimage::run_outside_the_mount);
    
    // Special case: Windows Service mode doesn't parse args normally
    #[cfg(windows)]
    if let Commands::Service = cli.command {
        // Minimal logging for service mode
        tracing_subscriber::registry()
            .with(tracing_subscriber::EnvFilter::new("info"))
            .with(tracing_subscriber::fmt::layer())
            .init();
        
        if let Err(e) = windows_service::run_as_service() {
            error!("Service failed: {}", e);
            std::process::exit(1);
        }
        return;
    }
    
    // Setup logging for non-service modes with log buffer for GUI
    let log_buffer = nanna_daemon::log_buffer::LogBuffer::new(
        nanna_daemon::LOG_BUFFER_LINES,
        nanna_daemon::log_buffer::LogSource::Daemon,
    );
    let log_layer = nanna_daemon::log_buffer::LogBufferLayer::new(log_buffer.clone());

    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(&cli.log_level));

    // Optional rotating file log. `Option<Layer>` is itself a `Layer` (None is a
    // no-op), so this composes cleanly whether or not file logging is enabled.
    // The worker guard is kept as a `main`-scoped local so it drops (flushing the
    // appender) on normal return; a `static` would never be dropped at exit.
    let (file_layer, _file_log_guard) = match file_log_writer(&cli) {
        Some((writer, guard)) => {
            let layer = tracing_subscriber::fmt::layer()
                .with_ansi(false)
                .with_span_events(FmtSpan::CLOSE)
                .with_writer(writer)
                .with_filter(filter_fn(own_spans_only));
            (Some(layer), Some(guard))
        }
        None => (None, None),
    };

    // Span CLOSE events carry what each span measured (P6): the agent loop's
    // `llm_call` / `tool_call` / `agent_run` spans record latency, token and
    // byte counts, and success as they settle, and this is the line that
    // prints them — with `time.busy`/`time.idle` — once per span.
    tracing_subscriber::registry()
        .with(filter)
        .with(
            tracing_subscriber::fmt::layer()
                .with_span_events(FmtSpan::CLOSE)
                .with_filter(filter_fn(own_spans_only)),
        )
        .with(log_layer)
        .with(file_layer)
        .init();

    // Store log_buffer so run_daemon can pass it to the DaemonBuilder
    LOG_BUFFER.set(log_buffer).ok();

    #[cfg(target_os = "linux")]
    match staging {
        Some(nanna_daemon::appimage::Outcome::Staged { exe }) => {
            info!("Running from {}, a copy outside the AppImage mount", exe.display());
        }
        Some(nanna_daemon::appimage::Outcome::Failed(reason)) => tracing::warn!(
            "Running from inside the AppImage mount ({reason}): if the app exits first, this daemon's shutdown can die of SIGBUS"
        ),
        Some(nanna_daemon::appimage::Outcome::NotInAppImage) | None => {}
    }
    
    let result = match cli.command {
        Commands::Run => run_daemon(&cli),
        Commands::Start => start_service(&cli),
        Commands::Stop => stop_service(&cli),
        Commands::Restart => restart_service(&cli),
        Commands::Status => {
            show_status(&cli);
            Ok(())
        }
        Commands::Install => install_service(&cli),
        Commands::Uninstall => uninstall_service(&cli),
        #[cfg(windows)]
        Commands::Service => unreachable!(), // Handled above
    };
    
    if let Err(e) = result {
        error!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run_daemon(cli: &Cli) -> Result<(), String> {
    info!("Starting Nanna daemon...");
    
    // Create tokio runtime
    let runtime = tokio::runtime::Runtime::new()
        .map_err(|e| format!("Failed to create runtime: {e}"))?;
    
    runtime.block_on(async {
        // Load config from Nanna config file (includes API keys)
        let mut builder = DaemonBuilder::from_nanna_config()
            .map_err(|e| format!("Failed to load config: {e}"))?
            .with_port(cli.port)
            .with_host(&cli.host)
            .with_log_level(&cli.log_level)
            .with_health_port(cli.health_port)
            .with_health_server(!cli.servers.no_health_server)
            .with_pid_file(!cli.servers.no_pid_file)
            .with_webhook_server(cli.servers.enable_webhooks)
            .with_webhook_port(cli.webhook_port);
        
        if let Some(ref data_dir) = cli.data_dir {
            builder = builder.with_data_dir(data_dir);
        }

        // Pass log buffer to daemon for serving logs via control plane
        if let Some(buf) = LOG_BUFFER.get() {
            builder = builder.with_log_buffer(buf.clone());
        }

        let mut daemon = builder.build();

        nanna_daemon::shutdown::install_signal_handlers(&daemon).map_err(|e| e.to_string())?;

        #[cfg(unix)]
        if cli.exit_with_parent {
            spawn_parent_watch(daemon.shutdown_handle(), daemon.exit_reason_handle());
        }

        daemon.run().await.map_err(|e| e.to_string())
    })
}

/// How often the daemon checks that its parent is still alive under
/// `--exit-with-parent`: one cheap syscall a second bounds how long an
/// orphaned sidecar lives before its graceful shutdown starts.
#[cfg(unix)]
const PARENT_POLL: std::time::Duration = std::time::Duration::from_secs(1);

/// Whether the parent that started us is gone: a process whose parent exits
/// is re-parented (to init or a subreaper), so its parent id changes. Pure.
#[cfg(unix)]
const fn parent_gone(original: u32, now: u32) -> bool {
    now != original
}

/// Watch for the parent's exit and then shut down the way a signal does:
/// record why, and send the drain broadcast (which, among other things,
/// closes the MCP servers this daemon started).
#[cfg(unix)]
fn spawn_parent_watch(
    shutdown: tokio::sync::broadcast::Sender<()>,
    exit_reason: nanna_daemon::exit_reason::ExitReasonFile,
) {
    let original = std::os::unix::process::parent_id();
    assert!(original > 0, "a running process always has a parent");
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(PARENT_POLL);
        loop {
            tick.tick().await;
            let now = std::os::unix::process::parent_id();
            if parent_gone(original, now) {
                info!("Parent process {original} exited (now re-parented to {now}); shutting down");
                exit_reason.record_exit("parent_exited", None);
                let _ = shutdown.send(());
                break;
            }
        }
    });
}

#[cfg(all(test, unix))]
mod parent_watch_tests {
    use super::parent_gone;

    #[test]
    fn a_changed_parent_id_means_the_parent_is_gone() {
        assert!(!parent_gone(4242, 4242));
        assert!(parent_gone(4242, 1), "re-parented to init");
        assert!(parent_gone(4242, 777), "re-parented to a subreaper");
    }
}

fn start_service(cli: &Cli) -> Result<(), String> {
    let manager = get_service_manager(cli);
    if manager.status() == ServiceStatus::Running {
        println!("Daemon is already running");
    } else {
        println!("Starting daemon...");
        manager.start()?;
        println!("Daemon started");
    }
    Ok(())
}

fn stop_service(cli: &Cli) -> Result<(), String> {
    let manager = get_service_manager(cli);
    if manager.status() == ServiceStatus::Stopped {
        println!("Daemon is not running");
    } else {
        println!("Stopping daemon...");
        manager.stop()?;
        println!("Daemon stopped");
    }
    Ok(())
}

fn restart_service(cli: &Cli) -> Result<(), String> {
    let manager = get_service_manager(cli);
    if manager.status() == ServiceStatus::Running {
        println!("Stopping daemon...");
        manager.stop()?;
        std::thread::sleep(std::time::Duration::from_secs(2));
    }
    println!("Starting daemon...");
    manager.start()?;
    println!("Daemon restarted");
    Ok(())
}

fn show_status(cli: &Cli) {
    let status = get_service_manager(cli).status();

    println!("Nanna Daemon Status");
    println!("==================");
    println!("Service: {}", match status {
        ServiceStatus::Running => "Running ✓",
        ServiceStatus::Stopped => "Stopped",
        ServiceStatus::Starting => "Starting...",
        ServiceStatus::Stopping => "Stopping...",
        ServiceStatus::Unknown => "Unknown",
    });
    println!("WebSocket Port: {}", cli.port);
    println!("Health Port: {}", cli.health_port);
    println!("Host: {}", cli.host);
    
    // Try to connect to health endpoint
    if status == ServiceStatus::Running || status == ServiceStatus::Unknown {
        let health_url = format!("http://{}:{}/health", cli.host, cli.health_port);
        println!("\nChecking health endpoint: {health_url}");
        
        // Try to fetch health endpoint
        match std::process::Command::new("curl")
            .args(["-s", "-m", "2", &health_url])
            .output()
        {
            Ok(output) if output.status.success() => {
                let body = String::from_utf8_lossy(&output.stdout);
                if body.contains("ok") {
                    println!("Health: OK ✓");
                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&body) {
                        if let Some(uptime) = json.get("uptime_secs").and_then(serde_json::Value::as_u64) {
                            let hours = uptime / 3600;
                            let mins = (uptime % 3600) / 60;
                            let secs = uptime % 60;
                            println!("Uptime: {hours}h {mins}m {secs}s");
                        }
                        if let Some(version) = json.get("version").and_then(|v| v.as_str()) {
                            println!("Version: {version}");
                        }
                    }
                } else {
                    println!("Health: Unknown (response: {})", body.chars().take(50).collect::<String>());
                }
            }
            Ok(_) => {
                println!("Health: Not responding (daemon may be starting or not running)");
            }
            Err(_) => {
                println!("Health: Could not check (curl not available)");
                println!("\nTrying WebSocket at ws://{}:{}...", cli.host, cli.port);
                println!("(WebSocket test not implemented)");
            }
        }
    }
}

fn install_service(cli: &Cli) -> Result<(), String> {
    let manager = get_service_manager(cli);
    println!("Installing Nanna daemon as system service...");
    manager.install()?;
    println!("Service installed successfully");
    println!("\nTo start the daemon, run:");
    println!("  nanna-daemon start");
    Ok(())
}

fn uninstall_service(cli: &Cli) -> Result<(), String> {
    let manager = get_service_manager(cli);

    // Stop if running
    if manager.status() == ServiceStatus::Running {
        println!("Stopping running daemon...");
        manager.stop()?;
        std::thread::sleep(std::time::Duration::from_secs(2));
    }

    println!("Uninstalling Nanna daemon service...");
    manager.uninstall()?;
    println!("Service uninstalled successfully");
    Ok(())
}

fn get_service_manager(_cli: &Cli) -> ServiceManager {
    ServiceManager::new(ServiceConfig::default())
}
