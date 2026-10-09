//! `daemon` subcommand handlers and background process management.
//!
//! The daemon owns its single-instance PID file: it claims the file under a
//! lock at startup and removes it on the way out. These commands only read
//! it, judged by the daemon's own [`PidFile::state`]. They never write or
//! delete it: a write from here ran outside the daemon's lock (and could
//! overwrite a live daemon's record with a child about to be refused), and a
//! liveness-only check of their own read a PID reused by another program as
//! a running daemon, which `stop` would then signal.

use crate::DaemonAction;
use nanna_config::Config;
use nanna_config::bind::LOOPBACK_HOST;
use nanna_daemon::DEFAULT_IPC_PORT;
use nanna_daemon::health::{DAEMON_MODE_FLAG, PidFile, PidFileState, ProcessProbe};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use tracing::info;

#[cfg(windows)]
use std::os::windows::process::CommandExt;

/// The data directory of the daemon these commands manage: `[general]
/// data_dir`, else the platform default. It is resolved as the daemon resolves
/// its own (`DaemonBuilder::from_nanna_config` → [`Config::resolve_data_dir`]),
/// so `status` and `stop` read the PID file the daemon wrote. These commands
/// used `Config::default_data_dir`, which ignores `[general] data_dir`, so on
/// a relocated install they looked at the wrong PID file.
fn daemon_data_dir(config: &Config) -> anyhow::Result<PathBuf> {
    Ok(config.resolve_data_dir()?)
}

/// Handle daemon subcommands
///
/// `config_path` is the CLI's `--config`: the daemon `start` launches runs on
/// the same file, so it resolves the same data directory as these commands.
pub async fn handle_daemon_command(
    action: DaemonAction,
    config: &Config,
    config_path: Option<&Path>,
) -> anyhow::Result<()> {
    let data_dir = daemon_data_dir(config)?;
    let pid_file = PidFile::new(&data_dir);

    match action {
        DaemonAction::Start { host, port } => {
            println!("🌙 Starting Nanna daemon...\n");
            if let Some(pid) = pid_file.state()?.live_daemon() {
                println!("⚠️  Daemon is already running (PID {pid})");
                println!("   Run 'nanna daemon status' to check details");
                println!("   Run 'nanna daemon stop' to stop it");
                return Ok(());
            }
            let (pid, log_file) = spawn_daemon_process(&host, port, &data_dir, config_path)?;
            println!("✅ Daemon started!");
            println!("   PID: {pid}");
            println!("   Address: ws://{host}:{port}/ws");
            println!("   Logs: {}", log_file.display());
            println!("\n   Use 'nanna daemon status' to check status");
            println!("   Use 'nanna daemon stop' to stop the daemon");
        }
        DaemonAction::Stop => {
            println!("🌙 Stopping Nanna daemon...\n");
            stop_daemon_process(&pid_file)?;
        }
        DaemonAction::Status => {
            print_daemon_status(&pid_file).await?;
        }
        DaemonAction::Restart { host, port } => {
            println!("🌙 Restarting Nanna daemon...\n");
            if pid_file.state()?.live_daemon().is_some() {
                println!("Stopping current daemon...");
                stop_daemon_process(&pid_file)?;
                await_stopped(
                    || Ok(pid_file.state()?.live_daemon()),
                    RESTART_STOP_DEADLINE,
                    RESTART_POLL_STEP,
                )
                .await?;
            }
            println!("Starting daemon...");
            if let Some(pid) = pid_file.state()?.live_daemon() {
                // Another daemon claimed the role while this one stopped: the
                // restart did not happen, so it must not exit 0.
                anyhow::bail!("another daemon started in the meantime (PID {pid}); not restarted");
            }
            let (pid, _) = spawn_daemon_process(&host, port, &data_dir, config_path)?;
            println!("✅ Daemon restarted!");
            println!("   PID: {pid}");
            println!("   Address: ws://{host}:{port}/ws");
        }
    }

    Ok(())
}

/// How long `restart` waits for the stopped daemon to exit before it gives up.
///
/// A SIGTERM'd daemon drains before it exits: in-flight turns are cancelled,
/// MCP servers get [`nanna_daemon::mcp_startup::MCP_SHUTDOWN_DEADLINE`] (3 s)
/// to close, and the store is flushed. That takes seconds, not the 500 ms
/// `restart` used to sleep — after which it found the old daemon still
/// running, printed "already running", started nothing and exited 0, leaving
/// no daemon once the old one finished. Ten times the one bounded component
/// leaves room for the flush on a loaded disk; a daemon still alive past that
/// is wedged, and the command fails saying so instead of claiming success.
const RESTART_STOP_DEADLINE: std::time::Duration = std::time::Duration::from_secs(30);

/// How often `restart` re-reads the PID file while it waits.
const RESTART_POLL_STEP: std::time::Duration = std::time::Duration::from_millis(100);

/// Wait until `live` (the PID file's live daemon) reports none, re-reading it
/// every `step` for at most `deadline`.
///
/// # Errors
///
/// When `live` fails, or a daemon is still alive at the deadline.
async fn await_stopped(
    mut live: impl FnMut() -> anyhow::Result<Option<u32>>,
    deadline: std::time::Duration,
    step: std::time::Duration,
) -> anyhow::Result<()> {
    assert!(!step.is_zero(), "a zero step would spin");
    assert!(step <= deadline, "the deadline allows at least one re-read");
    let until = tokio::time::Instant::now() + deadline;
    loop {
        let Some(pid) = live()? else {
            return Ok(());
        };
        if tokio::time::Instant::now() >= until {
            anyhow::bail!(
                "the daemon (PID {pid}) is still running {}s after it was told to stop; \
                 not restarted — run 'nanna daemon start' once it exits",
                deadline.as_secs()
            );
        }
        tokio::time::sleep(step).await;
    }
}

/// Spawn a daemon process in the background. Returns (PID, log file path).
///
/// The child claims the PID file itself, under the daemon's lock, once it
/// starts — or is refused if another daemon won the role in the meantime.
/// Its output goes to `daemon.log` in `data_dir`, beside the store it opens.
fn spawn_daemon_process(
    host: &str,
    port: u16,
    data_dir: &Path,
    config_path: Option<&Path>,
) -> anyhow::Result<(u32, PathBuf)> {
    use std::fs;
    use std::process::{Command, Stdio};

    let exe = std::env::current_exe()?;
    fs::create_dir_all(data_dir)?;
    let log_file = data_dir.join("daemon.log");
    let config_file = daemon_config_file(config_path)?;

    let log_handle = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_file)?;

    info!("Spawning daemon process...");

    let mut command = Command::new(exe);
    command
        .args(daemon_args(host, port))
        .stdout(Stdio::from(log_handle.try_clone()?))
        .stderr(Stdio::from(log_handle));
    if let Some(file) = config_file {
        command.env(Config::CONFIG_PATH_ENV, file);
    }
    // CREATE_NO_WINDOW: a console-less background process.
    #[cfg(windows)]
    command.creation_flags(0x0800_0000);
    let child = command.spawn()?;

    Ok((child.id(), log_file))
}

/// The command line (after the program) of the daemon `start` launches.
#[must_use]
pub fn daemon_args(host: &str, port: u16) -> Vec<OsString> {
    vec![
        DAEMON_MODE_FLAG.into(),
        "--host".into(),
        host.into(),
        "--port".into(),
        port.to_string().into(),
    ]
}

/// The config file the daemon `start` launches must run on, as its
/// `NANNA_CONFIG_PATH`: the `--config` file this command read, made absolute
/// so the daemon reads the same file wherever it resolves relative paths
/// from. `None` when no file was named: the daemon then inherits this
/// command's environment and finds the same default file.
///
/// Not a `--config` argument. The daemon reads its config file in more than
/// one place (the builder, and the control plane that serves and saves
/// Settings), and only the variable reaches all of them.
///
/// # Errors
///
/// When a relative `config_path` cannot be made absolute (no current
/// directory).
fn daemon_config_file(config_path: Option<&Path>) -> std::io::Result<Option<PathBuf>> {
    config_path.map(std::path::absolute).transpose()
}

/// Stop the daemon the PID file names — only a process the daemon's own probe
/// calls a daemon: a PID since reused by another program is never signalled.
fn stop_daemon_process(pid_file: &PidFile) -> anyhow::Result<()> {
    let state = pid_file.state()?;
    let Some(pid) = state.live_daemon() else {
        println!("❌ No running daemon found");
        match stale_note(&state) {
            Some(note) => println!("   ({note})"),
            None => println!("   No PID file at {}", pid_file.path().display()),
        }
        return Ok(());
    };

    #[cfg(windows)]
    {
        use std::process::Command;
        // /T kills the daemon's whole tree, not just the daemon: without it,
        // in-flight exec children (powershell.exe/bash.exe) survived every
        // `nanna daemon stop`/`restart` and accumulated across dev cycles.
        // A current daemon's kill-on-close Job Object reaps them anyway;
        // /T is belt-and-braces for daemons where adoption failed.
        let status = Command::new("taskkill")
            .args(["/T", "/F", "/PID", &pid.to_string()])
            .status()?;

        if status.success() {
            println!("✅ Daemon stopped (PID {pid})");
        } else {
            println!("⚠️  Failed to stop daemon (PID {pid})");
            println!("   It may have already been terminated");
        }
    }

    #[cfg(not(windows))]
    {
        // A PID file value beyond `pid_t` is no process this command started;
        // wrapping it negative would signal a whole process group (-1: every
        // process we may signal), so it is reported instead of sent.
        match libc::pid_t::try_from(pid) {
            // SAFETY: kill(2) is safe to call with a valid PID and signal number
            Ok(target) if unsafe { libc::kill(target, libc::SIGTERM) } == 0 => {
                println!("✅ Daemon stopped (PID {pid})");
            }
            Ok(_) => {
                let err = std::io::Error::last_os_error();
                println!("⚠️  Failed to stop daemon (PID {pid}): {err}");
                println!("   It may have already been terminated");
            }
            Err(_) => {
                println!("⚠️  Failed to stop daemon (PID {pid}): not a valid process id");
                println!("   It may have already been terminated");
            }
        }
    }

    // The PID file is the daemon's to remove on its way out. One left behind
    // by a hard kill records a dead PID, which the next start takes over.
    Ok(())
}

/// Print daemon status information.
async fn print_daemon_status(pid_file: &PidFile) -> anyhow::Result<()> {
    use nanna_client::{Client, ClientConfig};

    println!("🌙 Nanna Daemon Status\n");

    let state = pid_file.state()?;
    let Some(pid) = state.live_daemon() else {
        match stale_note(&state) {
            Some(note) => println!("   Status: ❌ Not running ({note})"),
            None => println!("   Status: Not running"),
        }
        println!("   Start with: nanna daemon start");
        return Ok(());
    };

    println!("   Status: ✅ Running");
    println!("   PID: {pid}");
    if matches!(state, PidFileState::Recorded(_, ProcessProbe::Unknown)) {
        println!("   (alive, but its program could not be identified — treated as the daemon)");
    }

    // Same constant `daemon start` binds, so status can never probe a different port
    // than the one the daemon was launched on — which is exactly what used to happen.
    let address = format!("ws://{LOOPBACK_HOST}:{DEFAULT_IPC_PORT}");
    let client_config = ClientConfig::new(&address);
    if let Ok(Ok(_)) = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        Client::connect(client_config)
    ).await {
        println!("   Connection: ✅ Healthy");
        println!("   Address: {address}");
    } else {
        println!("   Connection: ⚠️  Not responding");
        println!("   (Daemon may be starting up or misconfigured)");
    }
    Ok(())
}

/// Why a PID file that names no live daemon is stale; `None` when there is no
/// file, or when it does name a live daemon. A stale file is left in place:
/// the next daemon to start takes it over under the daemon's lock.
fn stale_note(state: &PidFileState) -> Option<String> {
    match state {
        PidFileState::Unparseable(content) => {
            Some(format!("stale PID file: it holds no PID ({content:?})"))
        }
        PidFileState::OwnPid(pid) | PidFileState::Recorded(pid, ProcessProbe::Dead) => {
            Some(format!("stale PID file: process {pid} has exited"))
        }
        PidFileState::Recorded(pid, ProcessProbe::Other) => {
            Some(format!("stale PID file: PID {pid} now belongs to another program"))
        }
        PidFileState::Absent
        | PidFileState::Recorded(_, ProcessProbe::Daemon | ProcessProbe::Unknown) => None,
    }
}

#[cfg(test)]
mod tests {

    /// A daemon that takes a while to drain is waited for, not raced: the
    /// restart starts its successor only once the old one is gone.
    #[tokio::test]
    async fn restart_waits_out_a_slow_drain() {
        let mut reads = 0_u32;
        let waited = await_stopped(
            || {
                reads += 1;
                Ok((reads < 40).then_some(7))
            },
            Duration::from_secs(30),
            Duration::from_millis(1),
        )
        .await;
        assert!(waited.is_ok(), "{waited:?}");
        assert_eq!(reads, 40, "re-read until the PID file named no live daemon");
    }

    /// A daemon still alive at the deadline fails the restart; it never
    /// reports success with nothing started.
    #[tokio::test]
    async fn restart_fails_when_the_old_daemon_never_exits() {
        let waited = await_stopped(
            || Ok(Some(7)),
            Duration::from_millis(20),
            Duration::from_millis(1),
        )
        .await;
        let message = waited.expect_err("a wedged daemon is an error").to_string();
        assert!(message.contains("PID 7"), "{message}");
        assert!(message.contains("still running"), "{message}");
        assert!(message.contains("not restarted"), "{message}");
    }
    use super::*;
    use std::time::Duration;

    /// A relocated install's daemon keeps its PID file under `[general]
    /// data_dir`. `status` and `stop` must read it there, not in the platform
    /// default, where they used to look.
    #[test]
    fn the_commands_read_the_pid_file_under_the_configured_data_dir() {
        let relocated = std::env::temp_dir().join("nanna-relocated-install");
        let mut config = Config::default();
        config.general.data_dir = Some(relocated.clone());

        let data_dir = daemon_data_dir(&config).expect("a configured data dir resolves");
        assert_eq!(data_dir, relocated);
        assert_eq!(
            PidFile::new(&data_dir).path(),
            relocated.join("nanna-daemon.pid").as_path()
        );
    }

    /// `nanna --config <file> daemon start` runs the daemon on that same
    /// file, made absolute, so the daemon resolves the data dir the commands
    /// did. Without `--config` the daemon is given nothing and finds the
    /// default file.
    #[test]
    fn the_daemon_is_started_on_the_config_file_the_command_read() {
        assert_eq!(daemon_config_file(None).unwrap(), None);

        let relative = Path::new("configs").join("nanna.toml");
        let file = daemon_config_file(Some(&relative))
            .unwrap()
            .expect("a named file is passed on");
        assert!(file.is_absolute(), "{file:?}");
        assert_eq!(file, std::env::current_dir().unwrap().join(&relative));
    }

    /// Daemon mode, host and port: the probe identifies the daemon by the
    /// first, and `start` reports the other two as its address.
    #[test]
    fn the_daemon_is_started_in_daemon_mode_on_the_requested_address() {
        assert_eq!(
            daemon_args("127.0.0.1", 5149),
            ["--daemon-mode", "--host", "127.0.0.1", "--port", "5149"]
        );
    }
}
