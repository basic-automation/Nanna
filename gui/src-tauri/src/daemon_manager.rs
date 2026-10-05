//! Daemon Manager - Manages the nanna-daemon sidecar lifecycle
//!
//! Responsibilities:
//! - Start daemon on app boot
//! - Monitor daemon health
//! - Restart on crash
//! - Stop on app exit

use serde::Serialize;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;
use tauri::{AppHandle, Runtime};
use tauri_plugin_shell::ShellExt;
use tokio::io::{AsyncBufReadExt, AsyncRead, BufReader};
use tokio::process::Child;
use tokio::sync::{Mutex, RwLock, mpsc, oneshot};
use tokio::time::sleep;
use tracing::{debug, error, info, warn};

/// Daemon process state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DaemonState {
    Stopped,
    Starting,
    Running,
    Stopping,
    Crashed,
}

/// Daemon manager configuration
#[derive(Debug, Clone)]
pub struct DaemonManagerConfig {
    /// Port for the daemon to listen on
    pub port: u16,
    /// Host to bind to
    pub host: String,
    /// Maximum restart attempts before giving up
    pub max_restarts: u32,
    /// Delay between restart attempts
    pub restart_delay: Duration,
    /// Health check interval
    pub health_check_interval: Duration,
    /// How long one health check may take to connect and complete the
    /// handshake.
    pub health_check_timeout: Duration,
}

/// The daemon port `NANNA_DAEMON_PORT` asks for, if it names a usable one.
///
/// For isolated runs (automated GUI verification above all): a GUI on the
/// default port attaches to whatever daemon already listens there — on a
/// developer's machine, their own, with their real data. Pure.
#[must_use]
pub fn port_override(value: Option<&str>) -> Option<u16> {
    value?
        .trim()
        .parse::<u16>()
        .ok()
        .filter(|port| *port >= 1024)
}

#[cfg(test)]
mod port_override_tests {
    use super::port_override;

    #[test]
    fn only_a_usable_port_overrides_the_default() {
        assert_eq!(port_override(Some("51997")), Some(51997));
        assert_eq!(port_override(Some(" 6000 ")), Some(6000));
        assert_eq!(port_override(None), None);
        assert_eq!(port_override(Some("")), None);
        assert_eq!(
            port_override(Some("80")),
            None,
            "privileged ports are not a daemon's"
        );
        assert_eq!(port_override(Some("70000")), None);
        assert_eq!(port_override(Some("fifty")), None);
    }
}

impl Default for DaemonManagerConfig {
    fn default() -> Self {
        Self {
            port: 5149,
            host: "127.0.0.1".to_string(),
            max_restarts: 3,
            restart_delay: Duration::from_secs(2),
            health_check_interval: Duration::from_secs(30),
            // The ceiling a version probe gets for a connect plus a request;
            // a check only connects.
            health_check_timeout: PROBE_TIMEOUT,
        }
    }
}

/// When a boot counts as slow enough to log. Readiness normally takes a few
/// seconds (claim the role, open storage, discover tools). Later notices come
/// at double the previous elapsed time, so even a boot that takes hours logs
/// only a handful of lines. The status footer switches to "Still starting"
/// at the same point (`backendLabels.ts`). Nothing here stops the wait.
const SLOW_START_NOTICE: Duration = Duration::from_secs(30);

/// Where the wait for a starting sidecar stands after one poll of its port.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BootWait {
    /// A daemon answered.
    Ready,
    /// Nobody answered yet and the sidecar is alive: it is still booting.
    KeepWaiting,
    /// Nobody answered and the sidecar is gone: the start failed.
    Failed,
    /// `stop()` took over while the wait was running.
    Cancelled,
}

/// What became of a ready-wait's answer (see [`DaemonManager::mark_ready`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Marked {
    /// The daemon is `Running`.
    Running,
    /// A stop, or a newer start, took over first.
    Superseded,
    /// The sidecar exited after the probe began. Probe again, now that the
    /// exit is known.
    ExitedSinceProbe,
}

/// The ready-wait's verdict after one poll of the daemon port. `answered`
/// says whether a daemon accepted the probe, and `sidecar_exited` whether the
/// sidecar had already exited when the probe began.
///
/// A live sidecar is still booting, however long that takes. The daemon
/// claims its PID file and reserves the port before it does anything slow,
/// and it opens the port only when it is ready. So a live process with a
/// closed port is a boot in progress. A cold model load, a big migration or a
/// slow provider can take minutes. The old 90 s timeout killed such a boot,
/// and every relaunch then did the same thing (2026-09-18).
///
/// A sidecar that has exited is a failure. That includes the `AlreadyRunning`
/// exit, where another daemon already holds the role. The poll that sees the
/// exit still probes the port, so a daemon that is already up gets attached.
/// A daemon that is still booting is not ours to wait for: the client's retry
/// loop attaches it when its port opens.
///
/// A stop wins over everything else. The wait must not leave behind a
/// `Running` or `Crashed` state after `stop()` has already reported `Stopped`.
const fn boot_wait_verdict(state: DaemonState, sidecar_exited: bool, answered: bool) -> BootWait {
    if matches!(state, DaemonState::Stopping | DaemonState::Stopped) {
        BootWait::Cancelled
    } else if answered {
        BootWait::Ready
    } else if sidecar_exited {
        BootWait::Failed
    } else {
        BootWait::KeepWaiting
    }
}

/// Kill the daemon sidecar's process tree by PID, best-effort. A current
/// daemon's own kill-on-close Job Object (adopted at startup) already reaps
/// its children when it dies; this is belt-and-braces for daemons where
/// adoption failed or that predate it.
///
/// Windows delegates to the shared `taskkill /T` walk in `nanna_proc` (which
/// suppresses the console window a windows-subsystem process like this one
/// would otherwise flash). Unix is a deliberate no-op — the sidecar is not
/// spawned as a process-group leader, so the process_group(0)-at-spawn
/// contract behind `nanna_proc`'s group kill does not hold here; the
/// caller's [`Sidecar::kill`] kills the direct child.
#[cfg(windows)]
async fn kill_sidecar_tree(pid: u32) {
    nanna_proc::kill_process_tree(pid).await;
}

/// The Unix no-op. It returns a ready future rather than being an `async fn`
/// with nothing to await, so callers `.await` both platforms the same way.
#[cfg(not(windows))]
fn kill_sidecar_tree(_pid: u32) -> std::future::Ready<()> {
    std::future::ready(())
}

/// Request id of the version probe; the reply is matched on it.
const VERSION_PROBE_ID: &str = "daemon-manager-version";

/// End-to-end ceiling on one version probe: connect, ask, read the reply.
/// The same 2 s the graceful-shutdown connect gets, plus 1 s for the reply —
/// `system version` is a constant built into the daemon binary, with no I/O.
const PROBE_TIMEOUT: Duration = Duration::from_secs(3);

/// How long an evicted daemon gets to release the port. A graceful stop
/// cancels any in-flight turn and saves stats before exiting; `stop()` allows
/// 5 s for the same thing on a daemon it owns and then tree-kills it. We did
/// not spawn this one and cannot kill it, so it gets twice that.
const EVICTION_TIMEOUT: Duration = Duration::from_secs(10);

/// What answered on the daemon port.
#[derive(Debug, Clone, PartialEq, Eq)]
enum PortOccupant {
    /// Nothing accepted a connection.
    Nobody,
    /// A daemon accepted. `version` is `None` when it did not report one in
    /// time — still mid-init, or too old to know the request.
    Daemon { version: Option<String> },
}

/// The occupant's version when it is a daemon from a *different* release.
/// An occupant that did not report a version is never called stale.
fn stale_version<'a>(occupant: &'a PortOccupant, ours: &str) -> Option<&'a str> {
    match occupant {
        PortOccupant::Daemon { version: Some(theirs) } if theirs != ours => Some(theirs),
        _ => None,
    }
}

/// The daemon's answer to the version probe.
#[derive(Debug, PartialEq, Eq)]
struct VersionReply {
    /// `None` when the reply carried no version (an error response).
    version: Option<String>,
}

/// Parse one IPC frame; `None` unless it is the reply to the probe.
fn version_from_reply(text: &str) -> Option<VersionReply> {
    let value: serde_json::Value = serde_json::from_str(text).ok()?;
    if value.get("id").and_then(serde_json::Value::as_str) != Some(VERSION_PROBE_ID) {
        return None;
    }
    let version = value
        .pointer("/result/data/version")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string);
    Some(VersionReply { version })
}

/// Connect to `url` and ask whatever is there for its version.
async fn probe_occupant(url: &str) -> PortOccupant {
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::Message;

    let deadline = tokio::time::Instant::now() + PROBE_TIMEOUT;
    let Ok(Ok((mut ws, _))) =
        tokio::time::timeout_at(deadline, tokio_tungstenite::connect_async(url)).await
    else {
        return PortOccupant::Nobody;
    };
    let request = serde_json::json!({
        "id": VERSION_PROBE_ID,
        "action": { "type": "system", "action": "version" }
    });
    let mut version = None;
    if ws.send(Message::Text(request.to_string().into())).await.is_ok() {
        // Unsolicited event frames may arrive first; skip until the reply.
        while let Ok(Some(Ok(frame))) = tokio::time::timeout_at(deadline, ws.next()).await {
            if let Message::Text(text) = frame
                && let Some(reply) = version_from_reply(&text)
            {
                version = reply.version;
                break;
            }
        }
    }
    let _ = ws.close(None).await;
    PortOccupant::Daemon { version }
}

/// Why a health check failed.
#[derive(Debug, Clone, PartialEq, Eq)]
enum CheckFailure {
    /// The connection did not complete in time. Something holds the port (on
    /// loopback a connection nobody listens for is refused at once) but does
    /// not answer: a wedged or starved daemon, or another program.
    NoAnswer(Duration),
    /// The connection failed: nothing listens on the port, or what listens is
    /// not a daemon.
    Failed(String),
}

impl std::fmt::Display for CheckFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoAnswer(timeout) => write!(f, "no answer within {timeout:?}"),
            Self::Failed(e) => f.write_str(e),
        }
    }
}

/// One health check: complete a WebSocket handshake with the daemon, then
/// close it again.
///
/// Bounded by `timeout` end to end. An unbounded check hung for good on a
/// port that accepts the TCP connection but never completes the handshake,
/// and the monitor it runs in stopped acting with the last state frozen.
async fn check_health(url: &str, timeout: Duration) -> Result<(), CheckFailure> {
    let check = async {
        let (mut ws, _) = tokio_tungstenite::connect_async(url)
            .await
            .map_err(|e| CheckFailure::Failed(e.to_string()))?;
        let _ = futures_util::SinkExt::close(&mut ws).await;
        Ok(())
    };
    tokio::time::timeout(timeout, check)
        .await
        .unwrap_or(Err(CheckFailure::NoAnswer(timeout)))
}

/// What the health monitor does about a failed check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AfterFailedCheck {
    /// Something holds the port but did not answer in time. It may only be
    /// slow: nothing is recorded, and the next tick checks again.
    CheckAgain,
    /// The daemon we started, process `pid`, is alive but refuses
    /// connections: `Crashed`, with the reason on record, and no restart.
    Report { pid: u32 },
    /// The daemon is gone: restart it.
    Restart,
}

/// What the health monitor does about a failed check. `sidecar` is the PID of
/// the sidecar we spawned while it is still running.
///
/// Only a daemon that is gone is restarted. A new daemon cannot start beside
/// one that is still there: while our sidecar lives it holds the daemon's PID
/// file, so a new one exits "Already running", and a port that took the
/// connection without answering is held, so a new one cannot bind it.
/// Replacing that daemon would mean killing it, and a daemon slow to answer
/// (a busy disk can starve it for seconds) is not killed on a timeout, just
/// as a slow boot is not (see [`boot_wait_verdict`]). A daemon that is alive
/// but not serving is replaced when someone asks for a restart or a retry.
///
/// The bound on the check made this matter. With it, a daemon that took more
/// than 3 s to answer was "restarted": the new sidecar took the old one's
/// place in the child slot, and the old daemon ran on with nothing left to
/// stop it.
const fn after_failed_check(failure: &CheckFailure, sidecar: Option<u32>) -> AfterFailedCheck {
    match (failure, sidecar) {
        (CheckFailure::NoAnswer(_), _) => AfterFailedCheck::CheckAgain,
        (CheckFailure::Failed(_), Some(pid)) => AfterFailedCheck::Report { pid },
        (CheckFailure::Failed(_), None) => AfterFailedCheck::Restart,
    }
}

/// What the health monitor makes of a check that found the daemon gone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FailedCheck {
    /// A stop, or someone else's start, took over while the check ran.
    NotOurs,
    /// Out of restarts: nothing more is tried.
    GaveUp,
    /// Restart; this is restart number `n` since a daemon was last ready.
    Restart(u32),
}

// =============================================================================
// Why a start failed
// =============================================================================

/// Why the daemon failed to start or to stay up.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StartFailureKind {
    /// The bundled `nanna-daemon` program could not be located.
    SidecarUnresolved,
    /// The program was located, but its process did not start.
    SpawnFailed,
    /// The sidecar exited before any daemon answered on the port.
    ExitedDuringBoot,
    /// The sidecar that was serving the app exited.
    ExitedAfterReady,
    /// The daemon stopped answering the health check.
    HealthCheckFailed,
    /// The health monitor used up its restarts and no longer tries.
    RestartsExhausted,
}

/// The most recent reason the daemon failed to start or to stay up, as
/// `BackendStatus::last_error` reports it. Cleared when a daemon becomes
/// ready.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StartFailure {
    pub kind: StartFailureKind,
    /// What went wrong, for a person to read. For an exit it is the daemon's
    /// own last error line when it printed one (see `exit_reason`), else
    /// the exit code or signal.
    pub message: String,
    pub exit_code: Option<i32>,
    pub signal: Option<i32>,
    /// When it happened, in milliseconds since the Unix epoch.
    pub at_ms: i64,
}

impl StartFailure {
    fn new(kind: StartFailureKind, message: String) -> Self {
        Self {
            kind,
            message,
            exit_code: None,
            signal: None,
            at_ms: chrono::Utc::now().timestamp_millis(),
        }
    }

    /// The failure a sidecar's exit amounts to.
    ///
    /// The exit code cannot say why: every error the daemon's `run` returns
    /// exits with 1, whether another daemon holds the role, another program
    /// holds the port, or the config does not load. The daemon says why in
    /// its last log line, so that line is the message whenever there is one.
    /// A boot that stopped at its command line never logged, and what it
    /// printed instead is the message.
    fn from_exit(kind: StartFailureKind, exit: &SidecarExit) -> Self {
        let during_boot = kind == StartFailureKind::ExitedDuringBoot;
        let own_words = exit
            .reason
            .clone()
            .or_else(|| exit.command_line_error.clone().filter(|_| during_boot));
        let message = own_words.unwrap_or_else(|| {
            let what = if during_boot {
                "The daemon exited during startup"
            } else {
                "The daemon exited"
            };
            match (exit.code, exit.signal) {
                (Some(code), _) => format!("{what} (exit code {code})"),
                (None, Some(signal)) => format!("{what} (signal {signal})"),
                (None, None) => what.to_string(),
            }
        });
        Self {
            exit_code: exit.code,
            signal: exit.signal,
            ..Self::new(kind, message)
        }
    }

    /// The monitor has given up. The failure before it stays in the message
    /// (and its exit details in the fields): giving up is the news, but the
    /// reason the restarts failed is what a person can act on.
    fn restarts_exhausted(max_restarts: u32, previous: &Self) -> Self {
        Self {
            exit_code: previous.exit_code,
            signal: previous.signal,
            ..Self::new(
                StartFailureKind::RestartsExhausted,
                format!(
                    "Stopped restarting the daemon after {max_restarts} failed restarts · {}",
                    previous.message
                ),
            )
        }
    }
}

// =============================================================================
// Boot log: the output of the current sidecar
// =============================================================================

/// The output lines of the longest boot on record, from "Starting Nanna
/// daemon…" to "Daemon ready", both counted: 450, on 2026-09-18 on the
/// operator's machine (tool-registry and skill lines, and
/// embedding-congestion warnings from a busy provider). The 14 boots in that
/// week's daemon logs (2026-09-14..18) that reached ready printed 197 to 450.
/// A boot with an empty data dir and a scratch config prints 270 (measured
/// the same day). The file log and stdout share one filter, so the file
/// counts are what the sidecar prints.
const LONGEST_MEASURED_BOOT_LINES: usize = 450;

/// How many output lines of one sidecar the boot log keeps: twice the
/// longest boot on record. Any boot that reaches ready is then held whole,
/// from its first line, with room for a setup with twice the tools and
/// skills. A boot that hangs keeps logging (the 2026-09-18 hang printed 5240
/// lines before it was restarted); the log then keeps the newest lines, which
/// hold the current phase and any fatal error. The same happens to a boot
/// logged below `info`: the sidecar inherits the app's `RUST_LOG`. The
/// measurement counts the daemon's own log only. The stdio MCP servers it
/// starts write to its stderr, and so into this log, on top of that: a
/// chatty one takes some of the headroom.
const BOOT_LOG_LINES: usize = 2 * LONGEST_MEASURED_BOOT_LINES;

/// The longest line the boot log keeps whole, in bytes. The longest of the
/// ~35,000 lines in the operator's daemon logs for 2026-09-14..18 is 1,104
/// bytes, so a real line fits nearly four times over. The cap bounds the log
/// at about [`BOOT_LOG_LINES`] × 4 KiB (3.7 MB) whatever the daemon prints.
const BOOT_LOG_LINE_BYTES: usize = 4096;

/// Which output stream of the sidecar a line came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum BootStream {
    Stdout,
    Stderr,
}

/// One output line of the sidecar, as `get_boot_log` returns it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BootLine {
    pub stream: BootStream,
    /// Plain text: no terminal escapes, no line ending.
    pub line: String,
}

/// The newest output lines of one sidecar, at most [`BOOT_LOG_LINES`].
#[derive(Debug, Default)]
struct BootLog {
    lines: VecDeque<BootLine>,
}

impl BootLog {
    fn push(&mut self, line: BootLine) {
        if self.lines.len() >= BOOT_LOG_LINES {
            self.lines.pop_front();
        }
        self.lines.push_back(line);
    }
}

/// `raw` as plain text: ANSI escape sequences removed, and every other
/// control character except tab (the line ending among them).
///
/// The sidecar is spawned with `NO_COLOR`, so the daemon's log should carry
/// no escapes. This keeps them out anyway: an older daemon, or a library
/// that writes to the terminal itself, shows up in the splash as text.
fn plain_text(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut chars = raw.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\u{1b}' => match chars.next() {
                // CSI (colours, cursor moves): parameter and intermediate
                // bytes, then one final byte in `@`..=`~`.
                Some('[') => {
                    for c in chars.by_ref() {
                        if ('@'..='~').contains(&c) {
                            break;
                        }
                    }
                }
                // OSC (titles, links): up to BEL or the string terminator
                // `ESC \`.
                Some(']') => {
                    while let Some(c) = chars.next() {
                        if c == '\u{7}' {
                            break;
                        }
                        if c == '\u{1b}' {
                            chars.next_if_eq(&'\\');
                            break;
                        }
                    }
                }
                // nF escapes (`ESC ( B` selects a character set):
                // intermediate bytes, then one final byte.
                Some(' '..='/') => {
                    while chars.next_if(|c| (' '..='/').contains(c)).is_some() {}
                    chars.next();
                }
                // Any other escape is ESC and one more character.
                _ => {}
            },
            '\t' => out.push(c),
            c if c.is_control() => {}
            c => out.push(c),
        }
    }
    out
}

/// `line`, cut to [`BOOT_LOG_LINE_BYTES`] on a character boundary. A cut
/// line says so and how much it lost.
fn fit_line(mut line: String) -> String {
    use std::fmt::Write as _;

    if line.len() <= BOOT_LOG_LINE_BYTES {
        return line;
    }
    let mut end = BOOT_LOG_LINE_BYTES;
    while !line.is_char_boundary(end) {
        end -= 1;
    }
    let cut = line.len() - end;
    line.truncate(end);
    let _ = write!(line, "… ({cut} more bytes)");
    line
}

/// The next whitespace-separated token of `s`, and what follows it.
fn next_token(s: &str) -> (&str, &str) {
    let s = s.trim_start();
    s.split_once(char::is_whitespace).unwrap_or((s, ""))
}

/// Whether `s` is a `tracing` target: a Rust module path.
fn is_log_target(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == ':')
}

/// The message of a daemon log line at ERROR level, or `None` for any other
/// line.
///
/// The daemon logs to stdout in `tracing`'s default format,
/// `<timestamp> <LEVEL> <target>: <message>`, e.g.
/// `2026-09-18T17:21:44.209179Z ERROR nanna_daemon: Error: Already running`.
/// The message is what follows the target: `Error: Already running`.
fn error_message(line: &str) -> Option<&str> {
    let (first, rest) = next_token(line);
    let rest = if first == "ERROR" {
        rest
    } else {
        let (second, rest) = next_token(rest);
        if second != "ERROR" {
            return None;
        }
        rest
    };
    let rest = rest.trim();
    let message = match rest.split_once(": ") {
        Some((target, message)) if is_log_target(target) => message,
        _ => rest,
    };
    (!message.is_empty()).then_some(message)
}

/// Why the sidecar exited, in its own words: the last thing it said, when
/// that was an error. `None` when its output ends in anything else, for
/// example a process killed mid-work by a signal.
///
/// The daemon reports every fatal error through its log: `run` failing
/// logs `Error: <what>` at ERROR level and exits 1 (nanna-daemon `main.rs`),
/// and its panic hook logs `PANIC: …`. So the last stdout line decides.
///
/// stderr is not the daemon's alone. Every stdio MCP server it starts
/// writes there too, and one may go on writing after the daemon died: its
/// chatter was taken for the reason a killed daemon exited. So stderr is
/// the reason only when it holds Rust's own fatal output (see
/// [`runtime_fatal`]) after the last stdout line, or anywhere when there is
/// no stdout line. A release daemon aborts on a panic, possibly before its
/// log line is written. A process that stopped at its command line says why
/// on stderr too, but that is read only for a boot (see
/// [`stopped_at_command_line`]).
///
/// The two streams are read separately, so their order is only
/// approximate; the daemon's own ERROR line therefore wins over stderr.
/// Rust's `note: run with RUST_BACKTRACE=1 …` hint is never the reason.
fn exit_reason(lines: &VecDeque<BootLine>) -> Option<String> {
    let Some(last_stdout) = lines.iter().rposition(|line| line.stream == BootStream::Stdout) else {
        return runtime_fatal(lines.iter().map(|line| line.line.as_str()));
    };
    if let Some(message) = error_message(&lines[last_stdout].line) {
        return Some(message.to_string());
    }
    runtime_fatal(
        lines
            .iter()
            .skip(last_stdout + 1)
            .filter(|line| line.stream == BootStream::Stderr)
            .map(|line| line.line.as_str()),
    )
}

/// What stopped a process that got no further than its command line: the
/// first line it printed, Rust's `note: run with RUST_BACKTRACE=1 …` hint
/// aside. That is a loader error, or an argument error followed by usage
/// lines. Such a process started nothing, so its stderr is its own.
///
/// `None` unless the process looks like one:
/// - it wrote no stdout line. The daemon's log is its stdout, and its first
///   line comes before it starts anything;
/// - it exited by itself, with `code`. A signal came from outside (a stop, a
///   restart, the OOM killer), so nothing it printed says why it ended. The
///   one it raises itself, an abort, comes with Rust's fatal output, which
///   [`exit_reason`] reads;
/// - it exited during its boot, which [`StartFailure::from_exit`] checks. A
///   daemon that became ready got past its command line.
///
/// The first condition alone was not enough. The sidecar inherits the app's
/// `RUST_LOG`, which can filter out the daemon's whole log, and the stdio
/// MCP servers the daemon starts write to its stderr. A daemon killed after
/// it started them was said to have exited because "Secure MCP Filesystem
/// Server running on stdio". One case still gets through: a daemon whose
/// log is filtered out, that starts its servers and then fails its boot with
/// an exit code. The boot log shows what happened there.
fn stopped_at_command_line(lines: &VecDeque<BootLine>, code: Option<i32>) -> Option<String> {
    if code.is_none() || lines.iter().any(|line| line.stream == BootStream::Stdout) {
        return None;
    }
    lines
        .iter()
        .find(|line| !line.line.starts_with("note: "))
        .map(|line| line.line.clone())
}

/// The first fatal error that Rust's runtime printed among `stderr` lines,
/// whatever the program's own log did:
/// - a panic: `thread 'main' panicked at src/x.rs:1:2:`, with the message
///   on the next line (older toolchains print both on one line);
/// - `thread 'main' has overflowed its stack`;
/// - `memory allocation of 1024 bytes failed`;
/// - any other `fatal runtime error: …`.
fn runtime_fatal<'a>(mut stderr: impl Iterator<Item = &'a str>) -> Option<String> {
    while let Some(line) = stderr.next() {
        let thread = line.starts_with("thread '");
        if thread && line.contains("' panicked at ") {
            return Some(match stderr.next() {
                Some(message) if line.ends_with(':') && !message.starts_with("note: ") => {
                    format!("{line} {message}")
                }
                _ => line.to_string(),
            });
        }
        if (thread && line.ends_with("has overflowed its stack"))
            || line.starts_with("fatal runtime error: ")
            || (line.starts_with("memory allocation of ") && line.ends_with(" failed"))
        {
            return Some(line.to_string());
        }
    }
    None
}

// =============================================================================
// One spawned sidecar
// =============================================================================

/// How a sidecar ended.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct SidecarExit {
    code: Option<i32>,
    signal: Option<i32>,
    /// See [`exit_reason`].
    reason: Option<String>,
    /// See [`stopped_at_command_line`]. A reason only for an exit during a
    /// boot.
    command_line_error: Option<String>,
}

impl SidecarExit {
    /// The exit, with `code` or by `signal`, of a sidecar that printed
    /// `lines`.
    fn new(code: Option<i32>, signal: Option<i32>, lines: &VecDeque<BootLine>) -> Self {
        Self {
            code,
            signal,
            reason: exit_reason(lines),
            command_line_error: stopped_at_command_line(lines, code),
        }
    }
}

/// How long a sidecar's exit waits, once its process is gone, for the
/// output the process wrote before it exited to be read. Only its
/// [`exit_reason`] depends on that output.
///
/// Normally both output streams end with the process, and the exit is
/// recorded at once. A stdio MCP server the daemon started holds a copy of
/// its stderr, and one that ignores stdin EOF keeps it open after the
/// daemon is gone, possibly for good. The exit is then recorded after this
/// long, with the lines read by then.
///
/// What is left to read when a process exits is at most one full pipe per
/// stream: a writer blocks once its pipe is full (64 KiB on Linux). Reading
/// and relaying a full pipe of daemon log lines, written just before the
/// exit, took 2.6 to 2.9 ms (20 runs, debug build, 2026-09-18). A second is
/// over 300 times that, for a machine whose disk and scheduler are
/// saturated. It stays well inside the 5 s a graceful stop allows for the
/// exit, and it delays what waits for an exit (the next boot, a failed
/// boot's report) only when a child holds the output.
const OUTPUT_DRAIN: Duration = Duration::from_secs(1);

/// What one spawned sidecar's event task records, shared with the manager.
/// A new one per spawn, so a sidecar's late output or exit can only ever
/// land in its own record.
#[derive(Default)]
struct SpawnWatch {
    /// The sidecar's process has exited. Distinguishes a live sidecar we own
    /// (stop = graceful shutdown, then tree-kill) from a dead one whose PID
    /// may have been recycled — e.g. the `AlreadyRunning` exit when we merely
    /// attached to a standalone daemon, which is not ours to stop.
    ///
    /// Set when the operating system reports the exit, once the output
    /// written before it has been read (see [`OUTPUT_DRAIN`]).
    exited: AtomicBool,
    /// How it exited. Written before `exited` is set, so whoever sees the
    /// flag finds the details.
    exit: Mutex<Option<SidecarExit>>,
    log: Mutex<BootLog>,
}

impl SpawnWatch {
    /// Log one output line (info level so it is visible in production) and
    /// keep it in the boot log. A blank line says nothing and is dropped.
    async fn relay(&self, stream: BootStream, raw: &[u8]) {
        let line = plain_text(&String::from_utf8_lossy(raw));
        if line.trim().is_empty() {
            return;
        }
        match stream {
            BootStream::Stdout => info!("daemon stdout: {}", line),
            BootStream::Stderr => error!("daemon stderr: {}", line),
        }
        self.log.lock().await.push(BootLine {
            stream,
            line: fit_line(line),
        });
    }
}

/// Relay `pipe`, one of the sidecar's output streams, line by line until it
/// ends.
async fn relay_output(watch: Arc<SpawnWatch>, stream: BootStream, pipe: impl AsyncRead + Unpin) {
    let mut reader = BufReader::new(pipe);
    let mut line = Vec::new();
    loop {
        line.clear();
        match reader.read_until(b'\n', &mut line).await {
            Ok(0) => break,
            Ok(_) => watch.relay(stream, &line).await,
            Err(e) => {
                error!("Could not read the daemon's {stream:?}: {e}");
                break;
            }
        }
    }
}

/// The signal that ended a process.
#[cfg(unix)]
fn exit_signal(status: std::process::ExitStatus) -> Option<i32> {
    use std::os::unix::process::ExitStatusExt;
    status.signal()
}

/// Windows has no signals: a killed process exits with a code.
#[cfg(not(unix))]
const fn exit_signal(_status: std::process::ExitStatus) -> Option<i32> {
    None
}

/// A request to the event task that owns a sidecar's process to kill it.
/// The event task answers once it has sent the kill. It drops the request
/// unanswered once the process has exited, and there is nothing to kill.
type KillRequest = oneshot::Sender<()>;

/// How long [`Sidecar::kill`] waits for the event task's answer. Answering
/// takes one wake-up of that task and one system call: 24 to 76 µs here (20
/// kills on each tokio runtime flavour, debug build, 2026-09-18), less than
/// the 2.6 to 2.9 ms the output relay needs at an exit. So the answer gets
/// the second [`OUTPUT_DRAIN`] gives the relay on a machine whose scheduler
/// is saturated, over 10,000 times the slowest answer measured. The bound is
/// there for a runtime that no longer runs the task at all, so that a quit
/// cannot hang on it.
const KILL_ANSWER: Duration = OUTPUT_DRAIN;

/// The sidecar process in the manager's child slot, with its record. The
/// process itself belongs to its event task, which waits for its exit.
struct Sidecar {
    pid: u32,
    /// Asks the event task to kill the process. It carries at most one
    /// request: a sidecar is taken out of the child slot to be killed.
    kill: mpsc::UnboundedSender<KillRequest>,
    watch: Arc<SpawnWatch>,
}

impl Sidecar {
    /// Kill the process (`SIGKILL` on Unix), and return once the kill has
    /// been sent.
    ///
    /// The event task that owns the process sends it. Asking that task used
    /// to be the whole kill, and a stop returned before the task had run. At
    /// a quit, Tauri then ended the app, the task never ran, and a wedged
    /// daemon outlived the app. A kill that has been sent needs nothing more
    /// from this process: the operating system carries it out. The exit is
    /// not waited for. A process blocked in disk I/O dies only once the I/O
    /// completes, which on a busy disk takes seconds, and the next spawn
    /// waits for it (see [`DaemonManager::previous_sidecar_gone`]).
    ///
    /// A process that has exited already is left alone: its PID may belong
    /// to another by now, and the event task never signals it once reaped.
    async fn kill(&self) {
        let (answer, answered) = oneshot::channel();
        if self.kill.send(answer).is_err() {
            // The event task has seen the exit.
            return;
        }
        // An `Err` inside means the request was dropped unanswered: the
        // process had exited.
        if tokio::time::timeout(KILL_ANSWER, answered).await.is_err() {
            error!(
                "The kill of the daemon process (PID {}) was not confirmed within {KILL_ANSWER:?}: it may outlive the app",
                self.pid
            );
        }
    }
}

/// What a sidecar's exit does to the manager's state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExitEffect {
    /// Nothing changes: a stop caused the exit, or a newer start owns the
    /// state.
    Nothing,
    /// The exit ends a boot. The ready-wait decides what it means: a daemon
    /// our sidecar deferred to may be answering.
    EndsBoot,
    /// The daemon the app was attached to is gone.
    Crash,
    /// A daemon still answers on the port, so the one the app attached to was
    /// never this sidecar: it deferred (the `AlreadyRunning` exit) after the
    /// attach. The app stays attached.
    StayAttached,
    /// The daemon is already down. If the health monitor reported it down
    /// while its process lived (see [`DaemonManager::not_serving`]), the
    /// exit replaces that report, which said it was still running. Any other
    /// record stays (see [`replaces_record`]).
    EndsReport,
}

/// The effect of a sidecar's exit. `current` says whether the sidecar
/// belongs to the current start attempt, and `still_answering` whether a
/// daemon answered on the port after the exit (probed only while `Running`).
///
/// Crashing on every exit while `Running` showed the app "crashed" while it
/// was attached to a healthy daemon, until the next health check 30 s later.
///
/// `Crashed` while the current sidecar lives has one cause: the monitor's
/// report of a daemon that is alive but not serving. Every other crash
/// comes after the sidecar's exit, or with no process at all. That includes
/// one recorded between the exit and this call, by whoever saw the exit
/// first (see [`replaces_record`]).
const fn exit_effect(current: bool, state: DaemonState, still_answering: bool) -> ExitEffect {
    if !current {
        return ExitEffect::Nothing;
    }
    match state {
        DaemonState::Starting => ExitEffect::EndsBoot,
        DaemonState::Running if still_answering => ExitEffect::StayAttached,
        DaemonState::Running => ExitEffect::Crash,
        DaemonState::Crashed => ExitEffect::EndsReport,
        DaemonState::Stopping | DaemonState::Stopped => ExitEffect::Nothing,
    }
}

/// Whether the exit of a daemon that is already down replaces `record`, the
/// failure on record. Only a failed health check is replaced, which the
/// exit explains better: the monitor's report of a live daemon that does not
/// serve, or its report of a refused connection after it saw the exit.
///
/// The event task flags the exit before it reads the state, so whoever sees
/// the flag in between can record the exit first: the ready-wait, as a boot
/// that exited, or the monitor, as it gives up on a daemon that is gone.
/// Taking every `Crashed` for the monitor's report replaced those records
/// with "exited after ready", which a boot that exited never was.
const fn replaces_record(record: Option<&StartFailure>) -> bool {
    matches!(
        record,
        Some(StartFailure {
            kind: StartFailureKind::HealthCheckFailed,
            ..
        })
    )
}

/// Where a sidecar's exit status goes when it dies without saying why: the
/// daemon's exit record (see `nanna_core::exit_record`), which the next boot
/// reads. A daemon killed by a signal no hook of its own sees (the `SIGBUS`
/// of an `AppImage` mount that went away, a `SIGKILL`) used to leave only
/// "it died through a path no hook could see"; the status this app got from
/// waiting on it was logged here and then lost with the app.
struct ExitRecordTarget {
    /// The daemon's data directory, resolved as the daemon resolves it.
    data_dir: std::path::PathBuf,
    pid: u32,
    /// Taken just before the spawn: a record written earlier is not this
    /// process's, whatever its PID says.
    spawned_at: chrono::DateTime<chrono::Utc>,
}

impl ExitRecordTarget {
    /// Add the exit to the record if it is still this process's `running`
    /// marker. Best effort: a failure is logged, and the exit is handled the
    /// same either way.
    async fn note(self, code: Option<i32>, signal: Option<i32>, killed_by_app: bool) {
        use nanna_core::exit_record::{NoteOutcome, ObservedExit, note_observed_exit};
        let observed = ObservedExit {
            code,
            signal,
            observer: "the app".to_string(),
            killed_by_observer: killed_by_app,
            at: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
        };
        let Self { data_dir, pid, spawned_at } = self;
        let noted = tokio::task::spawn_blocking(move || {
            note_observed_exit(&data_dir, pid, spawned_at, observed)
        })
        .await;
        match noted {
            Ok(Ok(NoteOutcome::Noted)) => warn!(
                "The daemon (PID {pid}) ended without recording why: its exit status is now on its exit record"
            ),
            Ok(Ok(outcome)) => debug!("Daemon exit record left as it was: {outcome:?}"),
            Ok(Err(e)) => warn!("Could not add the daemon's exit status to its exit record: {e}"),
            Err(e) => warn!("Noting the daemon's exit status did not finish: {e}"),
        }
    }
}

/// The event task of one spawned sidecar: it relays the output into the log
/// and the boot log, and records the exit.
struct SidecarEvents {
    /// The start attempt that spawned this sidecar.
    attempt: u64,
    watch: Arc<SpawnWatch>,
    state: Arc<RwLock<DaemonState>>,
    current_attempt: Arc<AtomicU64>,
    last_failure: Arc<RwLock<Option<StartFailure>>>,
    url: String,
}

impl SidecarEvents {
    /// Relay `child`'s output, kill it when `kills` asks (see
    /// [`Sidecar::kill`]), and record its exit.
    ///
    /// The exit is the one the operating system reports. It used to be the
    /// shell plugin's, which comes only once both output streams have ended,
    /// and an MCP server holding the daemon's stderr (see [`OUTPUT_DRAIN`])
    /// kept them from ending. A killed or dead daemon then never counted as
    /// exited: the next boot waited for it with no deadline, with nothing
    /// running and nothing retrying, and the health monitor reported the
    /// dead process as still running and never restarted it (the second
    /// splash review, 2026-09-18).
    async fn run(
        self,
        mut child: Child,
        mut kills: mpsc::UnboundedReceiver<KillRequest>,
        record: Option<ExitRecordTarget>,
    ) {
        // Held open until the exit, as the shell plugin held it.
        let _stdin = child.stdin.take();
        let readers = [
            child.stdout.take().map(|pipe| {
                tokio::spawn(relay_output(Arc::clone(&self.watch), BootStream::Stdout, pipe))
            }),
            child.stderr.take().map(|pipe| {
                tokio::spawn(relay_output(Arc::clone(&self.watch), BootStream::Stderr, pipe))
            }),
        ];
        let mut killed_by_app = false;
        let status = loop {
            tokio::select! {
                status = child.wait() => break status,
                Some(answer) = kills.recv() => {
                    killed_by_app = true;
                    if let Err(e) = child.start_kill() {
                        warn!("Failed to kill daemon process: {}", e);
                    }
                    let _ = answer.send(());
                }
            }
        };
        // A kill asked for from here on finds the process gone at once,
        // instead of waiting out the output drain below.
        drop(kills);
        let drained = tokio::time::timeout(OUTPUT_DRAIN, async {
            for reader in readers.into_iter().flatten() {
                let _ = reader.await;
            }
        })
        .await
        .is_ok();
        if !drained {
            warn!(
                "The daemon exited, but a process it started still holds its output open: its exit is recorded without waiting for that"
            );
        }
        // A process that cannot be waited for can no longer be watched, so
        // it counts as gone: a next boot waits for the previous sidecar's
        // exit (see `DaemonManager::previous_sidecar_gone`).
        let (code, signal) = match status {
            Ok(status) => (status.code(), exit_signal(status)),
            Err(e) => {
                error!("Could not wait for the daemon process: {e}");
                (None, None)
            }
        };
        // Before `on_exit` flags the exit: a next spawn waits for that flag,
        // so no new daemon can be writing the record meanwhile.
        if let Some(record) = record {
            record.note(code, signal, killed_by_app).await;
        }
        self.on_exit(code, signal).await;
    }

    /// Record the exit so any in-flight ready-wait bails out now and `stop()`
    /// can tell a graceful exit from a hang (and never tree-kills a recycled
    /// PID), then apply its [`ExitEffect`].
    ///
    /// The output the process wrote before its exit has been read by now
    /// (see [`Self::run`]), so the boot log holds its last line.
    async fn on_exit(&self, code: Option<i32>, signal: Option<i32>) {
        if let Some(code) = code {
            if code != 0 {
                error!("daemon terminated with exit code: {}", code);
            } else {
                info!("daemon terminated normally (code 0)");
            }
        } else if let Some(signal) = signal {
            warn!("daemon terminated by signal: {}", signal);
        } else {
            warn!("daemon terminated (unknown reason)");
        }

        let exit = SidecarExit::new(code, signal, &self.watch.log.lock().await.lines);
        *self.watch.exit.lock().await = Some(exit.clone());
        // Set before the probe below: a stop that runs meanwhile must see a
        // dead sidecar, and leave alone the daemon that may be answering.
        self.watch.exited.store(true, Ordering::SeqCst);

        let (current, state) = {
            let state = self.state.read().await;
            (self.current_attempt.load(Ordering::SeqCst) == self.attempt, *state)
        };
        let still_answering = current
            && state == DaemonState::Running
            && probe_occupant(&self.url).await != PortOccupant::Nobody;
        match exit_effect(current, state, still_answering) {
            ExitEffect::Crash => {
                let mut state = self.state.write().await;
                if *state == DaemonState::Running
                    && self.current_attempt.load(Ordering::SeqCst) == self.attempt
                {
                    error!("The daemon on {} exited and nothing answers there now", self.url);
                    *self.last_failure.write().await =
                        Some(StartFailure::from_exit(StartFailureKind::ExitedAfterReady, &exit));
                    *state = DaemonState::Crashed;
                }
            }
            ExitEffect::EndsReport => {
                let state = self.state.write().await;
                if *state == DaemonState::Crashed
                    && self.current_attempt.load(Ordering::SeqCst) == self.attempt
                {
                    let mut last = self.last_failure.write().await;
                    if replaces_record(last.as_ref()) {
                        *last = Some(StartFailure::from_exit(StartFailureKind::ExitedAfterReady, &exit));
                    }
                }
                drop(state);
            }
            ExitEffect::StayAttached => info!(
                "Our sidecar exited, but a daemon still answers on {}: the app stays attached to it",
                self.url
            ),
            ExitEffect::EndsBoot | ExitEffect::Nothing => {}
        }
    }
}

/// Wait (bounded) for a sidecar's exit, which its event task records.
async fn wait_for_terminated(watch: &SpawnWatch, timeout: Duration) -> bool {
    let deadline = tokio::time::Instant::now() + timeout;
    while tokio::time::Instant::now() < deadline {
        if watch.exited.load(Ordering::SeqCst) {
            return true;
        }
        sleep(Duration::from_millis(100)).await;
    }
    false
}

/// The data directory `NANNA_DEV_DATA_DIR` gives the sidecar, if set.
fn dev_data_dir() -> Option<String> {
    std::env::var("NANNA_DEV_DATA_DIR")
        .ok()
        .filter(|dir| !dir.trim().is_empty())
}

/// The sidecar's data directory, resolved as the daemon resolves its own:
/// `--data-dir` (which this app passes only from `NANNA_DEV_DATA_DIR`), then
/// `[general] data_dir` from the config file, then the platform default.
/// Without reading secrets, so no keyring prompt.
#[cfg(not(test))]
fn sidecar_data_dir() -> Option<std::path::PathBuf> {
    dev_data_dir()
        .map(std::path::PathBuf::from)
        .or_else(|| nanna_config::Config::data_dir_from_disk().ok())
}

/// `start`'s error when a stop took over. Public to the crate so a caller can
/// tell a start that was stopped on request (a restart, a quit, an update)
/// from one that failed.
pub(crate) const START_CANCELLED: &str = "Daemon start cancelled by a stop request";

/// Manages the daemon sidecar process
pub struct DaemonManager {
    config: DaemonManagerConfig,
    state: Arc<RwLock<DaemonState>>,
    restart_count: Arc<RwLock<u32>>,
    child: Arc<RwLock<Option<Sidecar>>>,
    /// When the current start began. Read only while the state is `Starting`.
    starting_since: RwLock<Option<tokio::time::Instant>>,
    /// The current start attempt. Each move to `Starting` opens a new one,
    /// under the state lock. A sidecar's event task changes the state only
    /// while its own attempt is still the current one, so a sidecar that a
    /// restart killed cannot mark the next boot crashed.
    attempt: Arc<AtomicU64>,
    /// Counts each stop twice: when it is requested and when it is done. A
    /// start carries the count its caller saw when it began, and is refused
    /// once a stop has come since (see [`Self::start`]), including one that
    /// was still running when the count was read.
    stops: AtomicU64,
    /// Runs stops one at a time, so each caller returns with its stop done.
    stop_turn: Mutex<()>,
    /// The record of the most recent spawn, kept after the sidecar exits and
    /// after a stop: its boot log is what `get_boot_log` returns.
    latest_spawn: RwLock<Option<Arc<SpawnWatch>>>,
    /// Why the daemon last failed to start or to stay up. Cleared when a
    /// daemon becomes ready.
    last_failure: Arc<RwLock<Option<StartFailure>>>,
    /// The command that runs in the bundled sidecar's place, program first:
    /// a `/bin/sh` script standing in for the daemon, or a program that
    /// does not exist.
    #[cfg(test)]
    stand_in: std::sync::Mutex<Option<Vec<String>>>,
}

impl DaemonManager {
    /// Create a new daemon manager
    #[must_use]
    pub fn new(config: DaemonManagerConfig) -> Self {
        Self {
            config,
            state: Arc::new(RwLock::new(DaemonState::Stopped)),
            restart_count: Arc::new(RwLock::new(0)),
            child: Arc::new(RwLock::new(None)),
            starting_since: RwLock::new(None),
            attempt: Arc::new(AtomicU64::new(0)),
            stops: AtomicU64::new(0),
            stop_turn: Mutex::new(()),
            latest_spawn: RwLock::new(None),
            last_failure: Arc::new(RwLock::new(None)),
            #[cfg(test)]
            stand_in: std::sync::Mutex::new(None),
        }
    }
    
    /// Get the daemon WebSocket URL
    #[must_use]
    pub fn ws_url(&self) -> String {
        format!("ws://{}:{}", self.config.host, self.config.port)
    }
    
    /// Get current daemon state
    pub async fn state(&self) -> DaemonState {
        *self.state.read().await
    }

    /// How long the daemon has been starting, or `None` when it is not
    /// `Starting`.
    pub async fn starting_for(&self) -> Option<Duration> {
        if *self.state.read().await != DaemonState::Starting {
            return None;
        }
        self.starting_since.read().await.map(|since| since.elapsed())
    }

    /// The stop count: each stop counts once when requested and once when
    /// done. Read it when a start is decided on, and pass it to
    /// [`Self::start`]. A count read after [`Self::stop`] returned is one no
    /// stop was running at.
    #[must_use]
    pub fn stop_epoch(&self) -> u64 {
        self.stops.load(Ordering::SeqCst)
    }

    /// Why the daemon last failed to start or to stay up, until a daemon
    /// becomes ready.
    pub async fn last_failure(&self) -> Option<StartFailure> {
        self.last_failure.read().await.clone()
    }

    /// The output lines of the current (or most recent) sidecar, oldest
    /// first. At most `BOOT_LOG_LINES`; each spawn starts a new log.
    pub async fn boot_log(&self) -> Vec<BootLine> {
        let Some(watch) = self.latest_spawn.read().await.clone() else {
            return Vec::new();
        };
        let log = watch.log.lock().await;
        log.lines.iter().cloned().collect()
    }

    /// Give automatic restarts a fresh budget. A restart the user asks for
    /// starts the daemon's story over; without this, a monitor that had used
    /// up its restarts earlier would report that it gave up instead of what
    /// the new boot did.
    pub async fn reset_restarts(&self) {
        *self.restart_count.write().await = 0;
    }

    /// The app attached to a daemon, so one answers, whoever started it. A
    /// crashed state reads running again, with nothing on record, at once:
    /// the status used to say connected and crashed together until the
    /// health monitor's next tick, up to 30 s later. Any other state has an
    /// owner that decides it, a boot's ready-wait or a stop.
    pub async fn attached(&self) {
        self.recovered(self.stop_epoch()).await;
    }

    /// Start the daemon sidecar
    ///
    /// Returns `Ok` straight away when the daemon is already `Running` or
    /// `Starting`. Otherwise it waits until a daemon answers on the port, for
    /// as long as the spawned sidecar stays alive. There is no deadline: a
    /// live sidecar whose port is still closed is booting (see
    /// `boot_wait_verdict`). Before the spawn it also waits for the
    /// previous sidecar to exit (see `Self::previous_sidecar_gone`).
    /// [`Self::starting_for`] reports how long it has been.
    ///
    /// `since_stop` is the [`Self::stop_epoch`] the caller read when it
    /// decided to start. A stop requested after that wins: the start is
    /// refused, or cancelled if it is under way. Without it, an init or a
    /// health-monitor restart that was already on its way when a stop landed
    /// (a restart, an update, quit) booted a daemon right after the stop.
    ///
    /// # Errors
    ///
    /// Each failure leaves the manager `Crashed` and recorded in
    /// [`Self::last_failure`], except a cancelled start.
    ///
    /// - `"Failed to create sidecar command: …"` when the bundled
    ///   `nanna-daemon` sidecar cannot be resolved;
    /// - `"Failed to spawn daemon: …"` when its process fails to start;
    /// - `"Daemon exited during startup"` when the sidecar exits and nothing
    ///   answers on the daemon port;
    /// - `"Daemon start cancelled by a stop request"` when [`Self::stop`] was
    ///   requested after `since_stop`, or while the start was stopping. The
    ///   state is then the one `stop` set.
    pub async fn start<R: Runtime>(&self, app: &AppHandle<R>, since_stop: u64) -> Result<(), String> {
        let Some(attempt) = self.claim_start(since_stop).await? else {
            return Ok(());
        };
        self.evict_stale_daemon(env!("CARGO_PKG_VERSION"), attempt).await;
        info!("Starting nanna-daemon sidecar...");
        let watch = self.spawn_sidecar(app, attempt).await?;

        // No deadline and no kill. The kill used to keep an embedded
        // in-process backend's fallback able to open nanna.db, and that
        // backend no longer exists. A live sidecar is still booting, and
        // killing it is what turned a slow boot into one that never finished.
        // The only way this wait fails is a sidecar that has exited. A dead
        // process holds no port and no lock, so there is nothing to kill.
        match self.wait_for_ready(attempt, &watch).await {
            BootWait::Ready => {
                info!("Daemon started successfully on {}", self.ws_url());
                Ok(())
            }
            BootWait::Cancelled => {
                info!("Daemon start abandoned: a stop request took over");
                Err(START_CANCELLED.to_string())
            }
            // `wait_for_ready` returns only a final verdict, never `KeepWaiting`.
            BootWait::Failed | BootWait::KeepWaiting => {
                let exit = watch.exit.lock().await.clone().unwrap_or_default();
                error!("Daemon exited during startup and nothing answers on {}", self.ws_url());
                let failure = StartFailure::from_exit(StartFailureKind::ExitedDuringBoot, &exit);
                self.fail_attempt(attempt, failure).await;
                Err("Daemon exited during startup".to_string())
            }
        }
    }

    /// Move to `Starting` and open a new start attempt, whose number is
    /// returned. `None` when a start is already under way or the daemon is
    /// running.
    ///
    /// The check and the move happen under one state lock, so two callers
    /// (an init and the health monitor, say) can never both spawn a sidecar.
    ///
    /// # Errors
    ///
    /// [`START_CANCELLED`] when a stop was requested after `since_stop`, or
    /// one is in progress.
    async fn claim_start(&self, since_stop: u64) -> Result<Option<u64>, String> {
        let mut state = self.state.write().await;
        if self.stops.load(Ordering::SeqCst) != since_stop || *state == DaemonState::Stopping {
            return Err(START_CANCELLED.to_string());
        }
        if matches!(*state, DaemonState::Running | DaemonState::Starting) {
            return Ok(None);
        }
        let attempt = self.attempt.fetch_add(1, Ordering::SeqCst) + 1;
        *self.starting_since.write().await = Some(tokio::time::Instant::now());
        *state = DaemonState::Starting;
        drop(state);
        Ok(Some(attempt))
    }

    /// Open a start attempt the way `start` does before its spawn, for tests
    /// of what runs around a boot (an init in flight, a restart).
    #[cfg(test)]
    pub(crate) async fn begin_start_for_test(&self) -> u64 {
        self.claim_start(self.stop_epoch())
            .await
            .expect("no stop since")
            .expect("nothing under way")
    }

    /// Open a start attempt and fail it with `failure`, as a boot that
    /// exited does: `Crashed`, with `failure` on record.
    #[cfg(test)]
    pub(crate) async fn crash_for_test(&self, failure: StartFailure) {
        let attempt = self.begin_start_for_test().await;
        self.fail_attempt(attempt, failure).await;
    }

    /// Whether `attempt` still owns a start in progress: no stop, and no
    /// newer start, has come since it began.
    async fn is_starting(&self, attempt: u64) -> bool {
        let state = self.state.read().await;
        *state == DaemonState::Starting && self.attempt.load(Ordering::SeqCst) == attempt
    }

    /// The state as `attempt` sees it: `Stopped` once a newer start has
    /// replaced it, since a stop came in between.
    async fn state_of(&self, attempt: u64) -> DaemonState {
        let state = self.state.read().await;
        if self.attempt.load(Ordering::SeqCst) == attempt {
            *state
        } else {
            DaemonState::Stopped
        }
    }

    /// Start a new boot log and spawn record. The previous sidecar's record
    /// stays with its event task, so its late output cannot mix into the
    /// new log.
    async fn new_spawn_watch(&self) -> Arc<SpawnWatch> {
        let watch = Arc::new(SpawnWatch::default());
        *self.latest_spawn.write().await = Some(Arc::clone(&watch));
        watch
    }

    /// Spawn the sidecar for `attempt` and start its event task. Returns the
    /// spawn's record, which the ready-wait watches.
    ///
    /// The child slot is held from the check to the store. `stop()` takes
    /// that slot after it sets `Stopping`, so a stop either lands before the
    /// check and cancels the spawn, or waits for the new child and stops it.
    /// A stop during the eviction used to miss the child spawned after it,
    /// and that sidecar then ran with no one to stop it.
    ///
    /// The previous sidecar goes first. A live one in the slot is killed: a
    /// [`Sidecar`] does not kill its process when dropped, so replacing it
    /// in the slot would leave it running with nothing left to stop it. The
    /// health monitor never restarts a daemon whose sidecar lives (see
    /// [`after_failed_check`]). A retry does, from the `Crashed` state the
    /// monitor reports such a daemon in, and asks for exactly this. Then the
    /// spawn waits until the previous sidecar has exited (see
    /// [`Self::previous_sidecar_gone`]).
    async fn spawn_sidecar<R: Runtime>(
        &self,
        app: &AppHandle<R>,
        attempt: u64,
    ) -> Result<Arc<SpawnWatch>, String> {
        let mut slot = self.child.write().await;
        if !self.is_starting(attempt).await {
            info!("Daemon start abandoned before the spawn: a stop request took over");
            return Err(START_CANCELLED.to_string());
        }
        if let Some(live) = slot.take_if(|sidecar| !sidecar.watch.exited.load(Ordering::SeqCst)) {
            warn!(
                "The previous sidecar (PID {}) is still running: killing it before the next one starts",
                live.pid
            );
            kill_sidecar_tree(live.pid).await;
            live.kill().await;
        }
        if !self.previous_sidecar_gone(attempt).await {
            info!("Daemon start abandoned before the spawn: a stop request took over");
            return Err(START_CANCELLED.to_string());
        }
        let watch = self.new_spawn_watch().await;

        // Spawn the sidecar. A failure must not leave the state `Starting`:
        // `start` returns early on `Starting`, so every later call (the Retry
        // button, the health monitor) would report success without spawning.
        // Nor may it leave the spawn's record waiting for an exit that no
        // process will make: the next spawn waits for that exit.
        info!("Creating sidecar command for nanna-daemon...");
        #[cfg(test)]
        let command = self.stand_in_command(app).map_or_else(|| app.shell().sidecar("nanna-daemon"), Ok);
        #[cfg(not(test))]
        let command = app.shell().sidecar("nanna-daemon");
        let sidecar = match command {
            Ok(sidecar) => sidecar,
            Err(e) => {
                error!("Failed to create sidecar command: {}", e);
                watch.exited.store(true, Ordering::SeqCst);
                let failure = StartFailure::new(
                    StartFailureKind::SidecarUnresolved,
                    format!("Could not locate the bundled nanna-daemon program: {e}"),
                );
                self.fail_attempt(attempt, failure).await;
                return Err(format!("Failed to create sidecar command: {e}"));
            }
        };

        let args = self.sidecar_args();
        info!("Spawning daemon with args: {:?}", args);
        // NO_COLOR: the daemon's console log otherwise colours every line
        // (tracing-subscriber turns ANSI on unless NO_COLOR is set, terminal
        // or not), and the relayed lines reached the log view and the boot
        // log full of escape codes.
        let command: std::process::Command = sidecar.args(args).env("NO_COLOR", "1").into();
        let spawned_at = chrono::Utc::now();
        // The plugin resolves the bundled program, pipes its stdio and, on
        // Windows, hides its console window. The process itself is ours to
        // wait for, not the plugin's (see `SidecarEvents::run`).
        let child = match tokio::process::Command::from(command).spawn() {
            Ok(child) => child,
            Err(e) => {
                error!("Failed to spawn daemon: {}", e);
                watch.exited.store(true, Ordering::SeqCst);
                let failure = StartFailure::new(
                    StartFailureKind::SpawnFailed,
                    format!("Could not start nanna-daemon: {e}"),
                );
                self.fail_attempt(attempt, failure).await;
                return Err(format!("Failed to spawn daemon: {e}"));
            }
        };
        // `Some` until the process has been waited for, which nothing has
        // done yet.
        let pid = child.id().unwrap_or_default();
        let (kill, kills) = mpsc::unbounded_channel();
        *slot = Some(Sidecar {
            pid,
            kill,
            watch: Arc::clone(&watch),
        });
        drop(slot);

        // The event task also records the termination, which is what ends
        // the ready-wait for a sidecar that dies while booting.
        // Tests run stand-in programs and must not touch a real data dir.
        #[cfg(test)]
        let record = None;
        #[cfg(not(test))]
        let record = sidecar_data_dir()
            .filter(|_| pid != 0)
            .map(|data_dir| ExitRecordTarget { data_dir, pid, spawned_at });
        #[cfg(test)]
        let _ = spawned_at;
        tokio::spawn(self.sidecar_events(attempt, Arc::clone(&watch)).run(child, kills, record));
        Ok(watch)
    }

    /// The command a test set to run in the sidecar's place, if any.
    #[cfg(test)]
    fn stand_in_command<R: Runtime>(
        &self,
        app: &AppHandle<R>,
    ) -> Option<tauri_plugin_shell::process::Command> {
        let stand_in = self.stand_in.lock().expect("the stand-in lock is never poisoned").clone()?;
        let (program, args) = stand_in.split_first()?;
        Some(app.shell().command(program).args(args))
    }

    /// Run `argv` (program first) in the sidecar's place from now on. Only
    /// the Unix tests call this, since they run `/bin/sh` scripts in the
    /// daemon's place, so it is Unix only too: a Windows test build would
    /// find it unused.
    #[cfg(all(test, unix))]
    pub(crate) fn replace_sidecar_with(&self, argv: &[&str]) {
        *self.stand_in.lock().expect("the stand-in lock is never poisoned") =
            Some(argv.iter().map(ToString::to_string).collect());
    }

    /// Wait until the most recent sidecar has exited, for as long as
    /// `attempt` still owns the start. `false` when a stop took over first.
    ///
    /// A sidecar that was killed (by a stop, a restart, or the backstop in
    /// [`Self::spawn_sidecar`]) can take a while to go. A process blocked in
    /// disk I/O dies only once that I/O completes, which on a busy disk takes
    /// seconds. Until then it holds the daemon's PID file, and maybe the
    /// port, so a new daemon beside it would exit "Already running" and the
    /// restart would fail. There is no deadline, for the reason the
    /// ready-wait has none: nothing can start before the process is gone, and
    /// a stop ends the wait at once. The exit waited for is the process's
    /// own, as the operating system reports it, not the end of its output,
    /// which a child of the daemon can hold open for good (see
    /// [`SidecarEvents::run`]).
    async fn previous_sidecar_gone(&self, attempt: u64) -> bool {
        let Some(previous) = self.latest_spawn.read().await.clone() else {
            return true;
        };
        if previous.exited.load(Ordering::SeqCst) {
            return true;
        }
        info!("Waiting for the previous sidecar to exit before the next one starts");
        loop {
            if previous.exited.load(Ordering::SeqCst) {
                return true;
            }
            if !self.is_starting(attempt).await {
                return false;
            }
            sleep(Duration::from_millis(100)).await;
        }
    }

    /// The PID of the sidecar we spawned, while it is still running.
    async fn live_sidecar_pid(&self) -> Option<u32> {
        self.child
            .read()
            .await
            .as_ref()
            .filter(|sidecar| !sidecar.watch.exited.load(Ordering::SeqCst))
            .map(|sidecar| sidecar.pid)
    }

    /// The event task for the sidecar `attempt` spawned.
    fn sidecar_events(&self, attempt: u64, watch: Arc<SpawnWatch>) -> SidecarEvents {
        SidecarEvents {
            attempt,
            watch,
            state: Arc::clone(&self.state),
            current_attempt: Arc::clone(&self.attempt),
            last_failure: Arc::clone(&self.last_failure),
            url: self.ws_url(),
        }
    }

    /// `attempt` reached a daemon: `Running`, with a fresh restart budget and
    /// no failure on record.
    ///
    /// `exited_before_probe` says whether `watch`'s sidecar had exited before
    /// the probe that got the answer began. If it exited after that, the
    /// answer may have been its last: a daemon that answered and then died
    /// at once. Its exit, seen while the state still read `Starting`, left
    /// the verdict to the ready-wait, and marking `Running` now would leave a
    /// dead daemon reading `running`, with nothing on record and, before any
    /// attach, no health monitor to restart it. The exit is flagged before
    /// the event task reads the state, so a flag this lock does not see is an
    /// exit that finds `Running` and is handled as one (see [`exit_effect`]).
    async fn mark_ready(&self, attempt: u64, watch: &SpawnWatch, exited_before_probe: bool) -> Marked {
        let mut state = self.state.write().await;
        if *state != DaemonState::Starting || self.attempt.load(Ordering::SeqCst) != attempt {
            return Marked::Superseded;
        }
        if !exited_before_probe && watch.exited.load(Ordering::SeqCst) {
            return Marked::ExitedSinceProbe;
        }
        *self.restart_count.write().await = 0;
        *self.last_failure.write().await = None;
        *state = DaemonState::Running;
        Marked::Running
    }

    /// `attempt` failed: `Crashed`, with `failure` on record. Nothing changes
    /// when a stop took over first: a stop is not a failure.
    async fn fail_attempt(&self, attempt: u64, failure: StartFailure) {
        let mut state = self.state.write().await;
        if *state == DaemonState::Starting && self.attempt.load(Ordering::SeqCst) == attempt {
            *self.last_failure.write().await = Some(failure);
            *state = DaemonState::Crashed;
        }
    }

    /// The sidecar's command line: the bind address, the optional dev-store
    /// isolation, then `run`.
    fn sidecar_args(&self) -> Vec<String> {
        // A dev build must never share the installed app's store: the daemon
        // takes an exclusive lock on nanna.db and runs migrations at startup,
        // so two builds pointed at one file is unsupported and unsafe.
        // NANNA_DEV_DATA_DIR isolates a dev run; unset in production, where
        // the daemon resolves its own default data dir as before.
        let port = self.config.port.to_string();
        let mut args: Vec<String> = vec![
            "--port".into(),
            port,
            "--host".into(),
            self.config.host.clone(),
        ];
        if let Some(dev_data_dir) = dev_data_dir() {
            info!("NANNA_DEV_DATA_DIR set — isolating daemon store at {dev_data_dir}");
            args.push("--data-dir".into());
            args.push(dev_data_dir);
        }
        // A killed GUI must not leave its daemon behind (Unix; on Windows
        // the Job Object already covers it, and the daemon ignores the flag).
        args.push("--exit-with-parent".into());
        args.push("run".into());
        args
    }

    /// Shut down a daemon from a different release that is holding our port.
    ///
    /// An app update replaces the GUI and its sidecar binary, but a daemon
    /// that outlived the old GUI keeps port 5149 and the `nanna.db` lock. The
    /// new sidecar then cannot bind, exits, and the GUI would attach to the old
    /// daemon — a v0.3.20 UI driving a v0.3.19 server, still running every bug
    /// the update shipped to fix (2026-09-17). Evicting it first lets the new
    /// sidecar start.
    ///
    /// Only a daemon that *reports* a different version is touched. One that
    /// cannot be identified is left alone: attaching to it is the old
    /// behaviour, and shutting down something we cannot name is worse.
    ///
    /// A stop ends the eviction early, so the start of `attempt` that runs it
    /// unwinds promptly: a restart waits for exactly that.
    async fn evict_stale_daemon(&self, ours: &str, attempt: u64) {
        let url = self.ws_url();
        let occupant = probe_occupant(&url).await;
        let Some(theirs) = stale_version(&occupant, ours) else {
            return;
        };
        if !self.is_starting(attempt).await {
            return;
        }
        warn!(
            "A v{theirs} daemon is running on {url}, but this app is v{ours} — asking it to shut down so the matching daemon can start"
        );
        if !self.request_graceful_shutdown().await {
            error!("Could not deliver a shutdown request to the v{theirs} daemon on {url}");
            return;
        }
        let deadline = tokio::time::Instant::now() + EVICTION_TIMEOUT;
        while tokio::time::Instant::now() < deadline {
            if !self.is_starting(attempt).await {
                return;
            }
            if probe_occupant(&url).await == PortOccupant::Nobody {
                info!("The v{theirs} daemon released {url}");
                return;
            }
            sleep(Duration::from_millis(200)).await;
        }
        error!(
            "The v{theirs} daemon on {url} is still running {}s after a shutdown request — the app will be talking to a server from a different release",
            EVICTION_TIMEOUT.as_secs()
        );
    }

    /// Poll the daemon port until [`boot_wait_verdict`] is final: a daemon
    /// answered (and the state is now `Running`, see [`Self::mark_ready`]),
    /// the sidecar exited with nobody answering, or a stop took over. There
    /// is no deadline while the sidecar is alive. A slow boot is logged at
    /// [`SLOW_START_NOTICE`], then at each doubling of the elapsed time.
    async fn wait_for_ready(&self, attempt: u64, watch: &SpawnWatch) -> BootWait {
        let url = self.ws_url();
        let mut evicted = false;
        let mut next_notice = SLOW_START_NOTICE;
        loop {
            // Read before the probe. An exit seen here means the probe below
            // ran after the exit, so a daemon that was already up when our
            // sidecar deferred to it (the AlreadyRunning exit) still gets
            // attached.
            let exited = watch.exited.load(Ordering::SeqCst);
            let occupant = probe_occupant(&url).await;
            // A daemon from another release can win the port between the
            // pre-spawn check and here. Evict it once; answering is not the
            // same as being ours.
            if stale_version(&occupant, env!("CARGO_PKG_VERSION")).is_some() && !evicted {
                evicted = true;
                self.evict_stale_daemon(env!("CARGO_PKG_VERSION"), attempt).await;
                continue;
            }
            let answered = occupant != PortOccupant::Nobody;
            let state = self.state_of(attempt).await;
            match boot_wait_verdict(state, exited, answered) {
                BootWait::KeepWaiting => {}
                BootWait::Ready => match self.mark_ready(attempt, watch, exited).await {
                    Marked::Running => {
                        if exited {
                            info!("Attached to an existing daemon instance on {}", url);
                        }
                        return BootWait::Ready;
                    }
                    Marked::Superseded => return BootWait::Cancelled,
                    Marked::ExitedSinceProbe => continue,
                },
                verdict => return verdict,
            }
            if let Some(elapsed) = self.starting_for().await
                && elapsed >= next_notice
            {
                let pid = self
                    .child
                    .read()
                    .await
                    .as_ref()
                    .map_or_else(|| "unknown".to_string(), |sidecar| sidecar.pid.to_string());
                warn!(
                    "Daemon still starting after {}s (PID {pid}): the process is alive and {url} is not open yet — waiting",
                    elapsed.as_secs()
                );
                next_notice = elapsed * 2;
            }
            sleep(Duration::from_millis(200)).await;
        }
    }
    
    /// Stop the daemon
    ///
    /// Returns once the stop is done. A stop requested while another runs
    /// waits for it. A sidecar that had to be killed has been sent the kill
    /// by then (see `Sidecar::kill`), so it cannot outlive an app that
    /// exits right after. It may take a while to exit; the next spawn waits
    /// for that (see `Self::previous_sidecar_gone`).
    ///
    /// # Errors
    ///
    /// Never returns `Err` today: an undeliverable or ignored shutdown request
    /// falls back to a tree-kill, a failed or unconfirmed kill is logged, and
    /// the manager always ends `Stopped`.
    pub async fn stop(&self) -> Result<(), String> {
        // Counted before anything else, even when there is nothing to stop:
        // from here on, every start decided on before this stop is refused
        // (see [`Self::start`]), including one that has not reached the state
        // lock yet.
        self.stops.fetch_add(1, Ordering::SeqCst);
        let turn = self.stop_turn.lock().await;
        self.stop_sidecar().await;
        drop(turn);
        Ok(())
    }

    /// The body of [`Self::stop`], which runs one at a time.
    ///
    /// The state reads `Stopping` for the whole stop, even when there was
    /// nothing to stop, and the stop is counted done (see
    /// [`Self::stop_epoch`]) in the same state lock that sets `Stopped`.
    /// [`Self::claim_start`] reads the count under that lock, so a start
    /// decided on while this stop ran finds `Stopping`, or `Stopped` with the
    /// stop counted done, and is refused either way. A stop that found
    /// `Stopped` used to leave the state alone and count itself done after
    /// the lock: a start that claimed in between got a boot nothing
    /// cancelled, and a restart then waited out that whole boot, which has
    /// no deadline.
    async fn stop_sidecar(&self) {
        let was = std::mem::replace(&mut *self.state.write().await, DaemonState::Stopping);
        let running = was != DaemonState::Stopped;
        if running {
            info!("Stopping nanna-daemon...");
        }

        // Held until the shutdown below completes, exactly as long as the guard
        // used to live in the `if let`: a racing `start()` can neither store a
        // new child nor kill one while this shutdown is in progress.
        let mut child_slot = self.child.write().await;
        if let Some(sidecar) = child_slot.take() {
            if sidecar.watch.exited.load(Ordering::SeqCst) {
                // The sidecar died earlier — usually its AlreadyRunning exit
                // after attaching to a standalone daemon. That daemon isn't
                // ours to stop, and the sidecar's PID may have been recycled,
                // so neither a shutdown request nor a tree-kill is safe here.
                info!("Sidecar already exited — leaving any attached daemon running");
            } else {
                // Prefer a graceful IPC shutdown: the daemon flushes state and
                // its kill-on-close Job Object reaps in-flight exec children on
                // exit. A pre-Job-Object daemon acks the request without
                // stopping — the bounded wait catches that and falls through.
                let exited = self.request_graceful_shutdown().await
                    && wait_for_terminated(&sidecar.watch, Duration::from_secs(5)).await;
                if exited {
                    info!("Daemon exited gracefully");
                } else {
                    // Hard stop: kill the TREE, not just the daemon — a bare
                    // kill() orphans in-flight exec children on daemons whose
                    // Job Object never adopted (or that predate it).
                    warn!("Graceful daemon shutdown failed — tree-killing PID {}", sidecar.pid);
                    kill_sidecar_tree(sidecar.pid).await;
                    sidecar.kill().await;
                }
            }
        }
        drop(child_slot);

        let mut state = self.state.write().await;
        self.stops.fetch_add(1, Ordering::SeqCst);
        *state = DaemonState::Stopped;
        drop(state);
        if running {
            info!("Daemon stopped");
        }
    }

    /// Ask the daemon to stop via its control plane. Returns whether the
    /// request was delivered — not whether the daemon actually exited.
    async fn request_graceful_shutdown(&self) -> bool {
        use futures_util::SinkExt;
        use tokio_tungstenite::tungstenite::Message;

        let url = self.ws_url();
        let Ok(Ok((mut ws, _))) =
            tokio::time::timeout(Duration::from_secs(2), tokio_tungstenite::connect_async(&url))
                .await
        else {
            return false;
        };
        let request = serde_json::json!({
            "id": "daemon-manager-shutdown",
            "action": { "type": "system", "action": "shutdown" }
        });
        // send() flushes before returning; the daemon acks, then stops.
        let sent = ws.send(Message::Text(request.to_string().into())).await.is_ok();
        let _ = ws.close(None).await;
        sent
    }

    /// Start health monitoring (call once after start)
    pub fn start_health_monitor<R: Runtime>(self: Arc<Self>, app: AppHandle<R>) {
        tokio::spawn(async move {
            loop {
                sleep(self.config.health_check_interval).await;
                self.health_tick(&app).await;
            }
        });
    }

    /// One health check, and what follows from it.
    ///
    /// `Crashed` is checked too: a daemon the app attached to may answer
    /// again (back to `Running`), or a first boot that failed stays down
    /// (restart it).
    async fn health_tick<R: Runtime>(&self, app: &AppHandle<R>) {
        // A stop from here on cancels this tick's restart.
        let since_stop = self.stop_epoch();
        let state = self.state().await;
        if !matches!(state, DaemonState::Running | DaemonState::Crashed) {
            return;
        }

        let url = self.ws_url();
        let failure = match check_health(&url, self.config.health_check_timeout).await {
            Ok(()) => {
                if state == DaemonState::Crashed {
                    // Our sidecar is gone, but a daemon is alive and
                    // answering.
                    self.recovered(since_stop).await;
                }
                debug!("Daemon health check: OK");
                return;
            }
            Err(failure) => failure,
        };
        match after_failed_check(&failure, self.live_sidecar_pid().await) {
            AfterFailedCheck::CheckAgain => {
                warn!(
                    "Daemon health check failed ({failure}): something holds {url} but did not answer in time. Checking again in {}s",
                    self.config.health_check_interval.as_secs()
                );
                return;
            }
            AfterFailedCheck::Report { pid } => {
                warn!(
                    "Daemon health check failed ({failure}), but the daemon we started (PID {pid}) is still running. A new one could not start beside it, so it is not restarted; restarting the daemon from the app replaces it"
                );
                let record = StartFailure::new(
                    StartFailureKind::HealthCheckFailed,
                    format!(
                        "The daemon stopped answering on {url} ({failure}) · its process (PID {pid}) is still running"
                    ),
                );
                self.not_serving(since_stop, record).await;
                return;
            }
            AfterFailedCheck::Restart => {}
        }

        warn!("Daemon health check failed: {failure}");
        let record = StartFailure::new(
            StartFailureKind::HealthCheckFailed,
            format!("The daemon stopped answering on {url} ({failure})"),
        );
        match self.on_failed_check(since_stop, record).await {
            // A stop or someone else's start took over while the check ran:
            // the restart is not ours to make.
            FailedCheck::NotOurs => {}
            FailedCheck::GaveUp => error!("Max daemon restarts exceeded, giving up"),
            FailedCheck::Restart(n) => {
                warn!("Attempting daemon restart ({n}/{})", self.config.max_restarts);
                sleep(self.config.restart_delay).await;
                if let Err(e) = self.start(app, since_stop).await {
                    error!("Daemon restart failed: {}", e);
                }
            }
        }
    }

    /// Whether the health monitor's tick that began at `since_stop` still
    /// owns the state: no stop since, and nobody else started a boot.
    fn monitor_owns(&self, since_stop: u64, state: DaemonState) -> bool {
        self.stops.load(Ordering::SeqCst) == since_stop
            && matches!(state, DaemonState::Running | DaemonState::Crashed)
    }

    /// A daemon answers again after a crash: `Running`, nothing on record.
    async fn recovered(&self, since_stop: u64) {
        let mut state = self.state.write().await;
        if *state == DaemonState::Crashed && self.stops.load(Ordering::SeqCst) == since_stop {
            info!("A daemon answers on {} again", self.ws_url());
            *self.last_failure.write().await = None;
            *state = DaemonState::Running;
        }
    }

    /// Record a health check that found the daemon gone as a crash, and count
    /// the restart it calls for, or give up once the restarts are used up.
    /// [`FailedCheck::NotOurs`], with nothing changed, when the state moved
    /// on while the check ran.
    ///
    /// The count is checked and raised in one step, under its lock. When it
    /// was read before the check and written after it, a restart's budget
    /// reset (see [`Self::reset_restarts`]) that landed in between was
    /// undone, and the next failure gave up at once.
    ///
    /// The check goes on record as [`record_check`] says. Giving up is
    /// recorded once, around the failure before it (or this check's, when
    /// there was none): the monitor logs every tick, and the record keeps the
    /// first time and reason.
    async fn on_failed_check(&self, since_stop: u64, failure: StartFailure) -> FailedCheck {
        let mut state = self.state.write().await;
        if !self.monitor_owns(since_stop, *state) {
            return FailedCheck::NotOurs;
        }
        let restart = {
            let mut restarts = self.restart_count.write().await;
            (*restarts < self.config.max_restarts).then(|| {
                *restarts += 1;
                *restarts
            })
        };
        {
            let mut last = self.last_failure.write().await;
            if restart.is_some() {
                record_check(&mut last, *state, failure);
            } else if last
                .as_ref()
                .is_none_or(|previous| previous.kind != StartFailureKind::RestartsExhausted)
            {
                let previous = last.take().unwrap_or(failure);
                *last = Some(StartFailure::restarts_exhausted(self.config.max_restarts, &previous));
            }
        }
        *state = DaemonState::Crashed;
        drop(state);
        restart.map_or(FailedCheck::GaveUp, FailedCheck::Restart)
    }

    /// Record a health check that found the daemon we started alive but not
    /// serving: `Crashed`, with the check on record as [`record_check`] says,
    /// and no restart counted, since none is made. Nothing changes when the
    /// state moved on while the check ran.
    async fn not_serving(&self, since_stop: u64, failure: StartFailure) {
        let mut state = self.state.write().await;
        if !self.monitor_owns(since_stop, *state) {
            return;
        }
        record_check(&mut *self.last_failure.write().await, *state, failure);
        *state = DaemonState::Crashed;
    }
}

/// Put a failed health check on `last`, the failure on record, given the
/// `state` the daemon was in. A daemon that was already down keeps its
/// failure: that says why (a boot that exited, say), and a check that cannot
/// connect to a daemon that is down adds nothing to it.
fn record_check(last: &mut Option<StartFailure>, state: DaemonState, failure: StartFailure) {
    if state == DaemonState::Running || last.is_none() {
        *last = Some(failure);
    }
}

impl Drop for DaemonManager {
    fn drop(&mut self) {
        // Note: async drop not possible, but child will be killed when dropped
        info!("DaemonManager dropped");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::Message;

    fn daemon(version: Option<&str>) -> PortOccupant {
        PortOccupant::Daemon { version: version.map(str::to_string) }
    }

    /// The 2026-09-18 case: the sidecar is alive and its port is still closed
    /// minutes into the boot. That means wait, not fail. Elapsed time is
    /// deliberately not an input to this function.
    #[test]
    fn a_live_sidecar_with_a_closed_port_is_still_booting() {
        assert_eq!(boot_wait_verdict(DaemonState::Starting, false, false), BootWait::KeepWaiting);
    }

    #[test]
    fn an_answer_on_the_port_is_ready_whether_or_not_our_sidecar_lives() {
        assert_eq!(boot_wait_verdict(DaemonState::Starting, false, true), BootWait::Ready);
        // The AlreadyRunning exit: our sidecar deferred to a daemon that is up.
        assert_eq!(boot_wait_verdict(DaemonState::Crashed, true, true), BootWait::Ready);
    }

    #[test]
    fn only_an_exited_sidecar_with_nobody_answering_is_a_failure() {
        // The exit handler leaves a boot's state to the ready-wait, so it
        // sees the exit while `Starting`; `Crashed` gets the same verdict.
        assert_eq!(boot_wait_verdict(DaemonState::Crashed, true, false), BootWait::Failed);
        assert_eq!(boot_wait_verdict(DaemonState::Starting, true, false), BootWait::Failed);
    }

    /// `stop()` during a boot owns the state it leaves behind. Neither a
    /// late answer nor the sidecar's exit (which the stop itself caused) may
    /// overwrite `Stopped` with `Running` or `Crashed`.
    #[test]
    fn a_stop_during_the_wait_cancels_it() {
        for state in [DaemonState::Stopping, DaemonState::Stopped] {
            for exited in [false, true] {
                for answered in [false, true] {
                    assert_eq!(boot_wait_verdict(state, exited, answered), BootWait::Cancelled);
                }
            }
        }
    }

    #[test]
    fn only_a_daemon_reporting_another_version_is_stale() {
        assert_eq!(stale_version(&daemon(Some("0.3.19")), "0.3.20"), Some("0.3.19"));
        assert_eq!(stale_version(&daemon(Some("0.3.20")), "0.3.20"), None);
        // Unidentified: attach as before rather than shut down a stranger.
        assert_eq!(stale_version(&daemon(None), "0.3.20"), None);
        assert_eq!(stale_version(&PortOccupant::Nobody, "0.3.20"), None);
    }

    #[test]
    fn version_is_read_only_from_the_probe_reply() {
        let reply = r#"{"id":"daemon-manager-version","result":{"status":"success","data":{"version":"0.3.19","name":"nanna-daemon"}}}"#;
        assert_eq!(version_from_reply(reply), Some(VersionReply { version: Some("0.3.19".to_string()) }));

        let error = r#"{"id":"daemon-manager-version","result":{"status":"error","code":"x","message":"y"}}"#;
        assert_eq!(version_from_reply(error), Some(VersionReply { version: None }));

        let other = r#"{"id":"something-else","result":{"status":"success","data":{"version":"9.9.9"}}}"#;
        assert_eq!(version_from_reply(other), None);
        assert_eq!(version_from_reply(r#"{"event":"heartbeat"}"#), None);
        assert_eq!(version_from_reply("not json"), None);
    }

    /// Serve one connection the way a daemon would: an unsolicited event
    /// first, then the reply to the version request.
    async fn fake_daemon(version: &'static str) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("ws://{}", listener.local_addr().unwrap());
        tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut ws = tokio_tungstenite::accept_async(stream).await.unwrap();
            let Some(Ok(Message::Text(request))) = ws.next().await else { return };
            let request: serde_json::Value = serde_json::from_str(&request).unwrap();
            assert_eq!(request.pointer("/action/action"), Some(&serde_json::json!("version")));
            let event = serde_json::json!({ "event": "heartbeat" });
            let reply = serde_json::json!({
                "id": request["id"],
                "result": { "status": "success", "data": { "version": version } }
            });
            let _ = ws.send(Message::Text(event.to_string().into())).await;
            let _ = ws.send(Message::Text(reply.to_string().into())).await;
        });
        url
    }

    #[tokio::test]
    async fn probe_reports_the_version_past_unsolicited_events() {
        let url = fake_daemon("0.3.19").await;
        assert_eq!(probe_occupant(&url).await, daemon(Some("0.3.19")));
    }

    #[tokio::test]
    async fn probe_of_a_closed_port_finds_nobody() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("ws://{}", listener.local_addr().unwrap());
        drop(listener);
        assert_eq!(probe_occupant(&url).await, PortOccupant::Nobody);
    }

    // -------------------------------------------------------------------------
    // Start attempts, failures, the boot log
    // -------------------------------------------------------------------------

    /// What a sidecar printed when another daemon held the PID file
    /// (captured 2026-09-18 from a real second daemon on one data dir).
    const PID_FILE_TAKEN: [&str; 3] = [
        r#"2026-09-18T17:21:44.208585Z  INFO nanna_daemon::server: Data directory: "/tmp/s/dataA""#,
        "2026-09-18T17:21:44.208739Z ERROR nanna_daemon::server: Another daemon instance is already running (PID: 298915)",
        "2026-09-18T17:21:44.209179Z ERROR nanna_daemon: Error: Already running",
    ];

    /// What a sidecar printed when another program held its port (captured
    /// the same way).
    const PORT_TAKEN: [&str; 3] = [
        "2026-09-18T17:21:44.224718Z ERROR nanna_daemon::server: IPC port 127.0.0.1:51993 unavailable: Address already in use (os error 98) — another daemon (or another program) holds it; exiting before touching storage",
        "2026-09-18T17:21:44.224786Z  INFO nanna_daemon::health: PID file removed",
        "2026-09-18T17:21:44.225193Z ERROR nanna_daemon: Error: IPC error: IPC port 127.0.0.1:51993 unavailable: Address already in use (os error 98)",
    ];

    /// A daemon's first log line as it reached the GUI before the sidecar
    /// got `NO_COLOR` (captured from a real boot).
    const COLOURED: &str = "\u{1b}[2m2026-09-18T17:21:21.595409Z\u{1b}[0m \u{1b}[32m INFO\u{1b}[0m \u{1b}[2mnanna_daemon\u{1b}[0m\u{1b}[2m:\u{1b}[0m Starting Nanna daemon...\n";

    fn line(stream: BootStream, text: &str) -> BootLine {
        BootLine { stream, line: text.to_string() }
    }

    fn stdout(lines: &[&str]) -> VecDeque<BootLine> {
        lines.iter().map(|text| line(BootStream::Stdout, text)).collect()
    }

    /// A manager for a daemon on `port`.
    fn manager_on(port: u16) -> DaemonManager {
        DaemonManager::new(DaemonManagerConfig { port, ..DaemonManagerConfig::default() })
    }

    /// A port nothing listens on.
    async fn free_port() -> u16 {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        listener.local_addr().unwrap().port()
    }

    /// Serve a daemon's handshake and version reply on a free port, on every
    /// connection, for as long as the test runs. Returns the port.
    async fn serve_daemon(version: &'static str) -> u16 {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                tokio::spawn(async move {
                    let Ok(mut ws) = tokio_tungstenite::accept_async(stream).await else { return };
                    while let Some(Ok(Message::Text(request))) = ws.next().await {
                        let request: serde_json::Value = serde_json::from_str(&request).unwrap();
                        let reply = serde_json::json!({
                            "id": request["id"],
                            "result": { "status": "success", "data": { "version": version } }
                        });
                        let _ = ws.send(Message::Text(reply.to_string().into())).await;
                    }
                });
            }
        });
        port
    }

    /// Open a start attempt the way `start` does, and return it.
    async fn claimed(manager: &DaemonManager) -> u64 {
        manager
            .claim_start(manager.stop_epoch())
            .await
            .expect("no stop since")
            .expect("nothing under way")
    }

    /// Two callers never both spawn, and a stop wins over every start that
    /// was decided on before it.
    #[tokio::test]
    async fn a_start_is_claimed_once_and_an_older_start_loses_to_a_stop() {
        let manager = manager_on(free_port().await);
        let before_stop = manager.stop_epoch();
        assert_eq!(manager.claim_start(before_stop).await, Ok(Some(1)));
        assert_eq!(
            manager.claim_start(before_stop).await,
            Ok(None),
            "a start under way is not claimed again"
        );
        assert_eq!(manager.state().await, DaemonState::Starting);

        manager.stop().await.unwrap();
        assert_eq!(manager.state().await, DaemonState::Stopped);
        assert!(!manager.is_starting(1).await, "the stop cancels the boot in progress");
        assert_eq!(
            manager.claim_start(before_stop).await,
            Err(START_CANCELLED.to_string()),
            "a start decided on before the stop must not undo it"
        );
        assert_eq!(manager.claim_start(manager.stop_epoch()).await, Ok(Some(2)));
    }

    /// 2026-09-18: why a boot failed was only a log line. The exit is kept,
    /// with the daemon's own reason, until a daemon is ready.
    #[tokio::test]
    async fn a_boot_exit_is_the_last_error_until_a_daemon_is_ready() {
        let manager = manager_on(free_port().await);
        let attempt = claimed(&manager).await;
        let watch = manager.new_spawn_watch().await;
        let events = manager.sidecar_events(attempt, Arc::clone(&watch));
        for text in PID_FILE_TAKEN {
            events.watch.relay(BootStream::Stdout, format!("{text}\n").as_bytes()).await;
        }
        events.on_exit(Some(1), None).await;

        // During a boot the exit decides nothing by itself: the ready-wait
        // does, since a daemon our sidecar deferred to may be answering.
        assert!(watch.exited.load(Ordering::SeqCst));
        assert_eq!(manager.state().await, DaemonState::Starting);
        assert_eq!(manager.last_failure().await, None);

        // The ready-wait found nobody answering: what `start` does next.
        let exit = watch.exit.lock().await.clone().expect("the exit is recorded");
        let failure = StartFailure::from_exit(StartFailureKind::ExitedDuringBoot, &exit);
        manager.fail_attempt(attempt, failure).await;
        assert_eq!(manager.state().await, DaemonState::Crashed);
        let failure = manager.last_failure().await.expect("the failure is recorded");
        assert_eq!(failure.kind, StartFailureKind::ExitedDuringBoot);
        assert_eq!(failure.message, "Error: Already running");
        assert_eq!((failure.exit_code, failure.signal), (Some(1), None));

        // A retry keeps it on show while it boots, and a daemon that answers
        // clears it.
        let retry = claimed(&manager).await;
        let retry_watch = manager.new_spawn_watch().await;
        assert_eq!(manager.last_failure().await, Some(failure));
        assert_eq!(manager.mark_ready(retry, &retry_watch, false).await, Marked::Running);
        assert_eq!(manager.state().await, DaemonState::Running);
        assert_eq!(manager.last_failure().await, None);
    }

    /// The answer came from a sidecar that exited while the probe ran: it
    /// may have been its last. Marking it `Running` left a dead daemon
    /// reading `running`, with nothing on record and nothing to restart it.
    #[tokio::test]
    async fn an_answer_from_a_sidecar_that_exited_meanwhile_is_not_ready_yet() {
        let manager = manager_on(free_port().await);
        let attempt = claimed(&manager).await;
        let watch = manager.new_spawn_watch().await;
        watch.exited.store(true, Ordering::SeqCst);

        assert_eq!(manager.mark_ready(attempt, &watch, false).await, Marked::ExitedSinceProbe);
        assert_eq!(manager.state().await, DaemonState::Starting, "left to the ready-wait");
        // A probe that began after the exit is trustworthy: the daemon that
        // answered is one our sidecar deferred to.
        assert_eq!(manager.mark_ready(attempt, &watch, true).await, Marked::Running);
    }

    /// The whole ready-wait: a daemon answers, and its sidecar exits during
    /// that very probe, taking the daemon with it. The wait looks again and
    /// reports the failed boot.
    #[tokio::test]
    async fn a_daemon_that_answers_and_dies_at_once_is_a_failed_boot() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let manager = manager_on(listener.local_addr().unwrap().port());
        let attempt = claimed(&manager).await;
        let watch = manager.new_spawn_watch().await;
        tokio::spawn({
            let watch = Arc::clone(&watch);
            async move {
                let (stream, _) = listener.accept().await.unwrap();
                // Nothing answers after this connection.
                drop(listener);
                watch.exited.store(true, Ordering::SeqCst);
                let mut ws = tokio_tungstenite::accept_async(stream).await.unwrap();
                if let Some(Ok(Message::Text(request))) = ws.next().await {
                    let request: serde_json::Value = serde_json::from_str(&request).unwrap();
                    let reply = serde_json::json!({
                        "id": request["id"],
                        "result": { "status": "success", "data": { "version": env!("CARGO_PKG_VERSION") } }
                    });
                    let _ = ws.send(Message::Text(reply.to_string().into())).await;
                }
            }
        });

        assert_eq!(manager.wait_for_ready(attempt, &watch).await, BootWait::Failed);
        assert_eq!(manager.state().await, DaemonState::Starting, "start() records the failure next");
    }

    /// The contract the splash reads: `snake_case` kinds, nullable exit fields.
    #[test]
    fn a_failure_serializes_as_the_status_reports_it() {
        let exit = SidecarExit { code: None, signal: Some(9), ..SidecarExit::default() };
        let failure = StartFailure::from_exit(StartFailureKind::ExitedAfterReady, &exit);
        let json = serde_json::to_value(&failure).unwrap();
        assert_eq!(json["kind"], "exited_after_ready");
        assert_eq!(json["message"], "The daemon exited (signal 9)");
        assert_eq!(json["exit_code"], serde_json::Value::Null);
        assert_eq!(json["signal"], 9);
        assert!(json["at_ms"].as_i64().is_some_and(|at| at > 0));
        let kinds = [
            (StartFailureKind::SidecarUnresolved, "sidecar_unresolved"),
            (StartFailureKind::SpawnFailed, "spawn_failed"),
            (StartFailureKind::ExitedDuringBoot, "exited_during_boot"),
            (StartFailureKind::HealthCheckFailed, "health_check_failed"),
            (StartFailureKind::RestartsExhausted, "restarts_exhausted"),
        ];
        for (kind, name) in kinds {
            assert_eq!(serde_json::to_value(kind).unwrap(), name);
        }
    }

    #[test]
    fn without_a_reason_the_message_is_the_exit_status() {
        let exit = |code, signal| SidecarExit { code, signal, ..SidecarExit::default() };
        let message = |kind, exit| StartFailure::from_exit(kind, &exit).message;
        assert_eq!(
            message(StartFailureKind::ExitedDuringBoot, exit(Some(1), None)),
            "The daemon exited during startup (exit code 1)"
        );
        assert_eq!(message(StartFailureKind::ExitedAfterReady, exit(None, Some(9))), "The daemon exited (signal 9)");
        assert_eq!(message(StartFailureKind::ExitedAfterReady, exit(None, None)), "The daemon exited");
    }

    /// The late `AlreadyRunning` exit (fix (a) of the splash work): the app
    /// attached to a daemon it did not spawn, then its own sidecar deferred.
    /// The state used to read `crashed` while connected, for up to 30 s.
    #[test]
    fn only_an_exit_while_running_with_nobody_answering_is_a_crash() {
        assert_eq!(exit_effect(true, DaemonState::Running, false), ExitEffect::Crash);
        assert_eq!(exit_effect(true, DaemonState::Running, true), ExitEffect::StayAttached);
        for answering in [false, true] {
            assert_eq!(exit_effect(true, DaemonState::Starting, answering), ExitEffect::EndsBoot);
            assert_eq!(exit_effect(true, DaemonState::Crashed, answering), ExitEffect::EndsReport);
            for state in [DaemonState::Stopping, DaemonState::Stopped] {
                assert_eq!(exit_effect(true, state, answering), ExitEffect::Nothing);
            }
        }
        for state in [
            DaemonState::Stopped,
            DaemonState::Starting,
            DaemonState::Running,
            DaemonState::Stopping,
            DaemonState::Crashed,
        ] {
            assert_eq!(
                exit_effect(false, state, false),
                ExitEffect::Nothing,
                "a sidecar from an older attempt owns nothing"
            );
        }
    }

    #[tokio::test]
    async fn our_sidecar_deferring_after_the_attach_leaves_the_app_attached() {
        let manager = manager_on(serve_daemon(env!("CARGO_PKG_VERSION")).await);
        let attempt = claimed(&manager).await;
        let watch = manager.new_spawn_watch().await;
        assert_eq!(manager.mark_ready(attempt, &watch, false).await, Marked::Running);

        manager.sidecar_events(attempt, Arc::clone(&watch)).on_exit(Some(1), None).await;

        assert_eq!(manager.state().await, DaemonState::Running);
        assert_eq!(manager.last_failure().await, None);
        assert!(watch.exited.load(Ordering::SeqCst), "stop() must leave that daemon alone");
    }

    #[tokio::test]
    async fn the_serving_sidecar_exiting_is_a_crash_with_its_exit_status() {
        let manager = manager_on(free_port().await);
        let attempt = claimed(&manager).await;
        let watch = manager.new_spawn_watch().await;
        assert_eq!(manager.mark_ready(attempt, &watch, false).await, Marked::Running);
        let events = manager.sidecar_events(attempt, Arc::clone(&watch));
        events
            .watch
            .relay(BootStream::Stdout, b"2026-09-18T17:30:00.000000Z  INFO nanna_core::scheduler: tick\n")
            .await;

        events.on_exit(None, Some(9)).await;

        assert_eq!(manager.state().await, DaemonState::Crashed);
        let failure = manager.last_failure().await.expect("the crash is recorded");
        assert_eq!(failure.kind, StartFailureKind::ExitedAfterReady);
        assert_eq!(failure.message, "The daemon exited (signal 9)");
        assert_eq!((failure.exit_code, failure.signal), (None, Some(9)));
    }

    /// The monitor reported the daemon alive but not serving, then its
    /// process exited. The report, which says the process is still running,
    /// used to stay on record through the restart's boot.
    #[tokio::test]
    async fn the_exit_of_a_daemon_reported_not_serving_replaces_the_report() {
        let manager = manager_on(free_port().await);
        let attempt = claimed(&manager).await;
        let watch = manager.new_spawn_watch().await;
        assert_eq!(manager.mark_ready(attempt, &watch, false).await, Marked::Running);
        let report = StartFailure::new(
            StartFailureKind::HealthCheckFailed,
            "The daemon stopped answering · its process (PID 7) is still running".to_string(),
        );
        manager.not_serving(manager.stop_epoch(), report).await;
        assert_eq!(manager.state().await, DaemonState::Crashed);

        manager.sidecar_events(attempt, watch).on_exit(None, Some(9)).await;

        assert_eq!(manager.state().await, DaemonState::Crashed);
        let failure = manager.last_failure().await.expect("recorded");
        assert_eq!(failure.kind, StartFailureKind::ExitedAfterReady);
        assert_eq!(failure.message, "The daemon exited (signal 9)");
    }

    /// The event task flags the exit before it reads the state, so whoever
    /// sees the flag in between can record the exit first: the ready-wait
    /// (a boot that exited) or the health monitor (a daemon that is gone,
    /// and no restarts left). The event task then read `Crashed` and took it
    /// for the monitor's report of a live daemon, and "exited after ready"
    /// replaced what was on record.
    #[tokio::test]
    async fn an_exit_already_on_record_is_not_replaced() {
        let manager = manager_on(free_port().await);
        let attempt = claimed(&manager).await;
        let events = manager.sidecar_events(attempt, manager.new_spawn_watch().await);
        let boot = StartFailure::new(
            StartFailureKind::ExitedDuringBoot,
            "Error: Already running".to_string(),
        );
        manager.fail_attempt(attempt, boot.clone()).await;
        events.on_exit(Some(1), None).await;
        assert_eq!(manager.last_failure().await, Some(boot), "the boot never became ready");

        let manager = manager_on(free_port().await);
        let attempt = claimed(&manager).await;
        let watch = manager.new_spawn_watch().await;
        assert_eq!(manager.mark_ready(attempt, &watch, false).await, Marked::Running);
        *manager.restart_count.write().await = manager.config.max_restarts;
        let check = StartFailure::new(StartFailureKind::HealthCheckFailed, "check".to_string());
        assert_eq!(manager.on_failed_check(manager.stop_epoch(), check).await, FailedCheck::GaveUp);
        let gave_up = manager.last_failure().await;
        manager.sidecar_events(attempt, watch).on_exit(None, Some(9)).await;
        assert_eq!(manager.last_failure().await, gave_up, "giving up is the news");
    }

    /// What stdio MCP servers print on the daemon's stderr, which they share.
    const MCP_CHATTER: [&str; 2] = [
        "Secure MCP Filesystem Server running on stdio",
        "Allowed directories: [ '/home/user' ]",
    ];

    /// The sidecar inherits the app's `RUST_LOG`, which can filter out the
    /// daemon's whole log (`RUST_LOG=nanna_gui=debug` names no daemon
    /// target), so the daemon may write no stdout line at all. The first
    /// line on its stderr, an MCP server's, was then taken for why a daemon
    /// that had started its servers exited.
    #[tokio::test]
    async fn a_daemon_whose_log_is_filtered_out_is_not_explained_by_its_mcp_servers() {
        let chatter = |events: &SidecarEvents| {
            let watch = Arc::clone(&events.watch);
            async move {
                for text in MCP_CHATTER {
                    watch.relay(BootStream::Stderr, format!("{text}\n").as_bytes()).await;
                }
            }
        };

        // Killed, and exited with a code, while it served.
        for (code, signal, expected) in [
            (None, Some(9), "The daemon exited (signal 9)"),
            (Some(1), None, "The daemon exited (exit code 1)"),
        ] {
            let manager = manager_on(free_port().await);
            let attempt = claimed(&manager).await;
            let watch = manager.new_spawn_watch().await;
            assert_eq!(manager.mark_ready(attempt, &watch, false).await, Marked::Running);
            let events = manager.sidecar_events(attempt, watch);
            chatter(&events).await;
            events.on_exit(code, signal).await;
            let failure = manager.last_failure().await.expect("the crash is recorded");
            assert_eq!(failure.message, expected);
        }

        // Killed during its boot: a restart of a boot that hung after it
        // started its servers.
        let manager = manager_on(free_port().await);
        let attempt = claimed(&manager).await;
        let watch = manager.new_spawn_watch().await;
        let events = manager.sidecar_events(attempt, Arc::clone(&watch));
        chatter(&events).await;
        events.on_exit(None, Some(9)).await;
        let exit = watch.exit.lock().await.clone().expect("the exit is recorded");
        assert_eq!(
            StartFailure::from_exit(StartFailureKind::ExitedDuringBoot, &exit).message,
            "The daemon exited during startup (signal 9)"
        );
    }

    /// A stop that finds nothing to stop still reads `Stopping` while it
    /// runs, so a start decided on meanwhile is refused. It used to return
    /// at once and count itself done afterwards: a start that claimed in
    /// between got a boot that nothing cancelled.
    #[tokio::test]
    async fn a_stop_with_nothing_to_stop_still_refuses_a_start_decided_on_meanwhile() {
        let manager = Arc::new(manager_on(free_port().await));
        // Holding the child slot holds the stop in its middle.
        let slot = manager.child.write().await;
        let stop = tokio::spawn({
            let manager = Arc::clone(&manager);
            async move { manager.stop().await }
        });
        tokio::time::timeout(Duration::from_secs(1), async {
            while manager.state().await != DaemonState::Stopping {
                sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("the stop reads Stopping");
        let meanwhile = manager.stop_epoch();
        assert_eq!(manager.claim_start(meanwhile).await, Err(START_CANCELLED.to_string()));

        drop(slot);
        stop.await.unwrap().unwrap();
        assert_eq!(manager.state().await, DaemonState::Stopped);
        assert_eq!(manager.claim_start(meanwhile).await, Err(START_CANCELLED.to_string()));
        assert!(manager.claim_start(manager.stop_epoch()).await.is_ok_and(|claim| claim.is_some()));
    }

    /// A restart kills the old sidecar and begins the next boot at once. The
    /// old sidecar's exit and late output arrive during that boot and must
    /// not fail it or mix into its log.
    #[tokio::test]
    async fn an_older_sidecar_cannot_touch_the_next_boot() {
        let manager = manager_on(free_port().await);
        let old = claimed(&manager).await;
        let old_events = manager.sidecar_events(old, manager.new_spawn_watch().await);
        manager.stop().await.unwrap();
        let new = claimed(&manager).await;
        let new_watch = manager.new_spawn_watch().await;

        old_events.watch.relay(BootStream::Stdout, b"a late line from the old sidecar\n").await;
        old_events.on_exit(None, Some(9)).await;

        assert!(manager.is_starting(new).await);
        assert_eq!(manager.last_failure().await, None);
        assert!(!new_watch.exited.load(Ordering::SeqCst));
        assert_eq!(manager.boot_log().await, Vec::new());
    }

    /// A killed sidecar blocked in disk I/O exits only once the I/O is done,
    /// and holds the PID file until then: a daemon spawned beside it exited
    /// "Already running". The next spawn waits for the exit, and a stop ends
    /// that wait.
    #[tokio::test]
    async fn the_next_spawn_waits_for_the_previous_sidecar_to_exit() {
        let manager = manager_on(free_port().await);
        let killed = manager.new_spawn_watch().await;
        let attempt = claimed(&manager).await;
        let began = tokio::time::Instant::now();
        let (gone, ()) = tokio::join!(manager.previous_sidecar_gone(attempt), async {
            sleep(Duration::from_millis(150)).await;
            killed.exited.store(true, Ordering::SeqCst);
        });
        assert!(gone);
        assert!(began.elapsed() >= Duration::from_millis(150), "it waited for the exit");

        manager.stop().await.unwrap();
        let _stuck = manager.new_spawn_watch().await;
        let attempt = claimed(&manager).await;
        let (gone, ()) = tokio::join!(manager.previous_sidecar_gone(attempt), async {
            sleep(Duration::from_millis(50)).await;
            manager.stop().await.unwrap();
        });
        assert!(!gone, "a stop ends the wait");
    }

    #[tokio::test]
    async fn the_boot_log_keeps_the_newest_lines_of_the_latest_spawn() {
        let manager = manager_on(free_port().await);
        assert_eq!(manager.boot_log().await, Vec::new(), "nothing spawned yet");

        let events = manager.sidecar_events(1, manager.new_spawn_watch().await);
        for n in 0..BOOT_LOG_LINES + 3 {
            events.watch.relay(BootStream::Stdout, format!("line {n}\n").as_bytes()).await;
        }
        events.watch.relay(BootStream::Stderr, b"\r\n").await;
        let log = manager.boot_log().await;
        assert_eq!(log.len(), BOOT_LOG_LINES, "bounded, and a blank line takes no slot");
        assert_eq!(log[0].line, "line 3");
        assert_eq!(log[BOOT_LOG_LINES - 1].line, format!("line {}", BOOT_LOG_LINES + 2));

        manager.new_spawn_watch().await;
        assert_eq!(manager.boot_log().await, Vec::new(), "each spawn starts a new log");
    }

    #[tokio::test]
    async fn relayed_lines_are_plain_text_tagged_with_their_stream() {
        let manager = manager_on(free_port().await);
        let events = manager.sidecar_events(1, manager.new_spawn_watch().await);
        events.watch.relay(BootStream::Stdout, COLOURED.as_bytes()).await;
        events
            .watch
            .relay(BootStream::Stderr, b"thread 'main' panicked at src/main.rs:1:2:\r\n")
            .await;
        assert_eq!(
            manager.boot_log().await,
            [
                line(
                    BootStream::Stdout,
                    "2026-09-18T17:21:21.595409Z  INFO nanna_daemon: Starting Nanna daemon..."
                ),
                line(BootStream::Stderr, "thread 'main' panicked at src/main.rs:1:2:"),
            ]
        );
    }

    #[test]
    fn plain_text_drops_escape_sequences_and_control_characters() {
        assert_eq!(plain_text("a\u{1b}[1;31mred\u{1b}[0m b"), "ared b");
        // OSC, ended by BEL and by the string terminator.
        assert_eq!(plain_text("\u{1b}]0;title\u{7}x\u{1b}]8;;http://a\u{1b}\\y"), "xy");
        // Two-character escapes, and a lone ESC at the end.
        assert_eq!(plain_text("\u{1b}(Bz\u{1b}"), "z");
        // Tabs stay; line endings, BEL and C1 controls go; text is untouched.
        assert_eq!(plain_text("k\tv\u{7}\u{85}é — ok\r\n"), "k\tvé — ok");
    }

    #[test]
    fn an_overlong_line_is_cut_on_a_character_boundary_and_says_so() {
        let whole = "x".repeat(BOOT_LOG_LINE_BYTES);
        assert_eq!(fit_line(whole.clone()), whole);

        // "é" is two bytes, and this one straddles the cut.
        let head = "x".repeat(BOOT_LOG_LINE_BYTES - 1);
        let long = format!("{head}é{}", "y".repeat(10));
        assert_eq!(fit_line(long), format!("{head}… (12 more bytes)"));
    }

    #[test]
    fn the_reason_is_the_daemons_own_last_error_line() {
        assert_eq!(exit_reason(&stdout(&PID_FILE_TAKEN)).as_deref(), Some("Error: Already running"));
        assert_eq!(
            exit_reason(&stdout(&PORT_TAKEN)).as_deref(),
            Some("Error: IPC error: IPC port 127.0.0.1:51993 unavailable: Address already in use (os error 98)")
        );
    }

    #[test]
    fn a_panic_is_reported_by_the_daemons_panic_line_over_stderr() {
        let mut lines = stdout(&[
            "2026-09-18T17:30:00.000000Z ERROR nanna_daemon::server: PANIC: boom location=src/server.rs:1:2",
        ]);
        lines.push_back(line(BootStream::Stderr, "thread 'main' panicked at src/server.rs:1:2:"));
        lines.push_back(line(BootStream::Stderr, "boom"));
        lines.push_back(line(
            BootStream::Stderr,
            "note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace",
        ));
        assert_eq!(
            exit_reason(&lines).as_deref(),
            Some("PANIC: boom location=src/server.rs:1:2")
        );

        // Without the daemon's line (a release daemon aborts at once), the
        // panic on stderr with its message, never the hint after it, nor what
        // an MCP server says after the daemon died.
        lines.pop_front();
        lines.push_front(line(BootStream::Stdout, "2026-09-18T17:30:00.000000Z  INFO nanna: working"));
        lines.push_back(line(BootStream::Stderr, "Secure MCP Filesystem Server running on stdio"));
        assert_eq!(
            exit_reason(&lines).as_deref(),
            Some("thread 'main' panicked at src/server.rs:1:2: boom")
        );
    }

    #[test]
    fn output_that_ends_in_ordinary_work_gives_no_reason() {
        // A signal mid-work: the exit status is the whole story.
        let mut lines = VecDeque::from([line(BootStream::Stderr, "an early warning")]);
        lines.push_back(line(BootStream::Stdout, "2026-09-18T17:30:00.000000Z  INFO nanna: ERROR is just a word here"));
        assert_eq!(exit_reason(&lines), None);
        assert_eq!(exit_reason(&VecDeque::new()), None);
    }

    /// The daemon passes its stderr on to the MCP servers it starts, and
    /// one goes on talking there after the daemon was killed. That used to
    /// be taken for the reason the daemon exited.
    #[test]
    fn an_mcp_server_talking_on_stderr_is_not_the_reason() {
        let mut lines = stdout(&["2026-09-18T17:30:00.000000Z  INFO nanna_mcp: Connected to filesystem"]);
        lines.push_back(line(BootStream::Stderr, "Secure MCP Filesystem Server running on stdio"));
        lines.push_back(line(BootStream::Stderr, "Error: stdin closed, shutting down"));
        assert_eq!(exit_reason(&lines), None);
        let exit = SidecarExit::new(None, Some(9), &lines);
        assert_eq!(
            StartFailure::from_exit(StartFailureKind::ExitedAfterReady, &exit).message,
            "The daemon exited (signal 9)"
        );
    }

    /// A program that stopped at its command line (a daemon too old for an
    /// argument, say) says why first, then prints usage. One the loader
    /// could not start says why and nothing else.
    #[test]
    fn a_boot_that_never_logged_is_stopped_by_the_first_thing_it_said() {
        let boot_failure = |texts: &[&str], code| {
            let lines = texts.iter().map(|text| line(BootStream::Stderr, text)).collect();
            let exit = SidecarExit::new(Some(code), None, &lines);
            StartFailure::from_exit(StartFailureKind::ExitedDuringBoot, &exit).message
        };
        let usage = [
            "error: unexpected argument '--exit-with-parent' found",
            "Usage: nanna-daemon [OPTIONS] <COMMAND>",
            "For more information, try '--help'.",
        ];
        assert_eq!(boot_failure(&usage, 2), usage[0]);
        let loader = "nanna-daemon: error while loading shared libraries: libx.so: cannot open shared object file";
        assert_eq!(boot_failure(&[loader], 127), loader);
    }

    #[test]
    fn rusts_fatal_runtime_errors_are_reasons() {
        let fatal = |texts: &[&str]| runtime_fatal(texts.iter().copied());
        assert_eq!(
            fatal(&["thread 'main' has overflowed its stack", "fatal runtime error: stack overflow"]).as_deref(),
            Some("thread 'main' has overflowed its stack")
        );
        assert_eq!(
            fatal(&["memory allocation of 1024 bytes failed"]).as_deref(),
            Some("memory allocation of 1024 bytes failed")
        );
        // An older toolchain's one-line panic, and a panic whose message
        // never came.
        assert_eq!(
            fatal(&["thread 'main' panicked at 'boom', src/main.rs:1:2"]).as_deref(),
            Some("thread 'main' panicked at 'boom', src/main.rs:1:2")
        );
        assert_eq!(
            fatal(&["thread 'x' panicked at src/a.rs:1:2:", "note: run with `RUST_BACKTRACE=1`"]).as_deref(),
            Some("thread 'x' panicked at src/a.rs:1:2:")
        );
        assert_eq!(fatal(&["Error: stdin closed", "it panicked"]), None);
    }

    #[test]
    fn an_error_line_is_read_with_or_without_its_timestamp() {
        assert_eq!(error_message("ERROR nanna_daemon: Error: x"), Some("Error: x"));
        assert_eq!(error_message("2026-09-18T17:30:00Z ERROR no target here"), Some("no target here"));
        assert_eq!(error_message("2026-09-18T17:30:00Z  WARN nanna: ERROR soon"), None);
        assert_eq!(error_message("2026-09-18T17:30:00Z ERROR"), None);
    }

    // -------------------------------------------------------------------------
    // Health monitor
    // -------------------------------------------------------------------------

    /// A port that accepts TCP but never completes the handshake used to
    /// hang the health monitor for good.
    #[tokio::test]
    async fn a_health_check_that_gets_no_handshake_times_out() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("ws://{}", listener.local_addr().unwrap());
        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                // Hold the connection open and never answer.
                tokio::spawn(async move {
                    let _stream = stream;
                    std::future::pending::<()>().await;
                });
            }
        });

        let started = std::time::Instant::now();
        let result = check_health(&url, Duration::from_millis(300)).await;
        assert_eq!(result, Err(CheckFailure::NoAnswer(Duration::from_millis(300))));
        assert_eq!(result.unwrap_err().to_string(), "no answer within 300ms");
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    #[tokio::test]
    async fn a_health_check_passes_against_a_daemon_and_fails_on_a_closed_port() {
        let serving = serve_daemon(env!("CARGO_PKG_VERSION")).await;
        assert_eq!(check_health(&format!("ws://127.0.0.1:{serving}"), PROBE_TIMEOUT).await, Ok(()));
        let closed = free_port().await;
        assert!(matches!(
            check_health(&format!("ws://127.0.0.1:{closed}"), PROBE_TIMEOUT).await,
            Err(CheckFailure::Failed(_))
        ));
    }

    /// Only a daemon that is gone is restarted. One that is there, alive but
    /// not answering, would only make a new one fail beside it.
    #[test]
    fn only_a_failed_check_with_our_sidecar_gone_restarts() {
        let refused = CheckFailure::Failed("Connection refused (os error 111)".to_string());
        let silent = CheckFailure::NoAnswer(PROBE_TIMEOUT);
        assert_eq!(after_failed_check(&refused, None), AfterFailedCheck::Restart);
        assert_eq!(
            after_failed_check(&refused, Some(7)),
            AfterFailedCheck::Report { pid: 7 },
            "our live sidecar holds the PID file"
        );
        for sidecar in [None, Some(7)] {
            assert_eq!(
                after_failed_check(&silent, sidecar),
                AfterFailedCheck::CheckAgain,
                "something holds the port, and may only be slow"
            );
        }
    }

    /// A tick whose check outlived a stop must neither overwrite the stop's
    /// state nor restart the daemon behind it.
    #[tokio::test]
    async fn a_health_check_that_outlived_a_stop_changes_nothing() {
        let manager = manager_on(free_port().await);
        let attempt = claimed(&manager).await;
        let watch = manager.new_spawn_watch().await;
        assert_eq!(manager.mark_ready(attempt, &watch, false).await, Marked::Running);
        let tick = manager.stop_epoch();
        manager.stop().await.unwrap();

        let failure = StartFailure::new(StartFailureKind::HealthCheckFailed, "x".to_string());
        assert_eq!(manager.on_failed_check(tick, failure).await, FailedCheck::NotOurs);
        assert_eq!(manager.state().await, DaemonState::Stopped);
        assert_eq!(manager.last_failure().await, None);
        assert!(manager.claim_start(tick).await.is_err(), "its restart is refused");
    }

    #[tokio::test]
    async fn a_failed_check_is_recorded_and_an_answer_clears_it() {
        let manager = manager_on(free_port().await);
        let attempt = claimed(&manager).await;
        let watch = manager.new_spawn_watch().await;
        assert_eq!(manager.mark_ready(attempt, &watch, false).await, Marked::Running);
        let tick = manager.stop_epoch();

        let failure = StartFailure::new(StartFailureKind::HealthCheckFailed, "x".to_string());
        assert_eq!(manager.on_failed_check(tick, failure.clone()).await, FailedCheck::Restart(1));
        assert_eq!(manager.state().await, DaemonState::Crashed);
        assert_eq!(manager.last_failure().await, Some(failure));

        manager.recovered(tick).await;
        assert_eq!(manager.state().await, DaemonState::Running);
        assert_eq!(manager.last_failure().await, None);
    }

    /// The retry loop attached to a daemon after a boot failed. The status
    /// read connected and crashed, with the boot's error, until the next
    /// health check up to 30 s later.
    #[tokio::test]
    async fn an_attach_clears_a_crash_at_once() {
        let manager = manager_on(free_port().await);
        let exit = SidecarExit { code: Some(1), ..SidecarExit::default() };
        manager
            .crash_for_test(StartFailure::from_exit(StartFailureKind::ExitedDuringBoot, &exit))
            .await;

        manager.attached().await;
        assert_eq!(manager.state().await, DaemonState::Running);
        assert_eq!(manager.last_failure().await, None);

        // A stop's state is the stop's, and a boot's is its ready-wait's.
        manager.stop().await.unwrap();
        manager.attached().await;
        assert_eq!(manager.state().await, DaemonState::Stopped);
        let attempt = claimed(&manager).await;
        manager.attached().await;
        assert!(manager.is_starting(attempt).await);
    }

    /// A boot that failed stays explained while the monitor retries it.
    #[tokio::test]
    async fn a_failed_check_on_a_crashed_daemon_keeps_the_reason_on_record() {
        let manager = manager_on(free_port().await);
        let attempt = claimed(&manager).await;
        let exit = SidecarExit {
            code: Some(1),
            reason: Some("Error: Already running".to_string()),
            ..SidecarExit::default()
        };
        let boot = StartFailure::from_exit(StartFailureKind::ExitedDuringBoot, &exit);
        manager.fail_attempt(attempt, boot.clone()).await;

        let check = StartFailure::new(StartFailureKind::HealthCheckFailed, "x".to_string());
        assert_eq!(
            manager.on_failed_check(manager.stop_epoch(), check).await,
            FailedCheck::Restart(1),
            "still a restart"
        );
        assert_eq!(manager.last_failure().await, Some(boot));
    }

    #[tokio::test]
    async fn giving_up_is_recorded_once_with_the_failure_before_it() {
        let manager = manager_on(free_port().await);
        let attempt = claimed(&manager).await;
        let spawn_failed = StartFailure::new(
            StartFailureKind::SpawnFailed,
            "Could not start nanna-daemon: No such file or directory (os error 2)".to_string(),
        );
        manager.fail_attempt(attempt, spawn_failed).await;
        *manager.restart_count.write().await = manager.config.max_restarts;
        let tick = manager.stop_epoch();
        let check = || StartFailure::new(StartFailureKind::HealthCheckFailed, "check".to_string());

        assert_eq!(manager.on_failed_check(tick, check()).await, FailedCheck::GaveUp);
        let gave_up = manager.last_failure().await.expect("recorded");
        assert_eq!(gave_up.kind, StartFailureKind::RestartsExhausted);
        assert_eq!(
            gave_up.message,
            "Stopped restarting the daemon after 3 failed restarts · Could not start nanna-daemon: No such file or directory (os error 2)"
        );

        assert_eq!(manager.on_failed_check(tick, check()).await, FailedCheck::GaveUp);
        assert_eq!(manager.last_failure().await, Some(gave_up), "the first record stays");
    }

    /// A restart resets the budget while a tick is between its check and
    /// its count. The count used to be read before and written after, so
    /// the stale count came back and the next failure gave up at once.
    #[tokio::test]
    async fn a_budget_reset_is_never_undone_by_a_tick() {
        let manager = manager_on(free_port().await);
        let attempt = claimed(&manager).await;
        let watch = manager.new_spawn_watch().await;
        assert_eq!(manager.mark_ready(attempt, &watch, false).await, Marked::Running);
        *manager.restart_count.write().await = manager.config.max_restarts - 1;
        let check = || StartFailure::new(StartFailureKind::HealthCheckFailed, "check".to_string());

        manager.reset_restarts().await;
        assert_eq!(
            manager.on_failed_check(manager.stop_epoch(), check()).await,
            FailedCheck::Restart(1),
            "counted from the reset"
        );
    }
}

/// The sidecar lifecycle with a real process in the sidecar's place, spawned
/// through the shell plugin of a mock app: what `start` and the health
/// monitor do around a spawn, which the tests above drive in pieces.
#[cfg(all(test, unix))]
mod spawned {
    use super::*;
    use futures_util::{SinkExt, StreamExt};
    use std::sync::atomic::AtomicU8;
    use tauri::test::{MockRuntime, mock_builder, mock_context, noop_assets};
    use tokio_tungstenite::tungstenite::Message;

    /// What a sidecar printed when another daemon held the PID file
    /// (captured 2026-09-18 from a real second daemon on one data dir).
    const ALREADY_RUNNING: &str = "\
printf '%s\\n' '2026-09-18T17:21:44.208739Z ERROR nanna_daemon::server: Another daemon instance is already running (PID: 298915)'
printf '%s\\n' '2026-09-18T17:21:44.209179Z ERROR nanna_daemon: Error: Already running'
exit 1";

    /// A daemon whose port stays closed: a boot, or a daemon, that never
    /// answers. Bounded, so a failed test leaves nothing running for long.
    const HANGS: &str = "exec sleep 60";

    fn mock_app() -> tauri::App<MockRuntime> {
        mock_builder()
            .plugin(tauri_plugin_shell::init())
            .build(mock_context(noop_assets()))
            .expect("the mock app builds")
    }

    /// What the test's daemon port does with a connection.
    const SERVE: u8 = 0;
    const SILENT: u8 = 1;
    const DROP: u8 = 2;

    /// A daemon port whose behaviour a test switches: `SERVE` answers like a
    /// daemon of this version, `SILENT` holds the connection and never
    /// answers, `DROP` closes it at once, so every probe finds nobody.
    struct Port {
        number: u16,
        mode: Arc<AtomicU8>,
    }

    impl Port {
        async fn open(mode: u8) -> Self {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let number = listener.local_addr().unwrap().port();
            let mode = Arc::new(AtomicU8::new(mode));
            let serving = Arc::clone(&mode);
            tokio::spawn(async move {
                while let Ok((stream, _)) = listener.accept().await {
                    match serving.load(Ordering::SeqCst) {
                        SERVE => {
                            tokio::spawn(async move {
                                let Ok(mut ws) = tokio_tungstenite::accept_async(stream).await else {
                                    return;
                                };
                                while let Some(Ok(Message::Text(request))) = ws.next().await {
                                    let request: serde_json::Value = serde_json::from_str(&request).unwrap();
                                    let reply = serde_json::json!({
                                        "id": request["id"],
                                        "result": {
                                            "status": "success",
                                            "data": { "version": env!("CARGO_PKG_VERSION") }
                                        }
                                    });
                                    let _ = ws.send(Message::Text(reply.to_string().into())).await;
                                }
                            });
                        }
                        SILENT => {
                            tokio::spawn(async move {
                                let _stream = stream;
                                std::future::pending::<()>().await;
                            });
                        }
                        _ => drop(stream),
                    }
                }
            });
            Self { number, mode }
        }

        fn set(&self, mode: u8) {
            self.mode.store(mode, Ordering::SeqCst);
        }
    }

    /// A manager on `port` that runs `script` with `/bin/sh` in the
    /// sidecar's place. A script runs as `sh -c` rather than from a file:
    /// executing a file a parallel test has just written can fail with
    /// "Text file busy".
    fn manager_running(script: &str, port: &Port) -> DaemonManager {
        let manager = DaemonManager::new(DaemonManagerConfig {
            port: port.number,
            restart_delay: Duration::from_millis(10),
            health_check_timeout: Duration::from_millis(300),
            ..DaemonManagerConfig::default()
        });
        manager.replace_sidecar_with(&["/bin/sh", "-c", script, "nanna-daemon"]);
        manager
    }

    /// The record of the most recent spawn.
    async fn latest(manager: &DaemonManager) -> Arc<SpawnWatch> {
        manager.latest_spawn.read().await.clone().expect("a spawn happened")
    }

    /// Stop the manager and wait for its sidecar's exit, so no test leaves
    /// a process behind.
    async fn stop_and_reap(manager: &DaemonManager) {
        let watch = latest(manager).await;
        manager.stop().await.unwrap();
        assert!(wait_for_terminated(&watch, Duration::from_secs(5)).await, "the sidecar exited");
    }

    /// `start` records a boot's exit with the daemon's own reason, and the
    /// sidecar ran with `NO_COLOR`. Then a spawn that fails is recorded, and
    /// starts a log of its own.
    #[tokio::test]
    async fn start_records_why_a_boot_failed_and_each_spawn_starts_a_new_log() {
        let app = mock_app();
        let port = Port::open(DROP).await;
        let script = format!("echo \"NO_COLOR=$NO_COLOR\"\n{ALREADY_RUNNING}");
        let manager = manager_running(&script, &port);

        let result = manager.start(app.handle(), manager.stop_epoch()).await;
        assert_eq!(result, Err("Daemon exited during startup".to_string()));
        assert_eq!(manager.state().await, DaemonState::Crashed);
        let failure = manager.last_failure().await.expect("the exit is recorded");
        assert_eq!(failure.kind, StartFailureKind::ExitedDuringBoot);
        assert_eq!(failure.message, "Error: Already running");
        assert_eq!((failure.exit_code, failure.signal), (Some(1), None));
        let log = manager.boot_log().await;
        assert_eq!(log.len(), 3);
        assert_eq!(log[0], BootLine { stream: BootStream::Stdout, line: "NO_COLOR=1".to_string() });

        manager.replace_sidecar_with(&["/nonexistent/nanna-daemon"]);
        let result = manager.start(app.handle(), manager.stop_epoch()).await;
        assert!(result.is_err_and(|e| e.starts_with("Failed to spawn daemon: ")));
        assert_eq!(manager.state().await, DaemonState::Crashed);
        let failure = manager.last_failure().await.expect("the spawn failure is recorded");
        assert_eq!(failure.kind, StartFailureKind::SpawnFailed);
        assert!(failure.message.starts_with("Could not start nanna-daemon: "));
        assert_eq!(manager.boot_log().await, Vec::new(), "the failed spawn has a log of its own");

        // Its record is not left waiting for an exit no process will make.
        manager.replace_sidecar_with(&["/bin/sh", "-c", ALREADY_RUNNING, "nanna-daemon"]);
        let next = tokio::time::timeout(PROBE_TIMEOUT * 2, manager.start(app.handle(), manager.stop_epoch()));
        assert!(next.await.expect("the next spawn does not wait").is_err());
    }

    /// The blocking finding of the splash review: a daemon we started that
    /// stops answering (wedged, or starved by a busy disk) is never
    /// restarted by the monitor. It used to be: the new sidecar took its
    /// slot, exited "Already running", and the old daemon ran on with nothing
    /// left to stop it.
    #[tokio::test]
    async fn the_health_monitor_never_replaces_our_live_daemon() {
        let app = mock_app();
        let port = Port::open(SERVE).await;
        let manager = manager_running(HANGS, &port);
        manager.start(app.handle(), manager.stop_epoch()).await.unwrap();
        let pid = manager.live_sidecar_pid().await.expect("the sidecar runs");

        // No answer in time: it may only be slow. Nothing changes.
        port.set(SILENT);
        manager.health_tick(app.handle()).await;
        assert_eq!(manager.state().await, DaemonState::Running);
        assert_eq!(manager.last_failure().await, None);

        // Refused: it is not serving, and the status says so.
        port.set(DROP);
        manager.health_tick(app.handle()).await;
        assert_eq!(manager.state().await, DaemonState::Crashed);
        let failure = manager.last_failure().await.expect("recorded");
        assert_eq!(failure.kind, StartFailureKind::HealthCheckFailed);
        assert!(
            failure.message.ends_with(&format!(" · its process (PID {pid}) is still running")),
            "{}",
            failure.message
        );
        assert_eq!(manager.live_sidecar_pid().await, Some(pid), "the same daemon, still ours");
        assert_eq!(*manager.restart_count.read().await, 0, "no restart was made");

        // It answers again.
        port.set(SERVE);
        manager.health_tick(app.handle()).await;
        assert_eq!(manager.state().await, DaemonState::Running);
        assert_eq!(manager.last_failure().await, None);

        port.set(DROP);
        stop_and_reap(&manager).await;
    }

    /// A retry of a daemon the monitor reported not serving: the live
    /// sidecar in the slot is killed, and has exited, before the next one
    /// starts.
    #[tokio::test]
    async fn a_new_spawn_never_leaves_a_live_sidecar_behind() {
        let app = mock_app();
        let port = Port::open(SERVE).await;
        let manager = manager_running(HANGS, &port);
        manager.start(app.handle(), manager.stop_epoch()).await.unwrap();
        let first = latest(&manager).await;
        let first_pid = manager.live_sidecar_pid().await.expect("the sidecar runs");

        port.set(DROP);
        manager.health_tick(app.handle()).await;
        assert_eq!(manager.state().await, DaemonState::Crashed, "reported, with our sidecar alive");
        port.set(SERVE);
        manager.start(app.handle(), manager.stop_epoch()).await.unwrap();

        assert!(first.exited.load(Ordering::SeqCst), "the first sidecar exited before the spawn");
        let exit = first.exit.lock().await.clone().expect("its exit is recorded");
        assert_eq!(exit.signal, Some(9), "killed");
        let second_pid = manager.live_sidecar_pid().await.expect("the new sidecar runs");
        assert_ne!(second_pid, first_pid);

        // Nothing to take a shutdown request, so the stop kills at once.
        port.set(DROP);
        stop_and_reap(&manager).await;
    }

    /// The monitor's call site: a daemon that is gone (nothing answers, and
    /// our sidecar is not running) is restarted, and the restart's outcome
    /// is what the status then reports.
    #[tokio::test]
    async fn the_health_monitor_restarts_a_daemon_that_is_gone() {
        let app = mock_app();
        let port = Port::open(SERVE).await;
        let manager = manager_running(ALREADY_RUNNING, &port);
        // Our sidecar deferred to the daemon already serving the port.
        manager.start(app.handle(), manager.stop_epoch()).await.unwrap();
        assert!(wait_for_terminated(&*latest(&manager).await, Duration::from_secs(5)).await);
        // The exit handler probes the port once more after the exit is
        // flagged. That probe is over within its timeout, and must meet the
        // daemon still serving.
        sleep(PROBE_TIMEOUT).await;
        assert_eq!(manager.state().await, DaemonState::Running);

        // A daemon that is there but silent is not restarted.
        port.set(SILENT);
        manager.health_tick(app.handle()).await;
        assert_eq!(manager.state().await, DaemonState::Running);
        assert_eq!(*manager.restart_count.read().await, 0);

        // That daemon is gone.
        port.set(DROP);
        manager.health_tick(app.handle()).await;
        assert_eq!(*manager.restart_count.read().await, 1);
        assert_eq!(manager.state().await, DaemonState::Crashed);
        let failure = manager.last_failure().await.expect("recorded");
        assert_eq!(failure.kind, StartFailureKind::ExitedDuringBoot, "the restart ran and failed");
        assert_eq!(failure.message, "Error: Already running");
    }

    /// A daemon that leaves a child holding its output open: the daemon
    /// passes its stderr on to every stdio MCP server it starts, and one that
    /// ignores stdin EOF outlives it. The child prints its PID, so the test
    /// can end it.
    const HOLDS_OUTPUT: &str = "sleep 30 &\necho \"holder=$!\"\nexec sleep 60";

    /// The PID the [`HOLDS_OUTPUT`] child printed.
    async fn holder_pid(manager: &DaemonManager) -> u32 {
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let pid = manager
                    .boot_log()
                    .await
                    .iter()
                    .find_map(|line| line.line.strip_prefix("holder=")?.parse().ok());
                if let Some(pid) = pid {
                    return pid;
                }
                sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("the child printed its PID")
    }

    /// Send `signal` to `pid`, as the OOM killer or the user would.
    fn signal(pid: u32, signal: &str) {
        let status = std::process::Command::new("kill")
            .args([signal, &pid.to_string()])
            .status()
            .expect("kill runs");
        assert!(status.success(), "kill {signal} {pid}");
    }

    /// The blocking finding of the second splash review: a killed boot whose
    /// child still held its output never counted as exited. The next boot
    /// waited for that exit with no deadline, and nothing ran or retried.
    #[tokio::test]
    async fn a_child_holding_the_output_does_not_hold_up_the_next_boot() {
        let app = mock_app();
        let port = Port::open(DROP).await;
        let manager = Arc::new(manager_running(HOLDS_OUTPUT, &port));
        let boot = tokio::spawn({
            let manager = Arc::clone(&manager);
            let app = app.handle().clone();
            async move { manager.start(&app, manager.stop_epoch()).await }
        });
        let holder = holder_pid(&manager).await;
        let hung = latest(&manager).await;

        // A restart: the boot is hung, so the stop kills it.
        manager.stop().await.unwrap();
        assert_eq!(boot.await.unwrap(), Err(START_CANCELLED.to_string()));
        assert!(
            wait_for_terminated(&hung, Duration::from_secs(5)).await,
            "the killed sidecar counts as exited while its child lives"
        );
        assert_eq!(hung.exit.lock().await.clone().expect("recorded").signal, Some(9));

        port.set(SERVE);
        manager.replace_sidecar_with(&["/bin/sh", "-c", HANGS, "nanna-daemon"]);
        let next = tokio::time::timeout(Duration::from_secs(5), manager.start(app.handle(), manager.stop_epoch()));
        assert_eq!(next.await, Ok(Ok(())), "the next boot spawns at once");

        signal(holder, "-KILL");
        port.set(DROP);
        stop_and_reap(&manager).await;
    }

    /// The same child behind a daemon that dies while it serves: the exit
    /// used to go unrecorded, so the state stayed running, and the monitor
    /// then reported the dead process as still running and never restarted
    /// it.
    #[tokio::test]
    async fn a_serving_daemon_that_dies_is_restarted_while_its_child_lives() {
        let app = mock_app();
        let port = Port::open(SERVE).await;
        let manager = manager_running(HOLDS_OUTPUT, &port);
        manager.start(app.handle(), manager.stop_epoch()).await.unwrap();
        let holder = holder_pid(&manager).await;
        let serving = latest(&manager).await;
        let pid = manager.live_sidecar_pid().await.expect("the sidecar runs");

        port.set(DROP);
        signal(pid, "-KILL");
        assert!(wait_for_terminated(&serving, Duration::from_secs(5)).await);
        // The exit handler's probe of the port ends within its timeout.
        tokio::time::timeout(PROBE_TIMEOUT * 2, async {
            while manager.state().await != DaemonState::Crashed {
                sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("the exit is a crash");
        let failure = manager.last_failure().await.expect("recorded");
        assert_eq!(failure.kind, StartFailureKind::ExitedAfterReady);
        assert_eq!((failure.exit_code, failure.signal), (None, Some(9)));
        assert_eq!(manager.live_sidecar_pid().await, None);

        manager.replace_sidecar_with(&["/bin/sh", "-c", ALREADY_RUNNING, "nanna-daemon"]);
        manager.health_tick(app.handle()).await;
        assert_eq!(*manager.restart_count.read().await, 1, "restarted");

        signal(holder, "-KILL");
    }

    /// Whether process `pid` has ended: it is gone, or it is a zombie that
    /// nothing has reaped yet.
    fn ended(pid: u32) -> bool {
        let ps = std::process::Command::new("ps")
            .args(["-o", "stat=", "-p", &pid.to_string()])
            .output()
            .expect("ps runs");
        let stat = String::from_utf8_lossy(&ps.stdout);
        stat.trim().is_empty() || stat.trim_start().starts_with('Z')
    }

    /// The user quits while the daemon is wedged: it takes the shutdown
    /// request's connection and never answers, so the stop kills it. The
    /// kill used to be only a request to the sidecar's event task, and the
    /// stop returned before that task ran. Tauri then ends the process, the
    /// task never runs again, and the wedged daemon outlived the app.
    ///
    /// The runtime here is the app's: once the stop returns it is abandoned,
    /// as the process exit abandons it, with nothing run or dropped again.
    #[test]
    fn a_stop_that_kills_the_daemon_has_killed_it_when_it_returns() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("a runtime");
        let pid = runtime.block_on(async {
            let app = mock_app();
            let port = Port::open(SERVE).await;
            let manager = manager_running(HANGS, &port);
            manager.start(app.handle(), manager.stop_epoch()).await.unwrap();
            let pid = manager.live_sidecar_pid().await.expect("the sidecar runs");
            port.set(SILENT);
            manager.stop().await.unwrap();
            pid
        });
        std::mem::forget(runtime);

        let killed = (0..50).any(|_| {
            let gone = ended(pid);
            if !gone {
                std::thread::sleep(Duration::from_millis(100));
            }
            gone
        });
        if !killed {
            signal(pid, "-KILL");
        }
        assert!(killed, "the daemon (PID {pid}) outlived the app");
    }
}
