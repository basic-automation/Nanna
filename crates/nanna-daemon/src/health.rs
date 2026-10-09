//! HTTP Health Endpoint and PID File Management
//!
//! Provides:
//! - `/health` endpoint for monitoring
//! - `/status` endpoint for detailed status
//! - `/metrics` endpoint (future: Prometheus)
//! - PID file management to prevent multiple instances

use axum::{
    extract::State,
    http::StatusCode,
    response::Json,
    routing::get,
    Router,
};
use serde::Serialize;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tower_http::cors::{Any, CorsLayer};
use tracing::{error, info, warn};

// =============================================================================
// PID File Management
// =============================================================================

/// File-name prefix every nanna daemon executable carries: `nanna-daemon`,
/// `nanna-daemon.exe`, or a Tauri sidecar still named `nanna-daemon-<triple>`.
#[cfg(any(target_os = "linux", windows, test))]
const DAEMON_EXE_PREFIX: &str = "nanna-daemon";

/// File name of the `nanna` CLI, which runs a full daemon — same data dir,
/// same PID file, same port — when started with [`DAEMON_MODE_FLAG`]. That is
/// what `nanna daemon start` launches.
#[cfg(any(target_os = "linux", windows, test))]
const CLI_EXE_NAME: &str = "nanna";

/// The flag that turns the `nanna` CLI into a daemon.
///
/// The probe looks for it on a `nanna` process's command line, so the CLI
/// launches its daemon with this very constant (and a CLI test pins that clap
/// parses it).
pub const DAEMON_MODE_FLAG: &str = "--daemon-mode";

/// What a PID recorded in the PID file refers to right now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessProbe {
    /// No such process — or a zombie awaiting its parent's reap, which runs no
    /// code and holds no port, lock or database.
    Dead,
    /// A live nanna daemon: a `nanna-daemon` executable, or the `nanna` CLI
    /// running [`DAEMON_MODE_FLAG`].
    Daemon,
    /// A live process running some other program: the recorded PID was reused
    /// after the daemon that wrote it died.
    Other,
    /// A live process whose identity could not be read (access denied, no
    /// `/proc`) — or, on Windows, a `nanna` CLI, whose command line (daemon
    /// mode or not) the probe does not read there. Treated as a daemon: a
    /// refused start is recoverable, two daemons against one store is not.
    Unknown,
}

/// What a PID file says about the daemon role, judged the one way
/// [`PidFile::acquire`] and the `nanna daemon` commands both judge it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PidFileState {
    /// No PID file.
    Absent,
    /// Content that is no PID: empty, torn by a death mid-write, or not a
    /// number at all (0 included: `kill(0, 0)` addresses a process GROUP).
    Unparseable(String),
    /// The reader's own PID, left by an earlier process that had it: nothing
    /// else can be running under it.
    OwnPid(u32),
    /// Another process's PID, and what that PID refers to right now.
    Recorded(u32, ProcessProbe),
}

impl PidFileState {
    /// The PID of the live daemon holding the role — `Some` exactly when
    /// [`PidFile::acquire`] refuses.
    #[must_use]
    pub const fn live_daemon(&self) -> Option<u32> {
        match *self {
            Self::Recorded(pid, ProcessProbe::Daemon | ProcessProbe::Unknown) => Some(pid),
            _ => None,
        }
    }
}

/// Single-instance guard: `nanna-daemon.pid` records the PID of THE daemon.
///
/// Incident (2026-09-16/17, Linux): a second daemon correctly refused to start
/// (`AlreadyRunning(521551)`) — and then its `Drop` deleted the live daemon's
/// PID file. Every later start found no file, "acquired" it, overwrote the
/// live daemon's exit record, and died only at the IPC bind. Three rules close
/// that class:
///
/// 1. **Refuse while the recorded PID is a live daemon.** Liveness alone is not
///    the question — a dead daemon's PID gets reused, and an unrelated program
///    holding it must not block startup forever — so the probe also checks the
///    process is a nanna daemon (see [`probe_process`]).
/// 2. **Only the owner removes the file.** `release` (and `Drop`) do nothing
///    unless this handle's `acquire` wrote the file, and even then remove it
///    only while it still records this process.
/// 3. **Read-decide-write is serialized** under an exclusive OS lock on the
///    sibling `nanna-daemon.pid.lock`. Without it two daemons starting together
///    can both take over one stale file: the second one's "remove the stale
///    file" deletes the first one's fresh record. The lock file is never
///    deleted — unlinking a lock file lets a waiter lock the orphaned inode
///    while a newcomer locks a new one.
pub struct PidFile {
    path: PathBuf,
    /// Set by a successful `acquire`, cleared by `release`.
    owned: AtomicBool,
}

impl PidFile {
    /// Create a new PID file manager
    #[must_use]
    pub fn new(data_dir: &Path) -> Self {
        Self {
            path: data_dir.join("nanna-daemon.pid"),
            owned: AtomicBool::new(false),
        }
    }

    /// Get the default PID file path
    #[must_use]
    pub fn default_path() -> PathBuf {
        nanna_config::project_dirs()
            .map_or_else(
                || PathBuf::from("."),
                |d| d.runtime_dir().map_or_else(
                    || d.data_dir().to_path_buf(),
                    Path::to_path_buf,
                ),
            )
            .join("nanna-daemon.pid")
    }

    /// Try to acquire the PID file.
    ///
    /// Refuses with `AlreadyRunning` while the recorded PID is a live nanna
    /// daemon (or a live process whose identity cannot be read); takes over a
    /// file whose PID is dead, reused by another program, unparseable, or this
    /// process's own (left by an earlier process that had the same PID).
    ///
    /// # Errors
    ///
    /// [`PidFileError::AlreadyRunning`] when the recorded PID is a live nanna
    /// daemon, or a live process whose program could not be identified.
    /// [`PidFileError::Io`] when the PID file's parent directory cannot be
    /// created, the existing file cannot be read, or this process's PID cannot
    /// be written.
    pub fn acquire(&self) -> Result<(), PidFileError> {
        self.acquire_with(probe_process)
    }

    /// [`Self::acquire`] with the process probe injected, so every verdict is
    /// testable without staging real processes.
    pub(crate) fn acquire_with(
        &self,
        probe: impl Fn(u32) -> ProcessProbe,
    ) -> Result<(), PidFileError> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| PidFileError::Io(e.to_string()))?;
        }

        let _section = self.lock_section();

        match self.read_state(probe)? {
            PidFileState::Absent => {}
            PidFileState::Unparseable(content) => {
                info!("Replacing unparseable PID file {:?} (content: {:?})", self.path, content);
            }
            PidFileState::OwnPid(pid) => {
                info!("PID file records this process's own PID {} (left by an earlier process that had it) — taking over", pid);
            }
            PidFileState::Recorded(pid, ProcessProbe::Daemon) => {
                return Err(PidFileError::AlreadyRunning(pid));
            }
            PidFileState::Recorded(pid, ProcessProbe::Unknown) => {
                warn!("PID {} from the PID file is alive but its program could not be identified — assuming it is a daemon", pid);
                return Err(PidFileError::AlreadyRunning(pid));
            }
            PidFileState::Recorded(pid, ProcessProbe::Dead) => {
                info!("Taking over stale PID file (process {} no longer exists)", pid);
            }
            PidFileState::Recorded(pid, ProcessProbe::Other) => {
                info!("Taking over stale PID file (PID {} now belongs to a different program)", pid);
            }
        }

        let own_pid = std::process::id();
        std::fs::write(&self.path, own_pid.to_string())
            .map_err(|e| PidFileError::Io(e.to_string()))?;
        self.owned.store(true, Ordering::Release);

        info!("PID file created at {:?} (PID: {})", self.path, own_pid);
        Ok(())
    }

    /// Judge the PID file without claiming it: whether a live daemon holds the
    /// role, decided exactly as [`Self::acquire`] decides it, with nothing
    /// written or removed. For observers — `nanna daemon start/stop/status` —
    /// which must not judge the file their own way: a liveness-only check
    /// reads a PID reused by another program as a running daemon, and `stop`
    /// would then signal it.
    ///
    /// # Errors
    ///
    /// [`PidFileError::Io`] when the existing file cannot be read.
    pub fn state(&self) -> Result<PidFileState, PidFileError> {
        self.state_with(probe_process)
    }

    /// [`Self::state`] with the process probe injected.
    pub(crate) fn state_with(
        &self,
        probe: impl Fn(u32) -> ProcessProbe,
    ) -> Result<PidFileState, PidFileError> {
        // A missing directory holds no PID file, and an observer creates
        // nothing — not even the lock file.
        let _section = if self.path.parent().is_none_or(Path::is_dir) {
            self.lock_section()
        } else {
            None
        };
        self.read_state(probe)
    }

    /// Read the file and probe the PID it records. The caller holds the lock
    /// section, when there is one.
    fn read_state(
        &self,
        probe: impl Fn(u32) -> ProcessProbe,
    ) -> Result<PidFileState, PidFileError> {
        let recorded = read_recorded_pid(&self.path).map_err(|e| PidFileError::Io(e.to_string()))?;
        Ok(match recorded {
            RecordedPid::Absent => PidFileState::Absent,
            RecordedPid::Unparseable(content) => PidFileState::Unparseable(content),
            RecordedPid::Pid(pid) if pid == std::process::id() => PidFileState::OwnPid(pid),
            RecordedPid::Pid(pid) => PidFileState::Recorded(pid, probe(pid)),
        })
    }

    /// Release the PID file — only if this handle acquired it AND the file
    /// still records this process. A refused instance, or one whose file was
    /// since taken over, leaves the file to its owner.
    pub fn release(&self) {
        if !self.owned.swap(false, Ordering::AcqRel) {
            return;
        }

        let _section = self.lock_section();
        let own_pid = std::process::id();
        match read_recorded_pid(&self.path) {
            Ok(RecordedPid::Pid(pid)) if pid == own_pid => match std::fs::remove_file(&self.path) {
                Ok(()) => info!("PID file removed"),
                Err(e) => warn!("Failed to remove PID file: {}", e),
            },
            Ok(RecordedPid::Absent) => {}
            Ok(other) => warn!(
                "PID file {:?} no longer records this process ({:?}) — leaving it for its owner",
                self.path, other
            ),
            Err(e) => warn!("Failed to read PID file before removal: {} — leaving it", e),
        }
    }

    /// Get the path to the PID file
    pub const fn path(&self) -> &PathBuf {
        &self.path
    }

    /// Hold an exclusive OS lock on `<pid file>.lock` for the guard's lifetime.
    ///
    /// Blocking is bounded by the holder's section: a read, one process probe
    /// and a write or remove on this same filesystem — nothing that waits on
    /// another process. The kernel drops the lock if the holder dies. `None`
    /// when the lock cannot be taken (a filesystem without lock support): the
    /// section then runs unserialized, and the IPC port reservation that
    /// follows at startup is still an atomic arbiter.
    fn lock_section(&self) -> Option<std::fs::File> {
        let lock_path = self.path.with_extension("pid.lock");
        let file = match std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)
        {
            Ok(file) => file,
            Err(e) => {
                warn!("Cannot open PID lock file {:?}: {} — PID file checks run unserialized", lock_path, e);
                return None;
            }
        };
        match file.lock() {
            Ok(()) => Some(file),
            Err(e) => {
                warn!("Cannot lock {:?}: {} — PID file checks run unserialized", lock_path, e);
                None
            }
        }
    }
}

impl Drop for PidFile {
    fn drop(&mut self) {
        self.release();
    }
}

/// What the PID file holds.
#[derive(Debug)]
enum RecordedPid {
    Absent,
    /// Empty, torn by a death mid-write, or not a PID at all (0 included:
    /// `kill(0, 0)` addresses a process GROUP and would read as alive).
    Unparseable(String),
    Pid(u32),
}

fn read_recorded_pid(path: &Path) -> std::io::Result<RecordedPid> {
    match std::fs::read(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(RecordedPid::Absent),
        Err(e) => Err(e),
        Ok(bytes) => {
            let text = String::from_utf8_lossy(&bytes);
            Ok(match text.trim().parse::<u32>() {
                Ok(pid) if pid != 0 => RecordedPid::Pid(pid),
                _ => RecordedPid::Unparseable(text.into_owned()),
            })
        }
    }
}

/// PID file errors
#[derive(Debug, thiserror::Error)]
pub enum PidFileError {
    #[error("Another daemon instance is already running (PID: {0})")]
    AlreadyRunning(u32),
    #[error("IO error: {0}")]
    Io(String),
}

/// The file-name part of an executable path, under either separator.
#[cfg(any(target_os = "linux", windows, test))]
fn file_name(executable: &str) -> &str {
    executable.rsplit(['/', '\\']).next().unwrap_or(executable)
}

/// True when an executable file name (or full path) names a nanna daemon.
#[cfg(any(target_os = "linux", windows, test))]
fn names_daemon(executable: &str) -> bool {
    file_name(executable).to_ascii_lowercase().starts_with(DAEMON_EXE_PREFIX)
}

/// True when an executable file name (or full path) names the `nanna` CLI:
/// `nanna`, or `nanna.exe`.
#[cfg(any(target_os = "linux", windows, test))]
fn names_cli(executable: &str) -> bool {
    let name = file_name(executable).to_ascii_lowercase();
    name.strip_suffix(".exe").unwrap_or(&name) == CLI_EXE_NAME
}

/// True when a `/proc/<pid>/cmdline` — argv, each NUL-terminated — passes
/// [`DAEMON_MODE_FLAG`]. argv\[0\] is the program, not an argument.
#[cfg(any(target_os = "linux", test))]
fn runs_daemon_mode(cmdline: &[u8]) -> bool {
    cmdline
        .split(|&byte| byte == 0)
        .skip(1)
        .any(|arg| arg == DAEMON_MODE_FLAG.as_bytes())
}

/// Classify a live process by its executable path alone (Windows).
///
/// The `nanna` CLI is a daemon only with [`DAEMON_MODE_FLAG`], and the
/// Windows probe reads the image path, not the command line (that takes the
/// native `NtQueryInformationProcess` API) — so a live `nanna.exe` is
/// `Unknown`: refused, never taken over.
#[cfg(any(windows, test))]
fn classify_image(image: &str) -> ProcessProbe {
    if names_daemon(image) {
        ProcessProbe::Daemon
    } else if names_cli(image) {
        ProcessProbe::Unknown
    } else {
        ProcessProbe::Other
    }
}

/// Split `/proc/<pid>/stat` into `(comm, state)`. `comm` is parenthesized and
/// may itself contain spaces and parentheses, so it ends at the LAST `)`.
#[cfg(any(target_os = "linux", test))]
fn parse_proc_stat(stat: &str) -> Option<(&str, char)> {
    let open = stat.find('(')?;
    let close = stat.rfind(')')?;
    if close < open {
        return None;
    }
    let state = stat[close + 1..].trim_start().chars().next()?;
    Some((&stat[open + 1..close], state))
}

/// Classify a recorded PID: dead, a live nanna daemon, a live other program,
/// or live but unidentifiable.
///
/// Linux reads `/proc/<pid>/stat`, which answers liveness AND identity in one
/// read: `comm` is the executable's file name (truncated to 15 bytes, which
/// keeps the whole `nanna-daemon` prefix), and state `Z`/`X` marks a process
/// that has already exited. A `nanna` process takes one more read: the CLI is
/// a daemon only in daemon mode, which its command line tells.
///
/// Reproduced live 2026-09-21: a `nanna --daemon-mode` daemon (what `nanna
/// daemon start` launches) held the PID file; a `nanna-daemon` probing it
/// by name alone read "reused by another program", took the file over, was
/// refused at the IPC bind — and, owning the file, removed it on the way out.
#[cfg(target_os = "linux")]
#[must_use]
pub fn probe_process(pid: u32) -> ProcessProbe {
    // The error arm is "not in /proc": gone — unless /proc hides other users'
    // processes (`hidepid`), which only the signal probe can tell apart.
    std::fs::read(format!("/proc/{pid}/stat")).map_or_else(
        |_| signal_probe(pid),
        |bytes| match parse_proc_stat(&String::from_utf8_lossy(&bytes)) {
            Some((_, 'Z' | 'X')) => ProcessProbe::Dead,
            Some((comm, _)) if names_daemon(comm) => ProcessProbe::Daemon,
            Some((comm, _)) if names_cli(comm) => probe_cli_mode(pid),
            Some(_) => ProcessProbe::Other,
            None => signal_probe(pid),
        },
    )
}

/// A live `nanna` process: a daemon when its command line passes
/// [`DAEMON_MODE_FLAG`], any other CLI command otherwise.
#[cfg(target_os = "linux")]
fn probe_cli_mode(pid: u32) -> ProcessProbe {
    match std::fs::read(format!("/proc/{pid}/cmdline")) {
        Ok(cmdline) if runs_daemon_mode(&cmdline) => ProcessProbe::Daemon,
        // Empty mid-execve (the process is already named `nanna`, but its
        // new arguments are not laid out yet) and again once it has released
        // its memory on the way out; unreadable once it is gone. Liveness
        // decides all three, as for an unreadable stat: a live one stays
        // `Unknown`, refused rather than taken over.
        Ok(cmdline) if cmdline.is_empty() => signal_probe(pid),
        Ok(_) => ProcessProbe::Other,
        Err(_) => signal_probe(pid),
    }
}

/// Non-Linux Unix has no `/proc` to name the program, so a live PID is
/// `Unknown` (conservatively a daemon).
#[cfg(all(unix, not(target_os = "linux")))]
#[must_use]
pub fn probe_process(pid: u32) -> ProcessProbe {
    signal_probe(pid)
}

/// `kill(pid, 0)`: existence without identity.
#[cfg(unix)]
fn signal_probe(pid: u32) -> ProcessProbe {
    // A PID above i32::MAX would wrap to a negative value, which addresses a
    // process group (-1: every process) and "succeeds".
    let Ok(pid) = i32::try_from(pid) else {
        return ProcessProbe::Dead;
    };
    if pid <= 0 {
        return ProcessProbe::Dead;
    }
    // SAFETY: signal 0 performs only the existence and permission check.
    if unsafe { libc::kill(pid, 0) } == 0 {
        return ProcessProbe::Unknown;
    }
    match std::io::Error::last_os_error().raw_os_error() {
        Some(libc::ESRCH) => ProcessProbe::Dead,
        // EPERM: the process exists but belongs to another user. The old
        // check read this as dead.
        _ => ProcessProbe::Unknown,
    }
}

/// Uses the Win32 API directly. An earlier `tasklist` subprocess check
/// returned "not running" whenever the subprocess itself failed, which let a
/// second daemon treat a LIVE daemon as dead and clobber its PID file
/// (observed live: a GUI sidecar overwrote the standalone daemon's lock, then
/// ran on as a storage-less zombie).
#[cfg(windows)]
#[must_use]
pub fn probe_process(pid: u32) -> ProcessProbe {
    use windows_sys::Win32::Foundation::{CloseHandle, GetLastError};
    use windows_sys::Win32::System::Threading::{
        OpenProcess, QueryFullProcessImageNameW, WaitForSingleObject,
    };

    // Named locally so a windows-sys reorganization cannot silently change
    // semantics: these are stable Win32 ABI values.
    const PROCESS_SYNCHRONIZE: u32 = 0x0010_0000;
    const PROCESS_QUERY_LIMITED_INFORMATION_CODE: u32 = 0x1000;
    const PROCESS_NAME_WIN32_CODE: u32 = 0;
    const WAIT_TIMEOUT_CODE: u32 = 0x102;
    const ERROR_ACCESS_DENIED_CODE: u32 = 5;
    // The NT path limit: an image path cannot be longer, so the query cannot
    // fail for want of buffer.
    const MAX_NT_PATH_UNITS: usize = 32_768;

    // SAFETY: plain Win32 calls; the handle is closed on every path, and the
    // name buffer outlives the call that fills it.
    unsafe {
        let handle = OpenProcess(
            PROCESS_SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION_CODE,
            0,
            pid,
        );
        if handle.is_null() {
            // Access denied means the process exists but isn't ours (e.g. an
            // elevated daemon). Any other failure (invalid parameter, not
            // found) means no such process.
            return if GetLastError() == ERROR_ACCESS_DENIED_CODE {
                ProcessProbe::Unknown
            } else {
                ProcessProbe::Dead
            };
        }
        let probe = if WaitForSingleObject(handle, 0) == WAIT_TIMEOUT_CODE {
            let mut name = vec![0u16; MAX_NT_PATH_UNITS];
            let mut len = MAX_NT_PATH_UNITS as u32;
            if QueryFullProcessImageNameW(handle, PROCESS_NAME_WIN32_CODE, name.as_mut_ptr(), &mut len) == 0 {
                ProcessProbe::Unknown
            } else {
                classify_image(&String::from_utf16_lossy(&name[..len as usize]))
            }
        } else {
            // Signaled: the process has exited.
            ProcessProbe::Dead
        };
        CloseHandle(handle);
        probe
    }
}

#[cfg(not(any(windows, unix)))]
#[must_use]
pub fn probe_process(_pid: u32) -> ProcessProbe {
    // Conservative: assume a daemon if we can't check
    ProcessProbe::Unknown
}

// =============================================================================
// Health HTTP Server
// =============================================================================

/// Memory service and durable-store status.
///
/// Flattened into [`StatusResponse`], so each field serializes under its
/// `memory_`-prefixed key exactly where the response always carried it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct MemoryHealth {
    /// Memory service status
    #[serde(rename = "memory_available")]
    pub available: bool,
    /// Durable memory store degraded (a corrupt row was skipped on load).
    #[serde(rename = "memory_degraded")]
    pub degraded: bool,
    /// Number of memory rows that were unreadable (corrupt) at load.
    #[serde(rename = "memory_corrupt_rows")]
    pub corrupt_rows: usize,
    /// The store was quarantined and rebuilt after page-level corruption.
    #[serde(rename = "memory_rebuilt")]
    pub rebuilt: bool,
    /// Memories salvaged into the rebuilt store (0 unless `rebuilt`).
    #[serde(rename = "memory_recovered_rows")]
    pub recovered_rows: usize,
    /// Memories the corrupt store held, when countable (None = unknown loss).
    #[serde(rename = "memory_expected_rows")]
    pub expected_rows: Option<usize>,
}

/// Health server state
pub struct HealthState {
    /// When the daemon started
    pub start_time: Instant,
    /// Number of active sessions
    pub session_count: Arc<RwLock<usize>>,
    /// Number of connected clients
    pub client_count: Arc<RwLock<usize>>,
    /// Memory service and durable-store status
    pub memory: MemoryHealth,
    /// Agent service status
    pub agent_available: bool,
    /// Last error message (if any)
    pub last_error: Arc<RwLock<Option<String>>>,
    /// Renders `GET /metrics`. `None` answers 404: a health server with no
    /// control plane behind it has nothing true to report.
    pub metrics: Option<MetricsFn>,
}

/// Produces the `/metrics` body on demand.
pub type MetricsFn = Arc<
    dyn Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = String> + Send>> + Send + Sync,
>;

impl HealthState {
    #[must_use]
    pub fn new(memory_available: bool, agent_available: bool) -> Self {
        Self {
            start_time: Instant::now(),
            session_count: Arc::new(RwLock::new(0)),
            client_count: Arc::new(RwLock::new(0)),
            memory: MemoryHealth {
                available: memory_available,
                ..MemoryHealth::default()
            },
            agent_available,
            last_error: Arc::new(RwLock::new(None)),
            metrics: None,
        }
    }

    /// Serve `GET /metrics` from `render`.
    #[must_use]
    pub fn with_metrics(mut self, render: MetricsFn) -> Self {
        self.metrics = Some(render);
        self
    }

    /// Seed the durable-memory-store health (from `MemoryService::store_health`).
    /// Builder-style so a corrupt store is visible on `/status` and `/health`
    /// instead of only a boot `error!` log.
    #[must_use]
    pub const fn with_memory_health(mut self, degraded: bool, corrupt_rows: usize) -> Self {
        self.memory.degraded = degraded;
        self.memory.corrupt_rows = corrupt_rows;
        self
    }

    /// Seed the store-rebuilt-after-corruption facts (from the startup
    /// `RecoveryReport`) so `/status` keeps reporting the rebuild — clients
    /// that connect after boot never saw the `MemoryStoreRebuilt` event.
    #[must_use]
    pub const fn with_memory_rebuild(mut self, recovered: usize, expected: Option<usize>) -> Self {
        self.memory.rebuilt = true;
        self.memory.recovered_rows = recovered;
        self.memory.expected_rows = expected;
        self
    }

    pub async fn set_session_count(&self, count: usize) {
        *self.session_count.write().await = count;
    }
    
    pub async fn set_client_count(&self, count: usize) {
        *self.client_count.write().await = count;
    }
    
    pub async fn set_last_error(&self, error: Option<String>) {
        *self.last_error.write().await = error;
    }

    /// Keep `last_error` at the newest [`Event::Error`] the daemon broadcasts,
    /// as `code: message`, until `events` closes.
    ///
    /// `set_last_error` had no caller, so `/status` reported `last_error: null`
    /// through any failure. The bus is the one place every error a client is
    /// told about passes. The write is a lock held for one assignment, so the
    /// loop never lags the bus; a lag (another consumer's backlog) skips to
    /// the newest events, which is what "last" wants anyway.
    #[must_use = "the handle ends with the bus; drop it to detach"]
    pub fn record_errors_from(
        self: Arc<Self>,
        mut events: tokio::sync::broadcast::Receiver<crate::protocol::Event>,
    ) -> tokio::task::JoinHandle<()> {
        use tokio::sync::broadcast::error::RecvError;
        tokio::spawn(async move {
            loop {
                match events.recv().await {
                    Ok(crate::protocol::Event::Error { code, message, .. }) => {
                        debug_assert!(!code.is_empty(), "an error event names its code");
                        self.set_last_error(Some(format!("{code}: {message}"))).await;
                    }
                    Ok(_) | Err(RecvError::Lagged(_)) => {}
                    Err(RecvError::Closed) => return,
                }
            }
        })
    }
}

/// Simple health check response
#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub uptime_secs: u64,
}

/// Detailed status response
#[derive(Serialize)]
pub struct StatusResponse {
    pub status: String,
    pub version: String,
    pub uptime_secs: u64,
    pub sessions: usize,
    pub clients: usize,
    #[serde(flatten)]
    pub memory: MemoryHealth,
    pub agent_available: bool,
    pub last_error: Option<String>,
}

/// Health check endpoint (GET /health)
async fn health(State(state): State<Arc<HealthState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_secs: state.start_time.elapsed().as_secs(),
    })
}

/// Prometheus scrape endpoint (GET /metrics).
async fn metrics(State(state): State<Arc<HealthState>>) -> axum::response::Response {
    use axum::response::IntoResponse;
    let Some(ref render) = state.metrics else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let body = render().await;
    (
        [(
            axum::http::header::CONTENT_TYPE,
            crate::metrics::METRICS_CONTENT_TYPE,
        )],
        body,
    )
        .into_response()
}

/// Kubernetes-style liveness probe (GET /healthz)
async fn healthz() -> StatusCode {
    StatusCode::OK
}

/// Kubernetes-style readiness probe (GET /readyz)
async fn readyz(State(state): State<Arc<HealthState>>) -> StatusCode {
    // Ready if agent is available
    if state.agent_available {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}

/// Detailed status endpoint (GET /status)
async fn status(State(state): State<Arc<HealthState>>) -> Json<StatusResponse> {
    let sessions = *state.session_count.read().await;
    let clients = *state.client_count.read().await;
    let last_error = state.last_error.read().await.clone();
    
    Json(StatusResponse {
        status: "running".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_secs: state.start_time.elapsed().as_secs(),
        sessions,
        clients,
        memory: state.memory,
        agent_available: state.agent_available,
        last_error,
    })
}

/// Health HTTP server
pub struct HealthServer {
    state: Arc<HealthState>,
    port: u16,
    host: String,
}

impl HealthServer {
    /// Create a new health server
    #[must_use]
    pub fn new(state: HealthState, host: &str, port: u16) -> Self {
        Self {
            state: Arc::new(state),
            port,
            host: host.to_string(),
        }
    }

    /// Create a health server that serves an **existing** shared state handle.
    ///
    /// Use this (not [`Self::new`]) when the daemon keeps updating the state
    /// (session/client counts, `last_error`) after the server is spawned: the
    /// server then reflects those live updates instead of serving a throwaway
    /// copy whose counters never move.
    #[must_use]
    pub fn from_shared(state: Arc<HealthState>, host: &str, port: u16) -> Self {
        Self {
            state,
            port,
            host: host.to_string(),
        }
    }

    /// Get a reference to the state (for updating from daemon)
    #[must_use]
    pub fn state(&self) -> Arc<HealthState> {
        self.state.clone()
    }
    
    /// Build the Axum router
    fn router(&self) -> Router {
        let cors = CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any);
        
        Router::new()
            .route("/health", get(health))
            .route("/healthz", get(healthz))
            .route("/readyz", get(readyz))
            .route("/status", get(status))
            .route("/metrics", get(metrics))
            .layer(cors)
            .with_state(self.state.clone())
    }
    
    /// Run the health server
    ///
    /// # Errors
    ///
    /// An `InvalidInput` error when `host:port` is not a socket address, the
    /// error from binding the listener (on Windows, after the retries), or the
    /// error that ends serving.
    pub async fn run(&self) -> Result<(), std::io::Error> {
        let addr: SocketAddr = format!("{}:{}", self.host, self.port)
            .parse()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
        
        info!("Health server listening on http://{}", addr);
        
        #[cfg(unix)]
        let listener = Self::bind_reuse_address(addr)?;
        #[cfg(windows)]
        let listener = Self::bind_with_retry(addr).await?;
        #[cfg(not(any(unix, windows)))]
        let listener = tokio::net::TcpListener::bind(addr).await?;
        axum::serve(listener, self.router()).await
    }

    /// Bind with `SO_REUSEADDR` (Unix), so a restart does not trip over the
    /// previous listener's `TIME_WAIT`.
    #[cfg(unix)]
    fn bind_reuse_address(addr: std::net::SocketAddr) -> Result<tokio::net::TcpListener, std::io::Error> {
        let socket = socket2::Socket::new(
            socket2::Domain::for_address(addr),
            socket2::Type::STREAM,
            Some(socket2::Protocol::TCP),
        )?;
        socket.set_reuse_address(true)?;
        socket.set_nonblocking(true)?;
        socket.bind(&addr.into())?;
        socket.listen(128)?;
        tokio::net::TcpListener::from_std(socket.into())
    }

    /// Bind with retry for Windows port conflicts: retries with delay.
    #[cfg(windows)]
    async fn bind_with_retry(addr: std::net::SocketAddr) -> Result<tokio::net::TcpListener, std::io::Error> {
        for attempt in 0..5 {
            match tokio::net::TcpListener::bind(addr).await {
                Ok(listener) => return Ok(listener),
                Err(e) if attempt < 4 => {
                    tracing::warn!("Health bind attempt {} failed ({}), retrying in 1s...", attempt + 1, e);
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                }
                Err(e) => return Err(e),
            }
        }
        unreachable!()
    }
    
    /// Spawn the health server as a background task
    #[must_use]
    pub fn spawn(self) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            if let Err(e) = self.run().await {
                error!("Health server error: {}", e);
            }
        })
    }
}

/// Default health server port (one below WebSocket port)
pub const DEFAULT_HEALTH_PORT: u16 = 5148;

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_pid_file_creation() {
        let temp_dir = TempDir::new().unwrap();
        let pid_file = PidFile::new(temp_dir.path());
        
        // Should acquire successfully
        assert!(pid_file.acquire().is_ok());
        
        // Check file exists
        assert!(pid_file.path().exists());
        
        // Read PID
        let content = std::fs::read_to_string(pid_file.path()).unwrap();
        assert_eq!(content.trim().parse::<u32>().unwrap(), std::process::id());
    }
    
    #[test]
    fn test_pid_file_release() {
        let temp_dir = TempDir::new().unwrap();
        let pid_file = PidFile::new(temp_dir.path());
        
        pid_file.acquire().unwrap();
        assert!(pid_file.path().exists());
        
        pid_file.release();
        assert!(!pid_file.path().exists());
    }

    /// A PID that is not this process's. The probe is injected in the tests
    /// that use it, so whatever really runs under it never matters.
    fn another_pid() -> u32 {
        std::process::id() + 1
    }

    fn recorded(pid_file: &PidFile) -> String {
        std::fs::read_to_string(pid_file.path()).unwrap()
    }

    /// Stage a PID file holding `content` and return a handle to it.
    fn staged(temp_dir: &TempDir, content: &str) -> PidFile {
        let pid_file = PidFile::new(temp_dir.path());
        std::fs::write(pid_file.path(), content).unwrap();
        pid_file
    }

    fn unreachable_probe(pid: u32) -> ProcessProbe {
        panic!("the probe must not run for this PID file (asked about {pid})")
    }

    #[test]
    fn a_live_daemon_pid_refuses_acquisition() {
        let temp_dir = TempDir::new().unwrap();
        let live = another_pid();
        let pid_file = staged(&temp_dir, &live.to_string());

        let result = pid_file.acquire_with(|pid| {
            assert_eq!(pid, live, "the probe must be asked about the recorded PID");
            ProcessProbe::Daemon
        });

        assert!(matches!(result, Err(PidFileError::AlreadyRunning(pid)) if pid == live));
        assert_eq!(recorded(&pid_file), live.to_string(), "the live daemon's record is untouched");
    }

    #[test]
    fn an_unidentifiable_live_pid_refuses_acquisition() {
        // Alive but unnamed (another user's process, no /proc): refusing is
        // recoverable, two daemons on one store is not.
        let temp_dir = TempDir::new().unwrap();
        let live = another_pid();
        let pid_file = staged(&temp_dir, &live.to_string());

        let result = pid_file.acquire_with(|_| ProcessProbe::Unknown);

        assert!(matches!(result, Err(PidFileError::AlreadyRunning(pid)) if pid == live));
        assert_eq!(recorded(&pid_file), live.to_string());
    }

    #[test]
    fn a_dead_pid_is_taken_over() {
        let temp_dir = TempDir::new().unwrap();
        let pid_file = staged(&temp_dir, &another_pid().to_string());

        pid_file.acquire_with(|_| ProcessProbe::Dead).unwrap();
        assert_eq!(recorded(&pid_file), std::process::id().to_string());

        // Taking over makes this handle the owner, so its release removes it.
        pid_file.release();
        assert!(!pid_file.path().exists());
    }

    #[test]
    fn a_pid_reused_by_another_program_is_taken_over() {
        // Liveness alone would block startup until that unrelated program exits.
        let temp_dir = TempDir::new().unwrap();
        let pid_file = staged(&temp_dir, &another_pid().to_string());

        pid_file.acquire_with(|_| ProcessProbe::Other).unwrap();
        assert_eq!(recorded(&pid_file), std::process::id().to_string());
    }

    #[test]
    fn an_unparseable_pid_file_is_taken_over_without_probing() {
        // "0" is here because kill(0, 0) addresses a process group and succeeds.
        for content in ["", "\n", "not a pid", "0", "-1", "99999999999"] {
            let temp_dir = TempDir::new().unwrap();
            let pid_file = staged(&temp_dir, content);

            pid_file.acquire_with(unreachable_probe).unwrap();
            assert_eq!(recorded(&pid_file), std::process::id().to_string(), "content {content:?}");
        }
    }

    #[test]
    fn this_process_own_pid_is_taken_over_without_probing() {
        // Nothing else can run under our own PID: an earlier process had it.
        let temp_dir = TempDir::new().unwrap();
        let pid_file = staged(&temp_dir, &std::process::id().to_string());

        pid_file.acquire_with(unreachable_probe).unwrap();
        assert_eq!(recorded(&pid_file), std::process::id().to_string());
    }

    #[test]
    fn a_refused_instance_never_removes_the_live_daemons_pid_file() {
        // The 2026-09-16 incident: instance B was correctly refused, then its
        // Drop deleted live instance A's PID file, so every later start found
        // no file and "acquired" the guard beside a running daemon.
        let temp_dir = TempDir::new().unwrap();
        let instance_a = another_pid();
        std::fs::write(temp_dir.path().join("nanna-daemon.pid"), instance_a.to_string()).unwrap();

        let instance_b = PidFile::new(temp_dir.path());
        assert!(instance_b.acquire_with(|_| ProcessProbe::Daemon).is_err());
        // Both shutdown routes: the explicit release and the Drop.
        instance_b.release();
        let path = instance_b.path().clone();
        drop(instance_b);

        assert_eq!(std::fs::read_to_string(path).unwrap(), instance_a.to_string());
    }

    #[test]
    fn shutdown_leaves_a_pid_file_that_no_longer_records_this_process() {
        // B acquired, then A took the file over (e.g. B was presumed dead). B's
        // shutdown must leave A's record alone.
        let temp_dir = TempDir::new().unwrap();
        let instance_b = PidFile::new(temp_dir.path());
        instance_b.acquire_with(unreachable_probe).unwrap();

        let instance_a = another_pid();
        std::fs::write(instance_b.path(), instance_a.to_string()).unwrap();

        instance_b.release();
        let path = instance_b.path().clone();
        drop(instance_b);

        assert_eq!(std::fs::read_to_string(path).unwrap(), instance_a.to_string());
    }

    #[test]
    fn probe_sees_the_current_process_alive() {
        // The regression that let two daemons run at once: the liveness
        // check mis-reporting a live PID as dead.
        assert_ne!(probe_process(std::process::id()), ProcessProbe::Dead);
    }

    #[test]
    fn test_stale_pid_file() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("nanna-daemon.pid");
        
        // Write a fake PID that definitely doesn't exist
        std::fs::write(&path, "999999999").unwrap();
        
        let pid_file = PidFile::new(temp_dir.path());

        // Should succeed because the old process doesn't exist
        assert!(pid_file.acquire().is_ok());
    }

    #[test]
    fn parse_proc_stat_ends_comm_at_the_last_paren() {
        assert_eq!(
            parse_proc_stat("521551 (nanna-daemon) S 1 521551 521551 0 -1"),
            Some(("nanna-daemon", 'S'))
        );
        // comm may hold spaces and parentheses of its own.
        assert_eq!(parse_proc_stat("7 (a) b (c)) Z 1 7"), Some(("a) b (c)", 'Z')));
        assert_eq!(parse_proc_stat("garbage"), None);
        assert_eq!(parse_proc_stat("7 (unterminated"), None);
    }

    #[test]
    fn names_daemon_matches_every_daemon_executable_form() {
        for name in [
            "nanna-daemon",
            "nanna-daemon.exe",
            "NANNA-DAEMON.EXE",
            "nanna-daemon-x8", // comm: a triple-suffixed sidecar, truncated to 15 bytes
            "/usr/lib/nanna/nanna-daemon",
            r"C:\Program Files\Nanna\nanna-daemon.exe",
        ] {
            assert!(names_daemon(name), "{name} is a daemon");
        }
        for name in ["nanna_daemon-4f3a", "nanna-gui", "sleep", "/opt/nanna-daemon/bin/other", ""] {
            assert!(!names_daemon(name), "{name} is not a daemon");
        }
    }

    #[test]
    fn names_cli_matches_only_the_cli_executable() {
        for name in [
            "nanna",
            "nanna.exe",
            "NANNA.EXE",
            "/home/me/.cargo/bin/nanna",
            r"C:\Program Files\Nanna\nanna.exe",
        ] {
            assert!(names_cli(name), "{name} is the CLI");
        }
        // "nanna-4f3a2b…" is a cargo test binary of the CLI crate.
        for name in ["nanna-daemon", "nanna-daemon.exe", "nanna-gui", "nanna-4f3a2b", "nannas", "anna", ".exe", ""] {
            assert!(!names_cli(name), "{name} is not the CLI");
        }
    }

    #[test]
    fn runs_daemon_mode_reads_the_arguments_not_the_program() {
        assert!(runs_daemon_mode(b"nanna\0--daemon-mode\0"));
        // `nanna daemon start` passes host and port after the flag.
        assert!(runs_daemon_mode(b"/usr/bin/nanna\0--daemon-mode\0--host\0localhost\0--port\0"));
        assert!(runs_daemon_mode(b"nanna\0--log-level\0debug\0--daemon-mode\0"));

        assert!(!runs_daemon_mode(b"nanna\0daemon\0status\0"));
        assert!(!runs_daemon_mode(b"--daemon-mode\0"), "argv[0] is the program name");
        assert!(!runs_daemon_mode(b"nanna\0--daemon-modes\0"));
        assert!(!runs_daemon_mode(b""));
    }

    #[test]
    fn classify_image_refuses_a_cli_whose_mode_it_cannot_read() {
        assert_eq!(classify_image(r"C:\Nanna\nanna-daemon.exe"), ProcessProbe::Daemon);
        // Maybe `nanna.exe --daemon-mode`: refusing is recoverable, a second
        // daemon on one store is not.
        assert_eq!(classify_image(r"C:\Nanna\nanna.exe"), ProcessProbe::Unknown);
        assert_eq!(classify_image(r"C:\Windows\System32\notepad.exe"), ProcessProbe::Other);
    }

    #[test]
    fn state_names_a_live_daemon_exactly_when_acquire_refuses() {
        // One judgement for the daemon's own claim and for the `nanna daemon`
        // commands: the CLI's own liveness-only check was how they drifted.
        for probe in [ProcessProbe::Dead, ProcessProbe::Daemon, ProcessProbe::Other, ProcessProbe::Unknown] {
            let temp_dir = TempDir::new().unwrap();
            let pid = another_pid();
            let pid_file = staged(&temp_dir, &pid.to_string());

            let state = pid_file.state_with(|_| probe).unwrap();
            assert_eq!(state, PidFileState::Recorded(pid, probe));
            assert_eq!(recorded(&pid_file), pid.to_string(), "judging writes nothing");

            let refused = match pid_file.acquire_with(|_| probe) {
                Ok(()) => None,
                Err(PidFileError::AlreadyRunning(holder)) => Some(holder),
                Err(e) => panic!("{probe:?}: {e}"),
            };
            assert_eq!(state.live_daemon(), refused, "{probe:?}");
        }
    }

    #[test]
    fn state_without_a_recorded_pid_names_no_daemon_and_never_probes() {
        let temp_dir = TempDir::new().unwrap();
        let pid_file = PidFile::new(temp_dir.path());
        assert_eq!(pid_file.state_with(unreachable_probe).unwrap(), PidFileState::Absent);

        std::fs::write(pid_file.path(), "not a pid").unwrap();
        let state = pid_file.state_with(unreachable_probe).unwrap();
        assert_eq!(state, PidFileState::Unparseable("not a pid".to_string()));
        assert_eq!(state.live_daemon(), None);

        let own_pid = std::process::id();
        std::fs::write(pid_file.path(), own_pid.to_string()).unwrap();
        let state = pid_file.state_with(unreachable_probe).unwrap();
        assert_eq!(state, PidFileState::OwnPid(own_pid));
        assert_eq!(state.live_daemon(), None);
        assert_eq!(recorded(&pid_file), own_pid.to_string(), "judging writes nothing");
    }

    #[test]
    fn state_of_a_missing_data_dir_creates_nothing() {
        // `nanna daemon status` on a machine that never ran a daemon.
        let temp_dir = TempDir::new().unwrap();
        let data_dir = temp_dir.path().join("never-created");
        let pid_file = PidFile::new(&data_dir);

        assert_eq!(pid_file.state_with(unreachable_probe).unwrap(), PidFileState::Absent);
        assert!(!data_dir.exists());
    }

    #[tokio::test]
    async fn from_shared_serves_live_updates() {
        // The daemon keeps updating this handle *after* the server is built;
        // the served `/status` must reflect those updates, not a frozen copy.
        let shared = Arc::new(HealthState::new(true, true));
        let server = HealthServer::from_shared(shared.clone(), "127.0.0.1", 0);

        // The server serves the very same state handle.
        assert!(Arc::ptr_eq(&shared, &server.state()));

        // A later update through the daemon's handle is visible via the server.
        shared.set_session_count(3).await;
        shared.set_client_count(2).await;
        shared.set_last_error(Some("boom".to_string())).await;

        let Json(status) = status(State(server.state())).await;
        assert_eq!(status.sessions, 3);
        assert_eq!(status.clients, 2);
        assert_eq!(status.last_error.as_deref(), Some("boom"));
    }

    /// `/status` reports the newest error event the daemon broadcast, and
    /// nothing else on the bus touches it.
    #[tokio::test]
    async fn status_reports_the_last_error_event() {
        use crate::protocol::Event;
        let shared = Arc::new(HealthState::new(true, true));
        let (tx, rx) = tokio::sync::broadcast::channel(16);
        let recorder = Arc::clone(&shared).record_errors_from(rx);
        for (code, message) in [("first", "older"), ("llm_error", "429 from provider")] {
            tx.send(Event::Error {
                code: code.to_string(),
                message: message.to_string(),
                session_id: None,
            })
            .expect("a receiver is subscribed");
        }
        tx.send(Event::Connected { client_id: "c".to_string() }).expect("send");
        drop(tx);
        recorder.await.expect("the recorder ends when the bus closes");
        assert_eq!(
            shared.last_error.read().await.as_deref(),
            Some("llm_error: 429 from provider")
        );
    }

    #[tokio::test]
    async fn status_surfaces_memory_degraded() {
        // A degraded store (corrupt rows skipped on load) must show on /status,
        // not just in a boot log.
        let shared = Arc::new(HealthState::new(true, true).with_memory_health(true, 2));
        let server = HealthServer::from_shared(shared, "127.0.0.1", 0);
        let Json(s) = status(State(server.state())).await;
        assert!(s.memory.degraded);
        assert_eq!(s.memory.corrupt_rows, 2);

        // A healthy store reports the negative.
        let clean = Arc::new(HealthState::new(true, true));
        let server2 = HealthServer::from_shared(clean, "127.0.0.1", 0);
        let Json(s2) = status(State(server2.state())).await;
        assert!(!s2.memory.degraded);
        assert_eq!(s2.memory.corrupt_rows, 0);
    }

    #[tokio::test]
    async fn status_surfaces_memory_rebuild() {
        // A store rebuilt after page-level corruption must stay visible on
        // /status for clients that connected after the boot-time event.
        let shared =
            Arc::new(HealthState::new(true, true).with_memory_rebuild(42, Some(50)));
        let server = HealthServer::from_shared(shared, "127.0.0.1", 0);
        let Json(s) = status(State(server.state())).await;
        assert!(s.memory.rebuilt);
        assert_eq!(s.memory.recovered_rows, 42);
        assert_eq!(s.memory.expected_rows, Some(50));

        // Without a rebuild the fields stay quiet.
        let clean = Arc::new(HealthState::new(true, true));
        let server2 = HealthServer::from_shared(clean, "127.0.0.1", 0);
        let Json(s2) = status(State(server2.state())).await;
        assert!(!s2.memory.rebuilt);
        assert_eq!(s2.memory.recovered_rows, 0);
        assert_eq!(s2.memory.expected_rows, None);
    }

    #[tokio::test]
    async fn new_serves_isolated_copy() {
        // Regression guard documenting *why* the daemon must use `from_shared`:
        // `new` wraps a fresh Arc, so an unrelated external handle can never
        // drive the served counters (this was the old server.rs bug).
        let server = HealthServer::new(HealthState::new(true, true), "127.0.0.1", 0);
        let external = Arc::new(HealthState::new(true, true));
        external.set_session_count(9).await;

        assert!(!Arc::ptr_eq(&external, &server.state()));
        let Json(status) = status(State(server.state())).await;
        assert_eq!(status.sessions, 0);
    }
}
