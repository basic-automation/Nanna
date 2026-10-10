//! Bridge between Nanna and scripted tools
//!
//! Exposes controlled Nanna functionality to JavaScript code.

use crate::{Result, ScriptError, tool::ToolPermissions};
use nanna_proc::{ChildJob, kill_process_tree};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::future::Future;
use std::path::{Path, PathBuf};

use std::sync::Mutex as StdMutex;
use std::collections::HashSet as StdHashSet;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::RwLock;

/// A service function callable from scripts via `Nanna.service(name, params)`.
///
/// Each closure receives a JSON `Value` of params and returns a JSON `Value` result.
pub type ServiceFn = Arc<
    dyn Fn(Value) -> Pin<Box<dyn Future<Output = std::result::Result<Value, String>> + Send>>
        + Send
        + Sync,
>;

/// The services a skill can reach, keyed by the name scripts pass to
/// `Nanna.service(name, params)`.
pub type ServiceMap = HashMap<String, ServiceFn>;

/// A synchronous ranked tool-search function backing `Nanna.searchTools(query)`.
///
/// Receives `(query, limit)` and returns a JSON array of
/// `{name, description, score}` hits, best first. Implementations must never
/// panic and should return an empty array when the underlying registry is
/// gone — the script-side contract is "empty results, never a throw".
pub type ToolSearchFn = Arc<dyn Fn(&str, usize) -> Value + Send + Sync>;

/// Default number of hits `Nanna.searchTools` returns when the script passes
/// no explicit limit — enough for a discovery response without flooding a
/// small model's context.
pub const DEFAULT_TOOL_SEARCH_LIMIT: usize = 10;

/// Bridge providing Nanna capabilities to scripts
#[derive(Clone)]
pub struct NannaBridge {
    pub permissions: ToolPermissions,
    http_client: reqwest::Client,
    logs: Arc<RwLock<Vec<LogEntry>>>,
    /// Tool definitions for `Nanna.listTools()` — set by the script engine when available
    tool_definitions: Option<Value>,
    /// Ranked tool search for `Nanna.searchTools()` — set by the script engine when available
    tool_search: Option<ToolSearchFn>,
    /// Service functions callable via `Nanna.service(name, params)`
    services: Arc<HashMap<String, ServiceFn>>,
    /// Default working directory for exec commands (overrides home dir fallback)
    default_workdir: Option<PathBuf>,
    /// Current session ID for session-scoped operations
    session_id: Option<String>,
}


/// Global advisory lock set for file writes.
/// Prevents concurrent sub-agents from writing to the same file simultaneously.
static FILE_WRITE_LOCKS: std::sync::LazyLock<StdMutex<StdHashSet<std::path::PathBuf>>> =
    std::sync::LazyLock::new(|| StdMutex::new(StdHashSet::new()));

/// Which Windows shell to run a model-issued command through.
#[cfg(windows)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WinShell {
    /// Explicit PowerShell — `powershell`/`pwsh` prefix, `$env:`, or a `Verb-Noun` cmdlet.
    PowerShell,
    /// The default. Models write POSIX/bash commands (`[ -f ]`, `cat`, `tail`, `&&`,
    /// `|`, forward-slash paths); we run those through Git Bash when available.
    Bash,
}

/// Classify a Windows command as explicit PowerShell, else Bash (the default).
///
/// Pure (no IO). `trimmed` must already be left-trimmed. Only *unambiguous*
/// PowerShell markers force PowerShell; everything else defaults to Bash, because
/// the previous `$ && |` heuristic mis-routed ordinary bash (`cat $f | grep`) to
/// PowerShell and broke it.
#[cfg(windows)]
fn classify_windows_command(trimmed: &str) -> WinShell {
    let is_powershell = trimmed.starts_with("powershell")
        || trimmed.starts_with("pwsh")
        || trimmed.contains("$env:")
        || trimmed.contains("Get-")
        || trimmed.contains("Set-")
        || trimmed.contains("New-")
        || trimmed.contains("Remove-")
        || trimmed.contains("Select-")
        || trimmed.contains("Where-Object")
        || trimmed.contains("ForEach-Object");
    if is_powershell {
        WinShell::PowerShell
    } else {
        WinShell::Bash
    }
}

/// Strip the outer quotes off a `python -c` payload when it has a matching pair.
///
/// `rest` is whatever followed `-c`, already trimmed. The length guard is the
/// whole point: a lone quote — `python -c "`, the unterminated one-liner a model
/// emits routinely — satisfies BOTH `starts_with` and `ends_with` on the SAME
/// character, so the bare `&rest[1..rest.len() - 1]` was the inverted range
/// `[1..0]`, and that panic unwound out of the daemon instead of letting python
/// answer for a malformed command. Below two characters there is no pair to
/// strip, so the payload passes through as written.
#[cfg(windows)]
fn strip_outer_quotes(rest: &str) -> &str {
    let quoted = rest.len() >= 2
        && ((rest.starts_with('"') && rest.ends_with('"'))
            || (rest.starts_with('\'') && rest.ends_with('\'')));
    if quoted {
        // Both bounds land on the ASCII quotes themselves, so they stay char
        // boundaries however multi-byte the code between them is.
        &rest[1..rest.len() - 1]
    } else {
        rest
    }
}

/// Translate `X:\a\b` drive paths into the `/x/a/b` form Git Bash understands.
///
/// In bash a backslash is an ESCAPE, so `cat > D:\Development\ws\minidb.sh`
/// silently becomes `cat > D:Developmentwsminidb.sh` — a file with a garbage
/// name, created without an error anyone can see. Observed live 2026-07-27: a
/// run's largest and newest work (3 KB) landed in a file literally named
/// `D:Developmentnanna-benchrun-qwen3-5-9bminidb.sh`, while the artifact the
/// acceptance tests read sat frozen and half-finished.
///
/// This is the shell-layer half of "handle absolute and relative paths
/// gracefully" — the tool layer already repairs argument paths; a command
/// string needs the same courtesy, because the model cannot tell that the
/// path form it was shown (native, with backslashes) is not the form the
/// shell accepts.
///
/// Narrow by construction:
/// - Only a drive-letter prefix (`X:\` or `X:/`) at a token boundary starts a
///   rewrite, so `\d`, `\n`, `C:` alone, and Windows paths inside SINGLE
///   quotes (bash's literal string) are untouched.
/// - The rewrite ends at whitespace or a shell metacharacter, so redirections
///   and pipes survive.
/// - Forward slashes are already valid; those paths are normalized to `/x/…`
///   too, which is the same location.
#[cfg(windows)]
fn normalize_drive_paths(command: &str) -> std::borrow::Cow<'_, str> {
    let b = command.as_bytes();
    let n = b.len();
    // Fast path: a drive path needs a colon.
    if !command.contains(':') {
        return std::borrow::Cow::Borrowed(command);
    }

    let mut out = String::with_capacity(n);
    let mut changed = false;
    let mut i = 0usize;
    let mut single_quote = false;
    let mut prev_boundary = true;

    while i < n {
        let c = b[i];
        if c == b'\'' {
            single_quote = !single_quote;
            out.push('\'');
            i += 1;
            prev_boundary = false;
            continue;
        }
        // A literal single-quoted string is exactly what a careful author uses
        // to protect a Windows path; never second-guess it.
        if single_quote {
            out.push(c as char);
            i += 1;
            continue;
        }

        let starts_drive = prev_boundary
            && c.is_ascii_alphabetic()
            && i + 2 < n
            && b[i + 1] == b':'
            && (b[i + 2] == b'\\' || b[i + 2] == b'/');

        if starts_drive {
            let drive = (c as char).to_ascii_lowercase();
            out.push('/');
            out.push(drive);
            i += 2; // skip "X:"
            while i < n {
                let p = b[i];
                if p == b' ' || p == b'\t' || p == b'"' || p == b'\'' || p == b'|'
                    || p == b'>' || p == b'<' || p == b'&' || p == b';' || p == b'\n'
                {
                    break;
                }
                out.push(if p == b'\\' { '/' } else { p as char });
                i += 1;
            }
            changed = true;
            prev_boundary = false;
            continue;
        }

        out.push(c as char);
        prev_boundary = matches!(c, b' ' | b'\t' | b'"' | b'|' | b'&' | b';' | b'(' | b'\n' | b'=');
        i += 1;
    }

    if changed {
        std::borrow::Cow::Owned(out)
    } else {
        std::borrow::Cow::Borrowed(command)
    }
}

/// Rewrite the small set of cmd.exe idioms that Git Bash rejects.
///
/// Currently only `cd /d <path>` → `cd <path>`. `/d` is a cmd.exe-only flag
/// ("also change drive"); Git Bash's `cd` treats it as a second argument and
/// dies with "cd: too many arguments" (observed in a real run log). We strip
/// `/d` (case-insensitive) ONLY when it is a whole token immediately after a
/// command-boundary `cd` AND a further token follows — the exact cmd idiom. A
/// bare `cd /d`, `cd /data`, `cd /d/x`, a quoted occurrence, or a non-boundary
/// `cd` are all left alone, so no currently-working command changes. Handles the
/// idiom at every command boundary in a compound command (`cd /d D:\x && …`).
///
/// Pure (no IO). Windows-only: on Unix `cd /d` legitimately means dir `/d`.
#[cfg(windows)]
fn normalize_cmdisms(command: &str) -> std::borrow::Cow<'_, str> {
    if !(command.contains("/d") || command.contains("/D")) {
        return std::borrow::Cow::Borrowed(command);
    }
    let b = command.as_bytes();
    let n = b.len();
    let mut out: Vec<u8> = Vec::with_capacity(n);
    let mut changed = false;
    let mut i = 0usize;
    let mut at_boundary = true; // start of string is a command boundary
    let mut quote: u8 = 0; // 0 = none, else b'\'' / b'"'
    while i < n {
        let c = b[i];
        if quote != 0 {
            out.push(c);
            if c == quote {
                quote = 0;
            }
            i += 1;
            continue;
        }
        if c == b'\'' || c == b'"' {
            quote = c;
            out.push(c);
            i += 1;
            at_boundary = false;
            continue;
        }
        if at_boundary {
            if c == b' ' || c == b'\t' {
                out.push(c);
                i += 1;
                continue;
            }
            if let Some(path_off) = match_cd_slash_d(&b[i..]) {
                out.extend_from_slice(b"cd ");
                i += path_off; // resume at the path token
                changed = true;
                at_boundary = false;
                continue;
            }
            at_boundary = false;
        }
        out.push(c);
        // These open a new command boundary for the NEXT char.
        if matches!(c, b';' | b'\n' | b'(' | b'|' | b'&') {
            at_boundary = true;
        }
        i += 1;
    }
    if changed {
        // Safe: we only ever copied original bytes verbatim plus ASCII "cd ".
        std::borrow::Cow::Owned(String::from_utf8(out).unwrap_or_else(|_| command.to_string()))
    } else {
        std::borrow::Cow::Borrowed(command)
    }
}

/// If `s` begins with the cmd idiom `cd /d <path…>`, return the byte offset of
/// the path token (so the caller can emit `cd ` then resume there). Requires
/// `cd`+ws, then `/d` or `/D` as a whole token (ws after), then a following
/// token. Otherwise `None`.
#[cfg(windows)]
fn match_cd_slash_d(s: &[u8]) -> Option<usize> {
    // `cd` (case-insensitive) then whitespace.
    if s.len() < 3 {
        return None;
    }
    if !(s[0] | 0x20 == b'c' && s[1] | 0x20 == b'd') {
        return None;
    }
    if !(s[2] == b' ' || s[2] == b'\t') {
        return None;
    }
    let mut j = 3;
    while j < s.len() && (s[j] == b' ' || s[j] == b'\t') {
        j += 1;
    }
    // `/d` or `/D` as a whole token.
    if j + 2 >= s.len() {
        return None;
    }
    if s[j] != b'/' || (s[j + 1] | 0x20) != b'd' {
        return None;
    }
    if !(s[j + 2] == b' ' || s[j + 2] == b'\t') {
        return None; // must be a whole token, not `/data` or `/d/x`
    }
    let mut k = j + 3;
    while k < s.len() && (s[k] == b' ' || s[k] == b'\t') {
        k += 1;
    }
    if k >= s.len() {
        return None; // no path token follows → not the idiom
    }
    Some(k)
}

/// Locate Git-for-Windows `bash.exe`, cached. Explicitly **not** WSL's
/// `C:\Windows\System32\bash.exe` (different filesystem semantics), only a real
/// Git Bash under a `Git\bin` install. Returns `None` if Git Bash isn't installed.
#[cfg(windows)]
fn git_bash_path() -> Option<&'static Path> {
    use std::sync::OnceLock;
    static CACHE: OnceLock<Option<PathBuf>> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            let mut candidates: Vec<PathBuf> = Vec::new();
            for var in ["ProgramFiles", "ProgramW6432", "ProgramFiles(x86)"] {
                if let Ok(base) = std::env::var(var) {
                    candidates.push(PathBuf::from(base).join("Git").join("bin").join("bash.exe"));
                }
            }
            if let Ok(local) = std::env::var("LOCALAPPDATA") {
                candidates
                    .push(PathBuf::from(local).join("Programs").join("Git").join("bin").join("bash.exe"));
            }
            candidates.into_iter().find(|p| p.is_file())
        })
        .as_deref()
}

/// Default per-command exec timeout, in seconds, for an unremarkable command.
pub const EXEC_DEFAULT_TIMEOUT_SECS: u64 = 30;

/// Auto-detected exec timeout, in seconds, for build/VCS tools (git, cargo,
/// npm, …) which routinely exceed [`EXEC_DEFAULT_TIMEOUT_SECS`] on large repos.
/// This is the widest window the bridge picks on its own; a caller can still
/// request more explicitly.
pub const EXEC_MAX_AUTODETECT_SECS: u64 = 120;

/// The per-command execution timeout (seconds) to use when the caller gives no
/// explicit override, auto-detected from the command text.
///
/// Pure (no IO). Extracted from `exec_with_timeout` so the outer script-engine
/// deadline can be sized against the *same* value the bridge will actually
/// enforce, and so the classification is unit-testable.
#[must_use]
pub fn default_exec_timeout_secs(command: &str) -> u64 {
    let cmd_lower = command.to_lowercase();
    if cmd_lower.starts_with("git ")
        || cmd_lower.starts_with("cargo ")
        || cmd_lower.starts_with("npm ")
        || cmd_lower.starts_with("pnpm ")
        || cmd_lower.starts_with("yarn ")
        || cmd_lower.starts_with("pip ")
        || cmd_lower.starts_with("make ")
        || cmd_lower.starts_with("cmake ")
        || cmd_lower.starts_with("dotnet ")
        || cmd_lower.starts_with("mvn ")
        || cmd_lower.starts_with("gradle ")
        || cmd_lower.contains("cargo ")
        || cmd_lower.contains("git checkout")
        || cmd_lower.contains("git clone")
        || cmd_lower.contains("git fetch")
        || cmd_lower.contains("git pull")
        || cmd_lower.contains("cargo build")
        || cmd_lower.contains("cargo check")
        || cmd_lower.contains("cargo clippy")
        || cmd_lower.contains("cargo test")
    {
        EXEC_MAX_AUTODETECT_SECS
    } else {
        EXEC_DEFAULT_TIMEOUT_SECS
    }
}

/// One incremental read from a live child pipe into `sink`. Returns `true`
/// once that pipe is finished — EOF, an unreadable pipe, or no pipe at all.
///
/// `read_buf` appends to `sink` and grows it itself, so the in-flight buffering
/// is exactly what `wait_with_output` did: no cap is imposed here and none is
/// needed — the exec deadline bounds the time, and the tool-result
/// summarize/truncate routing bounds the size. It is also cancel-safe, so the
/// deadline branch winning a `select!` race cannot swallow bytes that had
/// already been read.
async fn read_chunk<R>(reader: Option<&mut R>, sink: &mut Vec<u8>) -> bool
where
    R: tokio::io::AsyncRead + Unpin,
{
    use tokio::io::AsyncReadExt;
    match reader {
        Some(reader) => !matches!(reader.read_buf(sink).await, Ok(n) if n > 0),
        None => true,
    }
}

/// Collect whatever a killed command's pipe can hand over WITHOUT waiting.
///
/// Called only after the tree has been reaped, to pick up the last bytes it
/// wrote: on Windows the in-flight read completes on the blocking pool while
/// `taskkill` runs, on Unix the bytes simply sit in the pipe. The `ready`
/// branch ends the loop at the first read that would block, so this never
/// waits for a byte that has not already arrived — a writer that somehow
/// survived BOTH the tree walk and the Job Object cannot turn "killed at the
/// deadline" into a hang, which is why the drain is not joined to EOF here.
async fn drain_ready<R>(reader: Option<&mut R>, sink: &mut Vec<u8>)
where
    R: tokio::io::AsyncRead + Unpin,
{
    use tokio::io::AsyncReadExt;
    let Some(reader) = reader else {
        return;
    };
    loop {
        tokio::select! {
            biased;
            res = reader.read_buf(sink) => {
                if !matches!(res, Ok(n) if n > 0) {
                    return;
                }
            }
            () = std::future::ready(()) => return,
        }
    }
}

impl NannaBridge {
    /// Create a new bridge with the given permissions
    #[must_use]
    pub fn new(permissions: ToolPermissions) -> Self {
        Self {
            permissions,
            http_client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
            logs: Arc::new(RwLock::new(Vec::new())),
            tool_definitions: None,
            tool_search: None,
            services: Arc::new(HashMap::new()),
            default_workdir: None,
            session_id: None,
        }
    }

    /// Set the default working directory for exec commands.
    /// When set, this overrides the home directory fallback.
    #[must_use]
    pub fn with_default_workdir(mut self, workdir: impl Into<PathBuf>) -> Self {
        self.default_workdir = Some(workdir.into());
        self
    }

    /// Set the session ID for session-scoped operations.
    #[must_use]
    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Get the current session ID.
    #[must_use]
    pub fn session_id(&self) -> Option<&str> {
        self.session_id.as_deref()
    }

    /// Get the default working directory as a string.
    #[must_use]
    pub fn workdir(&self) -> Option<&str> {
        self.default_workdir.as_ref().and_then(|p| p.to_str())
    }

    /// Set tool definitions for `Nanna.listTools()`
    #[must_use]
    pub fn with_tool_definitions(mut self, defs: Value) -> Self {
        self.tool_definitions = Some(defs);
        self
    }

    /// Set the ranked tool-search function for `Nanna.searchTools()`
    #[must_use]
    pub fn with_tool_search(mut self, search: ToolSearchFn) -> Self {
        self.tool_search = Some(search);
        self
    }

    /// Resolve a path: expand `~` to home directory, resolve relative paths against
    /// the default working directory (if set) or home dir.
    /// Works on Windows, macOS, Linux, Android, and iOS.
    /// Falls back to the raw path if no base directory can be determined.
    fn resolve_path_with_workdir(path: &str, default_workdir: Option<&Path>) -> PathBuf {
        let path = path.trim();
        
        // Expand ~ or ~/ (but not ~username which we don't support)
        if (path == "~" || path.starts_with("~/") || path.starts_with("~\\"))
            && let Some(home) = Self::home_dir() {
                let rest = path.strip_prefix("~/")
                    .or_else(|| path.strip_prefix("~\\"))
                    .unwrap_or("");
                return home.join(rest);
            }
        
        // An MSYS drive path the shell itself printed (`/d/Development/x`).
        //
        // On Windows such a path HAS a root but no drive prefix, so it reports
        // `is_relative()` and the branch below joins it onto the workspace —
        // turning `/d/Development/x` into `<drive-of-workdir>\d\Development\x`.
        // Reads of a real file then answer "does not exist" while writes create
        // a phantom tree and report success, so the same string addresses two
        // different filesystems depending on which tool receives it. Git Bash
        // is the shell here, so every path `exec` prints comes back in this
        // shape and is routinely handed straight to a file tool.
        #[cfg(windows)]
        if let Some(native) = Self::msys_drive_path(path) {
            return native;
        }

        let p = Path::new(path);
        if p.is_relative() && !path.is_empty() {
            // Resolve relative to the active workspace working directory. The
            // registry seeds `default_workdir` from the active workspace (at boot,
            // on switch, and per-chat), so bare relative paths land in the workspace
            // the user is in — not the home dir. Home is only a last resort when
            // there is genuinely no active workspace (global mode). `~`-prefixed
            // paths were already expanded above.
            if let Some(wd) = default_workdir {
                let literal = wd.join(p);
                return Self::repair_redundant_prefix(p, wd).unwrap_or(literal);
            }
            if let Some(home) = Self::home_dir() {
                return home.join(p);
            }
        }
        p.to_path_buf()
    }

    /// Translate an MSYS drive path (`/d/Development/x`, or `/d:/Development/x`)
    /// into the native `D:\Development\x` form.
    ///
    /// Literal-first, the same precedence [`Self::repair_redundant_prefix`]
    /// uses: a genuine single-letter directory at the root of the current drive
    /// stays addressable, so this only rewrites when the literal path does not
    /// exist and the native one does. `/d` alone is left alone — a drive path
    /// needs something after the drive to name.
    #[cfg(windows)]
    fn msys_drive_path(path: &str) -> Option<PathBuf> {
        let rest = path.strip_prefix('/')?;
        let mut chars = rest.chars();
        let drive = chars.next().filter(char::is_ascii_alphabetic)?;
        // Accept both `/d/...` and the `/d:/...` form some tools emit.
        let tail = match chars.next()? {
            ':' => rest.get(3..)?,
            '/' => rest.get(2..)?,
            _ => return None,
        };
        if tail.is_empty() {
            return None;
        }

        // A real `/d/...` on this filesystem wins, exactly as a real nested
        // directory wins over prefix repair.
        if Path::new(path).exists() {
            return None;
        }

        let native = PathBuf::from(format!(
            "{}:\\{}",
            drive.to_ascii_uppercase(),
            tail.replace('/', "\\")
        ));
        native.exists().then_some(native)
    }

    /// Repair a relative path that redundantly repeats the tail of the working
    /// directory, e.g. `minidb-gui-qwen35/minidb` given a workdir that already
    /// ends in `minidb-gui-qwen35`.
    ///
    /// A model that learns its location (from `pwd`, a prompt line, or an
    /// earlier tool result) will sometimes address files by re-appending part
    /// of that path to a cwd that already includes it. Taken literally that
    /// silently creates a shadow tree one level deeper — observed live
    /// 2026-07-26: an hour of edits landed in `<ws>/<ws-leaf>/minidb` while the
    /// acceptance tests read `<ws>/minidb`, and the run scored 0.
    ///
    /// Rules, in order, so this never overrides a real file:
    /// 1. If the literal join exists, it wins — a genuine nested directory of
    ///    the same name is still addressable.
    /// 2. Otherwise, for the longest overlap first, if the de-duplicated
    ///    target exists, use it (reads and edits find the intended file).
    /// 3. Otherwise, if the de-duplicated *parent* exists while the literal
    ///    parent does not, use it (a write lands beside the files it belongs
    ///    with instead of conjuring a duplicate directory).
    ///
    /// Returns `None` when nothing needs repairing, leaving the literal join.
    fn repair_redundant_prefix(rel: &Path, workdir: &Path) -> Option<PathBuf> {
        let literal = workdir.join(rel);
        if literal.exists() {
            return None;
        }

        let rel_parts: Vec<_> = rel.components().collect();
        let wd_parts: Vec<_> = workdir.components().collect();
        // The overlap can cover at most all but the last component of the
        // relative path (something must remain to name the target), and at
        // most the whole working directory.
        let max_overlap = rel_parts.len().saturating_sub(1).min(wd_parts.len());

        let mut fallback = None;
        for k in (1..=max_overlap).rev() {
            let head = &rel_parts[..k];
            let wd_tail = &wd_parts[wd_parts.len() - k..];
            if head != wd_tail {
                continue;
            }
            let remainder: PathBuf = rel_parts[k..].iter().collect();
            let candidate = workdir.join(&remainder);
            if candidate.exists() {
                return Some(candidate);
            }
            if fallback.is_none() {
                let literal_parent_missing =
                    literal.parent().is_some_and(|p| !p.exists());
                let candidate_parent_exists =
                    candidate.parent().is_some_and(Path::exists);
                if literal_parent_missing && candidate_parent_exists {
                    fallback = Some(candidate);
                }
            }
        }
        fallback
    }

    /// Resolve a path using the instance's default working directory.
    fn resolve_path(&self, path: &str) -> PathBuf {
        Self::resolve_path_with_workdir(path, self.default_workdir.as_deref())
    }

    /// Get the user's home directory, cross-platform.
    /// Uses `directories` crate which supports Windows, macOS, Linux.
    /// On Android/iOS, falls back to current directory.
    fn home_dir() -> Option<PathBuf> {
        directories::BaseDirs::new()
            .map(|d| d.home_dir().to_path_buf())
            .or_else(|| std::env::current_dir().ok())
    }

    /// Set service functions for `Nanna.service()`
    #[must_use]
    pub fn with_services(mut self, services: HashMap<String, ServiceFn>) -> Self {
        self.services = Arc::new(services);
        self
    }

    /// Get stored tool definitions (for `Nanna.listTools()`)
    #[must_use]
    pub const fn list_tools(&self) -> Option<&Value> {
        self.tool_definitions.as_ref()
    }

    /// Run the ranked tool search (for `Nanna.searchTools()`).
    ///
    /// Returns `None` when no search function was attached (older embedding
    /// contexts, test harnesses) — the caller maps that to an empty array,
    /// never a thrown error.
    #[must_use]
    pub fn search_tools(&self, query: &str, limit: usize) -> Option<Value> {
        self.tool_search.as_ref().map(|f| f(query, limit))
    }

    /// Call a registered service by name
    ///
    /// # Errors
    ///
    /// Returns a message if no service named `name` is registered, or the
    /// service's own error message if the call fails.
    pub async fn call_service(&self, name: &str, params: Value) -> std::result::Result<Value, String> {
        let service = self.services.get(name)
            .ok_or_else(|| format!("Service not found: {name}"))?;
        service(params).await
    }

    /// Execute a shell command (if permitted)
    ///
    /// `timeout_secs`: optional override for the execution timeout (default: 30s).
    ///
    /// # Errors
    ///
    /// The same as [`Self::exec_with_timeout`].
    pub async fn exec(&self, command: &str, workdir: Option<&str>) -> Result<ExecResponse> {
        self.exec_with_timeout(command, workdir, None).await
    }

    /// Execute a shell command with an optional timeout override.
    ///
    /// A command that overruns its deadline is not an error: it is killed and
    /// returned with `timed_out: true` and whatever it had printed.
    ///
    /// # Errors
    ///
    /// Returns [`ScriptError::Permission`] if the tool lacks the `run`
    /// permission or the platform (Android, iOS) has no shell, and
    /// [`ScriptError::Bridge`] if the command cannot be spawned or waiting on
    /// it fails.
    pub async fn exec_with_timeout(&self, command: &str, workdir: Option<&str>, timeout_secs: Option<u64>) -> Result<ExecResponse> {
        tracing::debug!(target: "bridge", "exec called: command={}, run_permission={}", command, self.permissions.run);
        
        if !self.permissions.run {
            tracing::warn!(target: "bridge", "Shell execution denied - run permission is false");
            return Err(ScriptError::Permission(
                "Shell execution not permitted. Tool needs 'run' permission.".to_string(),
            ));
        }

        // On mobile platforms, shell exec is not available
        if cfg!(target_os = "android") || cfg!(target_os = "ios") {
            return Err(ScriptError::Permission(
                "Shell execution is not available on mobile platforms".to_string(),
            ));
        }

        // Rewrite cmd.exe idioms Git Bash rejects (e.g. `cd /d`) before routing,
        // so a model that emits cmd.exe habits doesn't hit "cd: too many arguments".
        // Every downstream use of `command` (python detect, classify, shell args,
        // timeout auto-detect, tracing) transparently sees the normalized string.
        #[cfg(windows)]
        let normalized = normalize_cmdisms(command);
        #[cfg(windows)]
        let command: &str = normalized.as_ref();
        // Then translate `D:\ws\file` → `/d/ws/file`, because bash reads those
        // backslashes as escapes and would write a garbage-named file instead
        // of failing loudly.
        #[cfg(windows)]
        let drive_normalized = normalize_drive_paths(command);
        #[cfg(windows)]
        let command: &str = drive_normalized.as_ref();

        // Determine shell based on OS (see `windows_shell_command` for the routing).
        #[cfg(windows)]
        let mut cmd = windows_shell_command(command);
        #[cfg(not(windows))]
        let mut cmd = {
            let mut c = tokio::process::Command::new("sh");
            c.args(["-c", command]);
            c
        };

        // No explicit workdir: run in the active workspace directory (the
        // registry seeds `default_workdir` from it). This is what makes exec
        // run "in the workspace you're in" rather than the home dir. The home
        // dir is the last resort, only when there is genuinely no workspace.
        let run_dir = workdir
            .map(|wd| self.resolve_path(wd))
            .or_else(|| self.default_workdir.clone())
            .or_else(Self::home_dir);
        if let Some(ref dir) = run_dir {
            // A missing directory fails the spawn with the same ENOENT a
            // missing program does, so it used to read "Failed to execute
            // command: No such file or directory" — which names neither the
            // directory nor the fact that the command itself was fine.
            if !dir.is_dir() {
                return Err(ScriptError::Bridge(missing_workdir_message(
                    dir,
                    workdir.is_some(),
                )));
            }
            cmd.current_dir(dir);
        }

        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        // Tell well-behaved tools to skip color output.
        // We also strip ANSI escapes from the output as a fallback.
        cmd.env("NO_COLOR", "1");
        cmd.env("TERM", "dumb");

        // Do not hand the child the AppImage's private library path. When the
        // daemon is a sidecar of an AppImage-launched GUI, AppRun has put the
        // bundle's own `usr/lib` at the FRONT of `LD_LIBRARY_PATH` so the GUI
        // finds its bundled WebKit. A child inherits that verbatim, and every
        // host binary it runs then loads the bundle's `liblzma`/`libpcre2`
        // instead of the system's: `git` warns "no version information
        // available" on every call, and a freshly built test binary dies at
        // load with "version `XZ_5.4' not found" (observed 2026-09-17, on a
        // `cargo test` run through `exec`). Only the bundle's own entries go —
        // whatever the user had set stays.
        #[cfg(unix)]
        {
            let appdir = std::env::var_os("APPDIR");
            let inherited = std::env::var_os("LD_LIBRARY_PATH");
            match scrub_appimage_library_path(appdir.as_deref(), inherited.as_deref()) {
                ScrubbedLibraryPath::Unchanged => {}
                ScrubbedLibraryPath::Cleared => {
                    cmd.env_remove("LD_LIBRARY_PATH");
                }
                ScrubbedLibraryPath::Narrowed(kept) => {
                    cmd.env("LD_LIBRARY_PATH", kept);
                }
            }
        }

        // Smart default timeout: longer for known slow commands.
        // Git, cargo, npm, pip, and build tools regularly exceed 30s on large repos.
        let timeout = timeout_secs.unwrap_or_else(|| default_exec_timeout_secs(command));

        // Reap our direct child if this future is dropped (e.g. the outer script
        // engine deadline fires). The per-child Job Object below widens this
        // backstop to the whole subtree on Windows.
        cmd.kill_on_drop(true);

        // Isolate the child from OUR process group.
        //
        // Without this the child shares the parent's console/process group, so a
        // script that terminates its group takes the agent down with it.
        // Observed live (2026-07-25): once models were told to run their own
        // acceptance checks, `exec sh tests/test_NN.sh` killed the harness
        // outright — exit 0xffffffff, no panic — reproduced across two models,
        // each dying seconds after its first self-check. The very same command
        // is safe through the acceptance runner, which spawns detached.
        //
        // This is not eval-specific: any agent that writes a script and runs it
        // can otherwise kill the daemon executing it. Isolation also makes the
        // timeout path honest — the tree-kill below now bounds exactly the
        // subtree we created, and can never walk up into us.
        #[cfg(windows)]
        {
            // CREATE_NEW_PROCESS_GROUP: console events (Ctrl+C / Ctrl+Break)
            // raised by or for the child stop at the child.
            const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
            cmd.creation_flags(CREATE_NEW_PROCESS_GROUP);
        }
        #[cfg(unix)]
        {
            // setsid(): new session + process group, so a child that signals its
            // group (or dies on a terminal signal) cannot reach us.
            cmd.process_group(0);
        }

        // Wall clock for the result's `elapsed_ms`. Started before the spawn so
        // it measures what the command actually cost the caller, and reported on
        // BOTH exits — a run that finished and a run that was killed are then
        // directly comparable.
        let started = std::time::Instant::now();

        let child = cmd
            .spawn()
            .map_err(|e| ScriptError::Bridge(format!("Failed to execute command: {e}")))?;
        Self::collect_exec(child, started, timeout).await
    }

    /// Wait for a spawned exec child under its deadline and collect its output.
    ///
    /// `started` is the wall clock taken before the spawn; `timeout` is the
    /// deadline in seconds.
    ///
    /// # Errors
    ///
    /// Returns [`ScriptError::Bridge`] if waiting on the child fails. An
    /// overrun is reported in the response, not as an error.
    async fn collect_exec(
        mut child: tokio::process::Child,
        started: std::time::Instant,
        timeout: u64,
    ) -> Result<ExecResponse> {
        // Capture the pid *before* the wait future consumes the child, so a timeout
        // can kill the whole process tree rooted here (not just the shell).
        let pid = child.id();
        // Windows: put the child in its OWN kill-on-close Job Object before it
        // can fork. The taskkill walk below only reaches descendants of a pid
        // it can still see — `foo &` leaves a grandchild holding our pipes
        // after the shell died, invisible to the walk (2026-08-08: such a
        // survivor pinned a blocking-pool pipe read and hung teardown 25+
        // min). Job membership is inherited, so terminating the job reaps the
        // whole subtree regardless; dropping this guard (future cancelled)
        // does the same. Unix needs none of this: the process group IS the
        // subtree, dead shell or not.
        let mut job = ChildJob::assign(&child);

        // Drain the pipes ourselves instead of `wait_with_output()`. The
        // buffering is identical — the reads grow these two vectors exactly as
        // `wait_with_output` grew its own, no cap is introduced (the deadline
        // is still the time bound, and the tool-result summarize/truncate
        // routing is still the size bound) — but the bytes accumulate in OUR
        // hands. `wait_with_output` owns its buffers *inside* the future and
        // drops them when the deadline cancels it, which is why every overrun
        // used to reach the model as "nothing ran" while its side effects sat
        // on disk.
        let mut out_pipe = child.stdout.take();
        let mut err_pipe = child.stderr.take();
        let mut out_done = out_pipe.is_none();
        let mut err_done = err_pipe.is_none();
        let mut stdout_bytes: Vec<u8> = Vec::new();
        let mut stderr_bytes: Vec<u8> = Vec::new();
        let wait = child.wait();
        tokio::pin!(wait);
        let deadline = tokio::time::sleep(std::time::Duration::from_secs(timeout));
        tokio::pin!(deadline);
        let mut status = None;

        // "Completed" means exactly what `wait_with_output` meant: the child
        // exited AND both pipes hit EOF. A backgrounded grandchild still
        // holding our write handles therefore still runs the deadline out —
        // that is the `foo &` shape the tree-kill and the Job Object exist for.
        let exit = loop {
            if out_done && err_done && status.is_some() {
                break status;
            }
            tokio::select! {
                done = read_chunk(out_pipe.as_mut(), &mut stdout_bytes), if !out_done => {
                    out_done = done;
                }
                done = read_chunk(err_pipe.as_mut(), &mut stderr_bytes), if !err_done => {
                    err_done = done;
                }
                res = &mut wait, if status.is_none() => {
                    status = Some(res.map_err(|e| {
                        ScriptError::Bridge(format!("Failed to execute command: {e}"))
                    })?);
                }
                () = &mut deadline => break None,
            }
        };

        let Some(exit_status) = exit else {
            // Overran the deadline. Kill the entire tree — the shell *and* any
            // long-running grandchild like cargo/git — so it can't outlive the
            // tool call and hold a workspace/build lock (the failure mode that
            // deadlocked repeated exec calls against each other). Walk first
            // while `wait` still owns the child (covers a live tree, including
            // anything spawned before the job assignment landed), then
            // terminate the job to sweep descendants the walk can't see.
            if let Some(pid) = pid {
                kill_process_tree(pid).await;
            }
            if let Some(job) = job.take() {
                job.terminate();
            }
            // Then tell the truth. This used to return `Err(Timeout)`, which the
            // exec tool rendered as "Nothing ran" — so the model retried a
            // command that HAD run, re-doing side effects already on disk. The
            // kill is unchanged; only the report is honest now: a result that
            // says it was killed, how long it lived, and what it printed first.
            // Tool errors return, they never throw.
            drain_ready(out_pipe.as_mut(), &mut stdout_bytes).await;
            drain_ready(err_pipe.as_mut(), &mut stderr_bytes).await;
            return Ok(ExecResponse {
                success: false,
                code: None,
                stdout: strip_ansi_escapes(&String::from_utf8_lossy(&stdout_bytes)),
                stderr: strip_ansi_escapes(&String::from_utf8_lossy(&stderr_bytes)),
                timed_out: true,
                elapsed_ms: u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
            });
        };

        // Completed: pipes EOF'd, shell exited. Anything still running is a
        // deliberately detached background process (output redirected — e.g. a
        // server the agent started on purpose). Spare it; the daemon-wide Job
        // Object still bounds it.
        if let Some(job) = job.take() {
            job.disarm();
        }

        let stdout = strip_ansi_escapes(&String::from_utf8_lossy(&stdout_bytes));
        let stderr = strip_ansi_escapes(&String::from_utf8_lossy(&stderr_bytes));

        Ok(ExecResponse {
            success: exit_status.success(),
            code: exit_status.code(),
            stdout,
            stderr,
            timed_out: false,
            elapsed_ms: u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
        })
    }

    /// Get the current platform (windows, darwin, linux)
    #[must_use]
    pub const fn platform() -> &'static str {
        if cfg!(windows) {
            "win32"
        } else if cfg!(target_os = "macos") {
            "darwin"
        } else if cfg!(target_os = "android") {
            "android"
        } else if cfg!(target_os = "ios") {
            "ios"
        } else {
            "linux"
        }
    }

    /// Fetch a URL (if permitted)
    ///
    /// # Errors
    ///
    /// Returns [`ScriptError::Permission`] if network access to the URL's host
    /// is not permitted, and [`ScriptError::Bridge`] if the URL does not parse,
    /// the method is not one of GET/POST/PUT/DELETE/PATCH, the request fails, or
    /// the response body cannot be read.
    pub async fn fetch(&self, url: &str, options: Option<FetchOptions>) -> Result<FetchResponse> {
        // Parse URL to check host
        let parsed = url::Url::parse(url)
            .map_err(|e| ScriptError::Bridge(format!("Invalid URL: {e}")))?;
        
        let host = parsed.host_str().unwrap_or("");
        
        if !self.permissions.allows_net(host) {
            return Err(ScriptError::Permission(format!(
                "Network access to '{host}' not permitted"
            )));
        }

        let options = options.unwrap_or_default();
        
        let mut request = match options.method.as_deref().unwrap_or("GET") {
            "GET" => self.http_client.get(url),
            "POST" => self.http_client.post(url),
            "PUT" => self.http_client.put(url),
            "DELETE" => self.http_client.delete(url),
            "PATCH" => self.http_client.patch(url),
            m => return Err(ScriptError::Bridge(format!("Unsupported method: {m}"))),
        };

        // Add headers
        if let Some(headers) = options.headers {
            for (key, value) in headers {
                request = request.header(&key, &value);
            }
        }

        // Add body
        if let Some(body) = options.body {
            request = request.body(body);
        }

        let response = request
            .send()
            .await
            .map_err(|e| ScriptError::Bridge(format!("Fetch failed: {e}")))?;

        let status = response.status().as_u16();
        let headers: std::collections::HashMap<String, String> = response
            .headers()
            .iter()
            .filter_map(|(k, v)| v.to_str().ok().map(|v| (k.to_string(), v.to_string())))
            .collect();
        
        let body = read_body_capped(response, READ_FILE_BYTES_MAX).await?;

        Ok(FetchResponse {
            status,
            headers,
            body,
        })
    }

    /// Read a file (if permitted)
    ///
    /// # Errors
    ///
    /// Returns [`ScriptError::Permission`] if the resolved path is outside the
    /// tool's read scope, and [`ScriptError::Bridge`] if the file cannot be read
    /// (including when its contents are not valid UTF-8, or it is larger than
    /// [`READ_FILE_BYTES_MAX`]).
    pub async fn read_file(&self, path: &str) -> Result<String> {
        let path = self.resolve_path(path);
        
        if !self.permissions.allows_read(&path) {
            return Err(ScriptError::Permission(format!(
                "Read access to '{}' not permitted",
                path.display()
            )));
        }

        read_to_string_capped(&path, READ_FILE_BYTES_MAX).await
    }

    /// Write a file (if permitted)
    ///
    /// # Errors
    ///
    /// Returns [`ScriptError::Bridge`] if another write to the same path is in
    /// progress in this process or the write itself fails, and
    /// [`ScriptError::Permission`] if the resolved path is outside the tool's
    /// write scope.
    pub async fn write_file(&self, path: &str, content: &str) -> Result<()> {
        // Ensure lock is released even on error
        struct FileGuard(std::path::PathBuf);
        impl Drop for FileGuard {
            fn drop(&mut self) {
                if let Ok(mut locks) = FILE_WRITE_LOCKS.lock() {
                    locks.remove(&self.0);
                }
            }
        }

        // Advisory file lock: prevent concurrent writes to the same path
        let canonical = self.resolve_path(path).canonicalize().unwrap_or_else(|_| self.resolve_path(path));
        {
            let mut locks = FILE_WRITE_LOCKS.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            if locks.contains(&canonical) {
                return Err(ScriptError::Bridge(
                    format!("File is being written by another agent: {path}")
                ));
            }
            locks.insert(canonical.clone());
        }
        let _guard = FileGuard(canonical);

        let path = self.resolve_path(path);
        
        if !self.permissions.allows_write(&path) {
            return Err(ScriptError::Permission(format!(
                "Write access to '{}' not permitted",
                path.display()
            )));
        }

        // Snapshot what this write is about to replace, so it can be undone.
        // Never blocks the write: losing the undo beats losing the work.
        if let Some(history) = crate::file_history::installed()
            && !crate::file_history::is_tool_housekeeping(&path)
            && let Err(e) = history
                .record_before_write(self.session_id.as_deref(), &path)
                .await
        {
            tracing::warn!(path = %path.display(), error = %e, "pre-write snapshot failed; writing without an undo");
        }

        // Create parent directories if needed
        if let Some(parent) = path.parent()
            && !parent.exists() {
                tokio::fs::create_dir_all(parent).await.ok();
            }

        // Name the cause machine-readably. The OS message alone is localized
        // prose ("Access is denied." / "Permission denied"), so a caller that
        // wants to tell a NON-RETRYABLE failure (a write-protected file) from a
        // transient one (Interrupted, WouldBlock, a sharing violation) has
        // nothing stable to match on. `io::ErrorKind`'s Debug name is stable
        // and locale-independent; the OS message still follows it verbatim.
        tokio::fs::write(&path, content).await.map_err(|e| {
            ScriptError::Bridge(format!(
                "Failed to write '{}' (kind={:?}): {e}",
                path.display(),
                e.kind()
            ))
        })
    }

    /// Log a message
    pub async fn log(&self, level: LogLevel, message: String) {
        let entry = LogEntry {
            level,
            message,
            timestamp: chrono::Utc::now(),
        };

        // Also emit to tracing
        match entry.level {
            LogLevel::Debug => tracing::debug!(target: "script", "{}", entry.message),
            LogLevel::Info => tracing::info!(target: "script", "{}", entry.message),
            LogLevel::Warn => tracing::warn!(target: "script", "{}", entry.message),
            LogLevel::Error => tracing::error!(target: "script", "{}", entry.message),
        }

        self.logs.write().await.push(entry);
    }

    /// List directory contents (if permitted).
    ///
    /// `max_entries` makes this a bounded query: at most that many entries are
    /// collected and the walk stops immediately once the bound is reached.
    /// A bounded *flat* listing is the first `max_entries` names in name
    /// order (see `list_dir_sorted_prefix`), so the same tree lists the same
    /// way on every filesystem; a recursive walk stops in walk order.
    /// This is NOT silent truncation — the caller asked for a bound and is
    /// responsible for announcing it (the convention is to request `budget + 1`
    /// so an overflow-length return proves more entries exist). The bound
    /// exists because every returned entry is marshalled into the script
    /// engine one JS object at a time; unbounded listings of real workspaces
    /// (hundreds of thousands of entries under `node_modules/.git/target`) blow
    /// the 30s script deadline before the script runs a single line.
    ///
    /// # Errors
    ///
    /// Returns [`ScriptError::Permission`] if the resolved path is outside the
    /// tool's read scope. A flat listing returns [`ScriptError::Bridge`] if the
    /// directory or one of its entries cannot be read; a recursive walk skips
    /// entries it cannot read.
    pub async fn list_dir(
        &self,
        path: &str,
        recursive: bool,
        max_entries: Option<usize>,
    ) -> Result<Vec<DirEntry>> {
        const IGNORE_DIRS: &[&str] = &[
            "node_modules", "target", ".git", "__pycache__", ".venv",
            "venv", "dist", "build", ".next", ".nuxt", ".cache",
        ];

        let path = self.resolve_path(path);

        if !self.permissions.allows_read(&path) {
            return Err(ScriptError::Permission(format!(
                "Read access to '{}' not permitted",
                path.display()
            )));
        }

        let cap = max_entries.unwrap_or(usize::MAX);
        let mut entries = Vec::new();

        if recursive {
            for result in walkdir::WalkDir::new(&path)
                .max_depth(10)
                .into_iter()
                .filter_entry(|e| {
                    e.file_name()
                        .to_str()
                        .is_none_or(|name| !IGNORE_DIRS.contains(&name))
                })
            {
                if entries.len() >= cap {
                    break;
                }
                let Ok(entry) = result else {
                    continue;
                };
                // Skip the root directory itself
                if entry.depth() == 0 {
                    continue;
                }
                if let Some(de) = dir_entry_from_walkdir(&entry) {
                    entries.push(de);
                }
            }
        } else if let Some(cap) = max_entries {
            entries = list_dir_sorted_prefix(&path, cap).await?;
        } else {
            let mut read_dir = tokio::fs::read_dir(&path)
                .await
                .map_err(|e| ScriptError::Bridge(format!("Failed to read directory: {e}")))?;

            while let Some(entry) = read_dir
                .next_entry()
                .await
                .map_err(|e| ScriptError::Bridge(format!("Failed to read entry: {e}")))?
            {
                let name = entry.file_name().to_string_lossy().to_string();
                let metadata = entry.metadata().await.ok();
                entries.push(dir_entry_from_metadata(name, metadata.as_ref()));
            }
        }

        Ok(entries)
    }

    /// Get file/directory metadata (if permitted)
    ///
    /// # Errors
    ///
    /// Returns [`ScriptError::Permission`] if the resolved path is outside the
    /// tool's read scope, and [`ScriptError::Bridge`] if its metadata cannot be
    /// read (for example, it does not exist).
    pub async fn stat(&self, path: &str) -> Result<FileStat> {
        let path = self.resolve_path(path);

        if !self.permissions.allows_read(&path) {
            return Err(ScriptError::Permission(format!(
                "Read access to '{}' not permitted",
                path.display()
            )));
        }

        let metadata = tokio::fs::metadata(&path)
            .await
            .map_err(|e| ScriptError::Bridge(format!("Failed to stat '{}': {e}", path.display())))?;

        let modified = metadata
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs());

        Ok(FileStat {
            path,
            size: metadata.len(),
            is_file: metadata.is_file(),
            is_dir: metadata.is_dir(),
            modified,
        })
    }

    /// Get environment variable (if permitted)
    ///
    /// # Errors
    ///
    /// Returns [`ScriptError::Permission`] if the tool lacks the `env`
    /// permission. An unset (or non-UTF-8) variable is `Ok(None)`.
    pub fn get_env(&self, key: &str) -> Result<Option<String>> {
        if !self.permissions.env {
            return Err(ScriptError::Permission(
                "Environment variable access not permitted".to_string(),
            ));
        }
        Ok(std::env::var(key).ok())
    }

    /// Get all logs from this execution
    pub async fn get_logs(&self) -> Vec<LogEntry> {
        self.logs.read().await.clone()
    }

    /// Clear logs
    pub async fn clear_logs(&self) {
        self.logs.write().await.clear();
    }
}

/// Exec response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecResponse {
    pub success: bool,
    pub code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    /// The command overran its deadline and was killed. `stdout`/`stderr` then
    /// carry what it had already printed: it RAN, and whatever side effects it
    /// reached are on disk — disk is truth. Never true for a command that
    /// finished on its own, and never true for a command that failed to start
    /// (that is still an `Err`, because then genuinely nothing ran).
    pub timed_out: bool,
    /// Wall-clock milliseconds the command was alive. Measured on both exits,
    /// so a killed command can say how long it ran before the deadline.
    pub elapsed_ms: u64,
}

/// Fetch request options
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FetchOptions {
    pub method: Option<String>,
    pub headers: Option<std::collections::HashMap<String, String>>,
    pub body: Option<String>,
}

/// Fetch response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FetchResponse {
    pub status: u16,
    pub headers: std::collections::HashMap<String, String>,
    pub body: String,
}

impl FetchResponse {
    /// Parse body as JSON
    pub fn json<T: serde::de::DeserializeOwned>(&self) -> Result<T> {
        serde_json::from_str(&self.body).map_err(Into::into)
    }
}

/// Directory entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirEntry {
    pub name: String,
    pub entry_type: String,
    pub size: u64,
    pub modified: Option<u64>,
}

/// File stat result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileStat {
    /// The path as resolved — `~` expanded, a relative path joined onto the
    /// workspace — and permitted. A skill hands THIS to a service: services
    /// take a raw path, so a relative one resolved against the daemon's own
    /// working directory and skipped the skill's read scope entirely.
    pub path: std::path::PathBuf,
    pub size: u64,
    pub is_file: bool,
    pub is_dir: bool,
    pub modified: Option<u64>,
}

/// How many names a bounded flat listing reads before choosing its prefix.
///
/// Bound justification: names alone are cheap (no `stat`, ~30 B each, so 2^18
/// of them is ~8 MiB); it is marshalling *entries* into the script engine
/// that the caller's cap protects against. Up to this many names the chosen
/// prefix is exactly the first `cap` in name order on every filesystem; past
/// it, the prefix is chosen among the first 2^18 the filesystem returned.
const LIST_DIR_NAME_SCAN_MAX: usize = 1 << 18;

/// The first `cap` entries of `path` in name order (byte order of the lossy
/// UTF-8 name), each `lstat`ed. A capped listing used to keep the first `cap`
/// in *readdir* order — reverse creation on tmpfs, creation order on btrfs —
/// so which entries a budget kept depended on the host, not the tree.
async fn list_dir_sorted_prefix(path: &std::path::Path, cap: usize) -> Result<Vec<DirEntry>> {
    let mut read_dir = tokio::fs::read_dir(path)
        .await
        .map_err(|e| ScriptError::Bridge(format!("Failed to read directory: {e}")))?;
    let mut names = Vec::new();
    while names.len() < LIST_DIR_NAME_SCAN_MAX
        && let Some(entry) = read_dir
            .next_entry()
            .await
            .map_err(|e| ScriptError::Bridge(format!("Failed to read entry: {e}")))?
    {
        names.push(entry.file_name().to_string_lossy().to_string());
    }
    names.sort_unstable();
    names.truncate(cap);
    let mut entries = Vec::with_capacity(names.len());
    for name in names {
        let metadata = tokio::fs::symlink_metadata(path.join(&name)).await.ok();
        entries.push(dir_entry_from_metadata(name, metadata.as_ref()));
    }
    debug_assert!(entries.len() <= cap, "the cap is honoured");
    debug_assert!(entries.windows(2).all(|w| w[0].name <= w[1].name), "name order");
    Ok(entries)
}

/// A flat listing's entry from its (`lstat`) metadata; `unknown` when the
/// entry vanished or could not be read between the listing and the stat.
fn dir_entry_from_metadata(name: String, metadata: Option<&std::fs::Metadata>) -> DirEntry {
    let entry_type = metadata
        .map_or("unknown", |m| {
            if m.is_dir() {
                "dir"
            } else if m.is_symlink() {
                "link"
            } else {
                "file"
            }
        })
        .to_string();
    let size = metadata.map_or(0, std::fs::Metadata::len);
    let modified = metadata
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs());
    DirEntry {
        name,
        entry_type,
        size,
        modified,
    }
}

/// Convert a walkdir entry to our `DirEntry`
fn dir_entry_from_walkdir(entry: &walkdir::DirEntry) -> Option<DirEntry> {
    let name = entry.path().to_string_lossy().to_string();
    let metadata = entry.metadata().ok()?;
    let entry_type = if metadata.is_dir() {
        "dir"
    } else if metadata.is_symlink() {
        "link"
    } else {
        "file"
    }
    .to_string();
    let size = metadata.len();
    let modified = metadata
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs());

    Some(DirEntry {
        name,
        entry_type,
        size,
        modified,
    })
}

/// Log level
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    Debug,
    Info,
    Warn,
    Error,
}

/// Log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub level: LogLevel,
    pub message: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Build the Windows child for `command`: `python[3] -c` one-liners run
/// python directly (cmd/quote mangling), explicit PowerShell goes to
/// `powershell.exe`, and everything else to Git Bash, falling back to
/// `cmd.exe` only when Git Bash is not installed.
#[cfg(windows)]
fn windows_shell_command(command: &str) -> tokio::process::Command {
    let trimmed = command.trim_start();

    // Detect `python[3] -c "..."` one-liners (route directly to avoid quote
    // mangling). Compound commands (&&, ||) are handled by the shell below.
    let is_python_c = (trimmed.starts_with("python3 -c") || trimmed.starts_with("python -c"))
        && !trimmed.contains("&&")
        && !trimmed.contains("||");

    if is_python_c {
        // Parse: "python[3] -c 'code'" or 'python[3] -c "code"'
        let (exe, rest) = if trimmed.starts_with("python3") {
            (
                "python3",
                trimmed.strip_prefix("python3 -c").unwrap_or("").trim(),
            )
        } else {
            (
                "python",
                trimmed.strip_prefix("python -c").unwrap_or("").trim(),
            )
        };
        // Strip outer quotes if present
        let code = strip_outer_quotes(rest);
        let mut c = tokio::process::Command::new(exe);
        c.args(["-c", code]);
        c
    } else {
        match classify_windows_command(trimmed) {
            WinShell::PowerShell => {
                let mut c = tokio::process::Command::new("powershell.exe");
                c.args(["-NoProfile", "-NonInteractive", "-Command", command]);
                c
            }
            WinShell::Bash => {
                // The model writes POSIX/bash — run through Git Bash if present
                // (handles `[ -f ]`, cat, tail, &&, |, quoting, forward-slash
                // paths); fall back to cmd.exe only when Git Bash isn't installed.
                if let Some(bash) = git_bash_path() {
                    let mut c = tokio::process::Command::new(bash);
                    c.args(["-c", command]);
                    c
                } else {
                    let mut c = tokio::process::Command::new("cmd");
                    c.args(["/S", "/C", command]);
                    c
                }
            }
        }
    }
}

/// What `scrub_appimage_library_path` decided about the child's
/// `LD_LIBRARY_PATH`.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ScrubbedLibraryPath {
    /// Not running from an `AppImage`, or the variable carries none of the
    /// bundle's directories: the child inherits it as-is.
    Unchanged,
    /// Every entry was the bundle's: the child gets no `LD_LIBRARY_PATH` at all,
    /// which is the state the system's own binaries expect.
    Cleared,
    /// The bundle's entries removed; these are the ones that remain, in order.
    Narrowed(std::ffi::OsString),
}

/// Drop the `AppImage` bundle's own directories from an inherited
/// `LD_LIBRARY_PATH`. `appdir` is the mount `AppRun` exported (`$APPDIR`); an
/// entry is the bundle's when it lies under that mount. Pure, so the three
/// outcomes can be pinned without an `AppImage` on the machine.
#[cfg_attr(not(unix), allow(dead_code))]
fn scrub_appimage_library_path(
    appdir: Option<&std::ffi::OsStr>,
    ld_library_path: Option<&std::ffi::OsStr>,
) -> ScrubbedLibraryPath {
    let (Some(appdir), Some(inherited)) = (appdir, ld_library_path) else {
        return ScrubbedLibraryPath::Unchanged;
    };
    let appdir = std::path::Path::new(appdir);
    if appdir.as_os_str().is_empty() {
        return ScrubbedLibraryPath::Unchanged;
    }
    let mut kept: Vec<std::path::PathBuf> = Vec::new();
    let mut dropped = 0usize;
    for entry in std::env::split_paths(inherited) {
        // AppRun writes its entries with a trailing slash; `starts_with` is
        // component-wise, so `/mnt/x/usr/lib/` is under `/mnt/x` either way.
        if entry.starts_with(appdir) {
            dropped += 1;
        } else {
            kept.push(entry);
        }
    }
    if dropped == 0 {
        return ScrubbedLibraryPath::Unchanged;
    }
    if kept.is_empty() {
        return ScrubbedLibraryPath::Cleared;
    }
    // A kept entry that contains the separator itself cannot be re-encoded:
    // leave the variable alone rather than hand the child a mangled one.
    std::env::join_paths(&kept).map_or(
        ScrubbedLibraryPath::Unchanged,
        ScrubbedLibraryPath::Narrowed,
    )
}

#[cfg(test)]
mod appimage_library_path_tests {
    use super::{ScrubbedLibraryPath, scrub_appimage_library_path};
    use std::ffi::OsStr;

    const APPDIR: &str = "/tmp/.mount_Nanna_kMlAkL";

    #[test]
    fn outside_an_appimage_the_variable_is_inherited_untouched() {
        assert_eq!(
            scrub_appimage_library_path(None, Some(OsStr::new("/opt/cuda/lib64"))),
            ScrubbedLibraryPath::Unchanged
        );
        assert_eq!(
            scrub_appimage_library_path(Some(OsStr::new(APPDIR)), None),
            ScrubbedLibraryPath::Unchanged
        );
        assert_eq!(
            scrub_appimage_library_path(Some(OsStr::new("")), Some(OsStr::new(APPDIR))),
            ScrubbedLibraryPath::Unchanged,
            "an empty APPDIR names no mount and must not match everything"
        );
    }

    #[test]
    fn the_bundles_own_entries_are_dropped_and_nothing_else() {
        // Exactly what AppRun exported in the 2026-09-17 incident, plus one
        // entry the user had set themselves.
        let inherited = format!(
            "{APPDIR}/usr/lib/:{APPDIR}/usr/lib/x86_64-linux-gnu/:/opt/cuda/lib64:{APPDIR}/lib64/:"
        );
        assert_eq!(
            scrub_appimage_library_path(Some(OsStr::new(APPDIR)), Some(OsStr::new(&inherited))),
            ScrubbedLibraryPath::Narrowed("/opt/cuda/lib64:".into()),
            "the user's entry (and the trailing empty one) survive; the bundle's go"
        );
    }

    #[test]
    fn a_variable_that_is_entirely_the_bundles_is_removed_outright() {
        let inherited = format!("{APPDIR}/usr/lib/:{APPDIR}/lib/");
        assert_eq!(
            scrub_appimage_library_path(Some(OsStr::new(APPDIR)), Some(OsStr::new(&inherited))),
            ScrubbedLibraryPath::Cleared
        );
    }

    #[test]
    fn a_variable_with_no_bundle_entries_is_left_alone() {
        assert_eq!(
            scrub_appimage_library_path(
                Some(OsStr::new(APPDIR)),
                Some(OsStr::new("/opt/cuda/lib64:/usr/local/lib"))
            ),
            ScrubbedLibraryPath::Unchanged
        );
        // A sibling mount is not this bundle: prefix, not substring.
        assert_eq!(
            scrub_appimage_library_path(
                Some(OsStr::new(APPDIR)),
                Some(OsStr::new("/tmp/.mount_Nanna_kMlAkL2/usr/lib"))
            ),
            ScrubbedLibraryPath::Unchanged
        );
    }
}

/// Ceiling for one `Nanna.readFile`, in bytes.
///
/// Every tool that reads whole files already refuses far below this
/// (`read_file` at 10 MB, file history snapshots at 8 MiB), so no working call
/// meets it. It exists for the calls that bypass those checks — a tool without
/// one, a file that grew between `stat` and read, or a device that never ends
/// (`/dev/zero` is valid UTF-8 forever) — each of which used to read until the
/// daemon ran out of memory. The engine holds text as UTF-16, so a read costs
/// about twice this before any copy.
pub const READ_FILE_BYTES_MAX: u64 = 64 * 1024 * 1024;

/// A response body as text, read chunk by chunk and refused past `cap` bytes.
///
/// `Response::text` buffered the whole body first, so a fast link could feed a
/// skill gigabytes within the 30 s client timeout — `web_fetch`'s `max_chars`
/// only applied after all of it was in memory. The cap is `readFile`'s: a
/// fetch must not be a way to load what a file read refuses. Decoded as UTF-8,
/// lossily (an invalid byte becomes U+FFFD rather than failing the call).
async fn read_body_capped(mut response: reqwest::Response, cap: u64) -> Result<String> {
    let cap_bytes = usize::try_from(cap).unwrap_or(usize::MAX);
    // The body's declared charset, read before the body consumes the response:
    // a `charset=windows-1252` / `iso-8859-1` / `shift_jis` page decoded as
    // UTF-8 came back as replacement characters, silently.
    let encoding = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(charset_of)
        .and_then(|label| encoding_rs::Encoding::for_label(label.as_bytes()))
        .unwrap_or(encoding_rs::UTF_8);
    if response
        .content_length()
        .is_some_and(|declared| declared > cap)
    {
        return Err(ScriptError::Bridge(format!(
            "Response body is larger than {cap} bytes; not read"
        )));
    }
    let mut bytes: Vec<u8> = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|e| ScriptError::Bridge(format!("Failed to read body: {e}")))?
    {
        if bytes.len().saturating_add(chunk.len()) > cap_bytes {
            return Err(ScriptError::Bridge(format!(
                "Response body exceeded {cap} bytes; stopped reading"
            )));
        }
        bytes.extend_from_slice(&chunk);
    }
    debug_assert!(bytes.len() <= cap_bytes, "never more than the cap is held");
    let (text, _, _) = encoding.decode(&bytes);
    Ok(text.into_owned())
}

/// The `charset` parameter of a `Content-Type` value, if it names one.
fn charset_of(content_type: &str) -> Option<&str> {
    content_type.split(';').skip(1).find_map(|param| {
        let (name, value) = param.split_once('=')?;
        name.trim()
            .eq_ignore_ascii_case("charset")
            .then(|| value.trim().trim_matches('"'))
            .filter(|label| !label.is_empty())
    })
}

/// Read `path` as UTF-8, refusing once more than `max_bytes` have arrived.
///
/// Reads at most `max_bytes + 1` bytes whatever the file claims its size is, so
/// neither a growing file nor an endless device can outrun the bound.
async fn read_to_string_capped(path: &Path, max_bytes: u64) -> Result<String> {
    use tokio::io::AsyncReadExt as _;
    assert!(max_bytes > 0, "a zero ceiling reads nothing");
    let fail = |e: &dyn std::fmt::Display| {
        ScriptError::Bridge(format!("Failed to read '{}': {e}", path.display()))
    };
    let file = tokio::fs::File::open(path).await.map_err(|e| fail(&e))?;
    let mut bytes = Vec::new();
    file.take(max_bytes.saturating_add(1))
        .read_to_end(&mut bytes)
        .await
        .map_err(|e| fail(&e))?;
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > max_bytes {
        return Err(ScriptError::Bridge(format!(
            "Failed to read '{}': it is larger than the {} MiB a script may read at once. \
             Nothing was returned — read it in parts (exec with head/tail/sed) instead.",
            path.display(),
            max_bytes / (1024 * 1024)
        )));
    }
    debug_assert!(u64::try_from(bytes.len()).is_ok_and(|n| n <= max_bytes));
    String::from_utf8(bytes).map_err(|e| fail(&e))
}

/// The error for an exec whose working directory is not a directory.
///
/// `explicit` is whether the caller passed `workdir`; otherwise the directory
/// is the active workspace's, and the fix is different (the workspace moved or
/// was deleted, so passing a real `workdir` works where retrying cannot).
fn missing_workdir_message(dir: &Path, explicit: bool) -> String {
    let what = if dir.exists() { "is not a directory" } else { "does not exist" };
    if explicit {
        format!(
            "The working directory {} {what}. Nothing ran — pass an existing directory as workdir.",
            dir.display()
        )
    } else {
        format!(
            "The active workspace directory {} {what}. Nothing ran — pass an existing directory as workdir.",
            dir.display()
        )
    }
}

/// Strip ANSI escape sequences from text.
/// Tools like cargo, git, and npm emit color codes that break text matching
/// (e.g., `findstr "error:"` fails because the actual text is `\x1b[31merror\x1b[0m:`).
///
/// Covers the ECMA-48 shapes a terminal program actually emits, not only
/// colour: CSI (`ESC [` params, ended by a final byte in `@`..=`~`, so
/// `ESC[2~` and `ESC[?25h` go whole), the string commands OSC/DCS/SOS/PM/APC
/// (`ESC ]`, `ESC P`, `ESC X`, `ESC ^`, `ESC _`, ended by BEL or `ESC \`) —
/// cargo and `ls --hyperlink` wrap paths in OSC 8 hyperlinks, which used to
/// leave `8;;file:///…` glued into the text — and the nF/Fp/Fe/Fs escapes
/// (`ESC (B`, `ESC 7`, `ESC =`). An unterminated sequence at the end of the
/// text is dropped, never echoed half-parsed.
fn strip_ansi_escapes(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '\x1b' {
            result.push(c);
            continue;
        }
        match chars.next() {
            // CSI: parameter and intermediate bytes, then one final byte.
            Some('[') => {
                for seq_char in chars.by_ref() {
                    if ('@'..='~').contains(&seq_char) {
                        break;
                    }
                }
            }
            // String commands: everything up to BEL or the ST `ESC \`.
            Some(']' | 'P' | 'X' | '^' | '_') => {
                while let Some(seq_char) = chars.next() {
                    if seq_char == '\x07' {
                        break;
                    }
                    if seq_char == '\x1b' {
                        if chars.peek() == Some(&'\\') {
                            chars.next();
                        }
                        break;
                    }
                }
            }
            // nF: intermediate bytes (`(`, `)`, `#`, …) then one final byte.
            Some(' '..='/') => {
                for seq_char in chars.by_ref() {
                    if !(' '..='/').contains(&seq_char) {
                        break;
                    }
                }
            }
            // Fp/Fe/Fs two-byte escapes (`ESC 7`, `ESC =`, `ESC c`), or a
            // lone ESC at the end of the text.
            Some(_) | None => {}
        }
    }
    debug_assert!(result.len() <= s.len(), "stripping never adds bytes");
    debug_assert!(!result.contains('\x1b'), "no escape introducer survives");
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fetched body is decoded in the charset its `Content-Type` declares;
    /// read as UTF-8, a Latin-1 page's accents came back as U+FFFD.
    #[test]
    fn a_declared_charset_decodes_the_body() {
        assert_eq!(
            charset_of("text/html; charset=ISO-8859-1"),
            Some("ISO-8859-1")
        );
        assert_eq!(
            charset_of("text/html;charset=\"windows-1252\""),
            Some("windows-1252")
        );
        assert_eq!(charset_of("text/html"), None);
        assert_eq!(charset_of("text/html; charset="), None);
        let encoding = encoding_rs::Encoding::for_label(b"iso-8859-1").expect("known label");
        let (text, _, _) = encoding.decode(b"caf\xe9");
        assert_eq!(text, "caf\u{e9}");
    }

    /// The invariant: a path the SHELL prints must name the same file when it
    /// is handed back to a file tool. Git Bash prints `/d/...`, and that used
    /// to be treated as relative and joined onto the workspace, so a read of a
    /// real file answered "does not exist" while a write conjured a shadow tree
    /// under `<workdir>\d\...` and reported success.
    #[cfg(windows)]
    #[test]
    fn an_msys_drive_path_from_the_shell_resolves_to_the_real_file() {
        let dir = std::env::temp_dir().join("nanna-msys-path-test");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let file = dir.join("real.txt");
        std::fs::write(&file, b"x").expect("write probe file");

        // Exactly what `pwd`/`realpath` hand back for this file.
        let native = file.to_string_lossy().replace('\\', "/");
        let msys = format!("/{}{}", &native[..1].to_lowercase(), &native[2..]);

        let workdir = std::path::PathBuf::from("D:/Development/some-workspace");
        let resolved = NannaBridge::resolve_path_with_workdir(&msys, Some(&workdir));

        assert_eq!(
            resolved, file,
            "the shell-printed path {msys} must resolve back to the file it names,              not be joined onto the workspace"
        );

        std::fs::remove_file(&file).ok();
    }

    // ---------------------------------------------------------------------
    // Tree-kill: the timeout path must reap the shell AND its grandchildren.
    // The walk itself (`kill_process_tree`) and its unit tests live in
    // nanna-proc; what belongs here is the exec-level contract — including
    // the shape the walk alone cannot cover.
    // ---------------------------------------------------------------------

    /// Reap bound: taskkill / `TerminateJobObject` is near-instant; 5s is a
    /// generous ceiling for a loaded CI machine, polled at 50ms.
    #[cfg(windows)] // only the Windows reap test below uses it
    const REAP_DEADLINE: std::time::Duration = std::time::Duration::from_secs(5);

    /// Liveness via `tasklist` — fine for a test; the daemon's runtime checks
    /// use Win32 directly (see nanna-daemon health.rs for why).
    #[cfg(windows)]
    async fn windows_pid_alive(pid: u32) -> bool {
        tokio::process::Command::new("tasklist")
            .args(["/FI", &format!("PID eq {pid}"), "/NH"])
            .output()
            .await
            .is_ok_and(|out| String::from_utf8_lossy(&out.stdout).contains(&pid.to_string()))
    }

    /// REGRESSION (2026-08-08, ministral endurance teardown hang): `foo &` —
    /// the shell exits immediately, the backgrounded grandchild inherits our
    /// stdout/stderr write handles, the pipe drains never see EOF, and
    /// the timeout fires with the shell already DEAD. The parent-walk
    /// tree-kill (`taskkill /T` from the dead pid) finds nothing, so pre-fix
    /// the grandchild survived holding the pipe — and because tokio
    /// child-pipe reads on Windows are synchronous reads on the blocking
    /// pool, it pinned a blocking thread until process exit (25+ min hung
    /// teardown). The per-child Job Object must reap the grandchild anyway,
    /// EOFing the pipe so those reads complete and runtime shutdown is
    /// prompt.
    #[cfg(windows)]
    #[test]
    fn exec_timeout_reaps_detached_grandchild_and_frees_pipe_reads() {
        if git_bash_path().is_none() {
            eprintln!(
                "skipping exec_timeout_reaps_detached_grandchild_and_frees_pipe_reads: Git Bash not installed"
            );
            return;
        }
        // A dedicated runtime so the test can measure its teardown — the
        // production symptom was Runtime::drop joining the pinned read.
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let dir = tempfile::tempdir().expect("tempdir");
        let pid_file = dir.path().join("grandchild.pid");
        // Bash-native forward slashes: this path never goes through the
        // model-facing drive normalization (that rewrites `D:\x` shapes).
        let pid_path = pid_file.display().to_string().replace('\\', "/");
        // Git Bash `$!` is an MSYS pid; /proc/<msys-pid>/winpid maps it to
        // the real Windows pid so tasklist can probe it.
        let command =
            format!("ping -n 60 127.0.0.1 & cat /proc/$!/winpid > '{pid_path}'; exit 0");

        let mut perms = ToolPermissions::none();
        perms.run = true;
        let bridge = NannaBridge::new(perms);

        let res = rt.block_on(bridge.exec_with_timeout(&command, None, Some(2)));
        assert!(
            res.as_ref().is_ok_and(|r| r.timed_out),
            "held pipes must force the deadline kill, got {res:?}"
        );

        // The shell wrote the pid file before exiting, well inside the 2s
        // deadline — readable by the time the call returns.
        let text = std::fs::read_to_string(&pid_file).expect("grandchild pid file");
        let grandchild_pid: u32 = text.trim().parse().expect("windows pid");

        rt.block_on(async {
            let deadline = tokio::time::Instant::now() + REAP_DEADLINE;
            while windows_pid_alive(grandchild_pid).await {
                assert!(
                    tokio::time::Instant::now() < deadline,
                    "detached grandchild survived the exec timeout"
                );
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
        });

        // With the grandchild dead the inherited pipe EOF'd and the blocking
        // reads finished, so teardown is prompt. Pre-fix this join waited out
        // the sleeper's full 60s.
        let teardown = std::time::Instant::now();
        drop(rt);
        assert!(
            teardown.elapsed() < REAP_DEADLINE,
            "runtime teardown pinned by a leaked pipe read: {:?}",
            teardown.elapsed()
        );
    }

    // ---------------------------------------------------------------------
    // Path resolution: absolute and relative must BOTH work, and a relative
    // path that redundantly repeats the workspace tail must not silently
    // create a shadow tree (2026-07-26: an hour of edits landed in
    // <ws>/<ws-leaf>/minidb while the tests read <ws>/minidb; run scored 0).
    // ---------------------------------------------------------------------

    #[test]
    fn bare_relative_path_resolves_into_the_workspace() {
        let ws = tempfile::tempdir().expect("tempdir");
        let resolved =
            NannaBridge::resolve_path_with_workdir("minidb", Some(ws.path()));
        assert_eq!(resolved, ws.path().join("minidb"));
    }

    #[test]
    fn absolute_path_is_used_as_given() {
        let ws = tempfile::tempdir().expect("tempdir");
        let target = ws.path().join("sub").join("minidb");
        let resolved = NannaBridge::resolve_path_with_workdir(
            &target.to_string_lossy(),
            Some(ws.path()),
        );
        assert_eq!(resolved, target, "an absolute path must not be re-rooted");
    }

    #[test]
    fn redundant_workspace_prefix_finds_the_existing_file() {
        let ws = tempfile::tempdir().expect("tempdir");
        let root = ws.path().join("minidb-bench");
        std::fs::create_dir_all(&root).expect("mkdir");
        std::fs::write(root.join("minidb"), "#!/bin/sh\n").expect("write");

        // The model addresses the file as if it were one level up.
        let resolved =
            NannaBridge::resolve_path_with_workdir("minidb-bench/minidb", Some(&root));
        assert_eq!(
            resolved,
            root.join("minidb"),
            "a redundantly prefixed path must resolve to the real file, \
             not <ws>/minidb-bench/minidb"
        );
    }

    #[test]
    fn redundant_prefix_write_lands_beside_existing_files() {
        // Nothing exists at either candidate yet (a fresh write). The target
        // whose PARENT exists is the intended one; the literal join would
        // conjure a duplicate directory.
        let ws = tempfile::tempdir().expect("tempdir");
        let root = ws.path().join("bench-ws");
        std::fs::create_dir_all(&root).expect("mkdir");

        let resolved =
            NannaBridge::resolve_path_with_workdir("bench-ws/newfile.sh", Some(&root));
        assert_eq!(resolved, root.join("newfile.sh"));
    }

    #[test]
    fn a_real_nested_directory_of_the_same_name_still_wins() {
        // The repair must never shadow a genuine file: if the literal path
        // exists, it is what the caller meant.
        let ws = tempfile::tempdir().expect("tempdir");
        let root = ws.path().join("dup");
        let nested = root.join("dup");
        std::fs::create_dir_all(&nested).expect("mkdir");
        std::fs::write(nested.join("f.txt"), "nested").expect("write");
        std::fs::write(root.join("f.txt"), "outer").expect("write");

        let resolved = NannaBridge::resolve_path_with_workdir("dup/f.txt", Some(&root));
        assert_eq!(
            resolved,
            nested.join("f.txt"),
            "an existing literal target must take precedence over the repair"
        );
    }

    #[test]
    fn multi_segment_overlap_is_repaired() {
        let ws = tempfile::tempdir().expect("tempdir");
        let root = ws.path().join("a").join("b");
        std::fs::create_dir_all(&root).expect("mkdir");
        std::fs::write(root.join("f.txt"), "x").expect("write");

        let resolved = NannaBridge::resolve_path_with_workdir("a/b/f.txt", Some(&root));
        assert_eq!(resolved, root.join("f.txt"));
    }

    #[test]
    fn unrelated_relative_path_is_left_alone() {
        let ws = tempfile::tempdir().expect("tempdir");
        let root = ws.path().join("ws");
        std::fs::create_dir_all(&root).expect("mkdir");

        let resolved =
            NannaBridge::resolve_path_with_workdir("tests/test_01.sh", Some(&root));
        assert_eq!(
            resolved,
            root.join("tests").join("test_01.sh"),
            "a normal subdirectory path must be untouched"
        );
    }

    /// REGRESSION (2026-07-25): once models were told to run their own
    /// acceptance checks, `exec sh tests/test_NN.sh` killed the HARNESS —
    /// exit 0xffffffff, no panic — reproduced across two models, each dying
    /// seconds after its first self-check. The child shared our process
    /// group, so a script that signalled its group reached us.
    ///
    /// This drives the real `exec` path with a child that deliberately
    /// terminates its own process group. Surviving to read the result IS the
    /// assertion: without isolation this test process dies with the child.
    #[tokio::test]
    async fn a_child_that_kills_its_process_group_cannot_kill_us() {
        let bridge = NannaBridge::new(ToolPermissions {
            run: true,
            ..ToolPermissions::default()
        });

        // Kill the whole group the child belongs to. Isolated, that group is
        // the child's own and we are untouched; unisolated, it is ours too.
        #[cfg(windows)]
        let suicidal = "taskkill /T /F /PID $$ 2>/dev/null; kill -TERM 0 2>/dev/null; exit 7";
        #[cfg(unix)]
        let suicidal = "kill -TERM 0; exit 7";

        let _ = bridge.exec_with_timeout(suicidal, None, Some(20)).await;

        // Reaching here at all means we survived. Prove we are still healthy
        // by running another command through the same bridge.
        let after = bridge
            .exec_with_timeout("echo still_alive", None, Some(20))
            .await
            .expect("bridge still usable after a self-killing child");
        assert!(
            after.stdout.contains("still_alive"),
            "parent survived but the bridge is broken: {after:?}"
        );
    }

    #[cfg(windows)]
    #[test]
    fn normalize_strips_cd_slash_d_idiom() {
        for (input, want) in [
            ("cd /d D:/x", "cd D:/x"),
            ("cd /D D:/x", "cd D:/x"),
            (
                "cd /d D:/Development/nanna && cargo build",
                "cd D:/Development/nanna && cargo build",
            ),
            ("  cd /d D:/x", "  cd D:/x"),                 // leading ws preserved
            ("git fetch && cd /d D:/x", "git fetch && cd D:/x"),
            ("cd /d D:\\x", "cd D:\\x"),                   // backslash path still stripped
        ] {
            assert_eq!(normalize_cmdisms(input).as_ref(), want, "input: {input}");
        }
    }

    #[cfg(windows)]
    #[test]
    fn normalize_leaves_legit_commands_untouched() {
        for input in [
            "cd /d",                           // bare: POSIX cd into dir /d
            "cd /data",                        // path starting /d, not the /d token
            "cd /d/x",                         // dir /d/x
            "ls -la | head -20",
            "cargo build",
            "echo cd /d not-command-initial",  // cd not at a boundary
            "echo \"cd /d x\"",                // inside quotes
        ] {
            assert_eq!(normalize_cmdisms(input).as_ref(), input, "input: {input}");
            // Borrowed (no allocation) when unchanged.
            assert!(
                matches!(normalize_cmdisms(input), std::borrow::Cow::Borrowed(_)),
                "input: {input}"
            );
        }
    }

    #[cfg(windows)]
    #[test]
    fn windows_bash_style_commands_route_to_bash() {
        // The exact shapes the model wrote that used to fail in cmd.exe.
        for cmd in [
            "if [ -f \"D:/x/y.lock\" ]; then echo LOCK; cat \"D:/x/y.lock\"; else echo NO; fi",
            "cd D:/Development/nanna && git fetch origin 2>&1 | tail -5",
            "ls -la | head -20",
            "cat $file | grep foo", // used to be stolen by the `$ && |` heuristic
            "grep -r 'pattern' . && echo done",
        ] {
            assert_eq!(
                classify_windows_command(cmd.trim_start()),
                WinShell::Bash,
                "should route to bash: {cmd}"
            );
        }
    }

    #[cfg(windows)]
    #[test]
    fn windows_explicit_powershell_routes_to_powershell() {
        for cmd in [
            "Get-ChildItem -Path .",
            "powershell -Command \"echo hi\"",
            "pwsh -c ls",
            "Write-Output $env:PATH",
            "Get-Content foo.txt | Select-Object -First 5",
            "Remove-Item bar -Recurse",
        ] {
            assert_eq!(
                classify_windows_command(cmd.trim_start()),
                WinShell::PowerShell,
                "should route to powershell: {cmd}"
            );
        }
    }

    /// REGRESSION (`python -c "`): an unterminated one-liner leaves `rest` as a
    /// single quote character, which satisfies BOTH halves of the outer-quote
    /// test — so the strip evaluated `&rest[1..0]`, an inverted range, and the
    /// panic took the whole daemon down over one malformed command. A payload
    /// with no real pair to strip must pass through and let python report it.
    #[cfg(windows)]
    #[test]
    fn lone_quote_python_payload_survives_the_outer_quote_strip() {
        for (input, want) in [
            ("\"", "\""),                      // the crash shape: one char, both ends
            ("'", "'"),
            ("", ""),
            ("\"\"", ""),                      // an empty but genuine pair still unwraps
            ("\"print(1)\"", "print(1)"),
            ("'print(1)'", "print(1)"),
            ("print(1)", "print(1)"),          // unquoted payloads are untouched
            ("\"—\"", "—"),                    // em dash: 3 bytes, right against both bounds
            ("\"print('—')\"", "print('—')"),
            ("\"mismatched'", "\"mismatched'"), // no pair, so nothing is stripped
        ] {
            assert_eq!(strip_outer_quotes(input), want, "input: {input}");
        }
    }

    /// Serve one canned HTTP response on a loopback port; its URL.
    async fn serve_once(head: &'static str, body: Vec<u8>) -> String {
        use tokio::io::AsyncWriteExt;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("addr");
        tokio::spawn(async move {
            if let Ok((mut socket, _)) = listener.accept().await {
                let mut request = [0_u8; 1024];
                let _ = tokio::io::AsyncReadExt::read(&mut socket, &mut request).await;
                let _ = socket.write_all(head.as_bytes()).await;
                let _ = socket.write_all(&body).await;
                let _ = socket.shutdown().await;
            }
        });
        format!("http://{addr}/")
    }

    /// A body past the cap is refused — declared up front or only discovered
    /// while streaming — instead of being buffered whole.
    #[tokio::test]
    async fn a_fetched_body_is_read_only_up_to_the_cap() {
        let client = reqwest::Client::new();
        let declared = serve_once(
            "HTTP/1.1 200 OK\r\nContent-Length: 32\r\nConnection: close\r\n\r\n",
            vec![b'a'; 32],
        )
        .await;
        let response = client.get(&declared).send().await.expect("send");
        assert!(
            read_body_capped(response, 16).await.is_err(),
            "declared too large"
        );

        let undeclared = serve_once(
            "HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n",
            vec![b'b'; 32],
        )
        .await;
        let response = client.get(&undeclared).send().await.expect("send");
        assert!(
            read_body_capped(response, 16).await.is_err(),
            "streamed past the cap"
        );

        let small = serve_once(
            "HTTP/1.1 200 OK\r\nContent-Length: 5\r\nConnection: close\r\n\r\n",
            b"hello".to_vec(),
        )
        .await;
        let response = client.get(&small).send().await.expect("send");
        assert_eq!(read_body_capped(response, 16).await.expect("read"), "hello");
    }

    /// `stat` hands back the path the bridge resolved — the one a skill passes
    /// on to a service, which would otherwise resolve a relative path against
    /// the daemon's own working directory and skip the skill's read scope.
    #[tokio::test]
    async fn stat_returns_the_resolved_permitted_path() {
        let workspace = tempfile::tempdir().expect("workspace");
        std::fs::write(workspace.path().join("spec.pdf"), b"%PDF").expect("file");
        let root = workspace.path().to_string_lossy().to_string();
        let bridge = NannaBridge::new(ToolPermissions::none().with_read([root.as_str()]))
            .with_default_workdir(workspace.path());
        let stat = bridge.stat("spec.pdf").await.expect("stat");
        assert_eq!(stat.path, workspace.path().join("spec.pdf"));
        assert!(stat.is_file);
        // Outside the read scope, stat refuses — so a skill cannot reach a
        // service with that path either.
        let refused = bridge.stat("../outside.pdf").await;
        assert!(
            matches!(refused, Err(ScriptError::Permission(_))),
            "{refused:?}"
        );
    }

    #[test]
    fn test_permission_check() {
        let bridge = NannaBridge::new(
            ToolPermissions::none()
                .with_net(["api.example.com"])
                .with_read(["/tmp"]),
        );

        assert!(bridge.permissions.allows_net("api.example.com"));
        assert!(!bridge.permissions.allows_net("evil.com"));
    }

    #[test]
    fn relative_path_resolves_against_active_workspace() {
        // A bare relative path resolves against the active workspace directory
        // (the registry seeds `default_workdir` from it) — the behavior that keeps
        // tools operating in the workspace the user is in, not the home dir.
        let ws = std::path::Path::new(if cfg!(windows) { "D:\\ws" } else { "/ws" });
        let resolved = NannaBridge::resolve_path_with_workdir("a/b.txt", Some(ws));
        assert_eq!(resolved, ws.join("a/b.txt"));
    }

    #[test]
    fn tilde_path_still_expands_to_home() {
        // `~` remains explicitly home-relative regardless of the CWD fallback.
        let home = NannaBridge::home_dir().unwrap();
        let resolved = NannaBridge::resolve_path_with_workdir("~/notes.txt", None);
        assert_eq!(resolved, home.join("notes.txt"));
    }

    #[test]
    fn absolute_path_is_unchanged() {
        let abs = if cfg!(windows) { "C:\\tmp\\x.txt" } else { "/tmp/x.txt" };
        let resolved = NannaBridge::resolve_path_with_workdir(abs, None);
        assert_eq!(resolved, std::path::PathBuf::from(abs));
    }

    #[test]
    fn default_exec_timeout_classifies_build_and_vcs_tools() {
        // Build/VCS tools get the wide window; everything else the default.
        for slow in [
            "cargo test",
            "cargo build --release",
            "git status",
            "git clone https://x",
            "npm install",
            "pnpm build",
            "make -j8",
            "cd repo && cargo check",
        ] {
            assert_eq!(
                default_exec_timeout_secs(slow),
                EXEC_MAX_AUTODETECT_SECS,
                "expected wide window for {slow:?}"
            );
        }
        for fast in ["ls -la", "echo hi", "cat file.txt", "grep -r foo ."] {
            assert_eq!(
                default_exec_timeout_secs(fast),
                EXEC_DEFAULT_TIMEOUT_SECS,
                "expected default window for {fast:?}"
            );
        }
    }

    /// A command that overruns its (explicit) timeout must be killed promptly —
    /// not run to completion — and must say so in the RESULT, not by throwing.
    /// Regression guard for the exec timeout actually being honored and the
    /// select-based deadline firing.
    #[cfg(unix)]
    #[tokio::test]
    async fn exec_times_out_and_returns_promptly() {
        let mut perms = ToolPermissions::none();
        perms.run = true;
        let bridge = NannaBridge::new(perms);

        let start = std::time::Instant::now();
        let res = bridge.exec_with_timeout("sleep 10", None, Some(1)).await;
        let elapsed = start.elapsed();

        assert!(res.as_ref().is_ok_and(|r| r.timed_out), "got {res:?}");
        // Must fire near the 1s deadline, well before the 10s sleep would end.
        assert!(elapsed < std::time::Duration::from_secs(5), "took {elapsed:?}");
    }

    /// Same guard on Windows, routed through Git Bash (skipped if absent).
    #[cfg(windows)]
    #[tokio::test]
    async fn exec_times_out_and_returns_promptly() {
        if git_bash_path().is_none() {
            eprintln!("skipping exec_times_out_and_returns_promptly: Git Bash not installed");
            return;
        }
        let mut perms = ToolPermissions::none();
        perms.run = true;
        let bridge = NannaBridge::new(perms);

        let start = std::time::Instant::now();
        let res = bridge.exec_with_timeout("sleep 10", None, Some(1)).await;
        let elapsed = start.elapsed();

        assert!(res.as_ref().is_ok_and(|r| r.timed_out), "got {res:?}");
        assert!(elapsed < std::time::Duration::from_secs(5), "took {elapsed:?}");
    }

    /// REGRESSION (P23): an overrun used to return `Err(Timeout)`, which the
    /// exec tool rendered to the model as "Nothing ran" — so the model retried
    /// a command that HAD run, re-doing side effects already on disk. The kill
    /// is unchanged; the report must now carry the truth: it was killed, it
    /// lived this long, and this is what it printed before it died.
    #[cfg(unix)]
    #[tokio::test]
    async fn exec_timeout_carries_partial_output_and_elapsed() {
        let mut perms = ToolPermissions::none();
        perms.run = true;
        let bridge = NannaBridge::new(perms);

        let res = bridge
            .exec_with_timeout("echo partial_before_deadline; sleep 10", None, Some(1))
            .await
            .expect("a deadline kill must RETURN a result, never throw");

        assert!(res.timed_out, "the result must admit the kill: {res:?}");
        assert!(!res.success, "a killed command did not succeed: {res:?}");
        assert!(res.code.is_none(), "there is no exit status: {res:?}");
        assert!(
            res.stdout.contains("partial_before_deadline"),
            "the output it managed to print must survive: {res:?}"
        );
        assert!(res.elapsed_ms > 0, "elapsed must be real: {res:?}");

        // The completed path is untouched: no kill, no timed_out flag.
        let done = bridge
            .exec_with_timeout("echo finished_cleanly", None, Some(20))
            .await
            .expect("a completed command still returns Ok");
        assert!(!done.timed_out, "nothing killed this one: {done:?}");
        assert!(done.success, "{done:?}");
        assert!(done.stdout.contains("finished_cleanly"), "{done:?}");
    }

    /// Same contract on Windows, routed through Git Bash (skipped if absent).
    #[cfg(windows)]
    #[tokio::test]
    async fn exec_timeout_carries_partial_output_and_elapsed() {
        if git_bash_path().is_none() {
            eprintln!(
                "skipping exec_timeout_carries_partial_output_and_elapsed: Git Bash not installed"
            );
            return;
        }
        let mut perms = ToolPermissions::none();
        perms.run = true;
        let bridge = NannaBridge::new(perms);

        let res = bridge
            .exec_with_timeout("echo partial_before_deadline; sleep 10", None, Some(1))
            .await
            .expect("a deadline kill must RETURN a result, never throw");

        assert!(res.timed_out, "the result must admit the kill: {res:?}");
        assert!(!res.success, "a killed command did not succeed: {res:?}");
        assert!(res.code.is_none(), "there is no exit status: {res:?}");
        assert!(
            res.stdout.contains("partial_before_deadline"),
            "the output it managed to print must survive: {res:?}"
        );
        assert!(res.elapsed_ms > 0, "elapsed must be real: {res:?}");

        // The completed path is untouched: no kill, no timed_out flag.
        let done = bridge
            .exec_with_timeout("echo finished_cleanly", None, Some(20))
            .await
            .expect("a completed command still returns Ok");
        assert!(!done.timed_out, "nothing killed this one: {done:?}");
        assert!(done.success, "{done:?}");
        assert!(done.stdout.contains("finished_cleanly"), "{done:?}");
    }

    /// A write that fails must name the cause in a form a caller can MATCH on:
    /// the OS message is localized prose, the `io::ErrorKind` name is not. The
    /// tool layer keys its "this is not retryable" copy off that name.
    #[tokio::test]
    async fn write_failure_names_the_error_kind() {
        let dir = tempfile::tempdir().expect("tempdir");
        let bridge = NannaBridge::new(ToolPermissions::none().with_write([dir.path()]));

        // Writing a directory AS A FILE is the portable way to provoke a real
        // `io::Error` through the same path a permission failure takes (the
        // kind itself differs per platform, which is exactly why the assertion
        // is on the shape, not on one kind).
        let err = bridge
            .write_file(&dir.path().to_string_lossy(), "x")
            .await
            .expect_err("writing a directory as a file must fail");
        let msg = err.to_string();
        assert!(
            msg.contains("(kind="),
            "the failure must name the io::ErrorKind: {msg}"
        );
        assert!(
            msg.contains(&*dir.path().to_string_lossy()),
            "the failure must still name the path: {msg}"
        );
    }

    /// End-to-end: a POSIX/bash command with a pipe + `tail` (which used to fail in
    /// cmd.exe with an empty error) actually runs and succeeds via Git Bash.
    #[cfg(windows)]
    #[tokio::test]
    async fn windows_exec_runs_posix_pipe_command_via_git_bash() {
        if git_bash_path().is_none() {
            eprintln!("skipping windows_exec_runs_posix_pipe_command_via_git_bash: Git Bash not installed");
            return;
        }
        let mut perms = ToolPermissions::none();
        perms.run = true;
        let bridge = NannaBridge::new(perms);

        // pipe + `tail` + POSIX `printf` — none of which cmd.exe can run.
        let out = bridge
            .exec("printf 'a\\nb\\nc\\n' | tail -1", None)
            .await
            .expect("exec should not error");
        assert!(out.success, "command should succeed, got {out:?}");
        assert_eq!(out.stdout.trim(), "c", "stdout was {:?}", out.stdout);
    }

    /// Every escape shape a terminal program emits is removed whole, and the
    /// text between them survives byte for byte. OSC 8 is the one that bit:
    /// cargo wraps paths in hyperlinks, and the old CSI-only stripper left
    /// `8;;file:///…` glued to the path the model then tried to open.
    #[test]
    fn strip_ansi_escapes_removes_every_sequence_shape_whole() {
        let cases = [
            ("\x1b[31merror\x1b[0m: x", "error: x"),
            ("a\x1b[2~b", "ab"),
            ("\x1b[?25hshown", "shown"),
            ("see \x1b]8;;file:///src/lib.rs\x1b\\src/lib.rs\x1b]8;;\x1b\\ now", "see src/lib.rs now"),
            ("\x1b]0;window title\x07text", "text"),
            ("\x1b(Bplain", "plain"),
            ("\x1b7saved\x1b8", "saved"),
            ("\x1bPdcs payload\x1b\\after", "after"),
            ("tail \x1b[", "tail "),
            ("tail \x1b", "tail "),
            ("no escapes — ünïcode", "no escapes — ünïcode"),
        ];
        for (input, want) in cases {
            assert_eq!(strip_ansi_escapes(input), want, "input {input:?}");
        }
    }

    /// A read stops at its ceiling instead of running the daemon out of
    /// memory: an endless device is refused, not drained, and a file at the
    /// ceiling still reads whole.
    #[cfg(unix)]
    #[tokio::test]
    async fn read_file_refuses_past_its_ceiling() {
        let err = read_to_string_capped(Path::new("/dev/zero"), 4096)
            .await
            .expect_err("an endless device is refused");
        assert!(err.to_string().contains("larger than"), "{err}");

        let dir = tempfile::tempdir().expect("temp dir");
        let at = dir.path().join("at.txt");
        std::fs::write(&at, "a".repeat(4096)).expect("seed");
        let text = read_to_string_capped(&at, 4096).await.expect("a file at the ceiling reads");
        assert_eq!(text.len(), 4096);

        std::fs::write(&at, "a".repeat(4097)).expect("seed");
        assert!(read_to_string_capped(&at, 4096).await.is_err(), "one byte over is refused");
    }

    /// A missing working directory is reported as the directory, not as a
    /// failed command: the spawn's ENOENT used to read "Failed to execute
    /// command: No such file or directory", and the model went looking for a
    /// missing program.
    #[cfg(unix)]
    #[tokio::test]
    async fn exec_in_a_missing_workdir_names_the_directory() {
        let dir = tempfile::tempdir().expect("temp dir");
        let gone = dir.path().join("gone");
        let mut perms = ToolPermissions::none();
        perms.run = true;
        let bridge = NannaBridge::new(perms);

        let err = bridge
            .exec("true", Some(&gone.to_string_lossy()))
            .await
            .expect_err("a missing workdir must fail");
        let msg = err.to_string();
        assert!(msg.contains("working directory"), "names the cause: {msg}");
        assert!(msg.contains(&*gone.to_string_lossy()), "names the path: {msg}");
        assert!(msg.contains("does not exist"), "says what is wrong: {msg}");

        let file = dir.path().join("file.txt");
        std::fs::write(&file, "x").expect("seed file");
        let bridge = bridge.with_default_workdir(&file);
        let err = bridge
            .exec("true", None)
            .await
            .expect_err("a workspace dir that is a file must fail");
        let msg = err.to_string();
        assert!(msg.contains("active workspace directory"), "names the source: {msg}");
        assert!(msg.contains("is not a directory"), "says what is wrong: {msg}");

        let ok = NannaBridge::new({
            let mut p = ToolPermissions::none();
            p.run = true;
            p
        })
        .exec("pwd", Some(&dir.path().to_string_lossy()))
        .await
        .expect("an existing workdir runs");
        assert!(ok.success, "ran: {ok:?}");
    }

    // ---------------------------------------------------------------------
    // list_dir bounded queries: max_entries makes listings safe to marshal
    // into the script engine (unbounded workspace walks blew the 30s Boa
    // deadline — explore tool, observed live 2026-08-02).
    // ---------------------------------------------------------------------

    /// Seed `count` files into `dir`, plus a `sub/` directory holding one file.
    fn seed_tree(dir: &std::path::Path, count: usize) {
        for i in 0..count {
            std::fs::write(dir.join(format!("f{i}.txt")), "x").expect("seed file");
        }
        std::fs::create_dir(dir.join("sub")).expect("seed subdir");
        std::fs::write(dir.join("sub").join("inner.txt"), "y").expect("seed inner");
    }

    fn read_bridge(dir: &std::path::Path) -> NannaBridge {
        NannaBridge::new(ToolPermissions::none().with_read([dir]))
    }

    #[tokio::test]
    async fn list_dir_unbounded_returns_everything() {
        let dir = tempfile::tempdir().expect("tempdir");
        seed_tree(dir.path(), 5);
        let bridge = read_bridge(dir.path());

        let flat = bridge
            .list_dir(&dir.path().to_string_lossy(), false, None)
            .await
            .expect("flat listing");
        assert_eq!(flat.len(), 6, "5 files + sub/");

        let deep = bridge
            .list_dir(&dir.path().to_string_lossy(), true, None)
            .await
            .expect("recursive listing");
        assert_eq!(deep.len(), 7, "5 files + sub/ + sub/inner.txt");
    }

    #[tokio::test]
    async fn list_dir_max_entries_bounds_flat_listing() {
        let dir = tempfile::tempdir().expect("tempdir");
        seed_tree(dir.path(), 10);
        let bridge = read_bridge(dir.path());

        let bounded = bridge
            .list_dir(&dir.path().to_string_lossy(), false, Some(3))
            .await
            .expect("bounded flat listing");
        assert_eq!(bounded.len(), 3, "must stop at the bound");

        // budget + 1 convention: an overflow-length return proves more exist.
        let probe = bridge
            .list_dir(&dir.path().to_string_lossy(), false, Some(12))
            .await
            .expect("probe listing");
        assert_eq!(probe.len(), 11, "10 files + sub/ fit under a 12-entry bound");
    }

    /// The bounded prefix is the first names in name order, whatever order
    /// the filesystem hands them out in — tmpfs returns newest first, btrfs
    /// and ext4 oldest first, so creating the alphabetically-first entry
    /// LAST makes a readdir-order prefix miss it on one of the two.
    #[tokio::test]
    async fn a_bounded_flat_listing_keeps_the_first_names_on_every_filesystem() {
        let dir = tempfile::tempdir().expect("tempdir");
        for name in ["m.txt", "z.txt", "c.txt", "q.txt"] {
            std::fs::write(dir.path().join(name), "x").expect("seed");
        }
        std::fs::create_dir(dir.path().join("a_dir")).expect("seed dir");
        let bridge = read_bridge(dir.path());

        let bounded = bridge
            .list_dir(&dir.path().to_string_lossy(), false, Some(3))
            .await
            .expect("bounded flat listing");
        let names: Vec<&str> = bounded.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names, ["a_dir", "c.txt", "m.txt"]);
        assert_eq!(bounded[0].entry_type, "dir", "entries are still stat'ed");
        assert_eq!(bounded[1].size, 1);
    }

    #[tokio::test]
    async fn list_dir_max_entries_bounds_recursive_walk() {
        let dir = tempfile::tempdir().expect("tempdir");
        seed_tree(dir.path(), 10);
        let bridge = read_bridge(dir.path());

        let bounded = bridge
            .list_dir(&dir.path().to_string_lossy(), true, Some(4))
            .await
            .expect("bounded recursive listing");
        assert_eq!(bounded.len(), 4, "recursive walk must stop at the bound");
    }
}

#[cfg(all(test, windows))]
mod drive_path_tests {
    use super::normalize_drive_paths;

    /// REGRESSION (2026-07-27): `cat > D:\ws\minidb.sh << 'EOF'` wrote a file
    /// named `D:wsminidb.sh` — bash ate the backslashes as escapes, no error,
    /// and 3 KB of the run's newest work went into a garbage name while the
    /// real artifact sat frozen.
    #[test]
    fn windows_paths_become_posix_paths() {
        for (input, want) in [
            (
                r"cat > D:\Development\ws\minidb.sh",
                "cat > /d/Development/ws/minidb.sh",
            ),
            (r"cd D:\ws && ls", "cd /d/ws && ls"),
            (r"sh C:\tools\run.sh", "sh /c/tools/run.sh"),
            // Forward-slash drive paths are valid already but normalize the same.
            ("cat D:/ws/f.txt", "cat /d/ws/f.txt"),
        ] {
            assert_eq!(normalize_drive_paths(input).as_ref(), want, "input: {input}");
        }
    }

    #[test]
    fn ordinary_commands_are_untouched() {
        for input in [
            "ls -la tests/",
            "sh tests/test_01.sh && echo PASS",
            r"grep -E '\d+' file.txt",          // a regex escape, not a path
            r"echo 'D:\ws\literal'",          // single-quoted: the author meant it
            r"printf 'a\tb\n'",
            "curl http://localhost:11434/api/tags",
        ] {
            assert_eq!(
                normalize_drive_paths(input).as_ref(),
                input,
                "must not rewrite: {input}"
            );
            assert!(
                matches!(normalize_drive_paths(input), std::borrow::Cow::Borrowed(_)),
                "no allocation for untouched input: {input}"
            );
        }
    }

    #[test]
    fn redirections_and_pipes_survive() {
        assert_eq!(
            normalize_drive_paths(r"cat D:\ws\a.txt | grep x > D:\ws\b.txt").as_ref(),
            "cat /d/ws/a.txt | grep x > /d/ws/b.txt"
        );
    }
}

#[cfg(test)]
mod env_containment_tests {
    use super::{NannaBridge, ToolPermissions};

    /// **A known gap, pinned so it cannot be closed by accident.**
    ///
    /// `permissions.env` gates exactly one thing: `NannaBridge::get_env`. It
    /// does **not** scope the environment a spawned child inherits — nothing in
    /// this crate calls `Command::env_clear`, so a tool with `run: true` reads
    /// any variable the daemon holds, including provider API keys, whatever
    /// `env` says.
    ///
    /// That matters because it answers a roadmap question: narrowing the
    /// written-on-your-behalf default from `env: true` to `env: false` would
    /// break `Nanna.getEnv` for undeclared tools while leaving this path wide
    /// open — a compatibility cost for no containment, since the same default
    /// also grants `run: true`.
    ///
    /// So this test asserts today's behaviour rather than the behaviour we
    /// want. **If it starts failing, that is good news**: somebody scoped the
    /// child environment, and the `env` default can then be narrowed for real.
    /// Update it then; do not "fix" it by loosening the assertion.
    #[cfg(unix)]
    #[tokio::test]
    async fn env_false_does_not_stop_a_tool_reading_the_environment_through_exec() {
        // SAFETY: single-threaded test setup before any child is spawned.
        unsafe {
            std::env::set_var("NANNA_ENV_CONTAINMENT_PROBE", "leaked");
        }

        let denied_env = ToolPermissions {
            run: true,
            env: false,
            ..ToolPermissions::default()
        };
        let bridge = NannaBridge::new(denied_env);

        // The gate `env: false` does enforce.
        assert!(
            bridge.get_env("NANNA_ENV_CONTAINMENT_PROBE").is_err(),
            "`env: false` must deny the scripting bridge's own env accessor",
        );

        // The path it does not.
        let result = bridge
            .exec("printf '%s' \"$NANNA_ENV_CONTAINMENT_PROBE\"", None)
            .await
            .expect("exec is permitted by `run: true`");
        assert_eq!(
            result.stdout.trim(),
            "leaked",
            "this test pins a known gap: if the child no longer inherits the \
             variable, the exec environment has been scoped and the roadmap's \
             `env` item can move — update this test rather than relaxing it",
        );

        unsafe {
            std::env::remove_var("NANNA_ENV_CONTAINMENT_PROBE");
        }
    }
}
