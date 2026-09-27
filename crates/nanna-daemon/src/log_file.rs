//! Rotating on-disk daemon logs.
//!
//! The in-memory [`crate::log_buffer`] serves the GUI's recent-logs view; this
//! module adds a *persistent*, size-bounded file log so a long-running daemon
//! leaves a reviewable trail without growing without bound. Files roll daily and
//! old ones are pruned so at most [`LOG_FILES_MAX`] remain on disk.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use tracing_appender::rolling::{RollingFileAppender, Rotation};

/// Filename prefix for rotated daemon log files (`nanna-daemon.YYYY-MM-DD.log`).
pub const LOG_FILE_PREFIX: &str = "nanna-daemon";

/// Filename suffix (extension) for rotated daemon log files.
pub const LOG_FILE_SUFFIX: &str = "log";

/// Maximum number of rotated log files kept on disk; older files are pruned by
/// the appender on rotation. Bounds unbounded log accumulation (P6).
pub const LOG_FILES_MAX: usize = 7;

/// Resolve the directory rotated logs are written to.
///
/// Precedence: an explicit `cli_log_dir` wins, otherwise `{data_dir}/logs`. The
/// result is always a dedicated sub-directory (never the data root itself) so
/// callers never scatter log files among data files.
#[must_use]
pub fn resolve_log_dir(cli_log_dir: Option<&Path>, data_dir: &Path) -> PathBuf {
    // Precondition: a caller-provided data_dir is required so we never default
    // logs into the process cwd.
    debug_assert!(
        !data_dir.as_os_str().is_empty(),
        "data_dir must be non-empty"
    );

    let dir = match cli_log_dir {
        Some(explicit) if !explicit.as_os_str().is_empty() => explicit.to_path_buf(),
        _ => data_dir.join("logs"),
    };

    // Postcondition: the resolved directory is always non-empty.
    debug_assert!(
        !dir.as_os_str().is_empty(),
        "resolved log dir must be non-empty"
    );
    dir
}

/// Build a daily-rotating file appender for `log_dir`, bounded to
/// [`LOG_FILES_MAX`] files. Creates `log_dir` if it does not exist.
///
/// # Errors
/// Returns an error string if `log_dir` can't be created or the rolling
/// appender can't be initialized, so the caller can fall back to console-only
/// logging instead of aborting startup.
///
/// # Panics
/// Panics if `log_dir` is empty; callers pass a resolved, non-empty path from
/// [`resolve_log_dir`].
pub fn build_appender(log_dir: &Path) -> Result<RollingFileAppender, String> {
    assert!(!log_dir.as_os_str().is_empty(), "log_dir must be non-empty");

    // The appender creates missing parents, but create eagerly so a bad path
    // surfaces here (fallback to console) rather than on the first write.
    std::fs::create_dir_all(log_dir)
        .map_err(|e| format!("create log dir {}: {e}", log_dir.display()))?;

    let appender = RollingFileAppender::builder()
        .rotation(Rotation::DAILY)
        .filename_prefix(LOG_FILE_PREFIX)
        .filename_suffix(LOG_FILE_SUFFIX)
        .max_log_files(LOG_FILES_MAX)
        .build(log_dir)
        .map_err(|e| format!("build rolling appender: {e}"))?;

    Ok(appender)
}

/// The directory forensic dumps are written to: the resolved log directory,
/// recorded once at startup by [`set_forensics_dir`].
static FORENSICS_DIR: OnceLock<PathBuf> = OnceLock::new();

/// Ceiling for one forensic dump file, in bytes. On reaching it the file rolls
/// to `<name>.1` (replacing the previous roll), so a dump costs at most twice
/// this on disk however long the daemon runs.
pub const FORENSIC_LOG_BYTES_MAX: u64 = 8 * 1024 * 1024;

/// Record where forensic dumps go. The first call wins; later calls are
/// ignored, so a second logging setup cannot move files mid-run.
pub fn set_forensics_dir(dir: &Path) {
    debug_assert!(
        !dir.as_os_str().is_empty(),
        "forensics dir must be non-empty"
    );
    // Ignoring the error is the first-call-wins rule above.
    let _ = FORENSICS_DIR.set(dir.to_path_buf());
}

/// The directory recorded by [`set_forensics_dir`], if the daemon set one.
///
/// `None` means "do not dump": a process that never resolved a log directory
/// (a test, an embedding caller) writes nothing rather than guessing — which
/// is how prompts used to land in the shared, world-readable temp directory.
#[must_use]
pub fn forensics_dir() -> Option<&'static Path> {
    FORENSICS_DIR.get().map(PathBuf::as_path)
}

/// Append `entry` to the forensic file `path`, bounded to `max_bytes`.
///
/// One entry is capped at a quarter of the budget (cut on a char boundary,
/// marked), and when the append would take the file past `max_bytes` the file
/// is first rolled to `<path>.1`. The file is created owner-only on Unix: a
/// dump holds whole prompts, which are the user's conversation.
///
/// # Errors
///
/// Returns the underlying I/O error; callers treat dumps as best-effort.
///
/// # Panics
///
/// Panics if `max_bytes` is below 4, a budget too small to hold any entry.
pub fn append_bounded(path: &Path, entry: &str, max_bytes: u64) -> std::io::Result<()> {
    assert!(max_bytes >= 4, "a budget too small to hold any entry");
    let entry_bytes_max = usize::try_from(max_bytes / 4).unwrap_or(usize::MAX);
    let entry = if entry.len() > entry_bytes_max {
        let marker = "\n[… entry truncated by the forensic log bound]\n";
        let mut cut = entry_bytes_max.saturating_sub(marker.len());
        while !entry.is_char_boundary(cut) {
            cut -= 1;
        }
        format!("{}{marker}", &entry[..cut])
    } else {
        entry.to_string()
    };
    debug_assert!(
        entry.len() <= entry_bytes_max.max(64),
        "one entry within its share"
    );

    let current = std::fs::metadata(path).map_or(0, |m| m.len());
    let entry_len = u64::try_from(entry.len()).unwrap_or(u64::MAX);
    if current > 0 && current.saturating_add(entry_len) > max_bytes {
        let mut rolled = path.as_os_str().to_owned();
        rolled.push(".1");
        std::fs::rename(path, PathBuf::from(rolled))?;
    }

    let mut options = std::fs::OpenOptions::new();
    options.create(true).append(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(path)?.write_all(entry.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn resolve_prefers_explicit_dir() {
        let explicit = PathBuf::from("/var/log/nanna");
        let data = PathBuf::from("/data/nanna");
        assert_eq!(resolve_log_dir(Some(&explicit), &data), explicit);
    }

    #[test]
    fn resolve_falls_back_to_data_logs_subdir() {
        let data = PathBuf::from("/data/nanna");
        assert_eq!(resolve_log_dir(None, &data), data.join("logs"));
    }

    #[test]
    fn resolve_ignores_empty_explicit_dir() {
        let empty = PathBuf::new();
        let data = PathBuf::from("/data/nanna");
        // An empty override must not win — fall through to the data subdir.
        assert_eq!(resolve_log_dir(Some(&empty), &data), data.join("logs"));
    }

    #[test]
    fn appender_creates_dir_and_writes_a_prefixed_file() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let log_dir = tmp.path().join("logs");
        let mut appender = build_appender(&log_dir).expect("appender");
        writeln!(appender, "hello nanna").expect("write");
        appender.flush().expect("flush");

        assert!(log_dir.is_dir(), "log dir should be created");
        let mut names: Vec<String> = std::fs::read_dir(&log_dir)
            .expect("read_dir")
            .filter_map(std::result::Result::ok)
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        assert_eq!(names.len(), 1, "exactly one log file, got {names:?}");
        assert!(
            names[0].starts_with(LOG_FILE_PREFIX) && names[0].ends_with(LOG_FILE_SUFFIX),
            "log file name {} should carry the prefix+suffix",
            names[0]
        );
    }

    /// A forensic file never grows past its budget: it rolls to `.1`, an
    /// oversized entry is cut to its share, and the file is owner-only.
    #[test]
    fn a_forensic_dump_is_bounded_and_private() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("dump.log");
        let max = 1_000;

        for i in 0..10 {
            append_bounded(&path, &format!("{i}:{}\n", "x".repeat(150)), max).expect("append");
            let len = std::fs::metadata(&path).expect("meta").len();
            assert!(len <= max, "file stays within its bound: {len}");
        }
        let rolled = dir.path().join("dump.log.1");
        assert!(rolled.exists(), "the old file was rolled, not deleted");
        assert!(std::fs::metadata(&rolled).expect("meta").len() <= max);

        let huge = "é".repeat(2_000);
        append_bounded(&path, &huge, max).expect("append huge");
        let text = std::fs::read_to_string(&path).expect("read");
        assert!(text.contains("entry truncated"), "a cut entry says so");
        assert!(std::fs::metadata(&path).expect("meta").len() <= max);

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).expect("meta").permissions().mode();
            assert_eq!(mode & 0o077, 0, "no group/other access: {mode:o}");
        }
    }
}
