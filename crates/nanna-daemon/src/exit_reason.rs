//! Terminal reason file: the daemon's own durable record of WHY it stopped.
//!
//! Motivation (2026-08-10 ministral bench leg): the daemon hard-died mid-turn
//! at 14:15:47Z and the only evidence was absence — the log simply ends, no
//! shutdown marker, no panic line, nothing on disk that says the process is
//! gone rather than thinking. The bench driver polled the corpse 14 times and
//! published a 0/42 score for a daemon that was dead for 96% of the window.
//!
//! Mechanism (a classic dirty bit):
//! - On startup, AFTER the PID file and the IPC port are claimed (so a
//!   losing duplicate can never clobber the live daemon's record), the daemon
//!   reads whatever the previous process left, logs it, and overwrites the
//!   file with a `state: running` marker.
//! - Every deliberate exit path (clean shutdown drain, panic hook, signal /
//!   ctrl handler, IPC-server hard exit) overwrites the marker with
//!   `state: exited` plus a reason. Last writer wins.
//! - A file still saying `running` for a process that no longer exists IS the
//!   unclean-exit verdict: the process died through a path no hook could see
//!   (`taskkill /F`, OOM kill, power loss, abort without unwinding).
//!
//! The writer is disarmed until the startup marker lands: `record_exit` and
//! the panic hook are no-ops in a process that never owned the file, so a
//! second instance that fails the PID or port claim (or is Ctrl-C'd while
//! losing it) cannot overwrite the live daemon's record.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

// The record itself is shared with the GUI, which adds the exit status it
// saw when the daemon died without writing a reason (see `nanna_core::exit_record`).
pub use nanna_core::exit_record::{EXIT_REASON_FILE, ExitReasonRecord, ExitState, ObservedExit};

/// What the previous process left behind.
#[derive(Debug, Clone)]
pub enum PreviousExit {
    /// No file — first boot, or the file was deleted.
    Absent,
    /// The file exists but does not parse. A torn write during a hard death
    /// lands here, so this is treated as an unclean exit.
    Corrupt(String),
    /// A parseable record.
    Record(ExitReasonRecord),
}

impl PreviousExit {
    /// True when the previous process died without recording a terminal
    /// reason — the log-just-ends case this file exists to catch.
    #[must_use]
    pub fn is_unclean(&self) -> bool {
        match self {
            Self::Absent => false,
            Self::Corrupt(_) => true,
            Self::Record(r) => r.state == ExitState::Running,
        }
    }

    /// One line for the startup log describing the previous exit.
    #[must_use]
    pub fn describe(&self) -> String {
        match self {
            Self::Absent => "no previous exit record (first boot or record deleted)".to_string(),
            Self::Corrupt(err) => format!(
                "previous exit record is unreadable ({err}) — treating as an UNCLEAN exit \
                 (a torn write during a hard death lands here)"
            ),
            Self::Record(r) => match r.state {
                ExitState::Running => {
                    // The process that spawned it may have seen how it ended.
                    let how = r.observed_exit.as_ref().map_or_else(
                        || {
                            " — it died through a path no hook could see (hard kill, OOM, \
                             power loss)"
                                .to_string()
                        },
                        |observed| format!("; {}", observed.describe()),
                    );
                    format!(
                        "previous daemon (PID {}) exited UNCLEANLY: it marked itself running at \
                         {} and never recorded a terminal reason{how}",
                        r.pid, r.at
                    )
                }
                ExitState::Exited => format!(
                    "previous daemon (PID {}) exited at {}: {}{}",
                    r.pid,
                    r.at,
                    r.reason.as_deref().unwrap_or("<no reason recorded>"),
                    r.detail
                        .as_deref()
                        .map(|d| format!(" — {d}"))
                        .unwrap_or_default(),
                ),
            },
        }
    }
}

/// Handle to the reason file. Cheap to clone; clones share the armed flag, so
/// arming the handle held by `run()` also arms the clones captured by the
/// panic hook and the signal handlers.
#[derive(Clone)]
pub struct ExitReasonFile {
    path: PathBuf,
    armed: Arc<AtomicBool>,
}

impl ExitReasonFile {
    #[must_use]
    pub fn new(data_dir: &Path) -> Self {
        Self {
            path: data_dir.join(EXIT_REASON_FILE),
            armed: Arc::new(AtomicBool::new(false)),
        }
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Read whatever the previous process left. Never errors: absence and
    /// corruption are verdicts, not failures.
    #[must_use]
    pub fn read_previous(&self) -> PreviousExit {
        match std::fs::read_to_string(&self.path) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => PreviousExit::Absent,
            Err(e) => PreviousExit::Corrupt(e.to_string()),
            Ok(text) => match serde_json::from_str::<ExitReasonRecord>(&text) {
                Ok(record) => PreviousExit::Record(record),
                Err(e) => PreviousExit::Corrupt(e.to_string()),
            },
        }
    }

    /// Write the `running` startup marker and arm the terminal writers.
    /// Call exactly once, after the PID file and the IPC port are claimed.
    pub fn mark_running(&self) {
        self.write(&ExitReasonRecord {
            state: ExitState::Running,
            pid: std::process::id(),
            reason: None,
            detail: None,
            at: now_rfc3339(),
            observed_exit: None,
        });
        self.armed.store(true, Ordering::Release);
    }

    /// Record a terminal reason. No-op until `mark_running` has armed the
    /// handle (a process that never owned the file must not write it).
    /// Best-effort and panic-free: safe to call from a panic hook.
    pub fn record_exit(&self, reason: &str, detail: Option<&str>) {
        if !self.armed.load(Ordering::Acquire) {
            return;
        }
        self.write(&ExitReasonRecord {
            state: ExitState::Exited,
            pid: std::process::id(),
            reason: Some(reason.to_string()),
            detail: detail.map(str::to_string),
            at: now_rfc3339(),
            observed_exit: None,
        });
    }

    /// Atomic-ish replace: write a sibling temp file, then rename over the
    /// destination (Windows rename refuses to clobber, so remove first). A
    /// death inside this window leaves either the old record or a temp file
    /// beside an old record — never a half-written destination. If the temp
    /// path itself is unwritable, fall back to a direct write: a torn record
    /// reads as Corrupt, which the reader already treats as unclean.
    fn write(&self, record: &ExitReasonRecord) {
        let Ok(json) = serde_json::to_string_pretty(record) else {
            return;
        };
        let tmp = self.path.with_extension("json.tmp");
        if std::fs::write(&tmp, &json).is_ok() {
            let _ = std::fs::remove_file(&self.path);
            if std::fs::rename(&tmp, &self.path).is_ok() {
                return;
            }
        }
        let _ = std::fs::write(&self.path, &json);
    }
}

fn now_rfc3339() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file_in(dir: &tempfile::TempDir) -> ExitReasonFile {
        ExitReasonFile::new(dir.path())
    }

    #[test]
    fn absent_on_first_boot() {
        let dir = tempfile::tempdir().unwrap();
        let f = file_in(&dir);
        let prev = f.read_previous();
        assert!(matches!(prev, PreviousExit::Absent));
        assert!(!prev.is_unclean());
    }

    #[test]
    fn running_marker_round_trips_with_our_pid() {
        let dir = tempfile::tempdir().unwrap();
        let f = file_in(&dir);
        f.mark_running();
        match f.read_previous() {
            PreviousExit::Record(r) => {
                assert_eq!(r.state, ExitState::Running);
                assert_eq!(r.pid, std::process::id());
                assert!(r.reason.is_none());
            }
            other => panic!("expected record, got {other:?}"),
        }
    }

    #[test]
    fn clean_exit_supersedes_running_marker() {
        let dir = tempfile::tempdir().unwrap();
        let f = file_in(&dir);
        f.mark_running();
        f.record_exit("clean_shutdown", None);
        let prev = f.read_previous();
        assert!(!prev.is_unclean());
        match prev {
            PreviousExit::Record(r) => {
                assert_eq!(r.state, ExitState::Exited);
                assert_eq!(r.reason.as_deref(), Some("clean_shutdown"));
            }
            other => panic!("expected record, got {other:?}"),
        }
    }

    #[test]
    fn stale_running_record_is_the_unclean_verdict() {
        // Simulate the ministral death: a previous process marked itself
        // running and then vanished. The next boot must call it unclean.
        let dir = tempfile::tempdir().unwrap();
        file_in(&dir).mark_running();
        let next_boot = file_in(&dir);
        let prev = next_boot.read_previous();
        assert!(prev.is_unclean());
        assert!(prev.describe().contains("UNCLEAN"));
    }

    #[test]
    fn corrupt_file_reads_as_unclean() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(EXIT_REASON_FILE), "{ torn wri").unwrap();
        let prev = file_in(&dir).read_previous();
        assert!(matches!(prev, PreviousExit::Corrupt(_)));
        assert!(prev.is_unclean());
        assert!(prev.describe().contains("UNCLEAN"));
    }

    #[test]
    fn record_exit_is_a_noop_until_armed() {
        // A duplicate instance that loses the PID race (or is Ctrl-C'd while
        // losing it) must not clobber the live daemon's record.
        let dir = tempfile::tempdir().unwrap();
        let live = file_in(&dir);
        live.mark_running();
        let loser = file_in(&dir); // separate handle: never armed
        loser.record_exit("signal", Some("ctrl_c"));
        match live.read_previous() {
            PreviousExit::Record(r) => assert_eq!(r.state, ExitState::Running),
            other => panic!("expected the running marker to survive, got {other:?}"),
        }
    }

    #[test]
    fn clones_share_the_armed_flag() {
        let dir = tempfile::tempdir().unwrap();
        let f = file_in(&dir);
        let hook_clone = f.clone(); // captured by a panic hook before arming
        f.mark_running();
        hook_clone.record_exit("panic", Some("boom at src/x.rs:1:1"));
        match f.read_previous() {
            PreviousExit::Record(r) => {
                assert_eq!(r.reason.as_deref(), Some("panic"));
                assert_eq!(r.detail.as_deref(), Some("boom at src/x.rs:1:1"));
            }
            other => panic!("expected panic record, got {other:?}"),
        }
    }

    #[test]
    fn the_next_boot_names_the_exit_the_app_saw() {
        // The 2026-09-28 AppImage death: the daemon is killed by a signal no
        // hook sees, its GUI waits on it and notes the status, and the next
        // boot's warning says what happened instead of guessing.
        let dir = tempfile::tempdir().unwrap();
        let spawned_at = chrono::Utc::now() - chrono::Duration::seconds(1);
        file_in(&dir).mark_running();
        let observed = ObservedExit {
            code: None,
            signal: Some(7),
            observer: "the app".to_string(),
            killed_by_observer: false,
            at: now_rfc3339(),
        };
        let outcome = nanna_core::exit_record::note_observed_exit(
            dir.path(),
            std::process::id(),
            spawned_at,
            observed,
        );
        assert_eq!(
            outcome.unwrap(),
            nanna_core::exit_record::NoteOutcome::Noted
        );

        let prev = file_in(&dir).read_previous();
        assert!(prev.is_unclean(), "the daemon still never said why");
        let text = prev.describe();
        assert!(text.contains("UNCLEAN"), "got: {text}");
        assert!(
            text.contains("the app saw it end by signal 7"),
            "got: {text}"
        );
        assert!(!text.contains("no hook could see"), "got: {text}");
    }

    #[test]
    fn describe_names_the_reason_for_a_clean_exit() {
        let dir = tempfile::tempdir().unwrap();
        let f = file_in(&dir);
        f.mark_running();
        f.record_exit("signal", Some("SIGTERM"));
        let text = f.read_previous().describe();
        assert!(text.contains("signal"), "got: {text}");
        assert!(text.contains("SIGTERM"), "got: {text}");
    }
}
