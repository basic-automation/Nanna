//! The daemon's exit record: one JSON file in its data directory that says
//! whether the last daemon process is running or why it stopped.
//!
//! The daemon owns the file (`nanna_daemon::exit_reason` writes a `running`
//! marker at boot and a terminal reason on every exit path it can see). A
//! record still saying `running` after its process is gone is the
//! unclean-exit verdict.
//!
//! That verdict could not say how the process died, and the deaths it covers
//! are exactly the ones worth explaining: a `SIGBUS` when an `AppImage`
//! mount went away under the daemon, a `SIGKILL` from nobody known (both
//! seen on the operator's machine, 2026-09-28). The process that spawned the
//! daemon does see that: the GUI waits on its sidecar and gets the exit
//! status. So it adds what it saw to the record with
//! [`note_observed_exit`], and the next boot's warning names the signal.
//!
//! The type lives here, not in the daemon, because the GUI links this crate
//! and not the daemon's: one definition, so the two cannot drift.

use std::path::Path;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// File name under the daemon data directory.
pub const EXIT_REASON_FILE: &str = "nanna-daemon.exit.json";

/// The largest record [`note_observed_exit`] reads. A record is a few
/// hundred bytes; the biggest field is a panic message, which the daemon's
/// panic hook writes whole, and panic messages are a line or two. 64 KiB
/// is over a hundred times that: anything larger is not a record.
const RECORD_MAX_BYTES: u64 = 64 * 1024;

/// Whether the recorded process believed itself alive or terminated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExitState {
    /// Startup marker: the process was running when this was written.
    Running,
    /// A terminal reason was recorded on the way out.
    Exited,
}

/// One record, serialized as JSON. The whole file is a single record; every
/// write replaces it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExitReasonRecord {
    pub state: ExitState,
    /// PID of the process that wrote the record.
    pub pid: u32,
    /// Terminal reason (`clean_shutdown`, `panic`, `signal`, ...). None while running.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// Free-form detail: panic payload + location, signal name, error text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    /// RFC3339 timestamp of the write.
    pub at: String,
    /// How the process ended, as the process that spawned it saw it. Only
    /// ever added to a `running` record (see [`note_observed_exit`]), so the
    /// record still reads as unclean: the daemon did not get to say why.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observed_exit: Option<ObservedExit>,
}

/// An exit seen from outside the process.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservedExit {
    /// The exit code, when the process exited by itself.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<i32>,
    /// The signal that ended it (Unix).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signal: Option<i32>,
    /// Who saw it, for the log line (`the app`).
    pub observer: String,
    /// The observer sent the kill itself: a stop that the daemon did not
    /// finish in time. Not a crash.
    #[serde(default)]
    pub killed_by_observer: bool,
    /// RFC3339 timestamp of the observation.
    pub at: String,
}

impl ObservedExit {
    /// One clause for the log line: `the app saw it end by signal 7
    /// (SIGBUS) at …`.
    #[must_use]
    pub fn describe(&self) -> String {
        let how = match (self.signal, self.code) {
            (Some(signal), _) => signal_name(signal).map_or_else(
                || format!("by signal {signal}"),
                |name| format!("by signal {signal} ({name})"),
            ),
            (None, Some(code)) => format!("with exit code {code}"),
            (None, None) => "with no exit status".to_string(),
        };
        let killed = if self.killed_by_observer {
            ", a kill it sent itself when a stop was not finished in time"
        } else {
            ""
        };
        format!("{} saw it end {how} at {}{killed}", self.observer, self.at)
    }
}

/// What [`note_observed_exit`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoteOutcome {
    /// The record was the dead process's `running` marker; the exit is on it now.
    Noted,
    /// No record: the process died before it wrote one.
    NoRecord,
    /// The process recorded its own reason, which stands.
    AlreadyExited,
    /// The record belongs to another process: a daemon our sidecar deferred
    /// to, one started since, or an older one whose PID was reused.
    NotThisProcess,
    /// The file is not a record (torn, foreign, oversized). Left alone: the
    /// daemon already reads that as an unclean exit.
    Unreadable,
}

/// Add `observed` to the record in `data_dir`, if it is the `running` marker
/// of the process `pid` spawned at `spawned_at`.
///
/// Both identities are checked because a PID alone is reused: a daemon that
/// died uncleanly long ago leaves a `running` record with its PID, which a
/// later process may get. A record written before the spawn cannot be this
/// process's.
///
/// Call only once the process has been waited for, and before a next one is
/// spawned; the record then has no other writer. A daemon of another app
/// instance would be another PID.
///
/// # Errors
///
/// Reading or writing the file failed.
///
/// # Panics
///
/// When `pid` is 0, which no spawned process has.
pub fn note_observed_exit(
    data_dir: &Path,
    pid: u32,
    spawned_at: DateTime<Utc>,
    observed: ObservedExit,
) -> std::io::Result<NoteOutcome> {
    assert!(pid != 0, "a spawned process has a PID");
    let path = data_dir.join(EXIT_REASON_FILE);
    let size_bytes = match std::fs::metadata(&path) {
        Ok(metadata) => metadata.len(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(NoteOutcome::NoRecord),
        Err(e) => return Err(e),
    };
    if size_bytes > RECORD_MAX_BYTES {
        return Ok(NoteOutcome::Unreadable);
    }
    let text = std::fs::read_to_string(&path)?;
    let Ok(mut record) = serde_json::from_str::<ExitReasonRecord>(&text) else {
        return Ok(NoteOutcome::Unreadable);
    };
    if record.state == ExitState::Exited {
        return Ok(NoteOutcome::AlreadyExited);
    }
    let written_after_spawn = DateTime::parse_from_rfc3339(&record.at)
        .is_ok_and(|at| at.with_timezone(&Utc) >= spawned_at);
    if record.pid != pid || !written_after_spawn {
        return Ok(NoteOutcome::NotThisProcess);
    }
    debug_assert!(record.observed_exit.is_none(), "one process exits once");
    record.observed_exit = Some(observed);
    let json = serde_json::to_string_pretty(&record).map_err(std::io::Error::other)?;
    // The daemon's own writer does the same: a sibling file renamed over the
    // record, so a death here leaves the old record, never half of one.
    let tmp = path.with_extension("json.observed.tmp");
    std::fs::write(&tmp, json)?;
    std::fs::rename(&tmp, &path)?;
    Ok(NoteOutcome::Noted)
}

/// The name of a common fatal signal on this platform. The numbers differ
/// between systems (`SIGBUS` is 7 on Linux, 10 on macOS), and the observer
/// and the reader are the same machine.
#[must_use]
pub const fn signal_name(signal: i32) -> Option<&'static str> {
    if cfg!(target_os = "linux") {
        match signal {
            1 => Some("SIGHUP"),
            2 => Some("SIGINT"),
            3 => Some("SIGQUIT"),
            4 => Some("SIGILL"),
            6 => Some("SIGABRT"),
            7 => Some("SIGBUS"),
            8 => Some("SIGFPE"),
            9 => Some("SIGKILL"),
            11 => Some("SIGSEGV"),
            13 => Some("SIGPIPE"),
            15 => Some("SIGTERM"),
            _ => None,
        }
    } else if cfg!(target_os = "macos") {
        match signal {
            1 => Some("SIGHUP"),
            2 => Some("SIGINT"),
            3 => Some("SIGQUIT"),
            4 => Some("SIGILL"),
            6 => Some("SIGABRT"),
            8 => Some("SIGFPE"),
            9 => Some("SIGKILL"),
            10 => Some("SIGBUS"),
            11 => Some("SIGSEGV"),
            13 => Some("SIGPIPE"),
            15 => Some("SIGTERM"),
            _ => None,
        }
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    fn record(state: ExitState, pid: u32, at: DateTime<Utc>) -> ExitReasonRecord {
        ExitReasonRecord {
            state,
            pid,
            reason: (state == ExitState::Exited).then(|| "clean_shutdown".to_string()),
            detail: None,
            at: at.to_rfc3339(),
            observed_exit: None,
        }
    }

    fn write(dir: &Path, record: &ExitReasonRecord) {
        let json = serde_json::to_string_pretty(record).unwrap_or_default();
        assert!(std::fs::write(dir.join(EXIT_REASON_FILE), json).is_ok());
    }

    fn read(dir: &Path) -> ExitReasonRecord {
        let text = std::fs::read_to_string(dir.join(EXIT_REASON_FILE)).unwrap_or_default();
        match serde_json::from_str(&text) {
            Ok(record) => record,
            Err(e) => panic!("the record must stay a record: {e}"),
        }
    }

    fn killed(signal: i32) -> ObservedExit {
        ObservedExit {
            code: None,
            signal: Some(signal),
            observer: "the app".to_string(),
            killed_by_observer: false,
            at: "2026-10-05T12:00:00.000Z".to_string(),
        }
    }

    fn note(dir: &Path, pid: u32, spawned_at: DateTime<Utc>) -> NoteOutcome {
        match note_observed_exit(dir, pid, spawned_at, killed(9)) {
            Ok(outcome) => outcome,
            Err(e) => panic!("noting must not fail on a writable dir: {e}"),
        }
    }

    #[test]
    fn the_running_marker_of_the_dead_process_gets_its_exit() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let spawned_at = Utc::now();
        write(
            dir.path(),
            &record(
                ExitState::Running,
                4242,
                spawned_at + Duration::milliseconds(80),
            ),
        );

        assert_eq!(note(dir.path(), 4242, spawned_at), NoteOutcome::Noted);

        let noted = read(dir.path());
        assert_eq!(
            noted.state,
            ExitState::Running,
            "still unclean: the daemon never said why"
        );
        assert_eq!(noted.observed_exit, Some(killed(9)));
        assert!(
            !dir.path()
                .join("nanna-daemon.exit.json.observed.tmp")
                .exists()
        );
    }

    #[test]
    fn a_record_of_another_process_or_an_earlier_one_is_left_alone() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let spawned_at = Utc::now();

        // A daemon our sidecar deferred to, or one started since.
        write(
            dir.path(),
            &record(ExitState::Running, 7, spawned_at + Duration::seconds(1)),
        );
        assert_eq!(
            note(dir.path(), 4242, spawned_at),
            NoteOutcome::NotThisProcess
        );

        // Our PID, but written before we spawned: a long-dead daemon whose
        // PID was reused by our sidecar, which died before its own marker.
        write(
            dir.path(),
            &record(ExitState::Running, 4242, spawned_at - Duration::hours(30)),
        );
        assert_eq!(
            note(dir.path(), 4242, spawned_at),
            NoteOutcome::NotThisProcess
        );
        assert_eq!(read(dir.path()).observed_exit, None);
    }

    #[test]
    fn a_reason_the_daemon_recorded_itself_stands() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let spawned_at = Utc::now();
        write(
            dir.path(),
            &record(ExitState::Exited, 4242, spawned_at + Duration::seconds(5)),
        );

        assert_eq!(
            note(dir.path(), 4242, spawned_at),
            NoteOutcome::AlreadyExited
        );
        assert_eq!(read(dir.path()).reason.as_deref(), Some("clean_shutdown"));
        assert_eq!(read(dir.path()).observed_exit, None);
    }

    #[test]
    fn no_record_and_a_file_that_is_not_one_are_reported_and_untouched() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(note(dir.path(), 4242, Utc::now()), NoteOutcome::NoRecord);

        let path = dir.path().join(EXIT_REASON_FILE);
        assert!(std::fs::write(&path, "{\"state\":\"runn").is_ok());
        assert_eq!(note(dir.path(), 4242, Utc::now()), NoteOutcome::Unreadable);
        assert_eq!(
            std::fs::read_to_string(&path).unwrap_or_default(),
            "{\"state\":\"runn"
        );

        let oversized = vec![b' '; usize::try_from(RECORD_MAX_BYTES + 1).unwrap_or(usize::MAX)];
        assert!(std::fs::write(&path, oversized).is_ok());
        assert_eq!(note(dir.path(), 4242, Utc::now()), NoteOutcome::Unreadable);
    }

    #[test]
    fn a_record_without_the_new_field_still_reads() {
        let old = r#"{"state":"running","pid":12,"at":"2026-09-28T15:31:00.000Z"}"#;
        let parsed: Result<ExitReasonRecord, _> = serde_json::from_str(old);
        assert!(parsed.is_ok_and(|r| r.observed_exit.is_none() && r.pid == 12));
    }

    #[test]
    fn the_clause_names_the_signal_or_the_code() {
        let by_signal = killed(9).describe();
        assert!(
            by_signal.starts_with("the app saw it end by signal 9"),
            "{by_signal}"
        );
        if cfg!(target_os = "linux") {
            assert!(by_signal.contains("(SIGKILL)"), "{by_signal}");
            assert_eq!(signal_name(7), Some("SIGBUS"));
        }
        assert_eq!(signal_name(64), None);

        let stopped = ObservedExit {
            killed_by_observer: true,
            ..killed(9)
        };
        assert!(
            stopped
                .describe()
                .ends_with("when a stop was not finished in time")
        );

        let exited = ObservedExit {
            code: Some(3),
            signal: None,
            ..killed(9)
        };
        assert!(exited.describe().contains("with exit code 3"));
    }
}
