//! Pre-write file snapshots, so a script tool's overwrite can be undone.
//!
//! `write_file`/`edit_file` mutate the user's files with no backup of their
//! own; the `.__prev__`/`.__best__` parks `write_file` leaves beside a file are
//! one slot each, live inside the user's tree, and `edit_file` parks nothing.
//! Hours of unattended work have been lost to a single fault-storm overwrite.
//!
//! **One chokepoint.** Every script write goes through
//! [`crate::NannaBridge::write_file`], which asks this store to copy the file's
//! current bytes first. So `write_file`, `edit_file`, `file_buffer` and any
//! user-authored tool are covered by construction. Shell writes (`exec`) are
//! not — the same honest boundary Claude Code's checkpoints draw.
//!
//! **Outside the workspace, per session.** Snapshots live under the daemon's
//! data dir (`file-history/<session>/`), never beside the file, and a session
//! restores only what its own writes displaced.
//!
//! **Bounded three ways.** The 100 most recent checkpoints per session are
//! kept (Claude Code's number); a file's FIRST checkpoint in a session is kept
//! as its baseline past that, up to 400 baselines; and a session never holds
//! more than 256 MiB of snapshot bytes. A file over 8 MiB is not snapshotted at
//! all — the write still happens, and the refusal is logged.
//!
//! A snapshot failure never blocks the write it precedes: losing the undo is
//! better than losing the work.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Most recent (non-baseline) checkpoints kept per session.
pub const RECENT_CHECKPOINTS_MAX: usize = 100;

/// Most baseline checkpoints — a file's first snapshot in a session — kept.
///
/// Four times the recent window: a session that touches more distinct files
/// than that is a bulk generation, where the first version of each file is
/// rarely the one anybody wants back.
pub const BASELINE_CHECKPOINTS_MAX: usize = 400;

/// Largest file snapshotted, in bytes.
///
/// A model writes a file in tool calls of a few thousand tokens; `file_buffer`
/// assembles larger ones, but a source or notes file past 8 MiB is generated
/// data, and copying it on every write would dominate the write itself.
pub const SNAPSHOT_BYTES_MAX: u64 = 8 * 1024 * 1024;

/// Most snapshot bytes one session holds: 32 maximum-size snapshots, or the
/// whole 500-checkpoint window at ~512 KiB each.
pub const SESSION_BYTES_MAX: u64 = 256 * 1024 * 1024;

/// Most snapshot bytes kept across ALL sessions.
///
/// Per-session bounds alone grow with the session count. The whole store is
/// held to four full sessions' worth; the least recently written sessions are
/// dropped first, when a session directory is created.
pub const TOTAL_BYTES_MAX: u64 = 4 * SESSION_BYTES_MAX;

const INDEX_FILE: &str = "index.json";
const UNSCOPED_SESSION: &str = "unscoped";

/// Suffixes of the files the editing skills keep beside a real one: the
/// recovery copies `write_file` parks (a park is itself a backup), and the
/// working files — `file_buffer`'s draft and its cleared marker, the Python
/// syntax gate's two scratch files. Snapshotting them spent the window on
/// copies of copies: a file built from 100 `file_buffer` appends pushed every
/// real checkpoint out of the 100 kept, so `restore` answered "pruned".
const RECOVERY_PARK_SUFFIXES: [&str; 6] = [
    ".__prev__",
    ".__best__",
    ".__buffer__",
    ".__cleared__",
    ".__chk.py",
    ".__chk_new.py",
];

/// One file's state immediately before a write replaced it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Session-local, increasing. The handle a restore names.
    pub seq: u64,
    pub path: PathBuf,
    /// `false` when the write created the file; restoring removes it.
    pub existed: bool,
    pub bytes: u64,
    /// FNV-1a 64 of the content, to skip a snapshot identical to the last.
    pub content_hash: u64,
    pub taken_at: DateTime<Utc>,
    /// The file's first checkpoint in this session.
    pub baseline: bool,
}

/// What a restore did.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum Restored {
    /// The file's content was put back.
    Rewrote { path: PathBuf, bytes: u64 },
    /// The write had created the file, so restoring removed it.
    Removed { path: PathBuf },
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct Index {
    next_seq: u64,
    checkpoints: Vec<Checkpoint>,
}

/// The snapshot store for one data directory.
#[derive(Debug)]
pub struct FileHistory {
    root: PathBuf,
    /// One index update at a time. Writes are serial per file already (the
    /// bridge's advisory locks); this covers two files in one session.
    index_lock: tokio::sync::Mutex<()>,
}

static INSTALLED: OnceLock<FileHistory> = OnceLock::new();

/// Install the process-wide store the bridge snapshots into. Returns `false`
/// when one was already installed (the first root wins).
pub fn install(root: PathBuf) -> bool {
    INSTALLED.set(FileHistory::new(root)).is_ok()
}

/// The process-wide store, if the host installed one.
#[must_use]
pub fn installed() -> Option<&'static FileHistory> {
    INSTALLED.get()
}

/// The directory the bundled skills keep their own bookkeeping in
/// (`write_hiwater.json`, `read_marks.json`, …).
const SKILL_STATE_DIR: &str = ".nanna";

/// Whether a write to `path` is the tools' own housekeeping, not the user's work.
///
/// That is one of `write_file`'s recovery parks, or anything under a
/// `.nanna/` state directory. Snapshotting those would fill the window with
/// entries nobody restores — measured on the real daemon, one `write_file`
/// produced three checkpoints, two of them bookkeeping. Pure.
#[must_use]
pub fn is_tool_housekeeping(path: &Path) -> bool {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    let is_park = RECOVERY_PARK_SUFFIXES
        .iter()
        .any(|suffix| name.ends_with(suffix));
    is_park
        || path
            .components()
            .any(|component| component.as_os_str() == SKILL_STATE_DIR)
}

/// FNV-1a 64. Stable across processes, which the persisted index needs and
/// `std`'s `DefaultHasher` does not promise.
fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// A directory name for a session id: readable, filesystem-safe, and unique.
fn session_dir_name(session: Option<&str>) -> String {
    let session = session
        .filter(|s| !s.trim().is_empty())
        .unwrap_or(UNSCOPED_SESSION);
    let readable: String = session
        .chars()
        .take(48)
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let name = format!("{readable}-{:016x}", fnv1a64(session.as_bytes()));
    debug_assert!(name.len() <= 48 + 17);
    name
}

/// Checkpoints to drop so the index fits every bound, oldest first. Pure.
///
/// Recent checkpoints go first; a baseline is dropped only when the baselines
/// alone break a bound.
fn evictions(checkpoints: &[Checkpoint]) -> Vec<u64> {
    let mut recent: Vec<&Checkpoint> = checkpoints.iter().filter(|c| !c.baseline).collect();
    let mut baselines: Vec<&Checkpoint> = checkpoints.iter().filter(|c| c.baseline).collect();
    recent.sort_by_key(|c| c.seq);
    baselines.sort_by_key(|c| c.seq);
    let mut bytes: u64 = checkpoints.iter().map(|c| c.bytes).sum();
    let mut dropped = Vec::new();
    let mut drop_oldest = |pool: &mut Vec<&Checkpoint>, bytes: &mut u64| {
        let oldest = pool.remove(0);
        *bytes -= oldest.bytes;
        dropped.push(oldest.seq);
    };
    while recent.len() > RECENT_CHECKPOINTS_MAX {
        drop_oldest(&mut recent, &mut bytes);
    }
    while baselines.len() > BASELINE_CHECKPOINTS_MAX {
        drop_oldest(&mut baselines, &mut bytes);
    }
    while bytes > SESSION_BYTES_MAX && !(recent.is_empty() && baselines.is_empty()) {
        if recent.is_empty() {
            drop_oldest(&mut baselines, &mut bytes);
        } else {
            drop_oldest(&mut recent, &mut bytes);
        }
    }
    debug_assert!(recent.len() <= RECENT_CHECKPOINTS_MAX);
    debug_assert!(bytes <= SESSION_BYTES_MAX);
    dropped
}

impl FileHistory {
    #[must_use]
    pub fn new(root: PathBuf) -> Self {
        Self {
            root,
            index_lock: tokio::sync::Mutex::new(()),
        }
    }

    fn session_dir(&self, session: Option<&str>) -> PathBuf {
        self.root.join(session_dir_name(session))
    }

    async fn load_index(dir: &Path) -> io::Result<Index> {
        match tokio::fs::read(dir.join(INDEX_FILE)).await {
            Ok(raw) => serde_json::from_slice(&raw)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e)),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(Index::default()),
            Err(e) => Err(e),
        }
    }

    /// Write the index atomically: a crash mid-write leaves the old one.
    async fn save_index(dir: &Path, index: &Index) -> io::Result<()> {
        let raw = serde_json::to_vec(index).map_err(io::Error::other)?;
        let staged = dir.join(format!("{INDEX_FILE}.tmp"));
        tokio::fs::write(&staged, raw).await?;
        tokio::fs::rename(&staged, dir.join(INDEX_FILE)).await
    }

    /// Snapshot `path` as it is now, before a write replaces it.
    ///
    /// `Ok(None)` when nothing was recorded, deliberately: a directory, a file
    /// over [`SNAPSHOT_BYTES_MAX`], or content identical to this path's last
    /// checkpoint.
    ///
    /// # Errors
    ///
    /// The file could not be read or the store could not be written. Callers
    /// log it and write anyway.
    pub async fn record_before_write(
        &self,
        session: Option<&str>,
        path: &Path,
    ) -> io::Result<Option<Checkpoint>> {
        let content = match tokio::fs::metadata(path).await {
            Ok(meta) if meta.is_dir() => return Ok(None),
            Ok(meta) if meta.len() > SNAPSHOT_BYTES_MAX => {
                tracing::info!(
                    path = %path.display(),
                    bytes = meta.len(),
                    "file too large to snapshot before writing; the write proceeds without an undo"
                );
                return Ok(None);
            }
            Ok(_) => Some(tokio::fs::read(path).await?),
            Err(e) if e.kind() == io::ErrorKind::NotFound => None,
            Err(e) => return Err(e),
        };
        let existed = content.is_some();
        let bytes = content.as_deref().unwrap_or_default();
        let content_hash = fnv1a64(bytes);

        let _guard = self.index_lock.lock().await;
        let dir = self.session_dir(session);
        if !tokio::fs::try_exists(&dir).await.unwrap_or(false) {
            // A new session is the only moment the total can have grown past
            // what the last prune left; checked here, not on every write.
            self.prune_sessions(TOTAL_BYTES_MAX).await;
            tokio::fs::create_dir_all(&dir).await?;
        }
        let mut index = Self::load_index(&dir).await?;
        let previous = index.checkpoints.iter().rev().find(|c| c.path == path);
        if previous.is_some_and(|c| c.existed == existed && c.content_hash == content_hash) {
            return Ok(None);
        }
        let checkpoint = Checkpoint {
            seq: index.next_seq,
            path: path.to_path_buf(),
            existed,
            bytes: bytes.len() as u64,
            content_hash,
            taken_at: Utc::now(),
            baseline: previous.is_none(),
        };
        if existed {
            tokio::fs::write(dir.join(format!("{:08}.snap", checkpoint.seq)), bytes).await?;
        }
        index.next_seq += 1;
        index.checkpoints.push(checkpoint.clone());
        for seq in evictions(&index.checkpoints) {
            index.checkpoints.retain(|c| c.seq != seq);
            // Already gone is fine; the index is what makes it unreachable.
            let _ = tokio::fs::remove_file(dir.join(format!("{seq:08}.snap"))).await;
        }
        Self::save_index(&dir, &index).await?;
        debug_assert!(index.checkpoints.len() <= RECENT_CHECKPOINTS_MAX + BASELINE_CHECKPOINTS_MAX);
        Ok(Some(checkpoint))
    }

    /// Drop whole session stores, least recently written first, until the
    /// store holds at most `keep_bytes`. Best effort: an unreadable entry is
    /// skipped, and a failed removal only means the next prune tries again.
    async fn prune_sessions(&self, keep_bytes: u64) {
        let Ok(mut entries) = tokio::fs::read_dir(&self.root).await else {
            return;
        };
        let mut sessions: Vec<(std::time::SystemTime, u64, PathBuf)> = Vec::new();
        while let Ok(Some(entry)) = entries.next_entry().await {
            let dir = entry.path();
            let Ok(index_meta) = tokio::fs::metadata(dir.join(INDEX_FILE)).await else {
                continue;
            };
            let written = index_meta.modified().unwrap_or(std::time::UNIX_EPOCH);
            sessions.push((written, dir_bytes(&dir).await, dir));
        }
        sessions.sort_by_key(|session| std::cmp::Reverse(session.0));
        let mut kept: u64 = 0;
        for (_, bytes, dir) in sessions {
            kept = kept.saturating_add(bytes);
            if kept > keep_bytes {
                tracing::info!(dir = %dir.display(), bytes, "dropping an old session's file history to stay within budget");
                let _ = tokio::fs::remove_dir_all(&dir).await;
            }
        }
    }

    /// A session's checkpoints, newest first, optionally for one path.
    ///
    /// # Errors
    ///
    /// The index exists but could not be read.
    pub async fn list(
        &self,
        session: Option<&str>,
        path: Option<&Path>,
    ) -> io::Result<Vec<Checkpoint>> {
        let _guard = self.index_lock.lock().await;
        let index = Self::load_index(&self.session_dir(session)).await?;
        let mut checkpoints: Vec<Checkpoint> = index
            .checkpoints
            .into_iter()
            .filter(|c| path.is_none_or(|p| c.path == p))
            .collect();
        checkpoints.sort_by_key(|checkpoint| std::cmp::Reverse(checkpoint.seq));
        Ok(checkpoints)
    }

    /// Put `path` back the way checkpoint `seq` found it.
    ///
    /// The current state is snapshotted first, so a restore is itself undoable.
    ///
    /// # Errors
    ///
    /// `NotFound` when this session has no checkpoint `seq`; otherwise the IO
    /// error that stopped the restore.
    pub async fn restore(&self, session: Option<&str>, seq: u64) -> io::Result<Restored> {
        let dir = self.session_dir(session);
        let checkpoint = {
            let _guard = self.index_lock.lock().await;
            Self::load_index(&dir)
                .await?
                .checkpoints
                .into_iter()
                .find(|c| c.seq == seq)
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::NotFound,
                        format!("no checkpoint {seq} in this session"),
                    )
                })?
        };
        let snapshot = if checkpoint.existed {
            Some(tokio::fs::read(dir.join(format!("{seq:08}.snap"))).await?)
        } else {
            None
        };
        // What the restore replaces is saved first — unless it is too large
        // to keep, in which case `record_before_write` skips it and the
        // restore would destroy it unsaved while reporting it saved. Refuse.
        if let Ok(meta) = tokio::fs::metadata(&checkpoint.path).await
            && meta.is_file()
            && meta.len() > SNAPSHOT_BYTES_MAX
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "{} is {} bytes, larger than file history keeps ({SNAPSHOT_BYTES_MAX}); \
                     restoring would replace it with no copy kept — copy it aside first. \
                     Nothing was changed.",
                    checkpoint.path.display(),
                    meta.len()
                ),
            ));
        }
        self.record_before_write(session, &checkpoint.path).await?;
        let Some(content) = snapshot else {
            match tokio::fs::remove_file(&checkpoint.path).await {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                Err(e) => return Err(e),
            }
            return Ok(Restored::Removed {
                path: checkpoint.path,
            });
        };
        debug_assert_eq!(content.len() as u64, checkpoint.bytes);
        if let Some(parent) = checkpoint.path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::write(&checkpoint.path, &content).await?;
        Ok(Restored::Rewrote {
            path: checkpoint.path,
            bytes: content.len() as u64,
        })
    }
}

/// Total size of the files directly inside `dir` (session stores are flat).
async fn dir_bytes(dir: &Path) -> u64 {
    let Ok(mut entries) = tokio::fs::read_dir(dir).await else {
        return 0;
    };
    let mut total: u64 = 0;
    while let Ok(Some(entry)) = entries.next_entry().await {
        if let Ok(meta) = entry.metadata().await {
            total = total.saturating_add(meta.len());
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> (tempfile::TempDir, FileHistory) {
        let dir = tempfile::tempdir().expect("tempdir");
        let history = FileHistory::new(dir.path().join("file-history"));
        (dir, history)
    }

    #[tokio::test]
    async fn an_overwrite_and_a_creation_can_both_be_undone() {
        let (dir, history) = store();
        let file = dir.path().join("notes.md");
        std::fs::write(&file, "the original").expect("seed");

        let first = history
            .record_before_write(Some("s"), &file)
            .await
            .expect("snap")
            .expect("recorded");
        assert!(first.existed && first.baseline);
        std::fs::write(&file, "a destructive rewrite").expect("overwrite");

        let restored = history
            .restore(Some("s"), first.seq)
            .await
            .expect("restore");
        assert_eq!(
            restored,
            Restored::Rewrote {
                path: file.clone(),
                bytes: 12
            }
        );
        assert_eq!(
            std::fs::read_to_string(&file).expect("read"),
            "the original"
        );

        // The restore snapshotted the rewrite first, so it is undoable too.
        let undo = history.list(Some("s"), Some(&file)).await.expect("list");
        assert_eq!(undo.len(), 2, "{undo:?}");
        assert_eq!(undo[0].bytes, "a destructive rewrite".len() as u64);

        let created = dir.path().join("new.txt");
        let before_create = history
            .record_before_write(Some("s"), &created)
            .await
            .expect("snap")
            .expect("recorded");
        assert!(!before_create.existed);
        std::fs::write(&created, "made by a tool").expect("create");
        let removed = history
            .restore(Some("s"), before_create.seq)
            .await
            .expect("restore");
        assert_eq!(
            removed,
            Restored::Removed {
                path: created.clone()
            }
        );
        assert!(!created.exists());
    }

    #[tokio::test]
    async fn identical_content_is_not_snapshotted_twice_and_sessions_are_separate() {
        let (dir, history) = store();
        let file = dir.path().join("a.txt");
        std::fs::write(&file, "same").expect("seed");
        assert!(
            history
                .record_before_write(Some("s"), &file)
                .await
                .expect("snap")
                .is_some()
        );
        assert!(
            history
                .record_before_write(Some("s"), &file)
                .await
                .expect("snap")
                .is_none()
        );
        assert!(
            history
                .record_before_write(Some("other"), &file)
                .await
                .expect("snap")
                .is_some()
        );
        assert_eq!(history.list(Some("s"), None).await.expect("list").len(), 1);
        let error = history
            .restore(Some("third"), 0)
            .await
            .expect_err("not this session's");
        assert_eq!(error.kind(), io::ErrorKind::NotFound);
    }

    #[tokio::test]
    async fn the_least_recently_written_sessions_go_first_when_over_budget() {
        let (dir, history) = store();
        let file = dir.path().join("a.txt");
        for (session, content) in [
            ("old", "x".repeat(600)),
            ("mid", "y".repeat(600)),
            ("new", "z".repeat(600)),
        ] {
            std::fs::write(&file, &content).expect("seed");
            history
                .record_before_write(Some(session), &file)
                .await
                .expect("snap");
            // mtime resolution: make the write order observable.
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        // Each store is 600 bytes of snapshot plus a few hundred of index: two
        // fit in 2000 bytes, three do not.
        history.prune_sessions(2000).await;
        assert_eq!(
            history.list(Some("new"), None).await.expect("list").len(),
            1
        );
        assert_eq!(
            history.list(Some("mid"), None).await.expect("list").len(),
            1
        );
        assert!(
            history
                .list(Some("old"), None)
                .await
                .expect("list")
                .is_empty(),
            "oldest dropped"
        );
    }

    #[tokio::test]
    async fn directories_and_oversized_files_are_not_snapshotted() {
        let (dir, history) = store();
        assert!(
            history
                .record_before_write(None, dir.path())
                .await
                .expect("dir")
                .is_none()
        );
        let big = dir.path().join("big.bin");
        let file = std::fs::File::create(&big).expect("create");
        file.set_len(SNAPSHOT_BYTES_MAX + 1).expect("sparse");
        assert!(
            history
                .record_before_write(None, &big)
                .await
                .expect("big")
                .is_none()
        );
    }

    /// A restore over a file too large to keep is refused: the replaced
    /// content could not be saved, and the restore used to destroy it while
    /// saying it was saved.
    #[tokio::test]
    async fn a_restore_over_an_unkeepable_file_is_refused() {
        let (dir, history) = store();
        let path = dir.path().join("data.bin");
        std::fs::write(&path, b"small, kept").expect("write");
        let saved = history
            .record_before_write(None, &path)
            .await
            .expect("record")
            .expect("a checkpoint");
        let file = std::fs::File::create(&path).expect("create");
        file.set_len(SNAPSHOT_BYTES_MAX + 1).expect("sparse");

        let err = history.restore(None, saved.seq).await.expect_err("refused");
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput, "{err}");
        assert_eq!(
            std::fs::metadata(&path).expect("stat").len(),
            SNAPSHOT_BYTES_MAX + 1,
            "the large file is untouched"
        );
    }

    fn checkpoint(seq: u64, baseline: bool, bytes: u64) -> Checkpoint {
        Checkpoint {
            seq,
            path: PathBuf::from(format!("/f{seq}")),
            existed: true,
            bytes,
            content_hash: seq,
            taken_at: Utc::now(),
            baseline,
        }
    }

    #[test]
    fn eviction_drops_the_oldest_recent_before_any_baseline() {
        let mut all: Vec<Checkpoint> = vec![checkpoint(0, true, 1)];
        all.extend((1..=RECENT_CHECKPOINTS_MAX as u64 + 2).map(|seq| checkpoint(seq, false, 1)));
        assert_eq!(
            evictions(&all),
            vec![1, 2],
            "the two oldest recent, never the baseline"
        );

        let at_limit = vec![
            checkpoint(0, true, SESSION_BYTES_MAX),
            checkpoint(1, false, 1),
            checkpoint(2, false, 1),
        ];
        assert_eq!(
            evictions(&at_limit),
            vec![1, 2],
            "bytes: recent go first; exactly the limit fits"
        );
        let over = vec![
            checkpoint(0, true, SESSION_BYTES_MAX + 1),
            checkpoint(1, false, 1),
        ];
        assert_eq!(
            evictions(&over),
            vec![1, 0],
            "then the baseline, when it alone is over"
        );

        let baselines: Vec<Checkpoint> = (0..=BASELINE_CHECKPOINTS_MAX as u64)
            .map(|seq| checkpoint(seq, true, 1))
            .collect();
        assert_eq!(evictions(&baselines), vec![0]);
    }

    #[test]
    fn recovery_parks_and_session_names_are_handled() {
        assert!(is_tool_housekeeping(Path::new("/w/main.rs.__prev__")));
        assert!(is_tool_housekeeping(Path::new("/w/main.rs.__best__")));
        for scratch in [
            "/w/app.py.__buffer__",
            "/w/app.py.__cleared__",
            "/w/app.py.__buffer__.__cleared__",
            "/w/app.py.__chk.py",
            "/w/app.py.__chk_new.py",
        ] {
            assert!(is_tool_housekeeping(Path::new(scratch)), "{scratch}");
        }
        assert!(
            !is_tool_housekeeping(Path::new("/w/buffer.py")),
            "a real file named like one"
        );
        assert!(is_tool_housekeeping(Path::new(
            "/home/u/.nanna/write_hiwater.json"
        )));
        assert!(!is_tool_housekeeping(Path::new("/w/main.rs")));
        assert!(
            !is_tool_housekeeping(Path::new("/w/nanna/notes.nanna.md")),
            "only a `.nanna` directory"
        );
        assert_ne!(
            session_dir_name(Some("a/b")),
            session_dir_name(Some("a_b")),
            "sanitizing must not merge sessions"
        );
        assert_eq!(session_dir_name(None), session_dir_name(Some("  ")));
        assert!(session_dir_name(Some("telegram:42:7")).starts_with("telegram_42_7-"));
    }
}
