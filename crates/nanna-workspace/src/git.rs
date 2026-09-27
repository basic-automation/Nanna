//! Git working-tree state for the workspace injection.
//!
//! The workspace slice already tells the model *what the project is* (README,
//! AGENTS, CONTRIBUTING, ROADMAP). It says nothing about *what state the tree is
//! in right now*, and that is the half that prevents damage: an agent that
//! cannot see uncommitted work has no way to know that the file it is about to
//! rewrite is the only copy of an hour of someone else's edits. It also spends
//! its first turns re-discovering, with tool calls, what one `git status` would
//! have told it for free.
//!
//! So: whenever the workspace context is assembled and the root is a
//! repository, inject a bounded snapshot of the branch, the dirty paths, and the
//! recent commits. The daemon reassembles that context on every chat turn, so
//! the snapshot is re-read per turn rather than frozen at boot.
//!
//! # It is a snapshot, and it says so
//!
//! The injected text states that it is not live and that the model must re-run
//! `git` itself for anything current. That matches how the rest of this codebase
//! treats injected state: the disk is the truth, and a summary of it must
//! announce what it is (see the workspace context header, and the registry's own
//! truncation notices).
//!
//! # Bounds
//!
//! Everything here reads a repository of unknown size, so every dimension is
//! capped — path count, commit count, per-line bytes, total bytes read, and
//! wall-clock — and any elision is *reported* rather than silent, because a
//! truncated file list that looks complete is worse than none at all.

use std::fmt::Write as _;
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::process::Command;
use tracing::debug;

/// Maximum changed paths listed.
///
/// A working tree dirtier than this is mid-refactor, and what the model needs
/// then is the *shape* of the change plus an honest count, not forty more lines
/// it will not read. At roughly 60 bytes a line this section stays near 2.5 KB —
/// well under a percent of even a 32k-token window.
pub const GIT_CHANGED_PATHS_MAX: usize = 40;

/// Maximum recent commits listed.
///
/// Enough to see what the branch has been doing and in what order; a deeper
/// history is what `git log` is for, and the model can run it.
pub const GIT_RECENT_COMMITS_MAX: usize = 10;

/// Maximum bytes of any single line kept.
///
/// A status line is a two-character code plus a path; a log line is a short hash
/// plus a subject. Both fit comfortably — anything longer is a pathological path
/// or a commit subject written as an essay, and neither should set the size of
/// this section.
pub const GIT_LINE_BYTES_MAX: usize = 200;

/// Maximum bytes read from one `git` invocation.
///
/// `git status --short` on a repository with a large untracked tree (a
/// `node_modules` nobody ignored, a build directory) emits megabytes. We only
/// ever keep the first [`GIT_CHANGED_PATHS_MAX`] lines, so reading past this is
/// pure waste — and unbounded waste, on the run's critical path.
pub const GIT_OUTPUT_BYTES_MAX: usize = 64 * 1024;

/// Wall-clock ceiling for one `git` invocation.
///
/// `git status` on a cold, large repository can take seconds. Beyond this the
/// snapshot is not worth stalling the user's turn for, so it is dropped and the
/// run proceeds without it — this is context, never a precondition.
pub const GIT_TIMEOUT: Duration = Duration::from_secs(5);

/// A bounded snapshot of a repository's working tree.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GitContext {
    /// The `## branch...upstream [ahead N, behind M]` line, without the `## `.
    /// `None` when git reported no branch line (a fresh repo with no commits).
    pub branch: Option<String>,
    /// `git status --short` lines, capped at [`GIT_CHANGED_PATHS_MAX`].
    pub changed: Vec<String>,
    /// How many changed paths were dropped by that cap.
    pub changed_elided: usize,
    /// `git log --oneline` lines, capped at [`GIT_RECENT_COMMITS_MAX`].
    pub recent_commits: Vec<String>,
}

impl GitContext {
    /// True when there is nothing worth injecting.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.branch.is_none() && self.changed.is_empty() && self.recent_commits.is_empty()
    }

    /// The injected section, or an empty string when there is nothing to say.
    ///
    /// Framed as state rather than instruction, and explicitly stamped as a
    /// snapshot: an agent that treats this as current will act on a stale tree,
    /// and the only defence against that is telling it plainly.
    #[must_use]
    pub fn to_system_context(&self) -> String {
        if self.is_empty() {
            return String::new();
        }

        let mut out = String::from(
            "## Git state (snapshot, not live)\n\
             This is how the repository looked when this context was assembled. \
             It is NOT live: if you are about to rely on it, run `git status` \
             yourself first. Uncommitted work listed here exists only in the \
             working tree — do not overwrite a file that appears below without \
             reading it.\n",
        );

        if let Some(ref branch) = self.branch {
            out.push_str("\nBranch: ");
            out.push_str(branch);
            out.push('\n');
        }

        if self.changed.is_empty() {
            out.push_str("\nWorking tree: clean.\n");
        } else {
            out.push_str("\nUncommitted changes:\n");
            for line in &self.changed {
                out.push_str("  ");
                out.push_str(line);
                out.push('\n');
            }
            if self.changed_elided > 0 {
                // Writing into a String cannot fail; the result is discarded so
                // a formatting slip could never cost the rest of the section.
                let _ = writeln!(
                    out,
                    "  ... and {} more changed path(s), not listed — run \
                     `git status --short` for the full list.",
                    self.changed_elided
                );
            }
        }

        if !self.recent_commits.is_empty() {
            out.push_str("\nRecent commits:\n");
            for line in &self.recent_commits {
                out.push_str("  ");
                out.push_str(line);
                out.push('\n');
            }
        }

        out
    }
}

/// Truncate to `max_bytes` without splitting a UTF-8 character.
fn bounded(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    debug_assert!(
        s.is_char_boundary(end),
        "truncation must land on a boundary"
    );
    &s[..end]
}

/// True when `root` looks like a git repository.
///
/// `.git` is a directory in a normal clone and a *file* in a worktree or
/// submodule, so this checks for existence rather than for a directory — a
/// worktree is exactly the case where knowing the branch matters most.
#[must_use]
pub fn is_git_repo(root: &Path) -> bool {
    root.join(".git").exists()
}

/// Run one `git` invocation under the byte and time ceilings.
///
/// Returns `None` on any failure — git missing, a non-zero exit, a timeout.
/// Every one of those means "no snapshot", never "fail the run": this is
/// context, and a turn that could not read the tree is still a turn.
///
/// `--no-pager` and a null stdin matter: git will otherwise start a pager when
/// it believes it has a terminal, which would leave a child waiting on input
/// that never comes. With no pager this is a leaf process, so `kill_on_drop`
/// is containment enough — there are no descendants for it to miss.
async fn run_git(root: &Path, args: &[&str]) -> Option<String> {
    let mut command = Command::new("git");
    command
        .arg("--no-pager")
        .arg("-C")
        .arg(root)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true);

    let child = match command.spawn() {
        Ok(child) => child,
        Err(e) => {
            debug!(error = %e, "git not available for workspace context");
            return None;
        }
    };

    let Ok(output) =
        tokio::time::timeout(GIT_TIMEOUT, read_capped(child, GIT_OUTPUT_BYTES_MAX)).await
    else {
        debug!(?args, "git timed out building workspace context");
        return None;
    };
    let Some(bytes) = output else {
        debug!(?args, "git exited non-zero");
        return None;
    };
    Some(String::from_utf8_lossy(&bytes).into_owned())
}

/// Read at most `cap` bytes of `child`'s stdout, then stop it.
///
/// Returns `None` when the child exits unsuccessfully (or cannot be read).
/// Output past `cap` is never read: once one byte beyond the cap arrives the
/// child is killed and the first `cap` bytes are the answer, exactly the bytes
/// the old read-everything-then-slice kept. That read buffered the whole
/// stream first — megabytes for `git status` over an unignored
/// `node_modules` — and a child that never stopped writing ran the whole
/// timeout out and produced nothing at all.
async fn read_capped(mut child: tokio::process::Child, cap: usize) -> Option<Vec<u8>> {
    use tokio::io::AsyncReadExt as _;
    assert!(cap > 0, "a zero cap reads nothing");
    let stdout = child.stdout.take()?;
    let limit = u64::try_from(cap).unwrap_or(u64::MAX).saturating_add(1);
    let mut bytes = Vec::with_capacity(cap.min(8 * 1024));
    stdout.take(limit).read_to_end(&mut bytes).await.ok()?;

    if bytes.len() > cap {
        bytes.truncate(cap);
        // More than we keep: stop it rather than drain the rest. The pipe is
        // already closed, so a killed or SIGPIPE'd exit is expected here and
        // says nothing about the bytes we have.
        let _ = child.start_kill();
        let _ = child.wait().await;
        debug_assert_eq!(bytes.len(), cap, "a truncated read is exactly the cap");
        return Some(bytes);
    }
    let status = child.wait().await.ok()?;
    debug_assert!(bytes.len() <= cap, "never more than the cap");
    status.success().then_some(bytes)
}

/// Split `git status --short --branch` output into the branch line and the
/// changed paths, applying the count and per-line caps.
///
/// Pure, so the caps are testable without a repository on disk.
#[must_use]
pub fn parse_status(raw: &str) -> (Option<String>, Vec<String>, usize) {
    let mut branch = None;
    let mut changed = Vec::new();
    let mut total_changed = 0usize;

    for line in raw.lines() {
        // `--branch` emits exactly one leading `## ` line. A path can also
        // legitimately start with `#`, but never at column 0 in short format —
        // status codes occupy the first two columns.
        if let Some(rest) = line.strip_prefix("## ")
            && branch.is_none()
        {
            branch = Some(bounded(rest.trim_end(), GIT_LINE_BYTES_MAX).to_string());
            continue;
        }
        if line.trim().is_empty() {
            continue;
        }
        total_changed += 1;
        if changed.len() < GIT_CHANGED_PATHS_MAX {
            changed.push(bounded(line.trim_end(), GIT_LINE_BYTES_MAX).to_string());
        }
    }

    let elided = total_changed.saturating_sub(changed.len());
    debug_assert!(
        changed.len() <= GIT_CHANGED_PATHS_MAX,
        "the listed paths must stay within their ceiling"
    );
    debug_assert_eq!(
        changed.len() + elided,
        total_changed,
        "every changed path is either listed or counted as elided"
    );
    (branch, changed, elided)
}

/// Apply the count and per-line caps to `git log --oneline` output.
#[must_use]
pub fn parse_log(raw: &str) -> Vec<String> {
    let commits: Vec<String> = raw
        .lines()
        .filter(|l| !l.trim().is_empty())
        .take(GIT_RECENT_COMMITS_MAX)
        .map(|l| bounded(l.trim_end(), GIT_LINE_BYTES_MAX).to_string())
        .collect();
    debug_assert!(
        commits.len() <= GIT_RECENT_COMMITS_MAX,
        "the listed commits must stay within their ceiling"
    );
    commits
}

/// Load the git snapshot for `root`, or `None` when it is not a repository (or
/// git could not answer within its bounds).
pub async fn load_git_context(root: &Path) -> Option<GitContext> {
    // Check the marker before spawning: the overwhelmingly common case for a
    // non-repo workspace is that there is nothing to ask, and paying for a
    // process launch to learn that on every run is the wrong trade.
    if !is_git_repo(root) {
        return None;
    }

    let status = run_git(root, &["status", "--short", "--branch"]).await?;
    let (branch, changed, changed_elided) = parse_status(&status);

    // `-n` carries the cap into git itself, so it stops at the limit rather
    // than streaming a full history we would only throw away. A repo with no
    // commits yet answers non-zero here; that is a `None` for this half only,
    // not for the whole snapshot.
    let depth = GIT_RECENT_COMMITS_MAX.to_string();
    let recent_commits = run_git(
        root,
        &["log", "--oneline", "--no-decorate", "-n", depth.as_str()],
    )
    .await
    .map_or_default(|raw| parse_log(&raw));

    let context = GitContext {
        branch,
        changed,
        changed_elided,
        recent_commits,
    };
    if context.is_empty() {
        None
    } else {
        Some(context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_status_reads_the_branch_line_and_the_paths() {
        let raw = "## main...origin/main [ahead 2]\n M src/lib.rs\n?? new.txt\n";
        let (branch, changed, elided) = parse_status(raw);
        assert_eq!(branch.as_deref(), Some("main...origin/main [ahead 2]"));
        assert_eq!(
            changed,
            vec![" M src/lib.rs".to_string(), "?? new.txt".to_string()]
        );
        assert_eq!(elided, 0);
    }

    #[test]
    fn parse_status_handles_a_clean_tree() {
        let (branch, changed, elided) = parse_status("## main\n");
        assert_eq!(branch.as_deref(), Some("main"));
        assert_eq!(changed, [] as [std::string::String; 0]);
        assert_eq!(elided, 0);
    }

    #[test]
    fn parse_status_handles_no_branch_line() {
        let (branch, changed, _) = parse_status(" M a.rs\n");
        assert_eq!(branch, None);
        assert_eq!(changed.len(), 1);
    }

    /// A truncated file list that looks complete is worse than no list: the
    /// model would conclude the paths it cannot see are clean.
    #[test]
    fn parse_status_counts_what_it_had_to_drop() {
        let mut raw = String::from("## main\n");
        let total = GIT_CHANGED_PATHS_MAX * 2 + 7;
        for i in 0..total {
            let _ = writeln!(raw, " M file{i}.rs");
        }
        let (_, changed, elided) = parse_status(&raw);
        assert_eq!(changed.len(), GIT_CHANGED_PATHS_MAX);
        assert_eq!(elided, total - GIT_CHANGED_PATHS_MAX);
        assert_eq!(changed.len() + elided, total);
    }

    #[test]
    fn the_elision_is_stated_in_the_injected_text() {
        let context = GitContext {
            branch: Some("main".into()),
            changed: vec![" M a.rs".into()],
            changed_elided: 12,
            recent_commits: Vec::new(),
        };
        let text = context.to_system_context();
        assert!(text.contains("12 more changed path(s)"), "{text}");
        assert!(
            text.contains("git status --short"),
            "must say how to see the rest: {text}"
        );
    }

    #[test]
    fn long_lines_are_clipped_on_a_character_boundary() {
        // A multi-byte path far over the ceiling: the cut must not panic and
        // must not exceed the cap.
        let path = "é".repeat(GIT_LINE_BYTES_MAX);
        let (_, changed, _) = parse_status(&format!(" M {path}\n"));
        assert_eq!(changed.len(), 1);
        assert!(changed[0].len() <= GIT_LINE_BYTES_MAX);
    }

    #[test]
    fn parse_log_applies_the_commit_ceiling() {
        let mut raw = String::new();
        for i in 0..GIT_RECENT_COMMITS_MAX * 3 {
            let _ = writeln!(raw, "abc{i:04} subject {i}");
        }
        assert_eq!(parse_log(&raw).len(), GIT_RECENT_COMMITS_MAX);
    }

    #[test]
    fn parse_log_ignores_blank_lines() {
        assert_eq!(parse_log("\n\nabc1234 a subject\n\n").len(), 1);
    }

    /// The header is the whole reason this is safe to inject: without the
    /// "snapshot" framing an agent treats a stale tree as current.
    #[test]
    fn the_injection_states_that_it_is_a_snapshot_and_not_live() {
        let context = GitContext {
            branch: Some("main".into()),
            changed: Vec::new(),
            changed_elided: 0,
            recent_commits: vec!["abc1234 a commit".into()],
        };
        let text = context.to_system_context();
        assert!(text.contains("snapshot, not live"), "{text}");
        assert!(text.contains("NOT live"), "{text}");
        assert!(
            text.contains("run `git status`"),
            "must say how to refresh: {text}"
        );
        assert!(text.contains("Working tree: clean."), "{text}");
        assert!(text.contains("abc1234 a commit"), "{text}");
    }

    #[test]
    fn an_empty_context_injects_nothing() {
        assert_eq!(GitContext::default().to_system_context(), "");
        assert!(GitContext::default().is_empty());
    }

    #[tokio::test]
    async fn a_directory_that_is_not_a_repository_yields_no_context() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!is_git_repo(dir.path()));
        assert_eq!(load_git_context(dir.path()).await, None);
    }

    /// A `.git` *file* is how a worktree and a submodule mark their root, and a
    /// worktree is exactly where knowing the branch matters most.
    #[test]
    fn a_git_file_counts_as_a_repository_marker() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(".git"), "gitdir: ../elsewhere\n").unwrap();
        assert!(is_git_repo(dir.path()));
    }

    /// A child that never stops writing is cut at the cap and answered at
    /// once. Buffering everything first, as the old read did, ran the whole
    /// timeout out on this shape and returned nothing.
    #[cfg(unix)]
    #[tokio::test]
    async fn an_endless_writer_is_cut_at_the_cap_not_the_timeout() {
        let child = Command::new("yes")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .unwrap();
        let started = std::time::Instant::now();
        let bytes = tokio::time::timeout(GIT_TIMEOUT, read_capped(child, 4096))
            .await
            .expect("answered before the timeout")
            .expect("a truncated read is still an answer");
        assert_eq!(bytes.len(), 4096);
        assert!(bytes.starts_with(b"y\ny\n"));
        assert!(
            started.elapsed() < GIT_TIMEOUT / 2,
            "{:?}",
            started.elapsed()
        );

        // Under the cap, the exit status decides.
        let failing = Command::new("sh")
            .args(["-c", "echo partial; exit 3"])
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        assert_eq!(
            read_capped(failing, 4096).await,
            None,
            "non-zero exit is no answer"
        );
        let ok = Command::new("sh")
            .args(["-c", "echo whole"])
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        assert_eq!(
            read_capped(ok, 4096).await.as_deref(),
            Some(&b"whole\n"[..])
        );
    }

    /// Run `git` synchronously for the fixture setup below. Returns false when
    /// git is not on PATH, which is the one legitimate reason to skip.
    fn git_in(dir: &Path, args: &[&str]) -> bool {
        std::process::Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(args)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    }

    /// The one test that actually spawns `git` and reads a real repository.
    ///
    /// The parsers above are pure and pinned, but they are fed by a subprocess
    /// whose flags, exit codes, and output shape are the part most likely to be
    /// wrong — `--no-pager`, `-C`, `--short --branch`, `--oneline`. This drives
    /// the whole path end to end against a repository built on disk.
    ///
    /// Skips (rather than fails) when git is absent: this crate must still be
    /// testable on a machine without it.
    #[tokio::test]
    async fn a_real_repository_yields_its_branch_dirty_paths_and_commits() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();

        if !git_in(root, &["init", "--initial-branch=main"]) {
            eprintln!("skipping: git is not available on PATH");
            return;
        }
        // A fixture repo has no identity; committing without one fails.
        assert!(git_in(
            root,
            &["config", "user.email", "test@example.invalid"]
        ));
        assert!(git_in(root, &["config", "user.name", "Nanna Test"]));

        std::fs::write(root.join("tracked.txt"), "one\n").unwrap();
        assert!(git_in(root, &["add", "tracked.txt"]));
        assert!(git_in(root, &["commit", "-m", "the first commit"]));

        // Dirty the tree two ways: a modification and an untracked file.
        std::fs::write(root.join("tracked.txt"), "one\ntwo\n").unwrap();
        std::fs::write(root.join("untracked.txt"), "new\n").unwrap();

        let context = load_git_context(root)
            .await
            .expect("a real repository must produce a snapshot");

        assert!(
            context
                .branch
                .as_deref()
                .is_some_and(|b| b.contains("main")),
            "branch line should name the branch: {:?}",
            context.branch
        );
        let changed = context.changed.join("\n");
        assert!(changed.contains("tracked.txt"), "{changed}");
        assert!(changed.contains("untracked.txt"), "{changed}");
        assert_eq!(context.changed_elided, 0);
        assert_eq!(
            context.recent_commits.len(),
            1,
            "{:?}",
            context.recent_commits
        );
        assert!(
            context.recent_commits[0].contains("the first commit"),
            "{:?}",
            context.recent_commits
        );

        // And the thing that actually reaches the model.
        let text = context.to_system_context();
        assert!(text.contains("## Git state (snapshot, not live)"), "{text}");
        assert!(text.contains("tracked.txt"), "{text}");
        assert!(text.contains("the first commit"), "{text}");
    }
}
