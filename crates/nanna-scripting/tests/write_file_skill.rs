#![warn(clippy::pedantic, clippy::nursery, clippy::all)]
//! Behavioral tests for the `write_file` default skill (v0.1.4), executed
//! for real through the Boa engine. Covers the structured guidance
//! failures, the versioned-copy-name refusal, and the shrink guard. The
//! Python syntax gate needs `Nanna.exec`, which this harness does not
//! grant — that path fails OPEN here by design and is exercised live.
//!
//! Tolerant by design: if the sibling default-skills tree isn't present,
//! the tests no-op.

#![cfg(feature = "boa")]

mod common;

use nanna_scripting::{ScriptEngine, ScriptedTool, ToolPermissions};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

fn skill_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../nanna-tools/default-skills/write_file/tool.ts")
}

/// The bridge workdir is pinned to the temp dir so the skill's own state
/// files (`.nanna/write_hiwater.json`, `.nanna/read_marks.json`) resolve
/// inside the fixture instead of the developer's home directory — reachable,
/// writable, and thrown away with the tempdir. Without it every run starts
/// with no read mark and the ratchet never persists, which is neither what
/// production does nor what these tests mean to exercise.
async fn run_write(input: Value, dir: &Path) -> Result<Value, String> {
    let tool = ScriptedTool::from_file(skill_path())
        .expect("read write_file tool.ts")
        .with_permissions(ToolPermissions::none().with_read([dir]).with_write([dir]))
        // Scaffolding, not an assertion — see `common::FIXTURE_TIMEOUT_MS`.
        .with_timeout(common::FIXTURE_TIMEOUT_MS);
    ScriptEngine::new()
        .execute_with_workdir(&tool, input, None, None, Some(dir.to_path_buf()))
        .await
        .map(|r| r.value)
        .map_err(|e| e.to_string())
}

/// Milliseconds since the epoch of a file's mtime, the unit read marks use.
fn mtime_ms(path: &Path) -> u64 {
    let modified = std::fs::metadata(path).expect("stat").modified().expect("mtime");
    u64::try_from(
        modified
            .duration_since(std::time::UNIX_EPOCH)
            .expect("mtime after epoch")
            .as_millis(),
    )
    .expect("mtime fits u64")
}

/// Seed `.nanna/read_marks.json` directly. With the workdir pinned above, the
/// skill's canonical key for a file inside the fixture is just its name.
fn seed_read_mark(dir: &Path, key: &str, at_ms: u64) {
    let state = dir.join(".nanna");
    std::fs::create_dir_all(&state).expect("mkdir .nanna");
    std::fs::write(
        state.join("read_marks.json"),
        json!({ key: { "at": at_ms } }).to_string(),
    )
    .expect("write read_marks.json");
}

async fn run_fail(input: Value, dir: &Path) -> String {
    let result = run_write(input, dir).await.expect("failures are returned, not thrown");
    assert_eq!(
        result["success"],
        Value::Bool(false),
        "expected failure, got: {result}"
    );
    result["content"].as_str().expect("content").to_string()
}

fn skill_missing() -> bool {
    if skill_path().is_file() {
        false
    } else {
        eprintln!("skipping: {} not present", skill_path().display());
        true
    }
}

#[tokio::test]
async fn writes_and_reports() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("a.txt").to_string_lossy().into_owned();

    let result = run_write(
        json!({ "file_path": target, "content": "hello" }),
        dir.path(),
    )
    .await
    .expect("write should succeed");
    let content = result["content"].as_str().expect("content");
    assert!(content.contains("Wrote 5 bytes"), "got: {content}");
    assert_eq!(std::fs::read_to_string(dir.path().join("a.txt")).unwrap(), "hello");
}

#[tokio::test]
async fn missing_content_is_a_structured_failure() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("b.txt").to_string_lossy().into_owned();

    // Round-6 live log showed this reaching the model as a thrown error
    // under five "Execution failed:" prefixes — it must be structured now.
    let err = run_fail(json!({ "file_path": target }), dir.path()).await;
    assert!(err.contains("missing content"), "got: {err}");
    assert!(err.contains("Nothing was written"), "got: {err}");
}

#[tokio::test]
async fn versioned_copy_names_are_refused() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();

    for name in [
        "new_notekeeper.py",
        "notekeeper_v2.py",
        "script_fixed.txt",
        "runner.py.new",
        "notes_backup.py",
    ] {
        let target = dir.path().join(name).to_string_lossy().into_owned();
        let err = run_fail(
            json!({ "file_path": target, "content": "print('hi')\n" }),
            dir.path(),
        )
        .await;
        assert!(err.contains("WRITE REFUSED"), "{name}: {err}");
        assert!(err.contains("versioned copy"), "{name}: {err}");
        assert!(!err.contains("force"), "must not advertise force: {err}");
        assert!(!dir.path().join(name).exists(), "{name} must not be created");
    }

    // force=true is the escape hatch for genuinely new standalone files.
    let target = dir.path().join("new_module.py").to_string_lossy().into_owned();
    let result = run_write(
        json!({ "file_path": target, "content": "print('hi')\n", "force": true }),
        dir.path(),
    )
    .await
    .expect("forced write succeeds");
    assert_eq!(result["success"], Value::Null, "write_file success result has no success:false");
    assert!(dir.path().join("new_module.py").exists());
}

#[tokio::test]
async fn valid_rewrite_over_a_parked_draft_is_accepted() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("a.py").to_string_lossy().into_owned();
    std::fs::write(dir.path().join("a.py.__buffer__"), "draft = (").unwrap();

    // Round-13 lesson: the old rail bounced even VALID regenerations while
    // a draft was parked — throwing away the model's one reliable move.
    // Valid content always wins now. (In this harness the syntax checker
    // fails open, which lands on the same accept path.)
    let result = run_write(
        json!({ "file_path": target, "content": "print('regenerated')\n" }),
        dir.path(),
    )
    .await
    .expect("valid rewrite must be accepted");
    let content = result["content"].as_str().expect("content");
    assert!(content.contains("Wrote"), "got: {content}");
    assert_eq!(
        std::fs::read_to_string(dir.path().join("a.py")).unwrap(),
        "print('regenerated')\n"
    );
}

#[tokio::test]
async fn shrink_guard_still_refuses_fragments() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let real = dir.path().join("big.txt");
    std::fs::write(&real, "y".repeat(1_000)).unwrap();
    let target = real.to_string_lossy().into_owned();
    // The session has read the file as it stands now, so the stale-shrink
    // hold has nothing to say and the 30% floor is what answers.
    seed_read_mark(dir.path(), "big.txt", mtime_ms(&real));

    let err = run_fail(
        json!({ "file_path": target, "content": "fragment" }),
        dir.path(),
    )
    .await;
    assert!(err.contains("WRITE REFUSED"), "got: {err}");
    assert!(err.contains("NOT modified"), "got: {err}");
    assert_eq!(std::fs::read_to_string(&real).unwrap().len(), 1_000);
}

/// P22 Tier 3 stale-shrink hold, NEVER-READ branch: no read mark exists at
/// all. The hold is right — you cannot shrink what you have not seen — but
/// the reason it states has to be the true one. These replies are read
/// literally by 9B-class local models, and "the file has CHANGED since you
/// last read it" sends one hunting for a change that never happened.
#[tokio::test]
async fn stale_shrink_hold_echoes_the_current_file() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let real = dir.path().join("notes.txt");
    let original = "alpha\nbeta\ngamma\n".repeat(50);
    std::fs::write(&real, &original).unwrap();
    let target = real.to_string_lossy().into_owned();

    // Well clear of the 30% floor, so the hold is the only thing in play.
    let merged = "alpha\nbeta\n".repeat(50);
    let err = run_fail(
        json!({ "file_path": target.clone(), "content": merged.clone() }),
        dir.path(),
    )
    .await;

    assert!(err.contains("WRITE HELD"), "got: {err}");
    assert!(
        err.contains("NEVER read this file in this session"),
        "must name the real reason: {err}"
    );
    assert!(
        !err.contains("CHANGED since you last read it"),
        "must not claim a change that never happened: {err}"
    );
    assert!(err.contains("gamma"), "the echo must carry the current content: {err}");
    assert!(
        err.contains("counts as your read"),
        "must say the bounce is the read: {err}"
    );
    assert_eq!(std::fs::read_to_string(&real).unwrap(), original);

    // The echo counted as the read, so the same write lands on the retry.
    run_write(json!({ "file_path": target, "content": merged.clone() }), dir.path())
        .await
        .expect("the retry after the echo must be accepted");
    assert_eq!(std::fs::read_to_string(&real).unwrap(), merged);
}

/// The other branch of the same hold: a read mark EXISTS but predates the
/// file's mtime, so the file genuinely moved under the model. Here the
/// "CHANGED since you last read it" sentence is the true one.
#[tokio::test]
async fn stale_shrink_hold_names_the_change_when_the_file_moved_under_us() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let real = dir.path().join("notes.txt");
    let original = "alpha\nbeta\ngamma\n".repeat(50);
    std::fs::write(&real, &original).unwrap();
    let target = real.to_string_lossy().into_owned();
    // Read ten minutes before the file was last written: seen, then changed.
    seed_read_mark(dir.path(), "notes.txt", mtime_ms(&real) - 600_000);

    let err = run_fail(
        json!({ "file_path": target, "content": "alpha\nbeta\n".repeat(50) }),
        dir.path(),
    )
    .await;

    assert!(err.contains("WRITE HELD"), "got: {err}");
    assert!(
        err.contains("CHANGED since you last read it"),
        "must name the real reason: {err}"
    );
    assert!(
        !err.contains("NEVER read this file"),
        "the session did read it: {err}"
    );
    assert!(err.contains("gamma"), "the echo must carry the current content: {err}");
    assert_eq!(std::fs::read_to_string(&real).unwrap(), original);
}

/// REGRESSION (2026-07-27): a run built `./minidb` to 8/42 acceptance checks
/// passing, then forked onto `./minidb.sh` and spent the rest of its time
/// improving a file the tests never read. The marker list catches renames
/// that announce themselves (`_v2`, `_backup`, `.new`); this is the quiet
/// fork — same name, extension added.
#[tokio::test]
async fn adding_an_extension_to_an_existing_file_is_refused() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let original = dir.path().join("minidb");
    std::fs::write(&original, "#!/bin/sh\necho original\n").unwrap();

    let fork = dir.path().join("minidb.sh").to_string_lossy().into_owned();
    let err = run_fail(
        json!({ "file_path": fork, "content": "#!/bin/sh\necho fork\n" }),
        dir.path(),
    )
    .await;

    assert!(err.contains("WRITE REFUSED"), "got: {err}");
    assert!(err.contains("minidb"), "must name the original: {err}");
    assert!(err.contains("edit_file"), "must offer the way forward: {err}");
    assert!(
        !dir.path().join("minidb.sh").exists(),
        "the fork must not be created"
    );
}

/// Conventional variants are siblings, not forks: `.env.local` beside `.env`
/// and `Dockerfile.dev` beside `Dockerfile` were refused as copies.
#[tokio::test]
async fn conventional_variants_are_not_forks() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join(".env"), "A=1\n").unwrap();
    std::fs::write(dir.path().join("Dockerfile"), "FROM scratch\n").unwrap();
    for (name, body) in [(".env.local", "A=2\n"), ("Dockerfile.dev", "FROM alpine\n")] {
        let path = dir.path().join(name).to_string_lossy().into_owned();
        let result = run_write(json!({ "file_path": path, "content": body }), dir.path())
            .await
            .expect("write returns");
        assert_ne!(result["success"], Value::Bool(false), "{name}: {result}");
        assert!(dir.path().join(name).is_file(), "{name} was written");
    }
}

/// The refusal must stay narrow: sibling FORMATS are legitimate. Their stems
/// are not themselves files, which is exactly what distinguishes them from a
/// copy of an extensionless original.
#[tokio::test]
async fn sibling_formats_are_still_allowed() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("config.yaml"), "a: 1\n").unwrap();

    let target = dir.path().join("config.json").to_string_lossy().into_owned();
    run_write(
        json!({ "file_path": target, "content": "{\"a\": 1}\n" }),
        dir.path(),
    )
    .await
    .expect("writing a sibling format must succeed");
    assert!(dir.path().join("config.json").exists());
}

/// A brand-new suffixed file with no extensionless original is ordinary work.
#[tokio::test]
async fn a_suffixed_name_with_no_original_is_fine() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("build.sh").to_string_lossy().into_owned();
    run_write(
        json!({ "file_path": target, "content": "#!/bin/sh\nmake\n" }),
        dir.path(),
    )
    .await
    .expect("a fresh script name must succeed");
    assert!(dir.path().join("build.sh").exists());
}

/// The rule registry is not a file the model may rewrite: `"{}"` erased every
/// rule the user had declared, and `force` must not open it either — only
/// `lift_invariant`, which asks the user, changes it.
#[tokio::test]
async fn the_declared_rules_cannot_be_rewritten_even_with_force() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().expect("temp dir");
    let registry = dir.path().join(".nanna/declared_invariants.json");
    std::fs::create_dir_all(registry.parent().expect("parent")).expect("mkdir");
    let rules = r#"{"invariants":[{"kind":"no_delete","glob":"tests"}]}"#;
    std::fs::write(&registry, rules).expect("seed registry");

    let content = run_fail(
        json!({ "file_path": ".nanna/declared_invariants.json", "content": "{}", "force": true }),
        dir.path(),
    )
    .await;
    assert!(content.contains("lift_invariant"), "{content}");
    assert_eq!(
        std::fs::read_to_string(&registry).expect("read"),
        rules,
        "untouched"
    );
}

/// Ledger keys fold case only where the filesystem does. On Linux
/// `README.md` and `readme.md` are two files; one lowercased key let a read
/// of either satisfy the blind-rewrite hold for the other.
#[tokio::test]
async fn the_ledger_keeps_a_files_case_where_the_filesystem_does() {
    if skill_missing() || cfg!(windows) || cfg!(target_os = "macos") {
        return;
    }
    let dir = tempfile::tempdir().expect("temp dir");
    let result = run_write(
        json!({ "file_path": "README.md", "content": "# Project\n\nSome text.\n" }),
        dir.path(),
    )
    .await
    .expect("write returns");
    assert_ne!(result["success"], Value::Bool(false), "{result}");
    let ledger = std::fs::read_to_string(dir.path().join(".nanna/write_hiwater.json"))
        .expect("the ledger is written");
    assert!(ledger.contains("\"README.md\""), "{ledger}");
    assert!(!ledger.contains("\"readme.md\""), "{ledger}");
}

/// A path with an apostrophe is one shell word to the checks that shell out.
/// Spliced between single quotes it re-split: the Python syntax gate ran
/// nothing, so invalid code was written, and its temp files were left behind.
#[tokio::test]
async fn an_apostrophe_in_the_path_does_not_switch_the_checks_off() {
    if skill_missing() || cfg!(windows) || which_python().is_none() {
        return;
    }
    let dir = tempfile::tempdir().expect("temp dir");
    // The gate shells out to `python`, so this run may spawn processes.
    let mut permissions = ToolPermissions::none()
        .with_read([dir.path()])
        .with_write([dir.path()]);
    permissions.run = true;
    let tool = ScriptedTool::from_file(skill_path())
        .expect("read write_file tool.ts")
        .with_permissions(permissions)
        .with_timeout(common::FIXTURE_TIMEOUT_MS);
    let result = ScriptEngine::new()
        .execute_with_workdir(
            &tool,
            json!({ "file_path": "it's.py", "content": "def broken(:\n    pass\n" }),
            None,
            None,
            Some(dir.path().to_path_buf()),
        )
        .await
        .expect("failures are returned, not thrown")
        .value;
    assert_eq!(result["success"], Value::Bool(false), "{result}");
    let content = result["content"].as_str().expect("content");
    assert!(
        content.contains("line 1"),
        "the gate named the bad line: {content}"
    );
    assert!(
        !dir.path().join("it's.py").exists(),
        "invalid code was not written"
    );
    let leftovers: Vec<String> = std::fs::read_dir(dir.path())
        .expect("list")
        .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().to_string()))
        .filter(|name| name.contains("__chk"))
        .collect();
    assert!(leftovers.is_empty(), "temp files leaked: {leftovers:?}");
}

/// The gate needs a `python` on PATH; without one it fails open by design.
fn which_python() -> Option<PathBuf> {
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|dir| dir.join("python"))
            .find(|candidate| candidate.is_file())
    })
}
