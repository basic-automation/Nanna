#![warn(clippy::pedantic, clippy::nursery, clippy::all)]
//! The `exec` skill's guards over Nanna's own state and the user's declared
//! rules, against a real shell in a temp workspace.
//!
//! `write_file` leaves `no_delete` to exec ("enforced where deletions happen")
//! and refuses rewriting the ratchet ledger — but exec read neither the
//! registry nor the state paths, so `rm tests/x.sh` under a declared
//! "never delete anything under tests/" ran, and `rm .nanna/write_hiwater.json`
//! reset every file's high-water.

#![cfg(feature = "boa")]

mod common;

use nanna_scripting::{ScriptEngine, ScriptedTool, ToolPermissions};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

fn skill_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../nanna-tools/default-skills/exec/tool.ts")
}

async fn run_exec(command: &str, dir: &Path) -> Value {
    let mut permissions = ToolPermissions::none().with_read([dir]).with_write([dir]);
    permissions.run = true;
    let tool = ScriptedTool::from_file(skill_path())
        .expect("read exec tool.ts")
        .with_permissions(permissions)
        // Scaffolding, not an assertion — see `common::FIXTURE_TIMEOUT_MS`.
        .with_timeout(common::FIXTURE_TIMEOUT_MS);
    ScriptEngine::new()
        .execute_with_workdir(
            &tool,
            json!({ "command": command }),
            None,
            None,
            Some(dir.to_path_buf()),
        )
        .await
        .map(|r| r.value)
        .expect("exec answers; it never throws")
}

fn refused(result: &Value) -> String {
    assert_eq!(
        result["success"],
        Value::Bool(false),
        "expected a refusal, got: {result}"
    );
    let content = result["content"].as_str().expect("content").to_string();
    assert!(content.starts_with("NOT EXECUTED"), "{content}");
    content
}

fn declare(dir: &Path, kind: &str, glob: &str) {
    let state = dir.join(".nanna");
    std::fs::create_dir_all(&state).expect("mkdir .nanna");
    std::fs::write(
        state.join("declared_invariants.json"),
        json!({ "invariants": [{ "kind": kind, "glob": glob, "source": "Never touch tests/" }] })
            .to_string(),
    )
    .expect("write registry");
}

#[tokio::test]
async fn nannas_own_records_cannot_be_deleted_or_overwritten() {
    if cfg!(windows) {
        return;
    }
    let dir = tempfile::tempdir().expect("temp dir");
    let ledger = dir.path().join(".nanna/write_hiwater.json");
    std::fs::create_dir_all(ledger.parent().expect("parent")).expect("mkdir");
    std::fs::write(&ledger, r#"{"main.py":{"hi":9000}}"#).expect("seed ledger");

    for command in [
        "rm -f .nanna/write_hiwater.json",
        "echo {} > .nanna/write_hiwater.json",
        "mv .nanna/declared_invariants.json /tmp/x",
        "rm -rf .nanna",
    ] {
        let content = refused(&run_exec(command, dir.path()).await);
        assert!(
            content.contains("Nanna's own record"),
            "{command}: {content}"
        );
    }
    assert!(ledger.is_file(), "the ledger is untouched");
}

#[tokio::test]
async fn a_declared_rule_holds_against_the_shell() {
    if cfg!(windows) {
        return;
    }
    let dir = tempfile::tempdir().expect("temp dir");
    std::fs::create_dir_all(dir.path().join("tests")).expect("mkdir tests");
    let kept = dir.path().join("tests/test_03.sh");
    std::fs::write(&kept, "echo ok\n").expect("seed test");

    declare(dir.path(), "no_delete", "tests");
    let content = refused(&run_exec("rm tests/test_03.sh", dir.path()).await);
    assert!(
        content.contains("Never touch tests/"),
        "the user's words: {content}"
    );
    assert!(kept.is_file(), "no_delete held");
    let wrote = run_exec("echo more >> tests/test_03.sh && echo done", dir.path()).await;
    assert_ne!(
        wrote["success"],
        Value::Bool(false),
        "no_delete does not forbid writing: {wrote}"
    );

    declare(dir.path(), "read_only", "tests");
    refused(&run_exec("echo x > tests/test_03.sh", dir.path()).await);
    refused(&run_exec("mv tests/test_03.sh old.sh", dir.path()).await);
    assert!(kept.is_file(), "read_only held");

    let elsewhere = run_exec("echo scratch > out.txt && cat out.txt", dir.path()).await;
    assert!(
        elsewhere.to_string().contains("scratch"),
        "other paths are untouched by the rule: {elsewhere}"
    );
}
