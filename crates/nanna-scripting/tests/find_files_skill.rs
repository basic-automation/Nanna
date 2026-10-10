#![warn(clippy::pedantic, clippy::nursery, clippy::all)]
//! Behavioral tests for the `find_files` default skill, executed for real
//! through the Boa engine with a bridge scoped to a temp directory.
//!
//! The contract: the two pattern shapes mean what the description says, every
//! bound that shaped the answer is stated in the answer, a search that ran and
//! found nothing SUCCEEDS (a `success: false` there earns a retry loop instead
//! of a different question), and a wrong call is corrected rather than thrown.
//!
//! The bound that matters most is the scan cap. A truncated walk makes "no
//! matches" a statement about the cap and not about the tree, so the result has
//! to say so — that is the whole reason this tool reports what it searched.

#![cfg(feature = "boa")]

mod common;

use nanna_scripting::{ScriptEngine, ScriptedTool, ToolPermissions};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

fn skill_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../nanna-tools/default-skills/find_files/tool.ts")
}

fn skill_missing() -> bool {
    if skill_path().is_file() {
        false
    } else {
        eprintln!("skipping: {} not present", skill_path().display());
        true
    }
}

async fn run_find(input: Value, dir: &Path) -> Value {
    let tool = ScriptedTool::from_file(skill_path())
        .expect("read find_files tool.ts")
        .with_permissions(ToolPermissions::none().with_read([dir]))
        // Scaffolding, not an assertion — see `common::FIXTURE_TIMEOUT_MS`.
        .with_timeout(common::FIXTURE_TIMEOUT_MS);
    ScriptEngine::new()
        .execute(&tool, input, None, None)
        .await
        .expect("find_files should not throw")
        .value
}

async fn find_ok(pattern: &str, dir: &Path) -> String {
    let result = run_find(
        json!({ "pattern": pattern, "path": dir.to_string_lossy() }),
        dir,
    )
    .await;
    assert_eq!(
        result["success"],
        Value::Bool(true),
        "expected success:true, got: {result}"
    );
    result["content"].as_str().expect("content").to_string()
}

fn seed(dir: &Path, name: &str) {
    let path = dir.join(name);
    std::fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
    std::fs::write(&path, "x").expect("seed");
}

fn tree() -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    for name in [
        "Cargo.toml",
        "src/main.rs",
        "src/lib.rs",
        "src/deep/nested/helper.rs",
        "crates/alpha/Cargo.toml",
        "crates/alpha/src/lib.rs",
        "docs/readme.md",
        "notes.txt",
    ] {
        seed(dir.path(), name);
    }
    dir
}

#[tokio::test]
async fn a_pattern_without_a_slash_matches_the_name_at_any_depth() {
    if skill_missing() {
        return;
    }
    let dir = tree();
    let out = find_ok("*.rs", dir.path()).await;
    for expected in [
        "src/main.rs",
        "src/lib.rs",
        "src/deep/nested/helper.rs",
        "crates/alpha/src/lib.rs",
    ] {
        assert!(out.contains(expected), "missing {expected} in:\n{out}");
    }
    assert!(!out.contains("notes.txt"), "matched a non-.rs file:\n{out}");
    assert!(out.contains("Found 4 file(s)"), "{out}");
}

#[tokio::test]
async fn an_exact_name_finds_every_copy_of_it() {
    if skill_missing() {
        return;
    }
    let dir = tree();
    let out = find_ok("Cargo.toml", dir.path()).await;
    assert!(out.contains("Cargo.toml"), "{out}");
    assert!(out.contains("crates/alpha/Cargo.toml"), "{out}");
    assert!(out.contains("Found 2 file(s)"), "{out}");
}

#[tokio::test]
async fn a_pattern_with_a_slash_is_anchored_to_the_search_root() {
    if skill_missing() {
        return;
    }
    let dir = tree();
    // `*` does not cross a separator, so this reaches src/ and no deeper.
    let out = find_ok("src/*.rs", dir.path()).await;
    assert!(out.contains("src/main.rs"), "{out}");
    assert!(out.contains("src/lib.rs"), "{out}");
    assert!(
        !out.contains("deep/nested/helper.rs"),
        "`*` must not cross a path separator:\n{out}"
    );
    assert!(
        !out.contains("crates/alpha/src/lib.rs"),
        "the pattern is anchored at the root:\n{out}"
    );
}

#[tokio::test]
async fn double_star_crosses_separators_and_matches_zero_directories() {
    if skill_missing() {
        return;
    }
    let dir = tree();
    let deep = find_ok("src/**/*.rs", dir.path()).await;
    assert!(deep.contains("src/deep/nested/helper.rs"), "{deep}");
    // `**/` must also match ZERO directories, or `src/**/*.rs` would skip the
    // files sitting directly in src/ — the classic off-by-one in glob engines.
    assert!(deep.contains("src/main.rs"), "`**/` must match zero dirs:\n{deep}");

    let anywhere = find_ok("**/Cargo.toml", dir.path()).await;
    assert!(anywhere.contains("crates/alpha/Cargo.toml"), "{anywhere}");
    assert!(
        anywhere.contains("Found 2 file(s)"),
        "`**/` must reach the root copy too:\n{anywhere}"
    );
}

#[tokio::test]
async fn a_question_mark_matches_exactly_one_character() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    seed(dir.path(), "a.rs");
    seed(dir.path(), "ab.rs");
    let out = find_ok("?.rs", dir.path()).await;
    assert!(out.contains("a.rs"), "{out}");
    assert!(out.contains("Found 1 file(s)"), "`?` matched more than one char:\n{out}");
}

/// Regex metacharacters that are not glob operators must be literal, or a
/// filename like `a+b.txt` would be read as a regex and match the wrong files.
#[tokio::test]
async fn regex_metacharacters_in_a_pattern_are_literal() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    seed(dir.path(), "a+b.txt");
    seed(dir.path(), "aab.txt");
    let out = find_ok("a+b.txt", dir.path()).await;
    assert!(out.contains("a+b.txt"), "{out}");
    assert!(
        !out.contains("aab.txt"),
        "`+` was treated as a regex quantifier:\n{out}"
    );

    // `.` likewise: `a.txt` must not match `axtxt`.
    let dir2 = tempfile::tempdir().expect("tempdir");
    seed(dir2.path(), "a.txt");
    seed(dir2.path(), "axtxt");
    let out2 = find_ok("a.txt", dir2.path()).await;
    assert!(out2.contains("Found 1 file(s)"), "`.` matched any char:\n{out2}");
}

/// A search that ran correctly and found nothing SUCCEEDED. Reporting
/// `success: false` there reads as a broken tool and earns a retry loop
/// instead of a different question.
#[tokio::test]
async fn finding_nothing_is_a_success_that_says_what_it_searched() {
    if skill_missing() {
        return;
    }
    let dir = tree();
    let result = run_find(
        json!({ "pattern": "*.zig", "path": dir.path().to_string_lossy() }),
        dir.path(),
    )
    .await;
    assert_eq!(result["success"], Value::Bool(true), "got: {result}");
    let content = result["content"].as_str().expect("content");
    assert!(content.contains("no file matched"), "{content}");
    assert!(content.contains("Searched 8 files"), "{content}");
}

/// An anchored pattern that finds nothing is the case most likely to be a
/// mistaken scope rather than an absent file, so the answer says how to widen.
#[tokio::test]
async fn an_anchored_miss_explains_how_to_search_the_whole_tree() {
    if skill_missing() {
        return;
    }
    let dir = tree();
    let out = find_ok("docs/*.rs", dir.path()).await;
    assert!(out.contains("no file matched"), "{out}");
    assert!(out.contains("**/"), "the advice must name the recursive form:\n{out}");
}

#[tokio::test]
async fn every_answer_states_the_pruning_and_depth_limit_it_ran_under() {
    if skill_missing() {
        return;
    }
    let dir = tree();
    for pattern in ["*.rs", "*.zig"] {
        let out = find_ok(pattern, dir.path()).await;
        assert!(out.contains("node_modules"), "pruning unstated for {pattern}:\n{out}");
        assert!(out.contains("10 levels deep"), "depth unstated for {pattern}:\n{out}");
    }
}

#[tokio::test]
async fn max_results_caps_the_list_and_the_cap_announces_itself() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    for i in 0..12 {
        seed(dir.path(), &format!("f{i}.rs"));
    }
    let result = run_find(
        json!({ "pattern": "*.rs", "path": dir.path().to_string_lossy(), "max_results": 5 }),
        dir.path(),
    )
    .await;
    let content = result["content"].as_str().expect("content");
    assert_eq!(result["success"], Value::Bool(true));
    assert!(content.contains("RESULTS TRUNCATED"), "{content}");
    assert!(content.contains("Found 5 file(s)"), "{content}");
}

#[tokio::test]
async fn a_missing_pattern_is_corrected_not_thrown() {
    if skill_missing() {
        return;
    }
    let dir = tree();
    let result = run_find(json!({ "path": dir.path().to_string_lossy() }), dir.path()).await;
    assert_eq!(result["success"], Value::Bool(false), "got: {result}");
    let content = result["content"].as_str().expect("content");
    assert!(content.contains("missing required parameter"), "{content}");
    assert!(content.contains("Nothing was searched"), "{content}");
}

/// `path` is the directory to search UNDER. Handed a file, the os error
/// ("Not a directory") teaches nothing, so it becomes the correction.
#[tokio::test]
async fn a_file_given_as_the_search_root_is_explained() {
    if skill_missing() {
        return;
    }
    let dir = tree();
    let file = dir.path().join("notes.txt");
    let result = run_find(
        json!({ "pattern": "*.rs", "path": file.to_string_lossy() }),
        dir.path(),
    )
    .await;
    let content = result["content"].as_str().expect("content");
    assert_eq!(result["success"], Value::Bool(false), "got: {content}");
    assert!(content.contains("is a FILE, not a directory"), "{content}");
    assert!(content.contains("Nothing was searched"), "{content}");
}

#[tokio::test]
async fn an_unreadable_glob_is_refused_rather_than_silently_reinterpreted() {
    if skill_missing() {
        return;
    }
    let dir = tree();
    let result = run_find(
        json!({ "pattern": "src/[unclosed.rs", "path": dir.path().to_string_lossy() }),
        dir.path(),
    )
    .await;
    assert_eq!(result["success"], Value::Bool(false), "got: {result}");
    let content = result["content"].as_str().expect("content");
    assert!(content.contains("could not read"), "{content}");
    assert!(content.contains("Nothing was searched"), "{content}");
}

#[tokio::test]
async fn a_character_class_selects_and_negates() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    seed(dir.path(), "a.rs");
    seed(dir.path(), "b.rs");
    seed(dir.path(), "c.rs");
    let picked = find_ok("[ab].rs", dir.path()).await;
    assert!(picked.contains("Found 2 file(s)"), "{picked}");
    let negated = find_ok("[!ab].rs", dir.path()).await;
    assert!(negated.contains("c.rs"), "{negated}");
    assert!(negated.contains("Found 1 file(s)"), "{negated}");
}

/// With no `path` (the default "."), or a relative one, results are relative
/// to the search root and a pattern with a slash matches. The bridge names
/// entries by their resolved path (`/ws/./src/main.rs`) and the raw "." never
/// prefixed them, so `src/**/*.rs` matched nothing and names printed absolute.
#[tokio::test]
async fn a_slash_pattern_matches_under_the_default_root() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().expect("temp dir");
    std::fs::create_dir_all(dir.path().join("src/bin")).expect("mkdir");
    std::fs::write(dir.path().join("src/main.rs"), "fn main() {}\n").expect("write");
    std::fs::write(dir.path().join("src/bin/tool.rs"), "fn main() {}\n").expect("write");
    let tool = ScriptedTool::from_file(skill_path())
        .expect("read find_files tool.ts")
        .with_permissions(ToolPermissions::none().with_read([dir.path()]))
        .with_timeout(common::FIXTURE_TIMEOUT_MS);
    let result = ScriptEngine::new()
        .execute_with_workdir(
            &tool,
            json!({ "pattern": "src/**/*.rs" }),
            None,
            None,
            Some(dir.path().to_path_buf()),
        )
        .await
        .expect("find_files should not throw")
        .value;
    let content = result["content"].as_str().expect("content");
    assert!(content.contains("src/main.rs"), "{content}");
    assert!(content.contains("src/bin/tool.rs"), "{content}");
    assert!(
        !content.contains(&*dir.path().to_string_lossy()),
        "paths are relative to the root: {content}"
    );
}
