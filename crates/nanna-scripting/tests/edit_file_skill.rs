#![warn(clippy::pedantic, clippy::nursery, clippy::all)]
//! Behavioral tests for the `edit_file` default skill, executed for real
//! through the Boa engine with a bridge scoped to a temp directory.
//!
//! These are the contract a 32k-window local model depends on: a failed edit
//! must leave the file byte-identical and return a short, instructive
//! `success: false` result (never a thrown error — engine wrapping stacks
//! scary "Execution failed:" prefixes that panic small models); a successful
//! edit must touch only the matched snippet (including when the file and
//! `old_string` disagree on LF vs CRLF line endings).
//!
//! Tolerant by design (same as `default_skills_params.rs`): if the sibling
//! `nanna-tools/default-skills` tree isn't present, the tests no-op.

#![cfg(feature = "boa")]

mod common;

use nanna_scripting::{ScriptEngine, ScriptedTool, ToolPermissions};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

fn skill_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../nanna-tools/default-skills/edit_file/tool.ts")
}

/// Execute the real `edit_file` tool.ts against `input`, sandboxed to `dir`.
/// Returns `Ok(result_value)` or `Err(error_string)`.
async fn run_edit(input: Value, dir: &Path) -> Result<Value, String> {
    let tool = ScriptedTool::from_file(skill_path())
        .expect("read edit_file tool.ts")
        .with_permissions(ToolPermissions::none().with_read([dir]).with_write([dir]))
        // Scaffolding, not an assertion — see `common::FIXTURE_TIMEOUT_MS`.
        .with_timeout(common::FIXTURE_TIMEOUT_MS);
    ScriptEngine::new()
        .execute(&tool, input, None, None)
        .await
        .map(|r| r.value)
        .map_err(|e| e.to_string())
}

/// Run and expect a STRUCTURED failure: the script must complete (no throw)
/// and return `success: false` with an instructive `content` message.
async fn run_edit_fail(input: Value, dir: &Path) -> String {
    let result = run_edit(input, dir)
        .await
        .expect("guidance failures must be returned, not thrown");
    assert_eq!(
        result["success"],
        Value::Bool(false),
        "expected success:false, got: {result}"
    );
    result["content"]
        .as_str()
        .expect("failure content string")
        .to_string()
}

/// Write `content` to `name` inside `dir`, returning the absolute path string.
fn seed(dir: &Path, name: &str, content: &str) -> String {
    let path = dir.join(name);
    std::fs::write(&path, content).expect("seed file");
    path.to_string_lossy().into_owned()
}

fn read(path: &str) -> String {
    std::fs::read_to_string(path).expect("read back")
}

/// Guard: skip (returning true) when the skills tree isn't present.
fn skill_missing() -> bool {
    if skill_path().is_file() {
        false
    } else {
        eprintln!("skipping: {} not present", skill_path().display());
        true
    }
}

#[tokio::test]
async fn unique_match_is_replaced_in_place() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = seed(dir.path(), "a.txt", "line one\nline two\nline three\n");

    let result = run_edit(
        json!({ "file_path": path, "old_string": "line two", "new_string": "line 2" }),
        dir.path(),
    )
    .await
    .expect("edit should succeed");

    assert_eq!(read(&path), "line one\nline 2\nline three\n");
    let content = result["content"].as_str().expect("content string");
    assert!(content.contains("replaced 1 occurrence"), "got: {content}");
    assert_eq!(result["success"], Value::Bool(true));
}

#[tokio::test]
async fn ambiguous_match_fails_and_leaves_file_unchanged() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let original = "foo\nbar\nfoo\n";
    let path = seed(dir.path(), "a.txt", original);

    let err = run_edit_fail(
        json!({ "file_path": path, "old_string": "foo", "new_string": "baz" }),
        dir.path(),
    )
    .await;

    assert_eq!(read(&path), original, "file must be untouched");
    assert!(err.contains("2 matches"), "got: {err}");
    assert!(err.contains("UNCHANGED"), "got: {err}");
    assert!(err.contains("replace_all"), "got: {err}");
    assert!(err.contains("occurrence"), "got: {err}");
}

#[tokio::test]
async fn not_found_fails_and_leaves_file_unchanged() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let original = "alpha\nbeta\n";
    let path = seed(dir.path(), "a.txt", original);

    let err = run_edit_fail(
        json!({
            "file_path": path,
            "old_string": "gamma\ndelta\nepsilon",
            "new_string": "x"
        }),
        dir.path(),
    )
    .await;

    assert_eq!(read(&path), original, "file must be untouched");
    // The miss is named without asserting a CAUSE. It used to claim "the
    // file's real content differs from your memory", which the tool never
    // checked.
    assert!(err.contains("does not appear in"), "got: {err}");
    assert!(
        !err.contains("differs from your memory"),
        "the message must not assert a cause it did not establish: {err}"
    );
    assert!(err.contains("UNCHANGED"), "got: {err}");
    // Re-anchoring guidance: shows what it looked for and tells the model
    // to call read_file first — without dumping the whole file.
    assert!(err.contains("gamma"), "got: {err}");
    assert!(err.contains("read_file"), "got: {err}");
    assert!(!err.contains("alpha"), "must not dump file content: {err}");
}

#[tokio::test]
async fn lf_old_string_matches_crlf_file_without_rewriting_endings() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = seed(dir.path(), "a.txt", "aaa\r\nbbb\r\nccc\r\n");

    run_edit(
        json!({ "file_path": path, "old_string": "aaa\nbbb", "new_string": "AAA\nBBB" }),
        dir.path(),
    )
    .await
    .expect("LF old_string must match CRLF file");

    // Replacement adopts the file's CRLF flavor; untouched tail keeps CRLF.
    assert_eq!(read(&path), "AAA\r\nBBB\r\nccc\r\n");
}

/// The commonest edit — lines added after a one-line anchor — writes in the
/// file's style too. A one-line `old_string` matches exactly, so the CRLF
/// conversion (which ran only when the exact match failed) was skipped and
/// the new lines landed as LF inside a CRLF file.
#[tokio::test]
async fn lines_added_after_a_one_line_anchor_keep_a_crlf_file_crlf() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = seed(dir.path(), "a.txt", "aaa\r\nbbb\r\n");

    run_edit(
        json!({ "file_path": path, "old_string": "aaa", "new_string": "aaa\nadded" }),
        dir.path(),
    )
    .await
    .expect("one-line anchor matches");

    assert_eq!(read(&path), "aaa\r\nadded\r\nbbb\r\n");
}

#[tokio::test]
async fn crlf_old_string_matches_lf_file_without_rewriting_endings() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = seed(dir.path(), "a.txt", "aaa\nbbb\nccc\n");

    run_edit(
        json!({ "file_path": path, "old_string": "aaa\r\nbbb", "new_string": "AAA\r\nBBB" }),
        dir.path(),
    )
    .await
    .expect("CRLF old_string must match LF file");

    assert_eq!(read(&path), "AAA\nBBB\nccc\n");
}

#[tokio::test]
async fn replace_all_replaces_every_occurrence() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = seed(dir.path(), "a.txt", "x foo y foo z");

    let result = run_edit(
        json!({
            "file_path": path,
            "old_string": "foo",
            "new_string": "bar",
            "replace_all": true
        }),
        dir.path(),
    )
    .await
    .expect("replace_all should succeed");

    assert_eq!(read(&path), "x bar y bar z");
    let content = result["content"].as_str().expect("content string");
    assert!(content.contains("replaced 2 occurrence"), "got: {content}");
}

#[tokio::test]
async fn replace_all_accepts_stringified_true() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = seed(dir.path(), "a.txt", "x foo y foo z");

    // Small models often emit booleans as strings; "true" must not fall
    // through to the ambiguity error (which would tell the model to pass
    // exactly what it already passed).
    run_edit(
        json!({
            "file_path": path,
            "old_string": "foo",
            "new_string": "bar",
            "replace_all": "true"
        }),
        dir.path(),
    )
    .await
    .expect("replace_all=\"true\" should succeed");

    assert_eq!(read(&path), "x bar y bar z");
}

#[tokio::test]
async fn occurrence_selects_the_nth_match() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = seed(dir.path(), "a.txt", "x foo y foo z");

    run_edit(
        json!({
            "file_path": path,
            "old_string": "foo",
            "new_string": "bar",
            "occurrence": 2
        }),
        dir.path(),
    )
    .await
    .expect("occurrence=2 should succeed");

    assert_eq!(read(&path), "x foo y bar z");
}

#[tokio::test]
async fn occurrence_out_of_range_fails_and_leaves_file_unchanged() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let original = "x foo y foo z";
    let path = seed(dir.path(), "a.txt", original);

    let err = run_edit_fail(
        json!({
            "file_path": path,
            "old_string": "foo",
            "new_string": "bar",
            "occurrence": 3
        }),
        dir.path(),
    )
    .await;

    assert_eq!(read(&path), original, "file must be untouched");
    assert!(err.contains("out of range"), "got: {err}");
    assert!(err.contains("UNCHANGED"), "got: {err}");
}

#[tokio::test]
async fn missing_params_return_instructive_failures() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = seed(dir.path(), "a.txt", "hello\n");

    // Nothing at all.
    let err = run_edit_fail(json!({}), dir.path()).await;
    assert!(err.contains("file_path"), "got: {err}");
    assert!(err.contains("Nothing was changed"), "got: {err}");
    // The corrected-call exemplar must not model backslashed paths a small
    // model would copy into JSON as invalid escapes.
    assert!(!err.contains("\\\\path"), "got: {err}");

    // File but no snippet.
    let err = run_edit_fail(json!({ "file_path": path }), dir.path()).await;
    assert!(err.contains("old_string"), "got: {err}");

    // File + old but no replacement.
    let err = run_edit_fail(
        json!({ "file_path": path, "old_string": "hello" }),
        dir.path(),
    )
    .await;
    assert!(err.contains("new_string"), "got: {err}");

    // Empty old_string is a whole-file write in disguise.
    let err = run_edit_fail(
        json!({ "file_path": path, "old_string": "", "new_string": "x" }),
        dir.path(),
    )
    .await;
    assert!(err.contains("write_file"), "got: {err}");

    assert_eq!(
        read(&path),
        "hello\n",
        "file must be untouched by failed calls"
    );
}

#[tokio::test]
async fn identical_old_new_is_a_noop_success_when_text_present() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = seed(dir.path(), "a.txt", "hello\nworld\n");

    // The model "confirms" content instead of diffing (observed live). If
    // the desired state already holds, that is success, not an error.
    let result = run_edit(
        json!({ "file_path": path, "old_string": "hello", "new_string": "hello" }),
        dir.path(),
    )
    .await
    .expect("no-op must succeed");
    assert_eq!(result["success"], Value::Bool(true));
    let content = result["content"].as_str().expect("content");
    assert!(content.contains("No change needed"), "got: {content}");
    assert_eq!(read(&path), "hello\nworld\n");
}

#[tokio::test]
async fn identical_old_new_with_absent_text_names_the_staleness() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = seed(dir.path(), "a.txt", "def real_function():\n    return 42\n");

    let err = run_edit_fail(
        json!({
            "file_path": path,
            "old_string": "def real_function():\n    return 99",
            "new_string": "def real_function():\n    return 99"
        }),
        dir.path(),
    )
    .await;
    assert!(err.contains("differs from your memory"), "got: {err}");
    assert!(err.contains("read_file"), "got: {err}");
    // Re-anchoring: the error quotes the closest REAL text.
    assert!(err.contains("return 42"), "got: {err}");
}

#[tokio::test]
async fn indentation_drift_still_matches_via_loose_span() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    // File is 4-space indented; the model remembers 2-space. Content is
    // right, whitespace is wrong — the loose span must match and replace
    // the ORIGINAL bytes.
    let path = seed(
        dir.path(),
        "a.py",
        "def foo():\n    x = 1\n    return x\n\nprint(foo())\n",
    );

    run_edit(
        json!({
            "file_path": path,
            "old_string": "def foo():\n  x = 1\n  return x",
            "new_string": "def foo():\n    x = 2\n    return x"
        }),
        dir.path(),
    )
    .await
    .expect("loose match should succeed");

    assert_eq!(read(&path), "def foo():\n    x = 2\n    return x\n\nprint(foo())\n");
}

#[tokio::test]
async fn not_found_error_quotes_closest_real_text() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = seed(
        dir.path(),
        "a.py",
        "def load_notes():\n    notes = []\n    return notes\n",
    );

    let err = run_edit_fail(
        json!({
            "file_path": path,
            "old_string": "def load_notes(path):\n    result = read(path)\n    return result",
            "new_string": "x"
        }),
        dir.path(),
    )
    .await;
    assert!(err.contains("UNCHANGED"), "got: {err}");
    // Real text from the file, either echoed whole (small files, the common
    // case) or as the closest snippet when the file is too large to fit a
    // failure message. What matters is that the model is shown actual content
    // instead of being sent on another read_file round-trip.
    assert!(
        err.contains("ACTUALLY in the file") || err.contains("Closest ACTUAL text"),
        "got: {err}"
    );
    assert!(err.contains("def load_notes():"), "got: {err}");
}

#[tokio::test]
async fn missing_file_points_to_write_file() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let ghost = dir
        .path()
        .join("no_such.txt")
        .to_string_lossy()
        .into_owned();

    let err = run_edit_fail(
        json!({ "file_path": ghost, "old_string": "a", "new_string": "b" }),
        dir.path(),
    )
    .await;

    assert!(err.contains("could not read"), "got: {err}");
    assert!(err.contains("write_file"), "got: {err}");
}

#[tokio::test]
async fn variant_param_names_are_accepted() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = seed(dir.path(), "a.txt", "one two three");

    // search/replacement variants; a stray boolean `replace` flag must not be
    // mistaken for the replacement text.
    run_edit(
        json!({
            "path": path,
            "search": "two",
            "replacement": "2",
            "replace": true
        }),
        dir.path(),
    )
    .await
    .expect("variant names should work");

    assert_eq!(read(&path), "one 2 three");
}

#[tokio::test]
async fn failure_messages_stay_small_for_small_context_models() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    // A big file and a long old_string: the not-found failure must stay
    // bounded (snippet capped at 3 lines / 120 chars) instead of echoing
    // either.
    let original = "filler\n".repeat(5_000);
    let path = seed(dir.path(), "big.txt", &original);
    let long_needle = format!("{}\nmore\nlines\nhere", "y".repeat(400));

    let err = run_edit_fail(
        json!({ "file_path": path, "old_string": long_needle, "new_string": "z" }),
        dir.path(),
    )
    .await;

    assert_eq!(read(&path), original, "file must be untouched");
    // The structured message is bounded to well under ~400 chars plus the
    // (caller-controlled) path — no engine prefix stack on this path.
    assert!(
        err.len() < 600,
        "failure message too large for a 32k-window model ({} chars): {err}",
        err.len()
    );
}

// "Diff presentation": a successful edit carries a bounded before/after view in
// `data.diff`, so the timeline can show what changed while the user was away.

#[tokio::test]
async fn a_successful_edit_reports_a_before_after_diff() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = seed(dir.path(), "a.txt", "line one\nline two\nline three\n");

    let result = run_edit(
        json!({ "file_path": path, "old_string": "line two", "new_string": "line 2" }),
        dir.path(),
    )
    .await
    .expect("edit should succeed");

    let diff = &result["data"]["diff"];
    assert_eq!(diff["start_line"], json!(2), "got: {result}");
    assert_eq!(diff["removed"], json!(["line two"]));
    assert_eq!(diff["added"], json!(["line 2"]));
    assert_eq!(diff["truncated"], json!(false));
}

#[tokio::test]
async fn a_crlf_file_diffs_without_carriage_returns() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = seed(dir.path(), "w.txt", "alpha\r\nbeta\r\ngamma\r\n");

    let result = run_edit(
        json!({ "file_path": path, "old_string": "beta", "new_string": "BETA" }),
        dir.path(),
    )
    .await
    .expect("edit should succeed");

    assert_eq!(
        read(&path),
        "alpha\r\nBETA\r\ngamma\r\n",
        "line endings untouched"
    );
    let diff = &result["data"]["diff"];
    assert_eq!(diff["removed"], json!(["beta"]), "got: {result}");
    assert_eq!(diff["added"], json!(["BETA"]));
}

#[tokio::test]
async fn a_large_edit_is_capped_and_says_so() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let body = (0..100).fold(String::new(), |mut acc, i| {
        use std::fmt::Write as _;
        writeln!(acc, "old {i}").expect("a String never refuses a write");
        acc
    });
    let path = seed(dir.path(), "big.txt", &format!("head\n{body}tail\n"));

    let result = run_edit(
        json!({ "file_path": path, "old_string": body, "new_string": body.replace("old", "new") }),
        dir.path(),
    )
    .await
    .expect("edit should succeed");

    let diff = &result["data"]["diff"];
    assert_eq!(diff["start_line"], json!(2), "got: {result}");
    assert_eq!(
        diff["removed"].as_array().map(Vec::len),
        Some(40),
        "capped per side"
    );
    assert_eq!(diff["truncated"], json!(true), "a capped view must say so");
}
