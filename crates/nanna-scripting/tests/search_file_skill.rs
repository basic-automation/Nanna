#![warn(clippy::pedantic, clippy::nursery, clippy::all)]
//! Behavioral tests for the `search_file` default skill, executed for real
//! through the Boa engine with a bridge scoped to a temp directory.
//!
//! The contract under test is the same failure class fixed in `code_search`:
//! an unbounded `content.split("\n")` that blows the 30s script deadline and
//! returns NOTHING. `search_file` accepted files up to 10 MB and split them
//! unconditionally; measured through this bridge, a 1 MiB file cost 184s in
//! that one line.
//!
//! - a whole-file regex prefilter keeps non-matching files out of the per-line
//!   pass, built with "m" so anchored patterns are not silently dropped;
//! - the split is done in bounded slices, which turns Boa's
//!   O(lines x length) split into O(lines x slice) and makes the walk
//!   interruptible;
//! - wall-clock against the real script deadline is the primary bound, with
//!   derived caps on read size, walked size, walked lines and result chars —
//!   and every one announces itself in the output when it trips;
//! - a bad parameter, a bad regex, a directory, a binary file or an
//!   inaccessible path returns a structured `success: false` result instead of
//!   throwing.
//!
//! Tolerant by design (same as `default_skills_params.rs`): if the sibling
//! `nanna-tools/default-skills` tree isn't present, the tests no-op.

#![cfg(feature = "boa")]

mod common;

use nanna_scripting::{ScriptEngine, ScriptedTool, ToolPermissions};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

fn skill_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../nanna-tools/default-skills/search_file/tool.ts")
}

/// Execute the real `search_file` tool.ts against `input`, sandboxed to `dir`.
async fn run_search(input: Value, dir: &Path) -> Result<Value, String> {
    let tool = ScriptedTool::from_file(skill_path())
        .expect("read search_file tool.ts")
        .with_permissions(ToolPermissions::none().with_read([dir]))
        // Scaffolding, not an assertion — see `common::FIXTURE_TIMEOUT_MS`.
        .with_timeout(common::FIXTURE_TIMEOUT_MS);
    ScriptEngine::new()
        .execute(&tool, input, None, None)
        .await
        .map(|r| r.value)
        .map_err(|e| e.to_string())
}

/// Run and expect success; returns the result content string.
async fn run_search_ok(input: Value, dir: &Path) -> String {
    let result = run_search(input, dir).await.expect("search should succeed");
    assert_eq!(
        result["success"],
        Value::Bool(true),
        "expected success:true, got: {result}"
    );
    result["content"]
        .as_str()
        .expect("content string")
        .to_string()
}

/// Run and expect a structured failure; returns the result content string.
async fn run_search_fail(input: Value, dir: &Path) -> String {
    let result = run_search(input, dir)
        .await
        .expect("failures are returned, not thrown");
    assert_eq!(
        result["success"],
        Value::Bool(false),
        "expected success:false, got: {result}"
    );
    result["content"]
        .as_str()
        .expect("content string")
        .to_string()
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

fn seed(dir: &Path, name: &str, content: &str) -> PathBuf {
    let path = dir.join(name);
    std::fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
    std::fs::write(&path, content).expect("seed file");
    path
}

fn tail(s: &str) -> &str {
    &s[s.len().saturating_sub(800)..]
}

const DEADLINE: std::time::Duration = std::time::Duration::from_secs(30);

#[tokio::test]
async fn finds_matches_with_line_numbers_and_context() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let f = seed(
        dir.path(),
        "a.rs",
        "one\ntwo\nthe needle here\nfour\nfive\n",
    );

    let content = run_search_ok(
        json!({ "file_path": f.to_string_lossy(), "pattern": "needle" }),
        dir.path(),
    )
    .await;

    assert!(content.contains("Found 1 match"), "got: {content}");
    // The match line is marked and numbered; context lines come with it.
    assert!(content.contains(" > 3 | the needle here"), "got: {content}");
    assert!(content.contains("   2 | two"), "got: {content}");
}

/// The whole-file prefilter that keeps non-matching files out of the expensive
/// per-line pass must never drop a file the per-line pass would have matched.
/// Anchored patterns are the case that catches it: `^needle` matches line 3 of
/// this file, but against the whole file text `^` would anchor to byte 0 —
/// which is exactly the bug caught and fixed in `code_search`. The prefilter is
/// therefore built with "m".
#[tokio::test]
async fn anchored_patterns_still_match_mid_file() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let f = seed(dir.path(), "a.rs", "one\ntwo\nneedle at line start\nfour\n");

    let start = run_search_ok(
        json!({ "file_path": f.to_string_lossy(), "pattern": "^needle" }),
        dir.path(),
    )
    .await;
    assert!(start.contains("Found 1 match"), "got: {start}");
    assert!(start.contains(" > 3 | needle"), "got: {start}");

    let end = run_search_ok(
        json!({ "file_path": f.to_string_lossy(), "pattern": "start$" }),
        dir.path(),
    )
    .await;
    assert!(end.contains("Found 1 match"), "got: {end}");
    assert!(end.contains(" > 3 | needle"), "got: {end}");
}

/// `$` matches at the end of a CRLF line too: split on `\n`, each line kept
/// its `\r`, and without the `m` flag `$` does not match before it.
#[tokio::test]
async fn end_anchored_patterns_match_in_a_crlf_file() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let f = seed(dir.path(), "a.rs", "fn main() {\r\n    run();\r\n}\r\n");

    let found = run_search_ok(
        json!({ "file_path": f.to_string_lossy(), "pattern": "\\{$" }),
        dir.path(),
    )
    .await;
    assert!(found.contains("Found 1 match"), "got: {found}");
    assert!(found.contains(" > 1 | fn main() {"), "got: {found}");
    assert!(
        !found.contains('\r'),
        "no carriage return is rendered: {found:?}"
    );
}

/// A match that falls inside an earlier match's context is still marked.
#[tokio::test]
async fn a_match_inside_another_matchs_context_is_marked() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let body = (1..=20).map(|n| format!("row {n}\n")).collect::<Vec<_>>().concat();
    let f = seed(
        dir.path(),
        "a.txt",
        &body
            .replace("row 10\n", "hit 10\n")
            .replace("row 12\n", "hit 12\n"),
    );

    let found = run_search_ok(
        json!({ "file_path": f.to_string_lossy(), "pattern": "^hit", "context_lines": 3 }),
        dir.path(),
    )
    .await;
    assert!(found.contains(" > 10 | hit 10"), "got: {found}");
    assert!(found.contains(" > 12 | hit 12"), "got: {found}");
}

#[tokio::test]
async fn no_matches_is_an_observation_not_an_error() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let f = seed(dir.path(), "a.rs", "fn main() {}\n");

    let content = run_search_ok(
        json!({ "file_path": f.to_string_lossy(), "pattern": "zzz_absent_zzz" }),
        dir.path(),
    )
    .await;

    assert!(content.contains("No matches"), "got: {content}");
    // A complete negative must say it is complete — otherwise "no matches"
    // and "gave up early" are indistinguishable to the model.
    assert!(
        content.contains("SUCCEEDED and is complete"),
        "got: {content}"
    );
}

/// The headline bound. A file over the walk budget that DOES contain the
/// pattern comes back with a bounded, self-announcing answer well inside the
/// script deadline, instead of spending minutes in one split and returning
/// nothing. The fact of the match survives; only its location is bounded away.
#[tokio::test]
async fn oversized_match_is_bounded_self_announcing_and_inside_the_deadline() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    // Over MAX_WALK_CHARS (1 MiB) but under MAX_READ_BYTES (10 MB).
    let mut big = String::from("the needle is here\n");
    while big.len() < 1_500_000 {
        big.push_str("padding line that does not match\n");
    }
    let f = seed(dir.path(), "huge.rs", &big);

    let started = std::time::Instant::now();
    let content = run_search_ok(
        json!({ "file_path": f.to_string_lossy(), "pattern": "needle" }),
        dir.path(),
    )
    .await;
    let elapsed = started.elapsed();

    assert!(
        !content.contains("No matches"),
        "a file that contains the pattern must not be reported as no-match: {content}"
    );
    assert!(content.contains("CONTAINS"), "got: {content}");
    assert!(
        content.contains("larger than the 1.0MB line-context budget"),
        "the skip must name its bound: {content}"
    );
    assert!(
        content.contains("SUCCEEDED"),
        "a budgeted stop must not read as a failure: {content}"
    );
    assert!(content.contains("rg -n"), "must offer a way through: {content}");
    assert!(
        elapsed < DEADLINE,
        "must return inside the script deadline, took {elapsed:?}"
    );
}

/// The prefilter answers a large NON-matching file definitively without ever
/// walking it — the common case, and the one that used to cost the most for
/// the least.
#[tokio::test]
async fn oversized_miss_is_answered_by_the_prefilter_inside_the_deadline() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let mut big = String::new();
    while big.len() < 2_000_000 {
        big.push_str("padding line that does not match\n");
    }
    let f = seed(dir.path(), "huge.rs", &big);

    let started = std::time::Instant::now();
    let content = run_search_ok(
        json!({ "file_path": f.to_string_lossy(), "pattern": "zzz_absent_zzz" }),
        dir.path(),
    )
    .await;
    let elapsed = started.elapsed();

    assert!(content.contains("No matches"), "got: {content}");
    assert!(
        content.contains("SUCCEEDED and is complete"),
        "a whole-file miss is a complete answer: {content}"
    );
    assert!(
        elapsed < DEADLINE,
        "must return inside the script deadline, took {elapsed:?}"
    );
}

/// The shape that breaks a byte-based cap: 128 KiB is small, but as 65,000
/// one-character lines it measured 84s in a single `split("\n")` — past the
/// deadline, for a file most tools would call tiny. The sliced walk has to
/// render this normally and quickly.
#[tokio::test]
async fn pathologically_short_lines_stay_inside_the_deadline() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let mut body = String::from("needle\n");
    for _ in 0..65_000 {
        body.push_str("a\n");
    }
    let f = seed(dir.path(), "short_lines.txt", &body);

    let started = std::time::Instant::now();
    let content = run_search_ok(
        json!({ "file_path": f.to_string_lossy(), "pattern": "needle" }),
        dir.path(),
    )
    .await;
    let elapsed = started.elapsed();

    assert!(content.contains("Found 1 match"), "got: {content}");
    assert!(content.contains("| needle"), "got: {content}");
    assert!(
        elapsed < DEADLINE,
        "65k short lines must beat the script deadline, took {elapsed:?}"
    );
}

#[tokio::test]
async fn max_results_stops_the_search_and_says_so() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let f = seed(dir.path(), "a.rs", &"needle\n".repeat(20));

    let content = run_search_ok(
        json!({ "file_path": f.to_string_lossy(), "pattern": "needle", "max_results": 3 }),
        dir.path(),
    )
    .await;

    assert!(content.contains("Found 3 matches"), "got: {content}");
    assert!(
        content.contains("Stopped at the requested max_results of 3"),
        "got: {content}"
    );
}

#[tokio::test]
async fn output_budget_trips_and_announces_itself() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    // Each match renders a 7-line section of ~2,000-char lines (~14,000
    // chars), so the second would pass the 16,000-char (5 memory chunk)
    // result budget. Blocks are 10 lines apart so the sections never merge.
    let pad = "p".repeat(2_000);
    let mut body = String::new();
    for _ in 0..4 {
        for j in 0..10 {
            if j == 4 {
                use std::fmt::Write as _;
                writeln!(body, "{pad} needle").expect("a String never refuses a write");
            } else {
                body.push_str(&pad);
                body.push('\n');
            }
        }
    }
    let f = seed(dir.path(), "wide.rs", &body);

    let content = run_search_ok(
        json!({ "file_path": f.to_string_lossy(), "pattern": "needle" }),
        dir.path(),
    )
    .await;

    assert!(
        content.contains("STOPPED: the next match section would pass the 16000-char result budget"),
        "got tail: {}",
        tail(&content)
    );
    assert!(
        content.contains("5 memory chunks"),
        "the budget must name the constraint it derives from: {}",
        tail(&content)
    );
    assert!(
        content.contains("SUCCEEDED"),
        "a budgeted stop must not read as a failure: {}",
        tail(&content)
    );
    assert!(
        content.contains("(PARTIAL"),
        "the header must flag the partial result: {}",
        &content[..200.min(content.len())]
    );
}

#[tokio::test]
async fn case_sensitivity_is_honored_on_both_paths() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let f = seed(dir.path(), "a.rs", "one\nNeedle\ntwo\n");

    let insensitive = run_search_ok(
        json!({ "file_path": f.to_string_lossy(), "pattern": "needle" }),
        dir.path(),
    )
    .await;
    assert!(insensitive.contains("Found 1 match"), "got: {insensitive}");

    // The prefilter must carry the same case flag as the matcher, or a
    // case-sensitive miss would be reported inconsistently.
    let sensitive = run_search_ok(
        json!({
            "file_path": f.to_string_lossy(),
            "pattern": "needle",
            "case_sensitive": true
        }),
        dir.path(),
    )
    .await;
    assert!(sensitive.contains("No matches"), "got: {sensitive}");
}

/// The prefilter matches whole text, the renderer matches single lines, so a
/// pattern spanning a line break is found by one and not the other. Reporting
/// that as "no matches" would contradict the prefilter; the skill has to say
/// what actually happened.
#[tokio::test]
async fn pattern_spanning_a_line_break_is_reported_not_denied() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let f = seed(dir.path(), "a.rs", "alpha\nbeta\ngamma\n");

    let content = run_search_ok(
        json!({ "file_path": f.to_string_lossy(), "pattern": "alpha\nbeta" }),
        dir.path(),
    )
    .await;

    assert!(
        !content.contains("No matches"),
        "the prefilter proved the text is there: {content}"
    );
    assert!(content.contains("CONTAINS"), "got: {content}");
    assert!(content.contains("no SINGLE"), "got: {content}");
    assert!(content.contains("SUCCEEDED"), "got: {content}");
}

/// An empty file has one line (`"".split("\n")` is `[""]`), and the sliced
/// walk has to agree with that — its loop body never runs, so the line is
/// appended afterwards and must still be scanned.
#[tokio::test]
async fn an_empty_file_is_searched_consistently() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let f = seed(dir.path(), "empty.rs", "");

    // A pattern that cannot match: an ordinary, complete miss.
    let miss = run_search_ok(
        json!({ "file_path": f.to_string_lossy(), "pattern": "needle" }),
        dir.path(),
    )
    .await;
    assert!(miss.contains("No matches"), "got: {miss}");
    assert!(miss.contains("SUCCEEDED and is complete"), "got: {miss}");

    // A pattern that matches the empty string matches that one empty line —
    // the prefilter says the file contains it, and the line scan must agree.
    let hit = run_search_ok(
        json!({ "file_path": f.to_string_lossy(), "pattern": "x*" }),
        dir.path(),
    )
    .await;
    assert!(
        hit.contains("Found 1 match"),
        "the line scan must agree with the prefilter: {hit}"
    );
}

#[tokio::test]
async fn missing_file_path_returns_structured_failure_not_a_throw() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();

    let content = run_search_fail(json!({ "pattern": "needle" }), dir.path()).await;
    assert!(content.contains("file_path"), "got: {content}");
    assert!(content.contains("Nothing was searched"), "got: {content}");
}

#[tokio::test]
async fn missing_pattern_returns_structured_failure_not_a_throw() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let f = seed(dir.path(), "a.rs", "fn main() {}\n");

    let content = run_search_fail(json!({ "file_path": f.to_string_lossy() }), dir.path()).await;
    assert!(content.contains("pattern"), "got: {content}");
    assert!(content.contains("Nothing was searched"), "got: {content}");
}

#[tokio::test]
async fn invalid_regex_returns_structured_failure_not_a_throw() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let f = seed(dir.path(), "a.rs", "fn main() {}\n");

    let content = run_search_fail(
        json!({ "file_path": f.to_string_lossy(), "pattern": "fn main(" }),
        dir.path(),
    )
    .await;
    assert!(content.contains("not a valid regex"), "got: {content}");
    assert!(content.contains("Nothing was searched"), "got: {content}");
}

#[tokio::test]
async fn missing_file_returns_structured_failure_not_a_throw() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let ghost = dir.path().join("no_such_file.rs");

    let content = run_search_fail(
        json!({ "file_path": ghost.to_string_lossy(), "pattern": "needle" }),
        dir.path(),
    )
    .await;
    assert!(content.contains("no_such_file.rs"), "got: {content}");
    assert!(content.contains("does not exist"), "got: {content}");
}

#[tokio::test]
async fn a_directory_returns_structured_failure_not_a_throw() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    seed(dir.path(), "sub/a.rs", "needle\n");

    let content = run_search_fail(
        json!({
            "file_path": dir.path().join("sub").to_string_lossy(),
            "pattern": "needle"
        }),
        dir.path(),
    )
    .await;
    assert!(content.contains("is a directory"), "got: {content}");
    assert!(content.contains("code_search"), "must point at the right tool: {content}");
}

#[tokio::test]
async fn binary_content_returns_structured_failure_not_a_throw() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();

    // NUL bytes are valid UTF-8, so this survives the read and is caught by
    // the NUL probe.
    let nul = dir.path().join("nul.bin");
    std::fs::write(&nul, b"\x00\x00needle\x00").expect("seed");
    let content = run_search_fail(
        json!({ "file_path": nul.to_string_lossy(), "pattern": "needle" }),
        dir.path(),
    )
    .await;
    assert!(content.contains("looks binary"), "got: {content}");

    // Invalid UTF-8 fails earlier, in read_to_string.
    let raw = dir.path().join("raw.bin");
    std::fs::write(&raw, b"\xff\xfe\x00needle").expect("seed");
    let content = run_search_fail(
        json!({ "file_path": raw.to_string_lossy(), "pattern": "needle" }),
        dir.path(),
    )
    .await;
    assert!(content.contains("could not be read as text"), "got: {content}");
}

#[tokio::test]
async fn file_over_the_read_budget_is_refused_with_a_way_through() {
    if skill_missing() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    // Over MAX_READ_BYTES (10 MB): never read, so never split.
    let mut big = String::with_capacity(11_000_000);
    while big.len() < 11_000_000 {
        big.push_str("needle padding line\n");
    }
    let f = seed(dir.path(), "enormous.rs", &big);

    let started = std::time::Instant::now();
    let content = run_search_fail(
        json!({ "file_path": f.to_string_lossy(), "pattern": "needle" }),
        dir.path(),
    )
    .await;
    let elapsed = started.elapsed();

    assert!(
        content.contains("over the 10.0MB read budget"),
        "the refusal must name its bound: {content}"
    );
    assert!(content.contains("Nothing was searched"), "got: {content}");
    assert!(content.contains("rg -n"), "must offer a way through: {content}");
    assert!(
        elapsed < DEADLINE,
        "a refusal must be immediate, took {elapsed:?}"
    );
}

/// Every other test seeds synthetic content. This one runs against a real
/// source file — the agent loop, the largest file in the repo — to check the
/// rendering holds up on ordinary code and that a real search is fast.
/// Ignored by default: wall-time on a shared box is not a stable assertion.
/// Run: cargo test -p nanna-scripting --features boa --test `search_file_skill` \
///        -- --ignored --nocapture `against_a_real`
#[tokio::test]
#[ignore = "manual check against a real repo file"]
async fn against_a_real_source_file() {
    if skill_missing() {
        return;
    }
    // Built by walking up, not with ".." segments: the bridge resolves the
    // path it is handed against the read permission literally, and a path
    // containing ".." does not compare as being under the root.
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf();
    let target = root.join("crates/nanna-agent/src/loop_runner.rs");
    if !target.is_file() {
        eprintln!("skipping: {} not present", target.display());
        return;
    }

    let start = std::time::Instant::now();
    let content = run_search_ok(
        json!({ "file_path": target.to_string_lossy(), "pattern": "^\\s*fn \\w+" }),
        &root,
    )
    .await;
    let elapsed = start.elapsed();

    println!("search_file({}) took {elapsed:?}", target.display());
    println!("--- output ({} bytes) ---\n{content}", content.len());
    assert!(
        elapsed < DEADLINE,
        "a real search must beat the script deadline, took {elapsed:?}"
    );
}

/// The measurements every budget in the skill is derived from. Ignored by
/// default — throughput on a shared box is not a stable assertion — but the
/// numbers quoted in the skill's budget comments came from exactly these runs,
/// and this is how to re-derive them when the engine or the machine changes.
///
/// Measured 2026-08-03, debug build, through the real Boa bridge.
///
/// Stage isolation over 32,768 one-char lines showed the split ITSELF is the
/// whole cost, not the per-line loop over its result:
///   split only 21.3s | split+touch 23.5s | split+regex 23.2s | empty loop 70ms
///
/// Cost is the PRODUCT of line count and string length (~1e-8 x lines x
/// chars s), so neither a byte cap nor a line cap alone can bound it:
///   128 KiB /    653 lines   0.91s      128 KiB /  2,428 lines   3.41s
///   128 KiB /  7,711 lines  10.29s      128 KiB / 26,215 lines  32.86s
///   128 KiB / 65,536 lines  83.96s        1 MiB / 19,419 lines 184.06s
///
/// Splitting in bounded slices breaks the product into O(lines x slice):
///   1 MiB / 19,419 lines:   one split 184.06s -> 4 KiB slices 3.05s (60x)
///   128 KiB / 65,536 lines: one split  82.53s -> 4 KiB slices 2.68s (31x)
///   128 KiB /  2,428 lines: one split   2.26s -> 4 KiB slices 0.13s (17x)
///
/// Run: cargo test -p nanna-scripting --features boa --test `search_file_skill` \
///        -- --ignored --nocapture `split_cost`
#[tokio::test]
#[ignore = "throughput measurement, not an assertion"]
async fn split_cost_model_and_the_slicing_that_breaks_it() {
    const ONE_SPLIT: &str = r#"
export default { name: "probe", execute: function(input) {
  var c = Nanna.readFile(String(input.path));
  return "lines=" + c.split("\n").length;
} }
"#;
    const SPLIT_THEN_REGEX: &str = r#"
export default { name: "probe", execute: function(input) {
  var c = Nanna.readFile(String(input.path));
  var re = new RegExp("zzz_no_such_token_zzz", "i");
  var lines = c.split("\n");
  var n = 0;
  for (var j = 0; j < lines.length; j++) { if (re.test(lines[j])) n++; }
  return "hits=" + n;
} }
"#;
    const EMPTY_LOOP: &str = r#"
export default { name: "probe", execute: function(input) {
  var n = 0;
  for (var j = 0; j < input.iters; j++) { n += 1; }
  return "n=" + n;
} }
"#;
    // The skill's own walk, in isolation.
    const SLICED_SPLIT: &str = r#"
export default { name: "probe", execute: function(input) {
  var c = Nanna.readFile(String(input.path));
  var SLICE = input.slice;
  var lines = [];
  var pos = 0;
  while (pos < c.length) {
    var end = pos + SLICE;
    if (end < c.length) {
      var nl = c.indexOf("\n", end);
      end = nl < 0 ? c.length : nl + 1;
    } else { end = c.length; }
    var part = c.slice(pos, end);
    pos = end;
    var got = part.split("\n");
    if (pos < c.length && got.length > 0 && got[got.length - 1] === "") got.pop();
    for (var i = 0; i < got.length; i++) lines.push(got[i]);
  }
  return "lines=" + lines.length;
} }
"#;

    async fn timed(src: &str, dir: &Path, arg: Value, label: &str) {
        let tool = ScriptedTool::new("probe.ts", src)
            .with_permissions(ToolPermissions::none().with_read([dir]))
            .with_timeout(600_000);
        let start = std::time::Instant::now();
        let out = ScriptEngine::new()
            .execute(&tool, arg, None, None)
            .await
            .expect("probe ran")
            .value;
        println!("{label}: {out} in {:?}", start.elapsed());
    }

    let dir = tempfile::tempdir().unwrap();
    let write_shape = |line_len: usize, target: usize| -> (PathBuf, usize, usize) {
        let line = "x".repeat(line_len);
        let mut body = String::new();
        while body.len() < target {
            body.push_str(&line);
            body.push('\n');
        }
        let n = body.matches('\n').count();
        let p = dir.path().join(format!("l{line_len}_{target}.txt"));
        std::fs::write(&p, &body).unwrap();
        (p, n, body.len())
    };

    // Where the cost lives.
    let (p, n, len) = write_shape(1, 64 * 1024);
    let arg = json!({ "path": p.to_string_lossy() });
    println!("--- stage isolation: {len}B, {n} lines ---");
    timed(ONE_SPLIT, dir.path(), arg.clone(), "  split only  ").await;
    timed(SPLIT_THEN_REGEX, dir.path(), arg.clone(), "  split+regex ").await;
    timed(EMPTY_LOOP, dir.path(), json!({ "iters": n }), "  empty loop  ").await;

    // The product model, and the slicing that breaks it.
    println!("--- one split vs 4 KiB slices ---");
    for &(line_len, target) in &[
        (53usize, 128 * 1024usize),
        (1, 128 * 1024),
        (53, 1024 * 1024),
    ] {
        let (p, n, len) = write_shape(line_len, target);
        let arg = json!({ "path": p.to_string_lossy() });
        let tag = format!("{len}B @ {line_len}c ({n} lines)");
        timed(ONE_SPLIT, dir.path(), arg.clone(), &format!("  one split      {tag}")).await;
        timed(
            SLICED_SPLIT,
            dir.path(),
            json!({ "path": p.to_string_lossy(), "slice": 4096 }),
            &format!("  4 KiB slices   {tag}"),
        )
        .await;
    }
}
