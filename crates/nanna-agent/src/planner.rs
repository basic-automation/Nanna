//! Plan-from-goal (P19): turn one natural-language request into the task
//! plan the long-horizon harness consumes.
//!
//! The harness ([`crate::harness`]) is a pure executor — it reads
//! [`crate::harness::TaskSource::next`] and stops the moment the store is
//! empty. Until now the store was filled by hand (the Tasks page) or by the
//! agent calling the `task` skill, which is why chat could not use the
//! harness: there was nothing to execute.
//!
//! This module closes that gap so **every** chat turn is a long-horizon run.
//! The design bet mirrors the harness's own: *the planner must never be the
//! reason a turn fails.* A model that returns prose instead of JSON, a model
//! that returns nothing, a model that is down — all of these degrade to the
//! single-task plan, which the harness executes as exactly one step. That
//! degenerate case is ordinary conversation, and it costs the same as the
//! pre-harness chat path did.
//!
//! Acceptance checks are emitted **only** when the request names a
//! machine-checkable outcome. "Explain borrow checking" has no verdict the
//! environment can render, and inventing one would manufacture false
//! `false_success_claims`. `acceptance: None` is a first-class answer — the
//! harness already reports those completions as `items_completed_unverified`.

use serde::{Deserialize, Serialize};

/// Maximum tasks admitted from one plan.
///
/// Bound justification: the plan is re-read from the store on every step, and
/// `build_step_prompt` pins the goal verbatim, so a long plan does not grow
/// the window — but it does grow the *time to first visible work*, and a
/// model asked for "a plan" will happily emit fifty bullet points. Twelve is
/// the largest decomposition observed to stay coherent on the reference
/// 9B tier in the P14 benchmark (v8 notekeeper: 11 stages); past that the
/// items stop being independently checkable. Replanning adds depth later, so
/// this caps breadth per planning call, not total plan size.
pub const PLAN_TASKS_MAX: usize = 12;

/// Maximum bytes of a task title.
///
/// Bound justification: the title is echoed into every step prompt for its
/// item and into the activity log. One line of a terminal at the repo's
/// 14px default is ~120 chars; 200 bytes covers that with UTF-8 headroom
/// without letting a model inline an essay as a "title".
pub const PLAN_TITLE_MAX_BYTES: usize = 200;

/// Maximum bytes of a task description.
///
/// Bound justification: the description rides in the step prompt's dynamic
/// tail alongside the goal, last result (`STEP_RESULT_TAIL_MAX_BYTES` = 2000)
/// and budget line. Matching that 2000-byte tail keeps the two largest
/// variable parts of the prompt symmetric, so a step prompt stays O(1) in
/// plan size.
pub const PLAN_DESCRIPTION_MAX_BYTES: usize = 2000;

/// Maximum bytes of the goal echoed into the planning prompt.
///
/// Bound justification: the goal is pinned verbatim and never summarized
/// (the harness's rule — intent is not compressible). This bound exists only
/// to stop a pasted logfile from becoming the planning prompt; 8 KiB is well
/// past any typed request and still a small fraction of the smallest
/// supported context window.
pub const PLAN_GOAL_MAX_BYTES: usize = 8 * 1024;

// ---------------------------------------------------------------------------
// Plan shape
// ---------------------------------------------------------------------------

/// One planned task, in the shape [`crate::harness::NewSubtask`] and the P15
/// store both accept.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlannedTask {
    pub title: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Raw acceptance JSON, validated against
    /// [`crate::harness::AcceptanceCheck`] before it is admitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acceptance: Option<serde_json::Value>,
    /// Tools this task needs. Kept short — small models degrade past ~5-10
    /// tool definitions, which is why the harness scopes tools per item.
    #[serde(default)]
    pub tool_scope: Vec<String>,
}

/// The result of planning, including how it was reached.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    pub tasks: Vec<PlannedTask>,
    pub origin: PlanOrigin,
}

/// How a plan was produced — recorded so an unplanned turn is visible rather
/// than silently indistinguishable from a planned one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanOrigin {
    /// The model returned a usable plan.
    Model,
    /// The model's output was unusable (unparseable, empty, or every task
    /// rejected) — the goal became one task.
    Fallback,
}

impl Plan {
    /// The one-task plan. Ordinary conversation and every planner failure
    /// land here, and the harness runs it as a single step.
    #[must_use]
    pub fn single(goal: &str) -> Self {
        Self {
            tasks: vec![PlannedTask {
                title: title_from_goal(goal),
                description: Some(clamp_bytes(goal, PLAN_DESCRIPTION_MAX_BYTES)),
                acceptance: None,
                tool_scope: Vec::new(),
            }],
            origin: PlanOrigin::Fallback,
        }
    }

    /// True when this is the degenerate single-step plan.
    #[must_use]
    pub const fn is_single_step(&self) -> bool {
        self.tasks.len() == 1
    }
}

/// Derive a task title from a free-text goal: first line, first sentence,
/// clamped. Used for the fallback plan and never shown in place of a
/// model-authored title.
#[must_use]
pub fn title_from_goal(goal: &str) -> String {
    let first_line = goal.lines().find(|l| !l.trim().is_empty()).unwrap_or("");
    let trimmed = first_line.trim();
    // Prefer a sentence boundary when one falls inside the budget.
    let candidate = trimmed
        .split_inclusive(['.', '?', '!'])
        .next()
        .unwrap_or(trimmed)
        .trim();
    let picked = if candidate.is_empty() { trimmed } else { candidate };
    if picked.is_empty() {
        return "Respond to the user".to_string();
    }
    clamp_bytes(picked, PLAN_TITLE_MAX_BYTES)
}

/// Truncate on a UTF-8 boundary, never mid-codepoint.
#[must_use]
pub fn clamp_bytes(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_string();
    }
    let mut end = max;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    text[..end].trim_end().to_string()
}

// ---------------------------------------------------------------------------
// Prompt
// ---------------------------------------------------------------------------

/// Build the planning prompt.
///
/// Deliberately asks for *one* JSON array and nothing else. The instruction
/// to prefer a single task is load-bearing: without it, small models
/// decompose "what time is it" into four steps.
#[must_use]
pub fn build_plan_prompt(goal: &str, context: Option<&str>) -> String {
    let mut prompt = String::with_capacity(2048);
    prompt.push_str(
        "You are planning how to satisfy one request. Reply with ONE JSON array and no other text.\n\
         \n\
         Each element: {\"title\": string, \"description\": string, \"acceptance\": object|null}\n\
         \n\
         RULES\n\
         1. Prefer ONE task. Emit multiple tasks only when the request truly has\n\
            separate, independently finishable parts. Conversation, questions,\n\
            explanations, and single edits are ALWAYS one task.\n\
         2. `acceptance` is a machine-run done-condition. Use null unless the\n\
            environment can actually render a verdict. Never invent one for a\n\
            question or an explanation.\n\
            Allowed forms:\n\
              {\"kind\":\"command\",\"command\":\"cargo test\",\"timeout_secs\":120}\n\
              {\"kind\":\"file_exists\",\"path\":\"src/foo.rs\"}\n\
              {\"kind\":\"regex\",\"pattern\":\"ok\",\"path\":\"out.txt\"}\n\
         3. Do NOT list tools for a task. Every step reaches whatever the work\n\
            turns out to need; planning tools ahead only ever guessed wrong.\n\
         4. Titles are imperative and specific. No numbering, no markdown.\n\
         \n",
    );
    if let Some(context) = context.filter(|c| !c.trim().is_empty()) {
        prompt.push_str("== CONTEXT ==\n");
        prompt.push_str(&clamp_bytes(context, PLAN_GOAL_MAX_BYTES));
        prompt.push_str("\n\n");
    }
    prompt.push_str("== REQUEST ==\n");
    prompt.push_str(&clamp_bytes(goal, PLAN_GOAL_MAX_BYTES));
    prompt.push_str("\n\nJSON array:");
    prompt
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// Parse a planner response into tasks.
///
/// Tolerant by design — this runs on every chat turn against whatever model
/// is configured, including small local ones. Accepts a bare array, a fenced
/// block, an object wrapping `tasks`/`plan`/`steps`, or a single bare task
/// object. Returns `Err` only when nothing task-shaped is present; callers
/// are expected to fall back rather than surface the error.
///
/// # Errors
///
/// Returns a human-readable reason when:
/// - the text contains no balanced JSON array or object;
/// - the first such span does not parse as JSON;
/// - it is an object with no `tasks`/`plan`/`steps` array and no `title`;
/// - it is neither an array nor an object;
/// - no element of the plan normalizes into a task with a non-empty title.
pub fn parse_plan(text: &str) -> Result<Vec<PlannedTask>, String> {
    let candidate = extract_json(text).ok_or_else(|| "no JSON found in planner output".to_string())?;
    let value: serde_json::Value =
        serde_json::from_str(&candidate).map_err(|e| format!("planner output is not valid JSON: {e}"))?;

    let array = match value {
        serde_json::Value::Array(items) => items,
        serde_json::Value::Object(ref map) => map
            .get("tasks")
            .or_else(|| map.get("plan"))
            .or_else(|| map.get("steps"))
            .and_then(|v| v.as_array().cloned())
            // A single bare task object is a plan of one.
            .or_else(|| map.contains_key("title").then(|| vec![value.clone()]))
            .ok_or_else(|| "planner object has no tasks array".to_string())?,
        _ => return Err("planner output is neither an array nor an object".to_string()),
    };

    let mut tasks = Vec::new();
    for item in array {
        if tasks.len() >= PLAN_TASKS_MAX {
            break;
        }
        if let Some(task) = normalize_task(&item) {
            tasks.push(task);
        }
    }
    if tasks.is_empty() {
        return Err("planner produced no usable tasks".to_string());
    }
    Ok(tasks)
}

/// Coerce one plan element into a [`PlannedTask`], dropping what cannot be
/// salvaged. A bare string is accepted as a title — small models emit
/// `["do the thing"]` often enough to be worth handling.
fn normalize_task(value: &serde_json::Value) -> Option<PlannedTask> {
    if let Some(title) = value.as_str() {
        let title = title.trim();
        if title.is_empty() {
            return None;
        }
        return Some(PlannedTask {
            title: clamp_bytes(title, PLAN_TITLE_MAX_BYTES),
            description: None,
            acceptance: None,
            tool_scope: Vec::new(),
        });
    }

    let map = value.as_object()?;
    let title = map
        .get("title")
        .or_else(|| map.get("name"))
        .or_else(|| map.get("task"))
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|t| !t.is_empty())?;

    let description = map
        .get("description")
        .or_else(|| map.get("detail"))
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|d| !d.is_empty())
        .map(|d| clamp_bytes(d, PLAN_DESCRIPTION_MAX_BYTES));

    // Acceptance is admitted only if it parses as a real check, and what is
    // admitted is the CANONICAL object, never the raw value: a check the
    // parser tolerates (e.g. the object handed over as a string) but the store
    // rejects would make `create` drop the whole task. A malformed one is
    // dropped, not passed through — an unparseable check would fail every
    // verdict and grind the item to abandonment.
    let acceptance = map
        .get("acceptance")
        .filter(|v| !v.is_null())
        .and_then(|v| crate::harness::AcceptanceCheck::canonicalize(v).ok());

    let tool_scope = map
        .get("tool_scope")
        .or_else(|| map.get("tools"))
        .and_then(|v| v.as_array())
        .map_or_default(|items| {
            items
                .iter()
                .filter_map(|t| t.as_str())
                .map(str::trim)
                .filter(|t| !t.is_empty())
                .map(String::from)
                .collect()
        });

    Some(PlannedTask {
        title: clamp_bytes(title, PLAN_TITLE_MAX_BYTES),
        description,
        acceptance,
        tool_scope,
    })
}

/// Pull the first JSON array or object out of arbitrary model prose.
///
/// Scans for the first `[` or `{` and matches to its partner, respecting
/// string literals and escapes so a brace inside a description cannot end
/// the scan early.
fn extract_json(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let start = bytes.iter().position(|b| *b == b'[' || *b == b'{')?;
    let open = bytes[start];
    let close = if open == b'[' { b']' } else { b'}' };

    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (offset, byte) in bytes[start..].iter().enumerate() {
        if in_string {
            if escaped {
                escaped = false;
            } else if *byte == b'\\' {
                escaped = true;
            } else if *byte == b'"' {
                in_string = false;
            }
            continue;
        }
        match *byte {
            b'"' => in_string = true,
            b if b == open => depth += 1,
            b if b == close => {
                depth -= 1;
                if depth == 0 {
                    let end = start + offset + 1;
                    return text.get(start..end).map(String::from);
                }
            }
            _ => {}
        }
    }
    None
}

/// Parse a planner response, falling back to the single-task plan.
///
/// This is the entry point callers should use: it cannot fail, so a planner
/// hiccup degrades a turn to ordinary one-step chat instead of breaking it.
#[must_use]
pub fn plan_or_fallback(goal: &str, planner_output: &str) -> Plan {
    parse_plan(planner_output).map_or_else(
        |_| Plan::single(goal),
        |tasks| Plan {
            tasks,
            origin: PlanOrigin::Model,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_bare_array() {
        let out = r#"[{"title":"Write the file","description":"make it","acceptance":null,"tool_scope":["write_file"]}]"#;
        let tasks = parse_plan(out).expect("parses");
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].title, "Write the file");
        assert_eq!(tasks[0].tool_scope, vec!["write_file".to_string()]);
    }

    #[test]
    fn parses_fenced_block_with_prose_around_it() {
        let out = "Sure! Here is the plan:\n```json\n[{\"title\":\"Do it\"}]\n```\nHope that helps.";
        let tasks = parse_plan(out).expect("parses");
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].title, "Do it");
    }

    #[test]
    fn parses_object_wrapper() {
        for key in ["tasks", "plan", "steps"] {
            let out = format!(r#"{{"{key}":[{{"title":"A"}},{{"title":"B"}}]}}"#);
            let tasks = parse_plan(&out).expect("parses");
            assert_eq!(tasks.len(), 2, "key {key}");
        }
    }

    #[test]
    fn parses_single_bare_task_object() {
        let tasks = parse_plan(r#"{"title":"Just one"}"#).expect("parses");
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].title, "Just one");
    }

    #[test]
    fn parses_bare_strings_as_titles() {
        let tasks = parse_plan(r#"["first thing","second thing"]"#).expect("parses");
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[1].title, "second thing");
    }

    #[test]
    fn brace_inside_a_description_does_not_end_the_scan() {
        let out = r#"[{"title":"Fix","description":"the map {a: 1} broke"},{"title":"Second"}]"#;
        let tasks = parse_plan(out).expect("parses");
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[1].title, "Second");
    }

    #[test]
    fn escaped_quote_inside_a_string_does_not_end_the_scan() {
        let out = r#"[{"title":"Say \"hi\"","description":"x"},{"title":"Second"}]"#;
        let tasks = parse_plan(out).expect("parses");
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0].title, r#"Say "hi""#);
    }

    #[test]
    fn valid_acceptance_is_kept() {
        let out = r#"[{"title":"Test","acceptance":{"kind":"command","command":"cargo test"}}]"#;
        let tasks = parse_plan(out).expect("parses");
        assert!(tasks[0].acceptance.is_some());
    }

    #[test]
    fn malformed_acceptance_is_dropped_not_passed_through() {
        // An unparseable check would fail every verdict and grind the item
        // to abandonment — dropping it completes unverified instead.
        let out = r#"[{"title":"Test","acceptance":{"kind":"vibes","feeling":"good"}}]"#;
        let tasks = parse_plan(out).expect("parses");
        assert!(tasks[0].acceptance.is_none());
    }

    /// A check that parses but the store would refuse is dropped from the
    /// task, not left to make `create` drop the whole task.
    #[test]
    fn a_check_the_store_would_refuse_is_dropped_and_the_task_kept() {
        for check in [
            r#"{"kind":"file_exists","path":""}"#,
            r#"{"kind":"command","command":""}"#,
            r#"{"kind":"regex","pattern":"ok"}"#,
        ] {
            let out = format!(r#"[{{"title":"Run tests","acceptance":{check}}}]"#);
            let tasks = parse_plan(&out).expect("parses");
            assert_eq!(tasks.len(), 1, "{check}");
            assert!(tasks[0].acceptance.is_none(), "{check}");
        }
        let good = r#"[{"title":"Write it","acceptance":{"kind":"file_exists","path":"out.md"}}]"#;
        assert!(parse_plan(good).expect("parses")[0].acceptance.is_some());
    }

    #[test]
    fn tasks_are_capped_at_the_bound() {
        let items: Vec<String> = (0..40).map(|i| format!(r#"{{"title":"t{i}"}}"#)).collect();
        let out = format!("[{}]", items.join(","));
        let tasks = parse_plan(&out).expect("parses");
        assert_eq!(tasks.len(), PLAN_TASKS_MAX);
    }

    #[test]
    fn oversized_title_and_description_are_clamped_on_char_boundaries() {
        let long_title = "é".repeat(PLAN_TITLE_MAX_BYTES);
        let long_desc = "ø".repeat(PLAN_DESCRIPTION_MAX_BYTES);
        let out = serde_json::json!([{ "title": long_title, "description": long_desc }]).to_string();
        let tasks = parse_plan(&out).expect("parses");
        assert!(tasks[0].title.len() <= PLAN_TITLE_MAX_BYTES);
        assert!(tasks[0].description.as_ref().unwrap().len() <= PLAN_DESCRIPTION_MAX_BYTES);
    }

    #[test]
    fn entries_without_a_title_are_skipped() {
        let out = r#"[{"description":"orphan"},{"title":"kept"},{"title":"   "}]"#;
        let tasks = parse_plan(out).expect("parses");
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].title, "kept");
    }

    #[test]
    fn prose_only_output_is_an_error() {
        assert!(parse_plan("I think we should start by reading the file.").is_err());
    }

    #[test]
    fn empty_array_is_an_error() {
        assert!(parse_plan("[]").is_err());
    }

    // ── Fallback: the planner must never break a turn ──

    #[test]
    fn fallback_on_prose_yields_one_task_carrying_the_goal() {
        let plan = plan_or_fallback("explain the borrow checker", "no json here");
        assert_eq!(plan.origin, PlanOrigin::Fallback);
        assert!(plan.is_single_step());
        assert_eq!(plan.tasks[0].title, "explain the borrow checker");
        assert!(plan.tasks[0].acceptance.is_none());
    }

    #[test]
    fn fallback_on_empty_output() {
        let plan = plan_or_fallback("hi", "");
        assert!(plan.is_single_step());
        assert_eq!(plan.origin, PlanOrigin::Fallback);
    }

    #[test]
    fn model_plan_is_marked_as_such() {
        let plan = plan_or_fallback("goal", r#"[{"title":"A"},{"title":"B"}]"#);
        assert_eq!(plan.origin, PlanOrigin::Model);
        assert!(!plan.is_single_step());
    }

    // ── Title derivation ──

    #[test]
    fn title_takes_the_first_sentence() {
        assert_eq!(
            title_from_goal("Fix the parser. Then ship it."),
            "Fix the parser."
        );
    }

    #[test]
    fn title_skips_leading_blank_lines() {
        assert_eq!(title_from_goal("\n\n  hello there  "), "hello there");
    }

    #[test]
    fn title_of_empty_goal_is_never_empty() {
        assert_eq!(title_from_goal("   "), "Respond to the user");
        assert_eq!(title_from_goal(""), "Respond to the user");
    }

    #[test]
    fn title_clamps_a_runaway_first_line() {
        let goal = "x".repeat(PLAN_TITLE_MAX_BYTES * 3);
        assert!(title_from_goal(&goal).len() <= PLAN_TITLE_MAX_BYTES);
    }

    // ── Prompt ──

    #[test]
    fn prompt_pins_the_goal_and_asks_for_one_task() {
        let prompt = build_plan_prompt("do the thing", None);
        assert!(prompt.contains("do the thing"));
        assert!(prompt.contains("Prefer ONE task"));
    }

    #[test]
    fn prompt_includes_context_when_given_and_omits_it_when_blank() {
        let with = build_plan_prompt("g", Some("prior turn"));
        assert!(with.contains("== CONTEXT =="));
        assert!(with.contains("prior turn"));
        assert!(!build_plan_prompt("g", Some("   ")).contains("== CONTEXT =="));
        assert!(!build_plan_prompt("g", None).contains("== CONTEXT =="));
    }

    #[test]
    fn prompt_clamps_a_pasted_logfile() {
        let goal = "L".repeat(PLAN_GOAL_MAX_BYTES * 2);
        let prompt = build_plan_prompt(&goal, None);
        assert!(prompt.len() < PLAN_GOAL_MAX_BYTES * 2);
    }
}
