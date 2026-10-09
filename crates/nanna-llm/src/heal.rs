//! Response healing for malformed LLM JSON.
//!
//! Free-form model output often wraps JSON in markdown fences, drops commas,
//! or concatenates objects. Call sites (chat tool args, embeddings HTTP bodies,
//! summarization / extraction) share this heuristic salvage path.

use serde_json::Value;

/// Strip common wrappers (markdown fences, leading prose) and salvage a JSON value.
///
/// Order of attempts:
/// 1. Direct parse of the trimmed input
/// 2. Strip ``` / ```json fences, retry
/// 3. Extract the first balanced `{...}` or `[...]` span, retry
/// 4. Apply cheap repairs (trailing commas, single quotes, bare keys), retry
///
/// Every attempt is lenient about bare control characters inside string
/// literals (a literal tab in TSV content — see
/// [`escape_bare_controls_in_strings`]): strict parse first, one escaped
/// retry on failure.
#[must_use]
pub fn heal_json(input: &str) -> Option<Value> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return None;
    }

    if let Some(v) = parse_lenient(trimmed) {
        return Some(v);
    }

    let unfenced = strip_code_fence(trimmed);
    if unfenced != trimmed
        && let Some(v) = parse_lenient(unfenced)
    {
        return Some(v);
    }

    if let Some(span) = extract_json_span(unfenced) {
        if let Some(v) = parse_lenient(span) {
            return Some(v);
        }
        let repaired = repair_common(span);
        if let Some(v) = parse_lenient(&repaired) {
            return Some(v);
        }
    }

    let repaired = repair_common(unfenced);
    parse_lenient(&repaired)
}

/// Strict parse, then — only if the input carries bare control characters
/// inside string literals — one retry with those escaped. The escaped retry
/// can only ever ACCEPT input the strict parse rejected for exactly that
/// reason; structurally malformed input fails both.
fn parse_lenient(s: &str) -> Option<Value> {
    if let Ok(v) = serde_json::from_str::<Value>(s) {
        return Some(v);
    }
    escape_bare_controls_in_strings(s)
        .and_then(|escaped| serde_json::from_str::<Value>(&escaped).ok())
}

/// Escape bare ASCII control characters (U+0000..U+001F) found inside JSON
/// string literals, so spec-strict parsers accept them.
///
/// Local models emitting TSV/multiline content write a literal tab or newline
/// inside a JSON string instead of `\t`/`\n`. That is spec-invalid — serde
/// rejects the whole call — but the intent is unambiguous: a bare control
/// character in a non-escape position inside a string can only mean that
/// character. The transform is:
///
/// - **bounded**: one pass; output ≤ 6× input bytes (each replaced character
///   becomes at most a 6-byte `\u00XX` escape);
/// - **lossless**: each replaced character becomes the JSON escape that
///   decodes back to exactly that character, and every byte outside string
///   literals is copied verbatim — provable by round-trip (parse the result,
///   the decoded strings equal the raw originals);
/// - **conservative**: a control character right after a backslash (an
///   already-invalid escape like `\<TAB>`) is ambiguous and left alone, and
///   `None` is returned when nothing was rewritten so callers skip the
///   redundant re-parse.
#[must_use]
pub fn escape_bare_controls_in_strings(s: &str) -> Option<String> {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(s.len() + 8);
    let mut changed = false;
    let mut in_string = false;
    let mut escaped = false;
    for c in s.chars() {
        if !in_string {
            if c == '"' {
                in_string = true;
            }
            out.push(c);
            continue;
        }
        if escaped {
            escaped = false;
            out.push(c);
            continue;
        }
        match c {
            '\\' => {
                escaped = true;
                out.push(c);
            }
            '"' => {
                in_string = false;
                out.push(c);
            }
            '\t' => {
                changed = true;
                out.push_str("\\t");
            }
            '\n' => {
                changed = true;
                out.push_str("\\n");
            }
            '\r' => {
                changed = true;
                out.push_str("\\r");
            }
            c if (c as u32) < 0x20 => {
                changed = true;
                // All remaining controls are < 0x20, so the escape is always
                // `\u00XX` — two hex digits suffice.
                let b = c as u32;
                out.push_str("\\u00");
                out.push(HEX[(b >> 4) as usize] as char);
                out.push(HEX[(b & 0xf) as usize] as char);
            }
            c => out.push(c),
        }
    }
    changed.then_some(out)
}

/// Like [`heal_json`], but typed.
#[must_use]
pub fn heal_json_as<T: serde::de::DeserializeOwned>(input: &str) -> Option<T> {
    heal_json(input).and_then(|v| serde_json::from_value(v).ok())
}

/// Heal a free-form model response expected to be a JSON object of tool args.
#[must_use]
pub fn heal_tool_args(input: &str) -> Value {
    match heal_json(input) {
        Some(Value::Object(map)) => Value::Object(map),
        Some(other) => serde_json::json!({ "value": other }),
        None => Value::Object(serde_json::Map::new()),
    }
}

fn strip_code_fence(s: &str) -> &str {
    let t = s.trim();
    if !t.starts_with("```") {
        return t;
    }
    let rest = t
        .strip_prefix("```json")
        .or_else(|| t.strip_prefix("```JSON"))
        .or_else(|| t.strip_prefix("```"))
        .unwrap_or(t);
    rest.strip_suffix("```").unwrap_or(rest).trim()
}

/// First top-level balanced JSON object/array span.
fn extract_json_span(s: &str) -> Option<&str> {
    let bytes = s.as_bytes();
    let start = bytes.iter().position(|&b| b == b'{' || b == b'[')?;
    let open = bytes[start];
    let close = if open == b'{' { b'}' } else { b']' };
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_string {
            if escape {
                escape = false;
            } else if b == b'\\' {
                escape = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b if b == open => depth += 1,
            b if b == close => {
                depth -= 1;
                if depth == 0 {
                    return Some(&s[start..=i]);
                }
            }
            _ => {}
        }
    }
    // Unterminated: take from start and try repair later
    if depth > 0 {
        return Some(s[start..].trim_end());
    }
    None
}

fn repair_common(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    let mut in_string = false;
    let mut escape = false;
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if in_string {
            out.push(c);
            if escape {
                escape = false;
            } else if c == '\\' {
                escape = true;
            } else if c == '"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        match c {
            '"' => {
                in_string = true;
                out.push(c);
            }
            '\'' => {
                // Treat single-quoted strings as double-quoted
                out.push('"');
                i += 1;
                while i < chars.len() {
                    let sc = chars[i];
                    if sc == '\'' && (i == 0 || chars[i - 1] != '\\') {
                        out.push('"');
                        break;
                    }
                    if sc == '"' {
                        out.push_str("\\\"");
                    } else {
                        out.push(sc);
                    }
                    i += 1;
                }
            }
            ',' => {
                // Drop trailing commas before } or ]
                let mut j = i + 1;
                while j < chars.len() && chars[j].is_whitespace() {
                    j += 1;
                }
                if j < chars.len() && (chars[j] == '}' || chars[j] == ']') {
                    // skip the comma
                } else {
                    out.push(c);
                }
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                // Bare key: word followed by colon → "word":
                let start = i;
                while i < chars.len()
                    && (chars[i].is_ascii_alphanumeric() || chars[i] == '_' || chars[i] == '-')
                {
                    i += 1;
                }
                let word: String = chars[start..i].iter().collect();
                let mut j = i;
                while j < chars.len() && chars[j].is_whitespace() {
                    j += 1;
                }
                if j < chars.len() && chars[j] == ':' {
                    out.push('"');
                    out.push_str(&word);
                    out.push('"');
                    continue; // i already advanced
                }
                out.push_str(&word);
                continue;
            }
            _ => out.push(c),
        }
        i += 1;
    }

    close_unbalanced(&mut out);
    out
}

/// Balance braces/brackets if truncated: close an open string, then every
/// unclosed container innermost first.
///
/// Closers come from a stack of the openers seen, not from two counters: a
/// fixed "all `]` then all `}`" order turned a truncated
/// `{"edits":[{"old":"x"` into `{"edits":[{"old":"x"]}}`, which does not
/// parse, so `heal_tool_args` fell back to `{}` and the call lost every
/// argument. A closer that does not match the innermost opener is malformed
/// input either way and is left alone. The stack is bounded by the input's
/// length.
fn close_unbalanced(out: &mut String) {
    let mut open: Vec<char> = Vec::new();
    let mut in_string = false;
    let mut escape = false;
    for c in out.chars() {
        if in_string {
            if escape {
                escape = false;
            } else if c == '\\' {
                escape = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        match c {
            '"' => in_string = true,
            '{' => open.push('}'),
            '[' => open.push(']'),
            '}' | ']' if open.last() == Some(&c) => {
                open.pop();
            }
            _ => {}
        }
    }
    debug_assert!(open.len() <= out.len(), "one closer per opener seen");
    if in_string {
        // A dangling escape would turn the closing quote into an escaped one.
        if escape {
            out.pop();
        }
        out.push('"');
    }
    while let Some(closer) = open.pop() {
        out.push(closer);
    }
}

/// Count balanced top-level JSON objects/arrays in `s`.
///
/// A count `> 1` is the signal that a single streamed `tool_use` argument buffer
/// holds *multiple* concatenated objects — i.e. several distinct tool calls
/// collapsed into one block (see the agent stream accumulator). String-aware, so
/// braces inside string literals do not inflate the count. Single pass, O(n).
#[must_use]
pub fn count_balanced_top_level_objects(s: &str) -> usize {
    let (mut count, mut depth, mut in_string, mut escape) = (0usize, 0i32, false, false);
    for &b in s.as_bytes() {
        if in_string {
            if escape {
                escape = false;
            } else if b == b'\\' {
                escape = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'{' | b'[' => depth += 1,
            b'}' | b']' => {
                depth -= 1;
                if depth == 0 {
                    count += 1;
                }
            }
            _ => {}
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_clean_json() {
        let v = heal_json(r#"{"a":1}"#).unwrap();
        assert_eq!(v["a"], 1);
    }

    #[test]
    fn strips_markdown_fence() {
        let raw = "```json\n[{\"content\":\"x\",\"category\":\"fact\"}]\n```";
        let v = heal_json(raw).unwrap();
        assert!(v.is_array());
        assert_eq!(v[0]["content"], "x");
    }

    #[test]
    fn extracts_object_from_prose() {
        let raw = "Sure! Here you go:\n{\"name\":\"Bob\",\"ok\":true}\nHope that helps.";
        let v = heal_json(raw).unwrap();
        assert_eq!(v["name"], "Bob");
    }

    #[test]
    fn repairs_trailing_comma() {
        let v = heal_json(r#"{"a":1,"b":2,}"#).unwrap();
        assert_eq!(v["b"], 2);
    }

    #[test]
    fn repairs_truncated_object() {
        let v = heal_json(r#"{"a": "hello", "b": 3"#).unwrap();
        assert_eq!(v["a"], "hello");
        assert_eq!(v["b"], 3);
    }

    #[test]
    fn truncated_nesting_closes_innermost_first() {
        // An object inside an array, cut off mid-string: the closers must be
        // `"}]}`, not `"]}}`.
        let v = heal_json(r#"{"edits":[{"old":"x"#).unwrap();
        assert_eq!(v["edits"][0]["old"], "x");
        let v = heal_json(r#"[{"a":[1,{"b":2"#).unwrap();
        assert_eq!(v[0]["a"][1]["b"], 2);
        let args = heal_tool_args(r#"{"path":"a.py","edits":[{"old":"f(","new":"g("#);
        assert_eq!(args["path"], "a.py");
        assert_eq!(args["edits"][0]["new"], "g(");
    }

    #[test]
    fn a_truncated_escape_does_not_swallow_the_closing_quote() {
        let v = heal_json(r#"{"content":"line"#).unwrap();
        assert_eq!(v["content"], "line");
    }

    #[test]
    fn tool_args_fallback_empty_object() {
        let v = heal_tool_args("not json at all");
        assert!(v.as_object().unwrap().is_empty());
    }

    #[test]
    fn typed_array_heal() {
        #[derive(serde::Deserialize)]
        struct Item {
            content: String,
        }
        let items: Vec<Item> = heal_json_as(r#"[{"content":"hi"}]"#).unwrap();
        assert_eq!(items[0].content, "hi");
    }

    #[test]
    fn last_object_concatenated_still_finds_first() {
        // Balanced extraction returns the first complete object
        let raw = r#"{"a":1} garbage {"b":2}"#;
        let v = heal_json(raw).unwrap();
        assert_eq!(v["a"], 1);
    }

    #[test]
    fn single_object_counts_one() {
        assert_eq!(count_balanced_top_level_objects(r#"{"a":1}"#), 1);
        assert_eq!(count_balanced_top_level_objects(r#"  {"a":{"b":2}}  "#), 1);
    }

    #[test]
    fn two_concatenated_objects_counts_two() {
        // The streaming-collapse shape: two tool calls' args in one buffer.
        let raw = r#"{"a":1}{"b":2}"#;
        assert_eq!(count_balanced_top_level_objects(raw), 2);
        // heal_json still salvages the FIRST object — count is the collapse
        // signal, heal is the (lossy) fallback.
        assert_eq!(heal_json(raw).unwrap()["a"], 1);
    }

    #[test]
    fn braces_inside_strings_ignored() {
        assert_eq!(count_balanced_top_level_objects(r#"{"a":"}{"}"#), 1);
        assert_eq!(count_balanced_top_level_objects(r#"{"a":"\""}"#), 1);
    }

    // ── bare control characters inside string literals ──
    // Observed 2026-08-14 (ornith UI leg): a local model writing TSV content
    // emits a literal 0x09 inside the JSON string instead of \t; strict serde
    // rejects the whole tool call.

    #[test]
    fn strict_serde_rejects_bare_tab_premise() {
        // The premise the lenient path exists for: spec-invalid, serde says no.
        assert!(serde_json::from_str::<Value>("{\"a\":\"x\ty\"}").is_err());
    }

    #[test]
    fn literal_tab_in_string_value_heals_to_real_tab() {
        let v = heal_json("{\"path\":\"db.tsv\",\"content\":\"key\tvalue\"}").unwrap();
        assert_eq!(v["content"], "key\tvalue");
    }

    #[test]
    fn literal_newline_and_cr_heal() {
        let v = heal_json("{\"content\":\"line1\nline2\rend\"}").unwrap();
        assert_eq!(v["content"], "line1\nline2\rend");
    }

    #[test]
    fn escaped_tab_is_untouched() {
        // A proper \t escape decodes to a real tab and the escape pass never
        // fires (nothing bare to rewrite).
        let raw = r#"{"content":"key\tvalue"}"#;
        assert_eq!(heal_json(raw).unwrap()["content"], "key\tvalue");
        assert!(escape_bare_controls_in_strings(raw).is_none());
    }

    #[test]
    fn controls_in_keys_and_short_escapeless_controls_heal() {
        // A tab inside a KEY string, and 0x01 (no short escape; becomes \u0001).
        let v = heal_json("{\"k\tey\":\"a\u{1}b\"}").unwrap();
        assert_eq!(v["k\tey"], "a\u{1}b");
    }

    #[test]
    fn fenced_json_with_literal_tab_heals() {
        let raw = "```json\n{\"content\":\"a\tb\"}\n```";
        assert_eq!(heal_json(raw).unwrap()["content"], "a\tb");
    }

    #[test]
    fn tool_args_with_literal_tab_reach_the_object() {
        let v = heal_tool_args("{\"path\":\"x\",\"content\":\"k\tv\"}");
        assert_eq!(v["content"], "k\tv");
    }

    #[test]
    fn escape_is_lossless_and_bounded() {
        let raw = "{\"a\":\"w\tx\ny\rz\u{1}\"}";
        let escaped = escape_bare_controls_in_strings(raw).unwrap();
        // Round-trip: parsing the escaped text restores the exact original
        // characters — the transform is lossless by construction.
        let v: Value = serde_json::from_str(&escaped).unwrap();
        assert_eq!(v["a"], "w\tx\ny\rz\u{1}");
        // Bounded: at most 6 output bytes per input byte.
        assert!(escaped.len() <= raw.len() * 6);
        // Controls OUTSIDE string literals are legal whitespace (or already
        // broken JSON) — never rewritten, so valid input stays byte-identical.
        assert!(escape_bare_controls_in_strings("{\t\"a\":\n1}").is_none());
    }

    #[test]
    fn malformed_input_with_tabs_still_rejected() {
        // Leniency covers exactly one defect class; garbage stays garbage.
        assert!(heal_json("not json \t at all").is_none());
        // `\<TAB>` is an invalid escape — ambiguous, left alone, still fails.
        assert!(heal_json("{\"a\":\"x\\\t\"}").is_none());
    }
}
