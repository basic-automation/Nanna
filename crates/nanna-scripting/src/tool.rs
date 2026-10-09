//! Scripted tool definitions

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt::Write as _;
use std::path::PathBuf;

/// A user-authored tool written in JavaScript/TypeScript
#[derive(Debug, Clone)]
pub struct ScriptedTool {
    /// Unique tool name
    pub name: String,
    /// Source code (JS or TS)
    pub source: String,
    /// Whether the source is TypeScript
    pub is_typescript: bool,
    /// Source file path (for error messages)
    pub source_path: Option<PathBuf>,
    /// Permissions granted to this tool
    pub permissions: ToolPermissions,
    /// Execution timeout in milliseconds
    pub timeout_ms: u64,
}

impl ScriptedTool {
    /// Create a new scripted tool from source code
    pub fn new(name: impl Into<String>, source: impl Into<String>) -> Self {
        let source = source.into();
        let name = name.into();
        // `rsplit_once` reads the text after the last `.`, which is exactly the
        // (case-sensitive) `.ts` suffix test this heuristic has always used.
        let is_typescript = name.rsplit_once('.').is_some_and(|(_, ext)| ext == "ts")
            || source.contains(": string")
            || source.contains(": number");

        Self {
            name,
            source,
            is_typescript,
            source_path: None,
            permissions: ToolPermissions::default(),
            timeout_ms: 30_000,
        }
    }

    /// Create from a file path
    ///
    /// # Errors
    ///
    /// Returns the I/O error if the file cannot be read, including when its
    /// contents are not valid UTF-8.
    pub fn from_file(path: impl Into<PathBuf>) -> std::io::Result<Self> {
        let path = path.into();
        let source = std::fs::read_to_string(&path)?;
        let name = path
            .file_stem().map_or_else(|| "unnamed".to_string(), |s| s.to_string_lossy().to_string());
        let is_typescript = path.extension().is_some_and(|e| e == "ts" || e == "tsx");

        Ok(Self {
            name,
            source,
            is_typescript,
            source_path: Some(path),
            permissions: ToolPermissions::default(),
            timeout_ms: 30_000,
        })
    }

    /// Set permissions
    #[must_use]
    pub fn with_permissions(mut self, permissions: ToolPermissions) -> Self {
        self.permissions = permissions;
        self
    }

    /// Set timeout
    #[must_use]
    pub const fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    /// Mark as TypeScript
    #[must_use]
    pub const fn typescript(mut self, is_ts: bool) -> Self {
        self.is_typescript = is_ts;
        self
    }
}

/// Permissions for scripted tools
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolPermissions {
    /// Allowed network hosts (empty = none)
    pub net: Vec<String>,
    /// Allowed read paths (empty = none)
    pub read: Vec<PathBuf>,
    /// Allowed write paths (empty = none)
    pub write: Vec<PathBuf>,
    /// Allow environment variable access
    pub env: bool,
    /// Allow subprocess execution
    pub run: bool,
}

impl ToolPermissions {
    /// No permissions (fully sandboxed)
    #[must_use]
    pub fn none() -> Self {
        Self::default()
    }

    /// Allow network access to specific hosts
    #[must_use]
    pub fn with_net(mut self, hosts: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.net = hosts.into_iter().map(Into::into).collect();
        self
    }

    /// Allow reading from specific paths
    #[must_use]
    pub fn with_read(mut self, paths: impl IntoIterator<Item = impl Into<PathBuf>>) -> Self {
        self.read = paths.into_iter().map(Into::into).collect();
        self
    }

    /// Allow writing to specific paths
    #[must_use]
    pub fn with_write(mut self, paths: impl IntoIterator<Item = impl Into<PathBuf>>) -> Self {
        self.write = paths.into_iter().map(Into::into).collect();
        self
    }

    /// Allow environment variable access
    #[must_use]
    pub const fn with_env(mut self) -> Self {
        self.env = true;
        self
    }

    /// Check if network access to a host is allowed
    #[must_use]
    pub fn allows_net(&self, host: &str) -> bool {
        self.net
            .iter()
            .any(|h| h == "*" || h == host || host.ends_with(&format!(".{h}")))
    }

    /// Check if reading a path is allowed.
    /// Supports `*` wildcard for unrestricted access and `~` for home directory.
    #[must_use]
    pub fn allows_read(&self, path: &std::path::Path) -> bool {
        self.read.iter().any(|scope| scope_covers(scope, path))
    }

    /// Check if writing a path is allowed.
    /// Supports `*` wildcard for unrestricted access and `~` for home directory.
    #[must_use]
    pub fn allows_write(&self, path: &std::path::Path) -> bool {
        self.write.iter().any(|scope| scope_covers(scope, path))
    }
}

/// The home directory, resolved once.
///
/// `allows_read` runs on every file a scripted tool touches, so the lookup is
/// cached rather than repeated per check.
static HOME_DIR: std::sync::LazyLock<Option<PathBuf>> = std::sync::LazyLock::new(|| {
    directories::UserDirs::new().map(|dirs| dirs.home_dir().to_path_buf())
});

/// Whether one declared scope covers `path`.
///
/// `*` is the whole filesystem and `~` is the home directory — which the doc
/// comments above have always promised and the check did **not** honour:
/// `Path::starts_with` is component-wise, so a literal `PathBuf("~")` matched
/// nothing and a `~`-scoped permission silently denied everything.
///
/// That was invisible in practice because `ScriptedToolWrapper::from_file`
/// expands `~` when it loads a `permissions.json`, one crate away — so every
/// production path arrived here already expanded, and only a caller
/// constructing `ToolPermissions` **programmatically** with `"~"` would have hit
/// the silent deny the comment said was supported. Honouring it here removes
/// the trap instead of documenting it, and changes nothing for an
/// already-expanded path.
fn scope_covers(scope: &std::path::Path, path: &std::path::Path) -> bool {
    // `Path::starts_with` compares components as written, so `~/../../etc`
    // "starts with" home. The path the bridge hands over is joined, never
    // normalised, and the OS resolves `..` when the file is opened — so the
    // check has to see the same path the open will.
    let normal = lexically_normal(path);
    let path = normal.as_path();
    let text = scope.to_string_lossy();
    if text == "*" {
        return true;
    }
    if text == "~" || text.starts_with("~/") {
        let Some(home) = HOME_DIR.as_ref() else {
            // No home directory to expand against: refuse rather than widen.
            return false;
        };
        let expanded = home.join(text.strip_prefix("~/").unwrap_or(""));
        return path.starts_with(expanded);
    }
    path.starts_with(scope)
}

/// `path` with `.` dropped and each `..` removing the component before it,
/// never climbing past the root — what the OS does on open, short of
/// symlinks: `a/link/..` is `a` here but the link's parent on disk, so a
/// symlink out of a scope is still the scope's own business. Pure: no
/// filesystem access, so a path that does not exist yet (a write) is checked
/// the same way.
fn lexically_normal(path: &std::path::Path) -> PathBuf {
    use std::path::Component;
    let mut normal = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                // At the root (or a prefix) there is nothing to climb to.
                if !matches!(
                    normal.components().next_back(),
                    None | Some(Component::RootDir | Component::Prefix(_))
                ) {
                    normal.pop();
                }
            }
            other => normal.push(other),
        }
    }
    debug_assert!(
        normal
            .components()
            .all(|c| !matches!(c, Component::ParentDir | Component::CurDir)),
        "nothing left to resolve"
    );
    normal
}

/// Where a tool's output should be routed after execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum OutputTarget {
    /// Large results are chunked and stored in memory; a stub replaces them in context (default).
    #[default]
    Memory,
    /// Results always stay in context. Large results are summarized (or truncated as fallback).
    Context,
}

/// Tool manifest extracted from source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolManifest {
    /// Tool name
    pub name: String,
    /// Semantic version string (e.g. "0.1.0")
    pub version: Option<String>,
    /// Tool description
    pub description: Option<String>,
    /// Input parameter schema (JSON Schema)
    pub parameters: Option<Value>,
    /// Where tool output should be routed
    #[serde(default)]
    pub output: OutputTarget,
    /// Timeout in seconds (None = use default 30s)
    pub timeout_secs: Option<u64>,
    /// Host services this tool CANNOT work without, e.g. `["memory.list"]`.
    ///
    /// A tool whose requirements are unmet is not registered at all, so it is
    /// never advertised to the model. Omit the field for tools that degrade
    /// gracefully (the `todo` skill calls `tasks.*` but falls back to a JSON
    /// file, so it must still load when storage is absent).
    ///
    /// Motivation, observed live 2026-07-26: with the memory service disabled
    /// (no embedding provider) `reflect` was still offered and failed on every
    /// single call with `Service not found: memory.list`; `list_reminders`
    /// likewise, since `schedule.*` is registered nowhere in the daemon. A
    /// model cannot tell "broken" from "try again", so it retried them in a
    /// loop instead of doing the work.
    #[serde(default)]
    pub requires: Vec<String>,
}

/// Extract manifest from tool source (looks for default export)
#[must_use]
pub fn extract_manifest(source: &str) -> Option<ToolManifest> {
    // Simple regex-free extraction for common patterns
    // export default { name: "...", description: "...", ... }

    let name = extract_string_field(source, "name")?;
    let version = extract_string_field(source, "version");
    let description = extract_string_field(source, "description");
    let output = match extract_string_field(source, "output").as_deref() {
        Some("context") => OutputTarget::Context,
        _ => OutputTarget::Memory,
    };

    let timeout_secs = extract_number_field(source, "timeout");

    Some(ToolManifest {
        name,
        version,
        description,
        parameters: extract_parameters_schema(source),
        output,
        timeout_secs,
        requires: extract_string_array_field(source, "requires"),
    })
}

/// Extract a manifest field declared as an array of string literals, e.g.
/// `requires: ["memory.list", "memory.search"]`.
///
/// Deliberately narrow: it reads the bracketed block that follows the key and
/// pulls out quoted items. Anything it cannot parse yields an empty list —
/// which means "no requirements", the permissive default, so a malformed
/// annotation can never silently hide a working tool.
fn extract_string_array_field(source: &str, field: &str) -> Vec<String> {
    let patterns = [format!("{field}:"), format!("{field} :")];
    let Some(start) = patterns.iter().find_map(|p| source.find(p.as_str())) else {
        return Vec::new();
    };
    let after = &source[start..];
    let Some(open) = after.find('[') else {
        return Vec::new();
    };
    let Some(close) = after[open..].find(']') else {
        return Vec::new();
    };
    let block = &after[open + 1..open + close];

    let mut out = Vec::new();
    let mut rest = block;
    while let Some(q) = rest.find(['"', '\'']) {
        let quote = rest.as_bytes()[q] as char;
        let value_start = q + 1;
        let Some(end) = rest[value_start..].find(quote) else {
            break;
        };
        let value = rest[value_start..value_start + end].trim().to_string();
        if !value.is_empty() {
            out.push(value);
        }
        rest = &rest[value_start + end + 1..];
    }
    out
}

/// Extract the JSON-Schema `parameters` object from a tool manifest source.
///
/// Manifests declare `parameters` as a JS object literal (unquoted keys, maybe
/// single-quoted strings, trailing commas, or comments), which is not valid
/// JSON. We locate the balanced `{..}` block following `parameters:` and
/// normalize it to strict JSON so the LLM-facing tool definition carries the
/// real input schema instead of an empty parameter list.
///
/// Returns `None` (falling back to no parameters — the prior behavior) if the
/// block is absent or can't be normalized into valid JSON; it never guesses.
pub fn extract_parameters_schema(source: &str) -> Option<Value> {
    let block = extract_object_after_key(source, "parameters")?;
    debug_assert!(block.starts_with('{') && block.ends_with('}'));
    let json = js_object_literal_to_json(block);
    serde_json::from_str::<Value>(&json).ok()
}

#[inline]
const fn is_ident_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_' || b == b'$'
}

#[inline]
const fn is_ident_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'$'
}

/// Find `key:` at identifier boundary followed by a `{`, and return the
/// balanced object block (including braces), respecting strings and comments.
fn extract_object_after_key<'a>(source: &'a str, key: &str) -> Option<&'a str> {
    let bytes = source.as_bytes();
    let mut from = 0usize;
    while let Some(rel) = source[from..].find(key) {
        let idx = from + rel;
        from = idx + key.len();
        // Must be a standalone identifier, not a substring of a longer one.
        let prev_ok = idx == 0 || !is_ident_char(bytes[idx - 1]);
        let next_ok = !bytes
            .get(idx + key.len())
            .is_some_and(|&b| is_ident_char(b));
        if !prev_ok || !next_ok {
            continue;
        }
        // Expect: optional ws, ':', optional ws, '{'.
        let mut j = idx + key.len();
        while j < bytes.len() && bytes[j].is_ascii_whitespace() {
            j += 1;
        }
        if bytes.get(j) != Some(&b':') {
            continue;
        }
        j += 1;
        while j < bytes.len() && bytes[j].is_ascii_whitespace() {
            j += 1;
        }
        if bytes.get(j) != Some(&b'{') {
            continue;
        }
        return extract_balanced_object(source, j);
    }
    None
}

/// Return the substring from the `{` at `open` to its matching `}` (inclusive),
/// skipping braces that appear inside string literals or comments.
fn extract_balanced_object(source: &str, open: usize) -> Option<&str> {
    let bytes = source.as_bytes();
    debug_assert_eq!(bytes.get(open), Some(&b'{'));
    let mut depth: usize = 0;
    let mut i = open;
    let mut in_str: Option<u8> = None;
    let mut escaped = false;
    while i < bytes.len() {
        let c = bytes[i];
        if let Some(q) = in_str {
            if escaped {
                escaped = false;
            } else if c == b'\\' {
                escaped = true;
            } else if c == q {
                in_str = None;
            }
            i += 1;
            continue;
        }
        match c {
            b'"' | b'\'' => in_str = Some(c),
            b'/' if bytes.get(i + 1) == Some(&b'/') => {
                i += 2;
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
                continue;
            }
            b'/' if bytes.get(i + 1) == Some(&b'*') => {
                i += 2;
                while i < bytes.len() && !(bytes[i] == b'*' && bytes.get(i + 1) == Some(&b'/')) {
                    i += 1;
                }
                i += 2;
                continue;
            }
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&source[open..=i]);
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// Normalize a JS object literal into strict JSON: convert single-quoted
/// strings to double-quoted, quote bare identifier keys, drop comments, and
/// drop trailing commas. Bare `true`/`false`/`null` values are left intact.
fn js_object_literal_to_json(block: &str) -> String {
    let bytes = block.as_bytes();
    let mut out = String::with_capacity(block.len() + 16);
    let mut i = 0usize;
    while i < bytes.len() {
        let c = bytes[i];
        match c {
            // Line comment.
            b'/' if bytes.get(i + 1) == Some(&b'/') => {
                i += 2;
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
            }
            // Block comment.
            b'/' if bytes.get(i + 1) == Some(&b'*') => {
                i += 2;
                while i < bytes.len() && !(bytes[i] == b'*' && bytes.get(i + 1) == Some(&b'/')) {
                    i += 1;
                }
                i += 2;
            }
            // String literal (either quote style) -> emit as JSON double-quoted.
            b'"' | b'\'' => {
                let (s, next) = read_js_string(block, i);
                push_json_string(&mut out, &s);
                i = next;
            }
            // Trailing comma: drop if the next significant char closes a container.
            b',' => {
                // A trailing comma is skipped; any other is kept.
                if !next_significant_is_close(block, i + 1) {
                    out.push(',');
                }
                i += 1;
            }
            // Bare identifier: a key (followed by ':') gets quoted; a literal
            // value (true/false/null) is copied verbatim.
            _ if is_ident_start(c) => {
                let start = i;
                i += 1;
                while i < bytes.len() && is_ident_char(bytes[i]) {
                    i += 1;
                }
                let ident = &block[start..i];
                if next_significant_is_colon(block, i) {
                    push_json_string(&mut out, ident);
                } else {
                    out.push_str(ident);
                }
            }
            _ => {
                // Copy a full UTF-8 char (structural tokens are all ASCII and
                // handled above; this arm may see multibyte text).
                let ch = block[i..].chars().next().unwrap_or(c as char);
                out.push(ch);
                i += ch.len_utf8();
            }
        }
    }
    out
}

/// Read a JS string starting at `open` (a quote char); returns the fully
/// **decoded** contents (escapes resolved to real characters, surrounding
/// quotes stripped) and the index just past the close. The caller re-encodes
/// via [`push_json_string`], so decoding here keeps escaping single-sourced.
fn read_js_string(block: &str, open: usize) -> (String, usize) {
    let bytes = block.as_bytes();
    let quote = bytes[open];
    let mut s = String::new();
    let mut i = open + 1;
    while i < bytes.len() {
        let c = bytes[i];
        if c == b'\\' {
            match bytes.get(i + 1) {
                Some(b'"') => s.push('"'),
                Some(b'\'') => s.push('\''),
                // A lone trailing backslash is kept as-is, like an escaped one.
                Some(b'\\') | None => s.push('\\'),
                Some(b'/') => s.push('/'),
                Some(b'n') => s.push('\n'),
                Some(b't') => s.push('\t'),
                Some(b'r') => s.push('\r'),
                Some(b'b') => s.push('\u{08}'),
                Some(b'f') => s.push('\u{0C}'),
                Some(b'u') => {
                    // \uXXXX -> the code point (malformed sequences pass through).
                    if let Some(cp) = block
                        .get(i + 2..i + 6)
                        .and_then(|hex| u32::from_str_radix(hex, 16).ok())
                        .and_then(char::from_u32)
                    {
                        s.push(cp);
                        i += 6;
                        continue;
                    }
                    s.push('\\');
                    i += 1;
                    continue;
                }
                Some(_) => {
                    // Unknown escape: keep the escaped char literally.
                    i += 1;
                    if let Some(ch) = block[i..].chars().next() {
                        s.push(ch);
                        i += ch.len_utf8();
                    }
                    continue;
                }
            }
            i += 2;
            continue;
        }
        if c == quote {
            i += 1;
            break;
        }
        // Copy a full UTF-8 char so multibyte text survives.
        let ch = block[i..].chars().next().unwrap_or(c as char);
        s.push(ch);
        i += ch.len_utf8();
    }
    (s, i)
}

/// Append `raw` to `out` as a valid JSON string literal (quotes + escaping).
fn push_json_string(out: &mut String, raw: &str) {
    out.push('"');
    for ch in raw.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if u32::from(c) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", u32::from(c));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// True if the next non-whitespace, non-comment byte from `i` is `:`.
fn next_significant_is_colon(block: &str, i: usize) -> bool {
    matches!(peek_significant(block, i), Some(b':'))
}

/// True if the next non-whitespace, non-comment byte from `i` closes a
/// container (`}` or `]`) — i.e. the preceding comma was trailing.
fn next_significant_is_close(block: &str, i: usize) -> bool {
    matches!(peek_significant(block, i), Some(b'}' | b']'))
}

/// Peek the next significant byte (skipping whitespace and comments).
fn peek_significant(block: &str, mut i: usize) -> Option<u8> {
    let bytes = block.as_bytes();
    while i < bytes.len() {
        match bytes[i] {
            b if b.is_ascii_whitespace() => i += 1,
            b'/' if bytes.get(i + 1) == Some(&b'/') => {
                i += 2;
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
            }
            b'/' if bytes.get(i + 1) == Some(&b'*') => {
                i += 2;
                while i < bytes.len() && !(bytes[i] == b'*' && bytes.get(i + 1) == Some(&b'/')) {
                    i += 1;
                }
                i += 2;
            }
            b => return Some(b),
        }
    }
    None
}

fn extract_number_field(source: &str, field: &str) -> Option<u64> {
    // Match `field: 300` / `field:300` at an identifier boundary, returning the
    // first occurrence that is actually followed by digits.
    //
    // Scanning *every* occurrence (not just the first) matters: a manifest may
    // declare the same key as a *parameter* too — e.g. the `exec` skill has both
    // a top-level `timeout: 180` and a `timeout: { type: "integer" }` inside
    // `parameters`. The numeric top-level value must win regardless of which
    // appears first in the source. The boundary check also stops `read_timeout:`
    // from matching a search for `timeout`.
    let bytes = source.as_bytes();
    let mut from = 0usize;
    while let Some(rel) = source[from..].find(field) {
        let idx = from + rel;
        from = idx + field.len();

        let prev_ok = idx == 0 || !is_ident_char(bytes[idx - 1]);
        if !prev_ok {
            continue;
        }

        // Expect: optional ws, ':', optional ws, then a run of digits.
        let mut j = idx + field.len();
        while j < bytes.len() && bytes[j].is_ascii_whitespace() {
            j += 1;
        }
        if bytes.get(j) != Some(&b':') {
            continue;
        }
        j += 1;
        while j < bytes.len() && bytes[j].is_ascii_whitespace() {
            j += 1;
        }
        let num_str: String = source[j..].chars().take_while(char::is_ascii_digit).collect();
        if let Ok(n) = num_str.parse::<u64>() {
            return Some(n);
        }
    }
    None
}

fn extract_string_field(source: &str, field: &str) -> Option<String> {
    // Match: name: "value" or name: 'value'
    let patterns = [
        format!(r#"{field}: ""#),
        format!(r"{field}: '"),
        format!(r#"{field}:""#),
        format!(r"{field}:'"),
    ];

    for pattern in &patterns {
        if let Some(start) = source.find(pattern) {
            let quote = if pattern.ends_with('"') { '"' } else { '\'' };
            let value_start = start + pattern.len();
            if let Some(end) = source[value_start..].find(quote) {
                return Some(source[value_start..value_start + end].to_string());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requires_is_parsed_from_the_manifest() {
        let src = r#"
export default {
  name: "reflect",
  requires: ["memory.list", "memory.delete"],
  version: "0.1.0",
  description: "Review stored memories.",
  parameters: { type: "object", properties: {} }
}
"#;
        let m = extract_manifest(src).expect("manifest");
        assert_eq!(m.name, "reflect");
        assert_eq!(m.requires, vec!["memory.list", "memory.delete"]);
    }

    #[test]
    fn a_tool_without_requires_declares_nothing() {
        // The permissive default: todo calls tasks.* but falls back to a JSON
        // file, so it must keep loading when storage is absent.
        let src = r#"
export default {
  name: "todo",
  description: "Task store.",
  parameters: { type: "object", properties: {} }
}
"#;
        let m = extract_manifest(src).expect("manifest");
        assert_eq!(m.requires, Vec::<String>::new());
    }

    #[test]
    fn a_malformed_requires_annotation_hides_nothing() {
        // Unparseable annotation must read as "no requirements" so a typo can
        // never silently remove a working tool from the surface.
        let src = r#"
export default {
  name: "weird",
  requires: not_an_array,
  description: "x"
}
"#;
        let m = extract_manifest(src).expect("manifest");
        assert_eq!(m.requires, Vec::<String>::new());
    }

    #[test]
    fn test_extract_manifest() {
        let source = r#"
            export default {
                name: "greet",
                description: "Greet someone by name",
                execute({ name }) {
                    return `Hello, ${name}!`;
                }
            }
        "#;

        let manifest = extract_manifest(source).unwrap();
        assert_eq!(manifest.name, "greet");
        assert_eq!(manifest.version, None);
        assert_eq!(
            manifest.description,
            Some("Greet someone by name".to_string())
        );
    }

    #[test]
    fn test_extract_manifest_with_version() {
        let source = r#"
            export default {
                name: "exec",
                version: "0.1.0",
                description: "Run a command",
                execute({ command }) {
                    return "done";
                }
            }
        "#;

        let manifest = extract_manifest(source).unwrap();
        assert_eq!(manifest.name, "exec");
        assert_eq!(manifest.version, Some("0.1.0".to_string()));
    }

    #[test]
    fn test_extract_manifest_with_output_context() {
        let source = r#"
            export default {
                name: "recall",
                description: "Search memory",
                output: "context",
                execute({ query }) {
                    return "results";
                }
            }
        "#;

        let manifest = extract_manifest(source).unwrap();
        assert_eq!(manifest.name, "recall");
        assert_eq!(manifest.output, OutputTarget::Context);
    }

    #[test]
    fn test_extract_manifest_output_defaults_to_memory() {
        let source = r#"
            export default {
                name: "exec",
                description: "Run a command",
                execute({ command }) {
                    return "done";
                }
            }
        "#;

        let manifest = extract_manifest(source).unwrap();
        assert_eq!(manifest.output, OutputTarget::Memory);
    }

    #[test]
    fn test_permissions() {
        let perms = ToolPermissions::none()
            .with_net(["api.example.com", "*.github.com"])
            .with_read(["/tmp"]);

        assert!(perms.allows_net("api.example.com"));
        assert!(!perms.allows_net("evil.com"));
        assert!(perms.allows_read(std::path::Path::new("/tmp/file.txt")));
        assert!(!perms.allows_read(std::path::Path::new("/etc/passwd")));
    }

    #[test]
    fn params_schema_matches_real_manifest_shape() {
        // Mirrors default-skills/read_file/tool.ts exactly.
        let source = r#"
            export default {
                name: "read_file",
                output: "context",
                description: "Read a file.",
                parameters: {
                    type: "object",
                    properties: {
                        file_path: { type: "string", description: "Path to the file." },
                        offset: { type: "integer", description: "Start line." },
                        limit: { type: "integer", description: "Max lines." }
                    },
                    required: ["file_path"]
                },
                execute: function(input) { return "x"; }
            }
        "#;
        let manifest = extract_manifest(source).unwrap();
        let params = manifest.parameters.expect("parameters must be parsed");
        assert_eq!(params["type"], "object");
        assert_eq!(params["properties"]["file_path"]["type"], "string");
        assert_eq!(
            params["properties"]["file_path"]["description"],
            "Path to the file."
        );
        assert_eq!(params["properties"]["offset"]["type"], "integer");
        assert_eq!(params["required"][0], "file_path");
    }

    #[test]
    fn params_schema_tolerates_trailing_commas_and_comments() {
        let source = r#"
            export default {
                name: "t",
                description: "d",
                parameters: {
                    // the input document
                    type: "object",
                    properties: {
                        q: { type: "string", }, /* the query */
                    },
                    required: ["q"],
                },
                execute(i) {}
            }
        "#;
        let params = extract_parameters_schema(source).expect("valid despite commas/comments");
        assert_eq!(params["properties"]["q"]["type"], "string");
        assert_eq!(params["required"][0], "q");
    }

    #[test]
    fn params_schema_handles_single_quotes_and_enum() {
        let source = r"
            export default {
                name: 't',
                description: 'd',
                parameters: {
                    type: 'object',
                    properties: {
                        mode: { type: 'string', enum: ['fast', 'slow'], default: 'fast' }
                    }
                },
                execute(i) {}
            }
        ";
        let params = extract_parameters_schema(source).expect("single quotes normalize");
        assert_eq!(params["properties"]["mode"]["type"], "string");
        assert_eq!(params["properties"]["mode"]["enum"][1], "slow");
        assert_eq!(params["properties"]["mode"]["default"], "fast");
    }

    #[test]
    fn params_schema_handles_escaped_quotes_in_descriptions() {
        // Mirrors default-skills/python/tool.ts: a double-quoted description
        // whose text embeds escaped double-quotes (a JSON example).
        let source = r#"
            export default {
                name: "python",
                description: "run python",
                parameters: {
                    type: "object",
                    properties: {
                        args: {
                            type: "string",
                            description: "JSON like: '{\"pattern\": \"*.rs\", \"dry_run\": true}'"
                        }
                    },
                    required: []
                },
                execute(i) {}
            }
        "#;
        let params = extract_parameters_schema(source).expect("escaped quotes normalize");
        assert_eq!(
            params["properties"]["args"]["description"],
            "JSON like: '{\"pattern\": \"*.rs\", \"dry_run\": true}'"
        );
        assert_eq!(params["required"], serde_json::json!([]));
    }

    #[test]
    fn params_schema_absent_yields_none() {
        // A tool with no `parameters` declaration keeps the prior behavior.
        let source = r#"
            export default {
                name: "noparams",
                description: "no schema",
                execute(i) { return "ok"; }
            }
        "#;
        let manifest = extract_manifest(source).unwrap();
        assert!(manifest.parameters.is_none());
    }

    #[test]
    fn params_schema_preserves_non_ascii_descriptions() {
        // Byte-wise normalization must not corrupt multibyte UTF-8 text.
        let source = r#"
            export default {
                name: "t",
                description: "d",
                parameters: {
                    type: "object",
                    properties: {
                        note: { type: "string", description: "café — déjà vu ✓" }
                    }
                },
                execute(i) {}
            }
        "#;
        let params = extract_parameters_schema(source).expect("utf-8 survives");
        assert_eq!(
            params["properties"]["note"]["description"],
            "café — déjà vu ✓"
        );
    }

    #[test]
    fn timeout_field_prefers_numeric_over_parameter_declaration() {
        // Mirrors the exec skill: a numeric top-level `timeout` alongside a
        // `timeout` *parameter*. The numeric one must be picked regardless of
        // order, and `read_timeout:` must not be mistaken for `timeout:`.
        let source = r#"
            export default {
                name: "exec",
                read_timeout: 999,
                parameters: {
                    properties: {
                        timeout: { type: "integer", description: "secs" }
                    }
                },
                timeout: 180,
                execute(i) {}
            }
        "#;
        assert_eq!(extract_number_field(source, "timeout"), Some(180));
    }

    #[test]
    fn timeout_field_absent_yields_none() {
        let source = r#"export default { name: "t", description: "d", execute(i) {} }"#;
        assert_eq!(extract_number_field(source, "timeout"), None);
    }

    #[test]
    fn params_schema_braces_inside_strings_dont_break_balance() {
        // A `}` inside a description string must not close the object early.
        let source = r#"
            export default {
                name: "t",
                description: "d",
                parameters: {
                    type: "object",
                    properties: {
                        tpl: { type: "string", description: "use {curly} braces }" }
                    }
                },
                execute(i) {}
            }
        "#;
        let params = extract_parameters_schema(source).expect("string braces ignored");
        assert_eq!(
            params["properties"]["tpl"]["description"],
            "use {curly} braces }"
        );
    }
}

#[cfg(test)]
mod scope_tests {
    use super::ToolPermissions;
    use std::path::{Path, PathBuf};

    fn home() -> PathBuf {
        directories::UserDirs::new()
            .map(|d| d.home_dir().to_path_buf())
            .expect("the test host has a home directory")
    }

    #[test]
    fn a_wildcard_scope_covers_everything() {
        let perms = ToolPermissions::none().with_read(["*"]);
        assert!(perms.allows_read(Path::new("/etc/passwd")));
        assert!(perms.allows_read(&home().join("notes.md")));
    }

    /// The documented behaviour that was not implemented: a literal `~` in a
    /// programmatically-built `ToolPermissions` used to match nothing.
    #[test]
    fn a_tilde_scope_covers_the_home_directory() {
        let perms = ToolPermissions::none().with_read(["~"]);
        assert!(
            perms.allows_read(&home().join("notes.md")),
            "a `~` scope denied a file inside home, which the doc comment \
             promised it would allow",
        );
    }

    #[test]
    fn a_tilde_scope_does_not_cover_outside_home() {
        let perms = ToolPermissions::none().with_read(["~"]);
        assert!(
            !perms.allows_read(Path::new("/etc/passwd")),
            "a `~` scope reached outside the home directory",
        );
    }

    #[test]
    fn a_tilde_subpath_scope_covers_only_that_subtree() {
        let perms = ToolPermissions::none().with_write(["~/projects"]);
        assert!(perms.allows_write(&home().join("projects").join("a.txt")));
        assert!(!perms.allows_write(&home().join("secrets.txt")));
    }

    #[test]
    fn an_absolute_scope_is_unchanged_by_the_tilde_handling() {
        let perms = ToolPermissions::none().with_read(["/tmp"]);
        assert!(perms.allows_read(Path::new("/tmp/file.txt")));
        assert!(!perms.allows_read(Path::new("/etc/passwd")));
    }

    /// `..` used to pass the prefix check unresolved: with the default `~`
    /// scope, `~/../../etc/passwd` was allowed while `/etc/passwd` was not.
    #[test]
    fn dot_dot_cannot_climb_out_of_a_scope() {
        let perms = ToolPermissions::none()
            .with_read(["~"])
            .with_write(["~/projects"]);
        let escape = home().join("..").join("..").join("etc").join("passwd");
        assert!(!perms.allows_read(&escape), "{}", escape.display());
        let sibling = home().join("projects").join("..").join("secrets.txt");
        assert!(!perms.allows_write(&sibling));
        // Climbing back inside is fine, and so is a `.`.
        let inside = home().join("projects").join("a").join("..").join("b.txt");
        assert!(perms.allows_write(&inside));
        assert!(perms.allows_read(&home().join(".").join("notes.md")));
        // `..` at the root stays at the root.
        assert_eq!(
            super::lexically_normal(Path::new("/../../etc")),
            Path::new("/etc")
        );
    }

    #[test]
    fn an_empty_scope_list_allows_nothing() {
        let perms = ToolPermissions::none();
        assert!(!perms.allows_read(Path::new("/tmp/file.txt")));
        assert!(!perms.allows_write(&home().join("notes.md")));
    }
}
