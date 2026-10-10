#![warn(clippy::pedantic, clippy::nursery, clippy::all)]
//! The `read_file` skill against real files.
//!
//! The skill's own header says every failure is RETURNED, never thrown: a
//! thrown error reaches the model under stacked "Execution failed:" prefixes
//! and reads as corruption. A file that is not UTF-8 text made the bridge's
//! `readFile` raise past an unguarded call.

#![cfg(feature = "boa")]

use nanna_scripting::{ScriptEngine, ScriptedTool, ToolPermissions};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

fn skill_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../nanna-tools/default-skills/read_file/tool.ts")
}

async fn run_read(input: Value, dir: &Path) -> Result<Value, String> {
    let tool = ScriptedTool::from_file(skill_path())
        .expect("read read_file tool.ts")
        .with_permissions(ToolPermissions::none().with_read([dir]).with_write([dir]))
        .with_timeout(30_000);
    ScriptEngine::new()
        .execute(&tool, input, None, None)
        .await
        .map(|r| r.value)
        .map_err(|e| e.to_string())
}

#[tokio::test]
async fn a_binary_file_is_answered_not_thrown() {
    let dir = tempfile::tempdir().expect("temp dir");
    let image = dir.path().join("logo.png");
    std::fs::write(
        &image,
        [
            0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a, 0xff, 0xfe, 0x00,
        ],
    )
    .expect("write binary");

    let result = run_read(json!({ "file_path": image.to_string_lossy() }), dir.path())
        .await
        .expect("a binary file is an answer, not a script error");
    assert_eq!(result["success"], Value::Bool(false), "{result}");
    let content = result["content"].as_str().expect("content");
    assert!(content.contains("could not be read as text"), "{content}");
    assert!(content.contains("Nothing was read"), "{content}");

    let text = dir.path().join("notes.txt");
    std::fs::write(&text, "hello\n").expect("write text");
    let result = run_read(json!({ "file_path": text.to_string_lossy() }), dir.path())
        .await
        .expect("a text file reads");
    assert!(result.to_string().contains("hello"), "{result}");
}
