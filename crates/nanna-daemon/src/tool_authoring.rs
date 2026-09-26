//! The `tools.create` / `tools.update` / `tools.list` services.
//!
//! Three bundled skills — `create_tool`, `edit_tool`, `list_user_tools` —
//! declare these and nothing registered them, so all three were withheld from
//! the model at every boot (see `tests/skill_services_are_registered.rs`, which
//! found them). The skills themselves are complete and specify the contract in
//! detail; this is the daemon half they were written against.
//!
//! **Storage shape.** A user tool is a skill directory —
//! `{tools_dir}/{name}/tool.ts` plus a `permissions.json` — which is what the
//! skills' own text describes and what the loader already reads. That is
//! deliberately *not* [`crate::user_tools::UserToolManager`]'s `{name}.json`
//! shape: that is a separate, older store with its own IPC surface, and a tool
//! written there would not be discovered by `discover_skills` at all.
//!
//! **Containment.** The only user-supplied path component is the tool name, and
//! `validate_tool_name` restricts it to `^[a-z][a-z0-9_]{0,63}$` — no `/`, no
//! `.`, no `..`, so traversal is impossible by construction rather than by
//! filtering. The symlink check on top covers the remaining case: a directory
//! somebody else planted that points out of the tools dir.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock, Weak};

use nanna_scripting::{ServiceFn, ServiceMap, extract_manifest};
use nanna_tools::ToolRegistry;
use serde_json::{Value, json};
use tracing::{info, warn};

use crate::user_tools::validate_tool_name;

/// Ceiling on an authored tool's source.
///
/// Not a round number picked for looks: it is the same 1 MiB the scripting
/// engine already refuses to compile beyond, so a larger source could only be
/// written and then fail to load. Rejecting it here means the refusal names the
/// real reason instead of surfacing later as a broken tool.
pub const TOOL_SOURCE_BYTES_MAX: usize = 1024 * 1024;

/// Resolve a tool's source file, refusing anything that is not a plain file
/// inside the tools directory.
///
/// Pure enough to test without a registry: it touches the filesystem only to
/// ask whether the directory is a symlink.
fn resolve_tool_dir(tools_dir: &Path, name: &str) -> Result<PathBuf, String> {
    validate_tool_name(name)?;
    let dir = tools_dir.join(name);
    assert!(
        dir.starts_with(tools_dir),
        "a validated tool name escaped the tools directory",
    );

    // A validated name cannot traverse, but the directory it points at can
    // still be a symlink somebody else planted.
    if let Ok(meta) = std::fs::symlink_metadata(&dir)
        && meta.file_type().is_symlink()
    {
        return Err(format!(
            "'{name}' is a symlink; refusing to write through it"
        ));
    }
    Ok(dir)
}

/// Apply an `old_string` → `new_string` edit, refusing anything but exactly one
/// match.
///
/// Zero matches means the caller is editing something that is not there; more
/// than one means it cannot know which it changed. Both are the caller's bug,
/// and both are reported rather than guessed at.
fn replace_exactly_once(current: &str, old: &str, new: &str) -> Result<String, String> {
    if old.is_empty() {
        return Err("old_string is empty; pass the text to replace".to_string());
    }
    let count = current.matches(old).count();
    if count == 0 {
        return Err(
            "old_string does not appear in the tool's current source; nothing was changed"
                .to_string(),
        );
    }
    if count > 1 {
        return Err(format!(
            "old_string appears {count} times; make it unique so the edit is \
             unambiguous. Nothing was changed"
        ));
    }
    let replaced = current.replacen(old, new, 1);
    debug_assert_ne!(replaced, current, "an applied edit changed nothing");
    Ok(replaced)
}

/// Reject a source the loader would not accept, before it reaches disk.
///
/// A tool whose source has no default export is written, registered and
/// advertised, and the first anyone hears of it is a failed call — the same
/// failure `UserToolManager::create_tool` was hardened against.
fn validate_source(source: &str) -> Result<(), String> {
    if source.trim().is_empty() {
        return Err("source is empty".to_string());
    }
    if source.len() > TOOL_SOURCE_BYTES_MAX {
        return Err(format!(
            "source is {} bytes; the scripting engine refuses anything over \
             {TOOL_SOURCE_BYTES_MAX}",
            source.len()
        ));
    }
    if extract_manifest(source).is_none() {
        return Err(
            "source must `export default` an object with `name` and `description`".to_string(),
        );
    }
    // The same parse `UserToolManager::create_tool` runs. Without it an edit
    // that broke the syntax was written, re-registered and advertised, and
    // failed only when called — a working tool replaced by a broken one.
    nanna_scripting::check_syntax(source).map_err(|e| {
        format!("source does not parse ({e}); nothing was written, the tool is unchanged")
    })
}

/// Whether `name` is a tool shipped in the binary.
///
/// Those are not the model's to rewrite: a bundled tool is part of what Nanna
/// IS (the file tools, `exec`, memory), boot re-extracts newer shipped
/// versions over it, and an edited one would run with the trust its shipped
/// version earned. Authoring tools may add tools, never alter these.
fn is_bundled(name: &str) -> bool {
    nanna_tools::skills::defaults::DEFAULT_SKILLS
        .iter()
        .any(|entry| entry.skill_name == name)
}

/// The refusal for a bundled `name`, addressed to the model.
fn refuse_bundled(name: &str) -> Result<(), String> {
    if is_bundled(name) {
        return Err(format!(
            "'{name}' ships with Nanna and cannot be changed by tool authoring. \
             Create a new tool under a different name instead."
        ));
    }
    Ok(())
}

/// Write a tool's source and its permissions, creating the directory.
///
/// The `permissions.json` is written here rather than left for
/// `ensure_permissions` at the next boot, so an authored tool carries a scope
/// chosen at creation time instead of one filled in later by a daemon restart.
fn write_tool(dir: &Path, source: &str) -> Result<PathBuf, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    let source_path = dir.join("tool.ts");
    std::fs::write(&source_path, source)
        .map_err(|e| format!("cannot write {}: {e}", source_path.display()))?;

    let permissions_path = dir.join("permissions.json");
    if !permissions_path.exists() {
        std::fs::write(
            &permissions_path,
            nanna_tools::skills::defaults::DEFAULT_PERMISSIONS_JSON,
        )
        .map_err(|e| format!("cannot write {}: {e}", permissions_path.display()))?;
    }
    Ok(source_path)
}

/// Register (or re-register) a tool from disk, reporting whether it took.
///
/// A failure here is not a failure of the write: the file is on disk and will
/// load at the next boot. The services say so rather than reporting success,
/// because "callable now" and "callable after a restart" are different answers
/// to the only question the caller asked.
async fn register_live(
    registry: &Weak<ToolRegistry>,
    services: &Arc<OnceLock<ServiceMap>>,
    dir: &Path,
) -> Result<(), String> {
    let Some(registry) = registry.upgrade() else {
        return Err("the tool registry is gone".to_string());
    };
    let empty = HashMap::new();
    let services = services.get().unwrap_or(&empty);
    let tool = nanna_tools::skills::load_skill_with_services(
        dir,
        services,
        Some(Arc::downgrade(&registry)),
    )
    .await
    .map_err(|e| e.to_string())?;
    registry.register_boxed(tool).await;
    Ok(())
}

/// Build the three tool-authoring services.
///
/// `services` is the map these services themselves live in, handed over as a
/// slot the caller fills once the map is complete. A tool authored at runtime
/// is loaded with the same services every bundled skill gets — without the
/// slot it would silently be the only tool in the daemon that cannot call one,
/// and the map cannot contain a closure that captures the finished map.
pub fn build_tool_authoring_services(
    tools_dir: PathBuf,
    registry: Weak<ToolRegistry>,
    services: Arc<OnceLock<ServiceMap>>,
) -> ServiceMap {
    let mut built: HashMap<String, ServiceFn> = HashMap::new();

    // --- tools.create ----------------------------------------------------
    let create_dir = tools_dir.clone();
    let create_registry = registry.clone();
    let create_services = services.clone();
    built.insert(
        "tools.create".to_string(),
        Arc::new(move |params: Value| {
            let tools_dir = create_dir.clone();
            let registry = create_registry.clone();
            let services = create_services.clone();
            Box::pin(async move {
                let name = string_arg(&params, "name")?;
                // A debug build loads bundled tools from the source tree, so
                // the tools dir may not hold one yet — and a user tool of the
                // same name would shadow it.
                refuse_bundled(&name)?;
                let source = string_arg(&params, "source")?;
                validate_source(&source)?;

                let dir = resolve_tool_dir(&tools_dir, &name)?;
                if dir.join("tool.ts").exists() || dir.join("tool.js").exists() {
                    return Err(format!(
                        "a tool named '{name}' already exists; use tools.update to change it"
                    ));
                }

                let path = write_tool(&dir, &source)?;
                info!(tool = %name, path = ?path, "Created a tool from tools.create");

                match register_live(&registry, &services, &dir).await {
                    Ok(()) => Ok(json!({
                        "path": path.to_string_lossy(),
                        "registered": true,
                    })),
                    Err(message) => {
                        warn!(tool = %name, %message, "Created tool did not register live");
                        Ok(json!({
                            "path": path.to_string_lossy(),
                            "registered": false,
                            "message": message,
                        }))
                    }
                }
            })
        }),
    );

    // --- tools.update ----------------------------------------------------
    let update_dir = tools_dir.clone();
    // The last consumer of each, so they are moved rather than cloned.
    let update_registry = registry;
    let update_services = services;
    built.insert(
        "tools.update".to_string(),
        Arc::new(move |params: Value| {
            let tools_dir = update_dir.clone();
            let registry = update_registry.clone();
            let services = update_services.clone();
            Box::pin(async move {
                let name = string_arg(&params, "name")?;
                refuse_bundled(&name)?;
                let dir = resolve_tool_dir(&tools_dir, &name)?;
                let source_path = existing_source_path(&dir).ok_or_else(|| {
                    format!("no tool named '{name}' on disk; use tools.create to add one")
                })?;

                // Whole-source replacement, or a single-occurrence edit against
                // the CURRENT file — never against what the caller remembers.
                let (updated, replacements) =
                    if let Some(whole) = params.get("source").and_then(Value::as_str) {
                        (whole.to_string(), 0_usize)
                    } else {
                        let old = string_arg(&params, "old_string")?;
                        let new = params
                            .get("new_string")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string();
                        let current = std::fs::read_to_string(&source_path)
                            .map_err(|e| format!("cannot read {}: {e}", source_path.display()))?;
                        (replace_exactly_once(&current, &old, &new)?, 1)
                    };
                validate_source(&updated)?;

                std::fs::write(&source_path, &updated)
                    .map_err(|e| format!("cannot write {}: {e}", source_path.display()))?;
                info!(tool = %name, path = ?source_path, "Updated a tool from tools.update");

                let mut result = json!({
                    "path": source_path.to_string_lossy(),
                    "replacements": replacements,
                    "registered": true,
                });
                if let Err(message) = register_live(&registry, &services, &dir).await {
                    warn!(tool = %name, %message, "Updated tool did not re-register live");
                    result["registered"] = json!(false);
                    result["message"] = json!(message);
                }
                Ok(result)
            })
        }),
    );

    // --- tools.list ------------------------------------------------------
    let list_dir = tools_dir;
    built.insert(
        "tools.list".to_string(),
        Arc::new(move |_params: Value| {
            let tools_dir = list_dir.clone();
            Box::pin(async move { Ok(list_user_tools(&tools_dir)) })
        }),
    );

    built
}

/// `tool.ts` or `tool.js`, in the loader's own precedence order.
fn existing_source_path(dir: &Path) -> Option<PathBuf> {
    let ts = dir.join("tool.ts");
    if ts.is_file() {
        return Some(ts);
    }
    let js = dir.join("tool.js");
    if js.is_file() { Some(js) } else { None }
}

/// Every skill directory that is not one of the bundled ones.
///
/// "User tool" means authored here rather than shipped in the binary, so the
/// embedded catalogue is the right thing to subtract — a name-list comparison,
/// not a guess from timestamps or a marker file that could go missing.
fn list_user_tools(tools_dir: &Path) -> Value {
    let Ok(entries) = std::fs::read_dir(tools_dir) else {
        return json!([]);
    };
    let mut listed = Vec::new();
    for entry in entries.flatten() {
        let dir = entry.path();
        if !dir.is_dir() {
            continue;
        }
        let Some(name) = dir.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if is_bundled(name) {
            continue;
        }
        let Some(source_path) = existing_source_path(&dir) else {
            continue;
        };
        let description = std::fs::read_to_string(&source_path)
            .ok()
            .and_then(|source| extract_manifest(&source))
            .map(|manifest| manifest.description)
            .unwrap_or_default();
        listed.push(json!({
            "name": name,
            "description": description,
            "enabled": true,
        }));
    }
    listed.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
    json!(listed)
}

/// A required string argument, refused by name rather than defaulted to empty.
fn string_arg(params: &Value, key: &str) -> Result<String, String> {
    let value = params
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or_default();
    if value.is_empty() {
        return Err(format!("`{key}` is required"));
    }
    Ok(value.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID_SOURCE: &str = r#"export default {
  name: "probe",
  description: "a probe tool",
  execute: function(input) { return "ok"; }
};
"#;

    fn authoring(dir: &Path) -> ServiceMap {
        build_tool_authoring_services(dir.to_path_buf(), Weak::new(), Arc::new(OnceLock::new()))
    }

    /// Shipped tools are not the model's to rewrite, through either verb.
    #[tokio::test]
    async fn a_bundled_tool_cannot_be_created_over_or_updated() {
        let dir = tempfile::tempdir().unwrap();
        let services = authoring(dir.path());
        let bundled = nanna_tools::skills::defaults::DEFAULT_SKILLS[0].skill_name;
        let create = services["tools.create"](json!({ "name": bundled, "source": VALID_SOURCE }))
            .await
            .expect_err("create over a bundled tool");
        assert!(create.contains("ships with Nanna"), "{create}");
        let update = services["tools.update"](json!({ "name": bundled, "source": VALID_SOURCE }))
            .await
            .expect_err("update of a bundled tool");
        assert!(update.contains("ships with Nanna"), "{update}");
        assert!(!dir.path().join(bundled).exists(), "nothing reached disk");
    }

    /// An edit that breaks the syntax is refused, and the working tool stays.
    /// It used to be written and re-registered, and fail only when called.
    #[tokio::test]
    async fn an_edit_that_breaks_the_syntax_leaves_the_working_tool() {
        let dir = tempfile::tempdir().unwrap();
        let services = authoring(dir.path());
        services["tools.create"](json!({ "name": "probe", "source": VALID_SOURCE }))
            .await
            .expect("create");
        let source_path = dir.path().join("probe").join("tool.ts");
        let before = std::fs::read_to_string(&source_path).unwrap();
        let broken = services["tools.update"](json!({
            "name": "probe",
            "old_string": "return \"ok\";",
            "new_string": "return \"ok\"; }}}",
        }))
        .await
        .expect_err("a syntax error is refused");
        assert!(broken.contains("does not parse"), "{broken}");
        let after = std::fs::read_to_string(&source_path).unwrap();
        assert_eq!(after, before, "the working tool is unchanged");
    }

    // --- name containment -------------------------------------------------

    #[test]
    fn a_traversing_name_is_refused_before_it_reaches_a_path() {
        let dir = tempfile::tempdir().unwrap();
        for name in ["../escape", "a/b", ".", "..", "Upper", "with space", ""] {
            assert!(
                resolve_tool_dir(dir.path(), name).is_err(),
                "{name:?} was accepted as a tool name",
            );
        }
    }

    #[test]
    fn a_valid_name_resolves_inside_the_tools_directory() {
        let dir = tempfile::tempdir().unwrap();
        let resolved = resolve_tool_dir(dir.path(), "my_tool_2").expect("a valid name resolves");
        assert!(resolved.starts_with(dir.path()));
        assert!(resolved.ends_with("my_tool_2"));
    }

    #[cfg(unix)]
    #[test]
    fn a_symlinked_tool_directory_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let elsewhere = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink(elsewhere.path(), dir.path().join("sneaky")).unwrap();

        let err = resolve_tool_dir(dir.path(), "sneaky")
            .expect_err("writing through a symlink must be refused");
        assert!(err.contains("symlink"), "unhelpful refusal: {err}");
    }

    // --- the edit grammar -------------------------------------------------

    #[test]
    fn an_edit_matching_once_applies() {
        let updated = replace_exactly_once("alpha beta", "beta", "gamma").unwrap();
        assert_eq!(updated, "alpha gamma");
    }

    #[test]
    fn an_edit_matching_nothing_is_refused() {
        let err = replace_exactly_once("alpha", "beta", "gamma").unwrap_err();
        assert!(err.contains("does not appear"), "unhelpful refusal: {err}");
    }

    #[test]
    fn an_ambiguous_edit_is_refused_rather_than_guessed() {
        let err = replace_exactly_once("beta beta", "beta", "gamma").unwrap_err();
        assert!(
            err.contains("2 times"),
            "the refusal must say how many: {err}"
        );
    }

    #[test]
    fn an_empty_old_string_is_refused() {
        assert!(replace_exactly_once("alpha", "", "x").is_err());
    }

    // --- source validation ------------------------------------------------

    #[test]
    fn a_source_without_a_default_export_is_refused_before_disk() {
        let err = validate_source("const x = 1;").unwrap_err();
        assert!(err.contains("export default"), "unhelpful refusal: {err}");
    }

    #[test]
    fn an_oversized_source_is_refused_with_the_engines_own_ceiling() {
        let huge = format!("{VALID_SOURCE}//{}", "x".repeat(TOOL_SOURCE_BYTES_MAX));
        let err = validate_source(&huge).unwrap_err();
        assert!(err.contains(&TOOL_SOURCE_BYTES_MAX.to_string()));
    }

    #[test]
    fn a_valid_source_passes() {
        assert!(validate_source(VALID_SOURCE).is_ok());
    }

    // --- writing ----------------------------------------------------------

    #[test]
    fn writing_a_tool_also_gives_it_a_permissions_file() {
        let dir = tempfile::tempdir().unwrap();
        let tool_dir = dir.path().join("probe");
        let path = write_tool(&tool_dir, VALID_SOURCE).unwrap();

        assert_eq!(std::fs::read_to_string(&path).unwrap(), VALID_SOURCE);
        let permissions = std::fs::read_to_string(tool_dir.join("permissions.json"))
            .expect("an authored tool must carry a scope chosen at creation time");
        assert!(
            !permissions.contains("\"*\"") || !permissions.contains("\"read\": [\"*\"]"),
            "an authored tool was given whole-filesystem scope",
        );
    }

    #[test]
    fn writing_a_tool_never_clobbers_an_authored_permissions_file() {
        let dir = tempfile::tempdir().unwrap();
        let tool_dir = dir.path().join("probe");
        std::fs::create_dir_all(&tool_dir).unwrap();
        let chosen = r#"{"read":[],"write":[],"run":false,"net":[],"env":false}"#;
        std::fs::write(tool_dir.join("permissions.json"), chosen).unwrap();

        write_tool(&tool_dir, VALID_SOURCE).unwrap();
        assert_eq!(
            std::fs::read_to_string(tool_dir.join("permissions.json")).unwrap(),
            chosen,
        );
    }

    // --- listing ----------------------------------------------------------

    #[test]
    fn listing_returns_authored_tools_and_skips_bundled_ones() {
        let dir = tempfile::tempdir().unwrap();
        write_tool(&dir.path().join("mine"), VALID_SOURCE).unwrap();
        // `exec` is bundled, so a directory of that name is not a user tool.
        write_tool(&dir.path().join("exec"), VALID_SOURCE).unwrap();

        let listed = list_user_tools(dir.path());
        let names: Vec<&str> = listed
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|t| t["name"].as_str())
            .collect();
        assert_eq!(names, vec!["mine"], "bundled skills leaked into the list");
        assert_eq!(listed[0]["description"], "a probe tool");
    }

    #[test]
    fn listing_a_missing_directory_is_empty_rather_than_an_error() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(list_user_tools(&dir.path().join("nope")), json!([]));
    }

    // --- argument handling ------------------------------------------------

    #[test]
    fn a_missing_argument_is_refused_by_name() {
        let err = string_arg(&json!({}), "name").unwrap_err();
        assert!(
            err.contains("name"),
            "the refusal must name the argument: {err}"
        );
        assert!(string_arg(&json!({ "name": "   " }), "name").is_err());
        assert_eq!(
            string_arg(&json!({ "name": " ok " }), "name").unwrap(),
            "ok"
        );
    }
}
