//! Tool directory resolution, loading, and bootstrapping helpers.
//!
//! Tools are loaded dynamically from the filesystem at runtime.
//! Default skills are embedded at compile time (via build.rs) and extracted
//! to the tools directory on first run in release builds.
//!
//! # Resolution Order
//!
//! 1. `NANNA_TOOLS_DIR` environment variable (for development)
//! 2. `config_tools_dir` (explicit configuration)
//! 3. The source tree's `default-skills/` (debug builds only)

use std::path::{Path, PathBuf};

// Include the build-script-generated embedded skills (all default-skills/ files).
include!(concat!(env!("OUT_DIR"), "/embedded_skills.rs"));

/// Parse a semver version string into (major, minor, patch) tuple.
/// Returns None if the string is not a valid semver triple.
// Release builds refresh extracted skills by version; tests cover it everywhere.
#[cfg(any(not(debug_assertions), test))]
fn parse_semver(v: &str) -> Option<(u64, u64, u64)> {
    // Strip leading 'v' if present
    let v = v.strip_prefix('v').unwrap_or(v);
    // Strip pre-release/build metadata (everything after - or +)
    let v = v.split(['-', '+']).next().unwrap_or(v);
    let parts: Vec<&str> = v.split('.').collect();
    if parts.len() != 3 {
        return None;
    }
    Some((
        parts[0].parse().ok()?,
        parts[1].parse().ok()?,
        parts[2].parse().ok()?,
    ))
}

/// Returns true if `embedded` version is strictly greater than `installed`.
// Release builds refresh extracted skills by version; tests cover it everywhere.
#[cfg(any(not(debug_assertions), test))]
fn is_newer_version(embedded: &str, installed: &str) -> bool {
    match (parse_semver(embedded), parse_semver(installed)) {
        (Some(e), Some(i)) => e > i,
        // If either fails to parse, don't overwrite
        _ => false,
    }
}

/// Extract the version field from a tool.ts source string.
/// Looks for `version: "x.y.z"` or `version: 'x.y.z'` in the source.
// Release builds refresh extracted skills by version; tests cover it everywhere.
#[cfg(any(not(debug_assertions), test))]
fn extract_version_from_source(source: &str) -> Option<String> {
    // Reuse the same pattern as extract_string_field in nanna-scripting
    let patterns = [
        r#"version: ""#,
        r"version: '",
        r#"version:""#,
        r"version:'",
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

/// A `permissions.json` Nanna once shipped and has since replaced, per skill.
///
/// Installed permission files are never overwritten, because the user may
/// have edited them — so a scope Nanna got wrong stayed wrong on every
/// existing install. These skills shipped home-only (`~`) scopes, but they
/// read the files of the user's workspace, which is often outside home (a
/// second drive, `/mnt/…`, `D:\…`): every such file was refused, and
/// `lift_invariant` answered "nothing was lifted" while the rule stood. A file
/// still byte-for-byte (as parsed JSON) what was shipped was never edited, so
/// it is replaced; any other is the user's and is kept.
// Release builds refresh extracted skills; tests cover it everywhere.
#[cfg(any(not(debug_assertions), test))]
const SUPERSEDED_PERMISSIONS: &[(&str, &str)] = &[
    ("analyze_image", HOME_SCOPED_RUN_NET_ENV),
    ("describe_image", HOME_SCOPED_RUN_NET_ENV),
    ("lift_invariant", HOME_SCOPED_RUN_NET_ENV),
    ("ocr", HOME_SCOPED_RUN_NET_ENV),
    ("read_pdf", HOME_SCOPED_RUN_NET_ENV),
    ("transcribe", HOME_SCOPED_RUN_NET_ENV),
    (
        "todo",
        r#"{"read":["~"],"write":["~"],"run":false,"net":[],"env":false}"#,
    ),
];

#[cfg(any(not(debug_assertions), test))]
const HOME_SCOPED_RUN_NET_ENV: &str =
    r#"{"read":["~"],"write":["~"],"run":true,"net":["*"],"env":true}"#;

/// Whether `installed` is a superseded scope of `skill`'s that `embedded`
/// corrects — never a file the user changed, never a no-op rewrite.
#[cfg(any(not(debug_assertions), test))]
fn replaces_superseded_permissions(skill: &str, installed: &str, embedded: &str) -> bool {
    let parse = |text: &str| serde_json::from_str::<serde_json::Value>(text).ok();
    let (Some(installed), Some(embedded)) = (parse(installed), parse(embedded)) else {
        return false;
    };
    installed != embedded
        && SUPERSEDED_PERMISSIONS
            .iter()
            .any(|(name, shipped)| *name == skill && parse(shipped).as_ref() == Some(&installed))
}

/// Directory name, under this crate, holding the bundled JS/TS skills.
/// Only the debug-build source-tree fallback reads it.
#[cfg(debug_assertions)]
const DEV_SKILLS_DIR_NAME: &str = "default-skills";

/// In debug builds, fall back to the source tree's `default-skills` directory.
/// Resolved relative to the `nanna-tools` crate, which `CARGO_MANIFEST_DIR`
/// pins at compile time.
///
/// This **joins** rather than concatenating a separator. It used to be
/// `concat!(env!("CARGO_MANIFEST_DIR"), "\\default-skills")`, which is a path
/// only on Windows: on Linux and macOS the backslash is an ordinary filename
/// character, so the constant named a file that has never existed, `is_dir()`
/// was false, and `resolve_tools_dir` fell through to `None` — silently, in the
/// one build profile where it is the *only* source of skills.
#[cfg(debug_assertions)]
#[must_use]
pub fn dev_tools_dir() -> Option<PathBuf> {
    Some(Path::new(env!("CARGO_MANIFEST_DIR")).join(DEV_SKILLS_DIR_NAME))
}

/// Release builds extract the embedded skills instead of reading the source
/// tree, so there is no development directory to fall back to.
#[cfg(not(debug_assertions))]
#[must_use]
pub fn dev_tools_dir() -> Option<PathBuf> {
    None
}

/// Scope written into a tool directory whose author declared none.
///
/// This is a grant made on somebody else's behalf, so it is deliberately not
/// the widest one. It used to be `read: ["*"], write: ["*"]` — whole-filesystem
/// access to any tool that simply forgot the file, which is how `edit_tool`
/// (the tool whose job is rewriting other tools' source) ran unscoped while its
/// sibling `create_tool` was confined to home.
///
/// `~` is the narrowest scope the reviewed corpus shows is commonly sufficient:
/// of the 44 bundled skills, **30 declare `~` and 13 declare `*`**, so home is
/// the modal authored choice and a tool that genuinely needs more now has to say
/// so. `ScriptedToolWrapper::from_file` expands `~` to the real home directory
/// at load time, so the scope is enforced, not decorative.
///
/// `run`, `net` and `env` are left as they were: 40 of the 44 declare exactly
/// `run: true, net: ["*"], env: true`, so those are not over-grants relative to
/// the corpus. (Narrowing `env` is a separate decision with a different
/// rationale — it has no scope vocabulary, only on/off — and is tracked in the
/// roadmap rather than ridden along here.)
///
/// **Existing installs are unaffected.** [`ensure_permissions`] writes only when
/// the file is absent, and the grant it writes is persisted, so a directory that
/// already received the old wide default keeps it until somebody edits it. The
/// narrowing reaches newly-created undeclared tools only.
pub const DEFAULT_PERMISSIONS_JSON: &str = r#"{
    "read": ["~"],
    "write": ["~"],
    "run": true,
    "net": ["*"],
    "env": true
}"#;

/// The scope string meaning "the whole filesystem".
///
/// Named because [`default_permissions_are_home_bounded`] and the test that pins
/// the decision must check for the same thing, not two spellings of it.
const FILESYSTEM_WILDCARD: &str = "*";

/// Whether [`DEFAULT_PERMISSIONS_JSON`]'s filesystem scope stays inside home.
///
/// Pure and cheap, so the always-on guard in [`ensure_permissions`] and the test
/// that pins this decision assert the identical property. A scope is home-bounded
/// when it is non-empty (an empty list would break every undeclared tool rather
/// than confine it) and names no `FILESYSTEM_WILDCARD`.
#[must_use]
pub fn default_permissions_are_home_bounded() -> bool {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(DEFAULT_PERMISSIONS_JSON) else {
        return false;
    };
    ["read", "write"].iter().all(|field| {
        value
            .get(field)
            .and_then(serde_json::Value::as_array)
            .is_some_and(|scopes| {
                !scopes.is_empty()
                    && scopes
                        .iter()
                        .all(|scope| scope.as_str() != Some(FILESYSTEM_WILDCARD))
            })
    })
}

/// Resolve the tools directory from environment, config, or the dev fallback.
///
/// Resolution order:
/// 1. `NANNA_TOOLS_DIR` environment variable
/// 2. `config_tools_dir` parameter (from config file)
/// 3. [`dev_tools_dir`] — the source tree's `default-skills/`, debug builds only
///
/// Returns `None` if none of those resolve; the caller decides the fallback
/// (the daemon uses `{data_dir}/tools/`). There is no `fallback` parameter —
/// the doc comment claimed one for a signature that never had it.
pub fn resolve_tools_dir(config_tools_dir: Option<&Path>) -> Option<PathBuf> {
    // 1. Environment variable (highest priority — useful for development)
    if let Ok(env_dir) = std::env::var("NANNA_TOOLS_DIR") {
        let p = PathBuf::from(env_dir);
        if p.is_dir() {
            tracing::info!("Using tools directory from NANNA_TOOLS_DIR: {:?}", p);
            return Some(p);
        }
        tracing::warn!("NANNA_TOOLS_DIR set but directory does not exist: {:?}", p);
    }

    // 2. Explicit config value
    if let Some(dir) = config_tools_dir {
        if dir.is_dir() {
            tracing::info!("Using tools directory from config: {:?}", dir);
            return Some(dir.to_path_buf());
        }
        // Return it even if it doesn't exist yet (caller may create it)
        tracing::info!("Using configured tools_dir (may not exist yet): {:?}", dir);
        return Some(dir.to_path_buf());
    }

    // 3. Development fallback: source tree's default-skills directory
    if let Some(dev_dir) = dev_tools_dir() {
        if dev_dir.is_dir() {
            tracing::info!("Using development tools directory: {:?}", dev_dir);
            return Some(dev_dir);
        }
        tracing::warn!(
            "development tools directory does not exist: {:?} — no JS/TS skills \
             will load unless NANNA_TOOLS_DIR or [tools].tools_dir is set",
            dev_dir
        );
    }

    None
}

/// Bootstrap the tools directory by copying bundled default skills into it.
///
/// In debug builds this is a no-op (we load directly from the source tree).
/// In release builds, default skills are embedded at compile time and extracted
/// to the target directory on first run.
///
/// Returns the number of skills bootstrapped (0 if already present).
pub fn bootstrap_default_skills(tools_dir: &Path) -> usize {
    // Create the tools directory if it doesn't exist
    if !tools_dir.exists() {
        if let Err(e) = std::fs::create_dir_all(tools_dir) {
            tracing::error!("Failed to create tools directory {:?}: {}", tools_dir, e);
            return 0;
        }
        tracing::info!("Created tools directory: {:?}", tools_dir);
    }

    // In debug builds, tools are loaded directly from the source tree via dev_tools_dir().
    // Only bootstrap in release builds where we need to populate {data_dir}/tools/.
    #[cfg(debug_assertions)]
    {
        let _ = tools_dir; // suppress unused warning
        0
    }

    #[cfg(not(debug_assertions))]
    {
        let mut count = 0;
        for entry in DEFAULT_SKILLS {
            let tool_dir = tools_dir.join(entry.skill_name);
            let target = tool_dir.join(entry.file_name);

            if target.exists() {
                // Only overwrite tool.ts files (not permissions.json etc.) when
                // the embedded version is strictly newer than the installed one.
                if entry.file_name == "tool.ts" || entry.file_name == "tool.js" {
                    let embedded_ver = extract_version_from_source(entry.content);
                    let installed_source = std::fs::read_to_string(&target).unwrap_or_default();
                    let installed_ver = extract_version_from_source(&installed_source);

                    match (&embedded_ver, &installed_ver) {
                        (Some(e), Some(i)) if is_newer_version(e, i) => {
                            tracing::info!(
                                "Upgrading default skill {}/{}: {} → {}",
                                entry.skill_name,
                                entry.file_name,
                                i,
                                e
                            );
                            // Fall through to write
                        }
                        (Some(e), None) => {
                            // Installed tool has no version — embedded does. Upgrade it.
                            tracing::info!(
                                "Upgrading unversioned skill {}/{} to {}",
                                entry.skill_name,
                                entry.file_name,
                                e
                            );
                            // Fall through to write
                        }
                        _ => {
                            // Same version, older embedded, or both unversioned — skip
                            continue;
                        }
                    }
                } else if entry.file_name == "permissions.json"
                    && std::fs::read_to_string(&target).is_ok_and(|installed| {
                        replaces_superseded_permissions(entry.skill_name, &installed, entry.content)
                    })
                {
                    tracing::info!(
                        "Upgrading {}/permissions.json: the installed scope is one Nanna shipped \
                         and has since corrected",
                        entry.skill_name
                    );
                    // Fall through to write
                } else {
                    // Non-tool files (permissions.json etc.) — don't overwrite:
                    // the user may have edited them.
                    continue;
                }
            }

            if let Err(e) = std::fs::create_dir_all(&tool_dir) {
                tracing::warn!("Failed to create skill directory {:?}: {}", tool_dir, e);
                continue;
            }

            if let Err(e) = std::fs::write(&target, entry.content) {
                tracing::warn!("Failed to write {:?}: {}", target, e);
                continue;
            }

            tracing::info!(
                "Bootstrapped default skill: {}/{}",
                entry.skill_name,
                entry.file_name
            );
            count += 1;
        }

        if count > 0 {
            // Ensure permissions are set for newly created skills. Every
            // bundled skill ships its own file, so a non-zero count here means
            // one regressed — `ensure_permissions` announces which.
            let granted_count = ensure_permissions(tools_dir);
            tracing::info!(
                granted_count,
                "Bootstrapped {} default skills into {:?}",
                count,
                tools_dir
            );
        }

        count
    }
}

/// Write [`DEFAULT_PERMISSIONS_JSON`] into every tool subdirectory lacking a
/// `permissions.json`, and report how many grants that made.
///
/// A directory that ships its own file is never touched, so this only ever
/// speaks for an author who declared nothing. Each such grant is announced at
/// `warn` naming the tool: a permission handed out on somebody's behalf has to
/// be reviewable, and before this it was written silently with nothing but the
/// file itself left to notice.
///
/// Returns the count of directories granted, so a caller can say so at boot and
/// so the announcement is testable without scraping logs.
///
/// # Panics
///
/// Panics if [`DEFAULT_PERMISSIONS_JSON`] is not home-bounded (see
/// [`default_permissions_are_home_bounded`]) — a defect in this build's
/// constant, never in the directory. The check stays on in release builds so a
/// fail-open default can never be written to disk.
pub fn ensure_permissions(tools_dir: &Path) -> usize {
    assert!(
        default_permissions_are_home_bounded(),
        "DEFAULT_PERMISSIONS_JSON grants whole-filesystem read or write to a \
         tool whose author declared nothing — exactly the fail-open this \
         default exists to prevent",
    );

    let Ok(entries) = std::fs::read_dir(tools_dir) else {
        return 0;
    };

    let mut granted_count: usize = 0;
    let mut directory_count: usize = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        directory_count += 1;

        let perms = path.join("permissions.json");
        if perms.exists() {
            continue;
        }

        if let Err(e) = std::fs::write(&perms, DEFAULT_PERMISSIONS_JSON) {
            tracing::debug!("Could not write permissions.json to {:?}: {}", path, e);
            continue;
        }

        granted_count += 1;
        tracing::warn!(
            tool = ?path.file_name().unwrap_or(path.as_os_str()),
            "Tool declared no permissions.json; wrote the home-scoped default \
             (read/write ~, run, net *, env). Declare the file to choose a \
             different scope.",
        );
    }

    debug_assert!(
        granted_count <= directory_count,
        "granted more permission files than there were tool directories",
    );
    granted_count
}

/// Load the `discover_tools` skill source from a tools directory.
///
/// Returns `None` if the file doesn't exist.
pub fn load_discover_tools_source(tools_dir: &Path) -> Option<String> {
    let path = tools_dir.join("discover_tools").join("tool.ts");
    match std::fs::read_to_string(&path) {
        Ok(source) => Some(source),
        Err(e) => {
            tracing::warn!("Could not load discover_tools from {:?}: {}", path, e);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An installed `permissions.json` equal to a scope Nanna shipped and has
    /// since corrected is replaced; one the user edited, another skill's, or a
    /// file already current is left alone.
    #[test]
    fn only_an_unedited_superseded_permissions_file_is_replaced() {
        let current = r#"{"read":["*"],"write":["*"],"run":true,"net":["*"],"env":true}"#;
        let shipped = "{\n  \"read\": [\"~\"], \"write\": [\"~\"],\n  \"run\": true, \"net\": [\"*\"], \"env\": true\n}";
        assert!(replaces_superseded_permissions(
            "read_pdf", shipped, current
        ));
        let edited =
            r#"{"read":["~","/mnt/data"],"write":["~"],"run":true,"net":["*"],"env":true}"#;
        assert!(
            !replaces_superseded_permissions("read_pdf", edited, current),
            "the user's edit stands"
        );
        assert!(
            !replaces_superseded_permissions("recall", shipped, current),
            "not a corrected skill"
        );
        assert!(
            !replaces_superseded_permissions("read_pdf", current, current),
            "already current"
        );
        assert!(!replaces_superseded_permissions(
            "read_pdf", "not json", current
        ));
    }

    /// Every superseded entry names a bundled skill whose shipped file now
    /// differs from it — otherwise the table would rewrite a file to itself
    /// or name a skill that no longer exists.
    #[test]
    fn every_superseded_scope_is_corrected_in_the_bundle() {
        for (skill, shipped) in SUPERSEDED_PERMISSIONS {
            let embedded = DEFAULT_SKILLS
                .iter()
                .find(|e| e.skill_name == *skill && e.file_name == "permissions.json")
                .unwrap_or_else(|| panic!("{skill} ships no permissions.json"));
            assert!(
                replaces_superseded_permissions(skill, shipped, embedded.content),
                "{skill}: the bundle still ships the superseded scope"
            );
        }
    }
    use tempfile::tempdir;

    /// Serializes tests that mutate `NANNA_TOOLS_DIR`.
    ///
    /// The environment is process-global while `cargo test` runs test functions on
    /// parallel threads, so two tests touching the same variable race: without this
    /// lock, `test_resolve_tools_dir_from_config`'s `remove_var` could land between
    /// the other test's `set_var` and its `resolve_tools_dir(None)` call, which then
    /// fell through to `dev_tools_dir()` and failed with the source-tree
    /// `default-skills` path instead of the temp dir.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Restores `NANNA_TOOLS_DIR` to its pre-test value on drop, so a panicking test
    /// cannot leak state into the next one (and so a developer's real env survives).
    struct EnvGuard {
        previous: Option<std::ffi::OsString>,
        _lock: std::sync::MutexGuard<'static, ()>,
    }

    impl EnvGuard {
        fn set(value: Option<&std::path::Path>) -> Self {
            // A poisoned lock only means some other test panicked; the env is still
            // ours to restore, so recover rather than cascade the failure.
            let lock = ENV_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            let previous = std::env::var_os("NANNA_TOOLS_DIR");
            match value {
                // SAFETY: every writer of this variable holds ENV_LOCK, so no other
                // thread reads or writes it concurrently.
                Some(path) => unsafe { std::env::set_var("NANNA_TOOLS_DIR", path) },
                None => unsafe { std::env::remove_var("NANNA_TOOLS_DIR") },
            }
            Self {
                previous,
                _lock: lock,
            }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            // SAFETY: the lock is still held until this guard is fully dropped.
            match self.previous.take() {
                Some(value) => unsafe { std::env::set_var("NANNA_TOOLS_DIR", value) },
                None => unsafe { std::env::remove_var("NANNA_TOOLS_DIR") },
            }
        }
    }

    #[test]
    fn test_resolve_tools_dir_from_env() {
        let dir = tempdir().unwrap();
        let path = dir.path().to_path_buf();

        let _env = EnvGuard::set(Some(&path));
        let resolved = resolve_tools_dir(None);

        assert_eq!(resolved, Some(path));
    }

    #[test]
    fn test_resolve_tools_dir_from_config() {
        let dir = tempdir().unwrap();
        let path = dir.path().to_path_buf();

        // The explicit config path must win even with no env var set.
        let _env = EnvGuard::set(None);
        let resolved = resolve_tools_dir(Some(&path));

        assert_eq!(resolved, Some(path));
    }

    /// The env var must take precedence over an explicit config path — the ordering
    /// `resolve_tools_dir` documents. Previously untested, and only safely testable
    /// now that env mutation is serialized.
    #[test]
    fn env_overrides_config_tools_dir() {
        let env_dir = tempdir().unwrap();
        let config_dir = tempdir().unwrap();

        let _env = EnvGuard::set(Some(env_dir.path()));
        let resolved = resolve_tools_dir(Some(config_dir.path()));

        assert_eq!(resolved, Some(env_dir.path().to_path_buf()));
        assert_ne!(resolved, Some(config_dir.path().to_path_buf()));
    }

    /// The development fallback must name a directory that **exists**.
    ///
    /// It did not on Linux or macOS: the constant behind it concatenated a
    /// literal `\\` onto `CARGO_MANIFEST_DIR`, which is a separator only on
    /// Windows. Everywhere else it produced a filename containing a backslash,
    /// `is_dir()` was false, and `resolve_tools_dir` returned `None` without
    /// saying so. Debug builds have no other source of JS/TS skills — release
    /// builds extract the embedded copies, debug builds read the source tree —
    /// so on Linux a developer daemon with no `NANNA_TOOLS_DIR` and no
    /// `[tools].tools_dir` ran with the Rust built-ins and nothing else, while
    /// the same commit on Windows loaded every skill.
    ///
    /// Asserting `is_dir()` rather than the spelling is what makes this
    /// portable: it is the property every platform needs and the one that was
    /// actually false.
    #[cfg(debug_assertions)]
    #[test]
    fn dev_tools_dir_names_a_real_directory() {
        let dir = dev_tools_dir().expect("debug builds must have a dev tools dir");
        assert!(
            dir.is_dir(),
            "dev tools dir {dir:?} does not exist — a hardcoded path separator?",
        );
        // Not merely *a* directory: the one holding the bundled skills.
        assert!(
            dir.join("discover_tools").join("tool.ts").is_file(),
            "{dir:?} exists but holds no discover_tools skill",
        );
    }

    /// With neither the env var nor a config path, the dev fallback is what
    /// `resolve_tools_dir` must return — the third documented step, and the
    /// step that silently produced `None` off Windows.
    #[cfg(debug_assertions)]
    #[test]
    fn resolve_falls_back_to_the_dev_tools_dir() {
        let _env = EnvGuard::set(None);
        assert_eq!(resolve_tools_dir(None), dev_tools_dir());
        assert!(
            resolve_tools_dir(None).is_some(),
            "debug builds resolved no tools directory at all — every JS/TS \
             skill would be missing",
        );
    }

    #[test]
    fn test_ensure_permissions() {
        let dir = tempdir().unwrap();
        let tool_dir = dir.path().join("my_tool");
        std::fs::create_dir_all(&tool_dir).unwrap();
        std::fs::write(tool_dir.join("tool.ts"), "// test").unwrap();

        // No permissions.json yet
        assert!(!tool_dir.join("permissions.json").exists());

        let granted_count = ensure_permissions(dir.path());

        // Now it should exist
        assert!(tool_dir.join("permissions.json").exists());
        assert_eq!(
            granted_count, 1,
            "the one undeclared tool was not counted as a grant",
        );
    }

    /// The decision this default encodes: a tool whose author declared nothing
    /// is confined to home, not handed the filesystem.
    ///
    /// Asserted on the bytes actually written rather than on the constant, so
    /// re-widening the constant fails here even if the guard is removed.
    #[test]
    fn undeclared_tools_are_granted_home_scope_not_the_filesystem() {
        let dir = tempdir().unwrap();
        let tool_dir = dir.path().join("undeclared_tool");
        std::fs::create_dir_all(&tool_dir).unwrap();

        assert_eq!(ensure_permissions(dir.path()), 1);

        let written = std::fs::read_to_string(tool_dir.join("permissions.json")).unwrap();
        let perms: serde_json::Value = serde_json::from_str(&written)
            .expect("the default this writes must deserialize, or the loader ignores it");

        for field in ["read", "write"] {
            let scopes = perms[field]
                .as_array()
                .unwrap_or_else(|| panic!("{field} is not an array"));
            assert!(
                !scopes.is_empty(),
                "{field} is empty, which denies an undeclared tool everything \
                 instead of confining it",
            );
            assert!(
                scopes.iter().all(|s| s.as_str() != Some("*")),
                "{field} grants the whole filesystem to a tool that declared \
                 nothing — the fail-open this default exists to prevent",
            );
        }

        assert!(default_permissions_are_home_bounded());
    }

    /// A directory that ships its own file keeps it verbatim. This is what
    /// makes the narrowing safe for existing installs: the grant is persisted,
    /// so a directory that already holds the old wide default is not rewritten.
    #[test]
    fn a_declared_permissions_file_is_never_overwritten() {
        let dir = tempdir().unwrap();
        let tool_dir = dir.path().join("declared_tool");
        std::fs::create_dir_all(&tool_dir).unwrap();

        let chosen = r#"{"read":["*"],"write":[],"run":false,"net":[],"env":false}"#;
        let perms_path = tool_dir.join("permissions.json");
        std::fs::write(&perms_path, chosen).unwrap();

        assert_eq!(
            ensure_permissions(dir.path()),
            0,
            "a tool that declared its own scope was counted as a grant",
        );
        assert_eq!(
            std::fs::read_to_string(&perms_path).unwrap(),
            chosen,
            "an authored permissions.json was overwritten",
        );
    }

    /// A file is a file, not a tool directory — walking it must not panic and
    /// must not count.
    #[test]
    fn ensure_permissions_ignores_loose_files() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("README.md"), "not a tool").unwrap();

        assert_eq!(ensure_permissions(dir.path()), 0);
        assert!(!dir.path().join("permissions.json").exists());
    }

    /// An unreadable directory yields no grants rather than a panic.
    #[test]
    fn ensure_permissions_on_a_missing_directory_grants_nothing() {
        let dir = tempdir().unwrap();
        assert_eq!(ensure_permissions(&dir.path().join("does_not_exist")), 0);
    }

    #[test]
    fn test_load_discover_tools_source() {
        let dir = tempdir().unwrap();
        let dt_dir = dir.path().join("discover_tools");
        std::fs::create_dir_all(&dt_dir).unwrap();
        std::fs::write(
            dt_dir.join("tool.ts"),
            "export default { name: 'discover_tools' }",
        )
        .unwrap();

        let source = load_discover_tools_source(dir.path());
        assert!(source.is_some());
        assert!(source.unwrap().contains("discover_tools"));
    }

    #[test]
    fn test_load_discover_tools_source_missing() {
        let dir = tempdir().unwrap();
        let source = load_discover_tools_source(dir.path());
        assert!(source.is_none());
    }

    #[test]
    fn test_parse_semver() {
        assert_eq!(parse_semver("0.1.0"), Some((0, 1, 0)));
        assert_eq!(parse_semver("1.2.3"), Some((1, 2, 3)));
        assert_eq!(parse_semver("v1.0.0"), Some((1, 0, 0)));
        assert_eq!(parse_semver("1.0.0-beta.1"), Some((1, 0, 0)));
        assert_eq!(parse_semver("1.0.0+build.123"), Some((1, 0, 0)));
        assert_eq!(parse_semver("not-a-version"), None);
        assert_eq!(parse_semver("1.0"), None);
        assert_eq!(parse_semver(""), None);
    }

    #[test]
    fn test_is_newer_version() {
        assert!(is_newer_version("0.2.0", "0.1.0"));
        assert!(is_newer_version("1.0.0", "0.9.9"));
        assert!(is_newer_version("0.1.1", "0.1.0"));
        assert!(!is_newer_version("0.1.0", "0.1.0")); // same = not newer
        assert!(!is_newer_version("0.1.0", "0.2.0")); // older
        assert!(!is_newer_version("bad", "0.1.0")); // unparseable
    }

    #[test]
    fn test_extract_version_from_source() {
        let source = r#"export default {
  name: "exec",
  version: "0.1.0",
  description: "Run stuff",
}"#;
        assert_eq!(
            extract_version_from_source(source),
            Some("0.1.0".to_string())
        );

        let source_single = r"export default {
  name: 'exec',
  version: '1.2.3',
}";
        assert_eq!(
            extract_version_from_source(source_single),
            Some("1.2.3".to_string())
        );

        let no_version = r#"export default { name: "exec" }"#;
        assert_eq!(extract_version_from_source(no_version), None);
    }
}
