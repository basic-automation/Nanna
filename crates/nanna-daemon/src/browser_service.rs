//! The four `browser.*` services.
//!
//! `browser_action`, `browser_evaluate`, `browser_extract` and
//! `browser_screenshot` declare these and nothing registered them, so all four
//! were withheld at every boot (found by `tests/skill_services_are_registered.rs`).
//!
//! **These are not thin wrappers.** The skills and `BrowserManager` were written
//! against each other and never run together, and their vocabularies disagreed:
//! `evaluate` was sent `expression` and read `script`; `action` is sent `value`
//! and `delay_ms` and reads `text`/`key`/`wait_ms`; `action`'s advertised enum
//! offered `scroll` and `navigate`, neither implemented; and `extract` is sent
//! an `attribute` for which there was no path at all. The manager gained the
//! three missing capabilities, and the translation between the two vocabularies
//! lives here, at the boundary, where the service speaks the skill's language.
//!
//! **Registered only when a browser is actually present.** The detected
//! executable is passed as `BrowserConfig.executable_path`, so the binary that
//! is probed and the binary that is launched are the same one by construction —
//! a gate that tests something other than what it guards is not a gate.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use nanna_scripting::ServiceFn;
use nanna_tools::{BrowserConfig, BrowserManager};
use serde_json::{Value, json};
use tracing::{info, warn};

/// Chromium-family executables to look for, in order.
///
/// The CDP backend drives Chromium only, so this is the set it can actually
/// use. Ordered plain-Chromium-first because it is the lighter install and the
/// one a headless host is more likely to have on purpose.
const CHROMIUM_CANDIDATES: &[&str] = &[
    "chromium",
    "chromium-browser",
    "google-chrome-stable",
    "google-chrome",
    "chrome",
];

/// Environment variables that name a browser explicitly, checked before the
/// candidate list. Both are what `chromiumoxide` itself honours.
const BROWSER_PATH_VARS: &[&str] = &["CHROME", "CHROME_PATH"];

/// Subdirectory of the data dir that page screenshots are written to.
const SCREENSHOT_DIR_NAME: &str = "screenshots";

/// Find a usable Chromium executable, or `None`.
///
/// An explicit `CHROME`/`CHROME_PATH` wins over the search, and is honoured
/// only if it points at something that exists — a stale override should fail
/// the gate here rather than at launch.
#[must_use]
pub fn find_browser_executable() -> Option<PathBuf> {
    for var in BROWSER_PATH_VARS {
        if let Ok(value) = std::env::var(var) {
            let path = PathBuf::from(value.trim());
            if path.is_file() {
                return Some(path);
            }
            if !value.trim().is_empty() {
                warn!(%var, path = ?path, "browser path override does not exist; ignoring it");
            }
        }
    }
    CHROMIUM_CANDIDATES
        .iter()
        .find_map(|name| which_on_path(name))
}

/// Resolve a bare executable name against `PATH`.
///
/// Hand-rolled rather than a dependency: it is a `PATH` split and an
/// `is_file` check, and adding a crate for that is not a trade worth making.
fn which_on_path(name: &str) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    std::env::split_paths(&path_var)
        .map(|dir| dir.join(name))
        .find(|candidate| candidate.is_file())
}

/// Where the services' browser is handed back to the daemon, so its shutdown
/// can close it. Nothing used to: Chromium was killed by the process exit
/// and its profile directory was left in the temp dir.
pub type BrowserSlot = Arc<std::sync::OnceLock<Arc<BrowserManager>>>;

/// Close the browser in `slot`, if one was ever built. Bounded by the
/// backend's own close (a graceful exit, then a kill, each at most
/// `nanna_browser::cdp::BROWSER_CLOSE_GRACE`).
pub async fn close_browser(slot: &BrowserSlot) {
    let Some(manager) = slot.get() else {
        return;
    };
    if let Err(e) = manager.close().await {
        warn!(error = %e, "closing the browser failed");
    }
}

/// Build the four `browser.*` services, or an empty map with no browser.
/// The browser behind them is put in `slot` for the daemon's shutdown.
pub fn build_browser_services(
    data_dir: &std::path::Path,
    slot: &BrowserSlot,
) -> HashMap<String, ServiceFn> {
    let Some(executable) = find_browser_executable() else {
        info!(
            "No Chromium-family browser found; the four browser.* services stay \
             unregistered and the browser_* skills stay withheld. Install \
             chromium or google-chrome, or set CHROME to its path."
        );
        return HashMap::new();
    };

    let config = BrowserConfig {
        // Probe and launch must agree: pass the very binary that passed the gate.
        executable_path: Some(executable.to_string_lossy().to_string()),
        ..BrowserConfig::default()
    };
    let manager = match BrowserManager::from_config(config) {
        Ok(manager) => Arc::new(manager),
        Err(e) => {
            warn!(error = %e, "Browser backend could not be created; browser.* stays unregistered");
            return HashMap::new();
        }
    };

    if slot.set(Arc::clone(&manager)).is_err() {
        // Services built twice: the second browser is closed by nobody. The
        // daemon builds them once; say so loudly if that ever changes.
        warn!("browser services were built twice; only the first browser is closed at shutdown");
    }
    info!(executable = ?executable, "Registering browser.action / evaluate / extract / screenshot");
    let screenshot_dir = data_dir.join(SCREENSHOT_DIR_NAME);

    let mut services: HashMap<String, ServiceFn> = HashMap::new();
    services.insert(
        "browser.extract".to_string(),
        extract_service(manager.clone()),
    );
    services.insert(
        "browser.evaluate".to_string(),
        evaluate_service(manager.clone()),
    );
    services.insert(
        "browser.action".to_string(),
        action_service(manager.clone()),
    );
    services.insert(
        "browser.screenshot".to_string(),
        screenshot_service(manager, screenshot_dir),
    );
    debug_assert_eq!(
        services.len(),
        4,
        "all four browser services register together"
    );
    services
}

/// The `url` every browser service needs, refused by name when absent.
fn required_url(params: &Value) -> Result<String, String> {
    let url = params
        .get("url")
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or_default();
    if url.is_empty() {
        return Err("a browser service requires a `url`".to_string());
    }
    Ok(url.to_string())
}

/// Copy the skill's own parameter names through, so the manager sees what it
/// expects without the skill having to learn the manager's vocabulary.
fn passthrough(params: &Value, keys: &[&str]) -> HashMap<String, Value> {
    let mut out = HashMap::new();
    for key in keys {
        if let Some(value) = params.get(*key)
            && !value.is_null()
        {
            out.insert((*key).to_string(), value.clone());
        }
    }
    out
}

fn extract_service(manager: Arc<BrowserManager>) -> ServiceFn {
    Arc::new(move |params: Value| {
        let manager = manager.clone();
        Box::pin(async move {
            let url = required_url(&params)?;
            let args = passthrough(&params, &["selector", "attribute", "mode"]);
            let text = manager.extract(&url, &args).await?;
            Ok(json!({ "text": text }))
        })
    })
}

fn evaluate_service(manager: Arc<BrowserManager>) -> ServiceFn {
    Arc::new(move |params: Value| {
        let manager = manager.clone();
        Box::pin(async move {
            let url = required_url(&params)?;
            // The skill sends `expression`; the manager now accepts it.
            let args = passthrough(&params, &["expression", "script"]);
            if args.is_empty() {
                return Err("browser.evaluate requires an `expression`".to_string());
            }
            let value = manager.evaluate(&url, &args).await?;
            Ok(json!({ "value": value }))
        })
    })
}

fn action_service(manager: Arc<BrowserManager>) -> ServiceFn {
    Arc::new(move |params: Value| {
        let manager = manager.clone();
        Box::pin(async move {
            let url = required_url(&params)?;
            let mut args = passthrough(&params, &["action", "selector", "value"]);
            if !args.contains_key("action") {
                return Err(
                    "browser.action requires an `action` (click, type, scroll, navigate, wait)"
                        .to_string(),
                );
            }
            // The two names the skill uses that the manager reads differently.
            // `type` takes its text from `value`, and `wait` its duration from
            // `delay_ms` — translated here rather than asking the skill to
            // learn the manager's spelling.
            if let Some(value) = params.get("value") {
                args.insert("text".to_string(), value.clone());
            }
            if let Some(delay) = params.get("delay_ms") {
                args.insert("wait_ms".to_string(), delay.clone());
            }
            let result = manager.action(&url, &args).await?;
            Ok(json!({ "result": result }))
        })
    })
}

fn screenshot_service(manager: Arc<BrowserManager>, screenshot_dir: PathBuf) -> ServiceFn {
    Arc::new(move |params: Value| {
        let manager = manager.clone();
        let screenshot_dir = screenshot_dir.clone();
        Box::pin(async move {
            let url = required_url(&params)?;
            let args = passthrough(&params, &["full_page", "width", "height", "selector"]);
            let image = manager.screenshot(&url, &args).await?;
            debug_assert!(
                !image.is_empty(),
                "the browser returned an empty screenshot"
            );

            // Same reasoning as `audio.tts`: the skill reported a byte count and
            // dropped the image, so the daemon drove a browser to produce a
            // screenshot nobody could look at.
            tokio::fs::create_dir_all(&screenshot_dir)
                .await
                .map_err(|e| format!("cannot create {}: {e}", screenshot_dir.display()))?;
            let captured_at_nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default();
            let path = screenshot_dir.join(format!("page-{captured_at_nanos}.png"));
            let size = image.len();
            tokio::fs::write(&path, image)
                .await
                .map_err(|e| format!("cannot write {}: {e}", path.display()))?;

            info!(path = ?path, size, %url, "Wrote page screenshot");
            Ok(json!({ "path": path.to_string_lossy(), "size": size }))
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_url_is_required_and_refused_by_name() {
        assert!(required_url(&json!({})).is_err());
        assert!(required_url(&json!({ "url": "   " })).is_err());
        assert_eq!(
            required_url(&json!({ "url": " https://example.com " })).unwrap(),
            "https://example.com"
        );
    }

    #[test]
    fn passthrough_keeps_present_keys_and_drops_absent_and_null_ones() {
        let params = json!({ "selector": "a", "attribute": null, "other": 1 });
        let out = passthrough(&params, &["selector", "attribute", "mode"]);
        assert_eq!(
            out.len(),
            1,
            "a null or absent key was passed through: {out:?}"
        );
        assert_eq!(out["selector"], "a");
    }

    /// The gate and the launch must name the same binary, or the gate is
    /// checking something it does not guard.
    #[test]
    fn a_detected_executable_is_a_real_file() {
        if let Some(found) = find_browser_executable() {
            assert!(
                found.is_file(),
                "detection returned a path that is not a file: {found:?}"
            );
        }
    }

    #[test]
    fn a_stale_browser_override_does_not_pass_the_gate() {
        // SAFETY: single-threaded test setup.
        unsafe {
            std::env::set_var("CHROME", "/definitely/not/a/browser");
        }
        let found = find_browser_executable();
        unsafe {
            std::env::remove_var("CHROME");
        }
        assert!(
            found.is_none_or(|p| p.as_os_str() != "/definitely/not/a/browser"),
            "a non-existent override was accepted, so the gate would pass and \
             the launch would fail",
        );
    }
}
