//! The `screenshot.capture` service.
//!
//! The bundled `screenshot` skill declares it and nothing registered it, so the
//! skill was withheld at every boot. The roadmap's note ("skill exists, service
//! missing, Rust tool is a stub") was accurate: `nanna-tools`' `ScreenshotTool`
//! was a placeholder that returned "not yet implemented" (deleted 2026-10-08),
//! so there was nothing to wire — this is the implementation, not a registration.
//!
//! **Shelling out to the desktop's own capture tool, rather than taking a
//! screen-capture dependency.** Screen capture is not portable library work:
//! on Wayland it goes through the compositor (`grim`), on X11 through the
//! server (`maim`/`scrot`/`import`), and on macOS through `screencapture`.
//! Every one of those already ships on the systems that have it, and each
//! speaks PNG to a file path, so the whole difference between them is a command
//! line.
//!
//! **Registered only when a capture tool and a display session both exist.** A
//! headless daemon has nothing to photograph, and a tool with no session fails
//! at call time with a confusing error — so both are checked at boot and the
//! skill stays withheld otherwise, with the boot line naming what is missing.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use nanna_scripting::ServiceFn;
use serde_json::{Value, json};
use tracing::{info, warn};

/// Subdirectory of the data dir that desktop captures are written to. Shared
/// with the browser services: both produce PNGs of something on screen.
const SCREENSHOT_DIR_NAME: &str = "screenshots";

/// How long a capture may take before it is abandoned.
///
/// Derived from what the command is: a screen grab is a single compositor
/// round-trip and completes in well under a second, so anything still running
/// after this is wedged — typically a tool waiting for an interactive region
/// selection that no one is there to make.
const CAPTURE_TIMEOUT_SECS: u64 = 15;

/// A desktop capture tool and how to ask it for a full-screen PNG.
///
/// One entry per tool rather than a generic "run a command" hook: the argument
/// order is the entire difference between them, and a table makes the supported
/// set reviewable.
struct CaptureTool {
    /// Executable name, resolved against `PATH`.
    name: &'static str,
    /// Arguments, with `{path}` replaced by the output file.
    args: &'static [&'static str],
    /// Session this tool needs: `Wayland`, `X11`, or `Any`.
    session: SessionKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SessionKind {
    Wayland,
    X11,
    Any,
}

/// Capture tools in preference order.
///
/// Wayland first because that is what a modern Linux desktop runs and an X11
/// tool under it captures nothing (or an `XWayland` surface, which is worse:
/// it succeeds and returns the wrong thing).
const CAPTURE_TOOLS: &[CaptureTool] = &[
    CaptureTool {
        name: "grim",
        args: &["{path}"],
        session: SessionKind::Wayland,
    },
    CaptureTool {
        name: "spectacle",
        args: &["-b", "-n", "-f", "-o", "{path}"],
        session: SessionKind::Any,
    },
    CaptureTool {
        name: "maim",
        args: &["{path}"],
        session: SessionKind::X11,
    },
    CaptureTool {
        name: "scrot",
        args: &["-o", "{path}"],
        session: SessionKind::X11,
    },
    CaptureTool {
        name: "import",
        args: &["-window", "root", "{path}"],
        session: SessionKind::X11,
    },
    CaptureTool {
        name: "screencapture",
        args: &["-x", "{path}"],
        session: SessionKind::Any,
    },
];

/// Whether a session of this kind is reachable from the daemon's environment.
fn session_available(kind: SessionKind) -> bool {
    let wayland = std::env::var_os("WAYLAND_DISPLAY").is_some_and(|v| !v.is_empty());
    let x11 = std::env::var_os("DISPLAY").is_some_and(|v| !v.is_empty());
    match kind {
        SessionKind::Wayland => wayland,
        SessionKind::X11 => x11,
        // macOS `screencapture` needs neither variable; on Linux a tool that
        // claims to work anywhere still needs *a* session.
        SessionKind::Any => cfg!(target_os = "macos") || wayland || x11,
    }
}

/// Resolve a bare executable name against `PATH`.
fn which_on_path(name: &str) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    std::env::split_paths(&path_var)
        .map(|dir| dir.join(name))
        .find(|candidate| candidate.is_file())
}

/// The first capture tool that is installed *and* has a session to capture.
fn find_capture_tool() -> Option<(&'static CaptureTool, PathBuf)> {
    CAPTURE_TOOLS.iter().find_map(|tool| {
        if !session_available(tool.session) {
            return None;
        }
        which_on_path(tool.name).map(|path| (tool, path))
    })
}

/// The command line for one capture, with `{path}` substituted.
fn capture_args(tool: &CaptureTool, output: &Path) -> Vec<String> {
    tool.args
        .iter()
        .map(|arg| arg.replace("{path}", &output.to_string_lossy()))
        .collect()
}

/// Why this capture request cannot be served, or `None` if it can.
///
/// Pure, so the contract is testable without a display. `target` is the skill's
/// only parameter and its schema advertises "'desktop' for full screen, or a
/// window title". **Window capture is not supported**, and that is said rather
/// than quietly served as a full-screen grab: handing back the whole desktop
/// when a specific window was asked for is a wrong answer, not a degraded one.
fn capture_refusal(target: &str) -> Option<String> {
    let target = target.trim();
    if target.is_empty() || target.eq_ignore_ascii_case("desktop") {
        return None;
    }
    Some(format!(
        "screenshot.capture can only capture the whole desktop; it cannot select \
         the window {target:?}. Ask for 'desktop', or use the browser_screenshot \
         skill to capture a specific web page."
    ))
}

/// Build `screenshot.capture`, or an empty map with no tool or no session.
pub fn build_screenshot_services(data_dir: &Path) -> HashMap<String, ServiceFn> {
    let Some((tool, executable)) = find_capture_tool() else {
        info!(
            "No usable desktop capture tool; screenshot.capture stays unregistered \
             and the screenshot skill stays withheld. It needs one of grim \
             (Wayland), spectacle, maim/scrot/import (X11) or screencapture \
             (macOS), and a display session the daemon can reach."
        );
        return HashMap::new();
    };

    info!(tool = tool.name, executable = ?executable, "Registering screenshot.capture");
    let screenshot_dir = data_dir.join(SCREENSHOT_DIR_NAME);

    let mut services: HashMap<String, ServiceFn> = HashMap::new();
    services.insert(
        "screenshot.capture".to_string(),
        Arc::new(move |params: Value| {
            let screenshot_dir = screenshot_dir.clone();
            let executable = executable.clone();
            Box::pin(async move {
                let target = params
                    .get("target")
                    .and_then(Value::as_str)
                    .unwrap_or("desktop");
                if let Some(refusal) = capture_refusal(target) {
                    return Err(refusal);
                }

                tokio::fs::create_dir_all(&screenshot_dir)
                    .await
                    .map_err(|e| format!("cannot create {}: {e}", screenshot_dir.display()))?;
                let captured_at_nanos = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or_default();
                let path = screenshot_dir.join(format!("desktop-{captured_at_nanos}.png"));

                let status = tokio::time::timeout(
                    std::time::Duration::from_secs(CAPTURE_TIMEOUT_SECS),
                    tokio::process::Command::new(&executable)
                        .args(capture_args(tool, &path))
                        // Without this the timeout below drops the future and
                        // leaves the child running: tokio's default is to
                        // detach, and the tool most likely to time out is one
                        // sitting on an interactive selection that will never
                        // come. This repo has paid for orphaned children before.
                        .kill_on_drop(true)
                        .status(),
                )
                .await
                .map_err(|_| {
                    format!(
                        "{} did not finish within {CAPTURE_TIMEOUT_SECS}s; it may be \
                         waiting for an interactive selection",
                        tool.name
                    )
                })?
                .map_err(|e| format!("could not run {}: {e}", tool.name))?;

                if !status.success() {
                    return Err(format!("{} exited with {status}", tool.name));
                }
                // The tool reporting success is not the same as a file existing:
                // several of these exit 0 after writing nothing when there is no
                // session to grab.
                let size = tokio::fs::metadata(&path)
                    .await
                    .map_err(|e| {
                        format!(
                            "{} reported success but wrote no file to {}: {e}",
                            tool.name,
                            path.display()
                        )
                    })?
                    .len();
                if size == 0 {
                    warn!(tool = tool.name, path = ?path, "capture wrote an empty file");
                    return Err(format!("{} wrote an empty screenshot", tool.name));
                }

                info!(path = ?path, size, tool = tool.name, "Captured the desktop");
                Ok(json!({
                    "path": path.to_string_lossy(),
                    "size": size,
                    "tool": tool.name,
                }))
            })
        }),
    );
    services
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn desktop_is_the_supported_target_and_the_default() {
        assert_eq!(capture_refusal("desktop"), None);
        assert_eq!(capture_refusal("Desktop"), None);
        assert_eq!(capture_refusal(""), None);
        assert_eq!(capture_refusal("   "), None);
    }

    /// Asking for a window must be refused, not quietly served as the whole
    /// screen — a wrong answer is worse than a refusal the caller can act on.
    #[test]
    fn a_window_title_is_refused_and_the_refusal_says_what_to_do() {
        let refusal = capture_refusal("Firefox").expect("a window title must be refused");
        assert!(refusal.contains("Firefox"), "unhelpful refusal: {refusal}");
        assert!(
            refusal.contains("browser_screenshot"),
            "the refusal should point at the tool that can do it: {refusal}"
        );
    }

    #[test]
    fn the_output_path_is_substituted_into_every_tools_arguments() {
        let output = Path::new("/tmp/probe.png");
        for tool in CAPTURE_TOOLS {
            let args = capture_args(tool, output);
            assert!(
                args.iter().any(|a| a.contains("/tmp/probe.png")),
                "{} never receives the output path: {args:?}",
                tool.name
            );
            assert!(
                !args.iter().any(|a| a.contains("{path}")),
                "{} has an unsubstituted placeholder: {args:?}",
                tool.name
            );
        }
    }

    /// An X11 tool under Wayland succeeds and captures the wrong thing, so
    /// preference order is load-bearing, not cosmetic.
    #[test]
    fn wayland_is_preferred_over_x11_in_the_table() {
        let wayland_at = CAPTURE_TOOLS
            .iter()
            .position(|t| t.session == SessionKind::Wayland)
            .expect("a Wayland tool is listed");
        let x11_at = CAPTURE_TOOLS
            .iter()
            .position(|t| t.session == SessionKind::X11)
            .expect("an X11 tool is listed");
        assert!(
            wayland_at < x11_at,
            "an X11 tool would be picked before a Wayland one",
        );
    }

    #[test]
    fn a_detected_tool_is_a_real_file_and_matches_its_table_entry() {
        if let Some((tool, path)) = find_capture_tool() {
            assert!(path.is_file(), "detection returned a non-file: {path:?}");
            assert!(
                path.ends_with(tool.name),
                "the path does not match the tool it was found for: {path:?} vs {}",
                tool.name
            );
            assert!(session_available(tool.session));
        }
    }

    #[tokio::test]
    async fn a_window_request_is_refused_before_anything_runs() {
        let dir = tempfile::tempdir().unwrap();
        let services = build_screenshot_services(dir.path());
        let Some(capture) = services.get("screenshot.capture") else {
            return; // No tool or no session on this host; nothing registered.
        };
        let err = capture(json!({ "target": "Some Window" }))
            .await
            .expect_err("a window title must be refused");
        assert!(err.contains("Some Window"));
        // Refused before anything ran means nothing was written.
        assert!(!dir.path().join(SCREENSHOT_DIR_NAME).exists());
    }
}
