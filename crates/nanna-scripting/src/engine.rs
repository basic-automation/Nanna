//! Script engine: runs a scripted tool on Boa, the one JS engine compiled in.

use crate::{Result, ScriptedTool, NannaBridge, bridge::{ServiceFn, ToolSearchFn}};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info};

/// Which engine executed the script. Boa is the only one: a V8 (Deno) path
/// sat behind a feature no crate enabled, and was deleted on 2026-10-08.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EngineKind {
    Boa,
}

impl std::fmt::Display for EngineKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Boa => write!(f, "Boa"),
        }
    }
}

/// Result of script execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    /// The return value
    pub value: Value,
    /// Which engine was used
    pub engine: EngineKind,
    /// Execution time in milliseconds
    pub duration_ms: u64,
}

/// The optional capabilities a script's `Nanna` bridge is built with.
///
/// A field left `None` leaves that capability off, so
/// `BridgeCapabilities::default()` is a bridge with none of them.
#[derive(Default)]
pub struct BridgeCapabilities {
    /// JSON array of tool definitions for `Nanna.listTools()`.
    pub tool_definitions: Option<Value>,
    /// Service functions callable via `Nanna.service()`.
    pub services: Option<HashMap<String, ServiceFn>>,
    /// Default working directory for exec commands, overriding the home
    /// directory fallback.
    pub default_workdir: Option<std::path::PathBuf>,
    /// Session ID for session-scoped operations.
    pub session_id: Option<String>,
    /// Ranked tool search behind `Nanna.searchTools()`.
    pub tool_search: Option<ToolSearchFn>,
}

/// Runs scripted tools on the compiled-in engine.
pub struct ScriptEngine;

impl ScriptEngine {
    /// Create a new script engine.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Execute a scripted tool
    ///
    /// `tool_definitions` is an optional JSON array of tool definitions for `Nanna.listTools()`.
    /// `services` is an optional map of service functions callable via `Nanna.service()`.
    ///
    /// # Errors
    ///
    /// The same as [`Self::execute_full`].
    pub async fn execute(
        &self,
        tool: &ScriptedTool,
        input: Value,
        tool_definitions: Option<Value>,
        services: Option<HashMap<String, ServiceFn>>,
    ) -> Result<ExecutionResult> {
        self.execute_with_workdir(tool, input, tool_definitions, services, None).await
    }

    /// Execute a scripted tool with an optional default working directory.
    ///
    /// # Errors
    ///
    /// The same as [`Self::execute_full`].
    pub async fn execute_with_workdir(
        &self,
        tool: &ScriptedTool,
        input: Value,
        tool_definitions: Option<Value>,
        services: Option<HashMap<String, ServiceFn>>,
        default_workdir: Option<std::path::PathBuf>,
    ) -> Result<ExecutionResult> {
        self.execute_with_workdir_and_session(tool, input, tool_definitions, services, default_workdir, None).await
    }

    /// Execute a scripted tool with optional working directory and session ID.
    ///
    /// # Errors
    ///
    /// The same as [`Self::execute_full`].
    pub async fn execute_with_workdir_and_session(
        &self,
        tool: &ScriptedTool,
        input: Value,
        tool_definitions: Option<Value>,
        services: Option<HashMap<String, ServiceFn>>,
        default_workdir: Option<std::path::PathBuf>,
        session_id: Option<String>,
    ) -> Result<ExecutionResult> {
        let capabilities = BridgeCapabilities {
            tool_definitions,
            services,
            default_workdir,
            session_id,
            tool_search: None,
        };
        self.execute_full(tool, input, capabilities).await
    }

    /// Execute a scripted tool with every optional bridge capability,
    /// including the ranked tool search behind `Nanna.searchTools()`.
    ///
    /// # Errors
    ///
    /// Returns the engine's error when the script fails — for example
    /// [`crate::ScriptError::Execution`] when the script throws, does not parse, or
    /// exports no callable `execute`, [`crate::ScriptError::Timeout`] when it overruns
    /// the tool's deadline, or [`crate::ScriptError::EngineNotAvailable`] when Boa is
    /// not compiled in.
    pub async fn execute_full(
        &self,
        tool: &ScriptedTool,
        input: Value,
        capabilities: BridgeCapabilities,
    ) -> Result<ExecutionResult> {
        let BridgeCapabilities {
            tool_definitions,
            services,
            default_workdir,
            session_id,
            tool_search,
        } = capabilities;
        let mut bridge = NannaBridge::new(tool.permissions.clone());
        if let Some(defs) = tool_definitions {
            bridge = bridge.with_tool_definitions(defs);
        }
        if let Some(search) = tool_search {
            bridge = bridge.with_tool_search(search);
        }
        if let Some(svcs) = services {
            bridge = bridge.with_services(svcs);
        }
        if let Some(wd) = default_workdir {
            bridge = bridge.with_default_workdir(wd);
        }
        if let Some(sid) = session_id {
            bridge = bridge.with_session_id(sid);
        }
        let bridge = Arc::new(bridge);
        let start = std::time::Instant::now();

        // The script-engine deadline wraps the *whole* script, including any shell
        // command it runs via the bridge. A tool that shells out (e.g. `exec`)
        // forwards an integer `timeout` (seconds) input to the bridge, which owns
        // the *command* deadline and can kill the child. If the engine deadline
        // were shorter than that command deadline it would preempt a legitimately
        // long command — and worse, orphan the child the bridge would otherwise
        // reap. So extend the engine deadline to cover the requested command
        // deadline. We only ever *extend*, never shorten, so tools without a
        // meaningful `timeout` input keep their configured deadline.
        let effective_timeout_ms = effective_timeout_ms(tool.timeout_ms, &input);
        let tool_owned;
        let tool: &ScriptedTool = if effective_timeout_ms == tool.timeout_ms {
            tool
        } else {
            tool_owned = tool.clone().with_timeout(effective_timeout_ms);
            &tool_owned
        };

        let engine = EngineKind::Boa;
        debug!(tool = %tool.name, engine = %engine, "Executing script");
        let value = Self::execute_with_engine(tool, &input, &bridge, engine).await?;
        let duration_ms = crate::elapsed_ms(start);
        info!(tool = %tool.name, engine = %engine, duration_ms, "Script executed successfully");
        Ok(ExecutionResult {
            value,
            engine,
            duration_ms,
        })
    }

    /// Execute with a specific engine
    async fn execute_with_engine(
        tool: &ScriptedTool,
        input: &Value,
        bridge: &Arc<NannaBridge>,
        engine: EngineKind,
    ) -> Result<Value> {
        match engine {
            EngineKind::Boa => {
                #[cfg(feature = "boa")]
                {
                    crate::boa_impl::execute(tool, input, bridge).await
                }
                #[cfg(not(feature = "boa"))]
                {
                    let _ = (tool, input, bridge);
                    Err(crate::ScriptError::EngineNotAvailable("Boa".to_string()))
                }
            }
        }
    }

    /// The deadline an OUTER supervisor must enforce for one call so that this
    /// engine's deadline always fires first.
    ///
    /// `base_ms` is the tool's configured deadline and `requested_timeout` is
    /// the call's raw `timeout` input, parsed here by the same rule the engine
    /// applies to itself, so a supervisor cannot disagree with the engine about
    /// what was asked for. The answer is the deadline this engine will really
    /// enforce plus one more `ENGINE_TIMEOUT_HANDOFF_MARGIN_MS`: every layer
    /// sits one handoff above the layer it supervises, which is what makes the
    /// inner, better-informed message win by construction instead of by luck.
    /// Only ever longer than `base_ms`, never shorter.
    #[must_use]
    pub fn supervising_timeout_ms(base_ms: u64, requested_timeout: Option<&Value>) -> u64 {
        extend_for_requested(base_ms, requested_timeout_secs(requested_timeout))
            .saturating_add(ENGINE_TIMEOUT_HANDOFF_MARGIN_MS)
    }

    /// Check which engines are available
    #[must_use]
    pub fn available_engines(&self) -> Vec<EngineKind> {
        vec![
            #[cfg(feature = "boa")]
            EngineKind::Boa,
        ]
    }
}

impl Default for ScriptEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Slack between the bridge's per-command deadline and the outer engine
/// deadline, in ms. The engine deadline must OUTLIVE the command deadline so the
/// bridge (which can kill the child) always fires first; this covers the
/// script/bridge handoff overhead (TS transpile, thread + runtime spawn, result
/// marshaling), all sub-second, with margin.
const ENGINE_TIMEOUT_HANDOFF_MARGIN_MS: u64 = 10_000;

/// The command deadline (seconds) a call asks the shell bridge for, if any.
///
/// Takes the raw input value so every layer that needs this number parses it by
/// the same rule: a non-integer or zero `timeout` is no request at all.
fn requested_timeout_secs(requested: Option<&Value>) -> Option<u64> {
    requested.and_then(Value::as_u64).filter(|&s| s > 0)
}

/// Extend `base_ms` to outlive a requested command deadline.
///
/// Adds [`ENGINE_TIMEOUT_HANDOFF_MARGIN_MS`] on top of the request. Only ever
/// extends; a call with no meaningful request keeps `base_ms`. An overflowing
/// request is treated as no request rather than saturating to eternity.
fn extend_for_requested(base_ms: u64, requested_secs: Option<u64>) -> u64 {
    let requested_ms = requested_secs
        .and_then(|s| s.checked_mul(1000))
        .map(|ms| ms.saturating_add(ENGINE_TIMEOUT_HANDOFF_MARGIN_MS));
    requested_ms.map_or(base_ms, |req| base_ms.max(req))
}

/// Compute the effective script-engine deadline for one execution.
///
/// Extends `base_ms` (the tool's configured deadline) to cover a `timeout`
/// (seconds) declared in `input` — the deadline the shell bridge will enforce
/// for a command — plus [`ENGINE_TIMEOUT_HANDOFF_MARGIN_MS`]. Only ever extends;
/// a tool with no `timeout` input (or a zero/absent one) keeps `base_ms`.
fn effective_timeout_ms(base_ms: u64, input: &Value) -> u64 {
    extend_for_requested(base_ms, requested_timeout_secs(input.get("timeout")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_display() {
        assert_eq!(EngineKind::Boa.to_string(), "Boa");
    }

    #[test]
    fn test_available_engines() {
        let engine = ScriptEngine::new();
        let available = engine.available_engines();

        #[cfg(feature = "boa")]
        assert!(available.contains(&EngineKind::Boa));
    }

    #[test]
    fn effective_timeout_extends_for_requested_command_timeout() {
        // A large explicit `timeout` input pushes the engine deadline past it by
        // the handoff margin, so the bridge's command deadline fires first.
        let base = 30_000;
        let input = serde_json::json!({ "command": "cargo build", "timeout": 600 });
        let eff = effective_timeout_ms(base, &input);
        assert_eq!(eff, 600_000 + ENGINE_TIMEOUT_HANDOFF_MARGIN_MS);
        assert!(eff > 600_000, "engine must outlive the 600s command deadline");
    }

    #[test]
    fn effective_timeout_never_shortens() {
        // A small requested timeout must not pull the engine deadline below the
        // tool's configured base (e.g. exec's 180s ceiling).
        let base = 180_000;
        let input = serde_json::json!({ "timeout": 5 });
        assert_eq!(effective_timeout_ms(base, &input), base);
    }

    #[test]
    fn supervising_timeout_outlives_the_engines_own_deadline() {
        // The registry's backstop must sit strictly above whatever this engine
        // will enforce, for BOTH shapes of call: one that requests a longer
        // command deadline, and one that requests nothing at all. The second is
        // the case that used to lose the race — two nominally equal deadlines,
        // and the outer one fires while the tool is still writing its answer.
        let base = 180_000;

        let long = serde_json::json!({ "command": "cargo build", "timeout": 600 });
        assert!(
            ScriptEngine::supervising_timeout_ms(base, long.get("timeout"))
                > effective_timeout_ms(base, &long)
        );

        let plain = serde_json::json!({ "command": "cargo build" });
        assert!(
            ScriptEngine::supervising_timeout_ms(base, plain.get("timeout"))
                > effective_timeout_ms(base, &plain)
        );
    }

    #[test]
    fn supervising_timeout_never_shortens() {
        // A small or malformed request must never pull the supervisor below the
        // tool's configured base.
        let base = 180_000;
        for requested in [
            serde_json::json!(5),
            serde_json::json!(0),
            serde_json::json!("soon"),
        ] {
            assert!(ScriptEngine::supervising_timeout_ms(base, Some(&requested)) > base);
        }
        assert!(ScriptEngine::supervising_timeout_ms(base, None) > base);
    }

    #[test]
    fn effective_timeout_ignores_absent_or_zero() {
        let base = 30_000;
        assert_eq!(effective_timeout_ms(base, &serde_json::json!({})), base);
        assert_eq!(
            effective_timeout_ms(base, &serde_json::json!({ "timeout": 0 })),
            base
        );
        // Non-integer timeouts are ignored (no partial/garbage parse).
        assert_eq!(
            effective_timeout_ms(base, &serde_json::json!({ "timeout": "soon" })),
            base
        );
    }
}
