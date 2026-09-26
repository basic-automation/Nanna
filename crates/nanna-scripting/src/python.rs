//! Embedded Python interpreter powered by `RustPython`.
//!
//! Provides a zero-dependency Python execution environment for AI tool use.
//! No system Python installation required — the interpreter is compiled into the binary.
//!
//! Supports standard library modules (os, json, re, pathlib, collections, etc.)
//! but not C-extension packages (numpy, requests, etc.).

use crate::{Result, ScriptError};
use serde::{Deserialize, Serialize};
use tracing::debug;

/// Native stack reserved for the thread running the `RustPython` interpreter.
///
/// `RustPython`'s VM recurses natively per Python frame, so how deep Python can go is
/// a function of this number.
///
/// **What this thread buys us:** Tokio's default worker/blocking stack is **2 MiB**,
/// which every `python.exec` overflowed — and a Rust stack overflow is an immediate
/// `abort()`, not a catchable panic, so `spawn_blocking`'s unwind guard could never
/// contain it. The interpreter therefore gets its own explicitly-sized thread rather
/// than inheriting the runtime's.
///
/// **Measured relationship** (rustpython 0.5, release, Windows `x86_64`) — the maximum
/// recursion depth reached before a *catchable* `RecursionError`:
///
/// | stack   | max depth reached | implied bytes/frame |
/// |---------|-------------------|---------------------|
/// | 64 MiB  | 32,727            | ~2,050              |
/// | 256 MiB | 131,004           | ~2,050              |
///
/// Depth scales **linearly** with the stack (4x stack → 4.00x depth), so a frame costs
/// about **2 KiB** and the usable depth is roughly `stack_bytes / 2 KiB`.
///
/// **`RustPython` guards on remaining stack, not just its `recursion_limit`.** Probing
/// with `sys.setrecursionlimit(1_000_000)` still raised `RecursionError` at the
/// stack-derived depth above (131,004 on 256 MiB) rather than overflowing. The
/// mechanism is `VirtualMachine::check_c_stack_overflow`
/// (`rustpython-vm-0.5.0/src/vm/mod.rs:1520`), a `CPython` `_Py_MakeRecCheck` port, run on
/// **every** recursive call: it compares the live stack pointer (`psm::stack_pointer()`)
/// against a soft limit computed from the thread's *actual* stack bounds. That is why
/// the reachable depth tracks `stack_size` at all. (At the *default* limit the limit
/// binds first and the guard never comes up; it only matters once user code raises the
/// limit past what the stack holds — which is the case this whole section is about.)
///
/// **That guard is not sufficient on its own, because it tests a *band*, not a floor.**
/// It fires only while the stack pointer sits within `2 x STACK_MARGIN_BYTES` (32 KiB)
/// above the stack base, so a frame that advances the pointer by more than the band
/// steps clean over the check and into the guard page — an uncatchable `abort()`.
/// Release frames (~2 KiB, above) land inside the band every time. **Debug frames do
/// not**: with `--features python` in a workspace build, raising the limit and recursing
/// overflows for real (`thread 'nanna-python' has overflowed its stack`).
///
/// That is why `build_wrapper` clamps `sys.setrecursionlimit` to the interpreter's own
/// startup default (256 debug / 1000 release) — the depth this stack is validated at.
/// User code may lower the limit, not raise it. `raising_the_recursion_limit_cannot_abort_the_process`
/// pins it, and **fails in debug without the clamp**, so the profile that catches this
/// is the one `cargo test --workspace` actually runs.
///
/// Cost is bounded and cheap: a thread stack is *reserved* address space committed
/// lazily by page, so a shallow execution touches only the pages it actually uses —
/// the 256 MiB is a ceiling on a 64-bit address space, not an allocation, and only
/// one such thread exists per in-flight call. At ~2 KiB/frame it is far more than the
/// stock `recursion_limit` (1000 release / 256 debug) can consume; it is retained
/// because it is measured-good and effectively free, not because the limit needs it.
///
/// Any change here must be re-measured under `--release` as well as `cargo test`: the
/// two profiles carry different default recursion limits (**256 debug / 1000 release**,
/// `rustpython_vm::VirtualMachine`), so debug-only evidence has been misleading here
/// before.
const PYTHON_STACK_BYTES: usize = 256 * 1024 * 1024;

/// Keeps the usable recursion depth comfortably above anything the stock interpreter
/// can reach: at the measured ~2 KiB/frame, this floor is ~65,000 frames, versus a
/// default `recursion_limit` of 1000 (release) / 256 (debug).
///
/// The floor is deliberately conservative rather than tight, and it is derived from the
/// **empirical overflow bisection**, not a round number: the `runaway_recursion` /
/// raised-limit guards below overflowed at **16 MiB (debug)** and **64 MiB (release)**
/// and passed at **128 MiB** — so 128 MiB is the measured "passes" point and
/// `PYTHON_STACK_BYTES` (256 MiB) is 2x it, all lazily committed (free).
///
/// Interpreter *setup* is the only term that could justify shrinking further (recursion
/// alone needs ~`1000 x 2 KiB` ≈ 2 MiB, and `python.exec` could not run `print('hello')`
/// on Tokio's 2 MiB stack, so setup dominates). But that setup high-water **cannot be
/// measured cheaply in-process on Windows** — a paint-and-scan probe faults
/// (`STATUS_ACCESS_VIOLATION`, verified) because a thread stack is *reserved* but
/// committed lazily through a guard page, so a deep write lands in uncommitted memory;
/// and a genuine overflow aborts the process uncatchably, defeating a binary-search
/// probe too. So the size stays anchored to the bisection above rather than a separate
/// setup number. Do not shrink toward the recursion-only budget.
const _: () = assert!(PYTHON_STACK_BYTES >= 128 * 1024 * 1024);

/// Result of executing Python code.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonResult {
    pub stdout: String,
    pub stderr: String,
    pub success: bool,
    pub error: Option<String>,
    pub duration_ms: u64,
}

/// Manages embedded `RustPython` execution.
///
/// Each execution creates a fresh interpreter to ensure isolation.
/// This is lightweight enough for tool-call frequency.
pub struct PythonEngine;

impl PythonEngine {
    /// Create a new Python engine.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Execute Python code and capture stdout/stderr.
    ///
    /// Each call gets a fresh interpreter — variables don't leak between calls.
    /// Working directory is set via `os.chdir()` if provided.
    ///
    /// A Python exception or syntax error is not an `Err`: it comes back as a
    /// [`PythonResult`] with `success: false` and the exception in `error`.
    ///
    /// # Errors
    ///
    /// Returns [`ScriptError::Execution`] if the interpreter thread cannot be
    /// spawned or ends without a result (it panicked), and
    /// [`ScriptError::Timeout`] if the code runs past `timeout_secs`.
    pub async fn execute(
        &self,
        code: &str,
        workdir: Option<&str>,
        timeout_secs: u64,
    ) -> Result<PythonResult> {
        let code = code.to_string();
        let workdir = workdir.map(String::from);

        let start = std::time::Instant::now();

        // RustPython is synchronous and !Send, and needs far more native stack than a
        // Tokio blocking thread provides (see PYTHON_STACK_BYTES), so it gets its own
        // explicitly-sized thread rather than the runtime's ambient one.
        let receiver = spawn_interpreter_thread(code, workdir)?;

        let result =
            tokio::time::timeout(std::time::Duration::from_secs(timeout_secs), receiver).await;

        let duration_ms = crate::elapsed_ms(start);

        match result {
            Ok(Ok(mut py_result)) => {
                py_result.duration_ms = duration_ms;
                debug!(
                    success = py_result.success,
                    duration_ms,
                    stdout_len = py_result.stdout.len(),
                    "Python execution complete"
                );
                Ok(py_result)
            }
            // The sender is only dropped without a value if the interpreter thread
            // panicked; the panic itself is reported by the default hook.
            Ok(Err(_)) => Err(ScriptError::Execution(
                "Python interpreter thread terminated without returning a result".to_string(),
            )),
            Err(_) => Err(ScriptError::Timeout(timeout_secs * 1000)),
        }
    }
}

impl Default for PythonEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Run one isolated interpreter on a dedicated thread with a stack large enough for
/// `RustPython`, returning a receiver for its result.
///
/// A timed-out caller simply stops awaiting the receiver; the thread runs to
/// completion and its result is dropped. That matches the previous `spawn_blocking`
/// behavior — a synchronous interpreter cannot be pre-empted mid-execution — and the
/// thread is bounded by the caller's timeout in practice.
fn spawn_interpreter_thread(
    code: String,
    workdir: Option<String>,
) -> Result<tokio::sync::oneshot::Receiver<PythonResult>> {
    // The stack invariant is enforced at compile time by the `const _` assert on
    // PYTHON_STACK_BYTES — stronger than a debug_assert here, and free.
    //
    // Empty source is *user* input (the daemon's python.exec defaults `code` to ""),
    // so it is handled by the interpreter rather than asserted on, and a failed spawn
    // is an operational error returned as Err below. Hence no runtime assertion.
    let (sender, receiver) = tokio::sync::oneshot::channel();

    std::thread::Builder::new()
        .name("nanna-python".to_string())
        .stack_size(PYTHON_STACK_BYTES)
        .spawn(move || {
            let result = execute_isolated(&code, workdir.as_deref());
            // A dropped receiver means the caller timed out — expected, not an error.
            drop(sender.send(result));
        })
        .map_err(|e| {
            ScriptError::Execution(format!("failed to spawn Python interpreter thread: {e}"))
        })?;

    Ok(receiver)
}

/// Execute code in a fresh, isolated interpreter.
///
/// Uses a two-phase approach:
/// 1. Wrap user code in a try/except that captures stdout, stderr, and errors
/// 2. Write results to temp files that we read back from Rust
///
/// This avoids needing to extract Python objects from the VM directly.
fn execute_isolated(code: &str, workdir: Option<&str>) -> PythonResult {
    use rustpython_vm as vm;
    use rustpython_vm::compiler::Mode;

    // Create a fresh interpreter with stdlib (native C-equivalent modules + frozen
    // Python bytecode). rustpython 0.5 replaced `Interpreter::with_init` with a
    // builder: stdlib module defs come from `stdlib_module_defs(&ctx)` and frozen
    // modules are added on the builder itself.
    let builder = vm::Interpreter::builder(vm::Settings::default());
    let stdlib_defs = rustpython_stdlib::stdlib_module_defs(&builder.ctx);
    let interp = builder
        .add_native_modules(&stdlib_defs)
        .add_frozen_modules(rustpython_pylib::FROZEN_STDLIB)
        .build();

    // Build wrapper code that captures everything
    let wrapper = build_wrapper(code, workdir);

    interp.enter(|vm| {
        let scope = vm.new_scope_with_builtins();

        match vm
            .compile(&wrapper, Mode::Exec, "<nanna>".to_owned())
            .map_err(|e| format!("SyntaxError: {e}"))
            .and_then(|code_obj| {
                vm.run_code_obj(code_obj, scope.clone())
                    .map_err(|exc| format_exception(vm, &exc))
            }) {
            Ok(_) => {
                // Extract results from the _nanna_result dict in scope
                let extract = r"
import json as _nj
_nanna_json = _nj.dumps(_nanna_result)
";
                if let Ok(code_obj) = vm.compile(extract, Mode::Exec, "<nanna-extract>".to_owned())
                {
                    let _ = vm.run_code_obj(code_obj, scope.clone());
                }

                // Get the JSON result by running a minimal expression
                let json_code = "_nanna_json";
                if let Ok(code_obj) = vm.compile(json_code, Mode::Eval, "<nanna-eval>".to_owned())
                    && let Ok(val) = vm.run_code_obj(code_obj, scope)
                        && let Ok(s) = val.str(vm) {
                            // rustpython 0.5: PyStr::as_str was replaced by
                            // to_string_lossy (strings may be non-UTF-8 kinds).
                            if let Ok(parsed) =
                                serde_json::from_str::<serde_json::Value>(&s.to_string_lossy())
                            {
                                return PythonResult {
                                    stdout: parsed
                                        .get("stdout")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("")
                                        .to_string(),
                                    stderr: parsed
                                        .get("stderr")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("")
                                        .to_string(),
                                    success: parsed
                                        .get("success")
                                        .and_then(serde_json::Value::as_bool)
                                        .unwrap_or(false),
                                    error: parsed
                                        .get("error")
                                        .and_then(|v| v.as_str())
                                        .map(String::from),
                                    duration_ms: 0,
                                };
                            }
                        }

                // Extraction failed: the outcome is unknown, so it is not reported
                // as a success (the caller would otherwise trust an empty result).
                PythonResult {
                    stdout: String::new(),
                    stderr: String::new(),
                    success: false,
                    error: Some("the script ran but its result could not be read back".into()),
                    duration_ms: 0,
                }
            }
            Err(error_msg) => {
                // The wrapper itself failed (shouldn't happen normally)
                PythonResult {
                    stdout: String::new(),
                    stderr: String::new(),
                    success: false,
                    error: Some(error_msg),
                    duration_ms: 0,
                }
            }
        }
    })
}

/// Build wrapper Python code that captures stdout, stderr, and exceptions.
fn build_wrapper(user_code: &str, workdir: Option<&str>) -> String {
    let chdir = workdir.map_or_else(String::new, |wd| {
        format!("import os; os.chdir({})\n", python_string_literal(wd))
    });

    // Escape the user code for embedding in a triple-quoted string
    // We use exec() with the code as a variable to avoid any escaping issues
    let escaped_code = user_code
        .replace('\\', "\\\\")
        .replace("\"\"\"", "\\\"\\\"\\\"");

    // Clamp `sys.setrecursionlimit` before any user code runs.
    //
    // The cap is the interpreter's *own* default limit, read at startup — not a number
    // we invent. That default (256 debug / 1000 release) is the configuration
    // PYTHON_STACK_BYTES was validated against, so it is exactly the depth we know the
    // stack survives. User code may lower the limit freely; it may not raise it past
    // what we tested. CPython's own default is 1000, so this is not an unusual ceiling.
    //
    // This is load-bearing, not belt-and-braces: RustPython's native stack guard tests a
    // *band* (see PYTHON_STACK_BYTES), and a build whose frames are larger than that band
    // — debug builds are — can step straight over it into the guard page, which aborts
    // the process uncatchably. `raising_the_recursion_limit_cannot_abort_the_process`
    // fails without this, in debug.
    //
    // The real function is captured in a closure and the installer name deleted, so the
    // un-clamped original is not reachable by name from the user's globals (`exec` below
    // shares them). That stops the ordinary footgun and casual abuse; it is not an
    // escape-proof sandbox — a determined script may still reach the closure cell by
    // introspection.
    let recursion_clamp = r"
def _nanna_install_recursion_clamp():
    _real = sys.setrecursionlimit
    _cap = sys.getrecursionlimit()
    def _clamped(limit):
        _real(min(limit, _cap))
    sys.setrecursionlimit = _clamped
_nanna_install_recursion_clamp()
del _nanna_install_recursion_clamp
";

    format!(
        r#"
import sys, io, traceback
{recursion_clamp}
{chdir}
_nanna_stdout_buf = io.StringIO()
_nanna_stderr_buf = io.StringIO()
_nanna_orig_stdout = sys.stdout
_nanna_orig_stderr = sys.stderr
sys.stdout = _nanna_stdout_buf
sys.stderr = _nanna_stderr_buf

_nanna_result = {{"stdout": "", "stderr": "", "success": True, "error": None}}

try:
    _nanna_user_code = """{escaped_code}"""
    exec(compile(_nanna_user_code, '<python>', 'exec'))
except SystemExit as _nanna_exit:
    _nanna_status = _nanna_exit.code
    if _nanna_status is not None and _nanna_status != 0:
        _nanna_result["success"] = False
        if isinstance(_nanna_status, int):
            _nanna_result["error"] = "SystemExit: exited with status " + str(_nanna_status)
        else:
            _nanna_result["error"] = "SystemExit: " + str(_nanna_status)
except:
    _nanna_result["success"] = False
    _nanna_result["error"] = traceback.format_exc()
finally:
    sys.stdout = _nanna_orig_stdout
    sys.stderr = _nanna_orig_stderr
    _nanna_result["stdout"] = _nanna_stdout_buf.getvalue()
    _nanna_result["stderr"] = _nanna_stderr_buf.getvalue()
"#
    )
}

/// Format a Python exception into a human-readable string.
fn format_exception(
    vm: &rustpython_vm::VirtualMachine,
    exc: &rustpython_vm::PyRef<rustpython_vm::builtins::PyBaseException>,
) -> String {
    // Convert to PyObject and call str()
    use rustpython_vm::AsObject;
    let obj = exc.as_object();
    let class_name = obj.class().name().to_string();
    let msg = obj
        .str(vm)
        .ok()
        .map_or_default(|s| s.to_string_lossy().into_owned());

    if msg.is_empty() {
        class_name
    } else {
        format!("{class_name}: {msg}")
    }
}

/// Escape a string for use as a Python string literal.
fn python_string_literal(s: &str) -> String {
    let escaped = s
        .replace('\\', "\\\\")
        .replace('\'', "\\'")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t");
    format!("'{escaped}'")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serializes the Python interpreter tests so `cargo test` is deterministic.
    ///
    /// Every test here spins a **fresh** `RustPython` interpreter, whose frozen-stdlib
    /// initialization is CPU-heavy (~2.7 s each, measured). Run in parallel — the
    /// `cargo test` default — a handful of them contend for cores, and a single
    /// `execute()` can then blow past its own `timeout_secs` wall-clock guard purely
    /// from scheduler starvation, surfacing a spurious `ScriptError::Timeout`. That is a
    /// **test-harness artifact, not a product bug**: a real `python.exec` sets its own
    /// per-call timeout and is unaffected by how many tests a CI box runs at once.
    ///
    /// A process-global async mutex forces one interpreter to build+run at a time, so
    /// each test's wall-clock tracks its solo cost (well under the smallest 10 s guard)
    /// and the suite passes regardless of `--test-threads`. `tokio::sync::Mutex` (not
    /// `std`): the guard is held across `.await`, so it must stay `Send` and must not
    /// trip clippy's `await_holding_lock`; it is also runtime-agnostic, so locking it
    /// from each `#[tokio::test]`'s own runtime (including the `current_thread` one) is
    /// fine, and it does **not** poison on a panicking test — a failed test releases the
    /// lock and the rest still run.
    static PYTHON_TEST_GUARD: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    /// Runaway recursion must surface as a catchable Python `RecursionError`, never a
    /// native stack overflow (which aborts the whole process, uncatchably).
    ///
    /// This is the guard on the sizing rule in `PYTHON_STACK_BYTES`: the stack must
    /// hold `recursion_limit x per-frame native cost`. It crashed at 16 MiB in debug
    /// and at 64 MiB in release (where the limit is 4x higher), which is what drove
    /// the constant to 256 MiB. **Run this under `--release` too** — it is the case
    /// that catches a debug-only sizing mistake.
    #[tokio::test]
    async fn runaway_recursion_errors_cleanly_without_aborting() {
        let _serialize = PYTHON_TEST_GUARD.lock().await;
        let engine = PythonEngine::new();
        let result = engine
            .execute(
                "def f(n):
    return f(n + 1)
f(0)",
                None,
                30,
            )
            .await
            .expect("runaway recursion must return a result, not kill the process");

        assert!(
            !result.success,
            "infinite recursion must not report success"
        );
        let error = result.error.unwrap_or_default();
        assert!(
            error.contains("RecursionError"),
            "expected a Python RecursionError, got: {error}"
        );
    }

    /// `sys.setrecursionlimit` is the one lever user code has over native stack depth,
    /// so it is the obvious way model-authored Python might try to abort the daemon (a
    /// Rust stack overflow is an uncatchable `abort()`, which no timeout or unwind
    /// guard can contain).
    ///
    /// It does not work, and this test pins why: `RustPython` guards on **remaining
    /// stack**, not merely on its own `recursion_limit`, so raising the limit to a
    /// million still lands on a catchable `RecursionError` at the stack-derived depth.
    /// No clamp on `setrecursionlimit` is needed — see `PYTHON_STACK_BYTES`. If a
    /// future rustpython drops that guard, this test fails by killing the test process
    /// rather than reporting an assertion — which is itself the signal.
    ///
    /// Must hold under `--release` too: the release recursion default is 4x debug's.
    #[tokio::test]
    async fn raising_the_recursion_limit_cannot_abort_the_process() {
        let _serialize = PYTHON_TEST_GUARD.lock().await;
        let engine = PythonEngine::new();
        let result = engine
            .execute(
                "import sys
sys.setrecursionlimit(1000000)
def f(n):
    return f(n + 1)
f(0)",
                None,
                60,
            )
            .await
            .expect("a raised recursion limit must still return a result, not kill the process");

        assert!(
            !result.success,
            "infinite recursion must not report success"
        );
        let error = result.error.unwrap_or_default();
        assert!(
            error.contains("RecursionError"),
            "a raised limit must stay in the catchable domain, got: {error}"
        );
    }

    /// The clamp lowers an over-large request rather than rejecting it, and pins it to
    /// the interpreter's own default — so legitimate code that raises the limit keeps
    /// running, it just cannot exceed the depth this stack is validated for.
    ///
    /// Asserted against the default read back at runtime rather than a literal, because
    /// that default is profile-dependent (256 debug / 1000 release).
    #[tokio::test]
    async fn raising_the_recursion_limit_is_clamped_to_the_interpreter_default() {
        let _serialize = PYTHON_TEST_GUARD.lock().await;
        let engine = PythonEngine::new();
        let result = engine
            .execute(
                "import sys
_default = sys.getrecursionlimit()
sys.setrecursionlimit(1000000)
print(sys.getrecursionlimit() == _default, sys.getrecursionlimit())",
                None,
                30,
            )
            .await
            .expect("execution completes");

        assert!(
            result.success,
            "clamping must not raise: {:?}",
            result.error
        );
        assert!(
            result.stdout.starts_with("True"),
            "an over-large request is pinned to the startup default, got: {}",
            result.stdout.trim()
        );
    }

    /// Below the cap, `setrecursionlimit` behaves exactly as stock Python does — the
    /// clamp is a ceiling, not a rewrite. 100 is under both profiles' defaults.
    #[tokio::test]
    async fn lowering_the_recursion_limit_is_untouched() {
        let _serialize = PYTHON_TEST_GUARD.lock().await;
        let engine = PythonEngine::new();
        let result = engine
            .execute(
                "import sys
sys.setrecursionlimit(100)
print(sys.getrecursionlimit())",
                None,
                30,
            )
            .await
            .expect("execution completes");

        assert!(
            result.success,
            "lowering the limit must work: {:?}",
            result.error
        );
        assert_eq!(
            result.stdout.trim(),
            "100",
            "a limit under the cap passes through"
        );
    }

    /// `sys.exit(1)` is a failure, as it is to any shell; it used to be swallowed
    /// and reported as success. `exit()`/`exit(0)`/`exit(None)` stay successes.
    #[tokio::test]
    async fn system_exit_carries_its_status() {
        let _serialize = PYTHON_TEST_GUARD.lock().await;
        let engine = PythonEngine::new();

        let failed = engine
            .execute("import sys\nprint('partial')\nsys.exit(3)", None, 30)
            .await
            .expect("runs");
        assert!(!failed.success, "a non-zero exit is a failure");
        assert_eq!(
            failed.stdout.trim(),
            "partial",
            "output before the exit is kept"
        );
        assert!(
            failed
                .error
                .as_deref()
                .is_some_and(|e| e.contains("status 3")),
            "{failed:?}"
        );

        let message = engine
            .execute("import sys\nsys.exit('bad input')", None, 30)
            .await
            .expect("runs");
        assert!(!message.success);
        assert!(
            message
                .error
                .as_deref()
                .is_some_and(|e| e.contains("bad input")),
            "{message:?}"
        );

        for clean in [
            "import sys\nsys.exit()",
            "import sys\nsys.exit(0)",
            "import sys\nsys.exit(None)",
        ] {
            let ok = engine.execute(clean, None, 30).await.expect("runs");
            assert!(ok.success, "{clean}: {ok:?}");
            assert!(ok.error.is_none(), "{clean}: {ok:?}");
        }
    }

    /// A single-threaded Tokio runtime is the worst case for the old design: there is
    /// no blocking pool thread with a bigger stack to fall back on. Executing here
    /// proves the interpreter's stack is independent of the runtime's.
    #[tokio::test(flavor = "current_thread")]
    async fn executes_under_current_thread_runtime() {
        let _serialize = PYTHON_TEST_GUARD.lock().await;
        let engine = PythonEngine::new();
        let result = engine
            .execute("import json\nprint(json.dumps({'ok': True}))", None, 10)
            .await
            .expect("execute must succeed on a current_thread runtime");

        assert!(result.success, "error: {:?}", result.error);
        assert!(result.stdout.contains("ok"));
    }

    #[tokio::test]
    async fn test_basic_execution() {
        let _serialize = PYTHON_TEST_GUARD.lock().await;
        let engine = PythonEngine::new();
        let result = engine
            .execute("print('hello world')", None, 10)
            .await
            .unwrap();
        assert!(result.success, "error: {:?}", result.error);
        assert_eq!(result.stdout.trim(), "hello world");
    }

    #[tokio::test]
    async fn test_error_handling() {
        let _serialize = PYTHON_TEST_GUARD.lock().await;
        let engine = PythonEngine::new();
        let result = engine
            .execute("raise ValueError('test error')", None, 10)
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.as_ref().unwrap().contains("ValueError"));
    }

    #[tokio::test]
    async fn test_multiline() {
        let _serialize = PYTHON_TEST_GUARD.lock().await;
        let engine = PythonEngine::new();
        let code = r#"
import json
data = {"name": "nanna", "version": "0.1.0"}
print(json.dumps(data, indent=2))
"#;
        let result = engine.execute(code, None, 10).await.unwrap();
        assert!(result.success, "error: {:?}", result.error);
        assert!(result.stdout.contains("nanna"));
    }

    #[tokio::test]
    async fn test_stdlib_modules() {
        let _serialize = PYTHON_TEST_GUARD.lock().await;
        let engine = PythonEngine::new();
        let code = r#"
import os, re, collections, pathlib
print("os:", hasattr(os, 'path'))
print("re:", bool(re.match(r'\d+', '42')))
print("Counter:", collections.Counter('aabbc'))
print("Path:", type(pathlib.Path('.')).__name__)
"#;
        let result = engine.execute(code, None, 10).await.unwrap();
        assert!(
            result.success,
            "stderr: {}, error: {:?}",
            result.stderr, result.error
        );
        assert!(result.stdout.contains("os: True"));
    }

    #[tokio::test]
    async fn test_file_operations() {
        let _serialize = PYTHON_TEST_GUARD.lock().await;
        let engine = PythonEngine::new();
        let code = r"
import tempfile, os
with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
    f.write('hello from python')
    name = f.name
with open(name) as f:
    print(f.read())
os.unlink(name)
";
        let result = engine.execute(code, None, 10).await.unwrap();
        assert!(result.success, "error: {:?}", result.error);
        assert!(result.stdout.contains("hello from python"));
    }

    #[tokio::test]
    async fn test_isolation() {
        let _serialize = PYTHON_TEST_GUARD.lock().await;
        let engine = PythonEngine::new();
        // Set a variable in first execution
        let _ = engine.execute("shared_var = 42", None, 10).await.unwrap();
        // Second execution should not see it
        let result = engine.execute("print(shared_var)", None, 10).await.unwrap();
        assert!(!result.success, "Variable leaked between executions!");
    }

    #[tokio::test]
    async fn test_syntax_error() {
        let _serialize = PYTHON_TEST_GUARD.lock().await;
        let engine = PythonEngine::new();
        let result = engine.execute("def foo(", None, 10).await.unwrap();
        assert!(!result.success);
        assert!(result.error.is_some());
    }

    #[tokio::test]
    async fn test_code_with_triple_quotes() {
        let _serialize = PYTHON_TEST_GUARD.lock().await;
        let engine = PythonEngine::new();
        let code = r"
msg = '''hello
world'''
print(msg)
";
        let result = engine.execute(code, None, 10).await.unwrap();
        assert!(result.success, "error: {:?}", result.error);
        assert!(result.stdout.contains("hello\nworld"));
    }
}
