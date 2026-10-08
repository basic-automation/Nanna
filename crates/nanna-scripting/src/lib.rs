#![warn(clippy::all)]
#![warn(clippy::pedantic, clippy::nursery)]

//! JavaScript/TypeScript scripting engine for Nanna
//!
//! Provides a unified interface for executing user-authored tools written in JS/TS.
//! Runs them on Boa, a pure-Rust engine.
//!
//! # Features
//!
//! - `boa` (default): Pure Rust JavaScript engine, lightweight (~5MB)
//! - `python`: an embedded Python interpreter (`RustPython`)
//!
//! # Example
//!
//! ```ignore
//! use nanna_scripting::{ScriptEngine, ScriptedTool};
//!
//! let engine = ScriptEngine::new();
//!
//! let tool = ScriptedTool::new("greet", r#"
//!     export default {
//!         name: "greet",
//!         description: "Greet someone",
//!         execute({ name }) {
//!             return `Hello, ${name}!`;
//!         }
//!     }
//! "#);
//!
//! let result = engine.execute(&tool, json!({"name": "World"})).await?;
//! ```

mod engine;
mod tool;
mod bridge;
/// Pre-write snapshots behind every script `writeFile`, for undo.
pub mod file_history;

#[cfg(feature = "boa")]
mod boa_impl;

/// Parse a tool's source exactly as the Boa runtime would, without running it.
/// See [`boa_impl::check_syntax`] — this is the workspace's only syntax gate for
/// shipped JS/TS skills.
#[cfg(feature = "boa")]
pub use boa_impl::check_syntax;

#[cfg(feature = "python")]
pub mod python;

pub use engine::{ScriptEngine, EngineKind, ExecutionResult, BridgeCapabilities};
pub use tool::{ScriptedTool, ToolManifest, ToolPermissions, OutputTarget, extract_manifest};
pub use bridge::{NannaBridge, ServiceFn, ServiceMap, ToolSearchFn, DEFAULT_TOOL_SEARCH_LIMIT};

use thiserror::Error;

#[derive(Error, Debug)]
pub enum ScriptError {
    #[error("Parse error: {0}")]
    Parse(String),

    #[error("Execution error: {0}")]
    Execution(String),

    #[error("Timeout after {0}ms")]
    Timeout(u64),

    #[error("Permission denied: {0}")]
    Permission(String),

    #[error("TypeScript transpilation failed: {0}")]
    Transpile(String),

    #[error("Tool export invalid: {0}")]
    InvalidExport(String),

    #[error("Engine not available: {0}")]
    EngineNotAvailable(String),

    #[error("Bridge error: {0}")]
    Bridge(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, ScriptError>;

/// Milliseconds since `start`, as the `u64` every result reports.
///
/// Saturates rather than truncating; a `u128` of milliseconds only exceeds
/// `u64::MAX` after some 584 million years.
pub(crate) fn elapsed_ms(start: std::time::Instant) -> u64 {
    u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX)
}

/// `n as f64`, spelled without a lossy cast.
///
/// Both 32-bit halves convert to `f64` exactly, scaling the high half by 2^32
/// is exact, and IEEE-754 addition rounds the exact sum once (to nearest, ties
/// to even) — the same single rounding `as` performs. The result is therefore
/// bit-identical to `n as f64` for every `u64`, not merely close to it.
#[cfg(feature = "boa")]
pub(crate) fn u64_to_f64(n: u64) -> f64 {
    let [h0, h1, h2, h3, l0, l1, l2, l3] = n.to_be_bytes();
    let high = f64::from(u32::from_be_bytes([h0, h1, h2, h3])) * 4_294_967_296.0;
    let low = f64::from(u32::from_be_bytes([l0, l1, l2, l3]));
    high + low
}

/// `n as f64` for a signed value; bit-identical for the same reason as
/// [`u64_to_f64`], since round-to-nearest-even is symmetric about zero.
#[cfg(feature = "boa")]
pub(crate) fn i64_to_f64(n: i64) -> f64 {
    let magnitude = u64_to_f64(n.unsigned_abs());
    if n < 0 { -magnitude } else { magnitude }
}

/// `n as u64`, spelled without a lossy cast: truncates toward zero, maps NaN
/// and everything below 1 to 0, and saturates at `u64::MAX`.
///
/// For `1 <= n < 2^64` the value is `mantissa * 2^(exponent - 52)` with the
/// implicit leading bit restored, so shifting the 53-bit mantissa by the
/// unbiased exponent yields exactly the integer part.
#[cfg(feature = "boa")]
pub(crate) fn f64_to_u64(n: f64) -> u64 {
    const MANTISSA_BITS: u64 = 52;
    const EXPONENT_BIAS: u64 = 1023;
    // NaN fails both comparisons below, so it reaches neither branch.
    if n.is_nan() || n < 1.0 {
        return 0;
    }
    if n >= 18_446_744_073_709_551_616.0 {
        return u64::MAX;
    }
    let bits = n.to_bits();
    let mantissa = (bits & ((1 << MANTISSA_BITS) - 1)) | (1 << MANTISSA_BITS);
    // 1 <= n < 2^64, so the biased exponent is in 1023..=1086.
    let exponent = ((bits >> MANTISSA_BITS) & 0x7ff) - EXPONENT_BIAS;
    if exponent >= MANTISSA_BITS {
        mantissa << (exponent - MANTISSA_BITS)
    } else {
        mantissa >> (MANTISSA_BITS - exponent)
    }
}

/// `n as usize`: [`f64_to_u64`], then saturated to `usize`, which is exactly
/// how `as` saturates on a target where `usize` is narrower than 64 bits.
#[cfg(feature = "boa")]
pub(crate) fn f64_to_usize(n: f64) -> usize {
    usize::try_from(f64_to_u64(n)).unwrap_or(usize::MAX)
}

#[cfg(all(test, feature = "boa"))]
mod conversion_tests {
    use super::{f64_to_u64, f64_to_usize, i64_to_f64, u64_to_f64};

    #[test]
    fn u64_to_f64_matches_as_at_the_rounding_edges() {
        assert_eq!(u64_to_f64(0).to_bits(), 0.0_f64.to_bits());
        assert_eq!(
            u64_to_f64(9_007_199_254_740_991).to_bits(),
            9_007_199_254_740_991.0_f64.to_bits()
        );
        // 2^53 + 1 and 2^53 + 3 are ties; `as` rounds both to the even mantissa.
        assert_eq!(
            u64_to_f64(9_007_199_254_740_993).to_bits(),
            9_007_199_254_740_992.0_f64.to_bits()
        );
        assert_eq!(
            u64_to_f64(9_007_199_254_740_995).to_bits(),
            9_007_199_254_740_996.0_f64.to_bits()
        );
        assert_eq!(
            u64_to_f64(u64::MAX).to_bits(),
            18_446_744_073_709_551_616.0_f64.to_bits()
        );
    }

    #[test]
    fn i64_to_f64_is_symmetric() {
        assert_eq!(i64_to_f64(-42).to_bits(), (-42.0_f64).to_bits());
        assert_eq!(
            i64_to_f64(-9_007_199_254_740_993).to_bits(),
            (-9_007_199_254_740_992.0_f64).to_bits()
        );
        assert_eq!(
            i64_to_f64(i64::MIN).to_bits(),
            (-9_223_372_036_854_775_808.0_f64).to_bits()
        );
    }

    #[test]
    fn f64_to_u64_truncates_and_saturates_like_as() {
        assert_eq!(f64_to_u64(f64::NAN), 0);
        assert_eq!(f64_to_u64(-1.5), 0);
        assert_eq!(f64_to_u64(-0.0), 0);
        assert_eq!(f64_to_u64(0.999), 0);
        assert_eq!(f64_to_u64(1.0), 1);
        assert_eq!(f64_to_u64(2.5), 2);
        assert_eq!(f64_to_u64(123_456.789), 123_456);
        assert_eq!(f64_to_u64(4_503_599_627_370_495.5), 4_503_599_627_370_495);
        assert_eq!(f64_to_u64(9_007_199_254_740_992.0), 9_007_199_254_740_992);
        assert_eq!(f64_to_u64(18_446_744_073_709_549_568.0), 18_446_744_073_709_549_568);
        assert_eq!(f64_to_u64(18_446_744_073_709_551_616.0), u64::MAX);
        assert_eq!(f64_to_u64(f64::INFINITY), u64::MAX);
        assert_eq!(f64_to_u64(f64::NEG_INFINITY), 0);
        assert_eq!(f64_to_usize(30.0), 30);
    }
}
