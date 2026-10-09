//! Numeric conversions with no lossless `From`/`TryFrom` form.
//!
//! Every lossy or range-changing conversion the daemon needs lives here, each
//! with the reason it is sound at the call sites that use it, so a reader
//! auditing precision or truncation has one file to read instead of casts
//! scattered through the crate. The float conversions are `nanna_numeric`'s,
//! re-exported so the crate keeps this single import path.

use std::time::Duration;

/// Whole milliseconds in `duration`, saturating at `u64::MAX`.
///
/// `Duration::as_millis` is `u128`. `u64::MAX` milliseconds is roughly 584
/// million years, so for any elapsed time or wall-clock reading the daemon
/// takes the saturation is unreachable; it only stands where an `as u64` cast
/// would have wrapped silently.
pub fn millis_u64(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

/// A collection length or index as an `i64`, saturating at `i64::MAX`.
///
/// Lengths and indices never exceed `isize::MAX`, which fits `i64` on every
/// supported target, so the saturation is unreachable.
pub fn i64_saturating(value: usize) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

/// A stored counter back to the `u32` it was written from, clamped to range.
///
/// Every write path binds `i64::from(u32)`, so an out-of-range value is
/// unreachable short of an external edit to the row; such a value clamps to
/// the nearest end of `u32` rather than wrapping into an arbitrary count.
pub fn u32_clamped(value: i64) -> u32 {
    u32::try_from(value).unwrap_or(if value < 0 { 0 } else { u32::MAX })
}

pub use nanna_numeric::{f32_from_f64, f64_from_u64, i64_from_f64, u64_from_f64};
