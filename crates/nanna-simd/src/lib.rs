#![warn(clippy::all)]
#![warn(clippy::pedantic, clippy::nursery)]

//! SIMD-accelerated operations for Nanna
//!
//! Provides runtime-dispatched SIMD operations with architecture-specific tiers:
//!
//! ## `x86_64`
//! - **AVX-512**: 16-wide f32 operations using `std::arch` intrinsics
//! - **AVX2**: 8-wide f32 operations using the `wide` crate
//! - **Scalar**: portable fallback
//!
//! ## `AArch64` (Apple Silicon, ARM servers, mobile)
//! - **NEON**: 4-wide f32 with dual accumulators and FMA (always available on `AArch64`)
//!
//! ## Other architectures
//! - **Scalar**: portable fallback
//!
//! Dispatch is automatic — the best available instruction set is detected at
//! runtime (`x86_64`) or compile time (`AArch64`) and cached for the process lifetime.

#[cfg(target_arch = "x86_64")]
mod avx512;

#[cfg(target_arch = "aarch64")]
mod neon;

use half::f16;

#[cfg(target_arch = "x86_64")]
use wide::f32x8;

// ---------------------------------------------------------------------------
// Runtime feature detection (cached)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod dispatch {
    use std::sync::atomic::{AtomicU8, Ordering};

    /// 0 = unprobed, 1 = AVX-512, 2 = AVX2, 3 = scalar
    static TIER: AtomicU8 = AtomicU8::new(0);

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum SimdTier {
        Avx512,
        Avx2,
        Scalar,
    }

    impl std::fmt::Display for SimdTier {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::Avx512 => write!(f, "AVX-512"),
                Self::Avx2 => write!(f, "AVX2"),
                Self::Scalar => write!(f, "scalar"),
            }
        }
    }

    #[inline]
    pub fn detect() -> SimdTier {
        let cached = TIER.load(Ordering::Relaxed);
        if cached != 0 {
            return match cached {
                1 => SimdTier::Avx512,
                2 => SimdTier::Avx2,
                _ => SimdTier::Scalar,
            };
        }

        let tier = probe();
        TIER.store(
            match tier {
                SimdTier::Avx512 => 1,
                SimdTier::Avx2 => 2,
                SimdTier::Scalar => 3,
            },
            Ordering::Relaxed,
        );
        tier
    }

    fn probe() -> SimdTier {
        if is_x86_feature_detected!("avx512f") {
            SimdTier::Avx512
        } else if is_x86_feature_detected!("avx2") {
            SimdTier::Avx2
        } else {
            SimdTier::Scalar
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod dispatch {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum SimdTier {
        Neon,
        Scalar,
    }

    impl std::fmt::Display for SimdTier {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::Neon => write!(f, "NEON"),
                Self::Scalar => write!(f, "scalar"),
            }
        }
    }

    /// NEON is mandatory on `AArch64` — no runtime probing needed.
    #[inline]
    pub const fn detect() -> SimdTier {
        SimdTier::Neon
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
mod dispatch {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum SimdTier {
        Scalar,
    }

    impl std::fmt::Display for SimdTier {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "scalar")
        }
    }

    #[inline]
    pub const fn detect() -> SimdTier {
        SimdTier::Scalar
    }
}

pub use dispatch::SimdTier;

/// Returns the SIMD tier in use on this machine.
#[cfg(target_arch = "x86_64")]
#[inline]
#[must_use]
pub fn simd_tier() -> SimdTier {
    dispatch::detect()
}

/// Returns the SIMD tier in use on this machine: fixed at compile time off
/// `x86_64`, so it can be `const` there.
#[cfg(not(target_arch = "x86_64"))]
#[inline]
#[must_use]
pub const fn simd_tier() -> SimdTier {
    dispatch::detect()
}

// ---------------------------------------------------------------------------
// Public API — unchanged signatures, architecture dispatch inside
// ---------------------------------------------------------------------------

/// SIMD-accelerated dot product for f32 vectors.
///
/// Dispatches to the best available instruction set:
/// - `x86_64`: AVX-512 (16-wide) → AVX2 (8-wide) → scalar
/// - `AArch64`: NEON (4-wide, dual accumulators)
///
/// # Panics
///
/// Panics if `a` and `b` have different lengths.
#[inline]
#[must_use]
pub fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vectors must have equal length");

    #[cfg(target_arch = "x86_64")]
    {
        match dispatch::detect() {
            SimdTier::Avx512 => unsafe { avx512::dot_product_f32_avx512(a, b) },
            SimdTier::Avx2 => dot_product_f32_avx2(a, b),
            SimdTier::Scalar => dot_product_f32_scalar(a, b),
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon::dot_product_f32_neon(a, b) }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    dot_product_f32_scalar(a, b)
}

/// SIMD-accelerated cosine similarity for f32 vectors.
///
/// # Panics
///
/// Panics if `a` and `b` have different lengths.
#[inline]
#[must_use]
pub fn cosine_similarity_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vectors must have equal length");

    #[cfg(target_arch = "x86_64")]
    {
        match dispatch::detect() {
            SimdTier::Avx512 => {
                unsafe { avx512::cosine_similarity_f32_avx512(a, b) }
            }
            SimdTier::Avx2 => cosine_similarity_f32_avx2(a, b),
            SimdTier::Scalar => cosine_similarity_f32_scalar(a, b),
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon::cosine_similarity_f32_neon(a, b) }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    cosine_similarity_f32_scalar(a, b)
}

/// SIMD-accelerated vector normalization (L2 in-place).
#[inline]
pub fn normalize_f32(v: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        match dispatch::detect() {
            SimdTier::Avx512 => {
                unsafe { avx512::normalize_f32_avx512(v) };
            }
            SimdTier::Avx2 => {
                normalize_f32_avx2(v);
            }
            SimdTier::Scalar => {
                normalize_f32_scalar(v);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon::normalize_f32_neon(v) };
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    normalize_f32_scalar(v);
}

/// SIMD-accelerated vector addition in-place: a += b.
///
/// # Panics
///
/// Panics if `a` and `b` have different lengths.
#[inline]
pub fn add_f32(a: &mut [f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    {
        match dispatch::detect() {
            SimdTier::Avx512 => {
                unsafe { avx512::add_f32_avx512(a, b) };
            }
            SimdTier::Avx2 => {
                add_f32_avx2(a, b);
            }
            SimdTier::Scalar => {
                add_f32_scalar(a, b);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon::add_f32_neon(a, b) };
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    add_f32_scalar(a, b);
}

/// SIMD-accelerated scalar multiplication in-place.
#[inline]
pub fn scale_f32(v: &mut [f32], scalar: f32) {
    #[cfg(target_arch = "x86_64")]
    {
        match dispatch::detect() {
            SimdTier::Avx512 => unsafe { avx512::scale_f32_avx512(v, scalar) },
            SimdTier::Avx2 => scale_f32_avx2(v, scalar),
            SimdTier::Scalar => scale_f32_scalar(v, scalar),
        }
    }

    #[cfg(target_arch = "aarch64")]
    unsafe { neon::scale_f32_neon(v, scalar) };

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    scale_f32_scalar(v, scalar);
}

// ---------------------------------------------------------------------------
// AVX2 tier (8-wide via `wide` crate) — x86_64 only
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
fn dot_product_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    let chunks = a.len() / 8;
    let remainder = a.len() % 8;
    let mut sum = f32x8::ZERO;

    for i in 0..chunks {
        let offset = i * 8;
        let va = f32x8::from(&a[offset..offset + 8]);
        let vb = f32x8::from(&b[offset..offset + 8]);
        sum += va * vb;
    }

    let mut result: f32 = sum.reduce_add();
    let remainder_start = chunks * 8;
    for i in 0..remainder {
        result = a[remainder_start + i].mul_add(b[remainder_start + i], result);
    }
    result
}

#[cfg(target_arch = "x86_64")]
fn cosine_similarity_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    let chunks = a.len() / 8;
    let remainder = a.len() % 8;

    let mut dot = f32x8::ZERO;
    let mut norm_a = f32x8::ZERO;
    let mut norm_b = f32x8::ZERO;

    for i in 0..chunks {
        let offset = i * 8;
        let va = f32x8::from(&a[offset..offset + 8]);
        let vb = f32x8::from(&b[offset..offset + 8]);
        dot += va * vb;
        norm_a += va * va;
        norm_b += vb * vb;
    }

    let mut dot_sum: f32 = dot.reduce_add();
    let mut mag_a: f32 = norm_a.reduce_add();
    let mut mag_b: f32 = norm_b.reduce_add();

    let remainder_start = chunks * 8;
    for i in 0..remainder {
        let ai = a[remainder_start + i];
        let bi = b[remainder_start + i];
        dot_sum = ai.mul_add(bi, dot_sum);
        mag_a = ai.mul_add(ai, mag_a);
        mag_b = bi.mul_add(bi, mag_b);
    }

    dot_sum / (mag_a.sqrt() * mag_b.sqrt())
}

#[cfg(target_arch = "x86_64")]
fn normalize_f32_avx2(v: &mut [f32]) {
    let norm = dot_product_f32_avx2(v, v).sqrt();
    if norm > f32::EPSILON {
        let inv_norm = 1.0 / norm;
        let chunks = v.len() / 8;
        let remainder = v.len() % 8;
        let inv_norm_vec = f32x8::splat(inv_norm);

        for i in 0..chunks {
            let offset = i * 8;
            let mut chunk = f32x8::from(&v[offset..offset + 8]);
            chunk *= inv_norm_vec;
            v[offset..offset + 8].copy_from_slice(chunk.as_array());
        }

        let remainder_start = chunks * 8;
        for i in 0..remainder {
            v[remainder_start + i] *= inv_norm;
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn add_f32_avx2(a: &mut [f32], b: &[f32]) {
    let chunks = a.len() / 8;
    let remainder = a.len() % 8;

    for i in 0..chunks {
        let offset = i * 8;
        let mut va = f32x8::from(&a[offset..offset + 8]);
        let vb = f32x8::from(&b[offset..offset + 8]);
        va += vb;
        a[offset..offset + 8].copy_from_slice(va.as_array());
    }

    let remainder_start = chunks * 8;
    for i in 0..remainder {
        a[remainder_start + i] += b[remainder_start + i];
    }
}

#[cfg(target_arch = "x86_64")]
fn scale_f32_avx2(v: &mut [f32], scalar: f32) {
    let chunks = v.len() / 8;
    let remainder = v.len() % 8;
    let scalar_vec = f32x8::splat(scalar);

    for i in 0..chunks {
        let offset = i * 8;
        let mut chunk = f32x8::from(&v[offset..offset + 8]);
        chunk *= scalar_vec;
        v[offset..offset + 8].copy_from_slice(chunk.as_array());
    }

    let remainder_start = chunks * 8;
    for i in 0..remainder {
        v[remainder_start + i] *= scalar;
    }
}

// ---------------------------------------------------------------------------
// Scalar tier — pure Rust fallback (x86_64 scalar path + generic architectures)
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "aarch64"))]
fn dot_product_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(not(target_arch = "aarch64"))]
fn cosine_similarity_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0_f32;
    let mut mag_a = 0.0_f32;
    let mut mag_b = 0.0_f32;
    for (ai, bi) in a.iter().zip(b.iter()) {
        dot += ai * bi;
        mag_a += ai * ai;
        mag_b += bi * bi;
    }
    dot / (mag_a.sqrt() * mag_b.sqrt())
}

#[cfg(not(target_arch = "aarch64"))]
fn normalize_f32_scalar(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        let inv = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn add_f32_scalar(a: &mut [f32], b: &[f32]) {
    for (x, y) in a.iter_mut().zip(b.iter()) {
        *x += y;
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn scale_f32_scalar(v: &mut [f32], scalar: f32) {
    for x in v.iter_mut() {
        *x *= scalar;
    }
}

// ---------------------------------------------------------------------------
// Non-dispatched utilities (unchanged)
// ---------------------------------------------------------------------------

/// Convert f16 array to f32 (for GPU compatibility).
///
/// # Panics
///
/// Panics if `src` and `dst` have different lengths.
#[inline]
pub fn f16_to_f32(src: &[f16], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len());
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = s.to_f32();
    }
}

/// Convert f32 array to f16 (for memory efficiency).
///
/// # Panics
///
/// Panics if `src` and `dst` have different lengths.
#[inline]
pub fn f32_to_f16(src: &[f32], dst: &mut [f16]) {
    assert_eq!(src.len(), dst.len());
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = f16::from_f32(*s);
    }
}

/// SIMD-accelerated argmax.
#[inline]
#[must_use]
pub fn argmax_f32(v: &[f32]) -> usize {
    if v.is_empty() {
        return 0;
    }

    let mut max_idx = 0;
    let mut max_val = v[0];

    // For argmax, scalar is often faster due to branch prediction.
    // SIMD horizontal reduction has overhead that rarely pays off here.
    for (i, &val) in v.iter().enumerate().skip(1) {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }

    max_idx
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_tier_detection() {
        let tier = simd_tier();
        eprintln!("detected SIMD tier: {tier}");
        // Should not panic; on x86_64 should be at least AVX2, on aarch64 always NEON
    }

    #[test]
    fn test_dot_product() {
        let a: Vec<f32> = (1..=20_u8).map(f32::from).collect();
        let b: Vec<f32> = (1..=20_u8).map(f32::from).collect();
        let expected: f32 = (1..=20_u16).map(|x| f32::from(x * x)).sum();
        let result = dot_product_f32(&a, &b);
        assert!(
            (result - expected).abs() < 1e-3,
            "got {result}, expected {expected}"
        );
    }

    #[test]
    fn test_dot_product_small() {
        // Smaller than one SIMD register — exercises remainder path
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = dot_product_f32(&a, &b);
        assert!((result - 32.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a: Vec<f32> = (1..=32_u8).map(f32::from).collect();
        let b = a.clone();
        let result = cosine_similarity_f32(&a, &b);
        assert!(
            (result - 1.0).abs() < 1e-5,
            "identical vectors should have cosine similarity ~1.0, got {result}"
        );
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let mut a = vec![0.0; 32];
        let mut b = vec![0.0; 32];
        a[0] = 1.0;
        b[1] = 1.0;
        let result = cosine_similarity_f32(&a, &b);
        assert!(
            result.abs() < 1e-5,
            "orthogonal vectors should have cosine similarity ~0.0, got {result}"
        );
    }

    #[test]
    fn test_normalize() {
        let mut v: Vec<f32> = (1..=20_u8).map(f32::from).collect();
        normalize_f32(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "normalized vector should have unit length, got {norm}"
        );
    }

    #[test]
    fn test_add() {
        let mut a: Vec<f32> = (1..=20_u8).map(f32::from).collect();
        let b: Vec<f32> = (1..=20_u8).map(f32::from).collect();
        add_f32(&mut a, &b);
        for (i, (&val, n)) in a.iter().zip(1_u8..).enumerate() {
            let expected = f32::from(n * 2);
            assert!(
                (val - expected).abs() < 1e-5,
                "a[{i}] = {val}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_scale() {
        let mut v: Vec<f32> = (1..=20_u8).map(f32::from).collect();
        scale_f32(&mut v, 3.0);
        for (i, (&val, n)) in v.iter().zip(1_u8..).enumerate() {
            let expected = f32::from(n * 3);
            assert!(
                (val - expected).abs() < 1e-5,
                "v[{i}] = {val}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_empty_vectors() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        assert_eq!(dot_product_f32(&a, &b), 0.0);
    }

    #[test]
    fn test_argmax() {
        let v = vec![1.0, 5.0, 3.0, 9.0, 2.0];
        assert_eq!(argmax_f32(&v), 3);
    }
}
