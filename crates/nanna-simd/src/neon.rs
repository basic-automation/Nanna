//! ARM NEON accelerated SIMD operations.
//!
//! NEON is mandatory on `AArch64`, so these functions are always available on
//! Apple Silicon, Raspberry Pi 4+, and all 64-bit ARM devices.
//!
//! Uses 128-bit registers (`float32x4_t` — 4 × f32) with dual accumulators
//! to keep the execution pipeline saturated. FMA (`vfmaq_f32`) is used where
//! applicable for fused multiply-add precision and throughput.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32, vmulq_f32, vst1q_f32,
};

// ---------------------------------------------------------------------------
// Dot product
// ---------------------------------------------------------------------------

/// NEON dot product: processes 8 floats per iteration (dual 4-wide accumulators).
///
/// # Safety
/// Caller must ensure this is running on an `AArch64` CPU (NEON is mandatory).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn dot_product_f32_neon(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let len = a.len();
    let chunks = len / 8;
    let remainder = len % 8;

    // Dual accumulators to exploit instruction-level parallelism
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let offset = i * 8;
        // SAFETY: offset + 8 <= len (loop bound), pointer arithmetic is in-bounds.
        unsafe {
            let va0 = vld1q_f32(a.as_ptr().add(offset));
            let vb0 = vld1q_f32(b.as_ptr().add(offset));
            let va1 = vld1q_f32(a.as_ptr().add(offset + 4));
            let vb1 = vld1q_f32(b.as_ptr().add(offset + 4));

            acc0 = vfmaq_f32(acc0, va0, vb0);
            acc1 = vfmaq_f32(acc1, va1, vb1);
        }
    }

    // Combine accumulators and reduce
    let mut result = vaddvq_f32(vaddq_f32(acc0, acc1));

    // Scalar remainder
    let rem_start = chunks * 8;
    for i in 0..remainder {
        result = a[rem_start + i].mul_add(b[rem_start + i], result);
    }

    result
}

// ---------------------------------------------------------------------------
// Cosine similarity
// ---------------------------------------------------------------------------

/// NEON cosine similarity: computes dot(a,b) / (|a| * |b|) with dual accumulators.
///
/// # Safety
/// Caller must ensure this is running on an `AArch64` CPU.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn cosine_similarity_f32_neon(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let len = a.len();
    let chunks = len / 8;
    let remainder = len % 8;

    let mut dot0 = vdupq_n_f32(0.0);
    let mut dot1 = vdupq_n_f32(0.0);
    let mut na0 = vdupq_n_f32(0.0);
    let mut na1 = vdupq_n_f32(0.0);
    let mut nb0 = vdupq_n_f32(0.0);
    let mut nb1 = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let offset = i * 8;
        // SAFETY: offset + 8 <= len (loop bound), pointer arithmetic is in-bounds.
        unsafe {
            let va0 = vld1q_f32(a.as_ptr().add(offset));
            let vb0 = vld1q_f32(b.as_ptr().add(offset));
            let va1 = vld1q_f32(a.as_ptr().add(offset + 4));
            let vb1 = vld1q_f32(b.as_ptr().add(offset + 4));

            dot0 = vfmaq_f32(dot0, va0, vb0);
            dot1 = vfmaq_f32(dot1, va1, vb1);
            na0 = vfmaq_f32(na0, va0, va0);
            na1 = vfmaq_f32(na1, va1, va1);
            nb0 = vfmaq_f32(nb0, vb0, vb0);
            nb1 = vfmaq_f32(nb1, vb1, vb1);
        }
    }

    let mut dot_sum = vaddvq_f32(vaddq_f32(dot0, dot1));
    let mut mag_a = vaddvq_f32(vaddq_f32(na0, na1));
    let mut mag_b = vaddvq_f32(vaddq_f32(nb0, nb1));

    // Scalar remainder
    let rem_start = chunks * 8;
    for i in 0..remainder {
        let ai = a[rem_start + i];
        let bi = b[rem_start + i];
        dot_sum = ai.mul_add(bi, dot_sum);
        mag_a = ai.mul_add(ai, mag_a);
        mag_b = bi.mul_add(bi, mag_b);
    }

    dot_sum / (mag_a.sqrt() * mag_b.sqrt())
}

// ---------------------------------------------------------------------------
// Normalize (L2 in-place)
// ---------------------------------------------------------------------------

/// NEON vector normalization (L2 in-place).
///
/// # Safety
/// Caller must ensure this is running on an `AArch64` CPU.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn normalize_f32_neon(v: &mut [f32]) {
    // SAFETY: dot_product_f32_neon has the same target_feature requirement.
    let norm = unsafe { dot_product_f32_neon(v, v) }.sqrt();

    if norm > f32::EPSILON {
        let inv_norm = 1.0 / norm;
        let inv_vec = vdupq_n_f32(inv_norm);

        let len = v.len();
        let chunks = len / 4;
        let remainder = len % 4;

        for i in 0..chunks {
            let offset = i * 4;
            // SAFETY: offset + 4 <= len (loop bound), pointer arithmetic is in-bounds.
            unsafe {
                let chunk = vld1q_f32(v.as_ptr().add(offset));
                let scaled = vmulq_f32(chunk, inv_vec);
                vst1q_f32(v.as_mut_ptr().add(offset), scaled);
            }
        }

        let rem_start = chunks * 4;
        for i in 0..remainder {
            v[rem_start + i] *= inv_norm;
        }
    }
}

// ---------------------------------------------------------------------------
// Add (in-place)
// ---------------------------------------------------------------------------

/// NEON vector addition in-place: a += b.
///
/// # Safety
/// Caller must ensure this is running on an `AArch64` CPU.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn add_f32_neon(a: &mut [f32], b: &[f32]) {
    debug_assert_eq!(a.len(), b.len());

    let len = a.len();
    let chunks = len / 4;
    let remainder = len % 4;

    for i in 0..chunks {
        let offset = i * 4;
        // SAFETY: offset + 4 <= len (loop bound), pointer arithmetic is in-bounds.
        unsafe {
            let va = vld1q_f32(a.as_ptr().add(offset));
            let vb = vld1q_f32(b.as_ptr().add(offset));
            let sum = vaddq_f32(va, vb);
            vst1q_f32(a.as_mut_ptr().add(offset), sum);
        }
    }

    let rem_start = chunks * 4;
    for i in 0..remainder {
        a[rem_start + i] += b[rem_start + i];
    }
}

// ---------------------------------------------------------------------------
// Scale (in-place)
// ---------------------------------------------------------------------------

/// NEON scalar multiplication in-place: v *= scalar.
///
/// # Safety
/// Caller must ensure this is running on an `AArch64` CPU.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn scale_f32_neon(v: &mut [f32], scalar: f32) {
    let scalar_vec = vdupq_n_f32(scalar);

    let len = v.len();
    let chunks = len / 4;
    let remainder = len % 4;

    for i in 0..chunks {
        let offset = i * 4;
        // SAFETY: offset + 4 <= len (loop bound), pointer arithmetic is in-bounds.
        unsafe {
            let chunk = vld1q_f32(v.as_ptr().add(offset));
            let scaled = vmulq_f32(chunk, scalar_vec);
            vst1q_f32(v.as_mut_ptr().add(offset), scaled);
        }
    }

    let rem_start = chunks * 4;
    for i in 0..remainder {
        v[rem_start + i] *= scalar;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    #[cfg(target_arch = "aarch64")]
    use super::*;

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_dot_product() {
        // 20 elements: 16 in SIMD (2 x 8-wide iterations) + 4 remainder
        let a: Vec<f32> = (1u16..=20).map(f32::from).collect();
        let b: Vec<f32> = (1u16..=20).map(f32::from).collect();
        let expected: f32 = (1u16..=20).map(|x| f32::from(x * x)).sum();
        let result = unsafe { dot_product_f32_neon(&a, &b) };
        assert!(
            (result - expected).abs() < 1e-3,
            "got {result}, expected {expected}"
        );
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_dot_product_small() {
        // Smaller than one dual-accumulator iteration
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = unsafe { dot_product_f32_neon(&a, &b) };
        assert!((result - 32.0).abs() < 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_cosine_similarity_identical() {
        let a: Vec<f32> = (1u16..=32).map(f32::from).collect();
        let b = a.clone();
        let result = unsafe { cosine_similarity_f32_neon(&a, &b) };
        assert!(
            (result - 1.0).abs() < 1e-5,
            "identical vectors should have cosine similarity ~1.0, got {result}"
        );
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_cosine_similarity_orthogonal() {
        let mut a = vec![0.0; 32];
        let mut b = vec![0.0; 32];
        a[0] = 1.0;
        b[1] = 1.0;
        let result = unsafe { cosine_similarity_f32_neon(&a, &b) };
        assert!(
            result.abs() < 1e-5,
            "orthogonal vectors should have cosine similarity ~0.0, got {result}"
        );
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_normalize() {
        let mut v: Vec<f32> = (1u16..=20).map(f32::from).collect();
        unsafe { normalize_f32_neon(&mut v) };
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "normalized vector should have unit length, got {norm}"
        );
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_add() {
        let mut a: Vec<f32> = (1u16..=20).map(f32::from).collect();
        let b: Vec<f32> = (1u16..=20).map(f32::from).collect();
        unsafe { add_f32_neon(&mut a, &b) };
        for ((i, &val), n) in a.iter().enumerate().zip(1u16..) {
            let expected = f32::from(n * 2);
            assert!(
                (val - expected).abs() < 1e-5,
                "a[{i}] = {val}, expected {expected}"
            );
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_scale() {
        let mut v: Vec<f32> = (1u16..=20).map(f32::from).collect();
        unsafe { scale_f32_neon(&mut v, 3.0) };
        for ((i, &val), n) in v.iter().enumerate().zip(1u16..) {
            let expected = f32::from(n * 3);
            assert!(
                (val - expected).abs() < 1e-5,
                "v[{i}] = {val}, expected {expected}"
            );
        }
    }
}
