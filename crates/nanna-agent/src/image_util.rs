//! Image preprocessing utilities for LLM API compliance.
//!
//! Resizes and compresses images to fit within provider-specific size limits
//! before sending them as base64-encoded content blocks.

use base64::Engine;
use tracing::{debug, warn};

use crate::numeric::{u32_from_f32, f32_from_u64};

/// Maximum image size (in decoded bytes) per provider.
/// Returns the limit for the given provider prefix, defaulting to the
/// most restrictive (Anthropic's 5 MB) when unknown.
fn max_image_bytes(provider: &str) -> usize {
    match provider {
        "openai" | "google" | "gemini" => 20 * 1024 * 1024,
        "mistral" => 10 * 1024 * 1024,
        // "anthropic", and the safe default for anything unknown.
        _ => 5 * 1024 * 1024,
    }
}

/// Extract the provider prefix from a model spec like `"anthropic/claude-sonnet-4-20250514"`.
/// Returns `""` if there is no slash (which will map to the safe default).
#[must_use]
pub fn provider_from_model(model: &str) -> &str {
    model.split_once('/').map_or("", |(p, _)| p)
}

/// Ensure the base64-encoded image fits within the provider's size limit.
///
/// If the decoded image is already small enough, returns the original
/// `(data, media_type)` unchanged.  Otherwise it progressively reduces
/// JPEG quality and resolution until the image fits.
///
/// # Arguments
/// * `base64_data` – the original base64-encoded image
/// * `media_type`  – MIME type, e.g. `"image/png"`, `"image/jpeg"`
/// * `model`       – full model spec (used to determine provider limit)
///
/// # Returns
/// `(base64_data, media_type)` — possibly re-encoded as `"image/jpeg"`.
pub fn fit_image_to_limit(
    base64_data: &str,
    media_type: &str,
    model: &str,
) -> (String, String) {
    let provider = provider_from_model(model);
    let limit = max_image_bytes(provider);

    // Fast path: check raw decoded size first
    let decoded_len = base64_data.len() * 3 / 4; // approximate
    if decoded_len <= limit {
        return (base64_data.to_owned(), media_type.to_owned());
    }

    // Decode the base64 data
    let raw = match base64::engine::general_purpose::STANDARD.decode(base64_data) {
        Ok(v) => v,
        Err(e) => {
            warn!("Failed to decode base64 image for resizing: {e}");
            return (base64_data.to_owned(), media_type.to_owned());
        }
    };

    // If decoded bytes are actually under limit, keep as-is
    if raw.len() <= limit {
        return (base64_data.to_owned(), media_type.to_owned());
    }

    debug!(
        original_bytes = raw.len(),
        limit,
        provider,
        "Image exceeds provider limit, resizing"
    );

    // Load with the `image` crate
    let img = match image::load_from_memory(&raw) {
        Ok(i) => i,
        Err(e) => {
            warn!("Failed to load image for resizing: {e}");
            return (base64_data.to_owned(), media_type.to_owned());
        }
    };

    if let Some(bytes) = shrink_to_jpeg(&img, limit) {
        let encoded = base64::engine::general_purpose::STANDARD.encode(&bytes);
        return (encoded, "image/jpeg".to_owned());
    }

    warn!(
        "Could not fit image under {limit} bytes after {SHRINK_ATTEMPTS_MAX} attempts, sending as-is"
    );
    (base64_data.to_owned(), media_type.to_owned())
}

/// How many encodes [`shrink_to_jpeg`] tries before giving up: four quality
/// steps at full size (90, 80, 70, 60), then a scale-down (0.75×) at quality
/// 85 and the ladder again.
const SHRINK_ATTEMPTS_MAX: usize = 8;

/// Re-encode `img` as a JPEG of at most `limit` bytes: lower the quality first
/// (90 → 60), then scale down, resetting the quality at each new size.
///
/// Two things made the old loop a no-op on the images that most needed it.
/// The quality ladder was only ever passed to the *fallback* encoder — the
/// first encode used the default quality every time, so "lower the quality"
/// changed nothing until the scale moved. And JPEG has no alpha channel: an
/// RGBA image (every PNG screenshot with transparency) failed that encode
/// *and* the fallback, and went to the provider over its limit. The image is
/// flattened to RGB once, up front, and every attempt uses the ladder.
fn shrink_to_jpeg(img: &image::DynamicImage, limit: usize) -> Option<Vec<u8>> {
    assert!(limit > 0, "a zero limit fits nothing");
    let rgb = image::DynamicImage::ImageRgb8(img.to_rgb8());
    let mut quality = 90u8;
    let mut scale: f32 = 1.0;

    for attempt in 0..SHRINK_ATTEMPTS_MAX {
        let resized = if (scale - 1.0).abs() < f32::EPSILON {
            rgb.clone()
        } else {
            let new_w = u32_from_f32(f32_from_u64(u64::from(rgb.width())) * scale);
            let new_h = u32_from_f32(f32_from_u64(u64::from(rgb.height())) * scale);
            rgb.resize(
                new_w.max(1),
                new_h.max(1),
                image::imageops::FilterType::Lanczos3,
            )
        };

        let mut bytes = Vec::new();
        let jpeg = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut bytes, quality);
        if let Err(e) = resized.write_with_encoder(jpeg) {
            warn!("JPEG encoding failed on attempt {attempt}: {e}");
            return None;
        }
        if bytes.len() <= limit {
            debug!(
                final_bytes = bytes.len(),
                quality,
                scale,
                attempts = attempt + 1,
                "Image resized successfully"
            );
            debug_assert!(bytes.starts_with(&[0xFF, 0xD8]), "a JPEG stream");
            return Some(bytes);
        }

        // Reduce quality first, then start scaling down
        if quality > 60 {
            quality -= 10;
        } else {
            scale *= 0.75;
            quality = 85; // reset quality when we scale down
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_from_model() {
        assert_eq!(provider_from_model("anthropic/claude-sonnet-4-20250514"), "anthropic");
        assert_eq!(provider_from_model("openai/gpt-4o"), "openai");
        assert_eq!(provider_from_model("claude-sonnet-4-20250514"), "");
    }

    #[test]
    fn test_small_image_passthrough() {
        // A tiny 1x1 white JPEG in base64
        let tiny = base64::engine::general_purpose::STANDARD.encode([
            0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46,
        ]);
        let (data, mt) = fit_image_to_limit(&tiny, "image/jpeg", "anthropic/claude-sonnet-4-20250514");
        assert_eq!(data, tiny);
        assert_eq!(mt, "image/jpeg");
    }

    /// Deterministic noise: incompressible enough that JPEG size tracks
    /// quality and resolution, so the ladder has something to do.
    fn noise_image(width: u32, height: u32, alpha: bool) -> image::DynamicImage {
        let mut state = 0x2545_F491_u32;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state.to_le_bytes()[0]
        };
        if alpha {
            image::DynamicImage::ImageRgba8(image::RgbaImage::from_fn(width, height, |_, _| {
                image::Rgba([next(), next(), next(), 128])
            }))
        } else {
            image::DynamicImage::ImageRgb8(image::RgbImage::from_fn(width, height, |_, _| {
                image::Rgb([next(), next(), next()])
            }))
        }
    }

    /// An RGBA image — every transparent PNG screenshot — shrinks like any
    /// other. JPEG has no alpha, and both of the old encodes failed on it, so
    /// such an image was sent to the provider over its limit.
    #[test]
    fn an_image_with_alpha_is_shrunk_to_the_limit() {
        let img = noise_image(256, 256, true);
        let limit = 40 * 1024;
        let bytes = shrink_to_jpeg(&img, limit).expect("an RGBA image fits after shrinking");
        assert!(bytes.len() <= limit, "{} > {limit}", bytes.len());
        let back = image::load_from_memory(&bytes).expect("a decodable JPEG");
        assert_eq!(back.color(), image::ColorType::Rgb8);
    }

    /// The quality ladder is really applied: a limit that full-size quality 90
    /// misses but quality 70 meets is met without losing any resolution. The
    /// old loop encoded at the default quality every time and only scaled.
    #[test]
    fn the_quality_ladder_is_tried_before_the_resolution() {
        let img = noise_image(128, 128, false);
        let size_at = |quality| {
            let mut out = Vec::new();
            img.write_with_encoder(image::codecs::jpeg::JpegEncoder::new_with_quality(
                &mut out, quality,
            ))
            .expect("encode");
            out.len()
        };
        let (q90, q70) = (size_at(90), size_at(70));
        assert!(q70 < q90, "quality changes the size: {q70} vs {q90}");

        let bytes = shrink_to_jpeg(&img, q70).expect("fits at quality 70");
        let back = image::load_from_memory(&bytes).expect("a decodable JPEG");
        assert_eq!(
            (back.width(), back.height()),
            (128, 128),
            "no scale-down was needed"
        );
    }
}
