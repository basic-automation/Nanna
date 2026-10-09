//! OCR tool - extract text from images
//!
//! # Tiered OCR Pipeline
//!
//! `OcrTool` uses a tiered approach:
//! - **Tier 0**: Embedded `ocrs` (pure-Rust ONNX engine, no external deps)
//! - **Tier 1+**: Vision model callbacks in a configurable priority list
//!
//! The tool tries Tier 0 first (if `use_embedded_ocr` is true), then falls
//! through the model list until one succeeds.  If no tier succeeds it returns
//! an error.
//!
//! # Model download
//!
//! The `ocrs` engine requires two `.rten` model files (detection + recognition).
//! On the first call to embedded OCR they are automatically downloaded from the
//! official ocrs release into `~/.cache/ocrs/`.

use crate::{Tool, ToolDefinition, ToolError, ToolResult};
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{info, warn};

// ---------------------------------------------------------------------------
// Public type aliases
// ---------------------------------------------------------------------------

/// Async callback for a vision-model OCR attempt.
///
/// Arguments: `(base64_image_data, prompt, media_type)` → `Result<text, err_msg>`
pub type OcrVisionFn = Arc<
    dyn Fn(String, String, String) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String, String>> + Send>>
        + Send
        + Sync,
>;

// ---------------------------------------------------------------------------
// OcrTool
// ---------------------------------------------------------------------------

/// Tool for extracting text from images using a tiered OCR pipeline.
pub struct OcrTool {
    /// Whether to attempt embedded `ocrs` before trying vision models.
    use_embedded_ocr: bool,
    /// Vision model callbacks in priority order: `(model_name, callback)`.
    vision_models: Vec<(String, OcrVisionFn)>,
}

impl OcrTool {
    /// Create a new `OcrTool` with the default configuration (embedded OCR
    /// enabled, no external vision models).
    #[must_use]
    pub fn new() -> Self {
        Self {
            use_embedded_ocr: true,
            vision_models: Vec::new(),
        }
    }

    /// Enable or disable the embedded `ocrs` tier.
    #[must_use]
    pub const fn with_embedded_ocr(mut self, enabled: bool) -> Self {
        self.use_embedded_ocr = enabled;
        self
    }

    /// Append a vision model callback at the end of the priority list.
    ///
    /// Models are tried in insertion order after embedded OCR (if enabled).
    #[must_use]
    pub fn with_vision_model(mut self, name: impl Into<String>, f: OcrVisionFn) -> Self {
        self.vision_models.push((name.into(), f));
        self
    }

    /// Replace the entire vision model priority list.
    #[must_use]
    pub fn with_vision_models(mut self, models: Vec<(String, OcrVisionFn)>) -> Self {
        self.vision_models = models;
        self
    }

    /// Convenience builder for backward-compatibility: set a single unnamed
    /// vision function (treated as the first and only vision model).
    #[must_use]
    pub fn with_vision_fn(self, f: OcrVisionFn) -> Self {
        self.with_vision_model("vision_model", f)
    }
}

impl Default for OcrTool {
    fn default() -> Self {
        Self::new()
    }
}

const OCR_PROMPT: &str = r"Extract ALL text from this image using OCR.

Instructions:
- Transcribe every piece of text you can see, exactly as written
- Preserve the original formatting, layout, and structure as much as possible
- Include headers, labels, captions, watermarks, handwritten text, etc.
- Use markdown formatting to represent structure (headers, lists, tables)
- If text is unclear or partially visible, indicate with [unclear] or [partial]
- For tables, use markdown table format
- For multi-column layouts, process left-to-right, top-to-bottom

Output the extracted text only, no additional commentary.";

#[async_trait]
impl Tool for OcrTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new("ocr", "Extract text from an image using OCR (optical character recognition)")
            .string_param("image", "Path to image file, URL, or base64-encoded image data", true)
            .string_param("media_type", "MIME type (image/jpeg, image/png, etc.) - required for base64", false)
            .string_param("language", "Expected language hint (e.g., 'english', 'chinese') - optional", false)
    }

    async fn execute(&self, params: HashMap<String, Value>) -> Result<ToolResult, ToolError> {
        let image = params
            .get("image")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidParams("Missing 'image' parameter".to_string()))?;

        let media_type = params
            .get("media_type")
            .and_then(|v| v.as_str())
            .unwrap_or("image/jpeg");

        let language = params.get("language").and_then(|v| v.as_str());

        // Prepare image bytes for embedded OCR / base64 data for vision models
        let (image_data, actual_media_type) = prepare_image(image, media_type).await?;

        // ------------------------------------------------------------------
        // Tier 0: embedded ocrs
        // ------------------------------------------------------------------
        if self.use_embedded_ocr {
            // Decode base64 back to bytes (prepare_image always encodes to b64
            // for file paths; URLs stay as URLs and can't be used by ocrs)
            if !image_data.starts_with("http://") && !image_data.starts_with("https://") {
                let raw_bytes = base64_simd::STANDARD
                    .decode_to_vec(image_data.as_bytes())
                    .unwrap_or_default();

                if !raw_bytes.is_empty() {
                    match embedded_ocr(&raw_bytes).await {
                        Ok(text) if !text.trim().is_empty() => {
                            return Ok(ToolResult::success(text));
                        }
                        Ok(_) => {
                            info!("Embedded OCR returned empty result — falling through to vision models");
                        }
                        Err(e) => {
                            warn!("Embedded OCR failed: {} — falling through to vision models", e);
                        }
                    }
                }
            } else {
                // URL — skip embedded OCR, it can't fetch remote images
                info!("Image is a URL; skipping embedded OCR tier");
            }
        }

        // ------------------------------------------------------------------
        // Tier 1+: vision model priority list
        // ------------------------------------------------------------------
        if self.vision_models.is_empty() && self.use_embedded_ocr {
            // Embedded OCR was the only option and it failed/returned empty
            return Err(ToolError::ExecutionFailed(
                "Embedded OCR returned no text and no vision models are configured".to_string(),
            ));
        }
        if self.vision_models.is_empty() {
            return Err(ToolError::ExecutionFailed(
                "No OCR methods configured (embedded OCR disabled, no vision models set)".to_string(),
            ));
        }

        let prompt = language.map_or_else(
            || OCR_PROMPT.to_string(),
            |lang| format!("{OCR_PROMPT}\n\nLanguage hint: {lang}"),
        );

        for (model_name, vision_fn) in &self.vision_models {
            match vision_fn(image_data.clone(), prompt.clone(), actual_media_type.clone()).await {
                Ok(text) if !text.trim().is_empty() => {
                    return Ok(ToolResult::success(text));
                }
                Ok(_) => {
                    warn!("OCR model '{}' returned empty text — trying next", model_name);
                }
                Err(e) => {
                    warn!("OCR model '{}' failed: {} — trying next", model_name, e);
                }
            }
        }

        Err(ToolError::ExecutionFailed(
            "All OCR methods failed (embedded OCR and all vision models exhausted)".to_string(),
        ))
    }
}

// ---------------------------------------------------------------------------
// Embedded OCR implementation (ocrs + rten)
// ---------------------------------------------------------------------------

/// Path to the ocrs models cache directory (`~/.cache/ocrs/` on Linux/macOS,
/// `%LOCALAPPDATA%\ocrs\` on Windows).
fn ocrs_cache_dir() -> PathBuf {
    use directories::BaseDirs;
    BaseDirs::new().map_or_else(|| PathBuf::from(".ocrs_cache"), |d| d.cache_dir().join("ocrs"))
}

const DETECTION_MODEL_URL: &str =
    "https://ocrs-models.s3-accelerate.amazonaws.com/text-detection.rten";
const RECOGNITION_MODEL_URL: &str =
    "https://ocrs-models.s3-accelerate.amazonaws.com/text-recognition.rten";

/// Download a file to a local path if it doesn't already exist.
async fn download_if_missing(url: &str, dest: &Path) -> Result<(), String> {
    if dest.exists() {
        return Ok(());
    }
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create cache dir: {e}"))?;
    }

    info!("Downloading OCR model from {} → {:?}", url, dest);

    let response = reqwest::get(url)
        .await
        .map_err(|e| format!("Failed to download {url}: {e}"))?;

    if !response.status().is_success() {
        return Err(format!("HTTP {} when downloading {}", response.status(), url));
    }

    let bytes = response
        .bytes()
        .await
        .map_err(|e| format!("Failed to read model bytes: {e}"))?;

    tokio::fs::write(dest, &bytes)
        .await
        .map_err(|e| format!("Failed to write model to {}: {e}", dest.display()))?;

    info!("OCR model saved to {:?} ({} bytes)", dest, bytes.len());
    Ok(())
}

/// Run embedded OCR on raw image bytes.
///
/// This function:
/// 1. Downloads the `.rten` model files on first use (into `~/.cache/ocrs/`)
/// 2. Decodes the image bytes via the `image` crate
/// 3. Feeds the pixels to `ocrs::OcrEngine`
/// 4. Returns the extracted text
///
/// The blocking model-load + inference is delegated to a `spawn_blocking`
/// thread to keep the async runtime happy.
async fn embedded_ocr(image_bytes: &[u8]) -> Result<String, String> {
    let cache = ocrs_cache_dir();
    let detection_path = cache.join("text-detection.rten");
    let recognition_path = cache.join("text-recognition.rten");

    // Ensure models are present (download if needed)
    download_if_missing(DETECTION_MODEL_URL, &detection_path).await?;
    download_if_missing(RECOGNITION_MODEL_URL, &recognition_path).await?;

    let image_bytes = image_bytes.to_vec(); // clone for move into blocking task
    let detection_path = detection_path.clone();
    let recognition_path = recognition_path.clone();

    tokio::task::spawn_blocking(move || {
        run_ocrs_sync(&image_bytes, &detection_path, &recognition_path)
    })
    .await
    .map_err(|e| format!("OCR task panicked: {e}"))?
}

/// Synchronous OCR extraction using ocrs.
fn run_ocrs_sync(
    image_bytes: &[u8],
    detection_path: &Path,
    recognition_path: &Path,
) -> Result<String, String> {
    use ocrs::{ImageSource, OcrEngine, OcrEngineParams};
    use rten::Model;

    // Load models
    let detection_model = Model::load_file(detection_path)
        .map_err(|e| format!("Failed to load detection model: {e}"))?;
    let recognition_model = Model::load_file(recognition_path)
        .map_err(|e| format!("Failed to load recognition model: {e}"))?;

    // Construct engine
    let engine = OcrEngine::new(OcrEngineParams {
        detection_model: Some(detection_model),
        recognition_model: Some(recognition_model),
        ..Default::default()
    })
    .map_err(|e| format!("Failed to create OcrEngine: {e}"))?;

    // Decode image using the `image` crate
    let img = image::load_from_memory(image_bytes)
        .map(image::DynamicImage::into_rgb8)
        .map_err(|e| format!("Failed to decode image: {e}"))?;

    // Create ImageSource from raw pixels
    let img_source = ImageSource::from_bytes(img.as_raw(), img.dimensions())
        .map_err(|e| format!("Failed to create ImageSource: {e}"))?;

    // Prepare input
    let ocr_input = engine
        .prepare_input(img_source)
        .map_err(|e| format!("Failed to prepare OCR input: {e}"))?;

    // Run detection + recognition (convenience API)
    let text = engine
        .get_text(&ocr_input)
        .map_err(|e| format!("OCR text extraction failed: {e}"))?;

    Ok(text)
}

// ---------------------------------------------------------------------------
// Image-preparation helper
// ---------------------------------------------------------------------------

/// Prepare image data — handles file paths, URLs, and base64.
///
/// Returns `(image_data, media_type)` where `image_data` is either a
/// base64-encoded string (for local files) or the original URL / base64 blob.
async fn prepare_image(
    image: &str,
    default_media_type: &str,
) -> Result<(String, String), ToolError> {
    // Check if it's a file path
    let path = Path::new(image);
    if path.exists() && path.is_file() {
        let bytes = tokio::fs::read(path)
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("Failed to read image file: {e}")))?;

        // Detect media type from extension
        let media_type = match path.extension().and_then(|e| e.to_str()) {
            Some("jpg" | "jpeg") => "image/jpeg",
            Some("png") => "image/png",
            Some("gif") => "image/gif",
            Some("webp") => "image/webp",
            Some("bmp") => "image/bmp",
            Some("svg") => "image/svg+xml",
            _ => default_media_type,
        };

        let base64_data = base64_simd::STANDARD.encode_to_string(&bytes);
        return Ok((base64_data, media_type.to_string()));
    }

    // Check if it's a URL
    if image.starts_with("http://") || image.starts_with("https://") {
        // Return URL as-is — vision models typically accept URLs directly
        return Ok((image.to_string(), default_media_type.to_string()));
    }

    // Assume it's already base64
    Ok((image.to_string(), default_media_type.to_string()))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ocr_tool_definition() {
        let tool = OcrTool::new();
        let def = tool.definition();
        assert_eq!(def.name, "ocr");
    }

    #[tokio::test]
    async fn test_ocr_tool_no_methods_configured() {
        let tool = OcrTool::new().with_embedded_ocr(false);
        let mut params = HashMap::new();
        params.insert(
            "image".to_string(),
            Value::String("dGVzdA==".to_string()), // base64 "test"
        );
        let result = tool.execute(params).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_ocr_tool_builder_vision_fn() {
        let fn_called = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let fn_called2 = fn_called.clone();

        let vision_fn: OcrVisionFn = Arc::new(move |_data, _prompt, _media| {
            fn_called2.store(true, std::sync::atomic::Ordering::SeqCst);
            Box::pin(async { Ok("Hello World".to_string()) })
        });

        // Disable embedded OCR so we go straight to the vision model
        let tool = OcrTool::new()
            .with_embedded_ocr(false)
            .with_vision_fn(vision_fn);

        let mut params = HashMap::new();
        params.insert(
            "image".to_string(),
            Value::String("dGVzdA==".to_string()),
        );
        let result = tool.execute(params).await.expect("should succeed");
        assert!(result.content.contains("Hello World"));
        assert!(fn_called.load(std::sync::atomic::Ordering::SeqCst));
    }
}
