//! Vision: the callback type the daemon's `vision.analyze` service holds, and
//! the image reading it shares with the bundled skills.

use std::sync::Arc;

/// Callback for analyzing images with a vision model
pub type VisionFn = Arc<
    dyn Fn(String, String, String) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String, String>> + Send>>
        + Send
        + Sync,
>;

/// Ceiling on an image read off disk for a vision call.
///
/// Derived, not chosen: the provider caps a request at **32 MB**, and base64
/// inflates bytes by 4/3, so a raw image above `32 MB × 3/4` cannot fit in a
/// request however it is framed. Refusing it here names that reason; sending it
/// spends the read and the encode to earn a provider-side rejection.
pub const IMAGE_BYTES_MAX: usize = 24 * 1024 * 1024;

/// The media type for an image path, by extension.
///
/// Extension rather than magic bytes on purpose: the media type is a *claim
/// made to the provider*, the provider accepts exactly these four, and a file
/// whose bytes disagree with its name fails at the provider with a clear error
/// rather than silently being relabelled here.
///
/// # Errors
/// Returns `Err` naming the accepted set, so a caller passing a PDF or a TIFF
/// learns what to convert to instead of getting a generic failure.
pub fn image_media_type(path: &std::path::Path) -> Result<&'static str, String> {
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    match extension.as_str() {
        "jpg" | "jpeg" => Ok("image/jpeg"),
        "png" => Ok("image/png"),
        "gif" => Ok("image/gif"),
        "webp" => Ok("image/webp"),
        "" => Err(format!(
            "{} has no extension, so its image type cannot be named; \
             vision accepts .jpg, .jpeg, .png, .gif and .webp",
            path.display()
        )),
        other => Err(format!(
            "'.{other}' is not an image type vision accepts (.jpg, .jpeg, .png, .gif, .webp)"
        )),
    }
}

/// Read an image off disk as base64 plus the media type to declare for it.
///
/// Checks the size from `metadata()` **before** the bytes are buffered, the
/// same order `read_pdf` uses — an oversized file should be refused, not read
/// and then refused.
///
/// # Errors
/// Returns a message naming the file when the extension is not an accepted
/// image type, the file cannot be stat'd or read, or it exceeds
/// [`IMAGE_BYTES_MAX`].
pub async fn read_image_as_base64(
    path: &std::path::Path,
) -> Result<(String, &'static str), String> {
    let media_type = image_media_type(path)?;

    let size_bytes = tokio::fs::metadata(path)
        .await
        .map_err(|e| format!("Cannot stat {}: {e}", path.display()))?
        .len();
    if size_bytes > IMAGE_BYTES_MAX as u64 {
        return Err(format!(
            "Image too large: {size_bytes} bytes (max {IMAGE_BYTES_MAX}); \
             above this it cannot fit a provider request once base64-encoded"
        ));
    }

    let bytes = tokio::fs::read(path)
        .await
        .map_err(|e| format!("Cannot read {}: {e}", path.display()))?;
    debug_assert!(
        bytes.len() as u64 == size_bytes,
        "the file changed size between stat and read",
    );
    Ok((base64_simd::STANDARD.encode_to_string(&bytes), media_type))
}

#[cfg(test)]
mod image_input_tests {
    use super::{IMAGE_BYTES_MAX, image_media_type, read_image_as_base64};
    use std::path::Path;

    #[test]
    fn the_accepted_image_types_map_to_their_media_types() {
        for (name, expected) in [
            ("a.png", "image/png"),
            ("a.jpg", "image/jpeg"),
            ("a.jpeg", "image/jpeg"),
            ("a.gif", "image/gif"),
            ("a.webp", "image/webp"),
            ("SHOUTING.PNG", "image/png"),
        ] {
            assert_eq!(image_media_type(Path::new(name)), Ok(expected), "{name}");
        }
    }

    #[test]
    fn a_non_image_extension_is_refused_naming_what_is_accepted() {
        let err = image_media_type(Path::new("scan.pdf")).unwrap_err();
        assert!(err.contains(".png"), "the refusal must list what works: {err}");
        assert!(err.contains("pdf"), "the refusal must name the input: {err}");
    }

    #[test]
    fn a_file_with_no_extension_is_refused_rather_than_guessed() {
        assert!(image_media_type(Path::new("screenshot")).is_err());
    }

    #[tokio::test]
    async fn a_real_image_round_trips_to_base64_and_its_media_type() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("probe.png");
        tokio::fs::write(&path, b"not really a png, but bytes are bytes")
            .await
            .unwrap();

        let (encoded, media_type) = read_image_as_base64(&path).await.unwrap();
        assert_eq!(media_type, "image/png");
        let decoded = base64_simd::STANDARD.decode_to_vec(&encoded).unwrap();
        assert_eq!(decoded, b"not really a png, but bytes are bytes");
    }

    #[tokio::test]
    async fn a_missing_file_is_refused_by_name() {
        let dir = tempfile::tempdir().unwrap();
        let err = read_image_as_base64(&dir.path().join("absent.png"))
            .await
            .unwrap_err();
        assert!(err.contains("absent.png"), "unhelpful refusal: {err}");
    }

    #[tokio::test]
    async fn an_oversized_image_is_refused_before_it_is_buffered() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("huge.png");
        // Sparse: the ceiling is checked from metadata, so this costs no RAM
        // and no disk beyond the file length.
        let file = std::fs::File::create(&path).unwrap();
        file.set_len(IMAGE_BYTES_MAX as u64 + 1).unwrap();
        drop(file);

        let err = read_image_as_base64(&path).await.unwrap_err();
        assert!(err.contains("too large"), "unhelpful refusal: {err}");
    }
}
