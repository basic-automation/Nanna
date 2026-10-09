//! PDF reading: text extraction with `lopdf`, page selection, and an OCR pass
//! over embedded images for pages that carry no text.
//!
//! These are functions, not a `Tool`: the daemon's `pdf.read` service
//! (`nanna-daemon` `server.rs`, `pdf_read_services`) calls [`read_pdf_text`]
//! and then [`ocr_empty_pages`] with an [`OcrFn`] bound to the first configured
//! vision model (`vision_service::bind_pdf_ocr_fn`).
//!
//! ## Future: pdfium page rendering
//! Rendering a whole PDF page to pixels (as opposed to extracting embedded
//! image objects) requires a PDF rendering library such as `pdfium-render`.
//! This is not implemented here to avoid a large C dependency; instead, we
//! fall back to extracting embedded image objects from the PDF stream.

use crate::ToolError;
use std::fmt::Write as _;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Public type aliases
// ---------------------------------------------------------------------------

/// OCR callback for the embedded images of pages that have no extractable text.
///
/// Arguments: `(base64_image_data, prompt, media_type)` → `Result<text, err_msg>`.
/// The daemon binds it to the first configured vision model.
pub type OcrFn = Arc<
    dyn Fn(
            String,
            String,
            String,
        )
            -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String, String>> + Send>>
        + Send
        + Sync,
>;

// ---------------------------------------------------------------------------
// Page selection
// ---------------------------------------------------------------------------

/// Which pages of a document to read.
///
/// The `read_pdf` skill has always sent a `pages` string (`"1-5"`, `"3"`) while
/// the tool only ever looked for an integer `max_pages`, so the selection was
/// parsed by the skill, serialised, and silently discarded: asking for page 3
/// of a forty-page contract returned all forty and never said the request had
/// been ignored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageSelection {
    /// Every page in the document.
    All,
    /// The first `n` pages — the legacy `max_pages` parameter.
    First(usize),
    /// An inclusive, 1-based range of page positions.
    Range { first: usize, last: usize },
}

impl PageSelection {
    /// The 1-based positions this selection admits in a document of
    /// `page_count` pages.
    ///
    /// Returns an empty range when the selection starts past the end, so a
    /// caller asking for page 90 of an 80-page file gets nothing plus the
    /// counts to see why — not a silent last page.
    fn positions(self, page_count: usize) -> std::ops::RangeInclusive<usize> {
        debug_assert!(page_count < usize::MAX, "page count must be representable");
        let (first, last) = match self {
            Self::All => (1, page_count),
            Self::First(n) => (1, n.min(page_count)),
            Self::Range { first, last } => (first, last.min(page_count)),
        };
        // `1..=0` is already empty, which is the right answer for an empty
        // document; `90..=80` likewise for a selection past the end.
        first..=last
    }
}

/// Parse a `pages` selection string.
///
/// Accepted forms: `"3"` (one page), `"2-5"` (inclusive range), `"4-"` (from
/// page 4 to the end), `"-4"` (up to page 4), and empty/whitespace (all pages).
///
/// # Errors
///
/// Returns a message naming the offending text when the string is none of
/// those, when a page number is zero (pages are 1-based), or when a range runs
/// backwards — reading `"5-2"` as `2-5` would return pages the caller did not
/// ask for, and reading it as empty would look like an empty document.
pub fn parse_page_selection(spec: &str) -> Result<PageSelection, String> {
    let spec = spec.trim();
    if spec.is_empty() {
        return Ok(PageSelection::All);
    }

    let number = |text: &str| -> Result<usize, String> {
        let n: usize = text
            .trim()
            .parse()
            .map_err(|_| format!("`{spec}` is not a page number or range"))?;
        if n == 0 {
            return Err(format!("`{spec}` names page 0; pages are numbered from 1"));
        }
        Ok(n)
    };

    let Some((head, tail)) = spec.split_once('-') else {
        let only = number(spec)?;
        return Ok(PageSelection::Range {
            first: only,
            last: only,
        });
    };

    let selection = match (head.trim().is_empty(), tail.trim().is_empty()) {
        // "-4": everything up to page 4.
        (true, false) => PageSelection::First(number(tail)?),
        // "4-": page 4 to the end.
        (false, true) => PageSelection::Range {
            first: number(head)?,
            last: usize::MAX,
        },
        // "2-5"
        (false, false) => {
            let first = number(head)?;
            let last = number(tail)?;
            if last < first {
                return Err(format!("`{spec}` runs backwards: {last} is before {first}"));
            }
            PageSelection::Range { first, last }
        }
        // A bare "-".
        (true, true) => return Err(format!("`{spec}` names no pages")),
    };
    Ok(selection)
}

/// Text extracted from a PDF, with the counts a caller needs to page through it.
///
/// Mirrors what `read_file` reports (`total_lines` / `lines_returned`): the
/// caller can see it did not get the whole document without having to infer it
/// from the text.
#[derive(Debug, Clone)]
pub struct PdfExtract {
    /// The formatted text, with a header per page read.
    pub text: String,
    /// Pages in the document.
    pub page_count: usize,
    /// Pages this call actually read.
    pub pages_read: usize,
    /// Page numbers that yielded no text — the OCR-fallback candidates.
    pub empty_pages: Vec<u32>,
}

/// Largest PDF this tool will read, in bytes.
///
/// Deliberately the same 10 MB ceiling `ReadFileTool` already applies, rather
/// than a new number: `read_pdf` must not be a way to load a file that
/// `read_file` refuses.
pub const PDF_MAX_BYTES: usize = 10 * 1024 * 1024;

// ---------------------------------------------------------------------------
// lopdf helpers
// ---------------------------------------------------------------------------

/// Extract text from PDF bytes for the given page selection.
///
/// # Errors
///
/// Returns `ToolError::ExecutionFailed` when the bytes are not a parseable PDF.
pub fn read_pdf_text(bytes: &[u8], selection: PageSelection) -> Result<PdfExtract, ToolError> {
    use lopdf::Document;

    // Enforced here, not merely upstream: this is a public entry point, and a
    // caller that skipped its own check should get an error, not a panic.
    if bytes.len() > PDF_MAX_BYTES {
        return Err(ToolError::ExecutionFailed(format!(
            "PDF too large: {} bytes (max: {PDF_MAX_BYTES} bytes)",
            bytes.len()
        )));
    }

    let doc = Document::load_mem(bytes)
        .map_err(|e| ToolError::ExecutionFailed(format!("Failed to parse PDF: {e}")))?;

    // `get_pages` is ordered by page number, so position N in this vector is
    // the Nth page of the document — which is what a `pages` range means.
    let all_pages: Vec<u32> = doc.get_pages().keys().copied().collect();
    let page_count = all_pages.len();

    let wanted = selection.positions(page_count);
    let selected: Vec<u32> = all_pages
        .iter()
        .enumerate()
        .filter(|(index, _)| wanted.contains(&(index + 1)))
        .map(|(_, page)| *page)
        .collect();
    let pages_read = selected.len();
    debug_assert!(pages_read <= page_count, "cannot read more than exists");

    let mut text = String::new();
    let _ = write!(
        text,
        "*{page_count} pages total, reading {pages_read}*\n\n"
    );

    // A selection that matched nothing must say so. An empty body plus a
    // "0 pages read" header is otherwise indistinguishable from a document
    // whose pages were all blank.
    if pages_read == 0 && page_count > 0 {
        let _ = writeln!(
            text,
            "*[No pages matched the requested selection; the document has {page_count} pages]*"
        );
    }

    let mut empty_pages: Vec<u32> = Vec::new();
    for page_num in &selected {
        let _ = writeln!(text, "--- Page {page_num} ---");

        match doc.extract_text(&[*page_num]) {
            Ok(page_text) => {
                let cleaned = page_text.trim();
                if cleaned.is_empty() {
                    text.push_str("*[No extractable text — may contain images only]*\n");
                    empty_pages.push(*page_num);
                } else {
                    text.push_str(cleaned);
                    text.push('\n');
                }
            }
            Err(e) => {
                let _ = writeln!(text, "*[Failed to extract: {e}]*");
                empty_pages.push(*page_num);
            }
        }
        text.push('\n');
    }

    // Announce the cut in counts rather than as "N more pages": the latter
    // reads as the tail, which is wrong for a range that skipped the front.
    if pages_read < page_count {
        let _ = write!(
            text,
            "\n*... {} of {page_count} pages not shown (selection read {pages_read})*",
            page_count - pages_read
        );
    }

    Ok(PdfExtract {
        text,
        page_count,
        pages_read,
        empty_pages,
    })
}

/// What OCR did, and what it found.
///
/// Three outcomes, kept apart on purpose. "No OCR pipeline is configured" and
/// "OCR ran and the page really is blank" produce the same empty text, and
/// collapsing them is the failure the `read_pdf` OCR item was held back on for
/// two runs: a caller cannot tell a scanned document it could read from one it
/// cannot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PdfOcrOutcome {
    /// No page was missing text, so OCR was never needed.
    NotNeeded,
    /// Pages were missing text but no OCR pipeline is available.
    Unavailable,
    /// OCR ran over `images_read` embedded images.
    Ran {
        /// Recovered text, empty when the images genuinely carried none.
        text: String,
        /// How many embedded images were sent.
        images_read: usize,
    },
    /// Pages were missing text and the document carried no embedded images to
    /// OCR — a different problem from either of the above, and the one that
    /// means "this PDF is not a scan".
    NoImages,
}

/// Run OCR over a document's embedded images, if it needs it and can.
///
/// Structured rather than markdown, so a caller that returns JSON can report
/// the outcome instead of embedding prose in a text field.
///
/// # Errors
/// Returns `ToolError` when the document cannot be parsed for images.
pub async fn ocr_empty_pages(
    bytes: &[u8],
    selection: PageSelection,
    empty_page_count: usize,
    ocr_fn: Option<&OcrFn>,
) -> Result<PdfOcrOutcome, ToolError> {
    if empty_page_count == 0 {
        return Ok(PdfOcrOutcome::NotNeeded);
    }
    let Some(ocr_fn) = ocr_fn else {
        return Ok(PdfOcrOutcome::Unavailable);
    };

    let images = extract_pdf_images(bytes, selection)?;
    if images.is_empty() {
        return Ok(PdfOcrOutcome::NoImages);
    }

    let images_read = images.len();
    let mut recovered = String::new();
    for (index, (image_data, media_type)) in images.into_iter().enumerate() {
        let encoded = base64_simd::STANDARD.encode_to_string(&image_data);
        let prompt = "Extract ALL text from this image using OCR. Output the \
                      extracted text only."
            .to_string();
        match ocr_fn(encoded, prompt, media_type).await {
            Ok(text) if !text.trim().is_empty() => {
                let _ = writeln!(recovered, "### Image {} (OCR)\n{text}\n", index + 1);
            }
            Ok(_) => {}
            Err(e) => {
                // A failed image is reported, not dropped: partial OCR that
                // silently omits a page reads as a page with no text.
                let _ = writeln!(
                    recovered,
                    "### Image {} (OCR failed)\nError: {e}\n",
                    index + 1
                );
            }
        }
    }
    debug_assert!(images_read > 0, "reported OCR over no images");
    Ok(PdfOcrOutcome::Ran {
        text: recovered,
        images_read,
    })
}

/// Extract images from PDF bytes.
/// Returns `Vec<(image_bytes, media_type)>`.
fn extract_pdf_images(
    bytes: &[u8],
    _selection: PageSelection,
) -> Result<Vec<(Vec<u8>, String)>, ToolError> {
    use lopdf::{Document, Object};

    let doc = Document::load_mem(bytes)
        .map_err(|e| ToolError::ExecutionFailed(format!("Failed to parse PDF: {e}")))?;

    let mut images = Vec::new();

    // Iterate through objects looking for images
    for object in doc.objects.values() {
        if images.len() >= 20 {
            // Limit to 20 images
            break;
        }

        if let Object::Stream(stream) = object {
            let dict = &stream.dict;

            // Check if this is an image
            let is_image = dict
                .get(b"Subtype")
                .is_ok_and(|o| matches!(o, Object::Name(n) if n == b"Image"));

            if is_image {
                // Try to get the image data
                if let Ok(data) = stream.decompressed_content() {
                    // Determine image type from filter
                    let media_type = dict
                        .get(b"Filter")
                        .map_or("image/png", |f| match f {
                            Object::Name(n) if n == b"DCTDecode" => "image/jpeg",
                            Object::Name(n) if n == b"FlateDecode" => "image/png",
                            Object::Name(n) if n == b"JPXDecode" => "image/jp2",
                            _ => "image/png",
                        });

                    images.push((data, media_type.to_string()));
                }
            }
        }
    }

    Ok(images)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bare_number_selects_exactly_that_page() {
        assert_eq!(
            parse_page_selection("3"),
            Ok(PageSelection::Range { first: 3, last: 3 })
        );
        assert_eq!(
            parse_page_selection("  7  "),
            Ok(PageSelection::Range { first: 7, last: 7 })
        );
    }

    #[test]
    fn ranges_parse_in_every_documented_form() {
        assert_eq!(
            parse_page_selection("2-5"),
            Ok(PageSelection::Range { first: 2, last: 5 })
        );
        assert_eq!(parse_page_selection("-4"), Ok(PageSelection::First(4)));
        assert_eq!(
            parse_page_selection("4-"),
            Ok(PageSelection::Range {
                first: 4,
                last: usize::MAX
            })
        );
        assert_eq!(parse_page_selection(""), Ok(PageSelection::All));
        assert_eq!(parse_page_selection("   "), Ok(PageSelection::All));
    }

    #[test]
    fn a_malformed_selection_is_refused_rather_than_guessed() {
        // Reading "5-2" as 2-5 would return pages nobody asked for.
        assert!(parse_page_selection("5-2").is_err());
        // Pages are 1-based; page 0 is a caller bug worth naming.
        assert!(parse_page_selection("0").is_err());
        assert!(parse_page_selection("0-3").is_err());
        assert!(parse_page_selection("-").is_err());
        assert!(parse_page_selection("first").is_err());
        assert!(parse_page_selection("1-2-3").is_err());
        // The message must quote the input so the failure is actionable.
        let err = parse_page_selection("banana").expect_err("not a range");
        assert!(err.contains("banana"), "unhelpful message: {err}");
    }

    #[test]
    fn positions_clamp_to_the_document_and_never_wrap() {
        assert_eq!(PageSelection::All.positions(4), 1..=4);
        assert_eq!(PageSelection::First(2).positions(4), 1..=2);
        // A cap larger than the document reads the document, not more.
        assert_eq!(PageSelection::First(99).positions(4), 1..=4);
        assert_eq!(
            PageSelection::Range { first: 2, last: 99 }.positions(4),
            2..=4
        );
        // Past the end selects nothing rather than silently the last page.
        let past = PageSelection::Range {
            first: 90,
            last: 95,
        }
        .positions(4);
        assert!(past.is_empty(), "a selection past the end must be empty");
        // An empty document yields nothing for every selection.
        assert!(PageSelection::All.positions(0).is_empty());
    }

    #[test]
    fn an_oversized_document_is_refused_instead_of_parsed() {
        // The ceiling exists so `read_pdf` cannot load what `read_file` turns
        // away. Bytes just past it are refused before lopdf ever sees them —
        // note this input is not a PDF at all, so reaching the parser would
        // produce the wrong error.
        let oversized = vec![0_u8; PDF_MAX_BYTES + 1];
        let err = read_pdf_text(&oversized, PageSelection::All)
            .expect_err("a document past the ceiling must be refused");
        let message = err.to_string();
        assert!(message.contains("too large"), "wrong reason: {message}");
        assert!(
            !message.contains("parse"),
            "the ceiling must be checked before parsing: {message}"
        );
    }
    /// Build a real three-page PDF in memory, each page carrying a distinct
    /// marker, so the selection is proven against a parsed document rather than
    /// against the parser being mocked out.
    fn three_page_pdf() -> Vec<u8> {
        use lopdf::content::{Content, Operation};
        use lopdf::{Document, Object, Stream, dictionary};

        let mut doc = Document::with_version("1.5");
        let pages_id = doc.new_object_id();
        let font_id = doc.add_object(dictionary! {
            "Type" => "Font",
            "Subtype" => "Type1",
            "BaseFont" => "Helvetica",
        });
        let resources_id = doc.add_object(dictionary! {
            "Font" => dictionary! { "F1" => font_id },
        });

        let mut kids: Vec<Object> = Vec::new();
        for page in 1..=3 {
            let content = Content {
                operations: vec![
                    Operation::new("BT", vec![]),
                    Operation::new("Tf", vec!["F1".into(), 24.into()]),
                    Operation::new("Td", vec![72.into(), 720.into()]),
                    Operation::new("Tj", vec![Object::string_literal(format!("MARKER{page}"))]),
                    Operation::new("ET", vec![]),
                ],
            };
            let content_id = doc.add_object(Stream::new(
                dictionary! {},
                content.encode().expect("content encodes"),
            ));
            let leaf_id = doc.add_object(dictionary! {
                "Type" => "Page",
                "Parent" => pages_id,
                "Contents" => content_id,
            });
            kids.push(leaf_id.into());
        }

        let count = i64::try_from(kids.len()).expect("three fits");
        doc.objects.insert(
            pages_id,
            Object::Dictionary(dictionary! {
                "Type" => "Pages",
                "Kids" => kids,
                "Count" => count,
                "Resources" => resources_id,
                "MediaBox" => vec![0.into(), 0.into(), 595.into(), 842.into()],
            }),
        );
        let catalog_id = doc.add_object(dictionary! {
            "Type" => "Catalog",
            "Pages" => pages_id,
        });
        doc.trailer.set("Root", catalog_id);

        let mut bytes = Vec::new();
        doc.save_to(&mut bytes).expect("document saves");
        bytes
    }

    #[test]
    fn a_page_range_reads_those_pages_and_only_those() {
        let pdf = three_page_pdf();

        let all = read_pdf_text(&pdf, PageSelection::All).expect("all pages read");
        assert_eq!(all.page_count, 3);
        assert_eq!(all.pages_read, 3);

        // The regression this fixes: `pages: "2"` used to be discarded and the
        // whole document came back. Page 2 must be the ONLY page present.
        let single = parse_page_selection("2").expect("valid selection");
        let one = read_pdf_text(&pdf, single).expect("page 2 read");
        assert_eq!(one.page_count, 3, "the document is still three pages");
        assert_eq!(one.pages_read, 1, "only one page was asked for");
        assert!(!one.text.contains("MARKER1"), "page 1 leaked: {}", one.text);
        assert!(!one.text.contains("MARKER3"), "page 3 leaked: {}", one.text);

        let span = parse_page_selection("2-3").expect("valid selection");
        let two = read_pdf_text(&pdf, span).expect("pages 2-3 read");
        assert_eq!(two.pages_read, 2);
        assert!(
            two.text.contains("not shown"),
            "a partial read must announce the cut: {}",
            two.text
        );
    }

    #[test]
    fn a_selection_past_the_end_says_so_instead_of_returning_a_page() {
        let pdf = three_page_pdf();
        let past = parse_page_selection("9").expect("valid selection");
        let extract = read_pdf_text(&pdf, past).expect("read succeeds");

        assert_eq!(extract.pages_read, 0, "nothing matched");
        assert_eq!(extract.page_count, 3);
        assert!(
            extract.text.contains("No pages matched"),
            "an empty selection must be distinguishable from a blank document: {}",
            extract.text
        );
        assert!(!extract.text.contains("MARKER"), "no page may leak");
    }
}
