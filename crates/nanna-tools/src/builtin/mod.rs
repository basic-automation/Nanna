//! Built-in tools

mod ask_parent;
mod audio;
mod curiosity;
mod echo;
mod exec;
mod file;
mod memory;
mod memory_storage;
mod ocr;
mod pdf;
mod schedule;
mod vision;
mod web;

#[cfg(feature = "browser")]
mod browser_wiring;

#[cfg(feature = "vision")]
mod vision_wiring;

mod audio_wiring;

pub use audio::{OpenAiTts, OpenAiWhisper, TranscribeFn, TtsFn};
pub use curiosity::{ExploreTool, WonderTool, StatusTool};
pub use echo::EchoTool;
pub use exec::ExecTool;
pub use file::{ReadFileTool, WriteFileTool, ListDirTool};
pub use memory::{InMemoryStorage, MemoryStorage, MemoryResult, StorageHandle, RememberTool, RecallTool, ReflectTool, MemoryServiceStorage, MemoryServiceAdapter};
pub use memory_storage::{EmbedFn, TursoMemoryStorage};
pub use ocr::{OcrTool, OcrVisionFn};
pub use pdf::{OcrFn as PdfOcrFn, PDF_MAX_BYTES, PageSelection, PdfExtract, PdfOcrOutcome, ocr_empty_pages, parse_page_selection, read_pdf_text};
pub use schedule::{ReminderStore, SchedulerState, RemindTool, ListRemindersTool, CancelReminderTool};
pub use ask_parent::AskParentTool;
pub use vision::{IMAGE_BYTES_MAX, VisionFn, image_media_type, read_image_as_base64};
pub use web::{WebSearchTool, WebFetchTool};

#[cfg(feature = "browser")]
pub use browser_wiring::BrowserManager;
#[cfg(feature = "browser")]
pub use nanna_browser::{BrowserConfig, BrowserType};

#[cfg(feature = "vision")]
pub use vision_wiring::create_vision_fn;

pub use audio_wiring::{create_transcribe_tool_fn, create_tts_fn};
