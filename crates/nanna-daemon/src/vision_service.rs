//! The `vision.analyze` service.
//!
//! Three bundled skills — `analyze_image`, `describe_image` and `ocr` — all
//! declare `requires: ["vision.analyze"]`, and nothing registered it, so all
//! three were withheld at every boot (found by
//! `tests/skill_services_are_registered.rs`). The Rust half has been complete
//! the whole time: `vision_wiring::create_vision_fn` builds the request and
//! `AnalyzeImageTool` wrapped it (deleted 2026-10-08), but the tool was
//! reachable from nowhere and the skills call a *service*, not a tool.
//!
//! **Registered only when a vision model is actually reachable.** The model
//! comes from `[memory] ocr_model_priority`, whose doc already says "only
//! vision-capable models should be listed here" and "models are tried in
//! order" — so this is an existing, documented setting, not a new one, and its
//! ordering semantics are honoured rather than invented. When the list is empty
//! or the boot-time router can serve none of it, the service is **not**
//! registered and the three skills stay withheld — with the boot warning naming
//! `vision.analyze`, which turns "three permanently dead skills" into "three
//! skills one config line away". Registering it unconditionally would advertise
//! tools that can only fail, which is what the withholding exists to prevent.

use std::collections::HashMap;
use std::sync::Arc;

use nanna_scripting::ServiceFn;
use nanna_tools::{VisionFn, create_vision_fn, read_image_as_base64};
use serde_json::{Value, json};
use tracing::{info, warn};

use crate::llm_router::LlmRouter;

/// Default prompt when a caller sends none. Matches `analyze_image`'s own
/// fallback so the two cannot drift.
const DEFAULT_VISION_PROMPT: &str = "Describe this image in detail";

/// One usable vision model: its name and the call bound to it.
struct VisionModel {
    name: String,
    call: VisionFn,
}

/// Bind each configured model the router can actually serve, in the configured
/// order.
///
/// The router is frozen at boot, so "can serve" is knowable here and does not
/// change later — which is exactly why this is decided at registration time
/// rather than per call.
fn bind_vision_models(router: &Arc<LlmRouter>, configured: &[String]) -> Vec<VisionModel> {
    let mut bound = Vec::new();
    for name in configured {
        let name = name.trim();
        if name.is_empty() {
            continue;
        }
        if let Some(client) = router.client_for_model(name) {
            bound.push(VisionModel {
                name: name.to_string(),
                call: create_vision_fn(client, LlmRouter::strip_model_prefix(name)),
            });
        } else {
            warn!(
                model = %name,
                "ocr_model_priority names a model no configured provider can serve; skipping it"
            );
        }
    }
    debug_assert!(
        bound.len() <= configured.len(),
        "bound more vision models than were configured",
    );
    bound
}

/// The first reachable vision model as a PDF OCR callback, if there is one.
///
/// `pdf.read` and `vision.analyze` want the same capability under two
/// signatures, so this binds it once from the same configured list rather than
/// giving the PDF path its own model setting to drift.
///
/// Only the first model, not the whole priority list: `ocr_empty_pages` sends
/// one request per embedded image, and falling through a list per image turns a
/// scanned document into a multiplied bill.
#[must_use]
pub fn bind_pdf_ocr_fn(
    router: &Arc<LlmRouter>,
    configured: &[String],
) -> Option<nanna_tools::PdfOcrFn> {
    let model = bind_vision_models(router, configured).into_iter().next()?;
    let call = model.call;
    Some(Arc::new(move |image, prompt, media_type| {
        let call = call.clone();
        Box::pin(async move { call(image, prompt, media_type).await })
    }))
}

/// Build `vision.analyze`, or an empty map when no configured vision model is
/// reachable.
pub fn build_vision_services(
    router: &Arc<LlmRouter>,
    configured: &[String],
) -> HashMap<String, ServiceFn> {
    let models = bind_vision_models(router, configured);
    if models.is_empty() {
        info!(
            configured_count = configured.len(),
            "No reachable vision model; vision.analyze stays unregistered and the \
             analyze_image / describe_image / ocr skills stay withheld. Set \
             [memory] ocr_model_priority to a vision-capable model a configured \
             provider can serve."
        );
        return HashMap::new();
    }

    info!(
        models = %models.iter().map(|m| m.name.as_str()).collect::<Vec<_>>().join(", "),
        "Registering vision.analyze"
    );

    let models = Arc::new(models);
    let mut services: HashMap<String, ServiceFn> = HashMap::new();
    services.insert(
        "vision.analyze".to_string(),
        Arc::new(move |params: Value| {
            let models = models.clone();
            Box::pin(async move {
                let path = params
                    .get("path")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .unwrap_or_default();
                if path.is_empty() {
                    return Err("vision.analyze requires a `path`".to_string());
                }
                let prompt = params
                    .get("prompt")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|p| !p.is_empty())
                    .unwrap_or(DEFAULT_VISION_PROMPT)
                    .to_string();

                // Read once, try each model against the same bytes: re-reading
                // per attempt would let the file change underneath a fallthrough.
                let (encoded, media_type) =
                    read_image_as_base64(std::path::Path::new(path)).await?;

                let mut failures = Vec::new();
                for model in models.iter() {
                    match (model.call)(
                        encoded.clone(),
                        prompt.clone(),
                        media_type.to_string(),
                    )
                    .await
                    {
                        Ok(text) => {
                            return Ok(json!({ "text": text, "model": model.name }));
                        }
                        Err(e) => {
                            warn!(model = %model.name, error = %e, "Vision model failed; trying the next");
                            failures.push(format!("{}: {e}", model.name));
                        }
                    }
                }

                // Every configured model failed. Name them all: the priority
                // list exists so a failure is a list of failures, and reporting
                // only the last one hides why the others were skipped.
                Err(format!(
                    "every configured vision model failed — {}",
                    failures.join("; ")
                ))
            })
        }),
    );
    services
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_router::LlmRouter;

    fn router_without_providers() -> Arc<LlmRouter> {
        Arc::new(LlmRouter::new())
    }

    #[test]
    fn no_configured_models_means_no_service() {
        let services = build_vision_services(&router_without_providers(), &[]);
        assert!(
            services.is_empty(),
            "vision.analyze was registered with nothing behind it, so the three \
             skills would load and fail at call time",
        );
    }

    #[test]
    fn a_model_no_provider_can_serve_does_not_register_the_service() {
        let services = build_vision_services(
            &router_without_providers(),
            &["anthropic/claude-opus-5".to_string()],
        );
        assert!(
            services.is_empty(),
            "a model name alone was taken as a reachable vision model",
        );
    }

    #[test]
    fn blank_entries_do_not_count_as_configured_models() {
        let services = build_vision_services(
            &router_without_providers(),
            &[String::new(), "   ".to_string()],
        );
        assert!(services.is_empty());
    }

    #[test]
    fn a_reachable_model_registers_the_service() {
        let router = Arc::new(LlmRouter::new().with_anthropic("test-key"));
        let services = build_vision_services(&router, &["anthropic/claude-opus-5".to_string()]);
        assert_eq!(
            services.len(),
            1,
            "a servable vision model did not produce vision.analyze",
        );
        assert!(services.contains_key("vision.analyze"));
    }

    #[test]
    fn unreachable_models_are_dropped_but_a_reachable_one_still_registers() {
        let router = Arc::new(LlmRouter::new().with_anthropic("test-key"));
        let bound = bind_vision_models(
            &router,
            &[
                "ollama/llava".to_string(),
                "anthropic/claude-opus-5".to_string(),
            ],
        );
        assert_eq!(
            bound.len(),
            1,
            "a model with no configured provider was bound anyway",
        );
        assert_eq!(bound[0].name, "anthropic/claude-opus-5");
    }

    #[tokio::test]
    async fn a_call_without_a_path_is_refused_by_name() {
        let router = Arc::new(LlmRouter::new().with_anthropic("test-key"));
        let services = build_vision_services(&router, &["anthropic/claude-opus-5".to_string()]);
        let analyze = services.get("vision.analyze").expect("registered");

        let err = analyze(json!({})).await.unwrap_err();
        assert!(err.contains("path"), "the refusal must name it: {err}");
    }

    #[tokio::test]
    async fn a_non_image_path_is_refused_before_any_model_is_called() {
        let router = Arc::new(LlmRouter::new().with_anthropic("test-key"));
        let services = build_vision_services(&router, &["anthropic/claude-opus-5".to_string()]);
        let analyze = services.get("vision.analyze").expect("registered");

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("notes.txt");
        std::fs::write(&path, "text").unwrap();

        let err = analyze(json!({ "path": path.to_string_lossy() }))
            .await
            .unwrap_err();
        assert!(
            err.contains(".png"),
            "the refusal must say what vision accepts: {err}"
        );
    }
}
