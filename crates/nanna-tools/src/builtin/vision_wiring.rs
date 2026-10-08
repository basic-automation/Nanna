//! Vision backend wiring
//!
//! Builds the vision callback on the Anthropic Claude vision API.

use super::vision::VisionFn;
use nanna_llm::{AnthropicMessage, AnthropicRequest, ContentBlock, LlmClient};
use std::sync::Arc;

/// The vision call on its own, without the tool wrapper.
///
/// Extracted so a caller that is not a [`Tool`](crate::Tool) — the daemon's
/// `vision.analyze` service, which the bundled `analyze_image` / `describe_image`
/// / `ocr` skills call — reaches the same request construction instead of a
/// second copy of it. That matters here specifically: the request has to run
/// `sampling_temperature_for_model`, because `temperature` is rejected outright
/// by current Claude models and a hand-rolled duplicate would not know that.
#[must_use]
pub fn create_vision_fn(llm: Arc<LlmClient>, model: String) -> VisionFn {
    let vision_fn: VisionFn = Arc::new(move |image: String, prompt: String, media_type: String| {
        let llm = llm.clone();
        let model = model.clone();

        Box::pin(async move {
            // Determine if image is URL or base64
            let content = if image.starts_with("http://") || image.starts_with("https://") {
                // URL-based image
                vec![
                    ContentBlock::Image {
                        source: nanna_llm::ImageSource::Url { url: image },
                    },
                    ContentBlock::Text { text: prompt },
                ]
            } else {
                // Base64-encoded image
                vec![
                    ContentBlock::Image {
                        source: nanna_llm::ImageSource::Base64 {
                            media_type,
                            data: image,
                        },
                    },
                    ContentBlock::Text { text: prompt },
                ]
            };

            let request = AnthropicRequest {
                messages: vec![AnthropicMessage::user(content)],
                max_tokens: 4096,
                temperature: nanna_llm::sampling_temperature_for_model(&model, 0.3),
                model,
                system: Some("You are a helpful vision assistant. Analyze the image and respond to the user's prompt accurately and concisely.".to_string()),
                tools: None,
                stream: None,
                thinking: None,
                cache_control: None,
                // Bit-rot caught the moment the `vision` feature was first
                // enabled (2026-09-15): `AnthropicRequest` grew this field and
                // this file did not, because nothing in the workspace compiled
                // it. `None` is the same value every other call site uses — it
                // is an Ollama `num_ctx` override, not a vision setting.
                context_limit: None,
            };

            let response = llm
                .complete_anthropic(&request)
                .await
                .map_err(|e| e.to_string())?;

            // Extract text from response
            let text: String = response
                .content
                .iter()
                .filter_map(|block| {
                    if let ContentBlock::Text { text } = block {
                        Some(text.as_str())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join("");

            Ok(text)
        })
    });

    vision_fn
}
