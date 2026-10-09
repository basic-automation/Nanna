//! Audio backend wiring
//!
//! Builds the TTS and transcription callbacks on `OpenAI`'s APIs.

use super::audio::{OpenAiTts, OpenAiWhisper, TranscribeFn, TtsFn};
use std::sync::Arc;

/// The speech call on its own, without the tool wrapper.
///
/// Extracted for the same reason as `create_vision_fn`: the daemon's
/// `audio.tts` service is not a [`Tool`](crate::Tool), and a second copy of the
/// client setup would be a second place for the default voice and the endpoint
/// to drift.
#[must_use]
pub fn create_tts_fn(api_key: impl Into<String>, default_voice: Option<&str>) -> TtsFn {
    let tts_client = Arc::new(
        OpenAiTts::new(api_key)
            .with_voice(default_voice.unwrap_or("nova"))
    );

    Arc::new(move |text: String, voice: Option<String>| {
        let client = tts_client.clone();
        Box::pin(async move {
            client.speak(&text, voice.as_deref()).await
        })
    })
}

/// The transcription call on its own, without the tool wrapper. See
/// [`create_tts_fn`] for why.
#[must_use]
pub fn create_transcribe_tool_fn(api_key: impl Into<String>) -> TranscribeFn {
    let whisper_client = Arc::new(OpenAiWhisper::new(api_key));

    Arc::new(move |audio: Vec<u8>, language: Option<String>| {
        let client = whisper_client.clone();
        Box::pin(async move {
            client.transcribe(&audio, language.as_deref()).await
        })
    })
}
