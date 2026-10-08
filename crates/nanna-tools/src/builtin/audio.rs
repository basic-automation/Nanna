//! Audio clients (`OpenAI` TTS and Whisper) and the callback types the
//! daemon's `audio.*` services hold.

use serde::Deserialize;
use std::sync::Arc;

/// Callback for text-to-speech
pub type TtsFn = Arc<
    dyn Fn(String, Option<String>) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<u8>, String>> + Send>>
        + Send
        + Sync,
>;

/// Callback for speech-to-text transcription
pub type TranscribeFn = Arc<
    dyn Fn(Vec<u8>, Option<String>) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String, String>> + Send>>
        + Send
        + Sync,
>;

/// `OpenAI` TTS client helper
pub struct OpenAiTts {
    api_key: String,
    model: String,
    voice: String,
}

impl OpenAiTts {
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: "tts-1".to_string(),
            voice: "alloy".to_string(),
        }
    }

    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    #[must_use]
    pub fn with_voice(mut self, voice: impl Into<String>) -> Self {
        self.voice = voice.into();
        self
    }

    /// Generate speech from text.
    ///
    /// # Errors
    ///
    /// Returns an error if the API request fails.
    pub async fn speak(&self, text: &str, voice_override: Option<&str>) -> Result<Vec<u8>, String> {
        let voice = voice_override.unwrap_or(&self.voice);

        let client = reqwest::Client::new();
        let response = client
            .post("https://api.openai.com/v1/audio/speech")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&serde_json::json!({
                "model": self.model,
                "input": text,
                "voice": voice,
            }))
            .send()
            .await
            .map_err(|e| format!("TTS request failed: {e}"))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(format!("TTS API error {status}: {body}"));
        }

        response
            .bytes()
            .await
            .map(|b| b.to_vec())
            .map_err(|e| format!("Failed to read audio: {e}"))
    }
}

/// `OpenAI` Whisper transcription client helper
pub struct OpenAiWhisper {
    api_key: String,
    model: String,
}

#[derive(Deserialize)]
struct WhisperResponse {
    text: String,
}

impl OpenAiWhisper {
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: "whisper-1".to_string(),
        }
    }

    /// Transcribe audio to text.
    ///
    /// # Errors
    ///
    /// Returns an error if the API request fails.
    pub async fn transcribe(&self, audio: &[u8], language: Option<&str>) -> Result<String, String> {
        let client = reqwest::Client::new();

        // Build multipart form
        let file_part = reqwest::multipart::Part::bytes(audio.to_vec())
            .file_name("audio.mp3")
            .mime_str("audio/mpeg")
            .map_err(|e| format!("Failed to create form part: {e}"))?;

        let mut form = reqwest::multipart::Form::new()
            .text("model", self.model.clone())
            .part("file", file_part);

        if let Some(lang) = language {
            form = form.text("language", lang.to_string());
        }

        let response = client
            .post("https://api.openai.com/v1/audio/transcriptions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .multipart(form)
            .send()
            .await
            .map_err(|e| format!("Transcription request failed: {e}"))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(format!("Whisper API error {status}: {body}"));
        }

        let result: WhisperResponse = response
            .json()
            .await
            .map_err(|e| format!("Failed to parse response: {e}"))?;

        Ok(result.text)
    }
}
