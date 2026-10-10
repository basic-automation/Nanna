//! Dreaming Integration
//!
//! Wires together the memory dreaming service with the LLM client and scheduler.

use nanna_llm::{CompletionRequest, EmbeddingClient, LlmClient, Message, RequestBuilder, Role};
use nanna_memory::{
    DreamingConfig, DreamingService, DreamingStats, EmbedFn, MemoryFeedback, MemoryService,
};
use std::sync::Arc;
use tracing::{info, warn};

/// Summarize text using an LLM client.
///
/// This is a helper function for memory consolidation.
async fn llm_summarize(llm: &LlmClient, model: &str, prompt: &str) -> Result<String, String> {
    let request = CompletionRequest::default()
        .with_model(model)
        .with_message(Message {
            role: Role::System,
            content: "You are a memory consolidation system. Summarize the given memories \
                      according to the instructions. Output only the consolidated memory, \
                      no explanations."
                .to_string(),
        })
        .with_message(Message {
            role: Role::User,
            content: prompt.to_string(),
        });

    llm.complete(&request).await.map_err(|e| e.to_string())
}

/// Configuration for the integrated dreaming runtime
#[derive(Debug, Clone)]
pub struct DreamingRuntimeConfig {
    /// Dreaming service configuration
    pub dreaming: DreamingConfig,
    /// Model to use for summarization
    pub summarization_model: String,
    /// Whether to auto-start the consolidation task
    pub auto_start: bool,
}

impl Default for DreamingRuntimeConfig {
    fn default() -> Self {
        Self {
            dreaming: DreamingConfig::default(),
            summarization_model: "claude-sonnet-5".to_string(),
            auto_start: true,
        }
    }
}

/// Integrated dreaming runtime that combines `DreamingService` with LLM
pub struct DreamingRuntime {
    service: DreamingService,
    llm: Arc<LlmClient>,
    model: String,
}

impl DreamingRuntime {
    /// Create a new dreaming runtime.
    ///
    /// # Arguments
    /// * `config` - Runtime configuration
    /// * `llm` - LLM client for summarization
    /// * `embed` - Embedding client for generating vector embeddings
    #[must_use]
    pub fn new(
        config: DreamingRuntimeConfig,
        llm: Arc<LlmClient>,
        embed: Arc<EmbeddingClient>,
    ) -> Self {
        // Create embedding function using the embedding client
        let embed_fn: EmbedFn = Arc::new(move |text: &str| {
            let embed = embed.clone();
            let text = text.to_string();
            Box::pin(async move { (*embed).embed_one(&text).await.map_err(|e| e.to_string()) })
        });

        let service = DreamingService::new(config.dreaming).with_embed_fn(embed_fn);

        Self {
            service,
            llm,
            model: config.summarization_model,
        }
    }

    /// Create a new dreaming runtime without embeddings.
    ///
    /// This is useful when you don't have an embedding client configured.
    /// Memory recall and smart ingest will not work without embeddings.
    #[must_use]
    pub fn new_without_embeddings(config: DreamingRuntimeConfig, llm: Arc<LlmClient>) -> Self {
        let service = DreamingService::new(config.dreaming);

        Self {
            service,
            llm,
            model: config.summarization_model,
        }
    }

    /// Create a dreaming runtime over an **existing, shared** memory store.
    ///
    /// This is the daemon-facing constructor (P13 unification): pass the live
    /// `Arc<MemoryService>` the agent's tools already read/write so consolidation
    /// dreams over the same store instead of a private one. The shared store must
    /// already carry its `embed_fn`/persistence (configure it before sharing).
    pub fn with_shared_memory(
        config: DreamingRuntimeConfig,
        llm: Arc<LlmClient>,
        memory: Arc<MemoryService>,
    ) -> Self {
        let service = DreamingService::with_shared_memory(config.dreaming, memory);

        Self {
            service,
            llm,
            model: config.summarization_model,
        }
    }

    /// Clone the shared memory-store handle this runtime dreams over.
    #[must_use]
    pub fn memory_arc(&self) -> Arc<MemoryService> {
        self.service.memory_arc()
    }

    /// Get reference to the underlying dreaming service
    #[must_use]
    pub const fn service(&self) -> &DreamingService {
        &self.service
    }

    /// Remember something
    ///
    /// # Errors
    ///
    /// Returns the errors of [`MemoryService::remember`]: a failed store write.
    /// A missing or failing embedding provider is not an error.
    pub async fn remember(
        &self,
        content: &str,
        metadata: std::collections::HashMap<String, String>,
    ) -> Result<String, nanna_memory::MemoryError> {
        self.service.remember(content, metadata).await
    }

    /// Recall memories similar to a query
    ///
    /// # Errors
    ///
    /// Returns the errors of [`MemoryService::recall`], e.g.
    /// [`MemoryError::NoEmbeddingProvider`](nanna_memory::MemoryError::NoEmbeddingProvider)
    /// when no embedding provider is configured.
    pub async fn recall(
        &self,
        query: &str,
    ) -> Result<Vec<nanna_memory::RecallResult>, nanna_memory::MemoryError> {
        self.service.recall(query).await
    }

    /// Record feedback for a memory (applied during dreaming)
    pub async fn record_feedback(&self, memory_id: &str, feedback: MemoryFeedback) {
        self.service.record_feedback(memory_id, feedback).await;
    }

    /// Apply feedback immediately without waiting for dreaming
    ///
    /// # Errors
    ///
    /// Returns the errors of [`MemoryService::promote`] or
    /// [`MemoryService::demote`] (chosen by the feedback's sign) when the FSRS
    /// update for `memory_id` fails, e.g. because no such memory exists.
    pub async fn apply_feedback(
        &self,
        memory_id: &str,
        feedback: MemoryFeedback,
    ) -> Result<(), nanna_memory::MemoryError> {
        self.service.apply_feedback(memory_id, feedback).await
    }

    /// Run the dreaming process (memory consolidation).
    ///
    /// This should be called periodically by the scheduler.
    ///
    /// # Errors
    ///
    /// Returns
    /// [`MemoryError::InvalidClusteringConfig`](nanna_memory::MemoryError::InvalidClusteringConfig)
    /// when the consolidation config would let unrelated memories merge. Failed
    /// feedback updates and summarizer calls are logged, not returned.
    pub async fn dream(&self) -> Result<DreamingStats, nanna_memory::MemoryError> {
        let llm = self.llm.clone();
        let model = self.model.clone();

        self.service
            .dream(|prompt| {
                let llm = llm.clone();
                let model = model.clone();
                async move { llm_summarize(&llm, &model, &prompt).await }
            })
            .await
    }

    /// Get memory statistics
    pub async fn stats(&self) -> nanna_memory::MemoryStats {
        self.service.stats().await
    }

    /// Save memories to file
    ///
    /// # Errors
    ///
    /// Returns the errors of [`MemoryService::save`]: the file cannot be written
    /// or the memories cannot be serialized.
    pub async fn save(&self, path: &std::path::Path) -> Result<(), nanna_memory::MemoryError> {
        self.service.save(path).await
    }

    /// Load memories from file
    ///
    /// # Errors
    ///
    /// Returns the errors of [`MemoryService::load`]: the file cannot be read or
    /// parsed.
    pub async fn load(&self, path: &std::path::Path) -> Result<(), nanna_memory::MemoryError> {
        self.service.load(path).await
    }
}

/// Create a consolidation task executor for the scheduler.
///
/// Returns a function that can be passed to the scheduler's task executor.
pub fn create_dreaming_executor(
    runtime: Arc<DreamingRuntime>,
) -> impl Fn(
    crate::ScheduledTask,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = crate::TaskResult> + Send>>
+ Send
+ Sync
+ 'static {
    move |task: crate::ScheduledTask| {
        let runtime = runtime.clone();
        Box::pin(async move {
            let start = std::time::Instant::now();

            // Check if this is a dreaming task
            if !crate::is_dreaming_task(&task) {
                let now = chrono::Utc::now();
                return crate::TaskResult {
                    task_id: task.id.clone(),
                    task_name: task.name.clone(),
                    success: false,
                    output: None,
                    error: Some("Not a dreaming task".to_string()),
                    duration_ms: elapsed_ms(start),
                    started_at: now,
                    finished_at: now,
                    skipped: false,
                };
            }

            info!("Starting memory consolidation (dreaming)...");

            let started_at = chrono::Utc::now();
            match runtime.dream().await {
                Ok(stats) => {
                    let finished_at = chrono::Utc::now();
                    let output = format!(
                        "Dreaming complete: {} processed, {} merged, {} expanded",
                        stats.consolidation.memories_processed,
                        stats.consolidation.memories_merged,
                        stats.consolidation.memories_expanded,
                    );
                    info!("{}", output);

                    crate::TaskResult {
                        task_id: task.id.clone(),
                        task_name: task.name.clone(),
                        success: true,
                        output: Some(output),
                        error: None,
                        duration_ms: elapsed_ms(start),
                        started_at,
                        finished_at,
                        skipped: false,
                    }
                }
                Err(e) => {
                    let finished_at = chrono::Utc::now();
                    warn!("Dreaming failed: {}", e);
                    crate::TaskResult {
                        task_id: task.id.clone(),
                        task_name: task.name.clone(),
                        success: false,
                        output: None,
                        error: Some(e.to_string()),
                        duration_ms: elapsed_ms(start),
                        started_at,
                        finished_at,
                        skipped: false,
                    }
                }
            }
        })
    }
}

/// Milliseconds since `start`, saturating: a task would need to run for ~584
/// million years to exceed `u64::MAX` milliseconds.
fn elapsed_ms(start: std::time::Instant) -> u64 {
    u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    

    #[tokio::test]
    async fn test_dreaming_runtime_creation() {
        // Would need a real LLM client to test properly
        // This just ensures the types compile correctly
    }
}
