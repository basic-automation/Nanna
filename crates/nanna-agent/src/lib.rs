#![warn(clippy::all)]
#![warn(clippy::pedantic, clippy::nursery)]

//! Agent system for Nanna
//!
//! Implements the agentic loop with tool calling, memory, and context management.

pub mod cancel;
mod chunker;
pub mod compressor;
mod context;
pub mod cost;
pub mod harness;
pub mod image_util;
mod loop_runner;
pub mod model_stats;
mod numeric;
pub mod planner;
pub mod prompts;
pub mod tool_stats;
pub mod spans;
mod summarizer;

#[cfg(feature = "mcp")]
pub mod mcp;

// Re-export workspace crate for convenience
pub use nanna_workspace;
pub use nanna_workspace::{Workspace, WorkspaceFiles, WorkspaceTemplate};

pub use cancel::CancelToken;
pub use context::{
    AgentContext, ContextIsolation, ContextSummary, ContextSummarizationConfig, plausible_summary,
};
pub use loop_runner::{
    Agent, AgentConfig, AgentResponse, CLAIM_NUDGE_REPEAT_AFTER_ITERATIONS, CLAIM_NUDGES_MAX,
    DegradationLedger, EmotionalContext, ExtractedMemory, MemoryCallback,
    MemoryProvenance, ModelTier, NudgeLevel, ReasoningBlock, ReasoningContent, RepeatLedger,
    RunAnalysis, RunOptions, STEERING_CONTINUATION, SharedRepeatLedger, StepKind, StreamCallback,
    TASK_ANCHOR_MAX_BYTES, TOOL_RESULT_CATEGORY, TaskComplexity,
    ThinkingCallback, ThinkingMode, ToolActivation, ToolCallRecord,
    budget_warning_message, claim_nudge_message, is_work_evidence_tool, min_viable_num_ctx,
    narration_nudge_message, repetition_nudge_message, thinking_spiral_nudge_message,
    tool_loop_nudge_message, wrapup_nudge_due, wrapup_nudge_message,
};
pub use summarizer::{
    Summarizer,
    SummarizerClients,
    SummarizerConfig,
};
pub use planner::{Plan, PlanOrigin, PlannedTask, build_plan_prompt, parse_plan, plan_or_fallback};
pub use cost::{ModelPricing, default_pricing, estimate_cost_usd};
pub use model_stats::{
    ModelCost, ModelStatsTracker, ModelStats, ModelStatsSummary, RequestModelStats,
    RequestObservation, StorableModelStats,
};
pub use tool_stats::{
    ToolStatsTracker, ToolStats, ToolStatsSummary, ToolObservation,
    GlobalToolStats, SessionStats as ToolSessionStats, SessionTotals,
};

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AgentError {
    #[error("LLM error: {0}")]
    Llm(#[from] nanna_llm::LlmError),
    #[error("Tool error: {0}")]
    Tool(#[from] nanna_tools::ToolError),
    #[error("Max iterations exceeded")]
    MaxIterations,
    #[error("Context too long")]
    ContextTooLong,
    #[error("Agent stopped")]
    Stopped,
    /// The in-flight LLM stream produced no event at all for longer than the
    /// transport could possibly stay silent. The HTTP client already declares
    /// its silence tolerance (`nanna_llm::STREAM_READ_TIMEOUT_SECS` between
    /// chunks), so on any truly dead socket that timeout fires first and takes
    /// the normal error path. Reaching a multiple of it means the transport
    /// believes the stream healthy while nothing arrives — the stream future
    /// itself is wedged (lost waker, swallowed pipeline stage). Observed
    /// 2026-08-10: a turn held its session silent for 50+ minutes with zero
    /// log output. The loop fails LOUDLY instead of hanging forever; the
    /// step-retry and failure-notice chain announce it to the session.
    #[error(
        "stream watchdog: no stream event from {model} for {silent_secs}s — {multiple}× the \
         transport's declared read timeout ({read_timeout_secs}s), which fires first on any \
         truly silent socket; the stream future itself is wedged, so the call is abandoned \
         loudly rather than hanging the turn"
    )]
    StreamWatchdog {
        /// How long the loop waited without a single stream event.
        silent_secs: u64,
        /// The transport's declared per-chunk silence tolerance.
        read_timeout_secs: u64,
        /// Watchdog multiple of that tolerance (the derivation, not a knob).
        multiple: u64,
        /// Model whose stream wedged.
        model: String,
    },
    /// The runner's effective context window was demoted below the smallest
    /// prompt this run can possibly send. Compression shrinks HISTORY; it can
    /// never shrink the system prompt, the tool definitions, the step frame,
    /// or the output reserve — so below their sum, adaptation is impossible
    /// and continuing would only produce silently-truncated prompts. The run
    /// stops LOUDLY instead; the step/eval is resumable once the window is
    /// restored (free GPU memory or restart the runner).
    ///
    /// Every term is already at its minimum when this fires: the tool term
    /// counts the PRESSURE-tier definitions (core baseline + the
    /// work-evidence set — the loop degrades to that tier and announces it
    /// before ever failing), and the output reserve is window-scaled with a
    /// derived per-step floor, not a fixed half-window claim.
    #[error(
        "effective context window ({effective_window} tokens) is below the minimum viable \
         window ({floor} tokens = system prompt {system_tokens} + tool definitions \
         {tool_tokens} + step frame {frame_tokens} + output reserve {output_reserve}); \
         {cause}",
        cause = below_floor_cause(*.effective_window, *.frame_tokens, *.output_reserve)
    )]
    ContextBelowFloor {
        /// The live window the runner will actually honour.
        effective_window: usize,
        /// Sum of the irreducible parts below — the derived minimum.
        floor: usize,
        /// Estimated tokens of the effective system prompt (workspace slice
        /// already re-capped for the shrunken window).
        system_tokens: usize,
        /// Estimated tokens of the smallest tool set the step could ship —
        /// the pressure tier (core baseline + work-evidence tools), which
        /// the loop already degraded to before failing.
        tool_tokens: usize,
        /// Estimated tokens of the step frame (the first user message, which
        /// truncation always preserves).
        frame_tokens: usize,
        /// Output tokens the request must reserve.
        output_reserve: usize,
    },
}

/// Why [`AgentError::ContextBelowFloor`] fired, and what the reader can do.
///
/// The step frame carries the request itself, which no compression may
/// shorten. When it alone, with room for a reply, exceeds the window, the
/// request is too long for this window — restoring GPU memory cannot help,
/// and blaming it sent a user who pasted a large log chasing a demotion that
/// never happened. Otherwise the fixed parts together outgrew a window that
/// was sized, or demoted, below them.
#[must_use]
pub const fn below_floor_cause(
    effective_window: usize,
    frame_tokens: usize,
    output_reserve: usize,
) -> &'static str {
    if frame_tokens.saturating_add(output_reserve) > effective_window {
        "the request itself is longer than this window can hold with room for a reply, and \
         no compression shortens a request — shorten it, split it across messages, or use a \
         model with a larger context window"
    } else {
        "the runner's num_ctx was demoted under GPU memory pressure and no amount of \
         history compression can fit this step — failing loudly instead of running \
         silently truncated; free GPU memory (or restart the runner) and resume"
    }
}

/// Message content types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentContent {
    Text { text: String },
    ToolUse { id: String, name: String, input: serde_json::Value },
    ToolResult { tool_use_id: String, content: String, is_error: bool },
}

impl AgentContent {
    pub fn text(content: impl Into<String>) -> Self {
        Self::Text { text: content.into() }
    }
}

#[cfg(test)]
mod below_floor_cause_tests {
    use super::below_floor_cause;

    #[test]
    fn a_request_longer_than_the_window_is_not_blamed_on_demotion() {
        let too_long = below_floor_cause(16_384, 77_526, 4_096);
        assert!(too_long.contains("request itself is longer"), "{too_long}");
        assert!(!too_long.contains("GPU"), "{too_long}");

        // A small request in a window the fixed parts outgrew: demotion.
        let demoted = below_floor_cause(4_096, 300, 1_024);
        assert!(demoted.contains("demoted"), "{demoted}");

        // The boundary: request plus reply room exactly filling the window
        // still fits them; one token more does not.
        assert!(below_floor_cause(8_192, 4_096, 4_096).contains("demoted"));
        assert!(below_floor_cause(8_192, 4_097, 4_096).contains("request itself"));
    }
}
