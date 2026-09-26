//! Session management
//!
//! Sessions represent conversations with the agent. Multiple channels
//! can subscribe to the same session.
//!
//! All session and message data is persisted to Turso via nanna-storage.
//! The in-memory `HashMap` serves as a hot cache for fast access.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

/// Unique session identifier
pub type SessionId = String;

/// Channel identifier (e.g., "gui:abc123", "telegram:456")
pub type ChannelId = String;

/// A message in a session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMessage {
    pub id: String,
    pub role: MessageRole,
    pub content: String,
    pub timestamp: DateTime<Utc>,
    #[serde(default)]
    pub tool_calls: Vec<ToolCallRecord>,
    #[serde(default)]
    pub attachments: Vec<AttachmentRecord>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    /// Chronological journal of the run that produced this message (thinking
    /// segments, tool calls, text segments, healed faults — in order). The
    /// flat `tool_calls`/`reasoning` fields above are kept for older
    /// messages and for model-context reconstruction; when `timeline` is
    /// non-empty the UI renders it instead.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub timeline: Vec<TimelineItem>,
    /// Token + wall-clock totals for the run (model benchmarking).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<RunUsage>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    User,
    Assistant,
    System,
    Tool,
}

impl MessageRole {
    /// Convert to the string format used in the database.
    #[must_use]
    pub const fn as_db_str(&self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::System => "system",
            Self::Tool => "tool",
        }
    }

    /// Parse from the string format used in the database.
    #[must_use]
    pub fn from_db_str(s: &str) -> Self {
        match s {
            "assistant" => Self::Assistant,
            "system" => Self::System,
            "tool" => Self::Tool,
            // "user" itself, and any role string this build does not know.
            _ => Self::User,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRecord {
    pub id: String,
    pub name: String,
    pub input: serde_json::Value,
    pub output: Option<String>,
    pub success: Option<bool>,
    pub duration_ms: Option<u64>,
}

/// Per-side line bound on a journaled [`EditDiff`]. Matches what the
/// `edit_file` skill itself emits, so the clamp is a guard against a skill
/// that stops honouring it, not the normal path.
pub const EDIT_DIFF_LINES_MAX: usize = 40;

/// Per-side byte bound on a journaled [`EditDiff`] — the same 4000 bytes the
/// journal allows a call's output (`TIMELINE_OUTPUT_CAP`), so an edit's view
/// costs the run record at most what its result already may.
pub const EDIT_DIFF_BYTES_MAX: usize = 4000;

/// The before/after view `edit_file` attaches to a successful edit (P18
/// "Diff presentation").
///
/// It holds the changed lines, where they start, and whether the view was
/// cut. Journaled on the tool item so a user who opens the session after
/// hours of unattended work can SEE each edit — the live event carries it
/// too, but only a client that was watching ever received that.
/// Observability, never an approval step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EditDiff {
    /// 1-based line, in the file before the edit, where the change starts.
    pub start_line: u32,
    pub removed: Vec<String>,
    pub added: Vec<String>,
    /// The view was cut — by the skill or by the journal bound. Rendered,
    /// never a silent cut.
    #[serde(default)]
    pub truncated: bool,
}

impl EditDiff {
    /// The diff to journal for a finished call, if any. Kept only for a call
    /// that succeeded AND really ran: a failed edit changed nothing, and a
    /// breaker replay never ran at all — a diff on either would be a false
    /// claim about the file. Both journal writers (the chat path and the
    /// task-run sink) call this, so the rule has one home.
    #[must_use]
    pub fn for_outcome(
        success: bool,
        short_circuited: bool,
        data: Option<&serde_json::Value>,
    ) -> Option<Self> {
        if !success || short_circuited {
            return None;
        }
        Self::from_tool_data(data)
    }

    /// Read and bound the view out of a tool result's structured data.
    ///
    /// The payload comes from a JS skill, so it is validated, not trusted:
    /// anything malformed is `None` — never a panic inside a spawned turn —
    /// and each side is clamped to [`EDIT_DIFF_LINES_MAX`] lines and
    /// [`EDIT_DIFF_BYTES_MAX`] bytes, with `truncated` set when that cut.
    #[must_use]
    pub fn from_tool_data(data: Option<&serde_json::Value>) -> Option<Self> {
        let raw = data?.get("diff")?;
        let parsed = Self::deserialize(raw).ok()?;
        if parsed.start_line == 0 {
            return None;
        }
        if parsed.removed.is_empty() && parsed.added.is_empty() {
            return None;
        }
        let (removed, removed_cut) = clamp_diff_side(parsed.removed);
        let (added, added_cut) = clamp_diff_side(parsed.added);
        let diff = Self {
            start_line: parsed.start_line,
            removed,
            added,
            truncated: parsed.truncated || removed_cut || added_cut,
        };
        debug_assert!(
            diff.removed.len() <= EDIT_DIFF_LINES_MAX,
            "removed side is bounded"
        );
        debug_assert!(
            diff.added.len() <= EDIT_DIFF_LINES_MAX,
            "added side is bounded"
        );
        Some(diff)
    }
}

/// Keep whole lines while they fit both bounds; the first line that would
/// overflow the byte bound is cut at a char boundary and ends the side.
/// Returns the kept lines and whether anything was dropped.
fn clamp_diff_side(lines: Vec<String>) -> (Vec<String>, bool) {
    let mut kept: Vec<String> = Vec::with_capacity(lines.len().min(EDIT_DIFF_LINES_MAX));
    let mut bytes_used: usize = 0;
    let mut cut_any = false;
    for line in lines {
        if kept.len() == EDIT_DIFF_LINES_MAX {
            cut_any = true;
            break;
        }
        let bytes_left = EDIT_DIFF_BYTES_MAX - bytes_used;
        if line.len() <= bytes_left {
            bytes_used += line.len();
            kept.push(line);
            continue;
        }
        let prefix = char_floor_prefix(&line, bytes_left);
        bytes_used += prefix.len();
        if !prefix.is_empty() {
            kept.push(prefix.to_string());
        }
        cut_any = true;
        break;
    }
    debug_assert!(
        bytes_used <= EDIT_DIFF_BYTES_MAX,
        "a side never exceeds the byte bound"
    );
    debug_assert!(kept.len() <= EDIT_DIFF_LINES_MAX, "…nor the line bound");
    (kept, cut_any)
}

/// The longest prefix of `text` of at most `bytes_max` bytes that ends on a
/// char boundary. Never slices through a multi-byte char — this repo has lost
/// a run to `&text[..n]` landing inside an em dash.
fn char_floor_prefix(text: &str, bytes_max: usize) -> &str {
    let mut end = bytes_max.min(text.len());
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    debug_assert!(end <= bytes_max, "the prefix respects the bound");
    text.get(..end).unwrap_or_default()
}

/// One entry in a run's chronological journal.
///
/// A long-horizon run is not
/// "one thinking blob + one flat tool list + one text blob" — it is an
/// interleaved sequence (think → call tools → think → speak → …), and for
/// runs that heal through provider faults it can span many attempts. The
/// timeline records events in the order they happened so the UI can replay
/// the run faithfully, including after navigation away and back.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TimelineItem {
    /// A contiguous burst of thinking/reasoning (closed by the next tool
    /// call or text output).
    Thinking { content: String, at: String },
    /// A contiguous burst of visible assistant text.
    Text { content: String, at: String },
    /// One tool call. `output`/`success`/`duration_ms` are back-filled when
    /// the call completes; a run that dies mid-call leaves them None.
    Tool {
        call_id: String,
        name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        input: Option<serde_json::Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        output: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        success: Option<bool>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        duration_ms: Option<u64>,
        /// Tokens spent on the action: input+output of the LLM request that
        /// issued this tool call (parallel calls from one request share it).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tokens: Option<u64>,
        /// Run-total tokens spent at the moment this call was issued.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        total_tokens: Option<u64>,
        /// `Some(true)` when the zero-information breaker replaced this call
        /// with a steering notice instead of running the tool. Back-filled
        /// with `success`/`output`, so an in-flight call is `None` ("not yet
        /// known"), never `Some(false)`.
        ///
        /// Without it the journal cannot tell a replay from a failure: a
        /// short circuit reports `success: false` (the model did not get what
        /// it asked for) with a notice as its output, so a timeline rebuilt
        /// from this record after a remount rendered a wall of steering as a
        /// wall of tool errors — while the same run rendered them correctly
        /// live, off the event's structured data. Omitted from the wire when
        /// `None`, so journals written before this field deserialize as-is.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        short_circuited: Option<bool>,
        /// `edit_file`'s before/after view of what a successful edit changed,
        /// bounded at record time (see [`EditDiff`]). Back-filled with the
        /// rest of the outcome. Absent for every other tool, for a failed or
        /// replayed call, and in journals written before the field existed —
        /// omitted from the wire when `None`, so those still deserialize.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        diff: Option<EditDiff>,
        at: String,
    },
    /// A provider fault the run healed through (stream drop, timeout, …).
    /// Recorded so the journal explains why thinking/text may restart:
    /// the attempt after a fault regenerates rather than resumes.
    Fault { message: String, at: String },
    /// The harness starting work on one plan item.
    ///
    /// This is run MECHANICS, not conversation. It used to be written into
    /// the assistant's message as literal `**[working]** …` markdown, which
    /// had three costs: the transcript read as a wall of banners instead of a
    /// reply, the banners were persisted into conversation history and fed
    /// back to the model as context on later turns, and they crowded out the
    /// narration a watcher actually wants. As a timeline item the GUI can
    /// render it as a status row that updates in place.
    Step {
        /// `working` | `planning` | `verifying`. Named `phase` because the
        /// enum is internally tagged on `kind`.
        phase: String,
        /// Human-readable item title, e.g. "Implement the get command".
        label: String,
        item_id: i64,
        at: String,
    },
}

/// Everything an assistant message can carry beyond its text: the tool calls
/// it made, its reasoning, the run's chronological journal, and usage totals.
#[derive(Debug, Clone, Default)]
pub struct MessageDetails {
    pub tool_calls: Vec<ToolCallRecord>,
    pub reasoning: Option<String>,
    pub timeline: Vec<TimelineItem>,
    pub usage: Option<RunUsage>,
}

/// Resource totals for one run, for benchmarking models against each other
/// on identical tasks: total tokens spent and wall-clock time taken.
///
/// Token totals accumulate across EVERY healing attempt via the per-request
/// usage callback — not just the attempt that finally succeeded. (Streams
/// that die before the provider reports usage still under-count slightly;
/// that loss is inherent to the protocol.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub duration_ms: u64,
    /// The model that finished the run (last model used).
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttachmentRecord {
    pub id: String,
    pub filename: String,
    pub content_type: String,
    pub url: Option<String>,
}

/// Metadata key holding a session's chat-model override.
///
/// Named `chat_model` rather than `model` because it names the ONE thing the
/// override covers: the absence of `sub_agent_model` / `summarization_model` /
/// `embedding_model` keys reads as deliberate rather than forgotten. Those stay
/// global — see `[llm]` and `[embedding]` in the config.
const CHAT_MODEL_KEY: &str = "chat_model";

/// Metadata key holding a session's user-selected extra tools.
///
/// Named `chat_tools` beside `chat_model` because it is the same kind of
/// per-chat override: a selection the user made for THIS conversation. The
/// semantics are strictly additive — these tools are unioned into the active
/// set a turn starts with, and an empty/absent list means "exactly the
/// default behavior", so a session that never picked is byte-identical to
/// before the feature existed.
const CHAT_TOOLS_KEY: &str = "chat_tools";

/// Metadata key holding the external channel a session's replies go back to:
/// `{"provider": "telegram", "id": "<chat id>"}`.
///
/// Metadata rather than `Session::owner` because metadata is persisted and
/// `owner` is not — a reply route that vanished on restart would strand every
/// reminder a channel user set before it.
const REPLY_CHANNEL_KEY: &str = "reply_channel";

/// Where a channel conversation's replies are sent: provider plus that
/// provider's chat/channel id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplyChannel {
    pub provider: String,
    pub id: String,
}

/// A conversation session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: SessionId,
    pub name: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub messages: Vec<SessionMessage>,
    /// Channels subscribed to this session (receive events)
    #[serde(default)]
    pub subscribers: HashSet<ChannelId>,
    /// Channel that "owns" this session (can clear, rename, etc.)
    pub owner: Option<ChannelId>,
    /// Session metadata
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
    /// Workspace this session belongs to (None = global)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workspace_id: Option<String>,
}

impl Session {
    /// Create a new session
    #[must_use]
    pub fn new(name: Option<String>) -> Self {
        let now = Utc::now();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            name,
            created_at: now,
            updated_at: now,
            messages: Vec::new(),
            subscribers: HashSet::new(),
            owner: None,
            metadata: HashMap::new(),
            workspace_id: None,
        }
    }
    
    /// Create a session with a specific ID
    pub fn with_id(id: impl Into<String>, name: Option<String>) -> Self {
        let now = Utc::now();
        Self {
            id: id.into(),
            name,
            created_at: now,
            updated_at: now,
            messages: Vec::new(),
            subscribers: HashSet::new(),
            owner: None,
            metadata: HashMap::new(),
            workspace_id: None,
        }
    }

    /// Set the workspace ID for this session
    #[must_use]
    pub fn with_workspace(mut self, workspace_id: impl Into<String>) -> Self {
        self.workspace_id = Some(workspace_id.into());
        self
    }
    
    /// This session's chat-model override, if the user picked one.
    ///
    /// `None` means "use the global default" — the absence of the key is the
    /// unset state, so a session that never picked behaves exactly as it did
    /// before overrides existed. The spec is stored verbatim in router form
    /// (`ollama/qwen3:14b`, `openrouter/…`, bare `claude-…`), the same shape
    /// `llm.model_priority` holds, so a caller can hand it straight to
    /// `LlmRouter::client_for_model`.
    pub fn chat_model(&self) -> Option<&str> {
        self.metadata.get(CHAT_MODEL_KEY).and_then(serde_json::Value::as_str)
    }

    /// The external channel this session's replies go back to, if it is a
    /// channel conversation. A malformed entry reads as none.
    #[must_use]
    pub fn reply_channel(&self) -> Option<ReplyChannel> {
        let route = self.metadata.get(REPLY_CHANNEL_KEY)?;
        let provider = route.get("provider")?.as_str()?;
        let id = route.get("id")?.as_str()?;
        (!provider.is_empty() && !id.is_empty()).then(|| ReplyChannel {
            provider: provider.to_string(),
            id: id.to_string(),
        })
    }

    /// Tools the user manually added to this chat's context, if any.
    ///
    /// Empty means "no manual selection" — the unset state — so a turn built
    /// from it behaves exactly as it did before per-chat tool selection
    /// existed. Selections only ever ADD to the default active set; they
    /// never remove or replace defaults.
    pub fn chat_tools(&self) -> Vec<String> {
        self.metadata
            .get(CHAT_TOOLS_KEY)
            .and_then(serde_json::Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(serde_json::Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Add a message to the session (in-memory only — use `SessionManager` for persistence)
    pub fn add_message(&mut self, role: MessageRole, content: impl Into<String>) -> String {
        let id = uuid::Uuid::new_v4().to_string();
        self.messages.push(SessionMessage {
            id: id.clone(),
            role,
            content: content.into(),
            timestamp: Utc::now(),
            tool_calls: Vec::new(),
            attachments: Vec::new(),
            reasoning: None,
            timeline: Vec::new(),
            usage: None,
        });
        self.updated_at = Utc::now();
        id
    }

    /// Add a message with tool calls, reasoning, run timeline, and usage
    /// totals to the session (in-memory only)
    pub fn add_full_message(
        &mut self,
        role: MessageRole,
        content: impl Into<String>,
        details: MessageDetails,
    ) -> String {
        let MessageDetails {
            tool_calls,
            reasoning,
            timeline,
            usage,
        } = details;
        let id = uuid::Uuid::new_v4().to_string();
        self.messages.push(SessionMessage {
            id: id.clone(),
            role,
            content: content.into(),
            timestamp: Utc::now(),
            tool_calls,
            attachments: Vec::new(),
            reasoning,
            timeline,
            usage,
        });
        self.updated_at = Utc::now();
        id
    }
    
    /// Subscribe a channel to this session
    pub fn subscribe(&mut self, channel_id: ChannelId) {
        self.subscribers.insert(channel_id);
    }
    
    /// Unsubscribe a channel from this session
    pub fn unsubscribe(&mut self, channel_id: &str) {
        self.subscribers.remove(channel_id);
    }
    
    /// Check if a channel is subscribed
    #[must_use]
    pub fn is_subscribed(&self, channel_id: &str) -> bool {
        self.subscribers.contains(channel_id)
    }
    
    /// Set the session owner
    pub fn set_owner(&mut self, channel_id: Option<ChannelId>) {
        self.owner = channel_id;
    }
    
    /// Clear all messages
    pub fn clear(&mut self) {
        self.messages.clear();
        self.updated_at = Utc::now();
    }

    /// Prepare the session to regenerate the last assistant response.
    ///
    /// Drops the most recent user message **and everything after it** (the
    /// stale assistant reply plus any trailing tool turns), returning that user
    /// message's content so the caller can replay the turn through the normal
    /// send path (which re-adds the user message and runs the agent afresh).
    ///
    /// Returns `None` and leaves the session unchanged when there is no user
    /// message to regenerate from.
    pub fn take_last_user_turn(&mut self) -> Option<String> {
        let idx = self
            .messages
            .iter()
            .rposition(|m| matches!(m.role, MessageRole::User))?;
        let content = self.messages[idx].content.clone();
        // Remove the user message and any messages that followed it.
        self.messages.truncate(idx);
        self.updated_at = Utc::now();
        Some(content)
    }

    /// Get display name (name or truncated ID)
    #[must_use]
    pub fn display_name(&self) -> String {
        self.name.clone().unwrap_or_else(|| {
            format!("Session {}", &self.id[..floor_boundary(&self.id, 8)])
        })
    }
    
    /// Get message count
    #[must_use]
    pub const fn message_count(&self) -> usize {
        self.messages.len()
    }
}

/// Session summary for listing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    pub id: SessionId,
    pub name: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub message_count: usize,
    pub subscriber_count: usize,
    pub owner: Option<ChannelId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workspace_id: Option<String>,
    /// Chat-model override, mirrored out of `metadata` so a client that lists
    /// sessions can render the pin without a `session.get` per row.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_model: Option<String>,
    /// User-selected extra tools, mirrored out of `metadata` for the same
    /// reason as `chat_model`. Empty = no manual selection (default behavior).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub chat_tools: Vec<String>,
}

impl From<&Session> for SessionSummary {
    fn from(session: &Session) -> Self {
        Self {
            id: session.id.clone(),
            name: session.name.clone(),
            created_at: session.created_at,
            updated_at: session.updated_at,
            message_count: session.messages.len(),
            subscriber_count: session.subscribers.len(),
            owner: session.owner.clone(),
            workspace_id: session.workspace_id.clone(),
            chat_model: session.chat_model().map(str::to_string),
            chat_tools: session.chat_tools(),
        }
    }
}

/// State of a sub-agent session
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubSessionState {
    Spawning,
    Running,
    Waiting,
    Completed,
    Failed,
    Killed,
}

/// Metadata for a sub-agent session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubSessionInfo {
    /// Session ID
    pub session_id: SessionId,
    /// Parent session ID (None for top-level sessions)
    pub parent_id: Option<SessionId>,
    /// Human-readable label
    pub label: Option<String>,
    /// Task description / initial prompt
    pub task: String,
    /// Current state
    pub state: SubSessionState,
    /// When it was spawned
    pub spawned_at: DateTime<Utc>,
    /// When it completed/failed/was killed
    pub finished_at: Option<DateTime<Utc>>,
    /// Model used
    pub model: Option<String>,
    /// Result summary (on completion)
    pub result: Option<String>,
    /// Error message (on failure)
    pub error: Option<String>,
}

/// A message in the sub-session mailbox
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MailboxMessage {
    pub from: String,
    pub content: String,
    pub timestamp: DateTime<Utc>,
}

/// Serialize a `SessionMessage`'s extra fields (`tool_calls`, attachments, reasoning, timeline, usage) to JSON metadata.
fn message_to_metadata(msg: &SessionMessage) -> Option<String> {
    let has_tool_calls = !msg.tool_calls.is_empty();
    let has_attachments = !msg.attachments.is_empty();
    let has_reasoning = msg.reasoning.is_some();
    let has_timeline = !msg.timeline.is_empty();
    let has_usage = msg.usage.is_some();

    if !has_tool_calls && !has_attachments && !has_reasoning && !has_timeline && !has_usage {
        return None;
    }

    let mut meta = serde_json::Map::new();
    if has_tool_calls {
        meta.insert("tool_calls".to_string(), serde_json::to_value(&msg.tool_calls).unwrap_or_default());
    }
    if has_attachments {
        meta.insert("attachments".to_string(), serde_json::to_value(&msg.attachments).unwrap_or_default());
    }
    if let Some(ref reasoning) = msg.reasoning {
        meta.insert("reasoning".to_string(), serde_json::Value::String(reasoning.clone()));
    }
    if has_timeline {
        meta.insert("timeline".to_string(), serde_json::to_value(&msg.timeline).unwrap_or_default());
    }
    if let Some(ref usage) = msg.usage {
        meta.insert("usage".to_string(), serde_json::to_value(usage).unwrap_or_default());
    }
    Some(serde_json::Value::Object(meta).to_string())
}

/// Deserialize a DB message row back into a `SessionMessage`.
fn db_message_to_session_message(
    message_id: &str,
    role: &str,
    content: &str,
    created_at: &str,
    metadata: Option<&serde_json::Value>,
) -> SessionMessage {
    let tool_calls = metadata
        .and_then(|m| m.get("tool_calls"))
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();
    let attachments = metadata
        .and_then(|m| m.get("attachments"))
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();
    let reasoning = metadata
        .and_then(|m| m.get("reasoning"))
        .and_then(|v| v.as_str())
        .map(String::from);
    let timeline = metadata
        .and_then(|m| m.get("timeline"))
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();
    let usage = metadata
        .and_then(|m| m.get("usage"))
        .and_then(|v| serde_json::from_value(v.clone()).ok());

    // Parse timestamp, fall back to now
    let timestamp = chrono::DateTime::parse_from_rfc3339(created_at)
        .map(|dt| dt.with_timezone(&Utc))
        .or_else(|_| {
            // Try Turso datetime format: "2026-03-31 14:09:49"
            chrono::NaiveDateTime::parse_from_str(created_at, "%Y-%m-%d %H:%M:%S")
                .map(|ndt| ndt.and_utc())
        })
        .unwrap_or_else(|_| Utc::now());

    SessionMessage {
        id: message_id.to_string(),
        role: MessageRole::from_db_str(role),
        content: content.to_string(),
        timestamp,
        tool_calls,
        attachments,
        reasoning,
        timeline,
        usage,
    }
}

/// The `sessions` row a persist writes, lifted out from under the map's guard.
///
/// A write-through has to survive the guard being dropped, because the database
/// round trip must not happen while the map is locked. Carrying the whole
/// [`Session`] across that await would copy its entire message history — the
/// largest thing a session owns — for what is often a one-key metadata change,
/// so only the columns the upsert actually binds come along.
struct SessionRow {
    id: SessionId,
    name: Option<String>,
    workspace_id: Option<String>,
    created_at: String,
    updated_at: String,
    metadata: HashMap<String, serde_json::Value>,
}

impl From<&Session> for SessionRow {
    fn from(session: &Session) -> Self {
        Self {
            id: session.id.clone(),
            name: session.name.clone(),
            workspace_id: session.workspace_id.clone(),
            created_at: session.created_at.to_rfc3339(),
            updated_at: session.updated_at.to_rfc3339(),
            metadata: session.metadata.clone(),
        }
    }
}

/// Manages all sessions with write-through to Turso.
pub struct SessionManager {
    sessions: Arc<RwLock<HashMap<SessionId, Session>>>,
    /// Default session ID (for new clients)
    default_session: Arc<RwLock<Option<SessionId>>>,
    /// Sub-session registry (`session_id` -> info)
    sub_sessions: Arc<RwLock<HashMap<SessionId, SubSessionInfo>>>,
    /// Per-session mailbox for inter-session messaging
    mailboxes: Arc<RwLock<HashMap<SessionId, Vec<MailboxMessage>>>>,
    /// Database storage for persistence (None = in-memory only, e.g. tests)
    storage: Option<Arc<nanna_storage::Storage>>,
}

impl SessionManager {
    /// Create a new session manager (no persistence)
    #[must_use]
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            default_session: Arc::new(RwLock::new(None)),
            sub_sessions: Arc::new(RwLock::new(HashMap::new())),
            mailboxes: Arc::new(RwLock::new(HashMap::new())),
            storage: None,
        }
    }

    /// Create a new session manager backed by Turso storage
    #[must_use]
    pub fn with_storage(storage: Arc<nanna_storage::Storage>) -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            default_session: Arc::new(RwLock::new(None)),
            sub_sessions: Arc::new(RwLock::new(HashMap::new())),
            mailboxes: Arc::new(RwLock::new(HashMap::new())),
            storage: Some(storage),
        }
    }

    /// Load all daemon sessions and their messages from Turso into the in-memory cache.
    /// Call this once at startup.
    pub async fn load_from_db(&self) -> usize {
        let Some(ref storage) = self.storage else {
            return 0;
        };

        let db_sessions = match storage.list_daemon_sessions().await {
            Ok(s) => s,
            Err(e) => {
                warn!("Failed to load sessions from database: {}", e);
                return 0;
            }
        };

        let count = db_sessions.len();
        let mut sessions = self.sessions.write().await;
        let mut default = self.default_session.write().await;

        for db_session in db_sessions {
            let session_id = db_session.session_id.clone();

            // Parse timestamps
            let created_at = chrono::DateTime::parse_from_rfc3339(&db_session.created_at)
                .map(|dt| dt.with_timezone(&Utc))
                .or_else(|_| {
                    chrono::NaiveDateTime::parse_from_str(&db_session.created_at, "%Y-%m-%d %H:%M:%S")
                        .map(|ndt| ndt.and_utc())
                })
                .unwrap_or_else(|_| Utc::now());
            let updated_at = chrono::DateTime::parse_from_rfc3339(&db_session.updated_at)
                .map(|dt| dt.with_timezone(&Utc))
                .or_else(|_| {
                    chrono::NaiveDateTime::parse_from_str(&db_session.updated_at, "%Y-%m-%d %H:%M:%S")
                        .map(|ndt| ndt.and_utc())
                })
                .unwrap_or_else(|_| Utc::now());

            // Load messages from DB
            let db_messages = match storage.load_daemon_messages(&session_id).await {
                Ok(msgs) => msgs,
                Err(e) => {
                    warn!("Failed to load messages for session {}: {}", session_id, e);
                    Vec::new()
                }
            };

            let messages: Vec<SessionMessage> = db_messages.iter().map(|m| {
                db_message_to_session_message(
                    m.tool_use_id.as_deref().unwrap_or(&m.id.to_string()),
                    &m.role,
                    &m.content,
                    &m.created_at,
                    m.metadata.as_ref(),
                )
            }).collect();

            let session = Session {
                id: session_id.clone(),
                name: db_session.name,
                created_at,
                updated_at,
                messages,
                subscribers: HashSet::new(),
                owner: None,
                metadata: db_session.metadata
                    .and_then(|v| {
                        if let serde_json::Value::Object(map) = v {
                            Some(map.into_iter().collect())
                        } else {
                            None
                        }
                    })
                    .unwrap_or_default(),
                workspace_id: db_session.workspace_id,
            };

            sessions.insert(session_id.clone(), session);

            // First session becomes default
            if default.is_none() {
                *default = Some(session_id);
            }
        }
        // Both guards span the whole load, so no reader ever sees a
        // half-loaded store or a default naming a session not yet inserted.
        drop(default);
        drop(sessions);

        info!("Loaded {} sessions from database", count);
        count
    }

    /// Persist session metadata to DB (fire-and-forget on errors)
    async fn persist_session(&self, session: &Session) {
        self.persist_row(&SessionRow::from(session)).await;
    }

    /// Write one `sessions` row, metadata included (fire-and-forget on errors).
    ///
    /// The metadata blob always travels, even when the map is empty. The SQL is
    /// `metadata = COALESCE(?6, metadata)`, so a NULL means "keep whatever is on
    /// disk" — and sending NULL for an empty map made a key the user just
    /// REMOVED survive the write, with `load_from_db` resurrecting it on the
    /// next daemon start. Un-pinning a chat's model would have looked like it
    /// worked right up until a restart put the pin back. The literal `{}` is
    /// what makes a removal land.
    async fn persist_row(&self, row: &SessionRow) {
        let Some(ref storage) = self.storage else {
            warn!("persist_session called but no storage backend — session {} will not be persisted", row.id);
            return;
        };
        let metadata = match serde_json::to_string(&row.metadata) {
            Ok(json) => json,
            Err(e) => {
                warn!("Failed to serialize metadata for session {}: {}", row.id, e);
                return;
            }
        };
        match storage.upsert_daemon_session(
            &row.id,
            row.name.as_deref(),
            row.workspace_id.as_deref(),
            &row.created_at,
            &row.updated_at,
            Some(&metadata),
        ).await {
            Ok(()) => info!("Persisted session {} to database", row.id),
            Err(e) => warn!("Failed to persist session {} to database: {}", row.id, e),
        }
    }

    /// Persist a single message to DB
    async fn persist_message(&self, session_id: &str, msg: &SessionMessage) {
        let Some(ref storage) = self.storage else {
            warn!("persist_message called but no storage backend — message {} will not be persisted", msg.id);
            return;
        };
        let created = msg.timestamp.to_rfc3339();
        let metadata = message_to_metadata(msg);
        match storage.add_daemon_message(
            session_id,
            &msg.id,
            msg.role.as_db_str(),
            &msg.content,
            &created,
            metadata.as_deref(),
        ).await {
            Ok(()) => info!("Persisted {} message {} in session {}", msg.role.as_db_str(), msg.id, session_id),
            Err(e) => warn!("Failed to persist message {} in session {}: {}", msg.id, session_id, e),
        }
    }

    /// Fork session `id`: a new session in the same workspace, with the same
    /// settings (`metadata`: pinned model, tools, …) and a copy of every
    /// message under a fresh id, stored like any other. `None` if `id` does
    /// not exist.
    ///
    /// The control plane used to copy only the messages — dropping the
    /// workspace and settings — through [`Self::update`], which never stored
    /// them: the fork came back empty after a restart.
    pub async fn fork(&self, id: &str, name: Option<String>) -> Option<Session> {
        let original = self.get(id).await?;
        let name = name.or_else(|| original.name.as_ref().map(|n| format!("{n} (copy)")));
        let mut forked = self
            .create_in_workspace(name, original.workspace_id.clone())
            .await;
        forked.metadata = original.metadata.clone();
        forked.messages = original
            .messages
            .iter()
            .map(|m| SessionMessage {
                id: uuid::Uuid::new_v4().to_string(),
                ..m.clone()
            })
            .collect();
        debug_assert_eq!(forked.messages.len(), original.messages.len());
        self.replace(forked.clone()).await;
        Some(forked)
    }

    /// Create a session and return it
    pub async fn create(&self, name: Option<String>) -> Session {
        self.create_in_workspace(name, None).await
    }

    /// Create a new session in a specific workspace
    pub async fn create_in_workspace(&self, name: Option<String>, workspace_id: Option<String>) -> Session {
        let mut session = Session::new(name);
        session.workspace_id = workspace_id;
        let id = session.id.clone();
        
        // Persist to DB first
        self.persist_session(&session).await;

        let mut sessions = self.sessions.write().await;
        sessions.insert(id.clone(), session.clone());
        
        // Set as default if it's the first session
        let mut default = self.default_session.write().await;
        if default.is_none() {
            *default = Some(id.clone());
        }
        // `sessions` stays held until the default is settled: a `delete` of
        // this id cannot interleave between the insert and the default write
        // and leave the default naming a session that no longer exists.
        drop(default);
        drop(sessions);
        
        info!("Created session: {} (workspace: {:?})", id, session.workspace_id);
        session
    }
    
    /// Make sure the session a channel conversation maps to exists and routes
    /// its replies to `route`, creating it under that exact id when missing.
    ///
    /// A channel conversation is addressed by a key derived from the channel,
    /// not by an id a client obtained from `session.create`, so nothing else
    /// creates it — and `chat.send` refuses a session that does not exist.
    /// Returns whether the session was created.
    ///
    /// # Panics
    ///
    /// On an empty session id or a route missing its provider or chat — both
    /// are built by the caller from a message that already has them.
    pub async fn ensure_channel_session(
        &self,
        session_id: &str,
        name: &str,
        route: &ReplyChannel,
    ) -> bool {
        assert!(!session_id.is_empty(), "a channel session needs an id");
        assert!(
            !route.provider.is_empty(),
            "a reply route names its provider"
        );
        assert!(!route.id.is_empty(), "a reply route names its chat");
        let value = serde_json::json!({ "provider": route.provider, "id": route.id });
        let (row, created) = {
            let mut sessions = self.sessions.write().await;
            if let Some(session) = sessions.get_mut(session_id) {
                if session.reply_channel().as_ref() == Some(route) {
                    return false;
                }
                session
                    .metadata
                    .insert(REPLY_CHANNEL_KEY.to_string(), value);
                session.updated_at = Utc::now();
                let row = SessionRow::from(&*session);
                drop(sessions);
                (row, false)
            } else {
                let mut session = Session::with_id(session_id, Some(name.to_string()));
                session
                    .metadata
                    .insert(REPLY_CHANNEL_KEY.to_string(), value);
                let row = SessionRow::from(&session);
                sessions.insert(session_id.to_string(), session);
                drop(sessions);
                (row, true)
            }
        };
        // Written with the map unlocked, as `set_chat_model` does.
        self.persist_row(&row).await;
        if created {
            info!(
                "Created channel session {session_id} ({}:{})",
                route.provider, route.id
            );
        }
        created
    }

    /// The reply route of a session, read without cloning its messages.
    pub async fn reply_channel(&self, session_id: &str) -> Option<ReplyChannel> {
        self.sessions.read().await.get(session_id)?.reply_channel()
    }

    /// Set or clear the workspace for an existing session
    pub async fn set_workspace(&self, session_id: &str, workspace_id: Option<String>) -> bool {
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.get_mut(session_id) {
            session.workspace_id.clone_from(&workspace_id);
            // Persist to DB
            if let Some(ref storage) = self.storage
                && let Err(e) = storage.set_daemon_session_workspace(session_id, workspace_id.as_deref()).await
            {
                warn!("Failed to persist workspace change for session {}: {}", session_id, e);
            }
            true
        } else {
            false
        }
    }

    /// Set or clear this session's chat model (`None` = follow the global
    /// `[llm]` default).
    ///
    /// Chat only. Sub-agent, summarization and embedding models stay global on
    /// purpose — this writes one key and touches nothing else.
    pub async fn set_chat_model(&self, session_id: &str, model: Option<String>) -> bool {
        let Some(row) = self.apply_chat_model(session_id, model).await else {
            return false;
        };
        // The database round trip runs with the map unlocked — see
        // `apply_chat_model` for why that split exists.
        self.persist_row(&row).await;
        true
    }

    /// Apply a chat-model pick to the cached session and hand back the row that
    /// still has to be written; `None` when there is no such session.
    ///
    /// The two halves are separate because the `sessions` WRITE guard must not
    /// be held across the storage await. It once was, and every concurrent
    /// `get`/`list` then queued behind a full database round trip — including
    /// the read an in-flight turn does in `prepare_chat_turn` to resolve this
    /// very pin. The guard dies with this function; only the row crosses into
    /// the write.
    async fn apply_chat_model(&self, session_id: &str, model: Option<String>) -> Option<SessionRow> {
        // Normalize at the boundary so every reader sees a single shape for
        // "no override". A blank pick from a client is a CLEAR, not a pin on
        // the empty string — which no provider serves, so it would only ever
        // reach the turn's fail-fast.
        let model = model.filter(|spec| !spec.trim().is_empty());

        let mut sessions = self.sessions.write().await;
        let session = sessions.get_mut(session_id)?;
        match model {
            Some(spec) => {
                session
                    .metadata
                    .insert(CHAT_MODEL_KEY.to_string(), serde_json::Value::String(spec));
            }
            None => {
                session.metadata.remove(CHAT_MODEL_KEY);
            }
        }
        session.updated_at = Utc::now();
        info!("Session {} chat model set to {:?}", session_id, session.chat_model());
        let row = SessionRow::from(&*session);
        drop(sessions);
        Some(row)
    }

    /// Set or clear this session's user-selected extra tools (empty = no
    /// manual selection, i.e. default tool behavior).
    ///
    /// Additive by contract: the turn unions these into its default active
    /// set — they never remove or replace defaults. This writes one metadata
    /// key and touches nothing else.
    pub async fn set_chat_tools(&self, session_id: &str, tools: Vec<String>) -> bool {
        let Some(row) = self.apply_chat_tools(session_id, tools).await else {
            return false;
        };
        // The database round trip runs with the map unlocked — see
        // `apply_chat_model` for why that split exists.
        self.persist_row(&row).await;
        true
    }

    /// Apply a tool selection to the cached session and hand back the row that
    /// still has to be written; `None` when there is no such session.
    ///
    /// Same lock discipline as `apply_chat_model`: the write guard dies with
    /// this function; only the row crosses into the persist.
    async fn apply_chat_tools(&self, session_id: &str, tools: Vec<String>) -> Option<SessionRow> {
        // Normalize at the boundary: blank names are noise, and an empty
        // selection is stored as ABSENCE of the key so "never picked" and
        // "cleared the picks" are the same unset state.
        let tools: Vec<String> = tools
            .into_iter()
            .map(|t| t.trim().to_string())
            .filter(|t| !t.is_empty())
            .collect();

        let mut sessions = self.sessions.write().await;
        let session = sessions.get_mut(session_id)?;
        if tools.is_empty() {
            session.metadata.remove(CHAT_TOOLS_KEY);
        } else {
            session.metadata.insert(
                CHAT_TOOLS_KEY.to_string(),
                serde_json::Value::Array(
                    tools.into_iter().map(serde_json::Value::String).collect(),
                ),
            );
        }
        session.updated_at = Utc::now();
        info!("Session {} chat tools set to {:?}", session_id, session.chat_tools());
        let row = SessionRow::from(&*session);
        drop(sessions);
        Some(row)
    }

    /// Get a session by ID
    pub async fn get(&self, id: &str) -> Option<Session> {
        let sessions = self.sessions.read().await;
        sessions.get(id).cloned()
    }
    
    /// Get or create the default session
    pub async fn get_or_create_default(&self) -> Session {
        // Check if default exists
        let default_id = {
            let default = self.default_session.read().await;
            default.clone()
        };
        
        if let Some(id) = default_id
            && let Some(session) = self.get(&id).await
        {
            return session;
        }
        
        // Create new default
        self.create(Some("Main".to_string())).await
    }
    
    /// List all sessions
    pub async fn list(&self) -> Vec<SessionSummary> {
        let sessions = self.sessions.read().await;
        sessions.values().map(SessionSummary::from).collect()
    }
    
    /// Update a session from a caller-held snapshot.
    ///
    /// The snapshot is not authoritative about the chat-model pin. Callers read
    /// a session, mutate their copy, and write it back — `ChatAction::Regenerate`
    /// does exactly that — so a `session.setModel` landing inside that window
    /// would be undone by the write: a stale snapshot resurrects a pin the user
    /// just cleared, or drops one they just set, and the chat then runs on a
    /// model the picker says it is not using. [`Self::set_chat_model`] is the
    /// single writer of that key and the live map is where its write landed, so
    /// the live value wins and `update` stays pin-neutral — it can neither set
    /// nor clear a pin. A session the map has never seen has no live value to
    /// reconcile against, so its snapshot carries through as-is.
    /// Replace a session wholesale: its row AND its messages, in memory and in
    /// the store.
    ///
    /// [`Self::update`] persists only the row, so a caller that changed the
    /// message list through it changed memory alone: Regenerate's dropped
    /// reply came back from the store on the next restart, and a fork's copied
    /// messages were never stored at all. Messages are matched by id: the ones
    /// no longer present are deleted, the new ones written; unchanged ones are
    /// left as they are.
    pub async fn replace(&self, session: Session) {
        let before: HashSet<String> = self
            .sessions
            .read()
            .await
            .get(&session.id)
            .map(|live| live.messages.iter().map(|m| m.id.clone()).collect())
            .unwrap_or_default();
        let after: HashSet<&str> = session.messages.iter().map(|m| m.id.as_str()).collect();
        let removed: Vec<String> = before
            .iter()
            .filter(|id| !after.contains(id.as_str()))
            .cloned()
            .collect();
        let added: Vec<SessionMessage> = session
            .messages
            .iter()
            .filter(|m| !before.contains(&m.id))
            .cloned()
            .collect();
        let id = session.id.clone();
        self.update(session).await;
        if let Some(ref storage) = self.storage {
            for message_id in &removed {
                if let Err(e) = storage.delete_daemon_message(&id, message_id).await {
                    warn!("Failed to delete message {message_id} in session {id}: {e}");
                }
            }
        }
        for message in &added {
            self.persist_message(&id, message).await;
        }
    }

    pub async fn update(&self, mut session: Session) {
        let row = {
            let mut sessions = self.sessions.write().await;
            if let Some(live) = sessions.get(&session.id) {
                match live.metadata.get(CHAT_MODEL_KEY) {
                    Some(pin) => {
                        session
                            .metadata
                            .insert(CHAT_MODEL_KEY.to_string(), pin.clone());
                    }
                    None => {
                        session.metadata.remove(CHAT_MODEL_KEY);
                    }
                }
            }
            let row = SessionRow::from(&session);
            sessions.insert(session.id.clone(), session);
            row
        };
        self.persist_row(&row).await;
    }
    
    /// Delete a session
    pub async fn delete(&self, id: &str) -> bool {
        let mut sessions = self.sessions.write().await;
        let removed = sessions.remove(id).is_some();

        if removed {
            // Delete from DB
            if let Some(ref storage) = self.storage
                && let Err(e) = storage.delete_daemon_session(id).await
            {
                warn!("Failed to delete session {} from DB: {}", id, e);
            }
            // Clear default if it was this session
            let mut default = self.default_session.write().await;
            if default.as_deref() == Some(id) {
                *default = sessions.keys().next().cloned();
            }
            drop(default);
            info!("Deleted session: {}", id);
        }
        // Held through the DB delete and the default reassignment, which reads
        // the remaining keys: map, database and default change as one step.
        drop(sessions);

        removed
    }

    /// Delete all sessions
    pub async fn delete_all(&self) -> usize {
        let mut sessions = self.sessions.write().await;
        let count = sessions.len();

        // Delete all from DB
        if let Some(ref storage) = self.storage {
            for id in sessions.keys() {
                if let Err(e) = storage.delete_daemon_session(id).await {
                    warn!("Failed to delete session {} from DB: {}", id, e);
                }
            }
        }

        sessions.clear();

        // Clear default session
        let mut default = self.default_session.write().await;
        *default = None;
        // `sessions` stays held until the default is cleared, so a concurrent
        // create cannot insert and become the default in between only to have
        // that default wiped here.
        drop(default);
        drop(sessions);

        info!("Deleted all {} sessions", count);
        count
    }

    /// Rename a session
    pub async fn rename(&self, id: &str, name: String) -> bool {
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.get_mut(id) {
            session.name = Some(name.clone());
            session.updated_at = Utc::now();
            // Persist to DB
            if let Some(ref storage) = self.storage
                && let Err(e) = storage.rename_daemon_session(id, &name).await
            {
                warn!("Failed to persist rename for session {}: {}", id, e);
            }
            true
        } else {
            false
        }
    }
    
    /// Clear a session's messages
    pub async fn clear(&self, id: &str) -> bool {
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.get_mut(id) {
            session.clear();
            // Clear from DB
            if let Some(ref storage) = self.storage
                && let Err(e) = storage.clear_daemon_session_messages(id).await
            {
                warn!("Failed to clear messages for session {} in DB: {}", id, e);
            }
            true
        } else {
            false
        }
    }
    
    /// Add a message to a session (with write-through to DB)
    pub async fn add_message(
        &self,
        session_id: &str,
        role: MessageRole,
        content: impl Into<String>,
    ) -> Option<String> {
        let content = content.into();
        let mut sessions = self.sessions.write().await;
        let session = sessions.get_mut(session_id)?;
        let msg_id = session.add_message(role, content);
        let msg = session.messages.last().cloned();
        // Persist after releasing the lock (see `add_full_message`).
        drop(sessions);
        if let Some(msg) = msg {
            self.persist_message(session_id, &msg).await;
        }
        Some(msg_id)
    }

    /// Append an assistant message outside any streamed turn and announce it
    /// with `session_message_added`, so an open client shows it and the channel
    /// reply forwarder delivers it to a channel-owned conversation.
    ///
    /// The one path for daemon-originated messages — a reminder coming due, a
    /// clarifying question, a scheduled job's result — so none of them can be
    /// persisted without being announced, or the reverse. `None` when the
    /// session does not exist; nothing is sent then.
    pub async fn post_assistant_message(
        &self,
        events: &tokio::sync::broadcast::Sender<crate::protocol::Event>,
        session_id: &str,
        content: String,
    ) -> Option<String> {
        debug_assert!(
            !content.trim().is_empty(),
            "a posted message says something"
        );
        let message_id = self
            .add_message(session_id, MessageRole::Assistant, content.clone())
            .await?;
        // No receiver is not a failure: the message is persisted and read back
        // from history by the next client that opens the session.
        let _ = events.send(crate::protocol::Event::SessionMessageAdded {
            session_id: session_id.to_string(),
            message_id: message_id.clone(),
            role: "assistant".to_string(),
            content,
        });
        Some(message_id)
    }

    /// Add a message with tool calls, reasoning, run timeline, and usage totals to a session (with write-through to DB)
    pub async fn add_full_message(
        &self,
        session_id: &str,
        role: MessageRole,
        content: impl Into<String>,
        details: MessageDetails,
    ) -> Option<String> {
        let content = content.into();
        let mut sessions = self.sessions.write().await;
        let session = sessions.get_mut(session_id)?;
        let msg_id = session.add_full_message(role, content, details);
        let msg = session.messages.last().cloned();
        // Persisted after releasing the lock. The write guard over EVERY
        // session used to be held across this database write, so each
        // appended message stalled every session read and write in the daemon
        // for the length of a disk write. Order is safe: the timestamp the
        // store sorts by (`created_at`) was assigned above, under the lock.
        drop(sessions);
        if let Some(msg) = msg {
            self.persist_message(session_id, &msg).await;
        }
        Some(msg_id)
    }
    
    /// Whether a session exists, without loading it.
    ///
    /// `Subscribe{Session}` needs to know a session is real before narrowing a
    /// connection to it, and reached for [`Self::subscribe`] to find out — which
    /// also filed the IPC connection id into the session's `subscribers` set,
    /// a `HashSet<ChannelId>`. Nothing routes on that set, so the only effect
    /// was a `subscriber_count` inflated by a different kind of id.
    pub async fn exists(&self, session_id: &str) -> bool {
        self.sessions.read().await.contains_key(session_id)
    }

    /// Subscribe a channel to a session
    pub async fn subscribe(&self, session_id: &str, channel_id: ChannelId) -> bool {
        let mut sessions = self.sessions.write().await;
        sessions.get_mut(session_id).is_some_and(|session| {
            session.subscribe(channel_id);
            true
        })
    }
    
    /// Unsubscribe a channel from a session
    pub async fn unsubscribe(&self, session_id: &str, channel_id: &str) -> bool {
        let mut sessions = self.sessions.write().await;
        sessions.get_mut(session_id).is_some_and(|session| {
            session.unsubscribe(channel_id);
            true
        })
    }
    
    /// Get all sessions a channel is subscribed to
    pub async fn get_subscriptions(&self, channel_id: &str) -> Vec<SessionId> {
        let sessions = self.sessions.read().await;
        sessions.values()
            .filter(|s| s.is_subscribed(channel_id))
            .map(|s| s.id.clone())
            .collect()
    }
    
    /// Get subscribers for a session
    pub async fn get_subscribers(&self, session_id: &str) -> Vec<ChannelId> {
        let sessions = self.sessions.read().await;
        sessions.get(session_id)
            .map(|s| s.subscribers.iter().cloned().collect())
            .unwrap_or_default()
    }
    
    /// Get session count
    pub async fn count(&self) -> usize {
        self.sessions.read().await.len()
    }
}

impl Default for SessionManager {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionManager {
    /// Restore a session from persistence (used during startup / migration)
    pub async fn restore(&self, session: Session) {
        let id = session.id.clone();
        // Persist to DB if storage is available (for migration from JSON)
        self.persist_session(&session).await;
        // Also persist all messages
        for msg in &session.messages {
            self.persist_message(&id, msg).await;
        }

        let mut sessions = self.sessions.write().await;
        sessions.insert(id.clone(), session);
        
        // Set as default if it's the first session
        let mut default = self.default_session.write().await;
        if default.is_none() {
            *default = Some(id);
        }
        // Same ordering as `create_in_workspace`: the insert and the default
        // settle together.
        drop(default);
        drop(sessions);
    }
    
    /// Set the default session ID
    pub async fn set_default(&self, id: &str) {
        let sessions = self.sessions.read().await;
        if sessions.contains_key(id) {
            let mut default = self.default_session.write().await;
            *default = Some(id.to_string());
        }
    }
    
    /// Get the internal sessions map (for legacy code that needs it)
    #[must_use]
    pub fn sessions_map(&self) -> Arc<RwLock<HashMap<SessionId, Session>>> {
        self.sessions.clone()
    }
    
    /// Get the default session ID holder
    #[must_use]
    pub fn default_session_id(&self) -> Arc<RwLock<Option<SessionId>>> {
        self.default_session.clone()
    }
}

impl SessionManager {
    // =========================================================================
    // Sub-Session Management (#72)
    // =========================================================================

    /// Register a new sub-session (called after creating the Session)
    pub async fn register_sub_session(&self, info: SubSessionInfo) {
        let id = info.session_id.clone();
        info!("Registered sub-session: {} (parent: {:?}, label: {:?})", id, info.parent_id, info.label);
        self.sub_sessions.write().await.insert(id.clone(), info);
        // Initialize empty mailbox
        self.mailboxes.write().await.entry(id).or_default();
    }

    /// Update sub-session state
    ///
    /// A killed sub-session stays killed: the run it belonged to may still
    /// report in as it winds down, and that must not revive it.
    pub async fn set_sub_session_state(&self, session_id: &str, state: SubSessionState) {
        let mut subs = self.sub_sessions.write().await;
        if let Some(info) = subs.get_mut(session_id) {
            if info.state == SubSessionState::Killed {
                return;
            }
            info.state = state;
            if matches!(state, SubSessionState::Completed | SubSessionState::Failed | SubSessionState::Killed) {
                info.finished_at = Some(Utc::now());
            }
        }
    }

    /// Set sub-session result (on completion)
    pub async fn set_sub_session_result(&self, session_id: &str, result: String) {
        let mut subs = self.sub_sessions.write().await;
        if let Some(info) = subs.get_mut(session_id) {
            info.result = Some(result);
            // Kept as what the run wrote, but a killed run did not complete.
            if info.state != SubSessionState::Killed {
                info.state = SubSessionState::Completed;
                info.finished_at = Some(Utc::now());
            }
        }
    }

    /// Set sub-session error (on failure)
    pub async fn set_sub_session_error(&self, session_id: &str, error: String) {
        let mut subs = self.sub_sessions.write().await;
        if let Some(info) = subs.get_mut(session_id) {
            info.error = Some(error);
            if info.state != SubSessionState::Killed {
                info.state = SubSessionState::Failed;
                info.finished_at = Some(Utc::now());
            }
        }
    }

    /// Get sub-session info
    pub async fn get_sub_session(&self, session_id: &str) -> Option<SubSessionInfo> {
        self.sub_sessions.read().await.get(session_id).cloned()
    }

    /// Find a sub-session by label
    pub async fn find_sub_session_by_label(&self, label: &str) -> Option<SubSessionInfo> {
        self.sub_sessions.read().await.values()
            .find(|s| s.label.as_deref() == Some(label))
            .cloned()
    }

    /// Resolve a sub-session target (label or ID) to a `SubSessionInfo`
    pub async fn resolve_sub_session(&self, target: &str) -> Option<SubSessionInfo> {
        let subs = self.sub_sessions.read().await;
        if let Some(info) = subs.get(target) {
            return Some(info.clone());
        }
        subs.values()
            .find(|s| s.label.as_deref() == Some(target))
            .cloned()
    }

    /// List sub-sessions, optionally filtered by parent
    pub async fn list_sub_sessions(&self, parent_id: Option<&str>) -> Vec<SubSessionInfo> {
        let subs = self.sub_sessions.read().await;
        subs.values()
            .filter(|s| parent_id.is_none_or(|pid| s.parent_id.as_deref() == Some(pid)))
            .cloned()
            .collect()
    }

    /// Mark a sub-session killed. Stopping its run is the caller's (the
    /// control plane cancels the agent's chat for this session).
    pub async fn kill_sub_session(&self, session_id: &str) -> bool {
        let mut subs = self.sub_sessions.write().await;
        if let Some(info) = subs.get_mut(session_id) {
            info.state = SubSessionState::Killed;
            info.finished_at = Some(Utc::now());
            info!("Killed sub-session: {}", session_id);
            true
        } else {
            false
        }
    }

    /// Send a message to a session's mailbox
    pub async fn send_to_mailbox(&self, session_id: &str, from: &str, content: String) -> bool {
        let mut mailboxes = self.mailboxes.write().await;
        mailboxes.get_mut(session_id).is_some_and(|mailbox| {
            mailbox.push(MailboxMessage {
                from: from.to_string(),
                content,
                timestamp: Utc::now(),
            });
            true
        })
    }

    /// Drain all messages from a session's mailbox
    pub async fn drain_mailbox(&self, session_id: &str) -> Vec<MailboxMessage> {
        let mut mailboxes = self.mailboxes.write().await;
        mailboxes.get_mut(session_id)
            .map(std::mem::take)
            .unwrap_or_default()
    }

    /// Peek at a session's mailbox **without** consuming it (returns a clone).
    /// Used for status/inspection so checking a mailbox never eats pending
    /// inter-session messages the way `drain_mailbox` does.
    pub async fn peek_mailbox(&self, session_id: &str) -> Vec<MailboxMessage> {
        let mailboxes = self.mailboxes.read().await;
        mailboxes.get(session_id).cloned().unwrap_or_default()
    }

    /// Clean up completed/failed/killed sub-sessions older than the given duration
    pub async fn cleanup_sub_sessions(&self, max_age: std::time::Duration) {
        let cutoff = Utc::now() - chrono::Duration::from_std(max_age).unwrap_or(chrono::Duration::hours(24));
        let mut subs = self.sub_sessions.write().await;
        let mut mailboxes = self.mailboxes.write().await;

        let to_remove: Vec<String> = subs.iter()
            .filter(|(_, info)| {
                matches!(info.state, SubSessionState::Completed | SubSessionState::Failed | SubSessionState::Killed)
                    && info.finished_at.is_some_and(|t| t < cutoff)
            })
            .map(|(id, _)| id.clone())
            .collect();

        for id in &to_remove {
            subs.remove(id);
            mailboxes.remove(id);
        }
        // Both maps are swept under one acquisition, so no reader sees a
        // sub-session without its mailbox or the reverse.
        drop(mailboxes);
        drop(subs);

        if !to_remove.is_empty() {
            info!("Cleaned up {} completed sub-sessions", to_remove.len());
        }
    }
}

/// Largest byte index at or below `max` that `s` may be sliced at.
///
/// A `SessionId` is a plain `String`, not a validated UUID: `with_id`,
/// restored legacy sessions and rows read back from the database all carry
/// whatever text their source held. Slicing one at a fixed byte index panics
/// when the id is shorter than the limit or when the index lands inside a
/// multi-byte character, and a panic here would take the whole daemon down.
const fn floor_boundary(s: &str, max: usize) -> usize {
    if s.len() <= max {
        return s.len();
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    end
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn take_last_user_turn_drops_reply_and_returns_content() {
        let mut s = Session::new(None);
        s.add_message(MessageRole::User, "hi");
        s.add_message(MessageRole::Assistant, "hello");

        assert_eq!(s.take_last_user_turn().as_deref(), Some("hi"));
        assert!(s.messages.is_empty(), "user msg + reply both removed");
    }

    #[test]
    fn take_last_user_turn_preserves_prior_history() {
        let mut s = Session::new(None);
        s.add_message(MessageRole::User, "q1");
        s.add_message(MessageRole::Assistant, "a1");
        s.add_message(MessageRole::User, "q2");
        s.add_message(MessageRole::Assistant, "a2");

        assert_eq!(s.take_last_user_turn().as_deref(), Some("q2"));
        // The earlier completed turn stays; only the latest turn is peeled off.
        assert_eq!(s.messages.len(), 2);
        assert!(matches!(s.messages[0].role, MessageRole::User));
        assert_eq!(s.messages[0].content, "q1");
        assert!(matches!(s.messages[1].role, MessageRole::Assistant));
    }

    #[test]
    fn take_last_user_turn_drops_trailing_tool_turns() {
        let mut s = Session::new(None);
        s.add_message(MessageRole::System, "sys");
        s.add_message(MessageRole::User, "do it");
        s.add_message(MessageRole::Assistant, "working");
        s.add_message(MessageRole::Tool, "tool output");

        assert_eq!(s.take_last_user_turn().as_deref(), Some("do it"));
        // Everything from the user message onward is gone; the system msg stays.
        assert_eq!(s.messages.len(), 1);
        assert!(matches!(s.messages[0].role, MessageRole::System));
    }

    #[test]
    fn take_last_user_turn_none_when_no_user_message() {
        let mut s = Session::new(None);
        s.add_message(MessageRole::System, "sys");
        s.add_message(MessageRole::Assistant, "greeting");
        let before = s.messages.len();

        assert!(s.take_last_user_turn().is_none());
        assert_eq!(s.messages.len(), before, "session left unchanged");
    }

    /// REGRESSION: the display name previews the first 8 bytes of the id, and
    /// an id is any string a caller hands us. When byte 8 falls inside a
    /// multi-byte character the raw slice panics — the same defect that killed
    /// the daemon mid-run from a preview elsewhere.
    #[test]
    fn display_name_clamps_multibyte_id_to_char_boundary() {
        // "session" is 7 bytes and the em dash occupies bytes 7..10, so the
        // 8-byte limit lands inside it.
        let s = Session::with_id("session\u{2014}id", None);

        assert_eq!(s.display_name(), "Session session");
    }

    /// REGRESSION: legacy sessions restored from disk carry short ids like
    /// "main", which the fixed-index slice ran off the end of.
    #[test]
    fn display_name_survives_id_shorter_than_the_preview() {
        let s = Session::with_id("main", None);

        assert_eq!(s.display_name(), "Session main");
    }

    /// REGRESSION (P19): a run's tool calls are first-class citizens of the
    /// chat — they must survive not just navigation (see
    /// `tasks::tests::tool_calls_survive_navigation_via_run_buffers`) but a
    /// full daemon restart. The timeline journal persisted with the message
    /// must round-trip through Turso intact: a fresh `SessionManager` over the
    /// same database restores the message with its tool call, input, output
    /// and verdict in place.
    #[tokio::test]
    async fn a_runs_tool_calls_survive_daemon_restart() {
        let storage = Arc::new(nanna_storage::Storage::in_memory().await.expect("storage"));
        let manager = SessionManager::with_storage(storage.clone());
        let session = manager.create(Some("restart-proof".to_string())).await;

        let at = Utc::now().to_rfc3339();
        let timeline = vec![
            TimelineItem::Thinking {
                content: "which files exist?".to_string(),
                at: at.clone(),
            },
            TimelineItem::Tool {
                call_id: "c1".to_string(),
                name: "exec".to_string(),
                input: Some(serde_json::json!({"cmd": "ls"})),
                output: Some("file.txt".to_string()),
                success: Some(true),
                duration_ms: Some(12),
                tokens: None,
                total_tokens: None,
                diff: None,
                short_circuited: Some(false),
                at: at.clone(),
            },
            // A breaker replay beside it: `success: false` with a notice for
            // output, distinguishable from a crash ONLY by the marker. This is
            // the entry a restored timeline used to render as a tool error.
            TimelineItem::Tool {
                call_id: "c2".to_string(),
                name: "exec".to_string(),
                input: Some(serde_json::json!({"cmd": "ls"})),
                output: Some("[replayed: you already ran this]".to_string()),
                success: Some(false),
                duration_ms: Some(0),
                tokens: None,
                total_tokens: None,
                diff: None,
                short_circuited: Some(true),
                at: at.clone(),
            },
            TimelineItem::Text {
                content: "there is one file: file.txt".to_string(),
                at,
            },
        ];
        manager
            .add_full_message(
                &session.id,
                MessageRole::Assistant,
                "there is one file: file.txt",
                MessageDetails {
                    timeline,
                    ..MessageDetails::default()
                },
            )
            .await;

        // "Restart": a brand-new manager over the same database, exactly what
        // daemon startup does via load_from_db.
        let reborn = SessionManager::with_storage(storage);
        assert!(reborn.load_from_db().await >= 1, "session loads from Turso");

        let restored = reborn.get(&session.id).await.expect("session survives restart");
        let msg = restored
            .messages
            .last()
            .expect("assistant message survives restart");
        assert_eq!(msg.content, "there is one file: file.txt");
        assert_eq!(msg.timeline.len(), 4, "the full journal round-trips");
        assert!(matches!(
            &msg.timeline[1],
            TimelineItem::Tool {
                call_id,
                input: Some(input),
                output: Some(output),
                success: Some(true),
                duration_ms: Some(12),
                short_circuited: Some(false),
                ..
            } if call_id == "c1" && input["cmd"] == "ls" && output == "file.txt"
        ));
        // The replay survives the restart AS a replay. Without the marker on
        // the persisted item this entry is `success: false` and nothing else,
        // so the rebuilt timeline had no way to render it as steering.
        assert!(matches!(
            &msg.timeline[2],
            TimelineItem::Tool {
                call_id,
                success: Some(false),
                short_circuited: Some(true),
                ..
            } if call_id == "c2"
        ));
    }

    fn diff_data(start_line: u64, removed: &[&str], added: &[&str]) -> serde_json::Value {
        serde_json::json!({ "diff": {
            "start_line": start_line, "removed": removed, "added": added, "truncated": false
        }})
    }

    /// P18 "Diff presentation": a well-formed view from `edit_file` reads back
    /// exactly, and a small one is not marked cut.
    #[test]
    fn an_edit_diff_reads_back_exactly() {
        let data = diff_data(2, &["line two"], &["line 2"]);
        let diff = EditDiff::from_tool_data(Some(&data)).expect("a well-formed view parses");
        assert_eq!(diff.start_line, 2);
        assert_eq!(diff.removed, vec!["line two"]);
        assert_eq!(diff.added, vec!["line 2"]);
        assert!(!diff.truncated, "nothing was cut");
    }

    /// The payload is skill output: every malformed shape is `None`, never a
    /// panic inside the spawned turn that journals it.
    #[test]
    fn a_malformed_edit_diff_is_dropped_not_trusted() {
        let cases = [
            serde_json::json!({}),
            serde_json::json!({ "diff": "not an object" }),
            serde_json::json!({ "diff": { "start_line": 0, "removed": ["a"], "added": ["b"] } }),
            serde_json::json!({ "diff": { "start_line": -3, "removed": ["a"], "added": ["b"] } }),
            serde_json::json!({ "diff": { "start_line": 1.5, "removed": ["a"], "added": ["b"] } }),
            serde_json::json!({ "diff": { "start_line": 1, "removed": "a", "added": ["b"] } }),
            serde_json::json!({ "diff": { "start_line": 1, "removed": [1], "added": ["b"] } }),
            serde_json::json!({ "diff": { "start_line": 1, "removed": [], "added": [] } }),
        ];
        for case in &cases {
            assert_eq!(
                EditDiff::from_tool_data(Some(case)),
                None,
                "must be dropped: {case}"
            );
        }
        assert_eq!(EditDiff::from_tool_data(None), None, "no data, no diff");
    }

    /// A skill that stops honouring the bound cannot bloat the journal: the
    /// side is cut to the line bound, and the view says it was cut.
    #[test]
    fn an_oversized_edit_diff_is_clamped_and_says_so() {
        let many: Vec<String> = (0..100).map(|i| format!("old {i}")).collect();
        let data = serde_json::json!({ "diff": {
            "start_line": 5, "removed": many, "added": ["new"], "truncated": false
        }});
        let diff = EditDiff::from_tool_data(Some(&data)).expect("an oversized view still parses");
        assert_eq!(
            diff.removed.len(),
            EDIT_DIFF_LINES_MAX,
            "cut to the line bound"
        );
        assert_eq!(diff.removed[0], "old 0", "the head of the change is kept");
        assert_eq!(diff.added, vec!["new"], "the other side is untouched");
        assert!(diff.truncated, "a clamped view must say so");
    }

    /// The byte bound cuts through a line only on a char boundary: a line of
    /// em dashes (3 bytes each) three times the bound must not panic, and
    /// keeps as much as the boundary allows.
    #[test]
    fn an_edit_diff_byte_cut_never_splits_a_char() {
        let dashes = "\u{2014}".repeat(EDIT_DIFF_BYTES_MAX);
        let data = serde_json::json!({ "diff": {
            "start_line": 1, "removed": [dashes], "added": ["x"]
        }});
        let diff = EditDiff::from_tool_data(Some(&data)).expect("parses; truncated defaults");
        let kept = &diff.removed[0];
        assert!(kept.len() <= EDIT_DIFF_BYTES_MAX, "within the byte bound");
        assert!(
            kept.len() > EDIT_DIFF_BYTES_MAX - 3,
            "…as close as a char boundary allows"
        );
        assert!(
            kept.chars().all(|c| c == '\u{2014}'),
            "no partial char survived"
        );
        assert!(diff.truncated, "a cut line must say so");
    }

    /// The journal rule: a diff is kept only for a call that succeeded and
    /// really ran — a failed edit changed nothing, a replay never ran.
    #[test]
    fn a_diff_is_journaled_only_for_an_edit_that_landed() {
        let data = diff_data(1, &["a"], &["b"]);
        assert!(EditDiff::for_outcome(true, false, Some(&data)).is_some());
        assert_eq!(
            EditDiff::for_outcome(false, false, Some(&data)),
            None,
            "a failed edit"
        );
        assert_eq!(
            EditDiff::for_outcome(true, true, Some(&data)),
            None,
            "a breaker replay"
        );
    }

    /// The point of journaling it: a user who opens the session after the run
    /// — after a daemon restart, even — still sees what each edit changed.
    #[tokio::test]
    async fn a_journaled_edit_diff_survives_daemon_restart() {
        let storage = Arc::new(nanna_storage::Storage::in_memory().await.expect("storage"));
        let manager = SessionManager::with_storage(storage.clone());
        let session = manager.create(Some("diff-survives".to_string())).await;
        let diff = EditDiff {
            start_line: 7,
            removed: vec!["let x = 1;".to_string()],
            added: vec!["let x = 2;".to_string()],
            truncated: false,
        };
        let timeline = vec![TimelineItem::Tool {
            call_id: "c1".to_string(),
            name: "edit_file".to_string(),
            input: None,
            output: Some("Edited a.rs".to_string()),
            success: Some(true),
            duration_ms: Some(4),
            tokens: None,
            total_tokens: None,
            short_circuited: Some(false),
            diff: Some(diff.clone()),
            at: Utc::now().to_rfc3339(),
        }];
        manager
            .add_full_message(
                &session.id,
                MessageRole::Assistant,
                "done",
                MessageDetails {
                    timeline,
                    ..MessageDetails::default()
                },
            )
            .await;

        let reborn = SessionManager::with_storage(storage);
        assert!(reborn.load_from_db().await >= 1, "session loads from Turso");
        let restored = reborn
            .get(&session.id)
            .await
            .expect("session survives restart");
        let msg = restored.messages.last().expect("message survives restart");
        match msg.timeline.first() {
            Some(TimelineItem::Tool {
                diff: Some(kept), ..
            }) => assert_eq!(*kept, diff),
            other => panic!("expected the edit with its diff, got {other:?}"),
        }
    }

    /// A session that never picked a model must be indistinguishable from one
    /// that existed before per-chat models did: no key, no override, and a
    /// listing that says nothing about a model.
    #[tokio::test]
    async fn an_unpicked_session_reports_no_chat_model() {
        let manager = SessionManager::new();
        let session = manager.create(None).await;

        assert_eq!(session.chat_model(), None);
        assert!(session.metadata.is_empty(), "no key is the unset state");
        assert_eq!(SessionSummary::from(&session).chat_model, None);
    }

    #[tokio::test]
    async fn a_chat_model_pick_shows_up_on_the_session_and_in_the_listing() {
        let manager = SessionManager::new();
        let session = manager.create(None).await;

        assert!(
            manager
                .set_chat_model(&session.id, Some("ollama/qwen3:14b".to_string()))
                .await
        );

        // Stored verbatim in router form, so the turn can hand it straight to
        // the router without re-deriving a provider prefix.
        let picked = manager.get(&session.id).await.expect("session");
        assert_eq!(picked.chat_model(), Some("ollama/qwen3:14b"));

        // `session.list` is what a client hydrates from; without this it would
        // need a `session.get` per row just to render the pin.
        let listed = manager.list().await;
        let summary = listed
            .iter()
            .find(|s| s.id == session.id)
            .expect("session is listed");
        assert_eq!(summary.chat_model.as_deref(), Some("ollama/qwen3:14b"));
    }

    /// A blank pick is a CLEAR, normalized here so every reader downstream sees
    /// one shape for "no override". Pinning the empty string would instead
    /// produce a chat no provider serves.
    #[tokio::test]
    async fn a_blank_chat_model_pick_clears_the_pin() {
        let manager = SessionManager::new();
        let session = manager.create(None).await;

        manager
            .set_chat_model(&session.id, Some("ollama/qwen3:14b".to_string()))
            .await;
        manager.set_chat_model(&session.id, Some("   ".to_string())).await;

        let cleared = manager.get(&session.id).await.expect("session");
        assert_eq!(cleared.chat_model(), None);
        assert!(!cleared.metadata.contains_key(CHAT_MODEL_KEY), "the key is removed, not blanked");
    }

    #[tokio::test]
    async fn setting_a_chat_model_on_an_unknown_session_reports_failure() {
        let manager = SessionManager::new();

        assert!(
            !manager
                .set_chat_model("no-such-session", Some("ollama/qwen3:14b".to_string()))
                .await
        );
    }

    #[tokio::test]
    async fn a_chat_model_pick_survives_a_daemon_restart() {
        let storage = Arc::new(nanna_storage::Storage::in_memory().await.expect("storage"));
        let manager = SessionManager::with_storage(storage.clone());
        let session = manager.create(Some("pinned".to_string())).await;

        manager
            .set_chat_model(&session.id, Some("ollama/qwen3:14b".to_string()))
            .await;

        let reborn = SessionManager::with_storage(storage);
        assert!(reborn.load_from_db().await >= 1, "session loads from Turso");
        let restored = reborn.get(&session.id).await.expect("session survives restart");

        assert_eq!(restored.chat_model(), Some("ollama/qwen3:14b"));
    }

    /// REGRESSION (the COALESCE trap): `upsert_daemon_session` writes
    /// `metadata = COALESCE(?6, metadata)`, so persisting an empty map as NULL
    /// leaves the old blob on disk and `load_from_db` resurrects it. Un-pinning
    /// a chat would then appear to work right up until the next daemon start
    /// put the pin back — a silent lie about which model the chat runs on.
    #[tokio::test]
    async fn clearing_a_chat_model_pick_survives_a_daemon_restart() {
        let storage = Arc::new(nanna_storage::Storage::in_memory().await.expect("storage"));
        let manager = SessionManager::with_storage(storage.clone());
        let session = manager.create(Some("un-pinned".to_string())).await;

        manager
            .set_chat_model(&session.id, Some("ollama/qwen3:14b".to_string()))
            .await;
        manager.set_chat_model(&session.id, None).await;

        let reborn = SessionManager::with_storage(storage);
        assert!(reborn.load_from_db().await >= 1, "session loads from Turso");
        let restored = reborn.get(&session.id).await.expect("session survives restart");

        assert_eq!(
            restored.chat_model(),
            None,
            "the clear must land on disk, not just in the hot cache"
        );
    }

    /// The metadata blob is shared with whatever else a session records, so the
    /// replacing write has to carry the siblings through — both when the key is
    /// added and when it is removed.
    #[tokio::test]
    async fn a_chat_model_pick_leaves_the_rest_of_the_metadata_alone() {
        let storage = Arc::new(nanna_storage::Storage::in_memory().await.expect("storage"));
        let manager = SessionManager::with_storage(storage.clone());
        let mut session = manager.create(None).await;
        session
            .metadata
            .insert("source".to_string(), serde_json::Value::String("telegram".to_string()));
        manager.update(session.clone()).await;

        manager
            .set_chat_model(&session.id, Some("ollama/qwen3:14b".to_string()))
            .await;
        manager.set_chat_model(&session.id, None).await;

        let reborn = SessionManager::with_storage(storage);
        reborn.load_from_db().await;
        let restored = reborn.get(&session.id).await.expect("session survives restart");

        assert_eq!(restored.chat_model(), None);
        assert_eq!(
            restored.metadata.get("source").and_then(serde_json::Value::as_str),
            Some("telegram"),
            "removing the pin must not take the neighbours with it"
        );
    }

    /// The chat-model pin as the DATABASE holds it, read past the hot cache —
    /// the only way to tell "written" from "merely remembered".
    async fn pinned_model_on_disk(
        storage: &nanna_storage::Storage,
        session_id: &str,
    ) -> Option<String> {
        storage
            .list_daemon_sessions()
            .await
            .expect("sessions load")
            .into_iter()
            .find(|row| row.session_id == session_id)
            .and_then(|row| row.metadata)
            .and_then(|meta| {
                meta.get(CHAT_MODEL_KEY)
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string)
            })
    }

    /// REGRESSION: a model pick used to hold the `sessions` WRITE guard across
    /// its database round trip, parking every concurrent `get`/`list` behind a
    /// disk write — the turn resolving this very pin among them. The pick and
    /// the write are now separate steps: `apply_chat_model` takes the guard,
    /// mutates, and gives it back, so by the time the row exists the map is
    /// readable, writable, and already telling the truth — with the database
    /// still untouched.
    #[tokio::test]
    async fn a_chat_model_pick_lands_in_memory_before_the_database_write() {
        let storage = Arc::new(nanna_storage::Storage::in_memory().await.expect("storage"));
        let manager = SessionManager::with_storage(storage.clone());
        let session = manager.create(None).await;

        let row = manager
            .apply_chat_model(&session.id, Some("ollama/qwen3:14b".to_string()))
            .await
            .expect("session exists");

        assert!(
            manager.sessions_map().try_write().is_ok(),
            "the guard is released before the write, not after it"
        );
        assert_eq!(
            manager.get(&session.id).await.expect("session").chat_model(),
            Some("ollama/qwen3:14b"),
            "readers see the pick without waiting on storage"
        );
        assert_eq!(
            pinned_model_on_disk(&storage, &session.id).await,
            None,
            "and all of that happened before the database was touched"
        );

        manager.persist_row(&row).await;
        assert_eq!(
            pinned_model_on_disk(&storage, &session.id).await.as_deref(),
            Some("ollama/qwen3:14b"),
            "the row the mutation handed back is what makes the pick durable"
        );
    }

    /// REGRESSION: `ChatAction::Regenerate` reads a session, peels the last turn
    /// off its copy, and writes that copy back. A `session.setModel` landing
    /// inside that window used to be undone — the snapshot still carried the old
    /// pin and put it straight back, so the chat kept running on a model the
    /// picker said it had stopped using.
    #[tokio::test]
    async fn update_cannot_resurrect_a_pin_cleared_after_its_snapshot() {
        let storage = Arc::new(nanna_storage::Storage::in_memory().await.expect("storage"));
        let manager = SessionManager::with_storage(storage.clone());
        let session = manager.create(Some("regenerating".to_string())).await;
        manager
            .set_chat_model(&session.id, Some("ollama/qwen3:14b".to_string()))
            .await;

        // The snapshot a regenerate is holding, taken while the pin was set.
        let mut snapshot = manager.get(&session.id).await.expect("session");
        assert_eq!(snapshot.chat_model(), Some("ollama/qwen3:14b"));

        // The user un-pins while the regenerate is still assembling.
        manager.set_chat_model(&session.id, None).await;

        // ...and the regenerate writes its snapshot back.
        snapshot.add_message(MessageRole::User, "again");
        manager.update(snapshot).await;

        let live = manager.get(&session.id).await.expect("session");
        assert_eq!(live.chat_model(), None, "the clear survives the write-back");
        assert_eq!(live.messages.len(), 1, "the caller's own edit still lands");

        let reborn = SessionManager::with_storage(storage);
        reborn.load_from_db().await;
        let restored = reborn.get(&session.id).await.expect("session survives restart");
        assert_eq!(
            restored.chat_model(),
            None,
            "and a restart does not bring the resurrected pin back"
        );
    }

    /// The mirror of the resurrection case: a pin set after the snapshot was
    /// taken must not be dropped by the write-back either. `update` is
    /// pin-neutral — [`SessionManager::set_chat_model`] is the only writer of
    /// that key.
    #[tokio::test]
    async fn update_cannot_drop_a_pin_set_after_its_snapshot() {
        let storage = Arc::new(nanna_storage::Storage::in_memory().await.expect("storage"));
        let manager = SessionManager::with_storage(storage.clone());
        let session = manager.create(Some("regenerating".to_string())).await;

        let mut snapshot = manager.get(&session.id).await.expect("session");
        assert_eq!(snapshot.chat_model(), None);

        manager
            .set_chat_model(&session.id, Some("ollama/qwen3:14b".to_string()))
            .await;

        snapshot.add_message(MessageRole::User, "again");
        manager.update(snapshot).await;

        let live = manager.get(&session.id).await.expect("session");
        assert_eq!(live.chat_model(), Some("ollama/qwen3:14b"));

        let reborn = SessionManager::with_storage(storage);
        reborn.load_from_db().await;
        let restored = reborn.get(&session.id).await.expect("session survives restart");
        assert_eq!(restored.chat_model(), Some("ollama/qwen3:14b"));
    }

    #[tokio::test]
    async fn test_peek_mailbox_is_non_destructive() {
        let manager = SessionManager::new();
        let info = SubSessionInfo {
            session_id: "sub-1".to_string(),
            parent_id: None,
            label: None,
            task: "t".to_string(),
            state: SubSessionState::Running,
            spawned_at: Utc::now(),
            finished_at: None,
            model: None,
            result: None,
            error: None,
        };
        manager.register_sub_session(info).await;

        assert!(
            manager
                .send_to_mailbox("sub-1", "parent", "hi".to_string())
                .await
        );
        assert!(
            manager
                .send_to_mailbox("sub-1", "parent", "again".to_string())
                .await
        );

        // Peeking twice returns the same messages without consuming them.
        assert_eq!(manager.peek_mailbox("sub-1").await.len(), 2);
        assert_eq!(manager.peek_mailbox("sub-1").await.len(), 2);

        // Draining consumes; a subsequent peek is empty.
        assert_eq!(manager.drain_mailbox("sub-1").await.len(), 2);
        assert_eq!(manager.peek_mailbox("sub-1").await.len(), 0);

        // Peeking an unknown session yields an empty vec (no panic).
        assert!(manager.peek_mailbox("nope").await.is_empty());
    }

    #[tokio::test]
    async fn test_session_manager() {
        let manager = SessionManager::new();
        
        // Create session
        let session = manager.create(Some("Test".to_string())).await;
        assert_eq!(session.name, Some("Test".to_string()));
        
        // Get session
        let retrieved = manager.get(&session.id).await.unwrap();
        assert_eq!(retrieved.id, session.id);
        
        // List sessions
        let list = manager.list().await;
        assert_eq!(list.len(), 1);
        
        // Subscribe channel
        manager.subscribe(&session.id, "gui:123".to_string()).await;
        let subs = manager.get_subscribers(&session.id).await;
        assert!(subs.contains(&"gui:123".to_string()));
        
        // Delete session
        manager.delete(&session.id).await;
        assert!(manager.get(&session.id).await.is_none());
    }
}
