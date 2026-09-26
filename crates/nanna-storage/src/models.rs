//! Database models

use serde::{Deserialize, Serialize};

/// Session model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: i64,
    pub session_id: String,
    pub channel: String,
    pub user_id: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    pub metadata: Option<serde_json::Value>,
    /// Optional workspace this session belongs to (None = global)
    pub workspace_id: Option<String>,
    /// Human-readable session name
    pub name: Option<String>,
}

/// Message model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: i64,
    pub session_id: String,
    pub role: String,
    pub content: String,
    pub content_type: String,
    pub tool_use_id: Option<String>,
    pub created_at: String,
    pub tokens_in: Option<i64>,
    pub tokens_out: Option<i64>,
    pub metadata: Option<serde_json::Value>,
}

/// Memory model (for vector search)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: i64,
    pub memory_id: String,
    pub content: String,
    pub embedding: Option<Vec<f32>>,
    pub embedding_model: Option<String>,
    pub session_id: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    pub metadata: Option<serde_json::Value>,
    pub tags: Vec<String>,
    /// Workspace scope (None = global)
    pub workspace_id: Option<String>,
    /// FSRS cognitive state fields
    pub fsrs_stability: f32,
    pub fsrs_difficulty: f32,
    pub fsrs_last_access: i64,
    pub fsrs_access_count: i64,
    pub fsrs_importance: f32,
    pub fsrs_storage_strength: f32,
    pub fsrs_generation: i64,
}

/// Cron job model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CronJob {
    pub id: i64,
    pub job_id: String,
    pub schedule: String,
    pub task: serde_json::Value,
    pub enabled: bool,
    pub last_run: Option<String>,
    pub next_run: Option<String>,
    pub created_at: String,
    pub metadata: Option<serde_json::Value>,
}

/// New message input
#[derive(Debug, Clone)]
pub struct NewMessage {
    pub session_id: String,
    pub role: String,
    pub content: String,
    pub content_type: String,
    pub tool_use_id: Option<String>,
    pub tokens_in: Option<i64>,
    pub tokens_out: Option<i64>,
    pub metadata: Option<serde_json::Value>,
}

/// New memory input
#[derive(Debug, Clone)]
pub struct NewMemory {
    pub memory_id: String,
    pub content: String,
    pub embedding: Option<Vec<f32>>,
    pub embedding_model: Option<String>,
    pub session_id: Option<String>,
    pub metadata: Option<serde_json::Value>,
    pub tags: Vec<String>,
    /// Workspace scope (None = global)
    pub workspace_id: Option<String>,
    /// FSRS cognitive state fields
    pub fsrs_stability: f32,
    pub fsrs_difficulty: f32,
    pub fsrs_last_access: i64,
    pub fsrs_access_count: i64,
    pub fsrs_importance: f32,
    pub fsrs_storage_strength: f32,
    pub fsrs_generation: i64,
}

/// FSRS state of one memory, as `MemoryRepository::update_fsrs` writes it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MemoryFsrsUpdate {
    pub stability: f32,
    pub difficulty: f32,
    pub last_access: i64,
    pub access_count: i64,
    pub importance: f32,
    pub storage_strength: f32,
    pub generation: i64,
}

/// New cron job input
#[derive(Debug, Clone)]
pub struct NewCronJob {
    pub job_id: String,
    pub schedule: String,
    pub task: serde_json::Value,
    pub enabled: bool,
    pub next_run: Option<String>,
    pub metadata: Option<serde_json::Value>,
}

/// Job run history entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRun {
    pub id: i64,
    pub job_id: String,
    pub started_at: String,
    pub finished_at: Option<String>,
    pub success: bool,
    pub output: Option<String>,
    pub error: Option<String>,
    pub duration_ms: Option<i64>,
}

/// New job run input
#[derive(Debug, Clone)]
pub struct NewJobRun {
    pub job_id: String,
    pub started_at: String,
    pub finished_at: Option<String>,
    pub success: bool,
    pub output: Option<String>,
    pub error: Option<String>,
    pub duration_ms: Option<i64>,
}

/// Registered workspace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceRecord {
    pub id: String,
    pub name: String,
    pub path: String,
    pub active: bool,
    pub created_at: String,
    pub last_accessed: String,
}

/// Task model (agent-grade task store, P15).
///
/// `status` is one of `pending | in_progress | done | cancelled`. `blocked` is
/// never stored: it is derived from `depends_on` at read time by
/// `TaskRepository` — a task is blocked while any dependency is not `done`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: i64,
    pub parent_id: Option<i64>,
    /// `session` | `workspace` | `global`
    pub scope: String,
    /// `session_id` or `workspace_id` depending on scope (None for global)
    pub scope_id: Option<String>,
    pub project: Option<String>,
    pub title: String,
    pub description: Option<String>,
    pub status: String,
    /// 1 (highest) ..= 4 (lowest), Todoist-style
    pub priority: i64,
    pub labels: Vec<String>,
    /// Tool names the current item scopes the agent to (P14 per-item tool hint)
    pub tool_scope: Vec<String>,
    /// The DEFER date (P25 decision 10): the card stays out of the inbox until
    /// this arrives. `None` means now.
    pub due_at: Option<String>,
    /// The bound: the card must be complete by then. `overdue` is measured
    /// against this and never against [`Self::due_at`].
    pub deadline_at: Option<String>,
    /// Cron expression executed by the existing scheduler (one recurrence engine)
    pub recurrence: Option<String>,
    pub depends_on: Vec<i64>,
    /// Machine-checkable done-condition, run by the harness — JSON
    /// `{kind: "command"|"file_exists"|"regex", ...}`
    pub acceptance: Option<serde_json::Value>,
    /// Which agent owns this item (parent vs sub-agent)
    pub assignee: Option<String>,
    pub sort_order: i64,
    pub created_at: String,
    pub updated_at: String,
    pub completed_at: Option<String>,
    /// Derived at read time: true while any dependency is not done.
    #[serde(default)]
    pub blocked: bool,
    /// When the sweep announced that [`Self::due_at`] arrived, so it announces
    /// once rather than every pass. Store-internal: no caller sets it, and
    /// changing `due_at` clears it.
    pub due_announced_at: Option<String>,
    /// When the sweep announced that [`Self::deadline_at`] passed. Same
    /// contract as [`Self::due_announced_at`]; changing `deadline_at` clears
    /// it, and reopening the card clears both.
    pub overdue_announced_at: Option<String>,
}

/// New task input
#[derive(Debug, Clone, Default)]
pub struct NewTask {
    pub parent_id: Option<i64>,
    pub scope: String,
    pub scope_id: Option<String>,
    pub project: Option<String>,
    pub title: String,
    pub description: Option<String>,
    pub priority: i64,
    pub labels: Vec<String>,
    pub tool_scope: Vec<String>,
    /// Defer date — see [`Task::due_at`].
    pub due_at: Option<String>,
    /// Completion bound — see [`Task::deadline_at`].
    pub deadline_at: Option<String>,
    pub recurrence: Option<String>,
    pub depends_on: Vec<i64>,
    pub acceptance: Option<serde_json::Value>,
    pub assignee: Option<String>,
    pub sort_order: i64,
}

/// Partial task update; `None` fields are left untouched.
///
/// `status` accepts `pending | in_progress` only — `done` must go through
/// `TaskRepository::complete` (acceptance is a verdict, not an assertion) and
/// `blocked` is derived, never written.
#[derive(Debug, Clone, Default)]
pub struct TaskPatch {
    pub parent_id: Option<Option<i64>>,
    pub project: Option<Option<String>>,
    pub title: Option<String>,
    pub description: Option<Option<String>>,
    pub status: Option<String>,
    pub priority: Option<i64>,
    pub labels: Option<Vec<String>>,
    pub tool_scope: Option<Vec<String>>,
    pub due_at: Option<Option<String>>,
    pub deadline_at: Option<Option<String>>,
    pub recurrence: Option<Option<String>>,
    pub depends_on: Option<Vec<i64>>,
    pub acceptance: Option<Option<serde_json::Value>>,
    pub assignee: Option<Option<String>>,
    pub sort_order: Option<i64>,
}

/// What a thread post is doing (P25 decision 2). The four kinds are what let
/// the router read a thread without re-deriving intent from prose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskNoteKind {
    /// Ordinary prose from a member.
    Comment,
    /// A step happened; noise to everyone but the board.
    Progress,
    /// Blocks: someone needs an answer.
    Question,
    /// The acceptance check's outcome.
    Verdict,
}

impl TaskNoteKind {
    /// The token stored in `task_notes.kind`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Comment => "comment",
            Self::Progress => "progress",
            Self::Question => "question",
            Self::Verdict => "verdict",
        }
    }

    /// Parse a stored token. `None` for a kind this version does not know, so
    /// a post written by a newer schema is rejected rather than flattened into
    /// a comment.
    #[must_use]
    pub fn parse(token: &str) -> Option<Self> {
        match token {
            "comment" => Some(Self::Comment),
            "progress" => Some(Self::Progress),
            "question" => Some(Self::Question),
            "verdict" => Some(Self::Verdict),
            _ => None,
        }
    }
}

/// One post in a card's thread. Append-only: there is no update path, because
/// the thread is the permanent record (P25 decision 14).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskNote {
    pub id: i64,
    pub task_id: i64,
    /// Pre-members actor string (`gui`, `harness`, an agent name). Legacy —
    /// Stage 4 removes it once every writer names a member.
    pub author: Option<String>,
    /// The board member who posted. `None` on rows written before members
    /// existed, which reads correctly: those posts predate the entity.
    pub author_member_id: Option<String>,
    pub kind: TaskNoteKind,
    pub content: String,
    pub created_at: String,
}

/// Task activity log entry (every transition, with actor)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskActivityEntry {
    pub id: i64,
    pub task_id: i64,
    pub actor: Option<String>,
    pub action: String,
    pub detail: Option<serde_json::Value>,
    pub created_at: String,
    /// The card's assignee when this row was written — stamped by the store,
    /// never a caller. `None` for rows older than migration 021, and for a
    /// card that had no assignee.
    #[serde(default)]
    pub assignee: Option<String>,
}

/// One member's acceptance verdicts, for one label or overall.
///
/// Read by the router when it picks a member for a card (P25 decision 4) and
/// when it adjusts capability tags at verdict time (decision 15). Attributed to
/// the member assigned when each verdict was judged, not the card's current
/// assignee — see migration 021.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerdictTally {
    pub member_id: String,
    /// `None` is the member's total across every card, labelled or not; a card
    /// with several labels counts once toward each of them.
    pub label: Option<String>,
    pub passed: u64,
    pub failed: u64,
}

/// One chunk waiting for a vector under some model: exactly what the
/// embedding queue knows, and nothing more.
///
/// The queue used to hand back a whole [`MemoryChunk`] with nine fields filled
/// by invention (`char_end: 0`, `created_at: ""`, …) — values a caller could
/// not tell from real ones. A narrow type makes the absence visible.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueuedChunk {
    pub id: i64,
    pub memory_id: String,
    pub ordinal: i64,
    pub content: String,
}

/// One embedded slice of a memory's content.
///
/// A memory's content is unbounded; an embedding model's input window is not.
/// Embedding a long row whole meant everything past the window was truncated
/// and the vector described only a prefix — silently, since nothing errors.
/// Chunks are cut to the model's window and embedded individually, so the whole
/// content is represented and a long memory is still findable by its tail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryChunk {
    pub id: i64,
    /// Parent `memories.memory_id`.
    pub memory_id: String,
    /// Position within the parent, 0-based. Stable across re-embeds so a
    /// pagination cursor survives one.
    pub ordinal: i64,
    pub content: String,
    /// Char range within the parent's content — lets a page be addressed
    /// without re-chunking.
    pub char_start: i64,
    pub char_end: i64,
    /// `None` while queued for backfill: the text exists, the vector does not.
    pub embedding: Option<Vec<f32>>,
    /// Which model produced `embedding`. Vectors from different models are not
    /// comparable, so this is what makes a model switch a resumable backfill
    /// rather than a full rebuild.
    pub embedding_model: Option<String>,
    /// The budget this chunk was cut to, and the chunker that cut it. Together
    /// they make re-chunking detectable: a parent whose chunks were cut under
    /// different parameters can be redone rather than left as a mixed set.
    pub chunk_max_chars: i64,
    pub chunker_version: i64,
    pub workspace_id: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

/// A chunk to write. `embedding` may be `None` — the row is then queued for the
/// backfill pass rather than being wrong.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewMemoryChunk {
    pub memory_id: String,
    pub ordinal: i64,
    pub content: String,
    pub char_start: i64,
    pub char_end: i64,
    pub embedding: Option<Vec<f32>>,
    pub embedding_model: Option<String>,
    pub chunk_max_chars: i64,
    pub chunker_version: i64,
    pub workspace_id: Option<String>,
}

/// One row of the episodic stream (`memory_events`).
///
/// This is the storage shape only — the timeline's *policy* (which kinds
/// exist, what bounds an append must respect) lives in `nanna-timeline`, so
/// this crate stays a schema mapping and nothing more.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEventRow {
    pub id: i64,
    pub event_id: String,
    /// Unix milliseconds. The wall-clock axis every timeline query walks.
    pub ts_unix_ms: i64,
    pub kind: String,
    /// Workspace scope (None = global).
    pub workspace_id: Option<String>,
    pub content: String,
    /// Character length of the content *before* the append-time cap. Greater
    /// than `content.chars().count()` exactly when the episode was truncated.
    pub content_len_chars: i64,
    pub embedding: Option<Vec<f32>>,
    pub embedding_model: Option<String>,
    pub salience: f32,
    /// Ids this event derives from (memories, messages, tool uses).
    pub source_ids: Vec<String>,
    pub created_at: String,
}

/// An episode to append. There is no `Update` counterpart by design: the
/// event log is append-only, so the only write shape is this one.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewMemoryEvent {
    pub event_id: String,
    pub ts_unix_ms: i64,
    pub kind: String,
    pub workspace_id: Option<String>,
    pub content: String,
    pub content_len_chars: i64,
    pub embedding: Option<Vec<f32>>,
    pub embedding_model: Option<String>,
    pub salience: f32,
    pub source_ids: Vec<String>,
}

/// What a board member *is*. The human and every agent are the same entity
/// (P25 decision 3); this is the only place the difference is recorded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemberKind {
    Human,
    Agent,
}

impl MemberKind {
    /// The token stored in `members.kind`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Human => "human",
            Self::Agent => "agent",
        }
    }

    /// Parse a stored token. `None` for anything this version does not know,
    /// so a row written by a newer schema is rejected rather than coerced.
    #[must_use]
    pub fn parse(token: &str) -> Option<Self> {
        match token {
            "human" => Some(Self::Human),
            "agent" => Some(Self::Agent),
            _ => None,
        }
    }
}

/// Who a member belongs to — not what it is. A `Workspace` member is shared by
/// everyone on that board; a `Human` member travels with its owner between
/// workspaces (P25 decisions 12 and 13).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemberOwner {
    Workspace,
    Human,
}

impl MemberOwner {
    /// The token stored in `members.owner_kind`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Workspace => "workspace",
            Self::Human => "human",
        }
    }

    /// Parse a stored token; `None` for an unknown one.
    #[must_use]
    pub fn parse(token: &str) -> Option<Self> {
        match token {
            "workspace" => Some(Self::Workspace),
            "human" => Some(Self::Human),
            _ => None,
        }
    }
}

/// All the board ever shows about a member's availability (P25 decision 3: you
/// see busy, never a queue).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemberStatus {
    Idle,
    Busy,
    Offline,
}

impl MemberStatus {
    /// The token stored in `members.status`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Busy => "busy",
            Self::Offline => "offline",
        }
    }

    /// Parse a stored token; `None` for an unknown one.
    #[must_use]
    pub fn parse(token: &str) -> Option<Self> {
        match token {
            "idle" => Some(Self::Idle),
            "busy" => Some(Self::Busy),
            "offline" => Some(Self::Offline),
            _ => None,
        }
    }
}

/// A board member: the human, an agent, or the per-workspace Task Management
/// Agent. `tasks.assignee` holds one of these ids.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Member {
    pub id: String,
    pub name: String,
    /// A reference (URL, emoji, short token) — never image bytes, see
    /// `MEMBER_AVATAR_MAX_BYTES`.
    pub avatar: Option<String>,
    pub kind: MemberKind,
    pub owner_kind: MemberOwner,
    /// `workspaces.id` for a workspace member, a human `members.id` for a
    /// personal agent, `None` for the install's own human.
    pub owner_id: Option<String>,
    pub status: MemberStatus,
    /// The router's input: model tier, capability tags, tools, skills, cost.
    pub profile: serde_json::Value,
    pub created_at: String,
    pub updated_at: String,
}

/// New member input.
#[derive(Debug, Clone)]
pub struct NewMember {
    pub id: String,
    pub name: String,
    pub avatar: Option<String>,
    pub kind: MemberKind,
    pub owner_kind: MemberOwner,
    pub owner_id: Option<String>,
    pub status: MemberStatus,
    pub profile: serde_json::Value,
}

/// Partial member update; `None` fields are left untouched.
///
/// `id`, `kind` and `owner_kind` are absent on purpose: a member's identity and
/// what it is do not change. Re-creating the member is the honest way to say
/// that, and it leaves the cards pointing at the old id visible instead of
/// silently re-attributed.
#[derive(Debug, Clone, Default)]
pub struct MemberPatch {
    pub name: Option<String>,
    pub avatar: Option<Option<String>>,
    pub owner_id: Option<Option<String>>,
    pub status: Option<MemberStatus>,
    pub profile: Option<serde_json::Value>,
}
