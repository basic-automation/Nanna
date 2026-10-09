//! Database migrations

/// List of migrations to apply in order
pub const MIGRATIONS: &[(&str, &str)] = &[
    ("001_initial", MIGRATION_001),
    ("002_memories", MIGRATION_002),
    ("003_config", MIGRATION_003),
    ("004_workspaces", MIGRATION_004),
    ("005_job_runs", MIGRATION_005),
    ("006_model_stats", MIGRATION_006),
    ("007_tool_stats", MIGRATION_007),
    ("008_workspace_registry", MIGRATION_008),
    ("009_memory_fsrs", MIGRATION_009),
    ("010_checkpoints", MIGRATION_010),
    ("011_tasks", MIGRATION_011),
    ("012_memory_chunks", MIGRATION_012),
    ("013_embedding_buckets", MIGRATION_013),
    ("014_memory_events", MIGRATION_014),
    ("015_model_stats_cache_ttl", MIGRATION_015),
    ("016_request_log_cache_ttl", MIGRATION_016),
    ("017_members", MIGRATION_017),
    ("018_task_deadline", MIGRATION_018),
    ("019_task_thread", MIGRATION_019),
    ("020_task_time_announcements", MIGRATION_020),
    ("021_task_activity_assignee", MIGRATION_021),
];

const MIGRATION_001: &str = r"
-- Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL UNIQUE,
    channel TEXT NOT NULL,
    user_id TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    metadata TEXT -- JSON
);

CREATE INDEX IF NOT EXISTS idx_sessions_channel ON sessions(channel);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);

-- Messages table
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL, -- 'user', 'assistant', 'system', 'tool'
    content TEXT NOT NULL,
    content_type TEXT NOT NULL DEFAULT 'text', -- 'text', 'tool_use', 'tool_result'
    tool_use_id TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    tokens_in INTEGER,
    tokens_out INTEGER,
    metadata TEXT, -- JSON
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at);
";

const MIGRATION_002: &str = r"
-- Vector memories table (for semantic search)
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL UNIQUE,
    content TEXT NOT NULL,
    embedding BLOB, -- f32 vector as bytes
    embedding_model TEXT,
    session_id TEXT, -- optional, for session-specific memories
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    metadata TEXT -- JSON
);

CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);

-- Memory tags for filtering
CREATE TABLE IF NOT EXISTS memory_tags (
    memory_id TEXT NOT NULL,
    tag TEXT NOT NULL,
    PRIMARY KEY (memory_id, tag),
    FOREIGN KEY (memory_id) REFERENCES memories(memory_id)
);

CREATE INDEX IF NOT EXISTS idx_memory_tags_tag ON memory_tags(tag);
";

const MIGRATION_003: &str = r"
-- Key-value config storage
CREATE TABLE IF NOT EXISTS config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Scheduled tasks / cron jobs
CREATE TABLE IF NOT EXISTS cron_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL UNIQUE,
    schedule TEXT NOT NULL, -- cron expression
    task TEXT NOT NULL, -- JSON task definition
    enabled INTEGER NOT NULL DEFAULT 1,
    last_run TEXT,
    next_run TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    metadata TEXT -- JSON
);

CREATE INDEX IF NOT EXISTS idx_cron_next ON cron_jobs(next_run) WHERE enabled = 1;
";

const MIGRATION_004: &str = r"
-- Add workspace support to sessions
-- workspace_id is optional (NULL = global/no workspace)
ALTER TABLE sessions ADD COLUMN workspace_id TEXT;
ALTER TABLE sessions ADD COLUMN name TEXT;

CREATE INDEX IF NOT EXISTS idx_sessions_workspace ON sessions(workspace_id);

-- Workspace memory links (memory can belong to multiple workspaces)
CREATE TABLE IF NOT EXISTS workspace_memories (
    workspace_id TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    PRIMARY KEY (workspace_id, memory_id),
    FOREIGN KEY (memory_id) REFERENCES memories(memory_id)
);

CREATE INDEX IF NOT EXISTS idx_workspace_memories_workspace ON workspace_memories(workspace_id);
";

const MIGRATION_005: &str = r"
-- Job run history for tracking cron executions
CREATE TABLE IF NOT EXISTS job_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    success INTEGER NOT NULL DEFAULT 0,
    output TEXT,
    error TEXT,
    duration_ms INTEGER,
    FOREIGN KEY (job_id) REFERENCES cron_jobs(job_id)
);

CREATE INDEX IF NOT EXISTS idx_job_runs_job ON job_runs(job_id);
CREATE INDEX IF NOT EXISTS idx_job_runs_started ON job_runs(started_at);

-- Add timezone support to cron jobs
ALTER TABLE cron_jobs ADD COLUMN timezone TEXT DEFAULT 'UTC';

-- Add target channel/session for cron results
ALTER TABLE cron_jobs ADD COLUMN target_channel TEXT;
ALTER TABLE cron_jobs ADD COLUMN target_session TEXT;
";

const MIGRATION_006: &str = r"
-- Model performance statistics (aggregated per model)
CREATE TABLE IF NOT EXISTS model_stats (
    model TEXT PRIMARY KEY,
    total_requests INTEGER NOT NULL DEFAULT 0,
    successful_requests INTEGER NOT NULL DEFAULT 0,
    failed_requests INTEGER NOT NULL DEFAULT 0,
    total_input_tokens INTEGER NOT NULL DEFAULT 0,
    total_output_tokens INTEGER NOT NULL DEFAULT 0,
    total_cache_read_tokens INTEGER NOT NULL DEFAULT 0,
    total_cache_creation_tokens INTEGER NOT NULL DEFAULT 0,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    last_success_epoch_ms INTEGER NOT NULL DEFAULT 0,
    last_failure_epoch_ms INTEGER NOT NULL DEFAULT 0,
    tier_successes_simple INTEGER NOT NULL DEFAULT 0,
    tier_successes_medium INTEGER NOT NULL DEFAULT 0,
    tier_successes_complex INTEGER NOT NULL DEFAULT 0,
    tier_failures_simple INTEGER NOT NULL DEFAULT 0,
    tier_failures_medium INTEGER NOT NULL DEFAULT 0,
    tier_failures_complex INTEGER NOT NULL DEFAULT 0,
    escalations INTEGER NOT NULL DEFAULT 0,
    -- Recent latencies/throughput stored as JSON arrays (ring buffer)
    latencies_ms_json TEXT NOT NULL DEFAULT '[]',
    throughput_tps_json TEXT NOT NULL DEFAULT '[]',
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Per-request model observations (detailed log for analysis)
CREATE TABLE IF NOT EXISTS model_request_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model TEXT NOT NULL,
    success INTEGER NOT NULL,
    latency_ms INTEGER NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens INTEGER NOT NULL DEFAULT 0,
    cache_creation_tokens INTEGER NOT NULL DEFAULT 0,
    tier TEXT,
    escalated INTEGER NOT NULL DEFAULT 0,
    session_id TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_model_request_log_model ON model_request_log(model);
CREATE INDEX IF NOT EXISTS idx_model_request_log_created ON model_request_log(created_at);
";

const MIGRATION_007: &str = r"
-- Tool performance statistics (aggregated per tool)
CREATE TABLE IF NOT EXISTS tool_stats (
    tool_name TEXT PRIMARY KEY,
    call_count INTEGER NOT NULL DEFAULT 0,
    success_count INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0,
    total_duration_ms INTEGER NOT NULL DEFAULT 0,
    last_called_epoch_ms INTEGER NOT NULL DEFAULT 0,
    -- Recent latencies stored as JSON array (ring buffer)
    latencies_ms_json TEXT NOT NULL DEFAULT '[]',
    -- Recent output sizes stored as JSON array (ring buffer)
    output_sizes_json TEXT NOT NULL DEFAULT '[]',
    -- Common errors as JSON: [{ message, count }]
    errors_json TEXT NOT NULL DEFAULT '[]',
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Per-invocation tool call log (time-series data for graphs)
CREATE TABLE IF NOT EXISTS tool_call_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tool_name TEXT NOT NULL,
    success INTEGER NOT NULL,
    duration_ms INTEGER NOT NULL,
    output_size INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    session_id TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_tool_call_log_tool ON tool_call_log(tool_name);
CREATE INDEX IF NOT EXISTS idx_tool_call_log_created ON tool_call_log(created_at);
CREATE INDEX IF NOT EXISTS idx_tool_call_log_tool_created ON tool_call_log(tool_name, created_at);

-- Hourly aggregated tool stats (for dashboard graphs over time)
CREATE TABLE IF NOT EXISTS tool_stats_hourly (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tool_name TEXT NOT NULL,
    hour TEXT NOT NULL,  -- ISO hour: '2026-03-11T15:00:00'
    call_count INTEGER NOT NULL DEFAULT 0,
    success_count INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0,
    total_duration_ms INTEGER NOT NULL DEFAULT 0,
    avg_duration_ms INTEGER NOT NULL DEFAULT 0,
    p95_duration_ms INTEGER NOT NULL DEFAULT 0,
    UNIQUE(tool_name, hour)
);

CREATE INDEX IF NOT EXISTS idx_tool_stats_hourly_hour ON tool_stats_hourly(hour);
CREATE INDEX IF NOT EXISTS idx_tool_stats_hourly_tool ON tool_stats_hourly(tool_name, hour);

-- Daily aggregated tool stats (for longer-term trends)
CREATE TABLE IF NOT EXISTS tool_stats_daily (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tool_name TEXT NOT NULL,
    day TEXT NOT NULL,  -- ISO date: '2026-03-11'
    call_count INTEGER NOT NULL DEFAULT 0,
    success_count INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0,
    total_duration_ms INTEGER NOT NULL DEFAULT 0,
    avg_duration_ms INTEGER NOT NULL DEFAULT 0,
    p95_duration_ms INTEGER NOT NULL DEFAULT 0,
    UNIQUE(tool_name, day)
);

CREATE INDEX IF NOT EXISTS idx_tool_stats_daily_day ON tool_stats_daily(day);
CREATE INDEX IF NOT EXISTS idx_tool_stats_daily_tool ON tool_stats_daily(tool_name, day);
";

const MIGRATION_008: &str = r"
-- Workspace registry: persists registered workspaces across restarts
CREATE TABLE IF NOT EXISTS workspaces (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    path TEXT NOT NULL UNIQUE,
    active INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_accessed TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_workspaces_path ON workspaces(path);
";

const MIGRATION_009: &str = r"
-- Add FSRS cognitive state columns and workspace scope to memories
ALTER TABLE memories ADD COLUMN workspace_id TEXT;
ALTER TABLE memories ADD COLUMN fsrs_stability REAL NOT NULL DEFAULT 1.0;
ALTER TABLE memories ADD COLUMN fsrs_difficulty REAL NOT NULL DEFAULT 5.0;
ALTER TABLE memories ADD COLUMN fsrs_last_access INTEGER NOT NULL DEFAULT 0;
ALTER TABLE memories ADD COLUMN fsrs_access_count INTEGER NOT NULL DEFAULT 0;
ALTER TABLE memories ADD COLUMN fsrs_importance REAL NOT NULL DEFAULT 1.0;
ALTER TABLE memories ADD COLUMN fsrs_storage_strength REAL NOT NULL DEFAULT 0.1;
ALTER TABLE memories ADD COLUMN fsrs_generation INTEGER NOT NULL DEFAULT 0;
CREATE INDEX IF NOT EXISTS idx_memories_workspace ON memories(workspace_id);
";

const MIGRATION_010: &str = r"
-- Checkpoints for crash recovery (replaces checkpoint-{id}.json files)
CREATE TABLE IF NOT EXISTS checkpoints (
    session_id TEXT PRIMARY KEY,
    data TEXT NOT NULL,  -- JSON checkpoint payload
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
";

const MIGRATION_011: &str = r"
-- Agent-grade task store (P15): hierarchy, dependencies, acceptance criteria.
-- No triggers (the migration runner splits statements on semicolons) --
-- parent auto-completion, dependency cycle checks, and the activity log are
-- enforced in TaskRepository.
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_id INTEGER,
    scope TEXT NOT NULL DEFAULT 'session',
    scope_id TEXT,
    project TEXT,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    priority INTEGER NOT NULL DEFAULT 3,
    labels TEXT NOT NULL DEFAULT '[]',
    tool_scope TEXT NOT NULL DEFAULT '[]',
    due_at TEXT,
    recurrence TEXT,
    depends_on TEXT NOT NULL DEFAULT '[]',
    acceptance TEXT,
    assignee TEXT,
    sort_order INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_tasks_scope ON tasks(scope, scope_id);
CREATE INDEX IF NOT EXISTS idx_tasks_parent ON tasks(parent_id);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);

-- Append-only working notes: where a sub-agent leaves findings for its parent.
CREATE TABLE IF NOT EXISTS task_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER NOT NULL,
    author TEXT,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_task_notes_task ON task_notes(task_id);

-- Activity log: every transition with actor + timestamp (drift post-mortems).
CREATE TABLE IF NOT EXISTS task_activity (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER NOT NULL,
    actor TEXT,
    action TEXT NOT NULL,
    detail TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_task_activity_task ON task_activity(task_id);
";

/// Per-chunk vectors, with the memory row as parent.
///
/// A memory's content is unbounded, but an embedding model's input window is
/// not. Embedding a row whole meant everything past the window was truncated
/// away and the resulting vector described only a prefix while claiming to
/// describe the row. Chunks are sized to the model's window and embedded
/// individually, so the whole content is represented.
///
/// Additive only. `memories` is untouched on purpose: the corruption-recovery
/// salvage path reads a fixed column list positionally out of a quarantined
/// file, so widening that table would break rebuilds of databases written
/// before the change. Existing rows simply have no chunks and keep matching on
/// `memories.embedding` until the backfill reaches them.
///
/// No ALTER, and no semicolon inside a comment -- migrations are split on
/// `;` with no comment awareness, run outside a transaction, and are recorded
/// only after every statement succeeds, so a statement that cannot apply is
/// retried on every boot forever.
const MIGRATION_012: &str = r"
CREATE TABLE IF NOT EXISTS memory_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    content TEXT NOT NULL,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    embedding BLOB,
    embedding_model TEXT,
    chunk_max_chars INTEGER NOT NULL,
    chunker_version INTEGER NOT NULL,
    workspace_id TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- One row per (parent, ordinal). Rewrites replace the whole set for a parent.
CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_chunks_parent
    ON memory_chunks(memory_id, ordinal);

-- Scoped search filters inline rather than joining, so the workspace is
-- denormalized onto the chunk.
CREATE INDEX IF NOT EXISTS idx_memory_chunks_workspace
    ON memory_chunks(workspace_id);

-- Finds chunks whose vector came from a model that is no longer active, which
-- is what makes an embedding-model switch a resumable backfill instead of a
-- full rebuild.
CREATE INDEX IF NOT EXISTS idx_memory_chunks_model
    ON memory_chunks(embedding_model);

-- Drives the backfill queue: chunks whose text exists but whose vector does
-- not yet.
CREATE INDEX IF NOT EXISTS idx_memory_chunks_pending
    ON memory_chunks(embedding_model, id);
";

const MIGRATION_013: &str = r"
-- One row per (memory, model). This is the bucket: a model's vectors live
-- beside every other model's rather than overwriting them, so returning to a
-- provider after an outage costs a lookup instead of a full re-embed.
CREATE TABLE IF NOT EXISTS memory_vectors (
    memory_id TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding BLOB NOT NULL,
    dim INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (memory_id, embedding_model)
);

-- Same, per chunk. Chunk re-embedding used to overwrite in place, destroying
-- the previous model's vector, which made a provider flap cost a full rebuild
-- of the chunk index.
CREATE TABLE IF NOT EXISTS memory_chunk_vectors (
    memory_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding BLOB NOT NULL,
    dim INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (memory_id, ordinal, embedding_model)
);

-- Durable work queue, keyed the same way the buckets are.
--
-- A derived queue cannot express this. The old one tested embedding_model
-- against the active model using a single column per row, and one column can
-- record only ONE model -- so it could not represent a chunk that has an
-- ollama vector but not an open-router one, which is the normal state once
-- buckets are real.
--
-- attempts and last_error exist so a provider answering 402 shows up as a
-- stuck queue with a reason attached, rather than as silence. That silence is
-- what let 2167 memories sit unembedded for a day.
CREATE TABLE IF NOT EXISTS embedding_queue (
    memory_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    embedding_model TEXT NOT NULL,
    enqueued_at TEXT NOT NULL DEFAULT (datetime('now')),
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    last_attempt_at TEXT,
    PRIMARY KEY (memory_id, ordinal, embedding_model)
);

-- Bucket selection counts vectors per model, and the backfill scans by model.
CREATE INDEX IF NOT EXISTS idx_memory_vectors_model
    ON memory_vectors(embedding_model);
CREATE INDEX IF NOT EXISTS idx_memory_chunk_vectors_model
    ON memory_chunk_vectors(embedding_model);

-- Drives the drain: oldest work first, per model.
CREATE INDEX IF NOT EXISTS idx_embedding_queue_model
    ON embedding_queue(embedding_model, attempts, enqueued_at);

-- Cascade deletes are manual everywhere in this schema because nothing sets
-- PRAGMA foreign_keys, so these carry no REFERENCES clause that would read as
-- enforcement it does not provide.
CREATE INDEX IF NOT EXISTS idx_memory_vectors_parent
    ON memory_vectors(memory_id);
CREATE INDEX IF NOT EXISTS idx_memory_chunk_vectors_parent
    ON memory_chunk_vectors(memory_id);
CREATE INDEX IF NOT EXISTS idx_embedding_queue_parent
    ON embedding_queue(memory_id);
";

const MIGRATION_014: &str = r"
-- The episodic stream: what happened, on a wall-clock axis.
--
-- `memories` is the SEMANTIC layer (facts, FSRS-weighted, never expiring).
-- This table is the RAW layer underneath it -- messages, tool calls, recalls
-- and outcomes as they occur -- which dreaming later consolidates *into*
-- facts. The two are deliberately separate: a fact has no single timestamp
-- (it is the residue of many episodes), and an episode has no FSRS state
-- (it is not independently recalled). Collapsing them would force one of
-- those two to lie.
--
-- Append-only by construction: there is no updated_at, and the repository
-- exposes no UPDATE or DELETE for a single row. Rewriting history is what
-- makes a timeline unable to answer 'what did I know, when'.
CREATE TABLE IF NOT EXISTS memory_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE,

    -- Unix milliseconds, not a datetime string. Every consumer of this table
    -- is arithmetic (resample into a series, decimate a window, detect a
    -- peak), and text timestamps would need parsing on every sample.
    ts_unix_ms INTEGER NOT NULL,

    kind TEXT NOT NULL,           -- 'message' | 'tool_call' | 'recall' | 'outcome'
    workspace_id TEXT,            -- NULL = global scope
    content TEXT NOT NULL,

    -- Length of the content BEFORE the append-time cap was applied. Equal to
    -- length(content) when nothing was dropped, so a truncated episode is
    -- detectable by comparison rather than by a flag nobody sets.
    content_len_chars INTEGER NOT NULL,

    embedding BLOB,               -- f32 little-endian, NULL until embedded
    embedding_model TEXT,

    -- [0.0, 1.0]. Drives which episodes survive DSP decimation of older
    -- windows and which get promoted to facts.
    salience REAL NOT NULL,

    -- JSON array of the ids this event derives from (memory ids, message ids,
    -- tool_use ids). Lineage, so a consolidated fact can be traced back.
    source_ids TEXT NOT NULL DEFAULT '[]',

    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- The wall-clock axis itself: every range scan and every resample walks it.
CREATE INDEX IF NOT EXISTS idx_memory_events_ts
    ON memory_events(ts_unix_ms);

-- Scoped ranges filter inline rather than joining, so the workspace is
-- denormalized onto the event exactly as it is onto memory_chunks.
CREATE INDEX IF NOT EXISTS idx_memory_events_workspace_ts
    ON memory_events(workspace_id, ts_unix_ms);

-- Per-signal resampling reads one kind at a time (salience(t) over tool
-- calls is a different series from salience(t) over messages).
CREATE INDEX IF NOT EXISTS idx_memory_events_kind_ts
    ON memory_events(kind, ts_unix_ms);

-- Finds events whose text exists but whose vector does not yet, which is what
-- makes embedding the backlog a resumable drain instead of a full rescan.
CREATE INDEX IF NOT EXISTS idx_memory_events_pending
    ON memory_events(embedding_model, id);
";

const MIGRATION_015: &str = r"
-- The 1-hour share of total_cache_creation_tokens. Anthropic bills a 1-hour
-- cache write at 2x input and a 5-minute one at 1.25x, so one write total
-- cannot be priced once [llm] prompt_cache_ttl can be 1h. A subset of
-- total_cache_creation_tokens, never added to it.
ALTER TABLE model_stats ADD COLUMN total_cache_creation_1h_tokens INTEGER NOT NULL DEFAULT 0;
";

const MIGRATION_016: &str = r"
-- The same 1-hour cache-write share, per request, so a day's spend can be
-- priced from the request log exactly as the lifetime totals are.
ALTER TABLE model_request_log ADD COLUMN cache_creation_1h_tokens INTEGER NOT NULL DEFAULT 0;
";

const MIGRATION_017: &str = r#"
-- P25 Stage 1: board members. The human, every agent and the Task Management
-- Agent share ONE entity, so a card's assignee is a member id regardless of
-- who is behind it and the board never has to branch on human-vs-agent to
-- render a row.
--
-- `owner_kind` records who the member belongs to, not what it is: a workspace
-- member is shared by everyone on that board, a human-owned agent travels with
-- its owner. `profile` is the router's input (model tier, capability tags,
-- tools, skills, cost) and is JSON because that shape is still being designed.
-- The columns above it are the ones the board and the assignment path read.
CREATE TABLE IF NOT EXISTS members (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    avatar TEXT,
    kind TEXT NOT NULL,
    owner_kind TEXT NOT NULL,
    owner_id TEXT,
    status TEXT NOT NULL DEFAULT 'idle',
    profile TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_members_owner ON members(owner_kind, owner_id);
CREATE INDEX IF NOT EXISTS idx_members_kind ON members(kind);

-- `tasks.assignee` now holds a `members.id`. The column is not re-declared as
-- a SQL foreign key: SQLite cannot add a constraint to an existing column, and
-- `PRAGMA foreign_keys` is off here anyway, so the reference would be decorative.
-- `TaskRepository` enforces it on write instead, which also covers the rows
-- this migration inherits. The index is what the board's assignee filter and
-- the per-member inbox read.
CREATE INDEX IF NOT EXISTS idx_tasks_assignee ON tasks(assignee);

-- The one human this install belongs to. Seeded rather than created on first
-- use so that `assignee = 'human'` resolves from the moment the schema exists.
INSERT OR IGNORE INTO members (id, name, kind, owner_kind, owner_id, status, profile)
VALUES ('human', 'You', 'human', 'human', NULL, 'idle', '{}');

-- One Task Management Agent per board, including the global one. It takes no
-- work of its own. It reads a card and decides who does.
INSERT OR IGNORE INTO members (id, name, kind, owner_kind, owner_id, status, profile)
VALUES ('router:global', 'Task Router', 'agent', 'workspace', NULL, 'idle', '{"role":"router"}');

INSERT OR IGNORE INTO members (id, name, kind, owner_kind, owner_id, status, profile)
SELECT 'router:' || workspaces.id, 'Task Router', 'agent', 'workspace', workspaces.id, 'idle', '{"role":"router"}'
FROM workspaces;
"#;

const MIGRATION_018: &str = r"
-- P25 decision 10: a date and a deadline are different things, and conflating
-- them is why a card that cannot start yet looks identical to one that is
-- late. `due_at` keeps its column but changes meaning -- it is now the DEFER
-- date, which keeps a card out of the inbox until it arrives. `deadline_at` is
-- the bound: the card must be complete by then, and `overdue` is measured
-- against it alone.
--
-- Existing rows are left as they are. A `due_at` written under the old meaning
-- reads as a defer date, which is the conservative reading: it hides the card
-- until its date rather than declaring it late.
ALTER TABLE tasks ADD COLUMN deadline_at TEXT;

CREATE INDEX IF NOT EXISTS idx_tasks_deadline ON tasks(deadline_at);
";

const MIGRATION_019: &str = r"
-- P25 decision 2: a conversation IS the card's thread. `task_notes` was an
-- agent scratchpad with a free-text `author` -- it becomes the forum, so a
-- post needs to say who made it (a board member) and what kind of post it is.
-- Distinguishing the four kinds is what lets the router read a thread without
-- re-deriving intent from prose: a `question` blocks, a `verdict` decides, a
-- `progress` line is noise to everyone but the board.
--
-- `author` is kept, unchanged, holding the pre-members actor string ('gui',
-- 'harness', an agent name). It is legacy that Stage 4 removes once every
-- writer names a member. `author_member_id` is the P25 field and is NULL on
-- every inherited row, which reads correctly: those posts predate members.
--
-- Nothing here makes a note mutable. The thread is the permanent record, so
-- there is no UPDATE path and this migration does not add one.
ALTER TABLE task_notes ADD COLUMN author_member_id TEXT;

ALTER TABLE task_notes ADD COLUMN kind TEXT NOT NULL DEFAULT 'comment';

CREATE INDEX IF NOT EXISTS idx_task_notes_author ON task_notes(author_member_id);
";

const MIGRATION_020: &str = r"
-- P25 Stage 1: `due` and `overdue` are TIME-driven lifecycle events, so unlike
-- every other kind they have no write to hang off -- a sweep notices them.
-- A sweep that re-derives 'this card is overdue' every pass would re-announce
-- the same card every 5 minutes forever, which is a notification bug rather
-- than an event stream. These two columns are the crossing marker: set when the
-- sweep announces, so the announcement happens once.
--
-- They are deliberately NOT part of `NewTask` or `TaskPatch`. Nobody outside
-- the store sets them -- they are the store's memory of what it has already
-- said, and letting a caller write one would let a caller suppress a
-- notification.
--
-- Re-arming is the half that is easy to get wrong, so it is explicit: moving
-- `due_at` clears `due_announced_at`, moving `deadline_at` clears
-- `overdue_announced_at`, and reopening a card clears BOTH -- a recurring card
-- comes back around and must be able to fall due again.
ALTER TABLE tasks ADD COLUMN due_announced_at TEXT;

ALTER TABLE tasks ADD COLUMN overdue_announced_at TEXT;
";

const MIGRATION_021: &str = r"
-- P25 Stage 1: the router reads each member's verdict history per label, and a
-- verdict belongs to the member who was assigned WHEN it was judged. The
-- current assignee is the wrong answer exactly when it matters: a failed
-- verdict sends the card back to the router, which may hand it to someone
-- else, and reading the assignee at query time would charge the first
-- member's failure to the second.
--
-- So every activity row carries the card's assignee at the moment it was
-- written. The store stamps it inside the INSERT itself, never a caller, so no
-- writer can forget it or name the wrong member. Rows written before this
-- migration are NULL, which reads correctly: their attribution is unknown, and
-- the rollup leaves them out rather than guessing.
ALTER TABLE task_activity ADD COLUMN assignee TEXT;

CREATE INDEX IF NOT EXISTS idx_task_activity_action ON task_activity(action, assignee);
";

/// Lexer state while splitting a migration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Lex {
    Code,
    /// Inside `'…'`, `"…"` or `` `…` ``, closed by the same character.
    Quoted(char),
    LineComment,
    BlockComment,
}

/// Split a migration into the statements `Storage::migrate` executes, one
/// `conn.execute` each.
///
/// A small SQL lexer rather than `sql.split(';')`, which does not know what a
/// comment is: a `;` inside a `--` comment used to cut the statement around
/// it in half (migration 014 nearly shipped that). Here a `;` ends a statement
/// only outside `--` and `/* */` comments and outside `'…'` strings and
/// `"…"`/`` `…` `` identifiers. Comments are dropped — the database never
/// needed them — so a note after the last `;` is no longer a comment-only
/// "statement" that fails the whole migration. SQL escapes a quote by
/// doubling it (`'it''s'`), which falls out of leaving and re-entering the
/// string. An unterminated string or comment is passed through for the
/// database to reject, never silently repaired.
pub fn split_statements(sql: &str) -> Vec<String> {
    let mut statements = Vec::new();
    let mut current = String::with_capacity(sql.len());
    let mut lex = Lex::Code;
    let mut chars = sql.chars().peekable();
    while let Some(ch) = chars.next() {
        match lex {
            Lex::Code => match ch {
                '-' if chars.peek() == Some(&'-') => {
                    chars.next();
                    lex = Lex::LineComment;
                }
                '/' if chars.peek() == Some(&'*') => {
                    chars.next();
                    // A space, so `a/* … */b` cannot glue two tokens.
                    current.push(' ');
                    lex = Lex::BlockComment;
                }
                '\'' | '"' | '`' => {
                    current.push(ch);
                    lex = Lex::Quoted(ch);
                }
                ';' => push_statement(&mut statements, &mut current),
                _ => current.push(ch),
            },
            Lex::Quoted(quote) => {
                current.push(ch);
                if ch == quote {
                    lex = Lex::Code;
                }
            }
            Lex::LineComment => {
                if ch == '\n' {
                    current.push('\n');
                    lex = Lex::Code;
                }
            }
            Lex::BlockComment => {
                if ch == '*' && chars.peek() == Some(&'/') {
                    chars.next();
                    lex = Lex::Code;
                }
            }
        }
    }
    push_statement(&mut statements, &mut current);
    debug_assert!(
        statements.iter().all(|s| !s.trim().is_empty()),
        "a blank or comment-only chunk is never executed"
    );
    debug_assert!(
        statements.len() <= sql.matches(';').count() + 1,
        "a statement ends only at a ';' or at the end of the migration"
    );
    statements
}

/// Close the statement being collected, keeping it only if it holds SQL.
fn push_statement(statements: &mut Vec<String>, current: &mut String) {
    let statement = current.trim();
    if !statement.is_empty() {
        statements.push(statement.to_string());
    }
    current.clear();
}

/// `(table, column)` when `statement` is `ALTER TABLE <table> ADD [COLUMN]
/// <column> …`, else `None`.
///
/// The one DDL form SQLite cannot make idempotent in the statement itself —
/// there is no `ADD COLUMN IF NOT EXISTS` — so a database that applied half a
/// migration before migrations ran in a transaction fails every later boot
/// with `duplicate column name`. `Storage::migrate` asks this, then probes the
/// live schema and skips an ADD whose column is already there. Identifiers are
/// matched case-insensitively and may be quoted with `"`, `` ` `` or `[]`.
#[must_use]
pub fn add_column_target(statement: &str) -> Option<(String, String)> {
    let mut words = statement.split_whitespace();
    let mut keyword = |expected: &str| {
        words
            .next()
            .is_some_and(|w| w.eq_ignore_ascii_case(expected))
    };
    if !(keyword("ALTER") && keyword("TABLE")) {
        return None;
    }
    let table = unquote(words.next()?);
    if !words.next()?.eq_ignore_ascii_case("ADD") {
        return None;
    }
    let mut column = words.next()?;
    if column.eq_ignore_ascii_case("COLUMN") {
        column = words.next()?;
    }
    let column = unquote(column);
    debug_assert!(
        !table.is_empty() && !column.is_empty(),
        "both names are words"
    );
    Some((table, column))
}

/// An identifier without its SQL quoting.
fn unquote(word: &str) -> String {
    word.trim_matches(|c| matches!(c, '"' | '`' | '[' | ']'))
        .to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use super::{MIGRATIONS, add_column_target, split_statements};

    #[test]
    fn add_column_targets_are_recognised_and_nothing_else_is() {
        let target = |t: &str, c: &str| Some((t.to_string(), c.to_string()));
        assert_eq!(
            add_column_target("ALTER TABLE tasks ADD COLUMN deadline_at TEXT"),
            target("tasks", "deadline_at")
        );
        assert_eq!(
            add_column_target("alter  table \"Tasks\"\n  add `Due` TEXT DEFAULT 'x'"),
            target("tasks", "due"),
            "case, quoting, COLUMN omitted and line breaks"
        );
        assert_eq!(add_column_target("ALTER TABLE tasks RENAME TO cards"), None);
        assert_eq!(add_column_target("CREATE TABLE t (x TEXT)"), None);
        assert_eq!(add_column_target("ALTER TABLE"), None);
    }

    /// Every ADD COLUMN a shipped migration runs must be one the probe can
    /// see, or a half-applied database would still wedge on it.
    #[test]
    fn every_shipped_add_column_is_probeable() {
        for (name, sql) in MIGRATIONS {
            for statement in split_statements(sql) {
                let lower = statement.to_ascii_lowercase();
                if lower.starts_with("alter table") && lower.contains(" add ") {
                    assert!(
                        add_column_target(&statement).is_some(),
                        "{name}: `{statement}` is an ADD the probe cannot parse"
                    );
                }
            }
        }
    }

    /// Strip `--` line comments the way a reader does, so what is left is the
    /// SQL the database would actually see.
    fn sql_only(chunk: &str) -> String {
        chunk
            .lines()
            .map(|line| line.split_once("--").map_or(line, |(code, _)| code))
            .collect::<Vec<_>>()
            .join("\n")
            .trim()
            .to_string()
    }

    /// SQL with comments removed and whitespace collapsed: what two splits
    /// must agree on for the database to see the same statement.
    fn normalized(chunk: &str) -> String {
        sql_only(chunk)
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// The runner used to execute `sql.split(';')`, and every shipped
    /// migration was written (and test-guarded) against that. Moving to a real
    /// lexer must not change what a single one of them executes — every
    /// existing install has applied them, but a fresh database runs them all.
    #[test]
    fn every_shipped_migration_splits_exactly_as_it_did_before() {
        for (name, sql) in MIGRATIONS {
            let before: Vec<String> = sql
                .split(';')
                .map(normalized)
                .filter(|s| !s.is_empty())
                .collect();
            let after: Vec<String> = split_statements(sql)
                .iter()
                .map(|s| normalized(s))
                .collect();
            assert!(!after.is_empty(), "{name} executes nothing");
            assert_eq!(after, before, "{name} now splits differently");
        }
    }

    /// The traps the old `split(';')` fell into, and the ones a naive
    /// comment-stripper would: a `;` in a comment (migration 014's near miss),
    /// a `;` or `--` inside a string, a doubled-quote escape, a block comment
    /// between statements, and a note after the last `;`.
    const TRICKY: &str = "CREATE TABLE t (a BLOB, -- f32 little-endian; NULL until embedded\n\
                          b TEXT DEFAULT 'x;y');\n\
                          INSERT INTO t (b) VALUES ('it''s; fine -- not a comment');\n\
                          /* a block; comment */ CREATE INDEX i ON t(b);\n\
                          -- a trailing note; never executed\n";

    #[test]
    fn a_semicolon_in_a_comment_or_a_string_does_not_end_a_statement() {
        let statements = split_statements(TRICKY);
        assert_eq!(statements.len(), 3, "{statements:#?}");
        assert!(
            statements[0].starts_with("CREATE TABLE t (a BLOB,")
                && statements[0].ends_with("b TEXT DEFAULT 'x;y')"),
            "{}",
            statements[0]
        );
        assert!(!statements[0].contains("f32"), "the comment is dropped");
        assert_eq!(
            statements[1],
            "INSERT INTO t (b) VALUES ('it''s; fine -- not a comment')"
        );
        assert_eq!(statements[2], "CREATE INDEX i ON t(b)");
    }

    #[test]
    fn blank_and_comment_only_input_executes_nothing() {
        let none: Vec<String> = Vec::new();
        assert_eq!(split_statements(""), none);
        assert_eq!(
            split_statements(" ;\n; -- just a note; really\n/* and; this */"),
            none
        );
        assert_eq!(split_statements("SELECT 1"), vec!["SELECT 1"]);
    }

    /// Names are the applied-migration key, so a duplicate would silently skip
    /// the second one forever.
    #[test]
    fn migration_names_are_unique_and_ordered() {
        let names: Vec<&str> = MIGRATIONS.iter().map(|(n, _)| *n).collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len(), "duplicate migration name");
        let mut in_order = names.clone();
        in_order.sort_unstable();
        assert_eq!(
            names, in_order,
            "migrations must be listed in applied order"
        );
    }

    /// The split statements are what the database accepts: every trap above,
    /// executed on a real turso connection, and the escaped value read back.
    #[tokio::test]
    async fn the_tricky_statements_run_on_a_real_database() {
        // The `Database` is dropped once connected, as `Storage::new` does.
        let conn = turso::Builder::new_local(":memory:")
            .build()
            .await
            .expect("open an in-memory database")
            .connect()
            .expect("connect");
        for statement in split_statements(TRICKY) {
            conn.execute(&statement, ()).await.expect(&statement);
        }
        let mut rows = conn
            .query("SELECT b FROM t", ())
            .await
            .expect("query the table");
        let row = rows.next().await.expect("read a row").expect("one row");
        let value: String = row.get(0).expect("a text value");
        assert_eq!(value, "it's; fine -- not a comment");
        assert!(
            rows.next().await.expect("read").is_none(),
            "exactly one row"
        );
    }

    /// A fresh database runs every shipped migration through the lexer: all of
    /// them must apply, each exactly once.
    #[tokio::test]
    async fn a_fresh_database_applies_every_migration_once() {
        let storage = crate::Storage::in_memory()
            .await
            .expect("every migration applies to a fresh database");
        let conn = storage.conn().lock().await;
        let mut rows = conn
            .query("SELECT COUNT(*), COUNT(DISTINCT name) FROM _migrations", ())
            .await
            .expect("query the ledger");
        drop(conn);
        let row = rows.next().await.expect("read a row").expect("one row");
        let applied: i64 = row.get(0).expect("a count");
        let distinct: i64 = row.get(1).expect("a count");
        assert_eq!(usize::try_from(applied).ok(), Some(MIGRATIONS.len()));
        assert_eq!(applied, distinct, "no migration recorded twice");
    }
}
