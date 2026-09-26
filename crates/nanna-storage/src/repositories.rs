//! Repository implementations using Turso

use crate::{
    CronJob, JobRun, Memory, MemoryChunk, MemoryEventRow, MemoryFsrsUpdate, Message, NewCronJob,
    NewJobRun, NewMemory, NewMemoryChunk, NewMemoryEvent, NewMessage, QueuedChunk, Session,
    StorageError, WorkspaceRecord,
};
use std::sync::Arc;
use tokio::sync::Mutex;
use turso::Connection;

/// Session repository
pub struct SessionRepository {
    conn: Arc<Mutex<Connection>>,
}

impl SessionRepository {
    pub const fn new(conn: Arc<Mutex<Connection>>) -> Self {
        Self { conn }
    }

    /// Create a session, or bump `updated_at` if `session_id` already exists.
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the upsert or the read-back fails,
    /// or [`StorageError::NotFound`] if the row cannot be read back.
    pub async fn create(
        &self,
        session_id: &str,
        channel: &str,
        user_id: Option<&str>,
    ) -> Result<Session, StorageError> {
        self.create_with_workspace(session_id, channel, user_id, None, None).await
    }

    /// Create a session in a workspace, or bump `updated_at` if `session_id`
    /// already exists (the existing row's other columns are kept).
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the upsert or the read-back fails,
    /// or [`StorageError::NotFound`] if the row cannot be read back.
    pub async fn create_with_workspace(
        &self,
        session_id: &str,
        channel: &str,
        user_id: Option<&str>,
        workspace_id: Option<&str>,
        name: Option<&str>,
    ) -> Result<Session, StorageError> {
        let conn = self.conn.lock().await;

        conn.execute(
            "INSERT INTO sessions (session_id, channel, user_id, workspace_id, name) VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(session_id) DO UPDATE SET updated_at = datetime('now')",
            turso::params![session_id, channel, user_id, workspace_id, name],
        )
        .await?;

        drop(conn);
        self.get(session_id).await
    }

    /// Fetch one session by id.
    ///
    /// # Errors
    /// Returns [`StorageError::NotFound`] if no session has `session_id`, or
    /// [`StorageError::Database`] if the query fails or a column does not
    /// decode. Unparseable metadata loads as `None`.
    pub async fn get(&self, session_id: &str) -> Result<Session, StorageError> {
        let conn = self.conn.lock().await;

        let mut rows = conn
            .query(
                "SELECT id, session_id, channel, user_id, created_at, updated_at, metadata, workspace_id, name 
                 FROM sessions WHERE session_id = ?1",
                turso::params![session_id],
            )
            .await?;

        let session = if let Some(row) = rows.next().await? {
            let metadata_str: Option<String> = row.get(6)?;
            Ok(Session {
                id: row.get(0)?,
                session_id: row.get(1)?,
                channel: row.get(2)?,
                user_id: row.get(3)?,
                created_at: row.get(4)?,
                updated_at: row.get(5)?,
                metadata: metadata_str.and_then(|s| serde_json::from_str(&s).ok()),
                workspace_id: row.get(7)?,
                name: row.get(8)?,
            })
        } else {
            Err(StorageError::NotFound(format!("Session: {session_id}")))
        };
        // Held until the cursor is gone: an open `Rows` on the shared
        // connection swallows later writes.
        drop(rows);
        drop(conn);
        session
    }

    /// The `limit` most recently updated sessions, newest first.
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the query fails or a column does
    /// not decode. Unparseable metadata loads as `None`.
    pub async fn list_recent(&self, limit: i64) -> Result<Vec<Session>, StorageError> {
        let conn = self.conn.lock().await;

        let mut rows = conn
            .query(
                "SELECT id, session_id, channel, user_id, created_at, updated_at, metadata, workspace_id, name 
                 FROM sessions ORDER BY updated_at DESC LIMIT ?1",
                turso::params![limit],
            )
            .await?;

        let mut sessions = Vec::new();
        while let Some(row) = rows.next().await? {
            let metadata_str: Option<String> = row.get(6)?;
            sessions.push(Session {
                id: row.get(0)?,
                session_id: row.get(1)?,
                channel: row.get(2)?,
                user_id: row.get(3)?,
                created_at: row.get(4)?,
                updated_at: row.get(5)?,
                metadata: metadata_str.and_then(|s| serde_json::from_str(&s).ok()),
                workspace_id: row.get(7)?,
                name: row.get(8)?,
            });
        }
        // Held until the cursor is gone: an open `Rows` on the shared
        // connection swallows later writes.
        drop(rows);
        drop(conn);

        Ok(sessions)
    }

    /// List sessions for a specific workspace (or global if `workspace_id` is None)
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the query fails or a column does
    /// not decode. Unparseable metadata loads as `None`.
    pub async fn list_by_workspace(&self, workspace_id: Option<&str>, limit: i64) -> Result<Vec<Session>, StorageError> {
        let conn = self.conn.lock().await;

        let mut rows = match workspace_id {
            Some(ws_id) => {
                conn.query(
                    "SELECT id, session_id, channel, user_id, created_at, updated_at, metadata, workspace_id, name 
                     FROM sessions WHERE workspace_id = ?1 ORDER BY updated_at DESC LIMIT ?2",
                    turso::params![ws_id, limit],
                ).await?
            }
            None => {
                conn.query(
                    "SELECT id, session_id, channel, user_id, created_at, updated_at, metadata, workspace_id, name 
                     FROM sessions WHERE workspace_id IS NULL ORDER BY updated_at DESC LIMIT ?1",
                    turso::params![limit],
                ).await?
            }
        };

        let mut sessions = Vec::new();
        while let Some(row) = rows.next().await? {
            let metadata_str: Option<String> = row.get(6)?;
            sessions.push(Session {
                id: row.get(0)?,
                session_id: row.get(1)?,
                channel: row.get(2)?,
                user_id: row.get(3)?,
                created_at: row.get(4)?,
                updated_at: row.get(5)?,
                metadata: metadata_str.and_then(|s| serde_json::from_str(&s).ok()),
                workspace_id: row.get(7)?,
                name: row.get(8)?,
            });
        }
        // Held until the cursor is gone: an open `Rows` on the shared
        // connection swallows later writes.
        drop(rows);
        drop(conn);

        Ok(sessions)
    }

    /// Update session name
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the update fails.
    pub async fn update_name(&self, session_id: &str, name: &str) -> Result<(), StorageError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE sessions SET name = ?1, updated_at = datetime('now') WHERE session_id = ?2",
            turso::params![name, session_id],
        ).await?;
        drop(conn);
        Ok(())
    }

    /// Update session workspace
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the update fails.
    pub async fn update_workspace(&self, session_id: &str, workspace_id: Option<&str>) -> Result<(), StorageError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE sessions SET workspace_id = ?1, updated_at = datetime('now') WHERE session_id = ?2",
            turso::params![workspace_id, session_id],
        ).await?;
        drop(conn);
        Ok(())
    }
}

/// Message repository
pub struct MessageRepository {
    conn: Arc<Mutex<Connection>>,
}

impl MessageRepository {
    pub const fn new(conn: Arc<Mutex<Connection>>) -> Self {
        Self { conn }
    }

    /// Insert a message, touch its session, and return the stored row.
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the insert, the session touch, or
    /// the read-back fails, or [`StorageError::NotFound`] if the read-back
    /// finds no row. Statements that succeeded before a failure stay applied.
    pub async fn create(&self, msg: NewMessage) -> Result<Message, StorageError> {
        let conn = self.conn.lock().await;

        let metadata_json = msg.metadata.as_ref().map(std::string::ToString::to_string);

        conn.execute(
            "INSERT INTO messages (session_id, role, content, content_type, tool_use_id, tokens_in, tokens_out, metadata)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            turso::params![
                msg.session_id.as_str(),
                msg.role.as_str(),
                msg.content.as_str(),
                msg.content_type.as_str(),
                msg.tool_use_id.as_deref(),
                msg.tokens_in,
                msg.tokens_out,
                metadata_json.as_deref(),
            ],
        )
        .await?;

        // Update session timestamp
        conn.execute(
            "UPDATE sessions SET updated_at = datetime('now') WHERE session_id = ?1",
            turso::params![msg.session_id.as_str()],
        )
        .await?;

        // Get the inserted message
        let mut rows = conn
            .query(
                "SELECT id, session_id, role, content, content_type, tool_use_id, created_at, tokens_in, tokens_out, metadata
                 FROM messages ORDER BY id DESC LIMIT 1",
                (),
            )
            .await?;

        let message = if let Some(row) = rows.next().await? {
            let metadata_str: Option<String> = row.get(9)?;
            Ok(Message {
                id: row.get(0)?,
                session_id: row.get(1)?,
                role: row.get(2)?,
                content: row.get(3)?,
                content_type: row.get(4)?,
                tool_use_id: row.get(5)?,
                created_at: row.get(6)?,
                tokens_in: row.get(7)?,
                tokens_out: row.get(8)?,
                metadata: metadata_str.and_then(|s| serde_json::from_str(&s).ok()),
            })
        } else {
            Err(StorageError::NotFound("Message just created".to_string()))
        };
        // Held from the insert through the read-back: "newest row" is only
        // this message while no other writer can interleave, and the cursor
        // must be gone before the guard is.
        drop(rows);
        drop(conn);
        message
    }

    /// Up to `limit` messages of a session, oldest first.
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the query fails or a column does
    /// not decode. Unparseable metadata loads as `None`.
    pub async fn get_by_session(
        &self,
        session_id: &str,
        limit: i64,
    ) -> Result<Vec<Message>, StorageError> {
        let conn = self.conn.lock().await;

        let mut rows = conn
            .query(
                "SELECT id, session_id, role, content, content_type, tool_use_id, created_at, tokens_in, tokens_out, metadata
                 FROM messages WHERE session_id = ?1 ORDER BY created_at ASC LIMIT ?2",
                turso::params![session_id, limit],
            )
            .await?;

        let mut messages = Vec::new();
        while let Some(row) = rows.next().await? {
            let metadata_str: Option<String> = row.get(9)?;
            messages.push(Message {
                id: row.get(0)?,
                session_id: row.get(1)?,
                role: row.get(2)?,
                content: row.get(3)?,
                content_type: row.get(4)?,
                tool_use_id: row.get(5)?,
                created_at: row.get(6)?,
                tokens_in: row.get(7)?,
                tokens_out: row.get(8)?,
                metadata: metadata_str.and_then(|s| serde_json::from_str(&s).ok()),
            });
        }
        // Held until the cursor is gone: an open `Rows` on the shared
        // connection swallows later writes.
        drop(rows);
        drop(conn);

        Ok(messages)
    }
}

/// Memory repository (for vector search)
pub struct MemoryRepository {
    conn: Arc<Mutex<Connection>>,
}

/// Helper to decode FSRS columns from row index 9..16
/// Expected column order after the base 9:
/// `9=workspace_id`, `10=fsrs_stability`, `11=fsrs_difficulty`,
/// `12=fsrs_last_access`, `13=fsrs_access_count`, `14=fsrs_importance`,
/// `15=fsrs_storage_strength`, `16=fsrs_generation`
fn decode_memory_row(
    row: &turso::Row,
    embedding: Option<Vec<f32>>,
    metadata_str: Option<String>,
    tags: Vec<String>,
) -> Result<Memory, turso::Error> {
    Ok(Memory {
        id: row.get(0)?,
        memory_id: row.get(1)?,
        content: row.get(2)?,
        embedding,
        embedding_model: row.get(4)?,
        session_id: row.get(5)?,
        created_at: row.get(6)?,
        updated_at: row.get(7)?,
        metadata: metadata_str.and_then(|s| serde_json::from_str(&s).ok()),
        tags,
        workspace_id: row.get(9)?,
        fsrs_stability: f32_from_real(row.get::<f64>(10)?),
        fsrs_difficulty: f32_from_real(row.get::<f64>(11)?),
        fsrs_last_access: row.get(12)?,
        fsrs_access_count: row.get(13)?,
        fsrs_importance: f32_from_real(row.get::<f64>(14)?),
        fsrs_storage_strength: f32_from_real(row.get::<f64>(15)?),
        fsrs_generation: row.get(16)?,
    })
}

/// Narrow a `REAL` column back to the `f32` it was written from.
///
/// Every `f32` this crate stores goes in as `f64::from(f32)`, which is exact,
/// so for those rows this narrowing is exact too. Any other value rounds to
/// the nearest `f32`, and one beyond the `f32` range becomes infinite — the
/// semantics of the `as` cast, implemented losslessly in `nanna_numeric`.
fn f32_from_real(value: f64) -> f32 {
    nanna_numeric::f32_from_f64(value)
}

fn decode_embedding(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>().0.iter()
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// Result of a salvaging bulk load: the rows that decoded, the ids skipped as
/// corrupt, and the total row count the table reported.
#[derive(Debug, Default)]
pub struct BulkLoadReport {
    pub memories: Vec<Memory>,
    pub corrupt_ids: Vec<i64>,
    pub expected: usize,
}

/// True if `err` looks like an on-disk corruption error — e.g. turso's
/// "inconsistent overflow chain observed during payload read".
///
/// Checks the typed variant first: turso maps `LimboError::Corrupt` to
/// `turso::Error::Corrupt`, but that variant *renders as the bare inner
/// message* (e.g. "Invalid page type: 0" — no "corrupt" substring), which is
/// exactly how the 2026-07-22 salvage failure slipped past a string-only
/// match. The string patterns remain for errors that crossed the
/// `StorageError` → `String` boundary that `MemoryError::Persistence` forces
/// (the enum variant isn't visible there). Pure, unit-testable.
#[must_use]
pub fn is_corruption_error(err: &StorageError) -> bool {
    if let StorageError::Database(e) = err
        && matches!(e, turso::Error::Corrupt(_) | turso::Error::NotAdb(_)) {
            return true;
        }
    let s = err.to_string().to_lowercase();
    s.contains("overflow chain")
        || s.contains("corrupt")
        || s.contains("malformed")
        || s.contains("invalid page")
}


const CHUNK_SELECT_BY_PARENT: &str = "SELECT id, memory_id, ordinal, content, char_start, char_end, embedding, embedding_model, chunk_max_chars, chunker_version, workspace_id, created_at, updated_at FROM memory_chunks WHERE memory_id = ?1 ORDER BY ordinal ASC";

const CHUNK_SELECT_PAGE: &str = "SELECT id, memory_id, ordinal, content, char_start, char_end, embedding, embedding_model, chunk_max_chars, chunker_version, workspace_id, created_at, updated_at FROM memory_chunks WHERE memory_id = ?1 ORDER BY ordinal ASC LIMIT ?2 OFFSET ?3";

const CHUNK_KNN_SCOPED: &str = "SELECT memory_id, ordinal, vector_distance_cos(embedding, ?1) AS dist FROM memory_chunks WHERE embedding IS NOT NULL AND embedding_model = ?2 AND octet_length(embedding) = ?3 AND (workspace_id = ?4 OR workspace_id IS NULL) ORDER BY dist ASC LIMIT ?5";

const CHUNK_KNN_GLOBAL: &str = "SELECT memory_id, ordinal, vector_distance_cos(embedding, ?1) AS dist FROM memory_chunks WHERE embedding IS NOT NULL AND embedding_model = ?2 AND octet_length(embedding) = ?3 ORDER BY dist ASC LIMIT ?4";


/// Ordinal used for a whole-row (parent) vector in the shared embedding queue.
///
/// Negative so it can never collide with a real chunk ordinal, which counts
/// from zero. One queue serves rows and chunks alike; a separate `kind` column
/// would be a second source of truth that could disagree with the key.
pub const ROW_ORDINAL: i64 = -1;

/// Longest error text kept per queue item.
///
/// A provider error is diagnostic, not a log: one line of it identifies the
/// cause (402, 404, connection refused), and a full HTML error page would put
/// kilobytes per row into a table that is read on every health check.
const QUEUE_ERROR_MAX_CHARS: usize = 500;

/// Trim a provider error to something a queue row can hold, marking the cut.
fn truncate_queue_error(error: &str) -> String {
    if error.chars().count() <= QUEUE_ERROR_MAX_CHARS {
        return error.to_string();
    }
    let head: String = error.chars().take(QUEUE_ERROR_MAX_CHARS).collect();
    format!("{head} [truncated]")
}

const PUT_MEMORY_VECTOR: &str = "INSERT INTO memory_vectors (memory_id, embedding_model, embedding, dim) VALUES (?1, ?2, ?3, ?4) ON CONFLICT(memory_id, embedding_model) DO UPDATE SET embedding = excluded.embedding, dim = excluded.dim, updated_at = datetime('now')";

const PUT_CHUNK_VECTOR: &str = "INSERT INTO memory_chunk_vectors (memory_id, ordinal, embedding_model, embedding, dim) VALUES (?1, ?2, ?3, ?4, ?5) ON CONFLICT(memory_id, ordinal, embedding_model) DO UPDATE SET embedding = excluded.embedding, dim = excluded.dim, updated_at = datetime('now')";

const BUCKET_CENSUS: &str = "SELECT embedding_model, COUNT(*) AS n FROM memory_vectors GROUP BY embedding_model ORDER BY n DESC";

const ENQUEUE_EMBEDDING: &str = "INSERT INTO embedding_queue (memory_id, ordinal, embedding_model) VALUES (?1, ?2, ?3) ON CONFLICT(memory_id, ordinal, embedding_model) DO NOTHING";

const PENDING_EMBEDDINGS: &str = "SELECT memory_id, ordinal FROM embedding_queue WHERE embedding_model = ?1 ORDER BY attempts ASC, enqueued_at ASC LIMIT ?2";

/// Adopt vectors already materialized on `memory_chunks` into the bucket table.
///
/// On an existing database every chunk vector lives only in
/// `memory_chunks.embedding`, and the bucket table starts empty. Without this
/// step the queue seed below would treat ALREADY EMBEDDED chunks as missing and
/// re-embed the entire store on first run — turning a migration into a full
/// re-index, against a provider that may be rate-limited or paid.
///
/// Idempotent, so it is safe to run on every drain.
const ADOPT_CHUNK_VECTORS: &str = "INSERT INTO memory_chunk_vectors (memory_id, ordinal, embedding_model, embedding, dim) SELECT memory_id, ordinal, embedding_model, embedding, octet_length(embedding) / 4 FROM memory_chunks WHERE embedding IS NOT NULL AND embedding_model IS NOT NULL ON CONFLICT(memory_id, ordinal, embedding_model) DO NOTHING";

/// Seed the queue from chunks that have no vector for `model`.
///
/// The queue is authoritative, but it starts empty on an existing database and
/// nothing retroactively enqueues the chunks written before it existed. Without
/// this the drain would read a permanently empty queue and report the store
/// complete while every pre-existing chunk sat unembedded — a derived queue
/// replaced by a durable one that knows about nothing.
///
/// `NOT EXISTS` against the bucket table rather than against
/// `memory_chunks.embedding_model`, because the bucket table is the truth about
/// which models have embedded a chunk.
const SEED_CHUNK_QUEUE: &str = "INSERT INTO embedding_queue (memory_id, ordinal, embedding_model) SELECT c.memory_id, c.ordinal, ?1 FROM memory_chunks c WHERE NOT EXISTS (SELECT 1 FROM memory_chunk_vectors v WHERE v.memory_id = c.memory_id AND v.ordinal = c.ordinal AND v.embedding_model = ?1) LIMIT ?2 ON CONFLICT(memory_id, ordinal, embedding_model) DO NOTHING";

/// Chunk work from the queue, with the text to embed.
///
/// Joined to `memory_chunks` for content: the queue stores identity, not a copy
/// of the text, so a chunk rewritten between enqueue and drain is embedded as it
/// is NOW rather than as it was when queued.
const QUEUED_CHUNKS: &str = "SELECT c.id, c.memory_id, c.ordinal, c.content FROM embedding_queue q JOIN memory_chunks c ON c.memory_id = q.memory_id AND c.ordinal = q.ordinal WHERE q.embedding_model = ?1 AND q.ordinal >= 0 ORDER BY q.attempts ASC, q.enqueued_at ASC LIMIT ?2";

/// Upsert, not UPDATE.
///
/// A bare UPDATE matches nothing when the work was never enqueued — and that is
/// the common case for the paths that most need a record, since a chunk written
/// before this table existed, or one whose enqueue itself failed, has no row to
/// update. The statement then affects zero rows and reports success, which is
/// the exact silent-failure shape this table was added to eliminate. Inserting
/// on conflict-miss also states the truth: work that just failed is work that
/// still needs doing.
const RECORD_EMBEDDING_FAILURE: &str = "INSERT INTO embedding_queue (memory_id, ordinal, embedding_model, attempts, last_error, last_attempt_at) VALUES (?1, ?2, ?3, 1, ?4, datetime('now')) ON CONFLICT(memory_id, ordinal, embedding_model) DO UPDATE SET attempts = attempts + 1, last_error = excluded.last_error, last_attempt_at = excluded.last_attempt_at";

const RESTORE_CHUNK_VECTORS: &str = "UPDATE memory_chunks SET embedding = (SELECT v.embedding FROM memory_chunk_vectors v WHERE v.memory_id = memory_chunks.memory_id AND v.ordinal = memory_chunks.ordinal AND v.embedding_model = ?1), embedding_model = ?1, updated_at = datetime('now') WHERE EXISTS (SELECT 1 FROM memory_chunk_vectors v WHERE v.memory_id = memory_chunks.memory_id AND v.ordinal = memory_chunks.ordinal AND v.embedding_model = ?1)";

/// The error shown per model is the MOST RECENT one, not `MAX(last_error)` —
/// MAX is lexicographic, so a stale "502 upstream" would forever outrank a
/// current "402 no credit" and the health line would name the wrong problem.
const EMBEDDING_QUEUE_HEALTH: &str = "SELECT q.embedding_model, COUNT(*) AS n, (SELECT q2.last_error FROM embedding_queue q2 WHERE q2.embedding_model = q.embedding_model AND q2.last_error IS NOT NULL ORDER BY q2.last_attempt_at DESC LIMIT 1) FROM embedding_queue q GROUP BY q.embedding_model ORDER BY n DESC";

impl MemoryRepository {
    // ------------------------------------------------------------------
    // Embedding buckets: one vector per (row | chunk, model)
    // ------------------------------------------------------------------

    /// Store `embedding` as `model`'s vector for `memory_id`, replacing only
    /// that model's bucket.
    ///
    /// Other models' vectors are untouched. That is the whole point: a provider
    /// that flaps away and back used to cost two full re-embeds of the store,
    /// because a single vector column meant adopting a new model destroyed the
    /// previous one's work.
    ///
    /// # Errors
    /// Returns [`StorageError`] if the write fails.
    ///
    /// # Panics
    /// Panics if `embedding` or `model` is empty — both would store a bucket
    /// that nothing can ever match.
    pub async fn put_memory_vector(
        &self,
        memory_id: &str,
        model: &str,
        embedding: &[f32],
    ) -> Result<(), StorageError> {
        assert!(!embedding.is_empty(), "an empty vector is not a bucket");
        assert!(!model.is_empty(), "a bucket must name the model that filled it");
        let blob: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();
        let dim = i64::try_from(embedding.len()).unwrap_or(i64::MAX);
        let conn = self.conn.lock().await;
        conn.execute(PUT_MEMORY_VECTOR, turso::params![memory_id.to_string(), model.to_string(), blob, dim])
            .await?;
        drop(conn);
        Ok(())
    }

    /// Drop every bucket for `memory_id` — row and chunk alike — and zero the
    /// bytes on the way out.
    ///
    /// Called when a memory's CONTENT is rewritten. Every existing vector
    /// describes the old text, so keeping any of them means the memory stays
    /// findable by words it no longer contains: a later rebind re-activates the
    /// stale bucket, and because the backfill skips entries that already have a
    /// bucket for the active model, nothing ever replaces it.
    ///
    /// The in-RAM path already enforced this (`update_content_and_embedding`
    /// clears the bucket map, with a test); the durable side did not, so a
    /// restart resurrected what the rewrite had deliberately discarded.
    ///
    /// Zeroed rather than merely deleted for the same reason deletes are: an
    /// embedding inverts back to its source text, so a superseded vector is a
    /// copy of the superseded text.
    ///
    /// # Errors
    /// Returns [`StorageError`] if the write fails.
    pub async fn clear_memory_buckets(&self, memory_id: &str) -> Result<(), StorageError> {
        let conn = self.conn.lock().await;
        for stmt in [
            "UPDATE memory_vectors SET embedding = zeroblob(octet_length(embedding)) WHERE memory_id = ?1",
            "UPDATE memory_chunk_vectors SET embedding = zeroblob(octet_length(embedding)) WHERE memory_id = ?1",
            "DELETE FROM memory_vectors WHERE memory_id = ?1",
            "DELETE FROM memory_chunk_vectors WHERE memory_id = ?1",
        ] {
            conn.execute(stmt, turso::params![memory_id.to_string()]).await?;
        }
        // Zeroing writes a NEW page version; the original still sits in the WAL,
        // which turso 0.6.1 never checkpoints on its own. Without truncating it
        // the overwrite is cosmetic — the pre-overwrite bytes remain readable
        // in the log. Same reasoning, and the same call, as the delete path.
        Self::checkpoint_truncate(&conn).await;
        // Held from the zeroing through the checkpoint: the guard is the
        // transaction.
        drop(conn);
        Ok(())
    }

    /// Every stored bucket for `memory_id`, as `(model, vector)`.
    ///
    /// # Errors
    /// Returns [`StorageError`] if the query fails.
    pub async fn memory_vectors(&self, memory_id: &str) -> Result<Vec<(String, Vec<f32>)>, StorageError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT embedding_model, embedding FROM memory_vectors WHERE memory_id = ?1",
                turso::params![memory_id.to_string()],
            )
            .await?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            let model: String = row.get(0)?;
            let blob: Vec<u8> = row.get(1)?;
            out.push((model, decode_embedding(&blob)));
        }
        // Held until the cursor is gone: an open `Rows` on the shared
        // connection swallows later writes.
        drop(rows);
        drop(conn);
        Ok(out)
    }

    /// Every bucket for every memory, as `(memory_id, model, vector)`.
    ///
    /// One scan: the loader needs all buckets for all rows, and N+1 point
    /// lookups over a few thousand memories is a startup stall.
    ///
    /// # Errors
    /// Returns [`StorageError`] if the query fails.
    pub async fn all_memory_vectors(&self) -> Result<Vec<(String, String, Vec<f32>)>, StorageError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query("SELECT memory_id, embedding_model, embedding FROM memory_vectors", ())
            .await?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            let id: String = row.get(0)?;
            let model: String = row.get(1)?;
            let blob: Vec<u8> = row.get(2)?;
            out.push((id, model, decode_embedding(&blob)));
        }
        // Held until the cursor is gone: an open `Rows` on the shared
        // connection swallows later writes.
        drop(rows);
        drop(conn);
        Ok(out)
    }

    /// How many vectors each model holds — the bucket census, largest first.
    ///
    /// # Errors
    /// Returns [`StorageError`] if the query fails.
    pub async fn bucket_counts(&self) -> Result<Vec<(String, usize)>, StorageError> {
        let conn = self.conn.lock().await;
        let mut rows = conn.query(BUCKET_CENSUS, ()).await?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            let model: String = row.get(0)?;
            let n: i64 = row.get(1)?;
            out.push((model, usize::try_from(n).unwrap_or(0)));
        }
        // Held until the cursor is gone: an open `Rows` on the shared
        // connection swallows later writes.
        drop(rows);
        drop(conn);
        Ok(out)
    }

    /// Store a chunk's vector under `model`, leaving other models' alone.
    ///
    /// # Errors
    /// Returns [`StorageError`] if the write fails.
    ///
    /// # Panics
    /// Panics if `embedding` or `model` is empty.
    pub async fn put_chunk_vector(
        &self,
        memory_id: &str,
        ordinal: i64,
        model: &str,
        embedding: &[f32],
    ) -> Result<(), StorageError> {
        assert!(!embedding.is_empty(), "an empty vector is not a bucket");
        assert!(!model.is_empty(), "a bucket must name the model that filled it");
        let blob: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();
        let dim = i64::try_from(embedding.len()).unwrap_or(i64::MAX);
        let conn = self.conn.lock().await;
        conn.execute(
            PUT_CHUNK_VECTOR,
            turso::params![memory_id.to_string(), ordinal, model.to_string(), blob, dim],
        )
        .await?;
        drop(conn);
        Ok(())
    }

    // ------------------------------------------------------------------
    // The durable embedding queue
    // ------------------------------------------------------------------

    /// Enqueue `(memory_id, ordinal, model)` for embedding.
    ///
    /// `ordinal` is [`ROW_ORDINAL`] for a whole-row vector and `>= 0` for a
    /// chunk, so one queue covers both without a discriminator column that
    /// could disagree with the key.
    ///
    /// Idempotent, and deliberately does NOT reset `attempts`: re-enqueueing
    /// must not let a repeatedly-failing item disguise itself as fresh work and
    /// jump the queue forever.
    ///
    /// # Errors
    /// Returns [`StorageError`] if the write fails.
    ///
    /// # Panics
    /// Panics if `model` is empty.
    pub async fn enqueue_embedding(
        &self,
        memory_id: &str,
        ordinal: i64,
        model: &str,
    ) -> Result<(), StorageError> {
        assert!(!model.is_empty(), "queued work must name the model it is for");
        let conn = self.conn.lock().await;
        conn.execute(
            ENQUEUE_EMBEDDING,
            turso::params![memory_id.to_string(), ordinal, model.to_string()],
        )
        .await?;
        drop(conn);
        Ok(())
    }

    /// Up to `limit` pending items for `model`: fewest attempts first, then
    /// oldest first.
    ///
    /// Attempts before age means an item that fails forever drifts to the back
    /// instead of blocking everything behind it — a dead-letter state without
    /// the extra column.
    ///
    /// # Errors
    /// Returns [`StorageError`] if the query fails.
    pub async fn pending_embeddings(
        &self,
        model: &str,
        limit: usize,
    ) -> Result<Vec<(String, i64)>, StorageError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                PENDING_EMBEDDINGS,
                turso::params![model.to_string(), i64::try_from(limit).unwrap_or(i64::MAX)],
            )
            .await?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            out.push((row.get(0)?, row.get(1)?));
        }
        // Held until the cursor is gone: an open `Rows` on the shared
        // connection swallows later writes.
        drop(rows);
        drop(conn);
        Ok(out)
    }

    /// Remove a completed item.
    ///
    /// # Errors
    /// Returns [`StorageError`] if the delete fails.
    pub async fn dequeue_embedding(
        &self,
        memory_id: &str,
        ordinal: i64,
        model: &str,
    ) -> Result<(), StorageError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "DELETE FROM embedding_queue WHERE memory_id = ?1 AND ordinal = ?2 AND embedding_model = ?3",
            turso::params![memory_id.to_string(), ordinal, model.to_string()],
        )
        .await?;
        drop(conn);
        Ok(())
    }

    /// Record a failed attempt, keeping the reason.
    ///
    /// The reason is what separates a queue that is stuck from one that is
    /// merely slow. A 402 with nothing attached reads as silence, and silence
    /// is how an entire day of writes went unembedded unnoticed.
    ///
    /// # Errors
    /// Returns [`StorageError`] if the update fails.
    pub async fn record_embedding_failure(
        &self,
        memory_id: &str,
        ordinal: i64,
        model: &str,
        error: &str,
    ) -> Result<(), StorageError> {
        let conn = self.conn.lock().await;
        conn.execute(
            RECORD_EMBEDDING_FAILURE,
            turso::params![
                memory_id.to_string(),
                ordinal,
                model.to_string(),
                truncate_queue_error(error)
            ],
        )
        .await?;
        drop(conn);
        Ok(())
    }

    /// Drop queued work for models not in `keep`.
    ///
    /// Rows for a model that left the priority list can never drain — the
    /// drain only ever runs for bound models — so they sit forever and pollute
    /// `embedding_queue_health` with depth that no configuration change short
    /// of re-adding the model will move. Safe to delete: the queue stores
    /// identity, not work product, and if the model returns to the list the
    /// seeding sweep re-derives exactly these rows from what its buckets are
    /// still missing.
    ///
    /// A caller with NOTHING resolved must not call this — an empty `keep`
    /// here would wipe the whole queue because one API key was missing at
    /// boot, so it is rejected loudly instead.
    ///
    /// # Errors
    /// Returns [`StorageError`] if the delete fails.
    ///
    /// # Panics
    /// Panics if `keep` is empty.
    pub async fn retain_queue_models(&self, keep: &[String]) -> Result<usize, StorageError> {
        assert!(
            !keep.is_empty(),
            "refusing to clear the whole queue — an empty keep-list means no provider resolved, \
             which is a configuration problem, not a cleanup request"
        );
        let placeholders: Vec<String> = (1..=keep.len()).map(|i| format!("?{i}")).collect();
        let sql = format!(
            "DELETE FROM embedding_queue WHERE embedding_model NOT IN ({})",
            placeholders.join(", ")
        );
        let params: Vec<turso::Value> =
            keep.iter().map(|m| turso::Value::Text(m.clone())).collect();
        let conn = self.conn.lock().await;
        let n = conn.execute(&sql, params).await?;
        drop(conn);
        Ok(usize::try_from(n).unwrap_or(0))
    }

    /// Queue depth per model with the most recent error for each.
    ///
    /// Exists so the daemon can SAY that work is piling up and why, rather than
    /// leaving it to be inferred from search results that quietly went missing.
    ///
    /// # Errors
    /// Returns [`StorageError`] if the query fails.
    pub async fn embedding_queue_health(&self) -> Result<Vec<(String, usize, Option<String>)>, StorageError> {
        let conn = self.conn.lock().await;
        let mut rows = conn.query(EMBEDDING_QUEUE_HEALTH, ()).await?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            let model: String = row.get(0)?;
            let n: i64 = row.get(1)?;
            let err: Option<String> = row.get(2).ok();
            out.push((model, usize::try_from(n).unwrap_or(0), err));
        }
        // Held until the cursor is gone: an open `Rows` on the shared
        // connection swallows later writes.
        drop(rows);
        drop(conn);
        Ok(out)
    }


    // ---------------------------------------------------------------
    // Per-chunk vectors
    //
    // A memory's content is unbounded; an embedding model's window is not.
    // These write and read the slices that make a long memory searchable by
    // any part of itself rather than only by its first window's worth.
    // ---------------------------------------------------------------

    /// Replace a parent's entire chunk set.
    ///
    /// Delete-then-insert rather than upsert: re-chunking under a different
    /// budget produces a different NUMBER of chunks, so leaving old ordinals
    /// behind would mix two chunkings of the same text and double-count it in
    /// search. Both statements run under one `conn.lock()` guard, which is the
    /// de-facto transaction here — nothing in this crate issues BEGIN/COMMIT.
    ///
    /// # Errors
    /// Returns [`StorageError`] if any statement fails.
    ///
    /// # Panics
    /// Panics if `memory_id` is empty, or if a chunk names a different parent.
    pub async fn replace_chunks(
        &self,
        memory_id: &str,
        chunks: &[NewMemoryChunk],
    ) -> Result<usize, StorageError> {
        assert!(!memory_id.is_empty(), "memory_id must not be empty");
        assert!(
            chunks.iter().all(|c| c.memory_id == memory_id),
            "every chunk must name the parent it is written under"
        );
        let conn = self.conn.lock().await;
        // The chunk set is being replaced, so every stored chunk VECTOR for
        // this parent describes spans of text that are about to stop existing.
        // They go in the same statement batch, zeroed first, because a stale
        // bucket is not merely wasteful — `restore_chunk_vectors` re-activates
        // whatever bucket exists on the next rebind and deletes the queue row
        // that would have re-embedded the new text, so one surviving stale
        // bucket makes the rewritten chunk permanently searchable by words it
        // no longer contains. That exact sequence was reproduced against a
        // live schema before this block existed.
        //
        // Zero-then-delete, matching `clear_memory_buckets`: an embedding
        // inverts back to its source text, so a superseded vector is a copy of
        // the superseded text. The checkpoint that makes the zeroing real is
        // paid only when there was something to zero — for a brand-new memory
        // (the overwhelmingly common caller, via `add`) both statements match
        // nothing and no fsync happens.
        let stale = conn
            .execute(
                "UPDATE memory_chunk_vectors SET embedding = zeroblob(octet_length(embedding)) WHERE memory_id = ?1",
                turso::params![memory_id],
            )
            .await?;
        conn.execute(
            "DELETE FROM memory_chunk_vectors WHERE memory_id = ?1",
            turso::params![memory_id],
        )
        .await?;
        // Queued chunk work names ordinals of the OLD splitting; the caller
        // re-enqueues against the new one. Row-level work (ordinal -1) is not
        // touched here — this primitive replaces chunks, not the row vector.
        conn.execute(
            "DELETE FROM embedding_queue WHERE memory_id = ?1 AND ordinal >= 0",
            turso::params![memory_id],
        )
        .await?;
        conn.execute(
            "DELETE FROM memory_chunks WHERE memory_id = ?1",
            turso::params![memory_id],
        )
        .await?;
        for c in chunks {
            let blob: Option<Vec<u8>> = c
                .embedding
                .as_ref()
                .map(|v| v.iter().flat_map(|f| f.to_le_bytes()).collect());
            conn.execute(
                "INSERT INTO memory_chunks (memory_id, ordinal, content, char_start, char_end, embedding, embedding_model, chunk_max_chars, chunker_version, workspace_id) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                turso::params![
                    c.memory_id.clone(),
                    c.ordinal,
                    c.content.clone(),
                    c.char_start,
                    c.char_end,
                    blob,
                    c.embedding_model.clone(),
                    c.chunk_max_chars,
                    c.chunker_version,
                    c.workspace_id.clone(),
                ],
            )
            .await?;
        }
        if stale > 0 {
            // Without the checkpoint the zeroing is cosmetic: turso 0.6.1 keeps
            // the pre-overwrite pages readable in the WAL indefinitely.
            Self::checkpoint_truncate(&conn).await;
        }
        // Held from the first delete through the checkpoint: the guard is the
        // transaction that keeps two chunkings from ever coexisting.
        drop(conn);
        Ok(chunks.len())
    }

    /// Every chunk of one parent, in ordinal order.
    ///
    /// # Errors
    /// Returns [`StorageError`] if the query fails.
    pub async fn chunks_for(&self, memory_id: &str) -> Result<Vec<MemoryChunk>, StorageError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(CHUNK_SELECT_BY_PARENT, turso::params![memory_id])
            .await?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            out.push(Self::decode_chunk_row(&row)?);
        }
        // Held until the cursor is gone: an open `Rows` on the shared
        // connection swallows later writes.
        drop(rows);
        drop(conn);
        debug_assert!(
            out.windows(2).all(|w| w[0].ordinal < w[1].ordinal),
            "ordinals must be strictly increasing (unique index + ORDER BY)"
        );
        Ok(out)
    }

    /// How many chunks a parent has, without loading their content.
    ///
    /// # Errors
    /// Returns [`StorageError`] if the query fails.
    pub async fn chunk_count(&self, memory_id: &str) -> Result<usize, StorageError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT COUNT(*) FROM memory_chunks WHERE memory_id = ?1",
                turso::params![memory_id],
            )
            .await?;
        let n = match rows.next().await? {
            Some(row) => row.get::<i64>(0)?,
            None => 0,
        };
        // Held until the cursor is gone: an open `Rows` on the shared
        // connection swallows later writes.
        drop(rows);
        drop(conn);
        Ok(usize::try_from(n).unwrap_or(0))
    }

    /// One page of a parent's chunks, for paginated recall.
    ///
    /// `page` is 1-based. Paging by ordinal rather than byte offset is what
    /// lets a cursor survive a re-embed: ordinals are stable, byte offsets move
    /// the moment the text is re-chunked.
    ///
    /// # Errors
    /// Returns [`StorageError`] if the query fails.
    ///
    /// # Panics
    /// Panics if `page` or `per_page` is 0 — both are caller errors.
    pub async fn chunk_page(
        &self,
        memory_id: &str,
        page: usize,
        per_page: usize,
    ) -> Result<Vec<MemoryChunk>, StorageError> {
        assert!(page > 0, "pages are 1-based");
        assert!(per_page > 0, "per_page must be positive");
        let offset = i64::try_from((page - 1) * per_page).unwrap_or(i64::MAX);
        let limit = i64::try_from(per_page).unwrap_or(i64::MAX);
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                CHUNK_SELECT_PAGE,
                turso::params![memory_id, limit, offset],
            )
            .await?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            out.push(Self::decode_chunk_row(&row)?);
        }
        // Held until the cursor is gone: an open `Rows` on the shared
        // connection swallows later writes.
        drop(rows);
        drop(conn);
        Ok(out)
    }

    /// Re-activate `model`'s stored chunk vectors.
    ///
    /// `memory_chunks.embedding` is the materialized ACTIVE vector that search
    /// compares against; `memory_chunk_vectors` is the durable per-model set.
    /// On a rebind this copies the new model's set into the active column,
    /// which is what makes returning to a provider free — without it, every
    /// chunk that model had already embedded would be queued and embedded
    /// again, and a provider that flaps away and back would pay twice for work
    /// it had already done.
    ///
    /// Returns how many chunks were re-activated.
    ///
    /// # Errors
    /// Returns [`StorageError`] if the update fails.
    ///
    /// # Panics
    /// Panics if `model` is empty.
    pub async fn restore_chunk_vectors(&self, model: &str) -> Result<usize, StorageError> {
        assert!(!model.is_empty(), "restoring must name a model");
        let conn = self.conn.lock().await;
        let n = conn.execute(RESTORE_CHUNK_VECTORS, turso::params![model.to_string()]).await?;
        // Work that is now satisfied must leave the queue, or the drain would
        // re-embed vectors it just restored.
        conn.execute(
            "DELETE FROM embedding_queue WHERE embedding_model = ?1 AND ordinal >= 0 AND EXISTS (SELECT 1 FROM memory_chunk_vectors v WHERE v.memory_id = embedding_queue.memory_id AND v.ordinal = embedding_queue.ordinal AND v.embedding_model = ?1)",
            turso::params![model.to_string()],
        )
        .await?;
        // Held across the restore and the dequeue: the guard is the
        // transaction, so the drain never sees restored work still queued.
        drop(conn);
        Ok(usize::try_from(n).unwrap_or(0))
    }

    /// Attach a vector to a chunk that was written without one.
    ///
    /// # Errors
    /// Returns [`StorageError`] if the update fails.
    ///
    /// # Panics
    /// Panics if `embedding` is empty.
    pub async fn set_chunk_embedding(
        &self,
        chunk_id: i64,
        embedding: &[f32],
        model: &str,
    ) -> Result<bool, StorageError> {
        assert!(!embedding.is_empty(), "embedding must not be empty");
        let blob: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();
        let conn = self.conn.lock().await;
        // Durable bucket first, then the materialized active copy. If the
        // process dies between them the bucket is authoritative and the restore
        // path re-activates it; the reverse order would leave an active vector
        // with no durable record, which is the state that made a flap cost a
        // full re-embed.
        conn.execute(
            "INSERT INTO memory_chunk_vectors (memory_id, ordinal, embedding_model, embedding, dim) SELECT memory_id, ordinal, ?2, ?1, ?4 FROM memory_chunks WHERE id = ?3 ON CONFLICT(memory_id, ordinal, embedding_model) DO UPDATE SET embedding = excluded.embedding, dim = excluded.dim, updated_at = datetime('now')",
            turso::params![blob.clone(), model.to_string(), chunk_id, i64::try_from(embedding.len()).unwrap_or(i64::MAX)],
        )
        .await?;
        // Work done leaves the queue in the SAME lock that recorded it done.
        // Leaving the dequeue to a separate caller is how work gets dropped or
        // repeated: a crash between the two would either re-embed a chunk that
        // is already finished, or — with the order reversed — lose the record
        // that it still needs doing.
        conn.execute(
            "DELETE FROM embedding_queue WHERE embedding_model = ?2 AND (memory_id, ordinal) IN (SELECT memory_id, ordinal FROM memory_chunks WHERE id = ?1)",
            turso::params![chunk_id, model.to_string()],
        )
        .await?;
        let n = conn
            .execute(
                "UPDATE memory_chunks SET embedding = ?1, embedding_model = ?2, updated_at = datetime('now') WHERE id = ?3",
                turso::params![blob, model.to_string(), chunk_id],
            )
            .await?;
        // Held across bucket, dequeue, and active copy: the guard is the
        // transaction.
        drop(conn);
        Ok(n > 0)
    }

    /// Chunks whose vector is missing, or came from a different model.
    ///
    /// This is why a model switch is a resumable backfill rather than a
    /// rebuild: the text is already stored and chunked, so only the vectors
    /// need recomputing — incrementally, and restartably after a crash.
    ///
    /// Chunk work for `model`, taken from the durable queue.
    ///
    /// Seeds the queue first so an existing database — whose chunks predate the
    /// queue entirely — produces work rather than reporting itself complete.
    /// Seeding is an idempotent upsert, so it costs one statement per drain
    /// once the store has caught up.
    ///
    /// # Errors
    /// Returns [`StorageError`] if the query fails.
    ///
    /// # Panics
    /// Panics if `model` is empty.
    pub async fn chunks_needing_embedding(
        &self,
        model: &str,
        limit: usize,
        seed: bool,
    ) -> Result<Vec<QueuedChunk>, StorageError> {
        assert!(!model.is_empty(), "chunk work must name the model it is for");
        let limit_i64 = i64::try_from(limit).unwrap_or(i64::MAX);
        let conn = self.conn.lock().await;
        if seed {
            // Adopt first: work already done must be visible as done before we
            // ask what is missing, or a migration re-embeds the whole store.
            //
            // Behind a flag because both statements scan all of
            // `memory_chunks`, and the drain calls this every 64 items — the
            // backfill was quadratic in store size. One seeding per (model,
            // process) suffices: new writes enqueue themselves at write time,
            // so the scan only exists to pick up rows from BEFORE this process
            // (or before the queue existed at all).
            conn.execute(ADOPT_CHUNK_VECTORS, ()).await?;
        // Seed generously relative to the batch: the drain takes `limit` items,
        // but the seed is what makes them exist at all, and under-seeding would
        // stall progress at a fraction of the batch per pass.
            // The seed cap is generous but finite; the caller re-seeds only
            // once per process, so a store larger than the cap finishes over
            // successive restarts rather than in one. Uncapped would hold the
            // shared connection through one giant insert.
            conn.execute(
                SEED_CHUNK_QUEUE,
                turso::params![model.to_string(), 100_000_i64],
            )
            .await?;
        }
        let mut rows = conn
            .query(QUEUED_CHUNKS, turso::params![model.to_string(), limit_i64])
            .await?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            out.push(QueuedChunk {
                id: row.get(0)?,
                memory_id: row.get(1)?,
                ordinal: row.get(2)?,
                content: row.get(3)?,
            });
        }
        // Held from the seeding through the drained cursor: seeding and
        // reading the queue are one transaction, and an open `Rows` on the
        // shared connection swallows later writes.
        drop(rows);
        drop(conn);
        Ok(out)
    }

    /// Parents with no chunks yet — the eager backfill queue.
    ///
    /// Every memory written before this table existed starts here, and stays
    /// searchable on its parent vector until the backfill reaches it.
    ///
    /// # Errors
    /// Returns [`StorageError`] if the query fails.
    ///
    /// # Panics
    /// Panics if `limit` is 0.
    pub async fn parents_without_chunks(&self, limit: usize) -> Result<Vec<String>, StorageError> {
        assert!(limit > 0, "limit must be positive");
        let limit_i64 = i64::try_from(limit).unwrap_or(i64::MAX);
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT m.memory_id FROM memories m WHERE NOT EXISTS (SELECT 1 FROM memory_chunks c WHERE c.memory_id = m.memory_id) ORDER BY m.id ASC LIMIT ?1",
                turso::params![limit_i64],
            )
            .await?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            out.push(row.get::<String>(0)?);
        }
        // Held until the cursor is gone: an open `Rows` on the shared
        // connection swallows later writes.
        drop(rows);
        drop(conn);
        Ok(out)
    }

    /// k-NN over CHUNK vectors, returning `(parent memory_id, ordinal, distance)`.
    ///
    /// Only chunks embedded by `model` are compared. This is the correctness
    /// gate, and width is NOT a substitute for it: two different models of the
    /// same dimension produce vectors that compare without complaint and mean
    /// nothing, so an `octet_length` guard alone lets a foreign embedding space
    /// be scored as if it were valid — silently, with no floor and no log.
    ///
    /// This filter was missing when chunk search first landed. Because a chunk
    /// hit can admit its parent on chunk evidence alone, every chunk still
    /// carrying a previous provider's vector — the entire not-yet-backfilled
    /// tail after any same-width failover — could rank its memory into results
    /// against a query from a different space. The whole-row path never had
    /// this hole, but only by accident: `rebind_to_model` clears an entry's
    /// active vector when the new model has no bucket for it, and chunks have
    /// no equivalent.
    ///
    /// Also keeps the `octet_length(embedding) = ?` guard, which is now purely
    /// a crash guard rather than an identity check: `vector_distance_cos`
    /// errors on a width mismatch and would abort the whole scan.
    ///
    /// Twin of [`Self::search_by_embedding_sql`], including the explicit
    /// `drop(rows)` — an unfinished cursor on the shared connection silently
    /// swallows later writes.
    ///
    /// `limit` counts CHUNKS, not parents: several chunks of one memory can
    /// occupy consecutive positions, so a caller wanting k parents over-fetches.
    ///
    /// # Errors
    /// Returns [`StorageError`] if the query fails.
    ///
    /// # Panics
    /// Panics if the query embedding is empty, `model` is empty, or `limit` is 0.
    pub async fn search_chunks_by_embedding_sql(
        &self,
        query_embedding: &[f32],
        model: &str,
        limit: usize,
        workspace_id: Option<&str>,
    ) -> Result<Vec<(String, i64, f64)>, StorageError> {
        assert!(
            !query_embedding.is_empty(),
            "query embedding must not be empty"
        );
        assert!(!model.is_empty(), "a chunk search must name the model it is searching");
        assert!(limit > 0, "limit must be positive");

        let query_blob: Vec<u8> = query_embedding
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let query_bytes = i64::try_from(query_blob.len()).unwrap_or(i64::MAX);
        let limit_i64 = i64::try_from(limit).unwrap_or(i64::MAX);

        let conn = self.conn.lock().await;
        let mut rows = match workspace_id {
            Some(ws) => {
                conn.query(
                    CHUNK_KNN_SCOPED,
                    turso::params![query_blob, model.to_string(), query_bytes, ws, limit_i64],
                )
                .await?
            }
            None => {
                conn.query(
                    CHUNK_KNN_GLOBAL,
                    turso::params![query_blob, model.to_string(), query_bytes, limit_i64],
                )
                .await?
            }
        };

        let mut out: Vec<(String, i64, f64)> = Vec::new();
        while let Some(row) = rows.next().await? {
            out.push((row.get(0)?, row.get(1)?, row.get(2)?));
        }
        drop(rows);

        debug_assert!(out.len() <= limit, "returned more neighbours than the limit");
        debug_assert!(
            out.windows(2).all(|w| w[0].2 <= w[1].2),
            "distances must be non-decreasing (ORDER BY dist ASC)"
        );
        Ok(out)
    }

    fn decode_chunk_row(row: &turso::Row) -> Result<MemoryChunk, StorageError> {
        let embedding = match row.get_value(6)? {
            turso::Value::Blob(b) => Some(
                b.as_chunks::<4>().0.iter()
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            ),
            _ => None,
        };
        let embedding_model = match row.get_value(7)? {
            turso::Value::Text(t) => Some(t),
            _ => None,
        };
        let workspace_id = match row.get_value(10)? {
            turso::Value::Text(t) => Some(t),
            _ => None,
        };
        Ok(MemoryChunk {
            id: row.get(0)?,
            memory_id: row.get(1)?,
            ordinal: row.get(2)?,
            content: row.get(3)?,
            char_start: row.get(4)?,
            char_end: row.get(5)?,
            embedding,
            embedding_model,
            chunk_max_chars: row.get(8)?,
            chunker_version: row.get(9)?,
            workspace_id,
            created_at: row.get(11)?,
            updated_at: row.get(12)?,
        })
    }

    pub const fn new(conn: Arc<Mutex<Connection>>) -> Self {
        Self { conn }
    }

    /// Insert a memory and its tags, and return the stored row.
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the memory insert, a tag insert,
    /// or the read-back fails (a duplicate `memory_id` fails the insert), or
    /// [`StorageError::NotFound`] if the read-back finds no row. Statements
    /// that succeeded before a failure stay applied.
    pub async fn create(&self, mem: NewMemory) -> Result<Memory, StorageError> {
        let conn = self.conn.lock().await;

        // Serialize embedding as bytes
        let embedding_bytes: Option<Vec<u8>> = mem.embedding.as_ref().map(|e| {
            e.iter().flat_map(|f| f.to_le_bytes()).collect()
        });

        let metadata_json = mem.metadata.as_ref().map(std::string::ToString::to_string);
        let memory_id = mem.memory_id.clone();

        conn.execute(
            "INSERT INTO memories (memory_id, content, embedding, embedding_model, session_id, metadata,
                                   workspace_id,
                                   fsrs_stability, fsrs_difficulty, fsrs_last_access, fsrs_access_count,
                                   fsrs_importance, fsrs_storage_strength, fsrs_generation)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
            turso::params![
                mem.memory_id.as_str(),
                mem.content.as_str(),
                embedding_bytes,
                mem.embedding_model.as_deref(),
                mem.session_id.as_deref(),
                metadata_json.as_deref(),
                mem.workspace_id.as_deref(),
                f64::from(mem.fsrs_stability),
                f64::from(mem.fsrs_difficulty),
                mem.fsrs_last_access,
                mem.fsrs_access_count,
                f64::from(mem.fsrs_importance),
                f64::from(mem.fsrs_storage_strength),
                mem.fsrs_generation,
            ],
        )
        .await?;

        // Add tags
        for tag in &mem.tags {
            conn.execute(
                "INSERT INTO memory_tags (memory_id, tag) VALUES (?1, ?2)",
                turso::params![mem.memory_id.as_str(), tag.as_str()],
            )
            .await?;
        }

        drop(conn);
        self.get(&memory_id).await
    }

    /// Fetch one memory, with its tags, by `memory_id`.
    ///
    /// # Errors
    /// Returns [`StorageError::NotFound`] if no memory has `memory_id`, or
    /// [`StorageError::Database`] if either query fails or a column does not
    /// decode. Unparseable metadata loads as `None`.
    pub async fn get(&self, memory_id: &str) -> Result<Memory, StorageError> {
        let conn = self.conn.lock().await;

        let mut rows = conn
            .query(
                "SELECT id, memory_id, content, embedding, embedding_model, session_id, created_at, updated_at, metadata,
                        workspace_id,
                        fsrs_stability, fsrs_difficulty, fsrs_last_access, fsrs_access_count,
                        fsrs_importance, fsrs_storage_strength, fsrs_generation
                 FROM memories WHERE memory_id = ?1",
                turso::params![memory_id],
            )
            .await?;

        let memory = if let Some(row) = rows.next().await? {
            let embedding_bytes: Option<Vec<u8>> = row.get(3)?;
            let embedding = embedding_bytes.as_deref().map(decode_embedding);
            let metadata_str: Option<String> = row.get(8)?;

            // Get tags
            let mut tag_rows = conn
                .query(
                    "SELECT tag FROM memory_tags WHERE memory_id = ?1",
                    turso::params![memory_id],
                )
                .await?;

            let mut tags = Vec::new();
            while let Some(tag_row) = tag_rows.next().await? {
                tags.push(tag_row.get(0)?);
            }

            Ok(decode_memory_row(&row, embedding, metadata_str, tags)?)
        } else {
            Err(StorageError::NotFound(format!("Memory: {memory_id}")))
        };
        // Held until the cursor is gone: an open `Rows` on the shared
        // connection swallows later writes.
        drop(rows);
        drop(conn);
        memory
    }

    /// The `limit` most recently updated memories, newest first, without tags.
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the query fails or any row does
    /// not decode — one bad row fails the whole list. Unparseable metadata
    /// loads as `None`.
    pub async fn list_all(&self, limit: i64) -> Result<Vec<Memory>, StorageError> {
        let conn = self.conn.lock().await;

        let mut rows = conn
            .query(
                "SELECT id, memory_id, content, embedding, embedding_model, session_id, created_at, updated_at, metadata,
                        workspace_id,
                        fsrs_stability, fsrs_difficulty, fsrs_last_access, fsrs_access_count,
                        fsrs_importance, fsrs_storage_strength, fsrs_generation
                 FROM memories ORDER BY updated_at DESC LIMIT ?1",
                turso::params![limit],
            )
            .await?;

        let mut memories = Vec::new();
        while let Some(row) = rows.next().await? {
            let embedding_bytes: Option<Vec<u8>> = row.get(3)?;
            let embedding = embedding_bytes.as_deref().map(decode_embedding);
            let metadata_str: Option<String> = row.get(8)?;
            memories.push(decode_memory_row(&row, embedding, metadata_str, Vec::new())?);
        }
        // Held until the cursor is gone: an open `Rows` on the shared
        // connection swallows later writes.
        drop(rows);
        drop(conn);

        Ok(memories)
    }

    /// Load ALL entries efficiently (no limit) — for populating the in-memory cache on startup.
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the query fails or any row does
    /// not decode — one corrupt row fails the whole load (see
    /// [`Self::bulk_load_salvage`]).
    pub async fn bulk_load(&self) -> Result<Vec<Memory>, StorageError> {
        let conn = self.conn.lock().await;

        let mut rows = conn
            .query(
                "SELECT id, memory_id, content, embedding, embedding_model, session_id, created_at, updated_at, metadata,
                        workspace_id,
                        fsrs_stability, fsrs_difficulty, fsrs_last_access, fsrs_access_count,
                        fsrs_importance, fsrs_storage_strength, fsrs_generation
                 FROM memories ORDER BY id ASC",
                (),
            )
            .await?;

        let mut memories = Vec::new();
        while let Some(row) = rows.next().await? {
            let embedding_bytes: Option<Vec<u8>> = row.get(3)?;
            let embedding = embedding_bytes.as_deref().map(decode_embedding);
            let metadata_str: Option<String> = row.get(8)?;
            memories.push(decode_memory_row(&row, embedding, metadata_str, Vec::new())?);
        }
        // Held until the cursor is gone: an open `Rows` on the shared
        // connection swallows later writes.
        drop(rows);
        drop(conn);

        Ok(memories)
    }

    /// Load all entries, skipping only the rows whose payload is corrupt.
    ///
    /// [`bulk_load`](Self::bulk_load) aborts the whole table on the first corrupt
    /// row (a single `?` on `rows.next()`), so one bad overflow chain silently
    /// zeroes the entire memory store. This variant first reads just the rowids —
    /// the rowid lives in the cell header and needs no record-body / overflow
    /// read, so `SELECT id` survives the corruption — then loads each row on its
    /// own and skips + records the ones that fail. If even the id scan fails
    /// (whole-btree corruption) it returns `Err`, so the caller can surface a
    /// fully-degraded store rather than a silent empty one.
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] only if the id scan fails; a row
    /// that fails to load is skipped and listed in
    /// [`BulkLoadReport::corrupt_ids`] instead.
    pub async fn bulk_load_salvage(&self) -> Result<BulkLoadReport, StorageError> {
        let conn = self.conn.lock().await;

        // rowid-only scan survives a corrupt payload/overflow chain.
        let mut id_rows = conn.query("SELECT id FROM memories ORDER BY id ASC", ()).await?;
        let mut ids: Vec<i64> = Vec::new();
        while let Some(row) = id_rows.next().await? {
            ids.push(row.get(0)?);
        }
        drop(id_rows);

        let expected = ids.len();
        let mut memories = Vec::with_capacity(expected);
        let mut corrupt_ids = Vec::new();

        for id in ids {
            let loaded: Result<Option<Memory>, StorageError> = async {
                let mut rows = conn
                    .query(
                        "SELECT id, memory_id, content, embedding, embedding_model, session_id, created_at, updated_at, metadata,
                                workspace_id,
                                fsrs_stability, fsrs_difficulty, fsrs_last_access, fsrs_access_count,
                                fsrs_importance, fsrs_storage_strength, fsrs_generation
                         FROM memories WHERE id = ?1",
                        turso::params![id],
                    )
                    .await?;
                if let Some(row) = rows.next().await? {
                    let embedding_bytes: Option<Vec<u8>> = row.get(3)?;
                    let embedding = embedding_bytes.as_deref().map(decode_embedding);
                    let metadata_str: Option<String> = row.get(8)?;
                    Ok(Some(decode_memory_row(&row, embedding, metadata_str, Vec::new())?))
                } else {
                    Ok(None)
                }
            }
            .await;

            match loaded {
                Ok(Some(mem)) => memories.push(mem),
                Ok(None) => {} // row vanished between the id scan and the fetch — ignore
                Err(e) => {
                    corrupt_ids.push(id);
                    tracing::warn!(id, error = %e, "Skipping corrupt memory row during salvage load");
                }
            }
        }
        // Held across the id scan and every per-row load; each load's cursor
        // is dropped when its async block ends, so none is open here.
        drop(conn);

        if !corrupt_ids.is_empty() {
            tracing::warn!(
                salvaged = memories.len(),
                expected,
                corrupt = corrupt_ids.len(),
                "Salvaged memories table with corrupt rows skipped: {:?}",
                corrupt_ids
            );
        }

        Ok(BulkLoadReport { memories, corrupt_ids, expected })
    }

    /// Count total memories in the database.
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the query fails.
    pub async fn count(&self) -> Result<i64, StorageError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query("SELECT COUNT(*) FROM memories", ())
            .await?;
        let count = if let Some(row) = rows.next().await? {
            row.get(0)?
        } else {
            0
        };
        // Held until the cursor is gone: an open `Rows` on the shared
        // connection swallows later writes.
        drop(rows);
        drop(conn);
        Ok(count)
    }

    /// Update FSRS state for a memory entry identified by `memory_id`.
    ///
    /// Returns `true` when a row was updated.
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the update fails.
    pub async fn update_fsrs(
        &self,
        memory_id: &str,
        fsrs: &MemoryFsrsUpdate,
    ) -> Result<bool, StorageError> {
        let conn = self.conn.lock().await;
        let result = conn
            .execute(
                "UPDATE memories SET
                    fsrs_stability = ?1, fsrs_difficulty = ?2, fsrs_last_access = ?3,
                    fsrs_access_count = ?4, fsrs_importance = ?5,
                    fsrs_storage_strength = ?6, fsrs_generation = ?7,
                    updated_at = datetime('now')
                 WHERE memory_id = ?8",
                turso::params![
                    f64::from(fsrs.stability),
                    f64::from(fsrs.difficulty),
                    fsrs.last_access,
                    fsrs.access_count,
                    f64::from(fsrs.importance),
                    f64::from(fsrs.storage_strength),
                    fsrs.generation,
                    memory_id,
                ],
            )
            .await?;
        drop(conn);
        Ok(result > 0)
    }

    /// Exact k-nearest-neighbour search **in SQL**, returning the `limit` closest
    /// memories to `query_embedding` by cosine distance, nearest first.
    ///
    /// The pinned `turso 0.6.1` computes `vector_distance_cos` over the stored
    /// **raw little-endian f32 BLOBs directly** — no `vector32()` wrapper and no
    /// trailing type byte are needed (that byte only matters to `vector_extract`),
    /// and the query vector is bound the same way. Proven end to end in
    /// `crates/nanna-storage/tests/vector_knn.rs`: identical → 0, orthogonal → 1,
    /// opposite → 2, and `ORDER BY` ranks correctly. This is O(N) distance
    /// computes but **O(1) RAM** — it streams rows instead of `bulk_load`-ing every
    /// embedding into memory, which is the cost behind the ~50k in-RAM ceiling.
    ///
    /// Scope mirrors the in-RAM `recall_scoped`: `Some(id)` → that workspace's +
    /// global memories; `None` → all memories.
    ///
    /// **Dimension guard:** `vector_distance_cos` *errors* on a dimension
    /// mismatch, so a single differently-sized embedding (e.g. left over from a
    /// prior embedding model) would abort the whole scan. Rows are therefore
    /// filtered to `octet_length(embedding) = query_dims * 4` — only same-dimension
    /// vectors are scored, which is also the only meaningful comparison. NULL
    /// embeddings are skipped.
    ///
    /// Returns `(memory_id, cosine_distance)` pairs, distance in `0.0..=2.0`
    /// (0 = identical). Cosine *similarity* is `1.0 - distance`.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError`] if the query fails.
    ///
    /// # Panics
    ///
    /// Panics if `query_embedding` is empty or `limit` is zero (programmer errors).
    pub async fn search_by_embedding_sql(
        &self,
        query_embedding: &[f32],
        limit: usize,
        workspace_id: Option<&str>,
    ) -> Result<Vec<(String, f64)>, StorageError> {
        assert!(
            !query_embedding.is_empty(),
            "query embedding must not be empty"
        );
        assert!(limit > 0, "limit must be positive");

        let query_blob: Vec<u8> = query_embedding
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let query_bytes = i64::try_from(query_blob.len()).unwrap_or(i64::MAX);
        let limit_i64 = i64::try_from(limit).unwrap_or(i64::MAX);

        let conn = self.conn.lock().await;
        // Push the scope branch up: two fixed statements rather than dynamic SQL.
        let mut rows = match workspace_id {
            Some(ws) => {
                conn.query(
                    "SELECT memory_id, vector_distance_cos(embedding, ?1) AS dist \
                     FROM memories \
                     WHERE embedding IS NOT NULL AND octet_length(embedding) = ?2 \
                       AND (workspace_id = ?3 OR workspace_id IS NULL) \
                     ORDER BY dist ASC LIMIT ?4",
                    turso::params![query_blob, query_bytes, ws, limit_i64],
                )
                .await?
            }
            None => {
                conn.query(
                    "SELECT memory_id, vector_distance_cos(embedding, ?1) AS dist \
                     FROM memories \
                     WHERE embedding IS NOT NULL AND octet_length(embedding) = ?2 \
                     ORDER BY dist ASC LIMIT ?3",
                    turso::params![query_blob, query_bytes, limit_i64],
                )
                .await?
            }
        };

        let mut out: Vec<(String, f64)> = Vec::new();
        while let Some(row) = rows.next().await? {
            let id: String = row.get(0)?;
            let dist: f64 = row.get(1)?;
            out.push((id, dist));
        }
        drop(rows);

        debug_assert!(
            out.len() <= limit,
            "returned more neighbours than the limit"
        );
        debug_assert!(
            out.windows(2).all(|w| w[0].1 <= w[1].1),
            "distances must be non-decreasing (ORDER BY dist ASC)"
        );
        Ok(out)
    }

    /// Update content text for a memory entry, **invalidating its vector**.
    ///
    /// Changing the text necessarily invalidates the embedding: a vector
    /// describes the words that produced it, so a row whose content moved on
    /// while its vector did not is findable by text it no longer contains, and
    /// silently — nothing about the row looks wrong. That is the same class as
    /// the 2026-08 stale-binding incident, and it is why there is no variant of
    /// this call that keeps the old vector.
    ///
    /// `embedding = NULL` is the ordinary queued-for-backfill state the loader
    /// already understands (`convert_rows` counts these as awaiting embedding),
    /// so the drain re-embeds the new text and the row is unsearchable rather
    /// than wrongly searchable in the meantime.
    ///
    /// The in-RAM half of this was fixed in 2026-08 (`VectorStore::update_content`
    /// clears `entry.embedding` and the buckets); this is the durable half,
    /// without which a restart reloaded the stale vector from disk and undid it.
    ///
    /// A caller that has a NEW vector for the new text writes it immediately
    /// afterwards with [`Self::update_embedding`]; a crash between the two
    /// leaves the row queued, which is the safe direction.
    ///
    /// Returns `true` when a row was updated.
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the update fails.
    pub async fn update_content(&self, memory_id: &str, content: &str) -> Result<bool, StorageError> {
        let conn = self.conn.lock().await;
        let result = conn
            .execute(
                "UPDATE memories SET content = ?1, embedding = NULL, embedding_model = NULL, \
                 updated_at = datetime('now') WHERE memory_id = ?2",
                turso::params![content, memory_id],
            )
            .await?;
        drop(conn);
        Ok(result > 0)
    }

    /// Update the embedding (and the model that produced it) for a memory.
    ///
    /// The save path is insert-then-update-on-conflict, and the update branch
    /// wrote only content and FSRS — so an embedding computed for an EXISTING
    /// row never reached disk. Every backfill after an embedding-model change
    /// therefore ran again on the next boot: 739 rows re-embedded on each
    /// restart, forever, with the log cheerfully reporting success each time.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError`] if the statement fails.
    pub async fn update_embedding(
        &self,
        memory_id: &str,
        embedding: &[f32],
        embedding_model: Option<&str>,
    ) -> Result<bool, StorageError> {
        let bytes: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();
        let conn = self.conn.lock().await;
        let result = conn
            .execute(
                "UPDATE memories SET embedding = ?1, embedding_model = ?2,                  updated_at = datetime('now') WHERE memory_id = ?3",
                turso::params![bytes, embedding_model, memory_id],
            )
            .await?;
        drop(conn);
        Ok(result > 0)
    }

    /// Delete multiple memories by their `memory_id`s, destroying their embeddings
    /// on disk (see [`Self::delete`] for why a plain `DELETE` leaks them).
    ///
    /// Each row is overwritten with `zeroblob`s and deleted, then the WAL is
    /// checkpoint-truncated **once** for the whole batch — the stale pre-overwrite
    /// frames of every removed row are discarded together, which is both correct
    /// (no ghost survives) and far cheaper than one checkpoint per row (the batch
    /// path a dream cycle takes when it folds duplicates).
    ///
    /// # Errors
    ///
    /// Returns [`StorageError`] if any overwrite or delete statement fails.
    ///
    /// # Panics
    ///
    /// Panics if any id in `ids` is empty (a programmer error).
    pub async fn bulk_delete(&self, ids: &[&str]) -> Result<u64, StorageError> {
        if ids.is_empty() {
            return Ok(0);
        }
        let conn = self.conn.lock().await;
        let mut deleted = 0u64;
        for id in ids {
            assert!(!id.is_empty(), "memory_id must not be empty");
            Self::overwrite_sensitive_columns(&conn, id).await?;
            // Delete tags first
            conn.execute(
                "DELETE FROM memory_tags WHERE memory_id = ?1",
                turso::params![*id],
            )
            .await?;
            conn.execute(
                "DELETE FROM memory_chunks WHERE memory_id = ?1",
                turso::params![*id],
            )
            .await?;
            // Buckets and queued work are children by the same convention, and
            // an orphaned bucket is worse than an orphaned chunk: it is a live
            // vector for text that no longer exists, so a deleted memory would
            // keep matching queries forever.
            conn.execute(
                "DELETE FROM memory_vectors WHERE memory_id = ?1",
                turso::params![*id],
            )
            .await?;
            conn.execute(
                "DELETE FROM memory_chunk_vectors WHERE memory_id = ?1",
                turso::params![*id],
            )
            .await?;
            conn.execute(
                "DELETE FROM embedding_queue WHERE memory_id = ?1",
                turso::params![*id],
            )
            .await?;
            let n = conn
                .execute(
                    "DELETE FROM memories WHERE memory_id = ?1",
                    turso::params![*id],
                )
                .await?;
            deleted += n as u64;
        }
        // One checkpoint for the whole batch (not per row).
        Self::checkpoint_truncate(&conn).await;
        // Held from the first overwrite through the checkpoint: the guard is
        // the transaction.
        drop(conn);
        debug_assert!(
            deleted <= ids.len() as u64,
            "cannot delete more rows than ids given"
        );
        Ok(deleted)
    }

    /// Overwrite a memory row's sensitive columns (`embedding`/`content`/`metadata`)
    /// with **same-length** `zeroblob`s so the record byte-size is unchanged and the
    /// newest WAL frame for its page carries zeros rather than the plaintext/vector.
    /// The caller is responsible for the subsequent `DELETE` and a WAL checkpoint.
    async fn overwrite_sensitive_columns(
        conn: &Connection,
        memory_id: &str,
    ) -> Result<(), StorageError> {
        // `octet_length` gives the exact stored byte count; nullable columns
        // COALESCE to 0 (a NULL column has no ghost to clear).
        conn.execute(
            "UPDATE memories SET \
                 embedding = zeroblob(COALESCE(octet_length(embedding), 0)), \
                 content = zeroblob(octet_length(content)), \
                 metadata = zeroblob(COALESCE(octet_length(metadata), 0)) \
             WHERE memory_id = ?1",
            turso::params![memory_id],
        )
        .await?;
        // The chunk rows hold the SAME secrets, split up: each carries a slice
        // of the content and its own embedding, and an embedding is invertible
        // back to its source text. Zeroing only the parent would leave the whole
        // memory reconstructible from its chunks — the Ghost-Vectors hole
        // reopened through a second table.
        conn.execute(
            "UPDATE memory_chunks SET \
                 embedding = zeroblob(COALESCE(octet_length(embedding), 0)), \
                 content = zeroblob(octet_length(content)) \
             WHERE memory_id = ?1",
            turso::params![memory_id],
        )
        .await?;
        // Per-model buckets are the same hole a third and fourth time. Each
        // bucket is an independent inversion of the same text, so leaving one
        // model's vector behind reconstructs the memory just as well as the one
        // we zeroed.
        conn.execute(
            "UPDATE memory_vectors SET embedding = zeroblob(octet_length(embedding)) \
             WHERE memory_id = ?1",
            turso::params![memory_id],
        )
        .await?;
        conn.execute(
            "UPDATE memory_chunk_vectors SET embedding = zeroblob(octet_length(embedding)) \
             WHERE memory_id = ?1",
            turso::params![memory_id],
        )
        .await?;
        Ok(())
    }

    /// Delete a memory, destroying its embedding on disk rather than leaking it.
    ///
    /// A plain `DELETE` only unlinks the row — Turso (like SQLite) does **not**
    /// zero freed pages, has no `secure_delete` pragma and no `VACUUM`, and in
    /// 0.6.1 keeps *all* committed data in the WAL (it does not checkpoint on
    /// close). So the deleted memory's embedding `f32` BLOB survives verbatim in
    /// the `-wal` file, and an embedding is invertible back to its source text by
    /// a Vec2Text-class model ("Ghost Vectors", arXiv 2606.18497) — a real
    /// exposure for a local-first product whose whole database sits in the user's
    /// filesystem and gets backed up.
    ///
    /// Two steps close it, both verified end to end in
    /// `crates/nanna-storage/tests/secure_delete.rs`:
    /// 1. Overwrite `embedding`/`content`/`metadata` with **same-length**
    ///    `zeroblob`s (keeps the record byte-size identical), so the newest WAL
    ///    frame for the row's page carries zeros, not the plaintext/vector.
    /// 2. `PRAGMA wal_checkpoint(TRUNCATE)` — collapse the WAL into the main file
    ///    (the latest, zeroed page image wins) **and truncate it to zero**,
    ///    discarding the stale pre-overwrite frame that still held the original
    ///    embedding. Without the truncate, the zeroed frame merely sits *behind*
    ///    the original in the same growing WAL and both remain on disk.
    ///
    /// The checkpoint is best-effort: a concurrent reader can make it report
    /// `busy`, in which case the stale frame lingers until a later checkpoint
    /// succeeds — the row is already deleted and overwritten regardless.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError`] if the overwrite or delete statements fail. A
    /// failed WAL checkpoint is logged, not returned.
    ///
    /// # Panics
    ///
    /// Panics if `memory_id` is empty (a programmer error — every stored memory
    /// has a non-empty id).
    pub async fn delete(&self, memory_id: &str) -> Result<bool, StorageError> {
        assert!(!memory_id.is_empty(), "memory_id must not be empty");
        let conn = self.conn.lock().await;

        // Step 1: overwrite the sensitive columns so the newest page frame is zeros.
        Self::overwrite_sensitive_columns(&conn, memory_id).await?;

        // Delete tags first
        conn.execute(
            "DELETE FROM memory_tags WHERE memory_id = ?1",
            turso::params![memory_id],
        )
        .await?;

        // Chunks are children by convention only: nothing in this crate sets
        // `PRAGMA foreign_keys = ON`, so the REFERENCES clause enforces nothing
        // and the cascade is ours to write. Without it a deleted memory keeps
        // matching queries through its orphaned chunk vectors.
        conn.execute(
            "DELETE FROM memory_chunks WHERE memory_id = ?1",
            turso::params![memory_id],
        )
        .await?;
        // Same convention, same reason: an orphaned bucket is a live vector for
        // text that no longer exists, which keeps a deleted memory matching
        // queries indefinitely.
        conn.execute(
            "DELETE FROM memory_vectors WHERE memory_id = ?1",
            turso::params![memory_id],
        )
        .await?;
        conn.execute(
            "DELETE FROM memory_chunk_vectors WHERE memory_id = ?1",
            turso::params![memory_id],
        )
        .await?;
        conn.execute(
            "DELETE FROM embedding_queue WHERE memory_id = ?1",
            turso::params![memory_id],
        )
        .await?;

        // Delete memory
        let result = conn
            .execute(
                "DELETE FROM memories WHERE memory_id = ?1",
                turso::params![memory_id],
            )
            .await?;

        debug_assert!(result <= 1, "memory_id is unique, at most one row deleted");

        // Step 2: collapse + truncate the WAL so the pre-overwrite frame is gone.
        Self::checkpoint_truncate(&conn).await;
        // Held from the overwrite through the checkpoint: the guard is the
        // transaction.
        drop(conn);

        Ok(result > 0)
    }

    /// Run `PRAGMA wal_checkpoint(TRUNCATE)` on `conn`, draining its status row.
    ///
    /// Best-effort by contract: logs and returns on any error or a `busy` result
    /// rather than propagating — the caller's row mutation has already committed,
    /// and a failed checkpoint only delays reclaiming the stale WAL frame.
    async fn checkpoint_truncate(conn: &Connection) {
        // A pragma that returns rows must go through `query`, not `execute`
        // (`execute` errors with "unexpected row during execution"). Leaving the
        // `Rows` cursor open on the shared connection would silently swallow the
        // next write, so it is drained and dropped before returning.
        match conn.query("PRAGMA wal_checkpoint(TRUNCATE)", ()).await {
            Ok(mut rows) => {
                let busy = match rows.next().await {
                    Ok(Some(row)) => row.get_value(0).ok().and_then(|v| match v {
                        turso::Value::Integer(i) => Some(i),
                        _ => None,
                    }),
                    _ => None,
                };
                drop(rows);
                if busy == Some(1) {
                    tracing::warn!(
                        "secure delete: WAL checkpoint reported busy; stale frame will clear on a later checkpoint"
                    );
                }
            }
            Err(e) => tracing::warn!("secure delete: WAL checkpoint failed: {e}"),
        }
    }
}

/// Config repository (key-value store)
pub struct ConfigRepository {
    conn: Arc<Mutex<Connection>>,
}

impl ConfigRepository {
    pub const fn new(conn: Arc<Mutex<Connection>>) -> Self {
        Self { conn }
    }

    /// The raw value stored under `key`, if any.
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the query fails or the value does
    /// not decode as text.
    pub async fn get(&self, key: &str) -> Result<Option<String>, StorageError> {
        let conn = self.conn.lock().await;

        let mut rows = conn
            .query("SELECT value FROM config WHERE key = ?1", turso::params![key])
            .await?;

        let value = if let Some(row) = rows.next().await? {
            Some(row.get(0)?)
        } else {
            None
        };
        // Held until the cursor is gone: an open `Rows` on the shared
        // connection swallows later writes.
        drop(rows);
        drop(conn);
        Ok(value)
    }

    /// Store `value` under `key`, replacing any previous value.
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the upsert fails.
    pub async fn set(&self, key: &str, value: &str) -> Result<(), StorageError> {
        let conn = self.conn.lock().await;

        conn.execute(
            "INSERT INTO config (key, value, updated_at) VALUES (?1, ?2, datetime('now'))
             ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = datetime('now')",
            turso::params![key, value],
        )
        .await?;
        drop(conn);

        Ok(())
    }

    /// Remove `key`; removing an absent key is not an error.
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the delete fails.
    pub async fn delete(&self, key: &str) -> Result<(), StorageError> {
        let conn = self.conn.lock().await;
        conn.execute("DELETE FROM config WHERE key = ?1", turso::params![key])
            .await?;
        drop(conn);
        Ok(())
    }

    /// The value under `key`, decoded from JSON.
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the read fails, or
    /// [`StorageError::Serialization`] if the stored text is not JSON that
    /// deserializes into `T`.
    pub async fn get_json<T: serde::de::DeserializeOwned>(
        &self,
        key: &str,
    ) -> Result<Option<T>, StorageError> {
        if let Some(value) = self.get(key).await? {
            Ok(Some(serde_json::from_str(&value)?))
        } else {
            Ok(None)
        }
    }

    /// Store `value` under `key` as JSON.
    ///
    /// `T: Sync` because `value` is borrowed across the write's `.await`;
    /// without it the returned future is not `Send`.
    ///
    /// # Errors
    /// Returns [`StorageError::Serialization`] if `T`'s `Serialize` impl fails
    /// (e.g. a map with non-string keys), or [`StorageError::Database`] if the
    /// write fails.
    pub async fn set_json<T: serde::Serialize + Sync>(&self, key: &str, value: &T) -> Result<(), StorageError> {
        let json = serde_json::to_string(value)?;
        self.set(key, &json).await
    }
}

/// Cron job repository
pub struct CronJobRepository {
    conn: Arc<Mutex<Connection>>,
}

impl CronJobRepository {
    pub const fn new(conn: Arc<Mutex<Connection>>) -> Self {
        Self { conn }
    }

    /// Create a job, or overwrite its schedule, task, enabled flag, next run,
    /// and metadata if `job_id` already exists.
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the upsert or the read-back fails,
    /// or [`StorageError::NotFound`] if the row cannot be read back.
    pub async fn create(&self, job: NewCronJob) -> Result<CronJob, StorageError> {
        let conn = self.conn.lock().await;

        let task_json = job.task.to_string();
        let metadata_json = job.metadata.as_ref().map(std::string::ToString::to_string);

        conn.execute(
            "INSERT INTO cron_jobs (job_id, schedule, task, enabled, next_run, metadata)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(job_id) DO UPDATE SET 
                schedule = excluded.schedule,
                task = excluded.task,
                enabled = excluded.enabled,
                next_run = excluded.next_run,
                metadata = excluded.metadata",
            turso::params![
                job.job_id.as_str(),
                job.schedule.as_str(),
                task_json.as_str(),
                job.enabled,
                job.next_run.as_deref(),
                metadata_json.as_deref(),
            ],
        )
        .await?;

        drop(conn);
        self.get(&job.job_id).await
    }

    /// Fetch one job by id.
    ///
    /// # Errors
    /// Returns [`StorageError::NotFound`] if no job has `job_id`, or
    /// [`StorageError::Database`] if the query fails or a column does not
    /// decode. An unparseable stored task loads as `null`, and unparseable
    /// metadata as `None`.
    pub async fn get(&self, job_id: &str) -> Result<CronJob, StorageError> {
        let conn = self.conn.lock().await;

        let mut rows = conn
            .query(
                "SELECT id, job_id, schedule, task, enabled, last_run, next_run, created_at, metadata
                 FROM cron_jobs WHERE job_id = ?1",
                turso::params![job_id],
            )
            .await?;

        let job = if let Some(row) = rows.next().await? {
            let task_str: String = row.get(3)?;
            let metadata_str: Option<String> = row.get(8)?;
            let enabled: i64 = row.get(4)?;

            Ok(CronJob {
                id: row.get(0)?,
                job_id: row.get(1)?,
                schedule: row.get(2)?,
                task: serde_json::from_str(&task_str).unwrap_or_default(),
                enabled: enabled != 0,
                last_run: row.get(5)?,
                next_run: row.get(6)?,
                created_at: row.get(7)?,
                metadata: metadata_str.and_then(|s| serde_json::from_str(&s).ok()),
            })
        } else {
            Err(StorageError::NotFound(format!("CronJob: {job_id}")))
        };
        // Held until the cursor is gone: an open `Rows` on the shared
        // connection swallows later writes.
        drop(rows);
        drop(conn);
        job
    }

    /// Every enabled job, soonest `next_run` first.
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the query fails or a column does
    /// not decode. Unparseable task/metadata JSON loads as `null`/`None`.
    pub async fn list_enabled(&self) -> Result<Vec<CronJob>, StorageError> {
        let conn = self.conn.lock().await;

        let mut rows = conn
            .query(
                "SELECT id, job_id, schedule, task, enabled, last_run, next_run, created_at, metadata
                 FROM cron_jobs WHERE enabled = 1 ORDER BY next_run ASC",
                (),
            )
            .await?;

        let mut jobs = Vec::new();
        while let Some(row) = rows.next().await? {
            let task_str: String = row.get(3)?;
            let metadata_str: Option<String> = row.get(8)?;
            let enabled: i64 = row.get(4)?;

            jobs.push(CronJob {
                id: row.get(0)?,
                job_id: row.get(1)?,
                schedule: row.get(2)?,
                task: serde_json::from_str(&task_str).unwrap_or_default(),
                enabled: enabled != 0,
                last_run: row.get(5)?,
                next_run: row.get(6)?,
                created_at: row.get(7)?,
                metadata: metadata_str.and_then(|s| serde_json::from_str(&s).ok()),
            });
        }
        // Held until the cursor is gone: an open `Rows` on the shared
        // connection swallows later writes.
        drop(rows);
        drop(conn);

        Ok(jobs)
    }

    /// Every job, newest first.
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the query fails or a column does
    /// not decode. Unparseable task/metadata JSON loads as `null`/`None`.
    pub async fn list_all(&self) -> Result<Vec<CronJob>, StorageError> {
        let conn = self.conn.lock().await;

        let mut rows = conn
            .query(
                "SELECT id, job_id, schedule, task, enabled, last_run, next_run, created_at, metadata
                 FROM cron_jobs ORDER BY created_at DESC",
                (),
            )
            .await?;

        let mut jobs = Vec::new();
        while let Some(row) = rows.next().await? {
            let task_str: String = row.get(3)?;
            let metadata_str: Option<String> = row.get(8)?;
            let enabled: i64 = row.get(4)?;

            jobs.push(CronJob {
                id: row.get(0)?,
                job_id: row.get(1)?,
                schedule: row.get(2)?,
                task: serde_json::from_str(&task_str).unwrap_or_default(),
                enabled: enabled != 0,
                last_run: row.get(5)?,
                next_run: row.get(6)?,
                created_at: row.get(7)?,
                metadata: metadata_str.and_then(|s| serde_json::from_str(&s).ok()),
            });
        }
        // Held until the cursor is gone: an open `Rows` on the shared
        // connection swallows later writes.
        drop(rows);
        drop(conn);

        Ok(jobs)
    }

    /// Record a run time and the next scheduled one.
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the update fails.
    pub async fn update_last_run(&self, job_id: &str, last_run: &str, next_run: Option<&str>) -> Result<(), StorageError> {
        let conn = self.conn.lock().await;

        conn.execute(
            "UPDATE cron_jobs SET last_run = ?1, next_run = ?2 WHERE job_id = ?3",
            turso::params![last_run, next_run, job_id],
        )
        .await?;
        drop(conn);

        Ok(())
    }

    /// Enable or disable a job.
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the update fails.
    pub async fn set_enabled(&self, job_id: &str, enabled: bool) -> Result<(), StorageError> {
        let conn = self.conn.lock().await;

        conn.execute(
            "UPDATE cron_jobs SET enabled = ?1 WHERE job_id = ?2",
            turso::params![enabled, job_id],
        )
        .await?;
        drop(conn);

        Ok(())
    }

    /// Delete a job; returns `true` when a row was removed.
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the delete fails.
    pub async fn delete(&self, job_id: &str) -> Result<bool, StorageError> {
        let conn = self.conn.lock().await;

        let result = conn
            .execute(
                "DELETE FROM cron_jobs WHERE job_id = ?1",
                turso::params![job_id],
            )
            .await?;
        drop(conn);

        Ok(result > 0)
    }
}

/// Job run history repository
pub struct JobRunRepository {
    conn: Arc<Mutex<Connection>>,
}

impl JobRunRepository {
    pub const fn new(conn: Arc<Mutex<Connection>>) -> Self {
        Self { conn }
    }

    /// Record a new job run
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the insert or the read-back
    /// fails, or [`StorageError::NotFound`] if the read-back finds no row.
    pub async fn create(&self, run: NewJobRun) -> Result<JobRun, StorageError> {
        let conn = self.conn.lock().await;

        conn.execute(
            "INSERT INTO job_runs (job_id, started_at, finished_at, success, output, error, duration_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            turso::params![
                run.job_id.as_str(),
                run.started_at.as_str(),
                run.finished_at.as_deref(),
                run.success,
                run.output.as_deref(),
                run.error.as_deref(),
                run.duration_ms,
            ],
        )
        .await?;

        // Get the inserted run
        let mut rows = conn
            .query(
                "SELECT id, job_id, started_at, finished_at, success, output, error, duration_ms
                 FROM job_runs ORDER BY id DESC LIMIT 1",
                (),
            )
            .await?;

        let job_run = if let Some(row) = rows.next().await? {
            let success: i64 = row.get(4)?;
            Ok(JobRun {
                id: row.get(0)?,
                job_id: row.get(1)?,
                started_at: row.get(2)?,
                finished_at: row.get(3)?,
                success: success != 0,
                output: row.get(5)?,
                error: row.get(6)?,
                duration_ms: row.get(7)?,
            })
        } else {
            Err(StorageError::NotFound("JobRun just created".to_string()))
        };
        // Held from the insert through the read-back: "newest row" is only
        // this run while no other writer can interleave, and the cursor must
        // be gone before the guard is.
        drop(rows);
        drop(conn);
        job_run
    }

    /// Get runs for a specific job
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the query fails or a column does
    /// not decode.
    pub async fn list_by_job(&self, job_id: &str, limit: i64) -> Result<Vec<JobRun>, StorageError> {
        let conn = self.conn.lock().await;

        let mut rows = conn
            .query(
                "SELECT id, job_id, started_at, finished_at, success, output, error, duration_ms
                 FROM job_runs WHERE job_id = ?1 ORDER BY started_at DESC LIMIT ?2",
                turso::params![job_id, limit],
            )
            .await?;

        let mut runs = Vec::new();
        while let Some(row) = rows.next().await? {
            let success: i64 = row.get(4)?;
            runs.push(JobRun {
                id: row.get(0)?,
                job_id: row.get(1)?,
                started_at: row.get(2)?,
                finished_at: row.get(3)?,
                success: success != 0,
                output: row.get(5)?,
                error: row.get(6)?,
                duration_ms: row.get(7)?,
            });
        }
        // Held until the cursor is gone: an open `Rows` on the shared
        // connection swallows later writes.
        drop(rows);
        drop(conn);

        Ok(runs)
    }

    /// Get recent runs across all jobs
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the query fails or a column does
    /// not decode.
    pub async fn list_recent(&self, limit: i64) -> Result<Vec<JobRun>, StorageError> {
        let conn = self.conn.lock().await;

        let mut rows = conn
            .query(
                "SELECT id, job_id, started_at, finished_at, success, output, error, duration_ms
                 FROM job_runs ORDER BY started_at DESC LIMIT ?1",
                turso::params![limit],
            )
            .await?;

        let mut runs = Vec::new();
        while let Some(row) = rows.next().await? {
            let success: i64 = row.get(4)?;
            runs.push(JobRun {
                id: row.get(0)?,
                job_id: row.get(1)?,
                started_at: row.get(2)?,
                finished_at: row.get(3)?,
                success: success != 0,
                output: row.get(5)?,
                error: row.get(6)?,
                duration_ms: row.get(7)?,
            });
        }
        // Held until the cursor is gone: an open `Rows` on the shared
        // connection swallows later writes.
        drop(rows);
        drop(conn);

        Ok(runs)
    }

    /// Delete old runs for a job (keep only the most recent N)
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the delete fails.
    pub async fn cleanup(&self, job_id: &str, keep: i64) -> Result<i64, StorageError> {
        let conn = self.conn.lock().await;

        let result = conn
            .execute(
                "DELETE FROM job_runs WHERE job_id = ?1 AND id NOT IN (
                    SELECT id FROM job_runs WHERE job_id = ?1 ORDER BY started_at DESC LIMIT ?2
                )",
                turso::params![job_id, keep],
            )
            .await?;
        drop(conn);

        // A row count cannot exceed `i64::MAX` (rowids are i64), so the
        // fallback is unreachable.
        Ok(i64::try_from(result).unwrap_or(i64::MAX))
    }

    /// Delete all runs for a job
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the delete fails.
    pub async fn delete_by_job(&self, job_id: &str) -> Result<i64, StorageError> {
        let conn = self.conn.lock().await;

        let result = conn
            .execute(
                "DELETE FROM job_runs WHERE job_id = ?1",
                turso::params![job_id],
            )
            .await?;
        drop(conn);

        // A row count cannot exceed `i64::MAX` (rowids are i64), so the
        // fallback is unreachable.
        Ok(i64::try_from(result).unwrap_or(i64::MAX))
    }
}

// ═══ Workspace Repository ═══

pub struct WorkspaceRepository {
    conn: Arc<Mutex<Connection>>,
}

impl WorkspaceRepository {
    pub const fn new(conn: Arc<Mutex<Connection>>) -> Self {
        Self { conn }
    }

    /// List all registered workspaces
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the query fails or a column does
    /// not decode.
    pub async fn list(&self) -> Result<Vec<WorkspaceRecord>, StorageError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT id, name, path, active, created_at, last_accessed FROM workspaces ORDER BY last_accessed DESC",
                turso::params![],
            )
            .await?;

        let mut workspaces = Vec::new();
        while let Some(row) = rows.next().await? {
            workspaces.push(WorkspaceRecord {
                id: row.get(0)?,
                name: row.get(1)?,
                path: row.get(2)?,
                active: row.get::<i64>(3)? != 0,
                created_at: row.get(4)?,
                last_accessed: row.get(5)?,
            });
        }
        // Held until the cursor is gone: an open `Rows` on the shared
        // connection swallows later writes.
        drop(rows);
        drop(conn);
        Ok(workspaces)
    }

    /// Get a workspace by ID
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the query fails or a column does
    /// not decode.
    pub async fn get(&self, id: &str) -> Result<Option<WorkspaceRecord>, StorageError> {
        let conn = self.conn.lock().await;
        let id = id.to_string();
        let mut rows = conn
            .query(
                "SELECT id, name, path, active, created_at, last_accessed FROM workspaces WHERE id = ?1",
                turso::params![id],
            )
            .await?;

        let workspace = if let Some(row) = rows.next().await? {
            Some(WorkspaceRecord {
                id: row.get(0)?,
                name: row.get(1)?,
                path: row.get(2)?,
                active: row.get::<i64>(3)? != 0,
                created_at: row.get(4)?,
                last_accessed: row.get(5)?,
            })
        } else {
            None
        };
        // Held until the cursor is gone: an open `Rows` on the shared
        // connection swallows later writes.
        drop(rows);
        drop(conn);
        Ok(workspace)
    }

    /// Insert or update a workspace (upsert by path)
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the upsert fails — including when
    /// a workspace with a different `id` already holds `record.path`, since
    /// `path` is unique and the conflict target is `id`.
    pub async fn upsert(&self, record: &WorkspaceRecord) -> Result<(), StorageError> {
        let conn = self.conn.lock().await;
        let id = record.id.clone();
        let name = record.name.clone();
        let path = record.path.clone();
        let active = i64::from(record.active);
        conn.execute(
            "INSERT INTO workspaces (id, name, path, active, last_accessed)
             VALUES (?1, ?2, ?3, ?4, datetime('now'))
             ON CONFLICT(id) DO UPDATE SET
               name = excluded.name,
               path = excluded.path,
               active = excluded.active,
               last_accessed = datetime('now')",
            turso::params![id, name, path, active],
        )
        .await?;
        drop(conn);
        Ok(())
    }

    /// Remove a workspace by ID
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the delete fails.
    pub async fn delete(&self, id: &str) -> Result<bool, StorageError> {
        let conn = self.conn.lock().await;
        let id = id.to_string();
        let affected = conn
            .execute(
                "DELETE FROM workspaces WHERE id = ?1",
                turso::params![id],
            )
            .await?;
        drop(conn);
        Ok(affected > 0)
    }

    /// Clear the active flag on all workspaces
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the update fails.
    pub async fn clear_active(&self) -> Result<(), StorageError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE workspaces SET active = 0",
            turso::params![],
        )
        .await?;
        drop(conn);
        Ok(())
    }

    /// Set a workspace as active (clears others first)
    ///
    /// Returns `false` when no workspace has `id` — every flag is cleared
    /// regardless.
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if either update fails.
    pub async fn set_active(&self, id: &str) -> Result<bool, StorageError> {
        let conn = self.conn.lock().await;
        // Clear all
        conn.execute("UPDATE workspaces SET active = 0", turso::params![])
            .await?;
        // Set this one
        let id = id.to_string();
        let affected = conn
            .execute(
                "UPDATE workspaces SET active = 1, last_accessed = datetime('now') WHERE id = ?1",
                turso::params![id],
            )
            .await?;
        // Held across clear-all and set-one: the guard is the transaction, so
        // no reader sees the moment between them with nothing active.
        drop(conn);
        Ok(affected > 0)
    }
}

/// Append-only access to `memory_events` — the episodic stream.
///
/// The API deliberately offers no single-row `update` or `delete`: an event
/// log that can be edited cannot answer "what did I know, and when", which is
/// the only reason to keep raw episodes beside the consolidated facts.
/// Compaction of *old windows* is a separate, whole-window operation and does
/// not belong to this type.
pub struct MemoryEventRepository {
    conn: Arc<Mutex<Connection>>,
}

/// Largest page any single timeline query will return.
///
/// Sized against the worst case a caller can hit: one event carries up to
/// 8 KiB of content plus a 768-dim f32 embedding (3 KiB), so 1000 rows is
/// ~11 MB resident. Above that a range scan stops being a query and becomes
/// a bulk load, which is what `bulk_load` on the memory side already proved
/// too expensive to do implicitly.
pub const MAX_EVENT_PAGE: usize = 1000;

impl MemoryEventRepository {
    pub const fn new(conn: Arc<Mutex<Connection>>) -> Self {
        Self { conn }
    }

    /// Append one episode.
    ///
    /// Idempotent on `event_id`: a redelivered event is ignored rather than
    /// duplicated, so an at-least-once producer (a channel retrying a message)
    /// cannot inflate the timeline.
    ///
    /// Returns `true` when the row was newly inserted.
    ///
    /// # Errors
    /// Returns [`StorageError`] if the insert fails.
    pub async fn append(&self, event: &NewMemoryEvent) -> Result<bool, StorageError> {
        // Validated, not asserted. This runs inside the daemon's spawned
        // turns, where a panic does not surface as a failure -- it wedges the
        // turn silently, which is how a `&text[..200]` slice once cost a whole
        // benchmark leg. A bad value here is worth an error, never a crash.
        if event.event_id.is_empty() {
            return Err(StorageError::Invalid(
                "an episode must carry a non-empty event_id".to_string(),
            ));
        }
        if !event.salience.is_finite() || !(0.0..=1.0).contains(&event.salience) {
            return Err(StorageError::Invalid(format!(
                "salience is a normalized weight in [0,1], got {}",
                event.salience
            )));
        }
        debug_assert!(
            event.content_len_chars >= 0,
            "a pre-truncation length cannot be negative"
        );
        debug_assert!(
            event.content_len_chars >= i64::try_from(event.content.chars().count()).unwrap_or(0),
            "the pre-truncation length can never be shorter than what is stored"
        );

        let blob: Option<Vec<u8>> = event
            .embedding
            .as_ref()
            .map(|e| e.iter().flat_map(|f| f.to_le_bytes()).collect());
        let source_ids = serde_json::to_string(&event.source_ids)?;

        let conn = self.conn.lock().await;
        let affected = conn
            .execute(
                APPEND_MEMORY_EVENT,
                turso::params![
                    event.event_id.clone(),
                    event.ts_unix_ms,
                    event.kind.clone(),
                    event.workspace_id.clone(),
                    event.content.clone(),
                    event.content_len_chars,
                    blob,
                    event.embedding_model.clone(),
                    f64::from(event.salience),
                    source_ids,
                ],
            )
            .await?;
        drop(conn);
        Ok(affected > 0)
    }

    /// Events with `start_ms <= ts < end_ms`, oldest first, capped at `limit`.
    ///
    /// Half-open on purpose: adjacent windows tile the axis exactly once, so
    /// resampling into fixed buckets cannot double-count an event that lands
    /// on a boundary.
    ///
    /// # Errors
    /// Returns [`StorageError`] if the query fails.
    pub async fn range(
        &self,
        start_ms: i64,
        end_ms: i64,
        workspace_id: Option<&str>,
        limit: usize,
    ) -> Result<Vec<MemoryEventRow>, StorageError> {
        check_window(start_ms, end_ms)?;
        check_page(limit)?;

        let limit_i64 = i64::try_from(limit).unwrap_or(i64::MAX);
        let conn = self.conn.lock().await;
        let mut rows = match workspace_id {
            Some(ws) => {
                conn.query(
                    RANGE_MEMORY_EVENTS_SCOPED,
                    turso::params![start_ms, end_ms, ws.to_string(), limit_i64],
                )
                .await?
            }
            None => {
                conn.query(
                    RANGE_MEMORY_EVENTS,
                    turso::params![start_ms, end_ms, limit_i64],
                )
                .await?
            }
        };

        // Drained fully before the guard drops: an open cursor on this shared
        // connection silently swallows the next write.
        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            out.push(row_to_memory_event(&row)?);
        }
        debug_assert!(out.len() <= limit, "the page cap must hold");
        Ok(out)
    }

    /// How many events fall in `[start_ms, end_ms)`.
    ///
    /// Separate from [`Self::range`] because sizing a resample window must not
    /// require materializing it — that is exactly the read the page cap
    /// forbids.
    ///
    /// # Errors
    /// Returns [`StorageError`] if the query fails.
    pub async fn count_in_range(&self, start_ms: i64, end_ms: i64) -> Result<i64, StorageError> {
        check_window(start_ms, end_ms)?;
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(COUNT_MEMORY_EVENTS, turso::params![start_ms, end_ms])
            .await?;
        let count = match rows.next().await? {
            Some(row) => row.get::<i64>(0)?,
            None => 0,
        };
        // Held until the cursor is gone: an open `Rows` on the shared
        // connection swallows later writes.
        drop(rows);
        drop(conn);
        debug_assert!(count >= 0, "a row count cannot be negative");
        Ok(count)
    }

    /// The `limit` most recent events, newest first.
    ///
    /// # Errors
    /// Returns [`StorageError`] if the query fails.
    pub async fn recent(&self, limit: usize) -> Result<Vec<MemoryEventRow>, StorageError> {
        check_page(limit)?;
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                RECENT_MEMORY_EVENTS,
                turso::params![i64::try_from(limit).unwrap_or(i64::MAX)],
            )
            .await?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            out.push(row_to_memory_event(&row)?);
        }
        // Held until the cursor is gone: an open `Rows` on the shared
        // connection swallows later writes.
        drop(rows);
        drop(conn);
        debug_assert!(out.len() <= limit, "the page cap must hold");
        Ok(out)
    }

    /// Up to `limit` events whose lineage names `source_id`, oldest first —
    /// one card's series (`task:<id>`) for the dream fold.
    ///
    /// Matched against the stored JSON array with the id's own quotes, so
    /// `task:12` never matches `task:120`.
    ///
    /// # Errors
    /// Returns [`StorageError`] if `limit` exceeds [`MAX_EVENT_PAGE`] or the
    /// query fails.
    pub async fn for_source(
        &self,
        source_id: &str,
        limit: usize,
    ) -> Result<Vec<MemoryEventRow>, StorageError> {
        check_page(limit)?;
        if source_id.is_empty() || source_id.contains('"') {
            return Err(StorageError::Invalid(format!(
                "a source id to match must be non-empty and unquoted, got {source_id:?}"
            )));
        }
        let pattern = format!("%\"{source_id}\"%");
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                MEMORY_EVENTS_FOR_SOURCE,
                turso::params![pattern, i64::try_from(limit).unwrap_or(i64::MAX)],
            )
            .await?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            out.push(row_to_memory_event(&row)?);
        }
        drop(rows);
        drop(conn);
        debug_assert!(out.len() <= limit, "the page cap must hold");
        Ok(out)
    }
}

/// Reject a backwards window rather than returning an empty page for it: an
/// inverted window is a caller bug, and answering it with "no events" hides
/// the bug behind a plausible result.
fn check_window(start_ms: i64, end_ms: i64) -> Result<(), StorageError> {
    if start_ms > end_ms {
        return Err(StorageError::Invalid(format!(
            "a time window cannot end before it starts ({start_ms} > {end_ms})"
        )));
    }
    Ok(())
}

/// Refuse an oversized page instead of quietly serving `MAX_EVENT_PAGE` of
/// them -- a silently clamped page reads as "that was all of it".
fn check_page(limit: usize) -> Result<(), StorageError> {
    if limit > MAX_EVENT_PAGE {
        return Err(StorageError::Invalid(format!(
            "a timeline page is bounded at {MAX_EVENT_PAGE} events, asked for {limit}"
        )));
    }
    Ok(())
}

fn row_to_memory_event(row: &turso::Row) -> Result<MemoryEventRow, StorageError> {
    let source_ids: String = row.get(11)?;
    Ok(MemoryEventRow {
        id: row.get(0)?,
        event_id: row.get(1)?,
        ts_unix_ms: row.get(2)?,
        kind: row.get(3)?,
        workspace_id: row.get(4)?,
        content: row.get(5)?,
        content_len_chars: row.get(6)?,
        embedding: row.get::<Option<Vec<u8>>>(7)?.as_deref().map(decode_embedding),
        embedding_model: row.get(8)?,
        salience: f32_from_real(row.get::<f64>(9)?),
        created_at: row.get(10)?,
        // A row whose lineage JSON is unreadable is still a real episode; the
        // timeline loses the link, not the event. Failing the whole page here
        // would make one bad row hide every good one behind it.
        source_ids: serde_json::from_str(&source_ids).unwrap_or_default(),
    })
}

const APPEND_MEMORY_EVENT: &str = "
INSERT OR IGNORE INTO memory_events
    (event_id, ts_unix_ms, kind, workspace_id, content, content_len_chars,
     embedding, embedding_model, salience, source_ids)
VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)";

const RANGE_MEMORY_EVENTS: &str = "
SELECT id, event_id, ts_unix_ms, kind, workspace_id, content,
       content_len_chars, embedding, embedding_model, salience, created_at, source_ids
FROM memory_events
WHERE ts_unix_ms >= ?1 AND ts_unix_ms < ?2
ORDER BY ts_unix_ms ASC, id ASC
LIMIT ?3";

const RANGE_MEMORY_EVENTS_SCOPED: &str = "
SELECT id, event_id, ts_unix_ms, kind, workspace_id, content,
       content_len_chars, embedding, embedding_model, salience, created_at, source_ids
FROM memory_events
WHERE ts_unix_ms >= ?1 AND ts_unix_ms < ?2 AND workspace_id = ?3
ORDER BY ts_unix_ms ASC, id ASC
LIMIT ?4";

const MEMORY_EVENTS_FOR_SOURCE: &str = "
SELECT id, event_id, ts_unix_ms, kind, workspace_id, content,
       content_len_chars, embedding, embedding_model, salience, created_at, source_ids
FROM memory_events
WHERE source_ids LIKE ?1
ORDER BY ts_unix_ms ASC, id ASC
LIMIT ?2";

const RECENT_MEMORY_EVENTS: &str = "
SELECT id, event_id, ts_unix_ms, kind, workspace_id, content,
       content_len_chars, embedding, embedding_model, salience, created_at, source_ids
FROM memory_events
ORDER BY ts_unix_ms DESC, id DESC
LIMIT ?1";

const COUNT_MEMORY_EVENTS: &str = "
SELECT COUNT(*) FROM memory_events WHERE ts_unix_ms >= ?1 AND ts_unix_ms < ?2";
