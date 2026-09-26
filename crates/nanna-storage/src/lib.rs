#![warn(clippy::all)]
#![warn(clippy::pedantic, clippy::nursery)]

//! Persistent storage for Nanna using Turso
//!
//! Turso is a Rust-native `SQLite` implementation.

mod members;
mod migrations;
mod models;
mod recovery;
mod repositories;
mod task_events;
pub mod task_filter;
mod tasks;

pub use members::*;
pub use models::*;
pub use recovery::*;
pub use repositories::*;
pub use task_events::*;
pub use tasks::*;

use std::sync::Arc;
use thiserror::Error;
use tokio::sync::Mutex;
use tracing::{info, warn};
use turso::{Builder, Connection};

#[derive(Error, Debug)]
pub enum StorageError {
    #[error("Database error: {0}")]
    Database(#[from] turso::Error),
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("Not found: {0}")]
    NotFound(String),
    #[error("Migration error: {0}")]
    Migration(String),
    #[error("Invalid: {0}")]
    Invalid(String),
}



/// Storage configuration
#[derive(Debug, Clone)]
pub struct StorageConfig {
    /// Path to local database file (or ":memory:")
    pub path: String,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            path: "nanna.db".to_string(),
        }
    }
}

/// Main storage interface
pub struct Storage {
    conn: Arc<Mutex<Connection>>,
    /// Where task lifecycle events go, once someone is listening.
    ///
    /// Write-once rather than a constructor argument: the daemon's event bus
    /// does not exist yet when storage is opened, and `Storage` is shared as
    /// an `Arc` immediately afterwards, so there is no later `&mut self` to
    /// set it through. A `OnceLock` also makes "attached twice" a detectable
    /// mistake instead of a silently dropped first sink.
    task_events: std::sync::OnceLock<Arc<dyn TaskEventSink>>,
}

impl Storage {
    /// Create a new storage instance
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the database at `config.path`
    /// cannot be opened or connected to, or if a pending migration fails.
    pub async fn new(config: &StorageConfig) -> Result<Self, StorageError> {
        info!("Opening database: {}", config.path);
        let db = Builder::new_local(&config.path).build().await?;
        let conn = db.connect()?;
        let storage = Self {
            conn: Arc::new(Mutex::new(conn)),
            task_events: std::sync::OnceLock::new(),
        };

        storage.migrate().await?;
        // Released once migrations have run; from here on the connection's
        // own reference keeps the database open.
        drop(db);
        Ok(storage)
    }

    /// Create an in-memory storage (for testing)
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the in-memory database cannot be
    /// created or connected to, or if a migration fails.
    pub async fn in_memory() -> Result<Self, StorageError> {
        let db = Builder::new_local(":memory:").build().await?;
        let conn = db.connect()?;
        let storage = Self {
            conn: Arc::new(Mutex::new(conn)),
            task_events: std::sync::OnceLock::new(),
        };
        storage.migrate().await?;
        // Released once migrations have run; from here on the connection's
        // own reference keeps the database open.
        drop(db);
        Ok(storage)
    }

    /// Run database migrations
    async fn migrate(&self) -> Result<(), StorageError> {
        let conn = self.conn.lock().await;

        // Create migrations table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS _migrations (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                applied_at TEXT NOT NULL
            )",
            (),
        )
        .await?;

        // Boxed as `dyn Future + Send` so proving `Storage::new` is `Send` stops
        // here. Unboxed, the proof walks every nested migration future and
        // overruns the solver's depth limit (the future-incompatible
        // `recursion_depth_exceeding_limit`, rust#159228) in every crate and
        // test that holds a `Storage` future — one allocation per boot instead.
        let run: std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<(), StorageError>> + Send + '_>,
        > = Box::pin(apply_migrations(&conn, migrations::MIGRATIONS));
        let result = run.await;
        // Held across every migration and its `_migrations` record.
        drop(conn);
        result
    }

    /// Get connection reference
    #[must_use] 
    pub const fn conn(&self) -> &Arc<Mutex<Connection>> {
        &self.conn
    }

    // Repository accessors
    #[must_use] 
    pub fn sessions(&self) -> SessionRepository {
        SessionRepository::new(self.conn.clone())
    }

    #[must_use] 
    pub fn messages(&self) -> MessageRepository {
        MessageRepository::new(self.conn.clone())
    }

    #[must_use] 
    pub fn memories(&self) -> MemoryRepository {
        MemoryRepository::new(self.conn.clone())
    }

    /// The episodic stream beneath [`Self::memories`] — raw events on a
    /// wall-clock axis, append-only.
    #[must_use]
    pub fn memory_events(&self) -> MemoryEventRepository {
        MemoryEventRepository::new(self.conn.clone())
    }

    #[must_use] 
    pub fn config_store(&self) -> ConfigRepository {
        ConfigRepository::new(self.conn.clone())
    }

    #[must_use] 
    pub fn cron_jobs(&self) -> CronJobRepository {
        CronJobRepository::new(self.conn.clone())
    }

    #[must_use] 
    pub fn job_runs(&self) -> JobRunRepository {
        JobRunRepository::new(self.conn.clone())
    }

    #[must_use]
    pub fn workspaces(&self) -> WorkspaceRepository {
        WorkspaceRepository::new(self.conn.clone())
    }

    /// Board members: the human, every agent, and each board's Task Management
    /// Agent. `tasks.assignee` holds one of their ids.
    #[must_use]
    pub fn members(&self) -> MemberRepository {
        MemberRepository::new(self.conn.clone())
    }

    /// Attach the sink task lifecycle events are published to.
    ///
    /// Returns `false` if a sink was already attached, in which case the new
    /// one is dropped and the existing one keeps receiving. Callers that care
    /// should treat `false` as a wiring bug, not as a condition to retry.
    pub fn set_task_events(&self, sink: Arc<dyn TaskEventSink>) -> bool {
        self.task_events.set(sink).is_ok()
    }

    #[must_use]
    pub fn tasks(&self) -> TaskRepository {
        // A fresh repository per call, so the sink is handed over each time
        // rather than held by whoever happened to build the first one.
        TaskRepository::new(self.conn.clone())
            .with_events(self.task_events.get().cloned())
    }

    // =========================================================================
    // Convenience methods for GUI
    // =========================================================================

    /// Create a new GUI session
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the insert or the read-back
    /// query fails, or [`StorageError::NotFound`] if the new row cannot be
    /// read back.
    pub async fn create_gui_session(&self, name: &str) -> Result<Session, StorageError> {
        self.create_gui_session_with_workspace(name, None).await
    }

    /// Create a new GUI session with optional workspace
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the insert or the read-back
    /// query fails, or [`StorageError::NotFound`] if the new row cannot be
    /// read back.
    pub async fn create_gui_session_with_workspace(
        &self,
        name: &str,
        workspace_id: Option<&str>,
    ) -> Result<Session, StorageError> {
        let session_id = uuid::Uuid::new_v4().to_string();
        let conn = self.conn.lock().await;

        conn.execute(
            "INSERT INTO sessions (session_id, channel, user_id, workspace_id, name, metadata) 
             VALUES (?1, 'gui', NULL, ?2, ?3, ?4)",
            turso::params![
                session_id.as_str(),
                workspace_id,
                name,
                // Serialized, never formatted: a name with a quote or a
                // backslash produced metadata that was not JSON at all.
                serde_json::json!({ "name": name }).to_string().as_str()
            ],
        )
        .await?;

        drop(conn);
        self.sessions().get(&session_id).await
    }

    /// List sessions for GUI (with names from metadata)
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the query fails or a row does not
    /// decode.
    pub async fn list_gui_sessions(&self, limit: i64) -> Result<Vec<Session>, StorageError> {
        self.sessions().list_recent(limit).await
    }

    /// List sessions for GUI filtered by workspace
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the query fails or a row does not
    /// decode.
    pub async fn list_gui_sessions_by_workspace(&self, workspace_id: Option<&str>, limit: i64) -> Result<Vec<Session>, StorageError> {
        self.sessions().list_by_workspace(workspace_id, limit).await
    }

    /// Get messages for a session
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the query fails or a row does not
    /// decode.
    pub async fn get_session_messages(&self, session_id: &str, limit: i64) -> Result<Vec<Message>, StorageError> {
        self.messages().get_by_session(session_id, limit).await
    }

    /// Add a message to a session
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the insert, the session touch, or
    /// the read-back fails, or [`StorageError::NotFound`] if the read-back
    /// finds no row.
    pub async fn add_message(&self, session_id: &str, role: &str, content: &str) -> Result<Message, StorageError> {
        self.messages().create(NewMessage {
            session_id: session_id.to_string(),
            role: role.to_string(),
            content: content.to_string(),
            content_type: "text".to_string(),
            tool_use_id: None,
            tokens_in: None,
            tokens_out: None,
            metadata: None,
        }).await
    }

    /// Add a message with tool calls to a session
    /// Tool calls are stored in the metadata field as JSON
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the insert, the session touch, or
    /// the read-back fails, or [`StorageError::NotFound`] if the read-back
    /// finds no row.
    pub async fn add_message_with_tool_calls(
        &self,
        session_id: &str,
        role: &str,
        content: &str,
        tool_calls: Option<serde_json::Value>,
    ) -> Result<Message, StorageError> {
        let metadata = tool_calls.map(|tc| serde_json::json!({ "tool_calls": tc }));
        self.messages().create(NewMessage {
            session_id: session_id.to_string(),
            role: role.to_string(),
            content: content.to_string(),
            content_type: "text".to_string(),
            tool_use_id: None,
            tokens_in: None,
            tokens_out: None,
            metadata,
        }).await
    }

    /// Count messages in a session
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the query fails or the count does
    /// not decode as an integer.
    pub async fn count_session_messages(&self, session_id: &str) -> Result<i64, StorageError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?1",
                turso::params![session_id],
            )
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

    /// Update session timestamp
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the update fails.
    pub async fn touch_session(&self, session_id: &str) -> Result<(), StorageError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE sessions SET updated_at = datetime('now') WHERE session_id = ?1",
            turso::params![session_id],
        )
        .await?;
        drop(conn);
        Ok(())
    }

    /// Rename a session
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the update fails.
    pub async fn rename_session(&self, session_id: &str, name: &str) -> Result<(), StorageError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE sessions SET metadata = json_set(COALESCE(metadata, '{}'), '$.name', ?1), updated_at = datetime('now') WHERE session_id = ?2",
            turso::params![name, session_id],
        )
        .await?;
        drop(conn);
        Ok(())
    }

    /// Delete a session and its messages
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if either delete fails; a failure on
    /// the session row leaves its messages already deleted.
    pub async fn delete_session(&self, session_id: &str) -> Result<(), StorageError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "DELETE FROM messages WHERE session_id = ?1",
            turso::params![session_id],
        )
        .await?;
        conn.execute(
            "DELETE FROM sessions WHERE session_id = ?1",
            turso::params![session_id],
        )
        .await?;
        // Held across both deletes: the guard is the transaction.
        drop(conn);
        Ok(())
    }

    /// Get session name - prefers name field, falls back to metadata, then generates one
    pub fn get_session_name(session: &Session) -> String {
        // First try the name column
        if let Some(name) = &session.name
            && !name.is_empty() {
                return name.clone();
            }
        // Fall back to metadata
        session.metadata
            .as_ref()
            .and_then(|m| m.get("name"))
            .and_then(|n| n.as_str()).map_or_else(|| {
                let end = truncate_boundary(&session.session_id, 8);
                format!("Session {}", &session.session_id[..end])
            }, String::from)
    }

    /// Update session's workspace
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the update fails.
    pub async fn set_session_workspace(&self, session_id: &str, workspace_id: Option<&str>) -> Result<(), StorageError> {
        self.sessions().update_workspace(session_id, workspace_id).await
    }
}

/// Apply every migration in `migrations` that `_migrations` does not record,
/// each inside its own transaction.
///
/// A migration and its `_migrations` row commit together or not at all. Before
/// this, a failure between two statements left half a migration applied and
/// unrecorded, so every later boot re-ran it from the top and died on the first
/// statement that had already landed (`duplicate column name` is the usual
/// one). Turso rolls DDL back (verified: a rolled-back `CREATE TABLE` and
/// `ADD COLUMN` both vanish), so the transaction is the real fix; the
/// `ADD COLUMN` probe covers databases that were half-migrated before it.
///
/// # Errors
/// Returns the first failing statement's error, after rolling its migration back.
pub(crate) async fn apply_migrations(
    conn: &Connection,
    migrations: &[(&str, &str)],
) -> Result<(), StorageError> {
    for (name, sql) in migrations {
        debug_assert!(!name.is_empty(), "a migration has a name");
        let mut rows = conn
            .query(
                "SELECT 1 FROM _migrations WHERE name = ?1",
                turso::params![*name],
            )
            .await?;
        let applied = rows.next().await?.is_some();
        // An open cursor on this connection swallows later writes.
        drop(rows);
        if applied {
            continue;
        }
        info!("Running migration: {}", name);
        conn.execute("BEGIN", ()).await?;
        match apply_one_migration(conn, name, sql).await {
            Ok(()) => {
                conn.execute("COMMIT", ()).await?;
            }
            Err(e) => {
                if let Err(rollback) = conn.execute("ROLLBACK", ()).await {
                    warn!("Rolling back migration {name} failed too: {rollback}");
                }
                return Err(e);
            }
        }
    }
    Ok(())
}

/// One migration's statements plus its `_migrations` row, inside the caller's
/// transaction.
async fn apply_one_migration(conn: &Connection, name: &str, sql: &str) -> Result<(), StorageError> {
    // Split by a lexer that knows comments and quotes, not by every ';'.
    for statement in migrations::split_statements(sql) {
        if let Some((table, column)) = migrations::add_column_target(&statement)
            && column_exists(conn, &table, &column).await?
        {
            warn!(
                "Migration {name}: {table}.{column} already exists (a partial earlier run); skipping its ADD COLUMN"
            );
            continue;
        }
        conn.execute(&statement, ()).await?;
    }
    conn.execute(
        "INSERT INTO _migrations (name, applied_at) VALUES (?1, datetime('now'))",
        turso::params![name],
    )
    .await?;
    Ok(())
}

/// Whether `table` currently has a column named `column` (case-insensitive,
/// as SQLite identifiers are).
async fn column_exists(conn: &Connection, table: &str, column: &str) -> Result<bool, StorageError> {
    let mut rows = conn
        .query(
            "SELECT 1 FROM pragma_table_info(?1) WHERE lower(name) = lower(?2)",
            turso::params![table, column],
        )
        .await?;
    let exists = rows.next().await?.is_some();
    drop(rows);
    Ok(exists)
}

/// Byte offset at or below `max_bytes` that a slice can end on.
///
/// A session id is whatever the caller stored: `nanna chat --session <ID>` and
/// `upsert_daemon_session` both take the string verbatim, so a fixed byte index
/// can land past the end of a short id or inside a multi-byte character, and
/// `str` indexing panics on either.
const fn truncate_boundary(s: &str, max_bytes: usize) -> usize {
    if s.len() <= max_bytes {
        return s.len();
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    end
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The cutoff is compared in `created_at`'s own format. An RFC 3339 cutoff
    /// sorted `' '` below `'T'`, so a row one second inside the window, on the
    /// cutoff DAY, was deleted with the rest of that day.
    #[tokio::test]
    async fn the_tool_log_prune_keeps_everything_inside_the_window() {
        let storage = Storage::in_memory().await.expect("storage");
        let conn = storage.conn().lock().await;
        for (tool, age) in [
            ("stale", "-31 days"),
            ("edge", "-30 days"),
            ("fresh", "-1 hours"),
        ] {
            conn.execute(
                "INSERT INTO tool_call_log \
                 (tool_name, success, duration_ms, error_message, session_id, created_at) \
                 VALUES (?1, 1, 5, '', '', datetime('now', ?2, '+2 seconds'))",
                turso::params![tool, age],
            )
            .await
            .expect("insert");
        }
        drop(conn);

        let deleted = storage.prune_tool_call_log(30).await.expect("prune");
        let left: Vec<String> = storage
            .get_tool_call_log(None, 10)
            .await
            .expect("log")
            .into_iter()
            .map(|e| e.tool_name)
            .collect();
        assert_eq!(deleted, 1, "the real count, not a placeholder 0");
        assert!(
            left.contains(&"edge".to_string()),
            "inside the window by 2 s: {left:?}"
        );
        assert!(!left.contains(&"stale".to_string()), "{left:?}");
    }

    /// A short-circuited call never ran. The hourly aggregate knew that; the
    /// daily one counted it as a success, so the two disagreed.
    #[tokio::test]
    async fn both_aggregates_agree_on_a_short_circuited_call() {
        let storage = Storage::in_memory().await.expect("storage");
        for short_circuited in [true, false] {
            storage
                .log_tool_call(&NewToolCall {
                    tool_name: "exec",
                    success: true,
                    short_circuited,
                    duration_ms: 3,
                    output_size: 0,
                    error_message: None,
                    session_id: None,
                })
                .await
                .expect("log");
        }
        let hourly = storage
            .get_tool_stats_hourly(Some("exec"), 2)
            .await
            .expect("hourly");
        let daily = storage
            .get_tool_stats_daily(Some("exec"), 2)
            .await
            .expect("daily");
        let counts = |b: &ToolStatsTimeBucket| (b.call_count, b.success_count, b.failure_count);
        assert_eq!(counts(&hourly[0]), (2, 1, 0), "{hourly:?}");
        assert_eq!(counts(&daily[0]), counts(&hourly[0]), "{daily:?}");
    }

    /// Session metadata is serialized, not formatted: a quote in the name used
    /// to produce a string that was not JSON.
    #[tokio::test]
    async fn a_session_name_with_quotes_is_stored_as_json() {
        let storage = Storage::in_memory().await.expect("storage");
        let name = r#"the "big" plan \ v2"#;
        let session = storage
            .create_gui_session_with_workspace(name, None)
            .await
            .expect("create");
        assert_eq!(
            session.metadata.as_ref().and_then(|m| m["name"].as_str()),
            Some(name),
            "{session:?}"
        );
    }

    async fn table_names(conn: &Connection) -> Vec<String> {
        let mut rows = conn
            .query(
                "SELECT name FROM sqlite_schema WHERE type = 'table' ORDER BY name",
                (),
            )
            .await
            .expect("schema");
        let mut names = Vec::new();
        while let Some(row) = rows.next().await.expect("row") {
            names.push(row.get::<String>(0).expect("name"));
        }
        names
    }

    async fn recorded(conn: &Connection, name: &str) -> bool {
        let mut rows = conn
            .query(
                "SELECT 1 FROM _migrations WHERE name = ?1",
                turso::params![name],
            )
            .await
            .expect("query");
        rows.next().await.expect("row").is_some()
    }

    /// A migration that fails part-way used to leave its first statements
    /// applied and itself unrecorded, so every later boot re-ran it and died on
    /// what had already landed. Now it commits whole or not at all.
    #[tokio::test]
    async fn a_failing_migration_leaves_no_trace() {
        let storage = Storage::in_memory().await.expect("storage");
        let conn = storage.conn().lock().await;
        let broken: &[(&str, &str)] = &[(
            "900_broken",
            "CREATE TABLE half_done (id INTEGER); ALTER TABLE half_done ADD COLUMN x TEXT; \
             INSERT INTO no_such_table VALUES (1);",
        )];
        assert!(apply_migrations(&conn, broken).await.is_err());
        assert!(
            !table_names(&conn).await.contains(&"half_done".to_string()),
            "rolled back"
        );
        assert!(!recorded(&conn, "900_broken").await);

        // The same name, fixed, applies cleanly on the next boot.
        let fixed: &[(&str, &str)] = &[(
            "900_broken",
            "CREATE TABLE half_done (id INTEGER); ALTER TABLE half_done ADD COLUMN x TEXT;",
        )];
        apply_migrations(&conn, fixed)
            .await
            .expect("the fixed migration applies");
        let is_recorded = recorded(&conn, "900_broken").await;
        drop(conn);
        assert!(is_recorded);
    }

    /// A database half-migrated before migrations ran in a transaction: the
    /// column landed, the record did not. The ADD COLUMN probe lets the rerun
    /// through instead of wedging every boot on `duplicate column name`.
    #[tokio::test]
    async fn a_half_applied_add_column_is_not_fatal() {
        let storage = Storage::in_memory().await.expect("storage");
        let conn = storage.conn().lock().await;
        conn.execute("CREATE TABLE legacy (id INTEGER)", ())
            .await
            .expect("create");
        conn.execute("ALTER TABLE legacy ADD COLUMN landed TEXT", ())
            .await
            .expect("the half that landed");
        let rerun: &[(&str, &str)] = &[(
            "901_partial",
            "ALTER TABLE legacy ADD COLUMN landed TEXT; ALTER TABLE legacy ADD COLUMN missing TEXT;",
        )];
        apply_migrations(&conn, rerun)
            .await
            .expect("the rerun completes");
        let is_recorded = recorded(&conn, "901_partial").await;
        let missing_added = column_exists(&conn, "legacy", "missing")
            .await
            .expect("probe");
        let landed_seen = column_exists(&conn, "LEGACY", "Landed")
            .await
            .expect("probe");
        drop(conn);
        assert!(is_recorded);
        assert!(missing_added, "the half that had not landed is applied");
        assert!(landed_seen, "the probe is case-insensitive");
    }

    /// Usage rolls up per day and per month, prices from the 1-hour column, and
    /// ignores what falls outside the window.
    #[tokio::test]
    async fn model_usage_rolls_up_by_day_and_month() {
        let storage = Storage::in_memory().await.expect("storage");
        for (model, input, writes, writes_1h) in [
            ("claude-opus-5", 100, 40, 10),
            ("claude-opus-5", 50, 0, 0),
            ("ollama/qwen3.5:9b", 7, 0, 0),
            ("claude-opus-5", 1, 0, 0),
        ] {
            storage
                .log_model_request(&NewModelRequest {
                    model,
                    success: true,
                    latency_ms: 10,
                    input_tokens: input,
                    output_tokens: 5,
                    cache_read_tokens: 0,
                    cache_creation_tokens: writes,
                    cache_creation_1h_tokens: writes_1h,
                    tier: None,
                    escalated: false,
                    session_id: Some("s"),
                })
                .await
                .expect("log");
        }
        // Row 3 happened three days ago; row 4 is older than any window asked for.
        let conn = storage.conn.lock().await;
        let three_days_ago = (chrono::Utc::now() - chrono::Duration::days(3))
            .format("%Y-%m-%d %H:%M:%S")
            .to_string();
        conn.execute(
            "UPDATE model_request_log SET created_at = ?1 WHERE id = 3",
            turso::params![three_days_ago.as_str()],
        )
        .await
        .expect("backdate");
        conn.execute(
            "UPDATE model_request_log SET created_at = '2001-01-01 00:00:00' WHERE id = 4",
            (),
        )
        .await
        .expect("backdate");
        drop(conn);

        let by_day = storage.model_usage_buckets(7, false).await.expect("rollup");
        assert_eq!(by_day.len(), 2, "{by_day:?}");
        let today = chrono::Utc::now().format("%Y-%m-%d").to_string();
        let opus = by_day
            .iter()
            .find(|b| b.model == "claude-opus-5")
            .expect("opus");
        assert_eq!(opus.period, today);
        assert_eq!(
            (
                opus.requests,
                opus.input_tokens,
                opus.cache_write_tokens,
                opus.cache_write_1h_tokens
            ),
            (2, 150, 40, 10)
        );
        let local = by_day
            .iter()
            .find(|b| b.model == "ollama/qwen3.5:9b")
            .expect("local");
        assert_eq!(local.period, three_days_ago[..10]);
        assert!(by_day[0].period <= by_day[1].period, "oldest first");

        let by_month = storage.model_usage_buckets(7, true).await.expect("rollup");
        assert!(by_month.iter().all(|b| b.period.len() == 7), "{by_month:?}");
        assert!(
            by_month.iter().all(|b| b.period != "2001-01"),
            "outside the window"
        );
        let by_session = storage
            .model_usage_by(7, UsagePeriod::Session)
            .await
            .expect("rollup");
        assert!(
            by_session.iter().all(|b| b.period == "s"),
            "every logged row named session s: {by_session:?}"
        );
        let everything = storage
            .model_usage_buckets(u32::MAX, true)
            .await
            .expect("clamped");
        assert!(
            everything.iter().all(|b| b.period != "2001-01"),
            "clamped to a year"
        );
    }
    #[tokio::test]
    async fn test_storage_creation() {
        let storage = Storage::in_memory().await.unwrap();

        let session = storage
            .sessions()
            .create("test-session", "cli", None)
            .await
            .unwrap();

        assert_eq!(session.session_id, "test-session");
    }

    #[tokio::test]
    async fn test_persistence_across_restarts() {
        // Create a temp file path
        let temp_dir = std::env::temp_dir();
        let db_path = temp_dir.join(format!("nanna_test_{}.db", std::process::id()));
        let db_path_str = db_path.to_string_lossy().to_string();

        // Clean up any existing test db
        let _ = std::fs::remove_file(&db_path);

        // Create storage, write data, then drop it
        {
            let config = StorageConfig { path: db_path_str.clone() };
            let storage = Storage::new(&config).await.unwrap();
            
            storage
                .sessions()
                .create("persistent-session", "cli", None)
                .await
                .unwrap();
        }

        // Reopen storage and verify data persisted
        {
            let config = StorageConfig { path: db_path_str.clone() };
            let storage = Storage::new(&config).await.unwrap();
            
            // get() returns Result<Session> - throws NotFound if missing
            let session = storage
                .sessions()
                .get("persistent-session")
                .await
                .expect("Session should persist across restarts");
            
            assert_eq!(session.session_id, "persistent-session");
        }

        // Cleanup
        let _ = std::fs::remove_file(&db_path);
    }

    fn sample_session(session_id: &str) -> Session {
        Session {
            id: 1,
            session_id: session_id.into(),
            channel: "cli".into(),
            user_id: None,
            created_at: String::new(),
            updated_at: String::new(),
            metadata: None,
            workspace_id: None,
            name: None,
        }
    }

    #[test]
    fn get_session_name_survives_multibyte_session_id() {
        // `nanna chat --session <ID>` stores the flag verbatim and writes
        // neither a name nor metadata, so the generated fallback is the only
        // reachable branch for such a session. Byte 8 of this id lands inside
        // the em dash (bytes 6..9) — the same cut that killed the daemon.
        assert_eq!(
            Storage::get_session_name(&sample_session("abcdef—ghij")),
            "Session abcdef"
        );

        // The same defect from the other side: an id shorter than the limit
        // puts the index out of range before boundaries even matter.
        assert_eq!(
            Storage::get_session_name(&sample_session("abc")),
            "Session abc"
        );

        // An ASCII id still keeps exactly the eight bytes it always did.
        assert_eq!(
            Storage::get_session_name(&sample_session("0123456789abcdef")),
            "Session 01234567"
        );
    }

    fn sample_new_memory(id: &str, content: &str) -> NewMemory {
        NewMemory {
            memory_id: id.into(),
            content: content.into(),
            embedding: Some(vec![0.1, 0.2, 0.3]),
            embedding_model: Some("test".into()),
            session_id: None,
            metadata: None,
            tags: vec![],
            workspace_id: None,
            fsrs_stability: 1.0,
            fsrs_difficulty: 5.0,
            fsrs_last_access: 0,
            fsrs_access_count: 0,
            fsrs_importance: 1.0,
            fsrs_storage_strength: 1.0,
            fsrs_generation: 0,
        }
    }

    #[test]
    fn is_corruption_error_matches_corruption_messages() {
        // The classifier's contract is a case-insensitive substring match on the
        // rendered message (the form it takes once it crosses into
        // `MemoryError::Persistence(String)`). NotFound is just a convenient
        // String-carrying variant to exercise that contract.
        assert!(is_corruption_error(&StorageError::NotFound(
            "inconsistent overflow chain observed during payload read".into()
        )));
        assert!(is_corruption_error(&StorageError::NotFound(
            "database disk image is CORRUPT".into()
        )));
        assert!(is_corruption_error(&StorageError::NotFound("malformed database page".into())));
        assert!(!is_corruption_error(&StorageError::NotFound("session xyz missing".into())));
    }

    #[tokio::test]
    async fn bulk_load_salvage_matches_bulk_load_on_clean_db() {
        let storage = Storage::in_memory().await.unwrap();
        let repo = storage.memories();
        for i in 0..5 {
            repo.create(sample_new_memory(&format!("m{i}"), &format!("content {i}")))
                .await
                .unwrap();
        }
        let bulk = repo.bulk_load().await.unwrap();
        let report = repo.bulk_load_salvage().await.unwrap();

        assert_eq!(report.expected, 5);
        assert_eq!(report.corrupt_ids, [] as [i64; 0]);
        assert_eq!(report.memories.len(), bulk.len());
        // Same memory_ids in the same order (both ORDER BY id ASC) — the per-id
        // reconstruction is lossless on a clean DB.
        let bulk_ids: Vec<_> = bulk.iter().map(|m| m.memory_id.clone()).collect();
        let salv_ids: Vec<_> = report.memories.iter().map(|m| m.memory_id.clone()).collect();
        assert_eq!(bulk_ids, salv_ids);
        assert_eq!(report.memories[0].embedding, bulk[0].embedding);
    }
}

// =============================================================================
// Model Stats Persistence
// =============================================================================

/// Stored model statistics row (mirrors `nanna-agent::ModelStats` without the dependency)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StoredModelStats {
    pub model: String,
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub total_cache_read_tokens: u64,
    pub total_cache_creation_tokens: u64,
    /// The 1-hour share of `total_cache_creation_tokens` (billed at 2x input, not 1.25x).
    #[serde(default)]
    pub total_cache_creation_1h_tokens: u64,
    pub consecutive_failures: u32,
    pub last_success_epoch_ms: u64,
    pub last_failure_epoch_ms: u64,
    pub tier_successes_simple: u64,
    pub tier_successes_medium: u64,
    pub tier_successes_complex: u64,
    pub tier_failures_simple: u64,
    pub tier_failures_medium: u64,
    pub tier_failures_complex: u64,
    pub escalations: u64,
    pub latencies_ms: Vec<u64>,
    pub throughput_tps: Vec<f64>,
}

impl Storage {
    /// Save model stats to the database (upsert).
    ///
    /// Counters are stored in `INTEGER` columns by bit-reinterpretation
    /// (`cast_signed`), and [`Self::load_model_stats`] reverses it with
    /// `cast_unsigned`, so every `u64` round-trips exactly.
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if an upsert fails; rows for models
    /// earlier in `stats` are already saved by then.
    pub async fn save_model_stats(&self, stats: &[StoredModelStats]) -> Result<(), StorageError> {
        let conn = self.conn.lock().await;
        for s in stats {
            let latencies_json = serde_json::to_string(&s.latencies_ms)?;
            let throughput_json = serde_json::to_string(&s.throughput_tps)?;
            conn.execute(
                "INSERT INTO model_stats (
                    model, total_requests, successful_requests, failed_requests,
                    total_input_tokens, total_output_tokens,
                    total_cache_read_tokens, total_cache_creation_tokens,
                    consecutive_failures, last_success_epoch_ms, last_failure_epoch_ms,
                    tier_successes_simple, tier_successes_medium, tier_successes_complex,
                    tier_failures_simple, tier_failures_medium, tier_failures_complex,
                    escalations, latencies_ms_json, throughput_tps_json, total_cache_creation_1h_tokens, updated_at
                ) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19,?20,?21,datetime('now'))
                ON CONFLICT(model) DO UPDATE SET
                    total_requests=?2, successful_requests=?3, failed_requests=?4,
                    total_input_tokens=?5, total_output_tokens=?6,
                    total_cache_read_tokens=?7, total_cache_creation_tokens=?8,
                    consecutive_failures=?9, last_success_epoch_ms=?10, last_failure_epoch_ms=?11,
                    tier_successes_simple=?12, tier_successes_medium=?13, tier_successes_complex=?14,
                    tier_failures_simple=?15, tier_failures_medium=?16, tier_failures_complex=?17,
                    escalations=?18, latencies_ms_json=?19, throughput_tps_json=?20, total_cache_creation_1h_tokens=?21, updated_at=datetime('now')",
                turso::params![
                    s.model.clone(),
                    s.total_requests.cast_signed(),
                    s.successful_requests.cast_signed(),
                    s.failed_requests.cast_signed(),
                    s.total_input_tokens.cast_signed(),
                    s.total_output_tokens.cast_signed(),
                    s.total_cache_read_tokens.cast_signed(),
                    s.total_cache_creation_tokens.cast_signed(),
                    i64::from(s.consecutive_failures),
                    s.last_success_epoch_ms.cast_signed(),
                    s.last_failure_epoch_ms.cast_signed(),
                    s.tier_successes_simple.cast_signed(),
                    s.tier_successes_medium.cast_signed(),
                    s.tier_successes_complex.cast_signed(),
                    s.tier_failures_simple.cast_signed(),
                    s.tier_failures_medium.cast_signed(),
                    s.tier_failures_complex.cast_signed(),
                    s.escalations.cast_signed(),
                    latencies_json,
                    throughput_json,
                    s.total_cache_creation_1h_tokens.cast_signed()
                ],
            ).await?;
        }
        // Held for the whole batch: the guard is the transaction.
        drop(conn);
        Ok(())
    }

    /// Load all model stats from the database.
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the query fails or a column does
    /// not decode as the expected type. Unparseable latency/throughput JSON is
    /// not an error: it loads as an empty series.
    pub async fn load_model_stats(&self) -> Result<Vec<StoredModelStats>, StorageError> {
        let conn = self.conn.lock().await;
        let mut rows = conn.query(
            "SELECT model, total_requests, successful_requests, failed_requests,
                    total_input_tokens, total_output_tokens,
                    total_cache_read_tokens, total_cache_creation_tokens,
                    consecutive_failures, last_success_epoch_ms, last_failure_epoch_ms,
                    tier_successes_simple, tier_successes_medium, tier_successes_complex,
                    tier_failures_simple, tier_failures_medium, tier_failures_complex,
                    escalations, latencies_ms_json, throughput_tps_json, total_cache_creation_1h_tokens
             FROM model_stats",
            (),
        ).await?;

        let mut result = Vec::new();
        while let Some(row) = rows.next().await? {
            let latencies_json: String = row.get::<String>(18)?;
            let throughput_json: String = row.get::<String>(19)?;
            result.push(StoredModelStats {
                model: row.get::<String>(0)?,
                total_requests: row.get::<i64>(1)?.cast_unsigned(),
                successful_requests: row.get::<i64>(2)?.cast_unsigned(),
                failed_requests: row.get::<i64>(3)?.cast_unsigned(),
                total_input_tokens: row.get::<i64>(4)?.cast_unsigned(),
                total_output_tokens: row.get::<i64>(5)?.cast_unsigned(),
                total_cache_read_tokens: row.get::<i64>(6)?.cast_unsigned(),
                total_cache_creation_tokens: row.get::<i64>(7)?.cast_unsigned(),
                // The only writer stores `i64::from(u32)`, so the fallback is
                // unreachable for any row this crate wrote.
                consecutive_failures: u32::try_from(row.get::<i64>(8)?).unwrap_or(u32::MAX),
                last_success_epoch_ms: row.get::<i64>(9)?.cast_unsigned(),
                last_failure_epoch_ms: row.get::<i64>(10)?.cast_unsigned(),
                tier_successes_simple: row.get::<i64>(11)?.cast_unsigned(),
                tier_successes_medium: row.get::<i64>(12)?.cast_unsigned(),
                tier_successes_complex: row.get::<i64>(13)?.cast_unsigned(),
                tier_failures_simple: row.get::<i64>(14)?.cast_unsigned(),
                tier_failures_medium: row.get::<i64>(15)?.cast_unsigned(),
                tier_failures_complex: row.get::<i64>(16)?.cast_unsigned(),
                escalations: row.get::<i64>(17)?.cast_unsigned(),
                latencies_ms: serde_json::from_str(&latencies_json).unwrap_or_default(),
                throughput_tps: serde_json::from_str(&throughput_json).unwrap_or_default(),
                total_cache_creation_1h_tokens: row.get::<i64>(20)?.cast_unsigned(),
            });
        }
        // Held until the cursor is gone: an open `Rows` on the shared
        // connection swallows later writes.
        drop(rows);
        drop(conn);
        Ok(result)
    }

    /// Log a single model request observation (detailed per-request log).
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the insert fails.
    pub async fn log_model_request(&self, request: &NewModelRequest<'_>) -> Result<(), StorageError> {
        debug_assert!(
            request.cache_creation_1h_tokens <= request.cache_creation_tokens,
            "the 1-hour share is a subset of the write total"
        );
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO model_request_log (model, success, latency_ms, input_tokens, output_tokens,
                cache_read_tokens, cache_creation_tokens, cache_creation_1h_tokens, tier,
                escalated, session_id)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            turso::params![
                request.model,
                i64::from(request.success),
                request.latency_ms.cast_signed(),
                i64::from(request.input_tokens),
                i64::from(request.output_tokens),
                i64::from(request.cache_read_tokens),
                i64::from(request.cache_creation_tokens),
                i64::from(request.cache_creation_1h_tokens),
                request.tier.unwrap_or(""),
                i64::from(request.escalated),
                request.session_id.unwrap_or("")
            ],
        ).await?;
        drop(conn);
        Ok(())
    }

    /// Model usage summed per day (`YYYY-MM-DD`) or month (`YYYY-MM`) and
    /// model, oldest first, over the last `days` days.
    ///
    /// Rows are bounded by periods × models: `days` is clamped to 366, and the
    /// model set is whatever was configured and used.
    ///
    /// # Errors
    ///
    /// The query failed or a row did not have the expected shape.
    pub async fn model_usage_buckets(
        &self,
        days: u32,
        by_month: bool,
    ) -> Result<Vec<ModelUsageBucket>, StorageError> {
        let period = if by_month {
            UsagePeriod::Month
        } else {
            UsagePeriod::Day
        };
        self.model_usage_by(days, period).await
    }

    /// Model usage over the last `days` days grouped by `period` and model.
    /// For [`UsagePeriod::Session`] the bucket's `period` is the session id
    /// (empty for requests made outside any conversation).
    ///
    /// # Errors
    ///
    /// The query failed or a row did not have the expected shape.
    pub async fn model_usage_by(
        &self,
        days: u32,
        period: UsagePeriod,
    ) -> Result<Vec<ModelUsageBucket>, StorageError> {
        let days = days.clamp(1, USAGE_BUCKET_DAYS_MAX);
        // `created_at` is `datetime('now')`: `YYYY-MM-DD HH:MM:SS`, so a day or
        // month is a prefix of it and the cutoff compares as text. The column
        // is chosen from a closed enum, never from caller text.
        let group_expr = match period {
            UsagePeriod::Day => "substr(created_at, 1, 10)",
            UsagePeriod::Month => "substr(created_at, 1, 7)",
            UsagePeriod::Session => "COALESCE(session_id, '')",
        };
        let since = (chrono::Utc::now() - chrono::Duration::days(i64::from(days)))
            .format("%Y-%m-%d 00:00:00")
            .to_string();
        let sql = format!(
            "SELECT {group_expr} AS period, model,
                    CAST(COUNT(*) AS INTEGER),
                    CAST(SUM(input_tokens) AS INTEGER),
                    CAST(SUM(output_tokens) AS INTEGER),
                    CAST(SUM(cache_read_tokens) AS INTEGER),
                    CAST(SUM(cache_creation_tokens) AS INTEGER),
                    CAST(SUM(cache_creation_1h_tokens) AS INTEGER)
             FROM model_request_log
             WHERE created_at >= ?1
             GROUP BY period, model
             ORDER BY period ASC, model ASC"
        );
        let conn = self.conn.lock().await;
        let mut rows = conn.query(&sql, turso::params![since]).await?;
        let mut buckets = Vec::new();
        while let Some(row) = rows.next().await? {
            let count = |index: usize| row.get::<i64>(index).map(|n| u64::try_from(n).unwrap_or(0));
            buckets.push(ModelUsageBucket {
                period: row.get::<String>(0)?,
                model: row.get::<String>(1)?,
                requests: count(2)?,
                input_tokens: count(3)?,
                output_tokens: count(4)?,
                cache_read_tokens: count(5)?,
                cache_write_tokens: count(6)?,
                cache_write_1h_tokens: count(7)?,
            });
        }
        drop(rows);
        drop(conn);
        Ok(buckets)
    }

    // =========================================================================
    // Tool Stats
    // =========================================================================

    /// Log a single tool call (time-series data for graphs).
    ///
    /// `short_circuited` (P22 Tier 4): the harness answered with a breaker
    /// replay instead of dispatching. The row is still logged (with a
    /// `[short_circuited]` error marker so it stays distinguishable in the
    /// time series — the schema has no outcome column), but the hourly
    /// aggregate counts it as neither success nor failure: the tool never
    /// ran, and a wall of replays must not read as a broken tool.
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the log insert or either
    /// aggregate upsert fails. The statements are not rolled back, so a
    /// failure in an aggregate leaves the earlier rows written.
    pub async fn log_tool_call(&self, call: &NewToolCall<'_>) -> Result<(), StorageError> {
        let NewToolCall {
            tool_name,
            success,
            short_circuited,
            duration_ms,
            output_size,
            error_message,
            session_id,
        } = *call;
        let conn = self.conn.lock().await;
        let logged_error = if short_circuited {
            "[short_circuited]"
        } else {
            error_message.unwrap_or("")
        };
        conn.execute(
            "INSERT INTO tool_call_log (tool_name, success, duration_ms, output_size, error_message, session_id)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            turso::params![
                tool_name,
                i64::from(success),
                duration_ms.cast_signed(),
                // A buffer length never exceeds `isize::MAX`, so the fallback
                // is unreachable.
                i64::try_from(output_size).unwrap_or(i64::MAX),
                logged_error,
                session_id.unwrap_or("")
            ],
        ).await?;

        // One instant for both aggregates, so a call at 23:59:59.999 cannot
        // land in one day's hourly row and the next day's daily row.
        let now = chrono::Utc::now();
        let hour = now.format("%Y-%m-%dT%H:00:00").to_string();
        // A short-circuited call never ran: it is neither a success nor a
        // failure in EITHER aggregate. The daily one used to count it as
        // `success`, so the two views disagreed about the same calls.
        let success_incr = i64::from(success && !short_circuited);
        let failure_incr = i64::from(!success && !short_circuited);
        conn.execute(
            "INSERT INTO tool_stats_hourly (tool_name, hour, call_count, success_count, failure_count, total_duration_ms, avg_duration_ms, p95_duration_ms)
             VALUES (?1, ?2, 1, ?3, ?4, ?5, ?5, ?5)
             ON CONFLICT(tool_name, hour) DO UPDATE SET
                call_count = call_count + 1,
                success_count = success_count + ?3,
                failure_count = failure_count + ?4,
                total_duration_ms = total_duration_ms + ?5,
                avg_duration_ms = (total_duration_ms + ?5) / (call_count + 1),
                p95_duration_ms = MAX(p95_duration_ms, ?5)",
            turso::params![
                tool_name,
                hour,
                success_incr,
                failure_incr,
                duration_ms.cast_signed()
            ],
        ).await?;

        let day = now.format("%Y-%m-%d").to_string();
        conn.execute(
            "INSERT INTO tool_stats_daily (tool_name, day, call_count, success_count, failure_count, total_duration_ms, avg_duration_ms, p95_duration_ms)
             VALUES (?1, ?2, 1, ?3, ?4, ?5, ?5, ?5)
             ON CONFLICT(tool_name, day) DO UPDATE SET
                call_count = call_count + 1,
                success_count = success_count + ?3,
                failure_count = failure_count + ?4,
                total_duration_ms = total_duration_ms + ?5,
                avg_duration_ms = (total_duration_ms + ?5) / (call_count + 1),
                p95_duration_ms = MAX(p95_duration_ms, ?5)",
            turso::params![
                tool_name,
                day,
                success_incr,
                failure_incr,
                duration_ms.cast_signed()
            ],
        ).await?;
        // Held across the log row and both aggregates: the guard is the
        // transaction.
        drop(conn);

        Ok(())
    }

    /// Get hourly tool stats for a given time range (for graphs).
    /// Returns data for the last `hours` hours.
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the query fails or a column does
    /// not decode as the expected type.
    pub async fn get_tool_stats_hourly(
        &self,
        tool_name: Option<&str>,
        hours: u32,
    ) -> Result<Vec<ToolStatsTimeBucket>, StorageError> {
        let conn = self.conn.lock().await;
        let since = chrono::Utc::now() - chrono::Duration::hours(i64::from(hours));
        let since_str = since.format("%Y-%m-%dT%H:00:00").to_string();

        let mut rows = if let Some(name) = tool_name {
            conn.query(
                "SELECT tool_name, hour, call_count, success_count, failure_count, 
                        total_duration_ms, avg_duration_ms, p95_duration_ms
                 FROM tool_stats_hourly
                 WHERE tool_name = ?1 AND hour >= ?2
                 ORDER BY hour ASC",
                turso::params![name, since_str],
            ).await?
        } else {
            conn.query(
                "SELECT 'all' as tool_name, hour, 
                        CAST(SUM(call_count) AS INTEGER),
                        CAST(SUM(success_count) AS INTEGER),
                        CAST(SUM(failure_count) AS INTEGER),
                        CAST(SUM(total_duration_ms) AS INTEGER),
                        CAST(AVG(avg_duration_ms) AS INTEGER),
                        CAST(MAX(p95_duration_ms) AS INTEGER)
                 FROM tool_stats_hourly
                 WHERE hour >= ?1
                 GROUP BY hour
                 ORDER BY hour ASC",
                turso::params![since_str],
            ).await?
        };
        let mut result = Vec::new();
        while let Some(row) = rows.next().await? {
            result.push(ToolStatsTimeBucket {
                tool_name: row.get::<String>(0)?,
                period: row.get::<String>(1)?,
                call_count: row.get::<i64>(2)?.cast_unsigned(),
                success_count: row.get::<i64>(3)?.cast_unsigned(),
                failure_count: row.get::<i64>(4)?.cast_unsigned(),
                total_duration_ms: row.get::<i64>(5)?.cast_unsigned(),
                avg_duration_ms: row.get::<i64>(6)?.cast_unsigned(),
                p95_duration_ms: row.get::<i64>(7)?.cast_unsigned(),
            });
        }
        // Held until the cursor is gone: an open `Rows` on the shared
        // connection swallows later writes.
        drop(rows);
        drop(conn);
        Ok(result)
    }

    /// Get daily tool stats for a given time range (for long-term graphs).
    /// Returns data for the last `days` days.
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the query fails or a column does
    /// not decode as the expected type.
    pub async fn get_tool_stats_daily(
        &self,
        tool_name: Option<&str>,
        days: u32,
    ) -> Result<Vec<ToolStatsTimeBucket>, StorageError> {
        let conn = self.conn.lock().await;
        let since = chrono::Utc::now() - chrono::Duration::days(i64::from(days));
        let since_str = since.format("%Y-%m-%d").to_string();

        let mut rows = if let Some(name) = tool_name {
            conn.query(
                "SELECT tool_name, day, call_count, success_count, failure_count,
                        total_duration_ms, avg_duration_ms, p95_duration_ms
                 FROM tool_stats_daily
                 WHERE tool_name = ?1 AND day >= ?2
                 ORDER BY day ASC",
                turso::params![name, since_str],
            ).await?
        } else {
            conn.query(
                "SELECT 'all' as tool_name, day,
                        CAST(SUM(call_count) AS INTEGER),
                        CAST(SUM(success_count) AS INTEGER),
                        CAST(SUM(failure_count) AS INTEGER),
                        CAST(SUM(total_duration_ms) AS INTEGER),
                        CAST(AVG(avg_duration_ms) AS INTEGER),
                        CAST(MAX(p95_duration_ms) AS INTEGER)
                 FROM tool_stats_daily
                 WHERE day >= ?1
                 GROUP BY day
                 ORDER BY day ASC",
                turso::params![since_str],
            ).await?
        };
        let mut result = Vec::new();
        while let Some(row) = rows.next().await? {
            result.push(ToolStatsTimeBucket {
                tool_name: row.get::<String>(0)?,
                period: row.get::<String>(1)?,
                call_count: row.get::<i64>(2)?.cast_unsigned(),
                success_count: row.get::<i64>(3)?.cast_unsigned(),
                failure_count: row.get::<i64>(4)?.cast_unsigned(),
                total_duration_ms: row.get::<i64>(5)?.cast_unsigned(),
                avg_duration_ms: row.get::<i64>(6)?.cast_unsigned(),
                p95_duration_ms: row.get::<i64>(7)?.cast_unsigned(),
            });
        }
        // Held until the cursor is gone: an open `Rows` on the shared
        // connection swallows later writes.
        drop(rows);
        drop(conn);
        Ok(result)
    }

    /// Get recent tool call log entries (for detail views).
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the query fails or a column does
    /// not decode as the expected type.
    pub async fn get_tool_call_log(
        &self,
        tool_name: Option<&str>,
        limit: u32,
    ) -> Result<Vec<ToolCallLogEntry>, StorageError> {
        let conn = self.conn.lock().await;

        let mut rows = if let Some(name) = tool_name {
            conn.query(
                "SELECT tool_name, success, duration_ms, output_size, error_message, session_id, created_at
                 FROM tool_call_log
                 WHERE tool_name = ?1
                 ORDER BY created_at DESC
                 LIMIT ?2",
                turso::params![name, i64::from(limit)],
            ).await?
        } else {
            conn.query(
                "SELECT tool_name, success, duration_ms, output_size, error_message, session_id, created_at
                 FROM tool_call_log
                 ORDER BY created_at DESC
                 LIMIT ?1",
                turso::params![i64::from(limit)],
            ).await?
        };
        let mut result = Vec::new();
        while let Some(row) = rows.next().await? {
            result.push(ToolCallLogEntry {
                tool_name: row.get::<String>(0)?,
                success: row.get::<i64>(1)? != 0,
                duration_ms: row.get::<i64>(2)?.cast_unsigned(),
                output_size: row.get::<i64>(3)?.cast_unsigned(),
                error_message: {
                    let s: String = row.get::<String>(4)?;
                    if s.is_empty() { None } else { Some(s) }
                },
                session_id: {
                    let s: String = row.get::<String>(5)?;
                    if s.is_empty() { None } else { Some(s) }
                },
                created_at: row.get::<String>(6)?,
            });
        }
        // Held until the cursor is gone: an open `Rows` on the shared
        // connection swallows later writes.
        drop(rows);
        drop(conn);
        Ok(result)
    }

    /// Delete raw tool-call rows older than `keep_days`; returns how many.
    ///
    /// The cutoff is computed by the database, in the format `created_at` is
    /// written in (`datetime('now')`, `YYYY-MM-DD HH:MM:SS`). It used to be an
    /// RFC 3339 string, and `' '` sorts below `'T'`, so every row of the cutoff
    /// DAY compared as older and the whole day was deleted at any hour. Only
    /// the raw log is pruned — the hourly and daily aggregates keep history.
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the delete fails.
    pub async fn prune_tool_call_log(&self, keep_days: u32) -> Result<u64, StorageError> {
        let modifier = format!("-{keep_days} days");
        let conn = self.conn.lock().await;
        let deleted = conn
            .execute(
                "DELETE FROM tool_call_log WHERE created_at < datetime('now', ?1)",
                turso::params![modifier],
            )
            .await?;
        drop(conn);
        Ok(deleted)
    }
}

// =========================================================================
// Checkpoints
// =========================================================================

impl Storage {
    /// Save a checkpoint for crash recovery.
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the upsert fails.
    pub async fn save_checkpoint(&self, session_id: &str, data: &str) -> Result<(), StorageError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO checkpoints (session_id, data, updated_at)
             VALUES (?1, ?2, datetime('now'))
             ON CONFLICT(session_id) DO UPDATE SET data = ?2, updated_at = datetime('now')",
            turso::params![session_id, data],
        ).await?;
        drop(conn);
        Ok(())
    }

    /// Load a checkpoint for a session.
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the query fails or the stored
    /// data does not decode as text.
    pub async fn load_checkpoint(&self, session_id: &str) -> Result<Option<String>, StorageError> {
        let conn = self.conn.lock().await;
        let mut rows = conn.query(
            "SELECT data FROM checkpoints WHERE session_id = ?1",
            turso::params![session_id],
        ).await?;
        let data = if let Some(row) = rows.next().await? {
            Some(row.get::<String>(0)?)
        } else {
            None
        };
        // Held until the cursor is gone: an open `Rows` on the shared
        // connection swallows later writes.
        drop(rows);
        drop(conn);
        Ok(data)
    }

    /// Delete a checkpoint after successful completion.
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the delete fails.
    pub async fn delete_checkpoint(&self, session_id: &str) -> Result<(), StorageError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "DELETE FROM checkpoints WHERE session_id = ?1",
            turso::params![session_id],
        ).await?;
        drop(conn);
        Ok(())
    }

    /// List all checkpoint session IDs (for recovery at startup).
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the query fails or a session id
    /// does not decode as text.
    pub async fn list_checkpoints(&self) -> Result<Vec<String>, StorageError> {
        let conn = self.conn.lock().await;
        let mut rows = conn.query(
            "SELECT session_id FROM checkpoints",
            (),
        ).await?;
        let mut ids = Vec::new();
        while let Some(row) = rows.next().await? {
            ids.push(row.get::<String>(0)?);
        }
        // Held until the cursor is gone: an open `Rows` on the shared
        // connection swallows later writes.
        drop(rows);
        drop(conn);
        Ok(ids)
    }

    // =========================================================================
    // Tool Stats (aggregated — replaces tool-stats.json)
    // =========================================================================

    /// Save aggregated tool stats to the `tool_stats` table (upsert).
    ///
    /// Anything that is not the expected shape is skipped rather than
    /// rejected: a missing `tools` object saves nothing, and a missing or
    /// non-integer counter is stored as 0.
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if an upsert fails; tools earlier in
    /// the map are already saved by then.
    pub async fn save_tool_stats_aggregated(&self, stats: &serde_json::Value) -> Result<(), StorageError> {
        let conn = self.conn.lock().await;
        // The JSON is expected to be { "tools": { "tool_name": { stats... }, ... }, "sessions": N }
        if let Some(tools) = stats.get("tools").and_then(|v| v.as_object()) {
            for (tool_name, tool_data) in tools {
                let call_count = tool_data.get("call_count").and_then(serde_json::Value::as_i64).unwrap_or(0);
                let success_count = tool_data.get("success_count").and_then(serde_json::Value::as_i64).unwrap_or(0);
                let failure_count = tool_data.get("failure_count").and_then(serde_json::Value::as_i64).unwrap_or(0);
                let total_duration_ms = tool_data.get("total_duration_ms").and_then(serde_json::Value::as_i64).unwrap_or(0);
                let last_called_epoch_ms = tool_data.get("last_called_epoch_ms").and_then(serde_json::Value::as_i64).unwrap_or(0);
                let latencies = tool_data.get("latencies_ms").map_or_else(|| "[]".to_string(), std::string::ToString::to_string);
                let output_sizes = tool_data.get("output_sizes").map_or_else(|| "[]".to_string(), std::string::ToString::to_string);
                let errors = tool_data.get("errors").map_or_else(|| "[]".to_string(), std::string::ToString::to_string);

                conn.execute(
                    "INSERT INTO tool_stats (tool_name, call_count, success_count, failure_count, total_duration_ms, last_called_epoch_ms, latencies_ms_json, output_sizes_json, errors_json, updated_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, datetime('now'))
                     ON CONFLICT(tool_name) DO UPDATE SET
                        call_count = ?2, success_count = ?3, failure_count = ?4,
                        total_duration_ms = ?5, last_called_epoch_ms = ?6,
                        latencies_ms_json = ?7, output_sizes_json = ?8, errors_json = ?9,
                        updated_at = datetime('now')",
                    turso::params![
                        tool_name.as_str(),
                        call_count,
                        success_count,
                        failure_count,
                        total_duration_ms,
                        last_called_epoch_ms,
                        latencies.as_str(),
                        output_sizes.as_str(),
                        errors.as_str()
                    ],
                ).await?;
            }
        }
        // Held for the whole batch: the guard is the transaction.
        drop(conn);
        Ok(())
    }

    /// Load aggregated tool stats from the `tool_stats` table (returns the JSON format `ToolStatsTracker` expects).
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the query fails or a column does
    /// not decode as the expected type. Unparseable stored JSON arrays load as
    /// empty arrays.
    pub async fn load_tool_stats_aggregated(&self) -> Result<serde_json::Value, StorageError> {
        let conn = self.conn.lock().await;
        let mut rows = conn.query(
            "SELECT tool_name, call_count, success_count, failure_count, total_duration_ms, last_called_epoch_ms, latencies_ms_json, output_sizes_json, errors_json
             FROM tool_stats",
            (),
        ).await?;

        let mut tools = serde_json::Map::new();
        let mut session_count: i64 = 0;
        while let Some(row) = rows.next().await? {
            let tool_name: String = row.get(0)?;
            let call_count: i64 = row.get(1)?;
            let success_count: i64 = row.get(2)?;
            let failure_count: i64 = row.get(3)?;
            let total_duration_ms: i64 = row.get(4)?;
            let last_called_epoch_ms: i64 = row.get(5)?;
            let latencies_str: String = row.get(6)?;
            let output_sizes_str: String = row.get(7)?;
            let errors_str: String = row.get(8)?;

            let latencies: serde_json::Value = serde_json::from_str(&latencies_str).unwrap_or(serde_json::json!([]));
            let output_sizes: serde_json::Value = serde_json::from_str(&output_sizes_str).unwrap_or(serde_json::json!([]));
            let errors: serde_json::Value = serde_json::from_str(&errors_str).unwrap_or(serde_json::json!([]));

            session_count += call_count;
            tools.insert(tool_name, serde_json::json!({
                "call_count": call_count,
                "success_count": success_count,
                "failure_count": failure_count,
                "total_duration_ms": total_duration_ms,
                "last_called_epoch_ms": last_called_epoch_ms,
                "latencies_ms": latencies,
                "output_sizes": output_sizes,
                "errors": errors,
            }));
        }
        // Held until the cursor is gone: an open `Rows` on the shared
        // connection swallows later writes.
        drop(rows);
        drop(conn);

        Ok(serde_json::json!({
            "tools": tools,
            "sessions": session_count
        }))
    }

    // =========================================================================
    // Daemon Session Persistence (replaces sessions.json)
    // =========================================================================

    /// Create or update a daemon session in the database.
    /// This handles the daemon's Session struct format (different from GUI sessions).
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the upsert fails.
    pub async fn upsert_daemon_session(
        &self,
        session_id: &str,
        name: Option<&str>,
        workspace_id: Option<&str>,
        created_at: &str,
        updated_at: &str,
        metadata: Option<&str>,
    ) -> Result<(), StorageError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO sessions (session_id, channel, name, workspace_id, created_at, updated_at, metadata)
             VALUES (?1, 'gui', ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(session_id) DO UPDATE SET
                name = COALESCE(?2, name),
                workspace_id = COALESCE(?3, workspace_id),
                updated_at = ?5,
                metadata = COALESCE(?6, metadata)",
            turso::params![session_id, name, workspace_id, created_at, updated_at, metadata],
        ).await?;
        drop(conn);
        Ok(())
    }

    /// Add a daemon message to the messages table.
    /// Stores `tool_calls`, attachments, and reasoning in the metadata JSON field.
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the insert or the session touch
    /// fails; a failed touch leaves the message already inserted.
    pub async fn add_daemon_message(
        &self,
        session_id: &str,
        message_id: &str,
        role: &str,
        content: &str,
        created_at: &str,
        metadata: Option<&str>,
    ) -> Result<(), StorageError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO messages (session_id, role, content, content_type, tool_use_id, created_at, metadata)
             VALUES (?1, ?2, ?3, 'text', ?4, ?5, ?6)",
            turso::params![session_id, role, content, message_id, created_at, metadata],
        ).await?;
        // Touch session
        conn.execute(
            "UPDATE sessions SET updated_at = ?1 WHERE session_id = ?2",
            turso::params![created_at, session_id],
        ).await?;
        // Held across the insert and the touch: the guard is the transaction.
        drop(conn);
        Ok(())
    }

    /// Load all sessions from the database (regardless of channel).
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the query fails or a column does
    /// not decode as the expected type. Unparseable metadata loads as `None`.
    pub async fn list_daemon_sessions(&self) -> Result<Vec<Session>, StorageError> {
        let conn = self.conn.lock().await;
        let mut rows = conn.query(
            "SELECT id, session_id, channel, user_id, created_at, updated_at, metadata, workspace_id, name
             FROM sessions ORDER BY updated_at DESC",
            (),
        ).await?;
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

    /// Load all messages for a daemon session, ordered by creation time.
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the query fails or a column does
    /// not decode as the expected type. Unparseable metadata loads as `None`.
    pub async fn load_daemon_messages(&self, session_id: &str) -> Result<Vec<Message>, StorageError> {
        let conn = self.conn.lock().await;
        let mut rows = conn.query(
            "SELECT id, session_id, role, content, content_type, tool_use_id, created_at, tokens_in, tokens_out, metadata
             FROM messages WHERE session_id = ?1 ORDER BY created_at ASC",
            turso::params![session_id],
        ).await?;
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

    /// Delete all messages for a daemon session.
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the delete fails.
    pub async fn clear_daemon_session_messages(&self, session_id: &str) -> Result<(), StorageError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "DELETE FROM messages WHERE session_id = ?1",
            turso::params![session_id],
        ).await?;
        drop(conn);
        Ok(())
    }

    /// Delete one message of a daemon session, by its message id.
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the delete fails.
    pub async fn delete_daemon_message(
        &self,
        session_id: &str,
        message_id: &str,
    ) -> Result<(), StorageError> {
        let conn = self.conn.lock().await;
        // The message id is stored in `tool_use_id` (see `add_daemon_message`).
        conn.execute(
            "DELETE FROM messages WHERE session_id = ?1 AND tool_use_id = ?2",
            turso::params![session_id, message_id],
        )
        .await?;
        drop(conn);
        Ok(())
    }

    /// Delete a daemon session and its messages.
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if either delete fails; a failure on
    /// the session row leaves its messages already deleted.
    pub async fn delete_daemon_session(&self, session_id: &str) -> Result<(), StorageError> {
        let conn = self.conn.lock().await;
        conn.execute("DELETE FROM messages WHERE session_id = ?1", turso::params![session_id]).await?;
        conn.execute("DELETE FROM sessions WHERE session_id = ?1", turso::params![session_id]).await?;
        // Held across both deletes: the guard is the transaction.
        drop(conn);
        Ok(())
    }

    /// Update daemon session name.
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the update fails.
    pub async fn rename_daemon_session(&self, session_id: &str, name: &str) -> Result<(), StorageError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE sessions SET name = ?1, updated_at = datetime('now') WHERE session_id = ?2",
            turso::params![name, session_id],
        ).await?;
        drop(conn);
        Ok(())
    }

    /// Update daemon session workspace.
    ///
    /// # Errors
    /// Returns [`StorageError::Database`] if the update fails.
    pub async fn set_daemon_session_workspace(&self, session_id: &str, workspace_id: Option<&str>) -> Result<(), StorageError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE sessions SET workspace_id = ?1, updated_at = datetime('now') WHERE session_id = ?2",
            turso::params![workspace_id, session_id],
        ).await?;
        drop(conn);
        Ok(())
    }
}

/// One model request, as [`Storage::log_model_request`] records it.
#[derive(Debug, Clone, Copy)]
pub struct NewModelRequest<'a> {
    pub model: &'a str,
    pub success: bool,
    pub latency_ms: u64,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cache_read_tokens: u32,
    pub cache_creation_tokens: u32,
    /// The 1-hour share of `cache_creation_tokens`.
    pub cache_creation_1h_tokens: u32,
    pub tier: Option<&'a str>,
    pub escalated: bool,
    pub session_id: Option<&'a str>,
}

/// One tool call, as [`Storage::log_tool_call`] records it.
#[derive(Debug, Clone, Copy)]
pub struct NewToolCall<'a> {
    pub tool_name: &'a str,
    pub success: bool,
    /// The harness answered with a breaker replay instead of dispatching.
    pub short_circuited: bool,
    pub duration_ms: u64,
    pub output_size: usize,
    pub error_message: Option<&'a str>,
    pub session_id: Option<&'a str>,
}

/// How a usage rollup groups requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UsagePeriod {
    Day,
    Month,
    Session,
}

/// Longest window a usage rollup covers: a year and a day.
pub const USAGE_BUCKET_DAYS_MAX: u32 = 366;

/// Model usage summed over one period for one model.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ModelUsageBucket {
    /// `YYYY-MM-DD` or `YYYY-MM` (UTC).
    pub period: String,
    pub model: String,
    pub requests: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_write_tokens: u64,
    /// The 1-hour share of `cache_write_tokens`.
    pub cache_write_1h_tokens: u64,
}

/// Time-bucketed tool statistics (hourly or daily).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolStatsTimeBucket {
    pub tool_name: String,
    pub period: String,
    pub call_count: u64,
    pub success_count: u64,
    pub failure_count: u64,
    pub total_duration_ms: u64,
    pub avg_duration_ms: u64,
    pub p95_duration_ms: u64,
}

/// A single tool call log entry.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolCallLogEntry {
    pub tool_name: String,
    pub success: bool,
    pub duration_ms: u64,
    pub output_size: u64,
    pub error_message: Option<String>,
    pub session_id: Option<String>,
    pub created_at: String,
}
