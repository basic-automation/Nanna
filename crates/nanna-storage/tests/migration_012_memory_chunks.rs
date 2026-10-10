#![warn(clippy::pedantic, clippy::nursery, clippy::all)]
//! Migration 012 must actually apply, and stay applied.
//!
//! Migrations run OUTSIDE a transaction, are split naively on `;` with no
//! comment awareness, and the migration name is recorded only after every
//! statement succeeds. So a statement that cannot apply — a stray semicolon
//! splitting one statement into two invalid halves, an `ALTER` that is not
//! idempotent, a typo — is not a one-time failure. It aborts the run, leaves
//! the name unrecorded, and is retried identically on every subsequent boot.
//! The database is then permanently stuck one migration behind, and until the
//! storage-open error was made loud, silently so.
//!
//! These tests pin the two properties that failure mode needs: the schema the
//! migration claims to create exists, and re-opening the same database is a
//! no-op rather than a second attempt.

use nanna_storage::{Storage, StorageConfig};
use turso::Builder;

/// One test's database file, in a fresh directory that is deleted on drop.
///
/// `/tmp` is tmpfs on the dev host, so a directory a test leaves behind is RAM
/// held until reboot: before this guard every suite run stranded 44 of them,
/// and 33 runs had accumulated 1.1 GB. Bind it before the `Storage` that opens
/// it, so the database closes before its directory goes.
struct TempDb {
    dir: std::path::PathBuf,
    path: String,
}

impl std::ops::Deref for TempDb {
    type Target = str;

    fn deref(&self) -> &str {
        &self.path
    }
}

impl Drop for TempDb {
    fn drop(&mut self) {
        // Only ever delete what `temp_db_path` created.
        assert!(self.dir.starts_with(std::env::temp_dir()));
        assert!(self.path.starts_with(&*self.dir.to_string_lossy()));
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// Fresh, isolated temp directory for one test's database file. Mirrors the
/// pattern in `secure_delete.rs` rather than adding a dev-dependency: this
/// crate pins its dependency set deliberately.
fn temp_db_path(tag: &str) -> TempDb {
    let dir = std::env::temp_dir().join(format!(
        "nanna_m012_{tag}_{}_{:p}",
        std::process::id(),
        &raw const tag
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let path = dir.join("mem.db").to_string_lossy().to_string();
    TempDb { dir, path }
}

async fn open(db_path: &str) -> Storage {
    Storage::new(&StorageConfig {
        path: db_path.to_string(),
    })
    .await
    .expect("storage opens and migrates")
}

async fn table_columns(db_path: &str, table: &str) -> Vec<String> {
    let conn = Builder::new_local(db_path)
        .build()
        .await
        .expect("open raw")
        .connect()
        .expect("connect raw");
    let mut rows = conn
        .query(&format!("PRAGMA table_info({table})"), ())
        .await
        .expect("table_info");
    let mut out = Vec::new();
    while let Some(row) = rows.next().await.expect("row") {
        // PRAGMA table_info: (cid, name, type, notnull, dflt_value, pk)
        if let Ok(turso::Value::Text(name)) = row.get_value(1) {
            out.push(name);
        }
    }
    out
}

#[tokio::test]
async fn memory_chunks_table_and_indexes_exist_after_migration() {
    let db_path = temp_db_path("schema");
    let _storage = open(&db_path).await;

    let cols = table_columns(&db_path, "memory_chunks").await;
    assert!(!cols.is_empty(), "memory_chunks table was not created");

    // The columns the chunk design depends on. `embedding_model` is what makes
    // a model switch a resumable backfill instead of a rebuild; the char range
    // is what lets paginated recall address a page without re-chunking.
    for expected in [
        "memory_id",
        "ordinal",
        "content",
        "char_start",
        "char_end",
        "embedding",
        "embedding_model",
        "chunk_max_chars",
        "chunker_version",
        "workspace_id",
    ] {
        assert!(
            cols.iter().any(|c| c == expected),
            "memory_chunks is missing column `{expected}` (has: {cols:?})"
        );
    }

    let conn = Builder::new_local(&db_path)
        .build()
        .await
        .expect("open raw")
        .connect()
        .expect("connect raw");
    let mut rows = conn
        .query(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'memory_chunks'",
            (),
        )
        .await
        .expect("index query");
    let mut indexes = Vec::new();
    while let Some(row) = rows.next().await.expect("row") {
        if let Ok(turso::Value::Text(name)) = row.get_value(0) {
            indexes.push(name);
        }
    }
    drop(rows);
    for expected in [
        "idx_memory_chunks_parent",
        "idx_memory_chunks_workspace",
        "idx_memory_chunks_model",
        "idx_memory_chunks_pending",
    ] {
        assert!(
            indexes.iter().any(|i| i == expected),
            "missing index `{expected}` (has: {indexes:?})"
        );
    }
}

/// The retry-forever signature: if any statement fails, the name is never
/// recorded, so a second open runs the whole migration again. Re-opening must
/// be a no-op, and the recorded name is the evidence it completed.
#[tokio::test]
async fn reopening_does_not_rerun_the_migration() {
    let db_path = temp_db_path("twice");

    {
        let _first = open(&db_path).await;
    }
    // Would fail here if the migration were not idempotent or had not recorded.
    {
        let _second = open(&db_path).await;
    }

    let conn = Builder::new_local(&db_path)
        .build()
        .await
        .expect("open raw")
        .connect()
        .expect("connect raw");
    let mut rows = conn
        .query(
            "SELECT COUNT(*) FROM _migrations WHERE name = '012_memory_chunks'",
            (),
        )
        .await
        .expect("count query");
    let row = rows.next().await.expect("row").expect("one row");
    let count = match row.get_value(0).expect("count") {
        turso::Value::Integer(n) => n,
        other => panic!("unexpected count value: {other:?}"),
    };
    drop(rows);
    assert_eq!(
        count, 1,
        "012_memory_chunks must be recorded exactly once — 0 means every boot \
         retries it, >1 means the ledger is not deduplicating"
    );
}

/// A memory row with no chunks must still be a valid, readable row. Existing
/// databases upgrade into exactly this state and keep matching on the parent
/// vector until the backfill reaches them, so "no chunks yet" cannot be an
/// error case.
#[tokio::test]
async fn a_memory_with_no_chunks_is_still_valid() {
    let db_path = temp_db_path("empty");
    let storage = open(&db_path).await;

    let memories = storage.memories();
    memories
        .create(nanna_storage::NewMemory {
            memory_id: "mem-no-chunks".to_string(),
            content: "a memory that predates chunking".to_string(),
            embedding: Some(vec![0.1, 0.2, 0.3, 0.4]),
            embedding_model: Some("probe".to_string()),
            session_id: None,
            metadata: None,
            tags: Vec::new(),
            workspace_id: None,
            fsrs_stability: 1.0,
            fsrs_difficulty: 5.0,
            fsrs_last_access: 0,
            fsrs_access_count: 0,
            fsrs_importance: 3.0,
            fsrs_storage_strength: 1.0,
            fsrs_generation: 0,
        })
        .await
        .expect("create memory");

    let found = memories.get("mem-no-chunks").await.expect("memory exists");
    assert_eq!(found.content, "a memory that predates chunking");

    let conn = Builder::new_local(&db_path)
        .build()
        .await
        .expect("open raw")
        .connect()
        .expect("connect raw");
    let mut rows = conn
        .query(
            "SELECT COUNT(*) FROM memory_chunks WHERE memory_id = 'mem-no-chunks'",
            (),
        )
        .await
        .expect("chunk count");
    let row = rows.next().await.expect("row").expect("one row");
    let chunks = match row.get_value(0).expect("count") {
        turso::Value::Integer(n) => n,
        other => panic!("unexpected: {other:?}"),
    };
    drop(rows);
    assert_eq!(chunks, 0, "a pre-chunking memory starts with no chunk rows");
}
