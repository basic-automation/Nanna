//! Corruption recovery for the Turso database file.
//!
//! When the memories table has **page-level** corruption (a broken btree page,
//! not just one unreadable row), both the bulk load and the row-by-row salvage
//! fail and the daemon used to boot with a silently empty memory store. This
//! module gives `Storage` a bounded self-healing open:
//!
//! 1. Open + migrate as usual, then **preflight** the memories table (the same
//!    scan the startup load performs).
//! 2. On a page-level corruption error: close the database, quarantine the
//!    file (`nanna.db` → `nanna.db.corrupt-<timestamp>`, WAL alongside), keep
//!    at most [`MAX_QUARANTINE_COPIES`] quarantine copies, and recreate a
//!    fresh database at the original path.
//! 3. Best-effort salvage every reachable row from the quarantined copy into
//!    the fresh database, skipping unreadable ones, and report counts.
//!
//! Row-level corruption (bulk load fails, salvage succeeds with skipped rows)
//! is deliberately NOT rebuilt here: the existing degraded-load path already
//! handles it, and rewriting the file for one bad row would trade bounded,
//! known data loss for churn.
//!
//! Bounds (Tiger Style — every bound derives from a real constraint):
//! - [`MAX_QUARANTINE_COPIES`] = 3: enough history to diagnose a recurring
//!   corruption (previous, previous-but-one, and the fresh case) without
//!   letting a crash-looping daemon eat the disk with full DB copies.
//! - Salvage scans are prefix scans: forward until the first unreadable page,
//!   then backward until the first unreadable page. Two passes per table,
//!   never more — a corrupt middle page cannot make salvage loop.
//! - The salvage table list is the static migration schema; salvage never
//!   discovers tables dynamically from a corrupt catalog.

use crate::is_corruption_error;
use crate::{Storage, StorageConfig, StorageError};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use tracing::{error, info, warn};
use turso::Value;

/// Maximum number of `<db>.corrupt-<timestamp>` quarantine copies kept on disk.
///
/// Each copy is a full database file; three give enough history to diagnose
/// repeated corruption while bounding disk growth from a crash-looping daemon
/// (see module docs).
pub const MAX_QUARANTINE_COPIES: usize = 3;

/// Tables copied from the quarantined file into the fresh database, in
/// foreign-key order (parents before children). Static on purpose: the schema
/// catalog of a corrupt file is not trustworthy, so salvage only ever touches
/// tables the compiled-in migrations define. `_migrations` is excluded — the
/// fresh database's own migration run owns it.
const SALVAGE_TABLES: &[&str] = &[
    "sessions",
    "messages",
    "memories",
    "memory_chunks",
    // Buckets and queued work salvage alongside their parents: recovering the
    // memories but not their vectors would leave a store that looks intact and
    // retrieves nothing until a full re-embed.
    "memory_vectors",
    "memory_chunk_vectors",
    "embedding_queue",
    "memory_tags",
    // The episodic stream beneath memories; append-only history nothing else
    // can rebuild.
    "memory_events",
    "config",
    "cron_jobs",
    "job_runs",
    "workspace_memories",
    "model_stats",
    "model_request_log",
    "tool_stats",
    "tool_call_log",
    "tool_stats_hourly",
    "tool_stats_daily",
    "workspaces",
    "checkpoints",
    // Before `tasks`: a card's assignee must resolve to a member, and the fresh
    // store seeds only `human` and the routers — every custom agent lives here.
    "members",
    "tasks",
    "task_notes",
    "task_activity",
];

/// Per-table salvage outcome.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TableSalvage {
    pub table: String,
    /// Rows successfully copied into the fresh database.
    pub recovered: usize,
    /// Total rows the quarantined table reported, when countable. `None` when
    /// even `COUNT(*)` failed on the corrupt table (loss size unknown).
    pub expected: Option<usize>,
}

/// Result of a quarantine + rebuild + salvage cycle.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RecoveryReport {
    /// Where the corrupt database file was moved.
    pub quarantine_path: PathBuf,
    /// Memories rows recovered into the fresh store.
    pub memories_recovered: usize,
    /// Memories rows the corrupt store held, when countable.
    pub memories_expected: Option<usize>,
    /// Every salvaged table's outcome, memories included.
    pub tables: Vec<TableSalvage>,
    /// Older quarantine copies deleted to hold the [`MAX_QUARANTINE_COPIES`] bound.
    pub quarantines_pruned: usize,
}

/// Preflight verdict on the memories table of a freshly opened store.
enum Preflight {
    /// Bulk load succeeded.
    Clean,
    /// Bulk load hit corruption but the row-by-row salvage works — the
    /// existing degraded-load path (skip + report corrupt rows) handles this.
    RowLevel,
    /// Both the bulk load and the salvage failed: the table's page structure
    /// is broken and nothing can be read through it.
    PageLevel(StorageError),
}

/// Open the database, verifying the memories table is readable; on page-level
/// corruption, quarantine the file, rebuild a fresh store, and salvage every
/// reachable row into it.
///
/// Returns the working storage plus `Some(RecoveryReport)` when a rebuild
/// happened. `:memory:` databases skip recovery entirely (no file to
/// quarantine, nothing persists).
///
/// The preflight repeats the scan the startup memory load will perform, so a
/// healthy boot pays one extra table scan. That cost is accepted: it is the
/// only deterministic way to know the store is readable *before* the rest of
/// the daemon takes connection clones, which is the last moment the file can
/// be safely renamed out from under the (single, exclusive) owner.
///
/// # Errors
///
/// Propagates non-corruption open/load failures (I/O, locking) unchanged —
/// quarantining the file on a transient error would destroy a healthy store —
/// and fails if the fresh post-quarantine store cannot be created.
pub async fn open_with_recovery(
    config: &StorageConfig,
) -> Result<(Storage, Option<RecoveryReport>), StorageError> {
    if config.path == ":memory:" {
        return Ok((Storage::new(config).await?, None));
    }

    let page_level = match Storage::new(config).await {
        Ok(storage) => match preflight_memories(&storage).await? {
            Preflight::Clean => return Ok((storage, None)),
            Preflight::RowLevel => {
                warn!(
                    "Memory store has row-level corruption; leaving file in place \
                     (degraded load will skip the unreadable rows)"
                );
                return Ok((storage, None));
            }
            Preflight::PageLevel(e) => {
                // `storage` holds the only handles on the file; drop them all
                // before the rename below or Windows refuses the move.
                drop(storage);
                e
            }
        },
        // The open/migration itself died on corruption (schema pages broken).
        Err(e) if is_corruption_error(&e) => e,
        Err(e) => return Err(e),
    };

    error!(
        "Memory store has page-level corruption ({page_level}); quarantining {} and rebuilding",
        config.path
    );

    let db_path = PathBuf::from(&config.path);
    let quarantine_path = match quarantine_database(&db_path) {
        Ok(p) => p,
        Err(e) => {
            // Could not move the corrupt file (still locked, permissions...).
            // Reopen and hand back the degraded store rather than refusing to
            // boot — this is exactly the pre-recovery behavior, made loud.
            error!(
                "Could not quarantine corrupt database {}: {e}. \
                 Continuing with the degraded store.",
                db_path.display()
            );
            return Ok((Storage::new(config).await?, None));
        }
    };
    let quarantines_pruned = prune_quarantines(&db_path);

    // Fresh, empty, migrated store at the original path. If even this fails
    // there is nothing left to fall back to — propagate.
    let fresh = Storage::new(config).await?;

    let tables = salvage_all_tables(&quarantine_path, &fresh).await;
    let (memories_recovered, memories_expected) = tables
        .iter()
        .find(|t| t.table == "memories")
        .map_or((0, None), |t| (t.recovered, t.expected));

    let report = RecoveryReport {
        quarantine_path: quarantine_path.clone(),
        memories_recovered,
        memories_expected,
        tables,
        quarantines_pruned,
    };

    warn!(
        "Memory store rebuilt after corruption: {} of {} memories recovered; \
         corrupt copy kept at {}",
        report.memories_recovered,
        report
            .memories_expected
            .map_or_else(|| "an unknown number of".to_string(), |n| n.to_string()),
        quarantine_path.display()
    );

    Ok((fresh, Some(report)))
}

/// Classify the memories table of `storage`: clean, row-level corrupt, or
/// page-level corrupt. A non-corruption bulk-load failure (I/O, locking)
/// propagates as `Err` — quarantining the file on a transient error would
/// destroy a healthy store.
async fn preflight_memories(storage: &Storage) -> Result<Preflight, StorageError> {
    let repo = storage.memories();
    match repo.bulk_load().await {
        Ok(_) => Ok(Preflight::Clean),
        Err(e) if is_corruption_error(&e) => match repo.bulk_load_salvage().await {
            // Salvage worked: only individual rows are unreadable.
            Ok(_) => Ok(Preflight::RowLevel),
            // Corruption is confirmed (bulk load already failed with a
            // corruption error), so ANY salvage failure means the page
            // structure itself is unreadable — rebuild.
            Err(se) => Ok(Preflight::PageLevel(se)),
        },
        Err(e) => Err(e),
    }
}

/// Move the database file (and its WAL/shm sidecars) to a timestamped
/// quarantine name next to it. Returns the quarantine path of the main file.
fn quarantine_database(db_path: &Path) -> std::io::Result<PathBuf> {
    let ts = chrono::Utc::now().format("%Y%m%d-%H%M%S");
    let base = format!("{}.corrupt-{ts}", db_path.display());

    // Bounded collision handling: two rebuilds in the same second get a
    // numeric suffix; ten in one second means something else is wrong.
    let mut quarantine = PathBuf::from(&base);
    for n in 1..=9u32 {
        if !quarantine.exists() {
            break;
        }
        quarantine = PathBuf::from(format!("{base}-{n}"));
    }
    if quarantine.exists() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            format!("quarantine name exhausted: {}", quarantine.display()),
        ));
    }

    std::fs::rename(db_path, &quarantine)?;

    // The WAL may hold committed pages not yet checkpointed into the main
    // file — move it with the database so the salvage read sees them. The
    // shm/coordination sidecar is only meaningful next to a live database.
    for suffix in ["-wal", "-tshm"] {
        let side = PathBuf::from(format!("{}{suffix}", db_path.display()));
        if side.exists() {
            let side_dst = PathBuf::from(format!("{}{suffix}", quarantine.display()));
            if let Err(e) = std::fs::rename(&side, &side_dst) {
                warn!("Could not move sidecar {}: {e}", side.display());
            }
        }
    }

    info!(
        "Quarantined corrupt database: {} -> {}",
        db_path.display(),
        quarantine.display()
    );
    Ok(quarantine)
}

/// Delete the oldest quarantine copies (and their sidecars) so at most
/// [`MAX_QUARANTINE_COPIES`] remain. Returns how many were deleted. The
/// timestamped names sort lexicographically in chronological order.
fn prune_quarantines(db_path: &Path) -> usize {
    let Some(dir) = db_path.parent().filter(|d| !d.as_os_str().is_empty()) else {
        return 0;
    };
    let Some(file_name) = db_path.file_name().and_then(|n| n.to_str()) else {
        return 0;
    };
    let prefix = format!("{file_name}.corrupt-");

    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    let mut copies: Vec<PathBuf> = entries
        .filter_map(std::result::Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.file_name().and_then(|n| n.to_str()).is_some_and(|n| {
                n.starts_with(&prefix) && !n.ends_with("-wal") && !n.ends_with("-tshm")
            })
        })
        .collect();
    if copies.len() <= MAX_QUARANTINE_COPIES {
        return 0;
    }

    // Newest last after sorting by name; keep the newest MAX_QUARANTINE_COPIES.
    copies.sort();
    let excess = copies.len() - MAX_QUARANTINE_COPIES;
    let mut pruned = 0usize;
    for old in &copies[..excess] {
        match std::fs::remove_file(old) {
            Ok(()) => {
                pruned += 1;
                for suffix in ["-wal", "-tshm"] {
                    let side = PathBuf::from(format!("{}{suffix}", old.display()));
                    if side.exists() {
                        let _ = std::fs::remove_file(&side);
                    }
                }
                info!("Pruned old quarantine copy {}", old.display());
            }
            Err(e) => warn!("Could not prune quarantine copy {}: {e}", old.display()),
        }
    }
    pruned
}

/// Best-effort copy of every reachable row of every known table from the
/// quarantined file into the fresh store. Never fails: a table (or the whole
/// quarantined file) that cannot be read simply contributes zero rows.
async fn salvage_all_tables(quarantine_path: &Path, fresh: &Storage) -> Vec<TableSalvage> {
    let src_conn = match turso::Builder::new_local(&quarantine_path.to_string_lossy())
        .build()
        .await
        .and_then(|db| db.connect())
    {
        Ok(conn) => conn,
        Err(e) => {
            error!(
                "Cannot open quarantined copy {} for salvage ({e}); starting empty",
                quarantine_path.display()
            );
            return SALVAGE_TABLES
                .iter()
                .map(|t| TableSalvage {
                    table: (*t).to_string(),
                    recovered: 0,
                    expected: None,
                })
                .collect();
        }
    };

    let mut results = Vec::with_capacity(SALVAGE_TABLES.len());
    for table in SALVAGE_TABLES {
        let salvage = salvage_table(&src_conn, fresh, table).await;
        if salvage.recovered > 0 || salvage.expected.is_none_or(|n| n > 0) {
            info!(
                "Salvaged {} of {} rows from table {table}",
                salvage.recovered,
                salvage
                    .expected
                    .map_or_else(|| "?".to_string(), |n| n.to_string()),
            );
        }
        results.push(salvage);
    }
    results
}

/// Copy one table's reachable rows from the quarantined connection into the
/// fresh store. Reads a forward prefix scan and, only if that scan aborted
/// early, a backward prefix scan (deduplicated by rowid) — so a single corrupt
/// middle page loses only the rows physically on it.
async fn salvage_table(src: &turso::Connection, fresh: &Storage, table: &str) -> TableSalvage {
    // COUNT may itself fail on a corrupt table; loss size is then unknown.
    let expected = count_rows(src, table).await;

    let select = format!("SELECT rowid, * FROM {table} ORDER BY rowid ASC");
    let (mut rows, forward_complete, source_columns) = read_prefix(src, &select).await;

    if !forward_complete {
        let select_desc = format!("SELECT rowid, * FROM {table} ORDER BY rowid DESC");
        let seen: HashSet<i64> = rows.iter().map(|(rid, _)| *rid).collect();
        let (tail_rows, _, _) = read_prefix(src, &select_desc).await;
        rows.extend(tail_rows.into_iter().filter(|(rid, _)| !seen.contains(rid)));
    }

    let mut recovered = 0usize;
    if !rows.is_empty() {
        let Some(insert) = salvage_insert(fresh, table, &source_columns).await else {
            return TableSalvage {
                table: table.to_string(),
                recovered: 0,
                expected,
            };
        };
        let conn = fresh.conn().lock().await;
        for (_rid, values) in rows {
            debug_assert_eq!(values.len(), source_columns.len(), "row width changed mid-table");
            let values: Vec<Value> = insert
                .source_indices
                .iter()
                .filter_map(|&i| values.get(i).cloned())
                .collect();
            match conn.execute(&insert.sql, values).await {
                Ok(_) => recovered += 1,
                Err(e) => {
                    warn!("Skipping unrestorable {table} row during salvage: {e}");
                }
            }
        }
    }

    TableSalvage {
        table: table.to_string(),
        recovered,
        expected,
    }
}

/// `SELECT COUNT(*)` that treats any failure as "unknown".
async fn count_rows(conn: &turso::Connection, table: &str) -> Option<usize> {
    let mut rows = conn
        .query(&format!("SELECT COUNT(*) FROM {table}"), ())
        .await
        .ok()?;
    let row = rows.next().await.ok()??;
    let n: i64 = row.get(0).ok()?;
    usize::try_from(n).ok()
}

/// Stream `sql`, collecting `(rowid, remaining column values)` per row until
/// the cursor errors or the table ends. Returns the collected prefix and
/// whether the scan reached the end without error. The cursor is dropped
/// before returning (an open cursor on a shared turso connection swallows
/// later statements).
async fn read_prefix(
    conn: &turso::Connection,
    sql: &str,
) -> (Vec<(i64, Vec<Value>)>, bool, Vec<String>) {
    let mut out = Vec::new();
    let Ok(mut rows) = conn.query(sql, ()).await else {
        return (out, false, Vec::new());
    };
    // The schema columns, in the SOURCE file's order (`rowid` is column 0 of
    // `SELECT rowid, *` and is not one of them).
    let columns: Vec<String> = rows.column_names().into_iter().skip(1).collect();
    loop {
        match rows.next().await {
            Ok(Some(row)) => {
                let Ok(rid) = row.get::<i64>(0) else {
                    return (out, false, columns);
                };
                let mut values = Vec::with_capacity(row.column_count().saturating_sub(1));
                for i in 1..row.column_count() {
                    match row.get_value(i) {
                        Ok(v) => values.push(v),
                        Err(_) => return (out, false, columns),
                    }
                }
                out.push((rid, values));
            }
            Ok(None) => return (out, true, columns),
            Err(_) => return (out, false, columns),
        }
    }
}

/// The statement that restores one salvaged row, matched by column NAME.
struct SalvageInsert {
    sql: String,
    /// For each placeholder, the index of its value in a source row.
    source_indices: Vec<usize>,
}

/// Build the salvage INSERT for `table` from the columns the source file has
/// and the fresh store has.
///
/// By name, not position: the quarantined file can predate a migration (it is
/// corrupt, so its own migration run may never have finished), and a column
/// appended since then made every positional INSERT fail on width — the whole
/// table lost, row by row. Columns only the fresh store has take their
/// defaults; columns only the source has are dropped, loudly.
///
/// `OR REPLACE`, because a fresh store is not empty: migrations seed rows
/// (`human`, the routers), and the user's own version of a seeded row — a
/// renamed human — must win over the default rather than be skipped as a
/// duplicate.
async fn salvage_insert(
    fresh: &Storage,
    table: &str,
    source_columns: &[String],
) -> Option<SalvageInsert> {
    let fresh_columns = {
        let conn = fresh.conn().lock().await;
        let rows = conn
            .query(&format!("SELECT * FROM {table} LIMIT 0"), ())
            .await;
        let names = rows
            .as_ref()
            .map(turso::Rows::column_names)
            .map_err(ToString::to_string);
        drop(rows);
        drop(conn);
        match names {
            Ok(names) => names,
            Err(e) => {
                error!("Fresh store has no readable {table} ({e}); nothing salvaged into it");
                return None;
            }
        }
    };
    let mut columns = Vec::new();
    let mut source_indices = Vec::new();
    for (i, name) in source_columns.iter().enumerate() {
        if fresh_columns.iter().any(|f| f.eq_ignore_ascii_case(name)) {
            columns.push(name.as_str());
            source_indices.push(i);
        } else {
            warn!("Salvage of {table}: column {name} no longer exists; its values are dropped");
        }
    }
    if columns.is_empty() {
        error!("Salvage of {table}: no column in common with the fresh store");
        return None;
    }
    let placeholders = (1..=columns.len())
        .map(|i| format!("?{i}"))
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!(
        "INSERT OR REPLACE INTO {table} ({}) VALUES ({placeholders})",
        columns.join(", ")
    );
    debug_assert_eq!(columns.len(), source_indices.len(), "one index per column");
    Some(SalvageInsert {
        sql,
        source_indices,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NewMemory;

    const PAGE_SIZE: usize = 4096;

    fn test_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("nanna_recovery_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn memory_row(i: usize, content: String) -> NewMemory {
        NewMemory {
            memory_id: format!("mem-{i:04}"),
            content,
            embedding: Some(vec![0.1, 0.2, 0.3, 0.4]),
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

    /// Build a populated on-disk store: `rows` memories whose content carries a
    /// unique needle, plus one session (proves cross-table salvage). Ensures
    /// everything is checkpointed into the main file, then closes it.
    async fn build_populated_db(dir: &Path, rows: usize, pad: usize) -> StorageConfig {
        let db_path = dir.join("nanna.db");
        let config = StorageConfig {
            path: db_path.to_string_lossy().to_string(),
        };
        let storage = Storage::new(&config).await.unwrap();

        storage
            .sessions()
            .create("salvage-me", "cli", None)
            .await
            .unwrap();
        let repo = storage.memories();
        for i in 0..rows {
            let content = format!("needle-{i:04} {}", "x".repeat(pad));
            repo.create(memory_row(i, content)).await.unwrap();
        }

        // Force WAL pages into the main file so page corruption below hits the
        // bytes reads actually use.
        checkpoint(&storage).await;
        drop(storage);

        let bytes = std::fs::read(&db_path).unwrap();
        assert!(
            find_needle_page(&bytes, b"needle-0000").is_some(),
            "test setup: rows must be checkpointed into the main db file"
        );
        config
    }

    async fn checkpoint(storage: &Storage) {
        let conn = storage.conn().lock().await;
        for sql in ["PRAGMA wal_checkpoint(TRUNCATE)", "PRAGMA wal_checkpoint"] {
            if let Ok(mut rows) = conn.query(sql, ()).await {
                while let Ok(Some(_)) = rows.next().await {}
                break;
            }
        }
    }

    /// Page index (0-based) of the first page containing `needle`.
    fn find_needle_page(bytes: &[u8], needle: &[u8]) -> Option<usize> {
        bytes
            .windows(needle.len())
            .position(|w| w == needle)
            .map(|off| off / PAGE_SIZE)
    }

    /// Zero the whole page holding `needle` — its page-type byte becomes 0, so
    /// any btree walk through it fails with a page-level corruption error.
    fn zero_needle_page(db_path: &Path, needle: &[u8]) {
        let mut bytes = std::fs::read(db_path).unwrap();
        let page = find_needle_page(&bytes, needle).expect("needle present in db file");
        assert!(page > 0, "refusing to zero page 1 (schema root)");
        bytes[page * PAGE_SIZE..(page + 1) * PAGE_SIZE].fill(0);
        std::fs::write(db_path, bytes).unwrap();
    }

    /// Every table a migration creates must be salvaged, or a page-level
    /// recovery silently drops it. `members` and `memory_events` were missing:
    /// the recovered board lost every custom agent, and its cards were left
    /// assigned to ids that no longer resolve.
    #[test]
    fn every_migrated_table_is_salvaged() {
        let mut missing = Vec::new();
        for (_, sql) in crate::migrations::MIGRATIONS {
            for statement in crate::migrations::split_statements(sql) {
                let words: Vec<&str> = statement.split_whitespace().collect();
                let name = match words.as_slice() {
                    [create, table, if_, not, exists, name, ..]
                        if create.eq_ignore_ascii_case("create")
                            && table.eq_ignore_ascii_case("table")
                            && if_.eq_ignore_ascii_case("if")
                            && not.eq_ignore_ascii_case("not")
                            && exists.eq_ignore_ascii_case("exists") =>
                    {
                        *name
                    }
                    [create, table, name, ..]
                        if create.eq_ignore_ascii_case("create")
                            && table.eq_ignore_ascii_case("table") =>
                    {
                        *name
                    }
                    _ => continue,
                };
                let name = name.trim_end_matches('(');
                if !SALVAGE_TABLES.contains(&name) {
                    missing.push(name.to_string());
                }
            }
        }
        assert!(missing.is_empty(), "tables no salvage copies: {missing:?}");
    }

    /// A quarantined file can predate a migration. Positional salvage failed
    /// every row of a table whose column count had since grown — here
    /// `task_activity` before migration 021 added `assignee`.
    #[tokio::test]
    async fn salvage_matches_columns_by_name_across_a_schema_change() {
        let dir = test_dir("by_name");
        let src_path = dir.join("old.db");
        let src = turso::Builder::new_local(&src_path.to_string_lossy())
            .build()
            .await
            .and_then(|db| db.connect())
            .unwrap();
        src.execute(
            "CREATE TABLE task_activity (id INTEGER PRIMARY KEY AUTOINCREMENT, \
             task_id INTEGER NOT NULL, actor TEXT, action TEXT NOT NULL, detail TEXT, \
             created_at TEXT NOT NULL DEFAULT (datetime('now')))",
            (),
        )
        .await
        .unwrap();
        src.execute(
            "INSERT INTO task_activity (task_id, actor, action) VALUES (7, 'gui', 'created')",
            (),
        )
        .await
        .unwrap();

        let fresh = Storage::in_memory().await.unwrap();
        let salvage = salvage_table(&src, &fresh, "task_activity").await;
        assert_eq!(salvage.recovered, 1, "{salvage:?}");
        let history = fresh.tasks().activity(7, 10).await.unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].action, "created");
        assert_eq!(
            history[0].assignee, None,
            "a column the source lacked takes its default"
        );

        drop(src);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The fresh store is not empty: migrations seed `human`. The user's own
    /// version of that row must win over the seed, not be skipped as a
    /// duplicate — and a custom agent must come back at all.
    #[tokio::test]
    async fn salvaged_members_replace_the_seeds_and_custom_agents_return() {
        let dir = test_dir("members");
        let config = StorageConfig {
            path: dir.join("old.db").to_string_lossy().to_string(),
        };
        let old = Storage::new(&config).await.unwrap();
        old.members()
            .update(
                crate::HUMAN_MEMBER_ID,
                crate::MemberPatch {
                    name: Some("Justin".to_string()),
                    ..crate::MemberPatch::default()
                },
            )
            .await
            .unwrap();
        old.members()
            .create(crate::NewMember {
                id: "reviewer".to_string(),
                name: "Reviewer".to_string(),
                avatar: None,
                kind: crate::MemberKind::Agent,
                owner_kind: crate::MemberOwner::Workspace,
                owner_id: None,
                status: crate::MemberStatus::Idle,
                profile: serde_json::json!({ "model": "qwen" }),
            })
            .await
            .unwrap();
        let src = old.conn().lock().await.clone();

        let fresh = Storage::in_memory().await.unwrap();
        let salvage = salvage_table(&src, &fresh, "members").await;
        assert_eq!(salvage.expected, Some(3), "{salvage:?}");
        assert_eq!(salvage.recovered, 3, "{salvage:?}");
        assert_eq!(
            fresh
                .members()
                .get(crate::HUMAN_MEMBER_ID)
                .await
                .unwrap()
                .name,
            "Justin"
        );
        assert_eq!(
            fresh.members().get("reviewer").await.unwrap().name,
            "Reviewer"
        );

        drop(src);
        drop(old);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn clean_db_opens_without_recovery() {
        let dir = test_dir("clean");
        let config = build_populated_db(&dir, 20, 50).await;

        let (storage, report) = open_with_recovery(&config).await.unwrap();
        assert!(report.is_none(), "clean store must not be rebuilt");
        assert_eq!(storage.memories().count().await.unwrap(), 20);

        drop(storage);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn page_corruption_quarantines_rebuilds_and_salvages() {
        let dir = test_dir("rebuild");
        // ~230 bytes per row spreads 200 rows over ~a dozen leaf pages, so one
        // zeroed middle leaf loses some rows while both neighbors stay readable.
        let config = build_populated_db(&dir, 200, 200).await;
        zero_needle_page(Path::new(&config.path), b"needle-0100");

        let (storage, report) = open_with_recovery(&config).await.unwrap();
        let report = report.expect("page-level corruption must trigger a rebuild");

        // The corrupt file was quarantined, not destroyed.
        assert!(report.quarantine_path.exists());
        assert_ne!(report.quarantine_path, PathBuf::from(&config.path));

        // Partial salvage: some rows live on the zeroed page (lost), the rest
        // were reachable from at least one scan direction.
        assert!(
            report.memories_recovered > 0,
            "salvage must recover reachable rows"
        );
        assert!(
            report.memories_recovered < 200,
            "rows on the zeroed page are gone"
        );
        assert_eq!(
            storage.memories().count().await.unwrap(),
            i64::try_from(report.memories_recovered).unwrap()
        );

        // Other tables ride along.
        let session = storage.sessions().get("salvage-me").await.unwrap();
        assert_eq!(session.session_id, "salvage-me");

        // The rebuilt store is fully working: reads and writes both succeed.
        storage
            .memories()
            .create(memory_row(9999, "post-recovery".into()))
            .await
            .unwrap();
        let loaded = storage.memories().bulk_load().await.unwrap();
        assert_eq!(loaded.len(), report.memories_recovered + 1);

        drop(storage);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn quarantine_copies_stay_bounded() {
        let dir = test_dir("bounded");
        let db_path = dir.join("nanna.db");

        // Simulate MAX+2 past rebuilds (name format matches quarantine_database)
        // plus a WAL sidecar on the oldest.
        for i in 0..(MAX_QUARANTINE_COPIES + 2) {
            let q = dir.join(format!("nanna.db.corrupt-20260101-00000{i}"));
            std::fs::write(&q, b"old corrupt copy").unwrap();
        }
        std::fs::write(dir.join("nanna.db.corrupt-20260101-000000-wal"), b"wal").unwrap();

        let pruned = prune_quarantines(&db_path);
        assert_eq!(pruned, 2);

        let remaining: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(Result::ok)
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.starts_with("nanna.db.corrupt-") && !n.ends_with("-wal"))
            .collect();
        assert_eq!(remaining.len(), MAX_QUARANTINE_COPIES);
        // The newest copies survive.
        assert!(remaining
            .iter()
            .all(|n| { n.ends_with('2') || n.ends_with('3') || n.ends_with('4') }));
        // The pruned copy's sidecar went with it.
        assert!(!dir.join("nanna.db.corrupt-20260101-000000-wal").exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn row_level_corruption_does_not_rebuild() {
        let dir = test_dir("rowlevel");
        // Rows with ~20KB content spill into overflow pages. Zeroing a page
        // that only holds overflow payload breaks that row's payload read but
        // leaves the leaf/interior structure (and the id scan) intact, so the
        // salvage path works and no rebuild happens.
        let config = build_populated_db(&dir, 12, 20_000).await;
        // An overflow page is 4 bytes of next-page pointer followed by payload.
        // A mid-chain page of a padded row is nonzero-pointer + all-'x' payload;
        // zeroing it breaks the chain (next -> page 0) without touching any
        // btree leaf, so only that one row becomes unreadable.
        let mut bytes = std::fs::read(&config.path).unwrap();
        let overflow_page = (1..bytes.len() / PAGE_SIZE)
            .find(|&p| {
                let page = &bytes[p * PAGE_SIZE..(p + 1) * PAGE_SIZE];
                page[..4] != [0, 0, 0, 0] && page[4..].iter().all(|&b| b == b'x')
            })
            .expect("a mid-chain overflow page of x-padding must exist");
        bytes[overflow_page * PAGE_SIZE..(overflow_page + 1) * PAGE_SIZE].fill(0);
        std::fs::write(&config.path, bytes).unwrap();

        let (storage, report) = open_with_recovery(&config).await.unwrap();
        assert!(
            report.is_none(),
            "row-level corruption is handled by the degraded load, not a rebuild"
        );
        // The file stayed in place — no quarantine copies appeared.
        let quarantined = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(Result::ok)
            .any(|e| e.file_name().to_string_lossy().contains(".corrupt-"));
        assert!(!quarantined);
        // And the salvage load still reads the intact rows.
        let salvage = storage.memories().bulk_load_salvage().await.unwrap();
        assert!(
            salvage.memories.len() < 12,
            "the corrupted row is unreadable"
        );
        assert_ne!(salvage.corrupt_ids, [] as [i64; 0]);

        drop(storage);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn corruption_classifier_matches_turso_page_errors() {
        // The 2026-07-22 incident error, as rendered through StorageError.
        assert!(is_corruption_error(&StorageError::Database(
            turso::Error::Corrupt("Invalid page type: 0".into())
        )));
        assert!(is_corruption_error(&StorageError::NotFound(
            "Database error: Invalid page type: 0".into()
        )));
        assert!(!is_corruption_error(&StorageError::Database(
            turso::Error::Error("no such table: foo".into())
        )));
    }
}
