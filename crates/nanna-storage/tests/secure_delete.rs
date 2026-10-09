#![warn(clippy::pedantic, clippy::nursery, clippy::all)]
//! Ghost Vectors regression: deleting a memory must destroy its embedding on disk.
//!
//! An embedding is invertible back to its source text by a Vec2Text-class model
//! ("Ghost Vectors", arXiv 2606.18497), and Turso — like SQLite — frees a deleted
//! row's page onto the free-list *without zeroing it*, with no `secure_delete`
//! pragma or `VACUUM` to force the issue. So a plain `DELETE` leaves the f32 BLOB
//! recoverable in the database file, which for a local-first product means it
//! survives on the user's disk (and in their backups) after they asked to forget it.
//!
//! `MemoryRepository::delete` overwrites the sensitive columns with same-length
//! `zeroblob`s before freeing the page. These tests pin that end to end through the
//! real dependency: the sentinel embedding is present on disk before the delete and
//! **absent after** — a constant-returning or no-op overwrite could not pass, because
//! the negative control (`raw_delete_*`, a plain `DELETE`) shows the bytes DO survive.

use nanna_storage::{NewMemory, Storage, StorageConfig};
use turso::Builder;

/// A distinctive embedding whose little-endian bytes are extremely unlikely to
/// occur by chance elsewhere in the database file (a 384-byte ramp). `seed`
/// shifts the ramp so callers can mint several mutually-distinct sentinels.
fn sentinel_embedding_seeded(seed: f32) -> Vec<f32> {
    (0..96_u16)
        .map(|i| seed.mul_add(101.7, f32::from(i).mul_add(7.531_9, 3.140_1)))
        .collect()
}

fn sentinel_embedding() -> Vec<f32> {
    sentinel_embedding_seeded(0.0)
}

fn embedding_bytes(embedding: &[f32]) -> Vec<u8> {
    embedding.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// True if `needle` appears as a contiguous byte subsequence of `haystack`.
fn contains_subsequence(haystack: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() || haystack.len() < needle.len() {
        return false;
    }
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}

/// Read and concatenate the bytes of every file whose name starts with the db
/// file's name (covers the main `.db`, plus any `-wal` / `-journal` sidecars).
fn read_all_db_bytes(db_path: &str) -> Vec<u8> {
    let path = std::path::Path::new(db_path);
    let dir = path.parent().expect("db path has a parent");
    let stem = path
        .file_name()
        .expect("db path has a file name")
        .to_string_lossy()
        .to_string();
    let mut bytes = Vec::new();
    let entries = std::fs::read_dir(dir).expect("temp dir is readable");
    for entry in entries {
        let entry = entry.expect("dir entry is readable");
        if !entry.file_name().to_string_lossy().starts_with(&stem) {
            continue;
        }
        if let Ok(data) = std::fs::read(entry.path()) {
            bytes.extend_from_slice(&data);
        }
    }
    bytes
}

fn new_memory(id: &str, embedding: Vec<f32>) -> NewMemory {
    NewMemory {
        memory_id: id.to_string(),
        content: "the user's home address is 1 sentinel road".to_string(),
        embedding: Some(embedding),
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
    }
}

/// Fresh, isolated temp directory for one test's database files.
fn temp_db_path(tag: &str) -> (std::path::PathBuf, String) {
    let dir = std::env::temp_dir().join(format!(
        "nanna_ghost_{tag}_{}_{:p}",
        std::process::id(),
        &raw const tag
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let db_path = dir.join("mem.db").to_string_lossy().to_string();
    (dir, db_path)
}

/// Insert the sentinel memory, then close so the row lands in the file, and
/// return whether the embedding bytes are present on disk (they must be — this
/// is the precondition that makes the "absent after delete" assertion meaningful).
async fn insert_and_confirm_present(db_path: &str, needle: &[u8]) {
    {
        let storage = Storage::new(&StorageConfig {
            path: db_path.to_string(),
        })
        .await
        .expect("open storage");
        storage
            .memories()
            .create(new_memory("ghost-1", sentinel_embedding()))
            .await
            .expect("insert memory");
    }
    let before = read_all_db_bytes(db_path);
    assert!(
        contains_subsequence(&before, needle),
        "precondition: the embedding must be on disk before the delete"
    );
}

#[tokio::test]
async fn secure_delete_removes_embedding_from_disk() {
    let (dir, db_path) = temp_db_path("secure");
    let needle = embedding_bytes(&sentinel_embedding());

    insert_and_confirm_present(&db_path, &needle).await;

    // Reopen and delete through the real (secure) path, then close.
    {
        let storage = Storage::new(&StorageConfig {
            path: db_path.clone(),
        })
        .await
        .expect("reopen storage");
        let deleted = storage.memories().delete("ghost-1").await.expect("delete");
        assert!(deleted, "delete should report a removed row");
    }

    let after = read_all_db_bytes(&db_path);
    assert!(
        !contains_subsequence(&after, &needle),
        "the embedding must NOT survive on disk after a secure delete"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn bulk_delete_removes_all_embeddings_from_disk() {
    let (dir, db_path) = temp_db_path("bulk");

    // Three memories, three mutually-distinct sentinel embeddings.
    let seeds = [1.0_f32, 2.0, 3.0];
    let needles: Vec<Vec<u8>> = seeds
        .iter()
        .map(|s| embedding_bytes(&sentinel_embedding_seeded(*s)))
        .collect();

    {
        let storage = Storage::new(&StorageConfig {
            path: db_path.clone(),
        })
        .await
        .expect("open storage");
        for (idx, seed) in seeds.iter().enumerate() {
            storage
                .memories()
                .create(new_memory(
                    &format!("bulk-{idx}"),
                    sentinel_embedding_seeded(*seed),
                ))
                .await
                .expect("insert memory");
        }
    }

    let before = read_all_db_bytes(&db_path);
    for needle in &needles {
        assert!(
            contains_subsequence(&before, needle),
            "precondition: every embedding must be on disk before the bulk delete"
        );
    }

    {
        let storage = Storage::new(&StorageConfig {
            path: db_path.clone(),
        })
        .await
        .expect("reopen storage");
        let ids = ["bulk-0", "bulk-1", "bulk-2"];
        let deleted = storage
            .memories()
            .bulk_delete(&ids)
            .await
            .expect("bulk delete");
        assert_eq!(deleted, 3, "all three rows should be deleted");
    }

    let after = read_all_db_bytes(&db_path);
    for needle in &needles {
        assert!(
            !contains_subsequence(&after, needle),
            "no embedding may survive on disk after a secure bulk delete"
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
}

/// An in-memory database has no WAL file, so the checkpoint step must degrade
/// gracefully (best-effort, logged) and the delete must still succeed.
#[tokio::test]
async fn secure_delete_is_graceful_on_in_memory_db() {
    let storage = Storage::in_memory().await.expect("open in-memory storage");
    storage
        .memories()
        .create(new_memory("mem-1", sentinel_embedding()))
        .await
        .expect("insert memory");
    let deleted = storage.memories().delete("mem-1").await.expect("delete");
    assert!(deleted, "in-memory delete should report a removed row");
    // A second delete of the same id removes nothing but must not error.
    let again = storage
        .memories()
        .delete("mem-1")
        .await
        .expect("delete again");
    assert!(
        !again,
        "re-deleting a missing id returns false, not an error"
    );
}

/// Negative control: a plain `DELETE` (bypassing the secure path) DOES leave the
/// embedding on disk. If this ever stops holding, the secure-delete test above is
/// no longer proving anything and both need revisiting.
#[tokio::test]
async fn raw_delete_leaves_embedding_on_disk() {
    let (dir, db_path) = temp_db_path("raw");
    let needle = embedding_bytes(&sentinel_embedding());

    insert_and_confirm_present(&db_path, &needle).await;

    // Reopen with a bare turso connection and issue a plain DELETE — no overwrite.
    {
        let conn = Builder::new_local(&db_path)
            .build()
            .await
            .expect("reopen raw")
            .connect()
            .expect("connect raw");
        conn.execute("DELETE FROM memories WHERE memory_id = ?1", ["ghost-1"])
            .await
            .expect("raw delete");
    }

    let after = read_all_db_bytes(&db_path);
    assert!(
        contains_subsequence(&after, &needle),
        "control: a plain DELETE is expected to leave the embedding recoverable on disk"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// Rewriting a memory's content (every dream consolidation does) must destroy
/// the superseded embedding on disk too — it is a copy of the text that was
/// just replaced.
#[tokio::test]
async fn a_content_rewrite_removes_the_old_embedding_from_disk() {
    let (dir, db_path) = temp_db_path("rewrite");
    let needle = embedding_bytes(&sentinel_embedding());

    insert_and_confirm_present(&db_path, &needle).await;
    {
        let storage = Storage::new(&StorageConfig {
            path: db_path.clone(),
        })
        .await
        .expect("reopen storage");
        let updated = storage
            .memories()
            .update_content("ghost-1", "rewritten by a dream")
            .await
            .expect("rewrite");
        assert!(updated);
    }
    let after = read_all_db_bytes(&db_path);
    assert!(
        !contains_subsequence(&after, &needle),
        "the superseded embedding must not survive a content rewrite on disk"
    );
    let _ = std::fs::remove_dir_all(dir);
}
