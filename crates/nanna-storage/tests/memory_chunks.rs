#![warn(clippy::pedantic, clippy::nursery, clippy::all)]
//! `memory_chunks` behaviour: replacement, paging, the backfill queue, k-NN,
//! and the two properties that would otherwise silently rot — cascade delete
//! and secure delete.
//!
//! The chunk table holds the same secrets as its parent, split up: each row
//! carries a slice of the content and its own embedding, and an embedding is
//! invertible back to its source text by a Vec2Text-class model. So "delete the
//! memory" has to mean the chunks too, in both senses — the rows go away AND
//! their bytes are overwritten before the page is freed.

use nanna_storage::{NewMemory, NewMemoryChunk, Storage, StorageConfig};
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

fn temp_db_path(tag: &str) -> TempDb {
    let dir = std::env::temp_dir().join(format!(
        "nanna_chunks_{tag}_{}_{:p}",
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
    .expect("storage opens")
}

fn memory(id: &str) -> NewMemory {
    NewMemory {
        memory_id: id.to_string(),
        content: "parent content".to_string(),
        embedding: Some(vec![0.5, 0.5, 0.5, 0.5]),
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

fn chunk(parent: &str, ordinal: i64, text: &str, embedding: Option<Vec<f32>>) -> NewMemoryChunk {
    NewMemoryChunk {
        memory_id: parent.to_string(),
        ordinal,
        content: text.to_string(),
        char_start: ordinal * 100,
        char_end: ordinal * 100 + i64::try_from(text.len()).expect("chunk text length fits i64"),
        embedding,
        embedding_model: Some("probe".to_string()),
        chunk_max_chars: 3200,
        chunker_version: 1,
        workspace_id: None,
    }
}

/// A distinctive vector whose little-endian bytes are very unlikely to occur by
/// chance elsewhere in the file.
fn sentinel(seed: f32) -> Vec<f32> {
    (0..96_u16).map(|i| seed.mul_add(101.7, f32::from(i).mul_add(7.531_9, 3.140_1))).collect()
}

fn file_contains(db_path: &str, needle: &[u8]) -> bool {
    let bytes = std::fs::read(db_path).unwrap_or_default();
    let wal = std::fs::read(format!("{db_path}-wal")).unwrap_or_default();
    bytes.windows(needle.len()).any(|w| w == needle)
        || wal.windows(needle.len()).any(|w| w == needle)
}

#[tokio::test]
async fn replacing_chunks_swaps_the_whole_set() {
    let db_path = temp_db_path("replace");
    let storage = open(&db_path).await;
    let repo = storage.memories();
    repo.create(memory("m1")).await.expect("create");

    repo.replace_chunks(
        "m1",
        &[
            chunk("m1", 0, "first", Some(vec![1.0, 0.0, 0.0, 0.0])),
            chunk("m1", 1, "second", Some(vec![0.0, 1.0, 0.0, 0.0])),
            chunk("m1", 2, "third", Some(vec![0.0, 0.0, 1.0, 0.0])),
        ],
    )
    .await
    .expect("write chunks");
    assert_eq!(repo.chunk_count("m1").await.expect("count"), 3);

    // Re-chunking under a different budget yields a different COUNT. Leaving
    // the old ordinals behind would mix two chunkings of the same text and
    // double-count it in search.
    repo.replace_chunks(
        "m1",
        &[chunk("m1", 0, "one bigger chunk", Some(vec![1.0, 1.0, 0.0, 0.0]))],
    )
    .await
    .expect("re-chunk");
    let chunks = repo.chunks_for("m1").await.expect("read");
    assert_eq!(chunks.len(), 1, "the previous set must be gone, not merged");
    assert_eq!(chunks[0].content, "one bigger chunk");
}

#[tokio::test]
async fn pages_are_ordinal_ordered_and_one_based() {
    let db_path = temp_db_path("page");
    let storage = open(&db_path).await;
    let repo = storage.memories();
    repo.create(memory("m1")).await.expect("create");
    let set: Vec<NewMemoryChunk> = (0..5)
        .map(|i| chunk("m1", i, &format!("chunk {i}"), Some(vec![nanna_numeric::f32_from_i64(i), 0.0, 0.0, 0.0])))
        .collect();
    repo.replace_chunks("m1", &set).await.expect("write");

    let page1 = repo.chunk_page("m1", 1, 2).await.expect("page 1");
    let page2 = repo.chunk_page("m1", 2, 2).await.expect("page 2");
    let page3 = repo.chunk_page("m1", 3, 2).await.expect("page 3");

    assert_eq!(
        page1.iter().map(|c| c.ordinal).collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert_eq!(
        page2.iter().map(|c| c.ordinal).collect::<Vec<_>>(),
        vec![2, 3]
    );
    assert_eq!(page3.len(), 1, "last page is short, not empty");
    assert!(
        repo.chunk_page("m1", 4, 2).await.expect("past end").is_empty(),
        "past the end is empty rather than an error"
    );
}

#[tokio::test]
async fn the_backfill_queue_finds_missing_and_stale_vectors() {
    let db_path = temp_db_path("backfill");
    let storage = open(&db_path).await;
    let repo = storage.memories();
    repo.create(memory("m1")).await.expect("create");

    let mut stale = chunk("m1", 1, "written by an older model", Some(vec![0.1, 0.2, 0.3, 0.4]));
    stale.embedding_model = Some("previous-model".to_string());

    repo.replace_chunks(
        "m1",
        &[
            chunk("m1", 0, "no vector yet", None),
            stale,
            chunk("m1", 2, "current", Some(vec![0.4, 0.3, 0.2, 0.1])),
        ],
    )
    .await
    .expect("write");

    let pending = repo
        .chunks_needing_embedding("probe", 10, true)
        .await
        .expect("queue");
    let ordinals: Vec<i64> = pending.iter().map(|c| c.ordinal).collect();
    assert_eq!(
        ordinals,
        vec![0, 1],
        "both the never-embedded chunk and the one from another model are stale; \
         vectors from different models are not comparable"
    );

    // Embedding it clears it from the queue — this is what makes a model switch
    // a resumable backfill instead of a rebuild.
    let id = pending[0].id;
    assert!(repo
        .set_chunk_embedding(id, &[9.0, 9.0, 9.0, 9.0], "probe")
        .await
        .expect("set"));
    let after = repo
        .chunks_needing_embedding("probe", 10, true)
        .await
        .expect("queue again");
    assert_eq!(after.len(), 1, "the refilled chunk left the queue");
}

#[tokio::test]
async fn parents_without_chunks_lists_pre_migration_rows() {
    let db_path = temp_db_path("parents");
    let storage = open(&db_path).await;
    let repo = storage.memories();
    repo.create(memory("old-1")).await.expect("create");
    repo.create(memory("old-2")).await.expect("create");
    repo.create(memory("chunked")).await.expect("create");
    repo.replace_chunks("chunked", &[chunk("chunked", 0, "x", Some(vec![1.0, 0.0, 0.0, 0.0]))])
        .await
        .expect("write");

    let pending = repo.parents_without_chunks(10).await.expect("queue");
    assert_eq!(pending, vec!["old-1".to_string(), "old-2".to_string()]);
}

#[tokio::test]
async fn chunk_knn_returns_parent_and_ordinal_nearest_first() {
    let db_path = temp_db_path("knn");
    let storage = open(&db_path).await;
    let repo = storage.memories();
    repo.create(memory("m1")).await.expect("create");
    repo.create(memory("m2")).await.expect("create");

    repo.replace_chunks("m1", &[chunk("m1", 0, "far", Some(vec![0.0, 1.0, 0.0, 0.0]))])
        .await
        .expect("write m1");
    repo.replace_chunks("m2", &[chunk("m2", 3, "near", Some(vec![1.0, 0.0, 0.0, 0.0]))])
        .await
        .expect("write m2");

    let hits = repo
        .search_chunks_by_embedding_sql(&[1.0, 0.0, 0.0, 0.0], "probe", 10, None)
        .await
        .expect("knn");
    assert_ne!(hits, [] as [(std::string::String, i64, f64); 0]);
    assert_eq!(hits[0].0, "m2", "nearest chunk's parent comes first");
    assert_eq!(hits[0].1, 3, "the winning chunk's ordinal is reported");
}

/// Nothing sets `PRAGMA foreign_keys = ON`, so the REFERENCES clause enforces
/// nothing. Without an explicit cascade a deleted memory keeps matching queries
/// through orphaned chunk vectors.
#[tokio::test]
async fn deleting_a_memory_deletes_its_chunks() {
    let db_path = temp_db_path("cascade");
    let storage = open(&db_path).await;
    let repo = storage.memories();
    repo.create(memory("m1")).await.expect("create");
    repo.replace_chunks(
        "m1",
        &[
            chunk("m1", 0, "a", Some(vec![1.0, 0.0, 0.0, 0.0])),
            chunk("m1", 1, "b", Some(vec![0.0, 1.0, 0.0, 0.0])),
        ],
    )
    .await
    .expect("write");

    assert!(repo.delete("m1").await.expect("delete"));
    assert_eq!(
        repo.chunk_count("m1").await.expect("count"),
        0,
        "orphaned chunks would keep a deleted memory searchable"
    );
    let hits = repo
        .search_chunks_by_embedding_sql(&[1.0, 0.0, 0.0, 0.0], "probe", 10, None)
        .await
        .expect("knn");
    assert!(hits.is_empty(), "no vector of a deleted memory may survive");
}

#[tokio::test]
async fn bulk_delete_also_deletes_chunks() {
    let db_path = temp_db_path("bulk_cascade");
    let storage = open(&db_path).await;
    let repo = storage.memories();
    for id in ["m1", "m2"] {
        repo.create(memory(id)).await.expect("create");
        repo.replace_chunks(id, &[chunk(id, 0, "x", Some(vec![1.0, 0.0, 0.0, 0.0]))])
            .await
            .expect("write");
    }
    repo.bulk_delete(&["m1", "m2"]).await.expect("bulk delete");
    assert_eq!(repo.chunk_count("m1").await.expect("c1"), 0);
    assert_eq!(repo.chunk_count("m2").await.expect("c2"), 0);
}

/// Ghost Vectors, through the new table. A chunk embedding is invertible back
/// to its source text, so deleting a memory must destroy the chunk vectors on
/// disk — not merely unlink the rows.
#[tokio::test]
async fn deleting_a_memory_destroys_its_chunk_embeddings_on_disk() {
    let db_path = temp_db_path("ghost");
    let vec_bytes: Vec<u8> = sentinel(1.0).iter().flat_map(|f| f.to_le_bytes()).collect();

    {
        let storage = open(&db_path).await;
        let repo = storage.memories();
        repo.create(memory("m1")).await.expect("create");
        repo.replace_chunks("m1", &[chunk("m1", 0, "the user's password hint", Some(sentinel(1.0)))])
            .await
            .expect("write");
    }
    assert!(
        file_contains(&db_path, &vec_bytes),
        "precondition: the chunk vector must be on disk before the delete, \
         or the assertion below proves nothing"
    );

    {
        let storage = open(&db_path).await;
        storage.memories().delete("m1").await.expect("delete");
    }

    assert!(
        !file_contains(&db_path, &vec_bytes),
        "the chunk embedding survived the delete — a Vec2Text-class model can \
         invert it back to the source text"
    );
}

/// The same, for the chunk's plaintext.
#[tokio::test]
async fn deleting_a_memory_destroys_its_chunk_text_on_disk() {
    let db_path = temp_db_path("ghost_text");
    let secret = "sentinel-chunk-plaintext-9c1f2a";

    {
        let storage = open(&db_path).await;
        let repo = storage.memories();
        repo.create(memory("m1")).await.expect("create");
        repo.replace_chunks("m1", &[chunk("m1", 0, secret, Some(sentinel(2.0)))])
            .await
            .expect("write");
    }
    assert!(
        file_contains(&db_path, secret.as_bytes()),
        "precondition: chunk text must be on disk first"
    );

    {
        let storage = open(&db_path).await;
        storage.memories().delete("m1").await.expect("delete");
    }

    let db = Builder::new_local(&db_path).build().await.expect("reopen");
    drop(db);
    assert!(
        !file_contains(&db_path, secret.as_bytes()),
        "chunk plaintext survived the delete"
    );
}

/// Equal width is not equal vector space: two different models produce vectors
/// that `vector_distance_cos` compares without complaint and that mean nothing.
/// The k-NN must therefore filter on the model, not merely on `octet_length` —
/// which it did not when chunk search first landed, so every chunk still
/// carrying a previous provider's vector could rank its parent into results
/// after any same-width failover.
#[tokio::test]
async fn chunk_knn_filters_on_model_not_only_width() {
    let db_path = temp_db_path("knn_model");
    let storage = open(&db_path).await;
    let repo = storage.memories();
    repo.create(memory("m1")).await.expect("create");

    // Same width, different model, and a near-perfect match on the query.
    let mut stale = chunk("m1", 0, "stale", Some(vec![1.0, 0.0, 0.0, 0.0]));
    stale.embedding_model = Some("prov:old".to_string());
    repo.replace_chunks("m1", &[stale]).await.expect("write");

    let foreign = repo
        .search_chunks_by_embedding_sql(&[1.0, 0.0, 0.0, 0.0], "prov:new", 10, None)
        .await
        .expect("knn");
    assert!(
        foreign.is_empty(),
        "a perfect-scoring vector from another model must not be returned"
    );

    let own = repo
        .search_chunks_by_embedding_sql(&[1.0, 0.0, 0.0, 0.0], "prov:old", 10, None)
        .await
        .expect("knn");
    assert_eq!(own.len(), 1, "the same row is returned under its own model");
}
