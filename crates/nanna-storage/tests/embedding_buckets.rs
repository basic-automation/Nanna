#![warn(clippy::pedantic, clippy::nursery, clippy::all)]
//! Per-model embedding buckets and the durable queue.
//!
//! The property under test throughout: **a model's vectors are never destroyed
//! by another model's**. Before buckets, adopting a new embedder overwrote the
//! single vector column, so a provider that flapped away and back cost two full
//! re-embeds of the entire store — and a provider that failed mid-session left
//! the store with no usable index at all.

use nanna_storage::{NewMemory, NewMemoryChunk, Storage, StorageConfig, ROW_ORDINAL};

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
        "nanna_buckets_{tag}_{}_{:p}",
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
        content: format!("content of {id}"),
        embedding: None,
        embedding_model: None,
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

/// Scan the database AND its write-ahead log.
///
/// turso 0.6.1 keeps every committed page in the WAL and does not checkpoint on
/// close, so a test that reads only the `.db` file proves nothing — the bytes
/// it is looking for have simply not been written there yet.
fn file_contains(db_path: &str, needle: &[u8]) -> bool {
    let bytes = std::fs::read(db_path).unwrap_or_default();
    let wal = std::fs::read(format!("{db_path}-wal")).unwrap_or_default();
    bytes.windows(needle.len()).any(|w| w == needle)
        || wal.windows(needle.len()).any(|w| w == needle)
}

/// A distinctive vector whose little-endian bytes will not occur by chance.
fn sentinel() -> Vec<f32> {
    (0..96_u16).map(|i| f32::from(i).mul_add(7.531_9, 3.140_1) + 404.5).collect()
}

fn chunk(parent: &str, ordinal: i64, text: &str) -> NewMemoryChunk {
    NewMemoryChunk {
        memory_id: parent.to_string(),
        ordinal,
        content: text.to_string(),
        char_start: ordinal * 100,
        char_end: ordinal * 100 + i64::try_from(text.len()).expect("chunk text length fits i64"),
        embedding: None,
        embedding_model: None,
        chunk_max_chars: 3200,
        chunker_version: 1,
        workspace_id: None,
    }
}

/// Two models' vectors coexist on one memory. This is the entire reason the
/// table exists: the previous single-column layout meant adopting a new model
/// destroyed the old one's work, so returning to a provider after an outage
/// cost a full re-embed instead of a lookup.
#[tokio::test]
async fn two_models_hold_separate_vectors_for_one_memory() {
    let db = temp_db_path("coexist");
    let storage = open(&db).await;
    let repo = storage.memories();
    repo.create(memory("m1")).await.expect("create");

    repo.put_memory_vector("m1", "openrouter:nemotron", &[1.0, 0.0, 0.0, 0.0])
        .await
        .expect("first bucket");
    repo.put_memory_vector("m1", "ollama:nomic", &[0.0, 1.0, 0.0])
        .await
        .expect("second bucket, different width");

    let mut buckets = repo.memory_vectors("m1").await.expect("read");
    buckets.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(buckets.len(), 2, "both models must survive");
    assert_eq!(buckets[0].0, "ollama:nomic");
    assert_eq!(buckets[0].1, vec![0.0, 1.0, 0.0], "widths may differ per bucket");
    assert_eq!(buckets[1].0, "openrouter:nemotron");
    assert_eq!(buckets[1].1, vec![1.0, 0.0, 0.0, 0.0]);
}

/// Re-embedding under the same model replaces that bucket only.
#[tokio::test]
async fn rewriting_one_bucket_leaves_the_others_intact() {
    let db = temp_db_path("rewrite");
    let storage = open(&db).await;
    let repo = storage.memories();
    repo.create(memory("m1")).await.expect("create");

    repo.put_memory_vector("m1", "a", &[1.0, 0.0]).await.expect("a");
    repo.put_memory_vector("m1", "b", &[0.0, 1.0]).await.expect("b");
    repo.put_memory_vector("m1", "a", &[0.5, 0.5]).await.expect("a again");

    let buckets = repo.memory_vectors("m1").await.expect("read");
    assert_eq!(buckets.len(), 2, "an update must not add a row");
    let a = buckets.iter().find(|(m, _)| m == "a").expect("a present");
    let b = buckets.iter().find(|(m, _)| m == "b").expect("b present");
    assert_eq!(a.1, vec![0.5, 0.5], "a was replaced");
    assert_eq!(b.1, vec![0.0, 1.0], "b was untouched");
}

/// The census drives bucket selection, so it has to be ordered and accurate.
#[tokio::test]
async fn the_bucket_census_counts_per_model_largest_first() {
    let db = temp_db_path("census");
    let storage = open(&db).await;
    let repo = storage.memories();
    for i in 0..5 {
        repo.create(memory(&format!("m{i}"))).await.expect("create");
        repo.put_memory_vector(&format!("m{i}"), "big", &[1.0, 0.0])
            .await
            .expect("big");
    }
    repo.put_memory_vector("m0", "small", &[0.0, 1.0]).await.expect("small");

    let census = repo.bucket_counts().await.expect("census");
    assert_eq!(census[0], ("big".to_string(), 5), "largest bucket first");
    assert_eq!(census[1], ("small".to_string(), 1));
}

/// A queue that lives only in RAM is rebuilt from scratch on every restart and
/// forgets what it already tried. This one is a table, so pending work and its
/// failure history survive the process.
#[tokio::test]
async fn queued_work_survives_and_carries_its_failure_reason() {
    let db = temp_db_path("queue");
    let storage = open(&db).await;
    let repo = storage.memories();
    repo.create(memory("m1")).await.expect("create");

    repo.enqueue_embedding("m1", ROW_ORDINAL, "ollama:nomic").await.expect("enqueue");
    repo.enqueue_embedding("m1", 0, "ollama:nomic").await.expect("enqueue chunk");

    let pending = repo.pending_embeddings("ollama:nomic", 10).await.expect("pending");
    assert_eq!(pending.len(), 2, "row work and chunk work share one queue");

    repo.record_embedding_failure("m1", 0, "ollama:nomic", "402 Insufficient credits")
        .await
        .expect("record");
    let health = repo.embedding_queue_health().await.expect("health");
    assert_eq!(health[0].0, "ollama:nomic");
    assert_eq!(health[0].1, 2, "depth is visible");
    assert!(
        health[0].2.as_deref().unwrap_or("").contains("402"),
        "the reason must survive, or a stuck queue is indistinguishable from a slow one"
    );

    repo.dequeue_embedding("m1", 0, "ollama:nomic").await.expect("dequeue");
    assert_eq!(
        repo.pending_embeddings("ollama:nomic", 10).await.expect("pending").len(),
        1
    );
}

/// Work is queued per model, so switching embedders does not disturb another
/// model's outstanding work.
#[tokio::test]
async fn the_queue_is_keyed_by_model() {
    let db = temp_db_path("queue_model");
    let storage = open(&db).await;
    let repo = storage.memories();
    repo.create(memory("m1")).await.expect("create");

    repo.enqueue_embedding("m1", 0, "model-a").await.expect("a");
    repo.enqueue_embedding("m1", 0, "model-b").await.expect("b");

    assert_eq!(repo.pending_embeddings("model-a", 10).await.expect("a").len(), 1);
    repo.dequeue_embedding("m1", 0, "model-a").await.expect("dequeue a");
    assert_eq!(repo.pending_embeddings("model-a", 10).await.expect("a").len(), 0);
    assert_eq!(
        repo.pending_embeddings("model-b", 10).await.expect("b").len(),
        1,
        "finishing one model's work must not clear another's"
    );
}

/// Re-enqueueing must not reset the attempt count, or an item that fails
/// forever could disguise itself as fresh work and hold the front of the queue.
#[tokio::test]
async fn re_enqueueing_does_not_erase_attempt_history() {
    let db = temp_db_path("requeue");
    let storage = open(&db).await;
    let repo = storage.memories();
    repo.create(memory("m1")).await.expect("create");
    repo.create(memory("m2")).await.expect("create");

    repo.enqueue_embedding("m1", 0, "m").await.expect("enqueue");
    repo.record_embedding_failure("m1", 0, "m", "boom").await.expect("fail");
    repo.record_embedding_failure("m1", 0, "m", "boom").await.expect("fail");
    repo.enqueue_embedding("m1", 0, "m").await.expect("re-enqueue");
    repo.enqueue_embedding("m2", 0, "m").await.expect("fresh work");

    let pending = repo.pending_embeddings("m", 10).await.expect("pending");
    assert_eq!(
        pending[0].0, "m2",
        "fresh work outranks an item that has already failed twice"
    );
}

/// Nothing sets `PRAGMA foreign_keys`, so the cascade is ours to write. An
/// orphaned bucket is worse than an orphaned chunk: it is a live vector for
/// text that no longer exists, and it keeps matching queries forever.
#[tokio::test]
async fn deleting_a_memory_takes_its_buckets_and_queue_with_it() {
    let db = temp_db_path("cascade");
    let storage = open(&db).await;
    let repo = storage.memories();
    repo.create(memory("m1")).await.expect("create");
    repo.replace_chunks("m1", &[chunk("m1", 0, "a")]).await.expect("chunks");
    repo.put_memory_vector("m1", "model", &[1.0, 0.0]).await.expect("row vector");
    repo.put_chunk_vector("m1", 0, "model", &[0.0, 1.0]).await.expect("chunk vector");
    repo.enqueue_embedding("m1", 0, "model").await.expect("queue");

    assert!(repo.delete("m1").await.expect("delete"));

    assert_eq!(repo.memory_vectors("m1").await.expect("read"), [] as [(std::string::String, std::vec::Vec<f32>); 0]);
    assert_eq!(repo.bucket_counts().await.expect("census").len(), 0);
    assert_eq!(repo.pending_embeddings("model", 10).await.expect("pending"), [] as [(std::string::String, i64); 0]);
}

/// Ghost Vectors (arXiv 2606.18497): an embedding inverts back to its source
/// text, so every bucket is an independent copy of the memory. Zeroing one and
/// leaving another reconstructs the memory just as well as leaving the first.
#[tokio::test]
async fn deleting_a_memory_zeroes_every_bucket_not_just_one() {
    let db = temp_db_path("ghost");
    let storage = open(&db).await;
    let repo = storage.memories();
    repo.create(memory("m1")).await.expect("create");
    repo.replace_chunks("m1", &[chunk("m1", 0, "a")]).await.expect("chunks");

    let secret = sentinel();
    repo.put_memory_vector("m1", "model-a", &secret).await.expect("a");
    repo.put_memory_vector("m1", "model-b", &secret).await.expect("b");
    repo.put_chunk_vector("m1", 0, "model-a", &secret).await.expect("chunk a");

    let needle: Vec<u8> = secret.iter().flat_map(|f| f.to_le_bytes()).collect();
    assert!(
        file_contains(&db, &needle),
        "precondition: the vector is on disk before the delete"
    );

    repo.delete("m1").await.expect("delete");

    assert!(
        !file_contains(&db, &needle),
        "a surviving bucket inverts back to the deleted memory's text — zeroing          one model's vector while another remains is not a delete"
    );
}

/// A failure recorded against work that was never enqueued must still leave a
/// trace. A bare UPDATE matches nothing there and reports success — the exact
/// silence this table exists to remove, reappearing inside the mechanism meant
/// to prevent it.
#[tokio::test]
async fn a_failure_on_unqueued_work_still_leaves_a_record() {
    let db = temp_db_path("unqueued_failure");
    let storage = open(&db).await;
    let repo = storage.memories();
    repo.create(memory("m1")).await.expect("create");

    // Never enqueued.
    repo.record_embedding_failure("m1", 0, "model", "402 Insufficient credits")
        .await
        .expect("record");

    let health = repo.embedding_queue_health().await.expect("health");
    assert_eq!(health.len(), 1, "the failure must create the work item it names");
    assert_eq!(health[0].1, 1);
    assert!(health[0].2.as_deref().unwrap_or("").contains("402"));
}

/// Repeated failures accumulate rather than resetting, so a poison item is
/// visibly poison.
#[tokio::test]
async fn repeated_failures_accumulate_attempts() {
    let db = temp_db_path("accumulate");
    let storage = open(&db).await;
    let repo = storage.memories();
    repo.create(memory("m1")).await.expect("create");

    for _ in 0..3 {
        repo.record_embedding_failure("m1", 0, "model", "boom")
            .await
            .expect("record");
    }
    let pending = repo.pending_embeddings("model", 10).await.expect("pending");
    assert_eq!(pending.len(), 1, "three failures are one work item, not three");
}

/// The queue has to be the thing the drain READS, not just a table it writes.
/// It also has to work on a database whose chunks predate it, or the durable
/// queue would replace a derived one and know about nothing.
#[tokio::test]
async fn the_drain_reads_the_queue_and_seeds_it_from_existing_chunks() {
    let db = temp_db_path("drain");
    let storage = open(&db).await;
    let repo = storage.memories();
    repo.create(memory("m1")).await.expect("create");
    repo.replace_chunks(
        "m1",
        &[chunk("m1", 0, "alpha"), chunk("m1", 1, "beta")],
    )
    .await
    .expect("chunks");

    // Nothing was enqueued explicitly — these chunks predate the queue.
    let work = repo.chunks_needing_embedding("model", 10, true).await.expect("work");
    assert_eq!(work.len(), 2, "the drain must seed itself from existing chunks");
    assert!(work.iter().any(|c| c.content == "alpha"));
}

/// A chunk already embedded must NOT be re-embedded on migration. Without
/// adopting the materialized vectors into the bucket table first, the seed sees
/// an empty bucket table and re-indexes the entire store against a provider
/// that may be paid or rate-limited.
#[tokio::test]
async fn migration_does_not_re_embed_work_already_done() {
    let db = temp_db_path("adopt");
    let storage = open(&db).await;
    let repo = storage.memories();
    repo.create(memory("m1")).await.expect("create");

    let mut done = chunk("m1", 0, "already embedded");
    done.embedding = Some(vec![1.0, 0.0]);
    done.embedding_model = Some("model".to_string());
    repo.replace_chunks("m1", &[done, chunk("m1", 1, "not yet")])
        .await
        .expect("chunks");

    let work = repo.chunks_needing_embedding("model", 10, true).await.expect("work");
    assert_eq!(work.len(), 1, "only the unembedded chunk is work");
    assert_eq!(work[0].content, "not yet");
}

/// Returning to a provider must be free. Its chunk vectors are still on disk,
/// so a rebind re-activates them instead of queueing them for a second embed.
#[tokio::test]
async fn returning_to_a_provider_reactivates_its_chunk_vectors() {
    let db = temp_db_path("flap");
    let storage = open(&db).await;
    let repo = storage.memories();
    repo.create(memory("m1")).await.expect("create");
    repo.replace_chunks("m1", &[chunk("m1", 0, "text")]).await.expect("chunks");

    // Provider A embeds it, then we fail over to B.
    let work = repo.chunks_needing_embedding("prov:a", 10, true).await.expect("work");
    repo.set_chunk_embedding(work[0].id, &[1.0, 0.0], "prov:a").await.expect("embed a");
    let b_work = repo.chunks_needing_embedding("prov:b", 10, true).await.expect("work b");
    assert_eq!(b_work.len(), 1, "B has not embedded this chunk");
    repo.set_chunk_embedding(b_work[0].id, &[0.0, 1.0], "prov:b").await.expect("embed b");

    // Back to A. Its vector is still stored, so nothing is owed.
    let restored = repo.restore_chunk_vectors("prov:a").await.expect("restore");
    assert_eq!(restored, 1, "A's stored vector is re-activated");
    assert!(
        repo.chunks_needing_embedding("prov:a", 10, true).await.expect("work").is_empty(),
        "returning to a provider must not re-embed what it already did"
    );

    let hits = repo
        .search_chunks_by_embedding_sql(&[1.0, 0.0], "prov:a", 10, None)
        .await
        .expect("knn");
    assert_eq!(hits.len(), 1, "the restored vector is searchable under A");
}

/// Rewriting a memory's content invalidates every vector for it — each one
/// describes text that no longer exists. Keeping any would leave the memory
/// findable by words it no longer contains, and the backfill would never
/// replace it because a bucket already exists for the active model.
#[tokio::test]
async fn rewriting_content_discards_every_stale_bucket() {
    let db = temp_db_path("prune");
    let storage = open(&db).await;
    let repo = storage.memories();
    repo.create(memory("m1")).await.expect("create");
    repo.replace_chunks("m1", &[chunk("m1", 0, "old text")]).await.expect("chunks");

    let secret = sentinel();
    repo.put_memory_vector("m1", "a", &secret).await.expect("a");
    repo.put_memory_vector("m1", "b", &secret).await.expect("b");
    repo.put_chunk_vector("m1", 0, "a", &secret).await.expect("chunk a");

    repo.clear_memory_buckets("m1").await.expect("clear");

    assert_eq!(repo.memory_vectors("m1").await.expect("read"), [] as [(std::string::String, std::vec::Vec<f32>); 0]);
    let needle: Vec<u8> = secret.iter().flat_map(|f| f.to_le_bytes()).collect();
    assert!(
        !file_contains(&db, &needle),
        "a superseded vector is a copy of the superseded text, so it must be zeroed, not just unlinked"
    );
}

/// The resurrection sequence, verbatim from the review that found it: embed a
/// chunk, rewrite the parent's text, then rebind to the same model. The stale
/// bucket used to win twice — `restore_chunk_vectors` copied the OLD text's
/// vector back into the active column, then deleted the queue row that would
/// have re-embedded the new text, and the seed refused to re-add it because a
/// bucket existed. The chunk was permanently searchable by words it no longer
/// contained, and the only log line said "Re-activated 1 chunk vectors".
#[tokio::test]
async fn a_rewrite_then_rebind_does_not_resurrect_the_old_texts_vector() {
    let db = temp_db_path("resurrect");
    let storage = open(&db).await;
    let repo = storage.memories();
    repo.create(memory("m1")).await.expect("create");

    // Embed the original text under prov:a.
    repo.replace_chunks("m1", &[chunk("m1", 0, "the harbor light was green")])
        .await
        .expect("chunks");
    let work = repo.chunks_needing_embedding("prov:a", 10, true).await.expect("work");
    repo.set_chunk_embedding(work[0].id, &[1.0, 0.0], "prov:a")
        .await
        .expect("embed");

    // Rewrite the text. The daemon adapter would re-enqueue; mirror that.
    repo.replace_chunks("m1", &[chunk("m1", 0, "TOTALLY DIFFERENT TEXT NOW")])
        .await
        .expect("rewrite");
    repo.enqueue_embedding("m1", 0, "prov:a").await.expect("enqueue");

    // The rebind that used to trigger the resurrection.
    let restored = repo.restore_chunk_vectors("prov:a").await.expect("restore");
    assert_eq!(restored, 0, "there is no valid stored vector to restore");

    let hits = repo
        .search_chunks_by_embedding_sql(&[1.0, 0.0], "prov:a", 10, None)
        .await
        .expect("knn");
    assert!(
        hits.is_empty(),
        "the old text's vector must not answer for the new text"
    );
    let work = repo.chunks_needing_embedding("prov:a", 10, true).await.expect("work");
    assert_eq!(work.len(), 1, "the re-embed must still be owed");
    assert_eq!(work[0].content, "TOTALLY DIFFERENT TEXT NOW");
}

/// The zeroing in `replace_chunks` must reach the disk, WAL included, for the
/// same Ghost-Vectors reason as a delete: the superseded vector is a copy of
/// the superseded text.
#[tokio::test]
async fn a_rewrite_destroys_the_old_chunk_vectors_on_disk() {
    let db = temp_db_path("rewrite_ghost");
    let storage = open(&db).await;
    let repo = storage.memories();
    repo.create(memory("m1")).await.expect("create");
    repo.replace_chunks("m1", &[chunk("m1", 0, "secret original text")])
        .await
        .expect("chunks");

    let secret = sentinel();
    let work = repo.chunks_needing_embedding("prov:a", 10, true).await.expect("work");
    repo.set_chunk_embedding(work[0].id, &secret, "prov:a").await.expect("embed");

    repo.replace_chunks("m1", &[chunk("m1", 0, "replacement text")])
        .await
        .expect("rewrite");

    let needle: Vec<u8> = secret.iter().flat_map(|f| f.to_le_bytes()).collect();
    assert!(
        !file_contains(&db, &needle),
        "the superseded vector inverts back to the superseded text"
    );
}

/// The pre-existing-work sweep is opt-in per call. Without the flag, only
/// explicitly enqueued work is returned — the sweep scans the whole chunk
/// table, and running it per drain batch made the backfill quadratic.
#[tokio::test]
async fn an_unseeded_drain_reads_only_the_queue() {
    let db = temp_db_path("unseeded");
    let storage = open(&db).await;
    let repo = storage.memories();
    repo.create(memory("m1")).await.expect("create");
    repo.replace_chunks("m1", &[chunk("m1", 0, "pre-existing, never enqueued")])
        .await
        .expect("chunks");
    // replace_chunks purges this parent's queue rows; nothing re-enqueued.
    repo.dequeue_embedding("m1", 0, "model").await.expect("ensure absent");

    let unseeded = repo.chunks_needing_embedding("model", 10, false).await.expect("work");
    assert!(unseeded.is_empty(), "no sweep, no derived work");

    let seeded = repo.chunks_needing_embedding("model", 10, true).await.expect("work");
    assert_eq!(seeded.len(), 1, "the sweep derives the pre-existing chunk");
}

/// Queue rows for a model that left the priority list can never drain; they
/// sit forever and pollute health. Pruning keeps only configured models — and
/// refuses an empty keep-list, which would wipe the queue over a missing key.
#[tokio::test]
async fn pruning_keeps_configured_models_and_refuses_to_wipe() {
    let db = temp_db_path("retain");
    let storage = open(&db).await;
    let repo = storage.memories();
    repo.create(memory("m1")).await.expect("create");
    repo.enqueue_embedding("m1", 0, "kept-model").await.expect("kept");
    repo.enqueue_embedding("m1", 0, "retired-model").await.expect("retired");

    let dropped = repo
        .retain_queue_models(&["kept-model".to_string()])
        .await
        .expect("retain");
    assert_eq!(dropped, 1);
    assert_eq!(repo.pending_embeddings("kept-model", 10).await.expect("kept").len(), 1);
    assert_eq!(repo.pending_embeddings("retired-model", 10).await.expect("gone"), [] as [(std::string::String, i64); 0]);

    let storage_for_panic = open(&db).await;
    let wipe = tokio::spawn(async move {
        storage_for_panic.memories().retain_queue_models(&[]).await
    });
    assert!(
        wipe.await.expect_err("must panic").is_panic(),
        "an empty keep-list must refuse, not silently clear the queue"
    );
}
