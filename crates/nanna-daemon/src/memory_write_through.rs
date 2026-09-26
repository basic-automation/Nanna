//! Every card and every thread post becomes a workspace memory (P25 Stage 1).
//!
//! P25 decision 14: "Every task is a memory. Every thread post is a memory."
//! The thread stays the permanent record and is never compacted; what dreaming
//! consolidates is these *copies*, each carrying a pointer back to the card and
//! the post it came from (`source_task_id`, `source_note_id`), so a recalled
//! memory can always be followed back to the words it was made from.
//!
//! **Fed by the store's own event sink, not by a bus subscriber.** A
//! `broadcast` receiver that falls behind is told it lagged and the skipped
//! events are gone — for a notification that is a missed toast, here it is a
//! memory that silently never exists. So the sink that already sees every task
//! write ([`crate::task_event_bridge::TaskEventBridge`]) also hands the few
//! kinds that matter to a bounded `mpsc` queue, which one worker drains in
//! order. A full queue is reported, never waited on: the sink is called from
//! the task store's write path, and that path must not block.
//!
//! **The same events feed the episodic timeline** (`memory_events`, via
//! `nanna_timeline`): every thread post as a `message` episode and every status
//! change or verdict as an `outcome`, each carrying the card (and post) ids as
//! its lineage. A card's thread is an event series — this is the raw series
//! the DSP compression step will fold (P25 "DSP timeline compression", step 1).
//! The timeline needs only storage, so it is fed even when memory is disabled.
//!
//! **Board cards only.** A `session`-scoped task is chat scaffolding — the
//! harness writes dozens per turn — and P25 decision 9 deletes that scope.
//! Copying those would flood memory with plan steps; they are skipped until
//! the session → workspace promotion makes every card a board card.

use nanna_memory::MemoryService;
use nanna_storage::{Storage, StorageError, Task, TaskEvent, TaskEventKind};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, warn};

/// Events the write-through queue holds before the sink starts reporting drops.
///
/// Bound justification: one scope holds at most
/// [`nanna_storage::TASKS_PER_SCOPE_MAX`] cards, and the largest single burst
/// the store emits is a cascade over one scope, so a whole scope's worth of
/// events fits even if the worker has not started draining yet (the queue is
/// created with the store, before the memory service exists). Each queued
/// event is a few hundred bytes, so the bound is a few MiB at worst.
pub const WRITE_THROUGH_QUEUE_MAX: usize = nanna_storage::TASKS_PER_SCOPE_MAX;

/// Importance given to a board copy, on the memory service's 1–5 scale.
///
/// The middle of the scale: a card is neither an offhand remark nor a stated
/// fact about the user, and dreaming — not this module — is what decides which
/// copies matter.
const COPY_IMPORTANCE: f32 = 3.0;

/// What a store event becomes in memory, decided without any I/O.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trigger {
    /// A card was created.
    Created,
    /// A card was closed: completed (a verdict) or cancelled.
    Closed { cancelled: bool },
    /// A post was added to a card's thread.
    Posted { note_id: i64 },
}

impl Trigger {
    /// The trigger for `event`, or `None` when it produces no memory.
    #[must_use]
    pub fn of(event: &TaskEvent) -> Option<Self> {
        if event.scope == "session" {
            return None;
        }
        match event.kind {
            TaskEventKind::Created => Some(Self::Created),
            TaskEventKind::Verdict => Some(Self::Closed { cancelled: false }),
            TaskEventKind::StatusChanged
                if event.detail.get("status").and_then(|s| s.as_str()) == Some("cancelled") =>
            {
                Some(Self::Closed { cancelled: true })
            }
            TaskEventKind::Posted => event
                .detail
                .get("note_id")
                .and_then(serde_json::Value::as_i64)
                .map(|note_id| Self::Posted { note_id }),
            _ => None,
        }
    }
}

/// Which timeline episode a store event becomes, if any.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpisodeTrigger {
    /// A thread post: a turn of the card's conversation.
    Posted { note_id: i64 },
    /// The card changed status, or a verdict closed it.
    Transition,
}

impl EpisodeTrigger {
    /// The episode trigger for `event`, or `None`. Board cards only, as for
    /// memory copies.
    #[must_use]
    pub fn of(event: &TaskEvent) -> Option<Self> {
        if event.scope == "session" {
            return None;
        }
        match event.kind {
            TaskEventKind::Posted => event
                .detail
                .get("note_id")
                .and_then(serde_json::Value::as_i64)
                .map(|note_id| Self::Posted { note_id }),
            TaskEventKind::StatusChanged | TaskEventKind::Verdict => Some(Self::Transition),
            _ => None,
        }
    }
}

/// Whether the write-through worker has anything to do for `event`: a memory
/// copy, a timeline episode, or both. The sink queues only these.
#[must_use]
pub fn is_board_record(event: &TaskEvent) -> bool {
    Trigger::of(event).is_some() || EpisodeTrigger::of(event).is_some()
}

/// Salience of a thread post by its kind.
///
/// Salience decides which episodes survive decimation, so it ranks by how much
/// a post changes the card's course: a verdict decides it and a question
/// blocks it; a comment informs; a progress line is the most compressible.
const fn post_salience(kind: nanna_storage::TaskNoteKind) -> f32 {
    match kind {
        nanna_storage::TaskNoteKind::Verdict => 0.9,
        nanna_storage::TaskNoteKind::Question => 0.8,
        nanna_storage::TaskNoteKind::Comment => 0.5,
        nanna_storage::TaskNoteKind::Progress => 0.3,
    }
}

/// The timeline episode for `trigger` on `task`. Pure.
#[must_use]
pub fn board_episode(
    task: &Task,
    event: &TaskEvent,
    trigger: EpisodeTrigger,
    post: Option<&nanna_storage::TaskNote>,
    ts_unix_ms: i64,
) -> nanna_timeline::Episode {
    let mut source_ids = vec![format!("task:{}", task.id)];
    let (kind, content, salience) =
        if let (EpisodeTrigger::Posted { note_id }, Some(note)) = (trigger, post) {
            source_ids.push(format!("task_note:{note_id}"));
            let by = note
                .author_member_id
                .as_deref()
                .or(note.author.as_deref())
                .unwrap_or("someone");
            (
                nanna_timeline::EventKind::Message,
                format!(
                    "#{} \"{}\" — {} by {by}: {}",
                    task.id,
                    task.title,
                    note.kind.as_str(),
                    note.content
                ),
                post_salience(note.kind),
            )
        } else {
            let verdict = event.kind == TaskEventKind::Verdict;
            let status = event
                .detail
                .get("status")
                .and_then(serde_json::Value::as_str)
                .unwrap_or(task.status.as_str());
            (
                nanna_timeline::EventKind::Outcome,
                if verdict {
                    format!("#{} \"{}\" — completed (verdict)", task.id, task.title)
                } else {
                    format!("#{} \"{}\" — status {status}", task.id, task.title)
                },
                if verdict { 0.9 } else { 0.4 },
            )
        };
    debug_assert!((0.0..=1.0).contains(&salience), "salience is normalized");
    nanna_timeline::Episode {
        kind,
        ts_unix_ms,
        workspace_id: (task.scope == "workspace")
            .then(|| task.scope_id.clone())
            .flatten(),
        content,
        salience,
        source_ids,
    }
}

/// The memory a trigger writes: its text and its pointers back to the board.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoardCopy {
    pub content: String,
    pub metadata: HashMap<String, String>,
    /// `Some(id)` for a workspace card; `None` for a global one.
    pub workspace_id: Option<String>,
}

/// Build the copy for `trigger` on `task`. Pure.
///
/// The card's title leads every copy so a recalled post still says what it
/// was about — a bare "done, tests pass" is useless without its card.
#[must_use]
pub fn board_copy(
    task: &Task,
    trigger: Trigger,
    post: Option<&nanna_storage::TaskNote>,
) -> BoardCopy {
    let mut metadata = HashMap::from([
        ("source".to_string(), "task_board".to_string()),
        ("source_task_id".to_string(), task.id.to_string()),
    ]);
    if !task.labels.is_empty() {
        metadata.insert("labels".to_string(), task.labels.join(","));
    }
    let header = format!("Card #{} \"{}\"", task.id, task.title);
    let content = match trigger {
        Trigger::Created => {
            metadata.insert("board_event".to_string(), "created".to_string());
            match task.description.as_deref().map(str::trim) {
                Some(description) if !description.is_empty() => {
                    format!("{header} was created.\n{description}")
                }
                _ => format!("{header} was created."),
            }
        }
        Trigger::Closed { cancelled } => {
            let outcome = if cancelled { "cancelled" } else { "completed" };
            metadata.insert("board_event".to_string(), outcome.to_string());
            format!("{header} was {outcome}.")
        }
        Trigger::Posted { note_id } => {
            metadata.insert("board_event".to_string(), "posted".to_string());
            metadata.insert("source_note_id".to_string(), note_id.to_string());
            let (kind, by, text) = post.map_or(("post", None, ""), |note| {
                (
                    note.kind.as_str(),
                    note.author_member_id.as_deref().or(note.author.as_deref()),
                    note.content.as_str(),
                )
            });
            metadata.insert("post_kind".to_string(), kind.to_string());
            // Only a real member id is recorded as one; a pre-members actor
            // string (`gui`, `harness`) still names the author in the text.
            if let Some(member) = post.and_then(|note| note.author_member_id.as_deref()) {
                metadata.insert("author_member_id".to_string(), member.to_string());
            }
            let by = by.map(|by| format!(" by {by}")).unwrap_or_default();
            format!("{header} — {kind}{by}:\n{text}")
        }
    };
    let workspace_id = if task.scope == "workspace" {
        task.scope_id.clone()
    } else {
        None
    };
    debug_assert!(!content.is_empty(), "a copy always has its header");
    debug_assert!(
        metadata.contains_key("source_task_id"),
        "a copy points at its card"
    );
    BoardCopy {
        content,
        metadata,
        workspace_id,
    }
}

/// What one event produced on the board's record.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BoardRecord {
    /// The memory copy's id, when one was written.
    pub memory_id: Option<String>,
    /// The timeline episode's id, when one was appended.
    pub episode_id: Option<String>,
}

/// Record one event: its memory copy (when `memory` is present) and its
/// timeline episode. Nothing is recorded for a card or post deleted before the
/// worker reached it.
///
/// # Errors
/// Returns a message when the store cannot be read or a write fails. The two
/// writes are independent: a failed memory copy still attempts the episode's
/// write only if it came first — each failure is reported, neither is retried.
pub async fn write_one(
    storage: &Storage,
    memory: Option<&MemoryService>,
    event: &TaskEvent,
) -> Result<BoardRecord, String> {
    let copy = Trigger::of(event).filter(|_| memory.is_some());
    let episode = EpisodeTrigger::of(event);
    if copy.is_none() && episode.is_none() {
        return Ok(BoardRecord::default());
    }
    let task = match storage.tasks().get(event.task_id).await {
        Ok(task) => task,
        Err(StorageError::NotFound(_)) => return Ok(BoardRecord::default()),
        Err(e) => return Err(format!("reading card #{}: {e}", event.task_id)),
    };
    let note_id = match (copy, episode) {
        (Some(Trigger::Posted { note_id }), _) | (_, Some(EpisodeTrigger::Posted { note_id })) => {
            Some(note_id)
        }
        _ => None,
    };
    let post = match note_id {
        Some(note_id) => match storage.tasks().note(note_id).await {
            Ok(note) => Some(note),
            Err(StorageError::NotFound(_)) => return Ok(BoardRecord::default()),
            Err(e) => return Err(format!("reading post #{note_id}: {e}")),
        },
        None => None,
    };

    let mut record = BoardRecord::default();
    if let (Some(trigger), Some(memory)) = (copy, memory) {
        let copy = board_copy(&task, trigger, post.as_ref());
        let (id, _) = memory
            .remember_deferred_vector(
                &copy.content,
                copy.metadata,
                COPY_IMPORTANCE,
                copy.workspace_id,
            )
            .await
            .map_err(|e| format!("writing the memory copy of card #{}: {e}", task.id))?;
        record.memory_id = Some(id);
    }
    if let Some(trigger) = episode {
        let now_ms = chrono::Utc::now().timestamp_millis();
        let episode = board_episode(&task, event, trigger, post.as_ref(), now_ms);
        let id = nanna_timeline::Timeline::new(storage)
            .append(&episode)
            .await
            .map_err(|e| format!("appending card #{}'s timeline episode: {e}", task.id))?;
        record.episode_id = Some(id);
    }
    Ok(record)
}

/// Drain the write-through queue until every sender is gone.
///
/// Sequential on purpose: one card's created → posted → completed records
/// land in the order they happened, and each write is a single durable insert.
/// `memory` is `None` when memory is disabled: the timeline is still fed.
pub async fn run(
    mut events: mpsc::Receiver<TaskEvent>,
    storage: Arc<Storage>,
    memory: Option<Arc<MemoryService>>,
) {
    while let Some(event) = events.recv().await {
        match write_one(&storage, memory.as_deref(), &event).await {
            Ok(record) => debug!("board record for card #{}: {record:?}", event.task_id),
            Err(e) => warn!("board write-through failed: {e}"),
        }
    }
}

/// Closed cards folded per dream cycle.
///
/// Bound justification: each fold is one bounded event read and one durable
/// memory insert, so 32 keeps a cycle's fold phase to well under a second
/// while a board closing a few dozen cards a day never falls behind.
pub const FOLDS_PER_CYCLE_MAX: usize = 32;

/// Most recently closed cards a cycle looks at for an unfolded one.
///
/// Bound justification: eight cycles' worth of folds — enough to catch up
/// after a busy day without ever scanning the whole closed history.
pub const FOLD_SCAN_MAX: usize = FOLDS_PER_CYCLE_MAX * 8;

/// A card with fewer events than this has nothing to compress: its episodes
/// already read as a summary.
pub const FOLD_MIN_EVENTS: usize = 3;

/// The dream fold (P25 "DSP timeline compression", step 3).
///
/// Each recently closed board card's episode series becomes ONE memory,
/// carrying the lineage back to the card. The thread itself is never touched
/// (decision 14); only its episodes are read.
///
/// Idempotent: a card whose fold already covers every one of its episodes is
/// skipped, so re-running a cycle writes nothing twice. A card reopened and
/// closed again has episodes its fold never saw; it is folded again and the
/// old fold removed — after the new one is written, so the card is never left
/// with none. Returns how many cards were folded.
///
/// # Errors
/// Returns a message when the store cannot be read, a memory write fails, or
/// a superseded fold cannot be removed; folds written before the failure stay
/// written.
pub async fn fold_closed_cards(storage: &Storage, memory: &MemoryService) -> Result<usize, String> {
    // Card id → (fold memory id, episodes it covered), for every existing fold.
    let already: HashMap<String, (String, usize)> = memory
        .list_all()
        .await
        .into_iter()
        .filter(|m| m.metadata.get("board_event").map(String::as_str) == Some("episode"))
        .filter_map(|m| {
            let card = m.metadata.get("source_task_id")?.clone();
            let covered = m
                .metadata
                .get("episode_events")
                .and_then(|n| n.parse().ok())
                .unwrap_or(0);
            Some((card, (m.id, covered)))
        })
        .collect();
    let cards = storage
        .tasks()
        .closed_board_cards(FOLD_SCAN_MAX)
        .await
        .map_err(|e| format!("listing closed cards: {e}"))?;
    let mut folded = 0usize;
    for card in cards {
        if folded == FOLDS_PER_CYCLE_MAX {
            break;
        }
        let previous = already.get(&card.id.to_string());
        let events = storage
            .memory_events()
            .for_source(&format!("task:{}", card.id), nanna_storage::MAX_EVENT_PAGE)
            .await
            .map_err(|e| format!("reading card #{}'s episodes: {e}", card.id))?;
        if events.len() < FOLD_MIN_EVENTS
            || previous.is_some_and(|(_, covered)| events.len() <= *covered)
        {
            continue;
        }
        let Some(fold) =
            nanna_timeline::compress_episode(&events, nanna_timeline::DREAM_FOLD_BUDGET)
        else {
            continue;
        };
        let metadata = HashMap::from([
            ("source".to_string(), "task_board".to_string()),
            ("source_task_id".to_string(), card.id.to_string()),
            ("board_event".to_string(), "episode".to_string()),
            ("episode_events".to_string(), events.len().to_string()),
            ("episode_kept".to_string(), fold.kept.to_string()),
        ]);
        let content = format!(
            "The story of card #{} \"{}\" ({} events, {} shown):\n{}",
            card.id,
            card.title,
            events.len(),
            fold.kept,
            fold.episode.content
        );
        memory
            .remember_deferred_vector(
                &content,
                metadata,
                FOLD_IMPORTANCE,
                fold.episode.workspace_id,
            )
            .await
            .map_err(|e| format!("writing card #{}'s folded episode: {e}", card.id))?;
        if let Some((superseded, _)) = previous {
            memory
                .forget(superseded)
                .await
                .map_err(|e| format!("removing card #{}'s superseded fold: {e}", card.id))?;
        }
        folded += 1;
    }
    debug_assert!(folded <= FOLDS_PER_CYCLE_MAX, "the fold phase is bounded");
    Ok(folded)
}

/// Importance of a folded card, above a single copy's: it is the account of a
/// whole card, the memory recall should prefer over any one of its posts.
const FOLD_IMPORTANCE: f32 = 4.0;

#[cfg(test)]
mod tests {
    use super::*;
    use nanna_storage::{NewTask, TaskEventSink, TaskNoteKind};

    fn event(kind: TaskEventKind, scope: &str, detail: serde_json::Value) -> TaskEvent {
        TaskEvent {
            kind,
            task_id: 1,
            scope: scope.to_string(),
            scope_id: Some("ws1".to_string()),
            actor: None,
            detail,
        }
    }

    #[test]
    fn only_board_creations_closures_and_posts_become_memories() {
        let none = serde_json::json!({});
        assert_eq!(
            Trigger::of(&event(TaskEventKind::Created, "workspace", none.clone())),
            Some(Trigger::Created)
        );
        assert_eq!(
            Trigger::of(&event(TaskEventKind::Verdict, "global", none.clone())),
            Some(Trigger::Closed { cancelled: false })
        );
        let cancelled = serde_json::json!({ "status": "cancelled" });
        assert_eq!(
            Trigger::of(&event(TaskEventKind::StatusChanged, "workspace", cancelled)),
            Some(Trigger::Closed { cancelled: true })
        );
        let started = serde_json::json!({ "status": "in_progress" });
        assert_eq!(
            Trigger::of(&event(TaskEventKind::StatusChanged, "workspace", started)),
            None
        );
        let posted = serde_json::json!({ "note_id": 9 });
        assert_eq!(
            Trigger::of(&event(TaskEventKind::Posted, "workspace", posted.clone())),
            Some(Trigger::Posted { note_id: 9 })
        );
        assert_eq!(
            Trigger::of(&event(TaskEventKind::Posted, "session", posted)),
            None,
            "chat scaffolding is not a board card"
        );
        assert_eq!(
            Trigger::of(&event(TaskEventKind::Assigned, "workspace", none)),
            None
        );
    }

    async fn wait_for_count(memory: &MemoryService, want: usize) -> usize {
        for _ in 0..200 {
            let have = memory.count().await;
            if have >= want {
                return have;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        memory.count().await
    }

    /// A store whose task events feed a running write-through worker.
    async fn board_with_write_through() -> (Arc<Storage>, Arc<MemoryService>) {
        let memory = Arc::new(MemoryService::new(
            nanna_memory::MemoryServiceConfig::default(),
        ));
        let storage = board_feeding(Some(Arc::clone(&memory))).await;
        (storage, memory)
    }

    /// A store whose task events feed a running worker; `memory` optional.
    async fn board_feeding(memory: Option<Arc<MemoryService>>) -> Arc<Storage> {
        let storage = Arc::new(Storage::in_memory().await.expect("storage"));
        let (bus, _keep) = tokio::sync::broadcast::channel(64);
        let (queue_tx, queue_rx) = mpsc::channel(WRITE_THROUGH_QUEUE_MAX);
        let bridge =
            crate::task_event_bridge::TaskEventBridge::new(bus).with_memory_write_through(queue_tx);
        assert!(storage.set_task_events(Arc::new(bridge) as Arc<dyn TaskEventSink>));
        tokio::spawn(run(queue_rx, Arc::clone(&storage), memory));
        storage
    }

    async fn wait_for_episodes(
        storage: &Storage,
        want: usize,
    ) -> Vec<nanna_storage::MemoryEventRow> {
        for _ in 0..200 {
            let rows = storage.memory_events().recent(50).await.expect("events");
            if rows.len() >= want {
                return rows;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        storage.memory_events().recent(50).await.expect("events")
    }

    /// Step 3: a closed card's series folds into ONE memory that points back
    /// at the card; a second cycle writes nothing twice; a card with too few
    /// events is left as it is; and `task:1`'s fold never reads `task:10`'s.
    #[tokio::test]
    async fn a_closed_cards_series_folds_once_into_one_memory() {
        let memory = Arc::new(MemoryService::new(
            nanna_memory::MemoryServiceConfig::default(),
        ));
        let storage = board_feeding(Some(Arc::clone(&memory))).await;
        let tasks = storage.tasks();
        let card = |title: &str| NewTask {
            scope: "workspace".to_string(),
            scope_id: Some("ws1".to_string()),
            title: title.to_string(),
            priority: 3,
            ..NewTask::default()
        };
        let busy = tasks.create(card("busy")).await.expect("card");
        for n in 0..4 {
            tasks
                .post(
                    busy.id,
                    Some("gui"),
                    None,
                    TaskNoteKind::Progress,
                    &format!("step {n}"),
                )
                .await
                .expect("post");
        }
        tasks
            .complete(busy.id, Some("gui"), None)
            .await
            .expect("complete");
        let quiet = tasks.create(card("quiet")).await.expect("card");
        tasks
            .complete(quiet.id, Some("gui"), None)
            .await
            .expect("complete");
        // busy: 4 posts + its verdict; quiet: its verdict.
        assert_eq!(wait_for_episodes(&storage, 6).await.len(), 6);

        let folded = fold_closed_cards(&storage, &memory).await.expect("fold");
        assert_eq!(folded, 1, "only the card with a series worth folding");
        let folds: Vec<_> = memory
            .list_all()
            .await
            .into_iter()
            .filter(|m| m.metadata.get("board_event").map(String::as_str) == Some("episode"))
            .collect();
        assert_eq!(folds.len(), 1);
        assert_eq!(
            folds[0].metadata.get("source_task_id"),
            Some(&busy.id.to_string())
        );
        assert!(folds[0].content.contains("step 3"), "{}", folds[0].content);
        assert!(!folds[0].content.contains("quiet"), "{}", folds[0].content);
        assert_eq!(folds[0].workspace_id.as_deref(), Some("ws1"));

        assert_eq!(
            fold_closed_cards(&storage, &memory).await.expect("fold"),
            0,
            "idempotent"
        );

        // Reopened, worked on, closed again: the fold is replaced, not kept
        // stale and not duplicated.
        tasks.reopen(busy.id, Some("gui")).await.expect("reopen");
        tasks
            .post(busy.id, Some("gui"), None, TaskNoteKind::Progress, "step 4")
            .await
            .expect("post");
        tasks
            .complete(busy.id, Some("gui"), None)
            .await
            .expect("complete");
        let before = wait_for_episodes(&storage, 8).await.len();
        assert!(before >= 8, "the second round's episodes arrived: {before}");
        assert_eq!(fold_closed_cards(&storage, &memory).await.expect("fold"), 1);
        let refolded: Vec<_> = memory
            .list_all()
            .await
            .into_iter()
            .filter(|m| m.metadata.get("board_event").map(String::as_str) == Some("episode"))
            .collect();
        assert_eq!(refolded.len(), 1, "the old fold is gone");
        assert!(
            refolded[0].content.contains("step 4"),
            "{}",
            refolded[0].content
        );
        assert_ne!(refolded[0].id, folds[0].id);
        assert_eq!(
            fold_closed_cards(&storage, &memory).await.expect("fold"),
            0,
            "idempotent again"
        );
    }

    /// The timeline half: a card's thread is an event series. Each post is a
    /// `message` episode and each transition an `outcome`, carrying the card
    /// (and post) as lineage — and it is fed with memory switched off too,
    /// since it needs only storage.
    #[tokio::test]
    async fn a_cards_thread_and_transitions_feed_the_timeline_without_memory() {
        let storage = board_feeding(None).await;
        let tasks = storage.tasks();
        let card = tasks
            .create(NewTask {
                scope: "workspace".to_string(),
                scope_id: Some("ws1".to_string()),
                title: "Ship it".to_string(),
                priority: 2,
                ..NewTask::default()
            })
            .await
            .expect("card");
        let note = tasks
            .post(
                card.id,
                Some("gui"),
                None,
                TaskNoteKind::Question,
                "Which tag?",
            )
            .await
            .expect("post");
        tasks
            .complete(card.id, Some("gui"), None)
            .await
            .expect("complete");

        let rows = wait_for_episodes(&storage, 2).await;
        let message = rows
            .iter()
            .find(|r| r.kind == "message")
            .expect("the post's episode");
        assert!(
            message.content.contains("Which tag?"),
            "{}",
            message.content
        );
        assert_eq!(
            message.source_ids,
            [
                format!("task:{}", card.id),
                format!("task_note:{}", note.id)
            ]
        );
        assert!(
            (message.salience - 0.8).abs() < f32::EPSILON,
            "a question blocks"
        );
        assert_eq!(message.workspace_id.as_deref(), Some("ws1"));
        let outcome = rows
            .iter()
            .find(|r| r.kind == "outcome")
            .expect("the verdict's episode");
        assert!(outcome.content.contains("completed"), "{}", outcome.content);
        assert_eq!(outcome.source_ids, [format!("task:{}", card.id)]);
    }

    /// The whole path: a card's life on the board lands in memory as three
    /// copies, in order, each pointing back at the card — and the post's copy
    /// at the post. A chat-scaffolding card on the same store leaves none.
    #[tokio::test]
    async fn a_cards_life_becomes_memories_that_point_back_at_it() {
        let (storage, memory) = board_with_write_through().await;
        let tasks = storage.tasks();
        let scaffolding = tasks
            .create(NewTask {
                scope: "session".to_string(),
                scope_id: Some("chat-1".to_string()),
                title: "plan step".to_string(),
                priority: 3,
                ..NewTask::default()
            })
            .await
            .expect("session card");
        let card = tasks
            .create(NewTask {
                scope: "workspace".to_string(),
                scope_id: Some("ws1".to_string()),
                title: "Ship the release".to_string(),
                description: Some("Tag, build, publish.".to_string()),
                priority: 2,
                labels: vec!["release".to_string()],
                ..NewTask::default()
            })
            .await
            .expect("board card");
        let note = tasks
            .post(
                card.id,
                Some("gui"),
                Some(nanna_storage::HUMAN_MEMBER_ID),
                TaskNoteKind::Question,
                "Which tag?",
            )
            .await
            .expect("post");
        tasks
            .complete(card.id, Some("gui"), None)
            .await
            .expect("complete");
        tasks
            .complete(scaffolding.id, Some("harness"), None)
            .await
            .expect("complete");

        assert_eq!(
            wait_for_count(&memory, 3).await,
            3,
            "three copies, none for the scaffolding"
        );
        let copies = memory.list_all().await;
        assert!(
            copies
                .iter()
                .all(|m| m.workspace_id.as_deref() == Some("ws1")),
            "{copies:?}"
        );
        assert!(
            copies
                .iter()
                .all(|m| m.metadata.get("source_task_id") == Some(&card.id.to_string())),
            "every copy points at its card: {copies:?}"
        );
        let post_copy = copies
            .iter()
            .find(|m| m.metadata.get("board_event").map(String::as_str) == Some("posted"))
            .expect("the post's copy");
        assert_eq!(
            post_copy.metadata.get("source_note_id"),
            Some(&note.id.to_string())
        );
        assert_eq!(
            post_copy
                .metadata
                .get("author_member_id")
                .map(String::as_str),
            Some(nanna_storage::HUMAN_MEMBER_ID)
        );
        assert!(
            post_copy.content.contains("Ship the release"),
            "{}",
            post_copy.content
        );
        assert!(
            post_copy.content.contains("Which tag?"),
            "{}",
            post_copy.content
        );
        let created = copies
            .iter()
            .find(|m| m.metadata.get("board_event").map(String::as_str) == Some("created"))
            .expect("the creation's copy");
        assert!(
            created.content.contains("Tag, build, publish."),
            "{}",
            created.content
        );
        assert_eq!(
            created.metadata.get("labels").map(String::as_str),
            Some("release")
        );
    }
}
