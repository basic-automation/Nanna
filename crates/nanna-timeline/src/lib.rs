#![warn(clippy::all)]
#![warn(clippy::pedantic, clippy::nursery)]

//! The episodic timeline: what happened, on a wall-clock axis.
//!
//! Nanna's memory has two layers and they answer different questions.
//! [`nanna_storage::Memory`] is the **semantic** layer — facts, FSRS-weighted,
//! never expiring, with no single timestamp because a fact is the residue of
//! many episodes. This crate is the **episodic** layer underneath it: the raw
//! stream of messages, tool calls, recalls and outcomes, each stamped with the
//! moment it happened. Dreaming consolidates episodes *into* facts; it never
//! rewrites them.
//!
//! Three properties hold by construction, and each one exists because losing
//! it would make a later phase impossible:
//!
//! 1. **Append-only.** There is no update or delete for a single event. A log
//!    that can be edited cannot answer "what did I know, and when".
//! 2. **Bounded.** Every append caps its content and its lineage; every query
//!    caps its page. The timeline grows without bound in *rows* — that is its
//!    job — so no single operation may grow without bound in *bytes*.
//! 3. **Truncation is visible.** A capped episode records its pre-truncation
//!    length, so a consumer can tell a short event from a shortened one
//!    instead of silently treating the remnant as the whole.

use nanna_storage::{MemoryEventRow, NewMemoryEvent, Storage, StorageError};

pub mod compress;
pub use compress::{CompressedEpisode, DREAM_FOLD_BUDGET, compress_episode};

/// Largest content an episode stores, in characters.
///
/// An episode is a record that something happened, not the artifact it
/// happened to: a 4 MB tool result belongs in the tool's own output routing,
/// and putting it here would make one event cost more than a day of them.
/// 8192 characters holds a full chat turn or a tool call with its arguments,
/// which is what the resampler and the consolidator actually read.
pub const MAX_EVENT_CONTENT_CHARS: usize = 8192;

/// Largest lineage an episode records.
///
/// `source_ids` is provenance — the handful of ids this event derives from.
/// A consolidation that draws on more than 64 sources has a cluster for a
/// parent, not a list, and should record the cluster.
pub const MAX_EVENT_SOURCE_IDS: usize = 64;

/// Largest page any timeline query returns. Re-exported from the storage
/// layer so callers bound themselves against one number, not two.
pub use nanna_storage::MAX_EVENT_PAGE;

/// What kind of thing happened.
///
/// Closed on purpose. Each variant is a distinct signal the resampler treats
/// separately — salience(t) over tool calls is not the same series as
/// salience(t) over messages — so an open string would let a typo create a
/// silent third series that nothing ever reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    /// A turn of conversation, from any channel.
    Message,
    /// A tool invocation with its arguments.
    ToolCall,
    /// A memory was retrieved and put in front of the model.
    Recall,
    /// A result: a tool returned, a task finished, an acceptance check ruled.
    Outcome,
}

impl EventKind {
    /// The stored discriminant. Stable — it is written to the database.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Message => "message",
            Self::ToolCall => "tool_call",
            Self::Recall => "recall",
            Self::Outcome => "outcome",
        }
    }

    /// Parse a stored discriminant back.
    #[must_use]
    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s {
            "message" => Some(Self::Message),
            "tool_call" => Some(Self::ToolCall),
            "recall" => Some(Self::Recall),
            "outcome" => Some(Self::Outcome),
            _ => None,
        }
    }
}

/// Everything that can go wrong appending to or reading the timeline.
#[derive(Debug, thiserror::Error)]
pub enum TimelineError {
    #[error("storage error: {0}")]
    Storage(#[from] StorageError),
    /// The caller asked for a window that runs backwards.
    #[error("time window ends before it starts: {start_ms} > {end_ms}")]
    InvertedWindow { start_ms: i64, end_ms: i64 },
    /// The caller asked for more rows than a single page may hold.
    #[error("page of {asked} events exceeds the cap of {cap}")]
    PageTooLarge { asked: usize, cap: usize },
    /// Salience is a normalized weight; anything else is a bug upstream, not
    /// a value to clamp silently.
    #[error("salience must be a finite value in [0,1], got {got}")]
    SalienceOutOfRange { got: f32 },
}

/// One episode to append.
#[derive(Debug, Clone, PartialEq)]
pub struct Episode {
    pub kind: EventKind,
    /// Unix milliseconds. Supplied by the caller rather than taken from the
    /// clock here, so replaying a channel backlog records when each thing
    /// *happened*, not when it was ingested.
    pub ts_unix_ms: i64,
    /// Workspace scope; `None` is global.
    pub workspace_id: Option<String>,
    pub content: String,
    /// Normalized importance in `[0,1]`.
    pub salience: f32,
    /// Ids this episode derives from.
    pub source_ids: Vec<String>,
}

/// The result of capping an episode's content.
struct CappedContent {
    text: String,
    original_chars: usize,
}

/// Truncate to at most `MAX_EVENT_CONTENT_CHARS` *characters*.
///
/// Character-wise, never byte-wise: slicing a `String` at a byte offset that
/// lands inside a multi-byte character panics, and episode content is
/// arbitrary user and tool text where an em dash at the wrong offset is not a
/// rare case. The original length is carried out so the caller can record it.
fn cap_content(content: &str) -> CappedContent {
    let original_chars = content.chars().count();
    if original_chars <= MAX_EVENT_CONTENT_CHARS {
        return CappedContent {
            text: content.to_string(),
            original_chars,
        };
    }
    let text: String = content.chars().take(MAX_EVENT_CONTENT_CHARS).collect();
    debug_assert!(
        text.chars().count() == MAX_EVENT_CONTENT_CHARS,
        "the cap must be exact in characters"
    );
    CappedContent {
        text,
        original_chars,
    }
}

/// Append-only view over the episodic stream.
pub struct Timeline<'a> {
    storage: &'a Storage,
}

impl<'a> Timeline<'a> {
    #[must_use]
    pub const fn new(storage: &'a Storage) -> Self {
        Self { storage }
    }

    /// Append one episode and return its generated event id.
    ///
    /// Content over [`MAX_EVENT_CONTENT_CHARS`] is truncated and the *original*
    /// character count is stored, so a consumer can tell it was shortened.
    /// Lineage over [`MAX_EVENT_SOURCE_IDS`] is truncated the same way.
    ///
    /// # Errors
    /// [`TimelineError::SalienceOutOfRange`] if salience is not a finite value
    /// in `[0,1]`; [`TimelineError::Storage`] if the insert fails.
    pub async fn append(&self, episode: &Episode) -> Result<String, TimelineError> {
        if !episode.salience.is_finite() || !(0.0..=1.0).contains(&episode.salience) {
            return Err(TimelineError::SalienceOutOfRange {
                got: episode.salience,
            });
        }

        let capped = cap_content(&episode.content);
        let mut source_ids = episode.source_ids.clone();
        source_ids.truncate(MAX_EVENT_SOURCE_IDS);

        debug_assert!(
            capped.text.chars().count() <= MAX_EVENT_CONTENT_CHARS,
            "content cap must hold"
        );
        debug_assert!(
            source_ids.len() <= MAX_EVENT_SOURCE_IDS,
            "lineage cap must hold"
        );

        let event_id = uuid::Uuid::new_v4().to_string();
        let row = NewMemoryEvent {
            event_id: event_id.clone(),
            ts_unix_ms: episode.ts_unix_ms,
            kind: episode.kind.as_str().to_string(),
            workspace_id: episode.workspace_id.clone(),
            content: capped.text,
            content_len_chars: i64::try_from(capped.original_chars).unwrap_or(i64::MAX),
            embedding: None,
            embedding_model: None,
            salience: episode.salience,
            source_ids,
        };
        self.storage.memory_events().append(&row).await?;
        Ok(event_id)
    }

    /// Episodes in `[start_ms, end_ms)`, oldest first.
    ///
    /// The window is half-open so adjacent windows tile the axis exactly once
    /// — a resampler stepping bucket to bucket cannot double-count an event
    /// that lands on a boundary.
    ///
    /// # Errors
    /// [`TimelineError::InvertedWindow`] if the window runs backwards;
    /// [`TimelineError::PageTooLarge`] above [`MAX_EVENT_PAGE`];
    /// [`TimelineError::Storage`] if the query fails.
    pub async fn range(
        &self,
        start_ms: i64,
        end_ms: i64,
        workspace_id: Option<&str>,
        limit: usize,
    ) -> Result<Vec<MemoryEventRow>, TimelineError> {
        Self::check_window(start_ms, end_ms)?;
        Self::check_page(limit)?;
        Ok(self
            .storage
            .memory_events()
            .range(start_ms, end_ms, workspace_id, limit)
            .await?)
    }

    /// How many episodes fall in `[start_ms, end_ms)`.
    ///
    /// Exists so sizing a window never requires materializing it — the one
    /// read the page cap is there to forbid.
    ///
    /// # Errors
    /// [`TimelineError::InvertedWindow`] if the window runs backwards;
    /// [`TimelineError::Storage`] if the query fails.
    pub async fn count_in_range(&self, start_ms: i64, end_ms: i64) -> Result<i64, TimelineError> {
        Self::check_window(start_ms, end_ms)?;
        Ok(self
            .storage
            .memory_events()
            .count_in_range(start_ms, end_ms)
            .await?)
    }

    /// The most recent `limit` episodes, newest first.
    ///
    /// # Errors
    /// [`TimelineError::PageTooLarge`] above [`MAX_EVENT_PAGE`];
    /// [`TimelineError::Storage`] if the query fails.
    pub async fn recent(&self, limit: usize) -> Result<Vec<MemoryEventRow>, TimelineError> {
        Self::check_page(limit)?;
        Ok(self.storage.memory_events().recent(limit).await?)
    }

    const fn check_window(start_ms: i64, end_ms: i64) -> Result<(), TimelineError> {
        if start_ms > end_ms {
            return Err(TimelineError::InvertedWindow { start_ms, end_ms });
        }
        Ok(())
    }

    const fn check_page(limit: usize) -> Result<(), TimelineError> {
        if limit > MAX_EVENT_PAGE {
            return Err(TimelineError::PageTooLarge {
                asked: limit,
                cap: MAX_EVENT_PAGE,
            });
        }
        Ok(())
    }
}

/// True when `row` was shortened at append time.
///
/// A caller that renders episode content needs this to say so, rather than
/// presenting a remnant as the whole thing.
#[must_use]
pub fn was_truncated(row: &MemoryEventRow) -> bool {
    let stored = i64::try_from(row.content.chars().count()).unwrap_or(i64::MAX);
    row.content_len_chars > stored
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_round_trips_through_its_stored_form() {
        for kind in [
            EventKind::Message,
            EventKind::ToolCall,
            EventKind::Recall,
            EventKind::Outcome,
        ] {
            assert_eq!(EventKind::from_str_opt(kind.as_str()), Some(kind));
        }
        assert_eq!(EventKind::from_str_opt("nonsense"), None);
    }

    #[test]
    fn content_under_the_cap_is_untouched() {
        let capped = cap_content("hello");
        assert_eq!(capped.text, "hello");
        assert_eq!(capped.original_chars, 5);
    }

    #[test]
    fn content_over_the_cap_is_cut_to_exactly_the_cap() {
        let long = "a".repeat(MAX_EVENT_CONTENT_CHARS + 500);
        let capped = cap_content(&long);
        assert_eq!(capped.text.chars().count(), MAX_EVENT_CONTENT_CHARS);
        assert_eq!(capped.original_chars, MAX_EVENT_CONTENT_CHARS + 500);
    }

    /// The regression this cap exists to avoid: byte-slicing arbitrary text
    /// panics when the offset lands inside a multi-byte character. Every
    /// character here is 3 bytes, so a byte-wise cut at the boundary would
    /// land mid-character.
    #[test]
    fn multibyte_content_is_cut_on_a_character_boundary_not_a_byte_one() {
        let long = "—".repeat(MAX_EVENT_CONTENT_CHARS + 10);
        assert!(long.len() > MAX_EVENT_CONTENT_CHARS, "chars must be wide");
        let capped = cap_content(&long);
        assert_eq!(capped.text.chars().count(), MAX_EVENT_CONTENT_CHARS);
        assert_eq!(capped.original_chars, MAX_EVENT_CONTENT_CHARS + 10);
        // The real assertion: it did not panic, and what came back is valid
        // UTF-8 whose every character is intact.
        assert!(capped.text.chars().all(|c| c == '—'));
    }

    #[test]
    fn truncation_is_detectable_from_the_stored_row() {
        let intact = MemoryEventRow {
            id: 1,
            event_id: "a".into(),
            ts_unix_ms: 0,
            kind: "message".into(),
            workspace_id: None,
            content: "short".into(),
            content_len_chars: 5,
            embedding: None,
            embedding_model: None,
            salience: 0.5,
            source_ids: Vec::new(),
            created_at: String::new(),
        };
        assert!(!was_truncated(&intact));

        let shortened = MemoryEventRow {
            content_len_chars: 9000,
            ..intact
        };
        assert!(was_truncated(&shortened));
    }
}
