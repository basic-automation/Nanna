//! Task lifecycle events (P25 Stage 1).
//!
//! These are **store mutations**, not run progress. The `TaskRun*` family on
//! the daemon's event bus describes a harness run; these describe what
//! happened to a card, and the board's router is the consumer they exist for.
//!
//! # Why this lives in `nanna-storage`
//!
//! [`TaskRepository`](crate::TaskRepository) is the only place that sees every
//! writer. Above it there are five independent ingress families — the `tasks.*`
//! tool services, the control plane's IPC handlers, the harness's `TaskSource`,
//! the seeding and recurrence sweeps, and the GUI — and two of them have no
//! event bus at all. Worse, some mutations have no caller to instrument:
//! `cascade_cancel` closes a whole subtree and ancestor auto-completion closes
//! parents, both touching rows nobody named. Emitting from the layer above the
//! repository would therefore be emitting from *one* of five doors while
//! claiming to describe the store.
//!
//! `nanna-storage` does not know the daemon's wire protocol, so it emits its
//! own shape through [`TaskEventSink`] and the daemon adapts it to
//! `protocol::Event`.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// What happened to a task in the store.
///
/// All nine of P25's kinds exist, and every one has a real emit point —
/// five from direct writes, two derived from dependency transitions, and two
/// from the time sweep — plus `Updated` and `Deleted`, so a client showing a
/// card hears every change to it, not only the ones the router acts on. Nothing here is declared ahead of the machinery that
/// sends it: a kind a consumer can match on but never receive is a dead field
/// wearing a feature's clothes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskEventKind {
    /// A card was created.
    Created,
    /// The assignee changed to a different member (never a no-op re-assign).
    Assigned,
    /// The status moved between two different values.
    StatusChanged,
    /// The card's derived `blocked` flag became true — a dependency it names
    /// is open again, or it gained one that already was.
    Blocked,
    /// The card's derived `blocked` flag became false — every dependency it
    /// names is now closed, gone, or no longer named.
    Unblocked,
    /// A thread post was appended.
    Posted,
    /// The card's defer date arrived, so it enters the inbox (P25 decision 11).
    /// Announced once per crossing, not once per sweep.
    Due,
    /// The card's deadline passed while it was still open. Measured against
    /// `deadline_at` alone — `due_at` defers, it does not bind.
    Overdue,
    /// A card was completed — the recorded verdict.
    Verdict,
    /// Fields other than status and assignee changed (title, labels, dates,
    /// priority, description, …); `detail.fields` names them. Before it, an
    /// edit reached no other open view until some unrelated event arrived.
    Updated,
    /// The card (and its subtree) was deleted.
    Deleted,
}

impl TaskEventKind {
    /// The wire name, matching `#[serde(rename_all = "snake_case")]` on the
    /// daemon's mirror of this enum.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Created => "created",
            Self::Assigned => "assigned",
            Self::StatusChanged => "status_changed",
            Self::Blocked => "blocked",
            Self::Unblocked => "unblocked",
            Self::Posted => "posted",
            Self::Due => "due",
            Self::Overdue => "overdue",
            Self::Verdict => "verdict",
            Self::Updated => "updated",
            Self::Deleted => "deleted",
        }
    }
}

/// A single task store mutation.
///
/// `detail` carries only what the kind needs — never the whole task. A
/// consumer that wants the card reads it; the event says what changed, and a
/// full row on every mutation would make the bus a replication channel.
#[derive(Debug, Clone)]
pub struct TaskEvent {
    pub kind: TaskEventKind,
    pub task_id: i64,
    pub scope: String,
    pub scope_id: Option<String>,
    /// Who caused it, as the activity log records actors. `None` when the
    /// mutation had no named actor (a cascade, an internal sweep).
    pub actor: Option<String>,
    pub detail: Value,
}

/// Where task store mutations are published.
///
/// # Contract
///
/// `publish` **must not block and must not panic.** It is called from the
/// repository on the write path, and a sink that waits turns every task write
/// in the process into a queue behind whatever it is waiting for. The daemon's
/// implementation is a `broadcast::Sender::send`, which is non-blocking and
/// drops on lag rather than applying back-pressure — the event-bus rule that a
/// slow consumer must never stall a producer.
///
/// Implementations are also called while other tasks may be mid-write, so they
/// must be `Send + Sync`.
pub trait TaskEventSink: Send + Sync {
    /// Publish one event. Non-blocking; see the trait contract.
    fn publish(&self, event: TaskEvent);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every kind, so a new variant has to be added here too.
    const ALL: [TaskEventKind; 11] = [
        TaskEventKind::Created,
        TaskEventKind::Assigned,
        TaskEventKind::StatusChanged,
        TaskEventKind::Blocked,
        TaskEventKind::Unblocked,
        TaskEventKind::Posted,
        TaskEventKind::Due,
        TaskEventKind::Overdue,
        TaskEventKind::Verdict,
        TaskEventKind::Updated,
        TaskEventKind::Deleted,
    ];

    #[test]
    fn serde_name_matches_as_str() {
        // The daemon puts this enum straight on the wire, so serde's
        // snake_case renaming IS the wire name. `as_str` exists for logs and
        // for callers that want it without serializing; if the two disagree,
        // a consumer matching on one silently stops seeing the other.
        for kind in ALL {
            let wire = serde_json::to_value(kind).expect("kind serializes");
            assert_eq!(wire, serde_json::Value::String(kind.as_str().to_string()));
        }
    }

    #[test]
    fn kind_wire_names_are_snake_case() {
        assert_eq!(TaskEventKind::Created.as_str(), "created");
        assert_eq!(TaskEventKind::Assigned.as_str(), "assigned");
        assert_eq!(TaskEventKind::StatusChanged.as_str(), "status_changed");
        assert_eq!(TaskEventKind::Blocked.as_str(), "blocked");
        assert_eq!(TaskEventKind::Unblocked.as_str(), "unblocked");
        assert_eq!(TaskEventKind::Posted.as_str(), "posted");
        assert_eq!(TaskEventKind::Due.as_str(), "due");
        assert_eq!(TaskEventKind::Overdue.as_str(), "overdue");
        assert_eq!(TaskEventKind::Verdict.as_str(), "verdict");
    }

    #[test]
    fn kind_wire_names_are_distinct() {
        let names: Vec<&str> = ALL.iter().map(|k| k.as_str()).collect();
        let unique: std::collections::HashSet<&str> = names.iter().copied().collect();
        assert_eq!(unique.len(), names.len(), "two kinds share a wire name");
    }
}
