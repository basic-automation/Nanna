//! Carries task store mutations from `nanna-storage` onto the daemon's event bus.
//!
//! `nanna-storage` emits [`nanna_storage::TaskEvent`] because the repository is
//! the only layer that sees every writer (see that module for why). It does not
//! know the daemon's wire protocol, so this is the adapter: one struct holding
//! the broadcast sender, translating each store event into
//! [`Event::TaskEvent`].

use crate::protocol::Event;
use nanna_storage::{TaskEvent as StoreTaskEvent, TaskEventSink};
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::{broadcast, mpsc};

/// Publishes storage task events onto the daemon event bus, hands the ones
/// that become memories to the board write-through queue, and the ones that
/// wake the board router to the route queue.
pub struct TaskEventBridge {
    events: broadcast::Sender<Event>,
    write_through: Option<mpsc::Sender<StoreTaskEvent>>,
    /// Copies the queue refused because it was full. Counted so the loss is
    /// visible (logged with a running total), never silent.
    write_through_dropped: AtomicU64,
    routes: Option<mpsc::Sender<crate::board_router_trigger::Wake>>,
    /// Cards the route queue refused because it was full — same contract as
    /// `write_through_dropped`.
    routes_dropped: AtomicU64,
    runs: Option<mpsc::Sender<crate::card_run_trigger::RunWake>>,
    /// Wakes the card-run queue refused because it was full.
    runs_dropped: AtomicU64,
}

impl TaskEventBridge {
    #[must_use]
    pub const fn new(events: broadcast::Sender<Event>) -> Self {
        Self {
            events,
            write_through: None,
            write_through_dropped: AtomicU64::new(0),
            routes: None,
            routes_dropped: AtomicU64::new(0),
            runs: None,
            runs_dropped: AtomicU64::new(0),
        }
    }

    /// Also queue the cards that wake the board router for
    /// [`crate::board_router_trigger::run`].
    #[must_use]
    pub fn with_router_queue(
        mut self,
        queue: mpsc::Sender<crate::board_router_trigger::Wake>,
    ) -> Self {
        self.routes = Some(queue);
        self
    }

    /// Queue `event`'s card for the router if the event wakes it. Never
    /// waits, for the same reason as [`Self::queue_copy`].
    fn queue_route(&self, event: &StoreTaskEvent) {
        let Some(queue) = self.routes.as_ref() else {
            return;
        };
        let Some(wake) = crate::board_router_trigger::wake_for(event) else {
            return;
        };
        debug_assert_eq!(wake.task_id, event.task_id, "the event's own card");
        // `Closed` means no worker (no storage-backed router on this daemon),
        // so nothing is owed a decision.
        if let Err(mpsc::error::TrySendError::Full(_)) = queue.try_send(wake) {
            let dropped = self.routes_dropped.fetch_add(1, Ordering::Relaxed) + 1;
            tracing::warn!(
                "board router queue is full; card #{} was not routed ({dropped} dropped so \
                 far) — the card itself is unaffected and can be assigned by hand",
                event.task_id
            );
        }
    }

    /// Also queue the cards that may have become workable for
    /// [`crate::card_run_trigger::run`].
    #[must_use]
    pub fn with_run_queue(mut self, queue: mpsc::Sender<crate::card_run_trigger::RunWake>) -> Self {
        self.runs = Some(queue);
        self
    }

    /// Queue `event`'s card for the card-run worker if it may now be
    /// workable. Never waits, for the same reason as [`Self::queue_copy`].
    fn queue_run(&self, event: &StoreTaskEvent) {
        let Some(queue) = self.runs.as_ref() else {
            return;
        };
        let Some(wake) = crate::card_run_trigger::run_wake_for(event) else {
            return;
        };
        if let Err(mpsc::error::TrySendError::Full(_)) = queue.try_send(wake) {
            let dropped = self.runs_dropped.fetch_add(1, Ordering::Relaxed) + 1;
            tracing::warn!(
                "card run queue is full; card #{} was not started ({dropped} dropped so far) \
                 — it starts when its member next frees, or by task.start_run",
                event.task_id
            );
        }
    }

    /// Also queue memory-producing events for
    /// [`crate::memory_write_through::run`].
    #[must_use]
    pub fn with_memory_write_through(mut self, queue: mpsc::Sender<StoreTaskEvent>) -> Self {
        self.write_through = Some(queue);
        self
    }

    /// Queue `event` for the write-through worker if it produces a memory.
    /// Never waits: this runs on the task store's write path.
    fn queue_copy(&self, event: &StoreTaskEvent) {
        let Some(queue) = self.write_through.as_ref() else {
            return;
        };
        if !crate::memory_write_through::is_board_record(event) {
            return;
        }
        // `Ok`, and `Closed` — no worker, because memory is not configured on
        // this daemon, so nothing is owed a copy — both need nothing done.
        if let Err(mpsc::error::TrySendError::Full(_)) = queue.try_send(event.clone()) {
            let dropped = self.write_through_dropped.fetch_add(1, Ordering::Relaxed) + 1;
            tracing::warn!(
                "board memory write-through queue is full; card #{}'s copy was not \
                 written ({dropped} dropped so far) — the card itself is unaffected",
                event.task_id
            );
        }
    }
}

impl TaskEventSink for TaskEventBridge {
    /// Forward one store mutation to the bus.
    ///
    /// `send` on a `broadcast::Sender` is non-blocking and returns `Err` when
    /// there is no subscriber — which is the normal state of a daemon nobody
    /// has attached a client to, not a failure. The result is deliberately
    /// discarded: the sink contract forbids blocking, and a task write must
    /// never fail because nothing was listening to its announcement.
    fn publish(&self, event: StoreTaskEvent) {
        self.queue_copy(&event);
        self.queue_route(&event);
        self.queue_run(&event);
        let _ = self.events.send(Event::TaskEvent {
            kind: event.kind,
            task_id: event.task_id,
            scope: event.scope,
            scope_id: event.scope_id,
            actor: event.actor,
            detail: event.detail,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nanna_storage::TaskEventKind;

    fn store_event(kind: TaskEventKind) -> StoreTaskEvent {
        StoreTaskEvent {
            kind,
            task_id: 7,
            scope: "workspace".to_string(),
            scope_id: Some("ws1".to_string()),
            actor: Some("gui".to_string()),
            detail: serde_json::json!({"title": "a card"}),
        }
    }

    #[test]
    fn publishes_every_field_onto_the_bus() {
        let (tx, mut rx) = broadcast::channel(4);
        let bridge = TaskEventBridge::new(tx);
        bridge.publish(store_event(TaskEventKind::Created));

        let event = rx.try_recv().expect("event reached the bus");
        let Event::TaskEvent {
            kind,
            task_id,
            scope,
            scope_id,
            actor,
            detail,
        } = event
        else {
            panic!("wrong variant: {event:?}");
        };
        assert_eq!(kind, TaskEventKind::Created);
        assert_eq!(task_id, 7);
        assert_eq!(scope, "workspace");
        assert_eq!(scope_id.as_deref(), Some("ws1"));
        assert_eq!(actor.as_deref(), Some("gui"));
        assert_eq!(detail, serde_json::json!({"title": "a card"}));
    }

    #[test]
    fn publish_with_no_subscriber_is_not_an_error() {
        // A daemon with no attached client is the ordinary case; a task write
        // must not fail because nobody was listening.
        let (tx, rx) = broadcast::channel(4);
        drop(rx);
        let bridge = TaskEventBridge::new(tx);
        bridge.publish(store_event(TaskEventKind::Verdict));
    }

    #[test]
    fn only_router_waking_events_reach_the_route_queue() {
        let (tx, _rx) = broadcast::channel(8);
        let (routes_tx, mut routes_rx) = mpsc::channel(4);
        let bridge = TaskEventBridge::new(tx).with_router_queue(routes_tx);

        bridge.publish(store_event(TaskEventKind::Created));
        bridge.publish(store_event(TaskEventKind::Posted));
        let mut by_harness = store_event(TaskEventKind::Created);
        by_harness.actor = Some("harness".to_string());
        bridge.publish(by_harness);

        assert_eq!(
            routes_rx.try_recv().ok().map(|wake| wake.task_id),
            Some(7),
            "the gui's card is queued"
        );
        assert!(
            routes_rx.try_recv().is_err(),
            "a post and the harness's own card are not"
        );
    }

    #[test]
    fn a_full_route_queue_counts_the_drop_and_never_waits() {
        let (tx, _rx) = broadcast::channel(8);
        let (routes_tx, _routes_rx) = mpsc::channel(1);
        let bridge = TaskEventBridge::new(tx).with_router_queue(routes_tx);

        bridge.publish(store_event(TaskEventKind::Created));
        bridge.publish(store_event(TaskEventKind::Created));
        bridge.publish(store_event(TaskEventKind::Created));

        assert_eq!(bridge.routes_dropped.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn task_events_belong_to_no_session() {
        // P25 decision 9: a card outlives every session, so a per-session
        // subscriber must not receive board events as if they were its own.
        let event = Event::TaskEvent {
            kind: TaskEventKind::Posted,
            task_id: 1,
            scope: "workspace".to_string(),
            scope_id: None,
            actor: None,
            detail: serde_json::Value::Null,
        };
        assert_eq!(event.session_id(), None);
    }
}
