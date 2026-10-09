//! Sessions are listed newest first whatever format their timestamp was
//! written in.

#![warn(clippy::all)]
#![warn(clippy::pedantic, clippy::nursery)]

use nanna_storage::Storage;

#[tokio::test]
async fn sessions_sort_by_time_not_by_timestamp_text() {
    let storage = Storage::in_memory().await.expect("storage");
    // RFC 3339, as the daemon writes it — 09:00.
    storage
        .upsert_daemon_session(
            "morning",
            None,
            None,
            "2026-10-09T08:00:00+00:00",
            "2026-10-09T09:00:00.123+00:00",
            None,
        )
        .await
        .expect("morning");
    // `datetime('now')` form, as a rename writes it — 15:00 the same day.
    storage
        .upsert_daemon_session(
            "afternoon",
            None,
            None,
            "2026-10-09 08:00:00",
            "2026-10-09 15:00:00",
            None,
        )
        .await
        .expect("afternoon");
    let order: Vec<String> = storage
        .list_daemon_sessions()
        .await
        .expect("list")
        .into_iter()
        .map(|session| session.session_id)
        .collect();
    assert_eq!(order, ["afternoon", "morning"], "newest first");
}
