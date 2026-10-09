//! The all-tools rows of the tool-stats time series.

#![warn(clippy::all)]
#![warn(clippy::pedantic, clippy::nursery)]

use nanna_storage::{NewToolCall, Storage};

async fn call(storage: &Storage, tool_name: &str, duration_ms: u64) {
    storage
        .log_tool_call(&NewToolCall {
            tool_name,
            success: true,
            short_circuited: false,
            duration_ms,
            output_size: 0,
            error_message: None,
            session_id: None,
        })
        .await
        .expect("logged");
}

/// The all-tools average is per call, not an average of each tool's average:
/// one slow call of one tool must not outweigh 99 fast calls of another.
#[tokio::test]
async fn the_all_tools_average_is_weighted_by_calls() {
    let storage = Storage::in_memory().await.expect("storage");
    call(&storage, "slow", 1_000).await;
    for _ in 0..99 {
        call(&storage, "fast", 10).await;
    }
    for buckets in [
        storage
            .get_tool_stats_hourly(None, 2)
            .await
            .expect("hourly"),
        storage.get_tool_stats_daily(None, 2).await.expect("daily"),
    ] {
        let bucket = buckets.last().expect("a bucket for now");
        assert_eq!(bucket.call_count, 100);
        // (1000 + 99·10) / 100 = 19.9 → 19; the unweighted mean was 505.
        assert_eq!(bucket.avg_duration_ms, 19, "{bucket:?}");
    }
}
