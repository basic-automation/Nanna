//! Handing freed heap back to the OS (Linux, glibc).
//!
//! glibc keeps freed pages in its arenas and reuses them, but never returns
//! them on its own below the top of the heap. The daemon is long-lived, and
//! its big allocations are bursts: the boot's `bulk_load`, and a large IPC
//! reply (a 3 730-memory `memory.list` is a 14 MB string built from a
//! `serde_json::Value` tree). Each burst left RSS at its high water. Measured
//! on the release daemon on a copy of the operator's store (2026-10-05): idle
//! at 142.5 MB, then 217 MB after one `memory.list` and ~425 MB after nine.
//!
//! So the daemon trims twice: once at boot ([`release_freed_heap`], before
//! "Daemon ready"), and after a large reply has been sent
//! ([`after_large_reply`]), at most once per [`TRIM_INTERVAL`]. Other
//! platforms and allocators do nothing here.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use tracing::debug;

/// The reply size, in bytes, past which sending it is followed by a trim.
///
/// Bound justification: ordinary replies (a status, a page of tasks, a chat
/// event) are kilobytes, so they never pay. A megabyte is where a reply's
/// transient heap (its `Value` tree, its string, its frame, several times the
/// reply) is large enough that keeping it shows in RSS; lists and exports of
/// a whole store are past it.
pub const LARGE_REPLY_BYTES: usize = 1024 * 1024;

/// The shortest time between two trims after large replies.
///
/// Bound justification: a trim walks the arenas under their locks, which is
/// milliseconds on a few hundred MB. One per 10 s caps that at a fraction of
/// a percent of one core however fast a client asks for large replies, and
/// a burst of them is covered by the trim after the burst.
pub const TRIM_INTERVAL: Duration = Duration::from_secs(10);

/// Seconds since [`epoch`] at the last trim after a large reply, plus one
/// (zero: none yet).
static LAST_TRIM_SECS: AtomicU64 = AtomicU64::new(0);

/// The process's reference instant for [`LAST_TRIM_SECS`].
fn epoch() -> Instant {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    *EPOCH.get_or_init(Instant::now)
}

/// Whether a reply of `reply_bytes`, sent `now_secs` after the epoch, is
/// followed by a trim, given the last one at `last_secs` (stored plus one,
/// zero for none). Pure.
#[must_use]
pub const fn trim_due(reply_bytes: usize, now_secs: u64, last_secs: u64) -> bool {
    if reply_bytes < LARGE_REPLY_BYTES {
        return false;
    }
    last_secs == 0
        || now_secs.saturating_add(1) >= last_secs.saturating_add(TRIM_INTERVAL.as_secs())
}

/// Return the arenas' free pages to the OS. `true` when some memory was
/// released. Blocking (it takes the arena locks): call it off the async
/// workers, or once where blocking is fine (the boot).
pub fn release_freed_heap() -> bool {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        let started = Instant::now();
        // SAFETY: `malloc_trim` takes no pointers and only returns free pages
        // to the OS; glibc documents it as callable at any time, from any thread.
        let released = unsafe { libc::malloc_trim(0) } != 0;
        debug!(
            released,
            took_us = started.elapsed().as_micros(),
            "Returned freed heap to the OS"
        );
        released
    }
    #[cfg(not(all(target_os = "linux", target_env = "gnu")))]
    {
        false
    }
}

/// A reply of `reply_bytes` has just been sent (and its buffers dropped):
/// trim on the blocking pool if it was large and no trim ran recently.
pub fn after_large_reply(reply_bytes: usize) {
    if reply_bytes < LARGE_REPLY_BYTES {
        return;
    }
    let now_secs = epoch().elapsed().as_secs();
    let last_secs = LAST_TRIM_SECS.load(Ordering::Relaxed);
    if !trim_due(reply_bytes, now_secs, last_secs) {
        return;
    }
    // One caller wins the slot; a concurrent large reply sees it taken.
    if LAST_TRIM_SECS
        .compare_exchange(
            last_secs,
            now_secs + 1,
            Ordering::Relaxed,
            Ordering::Relaxed,
        )
        .is_err()
    {
        return;
    }
    debug!(reply_bytes, "Large reply sent: trimming the heap");
    drop(tokio::task::spawn_blocking(release_freed_heap));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_large_reply_trims_and_at_most_once_per_interval() {
        let interval = TRIM_INTERVAL.as_secs();
        assert!(
            !trim_due(LARGE_REPLY_BYTES - 1, 100, 0),
            "small replies never pay"
        );
        assert!(
            trim_due(LARGE_REPLY_BYTES, 0, 0),
            "the first large reply trims"
        );
        // Last trim at second 50 (stored as 51).
        assert!(!trim_due(LARGE_REPLY_BYTES, 50, 51));
        assert!(!trim_due(LARGE_REPLY_BYTES, 50 + interval - 1, 51));
        assert!(trim_due(LARGE_REPLY_BYTES, 50 + interval, 51));
        assert!(
            trim_due(usize::MAX, u64::MAX - interval, 1),
            "no overflow at the extremes"
        );
    }

    #[tokio::test]
    async fn a_large_reply_claims_the_slot_and_a_second_one_waits() {
        after_large_reply(LARGE_REPLY_BYTES);
        let claimed = LAST_TRIM_SECS.load(Ordering::Relaxed);
        assert!(claimed > 0, "the first large reply took the slot");
        after_large_reply(LARGE_REPLY_BYTES);
        assert_eq!(
            LAST_TRIM_SECS.load(Ordering::Relaxed),
            claimed,
            "within the interval"
        );
        after_large_reply(10);
        assert_eq!(LAST_TRIM_SECS.load(Ordering::Relaxed), claimed);
    }
}
