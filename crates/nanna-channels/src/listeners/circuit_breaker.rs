//! Circuit breaker for channel listeners.
//!
//! Tracks consecutive connection and authentication failures, provides
//! exponential backoff, and stops the listener when authentication keeps
//! failing (a connection failure only ever backs off).  Reports state transitions via the optional [`StatusManager`].

use crate::status::{ConnectionState, StatusManager};
use std::sync::Arc;
use std::time::Duration;
use tracing::{error, info, warn};

/// Default number of consecutive auth failures before tripping the breaker.
const DEFAULT_MAX_AUTH_FAILURES: u32 = 3;

/// Default number of consecutive connection failures before tripping.
const DEFAULT_MAX_CONN_FAILURES: u32 = 10;

/// Default maximum backoff delay (seconds).
const DEFAULT_MAX_BACKOFF_SECS: u64 = 120;

/// Shared circuit-breaker state for a single channel listener.
///
/// Create one per listener and call the appropriate `record_*` / `backoff`
/// methods from the reconnect loop.  When a method returns
/// [`BreakerAction::Stop`] the caller should exit its loop and let the
/// listener task finish.
pub struct CircuitBreaker {
    /// Provider name used for status-manager updates and log messages.
    provider: String,

    /// Optional status manager — all state updates are no-ops when `None`.
    status_manager: Option<Arc<StatusManager>>,

    // ── Counters ────────────────────────────────────────────────────────
    /// Consecutive non-resumable auth / identity failures.
    auth_failures: u32,

    /// Consecutive transient connection failures.
    conn_failures: u32,

    // ── Thresholds ──────────────────────────────────────────────────────
    max_auth_failures: u32,
    max_conn_failures: u32,

    /// Cap for exponential backoff (seconds).
    max_backoff_secs: u64,
}

/// Action the caller should take after recording a failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreakerAction {
    /// Keep trying — wait out the backoff returned by [`CircuitBreaker::backoff`].
    Retry,
    /// The breaker has tripped — stop the listener entirely.
    Stop,
}

impl CircuitBreaker {
    /// Create a new breaker for the given provider name.
    pub fn new(provider: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            status_manager: None,
            auth_failures: 0,
            conn_failures: 0,
            max_auth_failures: DEFAULT_MAX_AUTH_FAILURES,
            max_conn_failures: DEFAULT_MAX_CONN_FAILURES,
            max_backoff_secs: DEFAULT_MAX_BACKOFF_SECS,
        }
    }

    // ── Builder helpers ─────────────────────────────────────────────────

    #[must_use]
    pub fn with_status_manager(mut self, sm: Arc<StatusManager>) -> Self {
        self.status_manager = Some(sm);
        self
    }

    #[must_use]
    pub const fn with_max_auth_failures(mut self, n: u32) -> Self {
        self.max_auth_failures = n;
        self
    }

    #[must_use]
    pub const fn with_max_conn_failures(mut self, n: u32) -> Self {
        self.max_conn_failures = n;
        self
    }

    #[must_use]
    pub const fn with_max_backoff_secs(mut self, secs: u64) -> Self {
        self.max_backoff_secs = secs;
        self
    }

    // ── Recording outcomes ──────────────────────────────────────────────

    /// Record a successful connection / authentication.
    /// Resets all failure counters and reports `Connected` state.
    pub async fn record_success(&mut self) {
        self.auth_failures = 0;
        self.conn_failures = 0;
        self.report(ConnectionState::Connected, None).await;
    }

    /// Record an authentication / identity failure (e.g. 401, invalid token,
    /// Discord op-9 non-resumable).
    ///
    /// Returns [`BreakerAction::Stop`] when the threshold is reached.
    pub async fn record_auth_failure(&mut self, detail: &str) -> BreakerAction {
        self.auth_failures += 1;
        self.conn_failures = 0; // auth failure is a different category

        if self.auth_failures >= self.max_auth_failures {
            let msg = format!(
                "Authentication failed after {} consecutive attempts: {}. \
                 Verify credentials and reconnect from the Channels page.",
                self.auth_failures, detail
            );
            error!(provider = %self.provider, "{}", msg);
            self.report(ConnectionState::AuthFailed, Some(msg)).await;
            BreakerAction::Stop
        } else {
            warn!(
                provider = %self.provider,
                attempt = self.auth_failures,
                max = self.max_auth_failures,
                "Auth failure ({}), will retry after backoff",
                detail
            );
            self.report(
                ConnectionState::Connecting,
                Some(format!(
                    "Auth failure (attempt {}/{}): {}",
                    self.auth_failures, self.max_auth_failures, detail
                )),
            )
            .await;
            BreakerAction::Retry
        }
    }

    /// Record a transient connection failure (network error, timeout, etc.).
    ///
    /// Always [`BreakerAction::Retry`]: past the threshold the channel is
    /// reported `Unavailable` and keeps retrying at the backoff cap. It used
    /// to stop — after about eight minutes of backoff (2 + 4 + … + 120 s) —
    /// and nothing could restart a stopped listener (no restart action, no
    /// "reconnect" on the Channels page its message pointed to), so a laptop
    /// that lost Wi-Fi for ten minutes left its channels deaf until the
    /// daemon restarted. A network that comes back is the common case;
    /// only a refused credential, which retrying cannot fix, stops.
    pub async fn record_conn_failure(&mut self, detail: &str) -> BreakerAction {
        self.conn_failures = self.conn_failures.saturating_add(1);

        if self.conn_failures >= self.max_conn_failures {
            let msg = format!(
                "Connection failed {} times in a row: {}. Still retrying every {} s; \
                 check network / service availability.",
                self.conn_failures,
                detail,
                self.backoff_duration().as_secs()
            );
            error!(provider = %self.provider, "{}", msg);
            self.report(ConnectionState::Unavailable, Some(msg)).await;
        } else {
            warn!(
                provider = %self.provider,
                attempt = self.conn_failures,
                max = self.max_conn_failures,
                "Connection failure ({}), will retry after backoff",
                detail
            );
            self.report(
                ConnectionState::Connecting,
                Some(format!(
                    "Connection failure (attempt {}/{}): {}",
                    self.conn_failures, self.max_conn_failures, detail
                )),
            )
            .await;
        }
        BreakerAction::Retry
    }

    // ── Backoff ─────────────────────────────────────────────────────────

    /// Sleep for an exponential backoff based on the total failure count.
    ///
    /// Call this after `record_auth_failure` / `record_conn_failure` returns
    /// [`BreakerAction::Retry`].
    pub async fn backoff(&self) {
        let failures = self.auth_failures.max(self.conn_failures);
        if failures == 0 {
            return;
        }
        let delay = self.backoff_duration();
        info!(
            provider = %self.provider,
            delay_secs = delay.as_secs(),
            "Backing off before next attempt"
        );
        tokio::time::sleep(delay).await;
    }

    /// Compute the backoff duration without sleeping.
    #[must_use]
    pub fn backoff_duration(&self) -> Duration {
        let failures = self.auth_failures.max(self.conn_failures);
        if failures == 0 {
            return Duration::ZERO;
        }
        // 2^failures seconds, capped at max
        let secs = 2u64.saturating_pow(failures).min(self.max_backoff_secs);
        Duration::from_secs(secs)
    }

    // ── State reporting ─────────────────────────────────────────────────

    /// Report `Connecting` state.
    pub async fn report_connecting(&self) {
        self.report(ConnectionState::Connecting, None).await;
    }

    /// Report an arbitrary state + detail string.
    pub async fn report(&self, state: ConnectionState, details: Option<String>) {
        if let Some(sm) = &self.status_manager {
            sm.set_state(&self.provider, state, details).await;
        }
    }

    // ── Accessors ───────────────────────────────────────────────────────

    #[must_use]
    pub const fn auth_failures(&self) -> u32 {
        self.auth_failures
    }

    #[must_use]
    pub const fn conn_failures(&self) -> u32 {
        self.conn_failures
    }

    #[must_use]
    pub fn provider(&self) -> &str {
        &self.provider
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_auth_failure_trips_after_threshold() {
        let mut cb = CircuitBreaker::new("test").with_max_auth_failures(2);
        assert_eq!(cb.record_auth_failure("bad token").await, BreakerAction::Retry);
        assert_eq!(cb.record_auth_failure("bad token").await, BreakerAction::Stop);
    }

    #[tokio::test]
    async fn test_conn_failures_never_stop_the_listener() {
        let mut cb = CircuitBreaker::new("test").with_max_conn_failures(3);
        for _ in 0..50 {
            assert_eq!(cb.record_conn_failure("timeout").await, BreakerAction::Retry);
        }
        assert_eq!(
            cb.backoff_duration(),
            Duration::from_secs(DEFAULT_MAX_BACKOFF_SECS),
            "retries settle at the cap"
        );
    }

    #[tokio::test]
    async fn test_success_resets_counters() {
        let mut cb = CircuitBreaker::new("test").with_max_auth_failures(3);
        cb.record_auth_failure("bad token").await;
        cb.record_auth_failure("bad token").await;
        assert_eq!(cb.auth_failures(), 2);
        cb.record_success().await;
        assert_eq!(cb.auth_failures(), 0);
        assert_eq!(cb.conn_failures(), 0);
    }

    #[test]
    fn test_backoff_duration() {
        let mut cb = CircuitBreaker::new("test").with_max_backoff_secs(60);
        assert_eq!(cb.backoff_duration(), Duration::ZERO);
        cb.conn_failures = 1;
        assert_eq!(cb.backoff_duration(), Duration::from_secs(2));
        cb.conn_failures = 3;
        assert_eq!(cb.backoff_duration(), Duration::from_secs(8));
        cb.conn_failures = 10;
        assert_eq!(cb.backoff_duration(), Duration::from_secs(60)); // capped
    }
}
