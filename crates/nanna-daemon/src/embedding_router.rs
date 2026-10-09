//! Multi-provider embedding router with automatic fallback and re-embedding
//!
//! Routes embedding requests to the primary provider, falling back to alternates
//! when the primary is unavailable. Tracks provider switches and signals when
//! re-embedding may be needed due to dimension changes.

use nanna_llm::EmbeddingClient;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Identifies an embedding provider
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EmbeddingProviderInfo {
    /// Provider name (e.g., "ollama", "openai")
    pub name: String,
    /// Model name (e.g., "nomic-embed-text", "text-embedding-3-small")
    pub model: String,
}

impl std::fmt::Display for EmbeddingProviderInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.name, self.model)
    }
}

/// Backoff schedule used when EVERY provider is congested at once.
///
/// Congestion on a free tier is a throughput limit, not a failure — the correct
/// response is to come back later, and roughly nine minutes of patience is a
/// fair price for not losing the memory entirely. It only applies when there is
/// nothing else to try: a provider that is merely busy is waited for, a provider
/// that is broken is skipped, and any healthy provider wins immediately.
const BACKOFF_SECS: [u64; 7] = [2, 5, 15, 30, 60, 120, 240];

/// How long a deterministically-failing provider stays benched before it gets
/// one probe call.
///
/// A non-rate-limit failure (401 bad key, 402 no credit, 404 wrong model, 422
/// unprocessable, network down) does not clear on its own — something outside
/// has to change, and in general there is no signal for when it does (the one
/// there is, a changed credential, lifts the bench at once — see [`Bench`]).
/// Retrying per call, as the 2026-08-02 incident did with a 422ing primary,
/// buys nothing but an error per write. The only honest recovery is a bounded
/// re-probe horizon, and the router already has one: the last step of
/// [`BACKOFF_SECS`] is the longest it is ever willing to assume provider state
/// stays put. Reuse that rather than invent a second number.
const DEMOTION_SECS: u64 = BACKOFF_SECS[BACKOFF_SECS.len() - 1];

/// When a bench of `cooldown` starting at `now` ends.
///
/// `cooldown` can come straight from a provider's `Retry-After`, which nothing
/// upstream caps, and `Instant + Duration` PANICS on overflow — under
/// `panic = "abort"` a buggy or hostile endpoint publishing ~1e19 seconds took
/// the daemon down. A horizon past what `Instant` can hold is no usable
/// instruction, so it is treated as none: the standard [`DEMOTION_SECS`].
fn bench_until(now: Instant, cooldown: Duration) -> Instant {
    let until = now
        .checked_add(cooldown)
        .unwrap_or_else(|| now + Duration::from_secs(DEMOTION_SECS));
    debug_assert!(until >= now, "a bench never ends before it starts");
    until
}

/// A provider held out after a deterministic failure.
#[derive(Debug, Clone, Copy)]
struct Bench {
    /// Not called again before this.
    until: Instant,
    /// The client's credential generation the failing call was made under.
    ///
    /// A refused token is the one deterministic failure whose cause the
    /// daemon itself can see change: once the user saves a new token, the
    /// bench stands on a call the provider will never see again, and holding
    /// the fixed provider out for the rest of the cooldown only stores
    /// memories without vectors. A bench counts only while the credential is
    /// still the one it was set under.
    credential: u64,
}

/// An embedding provider entry in the router
struct EmbeddingProviderEntry {
    info: EmbeddingProviderInfo,
    client: Arc<EmbeddingClient>,
}

/// Multi-provider embedding router with automatic fallback.
///
/// Tries the primary provider first, then falls back through alternates
/// in order. Tracks which provider is currently active so consumers can
/// detect provider switches and trigger re-embedding if needed.
pub struct EmbeddingRouter {
    /// Ordered list: index 0 is primary, rest are fallbacks
    providers: Vec<EmbeddingProviderEntry>,
    /// Index of the currently active provider
    active_index: RwLock<usize>,
    /// Generation counter — incremented on every provider switch.
    /// Consumers compare their last-seen generation to detect changes.
    generation: AtomicU64,
    /// Per-provider bench; `Some` means "do not call before its deadline, while
    /// its credential is unchanged". Same length as `providers`. See
    /// [`DEMOTION_SECS`] and [`Bench`].
    demoted_until: RwLock<Vec<Option<Bench>>>,
    /// Per-provider input window, memoized. Same length as `providers`.
    ///
    /// Three states, and the difference between the last two matters — see
    /// [`WindowMemo`].
    windows: RwLock<Vec<WindowMemo>>,
}

/// What is known about one provider's input window.
///
/// `Probed(None)` ("asked, the provider publishes no limit") must stay distinct
/// from `Unprobed`: collapsing the two would re-probe a provider that will never
/// answer, once per embedding call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WindowMemo {
    /// Never asked.
    Unprobed,
    /// Asked: `Some(n)` is a real limit, `None` means no published limit.
    Probed(Option<usize>),
}

impl EmbeddingRouter {
    /// Create a new router with a primary provider.
    #[must_use]
    pub fn new(info: EmbeddingProviderInfo, client: Arc<EmbeddingClient>) -> Self {
        Self {
            providers: vec![EmbeddingProviderEntry { info, client }],
            active_index: RwLock::new(0),
            generation: AtomicU64::new(0),
            demoted_until: RwLock::new(vec![None]),
            windows: RwLock::new(vec![WindowMemo::Unprobed]),
        }
    }

    /// Add a fallback provider. Order matters — tried in insertion order.
    #[must_use]
    pub fn with_fallback(mut self, info: EmbeddingProviderInfo, client: Arc<EmbeddingClient>) -> Self {
        self.providers.push(EmbeddingProviderEntry { info, client });
        self.demoted_until.get_mut().push(None);
        self.windows.get_mut().push(WindowMemo::Unprobed);
        self
    }

    /// Get the current generation counter.
    /// Consumers should store this and compare on subsequent calls.
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Relaxed)
    }

    /// The input window of `info`'s model, in tokens, or `None` when the
    /// provider publishes no limit.
    ///
    /// Memoized per provider: this costs one `POST /api/show` the first time a
    /// provider is asked about and nothing thereafter. It is deliberately NOT
    /// called on the embedding path — a provider switch is the only event that
    /// can change the answer, and switches are rare.
    ///
    /// An unknown provider (not in this router) answers `None` rather than
    /// panicking on a missing index; the caller then falls back to its own
    /// default, which is the same thing it does for a provider that publishes
    /// nothing.
    pub async fn context_window_for(&self, info: &EmbeddingProviderInfo) -> Option<usize> {
        let idx = self.providers.iter().position(|e| &e.info == info)?;
        let memo = self.windows.read().await[idx];
        if let WindowMemo::Probed(memoized) = memo {
            return memoized;
        }
        let probed = self.providers[idx].client.context_window().await;
        self.windows.write().await[idx] = WindowMemo::Probed(probed);
        if let Some(window) = probed { debug!("Embedding provider {info} accepts {window} tokens per input") } else { debug!(
            "Embedding provider {info} publishes no input limit; chunking falls back to the \
             retrieval-granularity default"
        ); }
        probed
    }

    /// Get info about the currently active provider.
    pub async fn active_provider(&self) -> EmbeddingProviderInfo {
        let idx = *self.active_index.read().await;
        self.providers[idx].info.clone()
    }

    /// Whether `idx` is currently benched (deterministic failure, cooldown not
    /// yet passed, credential unchanged since).
    async fn is_demoted(&self, idx: usize) -> bool {
        let credential = self.providers[idx].client.credential_generation();
        self.demoted_until.read().await[idx]
            .is_some_and(|bench| Instant::now() < bench.until && bench.credential == credential)
    }

    /// Bench `idx` for `cooldown` — it failed deterministically under
    /// `credential` (its generation when the call was made), so calling it
    /// again sooner with that credential is a guaranteed error per call.
    async fn demote(&self, idx: usize, cooldown: Duration, credential: u64) {
        self.demoted_until.write().await[idx] = Some(Bench {
            until: bench_until(Instant::now(), cooldown),
            credential,
        });
    }

    /// Record that the caller has just observed a healthy embedding from the
    /// provider at `idx`; if it differs from the live active provider, switch
    /// to it and bump the generation.
    ///
    /// The check-and-swap runs under the live lock, not against the index the
    /// sweep started from — so a burst of racing calls that all found the same
    /// fallback produces ONE switch and ONE generation bump, not one per call.
    /// (Each spurious bump used to trigger a full store rebind downstream.)
    ///
    /// Returns the provider's info when this call actually flipped the active
    /// provider, `None` when it was already active.
    async fn commit_success(&self, idx: usize) -> Option<EmbeddingProviderInfo> {
        self.demoted_until.write().await[idx] = None;
        let mut active = self.active_index.write().await;
        if *active == idx {
            return None;
        }
        let old_info = self.providers[*active].info.clone();
        *active = idx;
        drop(active);
        self.generation.fetch_add(1, Ordering::Relaxed);
        let entry = &self.providers[idx];
        info!(
            "Embedding provider switched: {} → {} (generation {})",
            old_info,
            entry.info,
            self.generation.load(Ordering::Relaxed)
        );
        Some(entry.info.clone())
    }

    /// Embed a single text with automatic fallback.
    ///
    /// Tries the current active provider first. On failure, iterates through
    /// all providers starting from the next one. On successful fallback,
    /// updates the active provider and increments the generation counter.
    ///
    /// Returns `(embedding, switched_to)` where `switched_to` names the
    /// provider that produced the vector IF this call flipped the active
    /// provider to it. Vector and provider identity travel together so the
    /// caller can rebind its store to a consistent `(model, width)` pair —
    /// reading the active provider back after the fact can race a second
    /// switch and tear the pair apart.
    ///
    /// # Errors
    ///
    /// Returns an error when no provider produced a vector: every provider is
    /// benched after deterministic failures, every non-benched provider failed
    /// without merely being congested (the message carries the last error), or
    /// providers were still congested after the whole `BACKOFF_SECS` schedule.
    pub async fn embed_one(
        &self,
        text: &str,
    ) -> Result<(Vec<f32>, Option<EmbeddingProviderInfo>), String> {
        self.embed_with_patience(text, &BACKOFF_SECS).await
    }

    /// [`Self::embed_one`] without waiting on congestion: one sweep over every
    /// provider, and an error if they were all merely busy.
    ///
    /// For callers on a path something else is waiting for — daemon startup
    /// above all. Nine minutes of patience is the right price for not losing
    /// a memory in the background; spent on the boot-time dimension probe it
    /// kept the IPC port closed until the app gave up on the daemon and killed
    /// it (2026-09-18, a congested free-tier embedding model: the daemon never
    /// started). A healthy provider still wins on this single sweep exactly as
    /// it would on the first sweep of the patient path.
    ///
    /// # Errors
    ///
    /// As [`Self::embed_one`], and also when every provider was congested on
    /// the one sweep made.
    pub async fn embed_one_now(
        &self,
        text: &str,
    ) -> Result<(Vec<f32>, Option<EmbeddingProviderInfo>), String> {
        self.embed_with_patience(text, &[]).await
    }

    /// The sweep behind both entry points. `backoff` is the wait before each
    /// retry sweep while everything is congested; empty means one sweep only.
    async fn embed_with_patience(
        &self,
        text: &str,
        backoff: &[u64],
    ) -> Result<(Vec<f32>, Option<EmbeddingProviderInfo>), String> {
        let active_idx = *self.active_index.read().await;
        let total = self.providers.len();

        // A fallback that wins once must not hold the binding forever.
        //
        // The sweep returns on the FIRST success starting from the active
        // provider, so once a fallback is active the primary is never called
        // again: the store stays bound to the fallback's model and width for
        // the life of the daemon. There was a `try_restore_primary` for this,
        // whose doc said "call this periodically" — and nothing anywhere
        // called it.
        //
        // Rather than a timer nobody owns, the restore rides THIS call: when a
        // fallback is active and the primary is out of its bench cooldown, the
        // sweep starts at the primary instead. That costs no extra request in
        // the healthy case (the primary embeds the real text and wins), and a
        // switch back is reported with the vector that produced it, so the
        // caller rebinds to a consistent (model, width) pair exactly as it
        // does for a switch away.
        //
        // The period is a bound already derived here: a provider whose probe
        // fails is held for `DEMOTION_SECS`, so a displaced primary is
        // re-probed at that same horizon and no sooner.
        let reprobing_primary = active_idx != 0 && !self.is_demoted(0).await;
        let current_idx = if reprobing_primary { 0 } else { active_idx };

        // Sweep EVERY provider before waiting on any of them.
        //
        // The earlier shape spent the whole nine-minute patience budget on the
        // active provider before it would so much as look at the next one. That
        // is wrong twice over. A provider that is out of credit is not going to
        // recover in nine minutes, so a paid model hitting its cap would strand
        // the run while a perfectly healthy free-tier model sat unused. And the
        // reason waiting was preferred to switching — that a switch forced a
        // full re-embed at the new width — stopped being true the moment
        // embeddings became retained per-model buckets: a switch is now a
        // rebind, and switching back is a lookup.
        //
        // So: try everyone, and only wait if EVERYONE is merely congested.
        // Congestion is the one condition that clears on its own.
        for (round, wait) in std::iter::once(0)
            .chain(backoff.iter().copied())
            .enumerate()
        {
            if wait > 0 {
                tokio::time::sleep(std::time::Duration::from_secs(wait)).await;
            }

            let mut soonest_retry: Option<u64> = None;
            let mut congested = 0usize;
            let mut benched = 0usize;
            let mut last_error = String::new();

            for offset in 0..total {
                let idx = (current_idx + offset) % total;
                let entry = &self.providers[idx];

                // A benched provider's last error was deterministic; calling it
                // again before the cooldown is a guaranteed failure. Skip it —
                // this is what turns "422 per call" into "422 once, then quiet
                // until the probe horizon".
                if self.is_demoted(idx).await {
                    benched += 1;
                    continue;
                }

                // Read before the call: a token replaced while it is in flight
                // must not have the old token's failure benched against it.
                let credential = entry.client.credential_generation();
                match entry.client.embed_one(text).await {
                    Ok(embedding) if embedding.is_empty() => {
                        // An empty vector can never be a usable embedding, and
                        // downstream it would rebind the store to width 0.
                        // Treat it as the deterministic fault it is.
                        warn!("Embedding provider {} returned an empty vector", entry.info);
                        self.demote(idx, Duration::from_secs(DEMOTION_SECS), credential)
                            .await;
                        last_error = format!("{} returned an empty vector", entry.info);
                    }
                    Ok(embedding) => {
                        let switched_to = self.commit_success(idx).await;
                        return Ok((embedding, switched_to));
                    }
                    Err(e) if e.is_rate_limit() => {
                        congested += 1;
                        // Wait only as long as the SOONEST provider needs, not
                        // as long as the slowest — the schedule is a fallback
                        // for providers that decline to say.
                        let published =
                            if let nanna_llm::LlmError::RateLimit { retry_after, .. } = &e {
                                *retry_after
                            } else {
                                None
                            };
                        if let Some(secs) = published {
                            soonest_retry =
                                Some(soonest_retry.map_or(secs, |cur: u64| cur.min(secs)));
                        }
                        // Congestion normally never benches: it clears on its
                        // own and waiting is the right answer. But the primary
                        // was called here OUT OF TURN, purely to see whether it
                        // had recovered — a healthy fallback is already
                        // serving. Paying that question once per embed is the
                        // per-call retry this file exists to avoid, so hold it
                        // for as long as it published, or the standard horizon
                        // when it published nothing.
                        if reprobing_primary && idx == 0 {
                            let cooldown = published.unwrap_or(DEMOTION_SECS);
                            debug!(
                                "Primary {} is still congested; next restore probe in {}s",
                                entry.info, cooldown
                            );
                            self.demote(0, Duration::from_secs(cooldown), credential)
                                .await;
                        }
                        debug!("Embedding provider {} is congested", entry.info);
                        last_error = e.to_string();
                    }
                    Err(e) if e.is_input_overflow() => {
                        // The CALLER's fault, not the provider's: an oversized
                        // input is rejected deterministically, but the provider
                        // is healthy for every other call — benching it would
                        // take the whole capability down for the full cooldown
                        // over one bad input (observed 2026-08-10). The client
                        // already shrinks-and-retries internally, so reaching
                        // here means the report carried no usable limit; the
                        // sweep continues because a fallback with a larger
                        // window may simply accept the input as-is.
                        warn!(
                            "Embedding provider {} rejected the input as too long \
                             (provider stays available): {}",
                            entry.info, e
                        );
                        last_error = e.to_string();
                    }
                    Err(e) => {
                        // Genuinely unavailable — no key, no credit, wrong
                        // model, unprocessable payload. Waiting will not fix
                        // it, so it does not count toward "everything is busy"
                        // — and it is benched so the next call does not pay to
                        // rediscover the same failure.
                        warn!(
                            "Embedding provider {} unavailable (benched {}s): {}",
                            entry.info, DEMOTION_SECS, e
                        );
                        self.demote(idx, Duration::from_secs(DEMOTION_SECS), credential)
                            .await;
                        last_error = e.to_string();
                    }
                }
            }

            if congested == 0 {
                if benched == total {
                    return Err(format!(
                        "All {total} embedding providers are benched after deterministic \
                         failures — next probe in at most {DEMOTION_SECS}s"
                    ));
                }
                return Err(format!(
                    "All {total} embedding providers failed (none merely congested) — {last_error}"
                ));
            }

            announce_congestion_wait(backoff.get(round).copied(), round, congested, total, soonest_retry);
        }

        Err(if backoff.is_empty() {
            format!("All {total} embedding provider(s) congested — not waiting for them here")
        } else {
            format!("All {total} embedding providers still congested after the full retry schedule")
        })
    }

    // There is no separate `try_restore_primary` here any more. It probed with
    // a throwaway "probe" string, committed the switch without a vector to
    // carry it, and — the reason it is gone — had no callers anywhere despite
    // a doc saying "call this periodically". Restoring the primary is now part
    // of the sweep in `embed_one`, so it happens on the real text, reports the
    // switch alongside the vector that produced it, and cannot be forgotten.

    /// Number of configured providers (primary + fallbacks)
    pub const fn provider_count(&self) -> usize {
        self.providers.len()
    }
}

/// Log the wait before the next sweep while every provider is congested.
///
/// Only when another sweep will actually follow (`upcoming` is the schedule's
/// next step), and naming the sleep the loop really takes. A provider's own
/// Retry-After is reported beside it, not in its place: this line used to print
/// that figure ("waiting 0s") ahead of a two-second sleep.
fn announce_congestion_wait(
    upcoming: Option<u64>,
    round: usize,
    congested: usize,
    total: usize,
    asked_for: Option<u64>,
) {
    let Some(upcoming) = upcoming else { return };
    let asked = asked_for.map_or_else(String::new, |secs| format!(" (a provider asked for {secs}s)"));
    warn!(
        "All {congested} embedding provider(s) congested ({congested} of {total}) — \
         waiting {upcoming}s before sweep {}{asked}",
        round + 2
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unrepresentable_retry_after_benches_for_the_standard_horizon() {
        let now = Instant::now();
        assert_eq!(
            bench_until(now, Duration::MAX),
            now + Duration::from_secs(DEMOTION_SECS)
        );
        assert_eq!(
            bench_until(now, Duration::from_secs(45)),
            now + Duration::from_secs(45)
        );
    }
    use std::sync::atomic::AtomicUsize;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// Serve every request on an ephemeral port with a fixed status + body,
    /// counting how many requests arrived. `Connection: close` forces one TCP
    /// connection per request, so the accept count IS the request count.
    async fn spawn_counting_server(status: u16, body: &'static str) -> (String, Arc<AtomicUsize>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind an ephemeral port");
        let addr = listener.local_addr().expect("read back the bound addr");
        let hits = Arc::new(AtomicUsize::new(0));
        let counter = hits.clone();
        tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    break;
                };
                counter.fetch_add(1, Ordering::SeqCst);
                let mut buf = [0_u8; 4096];
                let _ = stream.read(&mut buf).await;
                let response = format!(
                    "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len(),
                );
                let _ = stream.write_all(response.as_bytes()).await;
                let _ = stream.flush().await;
            }
        });
        (format!("http://{addr}"), hits)
    }

    /// OpenAI-shaped client so each `embed_one` is exactly ONE http request
    /// (the Ollama client falls back new-API→legacy and would count double).
    fn openai_provider(name: &str, base_url: &str) -> (EmbeddingProviderInfo, Arc<EmbeddingClient>) {
        (
            EmbeddingProviderInfo {
                name: name.into(),
                model: "test-embed".into(),
            },
            Arc::new(
                EmbeddingClient::openai("test-key")
                    .with_model("test-embed")
                    .with_base_url(base_url),
            ),
        )
    }

    const HEALTHY_BODY: &str = r#"{"data":[{"embedding":[0.1,0.2,0.3]}]}"#;

    /// The incident's per-call retry: a provider that deterministically 422s
    /// must be called ONCE, benched, and then skipped without a network call —
    /// not re-tried on every embed.
    #[tokio::test]
    async fn a_deterministic_422_provider_is_benched_not_retried_per_call() {
        let (url, hits) = spawn_counting_server(422, r#"{"error":{"message":"bad input"}}"#).await;
        let (info, client) = openai_provider("openrouter", &url);
        let router = EmbeddingRouter::new(info, client);

        for _ in 0..3 {
            let err = router
                .embed_one("text")
                .await
                .expect_err("a 422-only router cannot produce an embedding");
            assert!(
                err.contains("benched") || err.contains("failed"),
                "the error names the condition: {err}"
            );
        }
        assert_eq!(
            hits.load(Ordering::SeqCst),
            1,
            "the dead provider is hit once, then benched — never once per call"
        );
    }

    /// An input-overflow rejection is the CALLER's fault: the provider stays
    /// available — never benched — so the next call reaches the network again
    /// instead of being short-circuited by a cooldown. (Contrast with the
    /// generic-422 test above, where the second call must NOT hit the wire.)
    ///
    /// The one-char input makes the client's own shrink-and-retry bottom out
    /// immediately, which is the only way an overflow report still reaches
    /// the router.
    #[tokio::test]
    async fn an_input_overflow_never_benches_the_provider() {
        let (url, hits) = spawn_counting_server(
            422,
            r#"{"error":{"message":"input length 100 exceeds maximum context length 50"}}"#,
        )
        .await;
        let (info, client) = openai_provider("openrouter", &url);
        let router = EmbeddingRouter::new(info, client);

        for _ in 0..2 {
            let err = router
                .embed_one("x")
                .await
                .expect_err("an unhealable overflow surfaces");
            assert!(
                !err.contains("benched"),
                "the failure is reported without a bench: {err}"
            );
        }
        assert_eq!(
            hits.load(Ordering::SeqCst),
            2,
            "the provider is consulted on every call — an input fault never benches it"
        );
    }

    /// A switch reports the provider that produced the vector, exactly once —
    /// the vector and its provenance must travel together so the caller can
    /// rebind to a consistent (model, width) pair.
    #[tokio::test]
    async fn a_switch_reports_the_producing_provider_exactly_once() {
        let (dead_url, dead_hits) =
            spawn_counting_server(422, r#"{"error":{"message":"bad input"}}"#).await;
        let (healthy_url, _healthy_hits) = spawn_counting_server(200, HEALTHY_BODY).await;
        let (dead_info, dead_client) = openai_provider("openrouter", &dead_url);
        let (ok_info, ok_client) = openai_provider("ollama", &healthy_url);
        let router = EmbeddingRouter::new(dead_info, dead_client).with_fallback(ok_info, ok_client);

        let (embedding, switched) = router.embed_one("text").await.expect("fallback answers");
        assert_eq!(embedding.len(), 3, "the fallback's vector comes through");
        let switched = switched.expect("the first call flips the active provider");
        assert_eq!(
            switched.name, "ollama",
            "the reported provider is the one that produced the vector"
        );
        assert_eq!(router.generation(), 1, "one flip, one generation bump");

        let (_, switched_again) = router.embed_one("text").await.expect("still healthy");
        assert!(
            switched_again.is_none(),
            "an already-active provider is not reported as a switch"
        );
        assert_eq!(router.generation(), 1, "no spurious generation bump");
        assert_eq!(
            dead_hits.load(Ordering::SeqCst),
            1,
            "the dead primary was benched by the first sweep"
        );
    }

    /// A fallback that wins once must not hold the binding forever. Once the
    /// primary's bench expires, the next embed reaches for it again and hands
    /// the binding back — and reports the switch WITH the vector, so the store
    /// rebinds to a consistent (model, width) pair. Without this the store
    /// stayed bound to the fallback for the life of the daemon.
    #[tokio::test]
    async fn a_recovered_primary_takes_the_binding_back() {
        let (primary_url, primary_hits) = spawn_counting_server(200, HEALTHY_BODY).await;
        let (fallback_url, _fallback_hits) = spawn_counting_server(200, HEALTHY_BODY).await;
        let (primary_info, primary_client) = openai_provider("openrouter", &primary_url);
        let (fallback_info, fallback_client) = openai_provider("ollama", &fallback_url);
        let router = EmbeddingRouter::new(primary_info, primary_client)
            .with_fallback(fallback_info, fallback_client);

        // Put the router on the fallback the way an outage would: bench the
        // primary, then embed once so the fallback commits. (Credential
        // generation 0: a token never replaced.)
        router
            .demote(0, Duration::from_secs(DEMOTION_SECS), 0)
            .await;
        let (_, switched) = router.embed_one("text").await.expect("the fallback answers");
        assert_eq!(
            switched.expect("the first call flips the active provider").name,
            "ollama"
        );

        // Still benched — the fallback keeps answering and the primary is not
        // called at all.
        let (_, switched) = router.embed_one("text").await.expect("still healthy");
        assert!(switched.is_none(), "a benched primary is not re-probed");
        assert_eq!(
            primary_hits.load(Ordering::SeqCst),
            0,
            "the bench is what keeps the restore off the per-call path"
        );

        // The cooldown passes. The very next embed must hand the binding back.
        router.demote(0, Duration::ZERO, 0).await;
        let (embedding, switched) = router.embed_one("text").await.expect("the primary answers");
        assert_eq!(embedding.len(), 3, "the restore rides a real embedding");
        assert_eq!(
            switched.expect("the restore is reported as a switch").name,
            "openrouter",
            "a recovered primary reclaims the binding without an external timer"
        );
        assert_eq!(
            router.active_provider().await.name,
            "openrouter",
            "and stays active"
        );
        assert_eq!(
            primary_hits.load(Ordering::SeqCst),
            1,
            "restoring costs the one call that produced the vector, no probe on the side"
        );
    }

    /// The restore must not become the per-call retry it replaced: a primary
    /// that is merely CONGESTED publishes when to come back, and is not asked
    /// again until then. (When the primary is the active provider, congestion
    /// still never benches it — that path is unchanged.)
    #[tokio::test]
    async fn a_congested_primary_is_not_re_probed_on_every_embed() {
        let (busy_url, busy_hits) =
            spawn_counting_server(429, r#"{"error":{"message":"rate limited"}}"#).await;
        let (healthy_url, _healthy_hits) = spawn_counting_server(200, HEALTHY_BODY).await;
        let (busy_info, busy_client) = openai_provider("openrouter", &busy_url);
        let (ok_info, ok_client) = openai_provider("ollama", &healthy_url);
        let router = EmbeddingRouter::new(busy_info, busy_client).with_fallback(ok_info, ok_client);

        for _ in 0..3 {
            router.embed_one("text").await.expect("the fallback answers");
        }
        assert_eq!(
            router.active_provider().await.name,
            "ollama",
            "the healthy fallback is serving"
        );
        assert_eq!(
            busy_hits.load(Ordering::SeqCst),
            2,
            "once on the way to the fallback, once as the restore attempt — then held"
        );
    }

    /// The startup probe's sweep: a congested provider is asked ONCE and the
    /// call returns, instead of spending the nine-minute patience schedule.
    /// That schedule, on the boot-time dimension probe, kept the daemon's IPC
    /// port closed until the app killed it (2026-09-18) — each further sweep
    /// would be another request here, so one hit proves nothing waited.
    #[tokio::test]
    async fn embed_one_now_does_not_wait_on_a_congested_provider() {
        let (busy_url, busy_hits) =
            spawn_counting_server(429, r#"{"error":{"message":"rate limited"}}"#).await;
        let (busy_info, busy_client) = openai_provider("openrouter", &busy_url);
        let router = EmbeddingRouter::new(busy_info, busy_client);

        let err = router
            .embed_one_now("dimension probe")
            .await
            .expect_err("a congested-only router has nothing to return");
        assert!(err.contains("congested"), "the error says why: {err}");
        assert_eq!(busy_hits.load(Ordering::SeqCst), 1, "one sweep, no retries");
    }

    /// A healthy provider wins on the single sweep exactly as on the patient
    /// path, so the startup probe loses nothing when the provider is fine.
    #[tokio::test]
    async fn embed_one_now_returns_a_healthy_providers_vector() {
        let (url, hits) = spawn_counting_server(200, HEALTHY_BODY).await;
        let (info, client) = openai_provider("ollama", &url);
        let router = EmbeddingRouter::new(info, client);

        let (vector, switched) = router.embed_one_now("dimension probe").await.expect("healthy");
        assert_eq!(vector.len(), 3);
        assert!(switched.is_none(), "the active provider answered; nothing switched");
        assert_eq!(hits.load(Ordering::SeqCst), 1);
    }

    /// The bench expires: after the cooldown the provider is eligible again,
    /// so a fixed outage never becomes a permanent demotion.
    #[tokio::test]
    async fn a_bench_expires_after_its_cooldown() {
        let (url, _hits) = spawn_counting_server(200, HEALTHY_BODY).await;
        let (info, client) = openai_provider("ollama", &url);
        let router = EmbeddingRouter::new(info, client);

        router.demote(0, Duration::ZERO, 0).await;
        assert!(
            !router.is_demoted(0).await,
            "a zero cooldown is already expired — the provider must be eligible"
        );
        let (embedding, switched) = router.embed_one("text").await.expect("eligible again");
        assert_eq!(embedding.len(), 3);
        assert!(switched.is_none(), "it was the active provider all along");
    }

    /// A bench stands in for the signal nobody sends: "whatever made this
    /// provider fail has changed". For a refused token there IS one — the
    /// token changing — and a bench that outlived it kept the fixed provider
    /// out for the rest of its cooldown, every memory in between stored
    /// without a vector. A changed credential lifts the bench; saving the same
    /// one again does not.
    #[tokio::test]
    async fn a_changed_credential_lifts_the_bench() {
        let (url, seen) =
            crate::embedding_reload::test_ollama::spawn_token_gated("new-token").await;
        let bearer = nanna_llm::SharedBearer::new("old-token");
        let router = EmbeddingRouter::new(
            EmbeddingProviderInfo {
                name: "ollama".into(),
                model: "test-embed".into(),
            },
            Arc::new(
                EmbeddingClient::openai("")
                    .with_bearer(bearer.clone())
                    .with_model("test-embed")
                    .with_base_url(&url),
            ),
        );

        router
            .embed_one("text")
            .await
            .expect_err("the server refuses the old token");
        assert!(!bearer.replace("old-token"), "the same token saved again");
        router
            .embed_one("text")
            .await
            .expect_err("nothing changed, so the bench holds");
        assert_eq!(
            seen.lock().expect("record lock").len(),
            1,
            "a benched provider is not called"
        );

        assert!(bearer.replace("new-token"));
        let (embedding, _) = router
            .embed_one("text")
            .await
            .expect("the new token is tried at once, and accepted");
        assert_eq!(embedding.len(), 3);
        let seen = seen.lock().expect("record lock");
        assert_eq!(seen.len(), 2);
        assert_eq!(seen[1].authorization.as_deref(), Some("Bearer new-token"));
        drop(seen);
    }
}
