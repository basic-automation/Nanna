//! The summarizer the daemon's dream cycles call.
//!
//! Both consolidation paths — the scheduled cycle in [`crate::server`] and the
//! IPC `MemoryAction::Consolidate` in [`crate::control`] — need the same three
//! things: the user's configured summarization models, a **failover walk** over
//! that list, and a cluster byte budget that is safe for whichever model ends
//! up answering. They used to disagree: the scheduled path took only
//! `summarization_priority.first()` and made a single attempt, so one
//! unavailable or rate-limited model killed the whole nightly dream cycle,
//! while the IPC path already walked the list. This module is the one
//! implementation both now share.

use crate::llm_router::LlmRouter;
use nanna_llm::RequestBuilder;
use std::sync::Arc;

// NOTE: request pacing for rate-limited providers is NOT done here. It lives
// at the provider level in `nanna_llm::pace_openrouter`, awaited inside every
// OpenRouter-bound request path — so chat, dream summarization, and any future
// caller draw from one shared clock instead of pacing themselves and
// collectively burning the same quota.

/// Smallest summarizer context window we will ever size a cluster against.
///
/// Mirrors `nanna_memory`'s own fallback for an unknown model. A `ModelInfo`
/// this small means we failed to resolve the model at all; clamping here keeps
/// the byte-budget math meaningful instead of collapsing to zero.
const MIN_SUMMARIZER_CONTEXT_TOKENS: usize = 8_192;

/// The ordered list of models a memory consolidation may summarize with: the
/// Settings summarization list (`summarization_priority`) in its order, else
/// the chat models in their configured order.
///
/// The one rule for all three memory consumers — the scheduled dream cycle,
/// IPC consolidation and the `memory.summarize` script service. In-loop
/// summarization answers an empty Settings list by cutting to fit, but a
/// memory fold has nothing to cut, so it needs models from somewhere, and the
/// three used to take them from three different places (the single chat
/// model twice, the whole chat priority list once). The chat fallback is the
/// agent service's `configured_models`: exactly the models a chat walks, in
/// the order it walks them.
///
/// Blank entries are not models, in either list. Pure. The result is empty
/// only when nothing at all is configured, the case callers report as such.
#[must_use]
pub fn summarization_models(
    summarization_priority: &[String],
    chat_model: &str,
    chat_priority: &[String],
) -> Vec<String> {
    let listed = crate::agent_service::named_models(summarization_priority);
    let models = if listed.is_empty() {
        crate::agent_service::configured_models(chat_model, chat_priority)
    } else {
        listed
    };
    debug_assert!(
        models.iter().all(|m| !m.trim().is_empty()),
        "a blank entry names no model"
    );
    models
}

/// [`summarization_models`] read from the agent service's live config — the
/// view the dream cycle and `memory.summarize` hold.
///
/// The three fields are read here, once, rather than at each call site: a
/// site that passed an empty chat list, or no chat model, would still compile
/// and quietly narrow the fallback, and only this mapping is under test.
#[must_use]
pub fn for_agent_service(config: &crate::agent_service::AgentServiceConfig) -> Vec<String> {
    summarization_models(
        &config.summarization_priority,
        &config.model,
        &config.model_priority,
    )
}

/// [`summarization_models`] read from the user config's `[llm]` table — the
/// view IPC consolidation holds. See [`for_agent_service`] for why the fields
/// are read here.
#[must_use]
pub fn for_llm_config(llm: &nanna_config::LlmConfig) -> Vec<String> {
    summarization_models(&llm.summarization_priority, &llm.model, &llm.model_priority)
}

/// The cluster byte budget must hold for **whichever** model actually answers.
///
/// A dream cycle builds one prompt and then walks the failover list with it, so
/// sizing the budget to the *first* model would overflow a smaller fallback.
/// Taking the **minimum** `hard_input_limit` across every candidate makes the
/// prompt safe for all of them. The cost is only that a large model consolidates
/// a little less per pass — never lost content, because a cluster that would
/// breach the bound simply re-clusters on a later seed.
pub async fn summarizer_context_window_tokens(router: &LlmRouter, models: &[String]) -> usize {
    debug_assert!(
        !models.is_empty(),
        "context window must be resolved against at least one model"
    );

    // Seed with the floor rather than `usize::MAX`: an empty list must fall back
    // to the smallest safe window, never leave the budget effectively unbounded.
    // (Both callers guarantee a non-empty list today — this keeps the bound a
    // bound if a third one ever forgets.)
    let mut smallest_tokens = MIN_SUMMARIZER_CONTEXT_TOKENS;
    for (index, model) in models.iter().enumerate() {
        let limit = router.get_model_info(model).await.hard_input_limit();
        smallest_tokens = if index == 0 {
            limit
        } else {
            smallest_tokens.min(limit)
        };
    }

    let resolved = smallest_tokens.max(MIN_SUMMARIZER_CONTEXT_TOKENS);
    debug_assert!(
        resolved >= MIN_SUMMARIZER_CONTEXT_TOKENS,
        "resolved window must respect the floor"
    );
    resolved
}

/// Waits between congested rounds when the provider does not say how long.
const BACKOFF_SECS: [u64; 6] = [5, 15, 30, 60, 120, 240];

/// Longest wait taken on a provider's own `retry_after`.
///
/// Bound justification: a dream cycle holds its cluster while it waits, and a
/// provider asking for more than ten minutes is out for the night, not
/// congested — the cycle is better off failing the cluster and letting the
/// next cycle try.
const RETRY_AFTER_MAX_SECS: u64 = 600;

/// How long to wait after congested round `round`, or `None` when that was the
/// last round and the walk should give up now.
///
/// One wait per round. The provider's `retry_after` replaces the schedule
/// rather than adding to it: the loop used to sleep the provider's hint at the
/// end of a round and then the schedule's step at the top of the next, and
/// after the final round it announced a wait and gave up anyway.
fn next_wait_secs(round: usize, provider_retry_after: Option<u64>) -> Option<u64> {
    let scheduled = *BACKOFF_SECS.get(round)?;
    let wait = provider_retry_after.map_or(scheduled, |secs| secs.min(RETRY_AFTER_MAX_SECS));
    debug_assert!(wait <= RETRY_AFTER_MAX_SECS, "every wait is bounded");
    Some(wait)
}

/// Build the `summarize_fn` a dream cycle calls, walking `models` in order and
/// returning the first answer that has text.
///
/// Failure of one model is an *expected* operational condition (down,
/// rate-limited, out of credit, or answering with no text), so it is logged
/// and the walk continues; only
/// exhausting every candidate is an error, and that error names the last real
/// failure rather than a generic message.
///
/// The returned closure is `Send + Sync` and its future is `Send`, matching the
/// bounds `nanna_memory::DreamingService` requires.
pub fn summarize_with_failover(
    router: Arc<LlmRouter>,
    models: Vec<String>,
) -> impl Fn(
    String,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String, String>> + Send>>
+ Send
+ Sync {
    debug_assert!(
        !models.is_empty(),
        "a failover summarizer needs at least one model"
    );

    move |prompt: String| {
        let router = Arc::clone(&router);
        let models = models.clone();
        Box::pin(async move {
            // Wait out congestion rather than failing the cluster.
            //
            // A dream cycle asks for one summary per cluster, back to back, and
            // a free tier allows 20 requests a minute — so the first cycle after
            // this path started working burned its whole budget in one second
            // and lost 20 of 22 clusters to 429s. Those clusters are not
            // retried later; the consolidation simply does not happen, so the
            // store keeps growing uncompressed all night.
            //
            // Congestion is a throughput limit, not a failure. Waiting spreads
            // the cycle over a few minutes, which is the correct price. Only a
            // model that is genuinely broken gets failed over.
            let mut last_error = String::from("no summarization models configured");
            let mut wait_secs = 0u64;
            for round in 0..=BACKOFF_SECS.len() {
                if wait_secs > 0 {
                    tokio::time::sleep(std::time::Duration::from_secs(wait_secs)).await;
                }

                let mut soonest_retry: Option<u64> = None;
                let mut congested = 0usize;

                for model in &models {
                    let request = nanna_llm::CompletionRequest::default()
                        .with_model(model)
                        .with_message(nanna_llm::Message::user(&prompt));
                    match router.complete(model, request).await {
                        // No text is a failed call, not a summary: a runner
                        // stuck on a stop token and a reasoning model that
                        // spent its budget thinking both answer `""`, and a
                        // consolidation that took it would store an empty
                        // memory and delete the cluster it replaces. It is
                        // final for this model, like any non-congestion
                        // failure, so the next model on the list is asked.
                        //
                        // Emptiness is the only content test here. The
                        // in-loop plausibility floor (`plausible_summary`:
                        // 64 chars, 0.1% of the source) does not transfer —
                        // an `Essence` fold asks for "one short line", which
                        // is legitimately shorter, and each memory path
                        // already guards its own result (enrichment must
                        // contain the original).
                        Ok(summary) if summary.trim().is_empty() => {
                            tracing::warn!(
                                "Dream summarization model {model} answered with no text"
                            );
                            last_error = format!("{model}: answered with no text");
                        }
                        Ok(summary) => {
                            if round > 0 {
                                tracing::info!(
                                    "Dream summarization cleared congestion after {round} wait(s)"
                                );
                            }
                            return Ok(summary);
                        }
                        Err(e) if e.is_rate_limit() => {
                            congested += 1;
                            if let nanna_llm::LlmError::RateLimit { retry_after, .. } = &e
                                && let Some(secs) = retry_after
                            {
                                soonest_retry =
                                    Some(soonest_retry.map_or(*secs, |cur: u64| cur.min(*secs)));
                            }
                            last_error = format!("{model}: {e}");
                        }
                        Err(e) => {
                            tracing::warn!("Dream summarization model {model} failed: {e}");
                            last_error = format!("{model}: {e}");
                        }
                    }
                }

                if congested == 0 {
                    break;
                }
                let Some(next) = next_wait_secs(round, soonest_retry) else {
                    break;
                };
                tracing::info!(
                    "Dream summarization congested on all {} model(s) — waiting {}s",
                    models.len(),
                    next
                );
                wait_secs = next;
            }

            Err(format!(
                "all {} summarization model(s) failed; last error — {last_error}",
                models.len()
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_congested_round_waits_once_and_the_last_round_does_not_wait() {
        assert_eq!(
            next_wait_secs(0, None),
            Some(5),
            "the schedule when the provider is silent"
        );
        assert_eq!(
            next_wait_secs(0, Some(42)),
            Some(42),
            "the provider's word replaces it"
        );
        assert_eq!(
            next_wait_secs(1, Some(86_400)),
            Some(RETRY_AFTER_MAX_SECS),
            "bounded"
        );
        assert_eq!(next_wait_secs(BACKOFF_SECS.len() - 1, None), Some(240));
        assert_eq!(
            next_wait_secs(BACKOFF_SECS.len(), Some(1)),
            None,
            "after the final round there is nothing left to wait for"
        );
    }

    fn v(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn configured_priority_is_used_verbatim() {
        // The whole list, in order — this is the fix: the scheduled cycle used
        // to take only the head and make a single attempt.
        let priority = v(&["small-local", "big-cloud"]);
        let models = summarization_models(&priority, "main-model", &v(&["chat-a", "chat-b"]));
        assert_eq!(models, priority, "order and contents must be preserved");
    }

    /// Memory cannot be "truncated" instead of summarized, so with no
    /// summarization list the three memory consumers need models from
    /// somewhere — and used to take them from three different places: the
    /// dream cycle and `memory.summarize` the single chat model, IPC
    /// consolidation the whole chat priority list. One rule now: the chat
    /// models in their configured order, exactly the list a chat walks.
    #[test]
    fn with_no_settings_list_every_memory_consumer_uses_the_chat_models_in_order() {
        assert_eq!(
            summarization_models(&[], "main", &v(&["chat-a", "chat-b"])),
            v(&["chat-a", "chat-b"]),
            "the whole chat list, in order — not only its head"
        );
        assert_eq!(
            summarization_models(&[], "main", &[]),
            v(&["main"]),
            "no chat list: the single chat model"
        );
    }

    #[test]
    fn priority_wins_over_fallback() {
        // Negative space: the fallback must not leak in when a priority exists.
        let models = summarization_models(&v(&["chosen"]), "ignored", &v(&["ignored-too"]));
        assert_eq!(models, v(&["chosen"]));
    }

    /// A blank entry is not a model, in either list: a list of blanks is an
    /// empty list, and falls back the way an empty one does.
    #[test]
    fn blank_entries_are_not_models() {
        assert_eq!(
            summarization_models(&v(&["", "  "]), "main", &[]),
            v(&["main"])
        );
        assert_eq!(
            summarization_models(&v(&["", "chosen"]), "main", &[]),
            v(&["chosen"])
        );
    }

    /// Each consumer's config view feeds all three fields: the chat list
    /// wins over the single chat model, and the single chat model stands in
    /// when there is no list — in both views.
    #[test]
    fn both_config_views_read_the_settings_list_and_both_chat_fields() {
        let mut service = crate::agent_service::AgentServiceConfig {
            model: "main".to_string(),
            model_priority: v(&["chat-a", "chat-b"]),
            summarization_priority: Vec::new(),
            ..crate::agent_service::AgentServiceConfig::default()
        };
        let mut llm = nanna_config::LlmConfig {
            model: "main".to_string(),
            model_priority: v(&["chat-a", "chat-b"]),
            summarization_priority: Vec::new(),
            ..nanna_config::LlmConfig::default()
        };
        assert_eq!(for_agent_service(&service), v(&["chat-a", "chat-b"]));
        assert_eq!(for_llm_config(&llm), v(&["chat-a", "chat-b"]));

        service.model_priority.clear();
        llm.model_priority.clear();
        assert_eq!(for_agent_service(&service), v(&["main"]));
        assert_eq!(for_llm_config(&llm), v(&["main"]));

        service.summarization_priority = v(&["ollama/small:1b"]);
        llm.summarization_priority = v(&["ollama/small:1b"]);
        assert_eq!(for_agent_service(&service), v(&["ollama/small:1b"]));
        assert_eq!(for_llm_config(&llm), v(&["ollama/small:1b"]));
    }

    /// The `/api/chat` models the fake server was asked for, in order.
    fn asked(
        seen: &std::sync::Mutex<Vec<crate::embedding_reload::test_ollama::SeenRequest>>,
    ) -> Vec<String> {
        seen.lock()
            .expect("record lock")
            .iter()
            .filter(|request| request.path == "/api/chat")
            .filter_map(|request| request.model.clone())
            .collect()
    }

    /// An empty answer is not a summary. A runner that stops at once, or a
    /// reasoning model that spends its whole budget thinking, answers with
    /// no text — and a consolidation that took that as success would fold a
    /// cluster into an empty memory and delete its sources, while the next
    /// model on the Settings list was never asked.
    #[tokio::test(flavor = "multi_thread")]
    async fn an_empty_answer_hands_the_prompt_to_the_next_model() {
        const SUMMARY: &str = "The user keeps the build green.";
        let (host, seen) = crate::embedding_reload::test_ollama::spawn_answering(&[
            ("silent:1b", ""),
            ("blank:1b", " \n\t "),
            ("answers:1b", SUMMARY),
        ])
        .await;
        let router = Arc::new(LlmRouter::new().with_ollama(&host));
        let summarize = summarize_with_failover(
            router,
            v(&["ollama/silent:1b", "ollama/blank:1b", "ollama/answers:1b"]),
        );

        let summary = summarize("fold these memories".to_string())
            .await
            .expect("the third model answers");

        assert_eq!(summary, SUMMARY);
        assert_eq!(
            asked(&seen),
            v(&["silent:1b", "blank:1b", "answers:1b"]),
            "each model is asked once, in the Settings order"
        );
    }

    /// When every model answers with nothing, the walk fails — the
    /// consolidation keeps its sources — and says which model answered
    /// empty rather than reporting a success.
    #[tokio::test(flavor = "multi_thread")]
    async fn only_empty_answers_are_a_failure_that_says_so() {
        let (host, seen) = crate::embedding_reload::test_ollama::spawn_answering(&[
            ("silent:1b", ""),
            ("blank:1b", "   "),
        ])
        .await;
        let router = Arc::new(LlmRouter::new().with_ollama(&host));
        let summarize =
            summarize_with_failover(router, v(&["ollama/silent:1b", "ollama/blank:1b"]));

        let error = summarize("fold these memories".to_string())
            .await
            .expect_err("no model gave a summary");

        assert!(
            error.contains("ollama/blank:1b") && error.contains("no text"),
            "names the last model and why: {error}"
        );
        assert_eq!(
            asked(&seen),
            v(&["silent:1b", "blank:1b"]),
            "an empty answer is final for its model — no congestion retry"
        );
    }

    #[test]
    fn empty_only_when_nothing_is_configured() {
        // The single case callers must report as unconfigured.
        assert_eq!(summarization_models(&[], "", &[]), Vec::<String>::new());
        // …and never otherwise.
        assert_ne!(summarization_models(&v(&["a"]), "", &[]), Vec::<String>::new());
        assert_ne!(summarization_models(&[], "b", &[]), Vec::<String>::new());
        assert_ne!(summarization_models(&[], "", &v(&["c"])), Vec::<String>::new());
    }
}
