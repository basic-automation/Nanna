//! System handlers for the [`ControlPlane`].

use super::{json, info, ControlPlane, SystemAction, Value};
use nanna_config::same_ollama_server;

impl ControlPlane {
    /// Gather everything `GET /metrics` reports, in one pass.
    pub async fn metrics_snapshot(&self) -> crate::metrics::MetricsSnapshot {
        let memory_entries = match self.memory {
            Some(ref memory) => Some(memory.count().await),
            None => None,
        };
        let reminders_pending = match self.scheduler {
            Some(ref scheduler) => {
                let tasks = scheduler.read().await.list_tasks().await;
                let pending = tasks.iter().filter(|task| {
                    task.name == crate::reminder_service::REMINDER_TASK_NAME && task.enabled
                });
                Some(pending.count())
            }
            None => None,
        };
        let mcp_servers = match self.mcp_status {
            Some(ref status) => status.read().await.clone(),
            None => Vec::new(),
        };
        crate::metrics::MetricsSnapshot {
            uptime_secs: self.uptime_secs(),
            sessions: self.sessions.count().await,
            chat_runs_active: self.chat_runs.active_count().await,
            memory_entries,
            reminders_pending,
            tools: self.tool_stats.summaries().await,
            models: self.model_stats.summaries().await,
            mcp_servers,
            channels: self.channel_counters.snapshot(),
        }
    }

    // =========================================================================
    // System Handlers
    // =========================================================================
    
    pub(super) async fn handle_system(&self, _client_id: &str, action: SystemAction) -> Value {
        match action {
            SystemAction::Status => self.system_status().await,
            SystemAction::Restart => {
                info!("Restart requested");
                json!({ "status": "restarting" })
            }
            SystemAction::Shutdown => self.system_shutdown(),
            SystemAction::Version => {
                json!({
                    "version": env!("CARGO_PKG_VERSION"),
                    "name": "nanna-daemon",
                    "rust_version": env!("CARGO_PKG_RUST_VERSION"),
                })
            }
            SystemAction::CheckUpdate => {
                json!({ "update_available": false })
            }
            SystemAction::Update => {
                json!({ "error": "not_implemented" })
            }
            SystemAction::Logs { lines, level } => self.system_logs(lines, level.as_deref()),
            SystemAction::Health => self.system_health(),
            SystemAction::ModelStats => self.system_model_stats().await,
            SystemAction::ToolStats => {
                let summaries = self.tool_stats.summaries().await;
                json!({
                    "tools": summaries,
                })
            }
            SystemAction::GlobalStats => {
                let global = self.tool_stats.global_stats().await;
                json!(global)
            }
            SystemAction::ToolStatsHourly { tool_name, hours } => {
                if let Some(ref storage) = self.storage {
                    match storage.get_tool_stats_hourly(tool_name.as_deref(), hours.unwrap_or(24)).await {
                        Ok(data) => json!({ "buckets": data }),
                        Err(e) => json!({ "error": e.to_string() }),
                    }
                } else {
                    json!({ "buckets": [], "error": "Storage not available" })
                }
            }
            SystemAction::CostRollup { days, by } => self.system_cost_rollup(days, by).await,
            SystemAction::ToolStatsDaily { tool_name, days } => {
                if let Some(ref storage) = self.storage {
                    match storage.get_tool_stats_daily(tool_name.as_deref(), days.unwrap_or(30)).await {
                        Ok(data) => json!({ "buckets": data }),
                        Err(e) => json!({ "error": e.to_string() }),
                    }
                } else {
                    json!({ "buckets": [], "error": "Storage not available" })
                }
            }
            SystemAction::ToolCallLog { tool_name, limit } => {
                if let Some(ref storage) = self.storage {
                    match storage.get_tool_call_log(tool_name.as_deref(), limit.unwrap_or(50)).await {
                        Ok(entries) => json!({ "entries": entries }),
                        Err(e) => json!({ "error": e.to_string() }),
                    }
                } else {
                    json!({ "entries": [], "error": "Storage not available" })
                }
            }
            SystemAction::ProbeOllama { base_url, models } => {
                // Resolved under the read lock, which is released before the
                // probe's network round-trip.
                let (base_url, models, bearer) =
                    ollama_probe_target(&*self.config.read().await, base_url.as_deref(), models);
                let probe =
                    nanna_llm::probe_ollama(&base_url, bearer.as_deref(), OLLAMA_PROBE_TIMEOUT).await;
                ollama_probe_report(&base_url, &probe, &models)
            }
            SystemAction::ValidateApiKey { provider, key } => {
                // The key is used for this one request and dropped; it is
                // never written anywhere, and `SecretInput` keeps it out of
                // the IPC layer's request log.
                crate::validate_api_key::handle_validate_api_key(&provider, key.expose()).await
            }
        }
    }

    /// `SystemAction::Status`: the daemon's live service and store health.
    async fn system_status(&self) -> Value {
        let memory_stats = if let Some(ref memory) = self.memory {
            let stats = memory.stats().await;
            Some(json!({
                "total": stats.total,
                "active": stats.active,
            }))
        } else {
            None
        };
        
        let tool_count = if let Some(ref tools) = self.tools {
            Some(tools.definitions().await.len())
        } else {
            None
        };
        
        let workspace_count = self.workspaces.read().await.len();
        let scheduler_available = self.scheduler.is_some();

        // The router's live provider set. This is what actually routes
        // a chat request — the GUI model picker gates on this instead
        // of its own login state, so the two can't split-brain about
        // which providers exist.
        let llm_providers: Vec<&'static str> = self
            .router
            .as_ref()
            .map(|r| {
                r.available_providers_sorted()
                    .into_iter()
                    .map(crate::llm_router::ProviderId::name)
                    .collect()
            })
            .unwrap_or_default();

        // Durable-store health: a corrupt row skipped on load leaves the
        // store degraded — surface it rather than a silent empty store.
        let (memory_degraded, memory_corrupt_rows) = if let Some(ref m) = self.memory {
            let h = m.store_health().await;
            (h.degraded, h.corrupt_rows)
        } else {
            (false, 0)
        };

        // Startup quarantine + rebuild (page-level corruption): keep it
        // on /status for the daemon's lifetime — clients that connect
        // after boot never saw the MemoryStoreRebuilt event.
        let (memory_rebuilt, memory_recovered_rows, memory_expected_rows) = self
            .memory_recovery
            .as_ref()
            .map_or((false, 0, None), |r| {
                (true, r.memories_recovered, r.memories_expected)
            });

        json!({
            "status": "running",
            "version": env!("CARGO_PKG_VERSION"),
            "uptime_secs": self.uptime_secs(),
            "sessions": self.sessions.count().await,
            "workspaces": workspace_count,
            "agent_available": self.agent.is_some(),
            "memory_available": self.memory.is_some(),
            "memory_degraded": memory_degraded,
            "memory_corrupt_rows": memory_corrupt_rows,
            "memory_rebuilt": memory_rebuilt,
            "memory_recovered_rows": memory_recovered_rows,
            "memory_expected_rows": memory_expected_rows,
            "memory_stats": memory_stats,
            "tools_available": self.tools.is_some(),
            "tool_count": tool_count,
            "scheduler_available": scheduler_available,
            "llm_providers": llm_providers,
            // Each configured MCP server: starting / started (with its
            // tool count) / failed or not_started (with the reason).
            // A failed server's tools are simply absent otherwise.
            "mcp_servers": match self.mcp_status {
                Some(ref status) => json!(*status.read().await),
                None => json!([]),
            },
            "config_path": self.config_path,
        })
    }

    /// `SystemAction::Shutdown`: signal the shutdown handle, if one is wired.
    fn system_shutdown(&self) -> Value {
        info!("Shutdown requested");
        self.shutdown_tx.clone().map_or_else(
            // Minimal construction with no shutdown handle: say so
            // instead of pretending — callers fall back to a tree-kill.
            || json!({ "status": "unsupported", "error": "no shutdown handle wired" }),
            |tx| {
                // Fire after a short grace so the response below flushes to
                // the requesting client before the IPC server tears down.
                tokio::spawn(async move {
                    tokio::time::sleep(std::time::Duration::from_millis(250)).await;
                    let _ = tx.send(());
                });
                json!({ "status": "shutting_down" })
            },
        )
    }

    /// `SystemAction::Logs`: the tail of the in-memory log buffer, by level.
    fn system_logs(&self, lines: Option<usize>, level: Option<&str>) -> Value {
        self.log_buffer.as_ref().map_or_else(
            || json!({ "logs": [], "message": "Log buffer not available" }),
            |buf| {
                let entries = buf.get_recent(lines.unwrap_or(1000));
                // Filter by level if specified
                let filtered: Vec<_> = if let Some(lvl) = level {
                    let lvl = lvl.to_lowercase();
                    entries.into_iter().filter(|e| e.level == lvl).collect()
                } else {
                    entries
                };
                json!({ "logs": filtered })
            },
        )
    }

    /// `SystemAction::CostRollup`: priced request-log usage by day, month or
    /// session.
    async fn system_cost_rollup(&self, days: Option<u32>, by: Option<String>) -> Value {
        let Some(ref storage) = self.storage else {
            return json!({ "error": "storage_unavailable", "message": "Cost rollups need the request log in storage" });
        };
        let period = match by.as_deref() {
            None | Some("day") => nanna_storage::UsagePeriod::Day,
            Some("month") => nanna_storage::UsagePeriod::Month,
            Some("session") => nanna_storage::UsagePeriod::Session,
            Some(other) => {
                return json!({ "error": "invalid_period", "message": format!("`by` must be \"day\", \"month\" or \"session\" (got {other:?})") });
            }
        };
        match storage.model_usage_by(days.unwrap_or(30), period).await {
            Ok(usage) => json!(crate::cost_rollup::price_buckets(usage)),
            Err(e) => json!({ "error": "rollup_failed", "message": e.to_string() }),
        }
    }

    /// `SystemAction::Health`: per-service availability checks.
    fn system_health(&self) -> Value {
        let memory_ok = self.memory.is_some();
        let tools_ok = self.tools.is_some();
        let agent_ok = self.agent.is_some();
        let scheduler_ok = self.scheduler.is_some();
        let all_ok = agent_ok; // Agent is the critical service

        json!({
            "healthy": all_ok,
            "checks": {
                "sessions": "ok",
                "agent": if agent_ok { "ok" } else { "unavailable" },
                "memory": if memory_ok { "ok" } else { "unavailable" },
                "tools": if tools_ok { "ok" } else { "unavailable" },
                "scheduler": if scheduler_ok { "ok" } else { "unavailable" },
                "config": "ok",
                "workspaces": "ok",
            }
        })
    }

    /// `SystemAction::ModelStats`: per-model request stats and estimated spend.
    async fn system_model_stats(&self) -> Value {
        let summaries = self.model_stats.summaries().await;
        // Estimated USD spend per model (reference list prices); local
        // models come back flagged unpriced rather than as $0.
        let costs = self.model_stats.cost_report().await;
        let total_cost_usd = self.model_stats.total_cost_usd().await;
        json!({
            "models": summaries,
            "costs": costs,
            "total_cost_usd": total_cost_usd,
        })
    }
}

/// How long one Ollama probe may take, connect and answer together. A local
/// server answers `/api/tags` in milliseconds and a LAN one in tens; 3 s is
/// past both with room, and a dead host cannot stall the onboarding wizard
/// for longer than that. Same figure `nanna doctor --online` uses.
const OLLAMA_PROBE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(3);

/// What one `probe_ollama` checks: the server (`base_url`, else the configured
/// `[memory].ollama_host`), the models (as asked, else every Ollama model the
/// config names), and the bearer token to send.
///
/// The token is the configured server's, sent as chat sends it — a server
/// behind an authenticating proxy otherwise reads as down to the very check
/// meant to find it. Only to that server: a probe of any other URL gets none.
fn ollama_probe_target(
    config: &nanna_config::Config,
    base_url: Option<&str>,
    models: Vec<String>,
) -> (String, Vec<String>, Option<String>) {
    let base_url = base_url
        .map(str::trim)
        .filter(|url| !url.is_empty())
        .map_or_else(|| config.memory.ollama_host.clone(), str::to_string);
    let models = if models.is_empty() {
        configured_ollama_models(config)
    } else {
        models
    };
    let bearer = same_ollama_server(&base_url, &config.memory.ollama_host)
        .then(|| config.llm.ollama_api_key.clone())
        .flatten();
    (base_url, models, bearer)
}

/// Every Ollama model the configuration names — the chat model when the
/// provider is Ollama, the chat priority list, and the embedding priority
/// list — as the tags Ollama lists them under, deduplicated.
fn configured_ollama_models(config: &nanna_config::Config) -> Vec<String> {
    let mut wanted: Vec<String> = Vec::new();
    let mut push = |spec: &str| {
        if is_ollama_model_spec(spec) {
            let tag = ollama_tag(spec);
            if !wanted.iter().any(|have| have.eq_ignore_ascii_case(&tag)) {
                wanted.push(tag);
            }
        }
    };
    if config.llm.provider.eq_ignore_ascii_case("ollama") {
        push(&config.llm.model);
    }
    for spec in config
        .llm
        .model_priority
        .iter()
        .chain(&config.memory.embedding_priority)
    {
        push(spec);
    }
    wanted
}

/// Does a model spec name an Ollama model? `ollama/<model>`, or a bare
/// `name:tag` with no provider prefix (how Ollama ids look, and how the
/// router detects them).
fn is_ollama_model_spec(model: &str) -> bool {
    let model = model.trim();
    model.starts_with("ollama/") || (!model.contains('/') && model.contains(':'))
}

/// A model as the tag Ollama lists it under: no `ollama/` prefix, and
/// `:latest` when no tag is given — Ollama's own default.
fn ollama_tag(model: &str) -> String {
    let model = model.trim();
    let model = match model.split_once('/') {
        Some((prefix, rest)) if prefix.eq_ignore_ascii_case("ollama") => rest,
        _ => model,
    };
    let name = model.rsplit('/').next().unwrap_or(model);
    if name.contains(':') {
        model.to_string()
    } else {
        format!("{model}:latest")
    }
}

/// The wire shape of a `system.probe_ollama` answer. Pure, so the two
/// questions it must keep apart can be tested without a socket:
///
/// * `reachable: false` + `reason` — nothing usable answered (`missing` is
///   empty here on purpose: a dead server has told us nothing about models,
///   and listing every wanted model as "missing" would read as a pull problem).
/// * `reachable: true` + `missing` — the server answered and these wanted
///   models are not installed; each carries the `ollama pull` that fixes it.
///
/// `wanted` is echoed back normalized (`name:tag`) so a client can show what
/// was actually compared. `models` carries each installed model as
/// `{ name, size_bytes }` — the GUI's model picker shows sizes, and this
/// action is what it lists from.
fn ollama_probe_report(
    base_url: &str,
    probe: &nanna_llm::OllamaProbe,
    wanted: &[String],
) -> Value {
    let wanted: Vec<String> = wanted.iter().map(|m| ollama_tag(m)).collect();
    match probe {
        nanna_llm::OllamaProbe::Unreachable { reason } => json!({
            "base_url": base_url,
            "reachable": false,
            "reason": reason,
            "models": [],
            "wanted": wanted,
            "missing": [],
        }),
        nanna_llm::OllamaProbe::Reachable { models } => {
            // Compared with Ollama's default tag supplied on BOTH sides: Ollama
            // lists `name:latest`, but an Ollama-compatible server may list the
            // bare name (mummu's shim lists `qwen3.8-27b-ud-q4ks`), and that
            // model is installed — not missing.
            let missing: Vec<Value> = wanted
                .iter()
                .filter(|want| {
                    !models
                        .iter()
                        .any(|have| ollama_tag(&have.name).eq_ignore_ascii_case(want))
                })
                .map(|name| json!({ "name": name, "pull": format!("ollama pull {name}") }))
                .collect();
            let models: Vec<Value> = models
                .iter()
                .map(|m| json!({ "name": m.name, "size_bytes": m.size_bytes }))
                .collect();
            json!({
                "base_url": base_url,
                "reachable": true,
                "models": models,
                "wanted": wanted,
                "missing": missing,
            })
        }
    }
}

#[cfg(test)]
mod ollama_probe_tests {
    use super::*;
    use crate::protocol::Action;
    use nanna_llm::OllamaProbe;

    fn wanted(models: &[&str]) -> Vec<String> {
        models.iter().map(|m| (*m).to_string()).collect()
    }

    fn installed(models: &[(&str, u64)]) -> OllamaProbe {
        OllamaProbe::Reachable {
            models: models
                .iter()
                .map(|(name, size_bytes)| nanna_llm::OllamaModel {
                    name: (*name).to_string(),
                    size_bytes: *size_bytes,
                })
                .collect(),
        }
    }

    #[test]
    fn a_dead_server_is_unreachable_and_claims_nothing_about_models() {
        let probe = OllamaProbe::Unreachable {
            reason: "connection refused".to_string(),
        };
        let report = ollama_probe_report("http://localhost:11434", &probe, &wanted(&["qwen3.5:9b"]));
        assert_eq!(report["reachable"], json!(false));
        assert_eq!(report["reason"], json!("connection refused"));
        assert_eq!(report["models"], json!([]));
        // Down is not "every model is missing": the two must stay distinguishable.
        assert_eq!(report["missing"], json!([]));
        assert_eq!(report["wanted"], json!(["qwen3.5:9b"]));
    }

    #[test]
    fn a_live_server_names_each_missing_model_with_its_pull_command() {
        let probe = installed(&[("qwen3.5:9b", 6_000_000_000), ("nomic-embed-text:latest", 274_000_000)]);
        let report = ollama_probe_report(
            "http://localhost:11434",
            &probe,
            &wanted(&["ollama/qwen3.5:9b", "nomic-embed-text", "gemma4:12b"]),
        );
        assert_eq!(report["reachable"], json!(true));
        assert!(report.get("reason").is_none(), "a reachable server has no failure reason");
        // Sizes ride along: the GUI picker lists from this answer.
        assert_eq!(
            report["models"],
            json!([
                { "name": "qwen3.5:9b", "size_bytes": 6_000_000_000_u64 },
                { "name": "nomic-embed-text:latest", "size_bytes": 274_000_000_u64 }
            ])
        );
        // `ollama/` stripped, `:latest` supplied — compared as Ollama lists them.
        assert_eq!(
            report["wanted"],
            json!(["qwen3.5:9b", "nomic-embed-text:latest", "gemma4:12b"])
        );
        assert_eq!(
            report["missing"],
            json!([{ "name": "gemma4:12b", "pull": "ollama pull gemma4:12b" }])
        );
    }

    #[test]
    fn a_live_server_with_every_wanted_model_reports_nothing_missing() {
        let probe = installed(&[("QWEN3.5:9b", 0)]);
        let report = ollama_probe_report("http://gpu-box:11434", &probe, &wanted(&["qwen3.5:9b"]));
        assert_eq!(report["reachable"], json!(true));
        assert_eq!(report["missing"], json!([]), "tag comparison is case-insensitive");
        assert_eq!(report["base_url"], json!("http://gpu-box:11434"));
    }

    #[test]
    fn configured_ollama_models_collects_chat_priority_and_embeddings_once() {
        let mut config = nanna_config::Config::default();
        config.llm.provider = "ollama".to_string();
        config.llm.model = "qwen3.5:9b".to_string();
        config.llm.model_priority = vec![
            "ollama/qwen3.5:9b".to_string(),
            "anthropic/claude-sonnet-5".to_string(),
            "gemma4:12b".to_string(),
        ];
        config.memory.embedding_priority = vec!["ollama/nomic-embed-text".to_string()];
        assert_eq!(
            configured_ollama_models(&config),
            vec!["qwen3.5:9b", "gemma4:12b", "nomic-embed-text:latest"],
            "cloud specs are skipped, duplicates folded, tags normalized"
        );
    }

    #[test]
    fn a_server_listing_bare_names_has_them_installed() {
        // mummu's Ollama shim lists names without a tag; Ollama's default tag
        // is implied on both sides, so the configured model is not "missing".
        let probe = installed(&[("qwen3.8-27b-ud-q4ks", 15_358_213_024)]);
        let report =
            ollama_probe_report("https://host/ollama", &probe, &wanted(&["ollama/qwen3.8-27b-ud-q4ks"]));
        assert_eq!(report["missing"], json!([]), "{report}");
    }

    #[test]
    fn the_token_goes_only_to_the_configured_server() {
        assert!(same_ollama_server("https://Host/ollama/", "https://host/ollama"));
        assert!(same_ollama_server("http://localhost:11434", " http://127.0.0.1:11434/ "));
        assert!(!same_ollama_server("https://host/ollama", "https://host/api"));
        assert!(!same_ollama_server("https://evil.example", "https://host/ollama"));
        // A scheme's default port is the same server written or not, and
        // `[::1]` is this machine as much as `localhost` is.
        assert!(same_ollama_server("https://host:443/ollama", "https://host/ollama"));
        assert!(same_ollama_server("http://[::1]:11434", "http://localhost:11434"));
        assert!(!same_ollama_server("https://host:8443/ollama", "https://host/ollama"));
    }

    #[test]
    fn a_probe_with_no_address_checks_the_configured_server_with_its_token() {
        let mut config = nanna_config::Config::default();
        config.memory.ollama_host = "https://host/ollama".to_string();
        config.llm.ollama_api_key = Some("s3cret".to_string());

        // The GUI's model picker sends none: the daemon's own address is
        // current, the GUI's copy can be stale after a hand edit.
        let (url, _, bearer) = ollama_probe_target(&config, None, Vec::new());
        assert_eq!(url, "https://host/ollama");
        assert_eq!(bearer.as_deref(), Some("s3cret"));

        let (_, _, bearer) = ollama_probe_target(&config, Some(" https://HOST:443/ollama/ "), Vec::new());
        assert_eq!(bearer.as_deref(), Some("s3cret"), "the same server, spelled differently");

        let (url, _, bearer) = ollama_probe_target(&config, Some("https://elsewhere.example"), Vec::new());
        assert_eq!(url, "https://elsewhere.example");
        assert_eq!(bearer, None, "another server never gets the token");
    }

    #[test]
    fn a_null_address_on_the_wire_means_the_configured_server() {
        // What `get_ollama_models` sends: `"base_url": null`, not an absent key.
        let action: Action = serde_json::from_value(json!({
            "type": "system", "action": "probe_ollama", "base_url": null, "models": []
        }))
        .expect("parses");
        assert!(
            matches!(action, Action::System(SystemAction::ProbeOllama { base_url: None, .. })),
            "{action:?}"
        );
    }

    #[test]
    fn a_non_ollama_chat_provider_contributes_no_chat_model() {
        let mut config = nanna_config::Config::default();
        config.llm.provider = "anthropic".to_string();
        config.llm.model = "claude-sonnet-5".to_string();
        config.llm.model_priority.clear();
        config.memory.embedding_priority.clear();
        assert_eq!(configured_ollama_models(&config), Vec::<String>::new());
    }

    #[test]
    fn the_probe_action_parses_with_both_fields_absent() {
        let raw = json!({ "type": "system", "action": "probe_ollama" });
        match serde_json::from_value::<Action>(raw).expect("must parse") {
            Action::System(SystemAction::ProbeOllama { base_url, models }) => {
                assert_eq!(base_url, None);
                assert_eq!(models, Vec::<String>::new());
            }
            other => panic!("expected probe_ollama, got {other:?}"),
        }
    }
}
