//! `server` command and the legacy daemon-mode entry point.

use crate::setup::{create_scheduler, init_components, provider_chat_key};
use nanna_config::Config;
use nanna_core::{LlmClient, Nanna, NannaConfig};
use nanna_daemon::log_buffer::LogBuffer;
use nanna_daemon::server::DaemonBuilder;
use nanna_server::{AppStateBuilder, ServerConfig, start_server};
use tracing::{debug, info, warn};

/// The port `nanna server` listens on: the `--port` flag when given, otherwise
/// `[server].port` from the config — which the `PORT` env override and the
/// onboarding "Server port" answer both write, and which defaults to 3000.
///
/// Until 2026-09-11 the flag defaulted to the literal 3000 and the config key
/// was read by nothing, so the port a user chose during onboarding, the README's
/// documented `[server] port = …`, and `PORT` were all silently ignored. Pure,
/// so the precedence is pinned by a test rather than argued.
#[must_use]
pub fn server_port(flag: Option<u16>, config: &Config) -> u16 {
    let port = flag.unwrap_or(config.server.port);
    debug_assert!(
        flag.is_none_or(|f| f == port),
        "an explicit flag always wins"
    );
    debug_assert!(
        flag.is_some() || port == config.server.port,
        "…else the config decides"
    );
    port
}

/// The key `nanna serve`'s bot is built with: the chat client's
/// ([`provider_chat_key`]), so the two never run on different keys.
///
/// This read `[llm].api_key`, Anthropic's key, for every provider. Once
/// `nanna init` filed each key under its own provider (2026-09-21), an
/// `OpenAI` or `OpenRouter` bot found no key after the chat client beside it had
/// started, or was built with the Anthropic key.
fn bot_api_key(
    config: &Config,
    read_env: impl Fn(&str) -> Option<String>,
) -> anyhow::Result<String> {
    provider_chat_key(config, read_env)
}

/// A channel's bot token: the one its config holds, else `var` from `env`;
/// `None` when neither holds one that is not blank.
///
/// A configured channel's token is filled at load from the environment or the
/// secure store (it is never in `config.toml`), so a section whose token is in
/// neither holds an empty one — no token, not a bot to start with `""`.
fn channel_token(
    configured: Option<&str>,
    var: &str,
    env: impl Fn(&str) -> Option<String>,
) -> Option<String> {
    let set = |token: &String| !token.trim().is_empty();
    configured
        .map(str::to_owned)
        .filter(set)
        .or_else(|| env(var).filter(set))
}

/// The [`Nanna`] instance `nanna serve` keeps for backwards compatibility,
/// on the provider `config` names.
///
/// # Errors
///
/// When the provider has no key ([`bot_api_key`]), and whatever
/// [`Nanna::new`] reports for the built client.
async fn build_bot(config: &Config) -> anyhow::Result<Nanna> {
    let api_key = bot_api_key(config, |name| std::env::var(name).ok())?;

    // Create Nanna bot instance for backwards compatibility
    let bot_config = NannaConfig {
        name: config.general.name.clone(),
        default_model: config.llm.model.clone(),
        max_context_messages: 20,
        enable_gpu: true,
    };

    let bot_llm = match config.llm.provider.as_str() {
        "openai" => LlmClient::openai(&api_key),
        "openrouter" => LlmClient::openrouter(&api_key),
        _ => LlmClient::anthropic(&api_key), // anthropic or unknown
    };

    let bot = Nanna::new(bot_config, bot_llm).await?;

    if bot.has_gpu() {
        info!("GPU acceleration enabled");
    } else {
        info!("CPU-only mode (SIMD active)");
    }

    Ok(bot)
}

/// Run the HTTP server
pub async fn run_server(config: &Config, host: String, port: u16) -> anyhow::Result<()> {
    let (llm, tools, storage) = init_components(config).await?;

    let bot = build_bot(config).await?;

    // Get Telegram token from config or environment
    let telegram_token = channel_token(
        config
            .channels
            .telegram
            .as_ref()
            .map(|t| t.bot_token.as_str()),
        "TELEGRAM_BOT_TOKEN",
        |var| std::env::var(var).ok(),
    );

    if telegram_token.is_some() {
        info!("Telegram channel enabled");
    }

    // Get Discord config
    let discord_bot_token = channel_token(
        config
            .channels
            .discord
            .as_ref()
            .map(|d| d.bot_token.as_str()),
        "DISCORD_BOT_TOKEN",
        |var| std::env::var(var).ok(),
    );

    let discord_app_id = config
        .channels
        .discord
        .as_ref()
        .map(|d| d.application_id.clone())
        .or_else(|| std::env::var("DISCORD_APP_ID").ok());

    let discord_public_key = config
        .channels
        .discord
        .as_ref()
        .map(|d| d.public_key.clone());

    if discord_bot_token.is_some() && discord_app_id.is_some() {
        info!("Discord channel enabled");
    }

    // Build app state - pass Arcs directly
    let state = AppStateBuilder::new()
        .bot(bot)
        .storage_arc(storage.clone())
        .llm_arc(llm.clone())
        .tools_arc(tools.clone())
        .webhook_secret(config.server.webhook_secret.clone())
        .discord_public_key(discord_public_key)
        // Slack's signing secret was parsed and then never handed to the
        // server, so the `nanna serve` Slack webhook verified nothing. Now the
        // handler fails closed without it, so the wiring is load-bearing.
        .slack_signing_secret(
            config
                .channels
                .slack
                .as_ref()
                .map(|s| s.signing_secret.clone()),
        )
        .telegram_webhook_secret(
            config
                .channels
                .telegram
                .as_ref()
                .and_then(|t| t.webhook_secret.clone()),
        )
        .signal_webhook_secret(
            config
                .channels
                .signal
                .as_ref()
                .and_then(|s| s.webhook_secret.clone()),
        )
        .default_model(config.llm.model.clone())
        .telegram_token(telegram_token)
        .discord_config(discord_bot_token, discord_app_id)
        // ONE scheduler per process over the persisted job table. The server
        // state's own would run beside the one below, load the same jobs, and
        // "complete" every non-dreaming one by echoing its payload — so a
        // one-shot could be marked done, and deleted, without ever running.
        .scheduler(false)
        .build();

    // The scheduler for heartbeats, jobs and — through the server's dreaming
    // runtime — memory consolidation.
    let mut scheduler = start_scheduler(
        config,
        llm.clone(),
        tools.clone(),
        storage.clone(),
        state.dreaming.clone(),
    )
    .await;

    let server_config = ServerConfig {
        host: host.clone(),
        port,
        webhook_secret: config.server.webhook_secret.clone(),
    };

    info!("Server listening on {}:{}", host, port);
    start_server(server_config, state).await?;

    // Clean shutdown
    scheduler.stop().await;

    Ok(())
}

/// Load, top up and start the one scheduler `nanna serve` runs.
async fn start_scheduler(
    config: &Config,
    llm: std::sync::Arc<LlmClient>,
    tools: std::sync::Arc<nanna_tools::ToolRegistry>,
    storage: std::sync::Arc<nanna_storage::Storage>,
    dreaming: Option<std::sync::Arc<nanna_core::DreamingRuntime>>,
) -> nanna_core::Scheduler {
    let has_dreaming = dreaming.is_some();
    let mut scheduler = create_scheduler(config, llm, tools, storage, dreaming);

    // Load persisted cron jobs
    match scheduler.load_jobs().await {
        Ok(count) if count > 0 => info!("Loaded {} persisted cron jobs", count),
        Ok(_) => debug!("No persisted cron jobs found"),
        Err(e) => warn!("Failed to load cron jobs: {}", e),
    }
    // The server state used to register consolidation on its own scheduler.
    if has_dreaming
        && !scheduler
            .has_task_named(nanna_core::DREAMING_TASK_NAME)
            .await
    {
        scheduler
            .add_task(nanna_core::consolidation_task(None))
            .await;
        info!("Scheduled memory consolidation (every 1 hour)");
    }

    scheduler.start();
    info!("Scheduler started");
    scheduler
}

/// Run the daemon server (background mode)
///
/// A full daemon under the `nanna` executable's name, on the same PID file,
/// store and port as `nanna-daemon`. The single-instance probe recognizes it
/// by `nanna_daemon::health::DAEMON_MODE_FLAG` on its command line.
///
/// It is built and stopped the way `nanna-daemon run` (the GUI's sidecar) is:
/// configured by [`DaemonBuilder::from_nanna_config`] and stopped by the
/// signal handlers in [`nanna_daemon::shutdown`]. Until 2026-09-21 this path
/// hand-built its `DaemonConfig`, and the copy had drifted. It ignored
/// `[general] data_dir`, the model chain and routing
/// (`AgentServiceConfig::default()`), the channels, the Brave key and the
/// legacy `memories.json`. It had no signal handler, so `nanna daemon stop`
/// killed it without a drain and left its exit record reading `running`.
///
/// Its config file is the default one, as for `nanna-daemon`:
/// `NANNA_CONFIG_PATH`, else the platform config directory. `nanna --config
/// <file> daemon start` sets that variable for it. `log_buffer` holds the
/// lines `system.logs` serves (the GUI's Logs page).
///
/// # Errors
///
/// When a signal handler cannot be installed, or when
/// [`nanna_daemon::DaemonServer::run`] fails (another daemon holds the
/// instance, the port is taken).
pub async fn run_daemon(host: String, port: u16, log_buffer: LogBuffer) -> anyhow::Result<()> {
    let mut daemon = DaemonBuilder::from_nanna_config()?
        .with_host(host)
        .with_port(port)
        .with_log_buffer(log_buffer)
        .build();

    nanna_daemon::shutdown::install_signal_handlers(&daemon)?;

    daemon.run().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_env(_: &str) -> Option<String> {
        None
    }

    /// `nanna serve`'s bot runs on its provider's own key — where `nanna init`
    /// stores it since 2026-09-21 — like the chat client beside it, and never
    /// on the Anthropic key.
    #[test]
    fn the_bot_gets_its_providers_own_key() {
        for (provider, env_var) in [
            ("openai", "OPENAI_API_KEY"),
            ("openrouter", "OPENROUTER_API_KEY"),
        ] {
            let mut config = Config::default();
            config.llm.provider = provider.to_string();
            config.llm.api_key = Some("sk-ant-api03-anthropic".to_string());
            *config.llm.provider_api_key_mut() = Some("own-key".to_string());
            assert_eq!(
                bot_api_key(&config, no_env).expect(provider),
                "own-key",
                "{provider}"
            );

            *config.llm.provider_api_key_mut() = None;
            let error =
                bot_api_key(&config, no_env).expect_err("the Anthropic key is not this provider's");
            assert!(error.to_string().contains(env_var), "{provider}: {error}");
        }
    }

    /// The port a user chose during onboarding — written to `[server].port`,
    /// as is `PORT` — is the one `nanna server` listens on when no `--port`
    /// is given. Before 2026-09-11 it was silently ignored.
    #[test]
    fn server_port_falls_back_to_the_configured_port() {
        let mut config = Config::default();
        config.server.port = 4100;
        assert_eq!(server_port(None, &config), 4100);
    }

    /// An explicit `--port` still wins over the config.
    #[test]
    fn an_explicit_port_flag_overrides_the_config() {
        let mut config = Config::default();
        config.server.port = 4100;
        assert_eq!(server_port(Some(8080), &config), 8080);
    }

    /// With nothing configured the listening port is what it always was.
    #[test]
    fn the_default_port_is_unchanged() {
        assert_eq!(server_port(None, &Config::default()), 3000);
    }

    /// A channel section whose token is in neither the secure store nor the
    /// environment loads with an empty one. That is no token — not a bot to
    /// start with `""` — and the environment still gets its say.
    #[test]
    fn an_empty_channel_token_is_no_token() {
        let var = "TELEGRAM_BOT_TOKEN";
        let no_env = |_: &str| None;
        let env = |name: &str| (name == var).then(|| "env-token".to_string());
        assert_eq!(channel_token(Some(""), var, no_env), None);
        assert_eq!(channel_token(Some("  "), var, no_env), None);
        assert_eq!(channel_token(None, var, no_env), None);
        assert_eq!(
            channel_token(Some(""), var, env).as_deref(),
            Some("env-token")
        );
        assert_eq!(channel_token(None, var, env).as_deref(), Some("env-token"));
        assert_eq!(
            channel_token(Some("config-token"), var, env).as_deref(),
            Some("config-token")
        );
    }
}
