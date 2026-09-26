//! Interactive chat, one-shot prompt, and session listing commands.

use crate::setup::init_components;
use nanna_agent::{Agent, AgentConfig, AgentContext, RunOptions, Workspace};
use nanna_config::Config;
use nanna_daemon::llm_router::{AnthropicCredential, LlmRouter, ProviderCredentials};
use nanna_storage::{Storage, StorageConfig};
use std::io::{self, BufRead, Write};
use std::sync::Arc;
use tracing::{debug, info, warn};

const BANNER: &str = r"
         🌙
        /|\
       / | \
      /  |  \
     /   |   \
    /____|____\
       NANNA
";

/// Build the system prompt for CLI mode.
fn build_cli_system_prompt(cwd: &std::path::Path, workspace: Option<&Workspace>) -> String {
    let base = format!(
        r"You are Nanna — moon god of the digital realm.

You have tools at your disposal:
- exec: Execute shell commands
- read_file: Read file contents  
- write_file: Write content to files
- list_dir: List directory contents
- web_fetch: Fetch content from URLs

Current directory: {}

Be helpful. Be competent. Don't waste words.",
        cwd.display()
    );

    // Append workspace context if available
    if let Some(ws) = workspace {
        let ws_context = ws.system_context();
        if !ws_context.is_empty() {
            return format!("{base}\n\n{ws_context}");
        }
    }

    base
}

/// The credentials the CLI's summarizers may use: each key serves the
/// provider it is stored under, as in the daemon — `[llm].api_key` is
/// Anthropic's, `OpenAI`, `OpenRouter` and GitHub have their own, and Ollama
/// gets the token bound to `[memory].ollama_host`.
///
/// Until 2026-09-18 `nanna init` stored an `OpenRouter` or `OpenAI` key in
/// `[llm].api_key`, so this handed that field to the chat provider alone.
/// It now stores it under the provider's own name, and loading a config
/// moves a key the old layout left behind (`nanna_config`'s
/// `provider_key`), so the field holds Anthropic's key only.
///
/// The daemon's Anthropic chain (keyring entry, OAuth refresh, the Claude CLI
/// login) is not walked: the CLI's chat never used it, and walking it would
/// read — and may refresh and rewrite — a login on every start.
fn summarizer_credentials(config: &Config) -> ProviderCredentials {
    let non_blank = |value: Option<&String>| {
        value
            .map(|v| v.trim())
            .filter(|v| !v.is_empty())
            .map(str::to_string)
    };
    let anthropic = non_blank(config.llm.api_key.as_ref());
    // Carried into the error a skipped `anthropic/` entry logs, so the log
    // says why rather than only that Anthropic is missing.
    let anthropic_absent_reason = anthropic.is_none().then(|| {
        "no Anthropic API key is set (`nanna init`, or `ANTHROPIC_API_KEY`)".to_string()
    });
    ProviderCredentials {
        anthropic: anthropic.map(AnthropicCredential::ApiKey),
        anthropic_absent_reason,
        openai_api_key: non_blank(config.llm.openai_api_key.as_ref()),
        openrouter_api_key: non_blank(config.llm.openrouter_api_key.as_ref()),
        github_token: non_blank(config.llm.github_token.as_ref()),
        ollama_host: config.memory.ollama_host.clone(),
        ollama_api_key: non_blank(config.llm.ollama_api_key.as_ref()),
    }
}

/// How the CLI's agent reaches its summarization models: a chat router like
/// the daemon's, so an entry in `[llm].summarization_priority` routes by the
/// one grammar (`ProviderId::from_model`), over [`summarizer_credentials`].
///
/// `None` when the list names no model. Nothing would ever be resolved, so no
/// router is built and the CLI starts exactly as it did before summaries had
/// one; the agent then cuts to fit, and extracts memories on the chat model.
fn summarizer_clients(config: &Config) -> Option<nanna_agent::SummarizerClients> {
    if config
        .llm
        .summarization_priority
        .iter()
        .all(|spec| spec.trim().is_empty())
    {
        return None;
    }
    let router = Arc::new(LlmRouter::new());
    router.rebuild(&summarizer_credentials(config));
    Some(nanna_daemon::llm_router::summarizer_clients(&router))
}

/// The agent `nanna chat` and `nanna run` talk to: `agent_config` walking the
/// config's summarization list, resolved through [`summarizer_clients`] over
/// the same config — so the list the agent walks and the providers it
/// reaches come from one place.
fn cli_agent(
    config: &Config,
    agent_config: AgentConfig,
    llm: Arc<nanna_core::LlmClient>,
    tools: Arc<nanna_tools::ToolRegistry>,
    context: AgentContext,
) -> Agent {
    let agent_config = AgentConfig {
        summarization_priority: config.llm.summarization_priority.clone(),
        ..agent_config
    };
    let agent = Agent::new(agent_config, llm, tools).with_context(context);
    match summarizer_clients(config) {
        Some(clients) => agent.with_summarizer_clients(clients),
        None => agent,
    }
}

/// Print tool call results.
fn print_tool_calls(tool_calls: &[nanna_agent::ToolCallRecord]) {
    if tool_calls.is_empty() {
        return;
    }
    print!("\n[");
    for (i, call) in tool_calls.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        let status = if call.success { "✓" } else { "✗" };
        print!("{} {}", status, call.name);
    }
    println!("]");
}

/// Run interactive CLI mode.
pub async fn run_cli(
    config: &Config,
    session_id: Option<String>,
    model: Option<String>,
    stream: bool,
) -> anyhow::Result<()> {
    use nanna_agent::nanna_workspace::discover_workspace;

    let (llm, tools, storage) = init_components(config).await?;

    // Print banner
    println!("{BANNER}");
    println!(
        "  Moon god of the digital realm. v{}",
        env!("CARGO_PKG_VERSION")
    );
    if stream {
        println!("  Streaming enabled. Type 'quit' to exit, 'clear' to reset.\n");
    } else {
        println!("  Type 'quit' to exit, 'clear' to reset.\n");
    }

    // Try to detect workspace
    let cwd = std::env::current_dir()?;
    let workspace = if let Ok(root) = discover_workspace(Some(&cwd)) {
        match Workspace::load(root.clone()).await {
            Ok(ws) => {
                info!("Workspace detected: {} at {}", ws.name(), root.display());
                println!("  📂 Workspace: {}\n", ws.name());
                Some(ws)
            }
            Err(e) => {
                warn!("Failed to load workspace: {}", e);
                None
            }
        }
    } else {
        debug!("No workspace detected in {}", cwd.display());
        None
    };

    // Session setup
    let (session_id, is_resume) = session_id.map_or_else(
        || (uuid::Uuid::new_v4().to_string(), false),
        |id| (id, true),
    );
    info!("Session: {session_id}");
    let _ = storage.sessions().create(&session_id, "cli", None).await;

    // Agent config
    let agent_config = AgentConfig {
        model: model.unwrap_or_else(|| config.llm.model.clone()),
        max_tokens: config.llm.max_tokens,
        temperature: config.llm.temperature,
        max_iterations: Some(10),
        ..Default::default()
    };

    // Build context with system prompt (includes workspace context if available)
    let mut context = AgentContext::new(&session_id)
        .with_system_prompt(build_cli_system_prompt(&cwd, workspace.as_ref()));

    // Set workspace on context if detected
    if let Some(ref ws) = workspace {
        context = context.with_workspace(ws);
    }

    // Load session history if resuming
    if is_resume
        && let Ok(messages) = storage.messages().get_by_session(&session_id, 50).await {
            let msg_count = messages.len();
            for msg in messages {
                match msg.role.as_str() {
                    "user" => context.add_user_message(&msg.content),
                    "assistant" => context.add_assistant_message(&msg.content),
                    _ => {}
                }
            }
            if msg_count > 0 {
                info!("Resumed session with {msg_count} messages");
                println!("  Resumed session with {msg_count} previous messages.");
            }
        }

    let agent = cli_agent(config, agent_config, llm, tools, context);
    run_cli_loop(&agent, &storage, &session_id, stream).await
}

/// Main REPL loop for CLI mode.
async fn run_cli_loop(
    agent: &Agent,
    storage: &Arc<Storage>,
    session_id: &str,
    stream: bool,
) -> anyhow::Result<()> {
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("\n› ");
        stdout.flush()?;

        let mut input = String::new();
        stdin.lock().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        // Handle commands
        match input.to_lowercase().as_str() {
            "quit" | "exit" | "q" => {
                println!("\nThe moon sets. Until next time.");
                break;
            }
            "clear" => {
                agent.clear().await;
                println!("Context cleared.");
                continue;
            }
            _ => {}
        }

        // Store user message
        let _ = storage
            .messages()
            .create(nanna_storage::NewMessage {
                session_id: session_id.to_string(),
                role: "user".to_owned(),
                content: input.to_owned(),
                content_type: "text".to_owned(),
                tool_use_id: None,
                tokens_in: None,
                tokens_out: None,
                metadata: None,
            })
            .await;

        // Build run options
        let run_options = if stream {
            println!();
            stdout.flush()?;
            RunOptions {
                on_text: Some(Box::new(|text: &str| {
                    print!("{text}");
                    let _ = std::io::stdout().flush();
                })),
                ..Default::default()
            }
        } else {
            RunOptions::default()
        };

        // Run agent and handle response
        match agent.run(input, run_options).await {
            Ok(response) => {
                if stream {
                    println!();
                } else {
                    println!("\n{}", response.text);
                }

                // Store assistant response
                let _ = storage
                    .messages()
                    .create(nanna_storage::NewMessage {
                        session_id: session_id.to_string(),
                        role: "assistant".to_owned(),
                        content: response.text.clone(),
                        content_type: "text".to_owned(),
                        tool_use_id: None,
                        tokens_in: Some(i64::from(response.input_tokens)),
                        tokens_out: Some(i64::from(response.output_tokens)),
                        metadata: None,
                    })
                    .await;

                print_tool_calls(&response.tool_calls);
            }
            Err(err) => {
                eprintln!("\nError: {err}");
            }
        }
    }

    Ok(())
}

/// Run a single prompt and exit
pub async fn run_once(config: &Config, prompt: &str, model: Option<String>) -> anyhow::Result<()> {
    let (llm, tools, _storage) = init_components(config).await?;

    let agent_config = AgentConfig {
        model: model.unwrap_or_else(|| config.llm.model.clone()),
        max_tokens: config.llm.max_tokens,
        temperature: config.llm.temperature,
        max_iterations: Some(10),
        ..Default::default()
    };

    let cwd = std::env::current_dir()?;
    let context = AgentContext::new("oneshot").with_system_prompt(format!(
        r"You are Nanna — a helpful AI assistant.

You have tools at your disposal:
- exec: Execute shell commands
- read_file: Read file contents  
- write_file: Write content to files
- list_dir: List directory contents
- web_fetch: Fetch content from URLs

Current directory: {}

Be concise and direct.",
        cwd.display()
    ));

    let agent = cli_agent(config, agent_config, llm, tools, context);

    match agent.run(prompt, RunOptions::default()).await {
        Ok(response) => {
            println!("{}", response.text);
        }
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    }

    Ok(())
}

/// List recent sessions
pub async fn list_sessions(config: &Config, limit: i64) -> anyhow::Result<()> {
    // Initialize storage only (no LLM needed)
    let storage_path = crate::setup::cli_storage_path(config);

    let storage_config = StorageConfig {
        path: storage_path.to_string_lossy().to_string(),
    };
    let storage = Storage::new(&storage_config).await?;

    let sessions = storage.sessions().list_recent(limit).await?;

    if sessions.is_empty() {
        println!("No sessions found.");
        return Ok(());
    }

    println!("\n🌙 Recent Sessions\n");
    println!("{:<38} {:<8} {:<20}", "SESSION ID", "CHANNEL", "LAST ACTIVE");
    println!("{}", "-".repeat(70));

    for session in sessions {
        println!(
            "{:<38} {:<8} {:<20}",
            session.session_id,
            session.channel,
            session.updated_at
        );
    }

    println!("\nResume with: nanna chat --session <ID>");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHAT_KEY: &str = "sk-chat-provider-key";

    /// What `nanna init` leaves behind: `provider`'s key in that provider's
    /// own field, the only key the CLI has, and a summarization list naming
    /// Anthropic both ways a hand-edit might (the Settings picker's prefix,
    /// and the `OpenRouter` spelling of the same model).
    fn cli_config(provider: &str) -> Config {
        let mut config = Config::default();
        config.llm.provider = provider.to_string();
        config.llm.api_key = None;
        config.llm.openai_api_key = None;
        config.llm.openrouter_api_key = None;
        config.llm.github_token = None;
        config.llm.ollama_api_key = None;
        *config.llm.provider_api_key_mut() = Some(CHAT_KEY.to_string());
        config.llm.summarization_priority = vec![
            "anthropic/claude-3.5-haiku".to_string(),
            "openrouter/anthropic/claude-3.5-haiku".to_string(),
        ];
        config
    }

    fn resolved_model(clients: &nanna_agent::SummarizerClients, spec: &str) -> Result<String, String> {
        clients.resolve(spec).map(|(_, model)| model)
    }

    /// The live hazard: an `OpenRouter` or `OpenAI` user's key must never become
    /// an Anthropic client, which would post it to api.anthropic.com for an
    /// `anthropic/` summary or for a bare name the router sends there.
    #[test]
    fn the_chat_key_serves_the_chat_provider_and_no_other() {
        for (provider, own_prefix) in [
            ("openrouter", "openrouter/anthropic/claude-3.5-haiku"),
            ("openai", "openai/gpt-4o-mini"),
        ] {
            let config = cli_config(provider);
            let credentials = summarizer_credentials(&config);
            assert_eq!(credentials.anthropic, None, "{provider}");
            let slots = [
                ("openai", credentials.openai_api_key.as_deref()),
                ("openrouter", credentials.openrouter_api_key.as_deref()),
            ];
            for (slot, key) in slots {
                let expected = (slot == provider).then_some(CHAT_KEY);
                assert_eq!(key, expected, "{provider}: the {slot} slot");
            }

            let clients = summarizer_clients(&config).expect("the list names models");
            for spec in ["anthropic/claude-3.5-haiku", "llama3"] {
                let why = resolved_model(&clients, spec).expect_err(spec);
                assert!(
                    why.contains("no Anthropic API key is set"),
                    "{provider}: `{spec}` is skipped and the log says why: {why}"
                );
            }
            assert!(
                resolved_model(&clients, own_prefix).is_ok(),
                "{provider}: its own models still summarize"
            );
        }
    }

    #[test]
    fn an_anthropic_chat_key_summarizes_on_anthropic() {
        let config = cli_config("anthropic");
        let credentials = summarizer_credentials(&config);
        assert_eq!(
            credentials.anthropic,
            Some(AnthropicCredential::ApiKey(CHAT_KEY.to_string()))
        );
        assert_eq!(credentials.anthropic_absent_reason, None);
        assert_eq!(credentials.openrouter_api_key, None);

        let clients = summarizer_clients(&config).expect("the list names models");
        assert_eq!(
            resolved_model(&clients, "anthropic/claude-3.5-haiku").as_deref(),
            Ok("claude-3.5-haiku")
        );
        assert!(resolved_model(&clients, "openrouter/anthropic/claude-3.5-haiku").is_err());
    }

    /// `[llm].api_key` is Anthropic's alone, so an Anthropic key kept beside
    /// an `OpenRouter` chat summarizes on Anthropic, as it does in the daemon.
    #[test]
    fn an_anthropic_key_beside_another_chat_summarizes_on_anthropic() {
        let mut config = cli_config("openrouter");
        config.llm.api_key = Some("sk-ant-api03-own".to_string());

        let credentials = summarizer_credentials(&config);
        assert_eq!(
            credentials.anthropic,
            Some(AnthropicCredential::ApiKey("sk-ant-api03-own".to_string()))
        );
        assert_eq!(credentials.anthropic_absent_reason, None);
        assert_eq!(credentials.openrouter_api_key.as_deref(), Some(CHAT_KEY));

        let clients = summarizer_clients(&config).expect("the list names models");
        assert_eq!(
            resolved_model(&clients, "anthropic/claude-3.5-haiku").as_deref(),
            Ok("claude-3.5-haiku")
        );
    }

    /// A key stored under its own provider's name is that provider's whatever
    /// chat uses, and the Ollama token goes only to the server it is bound to.
    #[test]
    fn keys_stored_under_their_own_name_serve_their_own_provider() {
        let mut config = cli_config("openrouter");
        config.llm.openai_api_key = Some("sk-openai-own".to_string());
        config.llm.ollama_api_key = Some("ollama-bound-token".to_string());
        config.memory.ollama_host = "https://ollama.example.test".to_string();

        let credentials = summarizer_credentials(&config);
        assert_eq!(credentials.openai_api_key.as_deref(), Some("sk-openai-own"));
        assert_eq!(credentials.openrouter_api_key.as_deref(), Some(CHAT_KEY));
        assert_eq!(credentials.ollama_host, "https://ollama.example.test");
        assert_eq!(credentials.ollama_api_key.as_deref(), Some("ollama-bound-token"));
        assert_eq!(credentials.anthropic, None);
    }

    /// `nanna chat` and `nanna run` build their agent in one place: with a
    /// listed Ollama summarizer the agent resolves it to the configured
    /// server under its bare id; with no list it carries no resolver.
    #[test]
    fn the_cli_agent_summarizes_through_the_configured_server() {
        let mut config = cli_config("anthropic");
        config.memory.ollama_host = "http://ollama.example.test:11434".to_string();
        config.llm.summarization_priority = vec!["ollama/qwen3:4b".to_string()];
        let build = |config: &Config| {
            cli_agent(
                config,
                AgentConfig::default(),
                Arc::new(nanna_core::LlmClient::ollama("http://127.0.0.1:9")),
                Arc::new(nanna_tools::ToolRegistry::new()),
                AgentContext::new("cli-test"),
            )
        };

        let agent = build(&config);
        let (client, model) = agent
            .summarizer_clients()
            .expect("a listed model attaches a resolver")
            .resolve("ollama/qwen3:4b")
            .expect("the configured server serves it");
        assert_eq!(model, "qwen3:4b");
        assert_eq!(client.base_url(), "http://ollama.example.test:11434");

        config.llm.summarization_priority.clear();
        assert!(build(&config).summarizer_clients().is_none());
    }

    /// With nothing listed nothing would be resolved, so the CLI builds no
    /// router and starts as it always did.
    #[test]
    fn an_empty_list_builds_no_router() {
        let mut config = cli_config("anthropic");
        config.llm.summarization_priority = Vec::new();
        assert!(summarizer_clients(&config).is_none());
        config.llm.summarization_priority = vec!["  ".to_string()];
        assert!(summarizer_clients(&config).is_none());
    }
}
