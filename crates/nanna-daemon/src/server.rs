//! Daemon Server - Main daemon orchestrator
//!
//! Combines IPC server, control plane, sessions, persistence, and all subsystems.

use crate::agent_service::{AgentService, AgentServiceConfig};
use crate::channels::{ChannelManager, ChannelsConfig};
use crate::control::ControlPlane;
use crate::embedding_router::{EmbeddingProviderInfo, EmbeddingRouter};
use crate::health::{DEFAULT_HEALTH_PORT, HealthServer, HealthState, PidFile};
use crate::ipc::{IpcServer, IpcServerConfig};
use crate::llm_router::LlmRouter;
use crate::memory_persistence::TursoMemoryPersistence;
use crate::persistence::PersistenceManager;
use crate::protocol::Response;
use crate::session::SessionManager;
use crate::webhook::{DEFAULT_WEBHOOK_PORT, WebhookConfig, WebhookServer};
use async_trait::async_trait;
use nanna_channels::{
    ChannelId, IncomingMessage, MessageContent, MessageRouter as ChannelMessageRouter,
    Sender as ChannelSender, TelegramChannel,
};
use nanna_config::credentials::{self, SecureStore};
use nanna_memory::MemoryService;
use nanna_scripting::ServiceFn;
use nanna_tools::{AgentSpawner, ParentChannel, SpawnResult, ToolPolicy, ToolRegistry};
use std::collections::HashMap;
use std::path::PathBuf;

/// Heartbeat check-in prompt for the daemon scheduler.
///
/// Deliberately does **not** command the model to `Read HEARTBEAT.md` — that
/// drove a `read_file` tool call which, with no active workspace, resolved to
/// `~/HEARTBEAT.md` and hard-errored (`os error 2`) every heartbeat. P17 has
/// since retired the bespoke `HEARTBEAT.md` entirely (recurrence lives in
/// scheduled-task config), so the prompt now frames the heartbeat as running
/// due scheduled work — never reading instruction files off disk.
const DAEMON_HEARTBEAT_PROMPT: &str = "Heartbeat check-in. Run any due scheduled tasks. Do not read files from disk looking for instructions, and do not infer or repeat old tasks from prior chats. Review your current state, and if nothing needs attention, reply HEARTBEAT_OK.";

/// Concrete implementation of [`AgentSpawner`] that lives in the daemon. Runs
/// each sub-agent as a managed chat on the daemon [`ControlPlane`].
struct AgentSpawnerImpl {
    router: Arc<crate::llm_router::LlmRouter>,
    /// Read at spawn, never at construction: a sub-agent must run on the
    /// model and summarization list the user has NOW, not the ones the daemon
    /// booted with.
    agent_config_src: Arc<tokio::sync::RwLock<crate::agent_service::AgentServiceConfig>>,
    /// Filled once the daemon [`ControlPlane`] is live. Sub-agents are ordinary
    /// chats on that plane — same `run_chat_turn` path as a user turn.
    control: Arc<tokio::sync::RwLock<Option<Arc<ControlPlane>>>>,
}

#[async_trait]
impl AgentSpawner for AgentSpawnerImpl {
    async fn spawn(
        &self,
        prompt: &str,
        description: &str,
        max_iterations: Option<usize>,
    ) -> Result<SpawnResult, String> {
        info!(description = description, max_iterations = ?max_iterations, "Spawning sub-agent");

        // One live read for the whole spawn: the model list, the chat model
        // it falls back to, and the summarization list the sub-agent inherits
        // all come from the config as it stands right now.
        let (base_config, sub_agent_models) = {
            let live = self.agent_config_src.read().await;
            (
                crate::agent_service::agent_config_from(&live),
                live.sub_agent_models.clone(),
            )
        };

        // Fallback chain: first working model wins. A candidate whose
        // provider is missing is skipped; a candidate whose run fails hands
        // the prompt to the next (a fresh agent — sub-agent runs are
        // idempotent by contract, the parent only consumes the final text).
        let candidates = if sub_agent_models.is_empty() {
            vec![base_config.model.clone()]
        } else {
            sub_agent_models.clone()
        };

        let control = self.control.read().await.clone();
        let Some(control) = control else {
            return Err(
                "Sub-agent chat path is not bound yet (control plane not ready).".to_string(),
            );
        };

        let mut last_error = String::new();
        for (attempt, model_spec) in candidates.iter().enumerate() {
            if self.router.client_for_model(model_spec).is_none() {
                last_error = format!(
                    "No provider available for model '{}'. Available providers: {:?}",
                    model_spec,
                    self.router.available_providers()
                );
                warn!(model = %model_spec, "Sub-agent model has no provider — trying next");
                continue;
            }

            if attempt > 0 {
                info!(model = %model_spec, attempt, "Sub-agent model attempt");
            }

            let session = control
                .sessions
                .create(Some(format!("sub-agent: {description}")))
                .await;
            let session_id = session.id.clone();
            info!(
                description,
                session_id = %session_id,
                model = %model_spec,
                attempt = attempt + 1,
                of = candidates.len(),
                max_iterations = ?max_iterations,
                "Starting sub-agent as a managed chat"
            );

            match control.run_chat_turn(&session_id, prompt).await {
                Ok(_) => {
                    control.chat_runs.wait_until_idle(&session_id).await;
                    let text = last_assistant_text(&control.sessions, &session_id).await;
                    if text.trim().is_empty() {
                        last_error = format!(
                            "Sub-agent on '{model_spec}' finished with no output (session {session_id})"
                        );
                        warn!(model = %model_spec, session_id = %session_id, "{last_error}");
                        continue;
                    }
                    info!(
                        description,
                        session_id = %session_id,
                        model = %model_spec,
                        "Sub-agent chat completed"
                    );
                    return Ok(SpawnResult {
                        text,
                        iterations: 0,
                        tool_calls: 0,
                        input_tokens: 0,
                        output_tokens: 0,
                        model: model_spec.clone(),
                    });
                }
                Err(e) => {
                    last_error = format!("Sub-agent chat failed on '{model_spec}': {e}");
                    warn!(
                        model = %model_spec,
                        session_id = %session_id,
                        error = %e,
                        remaining = candidates.len() - attempt - 1,
                        "Sub-agent chat failed — falling back"
                    );
                }
            }
        }

        Err(format!(
            "All {} sub-agent model(s) failed ({:?}). Last error: {}",
            candidates.len(),
            candidates,
            last_error
        ))
    }
}

async fn last_assistant_text(sessions: &SessionManager, session_id: &str) -> String {
    let Some(session) = sessions.get(session_id).await else {
        return String::new();
    };
    session
        .messages
        .iter()
        .rev()
        .find(|m| m.role == crate::session::MessageRole::Assistant)
        .map(|m| m.content.clone())
        .unwrap_or_default()
}

/// Concrete implementation of [`ParentChannel`] that lives in the daemon.
/// Allows sub-agents to ask their parent questions.
///
/// Instead of blocking on mailbox polling, this makes a lightweight LLM call
/// with the parent session's conversation context to answer the sub-agent's
/// question directly. This avoids deadlocks (parent is blocked on the task
/// tool while the sub-agent waits for a reply).
struct ParentChannelImpl {
    sessions: Arc<SessionManager>,
    event_tx: Option<tokio::sync::broadcast::Sender<crate::protocol::Event>>,
    router: Arc<crate::llm_router::LlmRouter>,
    /// Model to use for answering sub-agent questions (e.g. cheap/fast model)
    model: String,
}

#[async_trait]
impl ParentChannel for ParentChannelImpl {
    async fn ask_parent(
        &self,
        sub_session_id: &str,
        question: &str,
        _timeout_secs: u64,
    ) -> Result<String, String> {
        // Look up the sub-session to find its parent and task context
        let sub_info = self
            .sessions
            .get_sub_session(sub_session_id)
            .await
            .ok_or_else(|| {
                format!(
                    "Sub-session '{sub_session_id}' not found — ask_parent is only available to \
                     sub-agents"
                )
            })?;

        let parent_id = sub_info
            .parent_id
            .clone()
            .ok_or_else(|| "This sub-agent has no parent session".to_string())?;

        let label = sub_info.label.clone();
        let task = sub_info.task.clone();

        // Emit event for GUI visibility
        if let Some(ref tx) = self.event_tx {
            let _ = tx.send(crate::protocol::Event::SubSessionQuestion {
                session_id: sub_session_id.to_string(),
                parent_id: Some(parent_id.clone()),
                label: label.clone(),
                question: question.to_string(),
            });
        }

        tracing::info!(
            sub_session = sub_session_id,
            parent = %parent_id,
            label = ?label,
            "Sub-agent asking parent: {}",
            question.chars().take(100).collect::<String>()
        );

        // Load parent session's recent conversation for context
        let parent_session = self
            .sessions
            .get(&parent_id)
            .await
            .ok_or_else(|| format!("Parent session '{parent_id}' not found"))?;

        let recent_messages: Vec<String> = parent_session
            .messages
            .iter()
            .rev()
            .take(20) // Last 20 messages for context
            .rev()
            .map(|m| {
                format!(
                    "[{}]: {}",
                    m.role.as_db_str(),
                    m.content.chars().take(500).collect::<String>()
                )
            })
            .collect();

        let context = recent_messages.join("\n");

        // Build a focused prompt to answer the sub-agent's question
        let prompt = format!(
            "You are answering a question from a sub-agent that was delegated a task.\n\n\
             ## Sub-agent task\n{}\n\n\
             ## Recent conversation context (parent session)\n{}\n\n\
             ## Sub-agent's question\n{}\n\n\
             Answer concisely and directly. Provide only the information the sub-agent needs \
             to continue its work. If you don't have enough context to answer, say so clearly.",
            task,
            if context.is_empty() {
                "(no prior conversation)".to_string()
            } else {
                context
            },
            question
        );

        // Make a lightweight LLM call to answer the question using parent context
        let model = &self.model;
        let llm_client = self
            .router
            .client_for_model(model)
            .ok_or_else(|| format!("No provider for model '{model}'"))?;

        let stripped_model = crate::llm_router::LlmRouter::strip_model_prefix(model);
        let request = nanna_llm::CompletionRequest {
            model: stripped_model,
            messages: vec![
                nanna_llm::Message::system(
                    "You are a helpful assistant answering questions from sub-agents. Be concise and precise.",
                ),
                nanna_llm::Message::user(&prompt),
            ],
            max_tokens: Some(2048),
            ..Default::default()
        };

        let answer = llm_client
            .complete(&request)
            .await
            .map_err(|e| format!("LLM call failed: {e}"))?;

        tracing::info!(
            sub_session = sub_session_id,
            answer_len = answer.len(),
            "Parent answered sub-agent question"
        );

        Ok(answer)
    }
}

use std::sync::Arc;
use std::time::Duration;

/// Build service closures for script tools.
///
/// These closures allow JS/TS tools to call back into Rust subsystems via `Nanna.service(name, params)`.
/// Shared session history that can be updated before each agent run.
/// This allows the `session.history` service to return messages for the current session.
pub type SharedSessionHistory = Arc<tokio::sync::RwLock<Vec<crate::session::SessionMessage>>>;

/// Stamp a `memory.store` call's declared provenance onto its tags.
///
/// Until now the service wrote only the caller's `tags`, so a memory saved
/// through the `remember` TOOL carried no `fact_type` at all — and the drift
/// pin (`nanna_memory::is_verbatim_pinned`) can only protect what it can
/// identify, so the memories a user most explicitly asked to keep were the ones
/// it could never protect. This closes that, without letting the tool
/// over-claim: the classification is `MemoryProvenance::from_label`, the SAME
/// rule the extraction path uses, so only an explicit case-insensitive
/// `"stated"` yields `stated` and everything else — absent, empty, misspelt —
/// is `observed`.
///
/// A caller-supplied `fact_type` in `tags` is respected as the declaration if
/// no explicit `provenance` field is given, so a script that already stamps the
/// key keeps working; either way the value is re-classified rather than
/// trusted verbatim, so `tags: {fact_type: "STATED-ish"}` cannot smuggle a pin.
fn tags_with_provenance(
    mut tags: HashMap<String, String>,
    params: &serde_json::Value,
) -> HashMap<String, String> {
    let declared = params
        .get("provenance")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .or_else(|| tags.get("fact_type").cloned())
        .unwrap_or_default();
    let provenance = nanna_agent::MemoryProvenance::from_label(&declared);
    tags.insert("fact_type".to_string(), provenance.as_str().to_string());
    debug_assert!(
        tags.get("fact_type").is_some_and(|f| f == "stated" || f == "observed"),
        "fact_type is a closed set"
    );
    tags
}

/// Resolve a memory handle — including one whose memory has since been
/// consolidated away by dreaming.
///
/// Context stubs quote a handle, so "the full result is in memory, use the
/// handle" is a promise the store must keep. But dreaming REPLACES clusters:
/// it writes one consolidated memory and forgets the originals. A handle
/// captured before that would dangle, and the model would be told a result it
/// was explicitly promised no longer exists — the worst possible answer,
/// because it reads as data loss when the content is actually still there in
/// generalised form.
///
/// Consolidation already records `consolidated_from` (the ids it absorbed),
/// so the trail exists; this follows it. Resolution order:
/// 1. the entry's own id,
/// 2. the `source_id` tag shared by the chunks of one tool result,
/// 3. **forwarding**: any memory that lists this handle in `consolidated_from`,
/// 4. an unambiguous id prefix (stubs carry a short id).
///
/// Forwarding is transitive by construction: a consolidation of a
/// consolidation carries the intermediate id, and the walk repeats. The hop
/// limit only stops a cycle that a corrupt store could otherwise turn into a
/// hang.
/// The whole text behind a handle: every chunk sharing its `source_id`, in order.
///
/// A large tool result is split into N rows that share one `source_id`, but
/// `resolve_memory_handle` returns a single row — so paging ran off the end of
/// chunk 1 and reported `truncated: false`, while the stub sitting in the
/// model's context promised the result "was stored whole in memory as {N}
/// chunk(s)" and that recall "returns the full text". A model reading a fifth
/// of a 42-case test run and being told nothing was missing will report on what
/// it saw. Reassemble, rather than keep a promise the retrieval path was not
/// keeping.
fn assemble_handle_content(
    all: &[nanna_memory::MemoryListEntry],
    entry: &nanna_memory::MemoryListEntry,
) -> String {
    let Some(source_id) = entry.metadata.get("source_id") else {
        return entry.content.clone();
    };
    // `chunk` is `"i/N"`: the position, and the count the stub promised.
    let mut expected_count = 0_usize;
    let mut chunks: Vec<(usize, &str)> = all
        .iter()
        .filter(|e| e.metadata.get("source_id").is_some_and(|s| s == source_id))
        .map(|e| {
            let mark = e.metadata.get("chunk");
            let idx = mark
                .and_then(|c| c.split('/').next()?.parse::<usize>().ok())
                .unwrap_or(1);
            if let Some(total) = mark.and_then(|c| c.split('/').nth(1)?.parse::<usize>().ok()) {
                expected_count = expected_count.max(total);
            }
            (idx, e.content.as_str())
        })
        .collect();
    if chunks.len() <= 1 {
        return entry.content.clone();
    }
    chunks.sort_by_key(|(idx, _)| *idx);
    let found_count = chunks.len();
    let assembled = chunks
        .into_iter()
        .map(|(_, content)| content)
        .collect::<Vec<_>>()
        .join("\n");

    // Say so when fewer rows came back than the stub promised. Dreaming
    // REPLACES clusters, so a result whose chunks have been partly
    // consolidated away reassembles short — and returning that silently is the
    // exact failure this function was written to end: a model reading a
    // fraction of a run and being told nothing was missing reports on what it
    // saw. `expected_count` is 0 when no row carried an `i/N` mark, which is
    // not evidence of loss, so that case says nothing.
    if expected_count > found_count {
        let missing_count = expected_count - found_count;
        return format!(
            "{assembled}\n\n[SYSTEM: {found_count} of {expected_count} stored chunks were \
             reassembled — {missing_count} are no longer in the store as separate rows, most \
             likely folded into a consolidated memory by a dream cycle. What is above is \
             complete for the chunks that remain, and the artifact itself is unaffected: read \
             it back off disk if you need the whole thing.]"
        );
    }
    assembled
}

/// The byte range of `content` that one `memory.get` page covers.
///
/// Rust panics when a byte index splits a char, and what is stored behind a
/// handle is tool and model output — an em dash in an `edit_file` error is
/// what killed the daemon — so both ends walk forward to the next boundary.
/// Forward, not back, because the walked `start` is handed to the caller as
/// the page's `offset`: a follow-up read resumes at the byte this one really
/// began at, and no char is dropped between two pages.
///
/// The range is only valid for the string it was computed from. Keeping that
/// pairing in one function is the point: the offsets were once proven against
/// the assembled text and then used to index a single chunk of it, which is
/// both out of bounds and off-boundary.
/// What a scheduled job posts into its conversation after a run. Pure.
///
/// Nothing for a heartbeat that found nothing to do (the `HEARTBEAT_OK`
/// sentinel is run mechanics, not news) or for an empty result — a chat app
/// would otherwise get a blank message on every quiet run.
fn scheduled_output_message(task_name: &str, content: &str, heartbeat_ok: bool) -> Option<String> {
    let content = content.trim();
    if heartbeat_ok || content.is_empty() {
        return None;
    }
    debug_assert!(!task_name.is_empty(), "every scheduled task is named");
    Some(format!("**Scheduled: {task_name}**\n\n{content}"))
}

/// The dreaming orchestrator, reached late — it is built after script services.
pub type DreamingSlot = Arc<std::sync::OnceLock<Arc<nanna_memory::DreamingService>>>;

/// Which memories a `memory.get` read demonstrably used, and how. Pure.
///
/// This is the first producer of `UsedSuccessfully`, which the FSRS feedback
/// loop priced and tallied but nothing ever sent. The attribution is the one
/// that is a fact about specific memories rather than a guess: a model resolved
/// a stored handle and got the stored text back, so those rows carried the work
/// forward. Only the FIRST page counts — paging through one result is one use,
/// and counting each page would reward long outputs for being long. A forwarded
/// handle credits the memory that absorbed the original, because that is what
/// answered.
fn recall_feedback(
    offset: usize,
    served_ids: &[String],
) -> Vec<(String, nanna_memory::MemoryFeedback)> {
    if offset > 0 {
        return Vec::new();
    }
    debug_assert!(
        !served_ids.is_empty(),
        "a resolved read served at least one row"
    );
    served_ids
        .iter()
        .map(|id| (id.clone(), nanna_memory::MemoryFeedback::UsedSuccessfully))
        .collect()
}

/// The ids of the rows a handle's content was assembled from.
fn served_row_ids(
    all: &[nanna_memory::MemoryListEntry],
    entry: &nanna_memory::MemoryListEntry,
) -> Vec<String> {
    let Some(source_id) = entry.metadata.get("source_id") else {
        return vec![entry.id.clone()];
    };
    let ids: Vec<String> = all
        .iter()
        .filter(|e| e.metadata.get("source_id").is_some_and(|s| s == source_id))
        .map(|e| e.id.clone())
        .collect();
    if ids.is_empty() {
        vec![entry.id.clone()]
    } else {
        ids
    }
}

fn handle_page_range(content: &str, offset: usize, limit: usize) -> (usize, usize) {
    let total = content.len();
    let start = offset.min(total);
    let end = start.saturating_add(limit).min(total);
    let mut s = start;
    while s < total && !content.is_char_boundary(s) {
        s += 1;
    }
    let mut e = end;
    while e < total && !content.is_char_boundary(e) {
        e += 1;
    }
    (s, e)
}

async fn resolve_memory_handle(
    memory: &Arc<MemoryService>,
    handle: &str,
) -> Result<nanna_memory::MemoryListEntry, String> {
    resolve_memory_handle_in(&memory.list_all().await, handle)
}

/// [`resolve_memory_handle`] over a snapshot the caller already holds, so a
/// caller that also assembles content reads ONE state of the store — not a
/// resolve from one snapshot and a reassembly from a later one, with a dream
/// cycle free to land in between.
fn resolve_memory_handle_in(
    all: &[nanna_memory::MemoryListEntry],
    handle: &str,
) -> Result<nanna_memory::MemoryListEntry, String> {
    const MAX_FORWARD_HOPS: usize = 8;

    let direct = |needle: &str| -> Option<nanna_memory::MemoryListEntry> {
        all.iter()
            .find(|e| e.id == needle)
            .or_else(|| {
                all.iter()
                    .find(|e| e.metadata.get("source_id").is_some_and(|s| s == needle))
            })
            .cloned()
    };

    if let Some(found) = direct(handle) {
        return Ok(found);
    }

    // Follow the consolidation trail: who absorbed this id?
    let mut needle = handle.to_string();
    for _ in 0..MAX_FORWARD_HOPS {
        let successor = all.iter().find(|e| {
            e.metadata
                .get("consolidated_from")
                .is_some_and(|sources| sources.split(',').any(|s| s.trim() == needle))
                || e.metadata
                    .get("sources")
                    .is_some_and(|sources| sources.split(',').any(|s| s.trim() == needle))
        });
        match successor {
            Some(entry) => {
                if let Some(found) = direct(&entry.id) {
                    return Ok(found);
                }
                needle = entry.id.clone();
            }
            None => break,
        }
    }

    all.iter()
        .find(|e| e.id.starts_with(handle))
        .cloned()
        .ok_or_else(|| {
            format!(
                "no memory matches handle '{handle}'. It was not found directly, and nothing \
                 in the store lists it as a source, so it was not consolidated into another \
                 memory either — it may predate this store or have been deleted outright."
            )
        })
}

// ---------------------------------------------------------------------------
// Param dialect readers for the `memory.*` script services
// ---------------------------------------------------------------------------
//
// Same contract as the task services (see `crate::tasks`): read what can be
// read losslessly, and answer what cannot with an error naming what ARRIVED.
// "id is required" said to a caller who supplied an id in the wrong shape is
// a false statement, and a model that reads it re-sends the identical call.

/// Coerce a scalar to the text a memory service wants.
///
/// A memory id and a memory body are both genuinely text, so there is nothing
/// to invent here: the only non-string shapes accepted are the scalars whose
/// rendering is exactly the characters the caller meant — a handle typed
/// without its quotes (`{"id": 12}`), a number, a boolean. An array or object
/// is not text under any reading and is refused.
fn as_text_lenient(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Number(n) => Some(n.to_string()),
        serde_json::Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

/// Read a required text param, keeping MISSING and UNINTERPRETABLE apart.
fn req_text(params: &serde_json::Value, key: &str) -> Result<String, String> {
    let Some(value) = params.get(key).filter(|v| !v.is_null()) else {
        return Err(format!("{key} is required"));
    };
    let text = as_text_lenient(value).ok_or_else(|| {
        format!(
            "{key} must be text (got {}).",
            crate::tasks::describe_value(value)
        )
    })?;
    if text.trim().is_empty() {
        return Err(format!("{key} is required — it arrived empty."));
    }
    Ok(text)
}

/// Read an optional text param. Absent or null yields `None`; a present value
/// that is not text errors instead of falling through to a default — a
/// silently defaulted `new` on `memory.replace` would delete the match.
fn opt_text(params: &serde_json::Value, key: &str) -> Result<Option<String>, String> {
    params
        .get(key)
        .filter(|v| !v.is_null())
        .map_or(Ok(None), |value| {
            as_text_lenient(value).map(Some).ok_or_else(|| {
                format!(
                    "{key} must be text (got {}).",
                    crate::tasks::describe_value(value)
                )
            })
        })
}

/// Read an optional count param (`limit`, `offset`).
///
/// Integers follow the same dialect rules as the task services — a stringified
/// `"50"` is read rather than dropped onto the default. A negative count is
/// refused rather than clamped: clamping answers a question nobody asked.
fn opt_count(params: &serde_json::Value, key: &str) -> Result<Option<usize>, String> {
    let Some(n) = crate::tasks::opt_i64(params, key)? else {
        return Ok(None);
    };
    usize::try_from(n)
        .map(Some)
        .map_err(|_| format!("{key} must be zero or a positive whole number (got {n})."))
}

/// The tools directory the authoring services write into, the live registry
/// they register into, and the slot they read the finished service map back
/// out of.
///
/// Named because the tuple is past `clippy::type_complexity`'s limit written
/// inline, and a reader of [`ScriptServiceDeps::tool_authoring`] should not have
/// to parse three nested generics to learn it is those three things.
type ToolAuthoringDeps = (
    PathBuf,
    std::sync::Weak<nanna_tools::ToolRegistry>,
    Arc<std::sync::OnceLock<HashMap<String, ServiceFn>>>,
);

/// What [`DaemonServer::load_script_tool_surface`] needs from `init_services`'s
/// locals.
///
/// A struct rather than nine positional references, for exactly the reason
/// [`ScriptServiceDeps`] is one: several are `Arc`s of different inner types
/// that the compiler would let a caller transpose in pairs.
struct ToolSurfaceDeps<'a> {
    tools: &'a Arc<ToolRegistry>,
    tools_dir: Option<&'a PathBuf>,
    router: &'a Arc<LlmRouter>,
    memory: &'a Option<Arc<MemoryService>>,
    session_history: &'a SharedSessionHistory,
    workspace_id_for_services: &'a Arc<tokio::sync::RwLock<Option<String>>>,
    turn_baselines: &'a Arc<crate::tasks::TurnBaselines>,
    shared_agent_config: &'a Arc<tokio::sync::RwLock<AgentServiceConfig>>,
    chat_runs: &'a Arc<crate::control::chat_harness::ChatRunRegistry>,
}

/// What [`DaemonServer::build_control_plane`] needs from `run`'s locals.
///
/// The control plane is assembled from fourteen collaborators that `run` has
/// already created, most of them `Arc`s; named fields keep the assembly honest
/// the way [`ScriptServiceDeps`] does for the script services.
struct ControlPlaneDeps {
    agent: Arc<AgentService>,
    memory: Option<Arc<MemoryService>>,
    tools: Arc<ToolRegistry>,
    router: Arc<LlmRouter>,
    tools_dir: Option<PathBuf>,
    workspace_id_for_services: Arc<tokio::sync::RwLock<Option<String>>>,
    turn_baselines: Arc<crate::tasks::TurnBaselines>,
    scheduler: Arc<tokio::sync::RwLock<nanna_core::Scheduler>>,
    model_stats: nanna_agent::ModelStatsTracker,
    chat_runs: Arc<crate::control::chat_harness::ChatRunRegistry>,
    degradations: Arc<nanna_agent::DegradationLedger>,
    activity_clock: Arc<nanna_memory::ActivityClock>,
    dreaming: Option<Arc<nanna_memory::DreamingService>>,
}

/// Everything [`build_script_services`] needs, named.
///
/// It grew from seven positional parameters to thirteen in one run (2026-09-15),
/// and three of the new ones are adjacent `Option<PathBuf>`s that the compiler
/// would happily let a caller transpose. They happen to carry the same value
/// today, which means a transposition would be silent *and* harmless — the worst
/// combination, because it would stay wrong the moment they diverge. Named
/// fields make that class of mistake unrepresentable, and `Default` lets a test
/// name only the dependency it exercises.
#[derive(Default)]
struct ScriptServiceDeps {
    memory: Option<Arc<MemoryService>>,
    spawner: Option<Arc<dyn AgentSpawner + Send + Sync>>,
    session_history: SharedSessionHistory,
    workspace_id: Arc<tokio::sync::RwLock<Option<String>>>,
    storage: Option<Arc<nanna_storage::Storage>>,
    turn_baselines: Arc<crate::tasks::TurnBaselines>,
    /// Router plus the LIVE config the model list is resolved from at call
    /// time. A `Vec<String>` here would be a boot snapshot, and this service
    /// outlives every `config.set` (2026-08-15).
    summarizer: Option<(
        Arc<crate::llm_router::LlmRouter>,
        Arc<tokio::sync::RwLock<crate::agent_service::AgentServiceConfig>>,
    )>,
    /// The tools directory and live registry the authoring services write into,
    /// plus the slot they read the finished service map back out of. `None`
    /// leaves `tools.{create,update,list}` unregistered, which withholds the
    /// three authoring skills rather than half-wiring them.
    tool_authoring: Option<ToolAuthoringDeps>,
    /// Router plus the configured vision-model priority list. `None`, an empty
    /// list, or a list the router cannot serve all leave `vision.analyze`
    /// unregistered, which withholds the three vision skills rather than
    /// advertising tools that can only fail.
    vision: Option<(Arc<crate::llm_router::LlmRouter>, Vec<String>)>,
    /// The same vision model, bound for `pdf.read`'s OCR fallback. `None` leaves
    /// image-only pages unread and the response says so.
    pdf_ocr: Option<nanna_tools::PdfOcrFn>,
    /// `OpenAI` key plus the data dir generated speech is written under. `None` or
    /// no key leaves `audio.tts` / `audio.transcribe` unregistered.
    audio: Option<(Option<String>, PathBuf)>,
    /// The data dir page screenshots are written under. `None`, or no
    /// Chromium-family browser on the host, leaves `browser.*` unregistered.
    browser_data_dir: Option<PathBuf>,
    /// The data dir desktop captures are written under. `None`, or no capture
    /// tool / display session, leaves `screenshot.capture` unregistered.
    screenshot_data_dir: Option<PathBuf>,
    /// The late-bound scheduler and the sessions reminders are delivered
    /// into. `None` leaves `schedule.*` unregistered, which withholds the three
    /// reminder skills.
    reminders: Option<(crate::reminder_service::SchedulerSlot, Arc<SessionManager>)>,
    /// Sessions, the event bus and the run registry `session.ask_user` posts
    /// through and waits on. `None` withholds the `ask_user` skill.
    ask_user: Option<crate::ask_user_service::AskUserDeps>,
    /// Where `memory.get` records that a stored memory was used. `None`, or a
    /// slot never filled, records nothing.
    feedback: Option<DreamingSlot>,
}

/// `memory.store` and its `memory.embed` alias: the two names a script can
/// write a memory under.
///
/// Both go through [`tags_with_provenance`], because provenance is what decides
/// whether a dream cycle may paraphrase the result, and a `remember` that
/// reached only one of these two entry points would be unprotected.
fn memory_write_services(
    mem: &Arc<MemoryService>,
    workspace_id: Arc<tokio::sync::RwLock<Option<String>>>,
) -> HashMap<String, ServiceFn> {
    use serde_json::{Value, json};

    let mut services: HashMap<String, ServiceFn> = HashMap::new();

    let mem_store = mem.clone();
    let ws_store = workspace_id.clone();
    services.insert(
        "memory.store".to_string(),
        Arc::new(move |params: Value| {
            let mem = mem_store.clone();
            let ws = ws_store.clone();
            Box::pin(async move {
                let content = params
                    .get("content")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let tags: HashMap<String, String> = params
                    .get("tags")
                    .and_then(|v| v.as_object())
                    .map(|obj| {
                        obj.iter()
                            .map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string()))
                            .collect()
                    })
                    .unwrap_or_default();
                let importance = crate::numeric::f32_from_f64(
                    params
                        .get("importance")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(1.0),
                );
                // Provenance is what decides whether a dream cycle may
                // paraphrase this memory, so it is written at the one place
                // every `remember` call passes through — both this service
                // and its `memory.embed` alias.
                let tags = tags_with_provenance(tags, &params);
                let workspace = ws.read().await.clone();
                match mem
                    .remember_scoped(&content, tags, importance, workspace)
                    .await
                {
                    Ok((id, _)) => Ok(json!({"id": id})),
                    Err(e) => Err(e.to_string()),
                }
            })
        }),
    );

    // Alias: some tool scripts may call memory.embed instead of memory.store
    let mem_embed = mem.clone();
    // Last reader of `workspace_id` in this function, so it moves.
    let ws_embed = workspace_id;
    services.insert(
        "memory.embed".to_string(),
        Arc::new(move |params: Value| {
            let mem = mem_embed.clone();
            let ws = ws_embed.clone();
            Box::pin(async move {
                let content = params
                    .get("content")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let tags: HashMap<String, String> = params
                    .get("tags")
                    .and_then(|v| v.as_object())
                    .map(|obj| {
                        obj.iter()
                            .map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string()))
                            .collect()
                    })
                    .unwrap_or_default();
                let importance = crate::numeric::f32_from_f64(
                    params
                        .get("importance")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(1.0),
                );
                // Provenance is what decides whether a dream cycle may
                // paraphrase this memory, so it is written at the one place
                // every `remember` call passes through — both this service
                // and its `memory.embed` alias.
                let tags = tags_with_provenance(tags, &params);
                let workspace = ws.read().await.clone();
                match mem
                    .remember_scoped(&content, tags, importance, workspace)
                    .await
                {
                    Ok((id, _)) => Ok(json!({"id": id})),
                    Err(e) => Err(e.to_string()),
                }
            })
        }),
    );

    services
}

/// `memory.search`: scoped similarity recall, one page of each hit.
fn memory_search_services(
    mem: &Arc<MemoryService>,
    workspace_id: Arc<tokio::sync::RwLock<Option<String>>>,
) -> HashMap<String, ServiceFn> {
    use serde_json::{Value, json};

    let mut services: HashMap<String, ServiceFn> = HashMap::new();

    let mem_search = mem.clone();
    // The only service registered here, so the binding moves rather than clones.
    let ws_search = workspace_id;
    services.insert(
        "memory.search".to_string(),
        Arc::new(move |params: Value| {
            let mem = mem_search.clone();
            let ws = ws_search.clone();
            Box::pin(async move {
                let query = params
                    .get("query")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                // Lenient: a JS number arrives as a float (`10.0`), which a
                // strict `as_u64` read as absent and silently replaced with
                // the default. `opt_count` takes it, and names a bad value.
                let limit = opt_count(&params, "limit")?.unwrap_or(10);
                // Per-result page budget. Storage is unbounded now, so a
                // recall that returned whole memories would put an
                // arbitrarily large payload into a fixed context window —
                // `limit` times over. The default is one embedding chunk's
                // worth of text: the same unit the memory was indexed in,
                // so a page corresponds to something the retrieval actually
                // reasoned about rather than to a round number of bytes.
                let page_chars = opt_count(&params, "page_chars")?
                    .unwrap_or(nanna_memory::MEMORY_CHUNK_TARGET_CHARS);
                let offset = opt_count(&params, "offset")?.unwrap_or(0);
                let workspace = ws.read().await;
                match mem
                    .recall_scoped_with_coverage(&query, workspace.as_deref())
                    .await
                {
                    // An empty answer from a scan that could not compare
                    // every stored memory (entries queued for embedding, or
                    // bound to another model) is not "nothing matched".
                    Ok((results, coverage)) if results.is_empty() && !coverage.is_complete() => {
                        keyword_search(
                            &mem,
                            &query,
                            workspace.as_deref(),
                            limit,
                            page_chars,
                            offset,
                        )
                        .await
                    }
                    Ok((results, _)) => {
                        let items: Vec<Value> = results
                            .into_iter()
                            .take(limit)
                            .map(|r| {
                                let (content, start, total) = r.excerpt(offset, page_chars);
                                let returned = content.chars().count();
                                json!({
                                    "id": r.id,
                                    "content": content,
                                    "score": r.score,
                                    "match": "semantic",
                                    // Always present, never inferred from
                                    // whether `content` "looks" cut off. A
                                    // page that does not announce itself is
                                    // indistinguishable from a whole
                                    // memory, and a reader that believes it
                                    // has the whole thing stops looking.
                                    "offset": start,
                                    "returned": returned,
                                    "total": total,
                                    "truncated": start + returned < total,
                                    "best_chunk": r.best_chunk,
                                })
                            })
                            .collect();
                        Ok(Value::Array(items))
                    }
                    Err(e) => {
                        // No embedder answering: the query cannot be embedded,
                        // so semantic recall cannot run at all. That used to
                        // answer an empty list — "no memories found" about a
                        // memory stored seconds earlier. Match words instead.
                        let msg = e.to_string();
                        if msg.contains("embedding") || msg.contains("No embedding function") {
                            tracing::info!("Memory search falling back to keywords: {}", msg);
                            keyword_search(
                                &mem,
                                &query,
                                workspace.as_deref(),
                                limit,
                                page_chars,
                                offset,
                            )
                            .await
                        } else {
                            Err(msg)
                        }
                    }
                }
            })
        }),
    );

    services
}

/// `memory.search`'s answer when semantic recall cannot give one: memories
/// ranked by the query's words ([`crate::keyword_recall`]), in the same
/// paged shape, each marked `"match": "keyword"`.
///
/// Scope follows `recall_scoped`: a workspace sees global memories plus its
/// own; no workspace sees everything. Nothing found is an error whose text
/// says semantic search was unavailable — an empty list would read as "no
/// such memory", which the fallback cannot know.
async fn keyword_search(
    mem: &MemoryService,
    query: &str,
    workspace: Option<&str>,
    limit: usize,
    page_chars: usize,
    offset: usize,
) -> Result<serde_json::Value, String> {
    use serde_json::{Value, json};

    let terms = crate::keyword_recall::query_terms(query);
    let entries: Vec<_> = mem
        .list_all()
        .await
        .into_iter()
        .filter(|entry| {
            workspace.is_none()
                || entry.workspace_id.is_none()
                || entry.workspace_id.as_deref() == workspace
        })
        .collect();
    let hits =
        crate::keyword_recall::rank(&terms, entries.iter().map(|e| e.content.as_str()), limit);
    if hits.is_empty() {
        return Err(format!(
            "semantic memory search is unavailable right now (no embedding provider is \
             answering), and no stored memory contains the words of \"{query}\" either — \
             {} memories were searched by keyword",
            entries.len()
        ));
    }
    let items: Vec<Value> = hits
        .iter()
        .filter_map(|hit| entries.get(hit.index).map(|entry| (hit, entry)))
        .map(|(hit, entry)| {
            let total = entry.content.chars().count();
            let start = offset.min(total);
            let content: String = entry
                .content
                .chars()
                .skip(start)
                .take(page_chars.max(1))
                .collect();
            let returned = content.chars().count();
            json!({
                "id": entry.id,
                "content": content,
                "score": hit.score,
                "match": "keyword",
                "offset": start,
                "returned": returned,
                "total": total,
                "truncated": start + returned < total,
                "best_chunk": Value::Null,
            })
        })
        .collect();
    debug_assert_eq!(items.len(), hits.len(), "every hit indexes a listed entry");
    Ok(Value::Array(items))
}

/// `memory.delete` and `memory.list`: the unscoped housekeeping pair.
fn memory_index_services(mem: &Arc<MemoryService>) -> HashMap<String, ServiceFn> {
    use serde_json::{Value, json};

    let mut services: HashMap<String, ServiceFn> = HashMap::new();

    let mem_delete = mem.clone();
    services.insert(
        "memory.delete".to_string(),
        Arc::new(move |params: Value| {
            let mem = mem_delete.clone();
            Box::pin(async move {
                let id = params
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                match mem.forget(&id).await {
                    Ok(()) => Ok(json!({"deleted": true})),
                    Err(e) => Err(e.to_string()),
                }
            })
        }),
    );

    let mem_list = mem.clone();
    services.insert(
        "memory.list".to_string(),
        Arc::new(move |params: Value| {
            let mem = mem_list.clone();
            Box::pin(async move {
                let limit = opt_count(&params, "limit")?.unwrap_or(20);
                let all = mem.list_all().await;
                let items: Vec<Value> = all
                    .into_iter()
                    .take(limit)
                    .map(|e| json!({"id": e.id, "content": e.content, "weight": e.weight}))
                    .collect();
                Ok(Value::Array(items))
            })
        }),
    );

    services
}

/// `memory.get`: read ONE memory by id, with a byte range.
///
/// The store had only "search by similarity" and "list everything", which is
/// why a tool result kept in memory could not be pointed at from context: a
/// stub naming an id had no way to dereference it. This is the first piece of a
/// file-like surface (read a range, later append/replace) so "the full result
/// lives in memory, a stub lives in context" actually has a retrieval path.
fn memory_read_services(
    mem: &Arc<MemoryService>,
    feedback: Option<DreamingSlot>,
) -> HashMap<String, ServiceFn> {
    use serde_json::{Value, json};

    let mut services: HashMap<String, ServiceFn> = HashMap::new();

    let mem_get = mem.clone();
    let feedback_for_get = feedback;
    services.insert(
        "memory.get".to_string(),
        Arc::new(move |params: Value| {
            let mem = mem_get.clone();
            let feedback = feedback_for_get.clone();
            Box::pin(async move {
                let id = req_text(&params, "id")?;
                let offset = opt_count(&params, "offset")?.unwrap_or(0);
                // Default cap keeps a huge tool result from re-flooding
                // the context the stub existed to protect.
                let limit = opt_count(&params, "limit")?.unwrap_or(4_000);

                // One snapshot for the resolve, the reassembly and the
                // feedback: it used to be three full `list_all()` clones.
                let all = mem.list_all().await;
                let entry = resolve_memory_handle_in(&all, &id)?;
                let content = assemble_handle_content(&all, &entry);
                if let Some(dreaming) = feedback.as_ref().and_then(|slot| slot.get()) {
                    let served = served_row_ids(&all, &entry);
                    for (memory_id, signal) in recall_feedback(offset, &served) {
                        dreaming.record_feedback(&memory_id, signal).await;
                    }
                }

                let total = content.len();
                // Never split a UTF-8 char, and index the same text the
                // range was measured against: every field below reports on
                // the assembled content, so that is what the page cuts.
                let (s, e) = handle_page_range(&content, offset, limit);

                // If the handle forwarded, SAY SO. Silently returning a
                // consolidated narration where raw output was asked for
                // is how a model concludes its data was corrupted; the
                // note explains that dreaming folded the original in and
                // that nothing was lost, only generalised.
                let forwarded = entry.id != id
                    && entry.metadata.get("source_id").is_none_or(|s| s != &id);
                let mut out = json!({
                    "id": entry.id,
                    "content": &content[s..e],
                    "offset": s,
                    "returned": e - s,
                    "total": total,
                    "truncated": e < total,
                });
                if forwarded {
                    out["forwarded_from"] = json!(id);
                    out["note"] = json!(format!(
                        "'{id}' was consolidated during dreaming; this is the memory that \
                         absorbed it ({}). The original text was generalised into this one, \
                         not deleted.",
                        entry.id
                    ));
                }
                Ok(out)
            })
        }),
    );

    services
}

/// `memory.append` / `memory.replace`: the write half of the file-like surface.
///
/// Without them a memory can only be created or forgotten, so a record that has
/// become WRONG (an action narrated as a fact, e.g. "creating minidb.sh at
/// D:\…" nine hours after that file stopped existing) can only be duplicated or
/// destroyed, never corrected. Append gives a running record per subject instead
/// of N disconnected islands.
fn memory_edit_services(mem: &Arc<MemoryService>) -> HashMap<String, ServiceFn> {
    use serde_json::{Value, json};

    let mut services: HashMap<String, ServiceFn> = HashMap::new();

    let mem_append = mem.clone();
    services.insert(
        "memory.append".to_string(),
        Arc::new(move |params: Value| {
            let mem = mem_append.clone();
            Box::pin(async move {
                let handle = req_text(&params, "id")?;
                let addition = req_text(&params, "content")?;
                let entry = resolve_memory_handle(&mem, &handle).await?;
                let combined = format!("{}\n{addition}", entry.content);
                mem.update_content(&entry.id, &combined)
                    .await
                    .map_err(|e| e.to_string())?;
                Ok(json!({ "id": entry.id, "total": combined.len() }))
            })
        }),
    );

    let mem_replace = mem.clone();
    services.insert(
        "memory.replace".to_string(),
        Arc::new(move |params: Value| {
            let mem = mem_replace.clone();
            Box::pin(async move {
                let handle = req_text(&params, "id")?;
                let old = req_text(&params, "old")?;
                let new = opt_text(&params, "new")?.unwrap_or_default();
                let entry = resolve_memory_handle(&mem, &handle).await?;
                let hits = entry.content.matches(&old).count();
                if hits == 0 {
                    // Same contract as edit_file: refuse rather than
                    // guess, and say what is actually there.
                    let preview: String = entry.content.chars().take(160).collect();
                    return Err(format!(
                        "'{old}' does not appear in memory {}. Nothing was changed. It \
                         begins: {preview}",
                        entry.id
                    ));
                }
                let updated = entry.content.replace(&old, &new);
                mem.update_content(&entry.id, &updated)
                    .await
                    .map_err(|e| e.to_string())?;
                Ok(json!({
                    "id": entry.id,
                    "replaced": hits,
                    "total": updated.len(),
                }))
            })
        }),
    );

    services
}

/// `memory.summarize`: concatenate texts and summarize them with the
/// summarization model chain.
///
/// The `day_dream` tool is the model-facing half: dreaming already does this on
/// a schedule, and this lets the model ask for it deliberately when it notices
/// related fragments piling up.
fn memory_summarize_services(
    router: Arc<crate::llm_router::LlmRouter>,
    summarizer_config: Arc<tokio::sync::RwLock<crate::agent_service::AgentServiceConfig>>,
) -> HashMap<String, ServiceFn> {
    use serde_json::{Value, json};

    let mut services: HashMap<String, ServiceFn> = HashMap::new();

    services.insert(
        "memory.summarize".to_string(),
        Arc::new(move |params: Value| {
            let router = router.clone();
            let summarizer_config = Arc::clone(&summarizer_config);
            Box::pin(async move {
                let texts: Vec<String> = params
                    .get("texts")
                    .and_then(|v| v.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_str().map(std::string::ToString::to_string))
                            .collect()
                    })
                    .unwrap_or_default();
                if texts.is_empty() {
                    return Err("texts must be a non-empty array".to_string());
                }
                let joined = texts.join("\n\n---\n\n");
                // Resolved per call: whichever summarization list the
                // user has set right now is the one that answers.
                let models =
                    crate::dream_summarizer::for_agent_service(&*summarizer_config.read().await);
                if models.is_empty() {
                    return Err(crate::agent_service::NO_MODEL_CONFIGURED.to_string());
                }
                let summarize =
                    crate::dream_summarizer::summarize_with_failover(router, models);
                let summary = summarize(joined).await?;
                Ok(json!({ "summary": summary }))
            })
        }),
    );

    services
}

/// `agent.spawn`: hand a prompt to a sub-agent and wait for its final text.
fn agent_spawn_services(
    spawner: Arc<dyn AgentSpawner + Send + Sync>,
) -> HashMap<String, ServiceFn> {
    use serde_json::{Value, json};

    let mut services: HashMap<String, ServiceFn> = HashMap::new();

    services.insert(
        "agent.spawn".to_string(),
        Arc::new(move |params: Value| {
            let spawner = spawner.clone();
            Box::pin(async move {
                let prompt = params
                    .get("prompt")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let description = params
                    .get("description")
                    .and_then(|v| v.as_str())
                    .unwrap_or("sub-task")
                    .to_string();
                let max_iterations = opt_count(&params, "max_iterations")?;
                match spawner.spawn(&prompt, &description, max_iterations).await {
                    Ok(result) => Ok(json!({
                        "text": result.text,
                        "iterations": result.iterations,
                        "tool_calls": result.tool_calls,
                        "model": result.model,
                    })),
                    Err(e) => Err(e),
                }
            })
        }),
    );

    services
}

/// Longest `python.exec` run a caller may ask for.
///
/// Bound justification: five minutes covers a real computation (a data
/// transform, a small simulation) while a model that asks for hours cannot
/// hold the call — and, until the engine's timeout kills, the interpreter —
/// for that long.
const PYTHON_EXEC_TIMEOUT_MAX_SECS: u64 = 300;

/// `python.exec`: the embedded interpreter (no system Python required).
fn python_exec_services() -> HashMap<String, ServiceFn> {
    use nanna_scripting::python::PythonEngine;
    use serde_json::{Value, json};

    let mut services: HashMap<String, ServiceFn> = HashMap::new();

    let python_engine = Arc::new(PythonEngine::new());
    services.insert(
        "python.exec".to_string(),
        Arc::new(move |params: Value| {
            let engine = python_engine.clone();
            Box::pin(async move {
                let code = params
                    .get("code")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                // Model-supplied, so bounded: an uncapped value held the call
                // (and its interpreter) for as long as the model asked.
                let timeout = opt_count(&params, "timeout")?
                    .map_or(30, |secs| u64::try_from(secs).unwrap_or(u64::MAX))
                    .min(PYTHON_EXEC_TIMEOUT_MAX_SECS);
                let workdir = params
                    .get("workdir")
                    .and_then(|v| v.as_str())
                    .map(String::from);

                match engine.execute(&code, workdir.as_deref(), timeout).await {
                    Ok(result) => Ok(json!({
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "success": result.success,
                        "error": result.error,
                        "duration_ms": result.duration_ms,
                    })),
                    Err(e) => Err(e.to_string()),
                }
            })
        }),
    );

    services
}

/// `session.history`: recent messages from the current session.
///
/// The [`SharedSessionHistory`] is populated before each agent run.
fn session_history_services(session_history: SharedSessionHistory) -> HashMap<String, ServiceFn> {
    use serde_json::{Value, json};

    let mut services: HashMap<String, ServiceFn> = HashMap::new();

    let history = session_history;
    services.insert(
        "session.history".to_string(),
        Arc::new(move |params: Value| {
            let history = history.clone();
            Box::pin(async move {
                let limit = opt_count(&params, "limit")?.unwrap_or(20);
                let history = history.read().await;
                let start = if history.len() > limit {
                    history.len() - limit
                } else {
                    0
                };
                let messages: Vec<Value> = history[start..]
                    .iter()
                    .map(|msg| {
                        json!({
                            "role": format!("{:?}", msg.role).to_lowercase(),
                            "content": msg.content,
                            "timestamp": msg.timestamp.to_rfc3339(),
                        })
                    })
                    .collect();
                // The snapshot is taken; nothing below reads the history, so
                // the read guard is released before the JSON is built.
                drop(history);
                Ok(json!(messages))
            })
        }),
    );

    services
}

/// `pdf.read`: text extraction, with an OCR fallback for image-only pages.
///
/// The bundled `read_pdf` skill declares `requires: ["pdf.read"]` and no such
/// service existed, so the tool loaded, advertised itself to the model, and
/// failed at call time with "PDF reading service not available". The Rust
/// extractor behind it has been complete and tested the whole time; only this
/// registration was missing.
///
/// OCR fallback for image-only pages, wired 2026-09-15 to the same vision model
/// `vision.analyze` uses. The thing that held it back for two runs was not the
/// pipeline but the ambiguity: a half-wired OCR path is indistinguishable from a
/// scanned document that genuinely has no text. So the response reports the
/// outcome as one of four named cases rather than returning an empty string for
/// all of them.
fn pdf_read_services(pdf_ocr: Option<nanna_tools::PdfOcrFn>) -> HashMap<String, ServiceFn> {
    use serde_json::{Value, json};

    let mut services: HashMap<String, ServiceFn> = HashMap::new();

    let pdf_ocr_fn = pdf_ocr;
    services.insert(
        "pdf.read".to_string(),
        Arc::new(move |params: Value| {
            let pdf_ocr_fn = pdf_ocr_fn.clone();
            Box::pin(async move {
                let path = params
                    .get("path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if path.trim().is_empty() {
                    return Err("pdf.read requires a `path`".to_string());
                }
                let selection = match params.get("pages").and_then(|v| v.as_str()) {
                    Some(spec) => nanna_tools::parse_page_selection(spec)?,
                    None => nanna_tools::PageSelection::All,
                };

                // Same ceiling `read_file` applies, checked before the read so
                // an oversized document is refused rather than buffered.
                let size_bytes = tokio::fs::metadata(&path)
                    .await
                    .map_err(|e| format!("Cannot stat {path}: {e}"))?
                    .len();
                let max_bytes = nanna_tools::PDF_MAX_BYTES;
                if size_bytes > max_bytes as u64 {
                    return Err(format!(
                        "PDF too large: {size_bytes} bytes (max: {max_bytes} bytes)"
                    ));
                }
                let bytes = tokio::fs::read(&path)
                    .await
                    .map_err(|e| format!("Cannot read {path}: {e}"))?;

                // lopdf parsing is synchronous and can take real time on a
                // large document, so it must not sit on a runtime worker.
                let parse_bytes = bytes.clone();
                let extracted = tokio::task::spawn_blocking(move || {
                    nanna_tools::read_pdf_text(&parse_bytes, selection)
                })
                .await
                .map_err(|e| format!("PDF read task failed: {e}"))?
                .map_err(|e| e.to_string())?;

                let ocr = nanna_tools::ocr_empty_pages(
                    &bytes,
                    selection,
                    extracted.empty_pages.len(),
                    pdf_ocr_fn.as_ref(),
                )
                .await
                .map_err(|e| e.to_string())?;

                // The counts ride alongside the text for the same reason
                // `read_file` reports `total_lines`: a caller must be able to
                // see it did not get the whole document without inferring it.
                // Four named outcomes, not one empty string: "no OCR
                // configured", "OCR ran and found nothing", "the document has
                // no images to OCR" and "no page was missing text" are
                // different answers, and a caller that cannot tell them apart
                // cannot tell an unreadable scan from an empty one.
                let mut response = json!({
                    "text": extracted.text,
                    "page_count": extracted.page_count,
                    "pages_read": extracted.pages_read,
                    "empty_pages": extracted.empty_pages.len(),
                });
                match ocr {
                    nanna_tools::PdfOcrOutcome::NotNeeded => {
                        response["ocr"] = json!("not_needed");
                    }
                    nanna_tools::PdfOcrOutcome::Unavailable => {
                        response["ocr"] = json!("unavailable");
                        response["ocr_note"] = json!(
                            "Pages had no extractable text and no OCR model is \
                             configured. Set [memory] ocr_model_priority to read \
                             image-only pages."
                        );
                    }
                    nanna_tools::PdfOcrOutcome::NoImages => {
                        response["ocr"] = json!("no_images");
                        response["ocr_note"] = json!(
                            "Pages had no extractable text and the document \
                             carries no embedded images, so this is not a scan \
                             OCR could recover."
                        );
                    }
                    nanna_tools::PdfOcrOutcome::Ran { text, images_read } => {
                        response["ocr"] = json!("ran");
                        response["ocr_images"] = json!(images_read);
                        response["ocr_text"] = json!(text);
                    }
                }
                Ok(response)
            })
        }),
    );

    services
}

fn build_script_services(deps: ScriptServiceDeps) -> HashMap<String, ServiceFn> {
    let ScriptServiceDeps {
        memory,
        spawner,
        session_history,
        workspace_id,
        storage,
        turn_baselines,
        summarizer,
        tool_authoring,
        vision,
        pdf_ocr,
        audio,
        browser_data_dir,
        screenshot_data_dir,
        reminders,
        ask_user,
        feedback,
    } = deps;
    let memory = &memory;

    let mut services: HashMap<String, ServiceFn> = HashMap::new();

    // Task store services (P15): the todo skill's backend. Only available
    // with storage — the skill falls back to its JSON file otherwise.
    if let Some(storage) = storage {
        services.extend(crate::tasks::build_task_services(
            storage,
            workspace_id.clone(),
            turn_baselines,
        ));
    }

    // Memory services
    if let Some(mem) = memory {
        services.extend(memory_write_services(mem, workspace_id.clone()));
        // Last reader of `workspace_id` in this function, so it moves.
        services.extend(memory_search_services(mem, workspace_id));
        services.extend(memory_index_services(mem));
    }

    // memory.get — read ONE memory by id, with a byte range.
    //
    // The store had only "search by similarity" and "list everything", which
    // is why a tool result kept in memory could not be pointed at from
    // context: a stub naming an id had no way to dereference it. This is the
    // first piece of a file-like surface (read a range, later append/replace)
    // so "the full result lives in memory, a stub lives in context" actually
    // has a retrieval path.
    if let Some(mem) = memory {
        services.extend(memory_read_services(mem, feedback));
    }

    // memory.append / memory.replace — the write half of the file-like
    // surface. Without them a memory can only be created or forgotten, so a
    // record that has become WRONG (an action narrated as a fact, e.g.
    // "creating minidb.sh at D:\…" nine hours after that file stopped
    // existing) can only be duplicated or destroyed, never corrected. Append
    // gives a running record per subject instead of N disconnected islands.
    if let Some(mem) = memory {
        services.extend(memory_edit_services(mem));
    }

    // memory.summarize — concatenate texts and summarize them with the
    // summarization model chain. The `day_dream` tool is the model-facing
    // half: dreaming already does this on a schedule, and this lets the model
    // ask for it deliberately when it notices related fragments piling up.
    if let Some((router, summarizer_config)) = summarizer {
        services.extend(memory_summarize_services(router, summarizer_config));
    }

    // Agent spawner service
    if let Some(spawner) = spawner {
        services.extend(agent_spawn_services(spawner));
    }

    // Embedded Python interpreter (no system Python required)
    services.extend(python_exec_services());

    // Session history service — returns recent messages from the current session.
    // The SharedSessionHistory is populated before each agent run.
    services.extend(session_history_services(session_history));

    // File history. The bridge snapshots before every script write once a
    // store is installed; these let the `file_history` skill list and restore.
    services.extend(crate::file_history_service::build_file_history_services(
        nanna_scripting::file_history::installed(),
    ));

    // Clarifying questions, answered by the user's next message to a live turn.
    if let Some(deps) = ask_user {
        services.extend(crate::ask_user_service::build_ask_user_services(deps));
    }

    // Reminders. `remind` / `list_reminders` / `cancel_reminder` declare
    // these; the scheduler is built after this map, so it arrives by slot.
    if let Some((scheduler, sessions)) = reminders {
        services.extend(crate::reminder_service::build_reminder_services(
            scheduler, sessions,
        ));
    }

    // Desktop capture. The `screenshot` skill declares this; the Rust tool
    // behind it was a stub, so this is the implementation, not a registration.
    if let Some(data_dir) = screenshot_data_dir {
        services.extend(crate::screenshot_service::build_screenshot_services(
            &data_dir,
        ));
    }

    // Browser. The four browser_* skills declare these; nanna-browser was
    // complete and the `browser` feature was enabled nowhere.
    if let Some(data_dir) = browser_data_dir {
        services.extend(crate::browser_service::build_browser_services(&data_dir));
    }

    // Audio. `text_to_speech` and `transcribe` declare these; the OpenAI
    // clients behind them were complete and reachable from nowhere.
    if let Some((openai_api_key, data_dir)) = audio {
        services.extend(crate::audio_service::build_audio_services(
            openai_api_key.as_deref(),
            &data_dir,
        ));
    }

    // Vision. The bundled `analyze_image`, `describe_image` and `ocr` skills all
    // declare `vision.analyze`, and nothing registered it.
    if let Some((router, models)) = vision {
        services.extend(crate::vision_service::build_vision_services(
            &router, &models,
        ));
    }

    // Tool authoring. The bundled `create_tool`, `edit_tool` and
    // `list_user_tools` skills declare `tools.create` / `tools.update` /
    // `tools.list`, and nothing registered any of them, so all three were
    // withheld at every boot (found by `skill_services_are_registered.rs`).
    if let Some((tools_dir, registry, slot)) = tool_authoring {
        services.extend(crate::tool_authoring::build_tool_authoring_services(
            tools_dir, registry, slot,
        ));
    }

    // PDF text extraction. The bundled `read_pdf` skill declares
    // `requires: ["pdf.read"]` and no such service existed, so the tool
    // loaded, advertised itself to the model, and failed at call time with
    // "PDF reading service not available". The Rust extractor behind it has
    // been complete and tested the whole time; only this registration was
    // missing.
    //
    // OCR fallback for image-only pages, wired 2026-09-15 to the same vision
    // model `vision.analyze` uses. The thing that held it back for two runs was
    // not the pipeline but the ambiguity: a half-wired OCR path is
    // indistinguishable from a scanned document that genuinely has no text. So
    // the response reports the outcome as one of four named cases rather than
    // returning an empty string for all of them.
    services.extend(pdf_read_services(pdf_ocr));

    services
}
use tokio::sync::broadcast;
use tracing::{debug, error, info, warn};

/// Which auxiliary listeners the daemon brings up, and whether it claims the
/// single-instance PID file.
///
/// Grouped rather than left as three loose `bool` fields on [`DaemonConfig`]:
/// they are all "does this process own a second thing besides the IPC socket",
/// they are all flipped together by the `--no-health-server` / `--no-pid-file`
/// / `--enable-webhooks` flags, and a flat row of same-typed booleans is
/// exactly the shape a caller can transpose silently.
#[derive(Debug, Clone, Copy)]
pub struct ServerSwitches {
    /// Enable HTTP health server
    pub health_server: bool,
    /// Enable webhook server for inbound messages
    pub webhook_server: bool,
    /// Enable PID file (prevents multiple instances)
    pub pid_file: bool,
}

impl Default for ServerSwitches {
    fn default() -> Self {
        Self {
            health_server: true,
            pid_file: true,
            // Disabled by default (needs configuration)
            webhook_server: false,
        }
    }
}

/// The per-call tool audit trail's two settings: whether it is written at all,
/// and whether it records argument values.
///
/// One is meaningless without the other — `log_values` does nothing while `log`
/// is false — so they travel together rather than as two independent booleans a
/// caller can set into a contradictory pair.
#[derive(Debug, Clone, Copy)]
pub struct ToolAuditSwitches {
    /// Append one JSON line per tool call to `{data_dir}/logs/tool-audit.jsonl`.
    /// Mirrors `[tools] audit_log`.
    pub log: bool,
    /// Include a bounded preview of tool arguments in that trail.
    /// Mirrors `[tools] audit_log_values`.
    pub log_values: bool,
}

impl Default for ToolAuditSwitches {
    fn default() -> Self {
        Self {
            log: true,
            log_values: false,
        }
    }
}

/// The scheduler's two master switches, mirroring `[scheduler]`.
///
/// The heartbeat is subordinate to the scheduler — `heartbeat_enabled` cannot
/// fire anything while `enabled` is false — so the pair is one setting with two
/// levels, not two unrelated flags.
#[derive(Debug, Clone, Copy)]
pub struct SchedulerSwitches {
    /// Master switch for the daemon's scheduler (mirrors `[scheduler] enabled`).
    /// `false` loads cron jobs but fires nothing.
    pub enabled: bool,
    /// Whether the periodic heartbeat runs (mirrors
    /// `[scheduler] heartbeat_enabled`). The heartbeat drives a full agent turn
    /// against the chat model, so on a single-slot local backend it competes
    /// with live conversation — hence a user-visible switch.
    pub heartbeat_enabled: bool,
}

impl Default for SchedulerSwitches {
    fn default() -> Self {
        // Mirror `nanna_config::SchedulerConfig::default()`.
        Self {
            enabled: true,
            heartbeat_enabled: true,
        }
    }
}

/// Configuration for the daemon server
#[derive(Debug, Clone)]
pub struct DaemonConfig {
    /// IPC server configuration
    pub ipc: IpcServerConfig,
    /// Data directory
    pub data_dir: PathBuf,
    /// Log level
    pub log_level: String,
    /// Auto-save interval in seconds
    pub auto_save_interval_secs: u64,
    /// LLM configuration
    pub llm: LlmConfig,
    /// Agent configuration
    pub agent: AgentServiceConfig,
    /// Enable memory service (requires embedding provider)
    pub enable_memory: bool,
    /// Which auxiliary listeners run, and whether the PID file is claimed
    pub servers: ServerSwitches,
    /// Health server port (default: 5148)
    pub health_port: u16,
    /// Webhook server port (default: 3000)
    pub webhook_port: u16,
    /// Webhook configuration
    pub webhook: WebhookConfig,
    /// Use TypeScript skill implementations instead of Rust builtins
    pub use_script_tools: bool,
    /// Directory containing tool scripts (resolved from env/config/default)
    pub tools_dir: Option<PathBuf>,
    /// Vision-capable models for `vision.analyze`, in priority order.
    ///
    /// Flattened from `[memory] ocr_model_priority` the same way the other
    /// `memory_*` values below are — that setting already means "vision-capable
    /// models, tried in order", so the vision service reads it rather than
    /// introducing a second list to keep in sync. Empty (the default) leaves
    /// `vision.analyze` unregistered and the vision skills withheld.
    pub vision_model_priority: Vec<String>,
    /// Tool names the agent may call. `None` (or a list containing `"*"`) means
    /// "no allowlist — every tool is permitted". Mirrors `[tools] enabled`.
    pub tool_allowlist: Option<Vec<String>>,
    /// Tool names the agent may never call. Takes precedence over the allowlist.
    /// Mirrors `[tools] disabled` — this is the setting that makes a disabled
    /// tool actually stop executing (previously the list was parsed but ignored).
    pub tool_denylist: Vec<String>,
    /// Whether the per-call tool audit trail is written, and how much of each
    /// call it records.
    pub tool_audit: ToolAuditSwitches,
    /// Channel configurations (Telegram, Discord, Slack, etc.)
    pub channels: Option<nanna_config::ChannelsConfig>,
    /// Max fraction of memories the scheduled dream cycle may merge away in one
    /// run (mirrors `[memory] max_compression_ratio`). Threaded so automatic
    /// consolidation honors the same user setting the IPC-triggered path does.
    pub memory_max_compression_ratio: f32,
    /// Floor the scheduled dream cycle leaves after consolidating (mirrors
    /// `[memory] min_remaining_memories`).
    pub memory_min_remaining_memories: usize,
    /// Seconds of idle (no chat activity) before the scheduled dream cycle may
    /// run (mirrors `[memory] dream_idle_threshold_secs`). Gated via the shared
    /// [`ActivityClock`](nanna_memory::ActivityClock) + `nanna_memory::dream_trigger`.
    pub dream_idle_threshold_secs: u64,
    /// Live memory count that forces a dream cycle regardless of idle time
    /// (mirrors `[memory] dream_memory_pressure_count`; `0` disables).
    pub dream_memory_pressure_count: usize,
    /// MCP servers to start at boot (`[mcp]`).
    pub mcp: nanna_config::McpConfig,
    /// The scheduler's master switch and its heartbeat switch (`[scheduler]`).
    pub scheduler: SchedulerSwitches,
    /// Seconds between heartbeats (mirrors `[scheduler] heartbeat_interval_secs`).
    /// Clamped up to [`nanna_core::MIN_HEARTBEAT_INTERVAL_SECS`] when applied.
    pub heartbeat_interval_secs: u64,
}

/// LLM provider configuration (multi-provider)
#[derive(Debug, Clone)]
pub struct LlmConfig {
    /// Primary provider (anthropic, openai, ollama) - used for single-provider mode
    pub provider: String,
    /// Anthropic API key
    pub anthropic_api_key: Option<String>,
    /// Anthropic OAuth access token (alternative to API key)
    pub anthropic_oauth_token: Option<String>,
    /// Whether to use OAuth token instead of API key for Anthropic
    pub anthropic_use_oauth: bool,
    /// `OpenAI` API key
    pub openai_api_key: Option<String>,
    /// `OpenRouter` API key
    pub openrouter_api_key: Option<String>,
    /// GitHub token (for GitHub Models)
    pub github_token: Option<String>,
    /// Ollama host
    pub ollama_host: String,
    /// Ollama API key (optional — for remote/authenticated instances)
    pub ollama_api_key: Option<String>,
    /// Legacy: API key field (for backwards compatibility)
    pub api_key: Option<String>,
}

/// Record `configured_host` as the server of a saved Ollama token that
/// records none — one saved by a build before tokens were bound.
///
/// Such a token has been going to the configured server, and every load reads
/// it as the configured server's. Unrecorded, that stays true of whatever
/// address is configured next, including one edited into `config.toml` while
/// the daemon is stopped. Recorded once at the first start, it stays with the
/// server it was going to. A failure is logged and changes nothing: the next
/// start tries again, and reloads while running attribute it to the running
/// server anyway.
fn record_legacy_ollama_token_server(configured_host: &str) {
    match SecureStore::new().bind_unbound_ollama_token(configured_host) {
        Ok(true) => info!(
            "Recorded {} as the server the saved Ollama token belongs to",
            nanna_config::ollama::redacted_ollama_host(configured_host)
        ),
        Ok(false) => {}
        Err(e) => warn!("Could not record which server the saved Ollama token belongs to: {e}"),
    }
}

/// A credential from the environment, falling back to the secure store.
///
/// The store is where the GUI puts keys the user types in, so a key that is
/// only ever read from the environment is a key the user cannot set. Anthropic
/// and `OpenAI` already got their store fallback further down this file; the
/// others never did, and `OpenRouter`'s absence was load-bearing — the dream
/// summarizer is configured to `OpenRouter` models by default, so every
/// consolidation failed with `Missing API key for provider: OpenRouter` while
/// the key sat in the store the whole time. Dreaming had never once run.
fn credential(env_var: &str, store_key: &str) -> Option<String> {
    if let Ok(value) = std::env::var(env_var)
        && !value.trim().is_empty()
    {
        return Some(value);
    }
    SecureStore::new()
        .get(store_key)
        .ok()
        .filter(|value| !value.trim().is_empty())
}

impl LlmConfig {
    /// Build the daemon's LLM credential view from the user config file.
    ///
    /// The single source of truth for this field mapping — used at boot
    /// ([`DaemonBuilder::from_nanna_config`]) and on every control-plane config
    /// mutation (`control::config`), so the provider set the router rebuilds
    /// from is always derived exactly the way boot derived it.
    #[must_use]
    pub fn from_nanna(config: &nanna_config::Config) -> Self {
        Self {
            provider: config.llm.provider.clone(),
            anthropic_api_key: config.llm.api_key.clone(),
            anthropic_oauth_token: config.llm.anthropic_oauth_token.clone(),
            anthropic_use_oauth: config.llm.anthropic_use_oauth,
            openai_api_key: config.llm.openai_api_key.clone(),
            openrouter_api_key: config.llm.openrouter_api_key.clone(),
            github_token: config.llm.github_token.clone(),
            // Ollama host is stored in memory config
            ollama_host: config.memory.ollama_host.clone(),
            ollama_api_key: config.llm.ollama_api_key.clone(),
            api_key: config.llm.api_key.clone(),
        }
    }
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            provider: "anthropic".to_string(),
            anthropic_api_key: credential("ANTHROPIC_API_KEY", credentials::keys::ANTHROPIC_API_KEY),
            anthropic_oauth_token: None,
            anthropic_use_oauth: false,
            openai_api_key: credential("OPENAI_API_KEY", credentials::keys::OPENAI_API_KEY),
            openrouter_api_key: credential(
                "OPENROUTER_API_KEY",
                credentials::keys::OPENROUTER_API_KEY,
            ),
            github_token: credential("GITHUB_TOKEN", credentials::keys::GITHUB_TOKEN),
            ollama_host: "http://localhost:11434".to_string(),
            ollama_api_key: std::env::var("OLLAMA_API_KEY").ok(),
            api_key: None, // Legacy
        }
    }
}

impl DaemonConfig {
    /// Where the per-call tool audit trail is written.
    ///
    /// One definition, because two consumers need it and they are built in
    /// different places: `init_services` hands it to the [`JsonlAuditSink`](nanna_tools::JsonlAuditSink), and
    /// the control plane needs it to serve the trail back to a client. Deriving
    /// it twice is how the reader ends up looking somewhere the writer never
    /// wrote — and note this hangs off `data_dir`, which `--data-dir` moves, so
    /// re-deriving from `Config::default_data_dir()` would read an empty trail
    /// on every isolated run.
    #[must_use]
    pub fn tool_audit_path(&self) -> PathBuf {
        self.data_dir.join("logs").join("tool-audit.jsonl")
    }
}

impl Default for DaemonConfig {
    fn default() -> Self {
        let data_dir = nanna_config::project_dirs()
            .map_or_else(|| PathBuf::from("./data"), |d| d.data_dir().to_path_buf());

        Self {
            ipc: IpcServerConfig::default(),
            data_dir,
            log_level: "info".to_string(),
            auto_save_interval_secs: 60,
            llm: LlmConfig::default(),
            agent: AgentServiceConfig::default(),
            enable_memory: true, // Enabled by default (requires embedding provider)
            servers: ServerSwitches::default(),
            health_port: DEFAULT_HEALTH_PORT,
            webhook_port: DEFAULT_WEBHOOK_PORT,
            webhook: WebhookConfig::default(),
            use_script_tools: true,
            tools_dir: None,
            vision_model_priority: Vec::new(),
            tool_allowlist: None,
            tool_denylist: Vec::new(),
            tool_audit: ToolAuditSwitches::default(),
            channels: None,
            // Mirror ConsolidationConfig::default() (== nanna-config defaults).
            memory_max_compression_ratio: 0.50,
            memory_min_remaining_memories: 20,
            // Mirror DreamingConfig::default() (== nanna-config defaults).
            dream_idle_threshold_secs: 300,
            dream_memory_pressure_count: 5000,
            mcp: nanna_config::McpConfig::default(),
            // Mirror nanna_config::SchedulerConfig::default().
            scheduler: SchedulerSwitches::default(),
            heartbeat_interval_secs: 1800,
        }
    }
}

/// Build the scheduled dream-cycle's [`ConsolidationConfig`] from the user's
/// memory settings, keeping automatic consolidation in lock-step with the
/// IPC-triggered path (see `control.rs`). The per-cluster content budget is
/// sized to the summarizer model's context window so the consolidation prompt
/// always fits it. Pure so it is unit-testable.
fn scheduled_consolidation_config(
    max_compression_ratio: f32,
    min_remaining_memories: usize,
    summarizer_context_window_tokens: usize,
) -> nanna_memory::ConsolidationConfig {
    nanna_memory::ConsolidationConfig {
        max_compression_ratio,
        min_remaining_memories,
        ..nanna_memory::ConsolidationConfig::default()
    }
    .with_summarizer_context_window(summarizer_context_window_tokens)
}

/// One provider request per backfill pass.
///
/// Bound justification: the drain's contention discipline is expressed PER
/// REQUEST — the admission gate must be able to interpose between any two
/// provider calls, and the repayment window is measured from one call's RTT.
/// A batch would put a live chat turn behind the whole batch (the 2026-08-10
/// storm was exactly three 64-entry batches: 201 POSTs in one minute) and
/// would average the RTT signal the pacing derives its bound from. The cost —
/// one pending-scan query per entry against local Turso — is microseconds
/// next to the network round-trip it now paces.
const DRAIN_STEP: usize = 1;

/// Fill `model`'s missing embeddings until the store is complete for it, the
/// provider stalls, or the binding moves on — the service refuses to fill a
/// bucket that is not the active binding, so a stale drain exits on its next
/// pass instead of poisoning a dead provider's bucket.
///
/// Contention discipline (P22 Tier 4; evidence: an embedding rebind fired 201
/// `POST /api/embed` in one minute while a mission's opening turn queued
/// behind them):
///
/// - **One drain process-wide** (`drain_serial`): the rebind path, the
///   startup bind, and the dimension-probe correction all spawn drains, and
///   concurrent drains would multiply the very request rate the pacing below
///   bounds. Queued drains re-check the binding and no-op fast when stale.
/// - **The drain yields to a live chat turn**: when the embedding provider is
///   the local one, every pass first waits until no harness run is live.
///   Priority, not a quota — it resumes the moment the turn releases, and a
///   turn arriving mid-pass waits at most one in-flight embed. Remote
///   embedding providers share nothing with a local generation slot, so they
///   skip the gate (their writes contend with nobody's GPU).
/// - **Each request repays its cost**: a pass that consulted the provider is
///   followed by an idle window equal to the time the provider spent serving
///   it. That caps the drain at half of the provider's wall-clock in ANY
///   window, with no fixed rate constant: the bound is derived from the
///   observed round-trip itself, so a loaded provider answering slowly earns
///   proportionally longer gaps — the brake tightens exactly when contention
///   rises, and a fast idle provider drains near full speed. In-flight stays
///   at one for the same reason: the ceiling is expressed in time, and a
///   second in-flight request would breach it by construction.
async fn drain_backfill(
    mem: &Arc<MemoryService>,
    model: &str,
    chat_runs: &Arc<crate::control::chat_harness::ChatRunRegistry>,
    drain_serial: &Arc<tokio::sync::Mutex<()>>,
) {
    // "provider:model" is the router's spec shape; the local provider is the
    // one that shares its single generation slot with chat.
    let local = model.starts_with("ollama:");

    loop {
        // The yield happens OUTSIDE `drain_serial`, and this is load-bearing.
        // Parking on `wait_idle` while holding the one process-wide drain lock
        // means a drain waiting for a mission to end holds that lock for the
        // length of the mission — starving every other drain behind it,
        // including the bounded foreground one ([`drain_queued_vectors`])
        // whose entire reason to exist is not to wait for that mission. The
        // lock still serializes the passes, and the repayment sleep still
        // happens under it, so both invariants above — one drain at a time,
        // at most half the provider's wall-clock — are unchanged.
        if local {
            chat_runs.wait_idle().await;
        }
        let _one_drain = drain_serial.lock().await;
        let pass_started = std::time::Instant::now();
        match mem.backfill_embeddings(model, DRAIN_STEP).await {
            Ok(0) => break,
            Ok(_) => tokio::time::sleep(pass_started.elapsed()).await,
            Err(e) => {
                warn!("Backfill for '{model}' halted: {e}");
                break;
            }
        }
    }

    // Chunk vectors drain in the same pass, after the whole-row ones.
    //
    // Row vectors first because they are what search falls back to: until a
    // memory has one, it is invisible to recall entirely, whereas a memory
    // with a row vector but no chunk vectors is merely coarser to retrieve.
    // Draining eagerly rather than on demand is deliberate — a lazy chunk
    // backfill would put the embed latency of a whole memory inside the first
    // recall that happened to touch it.
    loop {
        if local {
            chat_runs.wait_idle().await;
        }
        let _one_drain = drain_serial.lock().await;
        let pass_started = std::time::Instant::now();
        match mem.backfill_chunks(model, DRAIN_STEP).await {
            Ok((0, 0)) => break,
            Ok((embedded, _chunked)) => {
                // Chunking alone is local CPU — only a pass that actually
                // consulted the provider owes it an idle window.
                if embedded > 0 {
                    tokio::time::sleep(pass_started.elapsed()).await;
                }
            }
            Err(e) => {
                warn!("Chunk backfill for '{model}' halted: {e}");
                break;
            }
        }
    }
}

/// Embed the rows a LIVE turn parked, without making them wait for that turn
/// to end (P24.3 part 3).
///
/// Tool-result ingestion no longer embeds inline: a chunk is persisted
/// immediately and its vector is queued
/// ([`MemoryService::remember_deferred_vector`]). Something has to pick those
/// vectors up, and [`drain_backfill`] cannot be it — when the embedder is the
/// local provider that drain first waits for **no harness run to be live**, so
/// a row queued *by* a live run would wait for the run that queued it. During
/// a long autonomous mission that is hours, and for those hours the model is
/// being handed `recall(...)` handles to rows that similarity search cannot
/// see.
///
/// So this drain skips the yield gate — and pays for that with a bound the
/// yield gate was standing in for:
///
/// - **It may only embed rows this process parked.** The budget comes from
///   [`MemoryService::take_queued_vector_count`], so an inherited backlog (2167
///   rows, in the incident that motivated the queue) is NOT swept up at
///   foreground priority; that is still [`drain_backfill`]'s job at
///   [`drain_backfill`]'s priority. The exception is sized to the work the live
///   session created and nothing more.
/// - **It still repays every request.** Same rule as [`drain_backfill`]: a pass
///   that consulted the provider is followed by an idle window equal to that
///   call's RTT, so it can never exceed half the provider's wall-clock, and a
///   loaded provider answering slowly earns proportionally longer gaps.
/// - **It still takes `drain_serial`.** One drain's passes at a time,
///   process-wide, so this cannot multiply the request rate that pacing bounds.
/// - **Row vectors only.** Chunk vectors stay with [`drain_backfill`], for the
///   same reason that drain does rows before chunks: without a row vector a
///   memory is invisible to recall entirely, whereas one with a row vector and
///   no chunk vectors is merely coarser to retrieve. Coarser can wait for idle;
///   invisible cannot.
///
/// Net against what it replaces: the same embedding work, at half the duty
/// cycle, off the turn's critical path instead of blocking it.
async fn drain_queued_vectors(
    mem: &Arc<MemoryService>,
    drain_serial: &Arc<tokio::sync::Mutex<()>>,
) {
    loop {
        mem.vector_queue_notified().await;
        let mut budget_rows = mem.take_queued_vector_count();
        // A wake with an empty budget is not an error — `Notify` holds one
        // permit however many parks happened, so a drain that already took the
        // count can be woken once more with nothing to do.
        if budget_rows == 0 {
            continue;
        }
        let Some(model) = mem.active_binding().await.0 else {
            // No binding yet: the rows stay queued and the next binding event's
            // unconditional drain takes them. Putting the budget back would
            // spin, since nothing re-notifies.
            debug!("{budget_rows} queued vectors wait for an embedding binding");
            continue;
        };
        debug!("Draining {budget_rows} queued vectors for '{model}'");
        while budget_rows > 0 {
            let step = DRAIN_STEP.min(budget_rows);
            debug_assert!(step > 0, "a positive budget must yield a positive step");
            let _one_drain = drain_serial.lock().await;
            let pass_started = std::time::Instant::now();
            match mem.backfill_embeddings(&model, step).await {
                // Nothing left for this model — either the rows landed under a
                // different binding, or another drain got there first. Either
                // way the budget is stale, not owed.
                Ok(0) => break,
                Ok(embedded) => {
                    budget_rows = budget_rows.saturating_sub(embedded);
                    tokio::time::sleep(pass_started.elapsed()).await;
                }
                Err(e) => {
                    warn!("Queued-vector drain for '{model}' halted: {e}");
                    break;
                }
            }
        }
    }
}

/// How this differs from [`drain_queued_vectors`] above, since both end up
/// draining the same queue and only one of them is obvious:
///
/// `drain_queued_vectors` handles what THIS PROCESS deliberately deferred, at
/// foreground priority, during the turn -- and it is explicitly budgeted so it
/// will NOT sweep up an inherited backlog. Its own doc says that remainder is
/// "still `drain_backfill`'s job at `drain_backfill`'s priority".
///
/// The catch is that nothing was calling `drain_backfill` at that priority
/// during a session. Its only triggers are BINDING events -- daemon start,
/// provider switch, width reprobe -- and an ordinary session has none of them.
/// So a row parked by a *transient* embedding failure, or a backlog inherited
/// from a previous run, waited for a restart. That is the gap this closes, and
/// it is why the two coexist rather than one replacing the other: one drains
/// the work the turn created, the other drains everything else, at the first
/// moment no turn is live.
///
/// Drain rows that were parked mid-session, at the first moment no turn is live.
///
/// [`nanna_memory::MemoryService::store_unembedded`] is durable and honest —
/// the write lands, the row is reachable by its `source_id` handle, and it is
/// queued for backfill — but until now the ONLY things that started a drain
/// were **binding** events: daemon start, a provider switch, a width reprobe.
/// An ordinary session has none of those. So a row parked by a *transient*
/// embedding failure stayed unsearchable for the rest of the session, and the
/// store's own doc comment says exactly that: it "is recovered, not lost — but
/// the latency is a session, not a moment, and closing that needs a drain
/// trigger the memory crate does not own". This is that trigger, and it lives
/// here because the daemon is what owns both the run registry and the drain.
///
/// **Bound: one probe per active→idle cycle** — not per write, and not on a
/// timer. [`ChatRunRegistry::wait_active`] followed by
/// [`ChatRunRegistry::wait_idle`] is precisely one turn's lifetime, so the
/// probe rate is bounded above by the *turn* rate, which is bounded by a human
/// typing. A timer would have had to pick an interval that is either too slow
/// to matter or a poll; an edge does not.
///
/// The probe is what [`drain_backfill`] already costs on a store that is
/// complete for the bound model, and it is worth stating exactly rather than
/// as "a query": `entries_missing_model(model, 1)` is an **in-memory** scan of
/// the entries cache under a read lock — it short-circuits at the first entry
/// with no bucket for `model`, so it walks the whole cache only in the case
/// where there is nothing to do — followed by two `LIMIT 1` local Turso
/// queries (`parents_without_chunks`, `chunks_needing_embedding`). No provider
/// request is made unless work is actually found. At a 100k-entry store that
/// is single-digit milliseconds, once per turn.
///
/// **Worst case is one turn of latency, not one session.** `wait_active`
/// registers interest and then reads the flag, so a turn that begins AND ends
/// inside that window leaves no edge to observe and its rows wait for the next
/// turn. That is a bounded miss and a deliberate non-fix: the rows are durable
/// and handle-addressable the whole time, and the alternative — probing before
/// parking — turns the loop into a spin on an idle daemon, which is the exact
/// property this shape was chosen to avoid.
///
/// **It adds a trigger, never a second rate.** The drain it may start is the
/// existing [`drain_backfill`], so it inherits the process-wide `drain_serial`
/// mutex (one drain at a time), the `wait_idle` priority gate (a turn arriving
/// mid-drain takes the slot back within one in-flight embed) and the
/// per-request RTT repayment that caps the drain at half the provider's
/// wall-clock. Starting from an idle edge means the common case is that the
/// gate is already open and the queue is already empty.
async fn supervise_idle_backfill(
    memory: Arc<MemoryService>,
    chat_runs: Arc<crate::control::chat_harness::ChatRunRegistry>,
    drain_serial: Arc<tokio::sync::Mutex<()>>,
) {
    // Said once, at arm time, because the useful thing to know from a log is
    // that this exists at all. The drain it starts already announces its own
    // work ("Backfilled N embeddings"), and a per-turn line for the overwhelming
    // case -- nothing queued, nothing to do -- would be noise proportional to
    // conversation length.
    info!(
        "Idle-backfill supervisor armed: queued embedding rows now drain at the end of a turn"
    );
    loop {
        // A turn started, then every turn finished. Waiting on `active` FIRST is
        // what makes this an edge rather than a spin: with only `wait_idle`, an
        // already-idle daemon would loop as fast as the scheduler allows.
        chat_runs.wait_active().await;
        chat_runs.wait_idle().await;
        // Deliberately NOT asserted here: `!any_active()`. `wait_idle` certifies
        // that no run was live at the instant it returned -- a PAST instant. A
        // turn may legally claim the slot again before the next line runs, so
        // asserting it would fire on ordinary concurrency instead of on a
        // programmer error, and an assertion that can be false is worse than an
        // absent one. The property is real but belongs to the registry, where it
        // can be stated without a race: see the chat_harness test
        // `one_turn_drives_one_active_then_idle_cycle`. Nothing below needs the
        // daemon to still be idle -- the drain re-takes the gate itself, once
        // per provider request.

        // No binding means nothing can be drained INTO a bucket: the drain
        // refuses to fill a bucket that is not the active binding, so calling it
        // here would be a no-op with extra steps.
        let Some(model) = memory.active_embedding_model().await else {
            continue;
        };
        // Two things must hold about a binding before it is worth draining, and
        // both are far cheaper than the drain they guard, so they are checked
        // always rather than in debug only. The spec must name something; and
        // the drain must be able to tell a LOCAL provider from a remote one.
        // That second test is load-bearing rather than cosmetic --
        // `drain_backfill` decides whether to yield to a live chat turn by
        // reading the provider off this spec, so a spec with no provider segment
        // would silently classify the local embedder as remote and drain
        // straight through the contention gate that exists to keep it off the
        // shared generation slot.
        //
        // Both are ASSERTIONS rather than branches because both are structural,
        // not configuration-dependent: the binding string is produced by
        // `EmbeddingProviderInfo`'s `Display`, which is literally
        // `write!("{}:{}", name, model)`, and `drain_backfill` already reads the
        // prefix back off it. Neither can be made false by a config file, so
        // neither is a failure to handle -- only a way to notice if the shape
        // this depends on is ever changed somewhere else.
        assert!(
            !model.is_empty(),
            "the active embedding binding must name a model: an empty spec would \
             drain into an unaddressable bucket"
        );
        assert!(
            model.contains(':'),
            "the active embedding binding must be a `provider:model` spec (got \
             {model:?}): `drain_backfill` reads the provider off this string to \
             decide whether to yield the local generation slot to a live turn"
        );
        drain_backfill(&memory, &model, &chat_runs, &drain_serial).await;
    }
}

/// Run one scheduled agent prompt (heartbeat / cron), YIELDING the local
/// provider to a user turn that arrives mid-run (P22 Tier 4).
///
/// The evidence for existing at all: at one mission's start the daemon's own
/// heartbeat held the single local generation slot for 157 seconds while the
/// turn's planner timed out behind it. The tick-time gate cannot catch that
/// ordering — the heartbeat was already running when the user typed — so the
/// run itself must stand aside. Priority, not a quota: the user turn takes
/// the slot within one abortive cancel, and the yielded work is resumed by
/// the caller once the turn releases.
///
/// Mechanics: the agent run is SPAWNED, then raced against the registry's
/// became-active edge. Spawning matters — select-dropping the chat future
/// would skip its cleanup tail and leave the session registered as active
/// forever; instead, preemption cancels through the same abortive token the
/// Stop button uses (dropping the in-flight stream immediately) and then
/// AWAITS the run so it exits through its own tail.
///
/// Preemption is armed only when the agent's model is served by the local
/// single-slot provider: on a cloud provider a heartbeat and a chat coexist,
/// and cancelling one buys the other nothing.
///
/// Returns `None` when the run yielded (the caller owns resumption);
/// `Some(outcome)` when it finished on its own.
async fn run_scheduled_prompt_yielding(
    agent: &Arc<AgentService>,
    chat_runs: &Arc<crate::control::chat_harness::ChatRunRegistry>,
    session_id: &str,
    payload: &str,
) -> Option<Result<crate::agent_service::ChatResult, crate::agent_service::ChatError>> {
    let run_agent = agent.clone();
    let run_session = session_id.to_string();
    let run_payload = payload.to_string();
    let mut run = tokio::spawn(async move {
        // Session-scoped tools resolve their scope from this task-local
        // binding; carried by the run's own future so a chat starting during
        // this run cannot see the scheduled session (and vice versa).
        // Boxed: the chat future is ~23KB and would otherwise sit inline in
        // the spawned task's state machine.
        let run_span = tracing::info_span!("scheduled_run", session_id = %run_session);
        tracing::Instrument::instrument(
            Box::pin(ToolRegistry::with_run_session(
                run_session.clone(),
                run_agent.chat(&run_session, &run_payload, None, &[]),
            )),
            run_span,
        )
        .await
    });

    let local = crate::llm_router::ProviderId::from_model(&agent.agent_config().await.model)
        == crate::llm_router::ProviderId::Ollama;
    if !local {
        return Some(flatten_scheduled_join(run.await));
    }

    tokio::select! {
        joined = &mut run => Some(flatten_scheduled_join(joined)),
        () = chat_runs.wait_active() => {
            let yielding = std::time::Instant::now();
            info!(session_id, "Scheduled run yielding the local provider to a live chat turn");
            // The spawned run may not have registered its cancel token yet
            // (a turn can claim in the gap between spawn and registration),
            // and a cancel that lands on nothing would leave this arm
            // blocked behind the full run. Keep asking until the cancel
            // lands or the run ends on its own; `yield_now`, not a sleep —
            // registration is one task-poll away, so this converges in
            // microseconds and carries no tuned interval.
            while !agent.cancel(session_id).await && !run.is_finished() {
                tokio::task::yield_now().await;
            }
            // Orderly exit through the run's own cancel tail — never drop it.
            let _ = run.await;
            debug!(
                session_id,
                yield_secs = yielding.elapsed().as_secs_f64(),
                "Scheduled run yielded"
            );
            None
        }
    }
}

/// A panicked scheduled-run task is an error outcome, not a daemon crash.
fn flatten_scheduled_join(
    joined: Result<
        Result<crate::agent_service::ChatResult, crate::agent_service::ChatError>,
        tokio::task::JoinError,
    >,
) -> Result<crate::agent_service::ChatResult, crate::agent_service::ChatError> {
    joined.unwrap_or_else(|join_err| {
        Err(crate::agent_service::ChatError {
            message: format!("scheduled run task failed: {join_err}"),
            partial_result: None,
        })
    })
}

/// The user's `[memory]` dream-cycle settings, as the scheduled cycle reads them.
#[derive(Clone, Copy)]
struct DreamBudget {
    /// Mirrors `[memory] max_compression_ratio`.
    max_compression_ratio: f32,
    /// Mirrors `[memory] min_remaining_memories`.
    min_remaining_memories: usize,
    /// Carried for the skip log only — the gate *decision* lives in the
    /// [`nanna_memory::DreamingService`], built with both thresholds, so there
    /// is no second copy of the policy here.
    idle_threshold_secs: u64,
}

/// What the scheduled-task executor closure captures.
///
/// Ten collaborators, eight of them `Arc`s; named fields for the same reason
/// [`ScriptServiceDeps`] has them.
struct ScheduledTaskDeps {
    agent: Arc<AgentService>,
    router: Arc<LlmRouter>,
    dreaming: Option<Arc<nanna_memory::DreamingService>>,
    chat_runs: Arc<crate::control::chat_harness::ChatRunRegistry>,
    activity_clock: Arc<nanna_memory::ActivityClock>,
    storage: Option<Arc<nanna_storage::Storage>>,
    sessions: Arc<SessionManager>,
    events: broadcast::Sender<crate::protocol::Event>,
    /// The scheduler's own tick, which the reminder delivery window is derived
    /// from.
    reminder_tick: Duration,
    dream: DreamBudget,
}

/// What [`DaemonServer::start_scheduler`] needs from `run`'s locals.
struct SchedulerDeps {
    agent: Arc<AgentService>,
    router: Arc<LlmRouter>,
    dreaming: Option<Arc<nanna_memory::DreamingService>>,
    chat_runs: Arc<crate::control::chat_harness::ChatRunRegistry>,
    activity_clock: Arc<nanna_memory::ActivityClock>,
}

/// The `memory_consolidation` scheduled task: at most one dream cycle, gated
/// twice.
///
/// A live harness run is the opposite of idle however old the last user message
/// is — dreaming rewrites the very scoped memories the run is using, and doing
/// so mid-step deadlocked a live mission (2026-08-10, 316 tool-result memories
/// folded under a running step) — and the in-flight latch stops a tick firing a
/// second cycle while the first is still folding.
///
/// Returns the scheduler's `(success, output, error)` triple.
async fn run_scheduled_consolidation(
    dreaming: Option<&Arc<nanna_memory::DreamingService>>,
    agent: &Arc<AgentService>,
    router: &Arc<LlmRouter>,
    chat_runs: &Arc<crate::control::chat_harness::ChatRunRegistry>,
    activity: &Arc<nanna_memory::ActivityClock>,
    dream_in_flight: &Arc<std::sync::atomic::AtomicBool>,
    budget: DreamBudget,
) -> (bool, Option<String>, Option<String>) {
    let Some(dreaming) = dreaming else {
        return (
            true,
            Some("Skipped (memory service unavailable)".to_string()),
            None,
        );
    };
    if chat_runs.any_active().await {
        // A live harness run is the opposite of idle, however old the last user
        // message is: dreaming rewrites the very scoped memories the run is
        // using, and doing so mid-step deadlocked a live mission (2026-08-10,
        // 316 tool-result memories folded under a running step).
        return (true, Some("Skipped (mission live)".to_string()), None);
    }
    if dream_in_flight.swap(true, std::sync::atomic::Ordering::SeqCst) {
        return (
            true,
            Some("Skipped (dream already in flight)".to_string()),
            None,
        );
    }

    let outcome = dream_once(dreaming, agent, router, chat_runs, activity, budget).await;
    // Released whatever the cycle returned: a cycle that errored must not wedge
    // every later tick out of the latch.
    dream_in_flight.store(false, std::sync::atomic::Ordering::SeqCst);
    outcome
}

/// The board fold (P25 DSP step 3), run right after each scheduled dream.
///
/// It writes memories, so it takes the same guards the dream does: skipped
/// while a harness run is live, and only with the dream latch claimed.
/// Deterministic and model-free, so it runs whether or not the model-driven
/// cycle found anything to consolidate.
async fn run_board_fold(
    dreaming: Option<&Arc<nanna_memory::DreamingService>>,
    storage: Option<&Arc<nanna_storage::Storage>>,
    chat_runs: &Arc<crate::control::chat_harness::ChatRunRegistry>,
    dream_in_flight: &Arc<std::sync::atomic::AtomicBool>,
) {
    let (Some(dreaming), Some(storage)) = (dreaming, storage) else {
        return;
    };
    if chat_runs.any_active().await
        || dream_in_flight.swap(true, std::sync::atomic::Ordering::SeqCst)
    {
        return;
    }
    let folded =
        crate::memory_write_through::fold_closed_cards(storage, &dreaming.memory_arc()).await;
    dream_in_flight.store(false, std::sync::atomic::Ordering::SeqCst);
    match folded {
        Ok(0) => {}
        Ok(count) => info!("Dream fold: {count} closed cards folded into memory"),
        Err(e) => warn!("Dream fold failed: {e}"),
    }
}

/// One dream cycle, with the latch already claimed by the caller.
///
/// The idle gate AND the full cycle (feedback flush -> FSRS testing-effect
/// flush -> consolidate) both live in the one [`nanna_memory::DreamingService`].
/// The daemon supplies only what it owns: its lock-free activity clock and a
/// consolidation budget sized to *its* summarizer model. Dreaming competes with
/// the live agent for that model and rewrites the store, so it still only runs
/// during a lull (or under memory pressure).
async fn dream_once(
    dreaming: &Arc<nanna_memory::DreamingService>,
    agent: &Arc<AgentService>,
    router: &Arc<LlmRouter>,
    chat_runs: &Arc<crate::control::chat_harness::ChatRunRegistry>,
    activity: &Arc<nanna_memory::ActivityClock>,
    budget: DreamBudget,
) -> (bool, Option<String>, Option<String>) {
    let DreamBudget {
        max_compression_ratio: consolidation_max_ratio,
        min_remaining_memories: consolidation_min_remaining,
        idle_threshold_secs: dream_idle_threshold_secs,
    } = budget;

    // Read the summarization list LIVE, once per
    // cycle: whatever the user last set is what
    // this dream summarizes on. With none, the
    // chat models in their configured order — the
    // one rule every memory consumer shares.
    let summarization_models = {
        let handle = agent.config_handle();
        let live = handle.read().await;
        crate::dream_summarizer::for_agent_service(&live)
    };
    if summarization_models.is_empty() {
        return (
            true,
            Some(format!(
                "Skipped ({})",
                crate::agent_service::NO_MODEL_CONFIGURED
            )),
            None,
        );
    }

    // The idle gate AND the full dream cycle (feedback
    // flush -> FSRS testing-effect flush -> consolidate)
    // both live in the one `DreamingService`. The daemon
    // supplies only what it owns: its lock-free activity
    // clock and a consolidation budget sized to *its*
    // summarizer model. Dreaming competes with the live
    // agent for that model and rewrites the store, so it
    // still only runs during a lull (or under pressure).
    let idle = activity.idle();
    // Size the budget to the SMALLEST window across
    // the failover list: one prompt is built and
    // then offered to each candidate in turn, so a
    // budget fitted to the first model would
    // overflow a smaller fallback.
    let window_tokens =
        crate::dream_summarizer::summarizer_context_window_tokens(
            router,
            &summarization_models,
        )
        .await;
    let consolidation_config = scheduled_consolidation_config(
        consolidation_max_ratio,
        consolidation_min_remaining,
        window_tokens,
    );
    // The any_active skip above gates the dream's
    // START; this gates its MIDDLE (P22 Tier 4): a
    // chat turn arriving mid-dream pauses the NEXT
    // cluster's summarization until the turn
    // releases, instead of contending with it for
    // the model. Cluster boundaries are the natural
    // yield points — a fold already in flight
    // completes (the CAS guards own staleness),
    // only new provider work waits. Dreaming is
    // idle-time work by definition, so it pauses
    // for a live turn whatever provider it
    // summarizes on.
    let inner_summarize =
        crate::dream_summarizer::summarize_with_failover(
            router.clone(),
            summarization_models.clone(),
        );
    let summarize_gate = chat_runs.clone();
    let summarize = move |prompt: String| {
        let pending = inner_summarize(prompt);
        let gate = summarize_gate.clone();
        async move {
            gate.wait_idle().await;
            pending.await
        }
    };
    match dreaming
        .dream_if_triggered(idle, &consolidation_config, summarize)
        .await
    {
        Ok(None) => {
            let memory_count = dreaming.memory().count().await;
            debug!(
                "Skipping scheduled consolidation: system active \
                 (idle {idle:?} < {dream_idle_threshold_secs}s, \
                 {memory_count} memories)"
            );
            (
                true,
                Some(format!(
                    "Skipped (active; idle {}s, {memory_count} memories)",
                    idle.as_secs()
                )),
                None,
            )
        }
        Ok(Some((trigger, stats))) => {
            info!(
                "Scheduled dream ({trigger:?}): {} processed, \
                 {} merged, {} deduped, {} promoted, {} demoted",
                stats.consolidation.memories_processed,
                stats.consolidation.memories_merged,
                stats.consolidation.memories_deduped,
                stats.auto_promoted,
                stats.auto_demoted,
            );
            (
                true,
                Some(format!(
                    "Processed {} memories",
                    stats.consolidation.memories_processed
                )),
                None,
            )
        }
        Err(e) => {
            error!("Scheduled consolidation failed: {e}");
            (false, None, Some(e.to_string()))
        }
    }
}

/// A heartbeat or cron job, run as a full agent prompt (tools, memory, model
/// fallback) in a task-scoped session that is not persisted to the session
/// store.
///
/// Returns the scheduler's `(success, output, error)` triple.
async fn run_scheduled_agent_prompt(
    task: &nanna_core::ScheduledTask,
    agent: &Arc<AgentService>,
    chat_runs: &Arc<crate::control::chat_harness::ChatRunRegistry>,
    activity: &Arc<nanna_memory::ActivityClock>,
    scheduled_resume_parked: &Arc<std::sync::atomic::AtomicBool>,
    sessions: &Arc<SessionManager>,
    events: &broadcast::Sender<crate::protocol::Event>,
) -> (bool, Option<String>, Option<String>) {
    // Heartbeat and cron jobs run as full agent prompts
    // (tools, memory, model fallback) in a task-scoped
    // session that is not persisted to the session store.
    let session_id = format!("scheduled-{}", task.id);
    // Idle gate: never start an autonomous prompt on top
    // of a live run. A local model server serves one
    // generation at a time, so a heartbeat firing into a
    // streaming chat gets the slot time-shared and the
    // chat's generation CANCELLED — surfacing as a bogus
    // "provider incident" the harness then heals against
    // (observed live 2026-07-26). Skipping loses nothing:
    // a heartbeat exists to work during a lull, and the
    // next tick picks up whatever was due.
    if agent.any_run_active().await {
        debug!(
            "Skipping scheduled task '{}': a run is already in flight",
            task.name
        );
        return (true, Some("Skipped (a run is in flight)".to_string()), None);
    }
    if !agent.has_configured_model().await {
        // Nothing to run the prompt with: a daemon with
        // no model configured would only fail every tick
        // (and used to send a request naming no model).
        debug!(
            "Skipping scheduled task '{}': no model is configured",
            task.name
        );
        return (true, Some("Skipped (no model configured)".to_string()), None);
    }

    // An autonomous agent run (heartbeat / cron / task
    // prompt) is the daemon actively using the model, so
    // it counts as activity too — defer the dream cycle
    // while it runs. Heartbeats are infrequent (30 min)
    // vs the 5-min idle threshold, and memory pressure
    // still overrides, so dreaming is not starved.
    activity.record();
    // Session scoping (the `with_run_session` binding
    // that fixed 35 failed `todo` calls) now lives
    // inside `run_scheduled_prompt_yielding`, carried
    // by the run's own spawned future — a chat that
    // starts during this run cannot see the scheduled
    // session, nor vice versa.
    let outcome = run_scheduled_prompt_yielding(
        agent,
        chat_runs,
        &session_id,
        &task.payload,
    )
    .await;
    match outcome {
        Some(Ok(result)) => {
            let heartbeat_ok = task.name == "heartbeat"
                && result.content.trim().contains("HEARTBEAT_OK");
            if heartbeat_ok {
                debug!("Heartbeat: OK (nothing to do)");
            } else {
                info!(
                    "Scheduled task '{}' completed: {}",
                    task.name,
                    result.content.chars().take(200).collect::<String>()
                );
            }
            if let Some(message) = scheduled_output_message(
                &task.name,
                &result.content,
                heartbeat_ok,
            ) && let Some(ref target) = task.target_session
                && sessions
                    .post_assistant_message(events, target, message)
                    .await
                    .is_none()
            {
                warn!(
                    "Task '{}' posts to session {target}, which no longer \
                     exists; its result is in the run history only",
                    task.name
                );
            }
            (true, Some(result.content), None)
        }
        Some(Err(e)) => {
            error!("Scheduled task '{}' failed: {}", task.name, e.message);
            (false, None, Some(e.message))
        }
        None => park_scheduled_resume(
            task,
            agent,
            chat_runs,
            activity,
            scheduled_resume_parked,
            &session_id,
        ),
    }
}

/// The scheduled run yielded the local provider to a live chat turn (P22 Tier
/// 4). Resume on release.
///
/// ONE detached waiter re-runs the prompt when the registry goes idle —
/// promptly, not at the next tick — and if a fresh user turn preempts the
/// resumed run too, it parks again. That loop is bounded by user activity
/// itself, not by a counter: every extra lap requires a new turn to have
/// claimed the provider. The caller returns NOW because the heartbeat arm of
/// the scheduler loop awaits it inline — parking there would stall every other
/// scheduled task for as long as the chat runs.
fn park_scheduled_resume(
    task: &nanna_core::ScheduledTask,
    agent: &Arc<AgentService>,
    chat_runs: &Arc<crate::control::chat_harness::ChatRunRegistry>,
    activity: &Arc<nanna_memory::ActivityClock>,
    scheduled_resume_parked: &Arc<std::sync::atomic::AtomicBool>,
    session_id: &str,
) -> (bool, Option<String>, Option<String>) {
    // The run yielded the local provider to a
    // live chat turn (P22 Tier 4). Resume on
    // release: ONE detached waiter re-runs the
    // prompt when the registry goes idle —
    // promptly, not at the next tick — and if
    // a fresh user turn preempts the resumed
    // run too, it parks again. That loop is
    // bounded by user activity itself, not by
    // a counter: every extra lap requires a
    // new turn to have claimed the provider.
    // The executor returns NOW because the
    // heartbeat arm of the scheduler loop
    // awaits it inline — parking here would
    // stall every other scheduled task for as
    // long as the chat runs.
    if scheduled_resume_parked
        .compare_exchange(
            false,
            true,
            std::sync::atomic::Ordering::SeqCst,
            std::sync::atomic::Ordering::SeqCst,
        )
        .is_ok()
    {
        let resume_agent = agent.clone();
        let resume_runs = chat_runs.clone();
        let resume_activity = activity.clone();
        let resume_parked = scheduled_resume_parked.clone();
        let resume_session = session_id.to_string();
        let resume_payload = task.payload.clone();
        let resume_name = task.name.clone();
        tokio::spawn(async move {
            loop {
                resume_runs.wait_idle().await;
                // The release tail unregisters
                // the finished chat BEFORE
                // releasing the registry, so an
                // active run here is a NEW
                // claimant — the slot is taken
                // and the schedule covers the
                // rest.
                if resume_agent.any_run_active().await {
                    debug!(
                        task = %resume_name,
                        "Yielded run superseded — the slot is \
                         taken; the next tick owns the work"
                    );
                    break;
                }
                resume_activity.record();
                match run_scheduled_prompt_yielding(
                    &resume_agent,
                    &resume_runs,
                    &resume_session,
                    &resume_payload,
                )
                .await
                {
                    Some(Ok(result)) => {
                        let heartbeat_ok = resume_name
                            == "heartbeat"
                            && result
                                .content
                                .trim()
                                .contains("HEARTBEAT_OK");
                        if heartbeat_ok {
                            debug!(
                                "Heartbeat (resumed): OK \
                                 (nothing to do)"
                            );
                        } else {
                            info!(
                                "Scheduled task '{}' completed \
                                 after yielding: {}",
                                resume_name,
                                result
                                    .content
                                    .chars()
                                    .take(200)
                                    .collect::<String>()
                            );
                        }
                        break;
                    }
                    Some(Err(e)) => {
                        error!(
                            "Scheduled task '{}' failed after \
                             yielding: {}",
                            resume_name, e.message
                        );
                        break;
                    }
                    None => {
                        // Preempted again — user
                        // turns keep priority.
                    }
                }
            }
            resume_parked
                .store(false, std::sync::atomic::Ordering::SeqCst);
        });
        (
            true,
            Some(
                "Yielded to a live chat turn; resuming on release"
                    .to_string(),
            ),
            None,
        )
    } else {
        (
            true,
            Some(
                "Yielded to a live chat turn; a resume is \
                 already parked, the next tick covers this one"
                    .to_string(),
            ),
            None,
        )
    }
}

/// The `task_recurrence_sweep` scheduled task (P15): the scheduler is the one
/// recurrence engine, so recurring todo items are reopened here rather than by
/// a second clock inside the task store.
///
/// Returns the scheduler's `(success, output, error)` triple.
async fn run_recurrence_sweep(
    storage: Option<&Arc<nanna_storage::Storage>>,
) -> (bool, Option<String>, Option<String>) {
    let Some(storage) = storage else {
        return (true, Some("Skipped (no storage)".to_string()), None);
    };
    let reopened = crate::tasks::sweep_recurrences(storage).await;
    if reopened > 0 {
        info!("Recurrence sweep reopened {reopened} tasks");
    }
    // The time-driven task events ride this sweep rather than a second one:
    // both are periodic task maintenance on the same cadence, and the ORDER is
    // load-bearing — reopening a recurring card clears its announcement
    // markers, so a card that came back around this pass can fall due on the
    // same pass instead of waiting five minutes for the next.
    //
    // The scheduled task keeps the name `task_recurrence_sweep`. Renaming it
    // would not rename the row an existing daemon already has, so
    // `has_task_named` would register a SECOND sweep beside the first.
    let now = chrono::Utc::now().to_rfc3339();
    let (due, overdue) = match storage.tasks().announce_due(&now).await {
        Ok(counts) => counts,
        Err(e) => {
            warn!("task time sweep failed: {e}");
            (0, 0)
        }
    };
    if due > 0 || overdue > 0 {
        info!("Task time sweep announced {due} due and {overdue} overdue");
    }
    (
        true,
        Some(format!(
            "Reopened {reopened} recurring tasks; announced {due} due, {overdue} overdue"
        )),
        None,
    )
}

/// Name of the scheduled task that prunes the raw tool-call log.
pub const TOOL_CALL_LOG_PRUNE_TASK: &str = "tool_call_log_prune";

/// Name of the P15 recurrence sweep (which also announces due/overdue cards).
pub const TASK_RECURRENCE_SWEEP_TASK: &str = "task_recurrence_sweep";

/// Scheduled tasks that are the daemon's own machinery, not prompts.
///
/// They live in the same persisted `cron_jobs` table as the user's jobs, so any
/// other scheduler over that table — `nanna serve` runs one — must recognise
/// them and leave them alone. Unrecognised, a scheduler that runs payloads as
/// agent prompts hands "Reopen recurring tasks…" (or a reminder's text) to the
/// model, and one that fakes success consumes a one-shot reminder unsent.
pub const DAEMON_SYSTEM_TASKS: &[&str] = &[
    TASK_RECURRENCE_SWEEP_TASK,
    TOOL_CALL_LOG_PRUNE_TASK,
    crate::reminder_service::REMINDER_TASK_NAME,
];

/// Days of raw `tool_call_log` rows kept.
///
/// Bound justification: raw rows back only the "recent calls" view (the last
/// 50 by default); per-hour and per-day history lives in the aggregate tables,
/// which this does not touch. A month covers any debugging look-back, and at a
/// heavy few thousand calls a day holds the table near 100k rows instead of
/// growing for the daemon's whole life — which it did, because
/// `prune_tool_call_log` had never had a caller.
const TOOL_CALL_LOG_KEEP_DAYS: u32 = 30;

/// The daily `tool_call_log_prune` scheduled task.
///
/// Returns the scheduler's `(success, output, error)` triple.
async fn run_tool_call_log_prune(
    storage: Option<&Arc<nanna_storage::Storage>>,
) -> (bool, Option<String>, Option<String>) {
    let Some(storage) = storage else {
        return (true, Some("Skipped (no storage)".to_string()), None);
    };
    match storage.prune_tool_call_log(TOOL_CALL_LOG_KEEP_DAYS).await {
        Ok(deleted) => {
            if deleted > 0 {
                info!(
                    "Pruned {deleted} tool-call log rows older than {TOOL_CALL_LOG_KEEP_DAYS} days"
                );
            }
            (
                true,
                Some(format!("Pruned {deleted} tool-call log rows")),
                None,
            )
        }
        Err(e) => (
            false,
            None,
            Some(format!("tool-call log prune failed: {e}")),
        ),
    }
}

/// A reminder: delivered as a message, never run as a prompt.
///
/// A reminder needs no model, and must not wait for one (see
/// `reminder_service`).
///
/// Returns the scheduler's `(success, output, error)` triple.
async fn deliver_scheduled_reminder(
    sessions: &Arc<SessionManager>,
    events: &broadcast::Sender<crate::protocol::Event>,
    task: &nanna_core::ScheduledTask,
    reminder_tick: Duration,
) -> (bool, Option<String>, Option<String>) {
    match crate::reminder_service::deliver_reminder(
        sessions,
        events,
        task,
        chrono::Utc::now(),
        reminder_tick,
    )
    .await
    {
        Ok(content) => (true, Some(content), None),
        Err(e) => {
            warn!("{e}");
            (false, None, Some(e))
        }
    }
}

/// The main daemon server
pub struct DaemonServer {
    config: DaemonConfig,
    embedding: EmbeddingConfig,
    /// The part of `embedding` a config change may still move once the router
    /// is built (the Ollama token), shared by every Ollama embedding client
    /// and the control plane that applies config changes.
    live_embedding: Arc<crate::embedding_reload::LiveEmbeddingSettings>,
    memory_path: Option<PathBuf>,
    _brave_api_key: Option<String>,
    sessions: Arc<SessionManager>,
    /// Late-bound handle to the control plane for consumers created
    /// before it exists (filled in `run()`, read by the agent service).
    control_slot: Arc<tokio::sync::RwLock<Option<Arc<ControlPlane>>>>,
    /// The scheduler, for the `schedule.*` services built before it exists.
    /// Filled once in `run()`, right after the scheduler is constructed.
    scheduler_slot: crate::reminder_service::SchedulerSlot,
    /// Per-server MCP state: written by the boot task, read by `system.status`.
    mcp_status: crate::mcp_startup::McpStatus,
    /// The MCP background task, awaited by `finish_shutdown`.
    mcp_task: std::sync::Mutex<Option<tokio::task::JoinHandle<()>>>,
    /// The dreaming orchestrator, for `memory.get`'s use feedback. Filled once
    /// in `run()`, right after it is built.
    dreaming_slot: DreamingSlot,
    ipc: Arc<IpcServer>,
    persistence: Arc<PersistenceManager>,
    shutdown_tx: broadcast::Sender<()>,
    /// The serve loop's end of `shutdown_tx`, subscribed at construction and
    /// taken by `run()`. A broadcast reaches only the receivers that exist
    /// when it is sent, and `run()` reaches its serve loop only after the
    /// whole boot (storage, services, MCP). A loop that subscribed there
    /// missed any shutdown sent during boot, including a SIGTERM the signal
    /// handler had already caught, so the daemon neither stopped nor died.
    /// Held from here, every request made once the server exists is waiting
    /// when the loop starts.
    shutdown_rx: Option<broadcast::Receiver<()>>,
    /// PID file (prevents multiple instances)
    pid_file: Option<PidFile>,
    /// Log buffer for capturing daemon logs
    log_buffer: Option<crate::log_buffer::LogBuffer>,
    /// Shared storage for model stats persistence
    storage: Option<Arc<nanna_storage::Storage>>,
    /// Set when storage init quarantined + rebuilt a corrupt database file;
    /// surfaced on /status and broadcast as `Event::MemoryStoreRebuilt`.
    memory_recovery: Option<Arc<nanna_storage::RecoveryReport>>,
    /// Set when storage failed to open at all — memory is in-process only and
    /// will be lost on exit. Distinct from `memory_recovery`, which means the
    /// store WAS repaired and does persist. Surfaced on status so the state is
    /// observable rather than inferred from a log line at boot.
    storage_error: Option<String>,
    /// The board memory write-through queue's receiving end, created with the
    /// task event sink when storage opens and taken by `init_services` once the
    /// memory service exists (P25 Stage 1). Left `None` — and so dropped —
    /// when memory is disabled, which the sink reads as "no copies owed".
    board_copies: std::sync::Mutex<Option<tokio::sync::mpsc::Receiver<nanna_storage::TaskEvent>>>,
    /// Terminal reason file: a durable record of WHY this process stopped, so
    /// the next boot can tell a clean shutdown from a hard death whose only
    /// other evidence is a log that simply ends (2026-08-10 ministral leg).
    exit_reason: crate::exit_reason::ExitReasonFile,
}

impl DaemonServer {
    /// Resolve one `provider/model` spec to a live embedding client.
    ///
    /// `None` means the provider's credential is absent — the entry is skipped
    /// with a line saying so, rather than substituting a different model. A
    /// silent substitution is what put a hardcoded paid embedder in front of
    /// the free one the user had chosen.
    pub(crate) fn embedding_provider_for(
        &self,
        spec: &str,
    ) -> Option<(EmbeddingProviderInfo, Arc<nanna_llm::EmbeddingClient>)> {
        let (provider, model) = split_embedding_spec(spec)?;
        let info = EmbeddingProviderInfo {
            name: provider.clone(),
            model: model.clone(),
        };
        match provider.as_str() {
            // Config first, then env — matching the OpenRouter arm below.
            // Reading only the environment made this arm resolvable in the GUI
            // process (which exports the key) and unresolvable in the daemon
            // (which does not), so the same configuration behaved differently
            // depending on who started it.
            "openai" => {
                let key = self
                    .config
                    .llm
                    .openai_api_key
                    .clone()
                    .or_else(|| std::env::var("OPENAI_API_KEY").ok());
                let Some(key) = key else {
                    warn!("Embedding provider '{spec}' skipped: no OpenAI API key");
                    return None;
                };
                Some((
                    info,
                    Arc::new(nanna_llm::EmbeddingClient::openai(&key).with_model(&model)),
                ))
            }
            "openrouter" => {
                let key = self
                    .config
                    .llm
                    .openrouter_api_key
                    .clone()
                    .or_else(|| std::env::var("OPENROUTER_API_KEY").ok());
                let Some(key) = key else {
                    warn!("Embedding provider '{spec}' skipped: no OpenRouter API key");
                    return None;
                };
                Some((
                    info,
                    Arc::new(
                        nanna_llm::EmbeddingClient::openai(&key)
                            .with_model(&model)
                            .with_base_url("https://openrouter.ai/api"),
                    ),
                ))
            }
            // Every Ollama entry shares the one token handle, read per request:
            // a token saved while the daemon runs reaches these clients without
            // rebuilding them (see `LiveEmbeddingSettings`).
            "ollama" => Some((
                info,
                Arc::new(
                    nanna_llm::EmbeddingClient::ollama(&self.embedding.ollama_host)
                        .with_bearer(self.live_embedding.ollama_bearer())
                        .with_model(&model),
                ),
            )),
            other => {
                warn!("Embedding provider '{spec}' skipped: unknown provider '{other}'");
                None
            }
        }
    }

    /// Discover the embedding dimension by probing **the same router the live
    /// embed path uses**.
    ///
    /// Probing through the router (rather than a bespoke client built from
    /// `embedding.provider`) matters for two reasons:
    ///
    /// 1. The router resolves cloud keys from **config *or* env**, while a
    ///    bespoke client read only the env — so a key set in `config.toml`
    ///    probed as "missing" even though every real embed succeeded.
    /// 2. The router carries the Ollama fallback, so a probe survives an
    ///    unreachable/unkeyed cloud provider exactly like a real embed does.
    ///
    /// A failure here is **not** fatal — see the call site.
    async fn probe_embedding_dimension(router: &EmbeddingRouter) -> Result<usize, String> {
        // One sweep, no waiting on congestion: this runs before the IPC port
        // listens, so the daemon cannot be reached while it waits. A busy
        // provider leaves the store on the provisional width, and the
        // background `probe_and_align_dimension` — which does wait — corrects it.
        let (embedding, _switched_to) = router.embed_one_now("dimension probe").await?;
        if embedding.is_empty() {
            return Err("embedding provider returned an empty vector".to_string());
        }
        debug_assert!(!embedding.is_empty(), "probe returned a non-empty vector");
        Ok(embedding.len())
    }

    /// Create a new daemon server
    #[must_use]
    pub fn new(
        config: DaemonConfig,
        embedding: EmbeddingConfig,
        memory_path: Option<PathBuf>,
        brave_api_key: Option<String>,
    ) -> Self {
        let sessions = Arc::new(SessionManager::new());
        let ipc = Arc::new(IpcServer::new(config.ipc.clone()));
        let persistence = Arc::new(PersistenceManager::new(&config.data_dir));
        let (shutdown_tx, shutdown_rx) = broadcast::channel(1);

        // Create PID file if enabled
        let pid_file = if config.servers.pid_file {
            Some(PidFile::new(&config.data_dir))
        } else {
            None
        };

        let exit_reason = crate::exit_reason::ExitReasonFile::new(&config.data_dir);

        let live_embedding = Arc::new(crate::embedding_reload::LiveEmbeddingSettings::new(
            &embedding,
        ));

        Self {
            config,
            embedding,
            live_embedding,
            memory_path,
            _brave_api_key: brave_api_key,
            sessions,
            control_slot: Arc::new(tokio::sync::RwLock::new(None)),
            scheduler_slot: Arc::new(std::sync::OnceLock::new()),
            mcp_status: crate::mcp_startup::McpStatus::default(),
            mcp_task: std::sync::Mutex::new(None),
            dreaming_slot: Arc::new(std::sync::OnceLock::new()),
            ipc,
            persistence,
            shutdown_tx,
            shutdown_rx: Some(shutdown_rx),
            pid_file,
            log_buffer: None,
            storage: None,
            memory_recovery: None,
            storage_error: None,
            board_copies: std::sync::Mutex::new(None),
            exit_reason,
        }
    }

    /// What a config change may still update on the running embedding
    /// clients; the control plane applies every committed change to it.
    #[must_use]
    pub(crate) fn live_embedding(&self) -> Arc<crate::embedding_reload::LiveEmbeddingSettings> {
        Arc::clone(&self.live_embedding)
    }

    /// Recovery report from a startup quarantine + rebuild, if one happened.
    #[must_use]
    pub fn memory_recovery(&self) -> Option<Arc<nanna_storage::RecoveryReport>> {
        self.memory_recovery.clone()
    }

    /// Set the storage backend for model stats persistence and session persistence.
    pub fn set_storage(&mut self, storage: Arc<nanna_storage::Storage>) {
        // Replace the SessionManager with one that has storage
        self.sessions = Arc::new(SessionManager::with_storage(storage.clone()));
        self.storage = Some(storage);
    }

    /// Open (and if needed recover) the durable store at `<data_dir>/nanna.db`.
    ///
    /// [`Self::run`] calls this only after claiming the PID file and the IPC
    /// port: a live daemon holds nanna.db's exclusive lock, so an instance
    /// that opened it first reported a storage failure instead of the real
    /// conflict. Public so tests can drive the storage init path without
    /// binding sockets. Idempotent: a second call keeps the open store.
    pub async fn open_storage(&mut self) {
        if self.storage.is_some() {
            return;
        }

        // Initialize Turso storage. The recovering open verifies the memories
        // table is readable; on page-level corruption it quarantines the
        // damaged file, rebuilds a fresh store at the same path, and salvages
        // every reachable row — so the daemon boots with a working store
        // instead of a silently empty one.
        let db_path = self.config.data_dir.join("nanna.db");
        let storage_config = nanna_storage::StorageConfig {
            path: db_path.to_string_lossy().to_string(),
        };
        match nanna_storage::open_with_recovery(&storage_config).await {
            Ok((storage, recovery)) => {
                info!("Storage initialized at {:?}", db_path);
                if let Some(report) = recovery {
                    warn!(
                        "Memory store was REBUILT after corruption: {} memories recovered \
                         (corrupt copy: {})",
                        report.memories_recovered,
                        report.quarantine_path.display()
                    );
                    self.memory_recovery = Some(Arc::new(report));
                }
                let storage = Arc::new(storage);
                // Task lifecycle events reach the bus from the storage layer,
                // which is the only place that sees every writer — including
                // the cascades (subtree cancel, ancestor auto-complete) that
                // no caller ever names.
                let (copies_tx, copies_rx) = tokio::sync::mpsc::channel(
                    crate::memory_write_through::WRITE_THROUGH_QUEUE_MAX,
                );
                if storage.set_task_events(Arc::new(
                    crate::task_event_bridge::TaskEventBridge::new(self.ipc.event_sender())
                        .with_memory_write_through(copies_tx),
                )) {
                    *self
                        .board_copies
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(copies_rx);
                } else {
                    warn!("task event sink was already attached; keeping the existing one");
                }
                self.set_storage(storage);
            }
            Err(e) => {
                // Storage is where MEMORY lives, not just model stats. Without
                // it the daemon still runs and memory still "works" — in RAM,
                // for this session, discarded on exit — so the failure reads as
                // a good session until someone notices nothing was remembered.
                //
                // The old wording named model stats, the quietest thing lost,
                // at warn level. That is the wrong end of the blast radius and
                // the wrong severity: this is also the only signal a bad
                // migration produces, since a migration that fails leaves the
                // name unrecorded and gets retried on every boot forever.
                error!(
                    error = %e,
                    db_path = ?db_path,
                    "STORAGE UNAVAILABLE — memory is in-process only and will be LOST on exit; \
                     model stats, tasks and checkpoints will not persist. Most likely a failed \
                     migration or an unwritable database path."
                );
                self.storage_error = Some(e.to_string());
            }
        }
    }

    /// Get the shutdown sender (for signaling shutdown)
    ///
    /// A send at any point after construction stops the daemon: one made
    /// before `run()` reaches its serve loop is held until the loop starts,
    /// and the daemon then drains as soon as it has booted.
    #[must_use]
    pub fn shutdown_handle(&self) -> broadcast::Sender<()> {
        self.shutdown_tx.clone()
    }

    /// Handle to the terminal reason file, for exit paths that live outside
    /// `run()` (the signal handlers in [`crate::shutdown`], the binary's
    /// parent watch). Clones share the armed flag, so recording stays a no-op
    /// until `run()` has claimed the file by writing its startup marker.
    #[must_use]
    pub fn exit_reason_handle(&self) -> crate::exit_reason::ExitReasonFile {
        self.exit_reason.clone()
    }

    /// Get the IPC server address
    #[must_use]
    pub fn ipc_address(&self) -> String {
        self.ipc.address()
    }

    /// The address the IPC listener bound, `None` until `run` binds it. See
    /// [`IpcServer::bound_addr`]: with port 0 this reports the chosen port.
    #[must_use]
    pub fn ipc_bound_addr(&self) -> tokio::sync::watch::Receiver<Option<std::net::SocketAddr>> {
        self.ipc.bound_addr()
    }

    /// Start the cron runner.
    ///
    /// With daemon-first startup the daemon owns `nanna.db`, so it is the cron
    /// runner (the GUI scheduler only runs in embedded mode). Loads persisted
    /// jobs and runs heartbeat + memory consolidation, mirroring the GUI's
    /// embedded schedule.
    async fn start_scheduler(
        &self,
        deps: SchedulerDeps,
    ) -> Arc<tokio::sync::RwLock<nanna_core::Scheduler>> {
        let SchedulerDeps {
            agent,
            router,
            dreaming,
            chat_runs,
            activity_clock,
        } = deps;

        // These three come from `[scheduler]` in the user's config, not from
        // literals: the GUI's Scheduler tab writes them there and a config
        // reload re-applies them to this very loop (see the control plane's
        // config handler), so the toggles work without a daemon restart.
        let scheduler_config = nanna_core::SchedulerConfig {
            enabled: self.config.scheduler.enabled,
            heartbeat_interval: std::time::Duration::from_secs(
                nanna_core::clamp_heartbeat_secs(self.config.heartbeat_interval_secs),
            ),
            heartbeat_enabled: self.config.scheduler.heartbeat_enabled,
            heartbeat_prompt: DAEMON_HEARTBEAT_PROMPT.to_string(),
            max_concurrent: 4,
            check_interval: std::time::Duration::from_secs(30),
            default_timezone: "UTC".to_string(),
        };
        let mut scheduler = nanna_core::Scheduler::new(scheduler_config);
        if let Some(ref storage) = self.storage {
            scheduler = scheduler.with_storage(storage.clone());
            match scheduler.load_jobs().await {
                Ok(count) => info!("Loaded {count} cron jobs from storage"),
                Err(e) => warn!("Failed to load cron jobs: {e}"),
            }
        } else {
            info!("Scheduler running without persistence (no storage backend)");
        }

        let deduped = scheduler.deduplicate_by_name("memory_consolidation").await;
        if deduped > 0 {
            info!("Removed {deduped} duplicate consolidation tasks");
        }
        if !scheduler.has_task_named("memory_consolidation").await {
            scheduler
                .add_task(nanna_core::consolidation_task(Some(
                    std::time::Duration::from_secs(3600),
                )))
                .await;
            info!("Scheduled memory consolidation task (every 1 hour)");
        }

        // Task recurrence sweep (P15): the scheduler is the one recurrence
        // engine — recurring todo items are reopened here, not by a second
        // clock inside the task store.
        if self.storage.is_some() {
            let deduped = scheduler.deduplicate_by_name(TASK_RECURRENCE_SWEEP_TASK).await;
            if deduped > 0 {
                info!("Removed {deduped} duplicate recurrence sweep tasks");
            }
            if !scheduler.has_task_named(TASK_RECURRENCE_SWEEP_TASK).await {
                scheduler
                    .add_task(nanna_core::recurring_task(
                        TASK_RECURRENCE_SWEEP_TASK,
                        std::time::Duration::from_secs(300),
                        "Reopen recurring tasks whose next occurrence has arrived.",
                    ))
                    .await;
                info!("Scheduled task recurrence sweep (every 5 minutes)");
            }

            let deduped = scheduler.deduplicate_by_name(TOOL_CALL_LOG_PRUNE_TASK).await;
            if deduped > 0 {
                info!("Removed {deduped} duplicate tool-call log prune tasks");
            }
            if !scheduler.has_task_named(TOOL_CALL_LOG_PRUNE_TASK).await {
                scheduler
                    // Empty payload on purpose: the executor runs any task name
                    // it does not recognise as an AGENT PROMPT, so a binary older
                    // than this one (a rollback) would hand a prose payload to the
                    // model — "delete … rows" included. An empty one it skips.
                    .add_task(nanna_core::recurring_task(
                        TOOL_CALL_LOG_PRUNE_TASK,
                        std::time::Duration::from_hours(24),
                        "",
                    ))
                    .await;
                info!("Scheduled tool-call log prune (daily, keeping {TOOL_CALL_LOG_KEEP_DAYS} days)");
            }
        }

        let executor = Self::scheduled_task_executor(ScheduledTaskDeps {
            reminder_tick: scheduler.check_interval(),
            agent,
            router,
            dreaming,
            chat_runs,
            activity_clock,
            storage: self.storage.clone(),
            sessions: self.sessions.clone(),
            events: self.ipc.event_sender(),
            dream: DreamBudget {
                max_compression_ratio: self.config.memory_max_compression_ratio,
                min_remaining_memories: self.config.memory_min_remaining_memories,
                idle_threshold_secs: self.config.dream_idle_threshold_secs,
            },
        });
        scheduler = scheduler.with_executor(executor);
        scheduler.start();
        info!("Daemon scheduler started (heartbeat + cron runner)");
        let scheduler = Arc::new(tokio::sync::RwLock::new(scheduler));
        // `run` executes once per server, so the slot is empty here; a
        // second fill would leave the reminder services on the first
        // scheduler, which is the one still running.
        if self.scheduler_slot.set(Arc::clone(&scheduler)).is_err() {
            warn!("scheduler slot was already filled; reminders keep the first scheduler");
        }
        scheduler
    }

    /// The closure the scheduler calls for every due task.
    ///
    /// Nothing here is captured from a boot-time clone of the agent config: the
    /// summarization list is read from the LIVE config at the top of each cycle,
    /// because a boot clone survives every `config.set` — a user who repointed
    /// summarization kept dreaming on the model they had at startup, the whole
    /// class the P23 summarizer-pin fix closed on the chat path (2026-08-15:
    /// 171/171 summarizations in the benchmark series ran on the wrong model). A
    /// dream cycle runs minutes-to-hours apart, so one lock read per cycle costs
    /// nothing measurable.
    fn scheduled_task_executor(deps: ScheduledTaskDeps) -> nanna_core::TaskExecutor {
        let ScheduledTaskDeps {
            agent: agent_for_tasks,
            router: router_for_tasks,
            dreaming: dreaming_for_tasks,
            chat_runs: chat_runs_for_tasks,
            activity_clock: activity_for_tasks,
            storage: storage_for_tasks,
            sessions: sessions_for_tasks,
            events: events_for_tasks,
            reminder_tick,
            dream,
        } = deps;
        // At most one yielded scheduled run waiting to resume. Not a quota — a
        // dedup: reality already serializes scheduled runs (a live one makes
        // every other tick skip), so a second waiter could only arise from an
        // exotic interleaving, and dropping it costs one schedule period, which
        // the log names.
        let scheduled_resume_parked = Arc::new(std::sync::atomic::AtomicBool::new(false));
        // In-flight latch: the scheduler tick re-fired consolidation every 30s
        // while the previous one was still folding (observed 2026-08-10: a fresh
        // "Consolidation starting" per tick, none finishing) — one dream at a
        // time.
        let dream_in_flight_for_tasks = Arc::new(std::sync::atomic::AtomicBool::new(false));

        Arc::new(move |task| {
            let agent = agent_for_tasks.clone();
            let dreaming = dreaming_for_tasks.clone();
            let router = router_for_tasks.clone();
            let storage = storage_for_tasks.clone();
            let activity = activity_for_tasks.clone();
            let chat_runs = chat_runs_for_tasks.clone();
            let dream_in_flight = dream_in_flight_for_tasks.clone();
            let scheduled_resume_parked = scheduled_resume_parked.clone();
            let sessions = sessions_for_tasks.clone();
            let events = events_for_tasks.clone();
            Box::pin(async move {
                let start = std::time::Instant::now();
                let started_at = chrono::Utc::now();
                let (success, output, error) = match task.name.as_str() {
                    "memory_consolidation" => {
                        let outcome = run_scheduled_consolidation(
                            dreaming.as_ref(),
                            &agent,
                            &router,
                            &chat_runs,
                            &activity,
                            &dream_in_flight,
                            dream,
                        )
                        .await;
                        run_board_fold(
                            dreaming.as_ref(),
                            storage.as_ref(),
                            &chat_runs,
                            &dream_in_flight,
                        )
                        .await;
                        outcome
                    }
                    TASK_RECURRENCE_SWEEP_TASK => run_recurrence_sweep(storage.as_ref()).await,
                    TOOL_CALL_LOG_PRUNE_TASK => run_tool_call_log_prune(storage.as_ref()).await,
                    crate::reminder_service::REMINDER_TASK_NAME => {
                        deliver_scheduled_reminder(&sessions, &events, &task, reminder_tick).await
                    }
                    _ if task.payload.is_empty() => {
                        debug!("Skipping task with empty payload: {}", task.name);
                        (true, Some("Skipped (empty payload)".to_string()), None)
                    }
                    _ => {
                        run_scheduled_agent_prompt(
                            &task,
                            &agent,
                            &chat_runs,
                            &activity,
                            &scheduled_resume_parked,
                            &sessions,
                            &events,
                        )
                        .await
                    }
                };
                nanna_core::TaskResult {
                    task_id: task.id.clone(),
                    task_name: task.name.clone(),
                    success,
                    output,
                    error,
                    duration_ms: crate::numeric::millis_u64(start.elapsed()),
                    started_at,
                    finished_at: chrono::Utc::now(),
                }
            })
        })
    }

    /// Run the daemon server
    ///
    /// # Errors
    ///
    /// [`crate::DaemonError::AlreadyRunning`] when another daemon holds the PID
    /// file, [`crate::DaemonError::Io`] when the data directory cannot be
    /// created, [`crate::DaemonError::Ipc`] when the IPC request receiver has
    /// already been taken, and whatever `init_services` returns when no LLM
    /// provider is configured.
    pub async fn run(&mut self) -> Result<(), crate::DaemonError> {
        info!("Starting Nanna daemon...");
        info!("Data directory: {:?}", self.config.data_dir);

        self.install_panic_hook();

        // Adopt a kill-on-close Job Object BEFORE anything can spawn a child:
        // every exec/acceptance process inherits membership, so the OS reaps
        // the whole tree when this process dies for any reason — clean stop,
        // `taskkill /F`, or a crash. Closes the leak where daemon restarts
        // orphaned in-flight `powershell.exe`/`bash.exe` children (89 counted
        // on 2026-08-01). Survivable on failure: log and run uncontained.
        #[cfg(windows)]
        if crate::job::adopt_kill_on_close_job() {
            info!("Job Object adopted — child processes cannot outlive the daemon");
        } else {
            warn!("Job Object adoption failed — exec children may outlive an unclean daemon exit");
        }

        // Ensure data directory exists
        std::fs::create_dir_all(&self.config.data_dir)?;

        // Claim THE daemon role before touching anything a live daemon owns:
        // the PID file, then the IPC port, and only then the exit record and
        // nanna.db. 2026-09-17: a duplicate whose PID check had been defeated
        // opened storage first, so its first symptom was a turso lock error,
        // and it built the router and memory service before dying at the IPC
        // bind — after overwriting the live daemon's exit record.
        let ipc_port = self.claim_daemon_role().await?;

        self.open_storage().await;

        self.load_sessions().await;

        // ONE run registry, created before the services so the embedding
        // drain, the dream gate, the scheduler and the control plane all hold
        // the same handle: "a mission is live" must be one fact, not two.
        let chat_runs = Arc::new(crate::control::chat_harness::ChatRunRegistry::new());
        // ONE capability-transition ledger for the same reason — the provider
        // plumbing records into the very ledger the step runners drain.
        let degradations = Arc::new(nanna_agent::DegradationLedger::new());

        // Initialize services
        let (
            tools,
            memory,
            agent,
            router,
            tools_dir,
            workspace_id_for_services,
            turn_baselines,
            model_stats,
        ) = self.init_services(&chat_runs, &degradations).await?;

        self.recover_checkpoints(&agent).await;

        // Shared activity clock: stamped by the control plane on every chat
        // request, read by the scheduled dream cycle to gate on idleness. Made
        // here so it is in scope for both the scheduler executor (below) and the
        // control plane (built later); cloning the Arc shares the same clock.
        let activity_clock = Arc::new(nanna_memory::ActivityClock::new());

        let dreaming = self.build_dreaming(memory.as_ref(), &activity_clock);

        // Dreaming must observe the very store the agent writes to, never a
        // private copy — that identity is the whole point of the shared seam.
        debug_assert_eq!(
            dreaming.is_some(),
            memory.is_some(),
            "dreaming service must exist exactly when the memory store does"
        );

        // Scheduler: with daemon-first startup the daemon owns nanna.db, so it
        // is the cron runner (the GUI scheduler only runs in embedded mode).
        // Loads persisted jobs and runs heartbeat + memory consolidation,
        // mirroring the GUI's embedded schedule. (`chat_runs` was created
        // before the services above, and is the same handle everywhere.)
        // In-flight latch: the scheduler tick re-fired consolidation every
        // 30s while the previous one was still folding (observed 2026-08-10:
        // a fresh "Consolidation starting" per tick, none finishing) — one
        // dream at a time.
        let scheduler = self
            .start_scheduler(SchedulerDeps {
                agent: Arc::clone(&agent),
                router: Arc::clone(&router),
                dreaming: dreaming.clone(),
                chat_runs: Arc::clone(&chat_runs),
                activity_clock: Arc::clone(&activity_clock),
            })
            .await;

        let (control, channel_status_manager) = self
            .build_control_plane(ControlPlaneDeps {
                agent,
                memory: memory.clone(),
                tools,
                router,
                tools_dir,
                workspace_id_for_services,
                turn_baselines,
                scheduler,
                model_stats,
                chat_runs: Arc::clone(&chat_runs),
                degradations: Arc::clone(&degradations),
                activity_clock: Arc::clone(&activity_clock),
                dreaming: dreaming.clone(),
            })
            .await;

        // Take the request receiver from IPC server
        let mut request_rx =
            self.ipc.take_request_receiver().await.ok_or_else(|| {
                crate::DaemonError::Ipc("Request receiver already taken".to_string())
            })?;

        let mut shutdown_rx = self
            .shutdown_rx
            .take()
            .unwrap_or_else(|| self.shutdown_tx.subscribe());

        let ipc_handle = self.spawn_ipc_server(ipc_port);

        self.announce_memory_recovery();

        let _health_state = self.spawn_health_server(&control, memory.as_ref()).await;

        let channel_manager = self
            .start_channel_manager(&control, &channel_status_manager)
            .await;

        self.spawn_webhook_server(&control, channel_manager.as_ref())
            .await;

        // Sessions are now persisted via Turso write-through on every mutation.
        // No more periodic JSON auto-save — each create/message/delete/rename writes to DB immediately.

        let stats_save_handle = self.spawn_stats_autosave(&control);

        self.spawn_sub_agent_checkin();

        // The configured address: with port 0 the real port is only known once
        // the IPC task binds, and its own "listening" line reports that.
        info!(
            "Daemon ready. IPC server configured for ws://{}",
            self.ipc.address()
        );

        self.serve_requests(&control, &mut request_rx, &mut shutdown_rx)
            .await;

        self.finish_shutdown(channel_manager, stats_save_handle, ipc_handle)
            .await;
        Ok(())
    }

    /// Route every panic through tracing, and record it in the exit-reason file.
    fn install_panic_hook(&self) {
        // Route every panic through tracing BEFORE doing anything that can
        // spawn a task. The default hook prints to stderr, which a headless
        // daemon has nowhere useful to send — so a panicked task died with
        // ZERO log lines and the wedge it left (2026-08-10: a chat turn's
        // task panicked and its session went silent for 50+ minutes) was
        // undiagnosable from the log alone. Chains to the previous hook so
        // stderr output, where it exists, is preserved.
        let previous_hook = std::panic::take_hook();
        let panic_exit_reason = self.exit_reason.clone();
        std::panic::set_hook(Box::new(move |info| {
            let payload = info
                .payload()
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| info.payload().downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "<non-string panic payload>".to_string());
            let location = info.location().map_or_else(
                || "<unknown location>".to_string(),
                |l| format!("{}:{}:{}", l.file(), l.line(), l.column()),
            );
            // File first, log second: with panic=abort (the release profile)
            // this hook is the last code that runs, and the non-blocking log
            // writer may never flush — the reason file is the record that
            // survives. In debug a task panic doesn't kill the process; a
            // later clean shutdown overwrites this record, so last-writer-
            // wins keeps the file truthful either way.
            panic_exit_reason.record_exit("panic", Some(&format!("{payload} at {location}")));
            tracing::error!(%location, "PANIC: {payload}");
            previous_hook(info);
        }));
    }

    /// Claim the daemon role: the PID file, then the IPC port, then the exit
    /// record — in that order, before anything a live daemon owns is touched.
    ///
    /// Returns the reserved (bound, not listening) IPC port, which
    /// [`Self::spawn_ipc_server`] later listens on.
    ///
    /// # Errors
    ///
    /// [`crate::DaemonError::AlreadyRunning`] when another daemon holds the PID
    /// file, and [`crate::DaemonError::Ipc`] when the port is taken — in which
    /// case the PID file is released again, since this process is not THE
    /// daemon after all.
    async fn claim_daemon_role(&self) -> Result<crate::ipc::ReservedIpcPort, crate::DaemonError> {
        self.acquire_pid_file()?;

        // The port is the second, kernel-arbitrated claim: it catches a live
        // daemon the PID file cannot vouch for (a PID file I/O failure, an
        // instance run with --no-pid-file). Reserved, not listening — clients
        // keep getting "connection refused" until the daemon is ready.
        let ipc_port = match self.ipc.reserve_port().await {
            Ok(port) => port,
            Err(e) => {
                error!(
                    "IPC port {} unavailable: {} — another daemon (or another program) holds it; \
                     exiting before touching storage",
                    self.ipc.address(),
                    e
                );
                if let Some(ref pid_file) = self.pid_file {
                    pid_file.release();
                }
                return Err(crate::DaemonError::Ipc(format!(
                    "IPC port {} unavailable: {e}",
                    self.ipc.address()
                )));
            }
        };

        self.arm_exit_reason_file();
        Ok(ipc_port)
    }

    /// Claim the single-instance PID file, when one is configured.
    ///
    /// # Errors
    ///
    /// [`crate::DaemonError::AlreadyRunning`] when another daemon holds it. Any
    /// other failure is logged and the daemon continues uncontained — a
    /// stat/write error on the PID path is not a reason not to serve.
    fn acquire_pid_file(&self) -> Result<(), crate::DaemonError> {
        // Acquire PID file to prevent multiple instances
        if let Some(ref pid_file) = self.pid_file {
            match pid_file.acquire() {
                Ok(()) => {
                    info!("PID file acquired at {:?}", pid_file.path());
                }
                Err(crate::health::PidFileError::AlreadyRunning(pid)) => {
                    error!("Another daemon instance is already running (PID: {})", pid);
                    return Err(crate::DaemonError::AlreadyRunning);
                }
                Err(e) => {
                    warn!("Failed to acquire PID file: {}. Continuing anyway.", e);
                }
            }
        }

        Ok(())
    }

    /// Report how the PREVIOUS daemon died, then claim the reason file for this
    /// process.
    ///
    /// Called after the PID acquire so a duplicate instance that loses the race
    /// can never touch the live daemon's record.
    fn arm_exit_reason_file(&self) {
        // Terminal reason file: report how the PREVIOUS daemon died, then
        // claim the file for this process. Runs after the PID acquire so a
        // duplicate instance that loses the race can never touch the live
        // daemon's record. A previous record still saying `running` is the
        // unclean-exit verdict — the process died through a path no hook
        // could see, which is exactly the 2026-08-10 log-just-ends death.
        {
            let previous = self.exit_reason.read_previous();
            if previous.is_unclean() {
                warn!("Previous exit was UNCLEAN: {}", previous.describe());
            } else {
                info!("Previous exit: {}", previous.describe());
            }
            self.exit_reason.mark_running();
            info!(
                "Exit reason file armed at {:?} — every exit path now records why it fired",
                self.exit_reason.path()
            );
        }
    }

    /// Load sessions from Turso, migrating a legacy `sessions.json` if that is
    /// all there is, and create the default session when there are none.
    async fn load_sessions(&self) {
        // Load sessions from Turso database
        {
            let loaded = self.sessions.load_from_db().await;
            info!("Loaded {} sessions from database", loaded);
        }

        // If no sessions loaded from DB, check for legacy sessions.json migration
        if self.sessions.count().await == 0
            && let Some((sessions, default_id)) = self.persistence.load_legacy_sessions().await
            && !sessions.is_empty()
        {
            info!(
                "Migrating {} sessions from legacy sessions.json to database",
                sessions.len()
            );
            for session in sessions {
                self.sessions.restore(session).await;
            }
            if let Some(id) = default_id {
                self.sessions.set_default(&id).await;
            }
            // Mark as migrated
            self.persistence.mark_sessions_migrated().await;
        }

        // Create default session if none exist
        if self.sessions.count().await == 0 {
            let default_session = self.sessions.create(Some("Main".to_string())).await;
            info!("Created default session: {}", default_session.id);
        }
    }

    /// Append a crashed run's recovered output to its session. `false` when
    /// the session does not exist, so the caller keeps the checkpoint.
    async fn repost_recovered(
        &self,
        session_id: &str,
        partial: crate::agent_service::ChatResult,
    ) -> bool {
        let reasoning = partial.reasoning.clone();
        let posted = self
            .sessions
            .add_full_message(
                session_id,
                crate::session::MessageRole::Assistant,
                &partial.content,
                crate::session::MessageDetails {
                    tool_calls: partial.tool_calls,
                    reasoning,
                    timeline: partial.timeline,
                    usage: partial.usage,
                },
            )
            .await
            .is_some();
        if posted {
            info!("Recovered crashed run for session {}", session_id);
        } else {
            warn!(
                "Crashed run's checkpoint names session {} which does not exist — keeping the checkpoint",
                session_id
            );
        }
        posted
    }

    /// Re-post the partial output of any run that crashed mid-turn.
    ///
    /// A checkpoint is deleted only after a SUCCESSFUL recovery (or when it
    /// holds nothing recoverable) — deleting before or regardless of the parse
    /// made any recovery failure a permanent data loss.
    async fn recover_checkpoints(&self, agent: &Arc<AgentService>) {
        // Recover any orphaned checkpoints from the database.
        if let Some(ref storage) = self.storage {
            match storage.list_checkpoints().await {
                Ok(checkpoint_ids) => {
                    for session_id in checkpoint_ids {
                        // Load checkpoint data from DB and parse it. The
                        // checkpoint is deleted only after a SUCCESSFUL
                        // recovery (or when it holds nothing recoverable) —
                        // deleting before/regardless of the parse made any
                        // recovery failure a permanent data loss.
                        let mut recovered = false;
                        if let Ok(Some(data)) = storage.load_checkpoint(&session_id).await {
                            if let Some(partial) = agent.recover_checkpoint_from_data(&data) {
                                // `None` = no such session: the output had nowhere
                                // to go. It used to be reported recovered and its
                                // checkpoint deleted — the output lost for good.
                                recovered = self.repost_recovered(&session_id, partial).await;
                            } else if serde_json::from_str::<serde_json::Value>(&data).is_ok() {
                                // Parsed fine but held nothing recoverable —
                                // an empty checkpoint is safe to clean up.
                                recovered = true;
                            } else {
                                warn!(
                                    "Checkpoint for session {} did not parse — keeping it for manual inspection",
                                    session_id
                                );
                            }
                        }
                        if recovered
                            && let Err(e) = storage.delete_checkpoint(&session_id).await
                        {
                            warn!(
                                "Failed to delete checkpoint for session {}: {}",
                                session_id, e
                            );
                        }
                    }
                }
                Err(e) => warn!("Failed to list checkpoints: {}", e),
            }

            // Also migrate any legacy checkpoint JSON files
            let checkpoint_dir = self.config.data_dir.join("checkpoints");
            if checkpoint_dir.exists()
                && let Ok(entries) = std::fs::read_dir(&checkpoint_dir)
            {
                for entry in entries.flatten() {
                    let filename = entry.file_name();
                    let name = filename.to_string_lossy();
                    if name.starts_with("checkpoint-") && name.ends_with(".json") {
                        let session_id = name
                            .strip_prefix("checkpoint-")
                            .and_then(|s| s.strip_suffix(".json"))
                            .unwrap_or("");
                        if !session_id.is_empty() {
                            let reposted = match agent.recover_checkpoint(session_id) {
                                Some(partial) => self.repost_recovered(session_id, partial).await,
                                None => true,
                            };
                            // Remove the legacy file — unless its output found no
                            // session to go to, as for the stored checkpoints.
                            if reposted {
                                let _ = std::fs::remove_file(entry.path());
                            }
                        }
                    }
                }
            }
        }
    }

    /// The single dreaming orchestrator (P13 unification).
    ///
    /// Built once and shared by BOTH consolidation paths — the scheduled cycle
    /// and the IPC `MemoryAction::Consolidate` handler — so they run the same
    /// multi-phase body over the same live store and accumulate pending feedback
    /// in one place. It reads the very `activity_clock` the control plane stamps,
    /// so its idle gate cannot drift from the daemon's own notion of "in use".
    fn build_dreaming(
        &self,
        memory: Option<&Arc<MemoryService>>,
        activity_clock: &Arc<nanna_memory::ActivityClock>,
    ) -> Option<Arc<nanna_memory::DreamingService>> {
        let dreaming: Option<Arc<nanna_memory::DreamingService>> = memory.map(|memory| {
            let dreaming_config = nanna_memory::DreamingConfig {
                idle_threshold_secs: self.config.dream_idle_threshold_secs,
                memory_pressure_count: self.config.dream_memory_pressure_count,
                ..nanna_memory::DreamingConfig::default()
            };
            Arc::new(
                nanna_memory::DreamingService::with_shared_memory(
                    dreaming_config,
                    Arc::clone(memory),
                )
                .with_activity_clock(Arc::clone(activity_clock)),
            )
        });

        if let Some(ref dreaming) = dreaming
            && self.dreaming_slot.set(Arc::clone(dreaming)).is_err()
        {
            warn!("dreaming slot was already filled; memory use feedback keeps the first service");
        }

        dreaming
    }

    /// Assemble the control plane, restore persisted workspaces onto it, and
    /// publish it to the late-bound slot the sub-agent spawner reads.
    ///
    /// Returns the plane and the channel status manager attached to it, which
    /// [`Self::start_channel_manager`] must share so `ChannelAction::Status` and
    /// the channel listeners see the same state.
    async fn build_control_plane(
        &self,
        deps: ControlPlaneDeps,
    ) -> (Arc<ControlPlane>, Arc<nanna_channels::StatusManager>) {
        let ControlPlaneDeps {
            agent,
            memory,
            tools,
            router,
            tools_dir,
            workspace_id_for_services,
            turn_baselines,
            scheduler,
            model_stats,
            chat_runs,
            degradations,
            activity_clock,
            dreaming,
        } = deps;

        // Create control plane with all services (including router for consolidation).
        // The data dir is the one this daemon resolved (`[general] data_dir`,
        // then `--data-dir`), so user tools live beside the store it opened.
        let mut control = ControlPlane::with_all_services(
            self.sessions.clone(),
            agent,
            memory.clone(),
            Some(tools),
            Some(router),
            self.config.data_dir.clone(),
        )
        .with_tools_dir(tools_dir)
        .with_audit_log_path(
            self.config
                .tool_audit
                .log
                .then(|| self.config.tool_audit_path()),
        )
        .with_event_tx(self.ipc.event_sender())
        .with_session_filters(self.ipc.session_filters())
        .with_workspace_id(workspace_id_for_services)
        .with_turn_baselines(turn_baselines)
        .with_scheduler(scheduler)
        .with_mcp_status(Arc::clone(&self.mcp_status))
        .with_task_runs(Arc::new(crate::tasks::TaskRunManager::new()))
        .with_memory_recovery(self.memory_recovery.clone())
        .with_live_embedding(self.live_embedding())
        .with_chat_runs(chat_runs.clone())
        .with_degradations(degradations.clone())
        .with_shutdown(self.shutdown_tx.clone());
        if let Some(ref buf) = self.log_buffer {
            control = control.with_log_buffer(buf.clone());
        }
        // Make the tracker the agent + sub-agents record into (from
        // init_services) the canonical one the control plane owns. Must happen
        // BEFORE with_storage, which loads persisted stats via
        // import_from_storage — those now land in the shared tracker too.
        control.model_stats = model_stats;
        // Load persisted model stats from storage
        if let Some(ref storage) = self.storage {
            control = control.with_storage(storage.clone()).await;
        }

        self.restore_workspaces(&control).await;

        // Read back the user tools the Tools page created (`ToolAction::Create`,
        // `{data_dir}/user_tools/{name}.json`). Nothing did before, so a tool
        // was registered live and then, at the next restart, gone from the
        // registry, from `ListUser` and from the model — with its file still
        // on disk. Disabled ones join the store but stay unregistered. After
        // the skills, as a live `Create` is, so a name clash settles the same
        // way across a restart; before IPC, so no request sees a partial
        // store. Not the `{data_dir}/tools` skill dirs that `create_tool` and
        // `edit_tool` author: those load with the skills in `init_services`.
        match control.load_user_tools().await {
            Ok(load) => info!(
                "Loaded {} user tools, registered {} (disabled ones stay unregistered)",
                load.loaded, load.registered
            ),
            Err(e) => warn!("User tools not loaded: {e}"),
        }

        // Wire model stats tracker into the router for health-aware routing.
        // The control plane owns the canonical tracker; the router reads it.
        if let Some(router) = control.router() {
            router.set_stats(control.model_stats.clone()).await;
            info!("Stats-informed routing enabled on LLM router");
        }

        // Shared channel status manager — attached before the Arc wrap so
        // ChannelAction::Status and ChannelManager listeners see the same state.
        let channel_status_manager = Arc::new(nanna_channels::StatusManager::new());
        control.set_status_manager(Arc::clone(&channel_status_manager));

        // Share the activity clock so chat requests stamp the same clock the
        // scheduled dream cycle reads for its idle gate.
        control.set_activity_clock(Arc::clone(&activity_clock));

        // Share the ONE dreaming orchestrator, so an IPC-triggered consolidation
        // runs the same multi-phase cycle the scheduler does (P13 unification).
        if let Some(ref dreaming) = dreaming {
            control.set_dreaming(Arc::clone(dreaming));
        }

        let control = Arc::new(control);
        // Hand edits of config.toml apply without a restart.
        control.spawn_config_watcher(self.shutdown_tx.subscribe());
        *self.control_slot.write().await = Some(control.clone());

        (control, channel_status_manager)
    }

    /// Re-register the workspaces the last session had, and seed the tool
    /// working directory from whichever one was active.
    ///
    /// Without that seed a fresh daemon with a persisted active workspace left
    /// `default_workdir` at `None` until the user re-selected it, so tools fell
    /// back to the home dir instead of running "in the workspace you're in".
    async fn restore_workspaces(&self, control: &ControlPlane) {
        // Load persisted workspaces from database
        if let Some(ref storage) = self.storage {
            match storage.workspaces().list().await {
                Ok(records) if !records.is_empty() => {
                    let mut registry = control.workspaces().write().await;
                    let mut active_id = None;
                    for record in &records {
                        let path = PathBuf::from(&record.path);
                        if path.exists() {
                            let mut ws = nanna_core::Workspace::new(&path);
                            ws.id = record.id.clone();
                            if let Err(e) = ws.load_context().await {
                                warn!(
                                    "Failed to load workspace context for {}: {}",
                                    record.path, e
                                );
                            }
                            registry.register(ws);
                            if record.active {
                                active_id = Some(record.id.clone());
                            }
                        } else {
                            warn!("Persisted workspace path no longer exists: {}", record.path);
                        }
                    }
                    if let Some(id) = active_id {
                        registry.set_active(&id);
                        // Seed the tool working directory from the persisted active
                        // workspace so tools resolve against it from boot — not just
                        // after an interactive SetActive or the first workspace-scoped
                        // chat. Without this, a fresh daemon with a persisted active
                        // workspace left `default_workdir` at None until the user
                        // re-selected it, so tools fell back to the home dir instead of
                        // running "in the workspace you're in".
                        let active_path = registry.get(&id).map(|ws| ws.path.clone());
                        drop(registry);
                        if let (Some(tools), Some(path)) = (control.tools(), active_path) {
                            tools.set_default_workdir(Some(path.clone())).await;
                            info!(
                                "Seeded tool working directory from active workspace: {:?}",
                                path
                            );
                        }
                    } else {
                        drop(registry);
                    }
                    info!("Restored {} workspaces from database", records.len());
                }
                Ok(_) => {}
                Err(e) => {
                    warn!("Failed to load workspaces from database: {}", e);
                }
            }
        }
    }

    /// Run the IPC listener, taking the daemon down if it ever fails.
    ///
    /// An IPC-less daemon is unreachable (no control plane) but would keep
    /// running heartbeats and burning the LLM budget — observed live when a
    /// second instance lost the port race.
    fn spawn_ipc_server(&self, ipc_port: crate::ipc::ReservedIpcPort) -> tokio::task::JoinHandle<()> {
        let ipc_server = self.ipc.clone();
        let ipc_shutdown = self.shutdown_tx.clone();
        let ipc_exit_reason = self.exit_reason.clone();
        tokio::spawn(async move {
            if let Err(e) = ipc_server.serve(ipc_port).await {
                // An IPC-less daemon is unreachable (no control plane) but
                // would keep running heartbeats and burning the LLM budget —
                // observed live when a second instance lost the port race.
                // Take the daemon down instead of zombie-ing.
                error!(
                    "IPC server error: {} — shutting down (a daemon without IPC is unreachable)",
                    e
                );
                // Record the cause before either exit route. If the clean
                // drain completes it overwrites this with `clean_shutdown`;
                // if the hard exit below fires, this record is the terminal
                // one — and process::exit skips every Drop, so nothing else
                // would have written it.
                ipc_exit_reason.record_exit("ipc_server_error", Some(&e.to_string()));
                if ipc_shutdown.send(()).is_err() {
                    // No shutdown listener yet — exit hard rather than linger.
                    std::process::exit(1);
                }
            }
        })
    }

    /// Tell subscribed clients about a startup quarantine + rebuild.
    ///
    /// Boot usually precedes any subscriber, so `/status` (health server +
    /// control plane) carries the same facts for late-connecting clients.
    fn announce_memory_recovery(&self) {
        // Announce a startup quarantine + rebuild to subscribed clients. Boot
        // usually precedes any subscriber, so /status (health server + control
        // plane) carries the same facts for late-connecting clients.
        if let Some(ref report) = self.memory_recovery {
            self.ipc
                .broadcast_event(crate::protocol::Event::MemoryStoreRebuilt {
                    recovered: report.memories_recovered,
                    expected: report.memories_expected,
                    quarantine_path: report.quarantine_path.to_string_lossy().to_string(),
                });
        }
    }

    /// Serve `/status` and `/metrics`, seeded with the durable store's health.
    async fn spawn_health_server(
        &self,
        control: &Arc<ControlPlane>,
        memory: Option<&Arc<MemoryService>>,
    ) -> Option<Arc<HealthState>> {
        if self.config.servers.health_server {
            // Seed durable-memory-store health (load already ran in init_services),
            // so a corrupt/degraded store shows on /status, not just a boot log.
            //
            // A store that never opened is degraded too, and more severely than
            // one with corrupt rows: there is no durable store at all, so the
            // per-store health probe cannot report on it. Fold that in here or
            // the most complete failure is the one /status calls healthy.
            let (mem_degraded, mem_corrupt) = if let Some(m) = memory {
                let h = m.store_health().await;
                (h.degraded || self.storage_error.is_some(), h.corrupt_rows)
            } else {
                (self.storage_error.is_some(), 0)
            };
            if let Some(ref err) = self.storage_error {
                error!(
                    error = %err,
                    "reporting degraded health: memory has no durable store this session"
                );
            }
            let mut state = HealthState::new(
                memory.is_some(),
                true, // agent is available
            )
            .with_memory_health(mem_degraded, mem_corrupt);
            if let Some(ref report) = self.memory_recovery {
                state = state
                    .with_memory_rebuild(report.memories_recovered, report.memories_expected);
            }
            let control_for_metrics = Arc::clone(control);
            state = state.with_metrics(Arc::new(move || {
                let control = Arc::clone(&control_for_metrics);
                Box::pin(async move {
                    crate::metrics::render_metrics(&control.metrics_snapshot().await)
                })
            }));
            let health_state = Arc::new(state);

            // Update session count
            let sessions_for_health = self.sessions.clone();
            let health_state_clone = health_state.clone();
            tokio::spawn(async move {
                loop {
                    let count = sessions_for_health.count().await;
                    health_state_clone.set_session_count(count).await;
                    tokio::time::sleep(Duration::from_secs(5)).await;
                }
            });

            // Serve the SAME state the session-count loop above updates
            // (via `from_shared`), so `/status` reflects live counts instead of
            // a throwaway copy stuck at zero. The server logs its own
            // "listening" line from `run()` once the bind succeeds, so we don't
            // pre-log here (that duplicate line falsely implied a bind before
            // one had happened).
            let health_server = HealthServer::from_shared(
                health_state.clone(),
                &self.config.ipc.host,
                self.config.health_port,
            );
            // Detached on purpose: the health server runs for the daemon's whole
            // life and has no shutdown hook, so dropping the join handle — what
            // this code always did — is the intent, not an oversight.
            drop(health_server.spawn());

            Some(health_state)
        } else {
            None
        }
    }

    /// Start the channel listeners (Telegram, Discord, Slack, …) when any are
    /// configured, sharing `status_manager` with the control plane.
    async fn start_channel_manager(
        &self,
        control: &Arc<ControlPlane>,
        channel_status_manager: &Arc<nanna_channels::StatusManager>,
    ) -> Option<Arc<ChannelManager>> {
        // This handles listener-based inbound (polling) and routes responses back out.
        if let Some(ref channels_config) = self.config.channels {
            // Build a daemon-local ChannelsConfig from the nanna_config::ChannelsConfig.
            // We re-map from nanna_config types to the daemon-local types.
            let daemon_channels = build_daemon_channels_config(channels_config);

            let mut manager = ChannelManager::with_status_manager(
                Arc::clone(control),
                Arc::clone(channel_status_manager),
            );
            manager.configure(&daemon_channels).await;

            // Also register outbound channels for webhook-sourced providers that have
            // bot tokens in the channel config (Telegram, Discord, Slack).
            // The listener-based configure() already does this; this is a no-op guard.

            match manager.start().await {
                Ok(()) => {
                    info!("Channel manager started");
                    Some(Arc::new(manager))
                }
                Err(e) => {
                    error!("Failed to start channel manager: {}", e);
                    None
                }
            }
        } else {
            None
        }
    }

    /// Serve inbound webhooks, routing them through the same pipeline the
    /// channel listeners use.
    async fn spawn_webhook_server(
        &self,
        control: &Arc<ControlPlane>,
        channel_manager: Option<&Arc<ChannelManager>>,
    ) {
        // Spawn webhook HTTP server if enabled
        if self.config.servers.webhook_server {
            let mut webhook_config = self.config.webhook.clone();
            webhook_config.host = self.config.ipc.host.clone();
            webhook_config.port = self.config.webhook_port;

            // Keep a copy of the config for outbound channel registration below
            let webhook_config_copy = webhook_config.clone();

            let (webhook_server, mut webhook_rx) = WebhookServer::new(webhook_config);

            // Spawn the webhook server
            tokio::spawn(async move {
                if let Err(e) = webhook_server.run().await {
                    error!("Webhook server error: {}", e);
                }
            });

            // Build a shared router for the webhook event processor.
            // If a ChannelManager is running, share its router so outbound channels
            // (bot tokens) are already registered.  Otherwise create a standalone
            // router that may only cover providers registered via webhook config.
            let webhook_router = if let Some(mgr) = channel_manager {
                mgr.router()
            } else {
                // No channel manager — create a standalone router.
                // Outbound channels can be registered here from webhook config if
                // bot tokens are provided.
                let standalone_router =
                    Arc::new(tokio::sync::RwLock::new(ChannelMessageRouter::new()));

                // Register outbound channels from webhook config credentials
                {
                    let mut router = standalone_router.write().await;
                    if let Some(ref token) = webhook_config_copy.telegram_token {
                        router.register("telegram", Box::new(TelegramChannel::new(token)));
                        info!("Registered Telegram outbound channel from webhook config");
                    }
                    // Registration is complete; the Discord note below reads only
                    // the webhook config, so the write lock is released here
                    // rather than held across it.
                    drop(router);
                    if webhook_config_copy.discord_public_key.is_some() {
                        // discord_public_key is for verification; bot token for sending
                        // is not separately stored in WebhookConfig currently.
                        // Log a warning — users should configure channels.discord instead.
                        debug!(
                            "Discord public key found in webhook config; for outbound replies configure channels.discord with a bot_token"
                        );
                    }
                }

                // This router has no channel manager's forwarder behind it, so
                // replies to webhook conversations need one of their own.
                if let Some(events) = control.subscribe_events() {
                    crate::channels::spawn_reply_forwarder(
                        Arc::clone(&control.sessions),
                        events,
                        Arc::clone(&standalone_router),
                        Arc::clone(&control.channel_counters),
                    );
                } else {
                    warn!("No event bus attached; webhook conversations will get no replies");
                }

                standalone_router
            };

            // Spawn webhook event processor — routes events through the same pipeline
            // as channel listener messages.
            let control_for_webhooks = Arc::clone(control);
            tokio::spawn(async move {
                while let Some(event) = webhook_rx.recv().await {
                    debug!("Webhook event from {}: {:?}", event.source, event.message);

                    if let Some(ref msg) = event.message {
                        // Convert WebhookMessage → IncomingMessage
                        let incoming = IncomingMessage {
                            id: msg
                                .message_id
                                .clone()
                                .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                            channel: ChannelId::new(&event.source, &msg.chat_id),
                            sender: ChannelSender {
                                id: msg.sender_id.clone(),
                                name: msg.sender_name.clone(),
                                username: None,
                            },
                            content: MessageContent::Text {
                                text: msg.content.clone(),
                            },
                            timestamp: event.timestamp,
                            reply_to: None,
                        };

                        // Process through the same pipeline as channel listeners
                        let router_guard = webhook_router.read().await;
                        ChannelManager::process_message(
                            incoming,
                            &control_for_webhooks,
                            &router_guard,
                        )
                        .await;
                    }
                }
            });

            info!(
                "Webhook server listening on http://{}:{}",
                self.config.ipc.host, self.config.webhook_port
            );
        }
    }

    /// Persist model + tool stats every five minutes, and once more on shutdown.
    fn spawn_stats_autosave(&self, control: &Arc<ControlPlane>) -> tokio::task::JoinHandle<()> {
        let stats_control = control.clone();
        let mut stats_shutdown = self.shutdown_tx.subscribe();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(300));
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        stats_control.save_model_stats().await;
                        stats_control.save_tool_stats().await;
                    }
                    _ = stats_shutdown.recv() => {
                        // Final save on shutdown
                        stats_control.save_model_stats().await;
                        stats_control.save_tool_stats().await;
                        info!("Model + tool stats final save completed");
                        break;
                    }
                }
            }
        })
    }

    /// Drain parent mailboxes for running sub-agents so questions don't go stale.
    ///
    /// When a sub-agent uses `ask_parent`, [`ParentChannelImpl`] answers it
    /// directly via an LLM call; this task handles any orphaned mailbox messages
    /// and provides visibility into long-running sub-agents.
    fn spawn_sub_agent_checkin(&self) {
        // Spawn sub-agent check-in task: periodically check running sub-agents
        // and drain parent mailboxes so questions don't go stale.
        // When a sub-agent uses ask_parent, the ParentChannelImpl handles it directly
        // via an LLM call. This task handles any orphaned mailbox messages and provides
        // visibility into long-running sub-agents.
        {
            let sessions = self.sessions.clone();
            let ipc_events = self.ipc.event_sender();
            let mut checkin_shutdown = self.shutdown_tx.subscribe();
            tokio::spawn(async move {
                // Check every 30 seconds
                let mut interval = tokio::time::interval(Duration::from_secs(30));
                loop {
                    tokio::select! {
                        _ = interval.tick() => {
                            let running = sessions.list_sub_sessions(None).await;
                            let running: Vec<_> = running.into_iter()
                                .filter(|s| matches!(s.state, crate::session::SubSessionState::Running | crate::session::SubSessionState::Spawning))
                                .collect();

                            if running.is_empty() {
                                continue;
                            }

                            // Check each parent's mailbox for pending questions
                            for sub in &running {
                                if let Some(ref parent_id) = sub.parent_id {
                                    let messages = sessions.drain_mailbox(parent_id).await;
                                    for msg in messages {
                                        // Re-emit as events for any listening clients (GUI)
                                        let _ = ipc_events.send(crate::protocol::Event::SubSessionQuestion {
                                            session_id: sub.session_id.clone(),
                                            parent_id: Some(parent_id.clone()),
                                            label: sub.label.clone(),
                                            question: msg.content,
                                        });
                                    }
                                }
                            }
                        }
                        _ = checkin_shutdown.recv() => break,
                    }
                }
            });
        }
    }

    /// The main event loop: dispatch every request as a task and return when
    /// shutdown is signalled.
    ///
    /// The loop is purely a router — it never blocks. The multi-threaded runtime
    /// (the default for `Runtime::new()`) schedules the spawned tasks across
    /// worker threads, so concurrent requests (e.g. session creation while an
    /// agent is running) execute in parallel.
    async fn serve_requests(
        &self,
        control: &Arc<ControlPlane>,
        request_rx: &mut tokio::sync::mpsc::Receiver<(crate::ipc::ConnectionId, crate::protocol::Request)>,
        shutdown_rx: &mut broadcast::Receiver<()>,
    ) {
        loop {
            tokio::select! {
                Some((client_id, request)) = request_rx.recv() => {
                    let control = control.clone();
                    let ipc = self.ipc.clone();

                    tokio::spawn(async move {
                        let request_id = request.id.clone();
                        let result = control.handle(&client_id, request.action).await;
                        let response = Response::success(request_id, result);
                        if let Err(e) = ipc.send_response(&client_id, response).await {
                            warn!("Failed to send response to client {}: {}", client_id, e);
                        }
                    });
                }

                _ = shutdown_rx.recv() => {
                    info!("Shutdown signal received");
                    break;
                }
            }
        }
    }

    /// Start the configured MCP servers in the background and keep the task
    /// that owns them, for `finish_shutdown` to await.
    async fn start_mcp_servers(
        &self,
        tools: &Arc<nanna_tools::ToolRegistry>,
        chat_runs: &Arc<crate::control::chat_harness::ChatRunRegistry>,
    ) {
        let mcp = crate::mcp_startup::spawn_mcp_servers(
            &self.config.mcp,
            Arc::clone(tools),
            Arc::clone(&self.mcp_status),
            self.shutdown_tx.subscribe(),
            |key| nanna_config::credentials::SecureStore::new().get(key).ok(),
            // A server's question (MCP elicitation) goes to the user through
            // `ask_user`, in the conversation whose turn called the tool.
            Some(Arc::new(crate::ask_user_service::McpAskUser {
                deps: crate::ask_user_service::AskUserDeps {
                    sessions: Arc::clone(&self.sessions),
                    events: self.ipc.event_sender(),
                    chat_runs: Arc::clone(chat_runs),
                },
            })),
        )
        .await;
        if let Some(task) = mcp.task
            && let Ok(mut slot) = self.mcp_task.lock()
        {
            debug_assert!(slot.is_none(), "services are initialized once");
            *slot = Some(task);
        }
    }

    /// Drain and stop: shut the IPC server, let the stats task take its final
    /// save, release the PID file and record the clean exit.
    async fn finish_shutdown(
        &self,
        channel_manager: Option<Arc<ChannelManager>>,
        stats_save_handle: tokio::task::JoinHandle<()>,
        ipc_handle: tokio::task::JoinHandle<()>,
    ) {
        // Cleanup
        info!("Shutting down daemon...");
        // Re-announce to every task subscribed since the request. One made
        // during boot (held for the serve loop, see `shutdown_rx`) predates
        // the stats, MCP and config-watch receivers, which would otherwise
        // wait out their deadlines below. A task that heard the first one has
        // stopped listening, and every one of them matches any `recv` result.
        let _ = self.shutdown_tx.send(());
        self.ipc.shutdown();

        // Keep channel_manager alive until the end of run() so the spawned
        // listener task doesn't get prematurely shut down via shutdown_tx drop.
        // Explicit drop here makes the intent clear.
        drop(channel_manager);

        // Wait for stats auto-save task to complete final save
        let _ = tokio::time::timeout(Duration::from_secs(5), stats_save_handle).await;

        // The MCP task closes its servers on the same broadcast; wait for it,
        // or the process exits before the close runs and a server that
        // ignores stdin EOF outlives the daemon.
        let mcp_task = self.mcp_task.lock().ok().and_then(|mut slot| slot.take());
        if let Some(task) = mcp_task {
            crate::mcp_startup::await_mcp_shutdown(task).await;
        }

        ipc_handle.abort();

        // Release PID file
        if let Some(ref pid_file) = self.pid_file {
            pid_file.release();
        }

        // The clean-exit record supersedes the startup `running` marker (and
        // any earlier cause record, e.g. a survived debug-mode panic or the
        // signal that initiated this drain — the log carries those details).
        self.exit_reason.record_exit("clean_shutdown", None);

        info!("Daemon stopped");
    }

    /// Initialize all services
    ///
    /// `chat_runs` is THE liveness source for background-vs-user contention
    /// (P22 Tier 4): the embedding drain pauses on it. `degradations` is the
    /// capability-transition ledger the embed path records into and every
    /// step runner drains.
    async fn init_services(
        &self,
        chat_runs: &Arc<crate::control::chat_harness::ChatRunRegistry>,
        degradations: &Arc<nanna_agent::DegradationLedger>,
    ) -> Result<
        (
            Arc<ToolRegistry>,
            Option<Arc<MemoryService>>,
            Arc<AgentService>,
            Arc<LlmRouter>,
            Option<PathBuf>,                          // tools_dir
            Arc<tokio::sync::RwLock<Option<String>>>, // workspace_id for script services
            Arc<crate::tasks::TurnBaselines>,         // turn-start closed-task baselines
            nanna_agent::ModelStatsTracker,           // shared model-stats tracker
        ),
        crate::DaemonError,
    > {
        let router = self.init_llm_router().await?;

        // Create empty tool registry — all tools loaded from disk
        let tools = Arc::new(ToolRegistry::new());

        let tools_dir = self.resolve_tools_directory();

        let memory = self.init_memory_service(chat_runs, degradations).await;
        self.start_board_write_through(memory.as_ref());

        // ONE config for every long-lived collaborator below. The sub-agent
        // spawner and the script summarizer are constructed BEFORE the agent
        // service, so the service adopts this same lock (`with_shared_config`)
        // and a later `config.set` reaches all three at once — the boot-clone
        // staleness that ran a whole benchmark series on the wrong summarizer
        // (2026-08-15).
        let shared_agent_config = Arc::new(tokio::sync::RwLock::new(self.config.agent.clone()));

        // Shared session history for the recall_messages tool service
        let session_history: SharedSessionHistory = Arc::new(tokio::sync::RwLock::new(Vec::new()));

        // One shared model-stats tracker for the whole daemon: the main agent
        // (AgentService) and every sub-agent (managed chats on the control
        // plane) record into it, and the control plane makes it canonical
        // (persists it + feeds the router). Cloning shares state
        // (Arc<RwLock<_>> inside).
        let model_stats = nanna_agent::ModelStatsTracker::new();
        self.install_request_sink(&model_stats);

        // Build script services and load all tools from disk
        let workspace_id_for_services: Arc<tokio::sync::RwLock<Option<String>>> =
            Arc::new(tokio::sync::RwLock::new(None));
        // ONE registry, shared by the chat harness (which registers a turn's
        // baseline) and the `tasks.*` services (whose `tasks.add` reads it).
        // Two registries would compile and silently guard nothing.
        let turn_baselines = Arc::new(crate::tasks::TurnBaselines::new());
        self.load_script_tool_surface(ToolSurfaceDeps {
            tools: &tools,
            tools_dir: tools_dir.as_ref(),
            router: &router,
            memory: &memory,
            session_history: &session_history,
            workspace_id_for_services: &workspace_id_for_services,
            turn_baselines: &turn_baselines,
            shared_agent_config: &shared_agent_config,
            chat_runs,
        })
        .await;

        // Register common aliases for Claude Code compatibility (after tools are loaded)
        Self::register_tool_aliases(&tools).await;

        // MCP servers register their tools as each handshake completes; the
        // count above is the tool surface before them.
        self.start_mcp_servers(&tools, chat_runs).await;

        // Register discover_tools (JS/TS skill with registry access)
        if let Some(ref dir) = tools_dir {
            if let Some(source) = nanna_tools::skills::defaults::load_discover_tools_source(dir) {
                let wrapper = nanna_tools::skills::ScriptedToolWrapper::from_source(
                    "discover_tools",
                    &source,
                )
                .expect("discover_tools skill must parse")
                .with_registry(Arc::downgrade(&tools));
                tools.register(wrapper).await;
                info!("Registered discover_tools skill from {:?}", dir);
            } else {
                warn!("discover_tools not found in tools directory");
            }
        }

        // Register ask_parent tool for sub-agent ↔ parent communication
        {
            let parent_channel: Arc<dyn ParentChannel + Send + Sync> =
                Arc::new(ParentChannelImpl {
                    sessions: self.sessions.clone(),
                    event_tx: Some(self.ipc.event_sender()),
                    router: router.clone(),
                    model: self.config.agent.model.clone(),
                });
            tools
                .register(nanna_tools::AskParentTool::new(
                    parent_channel,
                    tools.clone(),
                ))
                .await;
            info!("Registered ask_parent tool for sub-agent communication");
        }

        self.apply_tool_policy_and_audit(&tools).await;

        // Create agent service with multi-provider router
        let event_tx = self.ipc.event_sender();
        let mut agent_service = AgentService::with_data_dir(
            self.config.agent.clone(),
            router.clone(),
            tools.clone(),
            memory.clone(),
            event_tx,
            Some(self.config.data_dir.clone()),
        )
        .with_session_history(session_history)
        .with_stats(model_stats.clone())
        .with_degradations(degradations.clone())
        // Adopt the lock the spawner and the script summarizer already hold,
        // so `config.set` moves all three at once instead of only this one.
        .with_shared_config(Arc::clone(&shared_agent_config));
        if let Some(ref storage) = self.storage {
            agent_service = agent_service.with_storage(storage.clone());
        }
        let agent = Arc::new(agent_service);

        info!("Agent service initialized");

        Ok((
            tools,
            memory,
            agent,
            router,
            tools_dir,
            workspace_id_for_services,
            turn_baselines,
            model_stats,
        ))
    }

    /// Build the script-service map and load every skill that can reach it.
    ///
    /// Both directories are loaded: the resolved tools dir (the source tree in a
    /// debug build, the extracted defaults in a release build) and the data
    /// dir's `tools/`, where runtime-authored skills are written. In a release
    /// build they are the same path and the second load is skipped; in a debug
    /// build the second is the other half of the tool surface, and without it an
    /// authored tool would be callable for one session and gone after a restart.
    async fn load_script_tool_surface(&self, deps: ToolSurfaceDeps<'_>) {
        let ToolSurfaceDeps {
            tools,
            tools_dir,
            router,
            memory,
            session_history,
            workspace_id_for_services,
            turn_baselines,
            shared_agent_config,
            chat_runs,
        } = deps;

        let spawner_arc: Option<Arc<dyn AgentSpawner + Send + Sync>> =
            if router.available_providers().is_empty() {
                None
            } else {
                Some(Arc::new(AgentSpawnerImpl {
                    router: router.clone(),
                    // The live config, not a snapshot of it — see
                    // `AgentSpawnerImpl::agent_config_src`.
                    agent_config_src: Arc::clone(shared_agent_config),
                    control: self.control_slot.clone(),
                }))
            };

        // `[memory] ocr_model_priority` already means "vision-capable
        // models, tried in order" — an existing documented setting, so
        // `vision.analyze` reads it rather than inventing a second one.
        let vision_models = self.config.vision_model_priority.clone();

        // The authoring services load a tool they just wrote with the same
        // services every bundled skill gets. That map is the one being
        // built, so it reaches them through a slot filled immediately
        // below — a runtime-authored tool must not be the only one in the
        // daemon that cannot call a service.
        let authoring_slot: Arc<std::sync::OnceLock<HashMap<String, ServiceFn>>> =
            Arc::new(std::sync::OnceLock::new());
        // Author into the data dir, never into wherever skills happen to be
        // *loaded* from. In a debug build `resolve_tools_dir` returns the
        // source tree, so `tools.create` would have written a new skill
        // directory into the checkout — untracked files appearing in the
        // repo because an agent made a tool. Loading from source is
        // deliberate and stays; writing to it is not.
        let authoring_dir = self.config.data_dir.join("tools");
        let tool_authoring = Some((
            authoring_dir.clone(),
            Arc::downgrade(tools),
            Arc::clone(&authoring_slot),
        ));

        // Before any skill runs: the bridge snapshots into this store on
        // every script write from here on.
        let file_history_root = self.config.data_dir.join("file-history");
        if nanna_scripting::file_history::install(file_history_root.clone()) {
            info!(
                "File history (pre-write snapshots) at {}",
                file_history_root.display()
            );
        }

        let services = build_script_services(ScriptServiceDeps {
            memory: memory.clone(),
            spawner: spawner_arc,
            session_history: session_history.clone(),
            workspace_id: workspace_id_for_services.clone(),
            storage: self.storage.clone(),
            turn_baselines: turn_baselines.clone(),
            summarizer: Some((router.clone(), Arc::clone(shared_agent_config))),
            tool_authoring,
            vision: Some((router.clone(), vision_models.clone())),
            pdf_ocr: crate::vision_service::bind_pdf_ocr_fn(router, &vision_models),
            audio: Some((
                self.config.llm.openai_api_key.clone(),
                self.config.data_dir.clone(),
            )),
            browser_data_dir: Some(self.config.data_dir.clone()),
            screenshot_data_dir: Some(self.config.data_dir.clone()),
            reminders: Some((Arc::clone(&self.scheduler_slot), Arc::clone(&self.sessions))),
            feedback: Some(Arc::clone(&self.dreaming_slot)),
            ask_user: Some(crate::ask_user_service::AskUserDeps {
                sessions: Arc::clone(&self.sessions),
                events: self.ipc.event_sender(),
                chat_runs: Arc::clone(chat_runs),
            }),
        });
        // Fill the slot before any skill can be executed. `set` returning
        // an error would mean the map was filled twice, which cannot
        // happen here and would leave the authoring services reading a
        // stale map if it did.
        if authoring_slot.set(services.clone()).is_err() {
            warn!(
                "tool-authoring service map was already filled; authored tools may not reach services"
            );
        }

        if let Some(dir) = tools_dir
            && dir.is_dir()
        {
            let loaded = tools.load_skills_with_services(dir, &services).await;
            info!("Loaded {} tools from {:?}", loaded, dir);
        }

        // Tools authored at runtime live in the data dir. In a release
        // build that is the same directory as above and this is skipped; in
        // a debug build it is the other half of the tool surface, and
        // without it an authored tool would be callable for one session and
        // gone after a restart.
        if authoring_dir.is_dir()
            && tools_dir != Some(&authoring_dir)
        {
            let loaded = tools
                .load_skills_with_services(&authoring_dir, &services)
                .await;
            if loaded > 0 {
                info!("Loaded {} authored tools from {:?}", loaded, authoring_dir);
            }
        }
    }

    /// The LLM router, rebuilt from every credential the config file and the
    /// secure store between them can resolve.
    ///
    /// # Errors
    ///
    /// [`crate::DaemonError::Config`] when not one provider resolved — a daemon
    /// with no model to call can only fail every request, so it refuses to boot
    /// rather than look healthy.
    async fn init_llm_router(&self) -> Result<Arc<LlmRouter>, crate::DaemonError> {
        // Create LLM router with all available providers. The same resolution
        // + construction runs again on every control-plane config mutation
        // (see `control::config`), so a provider the user authenticates after
        // boot registers without a daemon restart — registration used to be
        // boot-only, leaving the GUI and daemon split-brained about which
        // providers exist.
        let router = LlmRouter::new();
        let creds = crate::llm_router::ProviderCredentials::resolve(&self.config.llm).await;
        router.rebuild(&creds);

        let available = router.available_providers();
        if available.is_empty() {
            return Err(crate::DaemonError::Config(
                "No LLM providers configured. Please set up at least one provider (Anthropic, OpenAI, OpenRouter, or Ollama).".to_string()
            ));
        }
        info!(
            "LLM router initialized with {} providers: {:?}",
            available.len(),
            available
        );
        let router = Arc::new(router);

        Ok(router)
    }

    /// Where skills are loaded from: env var > config > dev fallback >
    /// `{data_dir}/tools/`, with the bundled defaults extracted on first run.
    /// `None` when `[tools] use_script_tools` is off.
    fn resolve_tools_directory(&self) -> Option<PathBuf> {
        if self.config.use_script_tools {
            let config_dir = self.config.tools_dir.as_deref();
            let resolved = nanna_tools::skills::defaults::resolve_tools_dir(config_dir)
                .unwrap_or_else(|| self.config.data_dir.join("tools"));

            // Bootstrap default skills into the tools directory if needed.
            // In debug builds this is a no-op (tools load from source tree).
            // In release builds, embedded skills are extracted on first run.
            let bootstrapped = nanna_tools::skills::defaults::bootstrap_default_skills(&resolved);
            if bootstrapped > 0 {
                info!(
                    "Bootstrapped {} default skills into {:?}",
                    bootstrapped, resolved
                );
            }

            if resolved.is_dir() {
                let granted_count = nanna_tools::skills::defaults::ensure_permissions(&resolved);
                if granted_count > 0 {
                    warn!(
                        granted_count,
                        "Wrote the default permissions.json for {} tool(s) that declared none",
                        granted_count
                    );
                }
                info!("Tools directory: {:?}", resolved);
            } else {
                warn!("Tools directory does not exist: {:?}", resolved);
            }
            Some(resolved)
        } else {
            None
        }
    }

    /// Start draining the board write-through queue (P25 Stage 1): memory
    /// copies when memory is enabled, timeline episodes whenever storage is.
    ///
    /// Runs at most once: the receiver is taken. Without storage the receiver
    /// is dropped here, and the sink reads the closed queue as "nothing owed"
    /// instead of filling it.
    fn start_board_write_through(&self, memory: Option<&Arc<MemoryService>>) {
        let queue = self
            .board_copies
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        if let (Some(queue), Some(storage)) = (queue, self.storage.as_ref()) {
            tokio::spawn(crate::memory_write_through::run(
                queue,
                Arc::clone(storage),
                memory.cloned(),
            ));
            info!(
                memory_copies = memory.is_some(),
                "Board write-through running: cards and posts feed the timeline and memory"
            );
        }
    }

    /// The memory service, or `None` when memory is switched off in config.
    ///
    /// Always `Some` once memory is enabled, even with no embedding provider —
    /// see [`Self::memory_without_embedder`] for why a missing key must not
    /// discard a session's memories.
    async fn init_memory_service(
        &self,
        chat_runs: &Arc<crate::control::chat_harness::ChatRunRegistry>,
        degradations: &Arc<nanna_agent::DegradationLedger>,
    ) -> Option<Arc<MemoryService>> {
        if !self.config.enable_memory {
            info!("Memory service disabled in config");
            return None;
        }

        // Resolve the ordered provider list the user actually configured.
        //
        // `embedding_priority` is authoritative and is walked in order:
        // the first entry whose credential resolves becomes primary, the
        // rest become fallbacks in the same order. Failover happens ONLY on
        // error, exactly like `model_priority` for chat, so one model
        // embeds at a time.
        //
        // What this replaces: a single configured provider plus THREE
        // hardcoded fallbacks appended by credential sniffing —
        // `text-embedding-3-small` (1536) for OpenAI and
        // `openai/text-embedding-3-small` (1536) for OpenRouter. Those were
        // injected whether or not the user wanted them, which is how a paid
        // 1536-dim embedder ended up live on an install whose config asked
        // for a free one first and a 768-dim local model second.
        if self.embedding.priority.is_empty() {
            info!(
                "No embedding_priority configured; using the single pair '{}/{}'",
                self.embedding.provider, self.embedding.model
            );
        }
        let specs = self.embedding.specs();

        let resolved: Vec<(EmbeddingProviderInfo, Arc<nanna_llm::EmbeddingClient>)> = specs
            .iter()
            .filter_map(|spec| self.embedding_provider_for(spec))
            .collect();

        // Queue rows for models no longer in the list can never drain and
        // would pollute queue health forever. Pruned only when something
        // resolved — an empty resolution means a missing key, and wiping
        // the queue over a missing key would turn a config hiccup into
        // lost work tracking.
        if !resolved.is_empty()
            && let Some(ref storage) = self.storage
        {
            let keep: Vec<String> = resolved.iter().map(|(info, _)| info.to_string()).collect();
            match storage.memories().retain_queue_models(&keep).await {
                Ok(0) => {}
                Ok(n) => info!("Dropped {n} queued embeddings for models no longer configured"),
                Err(e) => warn!("Could not prune the embedding queue: {e}"),
            }
        }

        let primary_client = resolved.first().cloned();
        let fallbacks: Vec<_> = resolved.into_iter().skip(1).collect();

        if primary_client.is_none() && !specs.is_empty() {
            warn!(
                "No embedding provider in embedding_priority could be resolved ({} entries tried) — \
                 memory will be written without vectors and queued for backfill",
                specs.len()
            );
        }

        let Some((primary_info, primary)) = primary_client else {
            return self.memory_without_embedder(degradations).await;
        };

        let embed_router = Self::build_embedding_router(&primary_info, primary, fallbacks);

        // Placeholder for memory service — set after construction via lazy init
        let memory_for_reembed: Arc<tokio::sync::OnceCell<Arc<MemoryService>>> =
            Arc::new(tokio::sync::OnceCell::new());
        // One drain at a time, process-wide — see `drain_backfill`.
        let drain_serial = Arc::new(tokio::sync::Mutex::new(()));
        let embed_fn = Self::build_embed_fn(
            &embed_router,
            &memory_for_reembed,
            &drain_serial,
            chat_runs,
            degradations,
        );

        let dimension = self.probe_or_seed_dimension(&embed_router).await;
        let config = nanna_memory::MemoryServiceConfig {
            dimension,
            ..Default::default()
        };

        // Wire up Turso persistence if storage is available.
        // The persistence adapter is constructed here and attached to the
        // MemoryService so all writes are automatically mirrored to Turso.
        let memory_service = if let Some(ref storage) = self.storage {
            let repo = storage.memories();
            let db = Arc::new(TursoMemoryPersistence::new(repo));
            nanna_memory::MemoryService::new(config)
                .with_embed_fn(embed_fn)
                .with_persistence(db)
        } else {
            warn!(
                "No storage backend available — memory will NOT be persisted to Turso"
            );
            nanna_memory::MemoryService::new(config).with_embed_fn(embed_fn)
        };

        self.load_persisted_memories(&memory_service).await;

        info!("Memory service initialized with Turso persistence and embedding router");
        let memory_arc = Arc::new(memory_service);

        // Probe the actual embedding dimension from the model IN THE
        // BACKGROUND. The probe's first embed call can take ~a minute
        // when the local embedding model is cold (Ollama loads it on
        // demand), and it used to block startup past the GUI's
        // daemon-ready timeout — forcing an embedded fallback while an
        // orphaned daemon kept running. `probe_and_align_dimension`
        // takes `&self` on the Arc'd service specifically so it can run
        // at runtime; a mismatched dimension is corrected (and entries
        // re-embedded) as soon as the probe completes.
        Self::spawn_embedding_maintenance(
            &memory_arc,
            &embed_router,
            dimension,
            &primary_info,
            chat_runs,
            &drain_serial,
        )
        .await;

        // Wire the memory service into the embed_fn's OnceCell
        // so provider-switch re-embedding can find it
        let _ = memory_for_reembed.set(memory_arc.clone());

        // One long-lived worker for the vectors a live turn parks.
        // It sleeps on the queue's notify and costs nothing until a
        // tool result is ingested; see `drain_queued_vectors` for
        // why it is a separate drain and what bounds it.
        let memory_for_queue = memory_arc.clone();
        let drain_serial_for_queue = drain_serial.clone();
        tokio::spawn(async move {
            drain_queued_vectors(&memory_for_queue, &drain_serial_for_queue).await;
        });

        Some(memory_arc)
    }

    /// The resolved list's first entry is primary; the rest are fallbacks in
    /// the user's own order.
    ///
    /// Fallbacks are the REST OF THE USER'S LIST, not a credential sweep: the
    /// list is the whole policy — what to try, and in what sequence.
    fn build_embedding_router(
        primary_info: &EmbeddingProviderInfo,
        primary: Arc<nanna_llm::EmbeddingClient>,
        fallbacks: Vec<(EmbeddingProviderInfo, Arc<nanna_llm::EmbeddingClient>)>,
    ) -> Arc<EmbeddingRouter> {
        // Build the embedding router with fallback providers
        let mut embed_router = EmbeddingRouter::new(primary_info.clone(), primary);

        // Fallbacks are the REST OF THE USER'S LIST, in their
        // order — not a credential sweep. The list is the whole
        // policy: what to try, and in what sequence.
        for (info, client) in fallbacks {
            info!("Embedding fallback: {info}");
            embed_router = embed_router.with_fallback(info, client);
        }

        info!(
            "Embedding router: {} providers configured",
            embed_router.provider_count()
        );
        Arc::new(embed_router)
    }

    /// The store's embed function: one seam every embedding outcome passes
    /// through, which is why the capability ledger and the provider-switch
    /// rebind both live here rather than at the call sites.
    fn build_embed_fn(
        embed_router: &Arc<EmbeddingRouter>,
        memory_for_reembed: &Arc<tokio::sync::OnceCell<Arc<MemoryService>>>,
        drain_serial: &Arc<tokio::sync::Mutex<()>>,
        chat_runs: &Arc<crate::control::chat_harness::ChatRunRegistry>,
        degradations: &Arc<nanna_agent::DegradationLedger>,
    ) -> nanna_memory::EmbedFn {
        // Create embedding function that routes through the EmbeddingRouter.
        let router_for_fn = embed_router.clone();
        let mem_cell_for_fn = memory_for_reembed.clone();
        let drain_serial_for_fn = drain_serial.clone();
        let chat_runs_for_fn = chat_runs.clone();
        let ledger_for_fn = degradations.clone();

        let embed_fn: nanna_memory::EmbedFn = Arc::new(move |text: &str| {
            let router = router_for_fn.clone();
            let text = text.to_string();
            let mem_cell = mem_cell_for_fn.clone();
            let drain_serial = drain_serial_for_fn.clone();
            let chat_runs = chat_runs_for_fn.clone();
            let ledger = ledger_for_fn.clone();
            Box::pin(async move {
                let attempted = router.embed_one(&text).await;

                // Capability transitions reach the model once, in
                // its next tool result (P22 Tier 4). The seam is
                // HERE — the one place every embed outcome passes —
                // because "no provider answered" is the moment
                // memory writes start landing vectorless, and the
                // first success afterwards is the moment they stop.
                // The ledger dedups by state, so the steady flow of
                // successes (and repeated failures) records nothing.
                match &attempted {
                    Err(reason) => ledger.set(
                        "memory-embeddings",
                        "degraded",
                        format!(
                            "[capability notice — memory embeddings DEGRADED: no \
                             embedding provider is answering ({reason}). Memory and \
                             tool-result writes still SUCCEED and are stored in full — \
                             the Turso store remains the source of truth — but new \
                             entries are queued for embedding backfill, so semantic \
                             recall may miss them until a provider recovers. This \
                             notice will not repeat unless the state changes.]"
                        ),
                    ),
                    Ok(_) => ledger.set(
                        "memory-embeddings",
                        "healthy",
                        "[capability notice — memory embeddings RESTORED: an \
                         embedding provider is answering again; queued entries are \
                         backfilling and new writes are searchable normally.]",
                    ),
                }

                let (embedding, switched_to) = attempted?;

                // Provider switched — realign the store BEFORE this
                // write is allowed to land, so it validates against
                // the new binding rather than the dead provider's.
                //
                // The router reports the switch to exactly ONE
                // caller (the one whose call flipped the live
                // active index), so this is stampede-safe without
                // any generation bookkeeping here. And the vector
                // in hand came from `switched_to` itself, so
                // `(model, embedding.len())` is a consistent pair
                // by construction — reading the active provider
                // back from the router here could race a second
                // switch and rebind the store to a torn
                // (new model, old width) state. That torn state is
                // the 2026-08-02 incident: model rebound, width
                // latch stale, every write failing
                // "expected 2048, got 768" for minutes.
                if let Some(provider) = switched_to
                    && let Some(mem) = mem_cell.get()
                {
                    let model = provider.to_string();
                    // The new provider's input window, memoized by
                    // the router — one `/api/show` per provider per
                    // process, and it travels WITH the model and
                    // width for the same reason those two do. A
                    // chunk sized for the old window is not
                    // rejected by the new embedder, it is silently
                    // truncated.
                    let window = router.context_window_for(&provider).await;
                    tracing::info!(
                        "Embedding provider changed — rebinding the store to \
                         '{}' ({} dims)",
                        model,
                        embedding.len()
                    );

                    // Rebinding is a hash lookup per entry: no
                    // network, no re-embed, and a switch BACK to a
                    // model used earlier is free because its bucket
                    // was retained.
                    let (_, missing) =
                        mem.rebind_embeddings(&model, embedding.len(), window).await;

                    // Whatever this model has never embedded gets
                    // filled in lazily, in bounded passes, while
                    // the run continues. It must not be done
                    // inline: the store can hold thousands of
                    // entries and the provider we just failed over
                    // to may be the rate-limited one.
                    //
                    // Unconditional for the same reason as the
                    // startup bind: `missing` counts ROW vectors,
                    // and the chunk queue is independent of it. A
                    // flap back to a model whose row buckets were
                    // all retained reports zero missing while its
                    // chunk vectors are still stamped with the
                    // other provider, and a drain interrupted
                    // partway leaves exactly that state.
                    let _ = missing;
                    let mem = mem.clone();
                    let chat_runs = chat_runs.clone();
                    let drain_serial = drain_serial.clone();
                    tokio::spawn(async move {
                        drain_backfill(&mem, &model, &chat_runs, &drain_serial)
                            .await;
                    });
                }

                Ok(embedding)
            })
        });

        embed_fn
    }

    /// The dimension the store starts on: the provider's own answer when it
    /// responds, the provisional seed when it does not.
    async fn probe_or_seed_dimension(&self, embed_router: &EmbeddingRouter) -> usize {
        // Seed the embedding dimension by probing the router.
        //
        // A probe failure must NOT stop the daemon: Nanna is
        // offline-capable by default, so an unreachable or unkeyed
        // embedding provider degrades memory — it does not refuse
        // to boot. The seed only has to be a valid positive
        // dimension: real vectors always come from the provider,
        // and the background `probe_and_align_dimension` below
        // corrects the store (re-embedding any mismatched entries)
        // as soon as a provider answers. Probing here is purely an
        // optimization — when it succeeds the store is right
        // immediately and nothing is ever re-embedded.
        let seed_dimension = nanna_memory::MemoryServiceConfig::default().dimension;
        let dimension = match Self::probe_embedding_dimension(embed_router).await {
            Ok(dim) => {
                info!(
                    "Memory service using probed dimension {} for model {}",
                    dim, self.embedding.model
                );
                dim
            }
            Err(e) => {
                warn!(
                    "Could not probe the embedding dimension ({e}). Starting anyway with a \
                     provisional dimension of {seed_dimension}; memory will re-align \
                     automatically once an embedding provider is reachable. To enable \
                     embeddings, run a local Ollama with `ollama pull {}` or set an \
                     OpenAI/OpenRouter key.",
                    self.embedding.model
                );
                seed_dimension
            }
        };
        assert!(dimension > 0, "embedding dimension must be positive");

        dimension
    }

    /// Fill the in-memory cache from Turso, migrating a legacy `memories.json`
    /// exactly once if one is there and the table is empty.
    async fn load_persisted_memories(&self, memory_service: &nanna_memory::MemoryService) {
        // One-time migration: if memories.json exists and Turso is empty,
        // load from JSON into in-memory cache then save each entry to Turso.
        let json_path = self.memory_path.as_ref();
        let should_migrate = if let (Some(path), Some(storage)) =
            (json_path, &self.storage)
        {
            if path.exists() {
                match storage.memories().count().await {
                    Ok(0) => true,
                    Ok(n) => {
                        info!(
                            "Turso already has {} memories — skipping JSON migration",
                            n
                        );
                        false
                    }
                    Err(e) => {
                        warn!("Could not check Turso memory count: {}", e);
                        false
                    }
                }
            } else {
                false
            }
        } else {
            false
        };

        if should_migrate {
            let path = json_path.unwrap();
            info!(
                "Migrating memories from {:?} to Turso (one-time migration)",
                path
            );
            match memory_service.load(path).await {
                Ok(()) => {
                    let count = memory_service.count().await;
                    info!("Loaded {} memories from JSON, flushing to Turso...", count);
                    // Flush all entries to Turso
                    match memory_service.flush_to_db().await {
                        Ok(n) => info!("Flushed {} memories to Turso", n),
                        Err(e) => warn!("Failed to flush memories to Turso: {}", e),
                    }
                    // Rename the JSON file so we don't re-migrate next time
                    let migrated_path = path.with_extension("json.migrated");
                    if let Err(e) = tokio::fs::rename(path, &migrated_path).await {
                        warn!("Could not rename migrated JSON file: {}", e);
                    } else {
                        info!(
                            "Renamed {:?} → {:?} (migration complete)",
                            path, migrated_path
                        );
                    }
                }
                Err(e) => {
                    warn!(
                        "JSON migration failed: {}. Will attempt to load from Turso.",
                        e
                    );
                }
            }
        }

        // Load from Turso into the in-memory cache (normal startup path).
        // Skipped if we just migrated (the entries are already in-memory from the JSON load above).
        if !should_migrate {
            match memory_service.load_from_db().await {
                Ok(count) => {
                    info!("Loaded {} memories from Turso", count);
                }
                Err(nanna_memory::MemoryError::Persistence(ref e))
                    if e.contains("No persistence backend") =>
                {
                    // No storage configured — silently skip
                }
                Err(e) => {
                    warn!("Failed to load memories from Turso: {}", e);
                }
            }
        }
    }

    /// Bind the store to the model that is about to write to it, arm the
    /// idle-backfill supervisor, and correct the dimension in the background.
    async fn spawn_embedding_maintenance(
        memory_arc: &Arc<MemoryService>,
        embed_router: &Arc<EmbeddingRouter>,
        dimension: usize,
        primary_info: &EmbeddingProviderInfo,
        chat_runs: &Arc<crate::control::chat_harness::ChatRunRegistry>,
        drain_serial: &Arc<tokio::sync::Mutex<()>>,
    ) {
        let memory_for_probe = memory_arc.clone();
        // Name the model the router actually bound — not the
        // legacy `embedding.model` config field, which the
        // priority list overrides. A log that names the wrong
        // model sends whoever reads it to debug a provider that
        // is not even running.
        let model_name = primary_info.to_string();
        // Bind the store to the model that is about to write to
        // it, BEFORE any probe or write. Without this the first
        // entries of a session get bucketed under `None` — they
        // would be re-embedded on the next switch instead of
        // being reusable, which is the whole point of buckets.
        let bind_provider = embed_router.active_provider().await;
        let bind_model = bind_provider.to_string();
        let memory_for_bind = memory_arc.clone();
        let router_for_bind = embed_router.clone();
        let chat_runs_for_bind = chat_runs.clone();
        let drain_serial_for_bind = drain_serial.clone();
        tokio::spawn(async move {
            // The probed (or seeded) dimension and the model's
            // input window travel WITH the model — the binding
            // is one triple, never three independently-updated
            // latches. Resolved inside the spawn because it
            // costs a request, and startup must not block on it.
            let window =
                router_for_bind.context_window_for(&bind_provider).await;
            let (_, _missing) = memory_for_bind
                .rebind_embeddings(&bind_model, dimension, window)
                .await;
            // Unconditional, NOT gated on missing row vectors.
            // The two queues are independent: after the chunk
            // migration every existing memory has a row vector
            // and no chunks at all, so `missing == 0` while the
            // entire store is unchunked. Gating on the row
            // count would leave it that way forever. Both
            // drains no-op immediately when their queue is
            // empty, so the unconditional call costs one query
            // each on a store that is already complete.
            drain_backfill(
                &memory_for_bind,
                &bind_model,
                &chat_runs_for_bind,
                &drain_serial_for_bind,
            )
            .await;
        });
        // One supervisor for the daemon's life: it turns the
        // end of every turn into a drain opportunity, which is
        // what closes the "parked until the next binding event"
        // gap `store_unembedded` documents -- the backlog
        // `drain_queued_vectors` is budgeted not to sweep.
        // Spawned beside the startup bind because that is where
        // the binding, the run registry and the drain mutex are
        // all in scope.
        let memory_for_supervisor = memory_arc.clone();
        let chat_runs_for_supervisor = chat_runs.clone();
        let drain_serial_for_supervisor = drain_serial.clone();
        tokio::spawn(supervise_idle_backfill(
            memory_for_supervisor,
            chat_runs_for_supervisor,
            drain_serial_for_supervisor,
        ));
        let chat_runs_for_probe = chat_runs.clone();
        let drain_serial_for_probe = drain_serial.clone();
        tokio::spawn(async move {
            match memory_for_probe.probe_and_align_dimension().await {
                Ok(actual_dim) => {
                    if actual_dim == dimension {
                        debug!(
                            "Embedding dimension confirmed: {actual_dim} for model {model_name}"
                        );
                    } else {
                        info!(
                            "Embedding dimension corrected: {dimension} → {actual_dim} for model {model_name}"
                        );
                        // Writes that landed under the stale
                        // width were queued for backfill, not
                        // failed — drain them now that the
                        // binding is honest.
                        if let Some(model) =
                            memory_for_probe.active_embedding_model().await
                        {
                            drain_backfill(
                                &memory_for_probe,
                                &model,
                                &chat_runs_for_probe,
                                &drain_serial_for_probe,
                            )
                            .await;
                        }
                    }
                }
                Err(e) => {
                    warn!(
                        "Could not probe embedding dimension (model may be loading): {e}. \
                         Using static dimension {dimension}."
                    );
                }
            }
        });
    }

    /// Memory with persistence and no embedder.
    ///
    /// Writes land in Turso with no vector — exactly the queued-for-backfill
    /// state the loader and the drain already handle — and they become
    /// searchable the moment a provider is configured and the daemon restarts.
    /// This arm used to return `None`, which contradicted the warning promising
    /// vectorless writes and quietly discarded every memory of a session that
    /// merely had a missing API key.
    async fn memory_without_embedder(
        &self,
        degradations: &Arc<nanna_agent::DegradationLedger>,
    ) -> Option<Arc<MemoryService>> {
        // No provider resolved — but memory does NOT switch off.
        // The service runs with persistence and no embedder:
        // writes land in Turso with no vector, exactly the
        // queued-for-backfill state the loader and the drain
        // already handle, and they become searchable the moment a
        // provider is configured and the daemon restarts. The old
        // arm returned `None` here, which contradicted the warn
        // above it promising vectorless writes — and quietly
        // discarded every memory of a session that merely had a
        // missing API key.
        warn!(
            "No embedding provider available — memory runs WITHOUT vectors: writes \
             persist and queue for backfill, recall is unavailable until an \
             embedding provider is configured"
        );
        // The model finds out the same way the operator does —
        // once, in its first tool result, not on every write.
        degradations.set(
            "memory-embeddings",
            "degraded",
            "[capability notice — memory embeddings are OFF: no embedding provider \
             is configured. Memory and tool-result writes still SUCCEED and persist \
             in full — the Turso store is the source of truth — but they carry no \
             vectors, so semantic recall is unavailable until a provider is \
             configured. This notice will not repeat unless the state changes.]",
        );
        let config = nanna_memory::MemoryServiceConfig::default();
        let memory_service = if let Some(ref storage) = self.storage {
            let repo = storage.memories();
            let db = Arc::new(TursoMemoryPersistence::new(repo));
            nanna_memory::MemoryService::new(config).with_persistence(db)
        } else {
            // No storage either: in-RAM only, still better than
            // dropping writes on the floor for the session.
            nanna_memory::MemoryService::new(config)
        };
        match memory_service.load_from_db().await {
            Ok(count) => info!("Loaded {count} memories from Turso (no embedder yet)"),
            Err(nanna_memory::MemoryError::Persistence(ref e))
                if e.contains("No persistence backend") => {}
            Err(e) => warn!("Failed to load memories from Turso: {e}"),
        }
        Some(Arc::new(memory_service))
    }

    /// Mirror every provider request into the per-request history table.
    ///
    /// Lifetime totals cannot be split back into days and `system.cost_rollup`
    /// prices days, so the table needs its own writer; it had none until this
    /// sink.
    fn install_request_sink(&self, model_stats: &nanna_agent::ModelStatsTracker) {
        // Keep the per-request history too: lifetime totals cannot be split
        // back into days, and `system.cost_rollup` prices days. The table
        // existed with no writer until this sink.
        if let Some(ref storage) = self.storage {
            let storage = Arc::clone(storage);
            model_stats.set_request_sink(Arc::new(move |obs: &nanna_agent::RequestObservation| {
                let storage = Arc::clone(&storage);
                let obs = obs.clone();
                // Read here, on the run's task: the task-local does not cross
                // the spawn below.
                let session_id = ToolRegistry::run_session_id();
                tokio::spawn(async move {
                    let latency_ms = u64::try_from(obs.latency.as_millis()).unwrap_or(u64::MAX);
                    let tier = obs.tier.map(|t| format!("{t:?}").to_lowercase());
                    if let Err(e) = storage
                        .log_model_request(&nanna_storage::NewModelRequest {
                            model: &obs.model,
                            success: obs.success,
                            latency_ms,
                            input_tokens: obs.input_tokens,
                            output_tokens: obs.output_tokens,
                            cache_read_tokens: obs.cache_read_tokens,
                            cache_creation_tokens: obs.cache_creation_tokens,
                            cache_creation_1h_tokens: obs
                                .cache_creation_1h_tokens
                                .min(obs.cache_creation_tokens),
                            tier: tier.as_deref(),
                            escalated: obs.escalated,
                            session_id: session_id.as_deref(),
                        })
                        .await
                    {
                        warn!("Failed to log model request: {e}");
                    }
                });
            }));
        }
    }

    /// Claude Code-compatible aliases, registered after the tools they point at.
    async fn register_tool_aliases(tools: &Arc<ToolRegistry>) {
        // Register common aliases for Claude Code compatibility (after tools are loaded)
        tools.register_alias("read", "read_file").await;
        tools.register_alias("Read", "read_file").await;
        tools.register_alias("write", "write_file").await;
        tools.register_alias("Write", "write_file").await;
        tools.register_alias("edit", "edit_file").await;
        tools.register_alias("Edit", "edit_file").await;
        tools.register_alias("bash", "exec").await;
        tools.register_alias("Bash", "exec").await;
        tools.register_alias("glob", "list_dir").await;
        tools.register_alias("Glob", "list_dir").await;
        tools.register_alias("ls", "list_dir").await;
        tools.register_alias("task", "sub_agent").await;
        tools.register_alias("Task", "sub_agent").await;
        tools.register_alias("sub-agent", "sub_agent").await;
        tools.register_alias("Sub-Agent", "sub_agent").await;

        {
            let tool_count = tools.definitions().await.len();
            info!("Tool registry: {} tools (including aliases)", tool_count);
        }
    }

    /// The allow/deny policy and the per-call audit trail, both attached to the
    /// registry rather than the agent loop: the loop is only one of the callers
    /// (chat harness, task tool, scheduled runs and the MCP bridge all call the
    /// registry directly), and an audit that saw only one of them would be worse
    /// than none — it would read as a complete account.
    async fn apply_tool_policy_and_audit(&self, tools: &Arc<ToolRegistry>) {
        // Apply the tool allow/deny policy. Built from `[tools] enabled/disabled`
        // and enforced by the registry AFTER alias/fuzzy resolution, so a denied
        // tool cannot be reached via an alias (`Bash` → `exec`), a case variant,
        // or a fuzzy near-miss. Skipped entirely when unrestricted.
        let policy = build_tool_policy(
            self.config.tool_allowlist.as_deref(),
            &self.config.tool_denylist,
        );
        if !policy.is_unrestricted() {
            tools.set_policy(policy).await;
        }

        // Attach the per-call audit trail. It sits on the registry rather than
        // on the agent loop deliberately: the loop is only one of the callers
        // (chat harness, task tool, scheduled runs and the MCP bridge all call
        // the registry directly), and an audit that saw only one of them would
        // be worse than none — it would read as a complete account.
        if self.config.tool_audit.log {
            let path = self.config.tool_audit_path();
            let sink = nanna_tools::JsonlAuditSink::new(
                path.clone(),
                nanna_tools::ToolAuditConfig {
                    include_values: self.config.tool_audit.log_values,
                    ..Default::default()
                },
            );
            tools.set_audit_sink(Some(Arc::new(sink))).await;
            info!(path = %path.display(), values = self.config.tool_audit.log_values,
                  "Tool audit trail enabled");
        }
    }
}

/// Build a [`ToolPolicy`] from the `[tools] enabled`/`disabled` config lists.
///
/// Semantics chosen to match the existing config surface:
/// - `enabled` acts as an allowlist. The conventional wildcard `"*"` (the
///   default) means "no allowlist in force" — every tool is permitted. An empty
///   list would deny everything, which nobody writes by accident because the
///   default is `["*"]`; we treat an empty allowlist as unrestricted too, so a
///   caller that forgets to populate it fails *open* on the allow side (the
///   denylist still applies) rather than silently muting every tool.
/// - `disabled` is the denylist and always wins (fail closed on conflict).
fn build_tool_policy(enabled: Option<&[String]>, disabled: &[String]) -> ToolPolicy {
    // Thin wrapper over the shared interpretation in `nanna-tools`, so the
    // daemon and `nanna mcp serve` cannot drift on what `[tools] enabled` /
    // `disabled` mean. The daemon's tests below pin the behaviour from this side.
    ToolPolicy::from_config_lists(enabled, disabled)
}

/// Split an `embedding_priority` entry into `(provider, model)`.
///
/// Splits on the FIRST slash only: model names routinely contain slashes of
/// their own (`openrouter/nvidia/nemotron-3-embed-1b:free` is provider
/// `openrouter`, model `nvidia/nemotron-3-embed-1b:free`), and splitting on the
/// last slash — or on all of them — silently addresses a different model than
/// the one written down.
///
/// An entry with no slash is treated as a bare model on the default local
/// provider, matching how a bare chat model name resolves.
pub(crate) fn split_embedding_spec(spec: &str) -> Option<(String, String)> {
    let spec = spec.trim();
    if spec.is_empty() {
        return None;
    }
    match spec.split_once('/') {
        // A slash was written, so a provider WAS stated. If the model half is
        // empty the entry is malformed and must be rejected — falling through
        // to the bare-name arm below would resolve `openrouter/` to the local
        // provider with the literal model name `openrouter/`, which is a
        // request that can only fail at the network, far from the typo.
        Some((provider, model)) => {
            let model = model.trim();
            if model.is_empty() || provider.trim().is_empty() {
                return None;
            }
            Some((provider.trim().to_ascii_lowercase(), model.to_string()))
        }
        // "nomic-embed-text:latest" — no provider stated.
        None => Some(("ollama".to_string(), spec.to_string())),
    }
}

#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    /// Provider (ollama, openai, openrouter)
    pub provider: String,
    /// Model name
    pub model: String,
    /// Ollama host (if using Ollama)
    pub ollama_host: String,
    /// Bearer token for that host — `[llm] ollama_api_key`, the one Ollama
    /// credential, which chat already sends. `None` sends no `Authorization`.
    pub ollama_api_key: Option<String>,
    /// Ordered `provider/model` specs, most preferred first.
    ///
    /// This is the user's stated order and it is authoritative: the router
    /// tries them in sequence and falls to the next ONLY on failure, exactly
    /// like `model_priority` does for chat. Empty means "not configured",
    /// which falls back to the single [`Self::provider`]/[`Self::model`] pair.
    ///
    /// This field existed in the config file and in the settings UI for a long
    /// time while nothing in the daemon read it. The list was therefore inert:
    /// a user could put a free embedder first and watch a hardcoded paid one
    /// run instead, with no way to tell from the outside.
    pub priority: Vec<String>,
}

impl EmbeddingConfig {
    /// The embedding settings as the config file states them — shared by boot
    /// and the reload path, so the two cannot derive different embedders from
    /// the same file.
    #[must_use]
    pub fn from_nanna(config: &nanna_config::Config) -> Self {
        Self {
            provider: config.memory.embedding_provider.clone(),
            model: config.memory.embedding_model.clone(),
            ollama_host: config.memory.ollama_host.clone(),
            // The one Ollama credential, blank meaning none (as chat reads it).
            ollama_api_key: config
                .llm
                .ollama_api_key
                .as_deref()
                .map(str::trim)
                .filter(|key| !key.is_empty())
                .map(str::to_string),
            priority: config.memory.embedding_priority.clone(),
        }
    }

    /// The ordered `provider/model` specs the router is built from:
    /// [`Self::priority`], or the single legacy pair when it is empty.
    #[must_use]
    pub fn specs(&self) -> Vec<String> {
        if self.priority.is_empty() {
            vec![format!("{}/{}", self.provider, self.model)]
        } else {
            self.priority.clone()
        }
    }
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            provider: "ollama".to_string(),
            model: "nomic-embed-text".to_string(),
            ollama_host: "http://localhost:11434".to_string(),
            ollama_api_key: None,
            priority: Vec::new(),
        }
    }
}

/// Builder for [`DaemonServer`]
pub struct DaemonBuilder {
    config: DaemonConfig,
    embedding: EmbeddingConfig,
    memory_path: Option<PathBuf>,
    brave_api_key: Option<String>,
    log_buffer: Option<crate::log_buffer::LogBuffer>,
}

/// Copy the signature-verification secrets from the user's channel config into a
/// [`WebhookConfig`]. Only providers the user configured are set, so unset
/// providers keep the previous value — and a provider that ends up with no
/// secret serves no webhook at all (`refuse_unconfigured` in `webhook.rs`).
///
/// Telegram used to be absent here on the reasoning that "Telegram
/// authenticates via the bot token in the URL". That was wrong about this
/// implementation: the route is the fixed path `/webhook/telegram` with no
/// token in it, so nothing was ever verified. `telegram_secret` now carries the
/// `setWebhook` secret token, which Telegram echoes as
/// `X-Telegram-Bot-Api-Secret-Token` on every POST.
fn apply_channel_webhook_secrets(
    webhook: &mut WebhookConfig,
    channels: &nanna_config::ChannelsConfig,
) {
    if let Some(ref discord) = channels.discord {
        webhook.discord_public_key = Some(discord.public_key.clone());
    }
    if let Some(ref slack) = channels.slack {
        webhook.slack_signing_secret = Some(slack.signing_secret.clone());
    }
    if let Some(ref whatsapp) = channels.whatsapp {
        webhook
            .whatsapp_verify_token
            .clone_from(&whatsapp.verify_token);
        webhook.whatsapp_app_secret.clone_from(&whatsapp.app_secret);
    }
    if let Some(ref telegram) = channels.telegram {
        webhook.telegram_token = Some(telegram.bot_token.clone());
        webhook.telegram_secret.clone_from(&telegram.webhook_secret);
    }
}

impl DaemonBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: DaemonConfig::default(),
            embedding: EmbeddingConfig::default(),
            memory_path: None,
            brave_api_key: None,
            log_buffer: None,
        }
    }

    /// Create builder from Nanna config file
    ///
    /// # Errors
    ///
    /// Never, today: a config file that fails to load falls back to
    /// [`nanna_config::Config::default`] with env overrides, and every field
    /// below is infallible. The `Result` is kept because this is the daemon's
    /// one construction seam and a future required setting (a credential that
    /// cannot be defaulted, say) must be able to refuse to boot here rather
    /// than panic.
    pub fn from_nanna_config() -> Result<Self, crate::DaemonError> {
        use nanna_config::Config;

        let config = match Config::load() {
            Ok(cfg) => {
                info!("Loaded Nanna config successfully");
                record_legacy_ollama_token_server(&cfg.memory.ollama_host);
                cfg.with_env_overrides()
            }
            Err(e) => {
                warn!(
                    "Failed to load Nanna config: {}, using defaults with env overrides",
                    e
                );
                Config::default().with_env_overrides()
            }
        };

        let mut builder = Self::new();

        // Set LLM configuration - copy all provider credentials (shared with
        // the control-plane reload path so both derive providers identically)
        builder.config.llm = LlmConfig::from_nanna(&config);

        // Split into one applier per `[section]` of the config file, called in
        // the order the sections used to appear inline. Each is a pure
        // field-for-field copy, so the sequence carries no dependency — but it
        // is kept anyway, because `apply_data_dir` logs before
        // `apply_agent_settings` does and a reordered boot log is a worse
        // diff to read than a reordered function.
        builder.apply_memory_settings(&config);
        builder.apply_scheduler_settings(&config);
        builder.apply_data_dir(&config);
        builder.apply_agent_settings(&config);
        builder.apply_tool_and_channel_settings(&config);
        builder.log_resolved_config();

        Ok(builder)
    }

    /// `[memory]`: the embedding binding, the dream cycle's compression and
    /// idle settings, the vision-model list and the MCP server list.
    fn apply_memory_settings(&mut self, config: &nanna_config::Config) {
        // Set embedding configuration from Nanna memory config
        self.embedding = EmbeddingConfig::from_nanna(config);

        // Thread the memory-compression settings so the scheduled dream cycle
        // honors them (previously only the IPC-triggered path did).
        self.config.memory_max_compression_ratio = config.memory.max_compression_ratio;
        self.config.memory_min_remaining_memories = config.memory.min_remaining_memories;
        // Idle gate for the scheduled dream cycle (defers dreaming to a lull).
        self.config.dream_idle_threshold_secs = config.memory.dream_idle_threshold_secs;
        self.config.dream_memory_pressure_count = config.memory.dream_memory_pressure_count;
        // `ocr_model_priority` already means "vision-capable models, in order";
        // `vision.analyze` reads it rather than adding a second list.
        self.config
            .vision_model_priority
            .clone_from(&config.memory.ocr_model_priority);
        self.config.mcp = config.mcp.clone();
    }

    /// `[scheduler]`, plus the webhook signature secrets that live under
    /// `[channels]` but are only ever read by the webhook server.
    fn apply_scheduler_settings(&mut self, config: &nanna_config::Config) {
        // Scheduler switches. The daemon owns the scheduler (P16), so without
        // this the GUI's Scheduler tab is dead UI and the heartbeat is
        // unconditional — which is how it kept stealing the local model's one
        // slot mid-chat.
        self.config.scheduler.enabled = config.scheduler.enabled;
        self.config.scheduler.heartbeat_enabled = config.scheduler.heartbeat_enabled;
        self.config.heartbeat_interval_secs = config.scheduler.heartbeat_interval_secs;

        // Wire webhook signature-verification secrets from the user's channel
        // config. Without this the inbound webhook server always ran with the
        // all-None default, so every provider's verification silently no-op'd —
        // the daemon never received the secrets it checks against.
        apply_channel_webhook_secrets(&mut self.config.webhook, &config.channels);
    }

    /// `[general] data_dir`, and the legacy `memories.json` path under it.
    fn apply_data_dir(&mut self, config: &nanna_config::Config) {
        // Set data directory from Nanna config (same location as GUI).
        //
        // `resolve_data_dir` — not `default_data_dir`: the latter ignores
        // `[general] data_dir` entirely, which is what made that setting inert
        // (it parsed, validated and round-tripped while changing nothing).
        // This is the single place the override is honoured; the daemon's
        // `--data-dir` flag is applied *after* the builder is constructed, in
        // `nanna-daemon/src/main.rs`, so an isolated run still wins.
        match config.resolve_data_dir() {
            Ok(data_dir) => {
                if config.has_custom_data_dir() {
                    // Say it out loud. Pointing this at a new folder does not
                    // move an existing store — the daemon opens whatever is
                    // there and creates an empty one if nothing is — and an
                    // empty store looks exactly like data loss to the person it
                    // happens to.
                    info!(
                        "Using CONFIGURED data directory: {:?} (set by [general] data_dir; \
                         the platform default is not in use)",
                        data_dir
                    );
                } else {
                    info!("Using Nanna data directory: {:?}", data_dir);
                }
                self.config.data_dir.clone_from(&data_dir);
                self.memory_path = Some(data_dir.join("memories.json"));
            }
            Err(e) => {
                warn!("Could not determine Nanna data dir: {}, using default", e);
            }
        }
    }

    /// `[llm]` + `[agent]`: the model chain, summarization, the loop's
    /// iteration policy and the routing/sub-agent settings.
    fn apply_agent_settings(&mut self, config: &nanna_config::Config) {
        // Set agent configuration from loaded config
        // Use user-configured model priority list for fallback
        self.config
            .agent
            .model_priority
            .clone_from(&config.llm.model_priority);
        info!("Model priority list: {:?}", config.llm.model_priority);

        if let Some(model) = config.llm.model_priority.first() {
            self.config.agent.model.clone_from(model);
        } else {
            self.config.agent.model.clone_from(&config.llm.model);
        }

        // Set summarization configuration. The list is all it takes: each
        // entry resolves through the chat router, with chat's server, token
        // and keys (`llm_router::summarizer_clients`).
        self.config
            .agent
            .summarization_priority
            .clone_from(&config.llm.summarization_priority);

        // Thinking mode is NOT read from config: it is always on (owner
        // directive 2026-08-04). `AgentServiceConfig::default` already carries
        // `ThinkingMode::default()`, and the `agent.thinking_enabled` flag that
        // used to gate it here is gone.

        // Agent-loop iteration policy: unbounded by default (long-horizon worker),
        // with late escalating soft nudges. All three are user-configurable.
        self.config.agent.max_iterations = config.agent.max_iterations;
        self.config.agent.nudge_after_iterations = config.agent.nudge_after_iterations;
        self.config.agent.nudge_interval_iterations = config.agent.nudge_interval_iterations;

        // Set model routing configuration
        self.config
            .agent
            .model_routing
            .clone_from(&config.llm.model_routing);
        self.config.agent.routing_first_turn_primary = config.llm.routing_first_turn_primary;
        self.config
            .agent
            .sub_agent_model
            .clone_from(&config.llm.sub_agent_model);
        // Resolved here (list > legacy single > main chat list) so every
        // consumer sees one authoritative, never-empty chain.
        self.config.agent.sub_agent_models = config.llm.effective_sub_agent_models();
        self.config.agent.prompt_cache_ttl =
            crate::agent_service::cache_ttl_from(config.llm.prompt_cache_ttl);
        if !config.llm.model_routing.is_empty() {
            info!("Model routing enabled: {:?}", config.llm.model_routing);
        }
        if !config.llm.sub_agent_models.is_empty() || config.llm.sub_agent_model.is_some() {
            info!("Sub-agent models: {:?}", self.config.agent.sub_agent_models);
        }
    }

    /// `[tools]` + `[channels]`: the Brave key, the script-tool surface, the
    /// allow/deny policy, the audit trail and the configured chat channels.
    fn apply_tool_and_channel_settings(&mut self, config: &nanna_config::Config) {
        // Set Brave API key for web search
        self.brave_api_key.clone_from(&config.tools.brave_api_key);

        // Set script tools flag and tools directory
        self.config.use_script_tools = config.tools.use_script_tools;
        self.config.tools_dir.clone_from(&config.tools.tools_dir);

        // Tool allow/deny policy — `[tools] enabled` is the allowlist ("*" = all),
        // `[tools] disabled` is the denylist. This is the wiring that makes a
        // disabled tool actually stop executing (the lists were previously
        // parsed into config but never enforced).
        self.config.tool_allowlist = Some(config.tools.enabled.clone());
        self.config.tool_denylist.clone_from(&config.tools.disabled);
        self.config.tool_audit.log = config.tools.audit_log;
        self.config.tool_audit.log_values = config.tools.audit_log_values;

        // Load channel configuration (Telegram, Discord, Slack, etc.)
        let has_channels = config.channels.telegram.is_some()
            || config.channels.discord.is_some()
            || config.channels.slack.is_some()
            || config.channels.signal.is_some()
            || config.channels.whatsapp.is_some();
        if has_channels {
            self.config.channels = Some(config.channels.clone());
            info!("Channel configuration loaded");
        }
    }

    /// The one-line boot summary: which providers resolved, and what the
    /// daemon is about to run on.
    fn log_resolved_config(&self) {
        // Log configured providers
        let mut providers = Vec::new();
        if self.config.llm.anthropic_api_key.is_some()
            || self.config.llm.anthropic_oauth_token.is_some()
        {
            providers.push("anthropic");
        }
        if self.config.llm.openai_api_key.is_some() {
            providers.push("openai");
        }
        if self.config.llm.openrouter_api_key.is_some() {
            providers.push("openrouter");
        }
        if self.config.llm.github_token.is_some() {
            providers.push("github");
        }
        providers.push("ollama"); // Always available

        info!(
            "Daemon config loaded: model={}, embedding={}:{}, providers=[{}], brave_key={}",
            self.config.agent.model,
            self.embedding.provider,
            self.embedding.model,
            providers.join(", "),
            if self.brave_api_key.is_some() {
                "set"
            } else {
                "none"
            }
        );
    }

    #[must_use]
    pub const fn with_port(mut self, port: u16) -> Self {
        self.config.ipc.port = port;
        self
    }

    #[must_use]
    pub fn with_host(mut self, host: impl Into<String>) -> Self {
        self.config.ipc.host = host.into();
        self
    }

    #[must_use]
    pub fn with_data_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.data_dir = path.into();
        self
    }

    #[must_use]
    pub fn with_log_level(mut self, level: impl Into<String>) -> Self {
        self.config.log_level = level.into();
        self
    }

    #[must_use]
    pub const fn with_auto_save_interval(mut self, secs: u64) -> Self {
        self.config.auto_save_interval_secs = secs;
        self
    }

    #[must_use]
    pub fn with_llm_provider(mut self, provider: impl Into<String>) -> Self {
        self.config.llm.provider = provider.into();
        self
    }

    #[must_use]
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.config.llm.api_key = Some(key.into());
        self
    }

    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.config.agent.model = model.into();
        self
    }

    /// The chat model priority list, head first — what `[llm]
    /// model_priority` sets, so a test can exercise the fallback walk.
    #[must_use]
    pub fn with_model_priority(mut self, models: Vec<String>) -> Self {
        if let Some(head) = models.first() {
            self.config.agent.model.clone_from(head);
        }
        self.config.agent.model_priority = models;
        self
    }

    /// Point the chat router's Ollama provider at `host` — the only provider
    /// a hermetic test can stand up locally, so it is what lets an end-to-end
    /// test run a real conversation turn against a scripted server.
    ///
    /// # Panics
    ///
    /// If `host` is not an `http(s)://` URL — a caller passing a bare
    /// `host:port` is a programmer error the router would otherwise turn into
    /// a connection failure on the first turn.
    #[must_use]
    pub fn with_ollama_host(mut self, host: impl Into<String>) -> Self {
        let host = host.into();
        assert!(
            host.starts_with("http://") || host.starts_with("https://"),
            "an Ollama host is an http(s) base URL, got {host:?}"
        );
        self.config.llm.ollama_host = host;
        self
    }

    /// Master switch for the scheduler (`[scheduler] enabled`). Off, no
    /// heartbeat or cron turn fires — so a test that counts a model's
    /// requests counts only the ones it caused.
    #[must_use]
    pub const fn with_scheduler(mut self, enabled: bool) -> Self {
        self.config.scheduler.enabled = enabled;
        self
    }

    /// The periodic heartbeat turn (`[scheduler] heartbeat_enabled`), under a
    /// running scheduler. Off, reminders and cron still fire but no heartbeat
    /// turn competes with the conversation a test is scripting.
    #[must_use]
    pub const fn with_heartbeat(mut self, enabled: bool) -> Self {
        self.config.scheduler.heartbeat_enabled = enabled;
        self
    }

    #[must_use]
    pub const fn with_memory(mut self, enable: bool) -> Self {
        self.config.enable_memory = enable;
        self
    }

    #[must_use]
    pub const fn with_health_server(mut self, enable: bool) -> Self {
        self.config.servers.health_server = enable;
        self
    }

    #[must_use]
    pub const fn with_health_port(mut self, port: u16) -> Self {
        self.config.health_port = port;
        self
    }

    #[must_use]
    pub const fn with_pid_file(mut self, enable: bool) -> Self {
        self.config.servers.pid_file = enable;
        self
    }

    #[must_use]
    pub const fn with_webhook_server(mut self, enable: bool) -> Self {
        self.config.servers.webhook_server = enable;
        self
    }

    #[must_use]
    pub const fn with_webhook_port(mut self, port: u16) -> Self {
        self.config.webhook_port = port;
        self
    }

    #[must_use]
    pub fn with_webhook_config(mut self, config: WebhookConfig) -> Self {
        self.config.webhook = config;
        self
    }

    #[must_use]
    pub const fn with_script_tools(mut self, enable: bool) -> Self {
        self.config.use_script_tools = enable;
        self
    }

    #[must_use]
    pub fn with_log_buffer(mut self, buffer: crate::log_buffer::LogBuffer) -> Self {
        self.log_buffer = Some(buffer);
        self
    }

    /// Assemble the server. Opens nothing: storage opens in
    /// [`DaemonServer::run`] once this process has claimed the instance.
    #[must_use]
    pub fn build(self) -> DaemonServer {
        let mut server = DaemonServer::new(
            self.config,
            self.embedding,
            self.memory_path,
            self.brave_api_key,
        );
        server.log_buffer = self.log_buffer;

        server
    }
}

impl Default for DaemonBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Helper: convert nanna_config::ChannelsConfig → daemon-local ChannelsConfig
// =============================================================================

/// Map the high-level `nanna_config::ChannelsConfig` (with its richer field names)
/// to the daemon-local `ChannelsConfig` used by `ChannelManager::configure()`.
///
/// Fields that exist in `nanna_config` but not in the daemon-local type are
/// silently dropped — the local type only covers what `ChannelManager` actually
/// needs at runtime.
fn build_daemon_channels_config(src: &nanna_config::ChannelsConfig) -> ChannelsConfig {
    use crate::channels::{
        DiscordConfig as DaemonDiscord, SlackConfig as DaemonSlack,
        TelegramConfig as DaemonTelegram,
    };

    ChannelsConfig {
        telegram: src.telegram.as_ref().map(|tg| DaemonTelegram {
            bot_token: tg.bot_token.clone(),
            // nanna_config::TelegramConfig uses webhook_url; treat presence of it as
            // "use webhooks" mode (listener-based polling is disabled when webhook URL set).
            allowed_chats: tg.allowed_users.clone().unwrap_or_default(),
            use_webhooks: tg.webhook_url.is_some(),
        }),
        discord: src.discord.as_ref().map(|dc| DaemonDiscord {
            bot_token: dc.bot_token.clone(),
            allowed_guilds: vec![], // nanna_config::DiscordConfig has no allowed_guilds yet
            intents: None,
        }),
        slack: src.slack.as_ref().and_then(|sl| {
            // Slack Socket Mode listener requires an app_token; fall back gracefully
            sl.app_token.as_ref().map(|app_token| DaemonSlack {
                app_token: app_token.clone(),
                bot_token: sl.bot_token.clone(),
                allowed_channels: vec![], // nanna_config::SlackConfig has no allowed_channels yet
            })
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A JS number reaches a service as a float (`2.0`). A strict `as_u64`
    /// read it as absent and silently used the default instead; the model's
    /// `limit` had no effect at all.
    #[tokio::test]
    async fn a_float_count_from_script_is_the_count_it_says() {
        let history: SharedSessionHistory = Arc::new(tokio::sync::RwLock::new(
            (0..5)
                .map(|n| crate::session::SessionMessage {
                    id: n.to_string(),
                    role: crate::session::MessageRole::User,
                    content: format!("message {n}"),
                    timestamp: chrono::Utc::now(),
                    tool_calls: Vec::new(),
                    attachments: Vec::new(),
                    reasoning: None,
                    timeline: Vec::new(),
                    usage: None,
                })
                .collect(),
        ));
        let services = session_history_services(history);
        let two = services["session.history"](serde_json::json!({ "limit": 2.0 }))
            .await
            .expect("a float count is read");
        assert_eq!(two.as_array().map(Vec::len), Some(2), "{two}");
        let refused = services["session.history"](serde_json::json!({ "limit": "lots" }))
            .await
            .expect_err("a count that is not a number is named, not defaulted");
        assert!(refused.contains("limit"), "{refused}");
    }

    /// With no Settings summarization list, `memory.summarize` walks the chat
    /// models in their configured order — the rule every memory consumer
    /// shares. It used to take the single chat model, so a first chat model
    /// with no provider failed the call while a second one could answer.
    #[tokio::test(flavor = "multi_thread")]
    async fn memory_summarize_walks_the_chat_models_when_settings_list_none() {
        use crate::embedding_reload::test_ollama::{ollama_credentials, spawn_token_gated};
        let (host, seen) = spawn_token_gated("bound-token").await;
        let router = Arc::new(crate::llm_router::LlmRouter::new());
        router.rebuild(&ollama_credentials(&host, "bound-token"));
        let chat_models = vec![
            // No OpenRouter key is registered: this one cannot answer.
            "openrouter/nanna-test/unreachable".to_string(),
            "ollama/nanna-test-chat:1b".to_string(),
        ];
        let config = Arc::new(tokio::sync::RwLock::new(
            crate::agent_service::AgentServiceConfig {
                model: chat_models[0].clone(),
                model_priority: chat_models,
                summarization_priority: Vec::new(),
                ..crate::agent_service::AgentServiceConfig::default()
            },
        ));
        let services = memory_summarize_services(router, config);
        let summarize = services.get("memory.summarize").expect("registered");

        let reply = summarize(serde_json::json!({ "texts": ["one fragment", "another"] }))
            .await
            .expect("the second chat model answers");

        assert!(reply["summary"].as_str().is_some_and(|s| !s.is_empty()), "{reply}");
        let reached = seen.lock().expect("record lock").clone();
        assert!(
            reached.iter().any(|r| r.path == "/api/chat"
                && r.model.as_deref() == Some("nanna-test-chat:1b")),
            "{reached:?}"
        );
    }

    /// A memory saved through the `remember` TOOL used to carry no `fact_type`
    /// at all, so the drift pin — which can only protect what it can identify —
    /// could never protect the memories a user most explicitly asked to keep.
    #[test]
    fn a_stated_declaration_pins_the_memory() {
        let tags = tags_with_provenance(
            HashMap::new(),
            &serde_json::json!({"provenance": "stated"}),
        );
        assert_eq!(tags.get("fact_type").map(String::as_str), Some("stated"));
        assert!(nanna_memory::is_verbatim_pinned(&tags));
    }

    /// The conservative default, and the whole reason this is classified rather
    /// than copied: absence of a declaration is not evidence the user said
    /// something. Only an explicit, case-insensitive "stated" pins.
    #[test]
    fn anything_that_is_not_stated_is_observed() {
        for params in [
            serde_json::json!({}),
            serde_json::json!({"provenance": ""}),
            serde_json::json!({"provenance": "observed"}),
            serde_json::json!({"provenance": "statedly"}),
            serde_json::json!({"provenance": "user-said"}),
        ] {
            let tags = tags_with_provenance(HashMap::new(), &params);
            assert_eq!(
                tags.get("fact_type").map(String::as_str),
                Some("observed"),
                "{params}"
            );
            assert!(!nanna_memory::is_verbatim_pinned(&tags));
        }
    }

    /// A caller that already stamps `fact_type` in its tags keeps working — but
    /// the value is re-classified, never trusted verbatim, so a near-miss
    /// spelling cannot smuggle a pin past the rule.
    #[test]
    fn a_fact_type_tag_is_reclassified_not_trusted() {
        let mut tags = HashMap::new();
        tags.insert("fact_type".to_string(), "  StAtEd ".to_string());
        let pinned = tags_with_provenance(tags, &serde_json::json!({}));
        assert_eq!(pinned.get("fact_type").map(String::as_str), Some("stated"));

        let mut tags = HashMap::new();
        tags.insert("fact_type".to_string(), "STATED-ish".to_string());
        let not_pinned = tags_with_provenance(tags, &serde_json::json!({}));
        assert_eq!(not_pinned.get("fact_type").map(String::as_str), Some("observed"));
    }

    /// An explicit `provenance` field wins over a `fact_type` tag: the field is
    /// the declaration this call is making, the tag may be inherited metadata.
    #[test]
    fn the_explicit_field_wins_over_an_inherited_tag() {
        let mut tags = HashMap::new();
        tags.insert("fact_type".to_string(), "stated".to_string());
        tags.insert("topic".to_string(), "deploys".to_string());
        let tags = tags_with_provenance(tags, &serde_json::json!({"provenance": "observed"}));
        assert_eq!(tags.get("fact_type").map(String::as_str), Some("observed"));
        assert_eq!(
            tags.get("topic").map(String::as_str),
            Some("deploys"),
            "unrelated tags are untouched"
        );
    }

    /// Seed one tool result's chunk rows: `stored_count` of a promised
    /// `promised_count`, all sharing a `source_id`.
    async fn seeded_chunk_store(
        stored_count: usize,
        promised_count: usize,
    ) -> Arc<nanna_memory::MemoryService> {
        let service = Arc::new(nanna_memory::MemoryService::new(
            nanna_memory::MemoryServiceConfig {
                dimension: 4,
                ..Default::default()
            },
        ));
        for idx in 1..=stored_count {
            let mut metadata = std::collections::HashMap::new();
            metadata.insert("source_id".to_string(), "abc123".to_string());
            metadata.insert("chunk".to_string(), format!("{idx}/{promised_count}"));
            service
                .add_entry(nanna_memory::MemoryEntry {
                    id: format!("chunk-{idx}"),
                    content: format!("part {idx}"),
                    // A distinct unit vector per chunk: the store asserts a
                    // non-empty embedding on add, and reassembly is keyed on
                    // metadata, so the direction is irrelevant here.
                    embedding: vec![1.0, 0.0, 0.0, 0.0],
                    embedding_model: None,
                    embeddings: std::collections::HashMap::new(),
                    metadata,
                    timestamp: 0,
                    fsrs: nanna_memory::FsrsState::new(),
                    workspace_id: None,
                })
                .await
                .expect("seed");
        }
        service
    }

    async fn first_entry(
        service: &Arc<nanna_memory::MemoryService>,
    ) -> nanna_memory::MemoryListEntry {
        service.list_all().await.into_iter().next().expect("seeded")
    }

    #[test]
    fn a_scheduled_result_is_posted_only_when_it_says_something() {
        assert_eq!(
            scheduled_output_message("inbox", "  3 new messages\n", false).as_deref(),
            Some("**Scheduled: inbox**\n\n3 new messages")
        );
        assert_eq!(
            scheduled_output_message("heartbeat", "HEARTBEAT_OK", true),
            None
        );
        assert_eq!(scheduled_output_message("inbox", "   ", false), None);
    }

    #[test]
    fn only_the_first_page_of_a_recall_counts_as_a_use() {
        let served = vec!["chunk-1".to_string(), "chunk-2".to_string()];
        let signals = recall_feedback(0, &served);
        assert_eq!(signals.len(), 2);
        assert!(
            signals
                .iter()
                .all(|(_, f)| *f == nanna_memory::MemoryFeedback::UsedSuccessfully)
        );
        assert!(
            recall_feedback(4000, &served).is_empty(),
            "a later page is the same use"
        );
    }

    /// End to end through the real service map: resolving a stub's handle
    /// credits every chunk row that served it, once.
    #[tokio::test]
    async fn resolving_a_stored_result_credits_the_rows_that_served_it() {
        let store = seeded_chunk_store(3, 3).await;
        let dreaming = Arc::new(nanna_memory::DreamingService::with_shared_memory(
            nanna_memory::DreamingConfig::default(),
            Arc::clone(&store),
        ));
        let slot: DreamingSlot = Arc::new(std::sync::OnceLock::new());
        assert!(slot.set(Arc::clone(&dreaming)).is_ok());
        let services = build_script_services(ScriptServiceDeps {
            memory: Some(Arc::clone(&store)),
            feedback: Some(slot),
            ..ScriptServiceDeps::default()
        });

        let first = services["memory.get"](serde_json::json!({ "id": "abc123", "limit": 7 }))
            .await
            .expect("resolved");
        assert_eq!(first["content"], "part 1\n");
        let next = services["memory.get"](serde_json::json!({ "id": "abc123", "offset": 7 }))
            .await
            .expect("next page");
        assert_eq!(next["offset"], 7);

        for chunk in ["chunk-1", "chunk-2", "chunk-3"] {
            assert_eq!(
                dreaming.pending_feedback_boost(chunk).await,
                Some(0.5),
                "{chunk}: one UsedSuccessfully, not one per page"
            );
        }
        assert!(
            services["memory.get"](serde_json::json!({ "id": "no-such-handle" }))
                .await
                .is_err()
        );
    }

    /// The whole result is present: the reassembly is exactly the chunks, in
    /// order, and says nothing extra. An announcement on a complete read would
    /// be noise, and worse, would teach the model to ignore the real one.
    #[tokio::test]
    async fn a_complete_reassembly_announces_nothing() {
        let service = seeded_chunk_store(3, 3).await;
        let entry = first_entry(&service).await;

        let assembled = assemble_handle_content(&service.list_all().await, &entry);

        assert_eq!(assembled, "part 1\npart 2\npart 3");
        assert!(!assembled.contains("[SYSTEM:"));
    }

    /// Dreaming REPLACES clusters, so a result whose chunks were partly
    /// consolidated away reassembles short. Returning that silently is the
    /// failure this function exists to end: the stub promised the result "was
    /// stored whole in memory as N chunk(s)", and a model reading a fraction
    /// while being told nothing is missing reports on what it saw.
    #[tokio::test]
    async fn a_short_reassembly_says_how_much_is_missing() {
        let service = seeded_chunk_store(2, 17).await;
        let entry = first_entry(&service).await;

        let assembled = assemble_handle_content(&service.list_all().await, &entry);

        assert!(assembled.starts_with("part 1\npart 2"), "content still comes first");
        assert!(assembled.contains("2 of 17 stored chunks"), "{assembled}");
        assert!(assembled.contains("15 are no longer in the store"), "{assembled}");
        assert!(
            assembled.contains("read it back off disk"),
            "the announcement must point at the thing that IS intact: {assembled}"
        );
    }

    /// Negative space: rows with no `i/N` mark carry no promise about a total,
    /// so there is nothing to be short of. Absence of evidence must not become
    /// an announcement of loss.
    #[tokio::test]
    async fn unmarked_chunks_never_claim_a_shortfall() {
        let service = seeded_chunk_store(2, 2).await;
        let mut entry = first_entry(&service).await;
        entry.metadata.remove("chunk");
        for stored in service.list_all().await {
            assert!(stored.metadata.contains_key("source_id"));
        }

        let assembled = assemble_handle_content(&service.list_all().await, &entry);
        assert!(!assembled.contains("[SYSTEM:"), "{assembled}");
    }

    /// The entry that motivated this: provider `openrouter`, model
    /// `nvidia/nemotron-3-embed-1b:free`. Splitting on the LAST slash would
    /// address model `free` on provider `openrouter/nvidia/nemotron-3-embed-1b`
    /// — a silent mis-address, not an error.
    #[test]
    fn a_spec_splits_on_the_first_slash_only() {
        assert_eq!(
            split_embedding_spec("openrouter/nvidia/nemotron-3-embed-1b:free"),
            Some(("openrouter".into(), "nvidia/nemotron-3-embed-1b:free".into()))
        );
        assert_eq!(
            split_embedding_spec("ollama/nomic-embed-text:latest"),
            Some(("ollama".into(), "nomic-embed-text:latest".into()))
        );
        assert_eq!(
            split_embedding_spec("openai/text-embedding-3-small"),
            Some(("openai".into(), "text-embedding-3-small".into()))
        );
    }

    /// A bare name resolves to the local provider, matching how a bare chat
    /// model name resolves — and NOT to a cloud provider, which would send the
    /// user's memories somewhere they never named.
    #[test]
    fn a_bare_model_name_stays_local() {
        assert_eq!(
            split_embedding_spec("nomic-embed-text:latest"),
            Some(("ollama".into(), "nomic-embed-text:latest".into()))
        );
    }

    #[test]
    fn a_provider_is_matched_case_insensitively_and_trimmed() {
        assert_eq!(
            split_embedding_spec("  OpenRouter/some-model  "),
            Some(("openrouter".into(), "some-model".into()))
        );
    }

    /// A trailing slash names no model. Treating it as a bare name would embed
    /// against provider `ollama` model `openrouter/`, which does not exist.
    #[test]
    fn an_empty_or_modelless_spec_is_rejected() {
        assert_eq!(split_embedding_spec(""), None);
        assert_eq!(split_embedding_spec("   "), None);
        assert_eq!(split_embedding_spec("openrouter/"), None);
    }

    /// `#[serde(default)]` sits on the `MemoryConfig` CONTAINER, so every
    /// config.toml that never wrote an `embedding_priority` key is handed this
    /// default — and the daemon treats a non-empty list as authoritative over
    /// `embedding_provider`. A default entry here therefore overrides the
    /// provider the user actually selected, silently.
    ///
    /// It was `["openai/text-embedding-3-small"]`. With no `OpenAI` key that
    /// resolved zero providers and switched the whole memory subsystem off, and
    /// the Settings dropdown writes provider/model without touching this list,
    /// so the state was one click away for anyone who chose Ollama.
    #[test]
    fn the_default_priority_must_stay_empty_or_it_overrides_the_chosen_provider() {
        let defaults = nanna_config::MemoryConfig::default();
        assert!(
            defaults.embedding_priority.is_empty(),
            "a non-empty default silently overrides embedding_provider for every config that \
             never wrote the key; got {:?}",
            defaults.embedding_priority
        );
    }

    /// REGRESSION: a scheduled run must see its own session — every one of the
    /// 35 logged "session scope requires `session_id`" `todo` failures came from
    /// a `scheduled-heartbeat-*` run that had a session id all along, it just
    /// never reached the registry `Nanna.sessionId()` reads.
    ///
    /// And it must see ONLY its own. The scheduler's idle gate
    /// (`agent.any_run_active()`) stops a scheduled run from STARTING during a
    /// live run; it does not stop a chat turn from starting during a scheduled
    /// run, because a chat claims per-session and rebinds the registry at once.
    /// This is the shape that arrangement produces: chat-a is bound, the
    /// heartbeat starts, chat-b starts mid-heartbeat. Nothing here may leave
    /// the heartbeat reading chat-b's session, chat-b reading the heartbeat's,
    /// or — the failure mode of a save/restore binding — chat-b's binding
    /// replaced by the dead chat-a when the heartbeat ends.
    #[tokio::test]
    async fn a_scheduled_run_and_a_concurrent_chat_keep_their_own_sessions() {
        let tools = Arc::new(ToolRegistry::new());
        tools.set_session_id(Some("chat-a".to_string())).await;

        // Barriers pin the interleaving so this fails for the right reason
        // rather than by timing luck: `chat_started` puts chat-b's rebind
        // strictly inside the heartbeat, `both_read` holds the heartbeat open
        // until chat-b has read.
        let chat_started = Arc::new(tokio::sync::Barrier::new(2));
        let both_read = Arc::new(tokio::sync::Barrier::new(2));

        let heartbeat = {
            let (tools, chat_started, both_read) =
                (tools.clone(), chat_started.clone(), both_read.clone());
            async move {
                chat_started.wait().await;
                let seen = tools.session_id().await;
                both_read.wait().await;
                seen
            }
        };

        let chat = {
            let (tools, chat_started, both_read) =
                (tools.clone(), chat_started.clone(), both_read.clone());
            async move {
                tools.set_session_id(Some("chat-b".to_string())).await;
                chat_started.wait().await;
                let seen = tools.session_id().await;
                both_read.wait().await;
                seen
            }
        };

        let (heartbeat_seen, chat_seen) = tokio::join!(
            ToolRegistry::with_run_session("scheduled-heartbeat".to_string(), heartbeat),
            chat,
        );

        assert_eq!(
            heartbeat_seen.as_deref(),
            Some("scheduled-heartbeat"),
            "the scheduled run's tools must see the run's own session"
        );
        assert_eq!(
            chat_seen.as_deref(),
            Some("chat-b"),
            "a chat starting mid-run must not be attributed to the scheduled run"
        );
        assert_eq!(
            tools.session_id().await.as_deref(),
            Some("chat-b"),
            "the ended run must not hand the live chat's binding back to a dead session"
        );
    }

    #[test]
    fn channel_webhook_secrets_flow_into_webhook_config() {
        use nanna_config::{ChannelsConfig, DiscordConfig, SlackConfig, WhatsAppConfig};

        let channels = ChannelsConfig {
            discord: Some(DiscordConfig {
                bot_token: "bot".into(),
                application_id: "app".into(),
                public_key: "pubkey-hex".into(),
            }),
            slack: Some(SlackConfig {
                bot_token: "bot".into(),
                app_token: None,
                signing_secret: "slack-signing".into(),
            }),
            whatsapp: Some(WhatsAppConfig {
                connection_method: "cloud-api".into(),
                phone_number_id: None,
                access_token: None,
                verify_token: Some("wa-verify".into()),
                app_secret: Some("wa-app-secret".into()),
                session_name: None,
                allowed_contacts: None,
            }),
            telegram: None,
            signal: None,
        };

        let mut webhook = WebhookConfig::default();
        apply_channel_webhook_secrets(&mut webhook, &channels);

        assert_eq!(webhook.discord_public_key.as_deref(), Some("pubkey-hex"));
        assert_eq!(
            webhook.slack_signing_secret.as_deref(),
            Some("slack-signing")
        );
        assert_eq!(webhook.whatsapp_verify_token.as_deref(), Some("wa-verify"));
        assert_eq!(
            webhook.whatsapp_app_secret.as_deref(),
            Some("wa-app-secret")
        );
    }

    #[test]
    fn absent_channels_leave_webhook_secrets_unset() {
        let channels = nanna_config::ChannelsConfig::default();
        let mut webhook = WebhookConfig::default();
        apply_channel_webhook_secrets(&mut webhook, &channels);
        assert!(webhook.discord_public_key.is_none());
        assert!(webhook.slack_signing_secret.is_none());
        assert!(webhook.whatsapp_app_secret.is_none());
    }

    /// The crash: a stored `edit_file` error whose em dash straddled the byte
    /// at the default 4 000-byte cap. An em dash is three bytes, `&s[..4_000]`
    /// landed in the middle of it, and the panic took the whole daemon down.
    #[test]
    fn a_page_never_splits_a_multi_byte_char_at_the_default_limit() {
        let content = format!("{}—tail", "a".repeat(3_999));
        assert!(!content.is_char_boundary(4_000), "the test string must straddle the cap");

        let (s, e) = handle_page_range(&content, 0, 4_000);
        let page = &content[s..e];

        assert_eq!(s, 0);
        assert_eq!(e, 4_002, "the em dash is carried whole rather than cut at 4 000");
        assert!(page.ends_with('—'));
    }

    /// The range must be usable on the very string it was measured from, at
    /// any caller-supplied offset and limit — those come straight off the wire
    /// via `opt_count`. Slicing is the operation that panicked, so slice.
    #[test]
    fn a_page_range_indexes_the_string_it_was_measured_from() {
        let content = "—".repeat(5);
        for offset in 0..content.len() + 4 {
            for limit in 0..content.len() + 4 {
                let (s, e) = handle_page_range(&content, offset, limit);
                assert!(s <= e && e <= content.len(), "offset {offset} limit {limit}");
                let _ = &content[s..e];
            }
        }
    }

    /// Paging must not drop a char between reads: the second page starts where
    /// the first ended, so walking both ends forward has to keep them joinable.
    #[test]
    fn consecutive_pages_reassemble_the_whole_content() {
        let content = format!("{}—{}", "a".repeat(10), "b".repeat(10));
        let (s1, e1) = handle_page_range(&content, 0, 11);
        let (s2, e2) = handle_page_range(&content, e1, 100);

        assert_eq!(s1, 0);
        assert_eq!(s2, e1, "the next page resumes at the byte this one ended on");
        assert_eq!(format!("{}{}", &content[s1..e1], &content[s2..e2]), content);
    }

    #[test]
    fn tool_policy_wildcard_enabled_is_unrestricted() {
        // The default `[tools] enabled = ["*"]` must not gate anything.
        let p = build_tool_policy(Some(&["*".to_string()]), &[]);
        assert!(p.is_unrestricted());
    }

    #[test]
    fn tool_policy_none_enabled_is_unrestricted() {
        assert!(build_tool_policy(None, &[]).is_unrestricted());
    }

    #[test]
    fn tool_policy_disabled_denies_even_with_wildcard() {
        // Regression: `disabled` was parsed but never enforced. With the wildcard
        // allowlist a disabled tool must still be refused.
        let p = build_tool_policy(Some(&["*".to_string()]), &["exec".to_string()]);
        assert!(!p.is_unrestricted());
        assert!(!p.permits("exec"));
        assert!(p.permits("read_file"));
    }

    #[test]
    fn tool_policy_explicit_allowlist_restricts() {
        let enabled = vec!["read_file".to_string(), "recall".to_string()];
        let p = build_tool_policy(Some(&enabled), &[]);
        assert!(p.permits("read_file"));
        assert!(!p.permits("exec"));
    }

    #[test]
    fn tool_policy_deny_wins_over_allow() {
        let enabled = vec!["exec".to_string(), "read_file".to_string()];
        let p = build_tool_policy(Some(&enabled), &["exec".to_string()]);
        assert!(!p.permits("exec"));
        assert!(p.permits("read_file"));
    }

    #[test]
    fn tool_policy_empty_allowlist_fails_open_on_allow_side() {
        // A forgotten/empty allowlist must not silently mute every tool; only the
        // denylist should bite.
        let p = build_tool_policy(Some(&[]), &["exec".to_string()]);
        assert!(p.permits("read_file"));
        assert!(!p.permits("exec"));
    }

    #[test]
    fn daemon_heartbeat_prompt_does_not_command_file_read() {
        // The daemon overrides the scheduler default with its own prompt; guard
        // it too so neither site reintroduces the erroring `Read HEARTBEAT.md`.
        let p = DAEMON_HEARTBEAT_PROMPT.to_lowercase();
        assert!(
            !p.contains("read heartbeat"),
            "must not command a file read: {p}"
        );
        assert!(
            !p.contains(".md"),
            "must not reference a bespoke .md file: {p}"
        );
        assert!(
            p.contains("heartbeat_ok"),
            "must keep the HEARTBEAT_OK sentinel: {p}"
        );
    }

    /// Serve exactly one HTTP request with a canned body, then exit.
    ///
    /// Bound on port 0 so the OS assigns a free port — the test never races a
    /// real Ollama or another test for a fixed port. Returns the base URL.
    async fn spawn_one_shot_embedding_server(body: &'static str) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind an ephemeral port");
        let addr = listener.local_addr().expect("read back the bound addr");
        tokio::spawn(async move {
            if let Ok((mut stream, _)) = listener.accept().await {
                use tokio::io::{AsyncReadExt, AsyncWriteExt};
                // Read whatever the client sends; we only need the socket
                // drained enough to reply.
                let mut buf = [0_u8; 4096];
                let _ = stream.read(&mut buf).await;
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(response.as_bytes()).await;
                let _ = stream.flush().await;
            }
        });
        format!("http://{addr}")
    }

    fn ollama_router_at(base_url: &str) -> EmbeddingRouter {
        EmbeddingRouter::new(
            EmbeddingProviderInfo {
                name: "ollama".into(),
                model: "nomic-embed-text".into(),
            },
            Arc::new(nanna_llm::EmbeddingClient::ollama(base_url).with_model("nomic-embed-text")),
        )
    }

    /// The probe reports the dimension the provider actually returned — it does
    /// not consult any per-model dimension table.
    #[tokio::test]
    async fn probe_reports_the_dimension_the_provider_returns() {
        let base =
            spawn_one_shot_embedding_server(r#"{"embeddings":[[0.1,0.2,0.3,0.4,0.5]]}"#).await;
        let dim = DaemonServer::probe_embedding_dimension(&ollama_router_at(&base))
            .await
            .expect("probe succeeds against a responsive provider");
        assert_eq!(dim, 5, "dimension comes from the response vector's length");
    }

    /// A provider that answers with an empty vector is an error, not a
    /// zero-dimension memory store.
    #[tokio::test]
    async fn probe_rejects_an_empty_embedding_vector() {
        let base = spawn_one_shot_embedding_server(r#"{"embeddings":[[]]}"#).await;
        let err = DaemonServer::probe_embedding_dimension(&ollama_router_at(&base))
            .await
            .expect_err("an empty vector must not pass as a valid dimension");
        assert!(!err.is_empty(), "the failure carries a reason");
    }

    /// The regression this fixes: an unreachable/unkeyed provider makes the
    /// probe fail, and the caller must be able to carry on. The probe returns
    /// `Err` rather than panicking, so boot can degrade to a provisional
    /// dimension instead of aborting.
    #[tokio::test]
    async fn probe_fails_cleanly_when_no_provider_answers() {
        // Bind then immediately drop the listener, so the port is (almost
        // certainly) closed — a stand-in for "no embedding provider running".
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("addr");
        drop(listener);

        let err =
            DaemonServer::probe_embedding_dimension(&ollama_router_at(&format!("http://{addr}")))
                .await
                .expect_err("probe must fail when nothing is listening");
        assert!(
            err.contains("embedding providers failed"),
            "the router reports exhausting its providers, got: {err}"
        );
    }

    /// The seed the daemon falls back to must itself be a usable dimension —
    /// this is the value memory runs on until a provider answers.
    #[test]
    fn provisional_seed_dimension_is_positive() {
        assert!(
            nanna_memory::MemoryServiceConfig::default().dimension > 0,
            "a zero seed would make every add() fail before the probe realigns"
        );
    }

    #[test]
    fn scheduled_consolidation_config_threads_user_memory_settings() {
        // The scheduled dream cycle must use the user's compression settings,
        // not ConsolidationConfig::default(), so automatic and IPC-triggered
        // consolidation behave identically.
        let cfg = scheduled_consolidation_config(0.25, 100, 8_192);
        assert!((cfg.max_compression_ratio - 0.25).abs() < f32::EPSILON);
        assert_eq!(cfg.min_remaining_memories, 100);
        // Untouched fields keep their defaults (e.g. the member-count cap).
        let default = nanna_memory::ConsolidationConfig::default();
        assert_eq!(cfg.max_cluster_memories, default.max_cluster_memories);
        assert!((cfg.cluster_threshold - default.cluster_threshold).abs() < f32::EPSILON);
    }

    #[test]
    fn scheduled_consolidation_config_sizes_content_budget_to_the_model() {
        // A large-context summarizer gets a proportionally larger per-cluster
        // content budget than a small one (so big models consolidate more per
        // pass) — the whole point of threading the model's context window.
        let small = scheduled_consolidation_config(0.5, 20, 8_192);
        let large = scheduled_consolidation_config(0.5, 20, 200_000);
        assert!(large.max_cluster_content_bytes > small.max_cluster_content_bytes);
        assert_eq!(
            large.max_cluster_content_bytes,
            nanna_memory::cluster_content_bytes_for_context(200_000)
        );
    }

    /// The P13 unification invariant: the orchestrator the daemon builds reads
    /// the **same** clock the control plane stamps, so a chat request moves the
    /// dream gate without any second bookkeeping call.
    #[tokio::test]
    async fn dreaming_service_gates_on_the_control_plane_clock() {
        let clock = Arc::new(nanna_memory::ActivityClock::new());
        let memory = Arc::new(nanna_memory::MemoryService::new(
            nanna_memory::MemoryServiceConfig::default(),
        ));
        // A 1-hour idle threshold: the gate is shut for as long as the clock
        // says the system was recently used.
        let dreaming = nanna_memory::DreamingService::with_shared_memory(
            nanna_memory::DreamingConfig {
                idle_threshold_secs: 3_600,
                memory_pressure_count: 0,
                ..nanna_memory::DreamingConfig::default()
            },
            memory,
        )
        .with_activity_clock(Arc::clone(&clock));

        // Stamping the clock the way the control plane does on a chat request
        // must be visible to the service…
        clock.record();
        assert!(
            dreaming.idle_duration() < std::time::Duration::from_secs(1),
            "the control plane's stamp must reset the service's idle timer"
        );
        // …and hold the gate shut.
        let ran = dreaming
            .dream_if_idle(|_p| async { Ok(String::new()) })
            .await
            .expect("gate must not error");
        assert!(ran.is_none(), "an actively-used daemon must not dream");
        // Both sides genuinely hold one clock, not two equal ones.
        assert!(Arc::ptr_eq(&dreaming.activity_clock(), &clock));
    }

    #[test]
    fn daemon_config_default_mirrors_consolidation_defaults() {
        let daemon = DaemonConfig::default();
        let cons = nanna_memory::ConsolidationConfig::default();
        assert!(
            (daemon.memory_max_compression_ratio - cons.max_compression_ratio).abs() < f32::EPSILON
        );
        assert_eq!(
            daemon.memory_min_remaining_memories,
            cons.min_remaining_memories
        );
    }

    #[test]
    fn daemon_config_default_mirrors_dreaming_idle_gate_defaults() {
        // The scheduled dream cycle's idle gate must default to the same policy
        // DreamingService uses, so wiring the gate changed no thresholds.
        let daemon = DaemonConfig::default();
        let dream = nanna_memory::DreamingConfig::default();
        assert_eq!(
            daemon.dream_idle_threshold_secs, dream.idle_threshold_secs,
            "daemon idle-gate default must mirror DreamingConfig"
        );
        assert_eq!(
            daemon.dream_memory_pressure_count, dream.memory_pressure_count,
            "daemon memory-pressure default must mirror DreamingConfig"
        );
    }

    #[test]
    fn daemon_config_default_mirrors_scheduler_config_defaults() {
        // A user who never opens Settings must get exactly the schedule the
        // config file documents — the daemon's fallback cannot drift from it.
        let daemon = DaemonConfig::default();
        let scheduler = nanna_config::SchedulerConfig::default();
        assert_eq!(daemon.scheduler.enabled, scheduler.enabled);
        assert_eq!(
            daemon.scheduler.heartbeat_enabled,
            scheduler.heartbeat_enabled
        );
        assert_eq!(
            daemon.heartbeat_interval_secs,
            scheduler.heartbeat_interval_secs
        );
    }

    #[test]
    fn scheduler_settings_come_from_config_not_literals() {
        // Regression: the daemon used to hardcode `heartbeat_enabled: true` and
        // a 1800s interval, which made Settings → Scheduler dead UI and left the
        // heartbeat competing with chat for the single local-model slot.
        let mut builder = DaemonBuilder::new();
        let user = nanna_config::SchedulerConfig {
            enabled: false,
            heartbeat_enabled: false,
            heartbeat_interval_secs: 600,
        };
        builder.config.scheduler.enabled = user.enabled;
        builder.config.scheduler.heartbeat_enabled = user.heartbeat_enabled;
        builder.config.heartbeat_interval_secs = user.heartbeat_interval_secs;

        // Mirrors the construction in `init_services`.
        let core = nanna_core::SchedulerConfig {
            enabled: builder.config.scheduler.enabled,
            heartbeat_interval: std::time::Duration::from_secs(nanna_core::clamp_heartbeat_secs(
                builder.config.heartbeat_interval_secs,
            )),
            heartbeat_enabled: builder.config.scheduler.heartbeat_enabled,
            ..nanna_core::SchedulerConfig::default()
        };

        assert!(!core.enabled);
        assert!(!core.heartbeat_enabled);
        assert_eq!(core.heartbeat_interval, std::time::Duration::from_secs(600));
    }

    /// `[general] data_dir` becomes `DaemonConfig::data_dir`, the one data dir
    /// the daemon hands to everything it builds — the store, and the control
    /// plane's user tools, which `e2e_daemon`'s
    /// `a_user_tool_is_kept_in_the_daemons_own_data_dir` follows from there.
    /// The control plane used to derive its own from the platform default, so
    /// a relocated install kept its user tools apart from its store.
    #[test]
    fn a_configured_data_dir_is_the_daemons_data_dir() {
        let relocated = PathBuf::from("relocated-nanna-data");
        let mut config = nanna_config::Config::default();
        config.general.data_dir = Some(relocated.clone());

        let mut builder = DaemonBuilder::new();
        builder.apply_data_dir(&config);

        assert_eq!(builder.config.data_dir, relocated);
        assert_eq!(builder.memory_path, Some(relocated.join("memories.json")));
    }

    /// A fake Ollama that answers everything EXCEPT a generation instantly,
    /// and holds `/api/chat` / `/api/generate` open forever — the shape of a
    /// generation occupying the single local slot.
    async fn spawn_stalling_ollama() -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind an ephemeral port");
        let addr = listener.local_addr().expect("read back the bound addr");
        tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    break;
                };
                tokio::spawn(async move {
                    use tokio::io::{AsyncReadExt, AsyncWriteExt};
                    let mut buf = vec![0_u8; 16384];
                    let n = stream.read(&mut buf).await.unwrap_or(0);
                    let head = String::from_utf8_lossy(&buf[..n]);
                    if head.contains("/api/chat") || head.contains("/api/generate") {
                        // Hold the slot forever: the test's preemption must be
                        // what ends this, never a server response.
                        std::future::pending::<()>().await;
                    }
                    let body = "{}";
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len(),
                    );
                    let _ = stream.write_all(response.as_bytes()).await;
                });
            }
        });
        format!("http://{addr}")
    }

    /// The P22 Tier 4 admission gate, end to end: a scheduled run whose
    /// generation is parked inside the local provider yields within moments
    /// of a user turn claiming the registry. Without the gate this test
    /// cannot pass — the stalled generation never returns on its own, which
    /// is exactly the 157-second slot squat the forensics measured (there it
    /// eventually finished; here it provably never would).
    #[tokio::test(flavor = "multi_thread")]
    async fn a_scheduled_run_yields_the_local_provider_to_a_live_chat_turn() {
        let base = spawn_stalling_ollama().await;
        let router = Arc::new(LlmRouter::new().with_ollama(&base));
        let config = AgentServiceConfig {
            // The `ollama/` prefix is what arms preemption — from_model
            // resolves it to the local single-slot provider.
            model: "ollama/stall-model".to_string(),
            ..AgentServiceConfig::default()
        };
        let (event_tx, _keep_events_alive) =
            tokio::sync::broadcast::channel::<crate::protocol::Event>(16);
        let tools = Arc::new(ToolRegistry::new());
        let agent = Arc::new(AgentService::new(config, router, tools, None, event_tx));
        let registry = Arc::new(crate::control::chat_harness::ChatRunRegistry::new());

        let helper_agent = agent.clone();
        let helper_registry = registry.clone();
        let helper = tokio::spawn(async move {
            run_scheduled_prompt_yielding(
                &helper_agent,
                &helper_registry,
                "scheduled-yield-test",
                "heartbeat check-in",
            )
            .await
        });

        // Let the scheduled run get genuinely in flight (request parked in
        // the stalled generation), then a user turn arrives. The gate is
        // correct even if the claim lands earlier — the cancel loop waits
        // for the run's registration — so this sleep widens coverage, it
        // does not carry the test.
        tokio::time::sleep(std::time::Duration::from_millis(400)).await;
        assert!(
            registry.try_claim("user-session").await,
            "the user turn claims the run slot"
        );

        let outcome = tokio::time::timeout(std::time::Duration::from_secs(30), helper)
            .await
            .expect("the yield completes in moments, never after the stalled generation")
            .expect("the helper task must not panic");
        assert!(
            outcome.is_none(),
            "a preempted run reports a yield, not a result"
        );
        registry.release("user-session").await;
    }
    /// The `read_pdf` skill declares `requires: ["pdf.read"]`. Without that
    /// service the tool still loads and still advertises itself to the model —
    /// it just fails at call time — so a compile cannot tell you whether it is
    /// wired. This calls it end to end against a real document.
    #[tokio::test]
    async fn the_pdf_read_service_is_registered_and_reads_a_real_document() {
        let services = build_script_services(ScriptServiceDeps::default());
        let pdf_read = services
            .get("pdf.read")
            .expect("the read_pdf skill's declared service must exist");

        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("probe.pdf");
        std::fs::write(&path, minimal_two_page_pdf()).expect("write probe pdf");
        let path_str = path.to_string_lossy().to_string();

        let whole = pdf_read(serde_json::json!({ "path": path_str }))
            .await
            .expect("reading the whole document succeeds");
        assert_eq!(whole["page_count"], 2);
        assert_eq!(whole["pages_read"], 2);

        // And the `pages` argument the skill sends is honoured, not dropped.
        let page_two = pdf_read(serde_json::json!({ "path": path_str, "pages": "2" }))
            .await
            .expect("reading one page succeeds");
        assert_eq!(page_two["page_count"], 2);
        assert_eq!(page_two["pages_read"], 1);

        // A selection it cannot read is refused rather than guessed at.
        let bad = pdf_read(serde_json::json!({ "path": path_str, "pages": "5-2" })).await;
        assert!(bad.is_err(), "a backwards range must not be reinterpreted");

        // Both pages are textless, and this daemon has no OCR model, so the
        // response has to say which of those two facts it is reporting.
        assert_eq!(whole["empty_pages"], 2);
        assert_eq!(
            whole["ocr"], "unavailable",
            "a textless PDF with no OCR configured must say so, not return an \
             empty string that looks like a blank document"
        );
        assert!(
            whole["ocr_note"]
                .as_str()
                .is_some_and(|note| note.contains("ocr_model_priority")),
            "the note must name what to configure: {:?}",
            whole["ocr_note"]
        );
    }

    /// The distinction the OCR item was held back on for two runs: "no OCR
    /// pipeline" and "OCR ran and there was nothing to read" produce the same
    /// empty text, so the response must name which one happened.
    #[tokio::test]
    async fn a_textless_pdf_distinguishes_no_ocr_from_nothing_to_ocr() {
        let ocr_called = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let calls = ocr_called.clone();
        let ocr_fn: nanna_tools::PdfOcrFn = Arc::new(move |_image, _prompt, _media| {
            calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Box::pin(async { Ok("stub ocr text".to_string()) })
        });

        let services = build_script_services(ScriptServiceDeps {
            pdf_ocr: Some(ocr_fn),
            ..ScriptServiceDeps::default()
        });
        let pdf_read = services.get("pdf.read").expect("pdf.read is registered");

        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("textless.pdf");
        std::fs::write(&path, minimal_two_page_pdf()).expect("write probe pdf");

        let read = pdf_read(serde_json::json!({ "path": path.to_string_lossy() }))
            .await
            .expect("reading succeeds");

        // The fixture carries no embedded images, so OCR has nothing to send —
        // which is a different answer from "no OCR model", and from a scan it
        // could have read.
        assert_eq!(
            read["ocr"], "no_images",
            "with an OCR model configured, a textless PDF with no images must \
             report that rather than reusing the unavailable case: {read}"
        );
        assert_eq!(
            ocr_called.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "OCR was called for a document with no embedded images"
        );
    }

    /// A two-page PDF with no text content — enough to exercise page counting
    /// and selection without depending on text extraction.
    fn minimal_two_page_pdf() -> Vec<u8> {
        use lopdf::{Document, Object, dictionary};

        let mut doc = Document::with_version("1.5");
        let pages_id = doc.new_object_id();
        let kids: Vec<Object> = (0..2)
            .map(|_| {
                doc.add_object(dictionary! {
                    "Type" => "Page",
                    "Parent" => pages_id,
                })
                .into()
            })
            .collect();
        let count = i64::try_from(kids.len()).expect("two fits");
        doc.objects.insert(
            pages_id,
            Object::Dictionary(dictionary! {
                "Type" => "Pages",
                "Kids" => kids,
                "Count" => count,
                "MediaBox" => vec![0.into(), 0.into(), 595.into(), 842.into()],
            }),
        );
        let catalog_id = doc.add_object(dictionary! {
            "Type" => "Catalog",
            "Pages" => pages_id,
        });
        doc.trailer.set("Root", catalog_id);

        let mut bytes = Vec::new();
        doc.save_to(&mut bytes).expect("document saves");
        bytes
    }
}
