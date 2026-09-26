#![warn(clippy::pedantic, clippy::nursery, clippy::all)]
// An integration test is its OWN crate root, so the crate-level attribute the
// six library roots carry does not reach it. This target drives a real daemon,
// so proving its futures are `Send` walks the same
// MemoryService -> VectorStore -> CosineSimilaritySearch -> wgpu graph that is
// deeper than the default limit of 128, and nightly-2026-08-25 makes that the
// future-incompatible `recursion_depth_exceeding_limit` warning (rust#159228)
// which is scheduled to become a hard error. Solver depth only.
#![recursion_limit = "256"]

//! End-to-end daemon tests: start a real daemon, attach a real client over the
//! WebSocket IPC, and drive it the way the GUI/CLI do.
//!
//! This is the P8 "daemon/embedded/reconnect story is untested" gap. Everything that
//! follows exercises the actual `DaemonServer` and the actual `Client` — no mocks, no
//! in-process shortcuts around the protocol.
//!
//! The daemon under test is deliberately **hermetic**: built through `DaemonBuilder`
//! with explicit settings rather than `from_nanna_config`, so a run never reads the
//! developer's `config.toml`, never touches their `.db`, and never needs an API key or
//! a reachable model. `with_memory(false)` keeps embeddings out of it entirely — these
//! tests are about the IPC/session/persistence path, and an LLM would make them
//! non-hermetic and slow.
//!
//! This test lives in `nanna-client` rather than `nanna-daemon` because the client
//! already depends on the daemon (for the shared protocol); putting it here keeps the
//! dependency edge pointing one way.

use nanna_client::{Client, ClientConfig};
use nanna_daemon::server::DaemonBuilder;
use std::time::Duration;

/// A daemon running on its own port and data dir for the duration of one test.
struct TestDaemon {
    port: u16,
    data_dir: tempfile::TempDir,
    handle: tokio::task::JoinHandle<Result<(), String>>,
}

/// Absolute ceiling on the readiness wait.
///
/// This is **not** the assertion — [`wait_until_ready`] decides success and
/// failure by watching the daemon task, not the clock. This only stops a daemon
/// that has genuinely wedged (deadlocked mid-boot, never binding and never
/// returning) from hanging CI until the job is killed.
///
/// It is therefore sized to be unreachable by scheduling: boot is subsecond in
/// practice, and two minutes is ~100× that. The previous 10 s *was* the
/// assertion, and on a machine running the whole workspace's test binaries at
/// once it fired on a healthy daemon — observed 2026-08-24, all 4 tests failing
/// a full `cargo test --workspace` and passing 4/4 in 1.54 s alone.
const READY_HANG_CEILING: Duration = Duration::from_secs(120);

/// How often to re-probe the port while waiting.
const READY_POLL_INTERVAL: Duration = Duration::from_millis(50);

/// The daemon's published IPC bind address (`None` until it binds).
type BoundAddr = tokio::sync::watch::Receiver<Option<std::net::SocketAddr>>;

/// Wait until the daemon's IPC listener has bound, and return the port it got.
///
/// Daemons start on port 0 and report the port the OS assigned. Picking a free
/// port up front and releasing it for the daemon to re-bind raced with every
/// parallel test doing the same: two daemons chose one port, the loser exited
/// ("returned cleanly without ever listening"), or — worse — the readiness probe
/// connected to the *other* test's daemon and the test carried on against it until
/// that daemon was stopped under it (observed 2026-09-17 under a full
/// `cargo test --workspace`).
///
/// **Watches the daemon, not the clock.** The task is the ground truth: if it
/// finishes before the listener binds, the daemon failed or panicked, and that is
/// knowable *immediately* and reported with the actual cause — no waiting out a
/// deadline to conclude something we already knew, and no "within Ns" message
/// that blames time for a crash.
///
/// Distinguishing the two mattered enough to restructure for: a daemon that died
/// and a daemon that is merely descheduled look identical to a bare timeout, and
/// collapsing them is what made this suite fail on healthy code under load while
/// giving a useless message when the code was genuinely broken.
async fn wait_until_ready(
    started: std::time::Instant,
    bound: &mut BoundAddr,
    handle: &mut tokio::task::JoinHandle<Result<(), String>>,
) -> u16 {
    loop {
        let current = *bound.borrow_and_update();
        if let Some(addr) = current {
            return addr.port();
        }

        // Ground truth: the task ended without ever binding the listener.
        if handle.is_finished() {
            match handle.await {
                Ok(Ok(())) => panic!("daemon returned cleanly without ever listening"),
                Ok(Err(e)) => panic!("daemon failed to run: {e}"),
                Err(join) if join.is_panic() => {
                    // Resume the panic so the original message and location
                    // survive, instead of being flattened into "task panicked".
                    std::panic::resume_unwind(join.into_panic());
                }
                Err(join) => panic!("daemon task ended: {join}"),
            }
        }

        assert!(
            started.elapsed() < READY_HANG_CEILING,
            "daemon is still running but never bound after {READY_HANG_CEILING:?} — \
             this is the hang ceiling, not a latency assertion, so treat it as a \
             wedged daemon rather than a slow one"
        );

        // Wake on the bind, or come back to re-check the task. A closed channel
        // means the server is gone; the task check above reports why.
        if let Ok(Err(_)) = tokio::time::timeout(READY_POLL_INTERVAL, bound.changed()).await {
            tokio::time::sleep(READY_POLL_INTERVAL).await;
        }
    }
}

impl TestDaemon {
    /// Boot a daemon and wait until its IPC port actually accepts connections.
    ///
    /// Reusing a `data_dir` across calls is how the restart test proves persistence
    /// survives a full process lifecycle.
    async fn start(data_dir: tempfile::TempDir) -> Self {
        Self::start_with(data_dir, |builder| builder).await
    }

    /// [`Self::start`], with `configure` applied to the builder after the
    /// hermetic defaults — how a test adds a model without giving up any of
    /// the isolation below.
    async fn start_with(
        data_dir: tempfile::TempDir,
        configure: impl FnOnce(DaemonBuilder) -> DaemonBuilder + Send + 'static,
    ) -> Self {
        let dir_path = data_dir.path().to_path_buf();
        let (bound_tx, bound_rx) = tokio::sync::oneshot::channel::<BoundAddr>();

        let mut handle = tokio::spawn(async move {
            let builder = DaemonBuilder::new()
                .with_host("127.0.0.1")
                // Port 0: the OS picks a free port at bind time and the daemon
                // reports it, so parallel tests cannot collide.
                .with_port(0)
                .with_data_dir(dir_path)
                // Keep the test hermetic and fast: no embeddings, no health/webhook
                // ports to collide on, no PID file to fight a real local daemon over.
                .with_memory(false)
                .with_health_server(false)
                .with_webhook_server(false)
                .with_pid_file(false)
                .with_log_level("warn");
            let mut server = configure(builder).build();
            // The receiver outlives this send; if it were gone the test has
            // already failed, so there is nothing to report here.
            let _ = bound_tx.send(server.ipc_bound_addr());
            // Returning the error rather than discarding it is what lets
            // `wait_until_ready` say *why* a daemon never came up.
            server.run().await.map_err(|e| e.to_string())
        });

        // One ceiling over the whole boot, `build()` included: a build that
        // wedges never sends its receiver, and must fail the test rather than
        // hang it.
        let started = std::time::Instant::now();
        let mut bound = match tokio::time::timeout(READY_HANG_CEILING, bound_rx).await {
            Ok(Ok(bound)) => bound,
            // The task ended before the server was even built: surface its cause.
            Ok(Err(_)) => match handle.await {
                Err(join) if join.is_panic() => std::panic::resume_unwind(join.into_panic()),
                other => panic!("daemon task ended before building the server: {other:?}"),
            },
            Err(elapsed) => panic!(
                "daemon is still building after {READY_HANG_CEILING:?} ({elapsed}) — this is the hang \
                 ceiling, not a latency assertion, so treat it as a wedged boot"
            ),
        };
        let port = wait_until_ready(started, &mut bound, &mut handle).await;
        Self { port, data_dir, handle }
    }

    fn url(&self) -> String {
        format!("ws://127.0.0.1:{}", self.port)
    }

    /// Connect a client with reconnect disabled.
    ///
    /// The reconnect test drives reconnection explicitly; leaving the automatic
    /// machinery on would make it unclear which path a passing assertion exercised.
    /// These two timeouts are scaffolding for the same reason as
    /// [`READY_HANG_CEILING`]: nothing here asserts how *fast* the daemon
    /// answers, only that it answers correctly. They exist so a wedged daemon
    /// fails the test instead of hanging it, and they are sized to be
    /// unreachable by scheduling delay on a saturated machine — a local
    /// round-trip over loopback to an in-process daemon is sub-millisecond.
    async fn connect_client(&self) -> Client {
        Client::connect(ClientConfig {
            url: self.url(),
            connect_timeout: READY_HANG_CEILING,
            request_timeout: READY_HANG_CEILING,
            ..Default::default()
        })
        .await
        .expect("client connects to the running daemon")
    }

    /// Stop the daemon, releasing the port.
    fn stop(self) -> tempfile::TempDir {
        self.handle.abort();
        self.data_dir
    }
}

/// Pull the session id out of a `sessions.create` response without assuming which
/// envelope shape it arrived in.
fn session_id_of(value: &serde_json::Value) -> String {
    value
        .get("session_id")
        .or_else(|| value.get("id"))
        .or_else(|| value.get("session").and_then(|s| s.get("id")))
        .and_then(|v| v.as_str())
        .unwrap_or_else(|| panic!("no session id in create response: {value}"))
        .to_string()
}

fn response_mentions(value: &serde_json::Value, needle: &str) -> bool {
    value.to_string().contains(needle)
}

/// The baseline the whole control plane rests on: a real daemon accepts a real client
/// and answers a request over the protocol.
#[tokio::test]
async fn daemon_starts_and_client_connects() {
    let daemon = TestDaemon::start(tempfile::tempdir().expect("temp dir")).await;
    let client = daemon.connect_client().await;

    assert!(client.is_connected().await, "client reports a live connection");

    let sessions = client.sessions().list().await.expect("sessions.list answers");
    assert!(sessions.is_object() || sessions.is_array(), "got a structured response: {sessions}");

    client.disconnect().await;
    daemon.stop();
}

/// A session created through the client is visible to a subsequent request — the
/// daemon is the one that owns session state, per "channels as control-plane clients".
#[tokio::test]
async fn created_session_is_visible_to_the_client() {
    let daemon = TestDaemon::start(tempfile::tempdir().expect("temp dir")).await;
    let client = daemon.connect_client().await;

    let created = client
        .sessions()
        .create(Some("e2e-session".to_string()))
        .await
        .expect("sessions.create succeeds");
    let session_id = session_id_of(&created);
    assert!(!session_id.is_empty(), "a created session has an id");

    let listed = client.sessions().list().await.expect("sessions.list succeeds");
    assert!(
        response_mentions(&listed, &session_id),
        "the created session appears in the list: {listed}"
    );

    client.disconnect().await;
    daemon.stop();
}

/// A reply larger than tungstenite's default 16 MiB frame must reach the
/// client. The daemon raised its OWN read limit to 128 MB because long
/// sessions exceeded the default and dropped the connection — but a read
/// limit protects only the side that applies it, and every large reply the
/// daemon SENDS (a whole-session `history`, a full run state, an export) is
/// read by this client. A session whose name is 20 MiB makes the `create`
/// reply that large, deterministically, with no model involved.
#[tokio::test]
async fn a_reply_over_the_default_frame_limit_reaches_the_client() {
    const REPLY_BYTES: usize = 20 * 1024 * 1024;
    let daemon = TestDaemon::start(tempfile::tempdir().expect("temp dir")).await;
    let client = daemon.connect_client().await;

    let huge_name = "n".repeat(REPLY_BYTES);
    let created = client
        .sessions()
        .create(Some(huge_name.clone()))
        .await
        .expect("a 20 MiB reply must be readable, not a dropped connection");
    assert_eq!(
        created["session"]["name"].as_str().map(str::len),
        Some(REPLY_BYTES),
        "the whole reply arrived"
    );
    assert!(
        client.is_connected().await,
        "and the connection survived it"
    );

    client.disconnect().await;
    daemon.stop();
}

/// Export end to end: the daemon — the session store's owner — renders the
/// document and hands it over the real protocol, in both formats, and an
/// unknown id is refused rather than exported as an empty document.
#[tokio::test]
async fn a_session_exports_over_the_protocol_in_both_formats() {
    let daemon = TestDaemon::start(tempfile::tempdir().expect("temp dir")).await;
    let client = daemon.connect_client().await;

    let created = client
        .sessions()
        .create(Some("Export Me".to_string()))
        .await
        .expect("sessions.create succeeds");
    let session_id = session_id_of(&created);

    let markdown = client
        .sessions()
        .export(&session_id, nanna_client::ExportFormat::Markdown)
        .await
        .expect("sessions.export answers");
    assert_eq!(markdown["filename"], "export-me.md", "{markdown}");
    let content = markdown["content"].as_str().expect("a markdown document");
    assert!(content.starts_with("# Export Me\n"), "{content}");
    assert!(
        content.contains(&session_id),
        "the document names its session: {content}"
    );

    let json = client
        .sessions()
        .export(&session_id, nanna_client::ExportFormat::Json)
        .await
        .expect("sessions.export answers");
    let document: serde_json::Value =
        serde_json::from_str(json["content"].as_str().expect("a json document"))
            .expect("valid JSON");
    assert_eq!(
        document["session"]["id"],
        serde_json::json!(session_id),
        "{document}"
    );

    let missing = client
        .sessions()
        .export("no-such-session", nanna_client::ExportFormat::Markdown)
        .await
        .expect("sessions.export answers");
    assert_eq!(missing["error"], "not_found", "{missing}");

    client.disconnect().await;
    daemon.stop();
}

/// Memory export routes through the protocol. A daemon running without memory
/// — as this hermetic one does — refuses with its reason instead of exporting
/// an empty store as though it were the user's.
#[tokio::test]
async fn memory_export_on_a_daemon_without_memory_says_why() {
    let daemon = TestDaemon::start(tempfile::tempdir().expect("temp dir")).await;
    let client = daemon.connect_client().await;

    let reply = client
        .memory()
        .export(None, nanna_client::ExportFormat::Json)
        .await
        .expect("memory.export answers");
    assert_eq!(reply["error"], "memory_unavailable", "{reply}");

    client.disconnect().await;
    daemon.stop();
}

/// The reconnection half of the P8 gap: a client that drops and attaches again must
/// find the daemon's state intact, because the daemon — not the client — owns it.
#[tokio::test]
async fn state_survives_a_client_reconnect() {
    let daemon = TestDaemon::start(tempfile::tempdir().expect("temp dir")).await;

    let first = daemon.connect_client().await;
    let created = first
        .sessions()
        .create(Some("survives-reconnect".to_string()))
        .await
        .expect("sessions.create succeeds");
    let session_id = session_id_of(&created);
    first.disconnect().await;
    assert!(!first.is_connected().await, "the first client is really gone");

    // A fresh client, not a resumed one — this is the GUI's reattach path.
    let second = daemon.connect_client().await;
    let listed = second.sessions().list().await.expect("the daemon still answers");
    assert!(
        response_mentions(&listed, &session_id),
        "the session created before the disconnect is still there: {listed}"
    );

    second.disconnect().await;
    daemon.stop();
}

/// A user tool outlives the daemon that created it. `ToolAction::Create`
/// writes `{data_dir}/user_tools/{name}.json` and registers the tool live, but
/// nothing read that directory back at boot, so after a restart the tool was
/// missing from `ListUser`, from `List` and from the registry the model calls
/// through, with its file still on disk. A disabled tool must come back too,
/// listed and re-enableable, but not callable.
#[tokio::test]
async fn a_user_tool_survives_a_daemon_restart() {
    const KEPT: &str = "e2e_restart_kept";
    const OFF: &str = "e2e_restart_off";
    let tool = |action| nanna_client::Action::Tool(action);
    let daemon = TestDaemon::start(tempfile::tempdir().expect("temp dir")).await;
    let client = daemon.connect_client().await;

    for name in [KEPT, OFF] {
        let created = client
            .request(tool(nanna_client::ToolAction::Create {
                name: name.to_string(),
                description: "restart probe".to_string(),
                code: format!(
                    "export default {{ name: \"{name}\", description: \"restart probe\", \
                     execute() {{ return \"{name} ran\"; }} }}"
                ),
                needs_shell: None,
            }))
            .await
            .expect("tool.create answers");
        assert_eq!(created["status"], "created", "{created}");
    }
    let disabled = client
        .request(tool(nanna_client::ToolAction::Disable {
            name: OFF.to_string(),
        }))
        .await
        .expect("tool.disable answers");
    assert_eq!(disabled["status"], "disabled", "{disabled}");

    client.disconnect().await;
    // Keep the data dir alive across the restart; dropping it would delete the store.
    let data_dir = daemon.stop();

    let restarted = TestDaemon::start(data_dir).await;
    let client = restarted.connect_client().await;

    let listed = client
        .request(tool(nanna_client::ToolAction::ListUser))
        .await
        .expect("tool.list_user answers");
    let enabled_of = |name: &str| {
        listed["tools"]
            .as_array()
            .and_then(|tools| tools.iter().find(|t| t["name"] == name))
            .map(|t| t["enabled"].clone())
    };
    assert_eq!(
        enabled_of(KEPT),
        Some(serde_json::json!(true)),
        "listed after the restart: {listed}"
    );
    assert_eq!(
        enabled_of(OFF),
        Some(serde_json::json!(false)),
        "listed, still disabled: {listed}"
    );

    // Registered: the direct-execute path resolves through the same registry
    // the model's calls do.
    let ran = client
        .request(tool(nanna_client::ToolAction::Execute {
            name: KEPT.to_string(),
            input: serde_json::json!({}),
            session_id: None,
        }))
        .await
        .expect("tool.execute answers");
    assert_eq!(ran["success"], true, "the restored tool runs: {ran}");
    assert_eq!(ran["output"], format!("{KEPT} ran"), "{ran}");

    let off = client
        .request(tool(nanna_client::ToolAction::Get {
            name: OFF.to_string(),
        }))
        .await
        .expect("tool.get answers");
    assert_eq!(
        off["error"], "not_found",
        "a restart must not re-register a disabled tool: {off}"
    );

    client.disconnect().await;
    restarted.stop();
}

/// The persistence half: state must outlive the daemon *process*, not just a client.
/// Restarting on the same data dir has to bring the session back — this is what makes
/// the daemon a durable control plane rather than a cache.
#[tokio::test]
async fn sessions_persist_across_a_daemon_restart() {
    let daemon = TestDaemon::start(tempfile::tempdir().expect("temp dir")).await;
    let client = daemon.connect_client().await;

    let created = client
        .sessions()
        .create(Some("survives-restart".to_string()))
        .await
        .expect("sessions.create succeeds");
    let session_id = session_id_of(&created);

    client.disconnect().await;
    // Keep the data dir alive across the restart; dropping it would delete the store.
    let data_dir = daemon.stop();

    let restarted = TestDaemon::start(data_dir).await;
    let client = restarted.connect_client().await;
    let listed = client.sessions().list().await.expect("the restarted daemon answers");
    assert!(
        response_mentions(&listed, &session_id),
        "the session survived a full daemon restart: {listed}"
    );

    client.disconnect().await;
    restarted.stop();
}

/// A user tool is kept in the data dir the daemon was started on — the same
/// `DaemonConfig::data_dir` that `--data-dir` and `[general] data_dir` set, and
/// that holds the store. The control plane used to pick its own from the
/// platform default, which knows neither, so an isolated or relocated daemon
/// kept its user tools beside some other install's store (observed 2026-09-21:
/// a scratch daemon created `$XDG_DATA_HOME/nanna/user_tools`).
#[tokio::test]
async fn a_user_tool_is_kept_in_the_daemons_owndata_dir() {
    const NAME: &str = "e2edata_dir_probe";
    let data_dir = tempfile::tempdir().expect("temp dir");
    let tools_dir = data_dir.path().join("user_tools");
    let daemon = TestDaemon::start(data_dir).await;
    let client = daemon.connect_client().await;

    let create = nanna_client::ToolAction::Create {
        name: NAME.to_string(),
        description: "probe".to_string(),
        code: format!(
            "export default {{ name: \"{NAME}\", description: \"probe\", \
             execute() {{ return \"ok\"; }} }}"
        ),
        needs_shell: None,
    };
    let created = client
        .request(nanna_client::Action::Tool(create))
        .await
        .expect("tool.create answers");
    assert_eq!(created["status"], "created", "{created}");
    let written = tools_dir.join(format!("{NAME}.json"));
    assert!(
        written.is_file(),
        "the tool is written to {}, under the daemon's data dir",
        written.display()
    );

    client.disconnect().await;
    daemon.stop();
}

/// P8 "Client API completeness": a job added through the typed scheduler wrapper is
/// the one the daemon lists, fetches and removes.
#[tokio::test]
async fn scheduler_jobs_round_trip_through_the_typed_api() {
    let daemon = TestDaemon::start(tempfile::tempdir().expect("temp dir")).await;
    let client = daemon.connect_client().await;
    let scheduler = client.scheduler();

    // The daemon's cron dialect is five fields: minute hour day month weekday.
    let added = scheduler
        .add("0 9 * * *", "summarize the inbox", Some("morning-digest"))
        .await
        .expect("scheduler.add answers");
    assert_eq!(added["status"], "created", "{added}");
    let id = added["id"]
        .as_str()
        .expect("a created job has an id")
        .to_string();

    let listed = scheduler.list().await.expect("scheduler.list answers");
    assert!(
        response_mentions(&listed, &id),
        "the job is listed: {listed}"
    );
    let fetched = scheduler.get(&id).await.expect("scheduler.get answers");
    assert_eq!(fetched["job"]["name"], "morning-digest", "{fetched}");

    let removed = scheduler
        .remove(&id)
        .await
        .expect("scheduler.remove answers");
    assert_eq!(removed["status"], "deleted", "{removed}");
    let after = scheduler.list().await.expect("scheduler.list answers");
    assert!(!response_mentions(&after, &id), "the job is gone: {after}");

    client.disconnect().await;
    daemon.stop();
}

/// A project directory registered through the typed workspace wrapper is listed,
/// fetchable by id, and gone after `close`.
#[tokio::test]
async fn workspaces_open_list_and_close_through_the_typed_api() {
    let daemon = TestDaemon::start(tempfile::tempdir().expect("temp dir")).await;
    let client = daemon.connect_client().await;
    let workspaces = client.workspaces();
    let project = tempfile::tempdir().expect("temp project dir");
    let path = project.path().to_str().expect("a UTF-8 temp path");

    let opened = workspaces.open(path).await.expect("workspace.open answers");
    assert_eq!(opened["status"], "opened", "{opened}");
    let id = opened["id"]
        .as_str()
        .expect("an opened workspace has an id")
        .to_string();

    let listed = workspaces.list().await.expect("workspace.list answers");
    assert!(
        response_mentions(&listed, &id),
        "the workspace is listed: {listed}"
    );
    let fetched = workspaces.get(&id).await.expect("workspace.get answers");
    assert_eq!(fetched["workspace"]["id"], id.as_str(), "{fetched}");

    let closed = workspaces
        .close(&id)
        .await
        .expect("workspace.close answers");
    assert_eq!(closed["status"], "closed", "{closed}");
    let after = workspaces.list().await.expect("workspace.list answers");
    assert!(
        !response_mentions(&after, &id),
        "the workspace is gone: {after}"
    );

    client.disconnect().await;
    daemon.stop();
}

/// The typed channel wrapper reaches the daemon's adapter inventory: all five chat
/// adapters are reported, each with a boolean `configured`.
#[tokio::test]
async fn channels_list_every_adapter_through_the_typed_api() {
    let daemon = TestDaemon::start(tempfile::tempdir().expect("temp dir")).await;
    let client = daemon.connect_client().await;

    let listed = client
        .channels()
        .list()
        .await
        .expect("channel.list answers");
    let channels = listed["channels"].as_array().expect("a channel array");
    let ids: Vec<&str> = channels.iter().filter_map(|c| c["id"].as_str()).collect();
    for expected in ["telegram", "discord", "slack", "signal", "whatsapp"] {
        assert!(ids.contains(&expected), "{expected} is listed: {listed}");
    }
    assert!(
        channels.iter().all(|c| c["configured"].is_boolean()),
        "every adapter says whether it is configured: {listed}"
    );
    let status = client
        .channels()
        .status(None)
        .await
        .expect("channel.status answers");
    assert!(status.is_object(), "{status}");

    client.disconnect().await;
    daemon.stop();
}

/// Session lifecycle events reach every client: a rename and a delete made by one
/// client arrive on another client's per-session stream. Before the daemon emitted
/// them, a session renamed from the CLI never reached an open GUI.
#[tokio::test]
async fn lifecycle_changes_by_one_client_reach_another_clients_session_stream() {
    let daemon = TestDaemon::start(tempfile::tempdir().expect("temp dir")).await;
    let watcher = daemon.connect_client().await;
    let actor = daemon.connect_client().await;

    let created = actor
        .sessions()
        .create(Some("before".to_string()))
        .await
        .expect("sessions.create succeeds");
    let session_id = session_id_of(&created);
    // Subscribe before acting: a broadcast only carries what is sent after it.
    let mut events = watcher.subscribe_session(session_id.clone());

    actor
        .sessions()
        .rename(&session_id, "after")
        .await
        .expect("sessions.rename succeeds");
    // The watcher's socket can still be carrying the `SessionCreated` that the
    // create broadcast before this subscription existed: a subscription filters
    // the live stream, it does not fence it. That one event — and only that one
    // — may precede the rename. Observed 2026-09-11 under a parallel suite:
    // `expected SessionRenamed, got SessionCreated { .. name: Some("before") }`.
    let mut created_skipped = 0_usize;
    let renamed = loop {
        let event = tokio::time::timeout(READY_HANG_CEILING, events.recv())
            .await
            .expect("the rename event arrives before the hang ceiling")
            .expect("the stream is open and not lagging");
        if matches!(&event, nanna_client::Event::SessionCreated { id, .. } if *id == session_id) {
            created_skipped += 1;
            assert!(
                created_skipped <= 1,
                "only the one create can predate the subscription"
            );
            continue;
        }
        break event;
    };
    match renamed {
        nanna_client::Event::SessionRenamed { id, name } => {
            assert_eq!(id, session_id);
            assert_eq!(name, "after");
        }
        other => panic!("expected SessionRenamed, got {other:?}"),
    }

    actor
        .sessions()
        .delete(&session_id)
        .await
        .expect("sessions.delete succeeds");
    let deleted = tokio::time::timeout(READY_HANG_CEILING, events.recv())
        .await
        .expect("the delete event arrives before the hang ceiling")
        .expect("the stream is open and not lagging");
    assert!(
        matches!(&deleted, nanna_client::Event::SessionDeleted { id } if *id == session_id),
        "expected SessionDeleted for {session_id}, got {deleted:?}"
    );

    watcher.disconnect().await;
    actor.disconnect().await;
    daemon.stop();
}

/// `Subscribe{Session}` used to be recorded and then ignored: every connection
/// was forwarded every session's events, so one attached client saw another
/// session's traffic on the wire whether it wanted it or not.
///
/// Proven against the real daemon rather than the registry: a narrowed
/// connection stops receiving a session it never named, while a connection that
/// sent no `Subscribe` at all keeps receiving everything — which is what makes
/// the change a no-op for every shipping client.
#[tokio::test]
async fn narrowing_a_connection_stops_another_sessions_events_on_the_wire() {
    let daemon = TestDaemon::start(tempfile::tempdir().expect("temp dir")).await;
    let narrowed = daemon.connect_client().await;
    let unfiltered = daemon.connect_client().await;
    let actor = daemon.connect_client().await;

    let watched = session_id_of(
        &actor
            .sessions()
            .create(Some("watched".to_string()))
            .await
            .expect("sessions.create succeeds"),
    );
    let unwatched = session_id_of(
        &actor
            .sessions()
            .create(Some("unwatched".to_string()))
            .await
            .expect("sessions.create succeeds"),
    );

    narrowed
        .narrow_to_session(watched.clone())
        .await
        .expect("the daemon narrows to a session it holds");

    // Both connections watch the session that was NOT named. Only the narrowed
    // one should go quiet.
    let mut narrowed_stream = narrowed.subscribe_session(unwatched.clone());
    let mut unfiltered_stream = unfiltered.subscribe_session(unwatched.clone());

    actor
        .sessions()
        .rename(&unwatched, "renamed")
        .await
        .expect("sessions.rename succeeds");

    // The connection that never subscribed still gets it: this is the
    // no-regression half, and it also proves the rename really was broadcast,
    // so the silence asserted below is filtering rather than a missing event.
    let seen = tokio::time::timeout(READY_HANG_CEILING, unfiltered_stream.recv())
        .await
        .expect("the rename reaches the unfiltered client before the ceiling")
        .expect("the stream is open and not lagging");
    assert!(
        matches!(&seen, nanna_client::Event::SessionRenamed { id, .. } if *id == unwatched),
        "expected SessionRenamed for {unwatched}, got {seen:?}"
    );

    // The narrowed connection must not have been sent it. Its socket may still
    // carry the two `SessionCreated`s that predate the narrowing — a narrowing
    // filters the live stream, it does not fence it — so drain those and insist
    // nothing else arrives.
    let mut predating_creates = 0_usize;
    loop {
        match tokio::time::timeout(Duration::from_secs(2), narrowed_stream.recv()).await {
            Err(_) => break,
            Ok(Ok(nanna_client::Event::SessionCreated { id, .. })) if id == unwatched => {
                predating_creates += 1;
                assert!(
                    predating_creates <= 1,
                    "only the one create can predate the narrowing"
                );
            }
            Ok(other) => panic!(
                "a narrowed connection was sent an event for a session it never \
                 named — the leak this filter exists to close: {other:?}"
            ),
        }
    }

    // Widening restores it, so the narrowing is reversible rather than a
    // one-way door for the connection.
    narrowed
        .widen_to_all_sessions()
        .await
        .expect("widening back to every session succeeds");
    let mut widened_stream = narrowed.subscribe_session(unwatched.clone());
    actor
        .sessions()
        .rename(&unwatched, "renamed-again")
        .await
        .expect("sessions.rename succeeds");
    let after_widening = tokio::time::timeout(READY_HANG_CEILING, widened_stream.recv())
        .await
        .expect("the rename reaches the widened client before the ceiling")
        .expect("the stream is open and not lagging");
    assert!(
        matches!(
            &after_widening,
            nanna_client::Event::SessionRenamed { id, name } if *id == unwatched && name == "renamed-again"
        ),
        "expected the widened connection to receive SessionRenamed, got {after_widening:?}"
    );

    narrowed.disconnect().await;
    unfiltered.disconnect().await;
    actor.disconnect().await;
    daemon.stop();
}

/// A daemon with no model configured answers a chat by naming the missing
/// setting, instead of running a turn on a model named "".
///
/// That blank name used to resolve to whichever provider claims unprefixed
/// names, so the turn's steps sent requests naming no model (a debug-assertion
/// panic in the agent loop, a provider error in release) and the user got no
/// explanation.
#[tokio::test]
async fn a_chat_with_no_model_configured_says_which_setting_is_missing() {
    let daemon = TestDaemon::start(tempfile::tempdir().expect("temp dir")).await;
    let client = daemon.connect_client().await;
    let session = session_id_of(
        &client
            .sessions()
            .create(Some("no model".to_string()))
            .await
            .expect("sessions.create succeeds"),
    );
    let mut events = client.subscribe_session(session.clone());

    client
        .chat()
        .send(&session, "hello")
        .await
        .expect("chat.send is accepted");

    let explained = tokio::time::timeout(READY_HANG_CEILING, async {
        loop {
            match events.recv().await {
                Ok(nanna_client::Event::MessageDelta { delta, .. })
                    if delta.contains("could not run") =>
                {
                    return delta;
                }
                Ok(_) => {}
                Err(e) => panic!("the session's event stream ended before the turn answered: {e:?}"),
            }
        }
    })
    .await
    .expect("the turn answers before the ceiling");
    assert!(
        explained.contains("No model is configured") && explained.contains("[llm] model"),
        "the reply names the missing setting: {explained}"
    );

    client.disconnect().await;
    daemon.stop();
}

/// The scripted model's `/api/show`: a 32K-context tool-calling model.
const SHOW_REPLY: &str = r#"{"model_info":{"general.architecture":"llama","llama.context_length":32768},"capabilities":["completion","tools"]}"#;

/// How `nanna_agent::planner::build_plan_prompt` opens — the stub's cue that a
/// request is the planner's rather than a step's.
const PLANNER_PROMPT_OPENING: &str = "You are planning how to satisfy one request";

/// The model a conversation-turn test configures; the `:tag` routes it to the
/// Ollama provider, which [`ScriptedOllama`] stands in for.
const STUB_MODEL: &str = "e2e-stub:1b";

/// A model the scripted server answers with Ollama's unknown-model 404.
const UNSERVED_MODEL: &str = "e2e-gone:1b";

/// A scripted Ollama server playing a well-behaved model, and keeping each
/// chat request body so a test can see what reached the model.
///
/// It answers the two prompts a chat turn sends the way the daemon asks them
/// to be answered: the planner gets ONE JSON task with no machine check (what
/// the planner's rules prescribe for a question), and step `n` gets
/// `steps[n]` verbatim (the last entry repeats) — a finishing step ends with
/// `TASK COMPLETE` on its own line, which is what the step prompt asks for
/// when no machine check exists. Anything else gets `{}`. A stub that
/// answered every prompt with the same prose would test the harness's
/// fallback ladder instead — the planner starves, no step ever completes, and
/// the turn is abandoned — which is a different test.
struct ScriptedOllama {
    base_url: String,
    chat_bodies: std::sync::Arc<tokio::sync::Mutex<Vec<String>>>,
}

impl ScriptedOllama {
    async fn start(steps: Vec<String>) -> Self {
        Self::start_with_plan(
            r#"[{"title":"Answer: {request}","description":"Reply directly.","acceptance":null}]"#,
            steps,
        )
        .await
    }

    /// [`Self::start`] with the planner's reply scripted too.
    async fn start_with_plan(plan: &str, steps: Vec<String>) -> Self {
        assert!(
            !steps.is_empty(),
            "a scripted model needs at least one step reply"
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind the scripted Ollama");
        let base_url = format!("http://{}", listener.local_addr().expect("stub address"));
        let chat_bodies = std::sync::Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let seen = std::sync::Arc::clone(&chat_bodies);
        // `{request}` in the planner's script is replaced by the request the
        // planner was asked about, so each turn's task is titled after its own
        // message — as a real planner's would be — rather than every turn
        // proposing the same title.
        let plan = plan.to_string();
        // `WAIT <ms> <script>` delays that reply — a step still in flight.
        let steps: std::sync::Arc<Vec<(u64, String)>> = std::sync::Arc::new(
            steps
                .iter()
                .map(|text| {
                    text.strip_prefix("WAIT ").map_or_else(
                        || (0, text.clone()),
                        |rest| {
                            let (ms, script) = rest.split_once(' ').unwrap_or((rest, ""));
                            (ms.parse().expect("WAIT takes milliseconds"), script.to_string())
                        },
                    )
                })
                .collect(),
        );
        let plan = std::sync::Arc::new(plan);
        let step_index = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        tokio::spawn(async move {
            // One task per connection, so a delayed reply never holds up the
            // next request (a cancel's follow-up turn, a model probe).
            while let Ok((mut socket, _)) = listener.accept().await {
                let (seen, plan, steps, step_index) = (
                    std::sync::Arc::clone(&seen),
                    std::sync::Arc::clone(&plan),
                    std::sync::Arc::clone(&steps),
                    std::sync::Arc::clone(&step_index),
                );
                tokio::spawn(async move {
                    let Some((request_line, body)) = read_http_request(&mut socket).await else {
                        return;
                    };
                    // Model info like a real 32K tool-calling model, so the
                    // daemon sizes its context the way it would in use.
                    if request_line.contains("/api/show") {
                        respond(&mut socket, SHOW_REPLY).await;
                        return;
                    }
                    if !request_line.contains("/api/chat") {
                        respond(&mut socket, "{}").await;
                        return;
                    }
                    // The model this server does not serve, answered the way
                    // Ollama answers an unknown model.
                    if body.contains(&format!("\"model\":\"{UNSERVED_MODEL}\"")) {
                        respond_status(
                            &mut socket,
                            "404 Not Found",
                            &format!("{{\"error\":\"model '{UNSERVED_MODEL}' not found\"}}"),
                        )
                        .await;
                        return;
                    }
                    let (delay_ms, line) = if body.contains(PLANNER_PROMPT_OPENING) {
                        (
                            0,
                            scripted_reply(&plan.replace("{request}", &planned_request(&body))),
                        )
                    } else {
                        let index = step_index.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        let (delay_ms, script) = &steps[index.min(steps.len() - 1)];
                        // `{goal}` names the request the step is working on,
                        // so concurrent sessions' replies can be told apart.
                        (
                            *delay_ms,
                            scripted_reply(&script.replace("{goal}", &step_goal(&body))),
                        )
                    };
                    seen.lock().await.push(body);
                    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                    // One NDJSON line serves both shapes: a streamed request
                    // reads it as its only (final) chunk.
                    respond(&mut socket, &format!("{line}\n")).await;
                });
            }
        });
        Self {
            base_url,
            chat_bodies,
        }
    }
}

/// One scripted model message as an Ollama NDJSON line. `CALL <tool> <json>`
/// is a tool call with those arguments; anything else is reply text.
fn scripted_reply(script: &str) -> String {
    let message = script.strip_prefix("CALL ").map_or_else(
        || serde_json::json!({ "role": "assistant", "content": script }),
        |call| {
            let (name, arguments) = call.split_once(' ').unwrap_or((call, "{}"));
            let arguments: serde_json::Value =
                serde_json::from_str(arguments).expect("scripted tool arguments are JSON");
            serde_json::json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{ "function": { "name": name, "arguments": arguments } }],
            })
        },
    );
    serde_json::json!({
        "model": STUB_MODEL,
        "message": message,
        "done": true,
        "done_reason": "stop",
        "prompt_eval_count": 20,
        "eval_count": 8,
    })
    .to_string()
}

/// The request a planner prompt asks about (the text between `== REQUEST ==`
/// and `JSON array:`), JSON-escaped for splicing into a scripted plan.
fn planned_request(body: &str) -> String {
    let prompt = serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|request| {
            request["messages"].as_array()?.last()?["content"]
                .as_str()
                .map(str::to_string)
        })
        .unwrap_or_default();
    let request = prompt
        .split("== REQUEST ==")
        .nth(1)
        .and_then(|rest| rest.split("JSON array:").next())
        .unwrap_or_default()
        .trim();
    let quoted = serde_json::to_string(request).unwrap_or_default();
    quoted.trim_matches('"').to_string()
}

/// The goal a step prompt is working on (the line after `== GOAL`),
/// JSON-escaped for splicing into a scripted reply.
fn step_goal(body: &str) -> String {
    let prompt = serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|request| {
            request["messages"].as_array()?.last()?["content"]
                .as_str()
                .map(str::to_string)
        })
        .unwrap_or_default();
    let goal = prompt
        .split("== GOAL")
        .nth(1)
        .and_then(|rest| rest.lines().nth(1))
        .unwrap_or_default()
        .trim()
        .to_string();
    let quoted = serde_json::to_string(&goal).unwrap_or_default();
    quoted.trim_matches('"').to_string()
}

/// Read one HTTP/1.1 request; returns (request line, body).
async fn read_http_request(stream: &mut tokio::net::TcpStream) -> Option<(String, String)> {
    use tokio::io::AsyncReadExt;
    let mut buf: Vec<u8> = Vec::new();
    let mut chunk = [0u8; 4096];
    let header_end = loop {
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break pos;
        }
        let n = stream.read(&mut chunk).await.ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&chunk[..n]);
    };
    let headers = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let request_line = headers.lines().next().unwrap_or("").to_string();
    let content_length = headers
        .lines()
        .find_map(|line| {
            let (key, value) = line.split_once(':')?;
            key.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())?
        })
        .unwrap_or(0);
    let body_start = header_end + 4;
    while buf.len() < body_start + content_length {
        let n = stream.read(&mut chunk).await.ok()?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
    }
    let body_end = (body_start + content_length).min(buf.len());
    Some((
        request_line,
        String::from_utf8_lossy(&buf[body_start..body_end]).to_string(),
    ))
}

async fn respond(stream: &mut tokio::net::TcpStream, body: &str) {
    respond_status(stream, "200 OK", body).await;
}

/// [`respond`] with an explicit status line.
async fn respond_status(stream: &mut tokio::net::TcpStream, status: &str, body: &str) {
    use tokio::io::AsyncWriteExt;
    let response = format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = stream.write_all(response.as_bytes()).await;
    let _ = stream.shutdown().await;
}

/// The rest of P8's end-to-end item: **a real conversation turn**, through
/// the real daemon, the real IPC and the real client — the message reaches the
/// model, the answer comes back as `message_end`, and it is persisted to the
/// session so a client that asks later reads the same reply. The model is the
/// one part that is scripted; everything between the client and the model's
/// socket is the shipping path.
#[tokio::test]
async fn a_conversation_turn_round_trips_and_persists_its_reply() {
    const USER_TEXT: &str = "What is the capital of France?";
    const REPLY: &str = "Paris is the capital of France.";

    let ollama = ScriptedOllama::start(vec![format!("{REPLY}\nTASK COMPLETE")]).await;
    let host = ollama.base_url.clone();
    let daemon = TestDaemon::start_with(tempfile::tempdir().expect("temp dir"), move |b| {
        b.with_model(STUB_MODEL)
            .with_ollama_host(host)
            // No heartbeat turn: every chat request the stub sees is ours.
            .with_scheduler(false)
    })
    .await;
    let client = daemon.connect_client().await;
    let session = session_id_of(
        &client
            .sessions()
            .create(Some("conversation".to_string()))
            .await
            .expect("sessions.create succeeds"),
    );
    let mut events = client.subscribe_session(session.clone());

    let ack = client
        .chat()
        .send(&session, USER_TEXT)
        .await
        .expect("chat.send is accepted");
    let message_id = ack["message_id"]
        .as_str()
        .unwrap_or_else(|| panic!("the ack names the turn's message: {ack}"))
        .to_string();

    let answered = tokio::time::timeout(READY_HANG_CEILING, async {
        loop {
            match events.recv().await {
                Ok(nanna_client::Event::MessageEnd {
                    message_id: id,
                    content,
                    ..
                }) if id == message_id => {
                    return content;
                }
                Ok(_) => {}
                Err(e) => panic!("the session's event stream ended before the turn did: {e:?}"),
            }
        }
    })
    .await
    .expect("the turn ends before the hang ceiling");
    // Exactly the answer: not the marker the harness consumes, not a
    // "could not finish" footer, not the reply repeated once per retried step.
    assert_eq!(
        answered.trim(),
        REPLY,
        "message_end carries the model's answer and nothing else"
    );

    // The message reached the model — not a canned daemon-side reply.
    let bodies = ollama.chat_bodies.lock().await.clone();
    assert!(
        bodies.iter().any(|body| body.contains(USER_TEXT)),
        "no chat request carried the user's message ({} requests)",
        bodies.len()
    );

    // And the turn is durable: a fresh read of the session holds both sides.
    let history = client
        .sessions()
        .history(&session, None)
        .await
        .expect("sessions.history answers");
    let text = history.to_string();
    assert!(
        text.contains(USER_TEXT),
        "the user's message is persisted: {history}"
    );
    assert!(
        text.contains(REPLY),
        "the assistant's reply is persisted: {history}"
    );

    client.disconnect().await;
    daemon.stop();
}

/// A chat whose head model fails is answered by the next model in the
/// priority list, and says so — the walk the retired direct-chat path did and
/// the harness path (every chat turn) did not: a failing head ended the turn
/// in "could not run" with working models configured below it (2026-09-21).
#[tokio::test]
async fn a_failing_head_model_falls_back_to_the_next_in_the_priority_list() {
    const REPLY: &str = "Answered by the fallback model.";

    let ollama = ScriptedOllama::start(vec![format!("{REPLY}\nTASK COMPLETE")]).await;
    let host = ollama.base_url.clone();
    let daemon = TestDaemon::start_with(tempfile::tempdir().expect("temp dir"), move |b| {
        b.with_model_priority(vec![
            format!("ollama/{UNSERVED_MODEL}"),
            format!("ollama/{STUB_MODEL}"),
        ])
        .with_ollama_host(host)
        .with_scheduler(false)
    })
    .await;
    let client = daemon.connect_client().await;
    let session = session_id_of(
        &client
            .sessions()
            .create(Some("fallback".to_string()))
            .await
            .expect("sessions.create succeeds"),
    );

    let answered = converse(&client, &session, "hello").await;

    assert_eq!(answered.trim(), REPLY, "the next model's answer is the reply");
    let bodies = ollama.chat_bodies.lock().await.clone();
    assert!(
        bodies.iter().any(|body| body.contains(STUB_MODEL)),
        "the fallback model was asked"
    );
    // The switch is on the record, not silent.
    let history = client
        .sessions()
        .history(&session, None)
        .await
        .expect("sessions.history answers")
        .to_string();
    assert!(
        history.contains(&format!("continuing on ollama/{STUB_MODEL}")),
        "the run journal names the fallback: {history}"
    );

    client.disconnect().await;
    daemon.stop();
}

/// A turn that takes two steps reads as two paragraphs, not one run-on line.
///
/// Each harness step streams into the same reply. With the step banner out
/// of the text (run mechanics are not content) nothing separated them, and
/// the reply — live and persisted — read `…the question.Paris is…`.
#[tokio::test]
async fn a_two_step_turn_reads_as_two_paragraphs() {
    const FIRST: &str = "Let me recall the answer.";
    const SECOND: &str = "Paris is the capital of France.";

    // Step one makes no claim to be finished, so the harness takes another
    // step; step two finishes.
    let ollama =
        ScriptedOllama::start(vec![FIRST.to_string(), format!("{SECOND}\nTASK COMPLETE")]).await;
    let host = ollama.base_url.clone();
    let daemon = TestDaemon::start_with(tempfile::tempdir().expect("temp dir"), move |b| {
        b.with_model(STUB_MODEL)
            .with_ollama_host(host)
            .with_scheduler(false)
    })
    .await;
    let client = daemon.connect_client().await;
    let session = session_id_of(
        &client
            .sessions()
            .create(Some("two steps".to_string()))
            .await
            .expect("sessions.create succeeds"),
    );
    let mut events = client.subscribe_session(session.clone());
    let ack = client
        .chat()
        .send(&session, "What is the capital of France?")
        .await
        .expect("chat.send is accepted");
    let message_id = ack["message_id"]
        .as_str()
        .unwrap_or_else(|| panic!("the ack names the turn's message: {ack}"))
        .to_string();

    let answered = tokio::time::timeout(READY_HANG_CEILING, async {
        loop {
            match events.recv().await {
                Ok(nanna_client::Event::MessageEnd {
                    message_id: id,
                    content,
                    ..
                }) if id == message_id => {
                    return content;
                }
                Ok(_) => {}
                Err(e) => panic!("the session's event stream ended before the turn did: {e:?}"),
            }
        }
    })
    .await
    .expect("the turn ends before the hang ceiling");
    // A multi-step turn ends with the harness's own `_N steps · …_` summary
    // line; the seam under test is the one between the two steps' text.
    assert!(
        answered.starts_with(&format!("{FIRST}\n\n{SECOND}")),
        "the two steps' text is separated by a paragraph break: {answered:?}"
    );

    let history = client
        .sessions()
        .history(&session, None)
        .await
        .expect("sessions.history answers");
    assert!(
        history
            .to_string()
            .contains(&format!("{FIRST}\\n\\n{SECOND}")),
        "the persisted reply keeps the paragraph break: {history}"
    );

    client.disconnect().await;
    daemon.stop();
}

/// A model that answers but never says `TASK COMPLETE` — common for small
/// local models — still gets a finished turn, not an abandoned one.
///
/// Before, the harness re-ran the step until the item's fruitless budget ran
/// out, streaming the same answer each time: a plain question came back as
/// its answer seven times, then `_could not finish: every planned task was
/// abandoned_`. Now the second identical tool-free answer closes the item.
#[tokio::test]
async fn a_model_that_never_claims_completion_still_finishes_the_turn() {
    const REPLY: &str = "Paris is the capital of France.";

    let ollama = ScriptedOllama::start(vec![REPLY.to_string()]).await;
    let host = ollama.base_url.clone();
    let daemon = TestDaemon::start_with(tempfile::tempdir().expect("temp dir"), move |b| {
        b.with_model(STUB_MODEL)
            .with_ollama_host(host)
            .with_scheduler(false)
    })
    .await;
    let client = daemon.connect_client().await;
    let session = session_id_of(
        &client
            .sessions()
            .create(Some("no claim".to_string()))
            .await
            .expect("sessions.create succeeds"),
    );
    let mut events = client.subscribe_session(session.clone());
    let ack = client
        .chat()
        .send(&session, "What is the capital of France?")
        .await
        .expect("chat.send is accepted");
    let message_id = ack["message_id"]
        .as_str()
        .unwrap_or_else(|| panic!("the ack names the turn's message: {ack}"))
        .to_string();

    let answered = tokio::time::timeout(READY_HANG_CEILING, async {
        loop {
            match events.recv().await {
                Ok(nanna_client::Event::MessageEnd {
                    message_id: id,
                    content,
                    ..
                }) if id == message_id => {
                    return content;
                }
                Ok(_) => {}
                Err(e) => panic!("the session's event stream ended before the turn did: {e:?}"),
            }
        }
    })
    .await
    .expect("the turn ends before the hang ceiling");
    assert!(
        !answered.contains("could not finish"),
        "a converged answer is a finished turn: {answered:?}"
    );
    // The stream shows the answer and its converging repeat; the reply the
    // user keeps (and the GUI's bubble is replaced by) holds it once.
    assert_eq!(
        answered.matches(REPLY).count(),
        1,
        "one copy kept — never the fruitless budget's worth: {answered:?}"
    );
    let step_requests = ollama
        .chat_bodies
        .lock()
        .await
        .iter()
        .filter(|body| !body.contains(PLANNER_PROMPT_OPENING))
        .count();
    assert_eq!(step_requests, 2, "two steps, then done");

    client.disconnect().await;
    daemon.stop();
}

/// Send `text` on `session` and wait for that turn's `message_end`.
///
/// Subscribes before sending: a broadcast only carries what is sent after it.
async fn converse(client: &Client, session: &str, text: &str) -> String {
    let mut events = client.subscribe_session(session.to_string());
    let ack = client
        .chat()
        .send(session, text)
        .await
        .expect("chat.send is accepted");
    let message_id = ack["message_id"]
        .as_str()
        .unwrap_or_else(|| panic!("the ack names the turn's message: {ack}"))
        .to_string();
    tokio::time::timeout(READY_HANG_CEILING, async {
        loop {
            match events.recv().await {
                Ok(nanna_client::Event::MessageEnd {
                    message_id: id,
                    content,
                    ..
                }) if id == message_id => {
                    return content;
                }
                Ok(_) => {}
                Err(e) => panic!("the session's event stream ended before the turn did: {e:?}"),
            }
        }
    })
    .await
    .expect("the turn ends before the hang ceiling")
}

/// A follow-up turn's planner is told what earlier turns closed — and an
/// answer that closed on the model's word (no check ran) must not be
/// presented as a passing done-condition it is told not to re-assess. That
/// is exactly the item a follow-up like "that's wrong" is about.
#[tokio::test]
async fn a_follow_up_turn_is_not_told_an_unchecked_answer_passed_a_check() {
    let ollama = ScriptedOllama::start(vec![
        "Paris is the capital of France.\nTASK COMPLETE".to_string(),
    ])
    .await;
    let host = ollama.base_url.clone();
    let daemon = TestDaemon::start_with(tempfile::tempdir().expect("temp dir"), move |b| {
        b.with_model(STUB_MODEL)
            .with_ollama_host(host)
            .with_scheduler(false)
    })
    .await;
    let client = daemon.connect_client().await;
    let session = session_id_of(
        &client
            .sessions()
            .create(Some("follow-up".to_string()))
            .await
            .expect("sessions.create succeeds"),
    );
    converse(&client, &session, "What is the capital of France?").await;
    converse(&client, &session, "Are you sure?").await;

    let bodies = ollama.chat_bodies.lock().await.clone();
    let follow_up_plan = bodies
        .iter()
        .filter(|body| body.contains(PLANNER_PROMPT_OPENING))
        .nth(1)
        .expect("the second turn was planned");
    assert!(
        follow_up_plan.contains("no check ran"),
        "the earlier answer is listed as unverified"
    );
    assert!(
        !follow_up_plan.contains("PASSING"),
        "nothing in this session passed a check, so nothing may be presented as one"
    );

    client.disconnect().await;
    daemon.stop();
}

/// A model that writes its reasoning inline — `<think>…</think>` in the
/// content, which Ollama passes through when it is not separating thinking —
/// must not put that reasoning in the user's reply. The non-streaming path
/// stripped it; chat streams, and the streaming path passed it through.
#[tokio::test]
async fn inline_reasoning_stays_out_of_the_reply() {
    let ollama = ScriptedOllama::start(vec![
        "<think>The user greets me; greet back.</think>\n\nHello there!\nTASK COMPLETE".to_string(),
    ])
    .await;
    let host = ollama.base_url.clone();
    let daemon = TestDaemon::start_with(tempfile::tempdir().expect("temp dir"), move |b| {
        b.with_model(STUB_MODEL)
            .with_ollama_host(host)
            .with_scheduler(false)
    })
    .await;
    let client = daemon.connect_client().await;
    let session = session_id_of(
        &client
            .sessions()
            .create(Some("inline think".to_string()))
            .await
            .expect("sessions.create succeeds"),
    );
    let reply = converse(&client, &session, "hi").await;
    assert_eq!(reply.trim(), "Hello there!");

    // Persisted the same way: the reply's content is the reply, and the
    // reasoning is kept — as a thinking entry in its timeline, where the GUI
    // renders reasoning, never as reply text.
    let history = client
        .sessions()
        .history(&session, None)
        .await
        .expect("sessions.history answers");
    let reply_message = history["messages"]
        .as_array()
        .and_then(|messages| messages.iter().find(|m| m["role"] == "assistant"))
        .unwrap_or_else(|| panic!("the reply is persisted: {history}"));
    assert_eq!(reply_message["content"], "Hello there!", "{history}");
    let timeline = reply_message["timeline"].as_array().expect("a timeline");
    assert!(
        timeline.iter().any(|item| item["kind"] == "thinking"
            && item["content"]
                .as_str()
                .is_some_and(|c| c.contains("greet back"))),
        "the reasoning is kept as thinking: {history}"
    );
    assert!(
        !history.to_string().contains("<think>"),
        "no tag survives anywhere: {history}"
    );

    client.disconnect().await;
    daemon.stop();
}

/// A model that answers with the bare `TASK COMPLETE` marker closes its item,
/// and the marker is stripped from the reply — which used to leave an empty
/// message the GUI hides: the user saw nothing at all. The turn now says it
/// finished without a reply.
#[tokio::test]
async fn a_claimed_completion_with_nothing_said_is_stated_not_silent() {
    let ollama = ScriptedOllama::start(vec!["TASK COMPLETE".to_string()]).await;
    let host = ollama.base_url.clone();
    let daemon = TestDaemon::start_with(tempfile::tempdir().expect("temp dir"), move |b| {
        b.with_model(STUB_MODEL)
            .with_ollama_host(host)
            .with_scheduler(false)
    })
    .await;
    let client = daemon.connect_client().await;
    let session = session_id_of(
        &client
            .sessions()
            .create(Some("silent".to_string()))
            .await
            .expect("sessions.create succeeds"),
    );
    let reply = converse(&client, &session, "hi").await;
    assert!(
        reply.contains("finished without a reply"),
        "an empty finished turn states itself: {reply:?}"
    );
    assert!(
        !reply.contains("TASK COMPLETE"),
        "the marker stays plumbing"
    );

    client.disconnect().await;
    daemon.stop();
}

/// A tool-using turn end to end: the model discovers a tool, calls it, and
/// reports — and the persisted reply records both calls, each under its own
/// id. Ollama sends no call ids; the synthesized ones used to restart per
/// response, so both calls here were `toolu_00000001`.
#[tokio::test]
async fn a_tool_using_turn_records_each_call_under_its_own_id() {
    let ollama = ScriptedOllama::start(vec![
        r#"CALL discover_tools {"query":"echo"}"#.to_string(),
        r#"CALL echo {"text":"hello from echo"}"#.to_string(),
        "The echo tool said hello from echo.\nTASK COMPLETE".to_string(),
    ])
    .await;
    let host = ollama.base_url.clone();
    let daemon = TestDaemon::start_with(tempfile::tempdir().expect("temp dir"), move |b| {
        b.with_model(STUB_MODEL)
            .with_ollama_host(host)
            .with_scheduler(false)
    })
    .await;
    let client = daemon.connect_client().await;
    let session = session_id_of(
        &client
            .sessions()
            .create(Some("tools".to_string()))
            .await
            .expect("sessions.create succeeds"),
    );
    let reply = converse(&client, &session, "Echo hello.").await;
    assert_eq!(reply.trim(), "The echo tool said hello from echo.");

    let history = client
        .sessions()
        .history(&session, None)
        .await
        .expect("sessions.history answers");
    let timeline = history["messages"]
        .as_array()
        .and_then(|messages| messages.iter().find(|m| m["role"] == "assistant"))
        .and_then(|reply| reply["timeline"].as_array())
        .unwrap_or_else(|| panic!("the reply is persisted with a timeline: {history}"));
    let calls: Vec<(&str, &str, bool)> = timeline
        .iter()
        .filter(|item| item["kind"] == "tool")
        .map(|item| {
            (
                item["name"].as_str().unwrap_or_default(),
                item["call_id"].as_str().unwrap_or_default(),
                item["success"].as_bool().unwrap_or(false),
            )
        })
        .collect();
    assert_eq!(calls.len(), 2, "{calls:?}");
    assert_eq!((calls[0].0, calls[1].0), ("discover_tools", "echo"));
    assert!(
        calls.iter().all(|call| call.2),
        "both calls succeeded: {calls:?}"
    );
    assert_ne!(
        calls[0].1, calls[1].1,
        "each call has its own id: {calls:?}"
    );

    client.disconnect().await;
    daemon.stop();
}

/// A call to a tool that does not exist is answered with a pointer to
/// `discover_tools`, and the turn carries on. What the model reads back is
/// the error stated once: the loop writes failures as `Error: …` and the
/// Ollama wire used to prefix `Error: ` again.
#[tokio::test]
async fn a_call_to_a_missing_tool_is_reported_once_and_the_turn_recovers() {
    let ollama = ScriptedOllama::start(vec![
        r#"CALL frobnicate {"x":1}"#.to_string(),
        "There is no such tool.\nTASK COMPLETE".to_string(),
    ])
    .await;
    let host = ollama.base_url.clone();
    let daemon = TestDaemon::start_with(tempfile::tempdir().expect("temp dir"), move |b| {
        b.with_model(STUB_MODEL)
            .with_ollama_host(host)
            .with_scheduler(false)
    })
    .await;
    let client = daemon.connect_client().await;
    let session = session_id_of(
        &client
            .sessions()
            .create(Some("missing tool".to_string()))
            .await
            .expect("sessions.create succeeds"),
    );
    let reply = converse(&client, &session, "Frobnicate it.").await;
    assert_eq!(reply.trim(), "There is no such tool.");

    let bodies = ollama.chat_bodies.lock().await.clone();
    let tool_result = bodies
        .iter()
        .filter_map(|body| serde_json::from_str::<serde_json::Value>(body).ok())
        .flat_map(|request| request["messages"].as_array().cloned().unwrap_or_default())
        .find(|message| message["role"] == "tool")
        .expect("the failed call's result was sent back to the model");
    let content = tool_result["content"].as_str().unwrap_or_default();
    assert!(
        content.starts_with("Error: Tool not found: frobnicate"),
        "{content:?}"
    );
    assert!(
        !content.contains("Error: Error:"),
        "stated once: {content:?}"
    );

    client.disconnect().await;
    daemon.stop();
}

/// Stop, end to end: a turn whose model is still generating is cancelled, its
/// late reply never reaches the transcript, and the session takes the next
/// message normally.
#[tokio::test]
async fn stop_ends_an_in_flight_turn_and_the_session_carries_on() {
    // The first step's reply is held back well past the cancel.
    let ollama = ScriptedOllama::start(vec![
        "WAIT 3000 Too late.\nTASK COMPLETE".to_string(),
        "Second answer.\nTASK COMPLETE".to_string(),
    ])
    .await;
    let host = ollama.base_url.clone();
    let daemon = TestDaemon::start_with(tempfile::tempdir().expect("temp dir"), move |b| {
        b.with_model(STUB_MODEL)
            .with_ollama_host(host)
            .with_scheduler(false)
    })
    .await;
    let client = daemon.connect_client().await;
    let session = session_id_of(
        &client
            .sessions()
            .create(Some("stop".to_string()))
            .await
            .expect("sessions.create succeeds"),
    );
    let mut events = client.subscribe_session(session.clone());
    let ack = client
        .chat()
        .send(&session, "first")
        .await
        .expect("chat.send is accepted");
    let first = ack["message_id"]
        .as_str()
        .unwrap_or_else(|| panic!("the ack names the turn's message: {ack}"))
        .to_string();

    // Cancel once the step is really in flight: the stub has seen it.
    tokio::time::timeout(READY_HANG_CEILING, async {
        while ollama.chat_bodies.lock().await.len() < 2 {
            tokio::time::sleep(READY_POLL_INTERVAL).await;
        }
    })
    .await
    .expect("the planner and the first step reach the model");
    client
        .chat()
        .cancel(&session)
        .await
        .expect("chat.cancel answers");

    let stopped = tokio::time::timeout(READY_HANG_CEILING, async {
        loop {
            match events.recv().await {
                Ok(nanna_client::Event::MessageEnd {
                    message_id,
                    content,
                    ..
                }) if message_id == first => return content,
                Ok(_) => {}
                Err(e) => panic!("the event stream ended before the stop landed: {e:?}"),
            }
        }
    })
    .await
    .expect("the stopped turn ends");
    assert!(
        !stopped.contains("Too late."),
        "the in-flight reply must not land after Stop: {stopped:?}"
    );
    // Persisted as the GUI showed it: the marker it puts on the live bubble,
    // not an empty message that erases it.
    assert_eq!(stopped.trim(), "[Stopped by user]");

    // Owner directive: an earlier turn's unfinished work is information
    // for the planner, not an instruction to resume. The stopped request
    // must not be worked on the user's next, unrelated message — it used to
    // be, and this reply read "Second answer.\n\nSecond answer.\n\n_2 steps ·
    // 2 items completed_".
    let next = converse(&client, &session, "second").await;
    assert_eq!(
        next.trim(),
        "Second answer.",
        "the next message is answered, and only it"
    );

    client.disconnect().await;
    daemon.stop();
}

/// A long non-ASCII reply arrives whole. Every provider stream decoded each
/// network chunk as UTF-8 on its own, so the first multibyte character a
/// chunk boundary split — here at byte 32767 of a 100 KB reply — failed the
/// stream with `Invalid UTF-8`, the retries hit the same wall, and the user
/// got `could not run` after more than a minute.
#[tokio::test]
async fn a_long_multibyte_reply_survives_chunk_boundaries() {
    let reply: String = "Ünïcödé 🌙 月 — ".repeat(4000);
    let ollama = ScriptedOllama::start(vec![format!("{reply}\nTASK COMPLETE")]).await;
    let host = ollama.base_url.clone();
    let daemon = TestDaemon::start_with(tempfile::tempdir().expect("temp dir"), move |b| {
        b.with_model(STUB_MODEL)
            .with_ollama_host(host)
            .with_scheduler(false)
    })
    .await;
    let client = daemon.connect_client().await;
    let session = session_id_of(
        &client
            .sessions()
            .create(Some("unicode".to_string()))
            .await
            .expect("sessions.create succeeds"),
    );
    let answered = converse(&client, &session, "Talk to me.").await;
    assert_eq!(
        answered.trim().len(),
        reply.trim().len(),
        "the whole reply arrived: {:?}",
        answered.chars().take(160).collect::<String>()
    );
    assert_eq!(answered.trim(), reply.trim());

    client.disconnect().await;
    daemon.stop();
}

/// A planner that answers in prose instead of the JSON array it was asked
/// for — routine for small local models — still gets the question answered:
/// the request itself becomes the one task, and the turn reads like chat.
#[tokio::test]
async fn a_planner_that_answers_in_prose_still_gets_the_question_answered() {
    let ollama = ScriptedOllama::start_with_plan(
        "Sure! I'll answer the question about France.",
        vec!["Paris.\nTASK COMPLETE".to_string()],
    )
    .await;
    let host = ollama.base_url.clone();
    let daemon = TestDaemon::start_with(tempfile::tempdir().expect("temp dir"), move |b| {
        b.with_model(STUB_MODEL)
            .with_ollama_host(host)
            .with_scheduler(false)
    })
    .await;
    let client = daemon.connect_client().await;
    let session = session_id_of(
        &client
            .sessions()
            .create(Some("prose plan".to_string()))
            .await
            .expect("sessions.create succeeds"),
    );
    let reply = converse(&client, &session, "What is the capital of France?").await;
    assert_eq!(reply.trim(), "Paris.", "no planning mechanics in the reply");

    client.disconnect().await;
    daemon.stop();
}

/// Reminders, end to end, including the promise the `remind` skill makes to
/// the model: the reminder "survives a restart of Nanna". The model sets one
/// through the real skill and service, the daemon is stopped and started on
/// the same data dir, and the reminder is posted into the conversation.
///
/// Slow by necessity: due reminders are swept every 30 s, so delivery lands
/// up to that long after the restart.
#[tokio::test]
async fn a_reminder_set_in_chat_survives_a_restart_and_is_delivered() {
    let ollama = ScriptedOllama::start(vec![
        r#"CALL remind {"message":"stretch your legs","delay_secs":2}"#.to_string(),
        "Reminder set.\nTASK COMPLETE".to_string(),
    ])
    .await;
    let host = ollama.base_url.clone();
    let configure = move |b: DaemonBuilder| {
        b.with_model(STUB_MODEL)
            .with_ollama_host(host.clone())
            // The scheduler must run (it delivers reminders); only the
            // heartbeat turn is kept out of the scripted conversation.
            .with_heartbeat(false)
    };
    let daemon =
        TestDaemon::start_with(tempfile::tempdir().expect("temp dir"), configure.clone()).await;
    let client = daemon.connect_client().await;
    let session = session_id_of(
        &client
            .sessions()
            .create(Some("reminder".to_string()))
            .await
            .expect("sessions.create succeeds"),
    );
    let reply = converse(&client, &session, "Remind me to stretch in two seconds.").await;
    assert_eq!(reply.trim(), "Reminder set.");
    client.disconnect().await;

    let daemon = TestDaemon::start_with(daemon.stop(), configure).await;
    let client = daemon.connect_client().await;
    let mut events = client.subscribe_session(session.clone());
    let delivered = tokio::time::timeout(READY_HANG_CEILING, async {
        loop {
            match events.recv().await {
                Ok(nanna_client::Event::SessionMessageAdded { content, .. }) => return content,
                Ok(_) => {}
                Err(e) => panic!("the event stream ended before the reminder: {e:?}"),
            }
        }
    })
    .await
    .expect("the reminder is delivered after the restart");
    assert!(delivered.contains("stretch your legs"), "{delivered:?}");

    let history = client
        .sessions()
        .history(&session, None)
        .await
        .expect("sessions.history answers")
        .to_string();
    assert!(
        history.contains("Reminder: stretch your legs"),
        "the delivery is persisted in the conversation: {history}"
    );

    client.disconnect().await;
    daemon.stop();
}

/// Memory with no embedder answering — this repo's own dev host, and any
/// install without an embedding provider. `remember` stores the memory whole
/// and says embeddings are degraded; `recall` used to answer "No memories
/// found matching" about it seconds later, because the search service turned
/// "cannot embed the query" into an empty list. It now falls back to a
/// keyword match and says that is what it did.
#[tokio::test]
async fn recall_without_an_embedder_finds_a_memory_by_keyword() {
    let ollama = ScriptedOllama::start(vec![
        r#"CALL remember {"content":"The user's cat is named Moonpie."}"#.to_string(),
        "Noted.\nTASK COMPLETE".to_string(),
        r#"CALL recall {"query":"what is my cat's name"}"#.to_string(),
        "Your cat is Moonpie.\nTASK COMPLETE".to_string(),
    ])
    .await;
    let host = ollama.base_url.clone();
    let daemon = TestDaemon::start_with(tempfile::tempdir().expect("temp dir"), move |b| {
        b.with_model(STUB_MODEL)
            .with_ollama_host(host)
            .with_scheduler(false)
            // The one test that wants memory: without an embedder, which is
            // the condition under test.
            .with_memory(true)
    })
    .await;
    let client = daemon.connect_client().await;
    let session = session_id_of(
        &client
            .sessions()
            .create(Some("memory".to_string()))
            .await
            .expect("sessions.create succeeds"),
    );
    converse(&client, &session, "My cat is named Moonpie. Remember that.").await;
    converse(&client, &session, "What's my cat's name?").await;

    let bodies = ollama.chat_bodies.lock().await.clone();
    let recalled = bodies
        .iter()
        .filter_map(|body| serde_json::from_str::<serde_json::Value>(body).ok())
        .flat_map(|request| request["messages"].as_array().cloned().unwrap_or_default())
        .filter(|message| message["role"] == "tool")
        .filter_map(|message| message["content"].as_str().map(str::to_string))
        .find(|content| content.contains("keyword") || content.contains("No memories found"))
        .expect("the recall result was sent back to the model");
    assert!(
        recalled.contains("Moonpie"),
        "the stored memory is found: {recalled:?}"
    );
    assert!(
        recalled.contains("by keyword match"),
        "and the result says how it was found: {recalled:?}"
    );

    client.disconnect().await;
    daemon.stop();
}

/// `ask_user` end to end — the one interruption the product wants: the model
/// asks a clarifying question mid-turn, the question is posted into the
/// conversation, the user's next message is handed to the waiting call as the
/// answer, and the turn finishes with it. The answer is consumed there: the
/// next turn does not work "Paris" again as a task of its own.
#[tokio::test]
async fn a_clarifying_question_is_answered_by_the_next_message() {
    let ollama = ScriptedOllama::start(vec![
        r#"CALL ask_user {"question":"Which city do you mean?","wait_secs":60}"#.to_string(),
        "Got it — Paris.\nTASK COMPLETE".to_string(),
        "You're welcome.\nTASK COMPLETE".to_string(),
    ])
    .await;
    let host = ollama.base_url.clone();
    let daemon = TestDaemon::start_with(tempfile::tempdir().expect("temp dir"), move |b| {
        b.with_model(STUB_MODEL)
            .with_ollama_host(host)
            .with_scheduler(false)
    })
    .await;
    let client = daemon.connect_client().await;
    let session = session_id_of(
        &client
            .sessions()
            .create(Some("clarify".to_string()))
            .await
            .expect("sessions.create succeeds"),
    );
    let mut events = client.subscribe_session(session.clone());
    let ack = client
        .chat()
        .send(&session, "Tell me about the city.")
        .await
        .expect("chat.send is accepted");
    let turn = ack["message_id"]
        .as_str()
        .unwrap_or_else(|| panic!("the ack names the turn's message: {ack}"))
        .to_string();

    let reply = tokio::time::timeout(READY_HANG_CEILING, async {
        let mut asked = false;
        loop {
            match events.recv().await {
                Ok(nanna_client::Event::SessionMessageAdded { content, .. })
                    if !asked && content.contains("Which city") =>
                {
                    asked = true;
                    client
                        .chat()
                        .send(&session, "Paris")
                        .await
                        .expect("the answer is accepted");
                }
                Ok(nanna_client::Event::MessageEnd {
                    message_id,
                    content,
                    ..
                }) if message_id == turn => return content,
                Ok(_) => {}
                Err(e) => panic!("the event stream ended before the turn did: {e:?}"),
            }
        }
    })
    .await
    .expect("the turn ends before the hang ceiling");
    assert_eq!(reply.trim(), "Got it — Paris.");

    let answered = ollama
        .chat_bodies
        .lock()
        .await
        .iter()
        .any(|body| body.contains("The user answered: Paris"));
    assert!(
        answered,
        "the waiting call received the reply as its result"
    );

    let before = ollama.chat_bodies.lock().await.len();
    let next = converse(&client, &session, "Thanks!").await;
    assert_eq!(next.trim(), "You're welcome.");
    // Every step the next turn ran, by the task line of its prompt.
    let tasks: Vec<String> = ollama.chat_bodies.lock().await[before..]
        .iter()
        .filter_map(|body| serde_json::from_str::<serde_json::Value>(body).ok())
        .filter_map(|request| {
            let prompt = request["messages"].as_array()?.last()?["content"]
                .as_str()?
                .to_string();
            prompt
                .lines()
                .find(|line| line.starts_with("Task #"))
                .map(str::to_string)
        })
        .collect();
    assert!(!tasks.is_empty(), "the next turn ran a step");
    assert!(
        tasks.iter().all(|task| !task.ends_with("Paris")),
        "the answer was consumed, not queued as new work: {tasks:?}"
    );

    client.disconnect().await;
    daemon.stop();
}

/// A tool call with a huge argument keeps the model's own call in context.
///
/// Only write tools had their bulk argument stripped from the stored turn;
/// here the scripted model passes 185 KB to `echo`, which sat verbatim in
/// the assistant message, pushed the estimate to ~56k tokens against a
/// 12k-token limit (the actual request was ~1.6k), and the hard cap dropped
/// the model's own tool call with a "history shortened" notice.
#[tokio::test]
async fn a_huge_tool_argument_does_not_evict_the_call_that_made_it() {
    let big = (0..6000).fold(String::new(), |mut acc, i| {
            use std::fmt::Write as _;
            writeln!(acc, "line {i}: the quick brown fox").expect("a String never refuses a write");
            acc
        });
    let call = format!("CALL echo {}", serde_json::json!({ "text": big }));
    let ollama = ScriptedOllama::start(vec![call, "Done.\nTASK COMPLETE".to_string()]).await;
    let host = ollama.base_url.clone();
    let daemon = TestDaemon::start_with(tempfile::tempdir().expect("temp dir"), move |b| {
        b.with_model(STUB_MODEL)
            .with_ollama_host(host)
            .with_scheduler(false)
    })
    .await;
    let client = daemon.connect_client().await;
    let session = session_id_of(
        &client
            .sessions()
            .create(Some("big argument".to_string()))
            .await
            .expect("sessions.create succeeds"),
    );
    let reply = converse(&client, &session, "Echo a lot.").await;
    assert_eq!(reply.trim(), "Done.");

    // The request after the call: it must still hold the model's own call.
    let after_call = ollama
        .chat_bodies
        .lock()
        .await
        .iter()
        .filter_map(|body| serde_json::from_str::<serde_json::Value>(body).ok())
        .find(|request| {
            request["messages"]
                .as_array()
                .is_some_and(|messages| messages.iter().any(|m| m["role"] == "tool"))
        })
        .expect("the tool result went back to the model");
    let messages = after_call["messages"].as_array().expect("messages");
    assert!(
        messages
            .iter()
            .any(|m| m["role"] == "assistant" && m["tool_calls"].is_array()),
        "the model's own tool call is still in context"
    );
    assert!(
        !after_call.to_string().contains("history shortened"),
        "nothing was dropped to make room"
    );

    client.disconnect().await;
    daemon.stop();
}

/// A model stuck repeating the same failing call — a classic small-model
/// loop — ends the turn by itself, and the report it leaves tells the truth
/// about how much ran: the abandoned item's step count and the turn's own
/// `N steps` line agree. The item's count used to be the no-progress CHARGE
/// counter (a repeated step is charged twice): "abandoned after 8 fruitless
/// steps" in a turn that ran 6.
#[tokio::test]
async fn a_repeated_failing_call_ends_bounded_and_reports_the_steps_it_ran() {
    let ollama = ScriptedOllama::start(vec![r#"CALL frobnicate {"x":1}"#.to_string()]).await;
    let host = ollama.base_url.clone();
    let daemon = TestDaemon::start_with(tempfile::tempdir().expect("temp dir"), move |b| {
        b.with_model(STUB_MODEL)
            .with_ollama_host(host)
            .with_scheduler(false)
    })
    .await;
    let client = daemon.connect_client().await;
    let session = session_id_of(
        &client
            .sessions()
            .create(Some("loop".to_string()))
            .await
            .expect("sessions.create succeeds"),
    );
    let reply = converse(&client, &session, "Frobnicate it.").await;
    assert!(
        reply.contains("could not finish"),
        "the failure is stated: {reply}"
    );

    let number_before = |marker: &str| -> Option<usize> {
        let end = reply.find(marker)?;
        // The line is italic markdown (`_6 steps · …_`): keep the digits.
        reply[..end]
            .split_whitespace()
            .last()?
            .trim_matches(|c: char| !c.is_ascii_digit())
            .parse()
            .ok()
    };
    let turn_steps = number_before(" steps · ").expect("the turn's step line");
    let item_steps = reply
        .split("abandoned after ")
        .nth(1)
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|n| n.parse::<usize>().ok())
        .expect("the abandoned item's step count");
    assert_eq!(item_steps, turn_steps, "one item, one count: {reply}");
    assert!(
        ollama.chat_bodies.lock().await.len() < 100,
        "the loop is bounded by the harness, not by the model"
    );
    assert!(
        !reply.contains("HARNESS NOTE") && !reply.contains("BREAKER"),
        "notices written for the model are not quoted to the user: {reply}"
    );

    client.disconnect().await;
    daemon.stop();
}

/// The repeat-completion escalation is for runs that ACT: a request re-sent
/// after a run that called tools, changed nothing, and declared itself done
/// again is exactly what it exists to call out. A conversational answer
/// completes with no side effects every time by design — asking the same
/// question twice, or pressing Regenerate, used to append "⚠️ repeat
/// completion … If you expected something to exist by now, it does not" to a
/// plain answer.
#[tokio::test]
async fn only_a_run_that_acted_is_told_its_repeat_changed_nothing() {
    async fn twice(steps: Vec<String>, request: &str) -> String {
        let ollama = ScriptedOllama::start(steps).await;
        let host = ollama.base_url.clone();
        let daemon = TestDaemon::start_with(tempfile::tempdir().expect("temp dir"), move |b| {
            b.with_model(STUB_MODEL)
                .with_ollama_host(host)
                .with_scheduler(false)
        })
        .await;
        let client = daemon.connect_client().await;
        let session = session_id_of(
            &client
                .sessions()
                .create(Some("repeat".to_string()))
                .await
                .expect("sessions.create succeeds"),
        );
        converse(&client, &session, request).await;
        let second = converse(&client, &session, request).await;
        client.disconnect().await;
        daemon.stop();
        second
    }

    let conversation = twice(
        vec!["Paris.\nTASK COMPLETE".to_string()],
        "What is the capital of France?",
    )
    .await;
    assert!(
        !conversation.contains("repeat completion"),
        "a repeated question is answered, not warned about: {conversation:?}"
    );

    // Calls a tool (it acted) but has no side effect — the shape of a
    // mission re-sent while nothing lands.
    let mission = twice(
        vec![
            r#"CALL echo {"text":"checked"}"#.to_string(),
            "All done.\nTASK COMPLETE".to_string(),
            r#"CALL echo {"text":"checked"}"#.to_string(),
            "All done.\nTASK COMPLETE".to_string(),
        ],
        "Make sure the build is done.",
    )
    .await;
    assert!(
        mission.contains("repeat completion"),
        "a run that acted and changed nothing is still told so: {mission:?}"
    );
}

/// Open tasks in a session's scope.
async fn open_tasks(client: &Client, session: &str) -> Vec<serde_json::Value> {
    let listed = client
        .request(nanna_client::Action::Task(nanna_client::TaskAction::List {
            scope: Some("session".to_string()),
            session_id: Some(session.to_string()),
            include_closed: Some(false),
        }))
        .await
        .expect("tasks.list answers");
    listed["tasks"]
        .as_array()
        .or_else(|| listed.as_array())
        .cloned()
        .unwrap_or_else(|| panic!("a task list: {listed}"))
}

/// Re-sending a request whose turn was stopped ADOPTS the item the stopped
/// turn left open — the planner proposed the same work again — instead of
/// creating a duplicate beside it: the work runs once and nothing is left
/// open. (A different next message leaves it open and unworked: see
/// `stop_ends_an_in_flight_turn_and_the_session_carries_on`.)
#[tokio::test]
async fn re_sending_a_stopped_request_adopts_its_open_item() {
    let ollama = ScriptedOllama::start(vec![
        "WAIT 3000 Too late.\nTASK COMPLETE".to_string(),
        "Resumed answer.\nTASK COMPLETE".to_string(),
    ])
    .await;
    let host = ollama.base_url.clone();
    let daemon = TestDaemon::start_with(tempfile::tempdir().expect("temp dir"), move |b| {
        b.with_model(STUB_MODEL)
            .with_ollama_host(host)
            .with_scheduler(false)
    })
    .await;
    let client = daemon.connect_client().await;
    let session = session_id_of(
        &client
            .sessions()
            .create(Some("adopt".to_string()))
            .await
            .expect("sessions.create succeeds"),
    );
    let mut events = client.subscribe_session(session.clone());
    let ack = client
        .chat()
        .send(&session, "What is the capital of France?")
        .await
        .expect("chat.send is accepted");
    let first = ack["message_id"].as_str().unwrap_or_default().to_string();
    tokio::time::timeout(READY_HANG_CEILING, async {
        while ollama.chat_bodies.lock().await.len() < 2 {
            tokio::time::sleep(READY_POLL_INTERVAL).await;
        }
    })
    .await
    .expect("the first step reaches the model");
    client
        .chat()
        .cancel(&session)
        .await
        .expect("chat.cancel answers");
    tokio::time::timeout(READY_HANG_CEILING, async {
        loop {
            if let Ok(nanna_client::Event::MessageEnd { message_id, .. }) = events.recv().await
                && message_id == first
            {
                return;
            }
        }
    })
    .await
    .expect("the stopped turn ends");
    assert_eq!(
        open_tasks(&client, &session).await.len(),
        1,
        "the stopped request's item is left open"
    );

    let reply = converse(&client, &session, "What is the capital of France?").await;
    assert_eq!(reply.trim(), "Resumed answer.", "worked once");
    assert!(
        open_tasks(&client, &session).await.is_empty(),
        "the adopted item was closed, not duplicated and left behind"
    );

    client.disconnect().await;
    daemon.stop();
}

/// An attached image reaches the model with the question; an attachment it
/// cannot read is named in the request so the model can say so. The harness
/// path used to drop every attachment with only a daemon-log warning — "what
/// is in this picture?" arrived as the bare question.
#[tokio::test]
async fn an_attached_image_reaches_the_model_and_an_unreadable_file_is_named() {
    // A valid 1x1 PNG: the loop resizes images for the model, so it must decode.
    const PNG: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";
    let ollama = ScriptedOllama::start(vec!["A single pixel.\nTASK COMPLETE".to_string()]).await;
    let host = ollama.base_url.clone();
    let daemon = TestDaemon::start_with(tempfile::tempdir().expect("temp dir"), move |b| {
        b.with_model(STUB_MODEL)
            .with_ollama_host(host)
            .with_scheduler(false)
    })
    .await;
    let client = daemon.connect_client().await;
    let session = session_id_of(
        &client
            .sessions()
            .create(Some("attachments".to_string()))
            .await
            .expect("sessions.create succeeds"),
    );
    let mut events = client.subscribe_session(session.clone());
    let ack = client
        .request(nanna_client::Action::Chat(nanna_client::ChatAction::Send {
            session_id: session.clone(),
            content: "What is in this picture?".to_string(),
            attachments: vec![
                nanna_client::Attachment {
                    filename: "pixel.png".to_string(),
                    content_type: "image/png".to_string(),
                    data: PNG.to_string(),
                },
                nanna_client::Attachment {
                    filename: "report.pdf".to_string(),
                    content_type: "application/pdf".to_string(),
                    data: "JVBERi0xLjQK".to_string(),
                },
            ],
        }))
        .await
        .expect("chat.send is accepted");
    let turn = ack["message_id"].as_str().unwrap_or_default().to_string();
    tokio::time::timeout(READY_HANG_CEILING, async {
        loop {
            if let Ok(nanna_client::Event::MessageEnd { message_id, .. }) = events.recv().await
                && message_id == turn
            {
                return;
            }
        }
    })
    .await
    .expect("the turn ends");

    let bodies = ollama.chat_bodies.lock().await.clone();
    let step = bodies
        .iter()
        .find(|body| !body.contains(PLANNER_PROMPT_OPENING))
        .expect("a step reached the model");
    let request: serde_json::Value = serde_json::from_str(step).expect("json");
    let images: Vec<&str> = request["messages"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|m| m["images"].as_array())
        .flatten()
        .filter_map(serde_json::Value::as_str)
        .collect();
    assert_eq!(images.len(), 1, "the image rides the step: {images:?}");
    assert!(
        step.contains("report.pdf (application/pdf)") && step.contains("cannot be read"),
        "the unreadable file is named for the model"
    );

    client.disconnect().await;
    daemon.stop();
}

/// Interjection end to end: a message sent while a turn is still working is
/// admitted into THAT turn at its next step boundary — acknowledged as
/// interjected, worked, and answered in the same reply — instead of waiting
/// for the run to end or starting a competing one.
#[tokio::test]
async fn a_message_sent_mid_turn_joins_the_running_turn() {
    let ollama = ScriptedOllama::start(vec![
        "WAIT 1500 First answer.\nTASK COMPLETE".to_string(),
        "Answer to the interjection.\nTASK COMPLETE".to_string(),
    ])
    .await;
    let host = ollama.base_url.clone();
    let daemon = TestDaemon::start_with(tempfile::tempdir().expect("temp dir"), move |b| {
        b.with_model(STUB_MODEL)
            .with_ollama_host(host)
            .with_scheduler(false)
    })
    .await;
    let client = daemon.connect_client().await;
    let session = session_id_of(
        &client
            .sessions()
            .create(Some("interject".to_string()))
            .await
            .expect("sessions.create succeeds"),
    );
    let mut events = client.subscribe_session(session.clone());
    let ack = client
        .chat()
        .send(&session, "first question")
        .await
        .expect("chat.send is accepted");
    let turn = ack["message_id"].as_str().unwrap_or_default().to_string();
    tokio::time::timeout(READY_HANG_CEILING, async {
        while ollama.chat_bodies.lock().await.len() < 2 {
            tokio::time::sleep(READY_POLL_INTERVAL).await;
        }
    })
    .await
    .expect("the first step is in flight");
    let joined = client
        .chat()
        .send(&session, "also, second question")
        .await
        .expect("chat.send is accepted");
    assert_eq!(joined["status"], "interjected", "{joined}");

    let reply = tokio::time::timeout(READY_HANG_CEILING, async {
        loop {
            match events.recv().await {
                Ok(nanna_client::Event::MessageEnd {
                    message_id,
                    content,
                    ..
                }) if message_id == turn => return content,
                Ok(_) => {}
                Err(e) => panic!("the event stream ended before the turn did: {e:?}"),
            }
        }
    })
    .await
    .expect("the turn ends");
    assert!(
        reply.starts_with("First answer.\n\nAnswer to the interjection."),
        "both answered, in order, in the one reply: {reply:?}"
    );

    client.disconnect().await;
    daemon.stop();
}

/// Two sessions' turns in flight at once stay apart: each reply answers its
/// own question and lands in its own history. Overlapping turns are the
/// shape behind the shared-workdir incident (one process-wide tool cwd let a
/// second chat re-root a live turn into another project), and the reason the
/// daemon's log lines now carry the session they came from.
#[tokio::test]
async fn overlapping_turns_in_two_sessions_stay_apart() {
    // Each reply names the goal of the step that produced it; the delay
    // keeps both turns in flight together.
    let ollama = ScriptedOllama::start(vec![
        "WAIT 400 Answer to: {goal}\nTASK COMPLETE".to_string(),
    ])
    .await;
    let host = ollama.base_url.clone();
    let daemon = TestDaemon::start_with(tempfile::tempdir().expect("temp dir"), move |b| {
        b.with_model(STUB_MODEL)
            .with_ollama_host(host)
            .with_scheduler(false)
    })
    .await;
    let client = daemon.connect_client().await;
    let mut sessions = Vec::new();
    for name in ["alpha", "beta"] {
        sessions.push(session_id_of(
            &client
                .sessions()
                .create(Some(name.to_string()))
                .await
                .expect("sessions.create succeeds"),
        ));
    }
    let (alpha, beta) = tokio::join!(
        converse(&client, &sessions[0], "alpha question"),
        converse(&client, &sessions[1], "beta question"),
    );
    assert_eq!(alpha.trim(), "Answer to: alpha question");
    assert_eq!(beta.trim(), "Answer to: beta question");

    for (session, own, other) in [
        (&sessions[0], "alpha", "beta"),
        (&sessions[1], "beta", "alpha"),
    ] {
        let history = client
            .sessions()
            .history(session, None)
            .await
            .expect("sessions.history answers")
            .to_string();
        assert!(
            history.contains(&format!("Answer to: {own} question")),
            "{history}"
        );
        assert!(
            !history.contains(other),
            "nothing of the other session leaked in: {history}"
        );
    }

    client.disconnect().await;
    daemon.stop();
}

/// A mission-shaped turn end to end: the plan's task carries a machine check
/// (`file_exists`), the scripted model writes the file with the real
/// `write_file` tool inside a workspace, the check verifies it, and nothing is
/// left open. A follow-up turn's planner is then told this item closed with
/// its done-condition passing — the verified counterpart of
/// `a_follow_up_turn_is_not_told_an_unchecked_answer_passed_a_check`.
#[tokio::test]
async fn a_checked_task_is_verified_by_the_environment_and_reported_as_passing() {
    let project = tempfile::tempdir().expect("temp project dir");
    let notes = project.path().join("notes.txt");
    let notes_path = notes.to_str().expect("a UTF-8 temp path").to_string();
    let plan = serde_json::json!([{
        "title": "Write the notes file",
        "description": "Create notes.txt",
        "acceptance": { "kind": "file_exists", "path": notes_path },
    }])
    .to_string();
    let write = format!(
        "CALL write_file {}",
        serde_json::json!({ "path": notes_path, "content": "hello notes\n" })
    );
    let ollama =
        ScriptedOllama::start_with_plan(&plan, vec![write, "Wrote the notes file.".to_string()])
            .await;
    let host = ollama.base_url.clone();
    let daemon = TestDaemon::start_with(tempfile::tempdir().expect("temp dir"), move |b| {
        b.with_model(STUB_MODEL)
            .with_ollama_host(host)
            .with_scheduler(false)
    })
    .await;
    let client = daemon.connect_client().await;
    // A workspace roots the file tools (and their bookkeeping) in the temp
    // project, never in the test's cwd or HOME.
    let opened = client
        .workspaces()
        .open(project.path().to_str().expect("a UTF-8 temp path"))
        .await
        .expect("workspace.open answers");
    let workspace_id = opened["id"].as_str().expect("a workspace id").to_string();
    let session = session_id_of(
        &client
            .request(nanna_client::Action::Session(
                nanna_client::SessionAction::CreateInWorkspace {
                    name: Some("mission".to_string()),
                    workspace_id: Some(workspace_id),
                },
            ))
            .await
            .expect("sessions.create_in_workspace answers"),
    );

    let reply = converse(&client, &session, "Write a notes file.").await;
    assert_eq!(reply.trim(), "Wrote the notes file.");
    assert_eq!(
        std::fs::read_to_string(&notes).ok().as_deref(),
        Some("hello notes\n"),
        "the tool really wrote the file"
    );
    assert!(
        open_tasks(&client, &session).await.is_empty(),
        "the check verified the task and closed it"
    );

    converse(&client, &session, "Is it done?").await;
    let follow_up_plan = ollama
        .chat_bodies
        .lock()
        .await
        .iter()
        .filter(|body| body.contains(PLANNER_PROMPT_OPENING))
        .nth(1)
        .cloned()
        .expect("the follow-up was planned");
    // The turn start re-runs the check and says so: the strongest form of
    // "this passed", and the opposite of the unverified block.
    assert!(
        follow_up_plan.contains("already PASS, verified by running them")
            && follow_up_plan.contains("Write the notes file"),
        "a verified item is reported as a passing done-condition"
    );
    assert!(
        !follow_up_plan.contains("no check ran"),
        "and not as the model's word"
    );

    client.disconnect().await;
    daemon.stop();
}

/// Deleting a session stops its running turn, as Stop would. It used to
/// leave the turn running: the model kept generating — and a mission would
/// have kept calling tools — for a conversation that no longer existed.
#[tokio::test]
async fn deleting_a_session_stops_its_running_turn() {
    let ollama = ScriptedOllama::start(vec![
        "WAIT 1500 Late answer.\nTASK COMPLETE".to_string(),
        "Other answer.\nTASK COMPLETE".to_string(),
    ])
    .await;
    let host = ollama.base_url.clone();
    let daemon = TestDaemon::start_with(tempfile::tempdir().expect("temp dir"), move |b| {
        b.with_model(STUB_MODEL)
            .with_ollama_host(host)
            .with_scheduler(false)
    })
    .await;
    let client = daemon.connect_client().await;
    let doomed = session_id_of(
        &client
            .sessions()
            .create(Some("doomed".to_string()))
            .await
            .expect("sessions.create succeeds"),
    );
    let mut events = client.subscribe_session(doomed.clone());
    client
        .chat()
        .send(&doomed, "question")
        .await
        .expect("chat.send is accepted");
    tokio::time::timeout(READY_HANG_CEILING, async {
        while ollama.chat_bodies.lock().await.len() < 2 {
            tokio::time::sleep(READY_POLL_INTERVAL).await;
        }
    })
    .await
    .expect("the step is in flight");
    client
        .sessions()
        .delete(&doomed)
        .await
        .expect("sessions.delete answers");

    let ended = tokio::time::timeout(READY_HANG_CEILING, async {
        loop {
            match events.recv().await {
                Ok(nanna_client::Event::MessageDelta { delta, .. })
                    if delta.contains("Late answer") =>
                {
                    panic!("the deleted session's turn kept generating: {delta:?}")
                }
                Ok(nanna_client::Event::MessageEnd { content, .. }) => return content,
                Ok(_) => {}
                Err(e) => panic!("the event stream ended before the turn did: {e:?}"),
            }
        }
    })
    .await
    .expect("the turn ends");
    assert!(!ended.contains("Late answer"), "{ended:?}");

    // And the daemon is fine: another session answers normally.
    let next = session_id_of(
        &client
            .sessions()
            .create(Some("next".to_string()))
            .await
            .expect("sessions.create succeeds"),
    );
    assert_eq!(
        converse(&client, &next, "hello").await.trim(),
        "Other answer."
    );

    client.disconnect().await;
    daemon.stop();
}

/// Clearing a conversation while a turn is running in it is refused, as the
/// chat apps' `/new` already was — otherwise the running turn's reply would
/// land in the freshly cleared conversation. Once the turn ends, clearing
/// works.
#[tokio::test]
async fn clearing_a_session_mid_turn_is_refused_until_the_turn_ends() {
    let ollama =
        ScriptedOllama::start(vec!["WAIT 800 An answer.\nTASK COMPLETE".to_string()]).await;
    let host = ollama.base_url.clone();
    let daemon = TestDaemon::start_with(tempfile::tempdir().expect("temp dir"), move |b| {
        b.with_model(STUB_MODEL)
            .with_ollama_host(host)
            .with_scheduler(false)
    })
    .await;
    let client = daemon.connect_client().await;
    let session = session_id_of(
        &client
            .sessions()
            .create(Some("clear".to_string()))
            .await
            .expect("sessions.create succeeds"),
    );
    let mut events = client.subscribe_session(session.clone());
    let ack = client
        .chat()
        .send(&session, "question")
        .await
        .expect("chat.send is accepted");
    let turn = ack["message_id"].as_str().unwrap_or_default().to_string();
    tokio::time::timeout(READY_HANG_CEILING, async {
        while ollama.chat_bodies.lock().await.len() < 2 {
            tokio::time::sleep(READY_POLL_INTERVAL).await;
        }
    })
    .await
    .expect("the step is in flight");

    let refused = client
        .sessions()
        .clear(&session)
        .await
        .expect("sessions.clear answers");
    assert_eq!(refused["error"], "busy", "{refused}");

    tokio::time::timeout(READY_HANG_CEILING, async {
        loop {
            if let Ok(nanna_client::Event::MessageEnd { message_id, .. }) = events.recv().await
                && message_id == turn
            {
                return;
            }
        }
    })
    .await
    .expect("the turn ends");
    // `message_end` goes out before the turn releases its run claim, so the
    // first clear after it may still be told "busy": poll, bounded.
    let cleared = tokio::time::timeout(READY_HANG_CEILING, async {
        loop {
            let answer = client
                .sessions()
                .clear(&session)
                .await
                .expect("sessions.clear answers");
            if answer["error"] != "busy" {
                return answer;
            }
            tokio::time::sleep(READY_POLL_INTERVAL).await;
        }
    })
    .await
    .expect("the claim is released after the turn");
    assert_eq!(cleared["status"], "cleared", "{cleared}");

    client.disconnect().await;
    daemon.stop();
}

/// A blank message is refused, not run. It used to be persisted and planned
/// as a whole turn — the model answering a request nobody made.
#[tokio::test]
async fn a_blank_message_is_refused_and_starts_no_turn() {
    let ollama = ScriptedOllama::start(vec!["Hello there.\nTASK COMPLETE".to_string()]).await;
    let host = ollama.base_url.clone();
    let daemon = TestDaemon::start_with(tempfile::tempdir().unwrap(), move |b| {
        b.with_model(STUB_MODEL)
            .with_ollama_host(host)
            .with_scheduler(false)
    })
    .await;
    let client = daemon.connect_client().await;
    let session = session_id_of(
        &client
            .sessions()
            .create(Some("blank".into()))
            .await
            .unwrap(),
    );

    let refused = client.chat().send(&session, " \n\t ").await.unwrap();
    assert_eq!(refused["error"], "empty_message", "{refused}");
    let history = client.sessions().history(&session, None).await.unwrap();
    assert_eq!(
        history["messages"].as_array().map_or(0, Vec::len),
        0,
        "nothing persisted: {history}"
    );

    assert_eq!(
        converse(&client, &session, "hi").await,
        "Hello there.",
        "the scripted reply was not spent on the blank message"
    );

    client.disconnect().await;
    daemon.stop();
}

/// A pasted log far longer than the model's window fails honestly. It used
/// to blame "`num_ctx` was demoted under GPU memory pressure … free GPU memory
/// and resume" — a demotion that never happened, and a remedy that cannot
/// help: the request itself does not fit.
#[tokio::test]
async fn a_request_longer_than_the_window_says_so() {
    let ollama = ScriptedOllama::start(vec!["Summarized.\nTASK COMPLETE".to_string()]).await;
    let host = ollama.base_url.clone();
    let daemon = TestDaemon::start_with(tempfile::tempdir().unwrap(), move |b| {
        b.with_model(STUB_MODEL)
            .with_ollama_host(host)
            .with_scheduler(false)
    })
    .await;
    let client = daemon.connect_client().await;
    let session = session_id_of(
        &client
            .sessions()
            .create(Some("paste".into()))
            .await
            .unwrap(),
    );
    let log_line = "2026-09-21 ERROR worker 7 failed to connect to db at 10.0.0.5:5432\n";
    let request = format!("Summarize this log:\n{}", log_line.repeat(4_000));

    let reply = converse(&client, &session, &request).await;
    assert!(
        reply.contains("request itself is longer than this window"),
        "{reply}"
    );
    assert!(
        !reply.contains("GPU memory"),
        "no demotion happened: {reply}"
    );

    client.disconnect().await;
    daemon.stop();
}

/// A sub-session that overruns its timeout is cancelled and wound down, not
/// dropped mid-flight. Dropped, its chat's `active_chats` entry was never
/// released: the daemon kept reporting the long-dead run as live — `chat.cancel`
/// on it answered `cancelled` forever — and the agent service read as busy.
#[tokio::test]
async fn a_timed_out_sub_session_leaves_no_live_run_behind() {
    use nanna_daemon::protocol::{Action, SessionAction};

    // Every model reply is held back far past the sub-session's 1 s timeout.
    let ollama =
        ScriptedOllama::start(vec!["WAIT 20000 Too late.\nTASK COMPLETE".to_string()]).await;
    let host = ollama.base_url.clone();
    let daemon = TestDaemon::start_with(tempfile::tempdir().expect("temp dir"), move |b| {
        b.with_model(STUB_MODEL)
            .with_ollama_host(host)
            .with_scheduler(false)
    })
    .await;
    let client = daemon.connect_client().await;
    let spawned = client
        .request(Action::Session(SessionAction::SpawnSubSession {
            task: "take your time".to_string(),
            label: Some("slow".to_string()),
            parent_id: None,
            model: None,
            max_iterations: None,
            timeout_secs: Some(1),
            system_prompt: None,
        }))
        .await
        .expect("sessions.spawn_sub_session answers");
    assert!(spawned.get("error").is_none(), "{spawned}");

    let status = tokio::time::timeout(READY_HANG_CEILING, async {
        loop {
            let status = client
                .request(Action::Session(SessionAction::GetSubSessionStatus {
                    target: "slow".to_string(),
                }))
                .await
                .expect("status answers");
            if status["error"].is_string() {
                return status;
            }
            tokio::time::sleep(READY_POLL_INTERVAL).await;
        }
    })
    .await
    .expect("the sub-session times out");
    assert!(
        status["error"]
            .as_str()
            .is_some_and(|e| e.contains("timed out")),
        "{status}"
    );
    let sub = status["session_id"].as_str().expect("an id").to_string();

    let cancel = client
        .chat()
        .cancel(&sub)
        .await
        .expect("chat.cancel answers");
    assert_eq!(
        cancel["status"], "not_active",
        "the timed-out run is gone: {cancel}"
    );
}

/// Killing a sub-session stops its run, and it stays killed. Kill used to set
/// a flag nothing read: the sub-agent ran on to the end, and its finish then
/// overwrote `killed` with `completed`.
#[tokio::test]
async fn a_killed_sub_session_stops_and_stays_killed() {
    use nanna_daemon::protocol::{Action, SessionAction};

    let ollama = ScriptedOllama::start(vec![
        "WAIT 20000 Finished anyway.\nTASK COMPLETE".to_string(),
    ])
    .await;
    let host = ollama.base_url.clone();
    let daemon = TestDaemon::start_with(tempfile::tempdir().expect("temp dir"), move |b| {
        b.with_model(STUB_MODEL)
            .with_ollama_host(host)
            .with_scheduler(false)
    })
    .await;
    let client = daemon.connect_client().await;
    client
        .request(Action::Session(SessionAction::SpawnSubSession {
            task: "work slowly".to_string(),
            label: Some("doomed".to_string()),
            parent_id: None,
            model: None,
            max_iterations: None,
            timeout_secs: None,
            system_prompt: None,
        }))
        .await
        .expect("spawn answers");
    // Kill once the run is really in flight: the stub has seen its request.
    tokio::time::timeout(READY_HANG_CEILING, async {
        while ollama.chat_bodies.lock().await.is_empty() {
            tokio::time::sleep(READY_POLL_INTERVAL).await;
        }
    })
    .await
    .expect("the sub-agent reaches the model");
    let killed = client
        .request(Action::Session(SessionAction::KillSubSession {
            target: "doomed".to_string(),
        }))
        .await
        .expect("kill answers");
    assert_eq!(killed["status"], "killed", "{killed}");

    // Without touching the run otherwise, it winds down well before the
    // stub's 20 s reply: the kill itself stopped it.
    let after = tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            let status = doomed_status(&client).await;
            if status["result"].is_string() || status["error"].is_string() {
                return status;
            }
            tokio::time::sleep(READY_POLL_INTERVAL).await;
        }
    })
    .await
    .expect("the killed run stopped instead of running on");
    assert!(
        !after.to_string().contains("Finished anyway"),
        "the run did not finish its work: {after}"
    );
    assert_eq!(
        after["state"], "killed",
        "still killed after the run ended: {after}"
    );
}

/// `sessions.get_sub_session_status` for the sub-session labelled `doomed`.
async fn doomed_status(client: &Client) -> serde_json::Value {
    client
        .request(nanna_daemon::protocol::Action::Session(
            nanna_daemon::protocol::SessionAction::GetSubSessionStatus {
                target: "doomed".to_string(),
            },
        ))
        .await
        .expect("status answers")
}

/// A fork's copied conversation is stored like any other: it survives a
/// restart, and the original keeps its own. The fork used to be written through
/// `update`, which stores only the session row — its messages existed in memory
/// alone and came back empty after a restart.
#[tokio::test]
async fn a_forked_conversation_survives_a_restart() {
    use nanna_daemon::protocol::{Action, SessionAction};

    let ollama =
        ScriptedOllama::start(vec!["The original answer.\nTASK COMPLETE".to_string()]).await;
    let host = ollama.base_url.clone();
    let daemon = TestDaemon::start_with(tempfile::tempdir().expect("temp dir"), {
        let host = host.clone();
        move |b| {
            b.with_model(STUB_MODEL)
                .with_ollama_host(host)
                .with_scheduler(false)
        }
    })
    .await;
    let client = daemon.connect_client().await;
    let original = session_id_of(
        &client
            .sessions()
            .create(Some("original".to_string()))
            .await
            .expect("create"),
    );
    converse(&client, &original, "ask something").await;
    let count = |history: &serde_json::Value| history["messages"].as_array().map_or(0, Vec::len);
    let before = count(
        &client
            .sessions()
            .history(&original, None)
            .await
            .expect("history"),
    );
    assert!(before >= 2, "a question and an answer");

    let forked = client
        .request(Action::Session(SessionAction::Fork {
            id: original.clone(),
            name: Some("the fork".to_string()),
        }))
        .await
        .expect("fork answers");
    let fork_id = forked["session"]["id"]
        .as_str()
        .expect("the fork's id")
        .to_string();

    client.disconnect().await;
    let restarted = TestDaemon::start_with(daemon.stop(), move |b| {
        b.with_model(STUB_MODEL)
            .with_ollama_host(host)
            .with_scheduler(false)
    })
    .await;
    let client = restarted.connect_client().await;
    let fork_after = count(
        &client
            .sessions()
            .history(&fork_id, None)
            .await
            .expect("history"),
    );
    let original_after = count(
        &client
            .sessions()
            .history(&original, None)
            .await
            .expect("history"),
    );
    assert_eq!(fork_after, before, "the fork's messages were stored");
    assert_eq!(original_after, before, "the original is untouched");
    client.disconnect().await;
    restarted.stop();
}

/// Regenerate's dropped reply leaves the store too: after a restart the
/// conversation holds the question and the NEW answer only. It was dropped
/// through `update`, which never deleted the old rows, so the stale answer
/// came back from the store on the next start.
#[tokio::test]
async fn a_regenerated_reply_stays_replaced_after_a_restart() {
    use nanna_daemon::protocol::{Action, ChatAction};

    let ollama = ScriptedOllama::start(vec![
        "The first answer.\nTASK COMPLETE".to_string(),
        "The second answer.\nTASK COMPLETE".to_string(),
    ])
    .await;
    let host = ollama.base_url.clone();
    let daemon = TestDaemon::start_with(tempfile::tempdir().expect("temp dir"), {
        let host = host.clone();
        move |b| {
            b.with_model(STUB_MODEL)
                .with_ollama_host(host)
                .with_scheduler(false)
        }
    })
    .await;
    let client = daemon.connect_client().await;
    let session = session_id_of(
        &client
            .sessions()
            .create(Some("regen".to_string()))
            .await
            .expect("create"),
    );
    converse(&client, &session, "ask once").await;
    let mut events = client.subscribe_session(session.clone());
    client
        .request(Action::Chat(ChatAction::Regenerate {
            session_id: session.clone(),
        }))
        .await
        .expect("regenerate answers");
    tokio::time::timeout(READY_HANG_CEILING, async {
        loop {
            if let Ok(nanna_client::Event::MessageEnd { .. }) = events.recv().await {
                return;
            }
        }
    })
    .await
    .expect("the regenerated turn ends");
    let live = client
        .sessions()
        .history(&session, None)
        .await
        .expect("history");

    client.disconnect().await;
    let restarted = TestDaemon::start_with(daemon.stop(), move |b| {
        b.with_model(STUB_MODEL)
            .with_ollama_host(host)
            .with_scheduler(false)
    })
    .await;
    let client = restarted.connect_client().await;
    let stored = client
        .sessions()
        .history(&session, None)
        .await
        .expect("history");
    let texts = |h: &serde_json::Value| -> Vec<String> {
        h["messages"]
            .as_array()
            .map(|m| {
                m.iter()
                    .filter_map(|m| m["content"].as_str().map(str::to_owned))
                    .collect()
            })
            .unwrap_or_default()
    };
    assert_eq!(
        texts(&stored),
        texts(&live),
        "the store matches what the client saw"
    );
    assert!(
        !texts(&stored).iter().any(|t| t.contains("first answer")),
        "the replaced reply stayed gone: {stored}"
    );
    client.disconnect().await;
    restarted.stop();
}
