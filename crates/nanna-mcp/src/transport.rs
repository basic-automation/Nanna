//! Transport layer for MCP communication
//!
//! Supports:
//! - stdio: Spawn a process and communicate via stdin/stdout
//! - HTTP/SSE: Connect to an HTTP server with Server-Sent Events

#[cfg(any(feature = "stdio", feature = "http"))]
use crate::McpError;
use crate::{JsonRpcNotification, JsonRpcRequest, JsonRpcResponse, Result};
use async_trait::async_trait;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(any(feature = "stdio", feature = "http"))]
use tokio::sync::Mutex;

/// Which MCP list a `.../list_changed` notification refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum McpList {
    /// The server's tool list.
    Tools,
    /// The server's resource list.
    Resources,
    /// The server's prompt list.
    Prompts,
}

/// Transport-agnostic "the server's list changed" flags, shared via `Arc`.
///
/// A transport that receives `notifications/{tools,resources,prompts}/list_changed`
/// marks the matching flag; the client consumes (and clears) it to lazily refresh
/// its cache on the next `list_*` call.
#[derive(Debug, Default)]
pub struct ListChangedFlags {
    tools: AtomicBool,
    resources: AtomicBool,
    prompts: AtomicBool,
    /// Wakes whoever waits in [`Self::changed`]. `notify_one` stores a permit
    /// when nobody is waiting, so a mark between two waits is never lost.
    wake: tokio::sync::Notify,
}

impl ListChangedFlags {
    /// Mark a list dirty (called by the transport on a `list_changed` notification).
    pub fn mark(&self, list: McpList) {
        self.flag(list).store(true, Ordering::Release);
        self.wake.notify_one();
    }

    /// Resolve once any list has been marked since the last call (or at once,
    /// if one was marked while nobody waited).
    pub async fn changed(&self) {
        self.wake.notified().await;
    }

    /// Atomically read-and-clear a list's dirty flag. Returns whether it was set.
    pub fn take(&self, list: McpList) -> bool {
        self.flag(list).swap(false, Ordering::AcqRel)
    }

    const fn flag(&self, list: McpList) -> &AtomicBool {
        match list {
            McpList::Tools => &self.tools,
            McpList::Resources => &self.resources,
            McpList::Prompts => &self.prompts,
        }
    }
}

/// Transport trait for MCP communication
#[async_trait]
pub trait Transport: Send + Sync {
    /// Send a request and wait for a response
    async fn request(&self, request: JsonRpcRequest) -> Result<JsonRpcResponse>;

    /// Send a notification (no response expected)
    async fn notify(&self, notification: JsonRpcNotification) -> Result<()>;

    /// Close the transport
    async fn close(&self) -> Result<()>;

    /// Shared list-changed flags this transport tracks, if any.
    ///
    /// Transports that receive server push notifications (stdio) return `Some`
    /// so the client can lazily refresh a stale cache. Request/response
    /// transports that never see `list_changed` return `None` (the default) and
    /// the client simply keeps serving its cache until an explicit refresh.
    fn list_changed_flags(&self) -> Option<Arc<ListChangedFlags>> {
        None
    }

    /// Open a `subscriptions/listen` stream (MCP 2026-07-28): a request whose
    /// answer is a long-lived stream of the change notifications it opted
    /// into, which the transport routes into its [`ListChangedFlags`]. It is
    /// never awaited like a normal request. The default does nothing — a
    /// transport without flags has nowhere to deliver them.
    ///
    /// # Errors
    ///
    /// Returns an error only if the request could not be sent at all.
    async fn open_listen(&self, request: JsonRpcRequest) -> Result<()> {
        let _ = request;
        Ok(())
    }
}

/// The reply to a server→client request. Pure.
///
/// `ping` MUST be answered with an empty result. Everything else — roots,
/// sampling, elicitation — is a capability this client never declares, so
/// it is refused with `-32601`. Unanswered, the server waits on it forever:
/// measured 2026-09-18, `@modelcontextprotocol/server-everything` holding an
/// unanswered `roots/list` did not exit on stdin EOF and outlived the daemon.
#[must_use]
pub fn reply_to_server_request(id: &serde_json::Value, method: &str) -> serde_json::Value {
    debug_assert!(!id.is_null(), "a request always has an id");
    if method == "ping" {
        return serde_json::json!({ "jsonrpc": "2.0", "id": id, "result": {} });
    }
    serde_json::json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": -32601, "message": format!("nanna does not serve `{method}`") },
    })
}

// ============================================================================
// Stdio Transport
// ============================================================================

#[cfg(feature = "stdio")]
pub mod stdio {
    use super::{
        Arc, JsonRpcNotification, JsonRpcRequest, JsonRpcResponse, ListChangedFlags, McpError,
        McpList, Mutex, Result, Transport, async_trait, reply_to_server_request,
    };
    use std::collections::HashMap;
    use std::process::Stdio;
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    use tokio::process::{Child, ChildStdin, ChildStdout, Command};
    use tokio::sync::{mpsc, oneshot};
    use tracing::{debug, error, trace, warn};

    /// Server → client notification categories from the MCP spec that we recognize.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum ServerNotification {
        /// `notifications/message` — a structured log record emitted by the server.
        LogMessage,
        /// `notifications/progress` — progress update for an in-flight request.
        Progress,
        /// A `.../list_changed` notification — a cache signal. Carries which list
        /// changed, or `None` if it's a `list_changed` we don't cache (e.g. roots).
        ListChanged(Option<McpList>),
        /// `notifications/cancelled` — the server cancelled a request.
        Cancelled,
        /// Anything not specifically handled.
        Other,
    }

    /// Map a `.../list_changed` method to the cached list it refers to. Pure.
    fn list_changed_kind(method: &str) -> Option<McpList> {
        match method {
            "notifications/tools/list_changed" => Some(McpList::Tools),
            "notifications/resources/list_changed" => Some(McpList::Resources),
            "notifications/prompts/list_changed" => Some(McpList::Prompts),
            _ => None,
        }
    }

    /// Classify an MCP notification by its JSON-RPC `method`. Pure.
    fn classify_server_notification(method: &str) -> ServerNotification {
        match method {
            "notifications/message" => ServerNotification::LogMessage,
            "notifications/progress" => ServerNotification::Progress,
            "notifications/cancelled" => ServerNotification::Cancelled,
            m if m.ends_with("/list_changed") => {
                ServerNotification::ListChanged(list_changed_kind(m))
            }
            _ => ServerNotification::Other,
        }
    }

    /// Whether an MCP log `level` (RFC 5424 severity keyword) is warning-or-worse
    /// and should surface at a higher tracing level. Pure.
    fn mcp_level_is_severe(level: &str) -> bool {
        matches!(
            level,
            "warning" | "error" | "critical" | "alert" | "emergency"
        )
    }

    /// Route a parsed server notification: log records go to tracing, and a
    /// `list_changed` marks the matching cache flag so the client refreshes lazily.
    fn handle_server_notification(notif: &JsonRpcNotification, list_changed: &ListChangedFlags) {
        match classify_server_notification(&notif.method) {
            ServerNotification::LogMessage => {
                let level = notif
                    .params
                    .as_ref()
                    .and_then(|p| p.get("level"))
                    .and_then(|l| l.as_str())
                    .unwrap_or("info");
                if mcp_level_is_severe(level) {
                    warn!(method = notif.method, level, "MCP server log");
                } else {
                    debug!(method = notif.method, level, "MCP server log");
                }
            }
            ServerNotification::ListChanged(kind) => {
                // A cached list changed server-side: mark it dirty so the next
                // client `list_*` call refreshes instead of serving a stale cache.
                if let Some(list) = kind {
                    list_changed.mark(list);
                    debug!(
                        method = notif.method,
                        "MCP list changed — cache marked dirty"
                    );
                } else {
                    debug!(method = notif.method, "MCP list changed (uncached list)");
                }
            }
            ServerNotification::Progress => {
                debug!(method = notif.method, "MCP progress notification");
            }
            ServerNotification::Cancelled => {
                debug!(method = notif.method, "MCP server cancelled a request");
            }
            ServerNotification::Other => {
                debug!(method = notif.method, "Unhandled MCP notification");
            }
        }
    }

    /// What one line from the server is, decided by JSON-RPC shape alone.
    ///
    /// The order matters: a server→client *request* (`id` + `method`) also
    /// deserializes as a [`JsonRpcResponse`] — `result` and `error` are both
    /// optional — so shape must be checked before the response parse, or a
    /// request is delivered to whichever pending client call shares its `id`.
    #[derive(Debug)]
    enum Incoming {
        Response(JsonRpcResponse),
        ServerRequest {
            id: serde_json::Value,
            method: String,
        },
        Notification(JsonRpcNotification),
        Unreadable(String),
    }

    /// Classify one line from the server. Pure.
    fn classify_incoming(line: &str) -> Incoming {
        let value: serde_json::Value = match serde_json::from_str(line) {
            Ok(value) => value,
            Err(error) => return Incoming::Unreadable(error.to_string()),
        };
        let method = value.get("method").and_then(serde_json::Value::as_str);
        let id = value.get("id").filter(|id| !id.is_null());
        match (method, id) {
            (Some(method), Some(id)) => Incoming::ServerRequest {
                id: id.clone(),
                method: method.to_string(),
            },
            (Some(_), None) => serde_json::from_value(value).map_or_else(
                |e| Incoming::Unreadable(e.to_string()),
                Incoming::Notification,
            ),
            (None, _) => serde_json::from_value(value)
                .map_or_else(|e| Incoming::Unreadable(e.to_string()), Incoming::Response),
        }
    }

    /// How long a stdio server gets to exit after its stdin closes before it
    /// is killed.
    ///
    /// The SDK servers exit in well under 100 ms (measured
    /// 2026-09-18); the grace is 20x that so a server flushing state is not
    /// cut off, and short enough that 16 servers closed concurrently keep the
    /// daemon's shutdown inside its existing 5 s stats-save window.
    pub const MCP_EXIT_GRACE: std::time::Duration = std::time::Duration::from_secs(2);

    /// Write one newline-delimited message. The lock is held across both
    /// writes and the flush so concurrent messages cannot interleave inside
    /// one line; a closed stdin is [`McpError::ConnectionClosed`].
    async fn write_line(stdin: &Mutex<Option<ChildStdin>>, line: &str) -> Result<()> {
        debug_assert!(
            !line.contains('\n'),
            "a stdio message must not embed a newline"
        );
        let mut guard = stdin.lock().await;
        let pipe = guard.as_mut().ok_or(McpError::ConnectionClosed)?;
        pipe.write_all(line.as_bytes()).await?;
        pipe.write_all(b"\n").await?;
        pipe.flush().await?;
        drop(guard);
        Ok(())
    }

    /// Stdio transport - spawns a process and communicates via stdin/stdout
    pub struct StdioTransport {
        /// Child process
        child: Arc<Mutex<Child>>,
        /// Stdin writer
        /// Stdin writer. `None` once `close` has dropped it — the EOF that is a
        /// stdio server's graceful-shutdown signal.
        stdin: Arc<Mutex<Option<ChildStdin>>>,
        /// Pending requests waiting for responses
        pending: Arc<Mutex<HashMap<String, oneshot::Sender<JsonRpcResponse>>>>,
        /// Shutdown signal
        shutdown_tx: mpsc::Sender<()>,
        /// Per-list "server said this changed" flags, set by the reader task and
        /// consumed by the client to refresh a stale cache lazily.
        list_changed: Arc<ListChangedFlags>,
    }

    impl StdioTransport {
        /// Spawn a new process and create a transport
        ///
        /// # Errors
        ///
        /// Returns error if process fails to spawn
        pub fn spawn(program: &str, args: &[&str]) -> Result<Self> {
            Self::spawn_with_env(program, args, &[])
        }

        /// Spawn with environment variables
        ///
        /// # Errors
        ///
        /// Returns error if process fails to spawn
        pub fn spawn_with_env(
            program: &str,
            args: &[&str],
            env: &[(&str, &str)],
        ) -> Result<Self> {
            debug!(program, ?args, "Spawning MCP server process");

            let mut cmd = Command::new(program);
            cmd.args(args)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::inherit()) // Let stderr pass through for debugging
                .kill_on_drop(true);

            for (key, value) in env {
                cmd.env(key, value);
            }

            let mut child = cmd.spawn()?;

            let stdin = child
                .stdin
                .take()
                .ok_or_else(|| McpError::Transport("Failed to open stdin".into()))?;

            let stdout = child
                .stdout
                .take()
                .ok_or_else(|| McpError::Transport("Failed to open stdout".into()))?;

            let pending: Arc<Mutex<HashMap<String, oneshot::Sender<JsonRpcResponse>>>> =
                Arc::new(Mutex::new(HashMap::new()));
            let (shutdown_tx, shutdown_rx) = mpsc::channel::<()>(1);
            let list_changed = Arc::new(ListChangedFlags::default());

            // Spawn reader task
            let pending_clone = pending.clone();
            let list_changed_clone = list_changed.clone();
            let stdin = Arc::new(Mutex::new(Some(stdin)));
            tokio::spawn(Self::reader_task(
                stdout,
                stdin.clone(),
                pending_clone,
                shutdown_rx,
                list_changed_clone,
            ));

            Ok(Self {
                child: Arc::new(Mutex::new(child)),
                stdin,
                pending,
                shutdown_tx,
                list_changed,
            })
        }

        /// Background task that reads the server's stdout: responses go to the
        /// waiting caller, notifications to the router, and server requests are
        /// answered on stdin.
        async fn reader_task(
            stdout: ChildStdout,
            stdin: Arc<Mutex<Option<ChildStdin>>>,
            pending: Arc<Mutex<HashMap<String, oneshot::Sender<JsonRpcResponse>>>>,
            mut shutdown_rx: mpsc::Receiver<()>,
            list_changed: Arc<ListChangedFlags>,
        ) {
            let mut reader = BufReader::new(stdout).lines();

            loop {
                tokio::select! {
                    _ = shutdown_rx.recv() => {
                        debug!("Reader task shutting down");
                        break;
                    }
                    line = reader.next_line() => {
                        match line {
                            Ok(Some(line)) => {
                                trace!(line, "Received from MCP server");
                                Self::route_line(&line, &stdin, &pending, &list_changed).await;
                            }
                            Ok(None) => {
                                debug!("MCP server closed stdout");
                                break;
                            }
                            Err(e) => {
                                error!(error = %e, "Error reading from MCP server");
                                break;
                            }
                        }
                    }
                }
            }
            // Nothing will answer the calls still waiting: drop their senders
            // so each fails now as `ConnectionClosed`, instead of every one of
            // them sitting out the full request timeout against a dead server.
            let orphaned = {
                let mut waiting = pending.lock().await;
                let orphaned = waiting.len();
                waiting.clear();
                orphaned
            };
            if orphaned > 0 {
                warn!(
                    orphaned,
                    "MCP server stream ended with calls in flight; failing them now"
                );
            }
        }

        /// Deliver one server line to where its shape says it belongs.
        async fn route_line(
            line: &str,
            stdin: &Mutex<Option<ChildStdin>>,
            pending: &Mutex<HashMap<String, oneshot::Sender<JsonRpcResponse>>>,
            list_changed: &ListChangedFlags,
        ) {
            match classify_incoming(line) {
                Incoming::Response(response) => {
                    let id = response.id.to_string();
                    let waiter = pending.lock().await.remove(&id);
                    if let Some(tx) = waiter {
                        let _ = tx.send(response);
                    } else {
                        warn!(id, "Received response for unknown request");
                    }
                }
                Incoming::Notification(notif) => handle_server_notification(&notif, list_changed),
                Incoming::ServerRequest { id, method } => {
                    debug!(%id, method, "MCP server sent a request; answering");
                    let reply = reply_to_server_request(&id, &method).to_string();
                    let written = write_line(stdin, &reply).await;
                    if let Err(e) = written {
                        warn!(error = %e, method, "Could not answer an MCP server request");
                    }
                }
                Incoming::Unreadable(error) => {
                    warn!(error, line, "Failed to parse MCP server line");
                }
            }
        }
    }

    #[async_trait]
    impl Transport for StdioTransport {
        async fn request(&self, request: JsonRpcRequest) -> Result<JsonRpcResponse> {
            let id = request.id.to_string();
            
            // Register pending request
            let (tx, rx) = oneshot::channel();
            {
                let mut pending = self.pending.lock().await;
                pending.insert(id.clone(), tx);
            }

            // Send request
            let line = serde_json::to_string(&request)?;
            trace!(line, "Sending to MCP server");
            
            if let Err(error) = write_line(&self.stdin, &line).await {
                // Nothing will ever answer an unsent request: do not leak its slot.
                self.pending.lock().await.remove(&id);
                return Err(error);
            }

            // Wait for response with timeout
            match tokio::time::timeout(crate::MCP_REQUEST_TIMEOUT, rx).await {
                Ok(Ok(response)) => Ok(response),
                Ok(Err(_)) => Err(McpError::ConnectionClosed),
                Err(_) => {
                    // Clean up pending request
                    self.pending.lock().await.remove(&id);
                    Err(McpError::Timeout)
                }
            }
        }

        async fn notify(&self, notification: JsonRpcNotification) -> Result<()> {
            let line = serde_json::to_string(&notification)?;
            trace!(line, "Sending notification to MCP server");

            write_line(&self.stdin, &line).await
        }

        /// Shut the server down the way the stdio binding asks: close its
        /// stdin (the primary, portable signal), give it [`MCP_EXIT_GRACE`] to
        /// exit, and only then kill it.
        async fn close(&self) -> Result<()> {
            let _ = self.shutdown_tx.send(()).await;
            let stdin = self.stdin.lock().await.take();
            drop(stdin);

            let mut child = self.child.lock().await;
            match tokio::time::timeout(MCP_EXIT_GRACE, child.wait()).await {
                Ok(Ok(status)) => debug!(%status, "MCP server exited on stdin EOF"),
                Ok(Err(e)) => warn!(error = %e, "Could not wait on the MCP server"),
                Err(_) => {
                    warn!(
                        grace_ms = MCP_EXIT_GRACE.as_millis(),
                        "MCP server ignored stdin EOF; killing it"
                    );
                    child.kill().await?;
                }
            }
            drop(child);
            Ok(())
        }

        fn list_changed_flags(&self) -> Option<Arc<ListChangedFlags>> {
            Some(self.list_changed.clone())
        }

        /// On stdio a listen stream shares the one channel: write the request
        /// and let the reader route the notifications that follow (they carry
        /// the subscription id in `_meta`, and the flags need no more than the
        /// method). The stream lives as long as the process.
        async fn open_listen(&self, request: JsonRpcRequest) -> Result<()> {
            debug_assert_eq!(request.method, "subscriptions/listen");
            let line = serde_json::to_string(&request)?;
            write_line(&self.stdin, &line).await
        }
    }

    #[cfg(test)]
    mod tests {
        use super::{
            super::reply_to_server_request, Incoming, JsonRpcNotification, JsonRpcResponse,
            ListChangedFlags, McpList, ServerNotification, Transport, classify_incoming,
            classify_server_notification, handle_server_notification, mcp_level_is_severe,
        };

        #[tokio::test]
        async fn a_mark_wakes_a_waiter_even_if_it_came_first() {
            let flags = ListChangedFlags::default();
            // Marked before anyone waits: the stored permit resolves the wait.
            flags.mark(McpList::Tools);
            tokio::time::timeout(std::time::Duration::from_secs(1), flags.changed())
                .await
                .expect("a mark made while nobody waited must not be lost");
            assert!(flags.take(McpList::Tools));
            // And with nothing marked, the wait really waits.
            let idle =
                tokio::time::timeout(std::time::Duration::from_millis(50), flags.changed()).await;
            assert!(idle.is_err(), "no mark, no wake");
        }

        #[cfg(unix)]
        #[tokio::test]
        async fn close_lets_a_server_that_honours_eof_exit_on_its_own() {
            // `cat` exits the moment its stdin closes: no kill should be needed.
            let transport = super::StdioTransport::spawn("sh", &["-c", "cat >/dev/null"]).unwrap();
            let started = std::time::Instant::now();
            transport.close().await.unwrap();
            assert!(
                started.elapsed() < super::MCP_EXIT_GRACE,
                "{:?}",
                started.elapsed()
            );
            let status = transport.child.lock().await.try_wait().unwrap();
            assert!(
                status.is_some_and(|s| s.success()),
                "exited cleanly on EOF: {status:?}"
            );
        }

        #[cfg(unix)]
        #[tokio::test]
        async fn close_kills_a_server_that_ignores_eof_after_the_grace() {
            // Never reads stdin, so EOF means nothing to it — the orphan case.
            let transport = super::StdioTransport::spawn("sh", &["-c", "sleep 30"]).unwrap();
            let started = std::time::Instant::now();
            transport.close().await.unwrap();
            let elapsed = started.elapsed();
            assert!(
                elapsed >= super::MCP_EXIT_GRACE,
                "waited the grace first: {elapsed:?}"
            );
            assert!(
                elapsed < super::MCP_EXIT_GRACE * 2,
                "then killed promptly: {elapsed:?}"
            );
            let status = transport.child.lock().await.try_wait().unwrap();
            assert!(
                status.is_some_and(|s| !s.success()),
                "killed, not exited: {status:?}"
            );
        }

        /// A server that reads the request and dies without answering. Its
        /// waiter used to stay in `pending` and sit out the whole request
        /// timeout; the reader now fails it the moment stdout ends.
        #[cfg(unix)]
        #[tokio::test]
        async fn a_server_that_dies_mid_call_fails_the_call_at_once() {
            let transport =
                super::StdioTransport::spawn("sh", &["-c", "read line; exit 0"]).unwrap();
            let started = std::time::Instant::now();
            let answer = transport
                .request(super::JsonRpcRequest::new(1_i64, "tools/list", None))
                .await;
            assert!(
                matches!(answer, Err(super::McpError::ConnectionClosed)),
                "{answer:?}"
            );
            assert!(
                started.elapsed() < std::time::Duration::from_secs(10),
                "failed on EOF, not after the {:?} timeout: {:?}",
                crate::MCP_REQUEST_TIMEOUT,
                started.elapsed()
            );
        }

        #[test]
        fn the_registry_deadline_sits_past_the_transport_timeout() {
            let transport = crate::MCP_REQUEST_TIMEOUT.as_secs();
            let registry = transport + crate::MCP_DEADLINE_MARGIN_SECS;
            assert!(
                registry > transport,
                "the transport's own timeout must fire first"
            );
            assert_eq!(
                crate::streamable_http::HTTP_REQUEST_TIMEOUT.as_secs(),
                transport
            );
        }

        #[cfg(unix)]
        #[tokio::test]
        async fn a_closed_transport_refuses_to_write() {
            let transport = super::StdioTransport::spawn("sh", &["-c", "cat >/dev/null"]).unwrap();
            transport.close().await.unwrap();
            let refused = transport
                .notify(JsonRpcNotification::new("notifications/initialized", None))
                .await;
            assert!(
                matches!(refused, Err(super::McpError::ConnectionClosed)),
                "{refused:?}"
            );
        }

        #[test]
        fn a_server_request_is_never_mistaken_for_a_response() {
            // The exact line @modelcontextprotocol/server-everything sends after
            // `initialized` when the client declared roots. It deserializes as a
            // JsonRpcResponse too, which is the bug this classification closes.
            let line = r#"{"method":"roots/list","jsonrpc":"2.0","id":0}"#;
            assert!(
                serde_json::from_str::<JsonRpcResponse>(line).is_ok(),
                "the trap is real"
            );
            match classify_incoming(line) {
                Incoming::ServerRequest { id, method } => {
                    assert_eq!(id, serde_json::json!(0));
                    assert_eq!(method, "roots/list");
                }
                other => panic!("expected a server request, got {other:?}"),
            }
        }

        #[test]
        fn responses_notifications_and_garbage_classify_by_shape() {
            assert!(matches!(
                classify_incoming(r#"{"jsonrpc":"2.0","id":3,"result":{}}"#),
                Incoming::Response(_)
            ));
            assert!(matches!(
                classify_incoming(r#"{"jsonrpc":"2.0","id":3,"error":{"code":-1,"message":"x"}}"#),
                Incoming::Response(_)
            ));
            assert!(matches!(
                classify_incoming(
                    r#"{"jsonrpc":"2.0","method":"notifications/tools/list_changed"}"#
                ),
                Incoming::Notification(_)
            ));
            // A null id is a notification's shape, not a request's.
            assert!(matches!(
                classify_incoming(
                    r#"{"jsonrpc":"2.0","method":"notifications/progress","id":null}"#
                ),
                Incoming::Notification(_)
            ));
            assert!(matches!(
                classify_incoming("not json"),
                Incoming::Unreadable(_)
            ));
        }

        #[test]
        fn ping_is_answered_and_everything_else_refused() {
            let id = serde_json::json!("abc");
            let pong = reply_to_server_request(&id, "ping");
            assert_eq!(pong["id"], "abc");
            assert_eq!(pong["result"], serde_json::json!({}));
            assert!(pong.get("error").is_none());

            for method in ["roots/list", "sampling/createMessage", "elicitation/create"] {
                let refusal = reply_to_server_request(&id, method);
                assert_eq!(refusal["error"]["code"], -32601, "{method}");
                assert!(refusal.get("result").is_none(), "{method}");
                assert!(
                    refusal["error"]["message"]
                        .as_str()
                        .unwrap()
                        .contains(method)
                );
            }
        }

        #[test]
        fn classifies_known_notifications() {
            assert_eq!(
                classify_server_notification("notifications/message"),
                ServerNotification::LogMessage
            );
            assert_eq!(
                classify_server_notification("notifications/progress"),
                ServerNotification::Progress
            );
            assert_eq!(
                classify_server_notification("notifications/cancelled"),
                ServerNotification::Cancelled
            );
            assert_eq!(
                classify_server_notification("notifications/tools/list_changed"),
                ServerNotification::ListChanged(Some(McpList::Tools))
            );
            assert_eq!(
                classify_server_notification("notifications/resources/list_changed"),
                ServerNotification::ListChanged(Some(McpList::Resources))
            );
            assert_eq!(
                classify_server_notification("notifications/prompts/list_changed"),
                ServerNotification::ListChanged(Some(McpList::Prompts))
            );
            // A `list_changed` for a list we don't cache (e.g. roots) → no McpList.
            assert_eq!(
                classify_server_notification("notifications/roots/list_changed"),
                ServerNotification::ListChanged(None)
            );
            assert_eq!(
                classify_server_notification("something/else"),
                ServerNotification::Other
            );
        }

        #[test]
        fn severity_mapping_is_correct() {
            for severe in ["warning", "error", "critical", "alert", "emergency"] {
                assert!(mcp_level_is_severe(severe), "{severe} should be severe");
            }
            for benign in ["debug", "info", "notice", "unknown"] {
                assert!(!mcp_level_is_severe(benign), "{benign} should not be severe");
            }
        }

        #[test]
        fn handle_notification_does_not_panic_on_missing_params() {
            // A log-message notification without params must not panic (defaults to info).
            let notif = JsonRpcNotification::new("notifications/message", None);
            let flags = ListChangedFlags::default();
            handle_server_notification(&notif, &flags);
        }

        #[test]
        fn list_changed_marks_only_the_named_list() {
            let flags = ListChangedFlags::default();

            // A tools/list_changed marks tools dirty and nothing else.
            let notif = JsonRpcNotification::new("notifications/tools/list_changed", None);
            handle_server_notification(&notif, &flags);
            assert!(!flags.take(McpList::Resources));
            assert!(!flags.take(McpList::Prompts));
            // take() is read-and-clear: the tools flag is set once, then cleared.
            assert!(flags.take(McpList::Tools));
            assert!(!flags.take(McpList::Tools));

            // An uncached list_changed (roots) marks nothing.
            let roots = JsonRpcNotification::new("notifications/roots/list_changed", None);
            handle_server_notification(&roots, &flags);
            assert!(!flags.take(McpList::Tools));
            assert!(!flags.take(McpList::Resources));
            assert!(!flags.take(McpList::Prompts));
        }
    }
}

#[cfg(feature = "stdio")]
pub use stdio::{MCP_EXIT_GRACE, StdioTransport};

/// Either transport the daemon starts servers over, so one manager can hold
/// stdio and Streamable HTTP servers side by side.
#[cfg(all(feature = "stdio", feature = "http"))]
pub enum AnyTransport {
    /// A child process speaking over stdin/stdout.
    Stdio(StdioTransport),
    /// A Streamable HTTP endpoint (boxed: it is several times the size of a
    /// stdio transport, and a manager holds many of either).
    Http(Box<crate::StreamableHttpTransport>),
    /// A server still on the deprecated 2024 HTTP+SSE transport.
    Sse(Box<crate::LegacySseTransport>),
}

#[cfg(all(feature = "stdio", feature = "http"))]
#[async_trait]
impl Transport for AnyTransport {
    async fn request(&self, request: JsonRpcRequest) -> Result<JsonRpcResponse> {
        match self {
            Self::Stdio(inner) => inner.request(request).await,
            Self::Http(inner) => inner.request(request).await,
            Self::Sse(inner) => inner.request(request).await,
        }
    }

    async fn notify(&self, notification: JsonRpcNotification) -> Result<()> {
        match self {
            Self::Stdio(inner) => inner.notify(notification).await,
            Self::Http(inner) => inner.notify(notification).await,
            Self::Sse(inner) => inner.notify(notification).await,
        }
    }

    async fn close(&self) -> Result<()> {
        match self {
            Self::Stdio(inner) => inner.close().await,
            Self::Http(inner) => inner.close().await,
            Self::Sse(inner) => inner.close().await,
        }
    }

    fn list_changed_flags(&self) -> Option<Arc<ListChangedFlags>> {
        match self {
            Self::Stdio(inner) => inner.list_changed_flags(),
            Self::Http(inner) => inner.list_changed_flags(),
            Self::Sse(inner) => inner.list_changed_flags(),
        }
    }

    async fn open_listen(&self, request: JsonRpcRequest) -> Result<()> {
        match self {
            Self::Stdio(inner) => inner.open_listen(request).await,
            Self::Http(inner) => inner.open_listen(request).await,
            Self::Sse(inner) => inner.open_listen(request).await,
        }
    }
}
