//! IPC Server - WebSocket-based communication
//!
//! Handles connections from channel clients (GUI, CLI, API, etc.)

use crate::protocol::{Event, Request, RequestId, Response};
use futures_util::{SinkExt, StreamExt};
use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{broadcast, mpsc, watch, RwLock};
use tokio_tungstenite::{accept_async_with_config, tungstenite::{protocol::WebSocketConfig, Message}};
use tracing::{debug, error, info, warn};

/// The IPC read limit, shared with every client — see its definition for why
/// a limit set on one end only ever protected that end.
pub use nanna_config::bind::IPC_MAX_MESSAGE_BYTES;

/// Server-initiated keepalive ping cadence. A live client answers with a pong
/// (an incoming frame), which resets the read deadline below.
const WS_PING_INTERVAL_SECS: u64 = 15;

/// Drop a connection whose peer has been silent this long — three missed
/// ping/pong cycles. Bounds the read await for force-killed clients that never
/// send a Close frame (Windows TCP keepalive is off by default).
const WS_READ_DEADLINE_SECS: u64 = 45;

/// Unique identifier for a connected client
pub type ConnectionId = String;

/// A request decoded off one connection, tagged with the connection it
/// arrived on — what the accept loop hands the daemon's request loop.
pub type TaggedRequest = (ConnectionId, Request);

/// One connection's read half, after the WebSocket stream is split.
type WsReader = futures_util::stream::SplitStream<
    tokio_tungstenite::WebSocketStream<TcpStream>,
>;

/// Configuration for the IPC server
#[derive(Debug, Clone)]
pub struct IpcServerConfig {
    /// Host to bind to
    pub host: String,
    /// Port to listen on
    pub port: u16,
    /// Maximum number of concurrent connections
    pub max_connections: usize,
}

/// The WebSocket IPC port every client is expected to reach the daemon on.
///
/// One definition on purpose: this used to be a literal repeated across the daemon, the CLI and
/// the docs, and the copies drifted — `nanna daemon start` defaulted to `9999` while
/// `nanna daemon status`, the GUI sidecar and the README all used `5149`, so a CLI-started daemon
/// reported itself as not running.
pub use nanna_config::{DEFAULT_IPC_PORT, default_daemon_ws_url};

impl Default for IpcServerConfig {
    fn default() -> Self {
        Self {
            host: nanna_config::bind::LOOPBACK_HOST.to_string(),
            port: DEFAULT_IPC_PORT,
            max_connections: 100,
        }
    }
}

/// Connected client state
#[derive(Debug)]
struct ClientConnection {
    _id: ConnectionId,
    _addr: SocketAddr,
    tx: mpsc::Sender<Message>,
}

/// Whether one connection should be sent one event.
///
/// The whole narrowing decision, as a pure function over the connection's
/// declared interest, so the policy is testable without a socket, a broadcast,
/// or a lock.
///
/// * `narrowed_to == None` — the connection never asked for less, so it gets
///   everything. This is every shipping client today.
/// * `narrowed_to == Some(set)` — the connection declared which sessions it
///   wants. It gets those, plus every event that carries **no** session at all
///   (config, memory, workspace, connection and channel events), because those
///   are not session-scoped and dropping them would silently break the
///   `System` and `ChannelStatus` topics.
///
/// An empty `set` therefore means "non-session events only", not "everything":
/// a connection that narrowed and then unsubscribed each session has not asked
/// to be widened again, and widening it there would silently re-open the very
/// stream it closed. `Subscribe{AllSessions}` is the explicit way back.
fn delivers(narrowed_to: Option<&HashSet<String>>, event: &Event) -> bool {
    let Some(sessions) = narrowed_to else {
        return true;
    };
    event
        .session_id()
        .is_none_or(|session_id| sessions.contains(session_id))
}

/// Per-connection event narrowing for the shared event broadcast.
///
/// `Subscribe{Session}` used to be recorded and then ignored: every connection
/// was forwarded the entire event stream, so one attached client saw every
/// other session's message deltas, tool calls and errors on the wire. Clients
/// filtered locally, which does nothing about the bytes having already left the
/// daemon — and with the North Star's phone-plus-desktop shape, "every client
/// sees every session" is a leak, not a convenience.
///
/// Narrowing is **opt-in**: a connection receives everything until it asks for
/// less. No shipping client sends `Subscribe` today, so this changes nothing
/// for any of them — and [`Self::is_empty`] makes that literal, short-circuiting
/// the lookup entirely while nobody has narrowed.
///
/// **Bounded** on both axes without inventing a cap: one entry per live
/// connection (the IPC server already bounds those by `max_connections`, and
/// [`Self::forget`] removes the entry on disconnect), and per entry at most the
/// number of live sessions, because `handle_subscribe` answers `not_found` for
/// a session the store does not hold.
#[derive(Debug, Default)]
pub struct SessionFilters {
    narrowed: RwLock<HashMap<ConnectionId, HashSet<String>>>,
    /// `narrowed.len()`, maintained inside the same write guard so it cannot
    /// drift, and read without the lock so an un-narrowed daemon pays one
    /// relaxed atomic load per event instead of a contended read lock.
    narrowed_count: AtomicUsize,
}

impl SessionFilters {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether no connection has narrowed itself — the fast path.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.narrowed_count.load(Ordering::Relaxed) == 0
    }

    /// Narrow `client_id` to `session_id`, in addition to any it already named.
    ///
    /// The write guard deliberately spans the `narrowed_count` update in this
    /// and every sibling below: releasing it first would let two writers
    /// compute their lengths and then store them out of order, which is exactly
    /// the drift the counter exists to be free of.
    ///
    /// # Panics
    /// Panics on an empty `session_id` — a caller that narrows to nothing has a
    /// bug, and silently narrowing to `""` would mute the connection instead.
    pub async fn narrow_to_session(&self, client_id: &str, session_id: String) {
        assert!(!session_id.is_empty(), "narrowed to an empty session id");
        let mut narrowed = self.narrowed.write().await;
        narrowed
            .entry(client_id.to_string())
            .or_default()
            .insert(session_id);
        debug_assert!(!narrowed.is_empty(), "narrowing left no entry behind");
        self.narrowed_count.store(narrowed.len(), Ordering::Relaxed);
    }

    /// Stop delivering `session_id` to `client_id`.
    ///
    /// A connection that has never narrowed stays un-narrowed: unsubscribing
    /// from one session is not a request to be cut off from the rest.
    pub async fn drop_session(&self, client_id: &str, session_id: &str) {
        let mut narrowed = self.narrowed.write().await;
        if let Some(sessions) = narrowed.get_mut(client_id) {
            sessions.remove(session_id);
        }
        self.narrowed_count.store(narrowed.len(), Ordering::Relaxed);
    }

    /// Widen `client_id` back to the whole stream.
    pub async fn widen_to_all(&self, client_id: &str) {
        let mut narrowed = self.narrowed.write().await;
        narrowed.remove(client_id);
        self.narrowed_count.store(narrowed.len(), Ordering::Relaxed);
    }

    /// Narrow `client_id` to nothing session-scoped — the inverse of
    /// [`Self::widen_to_all`], and what `Unsubscribe{AllSessions}` means.
    ///
    /// # Panics
    /// Panics if the entry did not land, which would leave the connection
    /// silently receiving every session it just asked to stop receiving.
    pub async fn drop_all_sessions(&self, client_id: &str) {
        let mut narrowed = self.narrowed.write().await;
        narrowed.insert(client_id.to_string(), HashSet::new());
        assert!(
            !narrowed.is_empty(),
            "dropping all sessions left no filter entry, so the connection \
             would silently keep receiving every session",
        );
        self.narrowed_count.store(narrowed.len(), Ordering::Relaxed);
    }

    /// Forget a disconnected connection, so the map is bounded by live
    /// connections rather than by every connection the daemon has ever seen.
    pub async fn forget(&self, client_id: &str) {
        let mut narrowed = self.narrowed.write().await;
        narrowed.remove(client_id);
        self.narrowed_count.store(narrowed.len(), Ordering::Relaxed);
    }

    /// Whether `event` should reach `client_id`.
    pub async fn delivers(&self, client_id: &str, event: &Event) -> bool {
        if self.is_empty() {
            return true;
        }
        let narrowed = self.narrowed.read().await;
        delivers(narrowed.get(client_id), event)
    }
}

/// The IPC port, bound but not yet listening.
///
/// Daemon startup reserves the port BEFORE opening storage, so an instance
/// that can never serve IPC (the port is held — usually by a live daemon)
/// exits before it touches `nanna.db`, instead of surfacing first as a turso
/// lock failure and dying only after the router and memory service were built
/// (2026-09-17). Bound-not-listening claims the port against a listening
/// daemon while connects are still refused, so a client polling for readiness
/// keeps reading "accepts a connection" as "serving".
///
/// On Unix the claim is exclusive against a LISTENING socket only: an instance
/// that skipped the PID guard could still bind with `SO_REUSEADDR` and listen
/// first, and this port's `listen` then fails — the pre-reservation failure,
/// arbitrated late instead of early.
pub struct ReservedIpcPort {
    socket: socket2::Socket,
    addr: SocketAddr,
}

impl ReservedIpcPort {
    fn bind(addr: SocketAddr) -> Result<Self, std::io::Error> {
        let socket = socket2::Socket::new(
            socket2::Domain::for_address(addr),
            socket2::Type::STREAM,
            Some(socket2::Protocol::TCP),
        )?;
        #[cfg(unix)]
        socket.set_reuse_address(true)?;
        socket.set_nonblocking(true)?;
        socket.bind(&addr.into())?;
        let addr = socket
            .local_addr()?
            .as_socket()
            .unwrap_or(addr);
        Ok(Self { socket, addr })
    }

    /// The bound address (the real port when the configured one was 0).
    #[must_use]
    pub const fn local_addr(&self) -> SocketAddr {
        self.addr
    }

    fn listen(self) -> Result<TcpListener, std::io::Error> {
        self.socket.listen(128)?;
        TcpListener::from_std(self.socket.into())
    }
}

/// IPC Server for daemon communication
pub struct IpcServer {
    config: IpcServerConfig,
    clients: Arc<RwLock<HashMap<ConnectionId, ClientConnection>>>,
    session_filters: Arc<SessionFilters>,
    request_tx: mpsc::Sender<TaggedRequest>,
    request_rx: Arc<RwLock<Option<mpsc::Receiver<TaggedRequest>>>>,
    event_tx: broadcast::Sender<Event>,
    shutdown_tx: broadcast::Sender<()>,
    /// The address the listener actually bound, published once it listens.
    /// With port 0 this is the only way to learn the port the OS chose.
    bound_tx: watch::Sender<Option<SocketAddr>>,
}

impl IpcServer {
    /// Create a new IPC server
    #[must_use]
    pub fn new(config: IpcServerConfig) -> Self {
        let (request_tx, request_rx) = mpsc::channel(1000);
        let (event_tx, _) = broadcast::channel(1000);
        let (shutdown_tx, _) = broadcast::channel(1);
        let (bound_tx, _) = watch::channel(None);
        
        Self {
            config,
            clients: Arc::new(RwLock::new(HashMap::new())),
            request_tx,
            request_rx: Arc::new(RwLock::new(Some(request_rx))),
            session_filters: Arc::new(SessionFilters::new()),
            event_tx,
            shutdown_tx,
            bound_tx,
        }
    }
    
    /// Claim the IPC port without accepting connections yet.
    ///
    /// On Unix, sets `SO_REUSEADDR` so `TIME_WAIT` sockets don't block restart
    /// (it does not permit binding over a LISTENING socket, which is what a
    /// live daemon holds). On Windows, `SO_REUSEADDR` has dangerous semantics
    /// (allows hijacking), so we retry with a short delay instead.
    ///
    /// # Errors
    ///
    /// An `InvalidInput` error when `host:port` is not a socket address, and
    /// the bind error when the port is already held — on Windows only after
    /// the retries above are spent.
    #[cfg_attr(
        not(windows),
        expect(
            clippy::unused_async,
            clippy::unused_async_trait_impl,
            reason = "the async is the Windows retry ladder's sleep; every caller \
                      awaits this on every platform, so the signature is shared"
        )
    )]
    pub async fn reserve_port(&self) -> Result<ReservedIpcPort, std::io::Error> {
        let addr: SocketAddr = self.address().parse()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;

        #[cfg(windows)]
        {
            for attempt in 0..5 {
                match ReservedIpcPort::bind(addr) {
                    Ok(port) => return Ok(port),
                    Err(e) if attempt < 4 => {
                        warn!("IPC bind attempt {} failed ({}), retrying in 1s...", attempt + 1, e);
                        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                    }
                    Err(e) => return Err(e),
                }
            }
            unreachable!()
        }

        #[cfg(not(windows))]
        ReservedIpcPort::bind(addr)
    }

    /// The address the listener bound: `None` until `serve` has bound it.
    ///
    /// Port 0 asks the OS for a free port, and this is how a caller learns
    /// which one. Choosing a port up front and releasing it before the server
    /// binds races with every other process picking ports the same way.
    #[must_use]
    pub fn bound_addr(&self) -> watch::Receiver<Option<std::net::SocketAddr>> {
        self.bound_tx.subscribe()
    }

    /// Get the address the server will bind to
    #[must_use]
    pub fn address(&self) -> String {
        format!("{}:{}", self.config.host, self.config.port)
    }
    
    /// Get a sender for broadcasting events to clients
    #[must_use]
    pub fn event_sender(&self) -> broadcast::Sender<Event> {
        self.event_tx.clone()
    }
    
    /// Get the request receiver (for the daemon to process requests)
    /// Can only be called once - returns None if already taken
    /// The per-connection narrowing registry, so the control plane's
    /// `Subscribe`/`Unsubscribe` handlers can drive what this server forwards.
    #[must_use]
    pub fn session_filters(&self) -> Arc<SessionFilters> {
        self.session_filters.clone()
    }

    pub async fn take_request_receiver(&self) -> Option<mpsc::Receiver<TaggedRequest>> {
        let mut rx_lock = self.request_rx.write().await;
        rx_lock.take()
    }
    
    /// Send a response to a specific client
    ///
    /// # Errors
    ///
    /// The rendered error string when `client_id` is not connected, when the
    /// response cannot be serialized, or when its connection's outgoing
    /// channel is closed (the client went away mid-send).
    pub async fn send_response(&self, client_id: &str, response: Response) -> Result<(), String> {
        // The sender is cloned out and the guard released BEFORE the send. The
        // channel is bounded, so a client that stops reading makes `send`
        // wait — and a read guard held across that wait blocked every writer:
        // no client could connect or disconnect until the slow one drained.
        let tx = self
            .clients
            .read()
            .await
            .get(client_id)
            .map(|client| client.tx.clone())
            .ok_or_else(|| format!("Client not found: {client_id}"))?;
        let msg = serde_json::to_string(&response).map_err(|e| e.to_string())?;
        tx.send(Message::Text(msg.into()))
            .await
            .map_err(|e| e.to_string())
    }
    
    /// Broadcast an event to all subscribed clients
    pub fn broadcast_event(&self, event: Event) {
        let _ = self.event_tx.send(event);
    }
    
    /// Get connected client count
    pub async fn client_count(&self) -> usize {
        self.clients.read().await.len()
    }
    
    /// Get list of connected client IDs
    pub async fn client_ids(&self) -> Vec<ConnectionId> {
        self.clients.read().await.keys().cloned().collect()
    }
    
    /// Shutdown the server
    pub fn shutdown(&self) {
        let _ = self.shutdown_tx.send(());
    }
    
    /// Run the IPC server
    ///
    /// # Errors
    ///
    /// Whatever [`Self::reserve_port`] and [`Self::serve`] return: the port is
    /// held by another process, or the reserved socket cannot start listening.
    pub async fn run(self: &Arc<Self>) -> Result<(), std::io::Error> {
        let port = self.reserve_port().await?;
        self.serve(port).await
    }

    /// Serve on a port claimed earlier by [`Self::reserve_port`]: start
    /// listening, then accept until shutdown.
    ///
    /// # Errors
    ///
    /// The error from `listen` on the reserved socket, or from handing it to
    /// tokio. Accept errors are logged and the loop continues, and shutdown
    /// ends it with `Ok`.
    pub async fn serve(self: &Arc<Self>, port: ReservedIpcPort) -> Result<(), std::io::Error> {
        let addr = port.addr;
        let listener = port.listen()?;
        // Published only once the listener accepts: a caller waiting on this
        // (a test on port 0, which learns the OS-chosen port here) must not
        // connect before then.
        self.bound_tx.send_replace(Some(addr));
        info!("IPC server listening on ws://{addr}");

        let mut shutdown_rx = self.shutdown_tx.subscribe();

        loop {
            tokio::select! {
                result = listener.accept() => {
                    match result {
                        Ok((stream, addr)) => {
                            // Check connection limit
                            if self.clients.read().await.len() >= self.config.max_connections {
                                warn!("Connection limit reached, rejecting {}", addr);
                                continue;
                            }

                            let client_id = uuid::Uuid::new_v4().to_string();
                            info!("New connection from {}: {}", addr, client_id);

                            // Spawn — NEVER await inline. The old inline await
                            // served one connection at a time, so a single
                            // silently-dead peer (force-killed client, no
                            // Close frame, no read deadline) parked the whole
                            // accept loop forever: new TCP connects completed
                            // in the kernel backlog but were never accepted
                            // (observed live; only a restart cleared it).
                            let this = Arc::clone(self);
                            tokio::spawn(async move {
                                this.handle_connection(client_id, stream, addr).await;
                            });
                        }
                        Err(e) => {
                            error!("Accept error: {}", e);
                        }
                    }
                }
                _ = shutdown_rx.recv() => {
                    info!("IPC server shutting down");
                    break;
                }
            }
        }

        Ok(())
    }
    
    async fn handle_connection(
        &self,
        client_id: ConnectionId,
        stream: TcpStream,
        addr: SocketAddr,
    ) {
        let mut ws_config = WebSocketConfig::default();
        ws_config.max_message_size = Some(IPC_MAX_MESSAGE_BYTES);
        ws_config.max_frame_size = Some(IPC_MAX_MESSAGE_BYTES);
        let ws_stream = match accept_async_with_config(stream, Some(ws_config)).await {
            Ok(ws) => ws,
            Err(e) => {
                error!("WebSocket handshake failed for {}: {}", addr, e);
                return;
            }
        };
        
        let (mut ws_tx, mut ws_rx) = ws_stream.split();
        let (msg_tx, mut msg_rx) = mpsc::channel::<Message>(100);
        
        // Store client connection
        {
            let mut clients = self.clients.write().await;
            clients.insert(client_id.clone(), ClientConnection {
                _id: client_id.clone(),
                _addr: addr,
                tx: msg_tx.clone(),
            });
        }
        
        // Broadcast connect event
        let _ = self.event_tx.send(Event::Connected { client_id: client_id.clone() });
        
        let clients = self.clients.clone();
        let request_tx = self.request_tx.clone();
        let event_rx = self.event_tx.subscribe();
        let client_id_clone = client_id.clone();
        let session_filters = self.session_filters.clone();
        let filtered_client_id = client_id.clone();
        
        // Spawn task to handle outgoing messages
        let outgoing_task = tokio::spawn(async move {
            let mut event_rx = event_rx;
            
            loop {
                tokio::select! {
                    // Forward messages from the channel to WebSocket
                    Some(msg) = msg_rx.recv() => {
                        if ws_tx.send(msg).await.is_err() {
                            break;
                        }
                    }
                    // Forward broadcast events to this client. Every receive result is
                    // handled: an `Ok(event)` pattern let a `Lagged` error fail the match,
                    // and the missed events vanished without the client ever knowing.
                    received = event_rx.recv() => {
                        let Some(event) = forwardable(received) else { break };
                        // A connection that narrowed itself gets only what it
                        // asked for. Until one does, this is a relaxed atomic
                        // load and nothing else.
                        if !session_filters.delivers(&filtered_client_id, &event).await {
                            continue;
                        }
                        if let Ok(json) = serde_json::to_string(&event)
                            && ws_tx.send(Message::Text(json.into())).await.is_err()
                        {
                            break;
                        }
                    }
                    else => break,
                }
            }
        });
        
        // Read until the peer closes, errors, or goes silent.
        pump_incoming(&client_id, &mut ws_rx, &msg_tx, &request_tx).await;
        
        // Cleanup
        outgoing_task.abort();
        
        {
            let mut clients_guard = clients.write().await;
            clients_guard.remove(&client_id_clone);
        }
        self.session_filters.forget(&client_id_clone).await;
        
        // Broadcast disconnect event
        let _ = self.event_tx.send(Event::Disconnected { client_id: client_id_clone.clone() });
        
        info!("Client {} disconnected", client_id_clone);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config_default() {
        let config = IpcServerConfig::default();
        assert_eq!(config.port, 5149);
        assert_eq!(config.host, "127.0.0.1");
    }

    #[tokio::test]
    async fn port_zero_publishes_the_port_the_os_chose() {
        let server = Arc::new(IpcServer::new(IpcServerConfig {
            host: "127.0.0.1".to_string(),
            port: 0,
            ..IpcServerConfig::default()
        }));
        let mut bound = server.bound_addr();
        assert_eq!(*bound.borrow(), None, "nothing is bound before run");

        let running = Arc::clone(&server);
        let task = tokio::spawn(async move { running.run().await });
        bound.changed().await.expect("run publishes the bound address");
        let addr = (*bound.borrow()).expect("a bound address");
        assert_ne!(addr.port(), 0, "the published port is the real one");
        tokio::net::TcpStream::connect(addr)
            .await
            .expect("the published address accepts connections");

        server.shutdown();
        task.await.expect("run task").expect("run ends cleanly on shutdown");
    }
}

/// Read one connection until it ends, forwarding decoded requests to the
/// daemon's request queue.
///
/// Every await here is bounded: the server pings every
/// `WS_PING_INTERVAL_SECS` (the pong resets the read deadline), and a peer
/// silent past `WS_READ_DEADLINE_SECS` is dropped. Without this, a
/// force-killed client (no Close frame, Windows TCP keepalive off) left
/// `ws_rx.next()` pending forever.
async fn pump_incoming(
    client_id: &ConnectionId,
    ws_rx: &mut WsReader,
    msg_tx: &mpsc::Sender<Message>,
    request_tx: &mpsc::Sender<TaggedRequest>,
) {
    let mut ping_interval =
        tokio::time::interval(std::time::Duration::from_secs(WS_PING_INTERVAL_SECS));
    ping_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    'incoming: loop {
        tokio::select! {
            maybe_msg = tokio::time::timeout(
                std::time::Duration::from_secs(WS_READ_DEADLINE_SECS),
                ws_rx.next(),
            ) => {
                let msg = match maybe_msg {
                    Err(_elapsed) => {
                        warn!(
                            "Read deadline ({WS_READ_DEADLINE_SECS}s) exceeded for {} — dropping dead connection",
                            client_id
                        );
                        break 'incoming;
                    }
                    Ok(None) => break 'incoming,
                    Ok(Some(m)) => m,
                };
                match msg {
                    Ok(Message::Text(text)) => {
                        match serde_json::from_str::<Request>(&text) {
                            Ok(request) => {
                                debug!("Request from {}: {:?}", client_id, request.action);
                                if request_tx.send((client_id.clone(), request)).await.is_err() {
                                    break 'incoming;
                                }
                            }
                            Err(e) => {
                                warn!("Invalid request from {}: {}", client_id, e);
                                // Answer under the request's own id when it has
                                // one: an error nobody can correlate reads, to
                                // the client, as silence until its timeout.
                                let error_response = Response::error(
                                    request_id_of(&text),
                                    "parse_error",
                                    format!("Invalid request: {e}"),
                                );
                                if let Ok(json) = serde_json::to_string(&error_response) {
                                    let _ = msg_tx.send(Message::Text(json.into())).await;
                                }
                            }
                        }
                    }
                    Ok(Message::Ping(data)) => {
                        let _ = msg_tx.send(Message::Pong(data)).await;
                    }
                    Ok(Message::Close(_)) => {
                        debug!("Client {} sent close", client_id);
                        break 'incoming;
                    }
                    Err(e) => {
                        debug!("WebSocket error for {}: {}", client_id, e);
                        break 'incoming;
                    }
                    _ => {}
                }
            }
            _ = ping_interval.tick() => {
                // Server-initiated keepalive: forces the OS to notice a
                // dead peer even with TCP keepalive off.
                if msg_tx.send(Message::Ping(Vec::new().into())).await.is_err() {
                    break 'incoming;
                }
            }
        }
    }
}

/// The id a request that failed to decode was sent under, so its error can
/// be matched to it: the raw JSON's `id` when it is a string or a number
/// (`RequestId` is a string; a numeric id is echoed as its decimal form), or
/// `"unknown"` when the text is not JSON or carries no usable id. Pure.
fn request_id_of(text: &str) -> RequestId {
    /// Bound on an echoed id: longer is not an id a client sent in earnest.
    const ECHOED_ID_BYTES_MAX: usize = 256;
    let id = serde_json::from_str::<serde_json::Value>(text)
        .ok()
        .and_then(|value| match value.get("id") {
            Some(serde_json::Value::String(id)) => Some(id.clone()),
            Some(serde_json::Value::Number(n)) => Some(n.to_string()),
            _ => None,
        })
        .filter(|id| !id.is_empty() && id.len() <= ECHOED_ID_BYTES_MAX);
    id.unwrap_or_else(|| "unknown".to_string())
}

/// What an IPC connection forwards for one receive from the shared event broadcast:
/// the event itself; for a connection that fell behind the bounded broadcast, an
/// `Error` event saying how many it missed so the client can resync; `None` once the
/// broadcast is closed.
fn forwardable(received: Result<Event, tokio::sync::broadcast::error::RecvError>) -> Option<Event> {
    use tokio::sync::broadcast::error::RecvError;
    match received {
        Ok(event) => Some(event),
        Err(RecvError::Lagged(missed)) => {
            debug_assert!(missed > 0, "a lag always skips at least one event");
            warn!("IPC client fell behind the event broadcast and missed {missed} events");
            Some(Event::Error {
                code: "events_lagged".to_string(),
                message: format!(
                    "this connection fell behind and missed {missed} events; re-fetch state to resync"
                ),
                session_id: None,
            })
        }
        Err(RecvError::Closed) => None,
    }
}

#[cfg(test)]
mod request_id_tests {
    use super::{Request, request_id_of};

    #[test]
    fn a_request_that_fails_to_decode_keeps_its_id() {
        // Each of these was answered under id "unknown" — silence to the
        // client that sent it, until its own timeout.
        for (text, id) in [
            (
                r#"{"id":"a","action":{"type":"tool","action":"execute","name":"nope"}}"#,
                "a",
            ),
            (
                r#"{"id":"b","action":{"type":"tool","action":"teleport"}}"#,
                "b",
            ),
            (r#"{"id":"c","action":{"type":"warp"}}"#, "c"),
            (r#"{"id":42,"action":{"type":"warp"}}"#, "42"),
        ] {
            assert!(
                serde_json::from_str::<Request>(text).is_err(),
                "must not decode: {text}"
            );
            assert_eq!(request_id_of(text), id, "{text}");
        }
    }

    #[test]
    fn no_usable_id_falls_back_to_unknown() {
        for text in [
            "this is not json",
            r#"{"action":{"type":"system","action":"status"}}"#,
            r#"{"id":null}"#,
            r#"{"id":""}"#,
            r#"{"id":{"nested":true}}"#,
        ] {
            assert_eq!(request_id_of(text), "unknown", "{text}");
        }
        let long = format!(r#"{{"id":"{}"}}"#, "x".repeat(300));
        assert_eq!(
            request_id_of(&long),
            "unknown",
            "an absurd id is not echoed"
        );
    }
}

#[cfg(test)]
mod session_filter_tests {
    use super::{Event, SessionFilters, delivers};
    use std::collections::HashSet;

    fn session_event(session_id: &str) -> Event {
        Event::MessageStart {
            session_id: session_id.to_string(),
            message_id: "m1".to_string(),
        }
    }

    // --- the pure decision ------------------------------------------------

    #[test]
    fn an_un_narrowed_connection_receives_everything() {
        assert!(delivers(None, &session_event("a")));
        assert!(delivers(None, &Event::ConfigChanged));
    }

    #[test]
    fn a_narrowed_connection_receives_only_its_sessions() {
        let wanted: HashSet<String> = std::iter::once("a".to_string()).collect();
        assert!(delivers(Some(&wanted), &session_event("a")));
        assert!(
            !delivers(Some(&wanted), &session_event("b")),
            "another session's events reached a connection that never asked \
             for them — the leak this filter exists to close",
        );
    }

    #[test]
    fn narrowing_never_drops_events_that_carry_no_session() {
        let wanted: HashSet<String> = std::iter::once("a".to_string()).collect();
        assert!(delivers(Some(&wanted), &Event::ConfigChanged));
        assert!(delivers(Some(&wanted), &Event::WorkspacesChanged));
    }

    #[test]
    fn an_empty_narrowing_is_not_a_widening() {
        let none_wanted: HashSet<String> = HashSet::new();
        assert!(
            !delivers(Some(&none_wanted), &session_event("a")),
            "unsubscribing every session silently re-opened the whole stream",
        );
        assert!(delivers(Some(&none_wanted), &Event::ConfigChanged));
    }

    // --- the registry -----------------------------------------------------

    #[tokio::test]
    async fn nobody_narrowed_means_the_fast_path_and_full_delivery() {
        let filters = SessionFilters::new();
        assert!(filters.is_empty());
        assert!(filters.delivers("client-1", &session_event("a")).await);
        assert!(filters.delivers("client-2", &session_event("b")).await);
    }

    #[tokio::test]
    async fn one_connection_narrowing_does_not_narrow_another() {
        let filters = SessionFilters::new();
        filters.narrow_to_session("client-1", "a".to_string()).await;

        assert!(!filters.is_empty());
        assert!(filters.delivers("client-1", &session_event("a")).await);
        assert!(!filters.delivers("client-1", &session_event("b")).await);
        assert!(
            filters.delivers("client-2", &session_event("b")).await,
            "a second connection was narrowed by the first one's Subscribe",
        );
    }

    #[tokio::test]
    async fn a_connection_can_narrow_to_several_sessions() {
        let filters = SessionFilters::new();
        filters.narrow_to_session("client-1", "a".to_string()).await;
        filters.narrow_to_session("client-1", "b".to_string()).await;

        assert!(filters.delivers("client-1", &session_event("a")).await);
        assert!(filters.delivers("client-1", &session_event("b")).await);
        assert!(!filters.delivers("client-1", &session_event("c")).await);
    }

    #[tokio::test]
    async fn widening_restores_the_whole_stream_and_the_fast_path() {
        let filters = SessionFilters::new();
        filters.narrow_to_session("client-1", "a".to_string()).await;
        assert!(!filters.delivers("client-1", &session_event("b")).await);

        filters.widen_to_all("client-1").await;
        assert!(
            filters.is_empty(),
            "the last narrowing left the slow path armed"
        );
        assert!(filters.delivers("client-1", &session_event("b")).await);
    }

    #[tokio::test]
    async fn dropping_one_session_keeps_the_rest() {
        let filters = SessionFilters::new();
        filters.narrow_to_session("client-1", "a".to_string()).await;
        filters.narrow_to_session("client-1", "b".to_string()).await;

        filters.drop_session("client-1", "a").await;
        assert!(!filters.delivers("client-1", &session_event("a")).await);
        assert!(filters.delivers("client-1", &session_event("b")).await);
    }

    #[tokio::test]
    async fn dropping_a_session_never_narrows_a_connection_that_had_not() {
        let filters = SessionFilters::new();
        // Keep the registry non-empty so `delivers` takes the slow path and
        // this asserts the lookup, not the short-circuit.
        filters.narrow_to_session("other", "z".to_string()).await;

        filters.drop_session("client-1", "a").await;
        assert!(
            filters.delivers("client-1", &session_event("a")).await,
            "unsubscribing one session cut a connection off from every session",
        );
    }

    #[tokio::test]
    async fn dropping_all_sessions_leaves_only_the_session_less_events() {
        let filters = SessionFilters::new();
        filters.drop_all_sessions("client-1").await;

        assert!(!filters.is_empty());
        assert!(!filters.delivers("client-1", &session_event("a")).await);
        assert!(filters.delivers("client-1", &Event::ConfigChanged).await);
    }

    #[tokio::test]
    async fn a_disconnected_connection_is_forgotten() {
        let filters = SessionFilters::new();
        filters.narrow_to_session("client-1", "a".to_string()).await;
        assert!(!filters.is_empty());

        filters.forget("client-1").await;
        assert!(
            filters.is_empty(),
            "a disconnected connection's filter outlived it, so the map grows \
             with every connection the daemon has ever seen",
        );
    }

    /// The reason this change is safe to ship without driving the GUI: a
    /// connection that never sends `Subscribe` follows the identical path it
    /// followed before, and no shipping client sends one.
    #[tokio::test]
    async fn a_silent_connection_is_unaffected_by_another_narrowing() {
        let filters = SessionFilters::new();
        filters.narrow_to_session("cli", "a".to_string()).await;

        for event in [session_event("a"), session_event("b"), Event::ConfigChanged] {
            assert!(
                filters.delivers("gui-never-subscribed", &event).await,
                "a client that never subscribed lost an event",
            );
        }
    }
}

#[cfg(test)]
mod forwardable_tests {
    use super::{Event, forwardable};
    use tokio::sync::broadcast::error::RecvError;

    #[test]
    fn a_received_event_is_forwarded_as_is() {
        let forwarded = forwardable(Ok(Event::ConfigChanged));
        assert!(matches!(forwarded, Some(Event::ConfigChanged)));
    }

    #[test]
    fn a_lag_becomes_an_error_event_naming_the_count() {
        match forwardable(Err(RecvError::Lagged(7))) {
            Some(Event::Error {
                code,
                message,
                session_id,
            }) => {
                assert_eq!(code, "events_lagged");
                assert!(message.contains("missed 7 events"), "{message}");
                assert_eq!(
                    session_id, None,
                    "a lag belongs to the connection, not a session"
                );
            }
            other => panic!("expected an events_lagged error, got {other:?}"),
        }
    }

    #[test]
    fn a_closed_broadcast_ends_the_forwarder() {
        assert!(forwardable(Err(RecvError::Closed)).is_none());
    }
}
