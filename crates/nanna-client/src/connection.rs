//! WebSocket connection to daemon

use crate::{ClientError, Result};
use futures_util::{SinkExt, StreamExt};
use nanna_daemon::protocol::{Event, Response, ResponseResult, Action, SubscribeAction, Request, SessionAction, ExportFormat, ChatAction, MemoryAction, ConfigAction, ToolAction, SystemAction, SchedulerAction, WorkspaceAction, ChannelAction};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpStream;
use tokio::sync::{broadcast, mpsc, oneshot, RwLock};
use tokio_tungstenite::{
    MaybeTlsStream, WebSocketStream, connect_async_with_config,
    tungstenite::{Message, protocol::WebSocketConfig},
};
use tracing::{debug, error, info, warn};

/// Client configuration
#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// Daemon URL (ws://host:port)
    pub url: String,
    /// Connection timeout
    pub connect_timeout: Duration,
    /// Request timeout
    pub request_timeout: Duration,
    /// Client identifier (for logging)
    pub client_id: Option<String>,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            url: nanna_daemon::ipc::default_daemon_ws_url(),
            connect_timeout: Duration::from_secs(10),
            request_timeout: Duration::from_secs(30),
            client_id: None,
        }
    }
}

impl ClientConfig {
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            ..Default::default()
        }
    }
}

/// Connection state
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConnectionState {
    Disconnected,
    Connecting,
    Connected,
}

/// Pending request waiting for response
struct PendingRequest {
    tx: oneshot::Sender<std::result::Result<Value, ClientError>>,
}

/// Client for interacting with nanna-daemon
pub struct Client {
    config: ClientConfig,
    state: Arc<RwLock<ConnectionState>>,
    msg_tx: mpsc::Sender<Message>,
    pending: Arc<RwLock<HashMap<String, PendingRequest>>>,
    event_tx: broadcast::Sender<Event>,
    shutdown_tx: broadcast::Sender<()>,
}

impl Client {
    /// Connect to the daemon
    ///
    /// # Errors
    /// Returns [`ClientError::Connection`] when the WebSocket handshake with
    /// `config.url` fails, or [`ClientError::Timeout`] when it does not complete
    /// within `config.connect_timeout`.
    pub async fn connect(config: ClientConfig) -> Result<Self> {
        let (msg_tx, msg_rx) = mpsc::channel::<Message>(100);
        let (event_tx, _) = broadcast::channel::<Event>(100);
        let (shutdown_tx, _) = broadcast::channel::<()>(1);
        let pending = Arc::new(RwLock::new(HashMap::<String, PendingRequest>::new()));
        let state = Arc::new(RwLock::new(ConnectionState::Connecting));
        
        let client = Self {
            config: config.clone(),
            state: state.clone(),
            msg_tx,
            pending: pending.clone(),
            event_tx: event_tx.clone(),
            shutdown_tx: shutdown_tx.clone(),
        };
        
        // Connect
        info!("Connecting to {}", config.url);
        let ws = Self::do_connect(&config).await?;
        
        *state.write().await = ConnectionState::Connected;
        info!("Connected to daemon");
        
        // Spawn connection handler
        Self::spawn_handler(ws, msg_rx, pending, event_tx, state, shutdown_tx.subscribe());
        
        Ok(client)
    }
    
    async fn do_connect(config: &ClientConfig) -> Result<WebSocketStream<MaybeTlsStream<TcpStream>>> {
        // Read with the same limits the daemon reads with. tungstenite's
        // default is a 16 MiB frame, and the daemon's large replies — a whole
        // session's history, a full run state, an export — exceed it: a bare
        // `connect_async` dropped them as `Connection("Disconnected")`.
        let mut ws_config = WebSocketConfig::default();
        ws_config.max_message_size = Some(nanna_daemon::ipc::IPC_MAX_MESSAGE_BYTES);
        ws_config.max_frame_size = Some(nanna_daemon::ipc::IPC_MAX_MESSAGE_BYTES);
        let connect_future = connect_async_with_config(&config.url, Some(ws_config), false);
        
        match tokio::time::timeout(config.connect_timeout, connect_future).await {
            Ok(Ok((ws, _))) => Ok(ws),
            Ok(Err(e)) => Err(ClientError::Connection(e.to_string())),
            Err(_) => Err(ClientError::Timeout),
        }
    }
    
    fn spawn_handler(
        ws: WebSocketStream<MaybeTlsStream<TcpStream>>,
        mut msg_rx: mpsc::Receiver<Message>,
        pending: Arc<RwLock<HashMap<String, PendingRequest>>>,
        event_tx: broadcast::Sender<Event>,
        state: Arc<RwLock<ConnectionState>>,
        mut shutdown_rx: broadcast::Receiver<()>,
    ) {
        tokio::spawn(async move {
            let (mut ws_tx, mut ws_rx) = ws.split();
            
            loop {
                tokio::select! {
                    // Send outgoing messages
                    Some(msg) = msg_rx.recv() => {
                        if ws_tx.send(msg).await.is_err() {
                            break;
                        }
                    }
                    
                    // Receive incoming messages
                    Some(msg) = ws_rx.next() => {
                        match msg {
                            Ok(Message::Text(text)) => {
                                Self::handle_message(&text, &pending, &event_tx).await;
                            }
                            Ok(Message::Ping(data)) => {
                                let _ = ws_tx.send(Message::Pong(data)).await;
                            }
                            Ok(Message::Close(_)) => {
                                debug!("Server sent close");
                                break;
                            }
                            Err(e) => {
                                error!("WebSocket error: {}", e);
                                break;
                            }
                            _ => {}
                        }
                    }
                    
                    // Shutdown signal
                    _ = shutdown_rx.recv() => {
                        debug!("Shutdown signal received");
                        let _ = ws_tx.close().await;
                        break;
                    }
                }
            }
            
            *state.write().await = ConnectionState::Disconnected;
            info!("Disconnected from daemon");
            
            // Fail all pending requests
            let mut pending = pending.write().await;
            for (_, req) in pending.drain() {
                let _ = req.tx.send(Err(ClientError::Connection("Disconnected".to_string())));
            }
        });
    }
    
    async fn handle_message(
        text: &str,
        pending: &Arc<RwLock<HashMap<String, PendingRequest>>>,
        event_tx: &broadcast::Sender<Event>,
    ) {
        // Try to parse as Response first
        if let Ok(response) = serde_json::from_str::<Response>(text) {
            let req = pending.write().await.remove(&response.id);
            if let Some(req) = req {
                let result = match response.result {
                    ResponseResult::Success { data } => Ok(data),
                    ResponseResult::Error { code, message } => {
                        Err(ClientError::Server { code, message })
                    }
                };
                let _ = req.tx.send(result);
            } else {
                warn!("Received response for unknown request: {}", response.id);
            }
            return;
        }
        
        // Try to parse as Event
        if let Ok(event) = serde_json::from_str::<Event>(text) {
            let _ = event_tx.send(event);
            return;
        }
        
        warn!("Unknown message format: {}", text);
    }
    
    /// Get current connection state
    pub async fn state(&self) -> ConnectionState {
        self.state.read().await.clone()
    }
    
    /// Check if connected
    pub async fn is_connected(&self) -> bool {
        *self.state.read().await == ConnectionState::Connected
    }
    
    /// Subscribe to events
    #[must_use]
    pub fn subscribe_events(&self) -> broadcast::Receiver<Event> {
        self.event_tx.subscribe()
    }

    /// Subscribe to the events of one session only, filtering **locally**.
    ///
    /// This narrows what this stream yields, not what the daemon sends: the
    /// connection still receives every event and discards the rest here, by
    /// [`Event::session_id`]. That is deliberate — several of these streams can
    /// be held at once over one connection, which server-side narrowing would
    /// break. To also stop the other sessions' events leaving the daemon, call
    /// [`Self::narrow_to_session`], which affects the whole connection.
    #[must_use]
    pub fn subscribe_session(&self, session_id: impl Into<String>) -> SessionEvents {
        SessionEvents::new(self.event_tx.subscribe(), session_id.into())
    }

    /// Ask the daemon to stop sending this **connection** any session's events
    /// but the ones it names.
    ///
    /// Call it once per session of interest; each call adds to the set. Events
    /// that carry no session (config, memory, workspace, channel, connection)
    /// keep arriving — narrowing is about sessions, not about going quiet.
    ///
    /// This is connection-wide, so every `SessionEvents` stream held over the
    /// same connection is affected: a stream for a session that was never named
    /// will simply stop yielding. [`Self::widen_to_all_sessions`] undoes it.
    ///
    /// # Errors
    /// Returns [`ClientError::NotConnected`] when the client has no live
    /// connection, or [`ClientError::Request`] when the daemon does not hold
    /// `session_id`.
    pub async fn narrow_to_session(&self, session_id: impl Into<String>) -> Result<()> {
        let session_id = session_id.into();
        let response = self
            .request(Action::Subscribe(SubscribeAction::Session {
                session_id: session_id.clone(),
            }))
            .await?;
        if response.get("error").is_some() {
            return Err(ClientError::Request(format!(
                "daemon refused to narrow to session {session_id}: {response}"
            )));
        }
        Ok(())
    }

    /// Undo [`Self::narrow_to_session`]: this connection receives every
    /// session's events again.
    ///
    /// # Errors
    /// Returns [`ClientError::NotConnected`] when the client has no live
    /// connection.
    pub async fn widen_to_all_sessions(&self) -> Result<()> {
        self.request(Action::Subscribe(SubscribeAction::AllSessions))
            .await?;
        Ok(())
    }
    
    /// Send a request and wait for response
    ///
    /// # Errors
    /// - [`ClientError::NotConnected`] when the client is not connected.
    /// - [`ClientError::Protocol`] when the request cannot be serialised.
    /// - [`ClientError::Request`] when the connection handler has already shut
    ///   down (the outgoing queue is closed) or the response channel is dropped
    ///   without an answer.
    /// - [`ClientError::Connection`] when the connection drops while the request
    ///   is pending.
    /// - [`ClientError::Server`] when the daemon answers with an error.
    /// - [`ClientError::Timeout`] when no response arrives within
    ///   `config.request_timeout`.
    pub async fn request(&self, action: Action) -> Result<Value> {
        if !self.is_connected().await {
            return Err(ClientError::NotConnected);
        }
        
        let id = uuid::Uuid::new_v4().to_string();
        let request = Request { id: id.clone(), action };
        
        let json = serde_json::to_string(&request)
            .map_err(|e| ClientError::Protocol(e.to_string()))?;
        
        // Create response channel
        let (tx, rx) = oneshot::channel();
        
        // Register pending request
        {
            let mut pending = self.pending.write().await;
            pending.insert(id.clone(), PendingRequest { tx });
        }
        
        // Send request. A send that fails leaves nothing that will ever answer
        // the entry just registered, so it goes too — it used to stay in
        // `pending` for the life of the client.
        if let Err(e) = self.msg_tx.send(Message::Text(json.into())).await {
            self.pending.write().await.remove(&id);
            return Err(ClientError::Request(e.to_string()));
        }
        
        // Wait for response with timeout
        match tokio::time::timeout(self.config.request_timeout, rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(ClientError::Request("Response channel closed".to_string())),
            Err(_) => {
                // Remove pending request on timeout
                self.pending.write().await.remove(&id);
                Err(ClientError::Timeout)
            }
        }
    }
    
    /// Disconnect from daemon.
    ///
    /// On return the client reports `Disconnected` and `request()` fails fast with
    /// `NotConnected`. The state is set here rather than left to the handler task:
    /// signalling shutdown only *asks* that task to stop, so a caller that checked
    /// `is_connected()` (or issued a request) immediately after `disconnect()` could
    /// still observe `Connected` and send into a socket that is going away. The
    /// handler sets the same state when it winds down; doing it twice is harmless.
    pub async fn disconnect(&self) {
        *self.state.write().await = ConnectionState::Disconnected;
        let _ = self.shutdown_tx.send(());
        debug_assert!(!self.is_connected().await, "disconnect() leaves the client disconnected");
    }
    
    // =========================================================================
    // Convenience methods
    // =========================================================================
    
    /// Get sessions API
    #[must_use]
    pub const fn sessions(&self) -> SessionsApi<'_> {
        SessionsApi { client: self }
    }
    
    /// Get chat API
    #[must_use]
    pub const fn chat(&self) -> ChatApi<'_> {
        ChatApi { client: self }
    }
    
    /// Get memory API
    #[must_use]
    pub const fn memory(&self) -> MemoryApi<'_> {
        MemoryApi { client: self }
    }
    
    /// Get config API
    #[must_use]
    pub const fn config(&self) -> ConfigApi<'_> {
        ConfigApi { client: self }
    }
    
    /// Get tools API
    #[must_use]
    pub const fn tools(&self) -> ToolsApi<'_> {
        ToolsApi { client: self }
    }
    
    /// Get system API
    #[must_use]
    pub const fn system(&self) -> SystemApi<'_> {
        SystemApi { client: self }
    }

    /// Get scheduler API
    #[must_use]
    pub const fn scheduler(&self) -> SchedulerApi<'_> {
        SchedulerApi { client: self }
    }

    /// Get workspaces API
    #[must_use]
    pub const fn workspaces(&self) -> WorkspacesApi<'_> {
        WorkspacesApi { client: self }
    }

    /// Get channels API
    #[must_use]
    pub const fn channels(&self) -> ChannelsApi<'_> {
        ChannelsApi { client: self }
    }
}

// =============================================================================
// API wrappers for ergonomic usage
// =============================================================================

/// Sessions API
pub struct SessionsApi<'a> {
    client: &'a Client,
}

impl SessionsApi<'_> {
    pub async fn list(&self) -> Result<Value> {
        self.client.request(Action::Session(SessionAction::List)).await
    }
    
    pub async fn get(&self, id: &str) -> Result<Value> {
        self.client.request(Action::Session(SessionAction::Get { id: id.to_string() })).await
    }
    
    pub async fn create(&self, name: Option<String>) -> Result<Value> {
        self.client.request(Action::Session(SessionAction::Create { name })).await
    }
    
    pub async fn delete(&self, id: &str) -> Result<Value> {
        self.client.request(Action::Session(SessionAction::Delete { id: id.to_string() })).await
    }
    
    pub async fn rename(&self, id: &str, name: &str) -> Result<Value> {
        self.client.request(Action::Session(SessionAction::Rename { 
            id: id.to_string(), 
            name: name.to_string() 
        })).await
    }
    
    pub async fn clear(&self, id: &str) -> Result<Value> {
        self.client.request(Action::Session(SessionAction::Clear { id: id.to_string() })).await
    }
    
    pub async fn history(&self, id: &str, limit: Option<usize>) -> Result<Value> {
        self.client.request(Action::Session(SessionAction::History { 
            id: id.to_string(), 
            limit,
            before: None,
        })).await
    }

    /// Export a session as a document (Markdown or JSON), rendered by the
    /// daemon: `{format, filename, content}`, or `{error, message}`.
    pub async fn export(&self, id: &str, format: ExportFormat) -> Result<Value> {
        self.client
            .request(Action::Session(SessionAction::Export {
                id: id.to_string(),
                format,
            }))
            .await
    }
}

/// Chat API
pub struct ChatApi<'a> {
    client: &'a Client,
}

impl ChatApi<'_> {
    pub async fn send(&self, session_id: &str, content: &str) -> Result<Value> {
        self.client.request(Action::Chat(ChatAction::Send {
            session_id: session_id.to_string(),
            content: content.to_string(),
            attachments: vec![],
        })).await
    }
    
    pub async fn cancel(&self, session_id: &str) -> Result<Value> {
        self.client.request(Action::Chat(ChatAction::Cancel {
            session_id: session_id.to_string(),
        })).await
    }
    
    pub async fn regenerate(&self, session_id: &str) -> Result<Value> {
        self.client.request(Action::Chat(ChatAction::Regenerate {
            session_id: session_id.to_string(),
        })).await
    }
}

/// Memory API
pub struct MemoryApi<'a> {
    client: &'a Client,
}

impl MemoryApi<'_> {
    pub async fn search(&self, query: &str, limit: Option<usize>) -> Result<Value> {
        self.client.request(Action::Memory(MemoryAction::Search {
            query: query.to_string(),
            limit,
            scope: None,
        })).await
    }
    
    pub async fn get(&self, id: &str) -> Result<Value> {
        self.client.request(Action::Memory(MemoryAction::Get { id: id.to_string() })).await
    }
    
    pub async fn create(&self, content: &str, tags: Option<Vec<String>>, importance: Option<u8>) -> Result<Value> {
        self.client.request(Action::Memory(MemoryAction::Create {
            content: content.to_string(),
            tags,
            importance,
        })).await
    }
    
    pub async fn delete(&self, id: &str) -> Result<Value> {
        self.client.request(Action::Memory(MemoryAction::Delete { id: id.to_string() })).await
    }
    
    pub async fn stats(&self) -> Result<Value> {
        self.client.request(Action::Memory(MemoryAction::Stats)).await
    }

    /// Export memories as a document (Markdown or JSON), rendered by the
    /// daemon: `{format, filename, content, count}`, or `{error, message}`.
    pub async fn export(&self, scope: Option<String>, format: ExportFormat) -> Result<Value> {
        self.client
            .request(Action::Memory(MemoryAction::Export { scope, format }))
            .await
    }
}

/// Config API
pub struct ConfigApi<'a> {
    client: &'a Client,
}

impl ConfigApi<'_> {
    pub async fn get(&self, path: Option<&str>) -> Result<Value> {
        self.client.request(Action::Config(ConfigAction::Get { 
            path: path.map(String::from) 
        })).await
    }
    
    pub async fn set(&self, path: &str, value: Value) -> Result<Value> {
        self.client.request(Action::Config(ConfigAction::Set {
            path: path.to_string(),
            value,
        })).await
    }
    
    pub async fn reload(&self) -> Result<Value> {
        self.client.request(Action::Config(ConfigAction::Reload)).await
    }
    
    pub async fn export(&self) -> Result<Value> {
        self.client.request(Action::Config(ConfigAction::Export)).await
    }
}

/// Tools API
pub struct ToolsApi<'a> {
    client: &'a Client,
}

impl ToolsApi<'_> {
    pub async fn list(&self) -> Result<Value> {
        self.client.request(Action::Tool(ToolAction::List)).await
    }
    
    pub async fn get(&self, name: &str) -> Result<Value> {
        self.client.request(Action::Tool(ToolAction::Get { name: name.to_string() })).await
    }
    
    pub async fn execute(&self, name: &str, input: Value) -> Result<Value> {
        self.client.request(Action::Tool(ToolAction::Execute {
            name: name.to_string(),
            input,
            session_id: None,
        })).await
    }
}

/// System API
pub struct SystemApi<'a> {
    client: &'a Client,
}

impl SystemApi<'_> {
    pub async fn status(&self) -> Result<Value> {
        self.client.request(Action::System(SystemAction::Status)).await
    }
    
    pub async fn version(&self) -> Result<Value> {
        self.client.request(Action::System(SystemAction::Version)).await
    }
    
    pub async fn health(&self) -> Result<Value> {
        self.client.request(Action::System(SystemAction::Health)).await
    }
    
    pub async fn restart(&self) -> Result<Value> {
        self.client.request(Action::System(SystemAction::Restart)).await
    }
    
    pub async fn shutdown(&self) -> Result<Value> {
        self.client.request(Action::System(SystemAction::Shutdown)).await
    }
}

/// Scheduler API: cron jobs, owned and run by the daemon.
pub struct SchedulerApi<'a> {
    client: &'a Client,
}

impl SchedulerApi<'_> {
    pub async fn list(&self) -> Result<Value> {
        self.client
            .request(Action::Scheduler(SchedulerAction::List))
            .await
    }

    pub async fn get(&self, id: &str) -> Result<Value> {
        let id = id.to_string();
        self.client
            .request(Action::Scheduler(SchedulerAction::Get { id }))
            .await
    }

    /// Add a cron job: `schedule` is a cron expression, `task` the prompt it runs.
    pub async fn add(&self, schedule: &str, task: &str, name: Option<&str>) -> Result<Value> {
        debug_assert!(!schedule.trim().is_empty(), "a job needs a schedule");
        debug_assert!(!task.trim().is_empty(), "a job needs a task");
        self.client
            .request(Action::Scheduler(SchedulerAction::Add {
                schedule: schedule.to_string(),
                task: task.to_string(),
                name: name.map(String::from),
                session_id: None,
            }))
            .await
    }

    /// Change a job's schedule and/or enabled state.
    ///
    /// The protocol also carries a `task`, but the daemon does not apply payload
    /// updates yet, so this wrapper does not offer an argument that would be ignored.
    pub async fn update(
        &self,
        id: &str,
        schedule: Option<&str>,
        enabled: Option<bool>,
    ) -> Result<Value> {
        self.client
            .request(Action::Scheduler(SchedulerAction::Update {
                id: id.to_string(),
                schedule: schedule.map(String::from),
                task: None,
                enabled,
            }))
            .await
    }

    pub async fn remove(&self, id: &str) -> Result<Value> {
        let id = id.to_string();
        self.client
            .request(Action::Scheduler(SchedulerAction::Remove { id }))
            .await
    }

    pub async fn run_now(&self, id: &str) -> Result<Value> {
        let id = id.to_string();
        self.client
            .request(Action::Scheduler(SchedulerAction::RunNow { id }))
            .await
    }

    pub async fn history(&self, id: &str, limit: Option<usize>) -> Result<Value> {
        let id = id.to_string();
        self.client
            .request(Action::Scheduler(SchedulerAction::History { id, limit }))
            .await
    }
}

/// Workspaces API: the daemon's registry of project directories.
pub struct WorkspacesApi<'a> {
    client: &'a Client,
}

impl WorkspacesApi<'_> {
    pub async fn list(&self) -> Result<Value> {
        self.client
            .request(Action::Workspace(WorkspaceAction::List))
            .await
    }

    pub async fn get(&self, id: &str) -> Result<Value> {
        let id = id.to_string();
        self.client
            .request(Action::Workspace(WorkspaceAction::Get { id }))
            .await
    }

    /// Register `path` as a workspace (the daemon creates its `.nanna` folder if absent).
    pub async fn open(&self, path: &str) -> Result<Value> {
        debug_assert!(!path.trim().is_empty(), "a workspace needs a path");
        let path = path.to_string();
        self.client
            .request(Action::Workspace(WorkspaceAction::Open { path }))
            .await
    }

    pub async fn close(&self, id: &str) -> Result<Value> {
        let id = id.to_string();
        self.client
            .request(Action::Workspace(WorkspaceAction::Close { id }))
            .await
    }

    pub async fn set_active(&self, id: &str) -> Result<Value> {
        let id = id.to_string();
        self.client
            .request(Action::Workspace(WorkspaceAction::SetActive { id }))
            .await
    }

    pub async fn clear_active(&self) -> Result<Value> {
        self.client
            .request(Action::Workspace(WorkspaceAction::ClearActive))
            .await
    }

    pub async fn reload(&self, id: &str) -> Result<Value> {
        let id = id.to_string();
        self.client
            .request(Action::Workspace(WorkspaceAction::Reload { id }))
            .await
    }

    pub async fn context(&self, id: &str) -> Result<Value> {
        let id = id.to_string();
        self.client
            .request(Action::Workspace(WorkspaceAction::GetContext { id }))
            .await
    }

    pub async fn update_context(&self, id: &str, file: &str, content: &str) -> Result<Value> {
        self.client
            .request(Action::Workspace(WorkspaceAction::UpdateContext {
                id: id.to_string(),
                file: file.to_string(),
                content: content.to_string(),
            }))
            .await
    }
}

/// Channels API: the chat adapters (Telegram, Discord, Slack, Signal, `WhatsApp`).
///
/// `list` and `status` are live. The daemon still answers `enable`, `disable`, `test`
/// and `send` with `"status": "not_implemented"`; they are wrapped so a client can
/// already speak the whole protocol, and they start working when the daemon does.
pub struct ChannelsApi<'a> {
    client: &'a Client,
}

impl ChannelsApi<'_> {
    pub async fn list(&self) -> Result<Value> {
        self.client
            .request(Action::Channel(ChannelAction::List))
            .await
    }

    /// Status of one channel, or of all of them (with a summary) when `id` is `None`.
    pub async fn status(&self, id: Option<&str>) -> Result<Value> {
        let id = id.map(String::from);
        self.client
            .request(Action::Channel(ChannelAction::Status { id }))
            .await
    }

    pub async fn enable(&self, id: &str) -> Result<Value> {
        let id = id.to_string();
        self.client
            .request(Action::Channel(ChannelAction::Enable { id }))
            .await
    }

    pub async fn disable(&self, id: &str) -> Result<Value> {
        let id = id.to_string();
        self.client
            .request(Action::Channel(ChannelAction::Disable { id }))
            .await
    }

    pub async fn test(&self, id: &str) -> Result<Value> {
        let id = id.to_string();
        self.client
            .request(Action::Channel(ChannelAction::Test { id }))
            .await
    }

    pub async fn send(&self, channel_id: &str, target: &str, content: &str) -> Result<Value> {
        self.client
            .request(Action::Channel(ChannelAction::Send {
                channel_id: channel_id.to_string(),
                target: target.to_string(),
                content: content.to_string(),
            }))
            .await
    }
}

/// The events of one session, filtered out of the client's shared event stream.
pub struct SessionEvents {
    receiver: broadcast::Receiver<Event>,
    session_id: String,
}

impl SessionEvents {
    fn new(receiver: broadcast::Receiver<Event>, session_id: String) -> Self {
        debug_assert!(
            !session_id.is_empty(),
            "a session filter needs a session id"
        );
        Self {
            receiver,
            session_id,
        }
    }

    /// The session this stream carries.
    #[must_use]
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// The next event for this session; other sessions' and session-less events are
    /// skipped.
    ///
    /// Lag is reported, not hidden: the shared stream is a bounded broadcast, and a
    /// slow reader that falls behind gets `RecvError::Lagged(n)` so it can decide to
    /// resync rather than silently missing events of its own session.
    ///
    /// # Errors
    ///
    /// `Lagged` when this reader fell behind the shared stream, `Closed` when the
    /// client shut down.
    pub async fn recv(&mut self) -> std::result::Result<Event, broadcast::error::RecvError> {
        loop {
            let event = self.receiver.recv().await?;
            if event.session_id() == Some(self.session_id.as_str()) {
                return Ok(event);
            }
        }
    }
}

#[cfg(test)]
mod session_events_tests {
    use super::{Event, SessionEvents};
    use tokio::sync::broadcast;

    fn delta(session_id: &str, text: &str) -> Event {
        Event::MessageDelta {
            session_id: session_id.to_string(),
            message_id: "m".to_string(),
            delta: text.to_string(),
        }
    }

    #[tokio::test]
    async fn only_the_subscribed_session_comes_through() {
        let (tx, rx) = broadcast::channel(16);
        let mut events = SessionEvents::new(rx, "mine".to_string());
        tx.send(delta("other", "not for me"))
            .expect("a receiver exists");
        tx.send(Event::ConfigChanged).expect("a receiver exists");
        tx.send(delta("mine", "for me")).expect("a receiver exists");

        match events.recv().await.expect("an event for this session") {
            Event::MessageDelta { delta, .. } => assert_eq!(delta, "for me"),
            other => panic!("expected this session's delta, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_lagging_reader_is_told_rather_than_silently_skipping() {
        let (tx, rx) = broadcast::channel(2);
        let mut events = SessionEvents::new(rx, "mine".to_string());
        for n in 0..5 {
            tx.send(delta("mine", &n.to_string()))
                .expect("a receiver exists");
        }
        assert!(
            matches!(
                events.recv().await,
                Err(broadcast::error::RecvError::Lagged(3))
            ),
            "a bounded stream that overflowed must say how much was lost"
        );
    }
}
