#![warn(clippy::all)]
#![warn(clippy::pedantic, clippy::nursery)]
// The auto-trait solver proves `DaemonServer::run`'s scheduler closure is
// `Send` by walking the whole state tuple, and that walk descends through
// `MemoryService` -> `VectorStore` -> `CosineSimilaritySearch` -> wgpu's
// `Global`/`Hub`/`Registry` graph, which is deeper than the default limit of
// 128. nightly-2026-08-25 turned the resulting overflow into the
// future-incompatible `recursion_depth_exceeding_limit` warning (rust#159228)
// — it is scheduled to become a hard error, so the limit is raised here rather
// than left to break a future toolchain bump. This costs solver depth only;
// it changes no behaviour and no generated code.
#![recursion_limit = "256"]

//! Nanna Daemon - Headless background service
//!
//! The daemon owns all state and exposes it via IPC to channel clients (GUI, CLI, etc.).
//! Every channel (including GUI) is a full control plane client with access to:
//! - Session management
//! - Memory browsing/editing
//! - Configuration
//! - Tool management
//! - Scheduler/cron

// `ActivityClock` now lives in `nanna-memory`, beside the dreaming code that
// reads it, so the control plane and `DreamingService` share **one** clock
// instead of each keeping a private notion of "last activity". Re-exported
// below because the daemon is where it is stamped.
pub mod agent_service;
pub mod channels;
pub mod control;
pub mod dream_summarizer;
pub mod embedding_reload;
pub mod embedding_router;
pub mod exit_reason;
pub mod export;
pub mod health;
pub mod liveness;
pub mod ipc;
#[cfg(windows)]
pub mod job;
pub mod keyword_recall;
pub mod llm_router;
// `log_buffer` now lives in `nanna-core` so pure daemon clients (the GUI) can
// capture their own lines without linking this crate. Re-exported here so
// `crate::log_buffer::…` and `nanna_daemon::log_buffer::…` paths keep resolving.
pub use nanna_core::log_buffer;
pub mod log_file;
pub mod memory_adapter;
pub mod memory_persistence;
pub mod memory_write_through;
mod numeric;
pub mod persistence;
pub mod protocol;
pub mod server;
pub mod service;
pub mod session;
pub mod shutdown;
pub mod task_event_bridge;
pub mod tasks;
pub mod tool_authoring;
pub mod vision_service;
pub mod audio_service;
pub mod browser_service;
pub mod screenshot_service;
pub mod reminder_service;
pub mod mcp_startup;
pub mod file_history_service;
pub mod metrics;
pub mod channel_counters;
pub mod ask_user_service;
pub mod cost_rollup;
pub mod user_tools;
pub mod validate_api_key;
pub mod webhook;

#[cfg(windows)]
pub mod windows_service;

pub use ipc::{IpcServer, IpcServerConfig, ConnectionId, DEFAULT_IPC_PORT};
pub use protocol::{Request, Response, Event, Action, SessionAction, MemoryAction, ConfigAction};
pub use server::{
    DaemonConfig, DaemonServer, SchedulerSwitches, ServerSwitches, ToolAuditSwitches,
};
pub use session::{Session, SessionManager, SessionId};
pub use control::ControlPlane;
pub use nanna_memory::ActivityClock;
pub use health::{PidFile, PidFileError, HealthServer, HealthState, DEFAULT_HEALTH_PORT};
pub use webhook::{WebhookServer, WebhookConfig, WebhookEvent, WebhookMessage, DEFAULT_WEBHOOK_PORT};
pub use log_buffer::{LogBuffer, LogEntry, LogSource};

/// Lines the daemon's in-memory log buffer holds: the tail `system.logs` serves
/// to a client (the GUI's Logs page). One value for every daemon entry point.
pub const LOG_BUFFER_LINES: usize = 5000;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum DaemonError {
    #[error("IPC error: {0}")]
    Ipc(String),
    
    #[error("Session error: {0}")]
    Session(String),
    
    #[error("Control error: {0}")]
    Control(String),
    
    #[error("Config error: {0}")]
    Config(String),
    
    #[error("Storage error: {0}")]
    Storage(String),
    
    #[error("Already running")]
    AlreadyRunning,
    
    #[error("Not running")]
    NotRunning,
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
