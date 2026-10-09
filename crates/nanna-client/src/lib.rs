#![warn(clippy::pedantic, clippy::nursery, clippy::all)]

//! Nanna Client - Library for connecting to the daemon
//!
//! Provides a high-level async API for interacting with nanna-daemon.
//! Used by GUI, CLI, and API clients.
//!
//! # Example
//!
//! ```ignore
//! use nanna_client::{Client, ClientConfig};
//!
//! let client = Client::connect(ClientConfig::default()).await?;
//!
//! // List sessions
//! let sessions = client.sessions().list().await?;
//!
//! // Send a message
//! let response = client.chat().send("session-id", "Hello!").await?;
//!
//! // Subscribe to events
//! let mut events = client.subscribe_session("session-id").await?;
//! while let Some(event) = events.next().await {
//!     println!("Event: {:?}", event);
//! }
//! ```

mod connection;
mod link;

pub use connection::{Client, ClientConfig, ConnectionState};
pub use link::DaemonLink;
pub use nanna_daemon::protocol::*;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum ClientError {
    #[error("Connection failed: {0}")]
    Connection(String),
    
    #[error("Request failed: {0}")]
    Request(String),
    
    #[error("Not connected")]
    NotConnected,
    
    #[error("Timeout")]
    Timeout,
    
    #[error("Protocol error: {0}")]
    Protocol(String),
    
    #[error("Server error: {code} - {message}")]
    Server { code: String, message: String },
}

pub type Result<T> = std::result::Result<T, ClientError>;
