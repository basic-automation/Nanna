#![warn(clippy::all)]
#![warn(clippy::pedantic, clippy::nursery)]
// Raised for the same reason as `nanna-daemon`'s: proving a future or closure
// is `Send` walks into wgpu's `Global`/`Hub`/`Registry` graph by way of
// `CosineSimilaritySearch`, which is deeper than the default limit of 128.
// nightly-2026-08-25 turned that overflow into the future-incompatible
// `recursion_depth_exceeding_limit` warning (rust#159228), which is scheduled
// to become a hard error. Solver depth only — no behaviour, no codegen change.
#![recursion_limit = "256"]

//! HTTP server for Nanna
//!
//! Provides REST API and webhook endpoints for channel integrations.

mod routes;
mod state;
mod webhooks;

pub use routes::create_router;
pub use state::{AppState, AppStateBuilder, ProcessedReply};

use std::net::SocketAddr;
use tokio::net::TcpListener;
use tower_http::compression::CompressionLayer;
use tower_http::trace::TraceLayer;
use tracing::info;

/// Server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub webhook_secret: Option<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            // Loopback by default — this HTTP surface has no authentication of
            // its own, so publishing it to every interface must be an explicit
            // choice, not an inherited default.
            host: nanna_config::LOOPBACK_HOST.to_string(),
            port: 3000,
            webhook_secret: None,
        }
    }
}

/// Start the HTTP server.
///
/// # Errors
///
/// Returns an error if the server fails to bind or start.
pub async fn start_server(config: ServerConfig, state: AppState) -> anyhow::Result<()> {
    // Start the scheduler for periodic tasks (dreaming, heartbeats)
    state.start_scheduler().await;

    // No CORS layer: `/api` runs the agent with its tools, and a permissive
    // one (`allow_origin(Any)`) let any web page the user opened POST to it
    // and read the reply. Native callers send no `Origin` and need none.
    let app = create_router(state.clone())
        .layer(TraceLayer::new_for_http())
        .layer(CompressionLayer::new());

    let addr: SocketAddr = format!("{}:{}", config.host, config.port).parse()?;
    let listener = TcpListener::bind(addr).await?;
    
    if nanna_config::is_loopback_host(&config.host) {
        info!("🚀 Server listening on {}", addr);
    } else {
        tracing::warn!(
            "🚀 Server listening on {addr} — this HTTP surface has NO authentication \
             and is reachable from other machines. Bind {} unless you intend to \
             expose it.",
            nanna_config::LOOPBACK_HOST
        );
    }
    info!("🧠 Memory dreaming enabled");
    
    axum::serve(listener, app).await?;

    // Stop scheduler on shutdown
    state.stop_scheduler().await;
    
    Ok(())
}
