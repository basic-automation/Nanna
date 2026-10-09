//! A daemon connection that comes back after the daemon does.

use crate::{Client, ClientConfig, ClientError, Result};
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;

/// A [`Client`] that is reopened when the daemon restarts.
///
/// A plain `Client` that lost its daemon stays disconnected: every later
/// request fails with `NotConnected` for the life of the process. A
/// long-lived proxy (`nanna mcp serve`) needs the next call after a daemon
/// restart to work, so each call here first makes sure the connection is up,
/// opening a new one if it is not.
///
/// A request is never re-sent once it went out: a tool call the daemon may
/// already have run must not run twice. Only a call that found the connection
/// down before sending is retried, once, on a fresh connection.
pub struct DaemonLink {
    config: ClientConfig,
    client: Mutex<Arc<Client>>,
}

impl DaemonLink {
    /// Wrap an open `client`; `config` is how to reopen it.
    #[must_use]
    pub fn new(client: Client, config: ClientConfig) -> Self {
        Self {
            config,
            client: Mutex::new(Arc::new(client)),
        }
    }

    /// The live client, reconnecting first when the current one has dropped.
    ///
    /// # Errors
    ///
    /// When the daemon cannot be reached.
    pub async fn client(&self) -> Result<Arc<Client>> {
        let mut current = self.client.lock().await;
        if !current.is_connected().await {
            let fresh = tokio::time::timeout(
                self.config.connect_timeout,
                Client::connect(self.config.clone()),
            )
            .await
            .map_err(|_| ClientError::Timeout)??;
            *current = Arc::new(fresh);
        }
        let live = Arc::clone(&current);
        // Held through the reconnect so two calls cannot both reopen it.
        drop(current);
        Ok(live)
    }

    /// [`crate::Client`]'s `tools().execute_until_answered`, over a live
    /// connection.
    ///
    /// # Errors
    ///
    /// When the daemon cannot be reached, or as the request itself.
    pub async fn execute_tool(&self, name: &str, input: Value) -> Result<Value> {
        let client = self.client().await?;
        match client.tools().execute_until_answered(name, input.clone()).await {
            // Dropped between the check and the send: nothing was sent.
            Err(ClientError::NotConnected) => {
                self.client()
                    .await?
                    .tools()
                    .execute_until_answered(name, input)
                    .await
            }
            other => other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DaemonLink;
    use crate::{Client, ClientConfig};
    use futures_util::{SinkExt, StreamExt};
    use std::time::Duration;

    /// A daemon that answers one request per connection, then hangs up —
    /// a daemon restarting between calls.
    async fn one_answer_per_connection(connections: usize) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("addr");
        tokio::spawn(async move {
            for n in 0..connections {
                let Ok((socket, _)) = listener.accept().await else {
                    return;
                };
                let Ok(mut ws) = tokio_tungstenite::accept_async(socket).await else {
                    return;
                };
                while let Some(Ok(message)) = ws.next().await {
                    let Ok(text) = message.to_text() else {
                        continue;
                    };
                    let Ok(request) = serde_json::from_str::<serde_json::Value>(text) else {
                        continue;
                    };
                    let id = request["id"].as_str().unwrap_or_default().to_string();
                    let reply = nanna_daemon::protocol::Response::success(
                        id,
                        serde_json::json!({ "connection": n }),
                    );
                    let json = serde_json::to_string(&reply).expect("serializes");
                    let _ = ws.send(json.into()).await;
                    let _ = ws.close(None).await;
                    break;
                }
            }
        });
        format!("ws://{addr}")
    }

    #[tokio::test]
    async fn the_next_call_after_a_daemon_restart_works() {
        let url = one_answer_per_connection(2).await;
        let config = ClientConfig::new(url);
        let client = Client::connect(config.clone()).await.expect("connect");
        let link = DaemonLink::new(client, config);

        let first = link.execute_tool("t", serde_json::json!({})).await.expect("first");
        assert_eq!(first["connection"], 0);
        // Let the client notice the hang-up.
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        while link.client.lock().await.is_connected().await {
            assert!(tokio::time::Instant::now() < deadline, "the hang-up was never seen");
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let second = link
            .execute_tool("t", serde_json::json!({}))
            .await
            .expect("answered over a new connection");
        assert_eq!(second["connection"], 1);
    }
}
