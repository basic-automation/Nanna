//! The deprecated HTTP+SSE transport (MCP 2024-11-05), as the last fallback
//! for a server URL.
//!
//! The client `GET`s the server URL and keeps the stream open; the server's
//! first event is `endpoint`, naming where to `POST` messages; every answer
//! then arrives as a `message` event on that one stream. Deprecated since
//! 2025-03-26 and eligible for removal, but older remote servers still speak
//! only this, and the Streamable HTTP binding tells a client to fall back to
//! it when a `POST` gets `400`/`404`/`405` without a modern error body.
//!
//! This replaced the old `HttpTransport` (which assumed `<url>/sse`,
//! "waited" for the endpoint with a sleep, and was deleted 2026-09-26 once
//! nothing used it); it is what the daemon uses. Source:
//! <https://modelcontextprotocol.io/specification/2024-11-05/basic/transports#http-with-sse>.

use crate::streamable_http::{HTTP_REQUEST_TIMEOUT, SseParser};
use crate::transport::reply_to_server_request;
use crate::{
    JsonRpcNotification, JsonRpcRequest, JsonRpcResponse, ListChangedFlags, McpError, McpList,
    Result, Transport,
};
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, oneshot};
use tracing::{debug, warn};

/// How long the server has to announce its message endpoint.
const ENDPOINT_TIMEOUT: Duration = Duration::from_secs(10);

/// Most requests awaiting an answer at once. A client sends one at a time
/// per tool call; the bound only stops a leak from growing without limit.
const PENDING_MAX: usize = 256;

/// Requests awaiting their answer on the stream, by id; `None` once the
/// stream has ended, so a later request fails at once instead of waiting
/// [`HTTP_REQUEST_TIMEOUT`] for an answer nothing can deliver.
type Pending = Arc<Mutex<Option<HashMap<String, oneshot::Sender<JsonRpcResponse>>>>>;

/// The 2024-11-05 HTTP+SSE transport.
pub struct LegacySseTransport {
    /// For the message `POST`s (bounded by [`HTTP_REQUEST_TIMEOUT`]).
    client: reqwest::Client,
    /// Where messages go, as the server's `endpoint` event named it.
    post_url: String,
    bearer_token: Option<String>,
    pending: Pending,
    list_changed: Arc<ListChangedFlags>,
    /// Ends the stream task when sent to, or when the transport is dropped.
    stop: tokio::sync::watch::Sender<bool>,
}

impl LegacySseTransport {
    /// Open the stream at `url` and wait for its `endpoint` event.
    ///
    /// # Errors
    ///
    /// [`McpError::HttpStatus`] if the `GET` is refused, [`McpError::Protocol`]
    /// if the answer is not an event stream or names no endpoint within
    /// 10 s, [`McpError::Transport`] if the server cannot be reached.
    pub async fn connect(url: &str, bearer_token: Option<String>) -> Result<Self> {
        let base = reqwest::Url::parse(url)
            .map_err(|e| McpError::Transport(format!("invalid MCP URL `{url}`: {e}")))?;
        let stream_client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .build()
            .map_err(|e| McpError::Transport(e.to_string()))?;
        let mut get = stream_client
            .get(base.clone())
            .header(reqwest::header::ACCEPT, "text/event-stream");
        if let Some(token) = &bearer_token {
            get = get.bearer_auth(token);
        }
        let response = get
            .send()
            .await
            .map_err(|e| McpError::Transport(e.to_string()))?;
        let status = response.status();
        let is_sse = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|ct| ct.starts_with("text/event-stream"));
        if !status.is_success() {
            return Err(McpError::HttpStatus {
                status: status.as_u16(),
                body: String::new(),
            });
        }
        if !is_sse {
            return Err(McpError::Protocol(format!(
                "`{url}` answered GET without an event stream; not an MCP server"
            )));
        }
        let mut stream = Box::pin(response.bytes_stream());
        let mut parser = SseParser::default();
        let endpoint = tokio::time::timeout(ENDPOINT_TIMEOUT, async {
            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(|e| McpError::Transport(e.to_string()))?;
                for event in parser.push_events(&chunk)? {
                    if event.event.as_deref() == Some("endpoint") {
                        return Ok(event.data);
                    }
                }
            }
            Err(McpError::ConnectionClosed)
        })
        .await
        .map_err(|_| McpError::Protocol(format!("`{url}` named no message endpoint")))??;
        let post_url = same_origin_endpoint(&base, &endpoint)?;
        debug!(post_url, "Legacy MCP SSE endpoint");

        let client = reqwest::Client::builder()
            .timeout(HTTP_REQUEST_TIMEOUT)
            .build()
            .map_err(|e| McpError::Transport(e.to_string()))?;
        let pending: Pending = Arc::new(Mutex::new(Some(HashMap::new())));
        let list_changed = Arc::new(ListChangedFlags::default());
        let (stop, stop_rx) = tokio::sync::watch::channel(false);
        let reader = Reader {
            client: client.clone(),
            post_url: post_url.clone(),
            bearer_token: bearer_token.clone(),
            pending: Arc::clone(&pending),
            list_changed: Arc::clone(&list_changed),
        };
        tokio::spawn(reader.run(stream, parser, stop_rx));
        Ok(Self {
            client,
            post_url,
            bearer_token,
            pending,
            list_changed,
            stop,
        })
    }

    /// Stop waiting for `id`'s answer.
    async fn forget(&self, id: &str) {
        if let Some(waiting) = self.pending.lock().await.as_mut() {
            waiting.remove(id);
        }
    }

    async fn post(&self, body: String) -> Result<()> {
        post_message(
            &self.client,
            &self.post_url,
            self.bearer_token.as_deref(),
            body,
        )
        .await
    }
}

/// `POST` one message to the endpoint; any 2xx (typically `202`) is accepted.
async fn post_message(
    client: &reqwest::Client,
    url: &str,
    bearer_token: Option<&str>,
    body: String,
) -> Result<()> {
    let mut builder = client
        .post(url)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(body);
    if let Some(token) = bearer_token {
        builder = builder.bearer_auth(token);
    }
    let response = builder
        .send()
        .await
        .map_err(|e| McpError::Transport(e.to_string()))?;
    let status = response.status();
    if status.is_success() {
        return Ok(());
    }
    let body = crate::streamable_http::read_error_body(response, 2048).await;
    Err(McpError::HttpStatus {
        status: status.as_u16(),
        body: body.chars().take(512).collect(),
    })
}

/// The stream task: routes every `message` event.
struct Reader {
    client: reqwest::Client,
    post_url: String,
    bearer_token: Option<String>,
    pending: Pending,
    list_changed: Arc<ListChangedFlags>,
}

impl Reader {
    async fn run<S, B>(
        self,
        mut stream: S,
        mut parser: SseParser,
        mut stop: tokio::sync::watch::Receiver<bool>,
    ) where
        S: futures::Stream<Item = reqwest::Result<B>> + Unpin + Send,
        B: AsRef<[u8]>,
    {
        loop {
            let chunk = tokio::select! {
                _ = stop.changed() => break,
                chunk = stream.next() => chunk,
            };
            let Some(Ok(chunk)) = chunk else {
                debug!("Legacy MCP SSE stream ended");
                break;
            };
            let Ok(events) = parser.push_events(chunk.as_ref()) else {
                warn!("Legacy MCP SSE stream sent an oversized or non-UTF-8 event; closing");
                break;
            };
            for event in events {
                self.route(&event.data).await;
            }
        }
        // Nothing will answer the requests still waiting, or any later one:
        // fail them now, and refuse the rest (dropping the senders ends
        // every waiter with `ConnectionClosed`).
        *self.pending.lock().await = None;
    }

    async fn route(&self, data: &str) {
        let Ok(message) = serde_json::from_str::<Value>(data) else {
            return;
        };
        let method = message.get("method").and_then(Value::as_str);
        let id = message.get("id").filter(|id| !id.is_null());
        match (method, id) {
            (Some(method), Some(id)) => {
                let reply = reply_to_server_request(id, method).to_string();
                if let Err(e) = post_message(
                    &self.client,
                    &self.post_url,
                    self.bearer_token.as_deref(),
                    reply,
                )
                .await
                {
                    warn!(error = %e, method, "Could not answer an MCP server request");
                }
            }
            (Some(method), None) => {
                let list = match method {
                    "notifications/tools/list_changed" => Some(McpList::Tools),
                    "notifications/resources/list_changed" => Some(McpList::Resources),
                    "notifications/prompts/list_changed" => Some(McpList::Prompts),
                    _ => None,
                };
                if let Some(list) = list {
                    self.list_changed.mark(list);
                }
            }
            (None, _) => {
                if let Ok(response) = serde_json::from_value::<JsonRpcResponse>(message)
                    && let Some(waiter) = self
                        .pending
                        .lock()
                        .await
                        .as_mut()
                        .and_then(|waiting| waiting.remove(&response.id.to_string()))
                {
                    let _ = waiter.send(response);
                }
            }
        }
    }
}

#[async_trait]
impl Transport for LegacySseTransport {
    async fn request(&self, request: JsonRpcRequest) -> Result<JsonRpcResponse> {
        let id = request.id.to_string();
        let (tx, rx) = oneshot::channel();
        let mut guard = self.pending.lock().await;
        let Some(waiting) = guard.as_mut() else {
            return Err(McpError::ConnectionClosed);
        };
        if waiting.len() >= PENDING_MAX {
            return Err(McpError::Protocol(format!(
                "{PENDING_MAX} MCP requests already await an answer"
            )));
        }
        waiting.insert(id.clone(), tx);
        drop(guard);
        if let Err(e) = self.post(serde_json::to_string(&request)?).await {
            self.forget(&id).await;
            return Err(e);
        }
        match tokio::time::timeout(HTTP_REQUEST_TIMEOUT, rx).await {
            Ok(Ok(response)) => Ok(response),
            Ok(Err(_)) => Err(McpError::ConnectionClosed),
            Err(_) => {
                self.forget(&id).await;
                Err(McpError::Timeout)
            }
        }
    }

    async fn notify(&self, notification: JsonRpcNotification) -> Result<()> {
        self.post(serde_json::to_string(&notification)?).await
    }

    async fn close(&self) -> Result<()> {
        let _ = self.stop.send(true);
        Ok(())
    }

    fn list_changed_flags(&self) -> Option<Arc<ListChangedFlags>> {
        Some(Arc::clone(&self.list_changed))
    }
}

/// The message endpoint a legacy SSE server named, resolved against the URL it
/// was reached at — refused unless it is on that same origin.
///
/// Every POST to it carries the server's bearer token, so taking the
/// `endpoint` event's URL as given let a server (or anything on its path)
/// redirect the configured credential to another host: `data:
/// https://attacker.example/x`, or the scheme-relative `//attacker.example/x`.
/// The TypeScript SDK refuses a cross-origin endpoint for the same reason.
fn same_origin_endpoint(base: &reqwest::Url, endpoint: &str) -> Result<String> {
    let joined = base
        .join(endpoint.trim())
        .map_err(|e| McpError::Protocol(format!("bad endpoint `{endpoint}`: {e}")))?;
    if joined.origin() != base.origin() {
        return Err(McpError::Protocol(format!(
            "the server named a message endpoint on another origin (`{}`, connected to `{}`); \
             refusing to send its credential there",
            joined.origin().ascii_serialization(),
            base.origin().ascii_serialization()
        )));
    }
    debug_assert_eq!(joined.origin(), base.origin(), "same origin only");
    Ok(joined.to_string())
}

#[cfg(test)]
mod origin_tests {
    use super::same_origin_endpoint;

    #[test]
    fn a_message_endpoint_must_stay_on_the_connections_origin() {
        let base = reqwest::Url::parse("https://mcp.example.com/sse").expect("url");
        assert_eq!(
            same_origin_endpoint(&base, "/messages?session=1").expect("relative"),
            "https://mcp.example.com/messages?session=1"
        );
        assert!(same_origin_endpoint(&base, "https://mcp.example.com/m").is_ok());
        for elsewhere in [
            "https://attacker.example/x",
            "//attacker.example/x",
            "http://mcp.example.com/m",
            "https://mcp.example.com:8443/m",
        ] {
            assert!(
                same_origin_endpoint(&base, elsewhere).is_err(),
                "{elsewhere}"
            );
        }
    }
}

#[cfg(test)]
mod stream_end_tests {
    use super::LegacySseTransport;
    use crate::transport::Transport as _;
    use crate::{JsonRpcRequest, McpError};

    /// Once the server's event stream has ended, a request fails at once:
    /// it used to be posted and then wait the full 60 s for an answer the
    /// dead stream could never deliver.
    #[tokio::test]
    async fn a_request_after_the_stream_ended_fails_at_once() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("addr");
        tokio::spawn(async move {
            if let Ok((mut socket, _)) = listener.accept().await {
                let mut request = [0_u8; 1024];
                let _ = socket.read(&mut request).await;
                let _ = socket
                    .write_all(
                        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
                          Connection: close\r\n\r\nevent: endpoint\ndata: /messages\n\n",
                    )
                    .await;
                let _ = socket.shutdown().await;
            }
        });
        let transport = LegacySseTransport::connect(&format!("http://{addr}/sse"), None)
            .await
            .expect("the endpoint event arrives before the stream ends");
        // Let the reader see the end of the stream.
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
        while transport.pending.lock().await.is_some() {
            assert!(tokio::time::Instant::now() < deadline, "the reader never saw the end");
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        let started = tokio::time::Instant::now();
        let refused = transport
            .request(JsonRpcRequest::new(1_i64, "tools/list", None))
            .await;
        assert!(matches!(refused, Err(McpError::ConnectionClosed)), "{refused:?}");
        assert!(started.elapsed() < std::time::Duration::from_secs(1));
    }
}
