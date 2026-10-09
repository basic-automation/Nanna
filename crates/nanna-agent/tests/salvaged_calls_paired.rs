#![warn(clippy::pedantic, clippy::nursery, clippy::all)]
//! A prose tool call the loop salvages is stored as a real `tool_use`, so an
//! exit before it runs must pair it with a result like any structured call.
//!
//! The token-budget (and Stop) exit paired only the structured calls: a
//! salvaged call stayed in history with no `tool_result`, which Anthropic
//! rejects on the next request. This drives the real loop against a scripted
//! Ollama stub: the first reply writes a call in prose and spends the budget;
//! the next run's request must carry that call's paired result.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use nanna_agent::{Agent, AgentConfig, RunOptions};
use nanna_llm::LlmClient;
use nanna_tools::{
    ParameterType, Tool, ToolDefinition, ToolError, ToolParameter, ToolRegistry, ToolResult,
};
use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

struct StubEcho;

#[async_trait]
impl Tool for StubEcho {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new("echo_stub", "Echo a word back.").param(ToolParameter {
            name: "word".to_string(),
            description: "The word".to_string(),
            param_type: ParameterType::String,
            required: true,
            default: None,
            enum_values: None,
        })
    }

    async fn execute(&self, _params: HashMap<String, Value>) -> Result<ToolResult, ToolError> {
        Ok(ToolResult::success("echoed"))
    }
}

/// Read one HTTP/1.1 request; returns (`request_line`, body).
async fn read_http_request(stream: &mut TcpStream) -> Option<(String, String)> {
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
    Some((request_line, String::from_utf8_lossy(&buf[body_start..body_end]).to_string()))
}

async fn respond(stream: &mut TcpStream, body: &str) {
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = stream.write_all(response.as_bytes()).await;
    let _ = stream.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_salvaged_call_cut_off_by_the_budget_is_paired() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind stub");
    let addr = listener.local_addr().expect("stub addr");
    let chat_bodies: Arc<tokio::sync::Mutex<Vec<String>>> =
        Arc::new(tokio::sync::Mutex::new(Vec::new()));
    let bodies = Arc::clone(&chat_bodies);
    tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                break;
            };
            let Some((request_line, body)) = read_http_request(&mut socket).await else {
                continue;
            };
            if !request_line.contains("/api/chat") {
                respond(&mut socket, "{}").await;
                continue;
            }
            let first = {
                let mut seen = bodies.lock().await;
                seen.push(body);
                seen.len() == 1
            };
            let content = if first {
                // A call written as prose: the salvage synthesizes it.
                r#"I will echo it. {\"name\": \"echo_stub\", \"arguments\": {\"word\": \"hi\"}}"#
            } else {
                "Done."
            };
            let reply = format!(
                r#"{{"model":"stub","message":{{"role":"assistant","content":"{content}"}},"done":true,"done_reason":"stop","prompt_eval_count":200,"eval_count":8}}"#
            );
            respond(&mut socket, &reply).await;
        }
    });

    let registry = Arc::new(ToolRegistry::new());
    registry.register(StubEcho).await;
    let config = AgentConfig {
        model: "salvage-pair-stub:1b".to_string(),
        ..Default::default()
    };
    let llm = Arc::new(LlmClient::ollama(format!("http://{addr}")));
    let agent = Agent::new(config, llm, registry);

    let budgeted = RunOptions {
        token_budget: Some(10),
        ..RunOptions::default()
    };
    agent
        .run("Echo hi.", budgeted)
        .await
        .expect("the budgeted run ends");
    agent
        .run("And now?", RunOptions::default())
        .await
        .expect("the next run completes");

    let seen = chat_bodies.lock().await;
    assert_eq!(seen.len(), 2, "one request per run");
    let parsed: Value = serde_json::from_str(&seen[1]).expect("json body");
    drop(seen);
    let messages = parsed["messages"].as_array().cloned().expect("messages");
    let called = messages.iter().any(|m| {
        m["role"] == "assistant"
            && m["tool_calls"]
                .as_array()
                .is_some_and(|calls| calls.iter().any(|c| c["function"]["name"] == "echo_stub"))
    });
    assert!(called, "precondition: the salvaged call was stored: {messages:#?}");
    let paired = messages.iter().any(|m| {
        m["role"] == "tool"
            && m["content"]
                .as_str()
                .is_some_and(|c| c.contains("Skipped") && c.contains("token budget"))
    });
    assert!(paired, "the salvaged call has its result: {messages:#?}");
}
