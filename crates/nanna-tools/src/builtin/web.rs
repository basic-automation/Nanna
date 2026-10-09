//! Web-related tools

use crate::{output_schemas, Tool, ToolDefinition, ToolError, ToolResult};
use async_trait::async_trait;
use scraper::{Html, Selector};
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use std::fmt::Write as _;

/// Web search tool using Brave Search API
pub struct WebSearchTool {
    pub api_key: Option<String>,
}

impl WebSearchTool {
    #[must_use]
    pub const fn new() -> Self {
        Self { api_key: None }
    }

    #[must_use]
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }
}

impl Default for WebSearchTool {
    fn default() -> Self {
        Self::new()
    }
}

/// Brave Search API response types
#[derive(Debug, Deserialize)]
struct BraveSearchResponse {
    web: Option<BraveWebResults>,
}

#[derive(Debug, Deserialize)]
struct BraveWebResults {
    results: Vec<BraveWebResult>,
}

#[derive(Debug, Deserialize)]
struct BraveWebResult {
    title: String,
    url: String,
    description: Option<String>,
}

#[async_trait]
impl Tool for WebSearchTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new("web_search", "Search the web using Brave Search API")
            .string_param("query", "Search query string", true)
            .int_param("count", "Number of results to return (1-10)", false)
            .enum_param(
                "output_mode",
                "text (default) or json — json returns structured result list",
                false,
                &["text", "json"],
            )
            .with_output_schema(output_schemas::web_search())
    }

    async fn execute(&self, params: HashMap<String, Value>) -> Result<ToolResult, ToolError> {
        let query = params
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidParams("Missing 'query' parameter".to_string()))?;

        let count = params
            .get("count")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(5)
            .min(10) as usize;

        let api_key = self.api_key.as_ref().ok_or_else(|| {
            ToolError::ExecutionFailed(
                "Brave Search API key not configured (BRAVE_API_KEY env var). \
                Use web_fetch to fetch content from specific URLs instead.".to_string()
            )
        })?;

        let client = reqwest::Client::new();
        let response = client
            .get("https://api.search.brave.com/res/v1/web/search")
            .header("Accept", "application/json")
            .header("X-Subscription-Token", api_key)
            .query(&[("q", query), ("count", &count.to_string())])
            .send()
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("Search request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(ToolError::ExecutionFailed(format!(
                "Brave Search API error: {status} - {body}"
            )));
        }

        let search_result: BraveSearchResponse = response
            .json()
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("Failed to parse response: {e}")))?;

        let results = search_result
            .web
            .map_or_default(|w| w.results);

        if results.is_empty() {
            return Ok(ToolResult::success("No results found."));
        }

        // Format results as readable text
        let mut output = String::new();
        for (i, result) in results.iter().take(count).enumerate() {
            let _ = write!(
                output,
                "{}. {}\n   {}\n   {}\n\n",
                i + 1,
                result.title,
                result.url,
                result.description.as_deref().unwrap_or("No description")
            );
        }

        let structured: Vec<serde_json::Value> = results
            .iter()
            .take(count)
            .map(|r| {
                serde_json::json!({
                    "title": r.title,
                    "url": r.url,
                    "description": r.description,
                })
            })
            .collect();

        Ok(ToolResult::success(output).with_data(serde_json::json!({
            "query": query,
            "result_count": structured.len(),
            "results": structured,
        })))
    }
}

/// Fetch web page content with readability extraction
pub struct WebFetchTool {
    pub max_size: usize,
    pub timeout_secs: u64,
}

impl WebFetchTool {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            max_size: 1024 * 1024, // 1MB
            timeout_secs: 30,
        }
    }
}

impl Default for WebFetchTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for WebFetchTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new("web_fetch", "Fetch and extract readable content from a URL")
            .string_param("url", "HTTP or HTTPS URL to fetch", true)
            .int_param("max_chars", "Maximum characters to return", false)
            .enum_param(
                "output_mode",
                "text (default) or json — json returns url/title/content",
                false,
                &["text", "json"],
            )
            .with_output_schema(output_schemas::web_fetch())
    }

    async fn execute(&self, params: HashMap<String, Value>) -> Result<ToolResult, ToolError> {
        let url = params
            .get("url")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidParams("Missing 'url' parameter".to_string()))?;

        let max_chars = crate::u64_to_usize(
            params
                .get("max_chars")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(50000),
        );

        // Validate URL
        if !url.starts_with("http://") && !url.starts_with("https://") {
            return Err(ToolError::InvalidParams(
                "URL must start with http:// or https://".to_string(),
            ));
        }

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(self.timeout_secs))
            .build()
            .map_err(|e| {
                ToolError::ExecutionFailed(format!("Failed to create HTTP client: {e}"))
            })?;

        let response = client
            .get(url)
            .header("User-Agent", "Mozilla/5.0 (compatible; Nanna/1.0)")
            .send()
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("Request failed: {e}")))?;

        let status = response.status();
        if !status.is_success() {
            return Err(ToolError::ExecutionFailed(format!("HTTP error: {status}")));
        }

        let content = response
            .text()
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("Failed to read response: {e}")))?;

        // Extract readable content
        let text = extract_readable_content(&content);
        let truncated = if text.len() > max_chars {
            format!(
                "{}...\n\n[Truncated at {} chars]",
                &text[..truncate_boundary(&text, max_chars)],
                max_chars
            )
        } else {
            text
        };

        Ok(ToolResult::success(truncated.clone()).with_data(serde_json::json!({
            "url": url,
            "status": status.as_u16(),
            "original_length": content.len(),
            "chars": truncated.len(),
            "content": truncated,
        })))
    }

    fn timeout_secs(&self) -> Option<u64> {
        Some(self.timeout_secs)
    }
}

/// Largest byte index at or below `max_bytes` that does not split a character.
///
/// Fetched pages are full of multi-byte characters — em dashes, curly quotes,
/// CJK — and slicing a string at a byte index inside one panics, taking the
/// whole process down, so the cut walks back to the nearest boundary.
const fn truncate_boundary(s: &str, max_bytes: usize) -> usize {
    if s.len() <= max_bytes {
        return s.len();
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    end
}

/// Extract readable content from HTML using a readability-style algorithm.
///
/// This removes scripts, styles, navigation, ads, and other boilerplate,
/// keeping the main content.
fn extract_readable_content(html: &str) -> String {
    let document = Html::parse_document(html);

    // Try to find main content areas
    let content_selectors = [
        "article",
        "main",
        "[role=\"main\"]",
        ".post-content",
        ".article-content",
        ".entry-content",
        ".content",
        "#content",
        ".post",
        ".article",
    ];

    // Try each content selector
    for selector_str in &content_selectors {
        if let Ok(selector) = Selector::parse(selector_str)
            && let Some(element) = document.select(&selector).next() {
                let text = extract_text_from_element(&element);
                if text.len() > 200 {
                    // Likely main content
                    return clean_extracted_text(&text);
                }
            }
    }

    // Fallback: extract from body, excluding common noise elements
    if let Ok(body_selector) = Selector::parse("body")
        && let Some(body) = document.select(&body_selector).next() {
            let text = extract_text_from_element(&body);
            return clean_extracted_text(&text);
        }

    // Last resort: strip all tags
    strip_html_basic(html)
}

/// Extract text from an HTML element, skipping script/style/nav elements.
fn extract_text_from_element(element: &scraper::ElementRef) -> String {
    let skip_tags = ["script", "style", "nav", "header", "footer", "aside", "noscript", "iframe"];

    let mut text = String::new();

    for node in element.descendants() {
        match node.value() {
            scraper::node::Node::Text(t) => {
                // Check if any ancestor is a skip tag
                let should_skip = node.ancestors().any(|ancestor| {
                    ancestor
                        .value()
                        .as_element()
                        .is_some_and(|el| skip_tags.contains(&el.name()))
                });

                if !should_skip {
                    let trimmed = t.trim();
                    if !trimmed.is_empty() {
                        if !text.is_empty() && !text.ends_with('\n') && !text.ends_with(' ') {
                            text.push(' ');
                        }
                        text.push_str(trimmed);
                    }
                }
            }
            scraper::node::Node::Element(el) => {
                // Add line breaks for block elements
                let block_tags = ["p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr"];
                if block_tags.contains(&el.name()) && !text.is_empty() && !text.ends_with('\n') {
                    text.push('\n');
                }
            }
            _ => {}
        }
    }

    text
}

/// Clean up extracted text (normalize whitespace, remove excess blank lines).
fn clean_extracted_text(text: &str) -> String {
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();

    // Collapse multiple blank-ish lines
    let mut result = String::new();
    let mut prev_empty = false;

    for line in lines {
        let is_short = line.len() < 3;
        if is_short && prev_empty {
            continue; // Skip consecutive short lines
        }
        if !result.is_empty() {
            result.push('\n');
        }
        result.push_str(line);
        prev_empty = is_short;
    }

    result
}

/// Basic HTML tag stripping (fallback).
fn strip_html_basic(html: &str) -> String {
    let mut result = String::with_capacity(html.len());
    let mut in_tag = false;
    let mut in_script = false;
    let mut in_style = false;
    let mut tag_name = String::new();

    let chars: Vec<char> = html.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];

        if c == '<' {
            in_tag = true;
            tag_name.clear();
            i += 1;
            continue;
        }

        if c == '>' && in_tag {
            in_tag = false;
            let tag_lower = tag_name.to_lowercase();
            if tag_lower.starts_with("script") {
                in_script = true;
            } else if tag_lower.starts_with("/script") {
                in_script = false;
            } else if tag_lower.starts_with("style") {
                in_style = true;
            } else if tag_lower.starts_with("/style") {
                in_style = false;
            }
            i += 1;
            continue;
        }

        if in_tag {
            tag_name.push(c);
        } else if !in_script && !in_style {
            result.push(c);
        }

        i += 1;
    }

    // Clean up whitespace
    let lines: Vec<&str> = result
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();

    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncation_backs_off_a_split_multibyte_character() {
        // The shape that killed the daemon: an em dash is 3 bytes, so a cut at
        // byte 80 lands one byte inside it and `&text[..80]` panics.
        let text = format!("{}—tail", "a".repeat(79));
        assert!(!text.is_char_boundary(80));

        let end = truncate_boundary(&text, 80);
        assert_eq!(end, 79);
        assert_eq!(&text[..end], "a".repeat(79).as_str());
    }

    #[test]
    fn truncation_keeps_the_limit_when_it_lands_on_a_boundary() {
        let text = "a".repeat(200);
        assert_eq!(truncate_boundary(&text, 80), 80);
        assert_eq!(truncate_boundary(&text, 200), 200);
        // Shorter than the limit means no cut at all.
        assert_eq!(truncate_boundary("short", 80), 5);
    }
}
