//! Browser backend wiring for tools
//!
//! Provides helpers to connect browser tools to actual browser backends
//! (CDP via chromiumoxide or Playwright).

use nanna_browser::{Browser, BrowserConfig, BrowserError, BrowserPage, ScreenshotOptions, ImageFormat};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
// Tracing available for future use

/// Browser manager that maintains a browser instance for tool use.
pub struct BrowserManager {
    browser: Arc<dyn Browser>,
    // Cache the current page for multi-step interactions
    current_page: RwLock<Option<Arc<dyn BrowserPage>>>,
}

impl BrowserManager {
    /// Create a new browser manager with the given browser instance.
    pub fn new(browser: Arc<dyn Browser>) -> Self {
        Self {
            browser,
            current_page: RwLock::new(None),
        }
    }

    /// Create from config using the default backend.
    ///
    /// # Errors
    ///
    /// Returns `BrowserError` if the browser cannot be created.
    pub fn from_config(config: BrowserConfig) -> Result<Self, BrowserError> {
        let browser = nanna_browser::create_browser(config)?;
        Ok(Self::new(browser))
    }

    /// Get or create a page for the given URL.
    async fn get_page(&self, url: &str) -> Result<Arc<dyn BrowserPage>, BrowserError> {
        // A cached page is never reused, even at the same URL: `page.url()`
        // might not work well, so we always navigate.
        let page = self.browser.navigate(url).await?;
        
        // Cache it
        *self.current_page.write().await = Some(page.clone());
        
        Ok(page)
    }

    /// Take a screenshot.
    ///
    /// # Errors
    ///
    /// Returns the backend's message if navigating to `url` or capturing the
    /// screenshot fails.
    pub async fn screenshot(
        &self,
        url: &str,
        params: &HashMap<String, Value>,
    ) -> Result<Vec<u8>, String> {
        let page = self.get_page(url).await.map_err(|e| e.to_string())?;

        let full_page = params
            .get("full_page")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);

        let format = params
            .get("format")
            .and_then(|v| v.as_str())
            .map_or(ImageFormat::Png, |f| match f.to_lowercase().as_str() {
                "jpeg" | "jpg" => ImageFormat::Jpeg,
                _ => ImageFormat::Png,
            });

        let options = ScreenshotOptions {
            full_page,
            format,
            // The low byte, exactly what the `as u8` this replaced kept.
            quality: params
                .get("quality")
                .and_then(serde_json::Value::as_u64)
                .map(|q| q.to_le_bytes()[0]),
            selector: params.get("selector").and_then(|v| v.as_str()).map(String::from),
        };

        page.screenshot(options).await.map_err(|e| e.to_string())
    }

    /// Extract content from a page.
    ///
    /// # Errors
    ///
    /// Returns a message if navigating to `url` fails, if `attribute` is given
    /// without a `selector`, or if reading the attribute, evaluating the
    /// selector script, or reading the page's HTML or text fails.
    pub async fn extract(
        &self,
        url: &str,
        params: &HashMap<String, Value>,
    ) -> Result<String, String> {
        let page = self.get_page(url).await.map_err(|e| e.to_string())?;

        let mode = params
            .get("mode")
            .and_then(|v| v.as_str())
            .unwrap_or("text");

        let selector = params.get("selector").and_then(|v| v.as_str());

        // `attribute` (e.g. `href`, `src`) is in the `browser_extract` skill's
        // parameter schema and had no path here at all, so asking for one
        // silently returned the element's text instead.
        if let Some(attribute) = params.get("attribute").and_then(|v| v.as_str()) {
            let Some(sel) = selector else {
                return Err(
                    "extracting an attribute needs a `selector` to read it from".to_string()
                );
            };
            return page
                .get_attribute(sel, attribute)
                .await
                .map(Option::unwrap_or_default)
                .map_err(|e| e.to_string());
        }

        if let Some(sel) = selector {
            // Extract from specific selector
            let script = match mode {
                "html" => format!(
                    "document.querySelector('{}')?.outerHTML || ''",
                    sel.replace('\'', "\\'")
                ),
                _ => format!(
                    "document.querySelector('{}')?.textContent || ''",
                    sel.replace('\'', "\\'")
                ),
            };
            let result = page.evaluate(&script).await.map_err(|e| e.to_string())?;
            Ok(result.as_str().unwrap_or("").to_string())
        } else {
            // Extract full page
            match mode {
                "html" => page.html().await.map_err(|e| e.to_string()),
                _ => page.text_content().await.map_err(|e| e.to_string()),
            }
        }
    }

    /// Perform an action on a page.
    ///
    /// # Errors
    ///
    /// Returns a message if navigating to `url` fails, if `action` is missing
    /// or unknown, if a parameter the action needs (`selector`, `text`, `key`,
    /// or the `value` URL for `navigate`) is missing, or if the page operation
    /// itself fails.
    pub async fn action(
        &self,
        url: &str,
        params: &HashMap<String, Value>,
    ) -> Result<String, String> {
        let page = self.get_page(url).await.map_err(|e| e.to_string())?;

        let action = params
            .get("action")
            .and_then(|v| v.as_str())
            .ok_or("Missing action")?;

        match action {
            "click" => {
                let selector = params
                    .get("selector")
                    .and_then(|v| v.as_str())
                    .ok_or("Click requires 'selector'")?;
                page.click(selector).await.map_err(|e| e.to_string())?;
                Ok(format!("Clicked '{selector}'"))
            }
            "type" => {
                let selector = params
                    .get("selector")
                    .and_then(|v| v.as_str())
                    .ok_or("Type requires 'selector'")?;
                let text = params
                    .get("text")
                    .and_then(|v| v.as_str())
                    .ok_or("Type requires 'text'")?;
                page.type_text(selector, text).await.map_err(|e| e.to_string())?;
                Ok(format!("Typed into '{selector}'"))
            }
            "fill" => {
                let selector = params
                    .get("selector")
                    .and_then(|v| v.as_str())
                    .ok_or("Fill requires 'selector'")?;
                let text = params
                    .get("text")
                    .and_then(|v| v.as_str())
                    .ok_or("Fill requires 'text'")?;
                page.fill(selector, text).await.map_err(|e| e.to_string())?;
                Ok(format!("Filled '{selector}'"))
            }
            "press" => {
                let selector = params
                    .get("selector")
                    .and_then(|v| v.as_str())
                    .ok_or("Press requires 'selector'")?;
                let key = params
                    .get("key")
                    .and_then(|v| v.as_str())
                    .ok_or("Press requires 'key'")?;
                page.press(selector, key).await.map_err(|e| e.to_string())?;
                Ok(format!("Pressed '{key}' on '{selector}'"))
            }
            "wait" => {
                let ms = params
                    .get("wait_ms")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(1000);
                tokio::time::sleep(std::time::Duration::from_millis(ms)).await;
                Ok(format!("Waited {ms}ms"))
            }
            "wait_selector" => {
                let selector = params
                    .get("selector")
                    .and_then(|v| v.as_str())
                    .ok_or("wait_selector requires 'selector'")?;
                page.wait_for_selector(selector).await.map_err(|e| e.to_string())?;
                Ok(format!("Found '{selector}'"))
            }
            // `scroll` and `navigate` are both in the `browser_action` skill's
            // advertised enum and neither was implemented, so the model was
            // told it could use two actions that always answered
            // "Unknown action".
            "scroll" => {
                // `value` is the skill's field for an action's argument; here
                // it is pixels, defaulting to one viewport-ish jump.
                let pixels = params
                    .get("value")
                    .and_then(|v| v.as_str().and_then(|s| s.parse::<i64>().ok()).or_else(|| v.as_i64()))
                    .unwrap_or(600);
                let script = format!("window.scrollBy(0, {pixels}); window.scrollY");
                page.evaluate(&script)
                    .await
                    .map_err(|e| e.to_string())?;
                Ok(format!("Scrolled {pixels}px"))
            }
            "navigate" => {
                // The skill documents `value` as "URL (for 'navigate')".
                let target = params
                    .get("value")
                    .and_then(|v| v.as_str())
                    .filter(|t| !t.trim().is_empty())
                    .ok_or("Navigate requires 'value' to be the URL to go to")?;
                page.goto(target).await.map_err(|e| e.to_string())?;
                Ok(format!("Navigated to '{target}'"))
            }
            _ => Err(format!("Unknown action: {action}")),
        }
    }

    /// Evaluate JavaScript on a page.
    ///
    /// # Errors
    ///
    /// Returns a message if navigating to `url` fails, if neither `expression`
    /// nor `script` is given, or if evaluating the script fails.
    pub async fn evaluate(
        &self,
        url: &str,
        params: &HashMap<String, Value>,
    ) -> Result<Value, String> {
        let page = self.get_page(url).await.map_err(|e| e.to_string())?;

        // The `browser_evaluate` skill sends `expression`; this read only
        // `script`, so every call answered "Missing script". Accept the name
        // the skill documents, and keep the old one working.
        let script = params
            .get("expression")
            .or_else(|| params.get("script"))
            .and_then(|v| v.as_str())
            .ok_or("evaluate requires an `expression` (the JavaScript to run)")?;

        page.evaluate(script).await.map_err(|e| e.to_string())
    }

    /// Close the browser.
    ///
    /// # Errors
    ///
    /// Returns the backend's [`BrowserError`] if closing the browser fails.
    pub async fn close(&self) -> Result<(), BrowserError> {
        self.browser.close().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_browser_manager_creation() {
        // Just test that the types are correct - actual browser tests need integration
        let _config = BrowserConfig::default();
        // Would need actual browser installed to test:
        // let manager = BrowserManager::from_config(config);
    }
}
