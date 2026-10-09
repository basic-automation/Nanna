//! Browser backend wiring for tools
//!
//! Provides helpers to connect browser tools to actual browser backends
//! (CDP via chromiumoxide or Playwright).

use nanna_browser::{Browser, BrowserConfig, BrowserError, BrowserPage, ScreenshotOptions, ImageFormat};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
// Tracing available for future use

/// Browser manager that maintains a browser instance for tool use.
pub struct BrowserManager {
    browser: Arc<dyn Browser>,
}

impl BrowserManager {
    /// Create a new browser manager with the given browser instance.
    pub fn new(browser: Arc<dyn Browser>) -> Self {
        Self { browser }
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

    /// A fresh page (tab) at `url`, for one call. The caller closes it with
    /// [`Self::release`] when the call is done.
    async fn get_page(&self, url: &str) -> Result<Arc<dyn BrowserPage>, BrowserError> {
        self.browser.navigate(url).await
    }

    /// Close a call's page. Every call opened a tab and none was ever closed
    /// (the page was cached, then overwritten): a long session accumulated
    /// one live tab per browser call, each still running its page's
    /// scripts, until Chromium ran out of memory.
    async fn release(page: Arc<dyn BrowserPage>) {
        if let Err(e) = page.close().await {
            tracing::debug!("closing a browser page failed: {e}");
        }
    }

    /// Take a screenshot.
    ///
    /// # Errors
    ///
    /// Returns the backend's message if navigating to `url` or capturing the
    /// screenshot fails.
    async fn screenshot_on(
        page: &dyn BrowserPage,
        params: &HashMap<String, Value>,
    ) -> Result<Vec<u8>, String> {

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
    async fn extract_on(
        page: &dyn BrowserPage,
        params: &HashMap<String, Value>,
    ) -> Result<String, String> {

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
            // A JSON string is a valid JS string literal with every quote,
            // backslash and line break escaped; escaping only `'` broke a
            // selector with a backslash (`#a\\:b`) or let one end the literal.
            let literal = serde_json::to_string(sel).map_err(|e| e.to_string())?;
            let script = match mode {
                "html" => format!("document.querySelector({literal})?.outerHTML || ''"),
                _ => format!("document.querySelector({literal})?.textContent || ''"),
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
    async fn action_on(
        page: &dyn BrowserPage,
        params: &HashMap<String, Value>,
        wait_max_ms: u64,
    ) -> Result<String, String> {

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
                // At most the browser's own operation deadline: an
                // unbounded `wait_ms` held the tool call for whatever the
                // model asked (10 000 000 ms is 2.8 hours).
                let ms = params
                    .get("wait_ms")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(1000)
                    .min(wait_max_ms);
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
    async fn evaluate_on(
        page: &dyn BrowserPage,
        params: &HashMap<String, Value>,
    ) -> Result<Value, String> {

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

    /// Take a screenshot of `url` (see `screenshot_on`).
    ///
    /// # Errors
    ///
    /// As `screenshot_on`, or when `url` cannot be opened.
    pub async fn screenshot(
        &self,
        url: &str,
        params: &HashMap<String, Value>,
    ) -> Result<Vec<u8>, String> {
        let page = self.get_page(url).await.map_err(|e| e.to_string())?;
        let result = Self::screenshot_on(page.as_ref(), params).await;
        Self::release(page).await;
        result
    }

    /// Extract content from `url` (see `extract_on`).
    ///
    /// # Errors
    ///
    /// As `extract_on`, or when `url` cannot be opened.
    pub async fn extract(&self, url: &str, params: &HashMap<String, Value>) -> Result<String, String> {
        let page = self.get_page(url).await.map_err(|e| e.to_string())?;
        let result = Self::extract_on(page.as_ref(), params).await;
        Self::release(page).await;
        result
    }

    /// Perform an action on `url` (see `action_on`).
    ///
    /// # Errors
    ///
    /// As `action_on`, or when `url` cannot be opened.
    pub async fn action(&self, url: &str, params: &HashMap<String, Value>) -> Result<String, String> {
        let page = self.get_page(url).await.map_err(|e| e.to_string())?;
        let wait_max_ms = self.browser.config().timeout_ms;
        let result = Self::action_on(page.as_ref(), params, wait_max_ms).await;
        Self::release(page).await;
        result
    }

    /// Evaluate JavaScript on `url` (see `evaluate_on`).
    ///
    /// # Errors
    ///
    /// As `evaluate_on`, or when `url` cannot be opened.
    pub async fn evaluate(&self, url: &str, params: &HashMap<String, Value>) -> Result<Value, String> {
        let page = self.get_page(url).await.map_err(|e| e.to_string())?;
        let result = Self::evaluate_on(page.as_ref(), params).await;
        Self::release(page).await;
        result
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

    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Counts the pages it opens and the ones closed, and records scripts.
    struct FakePage(Arc<AtomicUsize>, std::sync::Mutex<Vec<String>>);

    #[async_trait]
    impl BrowserPage for FakePage {
        fn url(&self) -> &'static str { "about:blank" }
        async fn goto(&self, _url: &str) -> Result<(), BrowserError> { Ok(()) }
        async fn screenshot(&self, _o: ScreenshotOptions) -> Result<Vec<u8>, BrowserError> { Ok(vec![1]) }
        async fn text_content(&self) -> Result<String, BrowserError> { Ok("text".into()) }
        async fn html(&self) -> Result<String, BrowserError> { Ok("<p/>".into()) }
        async fn click(&self, _s: &str) -> Result<(), BrowserError> { Ok(()) }
        async fn type_text(&self, _s: &str, _t: &str) -> Result<(), BrowserError> { Ok(()) }
        async fn fill(&self, _s: &str, _t: &str) -> Result<(), BrowserError> { Ok(()) }
        async fn press(&self, _s: &str, _k: &str) -> Result<(), BrowserError> { Ok(()) }
        async fn wait_for_selector(&self, _s: &str) -> Result<(), BrowserError> { Ok(()) }
        async fn evaluate(&self, s: &str) -> Result<Value, BrowserError> {
            self.1.lock().map_err(|_| BrowserError::NotInitialized)?.push(s.to_string());
            Ok(Value::Null)
        }
        async fn get_attribute(&self, _s: &str, _a: &str) -> Result<Option<String>, BrowserError> { Ok(None) }
        async fn exists(&self, _s: &str) -> Result<bool, BrowserError> { Ok(true) }
        async fn query_all_text(&self, _s: &str) -> Result<Vec<String>, BrowserError> { Ok(Vec::new()) }
        async fn close(&self) -> Result<(), BrowserError> {
            self.0.fetch_sub(1, Ordering::SeqCst);
            Ok(())
        }
    }

    struct FakeBrowser {
        open: Arc<AtomicUsize>,
        config: BrowserConfig,
    }

    #[async_trait]
    impl Browser for FakeBrowser {
        async fn launch(&self) -> Result<(), BrowserError> { Ok(()) }
        async fn new_page(&self) -> Result<Arc<dyn BrowserPage>, BrowserError> {
            self.navigate("about:blank").await
        }
        async fn navigate(&self, _url: &str) -> Result<Arc<dyn BrowserPage>, BrowserError> {
            self.open.fetch_add(1, Ordering::SeqCst);
            Ok(Arc::new(FakePage(Arc::clone(&self.open), std::sync::Mutex::default())))
        }
        async fn close(&self) -> Result<(), BrowserError> { Ok(()) }
        fn config(&self) -> &BrowserConfig { &self.config }
    }

    /// Every call closes the tab it opened — including one whose action
    /// failed — and a `wait` is held to the browser's deadline.
    #[tokio::test]
    async fn every_call_closes_the_tab_it_opened() {
        let open = Arc::new(AtomicUsize::new(0));
        let config = BrowserConfig { timeout_ms: 20, ..BrowserConfig::default() };
        let manager = BrowserManager::new(Arc::new(FakeBrowser { open: Arc::clone(&open), config }));
        let params = |pairs: &[(&str, Value)]| -> HashMap<String, Value> {
            pairs.iter().map(|(k, v)| ((*k).to_string(), v.clone())).collect()
        };
        manager.screenshot("https://a", &params(&[])).await.expect("shot");
        manager.extract("https://a", &params(&[])).await.expect("text");
        manager.evaluate("https://a", &params(&[("expression", "1".into())])).await.expect("eval");
        let refused = manager.action("https://a", &params(&[("action", "nope".into())])).await;
        assert!(refused.is_err());
        let started = std::time::Instant::now();
        let waited = manager
            .action("https://a", &params(&[("action", "wait".into()), ("wait_ms", 10_000_000.into())]))
            .await
            .expect("waits");
        assert_eq!(waited, "Waited 20ms");
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
        assert_eq!(open.load(Ordering::SeqCst), 0, "no tab left open");
    }

    /// A selector reaches the page as one intact JS string literal.
    #[tokio::test]
    async fn an_extract_selector_is_a_safe_string_literal() {
        let page = FakePage(Arc::new(AtomicUsize::new(1)), std::sync::Mutex::default());
        let selector = r"#a\:b, a[title='x']";
        let params = HashMap::from([("selector".to_string(), Value::from(selector))]);
        BrowserManager::extract_on(&page, &params).await.expect("extracts");
        let scripts = page.1.lock().expect("lock").clone();
        let expected = format!(
            "document.querySelector({})?.textContent || ''",
            serde_json::to_string(selector).expect("json")
        );
        assert_eq!(scripts, [expected]);
    }

    #[test]
    fn test_browser_manager_creation() {
        // Just test that the types are correct - actual browser tests need integration
        let _config = BrowserConfig::default();
        // Would need actual browser installed to test:
        // let manager = BrowserManager::from_config(config);
    }
}
