#![warn(clippy::pedantic, clippy::nursery, clippy::all)]
//! The four `browser.*` services against a real Chromium and a real page.
//!
//! Every other service this run wired — `vision.analyze`, `audio.*` — could be
//! tested up to the request and no further, because the host has no cloud
//! credential. Browser automation is the one whose backend is genuinely
//! available, so it gets the verification the others could not have: a page is
//! served, a browser is launched, and the services are asked about it.
//!
//! Where no Chromium-family browser exists the services are not registered at
//! all, and this asserts that case rather than skipping quietly — "no browser"
//! and "the test did nothing" look the same otherwise.

use std::collections::HashMap;

use nanna_daemon::browser_service::{build_browser_services, find_browser_executable};
use nanna_scripting::ServiceFn;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

const PAGE: &str = "<!doctype html><html><head><title>Probe</title></head><body>\
<h1 id=\"headline\">Hello from the probe</h1>\
<a id=\"link\" href=\"https://example.com/target\">a link</a>\
</body></html>";

/// Serve `PAGE` on a loopback port until the test drops the task.
///
/// A local listener rather than a real site: the point is to prove the browser
/// is driven, and reaching the internet would make the test a network check.
async fn serve_probe_page() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let port = listener.local_addr().expect("addr").port();
    tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            tokio::spawn(async move {
                let mut discard = [0_u8; 2048];
                let _ = socket.read(&mut discard).await;
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n\
                     Content-Length: {}\r\nConnection: close\r\n\r\n{PAGE}",
                    PAGE.len()
                );
                let _ = socket.write_all(response.as_bytes()).await;
                let _ = socket.flush().await;
            });
        }
    });
    format!("http://127.0.0.1:{port}/")
}

async fn call(services: &HashMap<String, ServiceFn>, name: &str, params: Value) -> Value {
    let service = services
        .get(name)
        .unwrap_or_else(|| panic!("{name} must be registered"));
    service(params)
        .await
        .unwrap_or_else(|e| panic!("{name} failed: {e}"))
}

#[tokio::test]
async fn the_browser_services_drive_a_real_page() {
    let data_dir = tempfile::tempdir().expect("temp data dir");
    let services = build_browser_services(data_dir.path());

    let Some(executable) = find_browser_executable() else {
        assert!(
            services.is_empty(),
            "no browser was found, so nothing should have registered",
        );
        eprintln!("no Chromium-family browser on this host; nothing to drive");
        return;
    };
    eprintln!("driving {}", executable.display());
    assert_eq!(services.len(), 4, "all four services register together");

    let url = serve_probe_page().await;

    // extract: the text of one element, by the selector the skill sends.
    let extracted = call(
        &services,
        "browser.extract",
        json!({ "url": url, "selector": "#headline" }),
    )
    .await;
    assert_eq!(
        extracted["text"].as_str().unwrap_or_default().trim(),
        "Hello from the probe",
        "browser.extract did not read the element: {extracted}"
    );

    // extract + attribute: the parameter the skill has always advertised and
    // that had no path through the manager at all until this run.
    let href = call(
        &services,
        "browser.extract",
        json!({ "url": url, "selector": "#link", "attribute": "href" }),
    )
    .await;
    assert_eq!(
        href["text"], "https://example.com/target",
        "an `attribute` request returned something other than the attribute: {href}"
    );

    // evaluate: the skill sends `expression`; the manager used to read only
    // `script` and answered "Missing script" every time.
    let evaluated = call(
        &services,
        "browser.evaluate",
        json!({ "url": url, "expression": "document.title" }),
    )
    .await;
    assert_eq!(
        evaluated["value"], "Probe",
        "browser.evaluate did not run the expression: {evaluated}"
    );

    // action + scroll: one of the two actions the skill advertised and nothing
    // implemented.
    let scrolled = call(
        &services,
        "browser.action",
        json!({ "url": url, "action": "scroll", "value": "120" }),
    )
    .await;
    assert!(
        scrolled["result"]
            .as_str()
            .is_some_and(|r| r.contains("120")),
        "scroll did not report what it did: {scrolled}"
    );

    // screenshot: written to a file, not reported as a byte count and dropped.
    let shot = call(&services, "browser.screenshot", json!({ "url": url })).await;
    let path = shot["path"].as_str().expect("a screenshot path");
    let bytes = std::fs::read(path).expect("the reported screenshot path exists");
    assert!(
        bytes.starts_with(b"\x89PNG"),
        "the screenshot is not a PNG ({} bytes)",
        bytes.len()
    );
    assert_eq!(
        usize::try_from(shot["size"].as_u64().unwrap_or_default()).expect("size fits usize"),
        bytes.len(),
        "the reported size does not match the file"
    );
}

#[tokio::test]
async fn a_service_call_without_a_url_is_refused_by_name() {
    let data_dir = tempfile::tempdir().expect("temp data dir");
    let services = build_browser_services(data_dir.path());
    if services.is_empty() {
        return;
    }
    for name in [
        "browser.extract",
        "browser.evaluate",
        "browser.action",
        "browser.screenshot",
    ] {
        let service = services.get(name).expect("registered");
        let err = service(json!({}))
            .await
            .expect_err("a call with no url must be refused");
        assert!(err.contains("url"), "{name} refused unhelpfully: {err}");
    }
}

/// Call `name` the way the scripting bridge does: on a throwaway
/// current-thread runtime of its own, dropped when the call returns.
async fn call_on_own_runtime(service: ServiceFn, name: &'static str, params: Value) -> Value {
    tokio::task::spawn_blocking(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("a throwaway runtime");
        runtime
            .block_on(service(params))
            .unwrap_or_else(|e| panic!("{name} failed: {e}"))
    })
    .await
    .expect("the call's thread")
}

/// Every `Nanna.service(...)` runs on a runtime built for that one call and
/// dropped after it. The browser used to launch on the first call's runtime, so
/// its connection died with it: every later call failed until the daemon
/// restarted, and the browser was never relaunched.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_browser_outlives_the_runtime_of_the_call_that_launched_it() {
    let data_dir = tempfile::tempdir().expect("temp data dir");
    let services = build_browser_services(data_dir.path());
    if find_browser_executable().is_none() {
        assert!(services.is_empty(), "no browser, nothing registered");
        return;
    }
    let url = serve_probe_page().await;
    let evaluate = services
        .get("browser.evaluate")
        .expect("registered")
        .clone();
    for round in 1..=2 {
        let evaluated = call_on_own_runtime(
            evaluate.clone(),
            "browser.evaluate",
            json!({ "url": url, "expression": "document.title" }),
        )
        .await;
        assert_eq!(evaluated["value"], "Probe", "call {round}: {evaluated}");
    }
}
