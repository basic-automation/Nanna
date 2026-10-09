//! `nanna export` — write a session, or the memory store, out as Markdown or
//! JSON.
//!
//! Goes through the daemon: it owns the session database and the memory store
//! (turso holds an exclusive lock on them) and renders the document itself
//! (`session.export` / `memory.export`), so the file is exactly what any other
//! client would get.

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context as _, bail};
use nanna_client::{Client, ClientConfig, ExportFormat};
use serde_json::Value;

/// Wait this long for the daemon to accept the connection — the budget
/// `nanna daemon status` already uses for the same probe.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(2);

/// The document format, as the CLI spells it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum ExportFormatArg {
    /// A readable transcript
    #[value(alias = "md")]
    Markdown,
    /// The whole record as stored — lossless
    Json,
}

impl From<ExportFormatArg> for ExportFormat {
    fn from(arg: ExportFormatArg) -> Self {
        match arg {
            ExportFormatArg::Markdown => Self::Markdown,
            ExportFormatArg::Json => Self::Json,
        }
    }
}

/// What to export.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExportTarget {
    /// One session, by id.
    Session(String),
    /// The memory store; `scope` filters as `memory.list` does.
    Memories { scope: Option<String> },
}

impl ExportTarget {
    /// The target the command line names: one session id, or `--memories`
    /// with an optional scope. clap already requires exactly one of the two;
    /// this says so rather than guess if that ever stops being true.
    pub fn from_cli(
        session: Option<String>,
        memories: bool,
        scope: Option<String>,
    ) -> anyhow::Result<Self> {
        match (session, memories) {
            (Some(id), false) => Ok(Self::Session(id)),
            (None, true) => Ok(Self::Memories { scope }),
            _ => bail!("name one session to export, or pass --memories"),
        }
    }

    fn describe(&self) -> String {
        match self {
            Self::Session(id) => format!("session {id}"),
            Self::Memories { scope: None } => "every memory".to_string(),
            Self::Memories { scope: Some(scope) } => format!("memories in scope {scope}"),
        }
    }
}

/// Export `target` to `output` — a file, or a directory that receives the
/// daemon's suggested file name — or to stdout when `output` is `None`, via
/// the daemon at `daemon` (the local default when `None`).
pub async fn export(
    target: ExportTarget,
    format: ExportFormatArg,
    output: Option<PathBuf>,
    daemon: Option<String>,
) -> anyhow::Result<()> {
    let address = daemon.unwrap_or_else(super::mcp::default_daemon_url);
    let client = connect(&address).await?;
    let reply = match &target {
        ExportTarget::Session(id) => client.sessions().export(id, format.into()).await,
        ExportTarget::Memories { scope } => {
            client.memory().export(scope.clone(), format.into()).await
        }
    }
    .context("the daemon did not answer the export request")?;
    client.disconnect().await;

    let (filename, content) = document_from_reply(&reply)?;
    match output {
        None => write_document(std::io::stdout().lock(), content)?,
        Some(path) => {
            let target_path = resolve_target(&path, filename);
            std::fs::write(&target_path, content)
                .with_context(|| format!("could not write {}", target_path.display()))?;
            eprintln!(
                "Exported {} to {}",
                target.describe(),
                target_path.display()
            );
        }
    }
    Ok(())
}

/// Write the document to `out` (stdout in practice). A reader that stops early
/// — `nanna export <id> | head` — closes the pipe; that is its choice, not a
/// failure, and `print!` would panic on it instead.
fn write_document(mut out: impl std::io::Write, content: &str) -> anyhow::Result<()> {
    match out.write_all(content.as_bytes()).and_then(|()| out.flush()) {
        Err(e) if e.kind() != std::io::ErrorKind::BrokenPipe => {
            Err(e).context("could not write the export to stdout")
        }
        _ => Ok(()),
    }
}

async fn connect(address: &str) -> anyhow::Result<Client> {
    match tokio::time::timeout(CONNECT_TIMEOUT, Client::connect(ClientConfig::new(address))).await {
        Ok(Ok(client)) => Ok(client),
        Ok(Err(e)) => bail!(
            "could not reach the Nanna daemon at {address}: {e}. Start it with \
             `nanna daemon start` — it owns the data, so export goes through it"
        ),
        Err(_) => bail!(
            "the Nanna daemon at {address} did not answer within {}s. Start it with \
             `nanna daemon start`",
            CONNECT_TIMEOUT.as_secs()
        ),
    }
}

/// Pull the document out of the daemon's reply. An `{error, message}` reply is
/// the daemon saying no — an unknown id, or memory switched off — and becomes
/// the command's error.
fn document_from_reply(reply: &Value) -> anyhow::Result<(&str, &str)> {
    if let Some(error) = reply.get("error").and_then(Value::as_str) {
        let message = reply
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or(error);
        bail!("{message}");
    }
    let filename = reply
        .get("filename")
        .and_then(Value::as_str)
        .context("the daemon's export reply carries no filename")?;
    let content = reply
        .get("content")
        .and_then(Value::as_str)
        .context("the daemon's export reply carries no content")?;
    Ok((filename, content))
}

/// A directory receives the suggested file name; any other path is the file.
/// Only the name's final component is used, so a reply can never steer the
/// write outside the directory the user chose.
fn resolve_target(path: &Path, filename: &str) -> PathBuf {
    if !path.is_dir() {
        return path.to_path_buf();
    }
    let name = Path::new(filename)
        .file_name()
        .map_or_else(|| "nanna-export".into(), std::ffi::OsStr::to_os_string);
    path.join(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_document_reply_yields_its_name_and_content() {
        let reply = json!({ "format": "markdown", "filename": "a.md", "content": "# a\n" });
        let (filename, content) = document_from_reply(&reply).expect("a document reply");
        assert_eq!(filename, "a.md");
        assert_eq!(content, "# a\n");
    }

    #[test]
    fn an_error_reply_becomes_the_commands_error() {
        let reply = json!({ "error": "not_found", "message": "Session nope not found" });
        let err = document_from_reply(&reply).expect_err("an error reply is an error");
        assert_eq!(err.to_string(), "Session nope not found");
    }

    #[test]
    fn a_directory_target_uses_only_the_suggested_names_last_component() {
        let dir = std::env::temp_dir().join(format!("nanna-export-target-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let target = resolve_target(&dir, "../../escape.md");
        assert_eq!(target, dir.join("escape.md"));
        let file = dir.join("chosen.md");
        assert_eq!(
            resolve_target(&file, "ignored.md"),
            file,
            "a file path is used as given"
        );
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn the_cli_spells_formats_as_the_protocol_does() {
        assert_eq!(
            ExportFormat::from(ExportFormatArg::Markdown),
            ExportFormat::Markdown
        );
        assert_eq!(
            ExportFormat::from(ExportFormatArg::Json),
            ExportFormat::Json
        );
    }

    #[test]
    fn a_target_describes_itself_for_the_confirmation_line() {
        assert_eq!(ExportTarget::Session("s1".into()).describe(), "session s1");
        assert_eq!(
            ExportTarget::Memories { scope: None }.describe(),
            "every memory"
        );
        assert_eq!(
            ExportTarget::Memories {
                scope: Some("global".into())
            }
            .describe(),
            "memories in scope global"
        );
    }

    #[test]
    fn the_command_line_names_exactly_one_target() {
        assert_eq!(
            ExportTarget::from_cli(Some("s1".into()), false, None).expect("a session"),
            ExportTarget::Session("s1".into())
        );
        assert_eq!(
            ExportTarget::from_cli(None, true, Some("ws".into())).expect("memories"),
            ExportTarget::Memories {
                scope: Some("ws".into())
            }
        );
        assert!(ExportTarget::from_cli(Some("s1".into()), true, None).is_err());
        assert!(ExportTarget::from_cli(None, false, None).is_err());
    }

    /// A writer that fails every write with `kind`.
    struct Failing(std::io::ErrorKind);

    impl std::io::Write for Failing {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::from(self.0))
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// `nanna export <id> | head` closes the pipe early: that is the reader's
    /// choice, not an error — but any other write failure still is.
    #[test]
    fn a_closed_pipe_is_not_a_failure_but_other_write_errors_are() {
        assert!(write_document(Failing(std::io::ErrorKind::BrokenPipe), "doc").is_ok());
        assert!(write_document(Failing(std::io::ErrorKind::StorageFull), "doc").is_err());
        let mut sink = Vec::new();
        write_document(&mut sink, "# doc\n").expect("a working writer");
        assert_eq!(sink, b"# doc\n");
    }
}
