//! Workspace templates — minimal project scaffolding only

use crate::{WorkspaceError, AGENTS_FILE, ROADMAP_FILE, WORKSPACE_MARKER_DIR};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tokio::fs;
use tracing::info;

/// Default root AGENTS.md content
pub const DEFAULT_AGENTS_MD: &str = r"# AGENTS.md

This is the project workspace. Treat it that way.

## Every Session

1. Read `README.md` — what this project is
2. Read this file — agent instructions for the repo
3. Check `ROADMAP.md` if present — current plan / next actions

## Safety

- Don't exfiltrate private data. Ever.
- Don't run destructive commands without asking.
- When in doubt, ask.

## Make It Yours

Capture build commands, architecture notes, and common pitfalls here so future
    sessions start with that knowledge. Keep it concise — this file is injected into
every prompt.";

/// A workspace template definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceTemplate {
    /// Template identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Description of what this template is for
    pub description: String,
    /// Files to create (filename -> content)
    pub files: HashMap<String, String>,
    /// Directories to create
    pub directories: Vec<String>,
}

impl WorkspaceTemplate {
    /// Minimal template: root AGENTS.md only (+ optional local-state dir)
    #[must_use]
    pub fn minimal() -> Self {
        let mut files = HashMap::new();
        files.insert(AGENTS_FILE.to_string(), DEFAULT_AGENTS_MD.to_string());

        Self {
            id: "minimal".to_string(),
            name: "Minimal".to_string(),
            description: "Root AGENTS.md only — no bespoke sidecar".to_string(),
            files,
            directories: vec![WORKSPACE_MARKER_DIR.to_string()],
        }
    }

    /// Project template: AGENTS.md + starter ROADMAP.md
    #[must_use]
    pub fn project() -> Self {
        let mut files = HashMap::new();
        files.insert(
            AGENTS_FILE.to_string(),
            "# AGENTS.md - Project Workspace

This is a code project workspace. Focus on:
- Understanding the codebase before making changes
- Writing clean, idiomatic code
- Running tests after modifications
- Committing changes with clear messages

## File Operations
- Read files to understand context
- Edit files with precision
- Run builds and tests to verify changes

## Safety
- Always run tests before claiming success
- Ask before major refactors
- Prefer safe operations (trash over rm)

## Self-Maintenance
When you discover project-relevant information during your work, **update this file**
so future sessions start with that knowledge.

Capture: build commands, file locations, architecture, common errors/fixes,
dependencies, testing strategies. Keep it concise — this file is injected into every prompt.
"
            .to_string(),
        );
        files.insert(
            ROADMAP_FILE.to_string(),
            "# Roadmap

> Project plan. Phases, checklists, dated notes.

**Last updated:** (add date)

---

## Immediate next actions

1. 
"
            .to_string(),
        );

        Self {
            id: "project".to_string(),
            name: "Project".to_string(),
            description: "AGENTS.md + ROADMAP.md for code projects".to_string(),
            files,
            directories: vec![WORKSPACE_MARKER_DIR.to_string()],
        }
    }
}

/// List all available templates
#[must_use]
pub fn list_templates() -> Vec<WorkspaceTemplate> {
    vec![
        WorkspaceTemplate::minimal(),
        WorkspaceTemplate::project(),
    ]
}

/// Create a workspace from a template
///
/// # Errors
///
/// Returns [`WorkspaceError::TemplateNotFound`] when no built-in template has
/// id `template_id`, and [`WorkspaceError::Io`] when the workspace directory,
/// one of the template's directories, or one of its files cannot be created or
/// written. Files written before the failure are left in place.
pub async fn create_from_template(path: &Path, template_id: &str) -> Result<(), WorkspaceError> {
    let template = list_templates()
        .into_iter()
        .find(|t| t.id == template_id)
        .ok_or_else(|| WorkspaceError::TemplateNotFound(template_id.to_string()))?;

    fs::create_dir_all(path).await?;

    for dir in &template.directories {
        fs::create_dir_all(path.join(dir)).await?;
    }

    // A file that already exists is the user's and is kept: the only guard
    // upstream is "no AGENTS.md yet", so initialising a repo that already had
    // a README.md or ROADMAP.md used to replace it with the starter text.
    for (filename, content) in &template.files {
        let file_path = path.join(filename);
        let created = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&file_path)
            .await;
        match created {
            Ok(mut file) => {
                use tokio::io::AsyncWriteExt;
                file.write_all(content.as_bytes()).await?;
                file.flush().await?;
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                info!("Kept the existing {}", file_path.display());
            }
            Err(e) => return Err(e.into()),
        }
    }

    info!(
        "Created workspace from template '{}' at {}",
        template.name,
        path.display()
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn an_existing_file_is_kept_not_replaced() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("ROADMAP.md"), "my own plan").unwrap();
        create_from_template(dir.path(), "project").await.unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.path().join("ROADMAP.md")).unwrap(),
            "my own plan"
        );
        assert!(
            dir.path().join("AGENTS.md").exists(),
            "the missing files are still created"
        );
    }

    #[test]
    fn test_list_templates() {
        let templates = list_templates();
        assert_eq!(templates.len(), 2);
        let ids: Vec<_> = templates.iter().map(|t| t.id.as_str()).collect();
        assert!(ids.contains(&"minimal"));
        assert!(ids.contains(&"project"));
    }

    #[tokio::test]
    async fn test_create_from_template() {
        let dir = tempdir().unwrap();
        let ws_path = dir.path().join("my_workspace");

        create_from_template(&ws_path, "minimal").await.unwrap();

        assert!(ws_path.join(AGENTS_FILE).exists());
        assert!(!ws_path.join("SOUL.md").exists());
        assert!(!ws_path.join("MEMORY.md").exists());
        assert!(!ws_path.join("memory").exists());
        assert!(ws_path.join(WORKSPACE_MARKER_DIR).exists());
    }

    #[tokio::test]
    async fn test_project_template_has_roadmap() {
        let dir = tempdir().unwrap();
        let ws_path = dir.path().join("proj");
        create_from_template(&ws_path, "project").await.unwrap();
        assert!(ws_path.join(AGENTS_FILE).exists());
        assert!(ws_path.join(ROADMAP_FILE).exists());
    }

    #[tokio::test]
    async fn test_invalid_template() {
        let dir = tempdir().unwrap();
        let result = create_from_template(dir.path(), "nonexistent").await;
        assert!(matches!(result, Err(WorkspaceError::TemplateNotFound(_))));
    }
}
