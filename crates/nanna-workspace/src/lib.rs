#![warn(clippy::all)]
#![warn(clippy::pedantic, clippy::nursery)]

//! Workspace management for Nanna
//!
//! Provides directory-based project context:
//! - Workspace detection from standard project signals
//! - Context from root `README.md` / `AGENTS.md` / `CONTRIBUTING.md` / `ROADMAP.md`
//! - System prompt injection
//! - Minimal workspace templates (no bespoke SOUL/USER/MEMORY sidecar)

mod discovery;
mod files;
mod git;
mod manager;
mod templates;

pub use discovery::{discover_workspace, find_workspace_root, get_markers, is_workspace, WorkspaceMarker};
pub use files::{WorkspaceFile, WorkspaceFiles};
pub use git::{
    GIT_CHANGED_PATHS_MAX, GIT_LINE_BYTES_MAX, GIT_OUTPUT_BYTES_MAX, GIT_RECENT_COMMITS_MAX,
    GIT_TIMEOUT, GitContext, is_git_repo, load_git_context,
};

pub use manager::{Workspace, WorkspaceConfig};
pub use templates::{create_from_template, list_templates, WorkspaceTemplate};

use std::path::PathBuf;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum WorkspaceError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Workspace not found at {0}")]
    NotFound(PathBuf),
    #[error("Invalid workspace: {0}")]
    Invalid(String),
    #[error("Template not found: {0}")]
    TemplateNotFound(String),
    #[error("File not found: {0}")]
    FileNotFound(PathBuf),
    #[error("Parse error: {0}")]
    Parse(String),
}

/// Standard project context files (workspace root)
pub const AGENTS_FILE: &str = "AGENTS.md";
pub const README_FILE: &str = "README.md";
pub const ROADMAP_FILE: &str = "ROADMAP.md";
pub const CONTRIBUTING_FILE: &str = "CONTRIBUTING.md";

/// Local non-md workspace state directory
pub const WORKSPACE_MARKER_DIR: &str = ".nanna";
