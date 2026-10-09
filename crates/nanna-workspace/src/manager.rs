//! A loaded workspace: its root, marker, files and `.nanna` config.

use crate::{
    find_workspace_root, WorkspaceError, WorkspaceFiles, WorkspaceMarker, AGENTS_FILE,
    WORKSPACE_MARKER_DIR,
};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use tokio::fs;
use tracing::{debug, info};

/// Configuration for a workspace (local non-md state in `.nanna/config.toml`)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Default)]
pub struct WorkspaceConfig {
    /// Workspace name (defaults to directory name)
    pub name: Option<String>,
    /// Maximum context tokens from workspace files
    pub max_context_tokens: Option<usize>,
    /// Custom file loading order/selection
    pub file_priority: Option<Vec<String>>,
}


/// A loaded workspace with its files and configuration
#[derive(Debug, Clone)]
pub struct Workspace {
    /// Root directory of the workspace
    pub root: PathBuf,
    /// Marker that was used to identify this workspace
    pub marker: WorkspaceMarker,
    /// Loaded workspace files
    pub files: WorkspaceFiles,
    /// Workspace configuration
    pub config: WorkspaceConfig,
}

impl Workspace {
    /// Load a workspace from a directory
    ///
    /// # Errors
    ///
    /// Returns [`WorkspaceError::NotFound`] when `root` does not exist. Unreadable
    /// workspace files and a missing or invalid `.nanna/config.toml` are not
    /// errors: they load as absent files and the default config.
    pub async fn load(root: PathBuf) -> Result<Self, WorkspaceError> {
        if !root.exists() {
            return Err(WorkspaceError::NotFound(root));
        }

        let marker = find_workspace_root(&root)
            .map_or(WorkspaceMarker::AgentsFile, |(_, m)| m);

        let files = WorkspaceFiles::load(&root).await;
        let config = Self::load_config(&root).await.unwrap_or_default();

        info!(
            "Loaded workspace from {} ({} files)",
            root.display(),
            files.existing_files().len()
        );

        Ok(Self {
            root,
            marker,
            files,
            config,
        })
    }

    /// Load workspace config from .nanna/config.toml
    async fn load_config(root: &Path) -> Option<WorkspaceConfig> {
        let config_path = root.join(WORKSPACE_MARKER_DIR).join("config.toml");
        if config_path.exists() {
            let content = fs::read_to_string(&config_path).await.ok()?;
            toml::from_str(&content).ok()
        } else {
            None
        }
    }

    /// Save workspace config to .nanna/config.toml
    ///
    /// # Errors
    ///
    /// Returns [`WorkspaceError::Io`] when the `.nanna` directory cannot be
    /// created or `config.toml` cannot be written, and [`WorkspaceError::Parse`]
    /// when the config cannot be serialized to TOML.
    pub async fn save_config(&self) -> Result<(), WorkspaceError> {
        let config_dir = self.root.join(WORKSPACE_MARKER_DIR);
        fs::create_dir_all(&config_dir).await?;

        let config_path = config_dir.join("config.toml");
        let content = toml::to_string_pretty(&self.config)
            .map_err(|e| WorkspaceError::Parse(e.to_string()))?;
        fs::write(&config_path, content).await?;
        Ok(())
    }

    /// Get the workspace name (config name, or directory name)
    #[must_use]
    pub fn name(&self) -> String {
        self.config.name.clone().unwrap_or_else(|| {
            self.root
                .file_name().map_or_else(|| "workspace".to_string(), |n| n.to_string_lossy().to_string())
        })
    }

    /// Reload workspace files from disk
    ///
    /// # Errors
    ///
    /// Never returns an error today: files that cannot be read reload as absent
    /// rather than failing the reload.
    pub async fn reload(&mut self) -> Result<(), WorkspaceError> {
        self.files = WorkspaceFiles::load(&self.root).await;
        debug!("Reloaded workspace files from {}", self.root.display());
        Ok(())
    }

    /// Generate system prompt context from workspace files
    #[must_use]
    pub fn system_context(&self) -> String {
        self.files.to_system_context()
    }

    /// Initialize workspace with a minimal root AGENTS.md (+ optional ROADMAP.md)
    ///
    /// # Errors
    ///
    /// Returns [`WorkspaceError::Io`] when the `.nanna` directory cannot be
    /// created or a missing `AGENTS.md` cannot be written.
    pub async fn initialize(&self) -> Result<(), WorkspaceError> {
        // Optional local-state dir (non-md)
        let marker_dir = self.root.join(WORKSPACE_MARKER_DIR);
        fs::create_dir_all(&marker_dir).await?;

        let agents_path = self.root.join(AGENTS_FILE);
        if !agents_path.exists() {
            fs::write(&agents_path, crate::templates::DEFAULT_AGENTS_MD).await?;
        }

        info!("Initialized workspace at {}", self.root.display());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::write;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_workspace_load() {
        let dir = tempdir().unwrap();
        write(dir.path().join(AGENTS_FILE), "# Test agents").unwrap();

        let workspace = Workspace::load(dir.path().to_path_buf()).await.unwrap();
        assert!(workspace.files.agents.is_some());
        assert_eq!(workspace.marker, WorkspaceMarker::AgentsFile);
    }

    #[tokio::test]
    async fn test_workspace_initialize() {
        let dir = tempdir().unwrap();

        let workspace = Workspace::load(dir.path().to_path_buf()).await.unwrap();
        workspace.initialize().await.unwrap();

        assert!(dir.path().join(AGENTS_FILE).exists());
        assert!(dir.path().join(WORKSPACE_MARKER_DIR).exists());
        assert!(!dir.path().join("SOUL.md").exists());
        assert!(!dir.path().join("memory").exists());
    }
}
