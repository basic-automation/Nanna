//! Executable tool wrapper (Python, shell, binary, command)

use crate::{Tool, ToolDefinition, ToolError, ToolResult, ParameterType, ToolParameter, OutputTarget};
use crate::skills::manifest::{SkillManifest, ExecutionMethod};
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use tokio::process::Command;
use tracing::{debug, warn};

/// A tool executed via external process (Python, shell, binary, or command)
pub struct ExecutableTool {
    manifest: SkillManifest,
    skill_dir: PathBuf,
}

impl ExecutableTool {
    /// Create from a manifest file path
    ///
    /// # Errors
    ///
    /// Returns [`ToolError::Io`] if the manifest cannot be read, and
    /// [`ToolError::InvalidParams`] if it is not a valid manifest or the path has
    /// no parent directory to run the skill from.
    pub fn from_manifest(manifest_path: &Path) -> Result<Self, ToolError> {
        let manifest = SkillManifest::from_file(manifest_path)?;
        let skill_dir = manifest_path.parent()
            .ok_or_else(|| ToolError::InvalidParams("Invalid manifest path".to_string()))?
            .to_path_buf();
        
        Ok(Self { manifest, skill_dir })
    }

    /// Build the command to execute
    fn build_command(&self, params: &HashMap<String, Value>) -> Result<Command, ToolError> {
        let workdir = self.manifest.resolve_workdir(&self.skill_dir);
        
        let mut cmd = match &self.manifest.execution {
            ExecutionMethod::Python(script) => {
                let script_path = self.skill_dir.join(script);
                let mut cmd = Command::new("python");
                cmd.arg(&script_path);
                cmd.arg("--json");
                cmd.arg(serde_json::to_string(params).map_err(|e| {
                    ToolError::InvalidParams(format!("Failed to serialize params: {e}"))
                })?);
                cmd
            }
            ExecutionMethod::Shell(script) => {
                let script_path = self.skill_dir.join(script);
                let shell = if cfg!(windows) { "cmd" } else { "bash" };
                let shell_arg = if cfg!(windows) { "/C" } else { "-c" };
                
                let mut cmd = Command::new(shell);
                cmd.arg(shell_arg);
                
                // Build: bash script.sh 'json_params'
                let json_params = serde_json::to_string(params).map_err(|e| {
                    ToolError::InvalidParams(format!("Failed to serialize params: {e}"))
                })?;
                
                if cfg!(windows) {
                    cmd.arg(format!("{} {}", script_path.display(), shell_escape(&json_params)));
                } else {
                    cmd.arg(format!("bash {} '{}'", script_path.display(), json_params.replace('\'', "'\\''")));
                }
                cmd
            }
            ExecutionMethod::Command(template) => {
                // Substitute {{param}} placeholders
                let expanded = substitute_params(template, params)?;
                
                let shell = if cfg!(windows) { "cmd" } else { "sh" };
                let shell_arg = if cfg!(windows) { "/C" } else { "-c" };
                
                let mut cmd = Command::new(shell);
                cmd.arg(shell_arg);
                cmd.arg(&expanded);
                cmd
            }
            ExecutionMethod::Binary(binary) => {
                let binary_path = self.skill_dir.join(binary);
                let mut cmd = Command::new(&binary_path);
                // JSON input via stdin
                cmd.stdin(Stdio::piped());
                cmd
            }
        };
        
        // Set working directory
        cmd.current_dir(&workdir);
        
        // Set environment variables
        for (key, value) in &self.manifest.env {
            cmd.env(key, value);
        }
        
        // Capture output
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        // Contain the process. `spawn` would otherwise inherit the daemon's stdin
        // (`output()` used to null it implicitly), a dropped call would leave the
        // child running, and a grandchild (`sh -c`, a script's own subprocesses)
        // would escape a kill aimed at the shell alone. On Unix the process group
        // IS the subtree; on Windows `run_contained` adds a kill-on-close job.
        if !matches!(self.manifest.execution, ExecutionMethod::Binary(_)) {
            cmd.stdin(Stdio::null());
        }
        cmd.kill_on_drop(true);
        #[cfg(unix)]
        cmd.process_group(0);

        Ok(cmd)
    }

    /// Spawn `cmd`, feed a binary its JSON on stdin, and collect its output
    /// under the manifest's own deadline.
    ///
    /// `None` means the deadline passed and the whole process tree was killed.
    /// The deadline covers the stdin write too: a binary that never reads would
    /// otherwise block a large write forever. The manifest timeout used to be
    /// enforced only by the registry dropping this future, which killed nothing
    /// — the process ran on, orphaned, for as long as it liked.
    ///
    /// # Errors
    ///
    /// [`ToolError::ExecutionFailed`] if the process cannot be spawned, written
    /// to, or waited on.
    async fn run_contained(
        &self,
        mut cmd: Command,
        params: &HashMap<String, Value>,
    ) -> Result<Option<std::process::Output>, ToolError> {
        use tokio::io::AsyncWriteExt;

        let mut child = cmd
            .spawn()
            .map_err(|e| ToolError::ExecutionFailed(format!("Failed to spawn process: {e}")))?;
        let pid = child.id();
        let mut job = nanna_proc::ChildJob::assign(&child);
        let stdin_json = match child.stdin.take() {
            Some(stdin) => Some((
                stdin,
                serde_json::to_vec(params).map_err(|e| {
                    ToolError::InvalidParams(format!("Failed to serialize params: {e}"))
                })?,
            )),
            None => None,
        };
        let run = async move {
            if let Some((mut stdin, json)) = stdin_json {
                stdin.write_all(&json).await.map_err(|e| {
                    ToolError::ExecutionFailed(format!("Failed to write to stdin: {e}"))
                })?;
                // Dropping stdin closes it, so a binary reading to EOF proceeds.
            }
            child
                .wait_with_output()
                .await
                .map_err(|e| ToolError::ExecutionFailed(format!("Failed to wait for process: {e}")))
        };
        tokio::pin!(run);
        let deadline = std::time::Duration::from_secs(self.manifest.timeout);
        tokio::select! {
            output = &mut run => {
                // Finished and its pipes closed: anything still alive was
                // deliberately detached. Spare it (the daemon-wide job bounds it).
                if let Some(job) = job.take() {
                    job.disarm();
                }
                output.map(Some)
            }
            () = tokio::time::sleep(deadline) => {
                warn!(tool = %self.manifest.name, "Executable tool timed out; killing its tree");
                // Walk while `run` still owns the child, then sweep the job.
                if let Some(pid) = pid {
                    nanna_proc::kill_process_tree(pid).await;
                }
                if let Some(job) = job.take() {
                    job.terminate();
                }
                Ok(None)
            }
        }
    }
}

#[async_trait]
impl Tool for ExecutableTool {
    fn definition(&self) -> ToolDefinition {
        // Convert JSON Schema parameters to ToolParameter format
        let parameters = self
            .manifest
            .parameters
            .as_ref()
            .map_or_else(Vec::new, parse_json_schema_params);
        
        ToolDefinition {
            name: self.manifest.name.clone(),
            description: self.manifest.description.clone(),
            parameters,
            output_schema: None,
        }
    }

    async fn execute(&self, params: HashMap<String, Value>) -> Result<ToolResult, ToolError> {
        debug!(tool = %self.manifest.name, "Executing executable tool");

        let cmd = self.build_command(&params)?;
        let Some(output) = self.run_contained(cmd, &params).await? else {
            return Ok(ToolResult::error(format!(
                "`{}` ran past its {}s timeout and was killed with everything it started; \
                 any partial work it did is on disk",
                self.manifest.name, self.manifest.timeout
            )));
        };

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        
        if !output.status.success() {
            let code = output.status.code().unwrap_or(-1);
            warn!(tool = %self.manifest.name, code, stderr = %stderr, "Tool execution failed");
            return Ok(ToolResult::error(format!(
                "Process exited with code {code}: {stderr}"
            )));
        }
        
        if !stderr.is_empty() {
            debug!(tool = %self.manifest.name, stderr = %stderr, "Tool stderr output");
        }
        
        Ok(ToolResult::success(stdout.trim().to_string()))
    }

    fn output_target(&self) -> OutputTarget {
        OutputTarget::from(&self.manifest.output)
    }

    fn timeout_secs(&self) -> Option<u64> {
        Some(self.manifest.timeout)
    }
}

/// Substitute `{{param}}` placeholders in a command template
fn substitute_params(template: &str, params: &HashMap<String, Value>) -> Result<String, ToolError> {
    let mut result = template.to_string();
    
    // Find all {{param}} patterns
    let re = regex::Regex::new(r"\{\{(\w+)\}\}").map_err(|e| {
        ToolError::InvalidParams(format!("Invalid regex: {e}"))
    })?;
    
    for cap in re.captures_iter(template) {
        let full_match = &cap[0];
        let param_name = &cap[1];
        
        let value = params.get(param_name).ok_or_else(|| {
            ToolError::InvalidParams(format!("Missing required parameter: {param_name}"))
        })?;
        
        let value_str = match value {
            Value::String(s) => shell_escape(s),
            Value::Number(n) => n.to_string(),
            Value::Bool(b) => b.to_string(),
            _ => shell_escape(&value.to_string()),
        };
        
        result = result.replace(full_match, &value_str);
    }
    
    Ok(result)
}

/// Escape a string for shell use
fn shell_escape(s: &str) -> String {
    if cfg!(windows) {
        // Windows: use double quotes, escape internal quotes
        format!("\"{}\"", s.replace('"', "\\\""))
    } else {
        // Unix: use single quotes, escape internal single quotes
        format!("'{}'", s.replace('\'', "'\\''"))
    }
}

/// Parse JSON Schema parameters into `ToolParameter` format
fn parse_json_schema_params(schema: &Value) -> Vec<ToolParameter> {
    let mut params = Vec::new();
    
    if let Some(properties) = schema.get("properties").and_then(|p| p.as_object()) {
        let required: Vec<&str> = schema.get("required")
            .and_then(|r| r.as_array())
            .map_or_default(|arr| arr.iter().filter_map(|v| v.as_str()).collect());
        
        for (name, prop) in properties {
            // `string`, and anything absent or unrecognised, reads as a string.
            let param_type = match prop.get("type").and_then(|t| t.as_str()) {
                Some("integer") => ParameterType::Integer,
                Some("number") => ParameterType::Number,
                Some("boolean") => ParameterType::Boolean,
                Some("array") => ParameterType::Array,
                Some("object") => ParameterType::Object,
                _ => ParameterType::String,
            };
            
            let description = prop.get("description")
                .and_then(|d| d.as_str())
                .unwrap_or("")
                .to_string();
            
            // Extract enum values if present
            let enum_values = prop.get("enum")
                .and_then(|e| e.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect());
            
            params.push(ToolParameter {
                name: name.clone(),
                description,
                param_type,
                required: required.contains(&name.as_str()),
                default: prop.get("default").cloned(),
                enum_values,
            });
        }
    }
    
    params
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_substitute_params() {
        let template = "convert {{input}} -resize {{size}} {{output}}";
        let mut params = HashMap::new();
        params.insert("input".to_string(), Value::String("test.jpg".to_string()));
        params.insert("size".to_string(), Value::String("800x600".to_string()));
        params.insert("output".to_string(), Value::String("out.jpg".to_string()));
        
        let result = substitute_params(template, &params).unwrap();
        // The exact format depends on platform (quoting style)
        assert!(result.contains("test.jpg"));
        assert!(result.contains("800x600"));
        assert!(result.contains("out.jpg"));
    }

    /// Dead or a zombie awaiting its reaper: either way, no longer running.
    #[cfg(target_os = "linux")]
    fn is_running(pid: &str) -> bool {
        std::fs::read_to_string(format!("/proc/{pid}/stat"))
            .ok()
            .and_then(|stat| {
                stat.rsplit(')')
                    .next()
                    .map(|rest| rest.trim_start().to_owned())
            })
            .is_some_and(|rest| !rest.starts_with('Z') && !rest.starts_with('X'))
    }

    /// The manifest timeout is enforced here, and it kills the whole tree: the
    /// registry used to drop the future, which killed nothing, so a `sleep`
    /// grandchild (and the shell) ran on orphaned.
    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn a_timed_out_skill_is_killed_with_its_grandchildren() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("tool.yaml"),
            "name: sleeper\ndescription: sleeps\ncommand: \"sleep 30 & echo $! > child.pid; wait\"\ntimeout: 1\n",
        )
        .expect("manifest");
        let tool = ExecutableTool::from_manifest(&dir.path().join("tool.yaml")).expect("loads");

        let started = std::time::Instant::now();
        let result = tool
            .execute(HashMap::new())
            .await
            .expect("a result, not an error");
        assert!(!result.success, "{result:?}");
        assert!(
            result
                .error
                .as_deref()
                .is_some_and(|e| e.contains("timeout")),
            "{result:?}"
        );
        assert!(
            started.elapsed() < std::time::Duration::from_secs(10),
            "the deadline held"
        );

        let pid = std::fs::read_to_string(dir.path().join("child.pid")).expect("pid written");
        let pid = pid.trim();
        let gone = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while is_running(pid) && std::time::Instant::now() < gone {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        assert!(
            !is_running(pid),
            "the grandchild {pid} outlived the timeout"
        );
    }

    /// A skill that finishes in time is untouched by the containment.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_quick_skill_still_returns_its_output() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("tool.yaml"),
            "name: echoer\ndescription: echoes\ncommand: \"echo {{word}}\"\ntimeout: 5\n",
        )
        .expect("manifest");
        let tool = ExecutableTool::from_manifest(&dir.path().join("tool.yaml")).expect("loads");
        let params = HashMap::from([("word".to_owned(), Value::String("hello".to_owned()))]);
        let result = tool.execute(params).await.expect("runs");
        assert!(result.success, "{result:?}");
        assert_eq!(result.content, "hello");
    }
}
