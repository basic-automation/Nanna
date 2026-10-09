//! User Tool Manager - Create, edit, and manage user-defined tools
//!
//! Moved from GUI to daemon so all clients can access user tools.

use async_trait::async_trait;
use nanna_scripting::{ScriptEngine, ScriptedTool, ToolPermissions, extract_manifest};
use nanna_tools::{Tool, ToolDefinition, ToolError, ToolRegistry, ToolResult, ParameterType, ToolParameter};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

/// User-created tool metadata (persisted to disk)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserToolMeta {
    pub name: String,
    pub description: String,
    pub source: String,
    pub language: String, // "javascript" or "typescript"
    pub parameters: Option<Value>, // JSON Schema
    pub permissions: UserToolPermissions,
    pub created_at: i64,
    pub updated_at: i64,
    pub enabled: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UserToolPermissions {
    pub net: Vec<String>,
    pub read: Vec<String>,
    pub write: Vec<String>,
    pub env: bool,
    #[serde(default)]
    pub run: bool,
}

impl From<UserToolPermissions> for ToolPermissions {
    fn from(p: UserToolPermissions) -> Self {
        let mut perms = Self::none()
            .with_net(p.net)
            .with_read(p.read.into_iter().map(PathBuf::from))
            .with_write(p.write.into_iter().map(PathBuf::from));
        perms.env = p.env;
        perms.run = p.run;
        perms
    }
}

/// What a load of the user tool store found, for the boot log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UserToolLoad {
    /// Tools read from disk, disabled ones included.
    pub loaded: usize,
    /// How many of those were registered: the enabled ones.
    pub registered: usize,
}

/// Manager for user-created tools
pub struct UserToolManager {
    tools_dir: PathBuf,
    engine: Arc<ScriptEngine>,
    tools: RwLock<HashMap<String, UserToolMeta>>,
}

/// Reject a tool whose source does not parse, before it reaches disk.
///
/// `extract_manifest` reads the manifest fields out of the source by scanning
/// it as text and never parses the JavaScript, so it accepts a file whose body
/// is unparseable garbage. This checks the body through the same engine that
/// will run it — including the `export default` rewrite and IIFE wrapper the
/// runtime applies, either of which can introduce a syntax error the source
/// does not itself contain.
///
/// Refusing here rather than at call time is the whole point: a registered tool
/// that cannot run costs a turn and reports as a broken tool, while a refused
/// `create` reports as a fixable call, with the engine's own line and column.
///
/// # Errors
/// Returns the parse diagnostic, addressed to whoever wrote the source.
fn validate_tool_source(source: &str) -> Result<(), String> {
    nanna_scripting::check_syntax(source).map_err(|e| {
        format!(
            "Source does not parse, so the tool was NOT created: {e}\n\
             Nothing was written to disk. Fix the syntax and call again — the position above is \
             from the engine that would have run it."
        )
    })
}

impl UserToolManager {
    /// Create a new manager with the given tools directory
    pub fn new(tools_dir: PathBuf) -> Self {
        // Ensure directory exists
        if let Err(e) = std::fs::create_dir_all(&tools_dir) {
            warn!("Failed to create tools directory: {}", e);
        }
        
        Self {
            tools_dir,
            engine: Arc::new(ScriptEngine::new()),
            tools: RwLock::new(HashMap::new()),
        }
    }

    /// Load all user tools from disk
    ///
    /// # Errors
    ///
    /// Returns the I/O error when the tools directory cannot be listed; the
    /// cache has already been cleared by then. Individual files that cannot be
    /// read or parsed are logged and skipped.
    pub async fn load_all(&self) -> Result<usize, std::io::Error> {
        let mut count = 0;
        let mut tools = self.tools.write().await;
        tools.clear();

        let entries = std::fs::read_dir(&self.tools_dir)?;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "json") {
                match std::fs::read_to_string(&path) {
                    Ok(content) => {
                        match serde_json::from_str::<UserToolMeta>(&content) {
                            Ok(meta) => {
                                info!("Loaded user tool: {}", meta.name);
                                tools.insert(meta.name.clone(), meta);
                                count += 1;
                            }
                            Err(e) => warn!("Failed to parse tool {:?}: {}", path, e),
                        }
                    }
                    Err(e) => warn!("Failed to read tool {:?}: {}", path, e),
                }
            }
        }
        // Held from the clear to the last insert: no reader ever sees the
        // cache emptied mid-reload.
        drop(tools);

        Ok(count)
    }

    /// Save a tool to disk
    fn save_tool(&self, meta: &UserToolMeta) -> Result<(), std::io::Error> {
        let path = self.tools_dir.join(format!("{}.json", meta.name));
        let content = serde_json::to_string_pretty(meta)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        std::fs::write(path, content)
    }

    /// Create a new user tool
    ///
    /// # Errors
    ///
    /// Returns an error, writing nothing, when the name is not a safe single
    /// filename component, the source does not parse, the source has no
    /// default export with a name and description, or a tool with that name
    /// already exists; and when writing the tool file fails.
    pub async fn create_tool(
        &self,
        name: String,
        description: String,
        source: String,
        language: Option<String>,
        parameters: Option<Value>,
        permissions: Option<UserToolPermissions>,
    ) -> Result<UserToolMeta, String> {
        // Security: the name becomes `{name}.json` on disk, so it must be a safe
        // single filename component — reject path traversal before anything else.
        validate_tool_name(&name)?;
        // A user tool registers under `name`; a bundled name would replace the
        // shipped tool (and its guards) in the live registry.
        crate::tool_authoring::refuse_bundled(&name)?;

        // Validate the source PARSES. The line this replaces was
        // `let _test_tool = ScriptedTool::new(&name, &source);` under the comment
        // "Validate the source compiles" — and `ScriptedTool::new` compiles
        // nothing; it stores the string. So a tool with a syntax error was
        // written to disk, registered, and advertised to the model, and the
        // first anyone heard of it was a failed call. This is the path where
        // that matters most: an agent writes these.
        validate_tool_source(&source)?;

        // Try to extract manifest to validate
        if extract_manifest(&source).is_none() {
            return Err("Source must export default with name and description. Example:\nexport default {\n  name: \"my_tool\",\n  description: \"Does something\",\n  execute(params) { return \"result\"; }\n}".to_string());
        }

        let now = chrono::Utc::now().timestamp();
        let meta = UserToolMeta {
            name: name.clone(),
            description,
            source,
            language: language.unwrap_or_else(|| "javascript".to_string()),
            parameters,
            permissions: permissions.unwrap_or_default(),
            created_at: now,
            updated_at: now,
            enabled: true,
        };

        // Hold the write lock across the duplicate check + disk write + insert so a
        // concurrent create with the same name can't race between them and clobber
        // an existing tool. `create` is a rare admin op, so blocking here is fine.
        let mut tools = self.tools.write().await;
        if tools.contains_key(&name) {
            return Err(format!(
                "A tool named '{name}' already exists; update or delete it first"
            ));
        }
        self.save_tool(&meta).map_err(|e| e.to_string())?;
        tools.insert(name.clone(), meta.clone());
        debug_assert!(tools.contains_key(&name));
        drop(tools); // release the write lock now that the critical section is done

        info!("Created user tool: {}", name);
        Ok(meta)
    }

    /// Update an existing tool
    ///
    /// # Errors
    ///
    /// Returns an error, changing nothing, when no tool has that name, or a new
    /// source does not parse or has no default export with a name and
    /// description; and when writing the tool file fails, in which case the
    /// cached tool is left as it was.
    pub async fn update_tool(
        &self,
        name: &str,
        description: Option<String>,
        source: Option<String>,
        parameters: Option<Value>,
        permissions: Option<UserToolPermissions>,
        enabled: Option<bool>,
    ) -> Result<UserToolMeta, String> {
        let mut tools = self.tools.write().await;
        // Mutate a *clone*, persist it, and only swap it into the cache on a
        // successful write. Mutating the live entry in place (the old behavior)
        // left RAM diverged from disk if source validation failed mid-way or the
        // disk write errored.
        let mut updated = tools
            .get(name)
            .cloned()
            .ok_or_else(|| format!("Tool not found: {name}"))?;

        // Validate the new source BEFORE applying any field, so a bad edit is a
        // clean no-op rather than a partial mutation.
        if let Some(ref src) = source {
            validate_tool_source(src)?;
            if extract_manifest(src).is_none() {
                return Err(
                    "Invalid source: must export default with name and description".to_string(),
                );
            }
        }

        if let Some(desc) = description {
            updated.description = desc;
        }
        if let Some(src) = source {
            updated.source = src;
        }
        if let Some(params) = parameters {
            updated.parameters = Some(params);
        }
        if let Some(perms) = permissions {
            updated.permissions = perms;
        }
        if let Some(en) = enabled {
            updated.enabled = en;
        }
        updated.updated_at = chrono::Utc::now().timestamp();
        debug_assert_eq!(updated.name, name, "update must not rename the tool");

        // Persist first; publish to the cache only after the write succeeds.
        self.save_tool(&updated).map_err(|e| e.to_string())?;
        tools.insert(name.to_string(), updated.clone());
        // Held from the read to the insert, so two concurrent updates of one
        // tool cannot both start from the same original and lose a write.
        drop(tools);

        info!("Updated user tool: {}", name);
        Ok(updated)
    }

    /// Delete a tool
    ///
    /// # Errors
    ///
    /// Returns an error when no tool has that name, or when its file exists but
    /// cannot be removed (the tool is already gone from the cache by then).
    pub async fn delete_tool(&self, name: &str) -> Result<(), String> {
        let mut tools = self.tools.write().await;
        tools.remove(name).ok_or_else(|| format!("Tool not found: {name}"))?;

        let path = self.tools_dir.join(format!("{name}.json"));
        if path.exists() {
            std::fs::remove_file(path).map_err(|e| e.to_string())?;
        }
        // Held through the file removal, so a concurrent create of the same
        // name cannot write its file in between and have it deleted here.
        drop(tools);

        info!("Deleted user tool: {}", name);
        Ok(())
    }

    /// Get all user tools
    pub async fn list_tools(&self) -> Vec<UserToolMeta> {
        self.tools.read().await.values().cloned().collect()
    }

    /// Get a specific tool
    pub async fn get_tool(&self, name: &str) -> Option<UserToolMeta> {
        self.tools.read().await.get(name).cloned()
    }

    /// Test a tool with given input (doesn't save)
    ///
    /// # Errors
    ///
    /// Returns the script engine's failure, prefixed `✗ Execution failed:`,
    /// when the source does not run or throws.
    pub async fn test_tool(
        &self,
        source: &str,
        input: HashMap<String, Value>,
    ) -> Result<String, String> {
        let tool = ScriptedTool::new("_test", source);
        let input_value = Value::Object(input.into_iter().collect());
        
        match self.engine.execute(&tool, input_value, None, None).await {
            Ok(result) => {
                let output = match result.value {
                    Value::String(s) => s,
                    Value::Null => "(no output)".to_string(),
                    other => serde_json::to_string_pretty(&other).unwrap_or_else(|_| other.to_string()),
                };
                Ok(format!("✓ Executed in {}ms via {}\n\n{}", result.duration_ms, result.engine, output))
            }
            Err(e) => Err(format!("✗ Execution failed: {e}")),
        }
    }

    /// Create a Tool implementation for a user tool
    ///
    /// # Errors
    ///
    /// None today: wrapping stored metadata cannot fail, and a source problem
    /// surfaces when the tool executes. The `Result` is what every caller
    /// already handles.
    pub fn create_tool_impl(&self, meta: &UserToolMeta) -> Result<Arc<dyn Tool>, String> {
        let tool = ScriptedTool::new(&meta.name, &meta.source)
            .with_permissions(meta.permissions.clone().into());
        
        let wrapper = UserToolWrapper {
            meta: meta.clone(),
            tool,
            engine: self.engine.clone(),
        };
        
        Ok(Arc::new(wrapper))
    }

    /// Register all enabled user tools with the registry
    pub async fn register_with_registry(&self, registry: &ToolRegistry) -> usize {
        let tools = self.tools.read().await;
        let mut count = 0;
        
        for meta in tools.values() {
            if !meta.enabled {
                continue;
            }
            
            match self.create_tool_impl(meta) {
                Ok(tool) => {
                    registry.register_boxed(tool).await;
                    count += 1;
                }
                Err(e) => {
                    warn!("Failed to register user tool {}: {}", meta.name, e);
                }
            }
        }
        // Held across the registrations, so the set registered is one
        // consistent snapshot of the cache.
        drop(tools);
        
        count
    }
}

/// Wrapper to make `UserToolMeta` implement Tool trait
struct UserToolWrapper {
    meta: UserToolMeta,
    tool: ScriptedTool,
    engine: Arc<ScriptEngine>,
}

#[async_trait]
impl Tool for UserToolWrapper {
    fn definition(&self) -> ToolDefinition {
        let parameters = self
            .meta
            .parameters
            .as_ref()
            .map_or_else(Vec::new, parse_params_from_schema);
        
        ToolDefinition {
            name: self.meta.name.clone(),
            description: self.meta.description.clone(),
            parameters,
            output_schema: None,
        }
    }

    async fn execute(&self, params: HashMap<String, Value>) -> Result<ToolResult, ToolError> {
        let input = Value::Object(params.into_iter().collect());
        
        let result = self.engine.execute(&self.tool, input, None, None).await.map_err(|e| {
            ToolError::ExecutionFailed(format!("Script execution failed: {e}"))
        })?;
        
        // If the script returns an object with { content, success } fields,
        // respect the success flag. This lets tools like `exec` signal failure
        // (e.g. command exited nonzero) while still returning output for the LLM.
        let (content, success, data) = match &result.value {
            Value::Object(map) if map.contains_key("content") => {
                let c = map.get("content")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let s = map.get("success")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(true);
                // Pass through extra fields as structured data (e.g. "written" for write_file)
                let mut extra = serde_json::Map::new();
                for (k, v) in map {
                    if k != "content" && k != "success" {
                        extra.insert(k.clone(), v.clone());
                    }
                }
                let d = if extra.is_empty() { None } else { Some(Value::Object(extra)) };
                (c, s, d)
            }
            Value::String(s) => {
                // Heuristic: strings starting with "Error:" indicate tool failure.
                // This catches tools that return error messages as plain strings
                // without needing to update every tool script.
                let failed = s.starts_with("Error:") || s.starts_with("Error ");
                (s.clone(), !failed, None)
            }
            Value::Null => (String::new(), true, None),
            other => (serde_json::to_string_pretty(other).unwrap_or_else(|_| other.to_string()), true, None),
        };
        
        if success {
            let mut r = ToolResult::success(content);
            r.data = data;
            Ok(r)
        } else {
            // Return with success=false so notifications fire, but include the
            // full output as content so the LLM can still read what happened.
            Ok(ToolResult {
                success: false,
                content,
                error: None,
                data,
            })
        }
    }

    fn timeout_secs(&self) -> Option<u64> {
        Some(30)
    }
}

/// Maximum length of a user-tool name, in bytes.
const TOOL_NAME_LEN_MAX: usize = 64;

/// Validate a user-tool name against `^[a-z][a-z0-9_]{0,63}$`.
///
/// Security guard: the name is used to build the on-disk file `{name}.json`, so
/// an unsanitized name like `../../etc/cron.d/evil` would escape the tools dir.
/// The pattern permits only lowercase ASCII letters, digits, and underscore, and
/// requires a leading letter — none of which can form a path separator, `.`, or
/// `..`, and the length is bounded per Tiger Style.
///
/// # Errors
/// Returns a human-readable message if the name is empty, too long, or contains
/// a disallowed character.
pub(crate) fn validate_tool_name(name: &str) -> Result<(), String> {
    if name.is_empty() || name.len() > TOOL_NAME_LEN_MAX {
        return Err(format!(
            "tool name must be 1..={TOOL_NAME_LEN_MAX} chars, got {} ({name:?})",
            name.len()
        ));
    }
    let mut chars = name.chars();
    let first_ok = chars.next().is_some_and(|c| c.is_ascii_lowercase());
    let rest_ok = chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_');
    if !first_ok || !rest_ok {
        return Err(format!(
            "tool name must match ^[a-z][a-z0-9_]{{0,63}}$ (lowercase letter first, then \
             lowercase/digit/underscore): {name:?}"
        ));
    }
    Ok(())
}

/// Parse parameters from a JSON Schema value
/// Convert a JSON Schema `enum` entry to its string form. Preserves non-string
/// scalars (numbers, booleans, null) that the previous `as_str`-only path silently
/// dropped — e.g. `"enum": [1, 2, 3]` or `[true, false]` now survive.
fn enum_value_to_string(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::Null => "null".to_string(),
        other => other.to_string(),
    }
}

fn parse_params_from_schema(schema: &Value) -> Vec<ToolParameter> {
    let mut params = Vec::new();
    
    if let Some(properties) = schema.get("properties").and_then(|p| p.as_object()) {
        let required: Vec<&str> = schema.get("required")
            .and_then(|r| r.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();
        
        for (name, prop) in properties {
            let param_type = match prop.get("type").and_then(|t| t.as_str()) {
                Some("integer") => ParameterType::Integer,
                Some("number") => ParameterType::Number,
                Some("boolean") => ParameterType::Boolean,
                Some("array") => ParameterType::Array,
                Some("object") => ParameterType::Object,
                // "string", and any type this parser does not model.
                _ => ParameterType::String,
            };
            
            let description = prop.get("description")
                .and_then(|d| d.as_str())
                .unwrap_or("")
                .to_string();
            
            let enum_values = prop.get("enum")
                .and_then(|e| e.as_array())
                .map(|arr| arr.iter().map(enum_value_to_string).collect());
            
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
    fn validate_tool_name_accepts_valid() {
        for good in ["a", "my_tool", "tool2", "read_file", "x_1_2_3"] {
            assert!(validate_tool_name(good).is_ok(), "should accept {good:?}");
        }
    }

    #[test]
    fn validate_tool_name_rejects_traversal_and_bad_chars() {
        for bad in [
            "",
            "../../etc/cron.d/evil",
            "../evil",
            "a/b",
            "a\\b",
            "Tool",      // uppercase leading
            "1tool",     // digit leading
            "_tool",     // underscore leading
            "my-tool",   // hyphen
            "my tool",   // space
            "tool.json", // dot
            "..",
        ] {
            assert!(validate_tool_name(bad).is_err(), "should reject {bad:?}");
        }
        let too_long = "a".repeat(TOOL_NAME_LEN_MAX + 1);
        assert!(validate_tool_name(&too_long).is_err());
    }

    #[tokio::test]
    async fn create_tool_rejects_traversal_name() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = UserToolManager::new(dir.path().to_path_buf());
        let src = "export default { name: \"x\", description: \"d\", execute() { return \"ok\"; } }";
        let res = mgr
            .create_tool("../escaped".to_string(), "d".to_string(), src.to_string(), None, None, None)
            .await;
        assert!(res.is_err());
        // Nothing must have been written outside the tools dir.
        assert!(!dir.path().parent().unwrap().join("escaped.json").exists());
    }

    /// The defect this replaced: `create_tool`'s "Validate the source compiles"
    /// was `let _test_tool = ScriptedTool::new(...)`, which compiles nothing. A
    /// tool with unparseable garbage in its body reached disk, registered, and
    /// was advertised to the model — the first sign of trouble was a failed call.
    #[tokio::test]
    async fn create_tool_refuses_source_that_does_not_parse() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = UserToolManager::new(dir.path().to_path_buf());
        let broken = "export default { name: \"broken\", description: \"d\", \
                      execute() { var x = [;;; not javascript ((( } }";

        let res = mgr
            .create_tool(
                "broken".to_string(),
                "d".to_string(),
                broken.to_string(),
                None,
                None,
                None,
            )
            .await;

        let err = res.expect_err("a tool that cannot parse must not be created");
        assert!(err.contains("does not parse"), "{err}");
        assert!(err.contains("NOT created"), "{err}");
        // Nothing on disk: a refused create is a clean no-op, not a half-write.
        assert!(
            !dir.path().join("broken.json").exists(),
            "a refused tool must leave nothing behind"
        );
    }

    /// The positive half, so the check cannot be a constant `Err`.
    #[tokio::test]
    async fn create_tool_still_accepts_well_formed_source() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = UserToolManager::new(dir.path().to_path_buf());
        let res = mgr
            .create_tool(
                "fine".to_string(),
                "d".to_string(),
                sample_source("fine"),
                None,
                None,
                None,
            )
            .await;
        assert!(res.is_ok(), "{res:?}");
        assert!(dir.path().join("fine.json").exists());
    }

    /// A bad edit must be a clean no-op — the tool that was working keeps working.
    #[tokio::test]
    async fn update_tool_refuses_a_broken_edit_without_touching_the_good_one() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = UserToolManager::new(dir.path().to_path_buf());
        mgr.create_tool(
            "keeper".to_string(),
            "d".to_string(),
            sample_source("keeper"),
            None,
            None,
            None,
        )
        .await
        .expect("create");

        let res = mgr
            .update_tool(
                "keeper",
                None,
                Some("export default { name: \"keeper\", execute() { [;;; ((( } }".to_string()),
                None,
                None,
                None,
            )
            .await;
        assert!(res.is_err(), "a broken edit must be refused");

        let kept = mgr.get_tool("keeper").await.expect("still registered");
        assert_eq!(
            kept.source,
            sample_source("keeper"),
            "the working source must survive a refused edit"
        );
    }

    fn sample_source(name: &str) -> String {
        format!(
            "export default {{ name: \"{name}\", description: \"d\", execute() {{ return \"ok\"; }} }}"
        )
    }

    #[tokio::test]
    async fn create_tool_rejects_duplicate_name() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = UserToolManager::new(dir.path().to_path_buf());

        mgr.create_tool(
            "greet".to_string(),
            "first".to_string(),
            sample_source("greet"),
            None,
            None,
            None,
        )
        .await
        .expect("first create succeeds");

        // A second create with the same name must be rejected, not silently clobber.
        let dup = mgr
            .create_tool(
                "greet".to_string(),
                "second".to_string(),
                sample_source("greet"),
                None,
                None,
                None,
            )
            .await;
        assert!(dup.is_err(), "duplicate name must be rejected");

        // The original tool is untouched (description not overwritten).
        assert_eq!(mgr.get_tool("greet").await.unwrap().description, "first");
    }

    #[tokio::test]
    async fn update_tool_bad_source_leaves_ram_and_disk_intact() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = UserToolManager::new(dir.path().to_path_buf());
        mgr.create_tool(
            "keep".to_string(),
            "original".to_string(),
            sample_source("keep"),
            None,
            None,
            None,
        )
        .await
        .unwrap();

        // An update carrying a new description AND an invalid source must fail
        // whole — the description change must NOT leak into RAM or disk.
        let res = mgr
            .update_tool(
                "keep",
                Some("changed".to_string()),
                Some("this is not a valid tool module".to_string()),
                None,
                None,
                None,
            )
            .await;
        assert!(res.is_err(), "invalid source must fail the update");

        // RAM is unchanged.
        assert_eq!(mgr.get_tool("keep").await.unwrap().description, "original");

        // Disk is unchanged: a fresh manager reloading from the same dir sees the
        // original description (proves nothing was persisted mid-mutation).
        let reloaded = UserToolManager::new(dir.path().to_path_buf());
        reloaded.load_all().await.unwrap();
        assert_eq!(
            reloaded.get_tool("keep").await.unwrap().description,
            "original"
        );
    }

    #[test]
    fn parse_params_preserves_non_string_enums() {
        // Integer/boolean enum values must survive schema parsing (were dropped).
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "retries": { "type": "integer", "enum": [0, 1, 5] },
                "verbose": { "type": "boolean", "enum": [true, false] }
            }
        });
        let params = parse_params_from_schema(&schema);
        let retries = params.iter().find(|p| p.name == "retries").unwrap();
        assert_eq!(
            retries.enum_values.as_deref(),
            Some(&["0".to_string(), "1".to_string(), "5".to_string()][..])
        );
        let verbose = params.iter().find(|p| p.name == "verbose").unwrap();
        assert_eq!(
            verbose.enum_values.as_deref(),
            Some(&["true".to_string(), "false".to_string()][..])
        );
    }
}
