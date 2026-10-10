//! `files.history` and `files.restore`: Nanna undoing her own writes.
//!
//! The snapshots are taken by the script bridge before every `writeFile` (see
//! `nanna_scripting::file_history`). These services are the other half — the
//! `file_history` skill lists what a session's writes displaced and puts a file
//! back. A restore snapshots the current state first, so it is undoable too.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use nanna_scripting::ServiceFn;
use nanna_scripting::file_history::{
    BASELINE_CHECKPOINTS_MAX, Checkpoint, FileHistory, RECENT_CHECKPOINTS_MAX,
};
use serde_json::{Value, json};

/// Checkpoints listed when the caller names no limit — a screenful for a model.
const LIST_DEFAULT: usize = 20;

/// Build both services over the process-wide store. Empty when none is
/// installed, which withholds the skill rather than advertising an undo that
/// has nothing behind it.
#[must_use]
pub fn build_file_history_services(
    history: Option<&'static FileHistory>,
) -> HashMap<String, ServiceFn> {
    let mut services: HashMap<String, ServiceFn> = HashMap::new();
    let Some(history) = history else {
        return services;
    };
    services.insert(
        "files.history".to_string(),
        Arc::new(move |params: Value| Box::pin(async move { list(history, &params).await })),
    );
    services.insert(
        "files.restore".to_string(),
        Arc::new(move |params: Value| Box::pin(async move { restore(history, &params).await })),
    );
    debug_assert_eq!(services.len(), 2);
    services
}

fn session_of(params: &Value) -> Option<&str> {
    params
        .get("session_id")
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
}

/// Whether checkpoint `c` is for `wanted`: the same path, or a relative
/// `wanted` naming its tail (the skill cannot resolve against the workdir).
/// Pure.
fn matches_path(c: &Checkpoint, wanted: &str) -> bool {
    let wanted = Path::new(wanted.trim());
    c.path == wanted || (wanted.is_relative() && c.path.ends_with(wanted))
}

/// List a session's checkpoints: `{session_id, path?, limit?}` →
/// `{checkpoints, total}`. Shared by the service and the `session.file_history`
/// IPC verb, so the skill and the GUI can never disagree about what exists.
pub(crate) async fn list(history: &FileHistory, params: &Value) -> Result<Value, String> {
    let limit = crate::tasks::opt_i64(params, "limit")?
        .and_then(|n| usize::try_from(n).ok())
        .unwrap_or(LIST_DEFAULT)
        // Everything a session can hold: recent checkpoints AND the baselines
        // kept beside them. Capped at the recent bound alone, "raise limit"
        // could never reach the oldest baselines it advertised.
        .clamp(1, RECENT_CHECKPOINTS_MAX + BASELINE_CHECKPOINTS_MAX);
    let path = params
        .get("path")
        .and_then(Value::as_str)
        .filter(|p| !p.trim().is_empty());
    let all = history
        .list(session_of(params), None)
        .await
        .map_err(|e| format!("The file history could not be read: {e}"))?;
    let matching: Vec<&Checkpoint> = all
        .iter()
        .filter(|c| path.is_none_or(|p| matches_path(c, p)))
        .collect();
    let shown: Vec<Value> = matching
        .iter()
        .take(limit)
        .map(|c| {
            json!({
                "checkpoint": c.seq,
                "path": c.path,
                "existed": c.existed,
                "bytes": c.bytes,
                "taken_at": c.taken_at.to_rfc3339(),
                "baseline": c.baseline,
            })
        })
        .collect();
    debug_assert!(shown.len() <= limit);
    Ok(json!({ "checkpoints": shown, "total": matching.len() }))
}

/// Restore one checkpoint: `{session_id, checkpoint}`. Shared like [`list`].
pub(crate) async fn restore(history: &FileHistory, params: &Value) -> Result<Value, String> {
    let Some(seq) = crate::tasks::opt_i64(params, "checkpoint")? else {
        return Err(
            "Nothing was restored: `checkpoint` is required. List them first; each has a number."
                .into(),
        );
    };
    let seq = u64::try_from(seq)
        .map_err(|_| format!("Nothing was restored: checkpoint {seq} cannot exist."))?;
    let restored = history
        .restore(session_of(params), seq)
        .await
        .map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => format!(
                "Nothing was restored: this conversation has no checkpoint {seq} (it may have been \
                 pruned, or belong to another conversation). List the checkpoints to see what exists."
            ),
            _ => format!("The restore of checkpoint {seq} failed: {e}"),
        })?;
    serde_json::to_value(restored).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leak_store(root: &Path) -> &'static FileHistory {
        Box::leak(Box::new(FileHistory::new(root.join("file-history"))))
    }

    #[tokio::test]
    async fn list_then_restore_through_the_services() {
        let dir = tempfile::tempdir().expect("tempdir");
        let history = leak_store(dir.path());
        let file = dir.path().join("plan.md");
        std::fs::write(&file, "v1").expect("seed");
        history
            .record_before_write(Some("s"), &file)
            .await
            .expect("snap");
        std::fs::write(&file, "v2 (bad)").expect("overwrite");

        let services = build_file_history_services(Some(history));
        let listed = services["files.history"](json!({ "session_id": "s", "path": "plan.md" }))
            .await
            .expect("list");
        assert_eq!(listed["total"], 1, "{listed}");
        let seq = listed["checkpoints"][0]["checkpoint"]
            .as_u64()
            .expect("seq");

        let other = services["files.history"](json!({ "session_id": "t" }))
            .await
            .expect("list");
        assert_eq!(other["total"], 0, "another conversation sees none of it");

        // Script engines send numbers as floats.
        let restored =
            services["files.restore"](json!({ "session_id": "s", "checkpoint": f64::from(u32::try_from(seq).expect("small seq")) }))
                .await
                .expect("restore");
        assert_eq!(restored["action"], "rewrote", "{restored}");
        assert_eq!(std::fs::read_to_string(&file).expect("read"), "v1");

        let missing =
            services["files.restore"](json!({ "session_id": "s", "checkpoint": 999 })).await;
        assert!(
            missing
                .expect_err("no such checkpoint")
                .contains("has no checkpoint 999")
        );
        let unnamed = services["files.restore"](json!({ "session_id": "s" })).await;
        assert!(
            unnamed
                .expect_err("required")
                .contains("`checkpoint` is required")
        );
    }

    #[test]
    fn without_a_store_nothing_is_registered() {
        assert!(build_file_history_services(None).is_empty());
    }
}
