//! Long-horizon harness (P14): drive hours of agent work from the task store
//! with an O(1) re-anchored prompt per step.
//!
//! The design bet: *the agent should never need to remember; the harness makes
//! forgetting survivable.* Every step rebuilds a small prompt from the pinned
//! goal + the one actionable task + the last result — never a growing
//! transcript. Acceptance checks are machine-run by the harness (`done` is a
//! verdict, not an assertion), progress is measured by checks flipping, and a
//! stalled item is re-planned instead of ground on.
//!
//! The engine is pure orchestration over two traits — [`TaskSource`] (the P15
//! store) and [`StepRunner`] (a fresh-context agent per step) — so the whole
//! control loop is deterministically testable without a model.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use crate::cancel::CancelToken;
use std::time::{Duration, Instant};

pub use crate::loop_runner::StepKind;

/// Default acceptance-check timeout.
///
/// Bound justification: a check runs between every step; one slower than the
/// step it verifies starves the loop. Two minutes covers a test-suite run on
/// the reference tier.
pub const ACCEPTANCE_TIMEOUT_SECS_DEFAULT: u64 = 120;

/// Hard ceiling on a single acceptance-check timeout.
///
/// Bound justification: a wedged verification command must never hang the run
/// — ten minutes is beyond any sane per-item check and keeps the loop live.
pub const ACCEPTANCE_TIMEOUT_SECS_MAX: u64 = 600;

/// Maximum bytes read from a file (or captured from a command) for a regex
/// acceptance check.
///
/// Bound justification: the harness loads the target into memory to match;
/// 4 MiB caps that memory and covers any log or report worth matching.
pub const ACCEPTANCE_READ_MAX_BYTES: usize = 4 * 1024 * 1024;

/// Maximum bytes of a step's output fed forward into the next prompt and
/// recorded as a task note.
///
/// Bound justification: the re-anchored prompt is O(1) by construction — the
/// last result is one screenful; anything larger belongs in task notes and
/// memory, not the window.
pub const STEP_RESULT_TAIL_MAX_BYTES: usize = 2000;

/// Chars of an acceptance command's own output kept on the verdict, at each
/// end.
///
/// Bound justification: this is the width the verdict detail has always used
/// for its output tail — one paragraph, enough to identify what the command
/// said without reproducing it (the full output is the command's to print
/// again, and re-running it is one `exec`). The HEAD is kept as well because
/// the two ends answer different questions: a test harness names WHAT it ran
/// first and WHETHER it passed last, and a completion record that carries
/// only the last line cannot say which artifact was verified.
pub const ACCEPTANCE_OUTPUT_EXCERPT_CHARS: usize = 400;

// ---------------------------------------------------------------------------
// Acceptance checks
// ---------------------------------------------------------------------------

/// A machine-checkable done-condition, run by the harness — never the model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AcceptanceCheck {
    /// Passes when the command exits 0.
    Command {
        command: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        timeout_secs: Option<u64>,
    },
    /// Passes when the path exists (relative paths resolve in the workdir).
    FileExists { path: String },
    /// Passes when the pattern matches the file content (if `path`) or the
    /// combined output of `command`.
    Regex {
        pattern: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        path: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        command: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        timeout_secs: Option<u64>,
    },
}

/// The harness's verdict after running an acceptance check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcceptanceVerdict {
    pub passed: bool,
    /// Human/model-readable evidence (exit code, missing path, match info).
    pub detail: String,
    /// The check was killed at its timeout without producing a verdict.
    ///
    /// `passed` stays `false` — a hung check can never CLOSE an item — but a
    /// timeout is **unknown, not failed**: the command said nothing about the
    /// work, so nothing downstream may treat the timeout as evidence the
    /// work is bad. Concretely: it mints no failure signature, never reopens
    /// a verified item, never counts as a refuted completion claim, and is
    /// never itself charged or counted into any verdict — accumulating
    /// unknowns into an escalation would just fabricate a failure from
    /// things that said nothing. While the check is silent, the step beside
    /// it is judged purely by its OWN evidence, exactly like a step with no
    /// check at all. What a timeout IS evidence of is a hang — the artifact
    /// (or the check) blocks forever — so the finding is carried forward
    /// first-class and the check's next run is cost-capped at this run's
    /// measured work cost (see `run_with_timeout_cap`). (Observed
    /// 2026-08-10: one leg spent 120 of 240 minutes inside 600s check
    /// timeouts and was abandoned as "fruitless" ten minutes after proving
    /// 20 of its checks passing.)
    #[serde(default)]
    pub timed_out: bool,
    /// The verdict was DEMOTED to unknown because the evidence it reads
    /// changed since this check was last baselined.
    ///
    /// Semantically identical to [`Self::timed_out`] everywhere a budget, a
    /// completion, or a fail→pass flip is decided — the check answered a
    /// question about inputs that are no longer the inputs it was verified
    /// against, so it says nothing about the work. It is emphatically NOT a
    /// hang: no hang finding is minted, the re-stake cap is not armed, and the
    /// next run of the check (against the re-baselined evidence) decides
    /// normally. See `EvidenceGuard`.
    #[serde(default)]
    pub evidence_changed: bool,
    /// First [`ACCEPTANCE_OUTPUT_EXCERPT_CHARS`] of what the check's command
    /// actually printed, when it ran one. `detail` carries the TAIL; this is
    /// the head, so a stored completion record names the artifact the command
    /// was talking about and not just its last line.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_head: Option<String>,
}

impl AcceptanceVerdict {
    const fn pass(detail: String) -> Self {
        Self {
            passed: true,
            detail,
            timed_out: false,
            evidence_changed: false,
            output_head: None,
        }
    }

    const fn fail(detail: String) -> Self {
        Self {
            passed: false,
            detail,
            timed_out: false,
            evidence_changed: false,
            output_head: None,
        }
    }

    const fn timeout(detail: String) -> Self {
        Self {
            passed: false,
            detail,
            timed_out: true,
            evidence_changed: false,
            output_head: None,
        }
    }

    /// Attach the head of the command's own output.
    fn with_output_head(mut self, combined: &str) -> Self {
        let head = text_head(combined.trim_start(), ACCEPTANCE_OUTPUT_EXCERPT_CHARS).trim();
        if !head.is_empty() {
            self.output_head = Some(head.to_string());
        }
        self
    }

    /// Did the check fail to say anything about the work — because it hung, or
    /// because its evidence moved under it?
    ///
    /// Unknown is not failure. Every caller that must not treat a silent check
    /// as a verdict tests THIS, so a new way of being silent can never be
    /// mistaken for a failing verdict by a site that only knew about hangs.
    #[must_use]
    pub const fn is_unknown(&self) -> bool {
        self.timed_out || self.evidence_changed
    }

    /// Fold an evidence-drift finding into this verdict: the sentence is
    /// appended to the detail, and a PASSING verdict is demoted to unknown.
    ///
    /// A failing verdict keeps its verdict — a fail is never made softer by
    /// noticing that the evidence moved; the drift sentence just tells the
    /// model which file changed under the check.
    fn with_evidence_drift(mut self, sentence: &str) -> Self {
        self.detail.push('\n');
        self.detail.push_str(sentence);
        if self.passed {
            self.passed = false;
            self.evidence_changed = true;
        }
        self
    }
}

/// An abandoned item whose acceptance still fails — carried on the report
/// so the mission loop can distinguish "nothing new to plan" from "done".
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbandonedUnmet {
    pub id: i64,
    pub title: String,
    /// The failing verdict's detail at the drain sweep (bounded at capture).
    pub detail: String,
}

/// An item this run walked away from that NO machine check can speak for —
/// it carried no acceptance condition, so neither sweep can ever revive it
/// or refute it.
///
/// [`AbandonedUnmet`] is the checked half of the same story ("we gave up and
/// the environment says it is still undone"); this is the unchecked half,
/// and it is the majority: across the task store 81% of items ever abandoned
/// carried no check at all. Without this list those abandonments left a
/// COUNT and no name — the closing message could say "1 item abandoned" but
/// never which one, and in one observed session the item that vanished that
/// way was the root goal itself.
///
/// It carries the item's own last step result because that is the only
/// evidence that exists: with no check, what the model last reported is the
/// whole record of why the harness stopped trying.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbandonedUnverifiable {
    pub id: i64,
    pub title: String,
    /// The abandonment reason the harness recorded on the store.
    pub reason: String,
    /// The item's last step result at the moment it was abandoned (already
    /// bounded by [`STEP_RESULT_TAIL_MAX_BYTES`] where it was captured).
    /// None when the item was abandoned before any step of it produced one.
    pub last_result: Option<String>,
}

/// An item this run closed on the environment's own evidence — the check
/// passed, whether after a step or before any ran.
///
/// Carried on the report so the continuation planner (and a resumed turn)
/// can build on what is KNOWN instead of re-deriving it. Discovering "this
/// is already done" is knowledge, not a dry round: throwing it away is what
/// made one leg re-seed "assess starting state" four times and close 11 of
/// 26 items as already-satisfied without the planner ever hearing why.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiedOutcome {
    pub id: i64,
    pub title: String,
    /// The passing verdict's detail (bounded at capture) — names the command
    /// and what it said, i.e. the artifact state the environment confirmed.
    pub detail: String,
    /// Closed by the pre-check, with zero steps run this turn.
    pub already_satisfied: bool,
}

// Every canonical acceptance shape, quoted verbatim in every parse error.
// Defined by the store — the write boundary owns what an acceptance check
// looks like — and used here so the reader's errors and the writer's errors
// can never drift apart.
use nanna_storage::ACCEPTANCE_SHAPES;

impl AcceptanceCheck {
    /// Normalize an acceptance payload to the object the store holds, without
    /// interpreting it as a typed check.
    ///
    /// This is the SAME normalization `create`/`update` apply, so anything
    /// admitted here is admitted by the store: an admission that used a looser
    /// rule than the write path silently dropped the task instead of running
    /// it.
    ///
    /// # Errors
    /// Returns a shape-carrying message when the payload cannot be normalized.
    pub fn canonicalize(value: &serde_json::Value) -> Result<serde_json::Value, String> {
        let check = Self::from_json(value)?;
        // Round-tripping through the typed check is what makes the result
        // canonical rather than merely object-shaped: unknown keys are gone,
        // absent options are absent, and `timeout_secs` is an integer.
        serde_json::to_value(&check)
            .map_err(|e| format!("invalid acceptance check: {e}. {ACCEPTANCE_SHAPES}"))
    }

    /// Parse the store's acceptance JSON (`{kind: ..., ...}`).
    ///
    /// Tolerates the one dialect models actually emit — the object handed over
    /// as a JSON *string* — by deferring to the store's own normalization, so
    /// the reader never accepts a shape the writer would reject.
    ///
    /// # Errors
    ///
    /// Returns a message naming the accepted shapes when the value is neither
    /// an object nor a string that decodes to one (the store's
    /// normalization), or when the object does not describe a valid check —
    /// an unknown `kind`, a missing required field, or a mistyped one such as
    /// a non-integer `timeout_secs`.
    pub fn from_json(value: &serde_json::Value) -> Result<Self, String> {
        let canonical = nanna_storage::canonicalize_acceptance(value)?;
        serde_json::from_value(canonical)
            .map_err(|e| format!("invalid acceptance check: {e}. {ACCEPTANCE_SHAPES}"))
    }

    /// Short human-readable description for the step prompt.
    #[must_use]
    pub fn describe(&self) -> String {
        match self {
            Self::Command { command, .. } => {
                format!("command `{command}` must exit 0")
            }
            Self::FileExists { path } => format!("file `{path}` must exist"),
            Self::Regex {
                pattern,
                path: Some(path),
                ..
            } => format!("file `{path}` must match /{pattern}/"),
            Self::Regex {
                pattern,
                command: Some(command),
                ..
            } => format!("output of `{command}` must match /{pattern}/"),
            Self::Regex { pattern, .. } => format!("output must match /{pattern}/"),
        }
    }

    /// The shell command the model can run itself to see the same verdict the
    /// harness will apply, or `None` when the check is pure inspection with
    /// no command to run (a bare file/regex test the model can just look at).
    ///
    /// Deliberately NOT synthesized for `FileExists`/path-`Regex`: inventing
    /// a `test -f`/`grep` line would put words in the check's mouth, and the
    /// description already states the condition plainly.
    #[must_use]
    pub fn self_check_command(&self) -> Option<&str> {
        match self {
            Self::Command { command, .. }
            | Self::Regex {
                command: Some(command),
                ..
            } => Some(command),
            Self::FileExists { .. } | Self::Regex { .. } => None,
        }
    }

    /// Could this check's verdict depend on `touched` — a path a write/edit
    /// tool call just targeted? Matched on the path's FINAL COMPONENT so the
    /// relative and absolute spellings of one file collide (the step may
    /// write `D:\ws\notes.md` while the check says `notes.md`).
    ///
    /// Best effort by design, with asymmetric costs shaping the rule: a
    /// false positive runs one bounded acceptance check for nothing, while a
    /// false negative merely postpones detection to the periodic re-sweep —
    /// so match loosely and never miss the obvious case.
    #[must_use]
    pub fn references_path(&self, touched: &str) -> bool {
        let normalized = touched.replace('\\', "/");
        let Some(file_name) = normalized.split('/').rfind(|c| !c.is_empty()) else {
            return false;
        };
        let mentions = |s: &str| s.replace('\\', "/").contains(file_name);
        match self {
            Self::Command { command, .. } => mentions(command),
            Self::FileExists { path } => mentions(path),
            Self::Regex { path, command, .. } => {
                path.as_deref().is_some_and(mentions) || command.as_deref().is_some_and(mentions)
            }
        }
    }

    /// Every workspace path this check NAMES — the subject it observes plus
    /// anything its command reads.
    ///
    /// The inverse view of [`Self::references_path`], over the same three
    /// arms: that answers "could a write to X change this verdict?", this
    /// answers "which files IS this verdict about?". Extraction itself lives
    /// at the store's write boundary
    /// ([`nanna_storage::acceptance_referenced_paths`]) so the writer and
    /// every reader agree on what a check names, exactly as they already
    /// agree on what a check looks like.
    #[must_use]
    pub fn referenced_paths(&self) -> Vec<String> {
        let canonical = serde_json::to_value(self).unwrap_or(serde_json::Value::Null);
        nanna_storage::acceptance_referenced_paths(&canonical)
    }

    /// The subset of [`Self::referenced_paths`] that is this check's
    /// INSTRUMENT rather than its subject — the files its command reads.
    ///
    /// Empty for `FileExists` and a path-flavored `Regex`: those observe the
    /// deliverable directly, so the file they name is the WORK, not the judge
    /// (see [`nanna_storage::acceptance_evidence_paths`]).
    #[must_use]
    pub fn evidence_paths(&self) -> Vec<String> {
        let canonical = serde_json::to_value(self).unwrap_or(serde_json::Value::Null);
        nanna_storage::acceptance_evidence_paths(&canonical)
    }

    fn effective_timeout(timeout_secs: Option<u64>) -> Duration {
        let secs = timeout_secs
            .unwrap_or(ACCEPTANCE_TIMEOUT_SECS_DEFAULT)
            .clamp(1, ACCEPTANCE_TIMEOUT_SECS_MAX);
        Duration::from_secs(secs)
    }

    /// Run the check against real environment state. This is deliberately the
    /// only place a "task is done" signal can originate when a check exists.
    pub async fn run(&self, workdir: &Path) -> AcceptanceVerdict {
        self.run_with_timeout_cap(workdir, None).await
    }

    /// [`Self::run`] with the harness's hang re-stake cap.
    ///
    /// `cap` is the largest cost the CALLER has measured for real work or a
    /// real answer this run; when set, the command timeout is
    /// `min(configured, max(cap, 1s))`. The harness passes it only for
    /// re-runs of a check whose previous run TIMED OUT: the loop's own docs
    /// say a check must be cheaper than the step it verifies
    /// ([`ACCEPTANCE_TIMEOUT_SECS_DEFAULT`]), and once a check has consumed
    /// its entire ceiling without answering, staking another full ceiling on
    /// the same unanswered question is how one leg burned 120 of 240 minutes
    /// in 600s timeouts. Both cap terms are measured, never configured; the
    /// first decided verdict lifts the cap. A capped timeout still yields
    /// UNKNOWN — never failure — and the verdict says the cap was applied.
    pub async fn run_with_timeout_cap(
        &self,
        workdir: &Path,
        cap: Option<Duration>,
    ) -> AcceptanceVerdict {
        let capped_timeout = |configured: Duration| -> (Duration, bool) {
            cap.map_or((configured, false), |cap| {
                let floored = cap.max(Duration::from_secs(1));
                (configured.min(floored), floored < configured)
            })
        };
        match self {
            Self::Command {
                command,
                timeout_secs,
            } => {
                let (timeout, capped) = capped_timeout(Self::effective_timeout(*timeout_secs));
                run_command_check(command, workdir, timeout, capped).await
            }
            Self::FileExists { path } => {
                let resolved = resolve_in_workdir(workdir, path);
                if resolved.exists() {
                    AcceptanceVerdict::pass(format!("file exists: {}", resolved.display()))
                } else {
                    AcceptanceVerdict::fail(format!("file does not exist: {}", resolved.display()))
                }
            }
            Self::Regex {
                pattern,
                path,
                command,
                timeout_secs,
            } => {
                let regex = match regex::Regex::new(pattern) {
                    Ok(r) => r,
                    Err(e) => {
                        return AcceptanceVerdict::fail(format!("invalid regex /{pattern}/: {e}"));
                    }
                };
                let haystack = if let Some(path) = path {
                    let resolved = resolve_in_workdir(workdir, path);
                    match read_bounded(&resolved) {
                        Ok(content) => content,
                        Err(e) => {
                            return AcceptanceVerdict::fail(format!(
                                "cannot read {}: {e}",
                                resolved.display()
                            ));
                        }
                    }
                } else if let Some(command) = command {
                    let (timeout, capped) =
                        capped_timeout(Self::effective_timeout(*timeout_secs));
                    let output = run_shell(command, workdir, timeout).await;
                    match output {
                        Ok((_, combined)) => combined,
                        Err(ShellRunError::Timeout { secs }) => {
                            return AcceptanceVerdict::timeout(timeout_detail(
                                command, secs, capped,
                            ));
                        }
                        Err(ShellRunError::Other(e)) => {
                            return AcceptanceVerdict::fail(format!("command failed: {e}"));
                        }
                    }
                } else {
                    return AcceptanceVerdict::fail(
                        "regex check has neither path nor command".to_string(),
                    );
                };
                if regex.is_match(&haystack) {
                    AcceptanceVerdict::pass(format!("pattern /{pattern}/ matched"))
                        .with_output_head(&haystack)
                } else {
                    AcceptanceVerdict::fail(format!(
                        "pattern /{pattern}/ did not match ({} bytes searched)",
                        haystack.len()
                    ))
                    .with_output_head(&haystack)
                }
            }
        }
    }
}

fn resolve_in_workdir(workdir: &Path, path: &str) -> PathBuf {
    let p = Path::new(path);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        workdir.join(p)
    }
}

fn read_bounded(path: &Path) -> std::io::Result<String> {
    use std::io::Read;
    let file = std::fs::File::open(path)?;
    let mut bytes = Vec::new();
    file.take(ACCEPTANCE_READ_MAX_BYTES as u64)
        .read_to_end(&mut bytes)?;
    // Lossy: a log truncated mid-char (or with stray binary) must still be
    // matchable — an acceptance check failing on encoding would wedge runs.
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

// ---------------------------------------------------------------------------
// Evidence drift: a verdict is only as trustworthy as its inputs
// ---------------------------------------------------------------------------

/// One evidence input of an acceptance check as the environment holds it right
/// now: the file the check reads, a hash of its content, and the identity
/// (size + modification time) naming WHICH version was seen.
#[derive(Debug, Clone, PartialEq, Eq)]
struct EvidenceFile {
    /// The path exactly as the check spells it — the identity the model can
    /// act on with no translation step.
    path: String,
    /// Hash of the bytes the check would read.
    hash: u64,
    len: u64,
    /// RFC3339 modification time; empty when the platform reported none.
    modified: String,
}

impl EvidenceFile {
    /// Short, stable rendering of the content hash. Identification only — the
    /// model never has to reproduce it, only to see that it differs.
    fn short_hash(&self) -> String {
        format!("{:016x}", self.hash)
    }
}

/// Fingerprint `paths` as the environment holds them right now, sorted by path.
///
/// Cost bound: the path set is what the acceptance text names, and each file is
/// read under [`read_bounded`] — the SAME bound the check's own read obeys.
/// Guarding a verdict therefore never costs more than reaching it. Anything
/// that does not resolve to a file is skipped: a check that names a directory,
/// a flag, or a path that does not exist has nothing to fingerprint.
fn fingerprint_paths(paths: Vec<String>, workdir: &Path) -> Vec<EvidenceFile> {
    use std::hash::{Hash, Hasher};
    let mut files: Vec<EvidenceFile> = Vec::new();
    for path in paths {
        let resolved = resolve_in_workdir(workdir, &path);
        let Ok(meta) = std::fs::metadata(&resolved) else {
            continue;
        };
        if !meta.is_file() {
            continue;
        }
        let Ok(content) = read_bounded(&resolved) else {
            continue;
        };
        let mut hasher = std::hash::DefaultHasher::new();
        content.hash(&mut hasher);
        files.push(EvidenceFile {
            path,
            hash: hasher.finish(),
            len: meta.len(),
            modified: meta.modified().ok().map_or_else(String::new, |t| {
                chrono::DateTime::<chrono::Utc>::from(t)
                    .to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
            }),
        });
    }
    files.sort_by(|a, b| a.path.cmp(&b.path));
    files
}

/// Final path component, matched the way [`AcceptanceCheck::references_path`]
/// matches: relative and absolute spellings of one file must collide.
fn final_component(path: &str) -> String {
    path.replace('\\', "/")
        .split('/').rfind(|c| !c.is_empty())
        .unwrap_or("")
        .to_string()
}

/// Did the INSTRUMENT a verdict is rendered through change since that check
/// was last baselined?
///
/// A passing check proves the work is good only if the check is still asking
/// the question it was verified against. When a step edits the test the test
/// is judged by, the pass says nothing — and the shape that motivates this is
/// not malice but the ordinary one: a rewrite spiral repairs the *evidence*
/// and closes the item in the same breath as the mutation.
///
/// Two narrowings keep the guard from firing on the WORK, and both are
/// derived, not tuned:
/// 1. Only a command check has an instrument
///    ([`AcceptanceCheck::evidence_paths`]). A `file_exists` or path-`regex`
///    check observes its deliverable directly, so the file it names is
///    precisely what the step is supposed to produce.
/// 2. Only a CONTENT CHANGE to a file that was already present counts. A file
///    appearing is the work creating something; a file vanishing makes the
///    check fail on its own evidence and needs no demotion to be heard.
///
/// The detector is a content hash rather than the write ledger because `exec`
/// mutations (a `sed -i`, a `chmod` then an append) never reach the ledger.
/// The ledger is used only for ATTRIBUTION: "modified by this session" is a
/// claim about authorship, and a claim needs a record.
///
/// Bounds: hashed set = the files the acceptance command names; hash cost ≤
/// what the check itself reads; re-baselining at every verdict means one
/// demotion per (path, modification) — never a standing veto, and a legitimate
/// test edit costs exactly one named re-verification.
#[derive(Debug, Default)]
struct EvidenceGuard {
    /// Last observation of each check's evidence inputs, by check identity.
    baselines: HashMap<u64, Vec<EvidenceFile>>,
    /// Final components of every path this run's write/edit tool calls
    /// targeted, for attribution only.
    session_touched: HashSet<String>,
}

impl EvidenceGuard {
    /// Record the paths a step's write/edit calls targeted.
    fn note_touched(&mut self, paths: &[String]) {
        for path in paths {
            let component = final_component(path);
            if !component.is_empty() {
                self.session_touched.insert(component);
            }
        }
    }

    /// Take the FIRST baseline for a check, if it has none yet.
    ///
    /// Called when the harness selects the item — before its step runs — so
    /// the baseline predates the work that could modify the evidence. (The
    /// truly first moment is the acceptance's canonicalization at write time,
    /// but the store has no workspace root to resolve paths against; this is
    /// the earliest point that holds both the canonical check and the
    /// workdir.)
    fn ensure_baseline(&mut self, check: &AcceptanceCheck, workdir: &Path) {
        let id = check_identity(check);
        if self.baselines.contains_key(&id) {
            return;
        }
        self.baselines
            .insert(id, fingerprint_paths(check.evidence_paths(), workdir));
    }

    /// Re-hash the check's evidence BEFORE it runs, compare against the
    /// baseline, and re-baseline so exactly the next verdict decides.
    ///
    /// Returns the structural sentence naming the transition, or `None` when
    /// the evidence is stable (the overwhelmingly common case, and the only
    /// one in which a fail→pass flip may replenish a fruitless budget).
    fn observe(&mut self, check: &AcceptanceCheck, workdir: &Path) -> Option<String> {
        let id = check_identity(check);
        let current = fingerprint_paths(check.evidence_paths(), workdir);
        let previous = self.baselines.insert(id, current.clone());
        let previous = previous?;
        let mut clauses: Vec<String> = Vec::new();
        // Authorship is a CLAIM, so it is made only against a record: the
        // write/edit ledger this run kept. Everything else is reported as a
        // modification time, with no author named.
        let touched = &self.session_touched;
        let attribution = |file: &EvidenceFile| {
            if file.modified.is_empty() {
                String::new()
            } else if touched.contains(&final_component(&file.path)) {
                format!(", modified by this session at {}", file.modified)
            } else {
                format!(", last modified {}", file.modified)
            }
        };
        // Only a file that was ALREADY THERE and whose content moved: an
        // instrument appearing or disappearing is not tampering with a
        // verdict, and treating either as such would demote the very step
        // that built the thing.
        for now in &current {
            let Some(old) = previous.iter().find(|old| old.path == now.path) else {
                continue;
            };
            if old.hash == now.hash {
                continue;
            }
            clauses.push(format!(
                "`{}` content {} → {} ({} bytes now{})",
                now.path,
                old.short_hash(),
                now.short_hash(),
                now.len,
                attribution(now)
            ));
        }
        if clauses.is_empty() {
            return None;
        }
        Some(format!(
            "EVIDENCE CHANGED SINCE THIS CHECK WAS LAST BASELINED: {}. This check renders \
             its verdict THROUGH the file(s) named there, so a verdict reached over new \
             content is UNKNOWN rather than a pass — it is not counted toward any budget, \
             it completes nothing, and it earns no progress credit. The new content is now \
             the baseline, so the very next run of this check decides. If you changed what \
             the check runs in order to make it pass, change the artifact instead and \
             re-run the check.",
            clauses.join("; ")
        ))
    }
}

/// What the environment confirmed AT THIS INSTANT, beside the verdict text:
/// each file the check reads, its size, and its modification time.
///
/// The failure this closes: a `file_exists` verdict stores "file exists:
/// …/notes.md" and nothing else, so a later turn reading the completion back
/// learns that *a* file was there and nothing about WHICH version — the exact
/// gap a from-scratch rewrite walks through. Size and mtime are artifact
/// identity, read from the filesystem rather than from anyone's memory.
///
/// Bound: the acceptance text names the set, and each entry is a path plus two
/// numbers. Best effort — a file that has since vanished simply does not
/// appear, which is itself the truth about that instant.
fn verdict_artifacts(check: &AcceptanceCheck, workdir: &Path) -> Vec<serde_json::Value> {
    fingerprint_paths(check.referenced_paths(), workdir)
        .into_iter()
        .map(|file| {
            serde_json::json!({
                "path": file.path,
                "bytes": file.len,
                "modified": file.modified,
            })
        })
        .collect()
}

/// The verdict line carried on a [`VerifiedOutcome`] — the clamped verdict
/// plus the artifact identity, so the do-not-regress digest says WHICH file
/// holds the verified work and not merely that something passed.
fn verified_detail(verdict: &AcceptanceVerdict, artifacts: &[serde_json::Value]) -> String {
    let mut detail = text_head(&verdict.detail, 240).to_string();
    let named: Vec<String> = artifacts
        .iter()
        .filter_map(|a| {
            let path = a.get("path").and_then(serde_json::Value::as_str)?;
            let bytes = a.get("bytes").and_then(serde_json::Value::as_u64)?;
            let modified = a
                .get("modified")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("");
            Some(if modified.is_empty() {
                format!("{path} ({bytes} bytes)")
            } else {
                format!("{path} ({bytes} bytes, modified {modified})")
            })
        })
        .collect();
    if !named.is_empty() {
        detail.push_str(" — on disk: ");
        detail.push_str(&named.join(", "));
    }
    detail
}

/// Materialize the user's declared file prohibitions into the workspace
/// registry the write tools consult.
///
/// Extraction is the store's ([`nanna_storage::extract_declared_invariants`])
/// and is deliberately conservative — only an imperative sentence with an
/// explicit path referent registers anything, because the tool side FAILS OPEN
/// on a missing registry: a missed constraint costs today's behavior, an
/// invented one blocks work the user asked for.
///
/// Invariants accumulate across turns (only the user lifts one, by saying so
/// in chat), and a turn that declares nothing new never rewrites the file.
/// Every failure path is silent by design: no workspace, an unwritable
/// `.nanna/`, a corrupted registry — each degrades to "no invariants", which
/// is exactly the behavior that shipped before this existed.
async fn materialize_declared_invariants(goal: &str, workdir: &Path) {
    let fresh = nanna_storage::extract_declared_invariants(goal, "session");
    if fresh.is_empty() {
        return;
    }
    // Canonical, workspace-relative spellings: one file has one identity on
    // both sides of the contract (the write ratchet's ledger key rule).
    let root = workdir.to_string_lossy().replace('\\', "/").to_lowercase();
    let prefix = if root.ends_with('/') {
        root
    } else {
        format!("{root}/")
    };
    let fresh: Vec<nanna_storage::DeclaredInvariant> = fresh
        .into_iter()
        .map(|mut invariant| {
            if let Some(rest) = invariant.glob.strip_prefix(&prefix) {
                invariant.glob = rest.to_string();
            }
            invariant
        })
        .collect();
    let registry = workdir.join(nanna_storage::DECLARED_INVARIANTS_FILE);
    let existing = tokio::fs::read_to_string(&registry).await.unwrap_or_default();
    let Some(document) = nanna_storage::merge_declared_invariants(&existing, &fresh) else {
        return;
    };
    if let Some(parent) = registry.parent() {
        let _ = tokio::fs::create_dir_all(parent).await;
    }
    match tokio::fs::write(&registry, document).await {
        Ok(()) => tracing::info!(
            invariants = fresh.len(),
            registry = %registry.display(),
            "registered user-declared file invariants — the write tools refuse mutations \
             under them and quote the user's own sentence"
        ),
        Err(error) => tracing::warn!(
            %error,
            registry = %registry.display(),
            "could not write the declared-invariant registry — the write tools fail open, \
             so this turn behaves exactly as before"
        ),
    }
}

/// The carried-forward finding for a verdict demoted by evidence drift — the
/// evidence-drift sibling of [`hanging_check_finding`], and deliberately NOT
/// that function: a hang is a diagnosis about the artifact, while this is a
/// diagnosis about the verification setup, and telling a model its code hangs
/// when a test file moved would send it fixing the wrong thing.
fn evidence_changed_finding(check: &AcceptanceCheck, verdict: &AcceptanceVerdict) -> String {
    let mut finding = format!(
        "ACCEPTANCE VERDICT NOT COUNTED (its evidence moved — this does NOT mean the work \
         failed): {}",
        verdict.detail
    );
    if let Some(command) = check.self_check_command() {
        let _ = write!(
            finding,
            "\nThe check is: `{command}` — run it again now that the changed files are the \
             baseline, and the next verdict counts."
        );
    }
    finding
}

/// The timeout verdict's detail line. When the run was `capped`, the cut
/// announces itself (a shortened leash must never read as the command's
/// configured allowance).
fn timeout_detail(command: &str, secs: u64, capped: bool) -> String {
    let mut detail = format!(
        "`{command}` ran {secs}s without finishing and was killed — no verdict"
    );
    if capped {
        detail.push_str(
            " (re-run capped at this run's measured work cost — the full ceiling was \
             already spent once on this check without an answer)",
        );
    }
    detail
}

async fn run_command_check(
    command: &str,
    workdir: &Path,
    timeout: Duration,
    capped: bool,
) -> AcceptanceVerdict {
    match run_shell(command, workdir, timeout).await {
        Ok((code, combined)) => {
            let passed = code == Some(0);
            let tail: String = combined
                .chars()
                .rev()
                .take(ACCEPTANCE_OUTPUT_EXCERPT_CHARS)
                .collect::<String>()
                .chars()
                .rev()
                .collect();
            let detail = format!(
                "`{command}` exited {} — {}",
                code.map_or_else(|| "signal".to_string(), |c| c.to_string()),
                tail.trim()
            );
            if passed {
                AcceptanceVerdict::pass(detail).with_output_head(&combined)
            } else {
                AcceptanceVerdict::fail(detail).with_output_head(&combined)
            }
        }
        Err(ShellRunError::Timeout { secs }) => {
            AcceptanceVerdict::timeout(timeout_detail(command, secs, capped))
        }
        Err(ShellRunError::Other(e)) => {
            AcceptanceVerdict::fail(format!("`{command}` failed to run: {e}"))
        }
    }
}

/// Why [`run_shell`] could not produce an exit code. `Timeout` is a distinct
/// arm because the two failures mean OPPOSITE things to the caller: a spawn
/// failure is a verdict about the command ("this cannot run"), while a
/// timeout is the absence of one ("this never answered") — and the fruitless
/// accounting downstream must never confuse the two.
enum ShellRunError {
    Timeout { secs: u64 },
    Other(String),
}

impl std::fmt::Display for ShellRunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Timeout { secs } => write!(f, "timed out after {secs}s"),
            Self::Other(message) => write!(f, "{message}"),
        }
    }
}

/// Run a shell command, returning (exit code, combined stdout+stderr).
///
/// On Windows this prefers Git Bash `sh` when on PATH (matching the exec
/// tool's POSIX routing) and falls back to `cmd /C`.
///
/// Lifecycle matches the exec tool (`bridge.rs`): the child runs in its own
/// process group so a check that signals its group can't reach us, and a
/// timeout kills the whole tree — the shell *and* any wedged grandchild —
/// via `nanna_proc::kill_process_tree` plus a per-child Job Object
/// (`nanna_proc::ChildJob`) that also reaps a detached grandchild whose
/// shell already exited, the `foo &` shape the walk can't see.
async fn run_shell(
    command: &str,
    workdir: &Path,
    timeout: Duration,
) -> Result<(Option<i32>, String), ShellRunError> {
    let mut cmd = shell_command(command);
    cmd.current_dir(workdir)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
    #[cfg(windows)]
    {
        // Console events (Ctrl+C / Ctrl+Break) raised by or for the child
        // stop at the child. Same flag as the exec tool.
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        cmd.creation_flags(CREATE_NEW_PROCESS_GROUP);
    }
    #[cfg(unix)]
    {
        // Own process group: isolates group signals AND gives the timeout
        // path a pgid to kill.
        cmd.process_group(0);
    }
    let child = cmd
        .spawn()
        .map_err(|e| ShellRunError::Other(e.to_string()))?;
    // Capture the pid before the wait future consumes the child, so a timeout
    // can kill the whole tree rooted here (not just the shell).
    let pid = child.id();
    // Windows: contain the whole subtree in its own kill-on-close Job Object
    // (see exec_with_timeout in nanna-scripting bridge.rs for the full
    // rationale). Unix relies on the process group above.
    let mut job = nanna_proc::ChildJob::assign(&child);
    let wait = child.wait_with_output();
    tokio::pin!(wait);
    let output = tokio::select! {
        res = &mut wait => {
            let output = res.map_err(|e| ShellRunError::Other(e.to_string()))?;
            // Completed: spare deliberate background survivors (the
            // daemon-wide Job Object still bounds them).
            if let Some(job) = job.take() {
                job.disarm();
            }
            output
        },
        () = tokio::time::sleep(timeout) => {
            // Walk the live tree first, then terminate the job to sweep
            // descendants the walk can't see (detached grandchildren).
            if let Some(pid) = pid {
                nanna_proc::kill_process_tree(pid).await;
            }
            if let Some(job) = job.take() {
                job.terminate();
            }
            return Err(ShellRunError::Timeout { secs: timeout.as_secs() });
        }
    };
    let mut combined = String::from_utf8_lossy(&output.stdout).into_owned();
    combined.push_str(&String::from_utf8_lossy(&output.stderr));
    if combined.len() > ACCEPTANCE_READ_MAX_BYTES {
        // Cut on a char boundary: String::truncate panics mid-char.
        let mut cut = ACCEPTANCE_READ_MAX_BYTES;
        while cut > 0 && !combined.is_char_boundary(cut) {
            cut -= 1;
        }
        combined.truncate(cut);
    }
    Ok((output.status.code(), combined))
}

/// Locate Git-for-Windows `bash.exe`, cached. Mirrors the exec tool's routing
/// in `nanna-scripting/src/bridge.rs` — explicitly NOT WSL's
/// `C:\Windows\System32\bash.exe` (different filesystem semantics).
#[cfg(windows)]
fn git_bash_path() -> Option<&'static Path> {
    use std::sync::OnceLock;
    static CACHE: OnceLock<Option<PathBuf>> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            let mut candidates: Vec<PathBuf> = Vec::new();
            for var in ["ProgramFiles", "ProgramW6432", "ProgramFiles(x86)"] {
                if let Ok(base) = std::env::var(var) {
                    candidates.push(PathBuf::from(base).join("Git").join("bin").join("bash.exe"));
                }
            }
            if let Ok(local) = std::env::var("LOCALAPPDATA") {
                candidates.push(
                    PathBuf::from(local)
                        .join("Programs")
                        .join("Git")
                        .join("bin")
                        .join("bash.exe"),
                );
            }
            candidates.into_iter().find(|p| p.is_file())
        })
        .as_deref()
}

#[cfg(windows)]
fn shell_command(command: &str) -> tokio::process::Command {
    // Acceptance commands are POSIX like every other command in this repo.
    // Route exactly like the exec tool: Git Bash first — a bare `sh` on PATH
    // is rare on Windows, and cmd cannot run `test`/`$(...)` at all, so a
    // silent cmd fallback makes POSIX checks unwinnable.
    if let Some(bash) = git_bash_path() {
        let mut cmd = tokio::process::Command::new(bash);
        cmd.arg("-c").arg(command);
        return cmd;
    }
    let sh_available = std::env::var_os("PATH")
        .is_some_and(|paths| std::env::split_paths(&paths).any(|dir| dir.join("sh.exe").exists()));
    if sh_available {
        let mut cmd = tokio::process::Command::new("sh");
        cmd.arg("-c").arg(command);
        cmd
    } else {
        let mut cmd = tokio::process::Command::new("cmd");
        cmd.arg("/C").arg(command);
        cmd
    }
}

#[cfg(not(windows))]
fn shell_command(command: &str) -> tokio::process::Command {
    let mut cmd = tokio::process::Command::new("sh");
    cmd.arg("-c").arg(command);
    cmd
}

// ---------------------------------------------------------------------------
// Task source + step runner traits
// ---------------------------------------------------------------------------

/// One actionable item as the harness sees it: exactly what fits in an O(1)
/// prompt — the task, its done-condition, its tool scope, and a bounded tail
/// of working notes.
#[derive(Debug, Clone)]
pub struct TaskStep {
    pub id: i64,
    pub title: String,
    pub description: Option<String>,
    pub acceptance: Option<AcceptanceCheck>,
    /// Tool names to activate for this step (P14 per-item tool scoping).
    pub tool_scope: Vec<String>,
    /// Recent working notes, oldest first (bounded by the source).
    pub notes_tail: Vec<String>,
}

/// A subtask emitted by a replan step.
#[derive(Debug, Clone)]
pub struct NewSubtask {
    pub title: String,
    pub description: Option<String>,
    pub acceptance: Option<serde_json::Value>,
    pub tool_scope: Vec<String>,
}

/// The task store as the harness consumes it (implemented over the P15
/// `TaskRepository` in production, in memory in tests).
#[async_trait::async_trait]
pub trait TaskSource: Send + Sync {
    /// The one actionable item (unblocked, highest priority, leaf), or None
    /// when the plan is finished.
    async fn next(&self) -> Result<Option<TaskStep>, String>;
    /// Mark a step's item started (idempotent).
    async fn start(&self, id: i64) -> Result<(), String>;
    /// Record the completion verdict for an item.
    async fn complete(&self, id: i64, detail: serde_json::Value) -> Result<(), String>;
    /// Append a working note (the durable scratchpad).
    async fn add_note(&self, id: i64, content: &str) -> Result<(), String>;
    /// Record a harness event in the item's activity log.
    async fn log(&self, id: i64, action: &str, detail: serde_json::Value) -> Result<(), String>;
    /// Give up on an item after repeated failed replans — close it so the run
    /// can move on instead of grinding.
    async fn abandon(&self, id: i64, reason: &str) -> Result<(), String>;

    /// Reopen a closed item because the environment's verdict changed after
    /// it closed: a verified completion whose acceptance now FAILS (later
    /// work un-did it), or an abandoned item whose acceptance now PASSES
    /// (later work fixed what it was stuck on). The world moved, so the old
    /// verdict is stale evidence — the store is the checkpoint and must say
    /// what is true NOW. Default refuses so sources without a reopen path
    /// keep their closed-is-closed semantics.
    async fn reopen(&self, _id: i64, _reason: &str) -> Result<(), String> {
        Err("this source does not support reopening closed items".to_string())
    }

    /// How many OPEN children an item has, or `None` when this source cannot
    /// report it.
    ///
    /// A replan step decomposes a stalled item by adding subtasks through the
    /// store, so this is how the runner tells a replan that produced work
    /// from one that produced nothing. `None` means "unknown", and the runner
    /// then treats every replan as productive — the pre-existing behavior, so
    /// a source that does not implement this is unaffected.
    async fn open_subtasks(&self, _id: i64) -> Result<Option<usize>, String> {
        Ok(None)
    }
}

/// Request for one step: everything the runner needs to build a fresh-context
/// agent run.
#[derive(Debug, Clone)]
pub struct StepRequest {
    pub item_id: i64,
    pub step_index: usize,
    pub step_kind: StepKind,
    /// The item's title, for status display.
    ///
    /// Carried explicitly rather than parsed back out of `prompt`: the old
    /// reader took everything after `Task #id: `, so a replan step — whose
    /// prompt line reads "…title has made no verifiable progress (5 steps
    /// without the done-condition flipping)" — put that whole sentence on
    /// screen as if it were the task name.
    pub item_title: String,
    pub prompt: String,
    /// Tools to activate for the step (≤ a handful — small models degrade
    /// past 5-10 tool definitions).
    pub tool_scope: Vec<String>,
    pub token_budget: Option<u64>,
    pub max_iterations: Option<usize>,
    pub max_wall_clock: Option<Duration>,
    /// The run's shared cancellation token — the SAME token the Stop button
    /// cancels. The harness polls it only at step boundaries, and a step can
    /// legitimately run for many minutes (a chat turn's one-task plan IS a
    /// single step); the runner must thread this into the in-step agent loop
    /// so Stop aborts the in-flight LLM stream and skips queued tool calls
    /// instead of waiting the step out.
    pub cancel: Option<CancelToken>,
}

/// One tool call as seen from outside a step (digests, not payloads — the
/// parent's context must not grow when a child runs).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepToolCall {
    pub name: String,
    pub input_digest: String,
    pub output_digest: String,
    /// Whether the tool itself reported success. Carried so the harness can
    /// recognize a step's own successful work as progress evidence (the
    /// replenish rule) without ever seeing the payloads.
    pub success: bool,
}

/// What came back from one step.
#[derive(Debug, Clone)]
pub struct StepOutcome {
    pub text: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub tool_calls: Vec<StepToolCall>,
    /// Filesystem paths the step's write/edit tool calls targeted
    /// (deduplicated, best effort — see [`touched_path_of`]; empty when the
    /// step wrote nothing or the runner cannot tell, e.g. `exec` side
    /// effects). The mid-run sweep uses these to re-check verified items
    /// whose acceptance references a just-touched path IMMEDIATELY, instead
    /// of waiting for the periodic cadence.
    pub touched_paths: Vec<String>,
    /// The step ended in a degenerate generation loop with ZERO tool calls:
    /// the in-step detectors (narration loop, repetitive output, thinking
    /// spiral) fired, their one-shot nudges did not recover the model, and
    /// the run exited having never acted on the world. That is a steering
    /// problem with the GENERATION, not evidence the task is unachievable —
    /// the harness routes it to its own escalation ladder instead of the
    /// fruitless budget. (Observed 2026-08-10: an item's last two steps
    /// before abandonment were pure narration; ~30 of one leg's 100 active
    /// minutes were discarded prose, each abort also burning one of the
    /// item's five lives.)
    pub degenerate_loop: bool,
}

/// Runs one re-anchored step in a fresh context (a new `Agent` + empty
/// `AgentContext` in production; scripted in tests).
#[async_trait::async_trait]
pub trait StepRunner: Send + Sync {
    async fn run_step(&self, request: StepRequest) -> Result<StepOutcome, String>;
}

/// Folds newly-arrived user input into the running plan.
///
/// A long-horizon run can last hours, so a user message that arrives mid-run
/// must not wait for the run to end — it is admitted at the **first available
/// opportunity**, which is a step boundary. Mid-step is not an opportunity:
/// a step owns a fresh context and an item, and interrupting it would strand
/// half-finished work the acceptance check would then refute.
///
/// The split of duties is deliberate: the harness owns *when* input may
/// enter (here, and only here); the implementation owns *what* entering
/// means — in the daemon it plans the message and inserts the tasks at top
/// priority, so the very next `next()` surfaces them ahead of existing work.
#[async_trait::async_trait]
pub trait Interjector: Send + Sync {
    /// Admit any pending input into the plan. Returns the number of items
    /// admitted (0 when nothing was waiting — the overwhelmingly common
    /// case, so implementations must make that path cheap).
    ///
    /// Errors are logged and the run continues: failing to admit a message
    /// must never kill work already in flight.
    async fn interject(&self) -> Result<usize, String>;
}

// ---------------------------------------------------------------------------
// Config / report
// ---------------------------------------------------------------------------

/// Harness configuration. Every bound is per-run blast radius, not model
/// truncation: the outer loop continues across steps, so step bounds set the
/// re-anchor cadence rather than cutting work short.
#[derive(Debug, Clone)]
pub struct LongHorizonConfig {
    /// Per-run wall-clock ceiling. Default 4h — the flagship benchmark is a
    /// "4-hour task" (ROADMAP P14); callers override per run.
    pub max_wall_clock: Duration,
    /// Per-run total token ceiling (None = unbounded).
    pub max_total_tokens: Option<u64>,
    /// Steps on one item with no acceptance flip before a replan step.
    /// Progress-or-replan: 5 fruitless attempts is grinding, not working.
    pub max_steps_per_item: usize,
    /// Replans per item before the harness abandons it and moves on.
    pub max_replans_per_item: usize,
    /// Optional token budget per step. None by default: a token cap
    /// proved to truncate PRODUCTIVE steps mid-work once the artifact under
    /// construction grew (observed live: steps dying at exactly the budget on
    /// later features, feeding stall → replan pressure). Set it only when a
    /// hard per-step spend ceiling matters more than step completion.
    ///
    /// There is deliberately NO per-step iteration cap to go with it. A fixed
    /// `step_iterations: 8` was the first domino of the P22 destruction
    /// chain: 99 truncations in one leg, 88 with a tool call in flight, 84
    /// with under 200 chars of final text — each charged as "fruitless", five
    /// of those abandoning the item, once 41 seconds after the run's all-time
    /// peak write. A step now ends the way the loop level always has (the
    /// "no hard cap" rule, applied one level down): on progress exhaustion —
    /// the runner closes the step when its recent iterations produced no new
    /// information, and reserves a final tools-off iteration so the step
    /// always ends with the model saying what it did. The run's wall-clock
    /// and token budgets remain the true bounds.
    pub step_token_budget: Option<u64>,
    /// Consecutive runner errors before the run stops (circuit breaker for a
    /// dead model endpoint).
    pub max_consecutive_errors: usize,
    /// Item ids whose acceptance check may run BEFORE a step: when one of
    /// these items' checks already passes with nothing run, complete it on
    /// that verdict and take no step.
    ///
    /// This is the harness's own philosophy applied one moment earlier — the
    /// check is the only judge of "done", so a check that already passes means
    /// the work is already done, and running a step to discover that burns a
    /// model round to learn what one command already said.
    ///
    /// **Empty by default, and the emptiness is the bound.** On the FIRST plan
    /// of a turn nothing has run yet, so a done-condition that already passes
    /// says the planner wrote a condition the world happened to satisfy — not
    /// that the user's request is moot. Completing there would delete real
    /// work. It is populated only for mission CONTINUATION rounds, where the
    /// run has already acted on the world this turn and the question "is this
    /// re-proposal already satisfied?" is exactly the mission's termination
    /// question. See `chat_harness`'s continuation loop.
    ///
    /// **A SET OF IDS, not a flag, and that is the invariant.** The pre-check
    /// exists to skip work the CONTINUATION PLANNER re-proposed that the
    /// environment already satisfies; it must NEVER apply to work a user just
    /// asked for. A flag scoped to the round would cover every item the round
    /// happens to select, and the harness polls the [`Interjector`] at the top
    /// of each iteration — so a message the user sent mid-round is planned
    /// into a new item and selected under the same flag. If its planned
    /// acceptance happened to pass already, that item would close with zero
    /// steps run and the user would never be answered. Interjections are the
    /// one thing that must never be skipped. Carrying the exact ids
    /// `seed_continuation` returned makes the wrong item unreachable rather
    /// than merely unlikely: leftovers, replan subtasks and interjected asks
    /// are all outside the set and all run normally.
    pub precheck_acceptance_items: HashSet<i64>,
    /// Actor name recorded in the activity log.
    pub actor: String,
}

impl Default for LongHorizonConfig {
    fn default() -> Self {
        Self {
            max_wall_clock: Duration::from_hours(4),
            max_total_tokens: None,
            max_steps_per_item: 5,
            max_replans_per_item: 2,
            step_token_budget: None,
            max_consecutive_errors: 3,
            precheck_acceptance_items: HashSet::new(),
            actor: "harness".to_string(),
        }
    }
}

/// Why the run stopped. Every exit produces a report — the harness analogue
/// of "always emit done:true on every exit".
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum StopReason {
    /// `next()` returned None — the plan is finished.
    AllTasksDone,
    WallClockExhausted,
    TokenBudgetExhausted,
    Cancelled,
    /// The task source failed (storage error).
    SourceError {
        message: String,
    },
    /// Too many consecutive step-runner errors.
    RunnerErrors {
        message: String,
    },
}

/// Final report: the governing metric is `tokens_per_completed_item`
/// (task success @ tokens), not tokens per turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LongHorizonReport {
    pub stop: StopReason,
    pub steps_taken: usize,
    /// Total tool calls made across every step.
    ///
    /// The signal for "this run ACTED on the world", which is what separates a
    /// conversational reply from work in progress. Step and item counts cannot
    /// carry that: a 42-feature build whose planner emits one task and whose
    /// first step completes it looks numerically identical to "hi".
    pub tool_calls: usize,
    /// The subset of `tool_calls` that CHANGED SOMETHING — the work-evidence
    /// set the harness-continuation and liveness rungs read
    /// (`is_work_evidence_tool`: writes, edits, the shell). The
    /// completion-claim rung uses a STRICTER predicate of its own
    /// (`claim_evidence_armed`), because a read-only shell success is proof a
    /// run acted but not proof it produced anything to claim. Reads and
    /// searches can run forever without
    /// touching anything outside the context window, so `tool_calls > 0`
    /// answers "did this run act?" while only this field answers "did this run
    /// change the world?".
    ///
    /// The caller that needs the distinction is the chat harness's
    /// mission-continuation loop: a round that changed nothing and closed
    /// nothing is not progress, whatever its tasks were titled.
    #[serde(default)]
    pub side_effect_tool_calls: usize,
    pub items_completed: usize,
    /// Items completed on the model's claim alone (no acceptance check
    /// existed) — logged so unverified completions are visible.
    pub items_completed_unverified: usize,
    /// Abandoned items the drain sweep revived because their acceptance now
    /// passes — later work fixed what they were stuck on, and the stale
    /// verdict was corrected without spending a model step.
    #[serde(default)]
    pub items_revived: usize,
    /// Verified completions a sweep reopened because their acceptance fails
    /// again — later work un-did them, and "done" must not outlive the
    /// evidence it was based on. Counted from BOTH sweep halves: the mid-run
    /// sweep at step boundaries (write-touch triggered or periodic) and the
    /// drain sweep at plan exhaustion.
    #[serde(default)]
    pub items_regressed_reopened: usize,
    /// The subset of `items_completed` closed by the acceptance PRE-CHECK:
    /// their done-condition already passed before any step ran, so the run
    /// completed them for free (see
    /// `LongHorizonConfig::precheck_acceptance_items`).
    ///
    /// Counted apart because these completions are evidence the goal was
    /// ALREADY met, not evidence this run advanced it. The chat harness's
    /// mission loop needs that distinction: a continuation round whose every
    /// item was already satisfied changed nothing, and must count dry.
    #[serde(default)]
    pub items_already_satisfied: usize,
    pub items_abandoned: usize,
    /// Abandoned items whose acceptance STILL FAILED at the drain sweep —
    /// live evidence the goal is provably unmet when the plan drained.
    /// The mission loop needs this: "the planner proposed nothing new" and
    /// "the goal is done" are different claims, and a failing check the run
    /// itself walked away from refutes the second (observed 2026-08-09: a
    /// turn ended dry at 5/42 with its abandoned item's check failing —
    /// the evidence existed and was discarded).
    #[serde(default)]
    pub abandoned_unmet: Vec<AbandonedUnmet>,
    /// Every item this run abandoned that carried NO acceptance check, with
    /// the reason and its last step result (see [`AbandonedUnverifiable`]).
    ///
    /// Disjoint from `abandoned_unmet` by construction: an abandoned item
    /// either has a check — in which case a sweep decides whether it is
    /// unmet or revivable — or it does not, and lands here. Together the two
    /// lists name EVERY abandonment, which is what the closing message needs
    /// to stop reporting walked-away work as a bare count.
    #[serde(default)]
    pub abandoned_unverifiable: Vec<AbandonedUnverifiable>,
    /// Every item this run closed on a PASSING check (post-step or
    /// pre-check), with the verdict that closed it. The knowledge half of the
    /// report: `abandoned_unmet` says where the goal is provably unmet, this
    /// says what is provably done — the continuation planner needs both to
    /// plan forward instead of re-litigating the past.
    #[serde(default)]
    pub verified_outcomes: Vec<VerifiedOutcome>,
    /// Acceptance runs that TIMED OUT this run — checks that produced no
    /// verdict. Unknown is not failure: none of these charged a fruitless
    /// budget, but each is a first-class hang finding carried on the item.
    #[serde(default)]
    pub acceptance_unknown: usize,
    /// The most recent step-runner error seen during the run, kept so a
    /// caller can say WHY when the plan drained through abandonment rather
    /// than completion. Poison containment turns a deterministic runner
    /// fault (e.g. "no provider for the configured model") into abandoned
    /// items and a clean `AllTasksDone` — without this field that failure is
    /// indistinguishable from a finished run.
    #[serde(default)]
    pub last_runner_error: Option<String>,
    pub replans: usize,
    /// Completion claims that the acceptance check refuted (the
    /// "false success" counter — the P14 anti-drift keystone at work).
    pub false_success_claims: usize,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub wall_clock_secs: u64,
    /// total tokens / items completed. None when nothing completed.
    pub tokens_per_completed_item: Option<u64>,
    /// Items admitted mid-run by an [`Interjector`] — user messages that
    /// joined the plan at a step boundary instead of waiting for the run.
    #[serde(default)]
    pub interjected_items: usize,
}

// ---------------------------------------------------------------------------
// Prompt building (pure, byte-stable prefix)
// ---------------------------------------------------------------------------

/// The DO-NOT-REGRESS digest: what this run has already proven working,
/// rendered for the model that EDITS rather than only the planner that plans.
///
/// Newest first — the freshest verdicts describe the current artifact best —
/// and bounded by [`STEP_RESULT_TAIL_MAX_BYTES`], the one-screenful prompt
/// bound the last-result block and the regression notice already obey, with
/// the cut announced. The digest's SIZE is evidence-derived: one line per item
/// this run closed on a passing check, each of which cost a real execution.
///
/// Voicing is the point. "Verified working" plus a stated cost for losing it
/// is what turns the block from background trivia into a constraint: the
/// observed failure was a full-file rewrite that silently dropped features
/// whose checks had passed minutes earlier, with nothing in the step's context
/// naming them.
#[must_use]
pub fn verified_digest(verified: &[VerifiedOutcome]) -> Option<String> {
    if verified.is_empty() {
        return None;
    }
    let mut text = String::from(
        "These are VERIFIED WORKING right now — each one's done-condition was run by the \
         harness and PASSED, and the verdict names the artifact state the environment \
         confirmed. A rewrite that loses any of these is a regression, not progress: if \
         your change must drop one, say which and why before you make it. Otherwise \
         preserve them — read the file and extend it, do not reconstruct it.\n",
    );
    let mut omitted = 0usize;
    for outcome in verified.iter().rev() {
        let line = format!("- #{} {}: {}\n", outcome.id, outcome.title, outcome.detail);
        if text.len() + line.len() > STEP_RESULT_TAIL_MAX_BYTES {
            omitted += 1;
            continue;
        }
        text.push_str(&line);
    }
    if omitted > 0 {
        let _ = writeln!(
            text,
            "- …and {omitted} more verified item(s) not shown here — they are equally \
             established; ask the task store for the full list before a wide rewrite."
        );
    }
    Some(text)
}

/// Build the re-anchored step prompt.
///
/// Layout is deliberate and load-bearing:
/// - The **prefix** (goal + working rules) is byte-stable across every step
///   of a run — the shape KV-prefix caching rewards. The goal is pinned
///   verbatim and never summarized: everything else is compressible, intent
///   is not.
/// - The **dynamic tail** (current task, verdict, do-not-regress digest,
///   budget) comes last, in recent attention where a small model actually
///   looks.
#[must_use]
pub fn build_step_prompt(
    goal: &str,
    step: &TaskStep,
    last_result: Option<&str>,
    verified_digest: Option<&str>,
    budget_line: &str,
) -> String {
    let mut prompt = stable_prefix(goal);
    prompt.push_str("== CURRENT TASK ==\n");
    let _ = writeln!(prompt, "Task #{}: {}", step.id, step.title);
    if let Some(description) = &step.description
        && !description.is_empty() {
            prompt.push_str(description);
            prompt.push('\n');
        }
    match &step.acceptance {
        Some(check) => {
            let _ = writeln!(
                prompt,
                "Done when (checked by the harness, not by you): {}",
                check.describe()
            );
            // Close the feedback loop INSIDE the step. Measured live
            // (gemma4:12b, 42-feature ladder): across two hours the model
            // never once ran its own acceptance command, even though the
            // line above names it — so it edited blind, learned the verdict
            // only at the step boundary, and re-submitted the same broken
            // implementation repeatedly. Checking costs one `exec`; not
            // checking costs a whole step.
            if let Some(command) = check.self_check_command() {
                let _ = writeln!(
                    prompt,
                    "Before you finish this step, RUN THAT CHECK YOURSELF: exec `{command}`. \
                     Read its output, fix what it reports, and run it again until it passes. \
                     Do not end the step on an unverified guess."
                );
            }
        }
        None => {
            prompt.push_str(
                "No machine check exists for this task. When it is genuinely finished, say \
                 TASK COMPLETE on its own line.\n",
            );
        }
    }
    if !step.notes_tail.is_empty() {
        prompt.push_str("Working notes so far:\n");
        for note in &step.notes_tail {
            let _ = writeln!(prompt, "- {note}");
        }
    }
    if let Some(last) = last_result
        && !last.is_empty() {
            let _ = writeln!(prompt, "\n== LAST RESULT ==\n{last}");
        }
    // Beside the last result, in the same recent-attention band: what must
    // survive this step. The digest is deliberately NOT part of the stable
    // prefix — it is state, it changes as work is verified, and it belongs
    // where a small model actually looks.
    if let Some(digest) = verified_digest
        && !digest.is_empty() {
            let _ = write!(prompt, "\n== VERIFIED WORKING (do not regress) ==\n{digest}");
        }
    let _ = writeln!(prompt, "\n{budget_line}");
    prompt
}

/// Build the replan prompt: decompose a stalled item into subtasks *in the
/// task store* (via the todo tool) — no fragile text parsing of plans.
#[must_use]
pub fn build_replan_prompt(goal: &str, step: &TaskStep, stall_summary: &str) -> String {
    let mut prompt = stable_prefix(goal);
    prompt.push_str("== REPLAN REQUIRED ==\n");
    let _ = writeln!(
        prompt,
        "Task #{}: {} has made no verifiable progress ({stall_summary}).",
        step.id, step.title
    );
    // Describes the OPERATION, not a tool signature. Naming a tool and its
    // exact arguments in a prompt that may not ship that tool's schema is how
    // the model ends up obeying blind and guessing parameter names — measured
    // at 53-66% malformed calls. Discovery is how it gets the real signature.
    let _ = writeln!(
        prompt,
        "Break it into 2-5 smaller subtasks, added to the task store as children of \
         task #{} — discover the task-management tool if you do not already have it. \
         Each subtask needs a machine-checkable acceptance condition (a command to run, \
         a file that must exist, or a pattern that must match) wherever possible. Make \
         the first subtask small enough to finish in one step. Do not attempt the work \
         itself in this step — only decompose.",
        step.id
    );
    prompt
}

/// The byte-stable prompt prefix: system-adjacent rules + the immutable goal.
fn stable_prefix(goal: &str) -> String {
    format!(
        "== GOAL (immutable — this is the whole point of the run) ==\n{goal}\n\n\
         == HOW TO WORK ==\n\
         You are executing one step of a long-running plan. You do not need to remember \
         anything between steps — the plan and everything you have already done are kept \
         for you.\n\
         - Advance the CURRENT TASK below by exactly one concrete action.\n\
         - SAY WHAT YOU ARE DOING: open with ONE short sentence naming the action you are \
           about to take and why it follows from the last result. Someone is watching this \
           run and that sentence is all they see.\n\
         - GET THE RIGHT TOOL FIRST. You start each run able to see only one tool, for \
           discovering the others. Everything you need — running commands, reading, \
           writing and editing files, searching your memory — exists and is waiting behind \
           it. Ask for what the task needs, then use the real tool.\n\
           Do not improvise around a tool you cannot see: writing a file by shell \
           redirection instead of the file tool, or piping a heredoc instead of editing, \
           skips the checks that keep this run alive and is how work gets silently lost.\n\
         - Use the tools you have; do not narrate actions you did not take. Describe what \
           actually happened, including failures — a wrong result reported honestly is worth \
           more than a confident guess.\n\
         - ONE artifact per job: change the real file in place. Do not build a parallel copy \
           (`thing_v2`, `thing.new`, `thing.backup`) — whatever reads the real file keeps \
           reading the original, so work in a copy never reaches the person who asked for \
           it. This applies to shell redirects too, not just the file tools.
\
         - Never mark work done yourself: the harness verifies the task's done-condition \
           and records completion.\n\n"
    )
}

/// One-line budget status the model can see — an agent that knows its budget
/// plans around it.
#[must_use]
pub fn budget_line(
    steps_taken: usize,
    items_completed: usize,
    tokens_used: u64,
    max_total_tokens: Option<u64>,
    elapsed: Duration,
    max_wall_clock: Duration,
) -> String {
    let token_part = max_total_tokens.map_or_else(
        || format!("{tokens_used} tokens used"),
        |max| format!("{tokens_used} of {max} tokens used"),
    );
    format!(
        "== BUDGET == step {}; {} items completed; {}; {} of {} minutes elapsed",
        steps_taken + 1,
        items_completed,
        token_part,
        elapsed.as_secs() / 60,
        max_wall_clock.as_secs() / 60
    )
}

/// Did the model claim the current task is finished?
///
/// Only honored for items *without* an acceptance check — with a check, the
/// environment is the only judge (false-success claims are the top documented
/// long-horizon failure mode).
#[must_use]
pub fn step_claims_completion(text: &str) -> bool {
    text.lines()
        .any(|line| line.trim().eq_ignore_ascii_case("TASK COMPLETE"))
}

/// Cross-step repetition: two consecutive steps on the same item made the
/// same non-empty tool-call sequence with the same results — semantically a
/// stall even when the narration varies.
#[must_use]
pub fn steps_repeat(previous: &[StepToolCall], current: &[StepToolCall]) -> bool {
    !current.is_empty() && previous == current
}

/// Fingerprint of a step's answer when the step is a pure answer: no tool
/// calls, not a degenerate (narration) loop, and some text. `None` otherwise.
///
/// Whitespace is collapsed so a re-flowed copy of the same answer matches;
/// anything else — a changed word, different casing — does not. Exact on
/// purpose: this feeds [`answer_converged`], which completes an item, and a
/// false match there ends a turn early while a missed one only costs a step.
#[must_use]
pub fn tool_free_answer_fingerprint(outcome: &StepOutcome) -> Option<u64> {
    use std::hash::{Hash, Hasher};
    if !outcome.tool_calls.is_empty() || outcome.degenerate_loop {
        return None;
    }
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    let mut words = 0_usize;
    for word in outcome.text.split_whitespace() {
        word.hash(&mut hasher);
        words += 1;
    }
    debug_assert!(
        words > 0 || outcome.text.trim().is_empty(),
        "non-blank text always has a word"
    );
    (words > 0).then(|| hasher.finish())
}

/// The first clause of an abandonment reason: how many steps ran on the item
/// (replans included, counted where the run counts its steps), and what the
/// no-progress budget charged.
///
/// The budget is a charge counter, not a step counter: a step that repeats
/// the previous one is charged twice, a step with fresh evidence resets it,
/// and replans are not charged at all. Reporting the charge as "N fruitless
/// steps" told the user a turn that ran 6 steps had run 8.
#[must_use]
pub fn abandonment_headline(steps_run: usize, fruitless_charges: usize, replans: usize) -> String {
    debug_assert!(replans <= steps_run, "a replan is a step");
    debug_assert!(fruitless_charges > 0, "abandonment spends the budget");
    format!(
        "abandoned after {steps_run} steps ({replans} of them replans) made no verifiable \
         progress; {fruitless_charges} no-progress charges spent (a step repeating the last \
         one is charged twice)"
    )
}

/// Did an unchecked item's answer converge — this step, like the one before
/// it on the same item, answered with no tool calls, and said the same thing?
///
/// For an item with no machine check the model's word is the verdict, and it
/// is asked to say `TASK COMPLETE`. Small models often answer and simply do
/// not say it; each further step then re-streams the same answer to the user
/// until the fruitless budget abandons the item — a plain question came back
/// as its answer seven times plus "could not finish". Repeating an answer
/// verbatim with nothing new done in between is the model saying it has
/// nothing to add, which is what `TASK COMPLETE` means.
#[must_use]
pub fn answer_converged(previous: Option<u64>, current: Option<u64>) -> bool {
    current.is_some() && previous == current
}

// ---------------------------------------------------------------------------
// The runner
// ---------------------------------------------------------------------------

/// Errors from the step runner charged to ONE item before that item is
/// abandoned as poisoned.
///
/// Bound justification: the runner already retries transient provider
/// failures internally, so an `Err` here means several consecutive failed
/// generations. When that happens twice on the same item while other items
/// proceed, the failure follows the item (something in its prompt/notes
/// deterministically breaks the model) — containing it at item granularity
/// costs one feature instead of the whole run. Provider-wide death still
/// trips the run-level breaker, whose counter spans items.
pub const ITEM_RUNNER_ERRORS_MAX: usize = 2;

/// Consecutive zero-tool-call degenerate-loop steps on one item that ride
/// the harness's steering ladder before they start charging the fruitless
/// budget.
///
/// Derivation: the ladder IS the bound — one rung per escalation level of
/// the existing wrap-up nudge ladder (gentle → firm → urgent,
/// [`crate::loop_runner::NudgeLevel`]). A generation loop is a steering
/// problem, so each such step gets one escalating steer; once the model has
/// ignored all three levels, further prose-only steps are evidence the item
/// cannot proceed here and charge normally so abandonment still converges.
pub const NARRATION_LADDER_STEPS: usize = 3;

/// File EVERY abandonment on the list that can speak for it, so no item the
/// harness walked away from survives only as a count.
///
/// The two lists are disjoint and exhaustive: an item with a check goes to
/// the sweep set, where a later run of that check decides whether it is
/// revivable or provably unmet; an item without one can never be re-judged,
/// so its name, the reason, and its own last result ARE the record. Both
/// abandonment sites call this — the earlier omission at the runner-error
/// site is exactly how a checked item could be abandoned and then reach no
/// sweep at all.
fn record_abandonment(
    step: &TaskStep,
    reason: &str,
    last_result: Option<String>,
    sweep_set: &mut Vec<(i64, String, AcceptanceCheck)>,
    unverifiable: &mut Vec<AbandonedUnverifiable>,
) {
    match &step.acceptance {
        Some(check) => sweep_set.push((step.id, step.title.clone(), check.clone())),
        None => unverifiable.push(AbandonedUnverifiable {
            id: step.id,
            title: step.title.clone(),
            reason: reason.to_string(),
            last_result,
        }),
    }
}

/// Per-item progress bookkeeping.
#[derive(Debug, Default, Clone)]
struct ItemProgress {
    steps_without_progress: usize,
    replans: usize,
    /// Replan steps on this item that added no subtasks at all.
    ///
    /// Non-zero means the decomposition ask has already come back empty
    /// here, so the harness stops asking it: the next stall rung on this
    /// item escalates instead of repeating the identical question (see
    /// [`stalled_escalation_text`]). Also what lets the abandonment reason
    /// say the ask returned nothing rather than describing the outcome as
    /// though a plan had been made.
    dry_replans: usize,
    /// Escalated next-action asks this item has taken (the rung that
    /// replaces a repeated decomposition ask). Reporting only — it is what
    /// lets the abandonment reason name the rung that actually ran.
    escalated_asks: usize,
    tokens_spent: u64,
    last_result: Option<String>,
    last_tool_calls: Vec<StepToolCall>,
    /// [`tool_free_answer_fingerprint`] of this item's previous step — what
    /// [`answer_converged`] compares against.
    last_answer_fingerprint: Option<u64>,
    /// Steps actually run on this item. Reporting only: the abandonment
    /// reason used to present `steps_without_progress` — a CHARGE counter,
    /// where a step repeating the last one is charged twice — as a count of
    /// steps, so a turn that ran 6 steps said "abandoned after 8 fruitless
    /// steps".
    steps_run: usize,
    runner_errors: usize,
    /// Consecutive acceptance runs that timed out (reset by any DECIDED
    /// verdict, pass or fail). Reporting and framing only — it sizes the
    /// hang finding, flavors the replan prompt and the abandonment reason.
    /// It is never a trigger: unknowns are not counted into any verdict;
    /// while the check is silent the step beside it is judged purely by its
    /// own evidence, exactly like a step with no check at all.
    consecutive_check_timeouts: usize,
    /// Zero-tool-call degenerate-loop steps this item has absorbed (the
    /// harness-level steering ladder, see [`NARRATION_LADDER_STEPS`]).
    narration_steps: usize,
    /// Digests of SUCCESSFUL side-effectful tool calls this item's steps have
    /// produced ([`crate::loop_runner::is_work_evidence_tool`]). A digest
    /// never seen before is the success mirror of the novel-failure rule: the
    /// step verifiably did NEW work on the world, which replenishes the
    /// fruitless budget by the environment's own evidence. Repeats (the
    /// byte-identical rewrite treadmill) earn nothing.
    seen_success_digests: std::collections::HashSet<u64>,
    /// Normalized signatures ([`failure_signature`]) of every acceptance
    /// failure this item has seen. A failure whose signature was NEVER seen
    /// before is evidence the work moved — the check now fails differently —
    /// and replenishes the fruitless budget (the same principle as stream
    /// retries replenishing on tool progress). Revisiting a signature from
    /// this set is oscillation, not progress, and charges normally — so
    /// genuine thrash still converges on the abandonment rung (observed
    /// 2026-08-08: one item advanced its failing assertion on the exact
    /// check where the progress-blind counter abandoned it, while a
    /// thrashing sibling cycled three known signatures without advancing).
    seen_failure_signatures: std::collections::HashSet<u64>,
    /// Has the acceptance pre-check already run for this item in this run?
    ///
    /// The bound on the pre-check's cost: ONE extra acceptance execution per
    /// item per run, taken at the item's first selection. Without it a
    /// grinding item would re-run its check before every step as well as
    /// after it, doubling the command executions of the slowest thing the
    /// harness does — and re-asking a question the post-step verdict answered
    /// moments earlier, from the same state.
    acceptance_prechecked: bool,
}

/// The long-horizon control loop.
pub struct LongHorizonRunner {
    pub config: LongHorizonConfig,
}

impl LongHorizonRunner {
    #[must_use]
    pub const fn new(config: LongHorizonConfig) -> Self {
        Self { config }
    }

    /// Drive the plan to completion (or to a budget/error stop). Every exit
    /// path returns a report.
    pub async fn run(
        &self,
        goal: &str,
        source: &dyn TaskSource,
        runner: &dyn StepRunner,
        workdir: &Path,
        cancel: Option<CancelToken>,
    ) -> LongHorizonReport {
        self.run_with_interjector(goal, source, runner, workdir, cancel, None)
            .await
    }

    /// [`Self::run`], plus a hook that admits new user input at each step
    /// boundary. This is the shape chat uses: a run is live for as long as
    /// the plan takes, and messages sent meanwhile join it rather than
    /// queueing behind it.
    pub async fn run_with_interjector(
        &self,
        goal: &str,
        source: &dyn TaskSource,
        runner: &dyn StepRunner,
        workdir: &Path,
        cancel: Option<CancelToken>,
        interjector: Option<&dyn Interjector>,
    ) -> LongHorizonReport {
        let mut run =
            HarnessRun::new(&self.config, goal, source, runner, workdir, cancel, interjector);

        // USER-DECLARED FILE INVARIANTS. Durable file prohibitions the user
        // stated in the goal itself ("never create, edit or delete anything
        // under tests/") are materialized to the workspace registry the write
        // tools consult before mutating. Best effort at every step: a
        // workspace that cannot be written simply keeps today's behavior,
        // because the tool side fails open on a missing registry.
        materialize_declared_invariants(goal, workdir).await;

        let stop = loop {
            if let ControlFlow::Break(IterationEnd::Stop(reason)) = run.iteration().await {
                break reason;
            }
        };
        run.into_report(stop)
    }
}

/// How one pass of the harness loop ended when it ended early.
enum IterationEnd {
    /// Go straight on to the next pass (the loop's `continue`).
    Next,
    /// End the run with this reason (the loop's `break`).
    Stop(StopReason),
}

/// The outcome of one phase of a harness pass: `Continue` carries the phase's
/// result on to the next phase, `Break` ends the pass (see [`IterationEnd`]).
type Phase<T = ()> = ControlFlow<IterationEnd, T>;

/// End the current pass and start the next one.
const fn next_iteration<T>() -> Phase<T> {
    ControlFlow::Break(IterationEnd::Next)
}

/// End the run with `reason`.
const fn stop<T>(reason: StopReason) -> Phase<T> {
    ControlFlow::Break(IterationEnd::Stop(reason))
}

/// Everything one [`LongHorizonRunner::run_with_interjector`] call keeps
/// between passes of its loop: the caller's inputs and the run's ledgers.
struct HarnessRun<'a> {
    cfg: &'a LongHorizonConfig,
    goal: &'a str,
    source: &'a dyn TaskSource,
    runner: &'a dyn StepRunner,
    workdir: &'a Path,
    cancel: Option<CancelToken>,
    interjector: Option<&'a dyn Interjector>,
    interjected_items: usize,
    started: Instant,
    steps_taken: usize,
    tool_calls_total: usize,
    side_effect_tool_calls: usize,
    items_completed: usize,
    items_completed_unverified: usize,
    items_already_satisfied: usize,
    items_abandoned: usize,
    items_revived: usize,
    items_regressed_reopened: usize,
    // Verdicts are per-moment; the drain sweep below and the MID-RUN
    // sweep at each step boundary re-ask the environment about items this
    // run closed with a check, so a later step that un-did verified work
    // (or fixed abandoned work) is caught while there is still budget to
    // act on it.
    verified_this_run: Vec<(i64, String, AcceptanceCheck)>,
    abandoned_this_run: Vec<(i64, String, AcceptanceCheck)>,
    // Rebuilt at every drain sweep; the FINAL sweep's state ships on the
    // report — an item revived later must not linger as "unmet".
    abandoned_unmet: Vec<AbandonedUnmet>,
    // Append-only, and never rebuilt: an item with no check cannot be
    // revived by any sweep, so what is recorded here at the moment of
    // abandonment stays true for the rest of the run.
    abandoned_unverifiable: Vec<AbandonedUnverifiable>,
    // Each item may be reopened at most once per run — the bound that
    // makes BOTH sweeps (mid-run and drain) a fixpoint instead of a loop.
    reopened_once: HashSet<i64>,
    // The mid-run sweep's spend ledger: verification time the run has
    // paid (prechecks + post-step verdicts) funds re-verification time
    // (see `resweep_due` — re-asking may never out-spend asking).
    check_cost_paid: Duration,
    check_runs: u32,
    sweep_cost_paid: Duration,
    // Pending "you un-did verified work" notice, consumed by the NEXT
    // execute step's prompt — whatever item that step serves.
    regression_notice: Option<String>,
    precheck_ids: HashSet<i64>,
    // Run-wide ledger of DECIDED check outcomes by check identity. A
    // fail→pass flip anywhere (a pre-check on a re-proposed item, a
    // post-step verdict, a sweep revival) is a verified environment
    // change and replenishes every open item's fruitless budget — a
    // check that flipped is progress even while another keeps failing
    // identically (P22: one leg climbed 12→16 passing while its selected
    // item's own check failed "the same way", earned zero credit, and was
    // abandoned one minute after ten passes were proven).
    check_outcomes: HashMap<u64, bool>,
    // The knowledge half of the report (see `LongHorizonReport`).
    verified_outcomes: Vec<VerifiedOutcome>,
    acceptance_unknown: usize,
    // Checks whose MOST RECENT run timed out, by identity. A member's
    // next run gets the hang re-stake cap (see `run_with_timeout_cap`):
    // the full ceiling was already spent once without an answer, so
    // further runs may stake at most what this run has MEASURED real
    // work (its longest step) or a real answer (its longest decided
    // check) to cost — both measured terms, no constant, and the first
    // decided verdict removes the member and lifts the cap. This is what
    // keeps a hanging check from eating the run 600s at a time (the
    // qwen leg lost 120 of 240 minutes exactly that way).
    hung_checks: HashSet<u64>,
    longest_step: Duration,
    longest_decided_check: Duration,
    replans: usize,
    false_success_claims: usize,
    input_tokens: u64,
    output_tokens: u64,
    consecutive_errors: usize,
    last_runner_error: Option<String>,
    progress: HashMap<i64, ItemProgress>,
    // Which items have seen which acceptance-failure signature, run-wide.
    // One root cause billed as N item failures is the cascade this
    // detects (observed 2026-08-08: 12 of 17 abandonments carried one
    // byte-identical syntax error a single earlier write had introduced;
    // another run lost 3 features to one broken helper) — when a
    // signature recurs across items, each affected item's step context
    // names the cluster so the model treats it as ONE bug.
    failure_owners: HashMap<u64, std::collections::BTreeSet<i64>>,
    // A verdict is only as trustworthy as the files it reads. The guard
    // fingerprints each check's evidence before every run of it and
    // demotes a pass whose inputs moved (see [`EvidenceGuard`]).
    evidence_guard: EvidenceGuard,
}

impl<'a> HarnessRun<'a> {
    fn new(
        cfg: &'a LongHorizonConfig,
        goal: &'a str,
        source: &'a dyn TaskSource,
        runner: &'a dyn StepRunner,
        workdir: &'a Path,
        cancel: Option<CancelToken>,
        interjector: Option<&'a dyn Interjector>,
    ) -> Self {
        Self {
            cfg,
            goal,
            source,
            runner,
            workdir,
            cancel,
            interjector,
            interjected_items: 0,
            started: Instant::now(),
            steps_taken: 0,
            tool_calls_total: 0,
            side_effect_tool_calls: 0,
            items_completed: 0,
            items_completed_unverified: 0,
            items_already_satisfied: 0,
            items_abandoned: 0,
            items_revived: 0,
            items_regressed_reopened: 0,
            verified_this_run: Vec::new(),
            abandoned_this_run: Vec::new(),
            abandoned_unmet: Vec::new(),
            abandoned_unverifiable: Vec::new(),
            reopened_once: HashSet::new(),
            check_cost_paid: Duration::ZERO,
            check_runs: 0,
            sweep_cost_paid: Duration::ZERO,
            regression_notice: None,
            precheck_ids: cfg.precheck_acceptance_items.clone(),
            check_outcomes: HashMap::new(),
            verified_outcomes: Vec::new(),
            acceptance_unknown: 0,
            hung_checks: HashSet::new(),
            longest_step: Duration::ZERO,
            longest_decided_check: Duration::ZERO,
            replans: 0,
            false_success_claims: 0,
            input_tokens: 0,
            output_tokens: 0,
            consecutive_errors: 0,
            last_runner_error: None,
            progress: HashMap::new(),
            failure_owners: HashMap::new(),
            evidence_guard: EvidenceGuard::default(),
        }
    }

    /// One pass of the loop: pick the next item, run a step on it, judge the
    /// step, and sweep what earlier verdicts claimed.
    async fn iteration(&mut self) -> Phase {
        if self.cancel.as_ref().is_some_and(CancelToken::is_cancelled) {
            return stop(StopReason::Cancelled);
        }
        if self.started.elapsed() >= self.cfg.max_wall_clock {
            return stop(StopReason::WallClockExhausted);
        }
        let tokens_used = self.input_tokens + self.output_tokens;
        if self.cfg.max_total_tokens.is_some_and(|max| tokens_used >= max) {
            return stop(StopReason::TokenBudgetExhausted);
        }

        // The first available opportunity: admit anything the user sent
        // while the previous step ran, BEFORE choosing the next item, so
        // a fresh request is picked up by this very iteration rather
        // than after the rest of the plan drains.
        if let Some(interjector) = self.interjector {
            match interjector.interject().await {
                Ok(0) => {}
                Ok(admitted) => {
                    self.interjected_items += admitted;
                }
                // A failed admission must not kill work in flight; the
                // message stays pending and is retried next boundary.
                Err(message) => {
                    tracing::warn!(%message, "interjection failed — retrying next step");
                }
            }
        }

        // Re-anchor: the store, not the transcript, is the state.
        let step = match self.source.next().await {
            Ok(Some(step)) => step,
            Ok(None) => return self.drain_sweep().await,
            Err(message) => return stop(StopReason::SourceError { message }),
        };
        // Scoped so the pre-check below (which may `progress.remove`) is
        // not fighting a live borrow of the same map.
        let (is_replan, fruitless_steps, item_replans, precheck_due, hang_timeouts) =
            self.item_gate(&step);

        if is_replan && item_replans >= self.cfg.max_replans_per_item {
            return self
                .abandon_exhausted(&step, fruitless_steps, item_replans, hang_timeouts)
                .await;
        }

        let _ = self.source.start(step.id).await;

        // Baseline this item's evidence BEFORE its step can touch it: the
        // guard's whole value is that the fingerprint predates the work it
        // judges. Idempotent — a check already baselined keeps the
        // baseline the last verdict left, so re-selection never launders a
        // modification made since.
        if let Some(check) = &step.acceptance {
            self.evidence_guard.ensure_baseline(check, self.workdir);
        }

        self.precheck(&step, precheck_due).await?;

        let item = self.progress.entry(step.id).or_default();

        // ESCALATION, not a second identical ask. The decomposition rung
        // already came back empty on this item, and re-asking the same
        // question produced the same nothing: 84 instrumented firings,
        // zero subtasks, split exactly half at the first attempt and half
        // at the second. So the next rung changes the question — it puts
        // the item's own last failing result in front of the model (the
        // replan prompt is the only step prompt that never receives it)
        // and asks for the single next concrete action instead of a plan.
        //
        // It runs as an EXECUTE step deliberately. The replan branch
        // `continue`s ahead of every escalation the harness owns, so a
        // zero-tool replan never rode the narration ladder and never got
        // steered; and the abandonment gate fires one iteration later
        // regardless, so a rung that could not replenish would be
        // decorative. As an execute step it earns its budget back the
        // ordinary way — a novel successful tool call or a check that
        // flips resets the fruitless counter — while still spending one
        // replan allowance below, so abandonment converges on exactly
        // the schedule the decompose-only ladder had.
        let escalate = is_replan && item.dry_replans > 0;

        let (prompt, step_kind) = self.step_prompt(&step, is_replan, escalate, tokens_used);

        // Open-children count BEFORE a replan runs, so "did this replan
        // actually decompose anything?" is answerable afterwards. Only
        // taken for replan steps — an execute step has no such contract
        // (the escalated rung included), and this is a store round-trip.
        let subtasks_before = if is_replan && !escalate {
            self.source.open_subtasks(step.id).await.unwrap_or(None)
        } else {
            None
        };

        let outcome = self.run_step(&step, prompt, step_kind).await?;
        self.record_step(&step, &outcome, is_replan, escalate, subtasks_before)
            .await?;
        let repeated = self.file_step_findings(&step, &outcome).await?;
        self.judge_step(&step, &outcome, repeated).await?;
        self.mid_run_sweep(&step, &outcome).await;
        ControlFlow::Continue(())
    }

    /// The plan is empty: re-ask the environment about every item this run
    /// closed, then either go on (something reopened) or end the run.
    async fn drain_sweep(&mut self) -> Phase {
        // DRAIN SWEEP. The plan is empty, but every closed
        // verdict was per-moment: later steps may have un-done
        // verified work (observed 2026-08-08: a record run's
        // final artifact held 8 of 37 verified sections) or
        // fixed what an abandoned item was stuck on (the same
        // logs show abandoned features whose commands were
        // implemented afterwards, with no path back). Re-ask the
        // environment — one cheap check per closed item — and
        // reopen what it contradicts; reopening at most once per
        // item makes this a fixpoint, not a loop.
        self.abandoned_unmet.clear();
        let revived = self.drain_abandoned().await;
        let regressed = self.drain_verified().await;
        if revived || regressed {
            return next_iteration();
        }
        stop(StopReason::AllTasksDone)
    }

    /// The drain sweep over abandoned items: revive those whose check now
    /// passes, and record those whose check still fails (or says nothing) as
    /// unmet. Returns whether anything was revived.
    async fn drain_abandoned(&mut self) -> bool {
        let mut reopened_any = false;
        for (id, title, check) in self.abandoned_this_run.clone() {
            if self.reopened_once.contains(&id) {
                continue;
            }
            let run_started = Instant::now();
            let hang_cap = self.hung_checks
                .contains(&check_identity(&check))
                .then(|| self.longest_step.max(self.longest_decided_check));
            // Re-hash BEFORE the run: the fingerprint has to
            // describe the inputs this run is about to read.
            let drift = self.evidence_guard.observe(&check, self.workdir);
            let mut verdict = check.run_with_timeout_cap(self.workdir, hang_cap).await;
            if let Some(sentence) = &drift {
                verdict = verdict.with_evidence_drift(sentence);
            }
            if !verdict.timed_out {
                self.hung_checks.remove(&check_identity(&check));
                self.longest_decided_check =
                    self.longest_decided_check.max(run_started.elapsed());
            }
            if verdict.is_unknown() {
                // Unknown: not proof the goal is unmet, but it
                // still blocks "done" — carry the finding rather
                // than a false failure verdict.
                if verdict.timed_out {
                    self.hung_checks.insert(check_identity(&check));
                }
                self.acceptance_unknown += 1;
                let detail = format!(
                    "check produced no verdict — {}",
                    text_head(&verdict.detail, 200)
                );
                self.abandoned_unmet.push(AbandonedUnmet { id, title, detail });
                continue;
            }
            note_check_outcome(&mut self.check_outcomes, &check, verdict.passed);
            if !verdict.passed {
                // Evidence for the mission loop: this walked-away
                // item's done-condition is still false, so the
                // goal is provably unmet however dry the planner
                // sounds. Bounded detail: one line is identity.
                let detail = text_head(&verdict.detail, 240).to_string();
                self.abandoned_unmet.push(AbandonedUnmet { id, title, detail });
            }
            if verdict.passed
                && self.source
                    .reopen(id, "acceptance now passes — reviving abandoned item")
                    .await
                    .is_ok()
            {
                self.reopened_once.insert(id);
                // The check already passed: route it through the
                // precheck door so it completes verified without
                // spending a model step.
                self.precheck_ids.insert(id);
                self.progress.remove(&id);
                self.items_revived += 1;
                self.items_abandoned = self.items_abandoned.saturating_sub(1);
                reopened_any = true;
                let _ = self.source
                    .log(
                        id,
                        "revived",
                        serde_json::json!({ "detail": verdict.detail }),
                    )
                    .await;
            }
        }
        reopened_any
    }

    /// The drain sweep over verified completions: reopen those whose check
    /// now fails. Returns whether anything was reopened.
    async fn drain_verified(&mut self) -> bool {
        let mut reopened_any = false;
        for (id, _title, check) in self.verified_this_run.clone() {
            if self.reopened_once.contains(&id) {
                continue;
            }
            let run_started = Instant::now();
            let hang_cap = self.hung_checks
                .contains(&check_identity(&check))
                .then(|| self.longest_step.max(self.longest_decided_check));
            let drift = self.evidence_guard.observe(&check, self.workdir);
            let mut verdict = check.run_with_timeout_cap(self.workdir, hang_cap).await;
            if let Some(sentence) = &drift {
                verdict = verdict.with_evidence_drift(sentence);
            }
            if !verdict.timed_out {
                self.hung_checks.remove(&check_identity(&check));
                self.longest_decided_check =
                    self.longest_decided_check.max(run_started.elapsed());
            }
            if verdict.is_unknown() {
                // Unknown: a silent re-check is not evidence the
                // verified work regressed — reopening on it would
                // charge the model for a wedged command (or for a
                // test file someone edited).
                if verdict.timed_out {
                    self.hung_checks.insert(check_identity(&check));
                }
                self.acceptance_unknown += 1;
                continue;
            }
            note_check_outcome(&mut self.check_outcomes, &check, verdict.passed);
            if !verdict.passed
                && self.source
                    .reopen(
                        id,
                        &format!(
                            "verified completion regressed — the acceptance \
                             check fails again: {}",
                            verdict.detail
                        ),
                    )
                    .await
                    .is_ok()
            {
                self.reopened_once.insert(id);
                self.forget_verified(id);
                self.progress.remove(&id);
                self.items_completed = self.items_completed.saturating_sub(1);
                self.items_regressed_reopened += 1;
                reopened_any = true;
                let _ = self.source
                    .log(
                        id,
                        "regressed",
                        serde_json::json!({ "detail": verdict.detail }),
                    )
                    .await;
            }
        }
        reopened_any
    }

    /// The selected item's standing, marking its one pre-check as taken:
    /// `(is_replan, fruitless_steps, item_replans, precheck_due,
    /// hang_timeouts)`.
    fn item_gate(&mut self, step: &TaskStep) -> (bool, usize, usize, bool, usize) {
        let item = self.progress.entry(step.id).or_default();
        let precheck_due = !item.acceptance_prechecked;
        item.acceptance_prechecked = true;
        (
            item.steps_without_progress >= self.cfg.max_steps_per_item,
            item.steps_without_progress,
            item.replans,
            precheck_due,
            item.consecutive_check_timeouts,
        )
    }

    /// Close an item that has spent its fruitless steps and its replans.
    async fn abandon_exhausted(
        &mut self,
        step: &TaskStep,
        fruitless_steps: usize,
        item_replans: usize,
        hang_timeouts: usize,
    ) -> Phase {
        // Grinding AND replanning failed — close the item and move on.
        let (dry_replans, escalated_asks, last_result, steps_run) =
            self.progress.get(&step.id).map_or((0, 0, None, 0), |item| {
                (
                    item.dry_replans,
                    item.escalated_asks,
                    item.last_result.clone(),
                    item.steps_run,
                )
            });
        let mut reason = abandonment_headline(steps_run, fruitless_steps, item_replans);
        // Say what the replan rungs actually returned. The old
        // sentence described the outcome as though decomposition had
        // happened: 84 instrumented firings produced ZERO subtasks,
        // and every one of the 42 items abandoned behind them read
        // "…and 2 replans" as if two plans had been made.
        if dry_replans > 0 {
            let _ = write!(
                reason,
                "; the decomposition ask came back with no subtasks {dry_replans} \
                 time(s)"
            );
            if escalated_asks > 0 {
                // Provably true wherever this gate fires: an item
                // whose check ever passed would have completed and
                // left `progress`, so it cannot reach abandonment.
                reason.push_str(
                    ", and the escalated next-action ask that followed never got it \
                     past its done-condition — both attempts to unstick it returned \
                     nothing",
                );
            }
        }
        if hang_timeouts > 0 {
            let _ = write!(
                reason,
                "; its acceptance check hung {hang_timeouts} consecutive time(s) with \
                 no verdict — see the hang finding in its notes"
            );
        }
        // Loud on purpose: abandonment used to reach only the store's
        // activity log, and a run that silently gave up its one item
        // read as "converged" in the daemon log (2026-08-08 forensics
        // — recovering WHY took a database query; it should take a
        // grep).
        tracing::warn!(item = step.id, title = %step.title, %reason, "abandoning item");
        if let Err(message) = self.source.abandon(step.id, &reason).await {
            return stop(StopReason::SourceError { message });
        }
        self.items_abandoned += 1;
        record_abandonment(
            step,
            &reason,
            last_result,
            &mut self.abandoned_this_run,
            &mut self.abandoned_unverifiable,
        );
        self.progress.remove(&step.id);
        next_iteration()
    }

    /// Run the item's acceptance check before spending a step on it, and
    /// complete the item without one when the check already passes.
    async fn precheck(&mut self, step: &TaskStep, precheck_due: bool) -> Phase {
        // ACCEPTANCE PRE-CHECK. The environment is the judge of "done";
        // ask it before spending a step, not only after. When the
        // done-condition already passes with nothing run, the work is
        // provably already there and a step could only rediscover that.
        //
        // Opt-in (`precheck_acceptance_items`) because on a first plan a
        // pre-passing condition means the check is wrong, not the work
        // done — see the field's docs. Scoped to the exact ids the caller
        // named rather than to "every item this run selects": the
        // interjection poll above can add a USER'S message to the plan
        // mid-round, and skipping that on a pre-passing condition would
        // leave the user unanswered. Bounded to one execution per item per
        // run by `acceptance_prechecked`, and each execution reuses
        // `AcceptanceCheck::run`, so it inherits the same workdir
        // resolution and the same clamped timeout as every other
        // acceptance run — no second execution path to keep in sync.
        let precheck = if precheck_due && self.precheck_ids.contains(&step.id) {
            step.acceptance.as_ref()
        } else {
            None
        };
        if let Some(check) = precheck {
            let check_started = Instant::now();
            let hang_cap = self.hung_checks
                .contains(&check_identity(check))
                .then(|| self.longest_step.max(self.longest_decided_check));
            let drift = self.evidence_guard.observe(check, self.workdir);
            let mut verdict = check.run_with_timeout_cap(self.workdir, hang_cap).await;
            if let Some(sentence) = &drift {
                verdict = verdict.with_evidence_drift(sentence);
            }
            self.check_cost_paid += check_started.elapsed();
            self.check_runs += 1;
            self.note_precheck_verdict(step, check, &verdict, check_started)
                .await;
            if verdict.passed {
                // What the environment confirmed AT THIS INSTANT, not just
                // that it confirmed something (see `verdict_artifacts`).
                let artifacts = verdict_artifacts(check, self.workdir);
                let detail = serde_json::json!({
                    "verified": true,
                    "verdict": verdict.detail,
                    "output_head": verdict.output_head,
                    "artifacts": artifacts,
                    "already_satisfied": true,
                    "steps_run": 0,
                    "tokens_spent": 0,
                });
                let _ = self.source
                    .log(
                        step.id,
                        "acceptance_already_satisfied",
                        serde_json::json!({
                            "detail": verdict.detail,
                            "check": check.describe(),
                        }),
                    )
                    .await;
                match self.source.complete(step.id, detail).await {
                    Ok(()) => {
                        self.consecutive_errors = 0;
                        self.items_completed += 1;
                        self.items_already_satisfied += 1;
                        self.verified_this_run.push((step.id, step.title.clone(), check.clone()));
                        // Knowledge, not a dry round: the passing verdict
                        // rides the report so the continuation planner
                        // hears WHAT is established instead of re-seeding
                        // "assess starting state".
                        self.verified_outcomes.push(VerifiedOutcome {
                            id: step.id,
                            title: step.title.clone(),
                            detail: verified_detail(&verdict, &artifacts),
                            already_satisfied: true,
                        });
                        self.progress.remove(&step.id);
                        tracing::info!(
                            item = step.id,
                            title = %step.title,
                            verdict = %verdict.detail,
                            "acceptance already passed before any step — completing \
                             without running one"
                        );
                        return next_iteration();
                    }
                    Err(message) => {
                        // Same containment as the post-step path: a
                        // completion can legitimately fail (a concurrent
                        // decomposition opened a child). Fall through and
                        // run the step normally.
                        let _ = self.source
                            .log(
                                step.id,
                                "complete_failed",
                                serde_json::json!({ "error": message }),
                            )
                            .await;
                    }
                }
            }
        }
        ControlFlow::Continue(())
    }

    /// Book a pre-check verdict: carry a finding into the step when it said
    /// nothing, or record the decided outcome.
    async fn note_precheck_verdict(
        &mut self,
        step: &TaskStep,
        check: &AcceptanceCheck,
        verdict: &AcceptanceVerdict,
        check_started: Instant,
    ) {
        if verdict.is_unknown() {
            // Unknown, not failure: the item runs its step normally,
            // but it starts KNOWING what silenced the check — the
            // finding is the most useful thing the step could hear.
            let hung = verdict.timed_out;
            if hung {
                self.hung_checks.insert(check_identity(check));
            } else {
                self.hung_checks.remove(&check_identity(check));
                self.longest_decided_check =
                    self.longest_decided_check.max(check_started.elapsed());
            }
            self.acceptance_unknown += 1;
            let item = self.progress.entry(step.id).or_default();
            let finding = if hung {
                item.consecutive_check_timeouts += 1;
                hanging_check_finding(check, verdict, item.consecutive_check_timeouts)
            } else {
                evidence_changed_finding(check, verdict)
            };
            item.last_result = Some(finding.clone());
            tracing::warn!(
                item = step.id,
                title = %step.title,
                hung,
                detail = %text_head(&verdict.detail, 240),
                "acceptance pre-check produced no usable verdict; carrying the \
                 finding into the step"
            );
            let _ = self.source.add_note(step.id, &finding).await;
            let _ = self.source
                .log(
                    step.id,
                    if hung {
                        "acceptance_timeout"
                    } else {
                        "acceptance_evidence_changed"
                    },
                    serde_json::json!({
                        "detail": verdict.detail,
                        "check": check.describe(),
                        "precheck": true,
                    }),
                )
                .await;
        } else {
            self.hung_checks.remove(&check_identity(check));
            self.longest_decided_check = self.longest_decided_check.max(check_started.elapsed());
            if note_check_outcome(&mut self.check_outcomes, check, verdict.passed) {
                // Re-proposed work whose condition NOW passes after
                // failing earlier this run: the environment moved.
                // Every open item's fruitless budget replenishes below
                // (the completion path also runs — flip and completion
                // are the same event seen at two granularities).
                for open_item in self.progress.values_mut() {
                    open_item.steps_without_progress = 0;
                }
                tracing::info!(
                    item = step.id,
                    check = %check.describe(),
                    "check flipped fail→pass — verified environment change; \
                     replenishing every open item's fruitless budget"
                );
            }
        }
    }

    /// The step's prompt and kind: a replan ask, or an execute step (the
    /// escalated rung included).
    fn step_prompt(
        &mut self,
        step: &TaskStep,
        is_replan: bool,
        escalate: bool,
        tokens_used: u64,
    ) -> (String, StepKind) {
        let item = self.progress.entry(step.id).or_default();
        if is_replan && !escalate {
            // A replan reached with the check currently hanging must aim
            // the decomposition at the hang first — it is the standing
            // reason no verdict can arrive.
            let stall_summary = if item.consecutive_check_timeouts > 0 {
                format!(
                    "the done-condition check has HUNG {} consecutive time(s) (killed at \
                     its timeout, no verdict). The artifact blocks forever on something \
                     this check runs — decompose so fixing the hang comes first",
                    item.consecutive_check_timeouts
                )
            } else {
                format!(
                    "{} steps without the done-condition flipping",
                    item.steps_without_progress
                )
            };
            (
                build_replan_prompt(self.goal, step, &stall_summary),
                StepKind::Plan,
            )
        } else {
            let line = budget_line(
                self.steps_taken,
                self.items_completed,
                tokens_used,
                self.cfg.max_total_tokens,
                self.started.elapsed(),
                self.cfg.max_wall_clock,
            );
            // A pending regression notice rides in with the last result:
            // the model must hear "you un-did verified work" on its very
            // NEXT step, whatever item that step serves — waiting for the
            // regressed item itself to be selected would let it wander
            // further from the wreckage first. One-shot: taken here,
            // durably recorded on the reopened tasks either way.
            let last_result = match (item.last_result.as_deref(), self.regression_notice.take()) {
                (Some(prev), Some(notice)) => Some(format!("{prev}\n\n{notice}")),
                (None, Some(notice)) => Some(notice),
                (prev, None) => prev.map(str::to_string),
            };
            // The escalation rides in the same band as the regression
            // notice, and for the same reason: it is what the model must
            // hear LAST, right after the result it is being asked to act
            // on. Everything else about the prompt stays the ordinary
            // execute prompt — the done-condition, the self-check line,
            // the notes and the digest all still apply, because this rung
            // is asking for work, not for a plan.
            let last_result = if escalate {
                let escalation = stalled_escalation_text(
                    item.steps_without_progress,
                    item.consecutive_check_timeouts,
                );
                Some(match last_result {
                    Some(prev) => format!("{prev}\n\n{escalation}"),
                    None => escalation,
                })
            } else {
                last_result
            };
            // The do-not-regress digest: verified state must reach the
            // model that EDITS, not only the planner that plans.
            let digest = verified_digest(&self.verified_outcomes);
            (
                build_step_prompt(
                    self.goal,
                    step,
                    last_result.as_deref(),
                    digest.as_deref(),
                    &line,
                ),
                StepKind::Execute,
            )
        }
    }

    /// Run one step, containing runner errors to the item or the run.
    async fn run_step(
        &mut self,
        step: &TaskStep,
        prompt: String,
        step_kind: StepKind,
    ) -> Phase<StepOutcome> {
        let remaining_wall = self.cfg.max_wall_clock.saturating_sub(self.started.elapsed());
        let request = StepRequest {
            item_id: step.id,
            step_index: self.steps_taken,
            step_kind,
            item_title: step.title.clone(),
            prompt,
            tool_scope: step.tool_scope.clone(),
            token_budget: self.cfg.step_token_budget,
            // No fixed iteration cap: the runner ends the step on
            // progress exhaustion (see `step_token_budget`'s docs for the
            // P22 evidence against a hard 8) and the run's wall clock
            // rides in below as the real bound.
            max_iterations: None,
            max_wall_clock: Some(remaining_wall),
            cancel: self.cancel.clone(),
        };

        let step_span =
            crate::spans::harness_step_span(request.step_index, request.item_id, step_kind);
        let step_started = Instant::now();
        match tracing::Instrument::instrument(self.runner.run_step(request), step_span).await {
            Ok(outcome) => {
                self.consecutive_errors = 0;
                // One term of the hang re-stake cap: the largest cost
                // this run has measured for real work.
                self.longest_step = self.longest_step.max(step_started.elapsed());
                ControlFlow::Continue(outcome)
            }
            Err(message) => {
                self.consecutive_errors += 1;
                self.last_runner_error = Some(message.clone());
                let item = self.progress.entry(step.id).or_default();
                item.runner_errors += 1;
                // Poison containment: when the failure follows one item
                // (its prompt/notes deterministically break the model),
                // abandon that item and keep the run alive. Provider-wide
                // death still trips the run-level breaker below, whose
                // counter spans items.
                if item.runner_errors >= ITEM_RUNNER_ERRORS_MAX {
                    let last_result = item.last_result.clone();
                    let reason =
                        format!("abandoned after persistent runner errors: {message}");
                    if let Err(message) = self.source.abandon(step.id, &reason).await {
                        return stop(StopReason::SourceError { message });
                    }
                    self.items_abandoned += 1;
                    // This site used to record NOTHING — not even for a
                    // checked item, which then never reached either sweep
                    // and so could be neither revived nor named as unmet.
                    // A poisoned prompt says nothing about the world: if
                    // later work satisfies this item's condition the sweep
                    // must still be able to find it.
                    record_abandonment(
                        step,
                        &reason,
                        last_result,
                        &mut self.abandoned_this_run,
                        &mut self.abandoned_unverifiable,
                    );
                    self.progress.remove(&step.id);
                    return next_iteration();
                }
                if self.consecutive_errors >= self.cfg.max_consecutive_errors {
                    return stop(StopReason::RunnerErrors { message });
                }
                next_iteration()
            }
        }
    }

    /// Book a finished step's spend, and settle the replan rungs; a replan
    /// step ends the pass here.
    async fn record_step(
        &mut self,
        step: &TaskStep,
        outcome: &StepOutcome,
        is_replan: bool,
        escalate: bool,
        subtasks_before: Option<usize>,
    ) -> Phase {
        self.steps_taken += 1;
        // Counted where the run counts, so the item's number and the turn's
        // `N steps` line can never disagree.
        self.progress.entry(step.id).or_default().steps_run += 1;
        self.input_tokens += outcome.input_tokens;
        self.output_tokens += outcome.output_tokens;
        // Attribution material for the evidence guard: which paths this
        // run has written through a tool that RECORDS what it wrote. Only
        // ever used to decide whether "modified by this session" is a
        // claim the run can back up.
        self.evidence_guard.note_touched(&outcome.touched_paths);

        let item = self.progress.entry(step.id).or_default();
        item.tokens_spent += outcome.input_tokens + outcome.output_tokens;
        if !outcome.tool_calls.is_empty() {
            // The narration ladder counts CONSECUTIVE prose-only steps; a
            // step that acted re-arms all three rungs.
            item.narration_steps = 0;
        }

        if escalate {
            // The escalated rung spends one replan allowance even though
            // it ran as an execute step. Without that the item would
            // escalate forever: `is_replan` is derived from the fruitless
            // counter, which an unproductive escalated step does not
            // reset, and only `replans` moves the abandonment gate. With
            // it, the ladder is decompose → escalate → abandon on the
            // default `max_replans_per_item: 2` — the same length it had
            // when both rungs asked the same question.
            item.replans += 1;
            item.escalated_asks += 1;
            self.replans += 1;
            let _ = self.source
                .log(
                    step.id,
                    "replan_escalated",
                    serde_json::json!({
                        "replans": item.replans,
                        "dry_replans": item.dry_replans,
                        "asked": "next concrete action",
                    }),
                )
                .await;
            // Falls through: the verdict, the fruitless accounting and
            // the narration ladder below all apply, which is the whole
            // point of running this rung as a step.
        } else if is_replan {
            // The replan step adds subtasks through the store; the next
            // next() will surface them. A replan that actually decomposed
            // resets the grind counter so the new work gets a fresh
            // allowance — but a DRY replan must not buy one.
            //
            // Observed live 2026-08-02 (session 05775d1d): a replan
            // re-proposed the title the run had just abandoned, the store
            // refused to resurrect it, and the reset handed the item
            // another full allowance of fruitless steps — the item span
            // grew by `max_steps_per_item` per empty replan instead of
            // converging. With the reset withheld the item is still at
            // its grind threshold, so the next iteration replans again
            // and the existing `max_replans_per_item` rung reaches
            // abandonment directly. A source that cannot report its
            // children (`None`) is treated as productive, exactly as
            // before.
            let produced_work = match (subtasks_before, self.source.open_subtasks(step.id).await) {
                (Some(before), Ok(Some(after))) => after > before,
                _ => true,
            };
            item.replans += 1;
            if produced_work {
                item.steps_without_progress = 0;
                item.last_result = None;
                item.last_tool_calls.clear();
            } else {
                // Recorded, not just logged: the next stall rung on this
                // item reads it and changes the question rather than
                // asking this one again.
                item.dry_replans += 1;
                tracing::info!(
                    item = step.id,
                    replans = item.replans,
                    "replan added no subtasks — not resetting the grind counter; the \
                     next stall rung escalates instead of re-asking"
                );
            }
            self.replans += 1;
            let _ = self.source
                .log(
                    step.id,
                    "replanned",
                    serde_json::json!({
                        "replans": item.replans,
                        "produced_work": produced_work,
                    }),
                )
                .await;
            return next_iteration();
        }

        ControlFlow::Continue(())
    }

    /// Note the step's findings, stop if the run was cancelled, and count its
    /// tool calls. Returns whether the step repeated the item's last one.
    async fn file_step_findings(
        &mut self,
        step: &TaskStep,
        outcome: &StepOutcome,
    ) -> Phase<bool> {
        // Leave the step's findings in the store (the durable scratchpad),
        // not in any transcript.
        let tail = text_tail(&outcome.text, STEP_RESULT_TAIL_MAX_BYTES);
        if !tail.is_empty() {
            let _ = self.source.add_note(step.id, &tail).await;
        }

        // A cancelled step is a truncated step: its text is not a
        // verdictable claim, and an acceptance command run now is work
        // after the user said stop. Findings are already noted above.
        if self.cancel.as_ref().is_some_and(CancelToken::is_cancelled) {
            return stop(StopReason::Cancelled);
        }

        let item = self.progress.entry(step.id).or_default();
        // Cross-step repetition = a stall the in-run detector cannot see.
        let repeated = steps_repeat(&item.last_tool_calls, &outcome.tool_calls);
        item.last_tool_calls.clone_from(&outcome.tool_calls);
        self.tool_calls_total += outcome.tool_calls.len();
        // Counted by NAME, not by outcome: a [`StepToolCall`] is a digest
        // and carries no success flag. An attempted-but-failed `exec`
        // therefore reads as "the step reached for the world", which errs
        // toward letting a mission continue — the cheap mistake, since the
        // opposite one ends a live mission a round early.
        self.side_effect_tool_calls += outcome
            .tool_calls
            .iter()
            .filter(|call| crate::loop_runner::is_work_evidence_tool(&call.name))
            .count();
        ControlFlow::Continue(repeated)
    }

    /// Judge the step: by its acceptance check when it has one, by its own
    /// claim and evidence when it does not.
    async fn judge_step(
        &mut self,
        step: &TaskStep,
        outcome: &StepOutcome,
        repeated: bool,
    ) -> Phase {
        // Verdict time. With a check, the environment is the only judge.
        match &step.acceptance {
            Some(check) => self.judge_checked(step, check, outcome, repeated).await,
            None => self.judge_unchecked(step, outcome, repeated).await,
        }
    }

    /// Run the item's acceptance check after its step and act on the verdict.
    async fn judge_checked(
        &mut self,
        step: &TaskStep,
        check: &AcceptanceCheck,
        outcome: &StepOutcome,
        repeated: bool,
    ) -> Phase {
        let check_started = Instant::now();
        let hang_cap = self.hung_checks
            .contains(&check_identity(check))
            .then(|| self.longest_step.max(self.longest_decided_check));
        // The step just ran; re-hash its check's evidence before
        // the verdict, so a step that edited what the check reads
        // cannot close the item in the same breath.
        let drift = self.evidence_guard.observe(check, self.workdir);
        let mut verdict = check.run_with_timeout_cap(self.workdir, hang_cap).await;
        if let Some(sentence) = &drift {
            verdict = verdict.with_evidence_drift(sentence);
        }
        self.check_cost_paid += check_started.elapsed();
        self.check_runs += 1;
        if !verdict.timed_out {
            self.hung_checks.remove(&check_identity(check));
            self.longest_decided_check =
                self.longest_decided_check.max(check_started.elapsed());
        }
        let _ = self.source
            .log(
                step.id,
                "acceptance_checked",
                serde_json::json!({
                    "passed": verdict.passed,
                    "detail": verdict.detail,
                    "unknown": verdict.is_unknown(),
                    "evidence_changed": verdict.evidence_changed,
                }),
            )
            .await;
        if verdict.is_unknown() {
            self.judge_unknown(step, check, outcome, repeated, &verdict)
                .await;
            ControlFlow::Continue(())
        } else if verdict.passed {
            self.judge_passed(step, check, &verdict).await
        } else {
            self.judge_failed(step, check, outcome, repeated, &verdict)
                .await;
            ControlFlow::Continue(())
        }
    }

    /// The check said nothing: carry the finding and judge the step on its own
    /// evidence.
    async fn judge_unknown(
        &mut self,
        step: &TaskStep,
        check: &AcceptanceCheck,
        outcome: &StepOutcome,
        repeated: bool,
        verdict: &AcceptanceVerdict,
    ) {
        let item = self.progress.entry(step.id).or_default();
        // UNKNOWN, not failed. The check said nothing about
        // the work, so nothing about the TIMEOUT may read as
        // failure: no failure signature, no refuted claim,
        // and never a charge for the hang itself. What the
        // run DID learn is that something hangs — surfaced as
        // a first-class carried finding — and the next run of
        // this check is cost-capped (`hung_checks`).
        //
        // The STEP beside the silent check is still a step,
        // and it is judged exactly like a step with no check
        // at all (the `None` arm below is the precedent):
        // novel successful evidence replenishes, a degenerate
        // loop rides the steering ladder, and an empty-handed
        // step charges as an empty-handed step. That is what
        // keeps the item converging on the NORMAL ladder
        // without ever counting unknowns into a verdict —
        // counting them (an earlier draft routed N
        // consecutive timeouts to the replan rung) is just
        // fabricating a failure from things that said
        // nothing. (P22 evidence both ways: qwen proved
        // tests 01–20 passing inside the very step whose
        // check then hung — novel evidence, replenishes; and
        // it burned 120 of 240 minutes re-staking 600s on
        // the same silent check — the cap's job.)
        //
        // The evidence-drift demotion enters here for exactly
        // the same reason and with exactly the same
        // consequences (no budget charge, no completion, no
        // flip credit) — but it is NOT a hang: no hang finding
        // is minted and the re-stake cap stays disarmed, or a
        // model whose test file simply changed would be sent
        // hunting a non-existent infinite loop.
        let hung = verdict.timed_out;
        if hung {
            self.hung_checks.insert(check_identity(check));
        }
        self.acceptance_unknown += 1;
        let fresh_success = novel_success_evidence(item, &outcome.tool_calls);
        let degenerate =
            outcome.degenerate_loop && outcome.tool_calls.is_empty();
        let steered = degenerate
            && item.narration_steps < NARRATION_LADDER_STEPS;
        if fresh_success {
            item.steps_without_progress = 0;
        } else if steered {
            item.narration_steps += 1;
        } else {
            item.steps_without_progress += 1;
            if repeated {
                item.steps_without_progress += 1;
            }
        }
        let mut finding = if hung {
            item.consecutive_check_timeouts += 1;
            hanging_check_finding(
                check,
                verdict,
                item.consecutive_check_timeouts,
            )
        } else {
            item.consecutive_check_timeouts = 0;
            evidence_changed_finding(check, verdict)
        };
        if degenerate {
            finding.push_str("\n\n");
            finding
                .push_str(&narration_steering_text(item.narration_steps.max(1)));
        }
        tracing::warn!(
            item = step.id,
            title = %step.title,
            hung,
            consecutive = item.consecutive_check_timeouts,
            step_charged = !fresh_success && !steered,
            detail = %text_head(&verdict.detail, 240),
            "acceptance check produced no usable verdict — carrying the \
             finding forward and judging the step on its own evidence"
        );
        let _ = self.source.add_note(step.id, &finding).await;
        let _ = self.source
            .log(
                step.id,
                if hung {
                    "acceptance_timeout"
                } else {
                    "acceptance_evidence_changed"
                },
                serde_json::json!({
                    "detail": verdict.detail,
                    "consecutive": item.consecutive_check_timeouts,
                }),
            )
            .await;
        item.last_result = Some(finding);
    }

    /// The check passed: complete the item on the environment's verdict.
    async fn judge_passed(
        &mut self,
        step: &TaskStep,
        check: &AcceptanceCheck,
        verdict: &AcceptanceVerdict,
    ) -> Phase {
        let item = self.progress.entry(step.id).or_default();
        item.consecutive_check_timeouts = 0;
        if note_check_outcome(&mut self.check_outcomes, check, true) {
            for open_item in self.progress.values_mut() {
                open_item.steps_without_progress = 0;
            }
            tracing::info!(
                item = step.id,
                check = %check.describe(),
                "check flipped fail→pass — verified environment change; \
                 replenishing every open item's fruitless budget"
            );
        }
        let artifacts = verdict_artifacts(check, self.workdir);
        let item = self.progress.entry(step.id).or_default();
        let detail = serde_json::json!({
            "verified": true,
            "verdict": verdict.detail,
            "output_head": verdict.output_head,
            "artifacts": artifacts,
            "tokens_spent": item.tokens_spent,
        });
        match self.source.complete(step.id, detail).await {
            Ok(()) => {
                self.consecutive_errors = 0;
                self.items_completed += 1;
                self.verified_this_run.push((
                    step.id,
                    step.title.clone(),
                    check.clone(),
                ));
                self.verified_outcomes.push(VerifiedOutcome {
                    id: step.id,
                    title: step.title.clone(),
                    detail: verified_detail(verdict, &artifacts),
                    already_satisfied: false,
                });
                self.progress.remove(&step.id);
            }
            Err(message) => {
                // A completion can legitimately fail (e.g. a
                // concurrent decomposition opened a child).
                // Retry via next() instead of killing the run.
                let _ = self.source
                    .log(
                        step.id,
                        "complete_failed",
                        serde_json::json!({ "error": message }),
                    )
                    .await;
                self.consecutive_errors += 1;
                if self.consecutive_errors >= self.cfg.max_consecutive_errors {
                    return stop(StopReason::SourceError { message });
                }
            }
        }
        ControlFlow::Continue(())
    }

    /// The check failed: charge or replenish the item's budget and hand the
    /// next step the verdict.
    async fn judge_failed(
        &mut self,
        step: &TaskStep,
        check: &AcceptanceCheck,
        outcome: &StepOutcome,
        repeated: bool,
        verdict: &AcceptanceVerdict,
    ) {
        let item = self.progress.entry(step.id).or_default();
        item.consecutive_check_timeouts = 0;
        note_check_outcome(&mut self.check_outcomes, check, false);
        if step_claims_completion(&outcome.text) {
            // The model said done; the environment disagrees.
            self.false_success_claims += 1;
            let _ = self.source
                .log(
                    step.id,
                    "false_success_claim",
                    serde_json::json!({ "verdict": verdict.detail }),
                )
                .await;
        }
        let signature = failure_signature(&verdict.detail);
        let novel = item.seen_failure_signatures.insert(signature);
        let fresh_success = novel_success_evidence(item, &outcome.tool_calls);
        let degenerate =
            outcome.degenerate_loop && outcome.tool_calls.is_empty();
        let steered = degenerate
            && item.narration_steps < NARRATION_LADDER_STEPS;
        if novel && item.seen_failure_signatures.len() > 1 {
            // The check fails DIFFERENTLY than every earlier
            // attempt: the work moved the failure, which is
            // progress by the environment's own evidence.
            // The fruitless budget replenishes (same
            // principle as stream retries replenishing on
            // tool progress). A first-ever failure is the
            // baseline, not progress; a revisited signature
            // is oscillation and charges normally.
            item.steps_without_progress = 0;
        } else if fresh_success {
            // The success mirror of the rule above: the step
            // verifiably did NEW work on the world (a
            // successful side-effectful call this item has
            // never seen), so the environment's own evidence
            // says the item is moving even though its check
            // still fails the same way. Byte-identical
            // repeats never take this arm, so the rewrite
            // treadmill still converges on abandonment.
            item.steps_without_progress = 0;
        } else if steered {
            // A zero-tool degenerate loop is a steering
            // problem, not task evidence — route it to the
            // harness's escalation ladder (bounded by
            // NARRATION_LADDER_STEPS) and count it apart.
            item.narration_steps += 1;
        } else {
            item.steps_without_progress += 1;
            if repeated {
                item.steps_without_progress += 1;
            }
        }
        let mut step_result =
            failed_acceptance_result(check, verdict, repeated);
        if degenerate {
            step_result.push_str("\n\n");
            step_result
                .push_str(&narration_steering_text(item.narration_steps.max(1)));
            let _ = self.source
                .log(
                    step.id,
                    "narration_step",
                    serde_json::json!({
                        "narration_steps": item.narration_steps,
                        "charged": !steered,
                    }),
                )
                .await;
        }
        let cluster = self.failure_owners.entry(signature).or_default();
        cluster.insert(step.id);
        if cluster.len() > 1 {
            let others: Vec<String> = cluster
                .iter()
                .filter(|id| **id != step.id)
                .map(|id| format!("#{id}"))
                .collect();
            let _ = self.source
                .log(
                    step.id,
                    "shared_failure",
                    serde_json::json!({
                        "items": cluster.iter().copied().collect::<Vec<_>>(),
                    }),
                )
                .await;
            let _ = write!(
                step_result,
                "\nNOTE: {} of this plan {} failing with this EXACT same \
                 error. That is one underlying bug, not {} separate ones — \
                 find and fix the shared cause, then re-run the checks.",
                others.join(", "),
                if others.len() == 1 { "is also" } else { "are also" },
                cluster.len(),
            );
        }
        item.last_result = Some(step_result);
    }

    /// No check exists: complete on the step's claim, or judge its evidence.
    async fn judge_unchecked(
        &mut self,
        step: &TaskStep,
        outcome: &StepOutcome,
        repeated: bool,
    ) -> Phase {
        let item = self.progress.entry(step.id).or_default();
        let answer = tool_free_answer_fingerprint(outcome);
        let converged = answer_converged(item.last_answer_fingerprint, answer);
        item.last_answer_fingerprint = answer;
        if step_claims_completion(&outcome.text) {
            return self.complete_unchecked(step, "completed_unverified").await;
        }
        if converged {
            return self.complete_unchecked(step, "completed_converged").await;
        }
        let fresh_success = novel_success_evidence(item, &outcome.tool_calls);
        let degenerate = outcome.degenerate_loop && outcome.tool_calls.is_empty();
        let steered = degenerate && item.narration_steps < NARRATION_LADDER_STEPS;
        if fresh_success {
            // No check exists, but the environment's evidence
            // still counts: the step did NEW successful work,
            // so the item is moving even without a claim.
            item.steps_without_progress = 0;
        } else if steered {
            // Zero-tool degenerate loop: steer, count apart,
            // charge only past the ladder (same routing as
            // the checked arm above).
            item.narration_steps += 1;
        } else {
            item.steps_without_progress += 1;
            if repeated {
                item.steps_without_progress += 1;
            }
        }
        let mut step_result = text_tail(&outcome.text, STEP_RESULT_TAIL_MAX_BYTES);
        if degenerate {
            if !step_result.is_empty() {
                step_result.push_str("\n\n");
            }
            step_result.push_str(&narration_steering_text(item.narration_steps.max(1)));
            let _ = self
                .source
                .log(
                    step.id,
                    "narration_step",
                    serde_json::json!({
                        "narration_steps": item.narration_steps,
                        "charged": !steered,
                    }),
                )
                .await;
        }
        item.last_result = Some(step_result);
        ControlFlow::Continue(())
    }

    /// Close an unchecked item on the model's word — an explicit claim, or an
    /// answer that converged — logging `event` so the two stay distinguishable.
    async fn complete_unchecked(&mut self, step: &TaskStep, event: &'static str) -> Phase {
        debug_assert!(
            step.acceptance.is_none(),
            "a checked item is judged by its check"
        );
        debug_assert!(
            event == "completed_unverified" || event == "completed_converged",
            "an unchecked completion names how the model's word was given"
        );
        let tokens_spent = self
            .progress
            .get(&step.id)
            .map_or(0, |item| item.tokens_spent);
        let detail = serde_json::json!({
            "verified": false,
            "tokens_spent": tokens_spent,
        });
        match self.source.complete(step.id, detail).await {
            Ok(()) => {
                self.consecutive_errors = 0;
                self.items_completed += 1;
                self.items_completed_unverified += 1;
                let _ = self
                    .source
                    .log(step.id, event, serde_json::Value::Null)
                    .await;
                self.progress.remove(&step.id);
            }
            Err(message) => {
                let _ = self
                    .source
                    .log(
                        step.id,
                        "complete_failed",
                        serde_json::json!({ "error": message }),
                    )
                    .await;
                self.consecutive_errors += 1;
                if self.consecutive_errors >= self.cfg.max_consecutive_errors {
                    return stop(StopReason::SourceError { message });
                }
            }
        }
        ControlFlow::Continue(())
    }

    /// The mid-run sweep at a step boundary (see the comment inside).
    async fn mid_run_sweep(&mut self, step: &TaskStep, outcome: &StepOutcome) {
        // MID-RUN SWEEP — the drain sweep's live half. Waiting for
        // AllTasksDone to re-ask the environment was proven insufficient
        // on 2026-08-10: continuously-live missions (continuation keeps
        // the plan non-empty for hours) destroyed their own verified work
        // late in the run with nothing catching it — one leg held 22/42
        // verified for three hours, then full-file rewrites collapsed it
        // to 1/42 in the final hour. So re-ask at the step boundary too:
        // IMMEDIATELY for items whose acceptance references a path this
        // step just wrote, and periodically for everything else on a
        // cadence derived from measured check cost (`resweep_due` — no
        // fixed N). The current item is excluded: its own check just ran
        // as this boundary's verdict. Reopens share `reopened_once` with
        // the drain sweep, so each item is reopened at most once per run
        // however it is caught.
        let eligible: Vec<(i64, String, AcceptanceCheck)> = self.verified_this_run
            .iter()
            .filter(|(id, _, _)| *id != step.id && !self.reopened_once.contains(id))
            .cloned()
            .collect();
        // The drain sweep's OTHER half, also run live: an abandoned item
        // whose check now passes was fixed by later work, and waiting for
        // plan exhaustion to notice both wasted the fix and hid the
        // strongest replenishment signal the run has (a fail→pass flip).
        let eligible_abandoned: Vec<(i64, String, AcceptanceCheck)> = self.abandoned_this_run
            .iter()
            .filter(|(id, _, _)| *id != step.id && !self.reopened_once.contains(id))
            .cloned()
            .collect();
        if !eligible.is_empty() || !eligible_abandoned.is_empty() {
            let estimated = if self.check_runs > 0 {
                // Saturating where `as` wrapped: the two differ only
                // past u32::MAX re-sweep candidates, which no plan holds.
                (self.check_cost_paid / self.check_runs)
                    * u32::try_from(eligible.len() + eligible_abandoned.len())
                        .unwrap_or(u32::MAX)
            } else {
                Duration::ZERO
            };
            let full_due = resweep_due(self.sweep_cost_paid, self.check_cost_paid, estimated);
            let targets =
                select_resweep_targets(eligible, full_due, &outcome.touched_paths);
            let abandoned_targets = select_resweep_targets(
                eligible_abandoned,
                full_due,
                &outcome.touched_paths,
            );
            self.revive_abandoned_mid_run(abandoned_targets).await;
            let regressions = self.recheck_verified_mid_run(targets).await;
            if !regressions.is_empty() {
                let notice = regression_notice_text(&regressions);
                // Merge with a still-pending notice (possible when the
                // steps in between were replans, which do not consume
                // it) rather than dropping either; both halves are
                // screenful-bounded and the next execute step takes all.
                self.regression_notice = Some(match self.regression_notice.take() {
                    Some(prev) => format!("{prev}\n\n{notice}"),
                    None => notice,
                });
            }
        }
    }

    /// Mid-run: revive abandoned items whose check now passes.
    async fn revive_abandoned_mid_run(
        &mut self,
        abandoned_targets: Vec<(i64, String, AcceptanceCheck)>,
    ) {
        for (id, title, check) in abandoned_targets {
            if self.cancel.as_ref().is_some_and(CancelToken::is_cancelled) {
                break;
            }
            let sweep_started = Instant::now();
            let hang_cap = self.hung_checks
                .contains(&check_identity(&check))
                .then(|| self.longest_step.max(self.longest_decided_check));
            let drift = self.evidence_guard.observe(&check, self.workdir);
            let mut verdict = check.run_with_timeout_cap(self.workdir, hang_cap).await;
            if let Some(sentence) = &drift {
                verdict = verdict.with_evidence_drift(sentence);
            }
            self.sweep_cost_paid += sweep_started.elapsed();
            if !verdict.timed_out {
                self.hung_checks.remove(&check_identity(&check));
                self.longest_decided_check =
                    self.longest_decided_check.max(sweep_started.elapsed());
            }
            if verdict.is_unknown() {
                if verdict.timed_out {
                    self.hung_checks.insert(check_identity(&check));
                }
                self.acceptance_unknown += 1;
                continue;
            }
            let flipped = note_check_outcome(&mut self.check_outcomes, &check, verdict.passed);
            if flipped {
                // Verified environment change — replenish regardless
                // of whether the reopen below succeeds; the evidence
                // is the environment's, not the store's.
                for open_item in self.progress.values_mut() {
                    open_item.steps_without_progress = 0;
                }
                tracing::info!(
                    item = id,
                    check = %check.describe(),
                    "mid-run sweep: check flipped fail→pass — verified \
                     environment change; replenishing every open item's \
                     fruitless budget"
                );
            }
            if !verdict.passed {
                continue;
            }
            if self.source
                .reopen(id, "acceptance now passes — reviving abandoned item")
                .await
                .is_ok()
            {
                self.reopened_once.insert(id);
                // Route through the precheck door: it completes
                // verified on its next selection without a step.
                self.precheck_ids.insert(id);
                self.progress.remove(&id);
                self.items_revived += 1;
                self.items_abandoned = self.items_abandoned.saturating_sub(1);
                tracing::info!(
                    item = id,
                    title = %title,
                    "mid-run sweep: abandoned item's check now passes — reviving"
                );
                let _ = self.source
                    .log(
                        id,
                        "revived",
                        serde_json::json!({
                            "detail": verdict.detail,
                            "mid_run": true,
                        }),
                    )
                    .await;
            }
        }
    }

    /// Drop item `id` from what the run calls verified, because its check
    /// just failed again.
    ///
    /// `verified_outcomes` feeds the do-not-regress digest, which tells the
    /// model that EDITS "#id … VERIFIED WORKING right now" — so a regressed
    /// item left in it was asserted as working in the very prompt that sent the
    /// model back to fix it, and a re-earned verdict was listed twice in the
    /// report. `verified_this_run` feeds later sweeps; the item re-enters both
    /// when its check passes again.
    fn forget_verified(&mut self, id: i64) {
        self.verified_outcomes.retain(|outcome| outcome.id != id);
        self.verified_this_run
            .retain(|(verified, _, _)| *verified != id);
        debug_assert!(
            self.verified_outcomes.iter().all(|o| o.id != id),
            "a regressed item is no longer claimed verified"
        );
    }

    /// Mid-run: reopen verified items whose check fails again. Returns the
    /// regressions as `(id, title, detail)`.
    async fn recheck_verified_mid_run(
        &mut self,
        targets: Vec<(i64, String, AcceptanceCheck)>,
    ) -> Vec<(i64, String, String)> {
        let mut regressions: Vec<(i64, String, String)> = Vec::new();
        for (id, title, check) in targets {
            // A sweep is bookkeeping, not work — stop mid-sweep the
            // moment the user says stop; the loop head reports it.
            if self.cancel.as_ref().is_some_and(CancelToken::is_cancelled) {
                break;
            }
            let sweep_started = Instant::now();
            let hang_cap = self.hung_checks
                .contains(&check_identity(&check))
                .then(|| self.longest_step.max(self.longest_decided_check));
            let drift = self.evidence_guard.observe(&check, self.workdir);
            let mut verdict = check.run_with_timeout_cap(self.workdir, hang_cap).await;
            if let Some(sentence) = &drift {
                verdict = verdict.with_evidence_drift(sentence);
            }
            self.sweep_cost_paid += sweep_started.elapsed();
            if !verdict.timed_out {
                self.hung_checks.remove(&check_identity(&check));
                self.longest_decided_check =
                    self.longest_decided_check.max(sweep_started.elapsed());
            }
            if verdict.is_unknown() {
                // Unknown — a silent re-check must not reopen verified
                // work as "regressed".
                if verdict.timed_out {
                    self.hung_checks.insert(check_identity(&check));
                }
                self.acceptance_unknown += 1;
                continue;
            }
            note_check_outcome(&mut self.check_outcomes, &check, verdict.passed);
            if verdict.passed {
                continue;
            }
            if self.source
                .reopen(
                    id,
                    &format!(
                        "verified completion regressed mid-run — the acceptance \
                         check fails again: {}",
                        verdict.detail
                    ),
                )
                .await
                .is_ok()
            {
                self.reopened_once.insert(id);
                self.forget_verified(id);
                self.progress.remove(&id);
                self.items_completed = self.items_completed.saturating_sub(1);
                self.items_regressed_reopened += 1;
                let detail = text_head(&verdict.detail, 240).to_string();
                tracing::warn!(
                    item = id,
                    title = %title,
                    detail = %detail,
                    "mid-run sweep: verified item regressed — reopening"
                );
                let _ = self.source
                    .log(
                        id,
                        "regressed",
                        serde_json::json!({
                            "detail": verdict.detail,
                            "mid_run": true,
                        }),
                    )
                    .await;
                // Durable context for whenever the item is next
                // selected, independent of the one-shot notice below.
                let _ = self.source
                    .add_note(
                        id,
                        &format!(
                            "REOPENED mid-run: later work un-did this verified \
                             item — {detail}. Disk is truth; restore it."
                        ),
                    )
                    .await;
                regressions.push((id, title, detail));
            }
        }
        regressions
    }

    /// The report for a run that stopped with `stop`.
    fn into_report(self, stop: StopReason) -> LongHorizonReport {
        let total_tokens = self.input_tokens + self.output_tokens;
        LongHorizonReport {
            stop,
            steps_taken: self.steps_taken,
            tool_calls: self.tool_calls_total,
            side_effect_tool_calls: self.side_effect_tool_calls,
            items_completed: self.items_completed,
            items_completed_unverified: self.items_completed_unverified,
            items_revived: self.items_revived,
            items_regressed_reopened: self.items_regressed_reopened,
            abandoned_unmet: self.abandoned_unmet,
            abandoned_unverifiable: self.abandoned_unverifiable,
            verified_outcomes: self.verified_outcomes,
            acceptance_unknown: self.acceptance_unknown,
            items_already_satisfied: self.items_already_satisfied,
            items_abandoned: self.items_abandoned,
            last_runner_error: self.last_runner_error,
            replans: self.replans,
            false_success_claims: self.false_success_claims,
            input_tokens: self.input_tokens,
            output_tokens: self.output_tokens,
            wall_clock_secs: self.started.elapsed().as_secs(),
            tokens_per_completed_item: if self.items_completed > 0 {
                Some(total_tokens / self.items_completed as u64)
            } else {
                None
            },
            interjected_items: self.interjected_items,
        }
    }
}


/// Feedback handed to the next step after a failed acceptance check —
/// rendered verbatim into the `== LAST RESULT ==` block of the retry prompt.
///
/// Measured live (gemma4:12b, 2026-08-02 endurance evals): 320+ failed
/// checks and ~130 steps per feature, because after a failure the model saw
/// only "Done-condition NOT met: `FAIL(test_04)`: ..." — it was never shown
/// WHAT command judges the task, nor told to read the test it must satisfy,
/// so it retried by guessing. qwen3.5:9b infers the command from context;
/// gemma4:12b does not. Naming the command and directing a read-first turns
/// the guess loop into a feedback loop.
///
/// Bounds: `verdict.detail` embeds planner-authored strings (command,
/// pattern, path) that carry no upstream byte cap — only the command-output
/// tail inside it is bounded (400 chars) — so it is clamped here to
/// [`STEP_RESULT_TAIL_MAX_BYTES`], the same one-screenful bound step notes
/// use, with the cut announced (the full detail is already durably logged
/// via the `acceptance_checked` entry before this runs). The command itself
/// is included verbatim: the model must be able to run and read it exactly,
/// truncating it would destroy the one thing this message exists to hand
/// over, and the step prompt already embeds it verbatim ("RUN THAT CHECK
/// YOURSELF"). Everything else is a fixed-size frame, so the whole message
/// is O(one screenful).
///
/// The file the command runs is deliberately NOT inlined here: the
/// directive names it via the command, and the model must exercise
/// `read_file` itself — inlining would bloat every retry and the file can
/// be large.
/// Normalized identity of an acceptance failure: the first non-empty line of
/// the verdict detail, whitespace-collapsed, hashed.
///
/// The first line is where an acceptance command names WHAT failed (the
/// failing assertion, the shell error, the missing path); later lines carry
/// context that can drift between otherwise-identical failures. Whitespace
/// collapse tolerates re-wrapping without inventing identity — two failures
/// share a signature only when they say the same thing.
fn failure_signature(detail: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let first_line = detail
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or("");
    let normalized: String = first_line.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut hasher = std::hash::DefaultHasher::new();
    // Hashed as `str`, which is exactly how `String` hashes itself.
    normalized.as_str().hash(&mut hasher);
    hasher.finish()
}

/// Stable identity of a check across items and rounds: the canonical JSON of
/// the typed check. Two items proposing the same done-condition (a
/// re-proposal after abandonment is exactly this) share one identity, which
/// is what lets a later observation of the SAME question flip the ledger.
fn check_identity(check: &AcceptanceCheck) -> u64 {
    use std::hash::{Hash, Hasher};
    let canonical = serde_json::to_string(check).unwrap_or_default();
    let mut hasher = std::hash::DefaultHasher::new();
    // Hashed as `str`, which is exactly how `String` hashes itself.
    canonical.as_str().hash(&mut hasher);
    hasher.finish()
}

/// Record a DECIDED verdict in the run-wide check ledger. Returns true when
/// this exact check was last seen failing and now passes — a **verified
/// environment change**, the run-level mirror of the per-item novel-failure
/// rule: the world moved toward the goal by the environment's own evidence,
/// so every still-open item's fruitless budget replenishes. Timeouts never
/// reach here (unknown teaches the ledger nothing).
fn note_check_outcome(
    outcomes: &mut HashMap<u64, bool>,
    check: &AcceptanceCheck,
    passed: bool,
) -> bool {
    let previous = outcomes.insert(check_identity(check), passed);
    passed && previous == Some(false)
}

/// The SUCCESS mirror of the novel-failure signature rule: digests of this
/// step's successful side-effectful tool calls that this item has never seen
/// before. New evidence of real work replenishes the fruitless budget; a
/// byte-identical repeat (the rewrite treadmill) earns nothing, so oscillation
/// still converges on the abandonment rung.
fn novel_success_evidence(item: &mut ItemProgress, calls: &[StepToolCall]) -> bool {
    use std::hash::{Hash, Hasher};
    let mut novel = false;
    for call in calls {
        if !call.success || !crate::loop_runner::is_work_evidence_tool(&call.name) {
            continue;
        }
        let mut hasher = std::hash::DefaultHasher::new();
        call.name.hash(&mut hasher);
        call.input_digest.hash(&mut hasher);
        call.output_digest.hash(&mut hasher);
        novel |= item.seen_success_digests.insert(hasher.finish());
    }
    novel
}

/// The carried-forward hang finding: a timed-out acceptance run reframed as
/// what it actually is — a diagnosis about the artifact, not a transport
/// error and not a failed verdict. Rendered into `== LAST RESULT ==` AND
/// recorded as a durable note so it survives step boundaries, compression,
/// and even abandonment. (Observed 2026-08-10: 24 exec timeouts on one
/// non-terminating `mset` path and the model never registered "my
/// implementation hangs" as a fact — it kept re-running the command and the
/// hang then poisoned every acceptance check that touched it.)
fn hanging_check_finding(
    check: &AcceptanceCheck,
    verdict: &AcceptanceVerdict,
    consecutive: usize,
) -> String {
    let mut finding = format!(
        "ACCEPTANCE CHECK HUNG (no verdict — this does NOT mean the work failed): {}",
        verdict.detail
    );
    if consecutive > 1 {
        let _ = write!(
            finding,
            "\nThat is {consecutive} consecutive hangs of this same check."
        );
    }
    finding.push_str(
        "\nTreat the hang itself as a FINDING about the artifact: something this command \
         runs blocks forever and never exits. Do not just re-run it — reproduce the hang \
         with a short timeout, find the blocking code path, fix it, and only then re-run \
         the check.",
    );
    if let Some(command) = check.self_check_command() {
        let _ = write!(
            finding,
            "\nThe hanging command is: `{command}` (wrap it, e.g. `timeout 10 {command}`, \
             to see where it sticks)."
        );
    }
    finding
}

/// Escalating steer for a zero-tool-call degenerate-loop step — the
/// harness-level rung of the same gentle → firm → urgent ladder the in-step
/// wrap-up nudges use. Level is the 1-based count of such steps this item
/// has absorbed.
fn narration_steering_text(level: usize) -> String {
    match level {
        1 => "NOTE: your previous step produced only narration — it described work but \
              called no tool, so nothing happened. Start this step with a tool call."
            .to_string(),
        2 => "IMPORTANT: two steps in a row have produced prose and ZERO tool calls. \
              Text does not change the world. Do not describe what you would do — \
              CALL A TOOL as your very first action."
            .to_string(),
        _ => "STOP NARRATING. Every step that emits only text is discarded. Your first \
              output this step MUST be a tool call (read_file, exec, write_file — \
              whichever the task needs). If you cannot decide, run the task's check \
              command with exec and act on its output."
            .to_string(),
    }
}

/// The escalated stall rung, appended to the LAST RESULT block of an
/// ordinary execute prompt after a decomposition ask returned nothing.
///
/// The decomposition ask is a fair first question — it has decomposed
/// successfully on record, in about 2% of instrumented attempts — but asking
/// it twice measured 84 firings and zero subtasks. So the second rung asks a
/// different question, and the change is not just wording: this prompt shows
/// the model the failing result it is stuck on, which the replan prompt (a
/// stable prefix plus a one-line stall summary) never did.
///
/// Voiced as one concrete action rather than a plan because "smallest thing
/// that changes the result" is answerable from the verdict above it, while
/// "break this into 2-5 subtasks" asks the model to invent structure for
/// work it has just proven it cannot start.
fn stalled_escalation_text(fruitless_steps: usize, check_timeouts: usize) -> String {
    let mut text = format!(
        "STALLED — READ THIS BEFORE ACTING. {fruitless_steps} steps on this task have gone \
         by without its done-condition flipping, and the plan-it-into-smaller-pieces step \
         that followed produced no new tasks. You are not being asked to plan again.\n\
         Take ONE concrete action this step, chosen from the result above: the smallest \
         thing that would make that verdict come out differently. Name it in one sentence, \
         do it with a tool, then re-run the check."
    );
    if check_timeouts > 0 {
        let _ = write!(
            text,
            "\nThe check has also hung {check_timeouts} consecutive time(s) with no verdict, \
             so the one action to take is finding what it blocks on — reproduce it under a \
             short timeout first."
        );
    }
    text.push_str(
        "\nIf the task truly cannot be advanced here, say exactly what blocks it and what \
         you tried — that is a useful answer; another description of the plan is not.",
    );
    text
}

fn failed_acceptance_result(
    check: &AcceptanceCheck,
    verdict: &AcceptanceVerdict,
    repeated: bool,
) -> String {
    let detail = if verdict.detail.len() <= STEP_RESULT_TAIL_MAX_BYTES {
        verdict.detail.clone()
    } else {
        format!(
            "[showing last {STEP_RESULT_TAIL_MAX_BYTES} of {} bytes — full verdict is in the \
             acceptance_checked run log] {}",
            verdict.detail.len(),
            text_tail(&verdict.detail, STEP_RESULT_TAIL_MAX_BYTES)
        )
    };
    let mut result = format!("Done-condition NOT met: {detail}");
    if repeated {
        result.push_str(
            " (you repeated the exact same tool calls as last step — change approach)",
        );
    }
    if let Some(command) = check.self_check_command() {
        let _ = write!(
            result,
            "\nThat verdict came from the harness running: `{command}`\n\
             Before changing anything, read the file(s) this command runs (use read_file), \
             understand every assertion, then make them pass. Do not guess."
        );
    }
    result
}

/// Last `max_bytes` of `text`, on a char boundary, trimmed.
fn text_tail(text: &str, max_bytes: usize) -> String {
    let trimmed = text.trim();
    if trimmed.len() <= max_bytes {
        return trimmed.to_string();
    }
    let mut start = trimmed.len() - max_bytes;
    while !trimmed.is_char_boundary(start) {
        start += 1;
    }
    trimmed[start..].to_string()
}

/// First `max_bytes` of `text`, on a char boundary. The safe form of
/// `String::truncate` for verdict details, which can embed file content and
/// therefore multi-byte characters at any offset.
fn text_head(text: &str, max_bytes: usize) -> &str {
    if text.len() <= max_bytes {
        return text;
    }
    let mut cut = max_bytes;
    while cut > 0 && !text.is_char_boundary(cut) {
        cut -= 1;
    }
    &text[..cut]
}

/// The filesystem path a tool call targeted, when the tool is one that
/// writes a caller-named path (the write/edit family).
///
/// `None` for everything
/// else — including `exec`, whose side effects are opaque to the caller; the
/// periodic re-sweep is the backstop for those.
///
/// Accepts both parameter spellings (`file_path` is canonical since the
/// Claude Code alignment; `path` remains accepted) — the same tolerance the
/// tools themselves apply.
#[must_use]
pub fn touched_path_of(name: &str, input: &serde_json::Value) -> Option<String> {
    if !matches!(
        name,
        "write_file" | "write" | "Write" | "edit_file" | "edit" | "Edit" | "file_buffer"
    ) {
        return None;
    }
    input
        .get("file_path")
        .and_then(serde_json::Value::as_str)
        .or_else(|| input.get("path").and_then(serde_json::Value::as_str))
        .map(str::to_string)
}

/// Is a FULL periodic re-sweep affordable at this step boundary?
///
/// The cadence is derived, not configured: **re-verification may spend at
/// most what verification has spent.** The run already pays one acceptance
/// check per verdict as the price of knowing anything; the sweep re-asks old
/// verdicts, and re-asking may never out-spend asking. That caps sweep
/// overhead at a doubling of check time — and since a check must already be
/// cheaper than the step it verifies (see [`ACCEPTANCE_TIMEOUT_SECS_DEFAULT`]),
/// the sweep can never dominate the loop. For the sub-second shell checks
/// missions actually use this converges on sweeping every few steps; for a
/// plan whose checks are minute-long test suites it backs off in exact
/// proportion. No magic N.
///
/// `estimated_sweep_cost` is the measured average check duration times the
/// eligible-item count. An estimate only — the ACTUAL cost of each sweep is
/// then paid into `sweep_cost_paid`, so a low estimate self-corrects by
/// pushing the next sweep further out.
fn resweep_due(
    sweep_cost_paid: Duration,
    check_cost_paid: Duration,
    estimated_sweep_cost: Duration,
) -> bool {
    sweep_cost_paid + estimated_sweep_cost <= check_cost_paid
}

/// Which verified items does this boundary's sweep re-check?
///
/// When the periodic budget covers a full sweep, all of them. Otherwise only
/// the targeted set: items whose acceptance references a path the step just
/// wrote — the full-file-rewrite regression shape, caught the moment it
/// happens instead of a cadence later.
fn select_resweep_targets(
    eligible: Vec<(i64, String, AcceptanceCheck)>,
    full_sweep_due: bool,
    touched_paths: &[String],
) -> Vec<(i64, String, AcceptanceCheck)> {
    if full_sweep_due {
        return eligible;
    }
    if touched_paths.is_empty() {
        return Vec::new();
    }
    eligible
        .into_iter()
        .filter(|(_, _, check)| touched_paths.iter().any(|path| check.references_path(path)))
        .collect()
}

/// The prompt-facing notice after the mid-run sweep reopened regressed items
/// — rendered into the very next execute step's `== LAST RESULT ==` block so
/// the model hears about the damage BEFORE wandering further from it.
///
/// Bounded to one screenful ([`STEP_RESULT_TAIL_MAX_BYTES`], the established
/// prompt bound): the notice exists to redirect the next step, not to carry
/// the ledger — every regression's full verdict is already durably recorded
/// on the reopened task (reopen reason, `regressed` log entry, note).
fn regression_notice_text(regressions: &[(i64, String, String)]) -> String {
    let mut text = String::from(
        "== VERIFIED WORK UN-DONE ==\n\
         The harness re-ran done-conditions that had already PASSED. Your recent changes \
         broke them:\n",
    );
    let footer = "Disk is truth: those items are reopened and their checks FAIL right now. \
                  Restore them with the smallest edit that makes the checks pass again \
                  before doing anything else — do not rewrite whole files.";
    // Reserve room for the footer and a possible overflow line so the total
    // stays one screenful however many items regressed at once.
    let reserve = footer.len() + 80;
    let mut omitted = 0usize;
    for (id, title, detail) in regressions {
        let line = format!("- you un-did verified work #{id} ({title}): {detail}\n");
        if text.len() + line.len() + reserve > STEP_RESULT_TAIL_MAX_BYTES {
            omitted += 1;
            continue;
        }
        text.push_str(&line);
    }
    if omitted > 0 {
        let _ = writeln!(
            text,
            "- …and {omitted} more (each reopened task carries its failing verdict)"
        );
    }
    text.push_str(footer);
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::sync::Arc;
    use tokio::sync::Mutex;

    // -----------------------------------------------------------------
    // In-memory task source + scripted step runners
    // -----------------------------------------------------------------

    #[derive(Debug, Clone)]
    struct MemItem {
        step: TaskStep,
        done: bool,
        abandoned: bool,
    }

    #[derive(Default)]
    struct MemorySource {
        items: Mutex<Vec<MemItem>>,
        completions: Mutex<Vec<(i64, serde_json::Value)>>,
        notes: Mutex<Vec<(i64, String)>>,
        log_entries: Mutex<Vec<(i64, String)>>,
        abandon_reasons: Mutex<Vec<(i64, String)>>,
        fail_next: Mutex<bool>,
        /// Open-children ledger. `None` (the default) means "this source
        /// cannot report", which is the pre-existing path every other test
        /// exercises: the runner then treats every replan as productive.
        subtasks: Option<Arc<Mutex<HashMap<i64, usize>>>>,
    }

    impl MemorySource {
        async fn push(&self, step: TaskStep) {
            self.items.lock().await.push(MemItem {
                step,
                done: false,
                abandoned: false,
            });
        }
    }

    #[async_trait::async_trait]
    impl TaskSource for MemorySource {
        async fn next(&self) -> Result<Option<TaskStep>, String> {
            if *self.fail_next.lock().await {
                return Err("storage exploded".to_string());
            }
            Ok(self
                .items
                .lock()
                .await
                .iter()
                .find(|i| !i.done && !i.abandoned)
                .map(|i| i.step.clone()))
        }
        async fn start(&self, _id: i64) -> Result<(), String> {
            Ok(())
        }
        async fn complete(&self, id: i64, detail: serde_json::Value) -> Result<(), String> {
            let mut items = self.items.lock().await;
            if let Some(item) = items.iter_mut().find(|i| i.step.id == id) {
                item.done = true;
            }
            drop(items);
            self.completions.lock().await.push((id, detail));
            Ok(())
        }
        async fn add_note(&self, id: i64, content: &str) -> Result<(), String> {
            self.notes.lock().await.push((id, content.to_string()));
            Ok(())
        }
        async fn log(
            &self,
            id: i64,
            action: &str,
            _detail: serde_json::Value,
        ) -> Result<(), String> {
            self.log_entries.lock().await.push((id, action.to_string()));
            Ok(())
        }
        async fn abandon(&self, id: i64, reason: &str) -> Result<(), String> {
            let mut items = self.items.lock().await;
            if let Some(item) = items.iter_mut().find(|i| i.step.id == id) {
                item.abandoned = true;
            }
            drop(items);
            self.abandon_reasons
                .lock()
                .await
                .push((id, reason.to_string()));
            Ok(())
        }
        async fn reopen(&self, id: i64, _reason: &str) -> Result<(), String> {
            let mut items = self.items.lock().await;
            if let Some(item) = items.iter_mut().find(|i| i.step.id == id) {
                item.done = false;
                item.abandoned = false;
            }
            drop(items);
            self.log_entries.lock().await.push((id, "reopened".to_string()));
            Ok(())
        }
        async fn open_subtasks(&self, id: i64) -> Result<Option<usize>, String> {
            match &self.subtasks {
                Some(ledger) => Ok(Some(ledger.lock().await.get(&id).copied().unwrap_or(0))),
                None => Ok(None),
            }
        }
    }

    /// Replays a script and, on every Plan step, adds `per_replan` open
    /// children to the shared ledger — what a replan step does when it
    /// actually decomposes something. At 0 it is the DRY replan: the step
    /// ran, and the store holds nothing new (every title it proposed was
    /// already closed this turn, so `tasks.add` refused to resurrect it).
    struct ReplanningRunner {
        inner: ScriptedRunner,
        subtasks: Arc<Mutex<HashMap<i64, usize>>>,
        per_replan: usize,
    }

    #[async_trait::async_trait]
    impl StepRunner for ReplanningRunner {
        async fn run_step(&self, request: StepRequest) -> Result<StepOutcome, String> {
            if request.step_kind == StepKind::Plan && self.per_replan > 0 {
                *self
                    .subtasks
                    .lock()
                    .await
                    .entry(request.item_id)
                    .or_insert(0) += self.per_replan;
            }
            self.inner.run_step(request).await
        }
    }

    /// Creates a file on every step — the environment changing UNDER the
    /// harness, which is the only thing an acceptance check exists to notice.
    struct FileWritingRunner {
        path: PathBuf,
        steps: Mutex<usize>,
    }

    impl FileWritingRunner {
        fn new(path: PathBuf) -> Self {
            Self {
                path,
                steps: Mutex::new(0),
            }
        }
    }

    #[async_trait::async_trait]
    impl StepRunner for FileWritingRunner {
        async fn run_step(&self, _request: StepRequest) -> Result<StepOutcome, String> {
            *self.steps.lock().await += 1;
            std::fs::write(&self.path, b"done").map_err(|e| e.to_string())?;
            Ok(outcome("wrote the artifact"))
        }
    }

    /// Replays a fixed script of outcomes; captures every request.
    struct ScriptedRunner {
        script: Mutex<VecDeque<Result<StepOutcome, String>>>,
        requests: Mutex<Vec<StepRequest>>,
    }

    impl ScriptedRunner {
        fn new(script: Vec<Result<StepOutcome, String>>) -> Self {
            Self {
                script: Mutex::new(script.into()),
                requests: Mutex::new(Vec::new()),
            }
        }
    }

    #[async_trait::async_trait]
    impl StepRunner for ScriptedRunner {
        async fn run_step(&self, request: StepRequest) -> Result<StepOutcome, String> {
            self.requests.lock().await.push(request);
            self.script
                .lock()
                .await
                .pop_front()
                .unwrap_or_else(|| Err("script exhausted".to_string()))
        }
    }

    fn outcome(text: &str) -> StepOutcome {
        StepOutcome {
            text: text.to_string(),
            input_tokens: 1000,
            output_tokens: 200,
            tool_calls: vec![],
            touched_paths: vec![],
            degenerate_loop: false,
        }
    }

    fn step(id: i64, title: &str, acceptance: Option<AcceptanceCheck>) -> TaskStep {
        TaskStep {
            id,
            title: title.to_string(),
            description: None,
            acceptance,
            tool_scope: vec!["exec".to_string()],
            notes_tail: vec![],
        }
    }

    fn fast_config() -> LongHorizonConfig {
        LongHorizonConfig {
            max_steps_per_item: 2,
            max_replans_per_item: 1,
            ..LongHorizonConfig::default()
        }
    }

    // -----------------------------------------------------------------
    // Acceptance checks
    // -----------------------------------------------------------------

    #[test]
    fn acceptance_parses_all_kinds_and_rejects_unknown() {
        let cmd = AcceptanceCheck::from_json(
            &serde_json::json!({"kind": "command", "command": "cargo test"}),
        )
        .unwrap();
        assert!(matches!(cmd, AcceptanceCheck::Command { .. }));
        let file = AcceptanceCheck::from_json(
            &serde_json::json!({"kind": "file_exists", "path": "out.txt"}),
        )
        .unwrap();
        assert!(matches!(file, AcceptanceCheck::FileExists { .. }));
        let re = AcceptanceCheck::from_json(
            &serde_json::json!({"kind": "regex", "pattern": "ok", "path": "log.txt"}),
        )
        .unwrap();
        assert!(matches!(re, AcceptanceCheck::Regex { .. }));
        assert!(AcceptanceCheck::from_json(&serde_json::json!({"kind": "vibes"})).is_err());
    }

    #[test]
    fn acceptance_accepts_the_object_handed_over_as_a_string() {
        // The dominant live failure: the model serializes the object itself.
        let cmd = AcceptanceCheck::from_json(&serde_json::json!(
            "{\"kind\": \"command\", \"command\": \"cargo check --all\"}"
        ))
        .expect("stringified command check");
        assert_eq!(
            cmd,
            AcceptanceCheck::Command {
                command: "cargo check --all".to_string(),
                timeout_secs: None,
            }
        );

        let file =
            AcceptanceCheck::from_json(&serde_json::json!("{\"kind\":\"file_exists\",\"path\":\"out.txt\"}"))
                .expect("stringified file_exists check");
        assert_eq!(
            file,
            AcceptanceCheck::FileExists {
                path: "out.txt".to_string(),
            }
        );

        let re = AcceptanceCheck::from_json(&serde_json::json!(
            "{\"kind\":\"regex\",\"pattern\":\"0 failed\",\"command\":\"cargo test\",\"timeout_secs\":30}"
        ))
        .expect("stringified regex check");
        assert_eq!(
            re,
            AcceptanceCheck::Regex {
                pattern: "0 failed".to_string(),
                path: None,
                command: Some("cargo test".to_string()),
                timeout_secs: Some(30),
            }
        );
    }

    #[test]
    fn acceptance_repairs_the_js_object_literal_flavor() {
        let check = AcceptanceCheck::from_json(&serde_json::json!(
            "{kind: 'command', command: 'cargo test -p nanna-agent', timeout_secs: 45}"
        ))
        .expect("unquoted keys + single-quoted values repair");
        assert_eq!(
            check,
            AcceptanceCheck::Command {
                command: "cargo test -p nanna-agent".to_string(),
                timeout_secs: Some(45),
            }
        );

        // Where a bare VALUE ends is a guess, so repair refuses and teaches
        // instead of inventing a check the model never wrote.
        let err = AcceptanceCheck::from_json(&serde_json::json!(
            "{kind: command, command: cargo check --all}"
        ))
        .expect_err("bare values are not repairable");
        assert!(err.contains(r#"{"kind":"command","command":"cargo test"}"#), "{err}");
    }

    #[test]
    fn acceptance_errors_show_the_expected_shapes() {
        // Double-encoded: one unwrap leaves a string, which is still not a check.
        let err = AcceptanceCheck::from_json(&serde_json::json!(
            "\"{\\\"kind\\\":\\\"command\\\",\\\"command\\\":\\\"cargo test\\\"}\""
        ))
        .expect_err("double-encoded acceptance must be rejected");
        for shape in [
            r#"{"kind":"command","command":"cargo test"}"#,
            r#"{"kind":"file_exists","path":"docs/plan.md"}"#,
            r#"{"kind":"regex","pattern":"0 failed","path":"build.log"}"#,
            r#"{"kind":"regex","pattern":"0 failed","command":"cargo test"}"#,
        ] {
            assert!(err.contains(shape), "error must show {shape}: {err}");
        }

        // A genuine object with the wrong kind teaches the same way.
        let err = AcceptanceCheck::from_json(&serde_json::json!({"kind": "vibes"}))
            .expect_err("unknown kind must be rejected");
        assert!(err.contains(r#"{"kind":"file_exists","path":"docs/plan.md"}"#), "{err}");
    }

    #[tokio::test]
    async fn file_exists_check_reflects_real_filesystem_state() {
        let dir = tempfile::tempdir().unwrap();
        let check = AcceptanceCheck::FileExists {
            path: "artifact.txt".to_string(),
        };
        let verdict = check.run(dir.path()).await;
        assert!(
            !verdict.passed,
            "missing file must fail: {}",
            verdict.detail
        );

        std::fs::write(dir.path().join("artifact.txt"), "x").unwrap();
        let verdict = check.run(dir.path()).await;
        assert!(
            verdict.passed,
            "existing file must pass: {}",
            verdict.detail
        );
    }

    #[tokio::test]
    async fn regex_check_matches_file_content() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("log.txt"), "42 tests passed, 0 failed").unwrap();
        let check = AcceptanceCheck::Regex {
            pattern: r"\d+ tests passed, 0 failed".to_string(),
            path: Some("log.txt".to_string()),
            command: None,
            timeout_secs: None,
        };
        assert!(check.run(dir.path()).await.passed);

        let no_match = AcceptanceCheck::Regex {
            pattern: "impossible-marker".to_string(),
            path: Some("log.txt".to_string()),
            command: None,
            timeout_secs: None,
        };
        assert!(!no_match.run(dir.path()).await.passed);
    }

    #[tokio::test]
    async fn regex_check_fails_cleanly_on_invalid_pattern_and_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let bad = AcceptanceCheck::Regex {
            pattern: "([unclosed".to_string(),
            path: Some("log.txt".to_string()),
            command: None,
            timeout_secs: None,
        };
        let verdict = bad.run(dir.path()).await;
        assert!(!verdict.passed);
        assert!(
            verdict.detail.contains("invalid regex"),
            "{}",
            verdict.detail
        );

        let missing = AcceptanceCheck::Regex {
            pattern: "x".to_string(),
            path: Some("nope.txt".to_string()),
            command: None,
            timeout_secs: None,
        };
        assert!(!missing.run(dir.path()).await.passed);
    }

    #[tokio::test]
    async fn command_check_uses_exit_code() {
        let dir = tempfile::tempdir().unwrap();
        // `exit 0` / `exit 1` work under both sh and cmd.
        let pass = AcceptanceCheck::Command {
            command: "exit 0".to_string(),
            timeout_secs: None,
        };
        assert!(pass.run(dir.path()).await.passed);
        let fail = AcceptanceCheck::Command {
            command: "exit 1".to_string(),
            timeout_secs: None,
        };
        assert!(!fail.run(dir.path()).await.passed);
    }

    #[tokio::test]
    async fn command_check_runs_posix_syntax() {
        // Regression from the first live eval: `test`/`$(...)` checks were
        // silently unwinnable when the shell fell back to cmd.exe because a
        // bare `sh` was not on PATH. The runner must route through Git Bash.
        #[cfg(windows)]
        if git_bash_path().is_none() {
            eprintln!("skipping command_check_runs_posix_syntax: Git Bash not installed");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let pass = AcceptanceCheck::Command {
            command: "test \"$(echo 4)\" = \"4\"".to_string(),
            timeout_secs: None,
        };
        let verdict = pass.run(dir.path()).await;
        assert!(verdict.passed, "POSIX check must pass: {}", verdict.detail);
        let fail = AcceptanceCheck::Command {
            command: "test \"$(echo 5)\" = \"4\"".to_string(),
            timeout_secs: None,
        };
        assert!(!fail.run(dir.path()).await.passed);
    }

    #[tokio::test]
    async fn run_shell_timeout_errors_and_returns_promptly() {
        let dir = tempfile::tempdir().unwrap();
        // A sleeper that works under every shell run_shell can route to.
        #[cfg(windows)]
        let command = "ping -n 30 127.0.0.1";
        #[cfg(not(windows))]
        let command = "sleep 30";

        let started = std::time::Instant::now();
        let result = run_shell(command, dir.path(), Duration::from_secs(1)).await;
        let err = result.expect_err("a 30s sleeper must time out at 1s");
        assert!(err.to_string().contains("timed out"), "unexpected error: {err}");
        // Bound: 1s timeout + tree-kill (taskkill subprocess on Windows).
        // 10s is a generous ceiling for a loaded CI machine; the pre-fix
        // hang mode was the full 30s sleeper duration.
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "timeout path must not wait for the workload"
        );
    }

    /// REGRESSION (2026-08-08): `foo &` under an acceptance check — the
    /// shell exits immediately, the backgrounded grandchild inherits the
    /// pipes so the timeout fires with the shell already dead, and the
    /// `taskkill /T` parent-walk from that dead pid finds nothing. The
    /// per-check Job Object must reap the grandchild anyway so the held
    /// pipes EOF (full pipe-read teardown coverage lives in the exec-side
    /// twin, nanna-scripting bridge.rs).
    #[cfg(windows)]
    #[tokio::test]
    async fn run_shell_timeout_reaps_detached_grandchild() {
        if git_bash_path().is_none() {
            eprintln!("skipping run_shell_timeout_reaps_detached_grandchild: Git Bash not installed");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let pid_file = dir.path().join("grandchild.pid");
        let pid_path = pid_file.display().to_string().replace('\\', "/");
        // Git Bash `$!` is an MSYS pid; /proc/<msys-pid>/winpid maps it to
        // the real Windows pid so tasklist can probe it.
        let command =
            format!("ping -n 60 127.0.0.1 & cat /proc/$!/winpid > '{pid_path}'; exit 0");

        let result = run_shell(&command, dir.path(), Duration::from_secs(2)).await;
        let err = result.expect_err("held pipes must force the timeout");
        assert!(err.to_string().contains("timed out"), "unexpected error: {err}");

        let text = std::fs::read_to_string(&pid_file).expect("grandchild pid file");
        let grandchild_pid: u32 = text.trim().parse().expect("windows pid");

        // Reap bound: `TerminateJobObject` is near-instant; 5s is a generous
        // ceiling for a loaded CI machine, polled at 50ms.
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        loop {
            let alive = tokio::process::Command::new("tasklist")
                .args(["/FI", &format!("PID eq {grandchild_pid}"), "/NH"])
                .output()
                .await
                .is_ok_and(|out| {
                    String::from_utf8_lossy(&out.stdout).contains(&grandchild_pid.to_string())
                });
            if !alive {
                break;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "detached grandchild survived the acceptance timeout"
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    #[test]
    fn acceptance_timeout_is_clamped_to_ceiling_and_floor() {
        assert_eq!(
            AcceptanceCheck::effective_timeout(Some(9999)),
            Duration::from_secs(ACCEPTANCE_TIMEOUT_SECS_MAX)
        );
        assert_eq!(
            AcceptanceCheck::effective_timeout(Some(0)),
            Duration::from_secs(1)
        );
        assert_eq!(
            AcceptanceCheck::effective_timeout(None),
            Duration::from_secs(ACCEPTANCE_TIMEOUT_SECS_DEFAULT)
        );
    }

    // -----------------------------------------------------------------
    // Prompt building
    // -----------------------------------------------------------------

    #[test]
    fn step_prompt_prefix_is_byte_stable_across_steps() {
        let goal = "Refactor the parser crate without breaking the test suite";
        let a = build_step_prompt(goal, &step(1, "first", None), None, None, "== BUDGET == x");
        let b = build_step_prompt(
            goal,
            &step(2, "second", None),
            Some("previous result"),
            Some("- #1 first: `sh t.sh` exited 0"),
            "== BUDGET == y",
        );
        let marker = "== CURRENT TASK ==";
        let prefix_a = &a[..a.find(marker).unwrap()];
        let prefix_b = &b[..b.find(marker).unwrap()];
        assert_eq!(
            prefix_a, prefix_b,
            "the prompt prefix must never move — KV-prefix caching depends on it"
        );
    }

    #[test]
    fn step_prompt_pins_goal_verbatim_and_puts_dynamics_last() {
        let goal = "Ship the exact goal text, unsummarized & unmodified.";
        let prompt = build_step_prompt(
            goal,
            &step(7, "do the thing", None),
            Some("last output"),
            None,
            "== BUDGET == step 3",
        );
        assert!(prompt.contains(goal), "goal must appear verbatim");
        let goal_pos = prompt.find(goal).unwrap();
        let task_pos = prompt.find("Task #7").unwrap();
        let last_pos = prompt.find("last output").unwrap();
        let budget_pos = prompt.find("== BUDGET ==").unwrap();
        assert!(goal_pos < task_pos && task_pos < last_pos && last_pos < budget_pos);
    }

    /// gemma4:12b ran its acceptance command ZERO times across two hours of
    /// the 42-feature ladder, so it edited blind and re-submitted the same
    /// broken implementation. The prompt now hands it the command.
    #[test]
    fn step_prompt_tells_the_model_to_run_the_check_itself() {
        let check = AcceptanceCheck::from_json(&serde_json::json!({
            "kind": "command", "command": "sh tests/test_05.sh"
        }))
        .unwrap();
        let prompt = build_step_prompt("g", &step(1, "t", Some(check)), None, None, "b");
        assert!(prompt.contains("RUN THAT CHECK YOURSELF"));
        assert!(prompt.contains("exec `sh tests/test_05.sh`"));
    }

    /// A pure file/regex check has no command to run — inventing one would
    /// put words in the check's mouth, so the instruction is omitted.
    #[test]
    fn a_file_check_gets_no_self_check_instruction() {
        let check = AcceptanceCheck::from_json(&serde_json::json!({
            "kind": "file_exists", "path": "greeting.txt"
        }))
        .unwrap();
        assert!(check.self_check_command().is_none());
        let prompt = build_step_prompt("g", &step(1, "t", Some(check)), None, None, "b");
        assert!(!prompt.contains("RUN THAT CHECK YOURSELF"));
        assert!(prompt.contains("must exist"), "the condition is still stated");
    }

    #[test]
    fn step_prompt_describes_acceptance_when_present() {
        let check = AcceptanceCheck::Command {
            command: "cargo test".to_string(),
            timeout_secs: None,
        };
        let with = build_step_prompt("g", &step(1, "t", Some(check)), None, None, "b");
        assert!(with.contains("cargo test"));
        assert!(with.contains("checked by the harness"));
        let without = build_step_prompt("g", &step(1, "t", None), None, None, "b");
        assert!(without.contains("TASK COMPLETE"));
    }

    #[test]
    fn completion_claim_requires_marker_on_own_line() {
        assert!(step_claims_completion("did the work\nTASK COMPLETE\n"));
        assert!(step_claims_completion("  task complete  "));
        assert!(!step_claims_completion("the task completes eventually"));
        assert!(!step_claims_completion("almost TASK COMPLETE but inline"));
    }

    #[test]
    fn steps_repeat_requires_identical_nonempty_sequences() {
        let call = StepToolCall {
            name: "exec".to_string(),
            input_digest: "a".to_string(),
            output_digest: "b".to_string(),
            success: true,
        };
        assert!(steps_repeat(
            std::slice::from_ref(&call),
            std::slice::from_ref(&call)
        ));
        assert!(
            !steps_repeat(&[], &[]),
            "empty sequences are not a loop signal"
        );
        let other = StepToolCall {
            output_digest: "c".to_string(),
            ..call.clone()
        };
        assert!(!steps_repeat(&[call], &[other]));
    }

    #[test]
    fn text_tail_bounds_and_respects_char_boundaries() {
        assert_eq!(text_tail("short", 100), "short");
        let long = format!("{}end", "x".repeat(5000));
        let tail = text_tail(&long, 100);
        assert!(tail.len() <= 100);
        assert!(tail.ends_with("end"));
        // Multi-byte chars must not be split.
        let emoji = "🌀".repeat(100);
        let tail = text_tail(&emoji, 10);
        assert!(tail.chars().all(|c| c == '🌀'));
    }

    // -----------------------------------------------------------------
    // Failed-acceptance feedback
    // -----------------------------------------------------------------

    /// 2026-08-02 endurance evals (gemma4:12b): 320+ failed checks at ~130
    /// steps/feature because the retry feedback never named the judging
    /// command. The failure message must hand the model the command and
    /// direct it to read the test before editing.
    #[test]
    fn failed_verdict_feedback_names_command_and_directs_read_first() {
        let check = AcceptanceCheck::Command {
            command: "sh tests/test_04.sh".to_string(),
            timeout_secs: None,
        };
        let verdict = AcceptanceVerdict {
            passed: false,
            detail: "FAIL(test_04): exit code should be 1, got 0".to_string(),
            timed_out: false,
            evidence_changed: false,
            output_head: None,
        };
        let result = failed_acceptance_result(&check, &verdict, false);
        assert!(
            result.starts_with("Done-condition NOT met: FAIL(test_04)"),
            "{result}"
        );
        assert!(
            result.contains("running: `sh tests/test_04.sh`"),
            "the exact acceptance command must be named: {result}"
        );
        assert!(
            result.contains("read the file(s) this command runs (use read_file)"),
            "{result}"
        );
        assert!(result.contains("Do not guess"), "{result}");
        assert!(
            !result.contains("repeated the exact same tool calls"),
            "no repetition admonition without repetition: {result}"
        );
    }

    #[test]
    fn failed_verdict_feedback_keeps_the_repeated_steps_admonition() {
        let check = AcceptanceCheck::Command {
            command: "sh run_tests.sh".to_string(),
            timeout_secs: None,
        };
        let verdict = AcceptanceVerdict {
            passed: false,
            detail: "`sh run_tests.sh` exited 1 — 3 failures".to_string(),
            timed_out: false,
            evidence_changed: false,
            output_head: None,
        };
        let result = failed_acceptance_result(&check, &verdict, true);
        assert!(
            result.contains("repeated the exact same tool calls as last step — change approach"),
            "{result}"
        );
        // The admonition must not displace the read-first directive.
        assert!(result.contains("Do not guess"), "{result}");
    }

    /// A pure file check has no command to read — the directive would name
    /// nothing, so it is omitted (the condition is already stated in the
    /// step prompt's "Done when" line).
    #[test]
    fn failed_file_check_feedback_omits_the_read_directive() {
        let check = AcceptanceCheck::FileExists {
            path: "out.txt".to_string(),
        };
        let verdict = AcceptanceVerdict {
            passed: false,
            detail: "file does not exist: out.txt".to_string(),
            timed_out: false,
            evidence_changed: false,
            output_head: None,
        };
        let result = failed_acceptance_result(&check, &verdict, false);
        assert!(result.starts_with("Done-condition NOT met:"), "{result}");
        assert!(
            !result.contains("Before changing anything"),
            "no command means no read-first directive: {result}"
        );
    }

    /// The verdict detail embeds planner-authored strings with no upstream
    /// byte cap; the feedback clamps it to `STEP_RESULT_TAIL_MAX_BYTES` —
    /// the step-note bound — keeping the newest evidence and announcing the
    /// cut, without displacing the command or the directive.
    #[test]
    fn failed_verdict_feedback_bounds_a_runaway_detail() {
        let check = AcceptanceCheck::Command {
            command: "sh tests/test_04.sh".to_string(),
            timeout_secs: None,
        };
        let verdict = AcceptanceVerdict {
            passed: false,
            detail: format!("{}NEWEST-EVIDENCE", "x".repeat(STEP_RESULT_TAIL_MAX_BYTES * 10)),
            timed_out: false,
            evidence_changed: false,
            output_head: None,
        };
        let result = failed_acceptance_result(&check, &verdict, true);
        assert!(
            result.len() <= STEP_RESULT_TAIL_MAX_BYTES + 512,
            "detail clamped to the step-note bound plus a fixed frame: {} bytes",
            result.len()
        );
        assert!(
            result.contains("NEWEST-EVIDENCE"),
            "tail truncation keeps the newest evidence"
        );
        assert!(
            result.contains("[showing last"),
            "the cut must announce itself: {result}"
        );
        assert!(result.contains("`sh tests/test_04.sh`"), "{result}");
        assert!(result.contains("Do not guess"), "{result}");
    }

    /// A short detail passes through byte-identical — no truncation marker.
    #[test]
    fn failed_verdict_feedback_leaves_short_detail_untouched() {
        let check = AcceptanceCheck::FileExists {
            path: "a.txt".to_string(),
        };
        let verdict = AcceptanceVerdict {
            passed: false,
            detail: "file does not exist: a.txt".to_string(),
            timed_out: false,
            evidence_changed: false,
            output_head: None,
        };
        let result = failed_acceptance_result(&check, &verdict, false);
        assert_eq!(result, "Done-condition NOT met: file does not exist: a.txt");
    }

    /// End-to-end: after a failed check, the NEXT step's prompt carries the
    /// command and the read-first directive in its LAST RESULT block.
    #[tokio::test]
    async fn failed_acceptance_enriches_the_next_step_prompt() {
        let dir = tempfile::tempdir().unwrap();
        let source = MemorySource::default();
        source
            .push(step(
                1,
                "make the test pass",
                Some(AcceptanceCheck::Command {
                    // `exit 1` works under both sh and cmd.
                    command: "exit 1".to_string(),
                    timeout_secs: None,
                }),
            ))
            .await;
        let runner = ScriptedRunner::new(vec![Ok(outcome("try 1")), Ok(outcome("try 2"))]);
        let _report = LongHorizonRunner::new(fast_config())
            .run("goal", &source, &runner, dir.path(), None)
            .await;
        let requests = runner.requests.lock().await;
        assert!(requests.len() >= 2, "the item must get a retry step");
        let retry = &requests[1].prompt;
        assert!(retry.contains("== LAST RESULT =="), "{retry}");
        assert!(retry.contains("Done-condition NOT met"), "{retry}");
        assert!(
            retry.contains("That verdict came from the harness running: `exit 1`"),
            "{retry}"
        );
        assert!(
            retry.contains("read the file(s) this command runs (use read_file)"),
            "{retry}"
        );
        // First attempt has no verdict yet — the enrichment is failure-only.
        assert!(!requests[0].prompt.contains("Done-condition NOT met"));
        drop(requests);
    }

    // -----------------------------------------------------------------
    // P23 — the do-not-regress digest
    // -----------------------------------------------------------------

    fn verified(id: i64, title: &str, detail: &str) -> VerifiedOutcome {
        VerifiedOutcome {
            id,
            title: title.to_string(),
            detail: detail.to_string(),
            already_satisfied: false,
        }
    }

    /// The digest is the thing the EDITING model sees, so it must name the
    /// items, state the cost of losing them, and bound itself to the same
    /// screenful every other prompt block obeys — with the cut announced.
    #[test]
    fn the_digest_names_verified_work_and_bounds_itself() {
        assert!(verified_digest(&[]).is_none(), "nothing verified, nothing said");

        let one = verified_digest(&[verified(3, "mset", "`sh t3.sh` exited 0")])
            .expect("one verified item renders");
        assert!(one.contains("#3 mset"), "{one}");
        assert!(one.contains("`sh t3.sh` exited 0"), "{one}");
        assert!(
            one.contains("VERIFIED WORKING") || one.contains("VERIFIED"),
            "the block must say what these facts ARE: {one}"
        );
        assert!(
            one.contains("regression, not progress"),
            "the cost of losing one must be stated: {one}"
        );

        // Newest first — the freshest verdicts describe the artifact best.
        let two = verified_digest(&[
            verified(1, "older", "old detail"),
            verified(2, "newer", "new detail"),
        ])
        .unwrap();
        assert!(
            two.find("#2 newer").unwrap() < two.find("#1 older").unwrap(),
            "{two}"
        );

        // Bounded like every other prompt block, and the cut announces itself.
        let many: Vec<VerifiedOutcome> = (0..200)
            .map(|i| verified(i, &format!("item {i}"), &"d".repeat(200)))
            .collect();
        let big = verified_digest(&many).unwrap();
        assert!(
            big.len() <= STEP_RESULT_TAIL_MAX_BYTES + 256,
            "digest must stay one screenful: {} bytes",
            big.len()
        );
        assert!(big.contains("not shown here"), "the cut must announce itself");
    }

    /// End-to-end: once an item is verified, every LATER step's prompt carries
    /// it beside the last result. The failure this closes is a full-file
    /// rewrite that dropped features whose checks had passed minutes earlier,
    /// with nothing in the step's own context naming them.
    #[tokio::test]
    async fn a_verified_item_reaches_the_next_steps_prompt() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("first.txt"), b"here").unwrap();
        let source = MemorySource::default();
        source
            .push(step(
                1,
                "make first.txt",
                Some(AcceptanceCheck::FileExists {
                    path: "first.txt".to_string(),
                }),
            ))
            .await;
        source
            .push(step(
                2,
                "make second.txt",
                Some(AcceptanceCheck::FileExists {
                    path: "second.txt".to_string(),
                }),
            ))
            .await;
        let runner =
            ScriptedRunner::new(vec![Ok(outcome("did one")), Ok(outcome("working on two"))]);
        let _report = LongHorizonRunner::new(fast_config())
            .run("goal", &source, &runner, dir.path(), None)
            .await;
        let requests = runner.requests.lock().await;
        assert!(requests.len() >= 2, "both items must get a step");
        assert!(
            !requests[0].prompt.contains("VERIFIED WORKING"),
            "nothing is verified before the first verdict: {}",
            requests[0].prompt
        );
        let second = &requests[1].prompt;
        assert!(second.contains("== VERIFIED WORKING (do not regress) =="), "{second}");
        assert!(second.contains("#1 make first.txt"), "{second}");
        assert!(
            second.contains("first.txt (4 bytes"),
            "the verdict must carry artifact identity, not just 'a file exists': {second}"
        );
        drop(requests);
    }

    /// The stored completion record must say what the environment confirmed at
    /// that instant — the command's own output head and the subject file's
    /// size and mtime — so a later turn reading it back knows WHICH version was
    /// verified instead of merely that something passed.
    #[tokio::test]
    async fn a_completion_record_carries_the_artifact_it_verified() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("report.txt"), b"PASS: all good").unwrap();
        let source = MemorySource::default();
        source
            .push(step(
                1,
                "produce the report",
                Some(AcceptanceCheck::Regex {
                    pattern: "PASS".to_string(),
                    path: Some("report.txt".to_string()),
                    command: None,
                    timeout_secs: None,
                }),
            ))
            .await;
        let runner = ScriptedRunner::new(vec![Ok(outcome("wrote it"))]);
        let report = LongHorizonRunner::new(fast_config())
            .run("goal", &source, &runner, dir.path(), None)
            .await;
        assert_eq!(report.items_completed, 1, "{report:?}");
        let completions = source.completions.lock().await;
        let (_, detail) = completions.first().expect("a completion was recorded");
        let artifacts = detail
            .get("artifacts")
            .and_then(serde_json::Value::as_array)
            .expect("the completion names the artifacts it verified");
        assert_eq!(artifacts.len(), 1, "{detail}");
        assert_eq!(
            artifacts[0].get("path").and_then(serde_json::Value::as_str),
            Some("report.txt")
        );
        assert_eq!(
            artifacts[0].get("bytes").and_then(serde_json::Value::as_u64),
            Some(14)
        );
        assert!(
            artifacts[0]
                .get("modified")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|m| !m.is_empty()),
            "artifact identity needs the modification time: {detail}"
        );
        assert_eq!(
            detail.get("output_head").and_then(serde_json::Value::as_str),
            Some("PASS: all good"),
            "the head of what the check actually read: {detail}"
        );
        drop(completions);
    }

    // -----------------------------------------------------------------
    // P23 — verdicts notice when their own evidence moved
    // -----------------------------------------------------------------

    /// Writes fixed content to one path every step and REPORTS it as touched —
    /// the shape of a step that edits the very file its check reads.
    struct EvidenceRewritingRunner {
        path: PathBuf,
        content: String,
    }

    #[async_trait::async_trait]
    impl StepRunner for EvidenceRewritingRunner {
        async fn run_step(&self, _request: StepRequest) -> Result<StepOutcome, String> {
            std::fs::write(&self.path, self.content.as_bytes()).map_err(|e| e.to_string())?;
            let mut step_outcome = outcome("updated the report");
            step_outcome.touched_paths = vec![self.path.display().to_string()];
            Ok(step_outcome)
        }
    }

    /// A pass produced in the same breath as a rewrite of the SCRIPT the check
    /// runs is UNKNOWN, not a completion — and it costs exactly ONE named
    /// re-verification, because the changed content immediately becomes the
    /// baseline the next verdict is judged against.
    #[tokio::test]
    async fn a_pass_whose_evidence_the_step_rewrote_is_unknown_then_decided() {
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("check.sh");
        std::fs::write(&script, b"exit 1\n").unwrap();
        let source = MemorySource::default();
        source
            .push(step(
                1,
                "make the check pass",
                Some(AcceptanceCheck::Command {
                    command: "sh check.sh".to_string(),
                    timeout_secs: Some(30),
                }),
            ))
            .await;
        // The step "fixes" the test instead of the artifact — the exact shape
        // the guard exists to notice.
        let runner = EvidenceRewritingRunner {
            path: script.clone(),
            content: "exit 0\n".to_string(),
        };
        let outcome_report = LongHorizonRunner::new(fast_config())
            .run("goal", &source, &runner, dir.path(), None)
            .await;

        assert_eq!(
            outcome_report.acceptance_unknown, 1,
            "exactly one verdict is demoted — the one whose inputs moved: {outcome_report:?}"
        );
        assert_eq!(
            outcome_report.steps_taken, 2,
            "the demotion costs one re-verification step, not the item: {outcome_report:?}"
        );
        assert_eq!(
            outcome_report.items_completed, 1,
            "the SECOND verdict, judged against the new baseline, decides: \
             {outcome_report:?}"
        );

        let notes = source.notes.lock().await;
        let drift = notes
            .iter()
            .find(|(_, text)| text.contains("EVIDENCE CHANGED"))
            .map(|(_, text)| text.clone())
            .expect("the demotion must announce itself in the item's notes");
        assert!(drift.contains("check.sh"), "{drift}");
        assert!(
            drift.contains("modified by this session at"),
            "the step's own write ledger attributes it: {drift}"
        );
        assert!(
            !drift.contains("HUNG"),
            "an evidence change is not a hang — that would send the model hunting a \
             non-existent infinite loop: {drift}"
        );
        drop(notes);

        let log = source.log_entries.lock().await;
        assert!(
            log.iter().any(|(_, action)| action == "acceptance_evidence_changed"),
            "the demotion is greppable and distinct from a timeout: {log:?}"
        );
        drop(log);
    }

    /// The mirror property, and the reason the guard is safe to run on every
    /// verdict: a check whose instrument never moves is never demoted, and
    /// PRODUCING the deliverable a check observes is never mistaken for
    /// tampering with it (a `file_exists`/path-`regex` check has no instrument
    /// at all — the file it names is the work).
    #[tokio::test]
    async fn producing_the_artifact_is_never_read_as_tampering() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("check.sh"), b"exit 0\n").unwrap();
        let report_path = dir.path().join("report.txt");
        let source = MemorySource::default();
        source
            .push(step(
                1,
                "run the check",
                Some(AcceptanceCheck::Command {
                    command: "sh check.sh".to_string(),
                    timeout_secs: Some(30),
                }),
            ))
            .await;
        source
            .push(step(
                2,
                "write the report",
                Some(AcceptanceCheck::Regex {
                    pattern: "PASS".to_string(),
                    path: Some("report.txt".to_string()),
                    command: None,
                    timeout_secs: None,
                }),
            ))
            .await;
        // The second item's step writes the very file its check reads — the
        // work, not tampering.
        let runner = EvidenceRewritingRunner {
            path: report_path,
            content: "PASS".to_string(),
        };
        let report = LongHorizonRunner::new(fast_config())
            .run("goal", &source, &runner, dir.path(), None)
            .await;
        assert_eq!(report.items_completed, 2, "{report:?}");
        assert_eq!(
            report.acceptance_unknown, 0,
            "neither an untouched instrument nor a freshly-produced deliverable may \
             demote a verdict: {report:?}"
        );
        assert_eq!(report.steps_taken, 2, "one step per item: {report:?}");
    }

    // -----------------------------------------------------------------
    // P23 — user-declared file invariants (the writer half)
    // -----------------------------------------------------------------

    /// The registry the write tools read is materialized from the user's own
    /// words, in the shape the tool side parses, with the sentence quotable
    /// verbatim.
    #[tokio::test]
    async fn a_declared_prohibition_reaches_the_workspace_registry() {
        let dir = tempfile::tempdir().unwrap();
        materialize_declared_invariants(
            "Fix the failing behaviour in ./minidb. Never create, edit or delete anything \
             under tests/.",
            dir.path(),
        )
        .await;
        let raw = std::fs::read_to_string(
            dir.path().join(nanna_storage::DECLARED_INVARIANTS_FILE),
        )
        .expect("the registry is written where the write tools look for it");
        let doc: serde_json::Value = serde_json::from_str(&raw).expect("valid JSON");
        assert_eq!(doc.get("version").and_then(serde_json::Value::as_u64), Some(1));
        let list = doc
            .get("invariants")
            .and_then(serde_json::Value::as_array)
            .expect("invariants array");
        assert!(
            list.iter().any(|i| i.get("kind").and_then(serde_json::Value::as_str)
                == Some("read_only")),
            "{raw}"
        );
        assert!(
            list.iter().all(|i| i.get("glob").and_then(serde_json::Value::as_str)
                == Some("tests")),
            "{raw}"
        );
        assert!(
            list.iter().all(|i| i
                .get("source")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|s| s.contains("Never create, edit or delete"))),
            "the refusal has to be able to quote the user: {raw}"
        );
    }

    /// A goal that declares nothing must leave the workspace exactly as it
    /// was: the tool side fails open on a missing registry, so an ordinary
    /// chat turn writes no file and behaves exactly as before.
    #[tokio::test]
    async fn an_ordinary_goal_writes_no_registry() {
        let dir = tempfile::tempdir().unwrap();
        materialize_declared_invariants(
            "Add a --version flag to the CLI and update the README.",
            dir.path(),
        )
        .await;
        assert!(
            !dir.path()
                .join(nanna_storage::DECLARED_INVARIANTS_FILE)
                .exists(),
            "no declaration, no registry"
        );
    }

    // -----------------------------------------------------------------
    // Control loop
    // -----------------------------------------------------------------

    /// The report must separate "called tools" from "changed something": the
    /// chat harness's mission loop treats a round that only READ as making no
    /// progress, and reads are most of a round's tool calls.
    #[tokio::test]
    async fn the_report_counts_side_effects_apart_from_tool_calls() {
        let dir = tempfile::tempdir().unwrap();
        let source = MemorySource::default();
        source
            .push(step(
                1,
                "look around",
                Some(AcceptanceCheck::FileExists {
                    path: "never.txt".to_string(),
                }),
            ))
            .await;
        let called = |names: &[&str]| {
            Ok(StepOutcome {
                text: "had a look".to_string(),
                input_tokens: 1000,
                output_tokens: 200,
                tool_calls: names
                    .iter()
                    .map(|name| StepToolCall {
                        name: (*name).to_string(),
                        input_digest: (*name).to_string(),
                        output_digest: String::new(),
                        success: true,
                    })
                    .collect(),
                touched_paths: vec![],
                degenerate_loop: false,
            })
        };
        let runner = ScriptedRunner::new(vec![
            called(&["read_file", "code_search", "list_dir"]),
            called(&["read_file", "exec"]),
        ]);
        let config = LongHorizonConfig {
            max_steps_per_item: 2,
            max_replans_per_item: 0,
            ..fast_config()
        };
        let report = LongHorizonRunner::new(config)
            .run("goal", &source, &runner, dir.path(), None)
            .await;
        assert_eq!(report.tool_calls, 5, "every call counts here");
        assert_eq!(
            report.side_effect_tool_calls, 1,
            "only the shell call changed anything: {report:?}"
        );
    }

    // -----------------------------------------------------------------
    // Acceptance pre-check: ask the environment before spending a step
    // -----------------------------------------------------------------

    /// The decisive convergence signal. A seeded item whose done-condition
    /// ALREADY passes is finished work: complete it on that verdict, run no
    /// step, and record it apart so the caller can tell "already met" from
    /// "this run met it".
    ///
    /// The runner's script is EMPTY on purpose — any step at all fails the
    /// test loudly instead of quietly costing a model round.
    #[tokio::test]
    async fn an_already_passing_acceptance_completes_without_running_a_step() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("artifact.txt"), b"already here").unwrap();
        let source = MemorySource::default();
        source
            .push(step(
                1,
                "produce the artifact",
                Some(AcceptanceCheck::FileExists {
                    path: "artifact.txt".to_string(),
                }),
            ))
            .await;
        let runner = ScriptedRunner::new(vec![]);
        let config = LongHorizonConfig {
            precheck_acceptance_items: HashSet::from([1]),
            ..fast_config()
        };
        let report = LongHorizonRunner::new(config)
            .run("goal", &source, &runner, dir.path(), None)
            .await;

        assert_eq!(report.stop, StopReason::AllTasksDone);
        assert_eq!(report.steps_taken, 0, "no step: {report:?}");
        assert!(
            runner.requests.lock().await.is_empty(),
            "the runner was never asked to do anything"
        );
        assert_eq!(report.items_completed, 1, "the item IS closed");
        assert_eq!(
            report.items_already_satisfied, 1,
            "…and closed as already-satisfied, which is not progress"
        );
        assert_eq!(report.input_tokens + report.output_tokens, 0);

        // The verdict says WHY it closed — a completion with no step behind it
        // must announce itself, in the store the user can read.
        let completions = source.completions.lock().await;
        let (id, detail) = completions.first().expect("one completion");
        assert_eq!(*id, 1);
        assert_eq!(detail["already_satisfied"], serde_json::json!(true));
        assert_eq!(detail["verified"], serde_json::json!(true));
        assert_eq!(detail["steps_run"], serde_json::json!(0));
        drop(completions);
        assert!(
            source
                .log_entries
                .lock()
                .await
                .iter()
                .any(|(_, action)| action == "acceptance_already_satisfied"),
            "the activity log records the pre-check"
        );
    }

    /// The other direction, and the larger half of the contract: a check that
    /// does NOT already pass changes nothing about how the item runs. The step
    /// happens, the post-step verdict closes it, and the completion is
    /// ordinary — `items_already_satisfied` stays 0.
    ///
    /// This also pins the pre-check's bound: it runs at most ONCE per item per
    /// run. The environment flips mid-run (the step writes the file), and the
    /// item still closes through the post-step verdict rather than a second
    /// pre-check.
    #[tokio::test]
    async fn a_failing_pre_check_runs_the_step_exactly_as_before() {
        let dir = tempfile::tempdir().unwrap();
        let artifact = dir.path().join("artifact.txt");
        let source = MemorySource::default();
        source
            .push(step(
                1,
                "produce the artifact",
                Some(AcceptanceCheck::FileExists {
                    path: "artifact.txt".to_string(),
                }),
            ))
            .await;
        let runner = FileWritingRunner::new(artifact.clone());
        let config = LongHorizonConfig {
            precheck_acceptance_items: HashSet::from([1]),
            ..fast_config()
        };
        let report = LongHorizonRunner::new(config)
            .run("goal", &source, &runner, dir.path(), None)
            .await;

        assert_eq!(report.stop, StopReason::AllTasksDone);
        assert_eq!(report.steps_taken, 1, "the step ran: {report:?}");
        assert_eq!(*runner.steps.lock().await, 1);
        assert_eq!(report.items_completed, 1);
        assert_eq!(
            report.items_already_satisfied, 0,
            "the work was done BY this run, not found already done"
        );
        assert!(artifact.exists());
        let completions = source.completions.lock().await;
        let (_, detail) = completions.first().expect("one completion");
        assert!(
            detail.get("already_satisfied").is_none(),
            "an ordinary completion, unmarked: {detail}"
        );
        drop(completions);
    }

    /// The pre-check is OPT-IN, and the default is the bound: on a first plan
    /// nothing has run yet, so a condition that already passes means the
    /// planner wrote a weak check — not that the user's request is moot.
    /// Completing there would delete real work, so the default runs the step.
    #[tokio::test]
    async fn the_pre_check_is_off_unless_the_caller_asks_for_it() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("artifact.txt"), b"already here").unwrap();
        let source = MemorySource::default();
        source
            .push(step(
                1,
                "produce the artifact",
                Some(AcceptanceCheck::FileExists {
                    path: "artifact.txt".to_string(),
                }),
            ))
            .await;
        let runner = ScriptedRunner::new(vec![Ok(outcome("had a go"))]);
        assert!(
            LongHorizonConfig::default()
                .precheck_acceptance_items
                .is_empty(),
            "off by default"
        );
        let report = LongHorizonRunner::new(fast_config())
            .run("goal", &source, &runner, dir.path(), None)
            .await;
        assert_eq!(report.steps_taken, 1, "the step ran: {report:?}");
        assert_eq!(report.items_completed, 1);
        assert_eq!(report.items_already_satisfied, 0);
    }

    /// THE PRE-CHECK MUST NEVER SWALLOW A LIVE USER MESSAGE.
    ///
    /// The pre-check is scoped to the ids the continuation planner seeded, and
    /// this is the reason. The harness polls the interjector at the top of
    /// every iteration, BEFORE `next()`, so a message the user sends during a
    /// continuation round is planned into a fresh item and selected inside the
    /// very round the pre-check is enabled for. A round-wide flag would cover
    /// it, and an interjected ask whose planned acceptance happened to pass
    /// already ("artifact.txt exists" — trivially true) would be completed
    /// with ZERO steps run: the user asked a question and got silence.
    ///
    /// Driven through the interjector rather than by calling the pre-check
    /// directly, because the path IS the bug: nothing about the item itself
    /// says "a user just asked for this", only where it came from.
    ///
    /// Both halves of the contract in one run — the seeded item is skipped and
    /// counted `already_satisfied` (so the caller's round still reads dry),
    /// while the interjected item, whose acceptance is the same trivially-true
    /// condition, runs its step.
    #[tokio::test]
    async fn an_interjected_item_runs_its_step_even_when_acceptance_already_passes() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("artifact.txt"), b"already here").unwrap();
        let passing = || {
            Some(AcceptanceCheck::FileExists {
                path: "artifact.txt".to_string(),
            })
        };
        let source = Arc::new(MemorySource::default());
        source
            .push(step(1, "the continuation re-proposal", passing()))
            .await;
        // Nothing waiting when the seeded item is selected; the user's message
        // lands on the next boundary, mid-round.
        let interjector = ScriptedInterjector::new(
            source.clone(),
            vec![vec![], vec![step(2, "and what about the logs?", passing())]],
        );
        let runner = ScriptedRunner::new(vec![Ok(outcome("here are the logs\nTASK COMPLETE"))]);
        let config = LongHorizonConfig {
            // Only the continuation planner's seed is in scope.
            precheck_acceptance_items: HashSet::from([1]),
            ..fast_config()
        };

        let report = LongHorizonRunner::new(config)
            .run_with_interjector(
                "goal",
                source.as_ref(),
                &runner,
                dir.path(),
                None,
                Some(&interjector),
            )
            .await;

        assert_eq!(report.stop, StopReason::AllTasksDone);
        assert_eq!(report.interjected_items, 1);
        assert_eq!(report.items_completed, 2, "both closed: {report:?}");
        assert_eq!(
            report.items_already_satisfied, 1,
            "ONLY the seeded re-proposal was skipped for free: {report:?}"
        );
        assert_eq!(
            report.steps_taken, 1,
            "one step, and it belongs to the user's message: {report:?}"
        );
        let prompts: Vec<String> = runner
            .requests
            .lock()
            .await
            .iter()
            .map(|r| r.prompt.clone())
            .collect();
        assert_eq!(prompts.len(), 1);
        assert!(
            prompts[0].contains("and what about the logs?"),
            "the user's mid-round message must reach a step prompt: {prompts:?}"
        );
        let completions = source.completions.lock().await;
        let seeded = completions
            .iter()
            .find(|(id, _)| *id == 1)
            .expect("the seeded item closed");
        assert_eq!(seeded.1["already_satisfied"], serde_json::json!(true));
        let interjected = completions
            .iter()
            .find(|(id, _)| *id == 2)
            .expect("the interjected item closed");
        assert!(
            interjected.1.get("already_satisfied").is_none(),
            "the user's item closed by running, not by being skipped: {}",
            interjected.1
        );
        drop(completions);
    }

    #[tokio::test]
    async fn empty_plan_stops_immediately_with_all_done() {
        let source = MemorySource::default();
        let runner = ScriptedRunner::new(vec![]);
        let dir = tempfile::tempdir().unwrap();
        let report = LongHorizonRunner::new(fast_config())
            .run("goal", &source, &runner, dir.path(), None)
            .await;
        assert_eq!(report.stop, StopReason::AllTasksDone);
        assert_eq!(report.steps_taken, 0);
        assert_eq!(report.tokens_per_completed_item, None);
        assert_eq!(report.last_runner_error, None, "no error, nothing to carry");
    }

    #[tokio::test]
    async fn verified_completion_requires_the_acceptance_check_to_pass() {
        struct Producer {
            artifact: PathBuf,
            calls: Mutex<usize>,
        }
        #[async_trait::async_trait]
        impl StepRunner for Producer {
            async fn run_step(&self, _request: StepRequest) -> Result<StepOutcome, String> {
                let mut calls = self.calls.lock().await;
                *calls += 1;
                if *calls == 2 {
                    std::fs::write(&self.artifact, "done").unwrap();
                }
                drop(calls);
                Ok(StepOutcome {
                    text: "worked on it".to_string(),
                    input_tokens: 1000,
                    output_tokens: 200,
                    tool_calls: vec![],
                    touched_paths: vec![],
                    degenerate_loop: false,
                })
            }
        }
        let dir = tempfile::tempdir().unwrap();
        let source = MemorySource::default();
        source
            .push(step(
                1,
                "produce artifact",
                Some(AcceptanceCheck::FileExists {
                    path: "artifact.txt".to_string(),
                }),
            ))
            .await;
        // Step 1: model works but produces nothing. Step 2: file appears.
        let artifact = dir.path().join("artifact.txt");
        let runner = Producer {
            artifact,
            calls: Mutex::new(0),
        };
        let report = LongHorizonRunner::new(fast_config())
            .run("goal", &source, &runner, dir.path(), None)
            .await;
        assert_eq!(report.stop, StopReason::AllTasksDone);
        assert_eq!(report.items_completed, 1);
        assert_eq!(report.steps_taken, 2);
        assert_eq!(report.items_completed_unverified, 0);
        let completions = source.completions.lock().await;
        assert_eq!(completions.len(), 1);
        assert_eq!(completions[0].1["verified"], serde_json::json!(true));
    }

    #[tokio::test]
    async fn drain_sweep_revives_an_abandoned_item_whose_check_now_passes() {
        struct LateProducer {
            artifact: PathBuf,
            calls: Mutex<usize>,
        }
        #[async_trait::async_trait]
        impl StepRunner for LateProducer {
            async fn run_step(&self, _request: StepRequest) -> Result<StepOutcome, String> {
                let mut calls = self.calls.lock().await;
                *calls += 1;
                if *calls == 2 {
                    std::fs::write(&self.artifact, "made by item 2").unwrap();
                    return Ok(outcome("all wrapped up\nTASK COMPLETE"));
                }
                drop(calls);
                Ok(outcome("grinding"))
            }
        }
        let dir = tempfile::tempdir().unwrap();
        let source = MemorySource::default();
        // Item 1 is stuck on a file that does not exist yet; item 2's work
        // creates it as a side effect — the exact cascade shape observed
        // live: an abandoned feature whose blocker was implemented later,
        // with no path back to the verdict.
        source
            .push(step(
                1,
                "blocked feature",
                Some(AcceptanceCheck::FileExists {
                    path: "x.txt".to_string(),
                }),
            ))
            .await;
        source.push(step(2, "later work", None)).await;
        let runner = LateProducer {
            artifact: dir.path().join("x.txt"),
            calls: Mutex::new(0),
        };
        let config = LongHorizonConfig {
            max_steps_per_item: 1,
            max_replans_per_item: 0,
            ..LongHorizonConfig::default()
        };
        let report = LongHorizonRunner::new(config)
            .run("goal", &source, &runner, dir.path(), None)
            .await;
        assert_eq!(report.stop, StopReason::AllTasksDone);
        // Item 1 burned its budget and was abandoned, item 2 created the
        // file — the drain sweep must revive item 1 and complete it verified
        // WITHOUT spending another model step.
        assert_eq!(report.items_revived, 1, "{report:?}");
        assert_eq!(report.items_abandoned, 0, "revival corrects the count");
        assert_eq!(report.items_completed, 2);
        assert_eq!(report.steps_taken, 2, "the revival costs checks, not steps");
        let completions = source.completions.lock().await;
        let item1 = completions.iter().find(|(id, _)| *id == 1).unwrap();
        assert_eq!(item1.1["verified"], serde_json::json!(true));
        drop(completions);
        let log = source.log_entries.lock().await;
        assert!(log.iter().any(|(id, a)| *id == 1 && a == "reopened"));
        assert!(log.iter().any(|(id, a)| *id == 1 && a == "revived"));
        drop(log);
    }

    #[tokio::test]
    async fn drain_sweep_reopens_a_verified_item_that_later_work_un_did() {
        struct Underminer {
            artifact: PathBuf,
            calls: Mutex<usize>,
        }
        #[async_trait::async_trait]
        impl StepRunner for Underminer {
            async fn run_step(&self, _request: StepRequest) -> Result<StepOutcome, String> {
                let mut calls = self.calls.lock().await;
                *calls += 1;
                match *calls {
                    1 => {
                        std::fs::write(&self.artifact, "v1").unwrap();
                        Ok(outcome("built it"))
                    }
                    2 => {
                        std::fs::remove_file(&self.artifact).unwrap();
                        Ok(outcome("cleaned up\nTASK COMPLETE"))
                    }
                    _ => {
                        std::fs::write(&self.artifact, "v2").unwrap();
                        Ok(outcome("restored it"))
                    }
                }
            }
        }
        let dir = tempfile::tempdir().unwrap();
        let source = MemorySource::default();
        source
            .push(step(
                1,
                "build artifact",
                Some(AcceptanceCheck::FileExists {
                    path: "y.txt".to_string(),
                }),
            ))
            .await;
        source.push(step(2, "destructive later work", None)).await;
        // Call 1 creates the artifact (item 1 verifies), call 2 DELETES it
        // (item 2 "succeeds" while silently un-doing verified work — the
        // rewrite-erosion shape), call 3 restores it when item 1 comes back.
        let runner = Underminer {
            artifact: dir.path().join("y.txt"),
            calls: Mutex::new(0),
        };
        let report = LongHorizonRunner::new(fast_config())
            .run("goal", &source, &runner, dir.path(), None)
            .await;
        assert_eq!(report.stop, StopReason::AllTasksDone);
        // "Done" must not outlive its evidence: the sweep caught the
        // regression, reopened item 1, and the loop re-earned the verdict.
        assert_eq!(report.items_regressed_reopened, 1, "{report:?}");
        assert_eq!(report.items_completed, 2);
        assert_eq!(report.steps_taken, 3);
        let log = source.log_entries.lock().await;
        assert!(log.iter().any(|(id, a)| *id == 1 && a == "regressed"));
        drop(log);
    }

    // -----------------------------------------------------------------
    // Mid-run sweep: regressions caught at the step boundary, not at drain
    // -----------------------------------------------------------------

    /// The rewrite-erosion shape, mid-run: call 1 builds item 1's artifact
    /// (verified), call 2 REWRITES it wholesale while working item 2 — the
    /// live-mission failure the drain sweep can never catch in time (observed
    /// 2026-08-10: 22/42 verified held for three hours, then full-file
    /// rewrites collapsed it to 1/42 while the plan stayed non-empty). Call 3
    /// restores it when item 1 comes back, call 4 finishes item 2.
    struct MidRunUnderminer {
        artifact: PathBuf,
        /// Report `y.txt` in the destructive step's `touched_paths` (the
        /// write/edit shape). False = the write is invisible to the runner
        /// (the exec-side-effect shape) and only the periodic sweep can see
        /// the damage.
        report_touch: bool,
        /// Clobber the artifact AGAIN on the final call — probing the
        /// one-reopen-per-item-per-run bound.
        final_clobber: bool,
        requests: Mutex<Vec<StepRequest>>,
        calls: Mutex<usize>,
    }

    impl MidRunUnderminer {
        fn new(artifact: PathBuf, report_touch: bool, final_clobber: bool) -> Self {
            Self {
                artifact,
                report_touch,
                final_clobber,
                requests: Mutex::new(Vec::new()),
                calls: Mutex::new(0),
            }
        }
    }

    #[async_trait::async_trait]
    impl StepRunner for MidRunUnderminer {
        async fn run_step(&self, request: StepRequest) -> Result<StepOutcome, String> {
            self.requests.lock().await.push(request);
            let mut calls = self.calls.lock().await;
            *calls += 1;
            let touched = vec!["y.txt".to_string()];
            Ok(match *calls {
                1 => {
                    std::fs::write(&self.artifact, "MARKER v1").unwrap();
                    let mut o = outcome("built it");
                    o.touched_paths = touched;
                    o
                }
                2 => {
                    std::fs::write(&self.artifact, "full rewrite, marker gone").unwrap();
                    let mut o = outcome("rewrote the doc");
                    if self.report_touch {
                        o.touched_paths = touched;
                    }
                    o
                }
                3 => {
                    std::fs::write(&self.artifact, "MARKER restored").unwrap();
                    let mut o = outcome("restored it");
                    o.touched_paths = touched;
                    o
                }
                _ => {
                    if self.final_clobber {
                        std::fs::write(&self.artifact, "clobbered again, marker gone").unwrap();
                    }
                    let mut o = outcome("all wrapped up\nTASK COMPLETE");
                    if self.final_clobber {
                        o.touched_paths = touched;
                    }
                    o
                }
            })
        }
    }

    fn marker_check() -> AcceptanceCheck {
        AcceptanceCheck::Regex {
            pattern: "MARKER".to_string(),
            path: Some("y.txt".to_string()),
            command: None,
            timeout_secs: None,
        }
    }

    async fn assert_mid_run_regression_caught(source: &MemorySource, runner: &MidRunUnderminer) {
        let requests = runner.requests.lock().await;
        assert_eq!(requests.len(), 4);
        // Caught MID-run: the reopened item is selected while item 2 is
        // still open, and its prompt carries the notice — at drain none of
        // this would exist.
        assert_eq!(requests[1].item_id, 2);
        assert_eq!(requests[2].item_id, 1, "reopened item comes back next");
        assert!(
            requests[2].prompt.contains("you un-did verified work #1"),
            "the very next step must hear about the damage: {}",
            requests[2].prompt
        );
        assert!(requests[2].prompt.contains("Disk is truth"));
        assert!(
            !requests[2].prompt.contains("- #1 build artifact:"),
            "the prompt that sends the model back to #1 must not call #1 verified: {}",
            requests[2].prompt
        );
        assert!(
            !requests[3].prompt.contains("un-did verified work"),
            "the notice is one-shot"
        );
        drop(requests);
        let log = source.log_entries.lock().await;
        assert!(log.iter().any(|(id, a)| *id == 1 && a == "regressed"));
        assert!(log.iter().any(|(id, a)| *id == 1 && a == "reopened"));
        drop(log);
        let notes = source.notes.lock().await;
        assert!(
            notes
                .iter()
                .any(|(id, n)| *id == 1 && n.contains("REOPENED mid-run")),
            "the reopened item carries durable context"
        );
        drop(notes);
    }

    #[tokio::test]
    async fn mid_run_sweep_reopens_a_verified_item_the_next_step_un_did() {
        let dir = tempfile::tempdir().unwrap();
        let source = MemorySource::default();
        source.push(step(1, "build artifact", Some(marker_check()))).await;
        source.push(step(2, "destructive later work", None)).await;
        // The destructive write REPORTS its touched path — the write/edit
        // trigger fires at that very boundary.
        let runner = MidRunUnderminer::new(dir.path().join("y.txt"), true, false);
        let report = LongHorizonRunner::new(fast_config())
            .run("goal", &source, &runner, dir.path(), None)
            .await;
        assert_eq!(report.stop, StopReason::AllTasksDone);
        assert_eq!(report.items_regressed_reopened, 1, "{report:?}");
        assert_eq!(report.items_completed, 2);
        assert_eq!(report.steps_taken, 4);
        assert_eq!(
            report
                .verified_outcomes
                .iter()
                .filter(|o| o.id == 1)
                .count(),
            1,
            "a re-earned verdict is listed once: {:?}",
            report.verified_outcomes
        );
        assert_mid_run_regression_caught(&source, &runner).await;
    }

    #[tokio::test]
    async fn mid_run_periodic_sweep_catches_a_regression_no_write_reported() {
        let dir = tempfile::tempdir().unwrap();
        let source = MemorySource::default();
        source.push(step(1, "build artifact", Some(marker_check()))).await;
        source.push(step(2, "destructive later work", None)).await;
        // The destructive write is INVISIBLE (exec-side-effect shape): only
        // the periodic cadence — funded by the check time already paid, so
        // due immediately for sub-second checks — can catch it.
        let runner = MidRunUnderminer::new(dir.path().join("y.txt"), false, false);
        let report = LongHorizonRunner::new(fast_config())
            .run("goal", &source, &runner, dir.path(), None)
            .await;
        assert_eq!(report.stop, StopReason::AllTasksDone);
        assert_eq!(report.items_regressed_reopened, 1, "{report:?}");
        assert_eq!(report.items_completed, 2);
        assert_eq!(report.steps_taken, 4);
        assert_mid_run_regression_caught(&source, &runner).await;
    }

    #[tokio::test]
    async fn mid_run_sweep_reopens_each_item_at_most_once_per_run() {
        let dir = tempfile::tempdir().unwrap();
        let source = MemorySource::default();
        source.push(step(1, "build artifact", Some(marker_check()))).await;
        source.push(step(2, "destructive later work", None)).await;
        // The final step clobbers the artifact a SECOND time. The reopen
        // bound must hold: item 1 was already reopened once this run, so
        // neither the mid-run sweep nor the drain sweep may reopen it again
        // — the fixpoint guarantee, mirrored from the drain sweep.
        let runner = MidRunUnderminer::new(dir.path().join("y.txt"), true, true);
        let report = LongHorizonRunner::new(fast_config())
            .run("goal", &source, &runner, dir.path(), None)
            .await;
        assert_eq!(report.stop, StopReason::AllTasksDone);
        assert_eq!(report.items_regressed_reopened, 1, "{report:?}");
        assert_eq!(report.items_completed, 2);
        assert_eq!(report.steps_taken, 4, "no reopen loop: {report:?}");
        let log = source.log_entries.lock().await;
        assert_eq!(
            log.iter().filter(|(id, a)| *id == 1 && a == "regressed").count(),
            1,
            "one reopen per item per run"
        );
        // The second clobber stands on disk — proof the bound held rather
        // than the sweep fighting the model forever.
        let content = std::fs::read_to_string(dir.path().join("y.txt")).unwrap();
        assert!(!content.contains("MARKER"));
    }

    #[test]
    fn periodic_resweep_cadence_derives_from_check_cost() {
        let ms = Duration::from_millis;
        // Sub-second checks: the verification time already paid covers a
        // full re-sweep — due immediately.
        assert!(resweep_due(Duration::ZERO, ms(300), ms(300)));
        // Expensive checks (minute-long test suites): a 60s estimated sweep
        // against 10s of paid verification is not affordable — the cadence
        // backs off in proportion, with no fixed N anywhere.
        assert!(!resweep_due(Duration::ZERO, ms(10_000), ms(60_000)));
        // The budget replenishes as further verification is paid for.
        assert!(resweep_due(ms(60_000), ms(120_000), ms(60_000)));
        assert!(!resweep_due(ms(60_000), ms(119_999), ms(60_000)));
    }

    #[test]
    fn resweep_targets_are_full_when_due_and_touched_paths_otherwise() {
        let check = |file: &str| AcceptanceCheck::Command {
            command: format!("sh checks/{file}"),
            timeout_secs: None,
        };
        let eligible = vec![
            (1i64, "a".to_string(), check("test_01.sh")),
            (2i64, "b".to_string(), check("test_02.sh")),
        ];
        // Full sweep due: everything, whatever was touched.
        assert_eq!(select_resweep_targets(eligible.clone(), true, &[]).len(), 2);
        // Not due, nothing touched: nothing to re-check.
        assert_eq!(select_resweep_targets(eligible.clone(), false, &[]), Vec::new());
        // Not due, one referenced path touched: exactly that item,
        // immediately — absolute spelling still collides with the check's
        // relative one.
        let touched = vec!["D:\\ws\\checks\\test_02.sh".to_string()];
        let targets = select_resweep_targets(eligible, false, &touched);
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].0, 2);
    }

    #[test]
    fn acceptance_path_references_match_by_final_component() {
        let cmd = AcceptanceCheck::Command {
            command: "sh checks/test_04.sh".to_string(),
            timeout_secs: None,
        };
        assert!(cmd.references_path("checks/test_04.sh"));
        assert!(cmd.references_path("D:\\ws\\checks\\test_04.sh"));
        assert!(!cmd.references_path("notekeeper.sh"));
        let file = AcceptanceCheck::FileExists {
            path: "data/notes.md".to_string(),
        };
        assert!(file.references_path("notes.md"));
        assert!(!file.references_path("other.md"));
        let rx = AcceptanceCheck::Regex {
            pattern: "x".to_string(),
            path: Some("y.txt".to_string()),
            command: None,
            timeout_secs: None,
        };
        assert!(rx.references_path("./y.txt"));
        assert!(!rx.references_path(""));
    }

    #[test]
    fn touched_path_extraction_covers_write_and_edit_tools_only() {
        let with_file_path = serde_json::json!({ "file_path": "notes.md", "content": "x" });
        assert_eq!(
            touched_path_of("write_file", &with_file_path).as_deref(),
            Some("notes.md")
        );
        let with_path = serde_json::json!({ "path": "notes.md" });
        assert_eq!(
            touched_path_of("edit_file", &with_path).as_deref(),
            Some("notes.md")
        );
        // exec is opaque — the periodic sweep is its backstop.
        assert_eq!(
            touched_path_of("exec", &serde_json::json!({ "command": "rm notes.md" })),
            None
        );
        assert_eq!(touched_path_of("read_file", &with_file_path), None);
    }

    #[test]
    fn regression_notice_stays_one_screenful_however_many_items_regressed() {
        let regressions: Vec<(i64, String, String)> = (0..40)
            .map(|i| {
                (
                    i,
                    format!("feature {i}"),
                    "pattern /MARKER/ did not match (2000 bytes searched)".to_string(),
                )
            })
            .collect();
        let notice = regression_notice_text(&regressions);
        assert!(
            notice.len() <= STEP_RESULT_TAIL_MAX_BYTES,
            "{} bytes",
            notice.len()
        );
        assert!(notice.contains("you un-did verified work #0"));
        assert!(notice.contains("more"), "the overflow is announced");
        assert!(notice.contains("Disk is truth"));
    }

    #[test]
    fn a_pure_answer_fingerprints_and_anything_else_does_not() {
        let answer = outcome("Paris is the capital.");
        let reflowed = outcome("  Paris is\nthe   capital.\n");
        assert!(tool_free_answer_fingerprint(&answer).is_some());
        assert_eq!(
            tool_free_answer_fingerprint(&answer),
            tool_free_answer_fingerprint(&reflowed),
            "re-flowed whitespace is the same answer"
        );
        assert_ne!(
            tool_free_answer_fingerprint(&answer),
            tool_free_answer_fingerprint(&outcome("Paris is the capital city.")),
            "a changed word is a different answer"
        );
        assert_eq!(
            tool_free_answer_fingerprint(&outcome("  \n ")),
            None,
            "blank"
        );
        let mut with_tools = outcome("Paris is the capital.");
        with_tools.tool_calls.push(StepToolCall {
            name: "web_search".to_string(),
            input_digest: "i".to_string(),
            output_digest: "o".to_string(),
            success: true,
        });
        assert_eq!(tool_free_answer_fingerprint(&with_tools), None, "did work");
        let mut looping = outcome("Paris is the capital.");
        looping.degenerate_loop = true;
        assert_eq!(
            tool_free_answer_fingerprint(&looping),
            None,
            "narration loop"
        );

        assert!(answer_converged(Some(7), Some(7)));
        assert!(!answer_converged(Some(7), Some(8)));
        assert!(
            !answer_converged(None, None),
            "two non-answers are not agreement"
        );
        assert!(!answer_converged(Some(7), None));
    }

    /// A model that answers but never says `TASK COMPLETE` used to have its
    /// answer re-run and re-streamed until the fruitless budget abandoned the
    /// item. The second identical tool-free answer now closes it.
    #[tokio::test]
    async fn an_unchecked_item_completes_when_its_answer_converges() {
        let dir = tempfile::tempdir().unwrap();
        let source = MemorySource::default();
        source
            .push(step(1, "What is the capital of France?", None))
            .await;
        let runner = ScriptedRunner::new(vec![
            Ok(outcome("Paris is the capital of France.")),
            Ok(outcome("Paris is the capital\nof France.")),
        ]);
        let report = LongHorizonRunner::new(LongHorizonConfig::default())
            .run("goal", &source, &runner, dir.path(), None)
            .await;
        assert_eq!(report.stop, StopReason::AllTasksDone, "{report:?}");
        assert_eq!(report.items_completed, 1);
        assert_eq!(
            report.items_completed_unverified, 1,
            "still the model's word, not a check's"
        );
        assert_eq!(report.steps_taken, 2, "two answers, then done: {report:?}");
        let log = source.log_entries.lock().await;
        assert!(
            log.iter().any(|(_, a)| a == "completed_converged"),
            "{log:?}"
        );
        assert!(
            !log.iter().any(|(_, a)| a == "completed_unverified"),
            "{log:?}"
        );
        drop(log);
    }

    /// Convergence needs the SAME answer on CONSECUTIVE pure-answer steps: a
    /// changed answer is still moving, and a step that did work in between
    /// breaks the chain — its world may have changed what the answer is.
    #[tokio::test]
    async fn a_changed_answer_or_work_in_between_is_not_convergence() {
        let dir = tempfile::tempdir().unwrap();
        let source = MemorySource::default();
        source.push(step(1, "fuzzy item", None)).await;
        let mut with_tools = outcome("Paris is the capital of France.");
        with_tools.tool_calls.push(StepToolCall {
            name: "web_search".to_string(),
            input_digest: "i".to_string(),
            output_digest: "o".to_string(),
            success: true,
        });
        let runner = ScriptedRunner::new(vec![
            Ok(outcome("Lyon, maybe.")),
            Ok(outcome("Paris is the capital of France.")),
            Ok(with_tools),
            Ok(outcome("Paris is the capital of France.")),
            Ok(outcome("done\nTASK COMPLETE")),
        ]);
        let report = LongHorizonRunner::new(LongHorizonConfig::default())
            .run("goal", &source, &runner, dir.path(), None)
            .await;
        assert_eq!(report.items_completed, 1);
        assert_eq!(
            report.steps_taken, 5,
            "no step before the claim converged: {report:?}"
        );
        let log = source.log_entries.lock().await;
        assert!(
            !log.iter().any(|(_, a)| a == "completed_converged"),
            "{log:?}"
        );
        assert!(
            log.iter().any(|(_, a)| a == "completed_unverified"),
            "{log:?}"
        );
        drop(log);
    }

    #[test]
    fn the_abandonment_headline_counts_steps_not_charges() {
        let headline = abandonment_headline(6, 8, 2);
        assert!(
            headline.starts_with("abandoned after 6 steps (2 of them replans)"),
            "{headline}"
        );
        assert!(headline.contains("8 no-progress charges"), "{headline}");
    }

    #[tokio::test]
    async fn unchecked_item_completes_on_claim_but_is_flagged_unverified() {
        let dir = tempfile::tempdir().unwrap();
        let source = MemorySource::default();
        source.push(step(1, "fuzzy item", None)).await;
        let runner = ScriptedRunner::new(vec![Ok(outcome("all wrapped up\nTASK COMPLETE"))]);
        let report = LongHorizonRunner::new(fast_config())
            .run("goal", &source, &runner, dir.path(), None)
            .await;
        assert_eq!(report.items_completed, 1);
        assert_eq!(report.items_completed_unverified, 1);
        let log = source.log_entries.lock().await;
        assert!(log.iter().any(|(_, a)| a == "completed_unverified"));
        drop(log);
    }

    #[tokio::test]
    async fn false_success_claim_is_refuted_replanned_then_abandoned() {
        // Suite 4 fixture: the model *claims* completion every step, the
        // environment never changes. The harness must never record a
        // completion — this is the anti-drift keystone.
        let dir = tempfile::tempdir().unwrap();
        let source = MemorySource::default();
        source
            .push(step(
                1,
                "impossible item",
                Some(AcceptanceCheck::FileExists {
                    path: "never-created.txt".to_string(),
                }),
            ))
            .await;
        let claim = || Ok(outcome("I finished!\nTASK COMPLETE"));
        let runner = ScriptedRunner::new(vec![claim(), claim(), claim(), claim(), claim()]);
        let report = LongHorizonRunner::new(fast_config())
            .run("goal", &source, &runner, dir.path(), None)
            .await;
        assert_eq!(
            report.stop,
            StopReason::AllTasksDone,
            "abandoned ⇒ plan drains"
        );
        assert_eq!(
            report.items_completed, 0,
            "false success must never complete"
        );
        assert_eq!(report.items_abandoned, 1);
        assert!(
            report.false_success_claims >= 2,
            "claims were counted: {report:?}"
        );
        assert_eq!(report.replans, 1);
        // 2 execute steps -> replan -> 2 more execute steps -> abandon = 5 steps max
        assert!(
            report.steps_taken <= 5,
            "grinding must be bounded: {report:?}"
        );
    }

    #[tokio::test]
    async fn replan_step_uses_plan_kind_and_replan_prompt() {
        let dir = tempfile::tempdir().unwrap();
        let source = MemorySource::default();
        source
            .push(step(
                1,
                "stubborn item",
                Some(AcceptanceCheck::FileExists {
                    path: "never.txt".to_string(),
                }),
            ))
            .await;
        let runner = ScriptedRunner::new(vec![
            Ok(outcome("try 1")),
            Ok(outcome("try 2")),
            Ok(outcome("decomposed")),
            Ok(outcome("try 3")),
            Ok(outcome("try 4")),
        ]);
        let report = LongHorizonRunner::new(fast_config())
            .run("goal", &source, &runner, dir.path(), None)
            .await;
        assert_eq!(report.replans, 1);
        let requests = runner.requests.lock().await;
        let plan_steps: Vec<_> = requests
            .iter()
            .filter(|r| r.step_kind == StepKind::Plan)
            .collect();
        assert_eq!(plan_steps.len(), 1, "exactly one replan step");
        assert!(plan_steps[0].prompt.contains("== REPLAN REQUIRED =="));
        // The stalled item must be named so the subtasks are parented to it —
        // but as the OPERATION, not as `parent_id=1` inside a literal tool
        // signature. A prompt that spells out a call it may not ship the schema
        // for teaches the model to guess argument names.
        assert!(plan_steps[0].prompt.contains("children of task #1"));
        assert!(
            !plan_steps[0].prompt.contains("todo("),
            "the replan prompt must not name a tool call verbatim"
        );
        // Execute steps carry the item's tool scope and step bounds.
        let exec = requests
            .iter()
            .find(|r| r.step_kind == StepKind::Execute)
            .unwrap();
        assert_eq!(exec.tool_scope, vec!["exec".to_string()]);
        // P22: no fixed per-step iteration cap — the runner ends the step on
        // progress exhaustion; wall clock is the bound that rides along.
        assert_eq!(exec.max_iterations, None);
        assert!(exec.max_wall_clock.is_some());
        drop(requests);
    }

    /// Build a stalled item whose acceptance can never pass, with an
    /// open-children ledger the runner can report from.
    fn replan_fixture(per_replan: usize) -> (MemorySource, ReplanningRunner) {
        let ledger = Arc::new(Mutex::new(HashMap::new()));
        let source = MemorySource {
            subtasks: Some(ledger.clone()),
            ..MemorySource::default()
        };
        let runner = ReplanningRunner {
            // Long enough that the script never runs dry first — the
            // assertions are about how many steps the LADDER takes, not
            // about the script running out.
            inner: ScriptedRunner::new((0..10).map(|i| Ok(outcome(&format!("try {i}")))).collect()),
            subtasks: ledger,
            per_replan,
        };
        (source, runner)
    }

    /// REGRESSION (live 2026-08-02, session 05775d1d): a replan re-proposed
    /// the title the run had just abandoned, the store refused to resurrect
    /// it, and the unconditional grind-counter reset handed the item another
    /// full allowance of fruitless steps anyway. A replan that decomposed
    /// NOTHING has produced no new work, so it earns no fresh allowance —
    /// the item stays at its grind threshold and the existing
    /// `max_replans_per_item` rung reaches abandonment directly.
    #[tokio::test]
    async fn a_dry_replan_does_not_buy_a_fresh_allowance() {
        let dir = tempfile::tempdir().unwrap();
        let (source, runner) = replan_fixture(0);
        source
            .push(step(
                1,
                "stubborn item",
                Some(AcceptanceCheck::FileExists {
                    path: "never.txt".to_string(),
                }),
            ))
            .await;

        let report = LongHorizonRunner::new(fast_config())
            .run("goal", &source, &runner, dir.path(), None)
            .await;

        // 2 execute steps reach the grind threshold, 1 replan produces
        // nothing, and the next iteration abandons — no second allowance.
        assert_eq!(report.replans, 1);
        assert_eq!(report.items_abandoned, 1);
        assert_eq!(
            report.steps_taken, 3,
            "a dry replan must not restart the grind allowance: {report:?}"
        );
    }

    /// The other half of the contract: a replan that DID decompose still
    /// resets the counter, so real new work gets its full allowance.
    #[tokio::test]
    async fn a_productive_replan_still_resets_the_grind_counter() {
        let dir = tempfile::tempdir().unwrap();
        let (source, runner) = replan_fixture(2);
        source
            .push(step(
                1,
                "stubborn item",
                Some(AcceptanceCheck::FileExists {
                    path: "never.txt".to_string(),
                }),
            ))
            .await;

        let report = LongHorizonRunner::new(fast_config())
            .run("goal", &source, &runner, dir.path(), None)
            .await;

        // 2 execute + 1 replan + 2 more execute, then abandonment.
        assert_eq!(report.replans, 1);
        assert_eq!(report.items_abandoned, 1);
        assert_eq!(
            report.steps_taken, 5,
            "a replan that added subtasks keeps its fresh allowance: {report:?}"
        );
    }

    /// A source that cannot report its children is unchanged: `None` means
    /// unknown, and unknown is treated as productive rather than starving a
    /// legitimate decomposition.
    #[tokio::test]
    async fn a_source_that_cannot_report_children_keeps_the_old_behavior() {
        let dir = tempfile::tempdir().unwrap();
        let source = MemorySource::default();
        source
            .push(step(
                1,
                "stubborn item",
                Some(AcceptanceCheck::FileExists {
                    path: "never.txt".to_string(),
                }),
            ))
            .await;
        let runner =
            ScriptedRunner::new((0..10).map(|i| Ok(outcome(&format!("try {i}")))).collect());

        let report = LongHorizonRunner::new(fast_config())
            .run("goal", &source, &runner, dir.path(), None)
            .await;
        assert_eq!(report.steps_taken, 5, "unreported children ⇒ productive");
    }

    /// REGRESSION (instrumented, 84 firings): the decomposition rung asked
    /// the SAME question twice — 42 first attempts and 42 second attempts,
    /// zero subtasks between them — and then abandoned every one of those
    /// items with a sentence describing the outcome as though decomposition
    /// had happened. After a dry attempt the harness must change the
    /// question: one EXECUTE step carrying the item's own failing result,
    /// which the replan prompt is the only step prompt never to receive.
    #[tokio::test]
    async fn a_dry_replan_escalates_instead_of_repeating_the_ask() {
        let dir = tempfile::tempdir().unwrap();
        let (source, runner) = replan_fixture(0);
        source
            .push(step(
                1,
                "stubborn item",
                Some(AcceptanceCheck::FileExists {
                    path: "never.txt".to_string(),
                }),
            ))
            .await;
        // The default replan allowance, where the second rung exists at all.
        let config = LongHorizonConfig {
            max_steps_per_item: 2,
            max_replans_per_item: 2,
            ..LongHorizonConfig::default()
        };

        let report = LongHorizonRunner::new(config)
            .run("goal", &source, &runner, dir.path(), None)
            .await;

        assert_eq!(report.items_abandoned, 1, "{report:?}");
        // 2 execute + 1 decomposition ask + 1 escalated ask, then
        // abandonment: the ladder is exactly as long as before, one rung of
        // it asks something different.
        assert_eq!(report.steps_taken, 4, "{report:?}");
        assert_eq!(
            report.replans, 2,
            "the escalated rung still spends one replan allowance, or the \
             abandonment gate never converges: {report:?}"
        );

        let requests = runner.inner.requests.lock().await;
        let plans = requests
            .iter()
            .filter(|r| r.step_kind == StepKind::Plan)
            .count();
        assert_eq!(plans, 1, "the decomposition ask is never repeated");

        let escalated = requests.last().expect("four steps ran");
        assert_eq!(escalated.step_kind, StepKind::Execute);
        assert!(
            escalated.prompt.contains("STALLED — READ THIS BEFORE ACTING"),
            "{}",
            escalated.prompt
        );
        assert!(
            !escalated.prompt.contains("== REPLAN REQUIRED =="),
            "the escalated rung must not re-ask for a plan: {}",
            escalated.prompt
        );
        assert!(
            escalated.prompt.contains("Done-condition NOT met"),
            "the escalated rung must show the failing result it is stuck on: {}",
            escalated.prompt
        );
        drop(requests);

        let reasons = source.abandon_reasons.lock().await;
        let (_, reason) = reasons.first().expect("the item was abandoned");
        assert!(
            reason.contains("came back with no subtasks")
                && reason.contains("escalated next-action ask"),
            "the reason must say BOTH attempts came back empty: {reason}"
        );
        drop(reasons);
    }

    /// REGRESSION: an abandoned item reached the report by NAME only if it
    /// carried a check, and 81% of everything the store has ever abandoned
    /// carries none — so the majority path left a count and nothing to read.
    /// In one observed session the item that vanished that way was the root
    /// goal itself.
    #[tokio::test]
    async fn an_unchecked_abandonment_is_named_not_just_counted() {
        let dir = tempfile::tempdir().unwrap();
        let (source, runner) = replan_fixture(0);
        source.push(step(1, "answer the question", None)).await;

        let report = LongHorizonRunner::new(fast_config())
            .run("goal", &source, &runner, dir.path(), None)
            .await;

        assert_eq!(report.items_abandoned, 1, "{report:?}");
        assert!(
            report.abandoned_unmet.is_empty(),
            "with no check nothing can be called provably unmet: {report:?}"
        );
        assert_eq!(report.abandoned_unverifiable.len(), 1, "{report:?}");
        let walked_away = &report.abandoned_unverifiable[0];
        assert_eq!(walked_away.id, 1);
        assert_eq!(walked_away.title, "answer the question");
        assert!(
            walked_away.reason.contains("abandoned after"),
            "{}",
            walked_away.reason
        );
        // With no check, what the model last said IS the whole record.
        assert_eq!(walked_away.last_result.as_deref(), Some("try 1"));
    }

    /// REGRESSION: the SECOND abandonment site recorded nothing at all — not
    /// even for an item that carries a check — so a poisoned item could
    /// never be revived by either sweep. A prompt the model chokes on says
    /// nothing about the world, and the world is what the check reads.
    #[tokio::test]
    async fn a_poisoned_item_with_a_check_still_reaches_the_sweep() {
        let dir = tempfile::tempdir().unwrap();
        // The condition is already true — the artifact is there, only the
        // model's prompt is broken.
        std::fs::write(dir.path().join("done.txt"), b"there").unwrap();
        let source = MemorySource::default();
        source
            .push(step(
                1,
                "poisoned but satisfied",
                Some(AcceptanceCheck::FileExists {
                    path: "done.txt".to_string(),
                }),
            ))
            .await;
        let runner = ScriptedRunner::new(vec![
            Err("empty completion".to_string()),
            Err("empty completion".to_string()),
        ]);

        let report = LongHorizonRunner::new(fast_config())
            .run("goal", &source, &runner, dir.path(), None)
            .await;

        assert_eq!(report.stop, StopReason::AllTasksDone, "{report:?}");
        assert_eq!(
            report.items_revived, 1,
            "the drain sweep must be able to see a poisoned item: {report:?}"
        );
        assert_eq!(report.items_abandoned, 0, "{report:?}");
        assert_eq!(report.items_completed, 1, "{report:?}");
        assert!(
            report.abandoned_unverifiable.is_empty(),
            "it had a check, so the sweep speaks for it: {report:?}"
        );
    }

    #[tokio::test]
    async fn repeated_tool_signatures_accelerate_the_stall_counter() {
        let dir = tempfile::tempdir().unwrap();
        let source = MemorySource::default();
        source
            .push(step(
                1,
                "loopy item",
                Some(AcceptanceCheck::FileExists {
                    path: "never.txt".to_string(),
                }),
            ))
            .await;
        let looped = || {
            Ok(StepOutcome {
                text: "trying the same thing".to_string(),
                input_tokens: 1000,
                output_tokens: 200,
                tool_calls: vec![StepToolCall {
                    name: "exec".to_string(),
                    input_digest: "same".to_string(),
                    output_digest: "same".to_string(),
                    success: true,
                }],
                touched_paths: vec![],
                degenerate_loop: false,
            })
        };
        let runner = ScriptedRunner::new(vec![looped(), looped(), looped(), looped()]);
        let config = LongHorizonConfig {
            max_steps_per_item: 4,
            max_replans_per_item: 0,
            ..LongHorizonConfig::default()
        };
        let report = LongHorizonRunner::new(config)
            .run("goal", &source, &runner, dir.path(), None)
            .await;
        // Identical signatures double the stall increment: 1 + 2 + 2 = 5 ≥ 4
        // after 3 steps, instead of 4 — the loop is cut short.
        assert_eq!(report.items_abandoned, 1);
        assert!(
            report.steps_taken < 4,
            "repetition must accelerate abandonment: {report:?}"
        );
    }

    #[tokio::test]
    async fn wall_clock_budget_stops_the_run() {
        let dir = tempfile::tempdir().unwrap();
        let source = MemorySource::default();
        source.push(step(1, "any", None)).await;
        let runner = ScriptedRunner::new(vec![]);
        let config = LongHorizonConfig {
            max_wall_clock: Duration::ZERO,
            ..LongHorizonConfig::default()
        };
        let report = LongHorizonRunner::new(config)
            .run("goal", &source, &runner, dir.path(), None)
            .await;
        assert_eq!(report.stop, StopReason::WallClockExhausted);
        assert_eq!(report.steps_taken, 0);
    }

    #[tokio::test]
    async fn token_budget_stops_the_run() {
        let dir = tempfile::tempdir().unwrap();
        let source = MemorySource::default();
        source.push(step(1, "endless", None)).await;
        let runner = ScriptedRunner::new(vec![
            Ok(outcome("no claim")),
            Ok(outcome("no claim")),
            Ok(outcome("no claim")),
        ]);
        let config = LongHorizonConfig {
            max_total_tokens: Some(1500),
            max_steps_per_item: 100,
            ..LongHorizonConfig::default()
        };
        let report = LongHorizonRunner::new(config)
            .run("goal", &source, &runner, dir.path(), None)
            .await;
        assert_eq!(report.stop, StopReason::TokenBudgetExhausted);
        assert_eq!(
            report.steps_taken, 2,
            "1200 tokens after step 1 < 1500; stop after step 2"
        );
    }

    #[tokio::test]
    async fn cancellation_flag_stops_before_the_next_step() {
        let dir = tempfile::tempdir().unwrap();
        let source = MemorySource::default();
        source.push(step(1, "any", None)).await;
        let runner = ScriptedRunner::new(vec![]);
        let cancel = CancelToken::new();
        cancel.cancel();
        let report = LongHorizonRunner::new(fast_config())
            .run("goal", &source, &runner, dir.path(), Some(cancel))
            .await;
        assert_eq!(report.stop, StopReason::Cancelled);
        assert_eq!(report.steps_taken, 0);
    }

    /// REGRESSION (GUI live drive, 2026-07-30): Stop had no effect for
    /// minutes because the flag reached only the harness boundary — the
    /// in-step agent loop got `cancel: None`. Every step request
    /// must carry the SAME flag the Stop button flips, so the runner can
    /// abort the in-flight stream, not just decline the next step.
    #[tokio::test]
    async fn every_step_request_carries_the_runs_cancel_flag() {
        let dir = tempfile::tempdir().unwrap();
        let source = MemorySource::default();
        source.push(step(1, "answer", None)).await;
        let runner = ScriptedRunner::new(vec![Ok(outcome("TASK COMPLETE"))]);
        let cancel = CancelToken::new();
        let report = LongHorizonRunner::new(fast_config())
            .run("goal", &source, &runner, dir.path(), Some(cancel.clone()))
            .await;
        assert_eq!(report.stop, StopReason::AllTasksDone, "{report:?}");

        let requests = runner.requests.lock().await;
        assert!(!requests.is_empty());
        for request in requests.iter() {
            let threaded = request
                .cancel
                .as_ref()
                .expect("the step request must carry the run's cancel flag");
            assert!(
                CancelToken::same_token(threaded, &cancel),
                "must be the SAME token the Stop button cancels, not a fresh one"
            );
        }
    }

    /// Cancels its step's own token mid-step and returns partial output —
    /// the scripted stand-in for the real agent loop observing
    /// `RunOptions.cancel` while a stream is in flight.
    struct CancelsDuringItsStep;

    #[async_trait::async_trait]
    impl StepRunner for CancelsDuringItsStep {
        async fn run_step(&self, request: StepRequest) -> Result<StepOutcome, String> {
            request
                .cancel
                .as_ref()
                .expect("the token is threaded into the step")
                .cancel();
            Ok(outcome("partial work\n\n[Cancelled by user]"))
        }
    }

    #[tokio::test]
    async fn a_cancellation_injected_mid_step_stops_the_run_at_that_step() {
        let dir = tempfile::tempdir().unwrap();
        let source = MemorySource::default();
        source.push(step(1, "first", None)).await;
        source.push(step(2, "second", None)).await;
        let cancel = CancelToken::new();
        let report = LongHorizonRunner::new(fast_config())
            .run("goal", &source, &CancelsDuringItsStep, dir.path(), Some(cancel))
            .await;
        assert_eq!(report.stop, StopReason::Cancelled);
        assert_eq!(report.steps_taken, 1, "the second item must never start");
        assert!(
            source.completions.lock().await.is_empty(),
            "a truncated step is not a verdictable claim — nothing completes"
        );
        assert_eq!(
            source.notes.lock().await.len(),
            1,
            "the cancelled step's partial findings still land in the store"
        );
    }

    #[tokio::test]
    async fn poisoned_item_is_abandoned_and_the_run_continues() {
        // An item whose prompt deterministically breaks the model must cost
        // one feature, not the whole run: errors follow the item, the item
        // gets abandoned, and the next item proceeds.
        let dir = tempfile::tempdir().unwrap();
        let source = MemorySource::default();
        source.push(step(1, "poisoned", None)).await;
        source.push(step(2, "healthy", None)).await;
        let runner = ScriptedRunner::new(vec![
            Err("empty completion".to_string()),
            Err("empty completion".to_string()),
            Ok(outcome("TASK COMPLETE")),
        ]);
        let report = LongHorizonRunner::new(fast_config())
            .run("goal", &source, &runner, dir.path(), None)
            .await;
        assert_eq!(report.stop, StopReason::AllTasksDone, "{report:?}");
        assert_eq!(report.items_abandoned, 1, "poisoned item abandoned");
        assert_eq!(report.items_completed, 1, "healthy item still completed");
        assert_eq!(
            report.last_runner_error.as_deref(),
            Some("empty completion"),
            "the error that poisoned the item survives into the report"
        );
    }

    #[tokio::test]
    async fn deterministic_runner_error_reports_why_the_plan_drained() {
        // The live 2026-07-31 failure: a model whose provider is not
        // configured fails every run_step call identically. Poison
        // containment abandons the only item and the run exits AllTasksDone
        // with zero steps — numerically identical to a finished run. The
        // report must still carry the error so the caller can say WHY.
        let dir = tempfile::tempdir().unwrap();
        let source = MemorySource::default();
        source.push(step(1, "the user's prompt", None)).await;
        let runner = ScriptedRunner::new(vec![
            Err("No provider available for model 'claude-fable-5'".to_string()),
            Err("No provider available for model 'claude-fable-5'".to_string()),
        ]);
        let report = LongHorizonRunner::new(fast_config())
            .run("goal", &source, &runner, dir.path(), None)
            .await;
        assert_eq!(report.stop, StopReason::AllTasksDone, "{report:?}");
        assert_eq!(report.steps_taken, 0);
        assert_eq!(report.items_completed, 0);
        assert_eq!(report.items_abandoned, 1);
        assert_eq!(
            report.last_runner_error.as_deref(),
            Some("No provider available for model 'claude-fable-5'"),
            "the report must name the fault that emptied the plan"
        );
    }

    #[tokio::test]
    async fn consecutive_runner_errors_trip_the_circuit_breaker() {
        // Provider-wide death: errors SPAN items (poison containment first
        // abandons the item they follow), so the run-level breaker trips
        // once the error streak crosses into a second item.
        let dir = tempfile::tempdir().unwrap();
        let source = MemorySource::default();
        source.push(step(1, "first", None)).await;
        source.push(step(2, "second", None)).await;
        let runner = ScriptedRunner::new(vec![
            Err("boom".to_string()),
            Err("boom".to_string()),
            Err("boom".to_string()),
        ]);
        let report = LongHorizonRunner::new(fast_config())
            .run("goal", &source, &runner, dir.path(), None)
            .await;
        assert!(matches!(report.stop, StopReason::RunnerErrors { .. }), "{report:?}");
        assert_eq!(report.steps_taken, 0, "failed steps are not progress");
        assert_eq!(report.items_abandoned, 1, "first item was contained as poisoned");
        assert_eq!(report.last_runner_error.as_deref(), Some("boom"));
    }

    #[tokio::test]
    async fn source_error_stops_the_run() {
        let dir = tempfile::tempdir().unwrap();
        let source = MemorySource::default();
        *source.fail_next.lock().await = true;
        let runner = ScriptedRunner::new(vec![]);
        let report = LongHorizonRunner::new(fast_config())
            .run("goal", &source, &runner, dir.path(), None)
            .await;
        assert!(matches!(report.stop, StopReason::SourceError { .. }));
    }

    #[tokio::test]
    async fn step_findings_are_written_to_task_notes() {
        let dir = tempfile::tempdir().unwrap();
        let source = MemorySource::default();
        source.push(step(1, "note me", None)).await;
        let runner = ScriptedRunner::new(vec![Ok(outcome(
            "found the config in crates/nanna-config\nTASK COMPLETE",
        ))]);
        let _ = LongHorizonRunner::new(fast_config())
            .run("goal", &source, &runner, dir.path(), None)
            .await;
        let notes = source.notes.lock().await;
        assert_eq!(notes.len(), 1);
        assert!(notes[0].1.contains("nanna-config"));
        drop(notes);
    }

    // -----------------------------------------------------------------
    // Suite 4 benchmark fixtures (deterministic; cited by bench/BASELINE.md)
    // -----------------------------------------------------------------

    /// Deterministic task-success @ tokens for a fully compliant scripted
    /// model: 3 items, 1 step each at 1200 tokens ⇒ 3600 total, 1200/item.
    #[tokio::test]
    async fn compliant_run_success_at_tokens_baseline() {
        let dir = tempfile::tempdir().unwrap();
        let source = MemorySource::default();
        for id in 1..=3 {
            source.push(step(id, "item", None)).await;
        }
        let runner = ScriptedRunner::new(vec![
            Ok(outcome("TASK COMPLETE")),
            Ok(outcome("TASK COMPLETE")),
            Ok(outcome("TASK COMPLETE")),
        ]);
        let report = LongHorizonRunner::new(LongHorizonConfig::default())
            .run("goal", &source, &runner, dir.path(), None)
            .await;
        assert_eq!(report.stop, StopReason::AllTasksDone);
        assert_eq!(report.items_completed, 3);
        assert_eq!(report.input_tokens + report.output_tokens, 3600);
        assert_eq!(report.tokens_per_completed_item, Some(1200));
    }

    /// Deterministic drift containment: a permanently-false-claiming model
    /// spends at most 6000 tokens (5 steps) before its item is closed, and
    /// records zero completions.
    #[tokio::test]
    async fn drift_containment_cost_baseline() {
        let dir = tempfile::tempdir().unwrap();
        let source = MemorySource::default();
        source
            .push(step(
                1,
                "impossible",
                Some(AcceptanceCheck::FileExists {
                    path: "never.txt".to_string(),
                }),
            ))
            .await;
        let claim = || Ok(outcome("TASK COMPLETE"));
        let runner =
            ScriptedRunner::new(vec![claim(), claim(), claim(), claim(), claim(), claim()]);
        let report = LongHorizonRunner::new(fast_config())
            .run("goal", &source, &runner, dir.path(), None)
            .await;
        assert_eq!(report.items_completed, 0);
        assert_eq!(report.items_abandoned, 1);
        assert!(
            report.input_tokens + report.output_tokens <= 6000,
            "drift cost must be bounded: {report:?}"
        );
    }

    // -----------------------------------------------------------------
    // Interjection — user input joining a run at a step boundary
    // -----------------------------------------------------------------

    /// Admits a scripted batch of items the first time it is polled, by
    /// pushing them into the shared source (as the daemon impl does).
    struct ScriptedInterjector {
        source: Arc<MemorySource>,
        /// Batches to admit, one per poll; empty batches mean "nothing waiting".
        batches: Mutex<VecDeque<Vec<TaskStep>>>,
        polls: Mutex<usize>,
        fail_first: Mutex<bool>,
    }

    impl ScriptedInterjector {
        fn new(source: Arc<MemorySource>, batches: Vec<Vec<TaskStep>>) -> Self {
            Self {
                source,
                batches: Mutex::new(batches.into()),
                polls: Mutex::new(0),
                fail_first: Mutex::new(false),
            }
        }
    }

    #[async_trait::async_trait]
    impl Interjector for ScriptedInterjector {
        async fn interject(&self) -> Result<usize, String> {
            *self.polls.lock().await += 1;
            {
                let mut fail = self.fail_first.lock().await;
                if *fail {
                    *fail = false;
                    return Err("store unavailable".to_string());
                }
                drop(fail);
            }
            let batch = self.batches.lock().await.pop_front().unwrap_or_default();
            let admitted = batch.len();
            for step in batch {
                self.source.push(step).await;
            }
            Ok(admitted)
        }
    }

    #[tokio::test]
    async fn interjected_task_is_executed_in_the_same_run() {
        let source = Arc::new(MemorySource::default());
        source.push(step(1, "original work", None)).await;
        // Nothing waiting on the first boundary; a message lands on the second.
        let interjector = ScriptedInterjector::new(
            source.clone(),
            vec![vec![], vec![step(2, "the interjected ask", None)]],
        );
        let runner = ScriptedRunner::new(vec![Ok(outcome("did 1
TASK COMPLETE")), Ok(outcome("did 2
TASK COMPLETE"))]);

        let report = LongHorizonRunner::new(fast_config())
            .run_with_interjector(
                "goal",
                source.as_ref(),
                &runner,
                Path::new("."),
                None,
                Some(&interjector),
            )
            .await;

        assert_eq!(report.stop, StopReason::AllTasksDone);
        assert_eq!(report.interjected_items, 1);
        assert_eq!(report.items_completed, 2, "both items ran: {report:?}");
        let titles: Vec<String> = runner
            .requests
            .lock()
            .await
            .iter()
            .map(|r| r.prompt.clone())
            .collect();
        assert!(
            titles.iter().any(|p| p.contains("the interjected ask")),
            "the mid-run message must reach a step prompt"
        );
    }

    #[tokio::test]
    async fn interjection_is_polled_before_every_step_selection() {
        let source = Arc::new(MemorySource::default());
        source.push(step(1, "a", None)).await;
        source.push(step(2, "b", None)).await;
        let interjector = ScriptedInterjector::new(source.clone(), vec![]);
        let runner = ScriptedRunner::new(vec![Ok(outcome("x
TASK COMPLETE")), Ok(outcome("y
TASK COMPLETE"))]);

        LongHorizonRunner::new(fast_config())
            .run_with_interjector(
                "goal",
                source.as_ref(),
                &runner,
                Path::new("."),
                None,
                Some(&interjector),
            )
            .await;

        // Two steps plus the final poll that discovers the empty plan.
        assert_eq!(*interjector.polls.lock().await, 3);
    }

    #[tokio::test]
    async fn a_failing_interjection_does_not_kill_work_in_flight() {
        let source = Arc::new(MemorySource::default());
        source.push(step(1, "keep going", None)).await;
        let interjector = ScriptedInterjector::new(source.clone(), vec![]);
        *interjector.fail_first.lock().await = true;
        let runner = ScriptedRunner::new(vec![Ok(outcome("done
TASK COMPLETE"))]);

        let report = LongHorizonRunner::new(fast_config())
            .run_with_interjector(
                "goal",
                source.as_ref(),
                &runner,
                Path::new("."),
                None,
                Some(&interjector),
            )
            .await;

        assert_eq!(report.stop, StopReason::AllTasksDone);
        assert_eq!(report.items_completed, 1);
        assert_eq!(report.interjected_items, 0);
    }

    #[tokio::test]
    async fn interjection_revives_a_plan_that_had_drained() {
        // The message lands exactly as the plan empties — the boundary poll
        // happens BEFORE next(), so the run continues instead of reporting
        // AllTasksDone and making the user start over.
        let source = Arc::new(MemorySource::default());
        source.push(step(1, "only item", None)).await;
        let interjector = ScriptedInterjector::new(
            source.clone(),
            vec![vec![], vec![step(2, "arrived at the buzzer", None)]],
        );
        let runner = ScriptedRunner::new(vec![Ok(outcome("first
TASK COMPLETE")), Ok(outcome("second
TASK COMPLETE"))]);

        let report = LongHorizonRunner::new(fast_config())
            .run_with_interjector(
                "goal",
                source.as_ref(),
                &runner,
                Path::new("."),
                None,
                Some(&interjector),
            )
            .await;

        assert_eq!(report.items_completed, 2);
        assert_eq!(report.interjected_items, 1);
    }

    #[tokio::test]
    async fn run_without_an_interjector_behaves_exactly_as_before() {
        let source = Arc::new(MemorySource::default());
        source.push(step(1, "a", None)).await;
        let runner = ScriptedRunner::new(vec![Ok(outcome("x
TASK COMPLETE"))]);

        let report = LongHorizonRunner::new(fast_config())
            .run("goal", source.as_ref(), &runner, Path::new("."), None)
            .await;

        assert_eq!(report.stop, StopReason::AllTasksDone);
        assert_eq!(report.items_completed, 1);
        assert_eq!(report.interjected_items, 0);
    }

    // -----------------------------------------------------------------
    // P22 Tier 1: step & budget semantics
    // -----------------------------------------------------------------

    /// A timed-out acceptance check is UNKNOWN, not failed — and unknowns
    /// are never counted into a verdict. The TIMEOUT itself charges nothing
    /// (no failure signature, no refuted claim); the step beside the silent
    /// check is judged purely by its own evidence, like a step with no check
    /// at all — here the steps are empty-handed, so they charge as
    /// empty-handed steps and the item converges on the NORMAL ladder, with
    /// the hang carried as a finding in notes, the next prompt, the
    /// abandonment reason, and the drain sweep's unmet detail.
    #[tokio::test]
    async fn timed_out_acceptance_is_unknown_not_failed() {
        let dir = tempfile::tempdir().unwrap();
        let source = MemorySource::default();
        #[cfg(windows)]
        let hang = "ping -n 30 127.0.0.1";
        #[cfg(not(windows))]
        let hang = "sleep 30";
        source
            .push(step(
                1,
                "hang victim",
                Some(AcceptanceCheck::Command {
                    command: hang.to_string(),
                    timeout_secs: Some(1),
                }),
            ))
            .await;
        let runner = ScriptedRunner::new(vec![
            Ok(outcome("looked around")),
            Ok(outcome("looked around more")),
        ]);
        let config = LongHorizonConfig {
            max_steps_per_item: 2,
            max_replans_per_item: 0,
            ..LongHorizonConfig::default()
        };
        let report = LongHorizonRunner::new(config)
            .run("goal", &source, &runner, dir.path(), None)
            .await;

        // Two empty-handed steps spend the 2-step allowance exactly as they
        // would under any silent verdict — the timeouts added nothing on
        // top, and no unknown-counter shortened the road either.
        assert_eq!(report.steps_taken, 2, "{report:?}");
        assert_eq!(report.items_abandoned, 1);
        // 2 post-step timeouts + 1 at the drain sweep.
        assert_eq!(report.acceptance_unknown, 3);
        // The hang is a first-class finding: durable note…
        let notes = source.notes.lock().await;
        assert!(
            notes.iter().any(|(_, n)| n.contains("ACCEPTANCE CHECK HUNG")),
            "hang finding must be recorded as a note"
        );
        drop(notes);
        // …in the NEXT step's prompt (carried via last_result)…
        let requests = runner.requests.lock().await;
        assert!(
            requests[1].prompt.contains("ACCEPTANCE CHECK HUNG"),
            "the step after a hang must hear about it: {}",
            requests[1].prompt
        );
        drop(requests);
        // …and in the abandonment reason — named, not laundered into
        // generic fruitlessness.
        let reasons = source.abandon_reasons.lock().await;
        assert!(
            reasons.iter().any(|(_, r)| r.contains("hung")),
            "abandonment must name the hang: {reasons:?}"
        );
        drop(reasons);
        // The drain sweep reports UNKNOWN, not a fabricated failure.
        assert_eq!(report.abandoned_unmet.len(), 1);
        assert!(
            report.abandoned_unmet[0].detail.contains("no verdict"),
            "unmet detail must carry the hang framing: {}",
            report.abandoned_unmet[0].detail
        );
        let log = source.log_entries.lock().await;
        assert!(log.iter().any(|(_, a)| a == "acceptance_timeout"));
        drop(log);
    }

    /// The other half of "judged by its own evidence": while the check
    /// hangs, steps that keep producing NOVEL successful side-effect
    /// evidence never charge at all — the item outlives its fruitless
    /// allowance for as long as the work is really moving, which is exactly
    /// the qwen shape (proved tests passing inside the very step whose
    /// check hung, then abandoned "fruitless" while passing).
    #[tokio::test]
    async fn a_hanging_check_never_outweighs_novel_evidence() {
        let dir = tempfile::tempdir().unwrap();
        let source = MemorySource::default();
        #[cfg(windows)]
        let hang = "ping -n 30 127.0.0.1";
        #[cfg(not(windows))]
        let hang = "sleep 30";
        source
            .push(step(
                1,
                "passing but unverifiable",
                Some(AcceptanceCheck::Command {
                    command: hang.to_string(),
                    timeout_secs: Some(1),
                }),
            ))
            .await;
        let working = |i: usize| {
            let mut out = outcome("ran another test");
            out.tool_calls = vec![StepToolCall {
                name: "exec".to_string(),
                input_digest: format!("test-{i}"),
                output_digest: format!("exit 0 #{i}"),
                success: true,
            }];
            Ok(out)
        };
        let runner = ScriptedRunner::new((0..3).map(working).collect());
        let config = LongHorizonConfig {
            max_steps_per_item: 2,
            max_replans_per_item: 0,
            ..LongHorizonConfig::default()
        };
        let report = LongHorizonRunner::new(config)
            .run("goal", &source, &runner, dir.path(), None)
            .await;

        // All 3 scripted steps ran (> the 2-step allowance): novel evidence
        // kept replenishing under the silent check; the item only died when
        // the script ran dry (runner-error containment), not as fruitless.
        assert_eq!(report.steps_taken, 3, "{report:?}");
        // 3 post-step timeouts plus one at the drain sweep. A runner-error
        // abandonment used to carry NO check onward, so the environment
        // never got a last word on an item whose steps had all succeeded;
        // it does now, and here it stays silent, so the verdict is still
        // unknown rather than a failure.
        assert_eq!(report.acceptance_unknown, 4, "{report:?}");
        assert_eq!(report.items_abandoned, 1);
        assert_eq!(
            report.abandoned_unmet.len(),
            1,
            "the walked-away item is NAMED, with the hang as its evidence: {report:?}"
        );
        assert!(
            report.abandoned_unverifiable.is_empty(),
            "it has a check, so the sweep speaks for it: {report:?}"
        );
        let reasons = source.abandon_reasons.lock().await;
        assert!(
            reasons.iter().all(|(_, r)| r.contains("runner errors")),
            "must die of script exhaustion, never of fruitlessness: {reasons:?}"
        );
        drop(reasons);
    }

    /// Once a check has consumed its ENTIRE ceiling without answering,
    /// re-runs are capped at the run's measured work cost — the cap is
    /// floored at 1s, honors the configured ceiling as an upper bound, and
    /// the verdict announces that the leash was shortened.
    #[tokio::test]
    async fn hang_restake_cap_shortens_the_timeout_and_announces() {
        let dir = tempfile::tempdir().unwrap();
        #[cfg(windows)]
        let hang = "ping -n 30 127.0.0.1";
        #[cfg(not(windows))]
        let hang = "sleep 30";
        // Configured allowance is the 120s default — a re-stake at the
        // measured cost (here ~zero, floored to 1s) must come back in
        // seconds, not minutes.
        let check = AcceptanceCheck::Command {
            command: hang.to_string(),
            timeout_secs: None,
        };
        let started = std::time::Instant::now();
        let verdict = check
            .run_with_timeout_cap(dir.path(), Some(Duration::ZERO))
            .await;
        assert!(verdict.timed_out);
        assert!(!verdict.passed);
        assert!(
            started.elapsed() < Duration::from_secs(20),
            "the cap must bound the run: {:?}",
            started.elapsed()
        );
        assert!(
            verdict.detail.contains("re-run capped"),
            "a shortened leash must announce itself: {}",
            verdict.detail
        );
        // Without a cap the configured/default ceiling stands — the detail
        // then carries no cap marker (checked via the fast 1s-configured
        // path so the test stays quick).
        let quick = AcceptanceCheck::Command {
            command: hang.to_string(),
            timeout_secs: Some(1),
        };
        let verdict = quick.run(dir.path()).await;
        assert!(verdict.timed_out);
        assert!(
            !verdict.detail.contains("re-run capped"),
            "an uncapped timeout must not claim it was capped: {}",
            verdict.detail
        );
    }

    /// Writes `flip.txt` (the abandoned sibling's done-condition) on its 4th
    /// step and reports the touch, while its own check never passes.
    struct FlipRunner {
        dir: PathBuf,
        calls: Mutex<usize>,
    }

    #[async_trait::async_trait]
    impl StepRunner for FlipRunner {
        async fn run_step(&self, _request: StepRequest) -> Result<StepOutcome, String> {
            let mut calls = self.calls.lock().await;
            *calls += 1;
            let mut out = outcome("kept working");
            if *calls == 4 {
                std::fs::write(self.dir.join("flip.txt"), b"now exists").unwrap();
                out.touched_paths = vec!["flip.txt".to_string()];
            }
            drop(calls);
            Ok(out)
        }
    }

    /// A check flipping fail→pass ANYWHERE is a verified environment change:
    /// the mid-run sweep revives the abandoned sibling whose condition now
    /// passes, and the flip replenishes the grinding item's fruitless budget
    /// — even though that item's own check keeps failing identically.
    #[tokio::test]
    async fn check_flip_replenishes_open_items_and_revives_mid_run() {
        let dir = tempfile::tempdir().unwrap();
        let source = MemorySource::default();
        source
            .push(step(
                1,
                "first",
                Some(AcceptanceCheck::FileExists {
                    path: "flip.txt".to_string(),
                }),
            ))
            .await;
        source
            .push(step(
                2,
                "second",
                Some(AcceptanceCheck::FileExists {
                    path: "never.txt".to_string(),
                }),
            ))
            .await;
        let runner = FlipRunner {
            dir: dir.path().to_path_buf(),
            calls: Mutex::new(0),
        };
        let config = LongHorizonConfig {
            max_steps_per_item: 2,
            max_replans_per_item: 0,
            ..LongHorizonConfig::default()
        };
        let report = LongHorizonRunner::new(config)
            .run("goal", &source, &runner, dir.path(), None)
            .await;

        // Item 1: 2 fruitless steps → abandoned. Item 2: 2 charged steps,
        // then its 4th call writes flip.txt → the sweep revives item 1
        // (completing it verified through the precheck door, no step spent)
        // and the flip resets item 2's counter, buying 2 more steps before
        // it abandons: 6 steps total, versus 4 without the replenish.
        assert_eq!(report.steps_taken, 6, "{report:?}");
        assert_eq!(report.items_revived, 1);
        assert_eq!(report.items_completed, 1);
        assert_eq!(report.items_already_satisfied, 1);
        assert_eq!(report.items_abandoned, 1);
        // The revival's verdict is knowledge on the report.
        assert_eq!(report.verified_outcomes.len(), 1);
        assert_eq!(report.verified_outcomes[0].id, 1);
        assert!(report.verified_outcomes[0].already_satisfied);
    }

    /// The success mirror of the novel-failure rule: a step whose successful
    /// side-effectful tool evidence is NEW replenishes the budget even while
    /// the check fails identically — the item grinds only when it stops
    /// producing new evidence, not after a fixed count.
    #[tokio::test]
    async fn novel_success_evidence_replenishes_the_fruitless_budget() {
        let dir = tempfile::tempdir().unwrap();
        let source = MemorySource::default();
        source
            .push(step(
                1,
                "climbing item",
                Some(AcceptanceCheck::FileExists {
                    path: "never.txt".to_string(),
                }),
            ))
            .await;
        let working = |i: usize| {
            let mut out = outcome("attempt");
            out.tool_calls = vec![StepToolCall {
                name: "exec".to_string(),
                input_digest: format!("cmd-{i}"),
                output_digest: format!("result-{i}"),
                success: true,
            }];
            Ok(out)
        };
        let runner =
            ScriptedRunner::new((0..6).map(working).collect());
        let config = LongHorizonConfig {
            max_steps_per_item: 2,
            max_replans_per_item: 0,
            ..LongHorizonConfig::default()
        };
        let report = LongHorizonRunner::new(config)
            .run("goal", &source, &runner, dir.path(), None)
            .await;

        // All 6 scripted steps ran (vs 2 without replenishment); the item
        // only died when the script ran dry (runner-error containment).
        assert_eq!(report.steps_taken, 6, "{report:?}");
        assert_eq!(report.items_abandoned, 1);
    }

    /// Zero-tool-call degenerate-loop steps ride the harness steering ladder
    /// (three escalating rungs, charged nothing) and only start consuming
    /// the fruitless budget once the ladder is exhausted.
    #[tokio::test]
    async fn degenerate_loop_steps_ride_the_ladder_before_charging() {
        let dir = tempfile::tempdir().unwrap();
        let source = MemorySource::default();
        source
            .push(step(
                1,
                "narrator",
                Some(AcceptanceCheck::FileExists {
                    path: "never.txt".to_string(),
                }),
            ))
            .await;
        let narration = || {
            let mut out = outcome("I would now call read_file and then fix things.");
            out.degenerate_loop = true;
            Ok(out)
        };
        let runner = ScriptedRunner::new((0..6).map(|_| narration()).collect());
        let config = LongHorizonConfig {
            max_steps_per_item: 2,
            max_replans_per_item: 0,
            ..LongHorizonConfig::default()
        };
        let report = LongHorizonRunner::new(config)
            .run("goal", &source, &runner, dir.path(), None)
            .await;

        // 3 ladder steps (uncharged) + 2 charged = 5 steps, then abandonment
        // — versus 2 steps if aborts were charged like genuine no-ops.
        assert_eq!(report.steps_taken, 5, "{report:?}");
        assert_eq!(report.items_abandoned, 1);
        let requests = runner.requests.lock().await;
        assert!(
            requests[1].prompt.contains("produced only narration"),
            "first rung must steer gently: {}",
            requests[1].prompt
        );
        assert!(
            requests[3].prompt.contains("STOP NARRATING"),
            "third rung must be urgent: {}",
            requests[3].prompt
        );
        drop(requests);
        let log = source.log_entries.lock().await;
        assert!(log.iter().any(|(_, a)| a == "narration_step"));
        drop(log);
    }

    /// An already-satisfied pre-check completion is knowledge: the passing
    /// verdict rides the report's `verified_outcomes` for the continuation
    /// planner, flagged as closed-by-evidence with zero steps run.
    #[tokio::test]
    async fn already_satisfied_precheck_records_a_verified_outcome() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("done.txt"), b"present").unwrap();
        let source = MemorySource::default();
        source
            .push(step(
                7,
                "already done",
                Some(AcceptanceCheck::FileExists {
                    path: "done.txt".to_string(),
                }),
            ))
            .await;
        let runner = ScriptedRunner::new(vec![]);
        let config = LongHorizonConfig {
            precheck_acceptance_items: std::iter::once(7).collect(),
            ..fast_config()
        };
        let report = LongHorizonRunner::new(config)
            .run("goal", &source, &runner, dir.path(), None)
            .await;

        assert_eq!(report.items_already_satisfied, 1);
        assert_eq!(report.verified_outcomes.len(), 1);
        let outcome = &report.verified_outcomes[0];
        assert_eq!(outcome.id, 7);
        assert!(outcome.already_satisfied);
        assert!(
            outcome.detail.contains("file exists"),
            "the verdict detail is the knowledge: {}",
            outcome.detail
        );
    }
}
