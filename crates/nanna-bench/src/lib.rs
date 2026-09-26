#![warn(clippy::pedantic, clippy::nursery, clippy::all)]
//! Unified benchmark harness for Nanna's six performance suites.
//!
//! Performance is a **gate**: a change ships only when a reproducible benchmark
//! holds or improves the budget recorded in `bench/BASELINE.md`. This crate
//! owns the suite taxonomy, deterministic fixtures, and (progressively) the
//! criterion bodies that replace the ad-hoc `nanna-gpu` benches.
//!
//! Suites (roadmap order):
//! 1. **Inference** — TTFT, tok/s, VRAM (P12 / Mummu)
//! 2. **Vector search** — SIMD vs GPU cosine (existing `nanna-gpu` benches)
//! 3. **Dreaming** — compression + recall retention (`nanna-memory::retention`)
//! 4. **Agent loop** — task-success @ tokens (long-horizon harness)
//! 5. **Guardrails** — syntax gates, breaker latency
//! 6. **Efficiency** — tokens and wall-clock per unit of work

/// One of the six benchmark suites that gate a perf-affecting change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Suite {
	/// Local-model decode: TTFT, tok/s, VRAM ceiling.
	Inference,
	/// Cosine similarity / k-NN: SIMD default path vs GPU at scale.
	VectorSearch,
	/// Dreaming & compression: information retention across a dream cycle.
	Dreaming,
	/// Long-horizon agent loop: task success @ token budget.
	AgentLoop,
	/// Safety / correctness gates: syntax gates, breaker latency.
	Guardrails,
	/// Efficiency: tokens and wall-clock per unit of work.
	Efficiency,
}

impl Suite {
	/// Every suite, in roadmap order.
	pub const ALL: [Self; 6] = [
		Self::Inference,
		Self::VectorSearch,
		Self::Dreaming,
		Self::AgentLoop,
		Self::Guardrails,
		Self::Efficiency,
	];

	/// Stable identifier used as the criterion group name and the
	/// `bench/BASELINE.md` row key.
	#[must_use]
	pub const fn id(self) -> &'static str {
		match self {
			Self::Inference => "inference",
			Self::VectorSearch => "vector_search",
			Self::Dreaming => "dreaming",
			Self::AgentLoop => "agent_loop",
			Self::Guardrails => "guardrails",
			Self::Efficiency => "efficiency",
		}
	}

	/// Whether this suite has real benchmark bodies yet.
	#[must_use]
	pub const fn implemented(self) -> bool {
		matches!(self, Self::VectorSearch)
	}
}

/// Direction a measured value must satisfy relative to its budget threshold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Direction {
	/// Measured value must be >= budget (`min` retention / compression).
	Min,
	/// Measured value must be <= budget (`max` latency / cost).
	Max,
	/// Measured value must equal budget exactly (correctness fixtures).
	Exact,
}

/// One machine-readable performance budget row from `bench/budgets.toml`.
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Budget {
	/// Stable key, dotted as `<suite>.<metric>`.
	pub id: String,
	/// Suite id matching [`Suite::id`].
	pub suite: String,
	/// Measured quantity name.
	pub metric: String,
	/// How the measured value is compared to [`Budget::value`].
	pub direction: Direction,
	/// Threshold the measurement must hold or improve.
	pub value: f64,
	/// Test or bench that produces the number.
	pub source: String,
	/// One-line context; full rationale lives in `bench/BASELINE.md`.
	#[serde(default)]
	pub note: String,
}

/// Parsed contents of `bench/budgets.toml`.
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct BudgetFile {
	/// Schema version.
	pub version: u32,
	/// Path to the human-readable baseline document.
	pub baseline_doc: String,
	/// Every committed budget row.
	#[serde(default)]
	pub budget: Vec<Budget>,
}

impl BudgetFile {
	/// Parse a budgets.toml document from its text contents.
	///
	/// # Errors
	///
	/// Returns the toml deserialize error when the document is malformed.
	pub fn parse(text: &str) -> Result<Self, toml::de::Error> {
		toml::from_str(text)
	}

	/// Load and parse `bench/budgets.toml` relative to the workspace root.
	///
	/// Walks up from `CARGO_MANIFEST_DIR` so the test works whether cargo is
	/// invoked from the workspace root or the crate directory.
	///
	/// # Errors
	///
	/// Returns a string describing an I/O or parse failure.
	pub fn load_default() -> Result<Self, String> {
		let path = budgets_toml_path()?;
		let text = std::fs::read_to_string(&path)
			.map_err(|e| format!("read {}: {e}", path.display()))?;
		Self::parse(&text).map_err(|e| format!("parse {}: {e}", path.display()))
	}

	/// Every budget id, in file order.
	#[must_use]
	pub fn ids(&self) -> Vec<&str> {
		self.budget.iter().map(|b| b.id.as_str()).collect()
	}

	/// Whether a budget with this exact id is present.
	#[must_use]
	pub fn contains_id(&self, id: &str) -> bool {
		self.budget.iter().any(|b| b.id == id)
	}
}

/// Resolve `bench/budgets.toml` from the crate manifest location.
fn budgets_toml_path() -> Result<std::path::PathBuf, String> {
	let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
	// crates/nanna-bench → workspace root
	let root = manifest_dir
		.parent()
		.and_then(|p| p.parent())
		.ok_or_else(|| "cannot locate workspace root from CARGO_MANIFEST_DIR".to_string())?;
	let path = root.join("bench").join("budgets.toml");
	if path.is_file() {
		Ok(path)
	} else {
		Err(format!("budgets.toml not found at {}", path.display()))
	}
}

/// Budget ids that must always be present — the machine-readable gate's
/// minimum surface. Adding a suite's first real baseline extends this list.
pub const REQUIRED_BUDGET_IDS: &[&str] = &[
	"dreaming.compression",
	"dreaming.recall_retention",
	"dreaming.summarizer_calls",
	"agent_loop.false_success_admitted",
	"agent_loop.drift_containment_tokens",
	"vector_search.simd_batch_1k_p95_ms",
	"vector_search.simd_batch_10k_p95_ms",
	"vector_search.simd_batch_50k_p95_ms",
];

/// Deterministic pseudo-random vectors for benchmark fixtures.
///
/// `SplitMix64`, matching the generator used by the memory retention harness,
/// so fixtures are reproducible across machines and across the Suite 3
/// retention tests.
#[must_use]
pub fn fixture_vectors(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
	let mut state = seed;
	let mut next = || {
		state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
		let mut z = state;
		z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
		z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
		z ^ (z >> 31)
	};

	(0..count)
		.map(|_| {
			(0..dim)
				.map(|_| {
					// 24 bits over 2^24 is [0, 1). The old divisor, 2^8 × 96 =
					// 24 576, gave [0, 683): nearly every component positive
					// and large, so the "random" vectors were near-parallel.
					let unit = top_24_bits_as_f32(next()) / 16_777_216.0;
					unit.mul_add(2.0, -1.0)
				})
				.collect()
		})
		.collect()
}

/// The top 24 bits of a draw (`draw >> 40`) as the `f32` that denotes them.
///
/// Every integer below 2^24 is exactly representable in an `f32`, but no
/// `From` conversion takes a 24-bit integer there. Split into its high 16 and
/// low 8 bits, both halves have lossless `From` conversions, and recombining
/// them lands on an integer below 2^24 again, so no step rounds: the result is
/// bit-identical to the plain cast.
fn top_24_bits_as_f32(draw: u64) -> f32 {
	let [high, middle, low, ..] = draw.to_be_bytes();
	f32::from(u16::from_be_bytes([high, middle])).mul_add(256.0, f32::from(low))
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn suite_ids_are_unique() {
		let mut ids: Vec<&str> = Suite::ALL.iter().map(|s| s.id()).collect();
		ids.sort_unstable();
		let len = ids.len();
		ids.dedup();
		assert_eq!(ids.len(), len);
	}

	#[test]
	fn fixture_components_are_in_the_unit_interval() {
		let vectors = fixture_vectors(64, 32, 7);
		assert!(
			vectors.iter().flatten().all(|x| (-1.0..1.0).contains(x)),
			"components must lie in [-1, 1)"
		);
		let negative = vectors.iter().flatten().filter(|x| **x < 0.0).count();
		assert!(negative > 64 * 32 / 4, "roughly half the components are negative: {negative}");
	}

	#[test]
	fn fixtures_are_deterministic() {
		let a = fixture_vectors(4, 8, 42);
		let b = fixture_vectors(4, 8, 42);
		assert_eq!(a, b);
		assert_eq!(a.len(), 4);
		assert_eq!(a[0].len(), 8);
	}

	#[test]
	fn top_24_bits_convert_exactly() {
		// `f64::from(u32)` is lossless, so it is an exact reference for the
		// 24-bit value the draw's top bits denote.
		for draw in [0, 1 << 40, (1 << 40) - 1, u64::MAX, 0xFFFF_FF00_0000_0000, 0x9E37_79B9_7F4A_7C15] {
			let expected = u32::try_from(draw >> 40).expect("a 24-bit value fits u32");
			assert_eq!(f64::from(top_24_bits_as_f32(draw)).to_bits(), f64::from(expected).to_bits(), "draw {draw:#x}");
		}
	}
}
