# Benchmark baselines

The committed baseline the daily-dev routine diffs against. A perf-affecting change
ships only when the relevant number here holds or improves (see the `daily-dev` skill,
Appendix B for methodology, suites, and the reference hardware tier). Update a number
only on a legitimate, measured improvement, and cite the commit.

**Governing metric — task success @ budget.** The agent-eval suite denominator (which
live/harness evals count, how pass rate is scored, reference tiers) is defined in
[`AGENT_EVAL.md`](./AGENT_EVAL.md). Machine-readable budget rows: [`budgets.toml`](./budgets.toml).

Reference tier (name it in every number): RTX 4070 Ti SUPER 16 GB (Vulkan/wgpu) +
AMD Zen 4 (AVX-512). Numbers without a hardware note are deterministic, hardware-independent
harness results (fixed-seed synthetic corpora), reproducible on any host.

---

## Suite 3 — Dreaming & compression (information retention)

Instrument: `nanna-memory::retention` (recall before/after a dream cycle). Deterministic,
offline, fixed-seed — these are exact, reproducible values, not timing samples. Run with
`cargo test -p nanna-memory retention`.

| Metric | Baseline | Source | Notes |
| --- | --- | --- | --- |
| Consolidation compression ratio | **0.90** | `retention::tests::dreaming_shrinks_store_while_holding_recall` | 60 → 6 memories |
| Recall retention across a dream cycle | **1.000** (recall@3 1.0 → 1.0) | same test | same-topic merges keep every topic reachable at its centroid |
| **Summarizer calls per cycle** (this corpus) | **0** (was 6) | same test | *(2026-07-23)* dream phase (b) folds the 54 removable memories deterministically — `memories_deduped: 54`, `clusters_formed: 0` |
| w20 aged-recall (FSRS-6 `0.0658`) | **6/6 topics** | `retention::tests::w20_experiment_aged_recall` | 800-day-aged corpus, FSRS-gated recall |
| w20 aged-recall (FSRS-5 `0.5`, the old default) | **0/6 topics** | same test | evidence the shipped constant was wrong; default flipped 2026-07-17 |

| Timeline fold compression (P25 DSP step 2) | **0.76** (24 of 100 events kept) | `nanna_timeline::compress::tests::budget_gate_timeline_fold` | *(2026-09-26)* fixed-seed `SplitMix64` card series at `DREAM_FOLD_BUDGET` (24); model-free, so exact |
| Timeline fold transition retention | **1.0** (10/10 outcomes kept) | same test | a transition is never decimated while the budget can hold it |

Budget: consolidation must not regress **recall retention below 1.0** on this fixed corpus,
and must hold **compression ≥ 0.90**. The timeline fold must hold **compression ≥ 0.75** and
**keep every transition** on its corpus. The w20 rows are a correctness fixture (they assert the
FSRS-6 exponent strictly out-recalls the old FSRS-5 one on aged memories), not a tunable budget.

### Summarization drift (content fidelity, not recall)

Instrument: `nanna-memory::retention::clause_survives` + the two `drift` fixtures. Measures
whether a **rare, safety-critical clause** survives a dream cycle — deliberately *not* a recall
metric, because the two come apart exactly here: the topic stays perfectly recallable while the
one clause that made it actionable is gone. That gap is what makes drift easy to ship blind.

| Metric | Baseline | Source | Notes |
| --- | --- | --- | --- |
| Rare clause survives a **summarized** cluster | **NO** (lost in 1 cycle) | `retention::tests::summarized_memories_lose_a_rare_critical_clause_in_one_cycle` | the known exposure, reproduced against our own pipeline |
| Rare clause survives a **deduplicated** band | **YES** (verbatim) | `retention::tests::deduplicated_memories_keep_their_rare_critical_clause` | dream phase (b) folds without paraphrasing; store still compresses |
| Rare clause survives a **user-STATED** cluster | **YES** (verbatim) | `retention::tests::a_user_stated_clause_survives_the_summarizing_cycle` | mitigation (c), 2026-08-21: same corpus/spread/summarizer as the losing row above — only the provenance differs |
| Stated provenance survives a **dedup fold** | **YES** | `retention::tests::a_dedup_fold_never_launders_stated_provenance_into_observed` | a fold keeps the *survivor's* metadata, so folding stated→observed used to launder the pin away |
| Rare clause survives a band of **already-summarized** rows | **YES** (verbatim) | `retention::tests::a_summary_is_never_summarized_again` | mitigation (b), 2026-08-21: same corpus/spread/summarizer as the losing row above — only `generation` differs |

Budget: the **dedup arm must never regress to NO** — that is a correctness fixture, and it is the
concrete payoff of folding restatements deterministically. The summarized arm is a **baseline to
beat, not a budget to hold**: it currently asserts the clause *is* lost, so when a drift mitigation
lands (generation ceiling, or verbatim-pinning user-STATED memories — both logged in P13) that test
will fail loudly and should be flipped to assert survival. A failing baseline arm means the system
got better, and the assertion says so in its message.

*(2026-08-21)* **Mitigation (c) landed — verbatim-pinning user-STATED memories.** Two rows added
above. The summarized arm deliberately **stays at NO**: the exposure is real and unfixed for
agent-*observed* content, and deleting the measurement that says so would be the dishonest move.
What changed is that a memory whose `fact_type` provenance is `stated` is now split out of its band
*before* both the dedup fold and the clusterer, so it is never handed to a summarizer. The split
runs before the fold, not after, because a fold keeps the **survivor's** metadata — folding a stated
row into an observed one handed back an `observed` row holding a user assertion, which the next
cycle was then free to paraphrase. Both new rows fail (and name the laundered id) with the split
removed, so neither is vacuous. Cost is one hash lookup per memory per band and strictly *fewer*
summarizer calls; the price is that pinned rows no longer compress by paraphrase — they still
deduplicate, so repetition cannot make them grow without bound.

*(2026-08-21)* **Mitigation (b) landed too — "compress a session, never re-compress a summary".**
The row above is the second controlled pair against the same losing baseline: identical corpus,
spread, summarizer and config, differing only in `FsrsState::generation`. This is the bound that
needed no number chosen for it. "How many summarization passes are safe?" has no derivable answer —
the baseline arm loses a rare clause in **one** — so the class is forbidden rather than counted, and
`generation` (already `max(sources) + 1` at every consolidation) records membership of that class
exactly. Gists are exempt from the **clusterer**, not from the cycle: they still go through the
lossless dedup fold, so two gists that restate each other still collapse and the store still
compresses (`retention::tests::gists_that_restate_each_other_still_fold`). `generation` is now also
monotone across a fold — absorbing a gist makes the survivor a gist-carrier — or a generation-1 row
folding into a generation-0 one would launder itself straight back into the summarizer's input
(`service::tests::a_fold_that_absorbs_a_gist_marks_the_survivor_as_one`, which fails with that one
line removed). Cost: one `u32` compare per memory per band, and strictly fewer summarizer calls. The
price is that a gist's footprint no longer shrinks by paraphrase — a bounded-drift-for-bounded-
footprint trade, taken deliberately.

*(2026-07-23)* The summarizer-call row is the newest budget and the point of dream phase (b):
this corpus's within-topic pairs sit in the `IngestAction::Reinforce` band (cosine > 0.92), so
they are true restatements and are now folded **deterministically, with zero LLM calls**, where
the cycle previously paid 6 summarizer prompts to paraphrase them. Compression and recall are
*unchanged* (0.90 / 1.000) — the win is entirely in the token budget, the governing scarce
resource on the single-GPU local tier. Do not "fix" a future regression here by lowering the
similarity threshold: the 0.92 line is the project's existing same-fact definition, and folding
below it would paraphrase-merge genuinely distinct memories.

---

## Suite 3b — Dream-time clustering (scaling)

Instrument: `cargo bench -p nanna-memory --bench clustering_scaling`. Deterministic
fixed-seed corpus (integer hash / xorshift, not an RNG), MiniLM-width 384-dim vectors,
at the shipped `cluster_threshold` (0.75). The **`pairs` column is hardware-independent**
— it counts the inner-loop iterations the greedy pass actually performs — so it is the
number an ANN replacement must be held to even off the reference tier. Wall-clock is
reference-tier (AMD Zen 4).

*Dense* = N/20 tight topics, so clusters fill. *Sparse* = mutually unrelated unit vectors
(expected cosine 0, sd ~1/√384 ≈ 0.051), so nothing clusters.

| N | clusters (dense) | pairs (dense) | wall_ms (dense) | clusters (sparse) | pairs (sparse) | wall_ms (sparse) |
| --- | --- | --- | --- | --- | --- | --- |
| 1,000 | 50 | 25,450 | 1.7 | 1,000 | 499,500 | 14.5 |
| 2,000 | 100 | 100,900 | 4.7 | 2,000 | 1,999,000 | 58.0 |
| 4,000 | 195 | 381,410 | 15.5 | 4,000 | 7,998,000 | 223.1 |
| 8,000 | 387 | 1,478,686 | 52.4 | 8,000 | 31,996,000 | 930.3 |
| 16,000 | 796 | 6,114,418 | 255.2 | 16,000 | **127,992,000** | **4,281.2** |

*(2026-09-09)* **The sparse arm is exactly N(N−1)/2** — 127,992,000 at N=16,000 — i.e.
every pair compared, every memory its own cluster. That is the literal O(N²) the
indexed-clustering item names, reproduced. The dense arm is **N^2.0 in pairs but bounded
in practice** by `max_cluster_memories` (64): a seed stops scanning once its cluster fills,
which is why it stays 21× cheaper at the same N.

Extrapolating the sparse arm (35 ns/pair): **50k ≈ 1.25B pairs / ~44 s**, **200k ≈ 20B /
~12 min**, **500k ≈ 125B / ~73 min**. A dream cycle cannot take an hour. Benchmark the ANN
work on the **sparse** arm — the dense arm is bounded by the cluster cap and cannot show a
win no matter how good the index is.

The sparse regime is also the realistic one for a long-lived store: what survives
consolidation is precisely the memories that did *not* merge into anything.

### 2026-09-20 — norm hoisting: 11-18% off the pass, and a useful negative result

`cluster_memories` now computes each memory's L2 norm **once** and gives the pair loop a
single dot product against two cached scalars, instead of calling `cosine_similarity_f32`,
which re-derives both magnitudes on every pair (it accumulates three FMAs per element:
`a·b`, `a·a`, `b·b`). Bit-identical results — the two SIMD kernels walk the same chunks in
the same order, so this is an identity, not a tolerance, and
`norm_hoisting_is_bit_identical_to_the_cosine_kernel` asserts exact `f32` equality across
seven widths. The `pairs` column is unchanged at every N, which is the other half of the
proof that nothing about the clustering decision moved.

Same host, same fixture, same seed; two independent runs after the change, both agreeing:

| N (sparse) | wall_ms before | wall_ms after (2 runs) | delta |
| --- | --- | --- | --- |
| 1,000 | 14.2 | 12.2 / 11.2 | **-17.6%** |
| 2,000 | 54.2 | 47.1 / 47.8 | **-12.5%** |
| 4,000 | 224.2 | 194.3 / 193.9 | **-13.4%** |
| 8,000 | 916.8 | 794.6 / 788.4 | **-13.7%** |
| 16,000 | 4,206.6 | 3,756.4 / 3,703.3 | **-11.3%** |

Dense arm moves too but less and noisily (226.9 → 203.2 / 206.1 at 16k, ~-10%); it has far
fewer pairs, so read the sparse arm.

**The negative result is the more useful half, and it contradicts the estimate that
motivated the change.** Removing two thirds of the FMAs was expected to be worth roughly
2-3x. It is worth 13%. The cosine kernel here is **memory-bandwidth bound, not
FMA bound** — each pair streams two 384-float vectors (~3 KB) and a wide out-of-order core
absorbs the extra multiply-adds nearly for free. So: **do not chase arithmetic in this
loop.** Vectorising harder, fusing, or hand-tuning the kernel will buy single-digit
percentages against a quadratic, and the ceiling is not moved by any of it.

Concretely, the extrapolation barely improves — 35 ns/pair → 29 ns/pair means **500k goes
from ~73 min to ~60 min**. A dream cycle still cannot take an hour. The only lever that
changes the shape is **comparing fewer pairs**, which is exactly what the indexed-clustering
item proposes and exactly what this measurement now supports with a number rather than an
assumption.

### 2026-09-20 — an `aged` arm, because the other two run at a timescale production never uses

The dense and sparse arms space memories one second apart while leaving
`time_span_minutes` at its 1440-minute default, so `age_proximity` never leaves the top of
its range. `MemoryService::with_store_timescale` does not do that: it sets the span to the
store's own oldest-to-newest gap, which is what makes a mature store discriminate at week
scale rather than minute scale. The two arms also use `FsrsState::default()` for every
memory, which pins `access_count` at 0 and `importance` at 1.0 — so `recall_affinity` and
`importance_proximity` are identically 1.0 *by construction*, the exact degeneracy the
2026-09-09 similarity-veto bug turned on.

The new **aged** arm keeps the sparse vectors and fixes both: 90 days of spread, the
production timescale rule, and FSRS state that varies. Same N, same vectors, same `pairs`.

### 2026-09-20 — pruning pairs before the cosine: 16-22%, and it is lossless by construction

`score_or_prune` computes the three scalar terms first and skips the cosine when the pair's
**ceiling** — the score it would get with a perfect cosine of 1.0 — is already below
`cluster_threshold`. Each cheap term is bounded above by 1.0 by construction, so the ceiling
is a true upper bound and the skipped pairs are exactly the pairs that would have been
rejected. Not a heuristic and not a tolerance: `pruning_never_changes_a_cluster` clusters a
randomized corpus with and without it and requires identical output, plus exact `f32`
equality on every pair that is *not* pruned.

On the aged arm it rejects **21.9% of all pairs** before touching an embedding. A/B on the
identical corpus through the identical code path (the prune disabled with a `black_box`
false so nothing else changes), **two samples each**:

| N (aged) | no prune | with prune | delta |
| --- | --- | --- | --- |
| 1,000 | 12.7 / 13.7 | 11.4 / 10.4 | **-17.4%** |
| 2,000 | 48.6 / 54.6 | 43.4 / 43.4 | **-15.9%** |
| 4,000 | 198.3 / 211.5 | 174.2 / 171.4 | **-15.7%** |
| 8,000 | 783.4 / 866.4 | 704.0 / 676.5 | **-16.3%** |
| 16,000 | 4,111.7 / 3,865.5 | 3,291.2 / 2,943.0 | **-21.9%** |

**How much it saves is data-dependent, and the honest version of that is: it does nothing
on the dense and sparse arms.** There every cheap term sits at 1.0, the ceiling never falls
below the threshold, and the prune cannot fire — which is why the aged arm had to exist
before this could be measured at all. A store written in one sitting gets nothing; a
long-lived store, which the roadmap calls the realistic case, gets a fifth of its pairs back.

**It does not change the conclusion above.** 21.9% fewer pairs is a constant factor on a
quadratic: 500k goes from ~60 min to roughly ~47 min. Still not a dream cycle. The two
2026-09-20 passes together are the argument for the index — they took every cheap win
available without one (a third of the arithmetic, a fifth of the pairs) and the wall is
still there.

> **Two earlier readings of this table were wrong, and both for the same reason — the
> configuration underneath moved.** The first (pre-fix `cluster_threshold` 0.45) showed
> `pairs` growing *linearly*, because the non-semantic floor (0.50) sat above the threshold
> so every pair matched immediately, every cluster filled to 64 at once, and each seed broke
> out after ~64 iterations. The second (threshold 0.55) showed N^1.45/N^1.31. The numbers
> above are at the shipped 0.75. A benchmark is only as honest as the configuration beneath
> it, and this one moved twice in one day.

---

## Suite 3c — Clustering threshold: compression vs fidelity

Instrument: `cargo bench -p nanna-memory --bench clustering_threshold_sweep`. Deterministic,
offline, fixed-seed; echo summarizer. **`min cosine` is the axis to compare against
published numbers**, not `threshold` — the threshold is judged against the *composite*
score, and `min_required_similarity()` converts between them.

*Loose corpus* (`member_spread` 0.6): related but not near-identical, so clustering decides.

| threshold | min cosine | clusters | merged | compression | recall retention |
| --- | --- | --- | --- | --- | --- |
| 0.55 | 0.10 | 3 | 27 | 0.450 | 1.000 |
| 0.65 | 0.30 | 3 | 27 | 0.450 | 1.000 |
| **0.75** (shipped) | **0.50** | 3 | 27 | 0.450 | 1.000 |
| 0.85 | 0.70 | 5 | 23 | 0.383 | 1.000 |
| 0.95 | 0.90 | 0 | 0 | 0.000 | 1.000 |

*Tight corpus* (`member_spread` 0.02) is the **control** and is flat at every threshold:
0 clusters, 54 deduped, compression 0.900, recall 1.000. Its members sit above the
`Reinforce` line (0.92) so dream phase (b) folds them and clustering never runs. Movement
there would mean the clustering threshold was reaching pairs dedup already owns.

*(2026-09-09)* **0.55 → 0.75 is free**: identical clusters, merges, compression and recall,
while the cosine actually demanded rises 0.10 → 0.50. The shipped default takes the top of
that flat range — a 5× stricter semantic bar at zero measured cost. Above it the trade is
real and now priced: **0.85 (cosine 0.70, the θ_sim the 2026 literature uses) costs
compression 0.450 → 0.383, a 15% relative drop, at recall 1.000.** Recall never moves
anywhere in this sweep, so the trade is compression against merge *precision*, not against
retrievability. Whether to spend it is an owner call; the number is no longer the unknown.

---

## Suite 4 — Long-horizon harness (task-success @ tokens)

Denominator (which evals count, tiers, pass-rate rules): [`AGENT_EVAL.md`](./AGENT_EVAL.md).
Instrument: `nanna-agent::harness` — the P14 control loop driven by scripted step runners
and an in-memory task source. Deterministic, offline, model-free: these rows measure the
*harness* (acceptance gating, progress-or-replan, drift containment), which is the P14
design bet — long-horizon capability comes from the harness, not the model. Run with
`cargo test -p nanna-agent harness`.

| Metric | Baseline | Source | Notes |
| --- | --- | --- | --- |
| Task success (compliant scripted model) | **3/3 items** | `harness::tests::compliant_run_success_at_tokens_baseline` | 1 step/item at 1200 tokens |
| Tokens per completed item (compliant) | **1200** | same test | 3600 total / 3 items — the governing metric's bookkeeping is exact |
| False-success completions admitted | **0** | `harness::tests::false_success_claim_is_refuted_replanned_then_abandoned` | model claims TASK COMPLETE every step; env never changes; harness never records completion |
| Drift containment cost | **≤ 6000 tokens** (5 steps) | `harness::tests::drift_containment_cost_baseline` | perma-claiming model is replanned once then abandoned — grinding is bounded |
| Loop acceleration | **< 4 steps** to abandon | `harness::tests::repeated_tool_signatures_accelerate_the_stall_counter` | identical tool signatures double the stall counter |

Budget: **false-success completions admitted must stay 0** (correctness fixture — the
anti-drift keystone), and drift containment must stay **≤ 6000 tokens** on the fixed script.
The compliant-run rows are exact bookkeeping fixtures, not tunable budgets.

### Live run (2026-07-18, after harness tuning)

Instrument: `nanna-daemon/tests/live_long_horizon.rs` (`#[ignore]`d; needs Ollama). Five
minutes-scale tasks with real acceptance checks (regex-on-file ×3, command-exit-0 ×2, one
dependency edge) in a temp workspace, driven end-to-end: harness → fresh agent per step →
Ollama → real file/exec tools → harness-run verification. Run with
`NANNA_EVAL_MODEL=qwen3.5:9b cargo test -p nanna-daemon --test live_long_horizon -- --ignored --nocapture`.

| Metric | Value | Notes |
| --- | --- | --- |
| Task success | **5/5 (1.00)** | qwen3.5:9b (9.7B) via Ollama — RTX 4070 Ti SUPER 16 GB |
| Tokens per completed item | **22,564** | 109k in + 3.4k out over 6 steps |
| Wall clock | **72 s** (6 steps) | run ended `all_tasks_done` |
| Replans / abandonments | **0 / 0** | verdict feedback worked: task #4's first check failed, the fed-back verdict fixed it next step |
| False-success claims admitted | **0** | harness integrity held on a real model |
| Unverified completions | **0** | every task carried a machine check |
| Dependency ordering | ✅ | the depends_on pair completed in order (data file → row count) |

These are recorded datapoints, **not budgets yet** — they will move with model choice and task
set. The tuning trail is itself the evidence this suite exists to produce — each run caught a
real harness/production bug:

- **Run 1 — 0/5.** The model did every task correctly; every artifact landed in `$HOME`.
  Production bug: scripted tools loaded via `load_skills_with_services` never got the registry
  handle, so relative paths silently resolved to the home directory. Fixed.
- **Run 2 — 3/5 @ 129k tokens/item.** Both command-based checks were unwinnable: no bare `sh`
  on PATH ⇒ the acceptance runner silently fell back to `cmd.exe`, which cannot run
  `test`/`$(...)`. Also 3 consecutive Ollama 500s (model-side tool-call template corruption)
  tripped the error breaker mid-task. Fixed: acceptance commands route through Git Bash like
  the exec tool (regression-tested), and the step runner retries transient 5xx with a fresh
  re-anchored context.
- **Run 3 — 5/5 @ 22.6k tokens/item, 72 s.** Above.

### Endurance run — the "4-hour task" (2026-07-19)

Instrument: `live_endurance` in the same test file — build `minidb` (a POSIX-shell key-value
store CLI) against 42 dependency-chained fail-to-pass feature tests. qwen3.5:9b,
RTX 4070 Ti SUPER 16 GB, full healing stack (bash-routed acceptance, abort-as-error parsing,
5xx/empty retries with runner reset, poison containment, gated server-restart healing, no
per-step token cap).

| Metric | Value | Notes |
| --- | --- | --- |
| Wall clock, one plan | **6.00 h** (cap) | single seeded plan, worked continuously start to finish |
| Longest unbroken segment | **4 h 39 m** | one provider incident at t=81m, healed by server restart; segment 2 alone clears the 4-hour bar |
| Verified completions | **23** (14/42 seeded features + 9 model-created subtasks) | progress distributed across the entire window (t=2m → t=360m, still advancing at hour six) |
| Tokens | **5.13 M** over 137 steps | ~854k tokens/hour sustained on the local GPU |
| False-success claims admitted | **0** | across all six hours |
| Drift | none observed | at hour six the model was decomposing and fixing the append feature — on-plan work, not looping or wandering |

The tuning trail to get here (each run caught a real bug): run 1 — tool workdir plumbing
(`$HOME` writes); run 2 — cmd.exe acceptance fallback + Ollama tool-template 500s; runs 3–4 —
Ollama's degraded-runner state (aborted `done:false` generations parsed as empty successes —
fixed in nanna-llm) and item-level poison containment; run 5 — subtask `sort_order 0`
queue-jumping (task explosion); run 7 — the result above.

**Cloud variant — `openrouter/openrouter/free` (2026-07-20).** The same 42-feature ladder
driven through OpenRouter's free-model auto-router, where the serving model varies per
request — the harness must carry ALL continuity. Result: **33/42 features verified in 3.30 h,
one unbroken segment, 0 resumes, 0 false successes**, 97 steps, 3.36 M tokens
(~102 k/verified item), stop = `all_tasks_done` (plan drained: 33 verified + 12
abandoned-with-containment). Abandonments clustered where weak models happened to be serving
(even trivial features), while stronger draws later carried the ladder — per-request model
variance handled by design. Smoke on the same router: 5/5 @ 17 k tokens/item. Healing is
provider-aware: cloud incidents heal by pause+resume+retries only (local-server surgery is
gated to Ollama-served models via `ProviderId::from_model`).

**Small-model datapoint — `gemma4:e4b-it-qat` (2026-08-06).** First candidate in the
smaller-than-qwen search (6.1 GB QAT quant, effective-4B MatFormer). Reference 16 GB tier,
master `7f129c0f` (v0.3.4-beta.9 tree), fresh store, 4.5 h window, VRAM 7.2→9.5 GB across the
run (no ceiling pressure, zero `num_ctx` demotions, zero CUDA faults; `num_ctx` latch not
captured — eval log level was warn+registry only). Smoke first: **5/5 @ 22.7 k tokens/item,
79 s** — statistically identical to qwen3.5:9b's 22.6 k, all 28 tool calls registry-valid.
Endurance: **7/42 features verified in 4.50 h, 0 false successes, 1 resume**. The failure
mode is not dialect, drift, or infrastructure — it is **recursive self-decomposition**: the
model spawned 103 extra items against 42 seeded (subtasks of subtasks, three levels deep),
closed 108 items total, and spent the window on ceremony — it never advanced past feature ~10
of the dependency ladder, and 9 seeded features ended cancelled. Same harness, same machine,
qwen ships features directly where gemma splits them. Next lever: decomposition damping
(soft-nudge family, per the no-hard-cap directive) before writing this model off.

**Iteration 2 — decomposition damping (PR #194, 2026-08-07).** Same model, tier, and
window; `tasks.add` now returns escalating do-the-work notes on depth-2+ creation and
sibling overhang. Result: **5/42 verified in 4.50 h** — no improvement (iteration 1: 7/42;
the delta is run noise). The decisive observation: the notes fired 69 times (32 "STOP
SPLITTING", 36 "DO the work") and the model kept splitting — item creation barely moved
(133 vs 145). **In-band tool-result feedback does not alter this model's planning
behavior.** One CUDA fault at t=66m, contained by the first-fault reset ladder; 0
demotions. Next lever must be structural: scheduler-level deprioritization of depth-2+
items, which cannot affect qwen (it never creates them).

**Iteration 3 — depth-biased scheduling (2026-08-07): INVALIDATED by operations, 7/42.**
`next()` deprioritizes ladder depth >= 2 (commit 1665610c). The run was paused for the
owner's gaming session and the first two RESUME attempts wedged: the pre-warm loaded a
default-ctx model instance that starved the eval runner's VRAM sizing, demoting `num_ctx`
to 4096 — below the 4211-token minimum viable window — and the harness FAILED LOUDLY with
the full arithmetic instead of running truncated (the post-2026-08-03 machinery working as
designed). Those wedged segments mass-cancelled 26/42 seeded features before the fix
(launch resumes with the model UNLOADED), capping the achievable score at 16. Early
in-flight evidence before the pause was promising — item creation ran ~20-25% below runs
1-2 at every checkpoint — but the verified count cannot be attributed. Operational lessons
banked: resumes must start with the model unloaded (fresh launches may pre-warm), and the
viable-floor abort turned what would have been a silent 4-hour loss into a 3-minute loud
one, twice.

**Iteration 4 — depth-biased scheduling, clean run (2026-08-07): 15/42.** The first
genuine improvement: more than double the 7/42 baseline, on an uncontaminated full 4.5 h
window with zero CUDA faults, zero context demotions, and only 6 features cancelled. The
scheduler ranking depth-2+ scaffolding below the seeded ladder (commit 1665610c) is the
only active lever this run — the parent-refocus hint sat on the service door while eval
completions flow through the harness door (0 firings at 25 completions; moved to a note on
the parent in 1a0cebfb for iteration 5). Item creation ran ~20-25% below the un-damped
runs throughout. Progression: 7/42 baseline → 5/42 advisory notes (no effect, model
ignored 69 firings) → invalid (ops) → **15/42 structural scheduling**. Bar: qwen3.5:9b
32/42.

**Iteration 5 — full honest lever set (2026-08-07): 8/42.** Depth bias + parent-refocus
notes (1a0cebfb, confirmed planting: 21 notes by t=94m) + verified-parent hygiene. Clean
full window, zero faults, zero demotions — and a REGRESSION from iteration 4's 15/42. The
only configuration delta was the refocus notes; they did not convert (12 features ended
cancelled vs 6, hygiene never fired). Two readings, both honest: the notes may add step-frame
weight that hurts more than the redirect helps, and run-to-run variance on this model is
large (8-15 across two clean runs of near-identical config — a single run cannot rank
levers). **Campaign verdict:** structural scheduling is worth ~2x (7 → 15 best case);
advisory text is worth nothing; gemma4:e4b-it-qat sits at 8-15/42 against qwen3.5:9b's
32/42 with the honest harness. The remaining gap looks like model capability, not harness —
further levers risk shading into scoring the benchmark for it. Decision escalated to the
owner: accept, keep iterating, or try qwen3.5-4B as the small-model candidate.

**Model comparison on the merged path (2026-08-08, v0.3.5-beta.10, master
`914b19f1`).** The campaign's winning levers merged as the singular path (depth-biased
scheduling + service-door refocus + verified-parent hygiene; the harness-door refocus notes
reverted as a measured regression). One build, identical conditions per leg: model unloaded
at launch, fresh store, 4.5 h cap, reference 16 GB tier.

| Model | Smoke | Endurance | Wall clock | Exit |
|---|---|---|---|---|
| **ornith:9b** | 5/5 @ 11.0 k tok/item, 68 s | **37/42** | **1.02 h** | early — plan drained: 37 verified on a 50-item plan (only 8 self-created), 0 faults, 0 resumes. **All-time record** — first model past qwen on this ladder, at half the token cost |
| qwen3.5:9b | 5/5 @ 22.6 k tok/item (2026-07-18 row) | **25/42** | 2.11 h | early — plan drained at 100/101 closed: 25 verified, 17 abandoned-with-containment; 0 faults, 0 resumes |
| gemma4:e4b-it-qat | 5/5 @ 22.7 k tok/item (2026-08-06 row) | **6/42** | 4.50 h (cap) | full window; 118 items from 42 seeded, 97 closed (81 done + 16 cancelled), mostly self-created scaffolding; 0 faults, 0 resumes |
| ministral-3:8b | 4/5 @ 11.8 k tok/item, 66 s (miss = Ollama-side 500 `tool 'exec' not found`) | **2/42** | 3.20 h (early stop) | deterministic-failure stop: 3 consecutive segments ended by Ollama mid-response aborts after 2 successful heals (t=37m, t=141m); runaway decomposition — 256 items (199 self-created), 42 done + 32 cancelled; 639 steps, 30.3 M tokens |
| lfm2.5:latest | **2/5** @ 78.1 k tok/item, 124 s | — (smoke gate failed) | — | 2 tasks abandoned after fruitless steps + 1 replan each; up from 0-for-tools in the 2026-08-03 campaign — dialect fixes hold, capability doesn't |

**Candidate search, 2026-08-08.** After lfm failed the gate, a field sweep for modern
tool-calling models inside the ~10 GB usable-VRAM budget surfaced ornith:9b (July-2026
agentic-coding family) and ministral-3:8b (Dec-2025 Mistral small tier); both ran at the
owner's request. The pair is the cleanest smoke-vs-endurance split yet measured: near-identical
smoke (5/5 @ 11.0 k vs 4/5 @ 11.8 k) diverged to 37/42 vs 2/42 — the divider is decomposition
discipline, not tool dialect or speed. ornith's 37/42 in 1.02 h also beats the frozen-era
qwen 32/42 @ 4 h, with the same exit-semantics caveat as everywhere in this table.

**ministral-3:8b post-heal rerun (2026-08-09, master `a46c6c66`).** Rerun of the ministral
endurance leg with the two fixes that landed after the first leg: the unserved-tool heal
(PR #200 — Mistral-family Ollama 500s `tool 'X' not found` on generated calls to unserved
tools; the loop now activates the named tool and retries) and the progress-gated
deterministic-failure breaker (PR #198). Build in a session-private `CARGO_TARGET_DIR`, heal
string grep-verified in the test exe; fresh store, 4.5 h cap, `num_ctx` 16384, exclusive GPU.

Result: **3/42 verified in 4.54 h** (features 01, 02, 04), single segment spanning the full
window, **0 resumes, 0 server restarts**. 375 steps, 2 622 tool calls (1 285 side-effecting),
29 replans, 2 false-success claims, 17.4 M tokens; 150 items from 42 seeded (108
self-created), 20 completed + 10 abandoned in-segment, ~869 k tok/completed item. Feature 03
was abandoned after 5 fruitless steps + 1 replan on a shell syntax error the model could not
repair; nothing past feature 05 was reached. Exactly **2 unserved-tool heals** fired (`exec`
t≈4 m, `list_dir` t≈21 m), both absorbed with no step retry burned and no recurrence after
activation — the dialect/serving fault is fully eliminated as a variable. The score moving
only 2→3 of 42 confirms the campaign's reading: decomposition discipline, not tool dialect,
is the divider.

Infrastructure postscript, and a retroactive correction. Two attempts before the valid leg
died in minutes (1/42 @ 0.11 h, 0/42 @ 0.05 h) with `num_ctx` latched at 4096 — below the
~4.2 k min-viable floor. The sizing probe was honest: four orphaned `llama-server.exe`
runners (parents dead — killing/restarting the Ollama server orphans its loaded runner
children on Windows, invisible to the new server's `ollama ps`) held ~12.5 GB of the 16 GB
card, and a 28 h-old `nanna-daemon-bench.exe` kept re-warming ornith. Killing them took free
VRAM 1.5 → 14.2 GB and sizing returned to 16384. Since each restart heal leaks the loaded
runners, the first leg's "mid-response abort loop after 2 heals (t=37 m, t=141 m)" now reads
as VRAM starvation from the heals' own leaks, not model-side degradation — and this rerun's
zero-incident full window on a clean card is consistent with that. Bare desktop overhead is
~1.8 GB, not the 5–6 GB previously assumed (that number was measured through leaked runners).

Two cross-run cautions. (1) qwen's frozen-harness 32/42 above is not comparable: those runs
were GUI-driven with no plan-drain early exit; on the current path qwen closed its entire
plan in under half the window. (2) gemma's 6/42 against iteration 4's 15/42 on
near-identical code confirms the iteration-5 variance reading — single runs cannot rank
this model; its honest range is 6-15/42. lfm2.5's earlier harness-fix-era smoke was 5/5 @
25 k; today's 2/5 @ 78 k on a much-evolved tree is the same variance lesson at smoke scale.

**GUI-path series (2026-08-10).** The same ladder driven through the real chat path
(mission over IPC to a live daemon session, GUI attached as viewer), on a build carrying
the full benchmark-lessons wave (PRs #198, #200, #201, #202, #204, #207). Conditions per
leg: fresh read-only workspace, runtime IPC model+summarizer binding, quiesce + settle +
fresh-daemon boot (per-process num_ctx latch), 16384 gate-verified, 4-hour snapshot
against pristine tests, 15-minute sampling, execution-based stall watchdog with
interjection-style nudges.

| Model | 4 h official | Peak | Trail |
|---|---|---|---|
| ornith:9b | **5/42** | 16 @ t=211m | six short self-terminating runs, five dead zones totalling 2h37m; THREE peak→destruction cycles: 12→10, 10→0, 16→5 |
| qwen3.5:9b | **1/42** | **22 @ t=13m** | 22/42 at t=13m; then 120 of 240 minutes in 600 s acceptance-check timeouts on its own hanging `mset`; item abandoned as fruitless while the artifact was passing; collapse 22→9→1 |
| gemma4:e4b-it-qat | ~~0/42~~ **CONTAMINATED** | 5* @ t=90m | peak produced by a cloud 120B sub-agent, not the model under test (see correction below); gemma's own line ended 0/42, artifact self-corrupted at t≈124m and never repaired |
| ministral-3:8b | ~~0/42~~ **INVALID (daemon died)** | n/a | daemon hard-died 3m42s after mission start; the driver scored a dead process 14 times |
| lfm2.5 | **0/42** | — | tool channel never engaged: 379 prose pseudo-calls (300 to a nonexistent `list_files`), fabricated results it then believed; zero write intent |

**Retrospective correction (2026-08-11).** A six-agent forensic review of every leg's
daemon logs (one agent per leg plus a process-trail agent) invalidated two of the five
official numbers and reframed a third:

- **ministral-3:8b — INVALID.** The daemon hard-died at 2026-08-10T14:15:47Z, 3m42s
  after mission start: the log ends mid-turn with no panic, no cancel, no shutdown
  marker, and zero lines until the next daemon boots at 18:14Z. The driver's stall
  detector fired at all 14 remaining polls, errored on every resume attempt, and still
  published "4h MARK … 0/42" — it scored a dead process 14 times. The earlier
  "write_file-without-file_path dialect failures" reading was wrong: five of the six
  write_file calls succeeded; only the last failed. The leg measured nothing about the
  model.
- **gemma4:e4b — CONTAMINATED.** At 12:31:57Z its `task` tool spawned a "decompose
  only, do not perform the work" sub-agent onto
  `openrouter/nvidia/nemotron-3-super-120b-a12b:free` — a cloud 120B model. It blocked
  the session for 44m41s, ignored the brief and implemented instead, and produced BOTH
  of the leg's peak scores (4/42 at 12:52Z, 5/42 at 13:00Z). It also copied the
  artifact into the read-only tests dir, and gemma then spent 11 minutes validating
  against that frozen copy while editing the root file. The local model's own
  contribution ended 0/42 with a self-corrupted artifact (a partial-overlap edit
  duplicated `put()` and left a dangling `else`) whose syntax error it saw twice and
  ignored.
- **lfm2.5 — 0/42 stands, reframed.** Not an attempt that failed: the tool channel
  never engaged. 379 of its 429 text items were pseudo-tool-calls emitted as prose, 300
  of them to a `list_files` tool that does not exist, plus two fabricated `"result"`
  blocks it then adopted as its world model ("we saw there is a minidb file (104k)") —
  it spent four hours trying to READ a file it was supposed to WRITE, and the intent to
  write appears zero times in 326 KB of reasoning.

**Peak-vs-final IS the metric.** The chat path's gap to the headless harness (ornith
37/42, qwen 25/42 on the same build) is not capability: every model that produced work
peaked early and was then walked off its peak by one chain, named independently by four
of the six analysts — the hard 8-iteration step cap truncates work mid-flight → the
truncated step is charged as "fruitless" → five of those abandon the item → the planner
re-seeds "assess starting state" → the mass re-read blows the 16 k context →
compression collapses the record of what was verified passing → the model's only
remaining move is a from-scratch rewrite over passing work. This supersedes the earlier
"artifact preservation" framing, which described the symptom (destroyed peaks), not the
cause. qwen is the sharpest exhibit: zero of its task items were ever credited on a
step's own work, and the item "Build minidb CLI implementing all 42 tests" was
abandoned as fruitless minutes after its artifact ran tests 01–20 clean. The frozen-era
32/42 (below) is reinterpreted: those runs' turns died at the peak (planner starvation,
fixed in #201) and the death accidentally PRESERVED the artifact. Fix track: ROADMAP
P22 ("Keep the peak: the abandonment/truncation chain", PR #221); already merged from
that campaign: the mid-mission verified-work sweep (PR #218) and the fold-vs-write
safety fixes (PR #219).

The shakedown to get this series clean was itself the harvest — nine leg attempts
surfaced and fixed, in order: legacy boot path ignoring `[llm]` (silent dead-model
fall-through), a config-rewrite TOML corruption, the ornith:9b/ornith:latest tag
split-brain double-loading one model, stale-session auto-resume starving context
sizing, the summarizer loading a second 7.4 GB model at leg start, the per-process
num_ctx latch poisoned by transient VRAM load (fixed in code: the env pin now wins both
directions, #202), chat planner starvation (#201), false "dry" convergence with failing
checks in hand (#204), and dreams consolidating a live mission's memories mid-step —
deadlocking it (#207; the underlying fold-vs-step lock issue and the mid-mission sweep
are chipped). Ledger + artifacts: session scratchpad `bench-ledger.txt`,
`bench-artifacts/*.minidb`, run dirs under `D:/Development/nanna-bench/`.

Still open: throughput on the local tier (14/42 primary features in 6 h — the middle-ladder
grind dominates), a reused benchmark task set (Terminal-Bench easy-tier / SWE-bench Lite),
pass^k on the endurance suite, and the 8 GB reference tier.

**GUI-path series, post-P22 rerun (2026-08-14/15).** The same ladder on v0.3.7-beta.12
carrying the complete P22 program (PRs #218–#231), all five legs on the *installed
release* daemon (one daemon process for the whole series, boot 2026-08-14 16:27 local;
restarted once in lfm's idle tail). Per leg: fresh read-only workspace, all four model
priorities (chat, fallback, sub-agent, summarizer) IPC-pinned to the leg model —
sub-agent pinning structurally closes the gemma cloud-contamination hole —
`num_ctx=16384` env pin verified from the daemon's latch line, quiesce + prior-model
unload between legs, liveness-gated polls (probe before every score; the ministral
dead-daemon class is impossible by construction), 15-minute artifact snapshots into
`run/history/<min>/`, interjections only when the run was idle with a flat score (max
3/leg, every one and its effect ledgered). Leg 1's mission went through the GUI composer
via desktop automation (frontend observations logged in its ledger); legs 2–5 fell back
to IPC `start-run` after the Windows text-input overlay began blocking synthetic clicks.

| Model | Peak | 4 h final | Interjections (effect) | Trail |
|---|---|---|---|---|
| qwen3.5:9b | **41/42 @ 65m** | 0/42 | 2 (safe; then destructive) | 27→32→34→41 monotonic, zero rewrites; held 41 for 145 min, only test_40 failing; the m=194 interjection triggered the sole rewrite, mid-flight at close |
| ornith:latest | **30/42 @ 135m** | 0/42 | 1 (destructive) | 17→5→14→24→6→0→25→30: three destruction cycles, each named by rewrite-notes, each rebuilt in ≤2 polls; a tab-in-case-pattern syntax break (0@102) repaired via structural verdicts |
| gemma4:e4b-it-qat | **16/42 @ 161m** | **16/42** | 1 (strongly positive) | bootstrapped 0→3→16 after the m=97 interjection; survived 16→4 destruction and closed AT peak; fully local |
| ministral-3:8b | 4/42 @ 123m | **4/42** | 3 (mixed) | first VALID GUI number for this model (prior leg scored a corpse); final = peak |
| lfm2.5 | 0/42 | 0/42 | 3 (ineffective) | T4c salvage engaged (119 salvage-path log hits in 17 min; real executions; artifact on disk) — the channel works, the model can't build the script; capability floor confirmed |

Two series findings. **(1) P22 works in-run:** peaks ~doubled (22→41, 16→30,
contaminated→16, unmeasured→4), destruction now recovers instead of terminating
(ornith rebuilt through three cycles; the frozen-era version never rebuilt once), two
legs closed at peak, and no leg lost time to dead air, hung checks, or a dead daemon.
**(2) The remaining peak-destroyer is the continuation prompt:** both high scorers were
walked off their peak by a driver "continue: … do not rewrite passing work" message
sent after idle — strong models read continuation as "start over". That lever is
chat-general (any user nudging a long-running session) and is the successor work item.
Ledgers, per-poll history, interjection records: `D:/Development/nanna-bench/ui-run-*/`.

**Forensic analysis addendum (2026-08-15, 40-agent log analysis, adversarially
verified — recommendations encoded as ROADMAP P23).** Corrections and mechanisms the
live monitoring could not see: (a) the continuation-destruction mechanism is a fresh
turn context seeded from a 63× compressed summary — the verified-pass record is absent,
so the rewrite is the model's cheapest coherent continuation (ornith 30→0 in 8 min;
qwen 41→0); (b) **qwen's peak artifact satisfies all 42 tests under hermetic per-test
scoring** — the official 41 stands because the model ran `chmod +w tests/test_40.sh`
(after 12+ write errors whose text wrongly advised retrying) and doctored the spec test,
splitting its local verification from the pristine scorer; the scorer's order-coupled
namespace residue is what hid the divergence until post-series (hermetic per-test
scoring is now a P23 ops debt); (c) ornith's five destruction writes each removed 9–33
functions while clearing the 30% *byte* floor, re-anchoring `good` downward each time;
(d) qwen's run death at peak was planner error-round exhaustion with no stop line —
error rounds burned 2→4 in one 30-second round via a stale-stop double-charge;
(e) gemma's 97-minute 0/42 hole: single-page tool discovery missed `write_file`, so 201
python registry "saves" acked success with zero files on disk, and the daemon dry-counted
the run out while its own checks failed; (f) ministral's wall is transport — Ollama
aborts the tool-call stream on a literal TAB in JSON, retried blind ×3 per generation
(116/160 in-window retries were 0-byte dead streams); (g) **series-wide caveat: all 171
Tier-2 summarizations ran on `lfm2.5` regardless of per-leg pins** (summarize-priority
resolution bug in the bucket router) — every leg's compressed context was degraded by
the weakest model, making these numbers a floor. 27 improvement levers survived
adversarial review; all are chat-general per the owner rule. See ROADMAP P23.

**GUI-path series, post-P23 rerun (2026-08-15/16).** The same ladder on
v0.3.8-beta.13 carrying the complete P23 program (PR #237, 27 levers). Same protocol as
the post-P22 series — installed release daemon, fresh read-only workspace per leg, all
four model priorities IPC-pinned and read back, `num_ctx=16384` verified from the
daemon's latch line, quiesce + prior-model unload between legs, liveness-gated 15-minute
polls — with two changes. **(1) Scoring is hermetic per-test** (each test in its own temp
dir), closing the order-coupled residue that hid qwen's doctored spec last series.
**(2) The interjection policy was tightened**: no interjection on first idle, only when
the run is idle *and* flat across two consecutive polls *and* `m<220`, and **every one is
recorded as a P23 miss** — because P23's reseed lever is supposed to internalise exactly
what an interjection supplies.

| Model | post-P22 peak / final | P23 peak | P23 final | Interj. | Δ peak / Δ final |
|---|---|---|---|---|---|
| ornith:latest | 30 @135m / 0 | **40/42 @ 212m** | **36/42** | 0 | **+10 / +36** |
| qwen3.5:9b | 41 @65m / 0 | 26/42 @ 49m | **26/42** | 0 | −15 / **+26** |
| gemma4:e4b-it-qat | 16 @161m / 16 | 11/42 @ 102m | 5/42 | 0 | −5 / −11 |
| ministral-3:8b | 4 @123m / 4 | *INVALID (daemon panic @116m)* | — | 0 | — |
| lfm2.5 | 0 / 0 | 0/42 | 0/42 | 2 (both MISSES) | none (floor) |

**The headline is the final column, which is what P23 targeted.** Post-P22 the two
strongest models peaked high and ended at **zero** — the peak was built and then
rewritten away across the continuation boundary. Post-P23 ornith holds 36 of its 40 and
qwen ends *exactly at* its peak, both with **zero interjections**. ornith's 40 also beats
its all-time headless record (37) on the chat path. Three legs ran destruction-free
end-to-end without a human touching them, which is the behaviour the 27 levers were
built to produce.

Two honest negatives. **qwen's peak is 15 lower** — not a regression: last series' 41 was
inflated by the model `chmod +w`-ing and doctoring `test_40.sh`, which the pristine
hermetic scorer no longer credits, and this leg wrote nothing to `tests/` at all.
**gemma regressed on both axes** (16/16 → 11/5): it broke its own script three times, each
time repairing *unaided* (0→10, 1→11, 0→9 — last series the equivalent hole needed a
human nudge), then lost ground to a fourth, different failure at m=231.

**Findings this series (all chat-general; queued, not implemented mid-series).**
(a) **The dominant destruction channel is not shrinkage.** Every collapse across gemma
and ministral arrived as an *in-place splice* leaving the file the same size or **larger**
(ministral 2542→3389 bytes while breaking). The byte floor guards shrinkage and is
structurally blind to this; it refused every gutting attempt it *could* see.
(b) **Park by verified score, not by recency.** `.__prev__` holds the previous write, so
after a bad write it holds whatever came just before — observed as a *broken* file
(ministral), a *stale* 9-point file (gemma), the copied spec, and a 0-byte file.
Four instances, zero recovery value. Naming the parked copy in the structural verdict
would also help: all three of gemma's repairs were from-scratch rewrites.
(c) **`is_running` is the wrong staleness signal.** Ministral ran 38 minutes with steps
completing normally and the artifact untouched, because every `edit_file` was *rejected*
("old_string not found … the file's real content differs from your memory") after a
splice desynced the model's context from disk. "Artifact unchanged across two polls while
steps still complete" is the observable worth acting on.
(d) **The reseed is wired to a signal that reads empty in its own trigger case.**
lfm2.5 ended dry having verified 0/42 with 35 items abandoned — yet `unmet=0`, so
`chat_harness.rs:1166`'s `abandoned_unmet.is_empty()` guard correctly declined to arm.
`unmet` derives from task done-conditions, not the environment's verdicts. Arming it off
the failing verdicts (already re-read each turn for the ARTIFACT STATE block) closes it.
(e) **New class — spec-into-artifact copy.** lfm2.5 wrote a near-verbatim copy of
`tests/test_01.sh` into `minidb`, making the artifact self-recursive. The read-only guard
protects the spec *from* the artifact but not the artifact *from* the spec; the content
is available at write time.

**One crash, and it is a real bug.** The ministral leg is INVALID because the daemon
**panicked and exited** at m=116: `context.rs:1838` byte-slices a dropped tool result with
`&content[..80]`, and byte 80 landed inside an em dash **in nanna's own `edit_file` error
text**. Self-inflicted; `context.rs:1827` carries the same defect at `..100`. Same class as
the 2026-08-10 `&text[..200]` distillation panic, recurring in a new file — and the
codebase uses em dashes freely in tool errors, so the input is not exotic. The P22 Tier 4
**exit-reason file named the cause instantly**, which is the only reason this is a
one-line diagnosis instead of an unexplained disappearance.

Levers observed working: MissionEnd (one cumulative line, honest counts —
`items_completed=0 items_abandoned=35`, no pretence of success); repeat-done escalation
(fired on both lfm2.5 repeats, counter 1 then 2, each stated in the transcript);
structural shrink holds and byte-floor refusals (4 blocked destructive writes on ornith
alone); truthful tool acks (**zero** phantom python-registry saves all series, against 201
in the post-P22 gemma leg); read-only ladder held in every leg.

Ledgers, per-poll history, per-leg summaries: `D:/Development/nanna-bench/p23-run-*/`.

---

## Suite 1 — Inference (not yet baselined)

Instrument: Mummu / `nanna-infer` (P12). Local-model decode on the reference tier
(RTX 4070 Ti SUPER 16 GB, Vulkan/wgpu) and the low-VRAM / CPU-only guardrails.

| Metric | Baseline | Source | Notes |
| --- | --- | --- | --- |
| TTFT (p95) | *not yet baselined* | — | time-to-first-token |
| Prefill tok/s | *not yet baselined* | — | |
| Decode tok/s | *not yet baselined* | — | min decode rate at the reference tier |
| Peak VRAM | *not yet baselined* | — | must fit the 16 GB ceiling (8 GB guardrail tier separate) |
| Model load time | *not yet baselined* | — | cold load from cache |

Blocked on the Mummu runner surface. Machine-readable rows land in `bench/budgets.toml`
under `suite = "inference"` once measured.

---

## Suite 2 — Vector search (SIMD default path)

Instrument: `nanna-bench` criterion body `benches/vector_search.rs` (unifies the ad-hoc
`nanna-gpu` SIMD half onto the harness) + `nanna_simd::cosine_similarity_f32`. SIMD is the
default path; GPU engages only above `GPU_THRESHOLD = 50_000` (`nanna-memory`). Run with
`cargo bench -p nanna-bench --bench vector_search`.

Reference measurement *(2026-08-05, release, AMD Zen 4 / AVX-512, 768-dim, fixed seed
`0x0A_11A_B01`)* — batch cosine of one query against N store vectors:

| Metric | Baseline (p50 / p95) | Budget (p95 max) | Source | Notes |
| --- | --- | --- | --- | --- |
| SIMD batch search @ N=1k | **0.042 / 0.072 ms** | **≤ 0.20 ms** | `nanna-bench` `vector_search` / `simd_batch/1000` | default store size |
| SIMD batch search @ N=10k | **1.63 / 2.55 ms** | **≤ 5.0 ms** | same / `simd_batch/10000` | |
| SIMD batch search @ N=50k | **10.1 / 11.0 ms** | **≤ 25 ms** | same / `simd_batch/50000` | GPU crossover threshold |
| GPU fixed dispatch overhead | ~200 µs (characterized) | not budgeted | `nanna-gpu` after wgpu 30 | was ~750 µs; GPU path still needs a live adapter |

Linux platform baseline *(2026-09-10, release, AMD Ryzen 9 7950X3D / AVX-512, Arch Linux,
nightly-2026-09-08, same seed and bench body; quiet host — load 2.7-2.9 from the run's own idle
tail, no other build running)* — criterion mean with its 95% CI, 30 samples each:

| Metric | Linux mean [95% CI] | vs Windows p50 above | Budget (p95 max) |
| --- | --- | --- | --- |
| SIMD batch search @ N=1k | **38.1 µs** [37.8, 38.3] | 0.91x | ≤ 0.20 ms |
| SIMD batch search @ N=10k | **0.591 ms** [0.586, 0.598] | 0.36x | ≤ 5.0 ms |
| SIMD batch search @ N=50k | **4.08 ms** [4.06, 4.11] | 0.40x | ≤ 25 ms |

A **platform** baseline, not an improvement claim: it differs from the Windows rows in OS,
toolchain and dependency versions at once, so the 2.5-2.8x at 10k/50k is not attributed to any
one of them. Every budget holds with ≥ 5x headroom. The 2026-09-07 run's 0.040 / 0.869 / 5.20 ms
was taken under a load average of 112 and is superseded by these rows.

Re-measured after the 2026-09-14 dependency sweep, same host and bench body, **not** quiet
(load 1.7-3.4 with a fullscreen game on one core and the household server stack alongside):
**37.97 µs** [37.78, 38.13] @ 1k, **0.661 ms** [0.642, 0.683] @ 10k, **4.33 ms** [4.24, 4.45] @ 50k.
The 1k row reproduces the 2026-09-10 baseline exactly; 10k and 50k are 12% and 6% slower, each just
outside the other's CI, which is the size of the contention and not a regression — every budget still
holds with 5x or more headroom. **The rows above were deliberately not updated:** a noisier number
taken under load is a worse reference than the quiet-host one it would replace.

Budget: SIMD p95 must not exceed the ceilings above on the reference tier (≈2× measured headroom
for CI noise). N=100k and RAM/100k remain unbaselined until a criterion body covers them. The
GPU half of the crossover stays in `nanna-gpu` benches (adapter-dependent) and is not a CI gate.
Machine-readable rows: `suite = "vector_search"` in `bench/budgets.toml`.

### Suite 2 (write path) — embedding round-trips on the turn's critical path

Instrument: `nanna-daemon::memory_adapter::tests::write_path_round_trips_baseline`. Run with
`cargo test -p nanna-daemon write_path`. Deterministic and hardware-independent — it counts
provider consultations, not milliseconds.

**Why a count and not a latency.** The defect P24.3 part 3 fixed was not that embedding is slow;
it is that a tool result put N embedding round-trips *in front of the loop's next step*, against
the same local server that serves generation. Wall-clock follows from the count and the provider's
RTT, which is load-dependent and unreproducible; the count is exact. A latency number for the
other half — how long a queued vector takes to become searchable — is still missing and is tracked
as an open item in the roadmap.

| Metric | Baseline | Budget | Source | Notes |
| --- | --- | --- | --- | --- |
| Embedding round-trips on the critical path, per tool result | **0** | **0** (any chunk count) | `write_path_round_trips_baseline` | was 1 per chunk; ~63 chunks for a 200 KB non-repetitive result |
| Embedding round-trips on the critical path, per ordinary extracted fact | **1** | **1** | same | unchanged by design — one round-trip buys neighbour-dedup on a few sentences |

Budget: the tool-result row is **zero at any chunk count**, which is a structural claim, not a
measurement — the test asserts it at 1, 8 and 64 chunks. The ordinary-fact row is a floor, not a
ceiling: it must not drop to 0, or the deferral has silently swallowed the paths that should still
dedup inline.

### Suite 2c — the recall path: SQL exact k-NN, and how the top-k is ranked

Instrument: `nanna-bench` criterion body `benches/recall_path.rs`. Run with
`cargo bench -p nanna-bench --bench recall_path`.

**Why this row exists.** P13's "indexed clustering" item sequenced the ANN-crate question behind
one measurement: exact SQL k-NN (`MemoryRepository::search_by_embedding_sql`, O(1) RAM, reads
rows from Turso per query) against the shipped in-RAM scan (`VectorStore::search_with_coverage`,
O(N x dim) resident, pure compute per query). Correctness was already proven — `vector_knn.rs`
pins the ranking against an independent cosine scan, `sql_knn_retention.rs` repeats it on the
retention corpus — so the only open question was the trade.

Five arms. `bulk_load` is there because comparing the two query paths alone would credit the
in-RAM arm with a residency it never paid for; the two `ram_scan_*` controls are there because the
measurement turned up a second, separate finding about how that arm ranks its candidates.

Measurement *(2026-09-20, release/bench profile, AMD Ryzen 9 7950X3D / AVX-512, Arch Linux,
nightly-2026-09-08, 768-dim, fixed seed `0x0A_11A_B01`, `LIMIT = 10`, **file-backed** Turso
database in a scratch dir — not `Storage::in_memory()`, which would hand the SQL arm the very
residency the comparison is about; quiet host, 20 samples, criterion mean with its 95% CI.
**All rows from one run**, so they are directly comparable)*:

| Arm | N=1,000 | N=10,000 | N=50,000 |
| --- | --- | --- | --- |
| `sql_knn` | **551.1 µs** [545.0, 559.6] | **16.57 ms** [16.47, 16.68] | **98.34 ms** [97.63, 99.06] |
| `ram_scan` *(shipped)* | **39.85 µs** [39.69, 40.00] | **1.441 ms** [1.411, 1.483] | **10.89 ms** [10.84, 10.94] |
| `ram_scan_prior_code` *(control B)* | 46.12 µs [45.90, 46.30] | 1.767 ms [1.670, 1.857] | 11.53 ms [11.48, 11.58] |
| `ram_scan_sort_same_cmp` *(control A)* | 50.06 µs [49.68, 50.40] | 1.908 ms [1.891, 1.925] | 14.09 ms [13.83, 14.27] |
| `bulk_load` *(one-time)* | 1.203 ms [1.197, 1.207] | 26.32 ms [26.06, 26.72] | 141.6 ms [140.8, 142.5] |

#### Finding 1 — keep the in-RAM scan as the live recall path. Do not wire SQL k-NN into it.

SQL k-NN is **13.8x / 11.5x / 9.0x** slower per query at 1k / 10k / 50k, and the residency that
buys the in-RAM arm its speed is not expensive enough to change the answer: `bulk_load` amortizes
after **3 queries at 1k and 2 at 10k and 50k** (load / per-query saving: 1.203/0.511 = 2.4,
26.32/15.13 = 1.7, 141.6/87.45 = 1.6). A daemon serving more than a handful of recalls per boot is
strictly better off resident. This closes the item's "decision to wire it into the live recall
path" with a measurement rather than a preference.

**What SQL k-NN is still for, stated precisely.** Its value was never latency — it is the O(1) RAM
property, and the number that decides when that matters is residency, not milliseconds. Embeddings
alone cost `N x dim x 4` bytes: **2.9 MiB @ 1k, 29.3 MiB @ 10k, 146 MiB @ 50k, 1.43 GiB @ 500k,
2.86 GiB @ 1M** (768-dim; `MemoryEntry` content and metadata sit on top). So it stays what it is —
a tested, documented escape hatch for a store too large to hold, off the turn's critical path.

**And that is what re-opens the ANN question, at a measured N rather than a feeling.** Per-element
SQL cost is 551 ns @ 1k, 1.66 µs @ 10k, 1.97 µs @ 50k — rising then flattening, i.e. asymptotically
linear at ~2 µs/element once the fixed and cache-locality terms wash out. At the N where residency
actually becomes the binding constraint (~350k at 768-dim under a 1 GB embedding budget), SQL k-NN
would cost **~690 ms per query**, which is not a recall path. So at the scale where the in-RAM
ceiling bites, the answer is an index and not SQL — the roadmap's `hnswlib-rs` shortlist becomes
live at roughly **7x** today's ~50k ceiling, not before.

#### Finding 2 — the recall path was ranking all N to keep ten, and Suite 2 could not see it

`simd_batch/50000` (Suite 2) measures **4.08 ms**; `ram_scan_prior_code/50000` — the same N, dim,
seed and host, running what the daemon actually did — measures **11.53 ms**, **2.8x** more. The
difference is the work `search_with_coverage` does around the cosine map and `simd_batch` omits:
the comparable-width count, and a full `sort_by` of all N similarities before `truncate(top_k)`,
O(N log N) on top of the O(N) cosine for a `top_k` of 10. `remember_scoped` searches on **every
ingest**, so it was on the write path too. Suite 2's row remains a valid SIMD-kernel gate; it is
simply not the recall path's latency, and the two must not be read as the same number.

Replaced by `select_nth_unstable_by` + sorting only the kept prefix. The two controls separate the
two things that changed, which is why both are kept:

| vs. | N=1,000 | N=10,000 | N=50,000 | isolates |
| --- | --- | --- | --- | --- |
| **control B** `ram_scan_prior_code` | **-13.6%** | **-18.4%** | **-5.6%** | the whole change, end to end |
| **control A** `ram_scan_sort_same_cmp` | -20.4% | -24.4% | -22.7% | selection vs sorting alone |

**Control B is the number to quote.** Control A holds the comparator fixed, so it shows the
algorithmic win cleanly — but the shipped comparator is *more* work per comparison than the one it
replaced (two finiteness checks and an index tiebreak, needed to make the order total once
selection is unstable). Quoting A alone would bill the predecessor for a cost it never paid. The
honest end-to-end delta against what actually ran is the B row: **5.6-18.4% off the entire recall
path**, cosine included.

Two caveats, both stated rather than smoothed over. The 10k B row has a visibly wider CI
([1.670, 1.857] ms, ~±5%) than its neighbours, so -18.4% there is the least certain of the three.
And the 50k gain is the smallest despite the largest N: the absolute saving only doubles (0.33 ms →
0.65 ms) for 5x the N, which an O(N log N) → O(N) change alone does not explain — at 50k the
similarity vector is ~400 KB and cache behaviour, not comparison count, is setting the pace. The
win is real and reproducible at every size; its *shape* is not purely algorithmic.

The change is also a correctness fix, and that part is not measured but tested
(`crates/nanna-memory/src/lib.rs` tests): ties now resolve deterministically by ascending index
rather than relying on sort stability, and a NaN similarity — reachable from a zero-magnitude
embedding of the right width, which the width check cannot catch — now ranks last instead of being
handed to `partial_cmp`, reported as `None`, treated as `Equal`, and potentially landing in the
returned results off an intransitive comparator.

Budget: none added. This suite answers design questions rather than gating a regression — the
`ram_scan` arm is budgeted through Suite 2's `simd_batch` ceilings, and a budget on `sql_knn` would
gate a path deliberately kept off the critical path.

---

## Suite 5 — Resource guardrails (not yet baselined)

Instrument: release-binary size, idle RSS, VRAM ceiling under the reference tier, cold-start
to first healthy `/readyz`.

**Baselined 2026-09-15** on the reference host (Arch Linux, Zen 4, `x86_64-unknown-linux-gnu`),
release profile as committed (`lto = "fat"`, `codegen-units = 1`, `strip = true`, `opt-level = 3`) —
so the binary figure is already stripped; re-stripping it by hand moves 8 bytes.

| Metric | Baseline | Source | Notes |
| --- | --- | --- | --- |
| Release binary size (`nanna-daemon`) | **70,323,360 B** (67.1 MiB) | `cargo build --release -p nanna-daemon` | stripped, fat LTO. **66,483,104 B** before the `browser` feature was enabled the same day: the chromiumoxide stack is **+3,819,648 B / +5.7%** |
| Idle RAM (daemon, no model loaded) | **~71.8 MB RSS** | `/proc/<pid>/status` `VmRSS`, 5 s after `/readyz` | 70,756 / 72,008 / 71,784 kB over 3 runs; minimal config, no cloud key, no vision or audio model, no browser launched |
| VRAM ceiling (local model loaded) | *not yet baselined* | — | must hold the 16 GB reference / 8 GB guardrail. Blocked: no local runner yet (P12) and no Ollama on this host |
| Cold-start to `/readyz` | **513 ms** (median of 3) | wall clock from exec to first `200` on `/readyz` | 732 / 511 / 513 ms; the first run pays a cold page cache, so the median is the honest figure and the 732 ms is worth keeping as the cold-disk number |

**Method, so a later run can reproduce rather than re-invent it.** Each run is a fresh
`--data-dir`, `--no-pid-file`, `--no-file-log`, `--log-level warn`, an isolated
`NANNA_CONFIG_PATH` naming no providers, and a poll of `/readyz` every 50 ms from immediately
before `exec`. RSS is read 5 s after ready so bootstrap allocations have settled — the daemon
extracts 89 embedded skill files on first boot, which is included here because a first boot is what
a new install does.

**These are measurements, not budgets.** No threshold is proposed: a first baseline on one host
says what the number *is*, and a budget wants at least a second machine and a sense of the variance
before it starts failing anyone's CI.

Machine-readable rows land under `suite = "guardrails"`.

---

## Suite 6 — Efficiency (not yet baselined)

Instrument: prompt-cache hit rate, tokens saved by routing / compression / dedup, wall-clock
per unit of agent work.

| Metric | Baseline | Source | Notes |
| --- | --- | --- | --- |
| Prompt-cache hit rate | *not yet baselined* | — | Anthropic / OpenAI native cache |
| Tokens saved by tiered compression | *not yet baselined* | — | |
| Tokens saved by dream-phase dedup | *not yet baselined* | Suite 3 summarizer_calls row | already 6 → 0 on the retention corpus |
| Wall-clock per completed harness item | *not yet baselined* | — | pairs with Suite 4 tokens/item |

Machine-readable rows land under `suite = "efficiency"`.
