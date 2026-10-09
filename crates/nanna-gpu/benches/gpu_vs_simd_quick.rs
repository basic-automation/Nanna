#![warn(clippy::pedantic, clippy::nursery, clippy::all)]
// Solver depth only, as on the daemon crate roots: proving these futures
// `Send` walks the wgpu/daemon type graph past the default limit of 128.
#![recursion_limit = "256"]
//! Quick GPU vs SIMD crossover benchmark (reduced matrix for faster runs)
//! Run with: cargo bench -p nanna-gpu --bench `gpu_vs_simd_quick`

use std::time::{Duration, Instant};

fn simd_batch_search(query: &[f32], vectors: &[Vec<f32>]) -> Vec<f32> {
    vectors.iter().map(|v| nanna_simd::cosine_similarity_f32(query, v)).collect()
}

async fn gpu_batch_search(
    pipeline: &nanna_gpu::CosineSimilaritySearch,
    ctx: &nanna_gpu::GpuContext,
    query: &[f32],
    vectors_flat: &[f32],
) -> Vec<f32> {
    pipeline.search(ctx, query, vectors_flat).await.unwrap()
}

fn generate_vectors(count: usize, dim: usize) -> Vec<Vec<f32>> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    (0..count)
        .map(|i| {
            let mut v: Vec<f32> = (0..dim)
                .map(|j| {
                    let mut h = DefaultHasher::new();
                    (i * dim + j).hash(&mut h);
                    let bits = h.finish();
                    nanna_numeric::f32_from_f64((nanna_numeric::f64_from_u64(bits) / nanna_numeric::f64_from_u64(u64::MAX)).mul_add(2.0, -1.0))
                })
                .collect();
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-10 { for x in &mut v { *x /= norm; } }
            v
        })
        .collect()
}

fn flatten_vectors(vectors: &[Vec<f32>]) -> Vec<f32> {
    vectors.iter().flat_map(|v| v.iter().copied()).collect()
}

/// One measurement series: its centre and its spread.
///
/// `min`/`max` were computed here and printed nowhere, which read to the
/// compiler as dead fields and to a reader as a mean with no error bar. That
/// is the wrong thing to discard in *this* bench: it is the source of the
/// GPU-vs-SIMD crossover number, and a crossover claimed from two means whose
/// ranges overlap is not a crossover. The spread is on every row now.
struct Stats { mean: Duration, min: Duration, max: Duration }

impl Stats {
    fn from(times: &[Duration]) -> Self {
        let sum: Duration = times.iter().sum();
        Self {
            mean: sum / u32::try_from(times.len()).expect("sample count fits u32"),
            min: *times.iter().min().unwrap(),
            max: *times.iter().max().unwrap(),
        }
    }
}

fn bench_sync<F: FnMut()>(mut f: F, warmup: usize, iters: usize) -> Stats {
    for _ in 0..warmup { f(); }
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        std::hint::black_box(&mut f)();
        times.push(t.elapsed());
    }
    Stats::from(&times)
}

fn bench_async_fn<F, Fut>(rt: &tokio::runtime::Runtime, mut f: F, warmup: usize, iters: usize) -> Stats
where F: FnMut() -> Fut, Fut: std::future::Future<Output = Vec<f32>>
{
    for _ in 0..warmup { rt.block_on(f()); }
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        std::hint::black_box(rt.block_on(f()));
        times.push(t.elapsed());
    }
    Stats::from(&times)
}

fn fmt_dur(d: Duration) -> String {
    let us = d.as_secs_f64() * 1e6;
    if us < 1000.0 { format!("{us:>8.1}µs") }
    else { format!("{:>8.2}ms", us / 1000.0) }
}

/// Spread as a percentage of the mean: the sampled range over the centre.
///
/// A ratio, not two more durations, because the column answers one question
/// at a glance: is the gap between these two means bigger than the noise? A
/// crossover read off means whose ranges overlap is not a crossover, and this
/// bench is where the `GPU_THRESHOLD` number comes from.
///
/// A zero mean is guarded rather than assumed away: it would divide by zero
/// and print `inf`, which reads as a measurement instead of an absence.
fn fmt_spread(s: &Stats) -> String {
    let mean_ns = s.mean.as_nanos();
    if mean_ns == 0 {
        return "  n/a".to_string();
    }
    let range = s.max.saturating_sub(s.min);
    let percent = range.as_secs_f64() / s.mean.as_secs_f64() * 100.0;
    format!("{percent:>4.0}%")
}

fn main() {
    let rt = tokio::runtime::Runtime::new().unwrap();

    println!();
    println!("══════════════════════════════════════════════════════════════");
    println!("  GPU vs SIMD Crossover Benchmark (quick)");
    println!("══════════════════════════════════════════════════════════════");
    println!();
    println!("SIMD tier: {:?}", nanna_simd::simd_tier());

    let gpu = rt.block_on(async { nanna_gpu::GpuContext::new().await });
    let (ctx, pipeline) = match gpu {
        Ok(ctx) => {
            println!("GPU: {} ({:?})", ctx.adapter_info.name, ctx.adapter_info.backend);
            match nanna_gpu::CosineSimilaritySearch::new(&ctx) {
                Ok(p) => (Some(ctx), Some(p)),
                Err(e) => { println!("Pipeline failed: {e}"); (None, None) }
            }
        }
        Err(e) => { println!("No GPU: {e}"); (None, None) }
    };
    println!();

    // Focused test matrix — one dimension (768, most common), key vector counts
    let dim = 768;
    let counts = [10, 50, 100, 250, 500, 1000, 2000, 5000, 10_000];
    let iters = 10;
    let warmup = 2;

    println!("  Dimension: {dim}  |  Iterations: {iters}  |  Warmup: {warmup}");
    println!("──────────────────────────────────────────────────────────────");
    println!("  Vectors │    SIMD mean │ ±sprd │     GPU mean │ ±sprd │  Ratio  │ Winner");
    println!("──────────┼─────────────┼──────────────┼─────────┼────────");

    let mut crossover: Option<usize> = None;

    for &count in &counts {
        let query = &generate_vectors(1, dim)[0];
        let vectors = generate_vectors(count, dim);
        let flat = flatten_vectors(&vectors);

        let simd = bench_sync(|| { std::hint::black_box(simd_batch_search(query, &vectors)); }, warmup, iters);

        if let (Some(ctx), Some(pipeline)) = (&ctx, &pipeline) {
            let gpu_stats = bench_async_fn(&rt, || gpu_batch_search(pipeline, ctx, query, &flat), warmup, iters);

            let ratio = gpu_stats.mean.as_secs_f64() / simd.mean.as_secs_f64();
            let winner = if ratio < 1.0 { "GPU" } else { "SIMD" };

            if ratio < 1.0 && crossover.is_none() {
                crossover = Some(count);
            }

            println!(
                "  {:>7} │ {} │ {} │ {} │ {} │ {:>6.2}× │ {winner}",
                count,
                fmt_dur(simd.mean),
                fmt_spread(&simd),
                fmt_dur(gpu_stats.mean),
                fmt_spread(&gpu_stats),
                ratio
            );
        } else {
            println!(
                "  {:>7} │ {} │ {} │     N/A      │   n/a │   N/A   │ SIMD",
                count,
                fmt_dur(simd.mean),
                fmt_spread(&simd)
            );
        }
    }

    println!("──────────────────────────────────────────────────────────────");
    println!();

    // Also test 1536-dim at a few key points
    if ctx.is_some() {
        let dim2 = 1536;
        let counts2 = [100, 500, 1000, 5000];
        println!("  Dimension: {dim2} (spot check)");
        println!("──────────────────────────────────────────────────────────────");
        println!("  Vectors │    SIMD mean │ ±sprd │     GPU mean │ ±sprd │  Ratio  │ Winner");
        println!("──────────┼─────────────┼──────────────┼─────────┼────────");

        for &count in &counts2 {
            let query = &generate_vectors(1, dim2)[0];
            let vectors = generate_vectors(count, dim2);
            let flat = flatten_vectors(&vectors);

            let simd = bench_sync(|| { std::hint::black_box(simd_batch_search(query, &vectors)); }, warmup, iters);

            if let (Some(ctx), Some(pipeline)) = (&ctx, &pipeline) {
                let gpu_stats = bench_async_fn(&rt, || gpu_batch_search(pipeline, ctx, query, &flat), warmup, iters);
                let ratio = gpu_stats.mean.as_secs_f64() / simd.mean.as_secs_f64();
                let winner = if ratio < 1.0 { "GPU" } else { "SIMD" };
                println!(
                    "  {:>7} │ {} │ {} │ {} │ {} │ {:>6.2}× │ {winner}",
                    count,
                    fmt_dur(simd.mean),
                    fmt_spread(&simd),
                    fmt_dur(gpu_stats.mean),
                    fmt_spread(&gpu_stats),
                    ratio
                );
            }
        }
        println!("──────────────────────────────────────────────────────────────");
        println!();
    }

    if let (Some(ctx), Some(pipeline)) = (&ctx, &pipeline) {
        measure_gpu_overhead(&rt, ctx, pipeline);
    }

    // Summary
    println!("══════════════════════════════════════════════════════════════");
    print_summary(crossover, ctx.is_some());
}

/// The fixed cost of one GPU dispatch, against one SIMD cosine of the same size.
fn measure_gpu_overhead(
    rt: &tokio::runtime::Runtime,
    ctx: &nanna_gpu::GpuContext,
    pipeline: &nanna_gpu::CosineSimilaritySearch,
) {
    println!("  GPU Fixed Overhead (1 vector, 768-dim):");
    let q = &generate_vectors(1, 768)[0];
    let sv = flatten_vectors(&generate_vectors(1, 768));
    let overhead = bench_async_fn(rt, || gpu_batch_search(pipeline, ctx, q, &sv), 3, 20);
    let simd1 = bench_sync(|| { std::hint::black_box(nanna_simd::cosine_similarity_f32(q, &sv)); }, 3, 20);
    println!("    GPU dispatch: {} (±{})", fmt_dur(overhead.mean), fmt_spread(&overhead).trim());
    println!("    SIMD single:  {} (±{})", fmt_dur(simd1.mean), fmt_spread(&simd1).trim());
    println!("    Overhead ratio: {:.0}×", overhead.mean.as_secs_f64() / simd1.mean.as_secs_f64());
    println!();
}

/// The verdict the crossover (or its absence) supports.
fn print_summary(crossover: Option<usize>, gpu_available: bool) {
    match crossover {
        Some(n) => {
            println!("  CROSSOVER: GPU becomes faster at ~{n} vectors (768-dim)");
            if n < 1000 {
                println!("  → Threshold of 1000 is CONSERVATIVE. Could lower to ~{n}.");
            } else {
                println!("  → Threshold of 1000 is appropriate or aggressive.");
            }
        }
        None if gpu_available => {
            println!("  GPU never beat SIMD in tested range (up to 10,000 vectors).");
            println!("  → GPU dispatch overhead dominates. Consider removing GPU path");
            println!("    or only using it for very large batches (>10k).");
        }
        _ => println!("  No GPU available — SIMD only."),
    }
    println!("══════════════════════════════════════════════════════════════");
}
