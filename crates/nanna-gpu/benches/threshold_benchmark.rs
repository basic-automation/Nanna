#![warn(clippy::pedantic, clippy::nursery, clippy::all)]
// Solver depth only, as on the daemon crate roots: proving these futures
// `Send` walks the wgpu/daemon type graph past the default limit of 128.
#![recursion_limit = "256"]
//! GPU vs SIMD Threshold Benchmark
//!
//! Measures the crossover point where GPU batch cosine similarity
//! becomes faster than SIMD for 384-dimensional vectors.
//!
//! Run with: cargo bench -p nanna-gpu --bench `threshold_benchmark`

use std::time::{Duration, Instant};

/// SIMD cosine similarity
fn simd_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    nanna_simd::cosine_similarity_f32(a, b)
}

/// SIMD batch search: compute cosine similarity of query against all vectors
fn simd_batch_search(query: &[f32], vectors: &[Vec<f32>]) -> Vec<f32> {
    vectors
        .iter()
        .map(|v| simd_cosine_similarity(query, v))
        .collect()
}

/// GPU batch search using wgpu compute shader
async fn gpu_batch_search(
    pipeline: &nanna_gpu::CosineSimilaritySearch,
    ctx: &nanna_gpu::GpuContext,
    query: &[f32],
    vectors_flat: &[f32],
) -> Vec<f32> {
    pipeline.search(ctx, query, vectors_flat).await.unwrap()
}

/// Generate deterministic normalized vectors using hash-based PRNG
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
            if norm > 1e-10 {
                for x in &mut v {
                    *x /= norm;
                }
            }
            v
        })
        .collect()
}

/// Flatten vectors for GPU (contiguous memory)
fn flatten_vectors(vectors: &[Vec<f32>]) -> Vec<f32> {
    vectors.iter().flat_map(|v| v.iter().copied()).collect()
}

/// Run a synchronous benchmark N times and return stats
fn bench_fn<F: FnMut() -> R, R>(mut f: F, warmup: usize, iterations: usize) -> BenchResult {
    for _ in 0..warmup {
        std::hint::black_box(f());
    }

    let mut times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        std::hint::black_box(f());
        times.push(start.elapsed());
    }

    BenchResult::from_durations(&times)
}

/// Run an async benchmark N times and return stats
fn bench_async<F, Fut, R>(
    rt: &tokio::runtime::Runtime,
    mut f: F,
    warmup: usize,
    iterations: usize,
) -> BenchResult
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = R>,
{
    for _ in 0..warmup {
        rt.block_on(async { std::hint::black_box(f().await) });
    }

    let mut times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        rt.block_on(async { std::hint::black_box(f().await) });
        times.push(start.elapsed());
    }

    BenchResult::from_durations(&times)
}

#[derive(Debug, Clone)]
struct BenchResult {
    mean: Duration,
    stddev: Duration,
    min: Duration,
    max: Duration,
}

impl BenchResult {
    fn from_durations(times: &[Duration]) -> Self {
        let nanos: Vec<f64> = times.iter().map(|t| t.as_secs_f64() * 1e9).collect();
        let mean = nanos.iter().sum::<f64>() / nanna_numeric::f64_from_usize(nanos.len());
        let variance = nanos.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / nanna_numeric::f64_from_usize(nanos.len());
        let stddev = variance.sqrt();
        let min = nanos.iter().copied().fold(f64::INFINITY, f64::min);
        let max = nanos.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        Self {
            mean: Duration::from_secs_f64(mean / 1e9),
            stddev: Duration::from_secs_f64(stddev / 1e9),
            min: Duration::from_secs_f64(min / 1e9),
            max: Duration::from_secs_f64(max / 1e9),
        }
    }
}

impl std::fmt::Display for BenchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:>10.2?} ± {:>8.2?}  (min {:>10.2?}, max {:>10.2?})",
            self.mean, self.stddev, self.min, self.max
        )
    }
}

fn main() {
    let rt = tokio::runtime::Runtime::new().unwrap();

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║        GPU vs SIMD Threshold Benchmark — nanna (384-dim)        ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // Detect SIMD tier
    println!("SIMD tier: {:?}", nanna_simd::simd_tier());
    println!();

    // Initialize GPU
    let gpu_available = rt.block_on(async { nanna_gpu::GpuContext::new().await });

    let (ctx, pipeline) = match gpu_available {
        Ok(ctx) => {
            println!("GPU: {} ({:?})", ctx.adapter_info.name, ctx.adapter_info.backend);
            match nanna_gpu::CosineSimilaritySearch::new(&ctx) {
                Ok(pipeline) => {
                    println!("GPU pipeline: ready");
                    (Some(ctx), Some(pipeline))
                }
                Err(e) => {
                    println!("GPU pipeline failed: {e} — running SIMD-only benchmarks");
                    (None, None)
                }
            }
        }
        Err(e) => {
            println!("No GPU available: {e} — running SIMD-only benchmarks");
            (None, None)
        }
    };
    println!();

    // Benchmark configuration
    let dim = 384;  // Standard embedding dimension
    let vector_counts = [100, 500, 1000, 5000, 10_000, 50_000, 100_000];
    let iterations = 20;
    let warmup = 3;

    println!("┌─────────┬──────────────────────────────────┬──────────────────────────────────┬──────────┐");
    println!("│ Vectors │           SIMD (AVX2/512)        │           GPU (wgpu)             │  Winner  │");
    println!("├─────────┼──────────────────────────────────┼──────────────────────────────────┼──────────┤");

    let query_vecs = generate_vectors(1, dim);
    let query = &query_vecs[0];
    let mut crossover_point: Option<usize> = None;
    let mut prev_gpu_faster = false;

    for &count in &vector_counts {
        let vectors = generate_vectors(count, dim);
        let vectors_flat = flatten_vectors(&vectors);

        // SIMD benchmark
        let simd_result = bench_fn(
            || simd_batch_search(query, &vectors),
            warmup,
            iterations,
        );

        // GPU benchmark (if available)
        let (gpu_result, winner) = match (&ctx, &pipeline) {
            (Some(ctx), Some(pipeline)) => {
                let gpu_result = bench_async(
                    &rt,
                    || gpu_batch_search(pipeline, ctx, query, &vectors_flat),
                    warmup,
                    iterations,
                );

                let speedup =
                    simd_result.mean.as_secs_f64() / gpu_result.mean.as_secs_f64();
                let gpu_faster = speedup > 1.0;

                if gpu_faster && !prev_gpu_faster && crossover_point.is_none() {
                    crossover_point = Some(count);
                }
                prev_gpu_faster = gpu_faster;

                let winner = if gpu_faster {
                    std::format!("GPU {speedup:.1}x")
                } else {
                    std::format!("SIMD {:.1}x", 1.0 / speedup)
                };

                (Some(gpu_result), winner)
            }
            _ => (None, "SIMD (no GPU)".to_string()),
        };

        let gpu_str = gpu_result.map_or_else(|| "N/A".to_string(), |r| std::format!("{r}"));

        println!(
            "│ {count:>7} │ {simd_result} │ {gpu_str:>32} │ {winner:>8} │"
        );
    }

    println!("└─────────┴──────────────────────────────────┴──────────────────────────────────┴──────────┘");
    println!();

    // ── Summary ────────────────────────────────────────────────────
    println!("═══ Summary ═══");
    println!();

    if let Some(threshold) = crossover_point {
        println!("✓ GPU becomes faster at: ~{threshold} vectors (384-dim)");
        println!();
        println!("RECOMMENDATION:");
        println!("  Set GPU_THRESHOLD = {threshold}");
        println!("  This enables GPU acceleration for searches with {threshold} or more vectors.");
    } else if ctx.is_some() {
        println!("GPU did not become faster than SIMD in the tested range (100-100k vectors).");
        println!("Consider increasing GPU_THRESHOLD or disabling GPU dispatch for 384-dim vectors.");
    } else {
        println!("No GPU available — SIMD-only results shown above.");
    }
}
