#![warn(clippy::pedantic, clippy::nursery, clippy::all)]
// Solver depth only, as on the daemon crate roots: proving these futures
// `Send` walks the wgpu/daemon type graph past the default limit of 128.
#![recursion_limit = "256"]
//! Extended GPU vs SIMD benchmark — find the TRUE crossover point
//! Run with: cargo bench -p nanna-gpu --bench `gpu_vs_simd_extended`

use std::time::{Duration, Instant};

fn simd_batch_search(query: &[f32], vectors_flat: &[f32], dim: usize) -> Vec<f32> {
    vectors_flat
        .chunks_exact(dim)
        .map(|v| nanna_simd::cosine_similarity_f32(query, v))
        .collect()
}

async fn gpu_batch_search(
    pipeline: &nanna_gpu::CosineSimilaritySearch,
    ctx: &nanna_gpu::GpuContext,
    query: &[f32],
    vectors_flat: &[f32],
) -> Vec<f32> {
    pipeline.search(ctx, query, vectors_flat).await.unwrap()
}

/// Fast vector generation — flat buffer, no per-vector Vec allocation
fn generate_flat(count: usize, dim: usize) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut buf = Vec::with_capacity(count * dim);
    for i in 0..count {
        let mut norm_sq = 0.0f32;
        let start = buf.len();
        for j in 0..dim {
            let mut h = DefaultHasher::new();
            (i * dim + j).hash(&mut h);
            let val = nanna_numeric::f32_from_f64((nanna_numeric::f64_from_u64(h.finish()) / nanna_numeric::f64_from_u64(u64::MAX)).mul_add(2.0, -1.0));
            buf.push(val);
            norm_sq = val.mul_add(val, norm_sq);
        }
        let inv_norm = 1.0 / norm_sq.sqrt();
        for x in &mut buf[start..] {
            *x *= inv_norm;
        }
    }
    buf
}

fn bench_sync<F: FnMut()>(mut f: F, iters: usize) -> Duration {
    f(); // warmup
    let mut total = Duration::ZERO;
    for _ in 0..iters {
        let t = Instant::now();
        std::hint::black_box(&mut f)();
        total += t.elapsed();
    }
    total / u32::try_from(iters).expect("iteration count fits u32")
}

fn bench_async_fn<F, Fut>(rt: &tokio::runtime::Runtime, mut f: F, iters: usize) -> Duration
where F: FnMut() -> Fut, Fut: std::future::Future<Output = Vec<f32>>
{
    rt.block_on(f()); // warmup
    let mut total = Duration::ZERO;
    for _ in 0..iters {
        let t = Instant::now();
        std::hint::black_box(rt.block_on(f()));
        total += t.elapsed();
    }
    total / u32::try_from(iters).expect("iteration count fits u32")
}

fn fmt_dur(d: Duration) -> String {
    let us = d.as_secs_f64() * 1e6;
    if us < 1000.0 { format!("{us:>8.1}µs") }
    else { format!("{:>8.2}ms", us / 1000.0) }
}

fn main() {
    let rt = tokio::runtime::Runtime::new().unwrap();

    println!();
    println!("══════════════════════════════════════════════════════════════════");
    println!("  GPU vs SIMD Extended Benchmark — Finding True Crossover");
    println!("══════════════════════════════════════════════════════════════════");
    println!();
    println!("SIMD tier: {:?}", nanna_simd::simd_tier());

    let gpu = rt.block_on(async { nanna_gpu::GpuContext::new().await });
    let (ctx, pipeline) = match gpu {
        Ok(ctx) => {
            println!("GPU: {} ({:?})", ctx.adapter_info.name, ctx.adapter_info.backend);
            match nanna_gpu::CosineSimilaritySearch::new(&ctx) {
                Ok(p) => (ctx, p),
                Err(e) => { println!("Pipeline failed: {e}"); return; }
            }
        }
        Err(e) => { println!("No GPU: {e}"); return; }
    };
    println!();

    let dim = 768;
    // Key question: does GPU EVER win? Test 10k, 25k, 50k with single iteration
    let counts = [10_000, 25_000, 50_000];

    println!("  768-dim, 1 warmup + 2 iterations each");
    println!("──────────────────────────────────────────────────────────────────");
    println!("  Vectors │    SIMD mean │     GPU mean │  Ratio  │ Winner");
    println!("──────────┼─────────────┼──────────────┼─────────┼────────");

    let query = generate_flat(1, dim);

    for &count in &counts {
        let flat = generate_flat(count, dim);

        let simd_mean = bench_sync(
            || { std::hint::black_box(simd_batch_search(&query, &flat, dim)); },
            2,
        );

        let gpu_mean = bench_async_fn(
            &rt,
            || gpu_batch_search(&pipeline, &ctx, &query, &flat),
            2,
        );

        let ratio = gpu_mean.as_secs_f64() / simd_mean.as_secs_f64();
        let winner = if ratio < 1.0 { "GPU ✓" } else { "SIMD" };

        println!("  {:>7} │ {} │ {} │ {:>6.2}× │ {winner}",
            count, fmt_dur(simd_mean), fmt_dur(gpu_mean), ratio);
    }

    println!("──────────────────────────────────────────────────────────────────");
    println!();

    // Also test 1536-dim at 10k and 25k
    let dim2 = 1536;
    let counts2 = [10_000, 25_000];

    println!("  1536-dim spot check");
    println!("──────────────────────────────────────────────────────────────────");
    println!("  Vectors │    SIMD mean │     GPU mean │  Ratio  │ Winner");
    println!("──────────┼─────────────┼──────────────┼─────────┼────────");

    let query2 = generate_flat(1, dim2);

    for &count in &counts2 {
        let flat = generate_flat(count, dim2);

        let simd_mean = bench_sync(
            || { std::hint::black_box(simd_batch_search(&query2, &flat, dim2)); },
            2,
        );

        let gpu_mean = bench_async_fn(
            &rt,
            || gpu_batch_search(&pipeline, &ctx, &query2, &flat),
            2,
        );

        let ratio = gpu_mean.as_secs_f64() / simd_mean.as_secs_f64();
        let winner = if ratio < 1.0 { "GPU ✓" } else { "SIMD" };

        println!("  {:>7} │ {} │ {} │ {:>6.2}× │ {winner}",
            count, fmt_dur(simd_mean), fmt_dur(gpu_mean), ratio);
    }

    println!("──────────────────────────────────────────────────────────────────");
    println!();
    println!("══════════════════════════════════════════════════════════════════");
    println!("  Based on quick + extended results, set GPU_THRESHOLD accordingly");
    println!("══════════════════════════════════════════════════════════════════");
}