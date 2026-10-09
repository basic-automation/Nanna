#![warn(clippy::all)]
#![warn(clippy::pedantic, clippy::nursery)]
// Raised for the same reason as `nanna-daemon`'s: proving a future or closure
// is `Send` walks into wgpu's `Global`/`Hub`/`Registry` graph by way of
// `CosineSimilaritySearch`, which is deeper than the default limit of 128.
// nightly-2026-08-25 turned that overflow into the future-incompatible
// `recursion_depth_exceeding_limit` warning (rust#159228), which is scheduled
// to become a hard error. Solver depth only — no behaviour, no codegen change.
#![recursion_limit = "256"]

//! GPU-accelerated compute for Nanna using wgpu
//!
//! Provides GPU acceleration for embedding operations, matrix multiplication,
//! and other compute-heavy tasks.

use bytemuck::Pod;
use std::sync::Arc;
use thiserror::Error;
use tracing::info;

pub mod memory_manager;

pub use memory_manager::{BatchedSearch, GpuMemoryStats, GpuVectorStore};

#[derive(Error, Debug)]
pub enum GpuError {
    #[error("No suitable GPU adapter found")]
    NoAdapter,
    #[error("Failed to request device: {0}")]
    DeviceRequest(#[from] wgpu::RequestDeviceError),
    #[error("Shader compilation error: {0}")]
    ShaderCompilation(String),
    #[error("Buffer mapping failed")]
    BufferMapping,
    #[error("GPU memory insufficient: {0}")]
    InsufficientMemory(String),
    #[error("Invalid search input: {0}")]
    InvalidInput(String),
}

/// GPU compute context
pub struct GpuContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub adapter_info: wgpu::AdapterInfo,
}

impl GpuContext {
    /// Initialize GPU context, preferring high-performance discrete GPU.
    ///
    /// # Errors
    ///
    /// Returns `GpuError::NoAdapter` if no GPU adapter is found.
    /// Returns `GpuError::DeviceCreation` if the GPU device cannot be created.
    pub async fn new() -> Result<Self, GpuError> {
        // wgpu 30: Instance::default() == all backends (what the old explicit
        // InstanceDescriptor { backends: all(), .. } spelled out).
        let instance = wgpu::Instance::default();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(|_| GpuError::NoAdapter)?;

        let adapter_info = adapter.get_info();
        info!(
            "GPU: {} ({:?})",
            adapter_info.name, adapter_info.backend
        );

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("nanna-gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            })
            .await?;

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter_info,
        })
    }

    /// Check if GPU supports compute shaders
    #[must_use] 
    pub const fn supports_compute(&self) -> bool {
        true // wgpu always supports compute on valid adapters
    }
}

/// GPU buffer for compute operations
pub struct GpuBuffer {
    _buffer: wgpu::Buffer,
    _size: u64,
}

impl GpuBuffer {
    /// Create a new GPU buffer from CPU data
    pub fn from_slice<T: Pod>(ctx: &GpuContext, data: &[T], usage: wgpu::BufferUsages) -> Self {
        let buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("compute_buffer"),
            contents: bytemuck::cast_slice(data),
            usage,
        });

        Self {
            _buffer: buffer,
            _size: std::mem::size_of_val(data) as u64,
        }
    }

    /// Create an empty buffer for output
    #[must_use] 
    pub fn empty(ctx: &GpuContext, size: u64, usage: wgpu::BufferUsages) -> Self {
        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output_buffer"),
            size,
            usage,
            mapped_at_creation: false,
        });

        Self { _buffer: buffer, _size: size }
    }
}

/// Dot product compute shader (WGSL)
const _DOT_PRODUCT_SHADER: &str = r"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

var<workgroup> partial_sums: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let idx = global_id.x;
    let local_idx = local_id.x;
    
    // Each thread computes one multiplication
    if (idx < arrayLength(&a)) {
        partial_sums[local_idx] = a[idx] * b[idx];
    } else {
        partial_sums[local_idx] = 0.0;
    }
    
    workgroupBarrier();
    
    // Parallel reduction
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (local_idx < stride) {
            partial_sums[local_idx] += partial_sums[local_idx + stride];
        }
        workgroupBarrier();
    }
    
    // First thread writes workgroup result
    if (local_idx == 0u) {
        result[workgroup_id.x] = partial_sums[0];
    }
}
";

/// Cosine similarity batch compute shader
const COSINE_SIMILARITY_BATCH_SHADER: &str = r"
struct Params {
    query_len: u32,
    num_vectors: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> query: array<f32>;
@group(0) @binding(2) var<storage, read> vectors: array<f32>;
@group(0) @binding(3) var<storage, read_write> similarities: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let vec_idx = global_id.x;
    if (vec_idx >= params.num_vectors) {
        return;
    }
    
    let offset = vec_idx * params.query_len;
    var dot: f32 = 0.0;
    var norm_q: f32 = 0.0;
    var norm_v: f32 = 0.0;
    
    for (var i = 0u; i < params.query_len; i++) {
        let q = query[i];
        let v = vectors[offset + i];
        dot += q * v;
        norm_q += q * q;
        norm_v += v * v;
    }
    
    similarities[vec_idx] = dot / (sqrt(norm_q) * sqrt(norm_v));
}
";

/// GPU-accelerated cosine similarity search
pub struct CosineSimilaritySearch {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

/// Parameters struct for the compute shader (must match WGSL layout)
#[repr(C)]
#[derive(Clone, Copy, Pod, bytemuck::Zeroable)]
struct SimilarityParams {
    query_len: u32,
    num_vectors: u32,
}

impl CosineSimilaritySearch {
    /// Create a new cosine similarity search pipeline.
    ///
    /// # Errors
    ///
    /// Returns `GpuError` if the compute pipeline cannot be created.
    pub fn new(ctx: &GpuContext) -> Result<Self, GpuError> {
        let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cosine_similarity_shader"),
            source: wgpu::ShaderSource::Wgsl(COSINE_SIMILARITY_BATCH_SHADER.into()),
        });

        let bind_group_layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cosine_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cosine_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            ..Default::default()
        });

        let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cosine_similarity_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            pipeline,
            bind_group_layout,
        })
    }

    /// Compute cosine similarity between a query vector and a batch of vectors.
    ///
    /// Returns a vector of similarity scores, one per input vector.
    ///
    /// The vectors are scored in as many dispatches as the device's limits
    /// require: one storage binding may not exceed
    /// `max_storage_buffer_binding_size`, and one dispatch may not launch more
    /// than `max_compute_workgroups_per_dimension` workgroups. A single dispatch
    /// past either limit is a wgpu validation error, which wgpu's default
    /// handler turns into a panic — under `panic = "abort"`, the process.
    ///
    /// # Arguments
    ///
    /// * `ctx` - GPU context
    /// * `query` - Query vector (normalized)
    /// * `vectors` - Batch of vectors to compare against (flattened, each same length as query)
    ///
    /// # Errors
    ///
    /// Returns `GpuError::BufferMapping` if a result buffer cannot be read,
    /// `GpuError::InvalidInput` if `vectors` is not a whole number of
    /// query-length vectors, and `GpuError::InsufficientMemory` if the counts do
    /// not fit the shader's `u32` parameters or one vector alone exceeds a
    /// storage binding.
    pub async fn search(
        &self,
        ctx: &GpuContext,
        query: &[f32],
        vectors: &[f32],
    ) -> Result<Vec<f32>, GpuError> {
        let (query_len, num_vectors) = shader_counts(query.len(), vectors.len())?;

        if num_vectors == 0 {
            return Ok(vec![]);
        }

        let per_dispatch = vectors_per_dispatch(query_len, &ctx.device.limits())?;
        self.search_in_dispatches(ctx, query, vectors, per_dispatch).await
    }

    /// Score `vectors` against `query`, at most `per_dispatch` vectors a dispatch.
    ///
    /// The query buffer is uploaded once and bound by every dispatch.
    async fn search_in_dispatches(
        &self,
        ctx: &GpuContext,
        query: &[f32],
        vectors: &[f32],
        per_dispatch: u32,
    ) -> Result<Vec<f32>, GpuError> {
        assert!(per_dispatch > 0, "a dispatch must score at least one vector");
        assert!(!query.is_empty() && vectors.len().is_multiple_of(query.len()));
        let vector_count = vectors.len() / query.len();
        let floats_per_dispatch = usize::try_from(per_dispatch)
            .map_or(usize::MAX, |count| count.saturating_mul(query.len()));

        let query_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("query_buffer"),
            contents: bytemuck::cast_slice(query),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let mut similarities = Vec::with_capacity(vector_count);
        for chunk in vectors.chunks(floats_per_dispatch) {
            let scores = self.dispatch(ctx, &query_buffer, query.len(), chunk).await?;
            similarities.extend_from_slice(&scores);
        }

        debug_assert_eq!(similarities.len(), vector_count);
        Ok(similarities)
    }

    /// One dispatch: score every vector of `chunk`, which fits the device's limits.
    async fn dispatch(
        &self,
        ctx: &GpuContext,
        query_buffer: &wgpu::Buffer,
        query_len: usize,
        chunk: &[f32],
    ) -> Result<Vec<f32>, GpuError> {
        let (query_len, num_vectors) = shader_counts(query_len, chunk.len())?;
        debug_assert!(num_vectors > 0, "a dispatch is never empty");
        debug_assert!(num_vectors.div_ceil(WORKGROUP_SIZE) <= ctx.device.limits().max_compute_workgroups_per_dimension);

        let params = SimilarityParams {
            query_len,
            num_vectors,
        };
        let params_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params_buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let vectors_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vectors_buffer"),
            contents: bytemuck::cast_slice(chunk),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_size = u64::from(num_vectors) * F32_BYTES;
        let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output_buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cosine_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: query_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: vectors_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: output_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cosine_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cosine_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(num_vectors.div_ceil(WORKGROUP_SIZE), 1, 1);
        }

        read_back(ctx, encoder, &output_buffer, output_size).await
    }
}

/// Invocations per workgroup — the shader's `@workgroup_size(64)`.
const WORKGROUP_SIZE: u32 = 64;

/// Bytes in one `f32` lane of every buffer the shader binds.
const F32_BYTES: u64 = 4;

/// Copy `output` into a staging buffer, submit `encoder`, and read the scores back.
async fn read_back(
    ctx: &GpuContext,
    mut encoder: wgpu::CommandEncoder,
    output: &wgpu::Buffer,
    output_size: u64,
) -> Result<Vec<f32>, GpuError> {
    debug_assert!(output_size > 0 && output_size.is_multiple_of(F32_BYTES));
    let staging_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging_buffer"),
        size: output_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(output, 0, &staging_buffer, 0, output_size);
    ctx.queue.submit(std::iter::once(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (tx, rx) = futures::channel::oneshot::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });

    // wgpu 30: Maintain → PollType; poll now returns a Result.
    ctx.device
        .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
        .map_err(|_| GpuError::BufferMapping)?;

    rx.await
        .map_err(|_| GpuError::BufferMapping)?
        .map_err(|_| GpuError::BufferMapping)?;

    // wgpu 30: get_mapped_range returns a Result.
    let data = buffer_slice
        .get_mapped_range()
        .map_err(|_| GpuError::BufferMapping)?;
    let results: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging_buffer.unmap();

    debug_assert_eq!(results.len() as u64 * F32_BYTES, output_size);
    Ok(results)
}

/// The query length and vector count as the `u32`s the shader indexes with.
///
/// A count past `u32::MAX` would need a storage binding of more than 16 GiB,
/// which no adapter's `max_storage_buffer_binding_size` allows, so refusing it
/// replaces a silent truncation that could never reach a valid dispatch anyway.
/// A trailing partial vector is refused too: it used to be dropped without a
/// word, shifting nothing but scoring nothing.
fn shader_counts(query_len: usize, vectors_len: usize) -> Result<(u32, u32), GpuError> {
    // An empty query has no dimensions to compare, so there are no vectors to
    // score; `vectors_len / 0` used to panic instead.
    let num_vectors = vectors_len.checked_div(query_len).unwrap_or(0);
    if query_len > 0 && !vectors_len.is_multiple_of(query_len) {
        return Err(GpuError::InvalidInput(format!(
            "{vectors_len} floats is not a whole number of {query_len}-wide vectors"
        )));
    }
    match (u32::try_from(query_len), u32::try_from(num_vectors)) {
        (Ok(query_len), Ok(num_vectors)) => Ok((query_len, num_vectors)),
        _ => Err(GpuError::InsufficientMemory(format!(
            "search input exceeds the shader's u32 indexing: query length {query_len}, {num_vectors} vectors"
        ))),
    }
}

/// How many `query_len`-wide vectors one dispatch may score on a device with `limits`.
///
/// Three limits bind, and the smallest wins: the vectors' storage binding and
/// the output's (both `max_storage_buffer_binding_size`, and no buffer past
/// `max_buffer_size`), and the workgroup count (`max_compute_workgroups_per_dimension`
/// × [`WORKGROUP_SIZE`] invocations, one per vector). On wgpu's default limits
/// a 1536-wide store is bound by its binding (21 845 vectors a dispatch) and a
/// 1-wide one by its workgroups (4 194 240).
fn vectors_per_dispatch(query_len: u32, limits: &wgpu::Limits) -> Result<u32, GpuError> {
    assert!(query_len > 0, "an empty query scores nothing and dispatches nothing");
    let binding_bytes = limits.max_storage_buffer_binding_size.min(limits.max_buffer_size);
    let vector_bytes = u64::from(query_len) * F32_BYTES;
    if vector_bytes > binding_bytes {
        return Err(GpuError::InsufficientMemory(format!(
            "one {query_len}-wide vector is {vector_bytes} bytes; this device binds at most {binding_bytes}"
        )));
    }

    let by_vectors = binding_bytes / vector_bytes;
    let by_output = binding_bytes / F32_BYTES;
    let by_workgroups = u64::from(limits.max_compute_workgroups_per_dimension) * u64::from(WORKGROUP_SIZE);
    let count = by_vectors.min(by_output).min(by_workgroups);

    debug_assert!(count >= 1, "a vector that fits its binding fits a dispatch");
    Ok(u32::try_from(count).unwrap_or(u32::MAX))
}

// wgpu buffer init descriptor helper
mod wgpu {
    pub use ::wgpu::*;
    
    pub mod util {
        use super::BufferUsages;
        
        pub struct BufferInitDescriptor<'a> {
            pub label: Option<&'a str>,
            pub contents: &'a [u8],
            pub usage: BufferUsages,
        }
    }
}

trait DeviceExt {
    fn create_buffer_init(&self, desc: &wgpu::util::BufferInitDescriptor) -> wgpu::Buffer;
}

impl DeviceExt for wgpu::Device {
    fn create_buffer_init(&self, desc: &wgpu::util::BufferInitDescriptor) -> wgpu::Buffer {
        let unpadded_size = desc.contents.len() as u64;
        let padding = (4 - (unpadded_size % 4)) % 4;
        let padded_size = unpadded_size + padding;
        
        let buffer = self.create_buffer(&wgpu::BufferDescriptor {
            label: desc.label,
            size: padded_size,
            usage: desc.usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        
        // wgpu 30: get_mapped_range_mut returns a Result, and BufferViewMut is
        // written via slice().copy_from_slice (no direct indexing). A buffer
        // created with `mapped_at_creation: true` is mapped by construction —
        // failure here is a programmer error, so the expect is an invariant assert.
        buffer
            .slice(..)
            .get_mapped_range_mut()
            .expect("buffer created with mapped_at_creation must be mappable")
            .slice(..desc.contents.len())
            .copy_from_slice(desc.contents);
        buffer.unmap();
        
        buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_query_scores_nothing_instead_of_dividing_by_zero() {
        assert_eq!(shader_counts(0, 12).unwrap(), (0, 0));
        assert_eq!(shader_counts(3, 12).unwrap(), (3, 4));
    }

    #[test]
    fn a_trailing_partial_vector_is_refused_not_dropped() {
        assert!(matches!(shader_counts(3, 13), Err(GpuError::InvalidInput(_))));
        assert_eq!(shader_counts(3, 0).unwrap(), (3, 0));
    }

    #[test]
    fn a_dispatch_holds_what_the_smallest_limit_admits() {
        let limits = wgpu::Limits::default();
        let binding = limits.max_storage_buffer_binding_size.min(limits.max_buffer_size);
        let workgroups = u64::from(limits.max_compute_workgroups_per_dimension) * u64::from(WORKGROUP_SIZE);

        // Wide vectors: the vectors' binding binds.
        let wide = vectors_per_dispatch(1536, &limits).unwrap();
        assert_eq!(u64::from(wide), binding / (1536 * F32_BYTES));
        // Narrow vectors: the workgroup count binds.
        let narrow = vectors_per_dispatch(1, &limits).unwrap();
        assert_eq!(u64::from(narrow), workgroups.min(binding / F32_BYTES));
        assert!(u64::from(narrow).div_ceil(u64::from(WORKGROUP_SIZE)) <= u64::from(limits.max_compute_workgroups_per_dimension));
    }

    #[test]
    fn a_vector_wider_than_a_binding_is_refused() {
        let limits = wgpu::Limits { max_storage_buffer_binding_size: 64, ..wgpu::Limits::default() };
        assert_eq!(vectors_per_dispatch(16, &limits).unwrap(), 1);
        assert!(matches!(vectors_per_dispatch(17, &limits), Err(GpuError::InsufficientMemory(_))));
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (norm(a) * norm(b))
    }

    /// Deterministic, non-degenerate vectors: component `i` of vector `n`.
    fn fixture(count: usize, width: usize) -> Vec<f32> {
        (0..count * width)
            .map(|i| {
                let lane = u16::try_from(i % 997).unwrap_or(0);
                f32::from(lane) / 997.0 + 0.01
            })
            .collect()
    }

    async fn gpu() -> Option<(GpuContext, CosineSimilaritySearch)> {
        match GpuContext::new().await {
            Ok(ctx) => {
                let search = CosineSimilaritySearch::new(&ctx).ok()?;
                Some((ctx, search))
            }
            Err(GpuError::NoAdapter) => {
                println!("No GPU adapter found, skipping test");
                None
            }
            Err(e) => panic!("Unexpected error: {e}"),
        }
    }

    #[tokio::test]
    async fn scores_split_across_dispatches_match_the_cpu() {
        let Some((ctx, search)) = gpu().await else { return };
        let width = 5;
        let query = fixture(1, width);
        let vectors = fixture(10, width);

        // Three vectors a dispatch: two full dispatches and a partial one.
        let scores = search.search_in_dispatches(&ctx, &query, &vectors, 3).await.unwrap();
        assert_eq!(scores.len(), 10);
        for (score, vector) in scores.iter().zip(vectors.chunks(width)) {
            assert!((score - cosine(&query, vector)).abs() < 1e-5);
        }
    }

    #[tokio::test]
    async fn a_store_past_the_binding_limit_is_scored_in_full() {
        let Some((ctx, search)) = gpu().await else { return };
        let width = 4;
        let binding = ctx.device.limits().max_storage_buffer_binding_size;
        // One vector more than a single storage binding holds: one dispatch
        // was a validation error, i.e. a panic.
        let count = usize::try_from(binding / (4 * F32_BYTES)).unwrap() + 1;
        let query = fixture(1, width);
        let vectors = fixture(count, width);

        let scores = search.search(&ctx, &query, &vectors).await.unwrap();
        assert_eq!(scores.len(), count);
        let last = &vectors[(count - 1) * width..];
        assert!((scores[count - 1] - cosine(&query, last)).abs() < 1e-5);
    }

    #[tokio::test]
    async fn a_store_past_the_workgroup_limit_is_scored_in_full() {
        let Some((ctx, search)) = gpu().await else { return };
        let limit = ctx.device.limits().max_compute_workgroups_per_dimension;
        // 1-wide vectors: one more than a dispatch can launch invocations for.
        let count = usize::try_from(u64::from(limit) * u64::from(WORKGROUP_SIZE)).unwrap() + 1;
        let query = vec![1.0_f32];
        let vectors = fixture(count, 1);

        let scores = search.search(&ctx, &query, &vectors).await.unwrap();
        assert_eq!(scores.len(), count);
        assert!(scores.iter().all(|score| (score - 1.0).abs() < 1e-5));
    }

    #[tokio::test]
    async fn test_gpu_context_creation() {
        // Skip if no GPU available (CI environments)
        match GpuContext::new().await {
            Ok(ctx) => {
                println!("GPU: {}", ctx.adapter_info.name);
                assert!(ctx.supports_compute());
            }
            Err(GpuError::NoAdapter) => {
                println!("No GPU adapter found, skipping test");
            }
            Err(e) => panic!("Unexpected error: {e}"),
        }
    }
}
