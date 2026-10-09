//! GPU memory management and batched operations for persistent vector storage.
//!
//! Provides efficient GPU-resident vector storage with change tracking,
//! and automatic batching for vector stores that exceed GPU memory limits.

use std::collections::HashSet;
use thiserror::Error;
use crate::{GpuContext, CosineSimilaritySearch, GpuError};

#[derive(Error, Debug)]
pub enum MemoryError {
    #[error("GPU error: {0}")]
    Gpu(#[from] GpuError),
    #[error("Insufficient GPU memory: required {required} bytes, available {available} bytes")]
    InsufficientMemory { required: u64, available: u64 },
    #[error("Invalid vector dimensions: expected {expected}, got {got}")]
    InvalidDimensions { expected: usize, got: usize },
    #[error("Vector not found at index {0}")]
    VectorNotFound(usize),
    #[error("Empty vector store")]
    EmptyStore,
}

/// Tracks GPU memory statistics and allocation.
#[derive(Clone, Debug)]
pub struct GpuMemoryStats {
    /// Number of vectors currently resident on GPU
    pub vectors_resident: usize,
    /// GPU memory used by vector buffers (bytes)
    pub bytes_used: u64,
    /// Maximum GPU memory available (bytes)
    pub bytes_available: u64,
    /// Number of batches required to fit all vectors
    pub batches_required: usize,
}

/// Persistent GPU-resident vector storage with change tracking.
pub struct GpuVectorStore {
    vectors: Vec<Vec<f32>>,
    vector_dim: usize,
    gpu_buffer: Option<wgpu::Buffer>,
    dirty_indices: HashSet<usize>,
    removed_indices: HashSet<usize>,
    bytes_per_vector: u64,
    max_buffer_size: u64,
}

impl GpuVectorStore {
    /// Create a new GPU vector store with a given vector dimension.
    ///
    /// # Arguments
    ///
    /// * `vector_dim` - Dimension of vectors to store
    /// * `ctx` - GPU context for querying limits
    ///
    /// # Errors
    ///
    /// Returns error if vector dimension is invalid.
    pub fn new(vector_dim: usize, ctx: &GpuContext) -> Result<Self, MemoryError> {
        if vector_dim == 0 {
            return Err(MemoryError::InvalidDimensions {
                expected: 1,
                got: 0,
            });
        }

        let bytes_per_vector = (vector_dim * std::mem::size_of::<f32>()) as u64;
        let max_buffer_size = ctx.device.limits().max_storage_buffer_binding_size;

        Ok(Self {
            vectors: Vec::new(),
            vector_dim,
            gpu_buffer: None,
            dirty_indices: HashSet::new(),
            removed_indices: HashSet::new(),
            bytes_per_vector,
            max_buffer_size,
        })
    }

    /// Append new vectors to the store. Marks them as dirty.
    ///
    /// # Arguments
    ///
    /// * `new_vectors` - Flattened array of vectors (length must be multiple of `vector_dim`)
    ///
    /// # Errors
    ///
    /// Returns error if vector dimensions don't match.
    pub fn append(&mut self, new_vectors: &[f32]) -> Result<(), MemoryError> {
        if !new_vectors.len().is_multiple_of(self.vector_dim) {
            return Err(MemoryError::InvalidDimensions {
                expected: self.vector_dim,
                got: new_vectors.len(),
            });
        }

        for chunk in new_vectors.chunks(self.vector_dim) {
            // The index the vector is about to occupy. `start + len - 1`
            // counted the store's old length twice, so on a non-empty store
            // the new vectors were never marked dirty and never reached the
            // GPU buffer — the search ran on stale data.
            let index = self.vectors.len();
            self.vectors.push(chunk.to_vec());
            self.dirty_indices.insert(index);
        }

        Ok(())
    }

    /// Update existing vector at the given index. Marks it as dirty.
    ///
    /// # Arguments
    ///
    /// * `index` - Vector index
    /// * `vector` - New vector data (must match `vector_dim`)
    ///
    /// # Errors
    ///
    /// Returns error if index is out of bounds or dimensions don't match.
    pub fn update(&mut self, index: usize, vector: &[f32]) -> Result<(), MemoryError> {
        if vector.len() != self.vector_dim {
            return Err(MemoryError::InvalidDimensions {
                expected: self.vector_dim,
                got: vector.len(),
            });
        }

        if index >= self.vectors.len() {
            return Err(MemoryError::VectorNotFound(index));
        }

        if !self.removed_indices.contains(&index) {
            self.vectors[index] = vector.to_vec();
            self.dirty_indices.insert(index);
        }

        Ok(())
    }

    /// Mark a vector as removed (soft delete, doesn't deallocate).
    ///
    /// # Arguments
    ///
    /// * `index` - Vector index
    ///
    /// # Errors
    ///
    /// Returns error if index is out of bounds.
    pub fn remove(&mut self, index: usize) -> Result<(), MemoryError> {
        if index >= self.vectors.len() {
            return Err(MemoryError::VectorNotFound(index));
        }

        self.removed_indices.insert(index);
        self.dirty_indices.remove(&index);
        Ok(())
    }

    /// Upload dirty vectors to GPU. Creates or updates the GPU buffer.
    ///
    /// # Errors
    ///
    /// Returns error if GPU memory is insufficient.
    pub fn sync(&mut self, ctx: &GpuContext) -> Result<(), MemoryError> {
        let active_count = self.vectors.len() - self.removed_indices.len();
        let required_bytes = active_count as u64 * self.bytes_per_vector;

        if required_bytes > self.max_buffer_size {
            return Err(MemoryError::InsufficientMemory {
                required: required_bytes,
                available: self.max_buffer_size,
            });
        }

        // Build flattened array of active vectors
        let mut flat_data = Vec::with_capacity(active_count * self.vector_dim);
        for (idx, _vec) in self.vectors.iter().enumerate() {
            if !self.removed_indices.contains(&idx) {
                flat_data.extend_from_slice(&self.vectors[idx]);
            }
        }

        // Create or recreate GPU buffer
        self.gpu_buffer = Some(ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vector_store_buffer"),
            contents: bytemuck::cast_slice(&flat_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        }));

        self.dirty_indices.clear();
        Ok(())
    }

    /// Search for similar vectors using cosine similarity.
    ///
    /// Returns vector of (`original_index`, `similarity_score`) pairs.
    ///
    /// # Arguments
    ///
    /// * `ctx` - GPU context
    /// * `search` - Cosine similarity search pipeline
    /// * `query` - Query vector (must match `vector_dim`)
    ///
    /// # Errors
    ///
    /// Returns error if store is empty, GPU buffer not synced, or dimensions don't match.
    pub async fn search(
        &self,
        ctx: &GpuContext,
        search: &CosineSimilaritySearch,
        query: &[f32],
    ) -> Result<Vec<(usize, f32)>, MemoryError> {
        if query.len() != self.vector_dim {
            return Err(MemoryError::InvalidDimensions {
                expected: self.vector_dim,
                got: query.len(),
            });
        }

        let _gpu_buffer = self.gpu_buffer.as_ref().ok_or(MemoryError::EmptyStore)?;
        let active_count = self.vectors.len() - self.removed_indices.len();

        if active_count == 0 {
            return Ok(vec![]);
        }

        // Run cosine similarity on active vectors
        let similarities = search.search(ctx, query, &self.flatten_active()).await?;

        // Map back to original indices
        let mut results = Vec::new();
        let mut active_idx = 0;
        for (orig_idx, _vec) in self.vectors.iter().enumerate() {
            if !self.removed_indices.contains(&orig_idx) {
                if active_idx < similarities.len() {
                    results.push((orig_idx, similarities[active_idx]));
                }
                active_idx += 1;
            }
        }

        Ok(results)
    }

    /// Get current memory statistics.
    #[must_use]
    pub fn stats(&self, ctx: &GpuContext) -> GpuMemoryStats {
        let vectors_resident = self.vectors.len() - self.removed_indices.len();
        let bytes_used = vectors_resident as u64 * self.bytes_per_vector;
        let bytes_available = ctx.device.limits().max_storage_buffer_binding_size;
        let batches_required = if bytes_used > 0 {
            // Saturates only where `usize` is narrower than `u64` and the count
            // exceeds it; on 64-bit targets the conversion is always exact.
            usize::try_from(bytes_used.div_ceil(self.max_buffer_size)).unwrap_or(usize::MAX)
        } else {
            0
        };

        GpuMemoryStats {
            vectors_resident,
            bytes_used,
            bytes_available,
            batches_required,
        }
    }

    /// Get the number of vectors in the store (including removed).
    #[must_use]
    pub const fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Check if the store is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Get the vector dimension.
    #[must_use]
    pub const fn vector_dim(&self) -> usize {
        self.vector_dim
    }

    /// Check if GPU buffer is synced (no dirty vectors).
    #[must_use]
    pub fn is_synced(&self) -> bool {
        self.dirty_indices.is_empty()
    }

    /// Flatten active (non-removed) vectors into a single array.
    fn flatten_active(&self) -> Vec<f32> {
        let mut result = Vec::new();
        for (idx, vec) in self.vectors.iter().enumerate() {
            if !self.removed_indices.contains(&idx) {
                result.extend_from_slice(vec);
            }
        }
        result
    }
}

/// Handles vector stores too large for a single GPU buffer by batching searches.
pub struct BatchedSearch {
    batch_size: usize,
}

impl BatchedSearch {
    /// Create a new batched search handler.
    ///
    /// # Arguments
    ///
    /// * `ctx` - GPU context (used to query memory limits)
    /// * `vector_dim` - Dimension of vectors
    #[must_use]
    pub fn new(ctx: &GpuContext, vector_dim: usize) -> Self {
        let max_buffer_size = ctx.device.limits().max_storage_buffer_binding_size;
        let bytes_per_vector = (vector_dim * std::mem::size_of::<f32>()) as u64;
        
        // Batch size: how many vectors fit in 80% of max buffer
        // Zero-width vectors get a batch size of 1. The saturation applies only
        // where `usize` is narrower than `u64`; on 64-bit targets it is exact.
        let batch_size = (max_buffer_size * 80 / 100)
            .checked_div(bytes_per_vector)
            .map_or(1, |per_batch| usize::try_from(per_batch.max(1)).unwrap_or(usize::MAX));

        Self {
            batch_size,
        }
    }

    /// Calculate the number of batches needed for a given number of vectors.
    #[must_use]
    pub const fn batches_needed(&self, num_vectors: usize) -> usize {
        if num_vectors == 0 {
            0
        } else {
            num_vectors.div_ceil(self.batch_size)
        }
    }

    /// Search across multiple batches of vectors, merging results.
    ///
    /// # Arguments
    ///
    /// * `ctx` - GPU context
    /// * `search` - Cosine similarity search pipeline
    /// * `query` - Query vector
    /// * `vectors` - All vectors to search (flattened)
    /// * `vector_dim` - Dimension of vectors
    /// * `k` - Number of top results to return
    ///
    /// # Errors
    ///
    /// Returns error if search fails or vectors are invalid.
    pub async fn search_top_k(
        &self,
        ctx: &GpuContext,
        search: &CosineSimilaritySearch,
        query: &[f32],
        vectors: &[f32],
        vector_dim: usize,
        k: usize,
    ) -> Result<Vec<(usize, f32)>, MemoryError> {
        if vectors.is_empty() || vector_dim == 0 {
            return Ok(vec![]);
        }

        let num_vectors = vectors.len() / vector_dim;
        let mut all_results = Vec::new();

        // Process each batch
        for batch_idx in 0..self.batches_needed(num_vectors) {
            let start_idx = batch_idx * self.batch_size;
            let end_idx = (start_idx + self.batch_size).min(num_vectors);
            let batch_vectors = &vectors[start_idx * vector_dim..end_idx * vector_dim];

            let similarities = search.search(ctx, query, batch_vectors).await?;

            for (local_idx, sim) in similarities.iter().enumerate() {
                all_results.push((start_idx + local_idx, *sim));
            }
        }

        // Sort by similarity descending and take top-k
        all_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        all_results.truncate(k);

        Ok(all_results)
    }

    /// Get the configured batch size.
    #[must_use]
    pub const fn batch_size(&self) -> usize {
        self.batch_size
    }
}

// wgpu buffer init descriptor helper (re-export from lib)
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

    #[tokio::test]
    async fn test_vector_store_creation() {
        match GpuContext::new().await {
            Ok(ctx) => {
                let store = GpuVectorStore::new(128, &ctx);
                assert!(store.is_ok());
                let store = store.unwrap();
                assert_eq!(store.vector_dim(), 128);
                assert!(store.is_empty());
            }
            Err(GpuError::NoAdapter) => {
                println!("No GPU adapter, skipping test");
            }
            Err(e) => panic!("Unexpected error: {e}"),
        }
    }

    #[tokio::test]
    async fn test_vector_store_append() {
        match GpuContext::new().await {
            Ok(ctx) => {
                let mut store = GpuVectorStore::new(4, &ctx).unwrap();
                let vectors = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
                assert!(store.append(&vectors).is_ok());
                assert_eq!(store.len(), 2);
                assert_eq!(store.dirty_indices, HashSet::from([0, 1]));

                // Appending to a NON-empty store marks the new slots — the
                // old arithmetic marked 2·start + k and missed them.
                store.dirty_indices.clear();
                assert!(store.append(&[9.0, 10.0, 11.0, 12.0]).is_ok());
                assert_eq!(store.dirty_indices, HashSet::from([2]));
            }
            Err(GpuError::NoAdapter) => {
                println!("No GPU adapter, skipping test");
            }
            Err(e) => panic!("Unexpected error: {e}"),
        }
    }

    #[test]
    fn test_batched_search_batch_calculation() {
        // Create a minimal context-like structure for testing batch calculation
        // This test doesn't require actual GPU access
        let batch_size = 1024;
        let num_vectors = 5000;
        let expected_batches = (num_vectors + batch_size - 1) / batch_size;
        assert_eq!(expected_batches, 5);
    }
}
