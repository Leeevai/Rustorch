use crate::tensor::Tensor;
use crate::error::TensorResult;
use crate::simd::SimdProcessor;
use rayon::prelude::*;
use std::sync::Arc;

pub enum ComputeMode {
    Single,
    MultiThread,
    SimdMultiThread,
}

pub struct TensorOps {
    simd_processor: Arc<SimdProcessor>,
    chunk_size: usize,
}

impl TensorOps {
    pub fn new() -> Self {
        let chunk_size = std::cmp::max(1000, num_cpus::get() * 100);
        TensorOps {
            simd_processor: Arc::new(SimdProcessor::new()),
            chunk_size,
        }
    }

    pub fn add(&self, a: &Tensor, b: &Tensor, mode: ComputeMode) -> TensorResult<Tensor> {
        a.check_same_shape(b)?;
        let mut result = Tensor::zeros(&a.shape);

        match mode {
            ComputeMode::Single => {
                for i in 0..a.data.len() {
                    result.data[i] = a.data[i] + b.data[i];
                }
            }
            ComputeMode::MultiThread => {
                result.data.par_iter_mut()
                    .enumerate()
                    .for_each(|(i, r)| *r = a.data[i] + b.data[i]);
            }
            ComputeMode::SimdMultiThread => {
                let processor = Arc::clone(&self.simd_processor);
                result.data.par_chunks_mut(self.chunk_size)
                    .enumerate()
                    .for_each(|(chunk_idx, chunk)| {
                        let start = chunk_idx * self.chunk_size;
                        let end = std::cmp::min(start + chunk.len(), a.data.len());
                        let len = end - start;
                        
                        processor.add_slice(
                            &a.data[start..start + len],
                            &b.data[start..start + len],
                            &mut chunk[..len]
                        );
                    });
            }
        }

        Ok(result)
    }

    pub fn multiply(&self, a: &Tensor, b: &Tensor, mode: ComputeMode) -> TensorResult<Tensor> {
        a.check_same_shape(b)?;
        let mut result = Tensor::zeros(&a.shape);

        match mode {
            ComputeMode::Single => {
                for i in 0..a.data.len() {
                    result.data[i] = a.data[i] * b.data[i];
                }
            }
            ComputeMode::MultiThread => {
                result.data.par_iter_mut()
                    .enumerate()
                    .for_each(|(i, r)| *r = a.data[i] * b.data[i]);
            }
            ComputeMode::SimdMultiThread => {
                let processor = Arc::clone(&self.simd_processor);
                result.data.par_chunks_mut(self.chunk_size)
                    .enumerate()
                    .for_each(|(chunk_idx, chunk)| {
                        let start = chunk_idx * self.chunk_size;
                        let end = std::cmp::min(start + chunk.len(), a.data.len());
                        let len = end - start;
                        
                        processor.mul_slice(
                            &a.data[start..start + len],
                            &b.data[start..start + len],
                            &mut chunk[..len]
                        );
                    });
            }
        }

        Ok(result)
    }

    pub fn scalar_multiply(&self, tensor: &Tensor, scalar: f32, mode: ComputeMode) -> Tensor {
        let mut result = Tensor::zeros(&tensor.shape);

        match mode {
            ComputeMode::Single => {
                for i in 0..tensor.data.len() {
                    result.data[i] = tensor.data[i] * scalar;
                }
            }
            ComputeMode::MultiThread => {
                result.data.par_iter_mut()
                    .enumerate()
                    .for_each(|(i, r)| *r = tensor.data[i] * scalar);
            }
            ComputeMode::SimdMultiThread => {
                let scalar_vec = vec![scalar; tensor.data.len()];
                let processor = Arc::clone(&self.simd_processor);
                result.data.par_chunks_mut(self.chunk_size)
                    .enumerate()
                    .for_each(|(chunk_idx, chunk)| {
                        let start = chunk_idx * self.chunk_size;
                        let end = std::cmp::min(start + chunk.len(), tensor.data.len());
                        let len = end - start;
                        
                        processor.mul_slice(
                            &tensor.data[start..start + len],
                            &scalar_vec[start..start + len],
                            &mut chunk[..len]
                        );
                    });
            }
        }

        result
    }

    pub fn sum(&self, tensor: &Tensor, mode: ComputeMode) -> f32 {
        match mode {
            ComputeMode::Single => {
                tensor.data.iter().sum()
            }
            ComputeMode::MultiThread | ComputeMode::SimdMultiThread => {
                tensor.data.par_iter().sum()
            }
        }
    }

    pub fn get_simd_info(&self) -> (usize, bool, bool) {
        (
            self.simd_processor.simd_width,
            self.simd_processor.supports_avx2,
            self.simd_processor.supports_avx512,
        )
    }
}