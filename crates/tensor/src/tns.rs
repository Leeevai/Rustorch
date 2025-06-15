use crate::error::{TensorError, TensorResult};
use crate::simd::SimdProcessor;
use rayon::prelude::*;
use std::sync::Arc;
use std::ops::{Add, Sub, Mul, Div};

#[derive(Debug, Clone)]
pub enum ComputeMode {
    Single,
    MultiThread,
    SimdMultiThread,
}

#[derive(Debug, Clone)]
pub struct Tensor {
    data: Vec<f32>,
    rank: usize,
    shape: Vec<usize>,
    simd_processor: Arc<SimdProcessor>,
    chunk_size: usize,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: &[usize]) -> TensorResult<Self> {
        let expected_size: usize = shape.iter().product();
        if data.len() != expected_size {
            return Err(TensorError::ShapeMismatch(format!(
                "Data length {} does not match shape {:?} (expected {})",
                data.len(), shape, expected_size
            )));
        }
        
        let chunk_size = std::cmp::max(1000, num_cpus::get() * 100);
        
        Ok(Tensor {
            data,
            shape: shape.to_vec(),
            rank: shape.len(),
            simd_processor: Arc::new(SimdProcessor::new()),
            chunk_size,
        })
    }

    pub fn zeros(shape: &[usize]) -> Self {
        let size: usize = shape.iter().product();
        let chunk_size = std::cmp::max(1000, num_cpus::get() * 100);
        
        Tensor {
            data: vec![0.0; size],
            shape: shape.to_vec(),
            rank: shape.len(),
            simd_processor: Arc::new(SimdProcessor::new()),
            chunk_size,
        }
    }

    pub fn ones(shape: &[usize]) -> Self {
        let size: usize = shape.iter().product();
        let chunk_size = std::cmp::max(1000, num_cpus::get() * 100);
        
        Tensor {
            data: vec![1.0; size],
            shape: shape.to_vec(),
            rank: shape.len(),
            simd_processor: Arc::new(SimdProcessor::new()),
            chunk_size,
        }
    }

    pub fn fill(shape: &[usize], value: f32) -> Self {
        let size: usize = shape.iter().product();
        let chunk_size = std::cmp::max(1000, num_cpus::get() * 100);
        
        Tensor {
            data: vec![value; size],
            shape: shape.to_vec(),
            rank: shape.len(),
            simd_processor: Arc::new(SimdProcessor::new()),
            chunk_size,
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    fn check_same_shape(&self, other: &Tensor) -> TensorResult<()> {
        if self.shape != other.shape {
            return Err(TensorError::ShapeMismatch(format!(
                "Tensors have different shapes: {:?} vs {:?}",
                self.shape, other.shape
            )));
        }
        Ok(())
    }

    // Element-wise addition
    pub fn add(&self, other: &Tensor, mode: ComputeMode) -> TensorResult<Tensor> {
        self.check_same_shape(other)?;
        let mut result = Tensor::zeros(&self.shape);

        match mode {
            ComputeMode::Single => {
                for i in 0..self.data.len() {
                    result.data[i] = self.data[i] + other.data[i];
                }
            }
            ComputeMode::MultiThread => {
                result.data.par_iter_mut()
                    .enumerate()
                    .for_each(|(i, r)| *r = self.data[i] + other.data[i]);
            }
            ComputeMode::SimdMultiThread => {
                let processor = Arc::clone(&self.simd_processor);
                result.data.par_chunks_mut(self.chunk_size)
                    .enumerate()
                    .for_each(|(chunk_idx, chunk)| {
                        let start = chunk_idx * self.chunk_size;
                        let end = std::cmp::min(start + chunk.len(), self.data.len());
                        let len = end - start;
                        
                        processor.add_slice(
                            &self.data[start..start + len],
                            &other.data[start..start + len],
                            &mut chunk[..len]
                        );
                    });
            }
        }

        Ok(result)
    }

    // Element-wise subtraction
    pub fn sub(&self, other: &Tensor, mode: ComputeMode) -> TensorResult<Tensor> {
        self.check_same_shape(other)?;
        let mut result = Tensor::zeros(&self.shape);

        match mode {
            ComputeMode::Single => {
                for i in 0..self.data.len() {
                    result.data[i] = self.data[i] - other.data[i];
                }
            }
            ComputeMode::MultiThread => {
                result.data.par_iter_mut()
                    .enumerate()
                    .for_each(|(i, r)| *r = self.data[i] - other.data[i]);
            }
            ComputeMode::SimdMultiThread => {
                // For subtraction, we'll use single-threaded approach as SIMD subtraction
                // would require additional SIMD implementation
                for i in 0..self.data.len() {
                    result.data[i] = self.data[i] - other.data[i];
                }
            }
        }

        Ok(result)
    }

    // Element-wise multiplication
    pub fn multiply(&self, other: &Tensor, mode: ComputeMode) -> TensorResult<Tensor> {
        self.check_same_shape(other)?;
        let mut result = Tensor::zeros(&self.shape);

        match mode {
            ComputeMode::Single => {
                for i in 0..self.data.len() {
                    result.data[i] = self.data[i] * other.data[i];
                }
            }
            ComputeMode::MultiThread => {
                result.data.par_iter_mut()
                    .enumerate()
                    .for_each(|(i, r)| *r = self.data[i] * other.data[i]);
            }
            ComputeMode::SimdMultiThread => {
                let processor = Arc::clone(&self.simd_processor);
                result.data.par_chunks_mut(self.chunk_size)
                    .enumerate()
                    .for_each(|(chunk_idx, chunk)| {
                        let start = chunk_idx * self.chunk_size;
                        let end = std::cmp::min(start + chunk.len(), self.data.len());
                        let len = end - start;
                        
                        processor.mul_slice(
                            &self.data[start..start + len],
                            &other.data[start..start + len],
                            &mut chunk[..len]
                        );
                    });
            }
        }

        Ok(result)
    }

    // Element-wise division
    pub fn divide(&self, other: &Tensor, mode: ComputeMode) -> TensorResult<Tensor> {
        self.check_same_shape(other)?;
        let mut result = Tensor::zeros(&self.shape);

        match mode {
            ComputeMode::Single => {
                for i in 0..self.data.len() {
                    if other.data[i] == 0.0 {
                        return Err(TensorError::InvalidOperation("Division by zero".to_string()));
                    }
                    result.data[i] = self.data[i] / other.data[i];
                }
            }
            ComputeMode::MultiThread => {
                // Check for zeros first
                if other.data.iter().any(|&x| x == 0.0) {
                    return Err(TensorError::InvalidOperation("Division by zero".to_string()));
                }
                result.data.par_iter_mut()
                    .enumerate()
                    .for_each(|(i, r)| *r = self.data[i] / other.data[i]);
            }
            ComputeMode::SimdMultiThread => {
                // Check for zeros first
                if other.data.iter().any(|&x| x == 0.0) {
                    return Err(TensorError::InvalidOperation("Division by zero".to_string()));
                }
                for i in 0..self.data.len() {
                    result.data[i] = self.data[i] / other.data[i];
                }
            }
        }

        Ok(result)
    }

    // Tensor multiplication (matrix multiplication for rank=2, generalized for higher ranks)
    pub fn matmul(&self, other: &Tensor) -> TensorResult<Tensor> {
        match (self.rank, other.rank) {
            (2, 2) => self.matrix_multiply(other),
            (1, 1) => self.vector_dot(other),
            (2, 1) => self.matrix_vector_multiply(other),
            (1, 2) => other.vector_matrix_multiply(self),
            _ => Err(TensorError::InvalidOperation(
                format!("Tensor multiplication not supported for shapes {:?} and {:?}", 
                       self.shape, other.shape)
            ))
        }
    }

    fn matrix_multiply(&self, other: &Tensor) -> TensorResult<Tensor> {
        if self.shape[1] != other.shape[0] {
            return Err(TensorError::ShapeMismatch(format!(
                "Cannot multiply matrices with shapes {:?} and {:?}", 
                self.shape, other.shape
            )));
        }

        let m = self.shape[0];
        let n = other.shape[1];
        let k = self.shape[1];
        
        let mut result = Tensor::zeros(&[m, n]);
        
        // Parallel matrix multiplication
        result.data.par_chunks_mut(n)
            .enumerate()
            .for_each(|(i, row)| {
                for j in 0..n {
                    let mut sum = 0.0;
                    for l in 0..k {
                        sum += self.data[i * k + l] * other.data[l * n + j];
                    }
                    row[j] = sum;
                }
            });

        Ok(result)
    }

    fn vector_dot(&self, other: &Tensor) -> TensorResult<Tensor> {
        if self.shape[0] != other.shape[0] {
            return Err(TensorError::ShapeMismatch(format!(
                "Cannot compute dot product of vectors with lengths {} and {}", 
                self.shape[0], other.shape[0]
            )));
        }

        let dot_product: f32 = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum();

        Tensor::new(vec![dot_product], &[1])
    }

    fn matrix_vector_multiply(&self, vector: &Tensor) -> TensorResult<Tensor> {
        if self.shape[1] != vector.shape[0] {
            return Err(TensorError::ShapeMismatch(format!(
                "Cannot multiply matrix {:?} with vector {:?}", 
                self.shape, vector.shape
            )));
        }

        let m = self.shape[0];
        let n = self.shape[1];
        let mut result = Tensor::zeros(&[m]);

        result.data.par_iter_mut()
            .enumerate()
            .for_each(|(i, r)| {
                let mut sum = 0.0;
                for j in 0..n {
                    sum += self.data[i * n + j] * vector.data[j];
                }
                *r = sum;
            });

        Ok(result)
    }

    fn vector_matrix_multiply(&self, vector: &Tensor) -> TensorResult<Tensor> {
        if vector.shape[0] != self.shape[0] {
            return Err(TensorError::ShapeMismatch(format!(
                "Cannot multiply vector {:?} with matrix {:?}", 
                vector.shape, self.shape
            )));
        }

        let m = self.shape[0];
        let n = self.shape[1];
        let mut result = Tensor::zeros(&[n]);

        result.data.par_iter_mut()
            .enumerate()
            .for_each(|(j, r)| {
                let mut sum = 0.0;
                for i in 0..m {
                    sum += vector.data[i] * self.data[i * n + j];
                }
                *r = sum;
            });

        Ok(result)
    }

    pub fn scalar_multiply(&self, scalar: f32, mode: ComputeMode) -> Tensor {
        let mut result = Tensor::zeros(&self.shape);

        match mode {
            ComputeMode::Single => {
                for i in 0..self.data.len() {
                    result.data[i] = self.data[i] * scalar;
                }
            }
            ComputeMode::MultiThread => {
                result.data.par_iter_mut()
                    .enumerate()
                    .for_each(|(i, r)| *r = self.data[i] * scalar);
            }
            ComputeMode::SimdMultiThread => {
                let scalar_vec = vec![scalar; self.data.len()];
                let processor = Arc::clone(&self.simd_processor);
                result.data.par_chunks_mut(self.chunk_size)
                    .enumerate()
                    .for_each(|(chunk_idx, chunk)| {
                        let start = chunk_idx * self.chunk_size;
                        let end = std::cmp::min(start + chunk.len(), self.data.len());
                        let len = end - start;
                        
                        processor.mul_slice(
                            &self.data[start..start + len],
                            &scalar_vec[start..start + len],
                            &mut chunk[..len]
                        );
                    });
            }
        }

        result
    }

    pub fn sum(&self, mode: ComputeMode) -> f32 {
        match mode {
            ComputeMode::Single => {
                self.data.iter().sum()
            }
            ComputeMode::MultiThread | ComputeMode::SimdMultiThread => {
                self.data.par_iter().sum()
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

// Operator overloading
impl Add for &Tensor {
    type Output = TensorResult<Tensor>;

    fn add(self, other: &Tensor) -> Self::Output {
        self.add(other, ComputeMode::SimdMultiThread)
    }
}

impl Sub for &Tensor {
    type Output = TensorResult<Tensor>;

    fn sub(self, other: &Tensor) -> Self::Output {
        self.sub(other, ComputeMode::SimdMultiThread)
    }
}

impl Mul for &Tensor {
    type Output = TensorResult<Tensor>;

    fn mul(self, other: &Tensor) -> Self::Output {
        self.multiply(other, ComputeMode::SimdMultiThread)
    }
}

impl Div for &Tensor {
    type Output = TensorResult<Tensor>;

    fn div(self, other: &Tensor) -> Self::Output {
        self.divide(other, ComputeMode::SimdMultiThread)
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && 
        self.data.iter().zip(other.data.iter()).all(|(a, b)| (a - b).abs() < 1e-6)
    }
}
