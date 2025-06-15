use std::ops::{Add, Sub};
use std::thread;
use crate::tensor::Tensor;
use crate::error::{TensorError, TensorResult};
use crate::simd::{SIMDOps, RawPointerWrapper};
use crate::ExecutionMode;

impl Add for &Tensor {
    type Output = TensorResult<Tensor>;

    fn add(self, rhs: &Tensor) -> TensorResult<Tensor> {
        self.check_same_shape(rhs)?;
        let data = self.data.iter().zip(rhs.data.iter()).map(|(a, b)| a + b).collect();
        Tensor::new(data, &self.shape)
    }
}

impl Sub for &Tensor {
    type Output = TensorResult<Tensor>;

    fn sub(self, rhs: &Tensor) -> TensorResult<Tensor> {
        self.check_same_shape(rhs)?;
        let data = self.data.iter().zip(rhs.data.iter()).map(|(a, b)| a - b).collect();
        Tensor::new(data, &self.shape)
    }
}

impl Sub for Tensor {
    type Output = TensorResult<Tensor>;

    fn sub(self, rhs: Tensor) -> TensorResult<Tensor> {
        self.check_same_shape(&rhs)?;
        let data = self.data.iter().zip(rhs.data.iter()).map(|(a, b)| a - b).collect();
        Tensor::new(data, &self.shape)
    }
}

impl Tensor {
    pub fn multiply(&self, other: &Tensor, mode: ExecutionMode) -> TensorResult<Tensor> {
        match mode {
            ExecutionMode::Sequential => self.multiply_sequential(other),
            ExecutionMode::Parallel => self.multiply_parallel(other, 6),
            ExecutionMode::SIMD => self.multiply_simd(other),
            ExecutionMode::ParallelSIMD => self.multiply_simd_parallel(other, 6),
        }
    }

    fn multiply_sequential(&self, other: &Tensor) -> TensorResult<Tensor> {
        if !self.is_matrix() || !other.is_matrix() {
            return Err(TensorError::DimensionError(
                "Sequential multiplication only supports 2D matrices".to_string()
            ));
        }

        if self.shape()[1] != other.shape()[0] {
            return Err(TensorError::ShapeMismatch(format!(
                "Matrix dimensions don't match: {}x{} * {}x{}",
                self.shape()[0], self.shape()[1], other.shape()[0], other.shape()[1]
            )));
        }

        let mut result = vec![0.0; self.rows() * other.cols()];

        for i in 0..self.rows() {
            for j in 0..other.cols() {
                let mut sum = 0.0;
                for k in 0..self.cols() {
                    sum += self.data[i * self.cols() + k] * other.data[k * other.cols() + j];
                }
                result[i * other.cols() + j] = sum;
            }
        }
        Tensor::new(result, &[self.rows(), other.cols()])
    }

    fn multiply_parallel(&self, other: &Tensor, nb_threads: usize) -> TensorResult<Tensor> {
        if !self.is_matrix() || !other.is_matrix() {
            return Err(TensorError::DimensionError(
                "Parallel multiplication only supports 2D matrices".to_string()
            ));
        }

        if self.shape()[1] != other.shape()[0] {
            return Err(TensorError::ShapeMismatch(format!(
                "Matrix dimensions don't match: {}x{} * {}x{}",
                self.shape()[0], self.shape()[1], other.shape()[0], other.shape()[1]
            )));
        }

        let mut result = vec![0.0; self.rows() * other.cols()];
        let chunk_size = self.rows() / nb_threads;
        let mut handles = vec![];

        for t in 0..nb_threads {
            let start = t * chunk_size;
            let end = if t == nb_threads - 1 { self.rows() } else { start + chunk_size };

            let raw_pointer = RawPointerWrapper { raw: result.as_mut_ptr() };
            let a = self.data.clone();
            let b = other.data.clone();
            let a_cols = self.cols();
            let b_cols = other.cols();

            let handle = thread::spawn(move || {
                for i in start..end {
                    for j in 0..b_cols {
                        let mut sum = 0.0;
                        for k in 0..a_cols {
                            sum += a[i * a_cols + k] * b[k * b_cols + j];
                        }
                        unsafe {
                            raw_pointer.modify_at(i * b_cols + j, sum);
                        }
                    }
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        Tensor::new(result, &[self.rows(), other.cols()])
    }

    fn multiply_simd(&self, other: &Tensor) -> TensorResult<Tensor> {
        if other.is_column_vector() && self.is_matrix() {
            SIMDOps::matrix_vector_multiply(self, other)
        } else if self.is_matrix() && other.is_matrix() {
            SIMDOps::matrix_multiply(self, other)
        } else {
            Err(TensorError::DimensionError(
                "SIMD multiplication only supports matrix-vector or matrix-matrix operations".to_string()
            ))
        }
    }

    fn multiply_simd_parallel(&self, other: &Tensor, nb_threads: usize) -> TensorResult<Tensor> {
        if other.is_column_vector() && self.is_matrix() {
            SIMDOps::matrix_vector_multiply_parallel(self, other, nb_threads)
        } else if self.is_matrix() && other.is_matrix() {
            SIMDOps::matrix_multiply_parallel(self, other, nb_threads)
        } else {
            Err(TensorError::DimensionError(
                "SIMD parallel multiplication only supports matrix-vector or matrix-matrix operations".to_string()
            ))
        }
    }
}