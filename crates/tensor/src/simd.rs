<<<<<<< HEAD
use std::arch::x86_64::*;

pub struct SimdProcessor {
    pub simd_width: usize,
    pub supports_avx2: bool,
    pub supports_avx512: bool,
}

impl SimdProcessor {
    pub fn new() -> Self {
        let supports_avx2 = is_x86_feature_detected!("avx2");
        let supports_avx512 = is_x86_feature_detected!("avx512f");
        
        let simd_width = if supports_avx512 {
            16 // 512 bits / 32 bits per f32
        } else if supports_avx2 {
            8  // 256 bits / 32 bits per f32
        } else {
            4  // 128 bits / 32 bits per f32 (SSE)
        };

        SimdProcessor {
            simd_width,
            supports_avx2,
            supports_avx512,
        }
    }

    #[inline]
    pub fn add_slice(&self, a: &[f32], b: &[f32], result: &mut [f32]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), result.len());

        if self.supports_avx2 {
            unsafe {
                self.add_avx2(a, b, result);
            }
        } else {
            self.add_scalar(a, b, result);
        }
    }

    #[inline]
    pub fn mul_slice(&self, a: &[f32], b: &[f32], result: &mut [f32]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), result.len());

        if self.supports_avx2 {
            unsafe {
                self.mul_avx2(a, b, result);
            }
        } else {
            self.mul_scalar(a, b, result);
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn add_avx2(&self, a: &[f32], b: &[f32], result: &mut [f32]) {
        let len = a.len();
        let simd_len = len - (len % 8);

        // Process 8 elements at a time
        for i in (0..simd_len).step_by(8) {
            unsafe {
                let va = _mm256_loadu_ps(a.as_ptr().add(i));
                let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                let vr = _mm256_add_ps(va, vb);
                _mm256_storeu_ps(result.as_mut_ptr().add(i), vr);
            }
        }

        for i in simd_len..len {
            result[i] = a[i] + b[i];
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn mul_avx2(&self, a: &[f32], b: &[f32], result: &mut [f32]) {
        let len = a.len();
        let simd_len = len - (len % 8);

        // Process 8 elements at a time
        for i in (0..simd_len).step_by(8) {
            unsafe {
                let va = _mm256_loadu_ps(a.as_ptr().add(i));
                let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                let vr = _mm256_mul_ps(va, vb);
                _mm256_storeu_ps(result.as_mut_ptr().add(i), vr);
            }
        }

        // Handle remaining elements
        for i in simd_len..len {
            result[i] = a[i] * b[i];
        }
    }

    #[inline]
    fn add_scalar(&self, a: &[f32], b: &[f32], result: &mut [f32]) {
        for i in 0..a.len() {
            result[i] = a[i] + b[i];
        }
    }

    #[inline]
    fn mul_scalar(&self, a: &[f32], b: &[f32], result: &mut [f32]) {
        for i in 0..a.len() {
            result[i] = a[i] * b[i];
        }
=======
use std::arch::x86_64::{_mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_setzero_ps, _mm256_storeu_ps};
use std::sync::Arc;
use std::thread;
use crate::tensor::Tensor;
use crate::error::{TensorError, TensorResult};

#[derive(Clone, Copy)]
pub(crate) struct RawPointerWrapper {
    pub raw: *mut f32,
}

unsafe impl Send for RawPointerWrapper {}
unsafe impl Sync for RawPointerWrapper {}

impl RawPointerWrapper {
    pub unsafe fn modify_at(&self, index: usize, value: f32) {
        let ptr = self.raw.add(index);
        *ptr = value;
    }
}

pub struct SIMDOps;

impl SIMDOps {
    pub fn matrix_vector_multiply(matrix: &Tensor, vector: &Tensor) -> TensorResult<Tensor> {
        if !matrix.is_matrix() || !vector.is_column_vector() {
            return Err(TensorError::DimensionError(
                "Expected matrix and column vector".to_string()
            ));
        }

        if matrix.shape()[1] != vector.shape()[0] {
            return Err(TensorError::ShapeMismatch(format!(
                "Matrix cols {} must match vector rows {}",
                matrix.shape()[1], vector.shape()[0]
            )));
        }

        let mut res = vec![0.0f32; matrix.rows()];
        let cols = matrix.cols();

        for i in 0..matrix.rows() {
            unsafe {
                let mut total = 0.0f32;
                let mut elem = _mm256_setzero_ps();
                
                let complete_chunks = cols / 8;
                for j in 0..complete_chunks {
                    let offset = j * 8;
                    let a_vec = _mm256_loadu_ps(matrix.data.as_ptr().add(i * cols + offset));
                    let b_vec = _mm256_loadu_ps(vector.data.as_ptr().add(offset));
                    let prod = _mm256_mul_ps(a_vec, b_vec);                   
                    elem = _mm256_add_ps(prod, elem);
                }

                let remaining = cols % 8;
                if remaining > 0 {
                    let offset = complete_chunks * 8;
                    for j in 0..remaining {
                        total += matrix.data[i * cols + offset + j] * vector.data[offset + j];
                    }
                }

                let mut values = vec![0.0f32; 8];
                _mm256_storeu_ps(values.as_mut_ptr(), elem);
                total += values[0] + values[1] + values[2] + values[3] + 
                        values[4] + values[5] + values[6] + values[7];

                res[i] = total;
            }
        }
        Tensor::new(res, &[matrix.rows(), 1])
    }

    pub fn matrix_vector_multiply_parallel(matrix: &Tensor, vector: &Tensor, nb_threads: usize) -> TensorResult<Tensor> {
        if !matrix.is_matrix() || !vector.is_column_vector() {
            return Err(TensorError::DimensionError(
                "Expected matrix and column vector".to_string()
            ));
        }

        if matrix.shape()[1] != vector.shape()[0] {
            return Err(TensorError::ShapeMismatch(format!(
                "Matrix cols {} must match vector rows {}",
                matrix.shape()[1], vector.shape()[0]
            )));
        }

        let mut res = vec![0.0f32; matrix.rows()];
        let raw_ptr = RawPointerWrapper { raw: res.as_mut_ptr() };

        let rows_per_thread = matrix.rows() / nb_threads;
        let self_data: Arc<Vec<f32>> = Arc::from(matrix.data.clone());
        let vec_data: Arc<Vec<f32>> = Arc::from(vector.data.clone());
        let mut handles = vec![];

        for i in 0..nb_threads {
            let start = i * rows_per_thread;
            let mut end = start + rows_per_thread;
            if i == nb_threads - 1 {
                end = matrix.rows();
            }

            let self_data = Arc::clone(&self_data);
            let vec_data = Arc::clone(&vec_data);
            let cols = matrix.cols();

            let handle = thread::spawn(move || {
                for k in start..end {
                    unsafe {
                        let mut total = 0.0f32;
                        let mut elem = _mm256_setzero_ps();
                        
                        let complete_chunks = cols / 8;
                        for j in 0..complete_chunks {
                            let offset = j * 8;
                            let a_vec = _mm256_loadu_ps(self_data.as_ptr().add(k * cols + offset));
                            let b_vec = _mm256_loadu_ps(vec_data.as_ptr().add(offset));
                            let prod = _mm256_mul_ps(a_vec, b_vec);                   
                            elem = _mm256_add_ps(prod, elem);
                        }

                        let remaining = cols % 8;
                        if remaining > 0 {
                            let offset = complete_chunks * 8;
                            for j in 0..remaining {
                                total += self_data[k * cols + offset + j] * vec_data[offset + j];
                            }
                        }
        
                        let mut values = vec![0.0f32; 8];
                        _mm256_storeu_ps(values.as_mut_ptr(), elem);
                        total += values[0] + values[1] + values[2] + values[3] + 
                                values[4] + values[5] + values[6] + values[7];
        
                        raw_ptr.modify_at(k, total);
                    }
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }
        Tensor::new(res, &[matrix.rows(), 1])
    }

    pub fn matrix_multiply(a: &Tensor, b: &Tensor) -> TensorResult<Tensor> {
        if !a.is_matrix() || !b.is_matrix() {
            return Err(TensorError::DimensionError(
                "Both tensors must be 2D matrices".to_string()
            ));
        }

        if a.shape()[1] != b.shape()[0] {
            return Err(TensorError::ShapeMismatch(format!(
                "Matrix dimensions don't match: {}x{} * {}x{}",
                a.shape()[0], a.shape()[1], b.shape()[0], b.shape()[1]
            )));
        }

        let mut res = vec![0.0f32; a.rows() * b.cols()];
        let transposed = b.transpose()?;

        for i in 0..a.rows() {
            for k in 0..b.cols() {
                unsafe {
                    let mut total = 0.0f32;
                    let mut elem = _mm256_setzero_ps();
                    
                    let complete_chunks = a.cols() / 8;
                    for j in 0..complete_chunks {
                        let offset = j * 8;
                        let a_vec = _mm256_loadu_ps(a.data.as_ptr().add(i * a.cols() + offset));
                        let b_vec = _mm256_loadu_ps(transposed.data.as_ptr().add(k * transposed.cols() + offset));
                        let prod = _mm256_mul_ps(a_vec, b_vec);                   
                        elem = _mm256_add_ps(prod, elem);
                    }
    
                    let remaining = a.cols() % 8;
                    if remaining > 0 {
                        let offset = complete_chunks * 8;
                        for j in 0..remaining {
                            total += a.data[i * a.cols() + offset + j] * transposed.data[k * transposed.cols() + offset + j];
                        }
                    }
    
                    let mut values = [0.0f32; 8];
                    _mm256_storeu_ps(values.as_mut_ptr(), elem);
                    total += values[0] + values[1] + values[2] + values[3] + 
                            values[4] + values[5] + values[6] + values[7];
    
                    res[i * b.cols() + k] = total;
                }
            }
        }
        Tensor::new(res, &[a.rows(), b.cols()])
    }

    pub fn matrix_multiply_parallel(a: &Tensor, b: &Tensor, nb_threads: usize) -> TensorResult<Tensor> {
        if !a.is_matrix() || !b.is_matrix() {
            return Err(TensorError::DimensionError(
                "Both tensors must be 2D matrices".to_string()
            ));
        }

        if a.shape()[1] != b.shape()[0] {
            return Err(TensorError::ShapeMismatch(format!(
                "Matrix dimensions don't match: {}x{} * {}x{}",
                a.shape()[0], a.shape()[1], b.shape()[0], b.shape()[1]
            )));
        }

        let transposed = b.transpose()?;
        let mut res = vec![0.0f32; a.rows() * b.cols()];
        let raw_ptr = RawPointerWrapper { raw: res.as_mut_ptr() };

        let rows_per_thread = a.rows() / nb_threads;
        let a_data: Arc<Vec<f32>> = Arc::from(a.data.clone());
        let b_data: Arc<Vec<f32>> = Arc::from(transposed.data.clone());
        let mut handles = vec![];

        for i in 0..nb_threads {
            let start = i * rows_per_thread;
            let mut end = start + rows_per_thread;
            if i == nb_threads - 1 {
                end = a.rows();
            }

            let a_data = Arc::clone(&a_data);
            let b_data = Arc::clone(&b_data);
            let a_cols = a.cols();
            let b_cols = b.cols();
            let b_rows = transposed.cols();

            let handle = thread::spawn(move || {
                for i in start..end {
                    for k in 0..b_cols {
                        unsafe {
                            let mut total = 0.0f32;
                            let mut elem = _mm256_setzero_ps();
                            
                            let complete_chunks = a_cols / 8;
                            for j in 0..complete_chunks {
                                let offset = j * 8;
                                let a_vec = _mm256_loadu_ps(a_data.as_ptr().add(i * a_cols + offset));
                                let b_vec = _mm256_loadu_ps(b_data.as_ptr().add(k * b_rows + offset));
                                let prod = _mm256_mul_ps(a_vec, b_vec);                   
                                elem = _mm256_add_ps(prod, elem);
                            }
            
                            let remaining = a_cols % 8;
                            if remaining > 0 {
                                let offset = complete_chunks * 8;
                                for j in 0..remaining {
                                    total += a_data[i * a_cols + offset + j] * b_data[k * b_rows + offset + j];
                                }
                            }
            
                            let mut values = [0.0f32; 8];
                            _mm256_storeu_ps(values.as_mut_ptr(), elem);
                            total += values[0] + values[1] + values[2] + values[3] + 
                                    values[4] + values[5] + values[6] + values[7];
            
                            raw_ptr.modify_at(i * b_cols + k, total);
                        }
                    }
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }
        Tensor::new(res, &[a.rows(), b.cols()])
>>>>>>> 3-tensor-crate-v2
    }
}