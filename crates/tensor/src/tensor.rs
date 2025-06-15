use crate::error::{TensorError, TensorResult};
use rand_pcg::Pcg64;
use rand::distributions::{Distribution, Uniform};
use rand::SeedableRng;

#[derive(Debug, Clone)]
pub struct Tensor {
    pub(crate) data: Vec<f32>,
    pub(crate) rank: usize,
    pub(crate) shape: Vec<usize>,
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
        
        Ok(Tensor {
            data,
            shape: shape.to_vec(),
            rank: shape.len(),
        })
    }

    pub fn zeros(shape: &[usize]) -> Self {
        let size: usize = shape.iter().product();
        Tensor {
            data: vec![0.0; size],
            shape: shape.to_vec(),
            rank: shape.len(),
        }
    }

    pub fn ones(shape: &[usize]) -> Self {
        let size: usize = shape.iter().product();
        Tensor {
            data: vec![1.0; size],
            shape: shape.to_vec(),
            rank: shape.len(),
        }
    }

    pub fn fill(shape: &[usize], value: f32) -> Self {
        let size: usize = shape.iter().product();
        Tensor {
            data: vec![value; size],
            shape: shape.to_vec(),
            rank: shape.len(),
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

    pub (crate) fn check_same_shape(&self, other: &Tensor) -> TensorResult<()> {
        if self.shape != other.shape {
            return Err(TensorError::ShapeMismatch(format!(
                "Tensors have different shapes: {:?} vs {:?}",
                self.shape, other.shape
            )));
        }
        Ok(())
    }
     // Legacy compatibility methods
     pub fn rows(&self) -> usize {
        if self.rank >= 1 { self.shape[0] } else { 1 }
    }

    pub fn cols(&self) -> usize {
        if self.rank >= 2 { self.shape[1] } else { 1 }
    }

    pub fn dims(&self) -> (usize, usize) {
        (self.rows(), self.cols())
    }

    // Creation methods
    pub fn scalar(value: f32) -> Self {
        Tensor {
            data: vec![value],
            shape: vec![1],
            rank: 1,
        }
    }

    pub fn random(shape: &[usize], seed: u64) -> Self {
        let mut rng = Pcg64::seed_from_u64(seed);
        let uniform = Uniform::new(0.0, 1.0);
        let size: usize = shape.iter().product();
        let data = (0..size)
            .map(|_| uniform.sample(&mut rng))
            .collect::<Vec<f32>>();

        Tensor {
            data,
            shape: shape.to_vec(),
            rank: shape.len(),
        }
    }

    // Utility methods
    pub fn print(&self) {
        if self.rank == 2 {
            for r in 0..self.rows() {
                for c in 0..self.cols() {
                    print!("{:.5} ", self.data[r * self.cols() + c]);
                }
                println!();
            }
        } else {
            println!("Tensor shape: {:?}", self.shape);
            println!("Data: {:?}", self.data);
        }
    }

    pub fn transpose(&self) -> TensorResult<Tensor> {
        if self.rank != 2 {
            return Err(TensorError::DimensionError(
                "Transpose only supported for 2D tensors".to_string()
            ));
        }

        let rows = self.shape[0];
        let cols = self.shape[1];
        let mut data = vec![0.0; rows * cols];
        
        for i in 0..rows {
            for j in 0..cols {
                data[j * rows + i] = self.data[i * cols + j];
            }
        }
        
        Tensor::new(data, &[cols, rows])
    }

    pub fn scale(&self, scalar: f32) -> Self {
        let data = self.data.iter().map(|&x| x * scalar).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
            rank: self.rank,
        }
    }

    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    pub fn square(&self) -> Self {
        let data = self.data.iter().map(|x| x * x).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
            rank: self.rank,
        }
    }

    pub fn argmax(&self) -> TensorResult<usize> {
        if self.rank != 2 || self.shape[1] != 1 {
            return Err(TensorError::DimensionError(
                "argmax only works on column vectors (rx1 tensors)".to_string()
            ));
        }
        
        let mut max_idx = 0;
        let mut max_val = self.data[0];
        for (i, &val) in self.data.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }
        Ok(max_idx)
    }

    // Element-wise multiplication (Hadamard product)
    pub fn hadamard(&self, other: &Tensor) -> TensorResult<Tensor> {
        self.check_same_shape(other)?;
        let data = self.data.iter().zip(other.data.iter()).map(|(a, b)| a * b).collect();
        Tensor::new(data, &self.shape)
    }

    // Check if tensor is a vector (column vector for matrix operations)
    pub(crate) fn is_column_vector(&self) -> bool {
        self.rank == 2 && self.shape[1] == 1
    }

    pub(crate) fn is_matrix(&self) -> bool {
        self.rank == 2
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && 
        self.data.iter().zip(other.data.iter()).all(|(a, b)| (a - b).abs() < 1e-6)
    }
}
