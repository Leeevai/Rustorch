use crate::error::{TensorError, TensorResult};

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
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && 
        self.data.iter().zip(other.data.iter()).all(|(a, b)| (a - b).abs() < 1e-6)
    }
}
