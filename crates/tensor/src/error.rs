use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum TensorError {
    ShapeMismatch(String),
    DimensionError(String),
    IndexOutOfBounds(String),
    InvalidOperation(String),
    MatrixMultiplicationError(String),
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorError::ShapeMismatch(msg) => write!(f, "Shape mismatch: {}", msg),
            TensorError::DimensionError(msg) => write!(f, "Dimension error: {}", msg),
            TensorError::IndexOutOfBounds(msg) => write!(f, "Index out of bounds: {}", msg),
            TensorError::InvalidOperation(msg) => write!(f, "Invalid operation: {}", msg),
            TensorError::MatrixMultiplicationError(msg) => write!(f, "Matrix multiplication error: {}", msg),
        }
    }
}

impl std::error::Error for TensorError {}

pub type TensorResult<T> = Result<T, TensorError>;