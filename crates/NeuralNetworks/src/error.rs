use std::fmt;
use matrix::MatrixError;

#[derive(Debug, Clone)]
pub enum NeuralNetworkError {
    InvalidArchitecture(String),
    InvalidInputSize { expected: usize, actual: usize },
    InvalidOutputSize { expected: usize, actual: usize },
    MatrixError(MatrixError),
    InitializationError(String),
    ForwardPropagationError(String),
    InvalidActivationFunction,
    EmptyNetwork,
    LayerIndexOutOfBounds { index: usize, max: usize },
}

impl fmt::Display for NeuralNetworkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NeuralNetworkError::InvalidArchitecture(msg) => {
                write!(f, "Invalid neural network architecture: {}", msg)
            }
            NeuralNetworkError::InvalidInputSize { expected, actual } => {
                write!(f, "Invalid input size: expected {}, got {}", expected, actual)
            }
            NeuralNetworkError::InvalidOutputSize { expected, actual } => {
                write!(f, "Invalid output size: expected {}, got {}", expected, actual)
            }
            NeuralNetworkError::MatrixError(err) => {
                write!(f, "Matrix operation error: {}", err)
            }
            NeuralNetworkError::InitializationError(msg) => {
                write!(f, "Network initialization error: {}", msg)
            }
            NeuralNetworkError::ForwardPropagationError(msg) => {
                write!(f, "Forward propagation error: {}", msg)
            }
            NeuralNetworkError::InvalidActivationFunction => {
                write!(f, "Invalid activation function")
            }
            NeuralNetworkError::EmptyNetwork => {
                write!(f, "Neural network is empty")
            }
            NeuralNetworkError::LayerIndexOutOfBounds { index, max } => {
                write!(f, "Layer index {} out of bounds (max: {})", index, max)
            }
        }
    }
}

impl std::error::Error for NeuralNetworkError {}

impl From<MatrixError> for NeuralNetworkError {
    fn from(error: MatrixError) -> Self {
        NeuralNetworkError::MatrixError(error)
    }
}

pub type NeuralNetworkResult<T> = Result<T, NeuralNetworkError>;