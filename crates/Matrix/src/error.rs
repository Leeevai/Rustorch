use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub enum MatrixError {
    InvalidMatrixDimension,
    InvalidRowDimension,
    InvalidColumnDimension,
    InvalidDimensions,
    IndexOutOfBounds { row: usize, col: usize, max_row: usize, max_col: usize },
    DimensionMismatch { expected: (usize, usize), actual: (usize, usize) },
    IncompatibleDimensions { op: String, dim1: (usize, usize), dim2: (usize, usize) },
    SingularMatrix,
    NotSquareMatrix { rows: usize, cols: usize },
    EmptyMatrix,
    DivisionByZero,
    InvalidOperation(String),
}

impl fmt::Display for MatrixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatrixError::InvalidMatrixDimension => {
                write!(f, "Invalid matrix dimensions")
            }
            MatrixError::InvalidRowDimension => {
                write!(f, "Invalid row dimension")
            }
            MatrixError::InvalidColumnDimension => {
                write!(f, "Invalid column dimension")
            }
            MatrixError::InvalidDimensions => {
                write!(f, "Invalid dimensions provided")
            }
            MatrixError::IndexOutOfBounds { row, col, max_row, max_col } => {
                write!(f, "Index ({}, {}) out of bounds for matrix of size {}x{}", 
                       row, col, max_row, max_col)
            }
            MatrixError::DimensionMismatch { expected, actual } => {
                write!(f, "Dimension mismatch: expected {}x{}, got {}x{}", 
                       expected.0, expected.1, actual.0, actual.1)
            }
            MatrixError::IncompatibleDimensions { op, dim1, dim2 } => {
                write!(f, "Incompatible dimensions for {}: {}x{} and {}x{}", 
                       op, dim1.0, dim1.1, dim2.0, dim2.1)
            }
            MatrixError::SingularMatrix => {
                write!(f, "Matrix is singular (determinant is zero)")
            }
            MatrixError::NotSquareMatrix { rows, cols } => {
                write!(f, "Operation requires square matrix, got {}x{}", rows, cols)
            }
            MatrixError::EmptyMatrix => {
                write!(f, "Matrix is empty")
            }
            MatrixError::DivisionByZero => {
                write!(f, "Division by zero")
            }
            MatrixError::InvalidOperation(msg) => {
                write!(f, "Invalid operation: {}", msg)
            }
        }
    }
}

impl std::error::Error for MatrixError {}

pub type MatrixResult<T> = Result<T, MatrixError>;