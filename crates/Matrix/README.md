# Matrix

A high-performance, thread-safe matrix library for Rust with support for concurrent operations.

## Features

- Thread-safe matrix operations with concurrent and sequential execution modes
- Comprehensive matrix operations (addition, subtraction, multiplication, etc.)
- Support for determinants, transpose, and other linear algebra operations
- Efficient memory layout using a one-dimensional vector
- Robust error handling with detailed error types
- Implements standard Rust traits (Index, IndexMut, Add, Sub, Mul, Div, Neg)

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
Matrix = "0.1.0"
```

Or use `cargo add`:

```bash
cargo add Matrix
```

## Usage

### Creating Matrices

```rust
use Matrix::Matrix;

// Create a new 3x3 matrix initialized with default values
let mat = Matrix::<i32>::new(3, 3).unwrap();

// Create a matrix from a vector
let data = vec![1, 2, 3, 4, 5, 6];
let mat = Matrix::from_vec(2, 3, data).unwrap();

// Create special matrices
let zeros = Matrix::<i32>::zeros(3, 3).unwrap();
let ones = Matrix::<i32>::ones(3, 3).unwrap();
let identity = Matrix::<i32>::identity(3).unwrap();
```

### Accessing and Modifying Elements

```rust
let mut mat = Matrix::<i32>::new(2, 2).unwrap();

// Using index notation
mat[(0, 0)] = 1;
mat[(0, 1)] = 2;
mat[(1, 0)] = 3;
mat[(1, 1)] = 4;

let value = mat[(0, 1)]; // Gets 2

// Using safe methods with error handling
mat.set(0, 0, 5).unwrap();
let value = mat.get(0, 0).unwrap(); // Gets 5
```

### Matrix Operations

```rust
let mat1 = Matrix::from_vec(2, 2, vec![1, 2, 3, 4]).unwrap();
let mat2 = Matrix::from_vec(2, 2, vec![5, 6, 7, 8]).unwrap();

// Addition
let sum = (mat1.clone() + mat2.clone()).unwrap();

// Subtraction
let diff = (mat1.clone() - mat2.clone()).unwrap();

// Matrix multiplication
let product = (mat1.clone() * mat2.clone()).unwrap();

// Element-wise multiplication
let dot_product = mat1.dot_product(&mat2).unwrap();

// Scalar operations
let scaled = (mat1.clone() * 2).unwrap();
let divided = (mat1.clone() / 2).unwrap();

// Transpose
let transposed = mat1.transpose().unwrap();

// Determinant (for square matrices)
let det = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0])
    .unwrap()
    .determinant()
    .unwrap();
```

### Concurrency Control

```rust
let mut mat = Matrix::<i32>::new(100, 100).unwrap();

// By default, matrices use concurrent operations when possible
assert!(mat.is_concurrent());

// Disable concurrency for this matrix
mat.set_concurrent(false);
assert!(!mat.is_concurrent());

// Create a matrix with concurrency disabled from the start
let mat_seq = Matrix::<i32>::new_sequential(100, 100).unwrap();
assert!(!mat_seq.is_concurrent());
```

## API Documentation

### Matrix Creation

- `new(rows: usize, cols: usize) -> MatrixResult<Matrix<T>>` - Create a new matrix with concurrent operations enabled
- `new_sequential(rows: usize, cols: usize) -> MatrixResult<Matrix<T>>` - Create a new matrix with concurrent operations disabled
- `from_vec(rows: usize, cols: usize, data: Vec<T>) -> MatrixResult<Matrix<T>>` - Create a matrix from a vector with concurrent operations enabled
- `from_vec_sequential(rows: usize, cols: usize, data: Vec<T>) -> MatrixResult<Matrix<T>>` - Create a matrix from a vector with concurrent operations disabled
- `zeros(rows: usize, cols: usize) -> MatrixResult<Matrix<T>>` - Create a matrix filled with zeros
- `ones(rows: usize, cols: usize) -> MatrixResult<Matrix<T>>` - Create a matrix filled with ones
- `identity(size: usize) -> MatrixResult<Matrix<T>>` - Create an identity matrix

### Element Access

- `get(row: usize, col: usize) -> MatrixResult<&T>` - Get a reference to an element
- `get_mut(row: usize, col: usize) -> MatrixResult<&mut T>` - Get a mutable reference to an element
- `set(row: usize, col: usize, value: T) -> MatrixResult<()>` - Set an element's value
- `row(row: usize) -> MatrixResult<Vec<T>>` - Get a copy of a row
- `col(col: usize) -> MatrixResult<Vec<T>>` - Get a copy of a column

### Matrix Properties

- `dimensions() -> (usize, usize)` - Get the dimensions of the matrix
- `rows() -> usize` - Get the number of rows
- `cols() -> usize` - Get the number of columns
- `is_square() -> bool` - Check if the matrix is square
- `is_empty() -> bool` - Check if the matrix is empty
- `is_concurrent() -> bool` - Check if concurrent operations are enabled
- `set_concurrent(concurrent: bool)` - Enable or disable concurrent operations

### Matrix Operations

- `transpose() -> MatrixResult<Matrix<T>>` - Transpose the matrix
- `trace() -> MatrixResult<T>` - Calculate the trace (sum of diagonal elements)
- `determinant() -> MatrixResult<T>` - Calculate the determinant
- `dot_product(&self, other: &Matrix<T>) -> MatrixResult<Matrix<T>>` - Calculate the element-wise product
- `matrix_multiply(&self, other: &Matrix<T>) -> MatrixResult<Matrix<T>>` - Perform matrix multiplication
- `cofactor_matrix() -> MatrixResult<Matrix<T>>` - Calculate the cofactor matrix
- `minor_matrix(exclude_row: usize, exclude_col: usize) -> MatrixResult<Matrix<T>>` - Calculate a minor matrix

### Operator Overloads

- `+` - Matrix addition
- `-` - Matrix subtraction
- `*` - Matrix multiplication or scalar multiplication
- `/` - Scalar division
- `-` (unary) - Negation
- `[]` - Index access

## Error Handling

All operations that can fail return a `MatrixResult<T>` which is a type alias for `Result<T, MatrixError>`. The `MatrixError` enum provides detailed information about what went wrong.

Common error types include:

- `InvalidDimensions` - Invalid matrix dimensions
- `IndexOutOfBounds` - Attempted to access an element outside the matrix
- `DimensionMismatch` - Dimensions don't match expected values
- `IncompatibleDimensions` - Matrices have incompatible dimensions for an operation
- `NotSquareMatrix` - Operation requires a square matrix
- `DivisionByZero` - Attempted to divide by zero

## Thread Safety

The Matrix implementation is designed to be thread-safe and can efficiently utilize multiple CPU cores for operations when concurrent mode is enabled. It uses the [rayon](https://crates.io/crates/rayon) crate for parallel iterators.

## License

This project is licensed under the MIT License - see the LICENSE file for details.