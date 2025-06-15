# Tensor Computation Crate

A high-performance Rust library for tensor operations with support for SIMD vectorization, parallel processing, and multiple execution modes.

## Features

- **Multi-dimensional tensor operations** with flexible shape handling
- **SIMD acceleration** using AVX2 instructions for optimal performance
- **Parallel processing** with configurable thread counts
- **Multiple execution modes** for different performance requirements
- **Memory-safe operations** with comprehensive error handling
- **Matrix multiplication** optimized for both matrix-vector and matrix-matrix operations

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
tensor-crate = "0.1.0"
rand = "0.8"
rand_pcg = "0.3"

```

## Quick Start

```rust
use tensor_crate::{Tensor, ExecutionMode, TensorResult};

fn main() -> TensorResult<()> {
    // Create tensors
    let matrix = Tensor::random(&[3, 4], 42);
    let vector = Tensor::ones(&[4, 1]);

    // Perform matrix-vector multiplication with SIMD
    let result = matrix.multiply(&vector, ExecutionMode::SIMD)?;

    // Print the result
    result.print();

    Ok(())
}

```

## Core Types

### Tensor

The main data structure representing an n-dimensional array of f32 values.

```rust
// Create tensors with different initialization methods
let zeros = Tensor::zeros(&[2, 3]);           // All zeros
let ones = Tensor::ones(&[2, 3]);             // All ones
let filled = Tensor::fill(&[2, 3], 5.0);      // Fill with specific value
let random = Tensor::random(&[2, 3], 42);     // Random values with seed
let custom = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?; // From data

```

### ExecutionMode

Controls how operations are executed:

- `Sequential` - Single-threaded, basic implementation
- `Parallel` - Multi-threaded parallelization
- `SIMD` - Single-threaded with SIMD vectorization
- `ParallelSIMD` - Multi-threaded with SIMD vectorization

### Error Handling

The crate uses a comprehensive error system with `TensorResult<T>`:

- `ShapeMismatch` - When tensor shapes are incompatible
- `DimensionError` - When operations require specific dimensions
- `IndexOutOfBounds` - When accessing invalid indices
- `InvalidOperation` - When operations are not supported
- `MatrixMultiplicationError` - Matrix multiplication specific errors

## Tensor Operations

### Basic Operations

```rust
// Element-wise operations
let a = Tensor::ones(&[2, 2]);
let b = Tensor::fill(&[2, 2], 2.0);

let sum = (&a + &b)?;                    // Addition
let diff = (&a - &b)?;                   // Subtraction
let hadamard = a.hadamard(&b)?;          // Element-wise multiplication
let scaled = a.scale(3.0);               // Scalar multiplication

```

### Matrix Operations

```rust
// Matrix multiplication with different execution modes
let matrix_a = Tensor::random(&[100, 50], 123);
let matrix_b = Tensor::random(&[50, 75], 456);

// Choose execution mode based on your needs
let result_seq = matrix_a.multiply(&matrix_b, ExecutionMode::Sequential)?;
let result_simd = matrix_a.multiply(&matrix_b, ExecutionMode::SIMD)?;
let result_parallel = matrix_a.multiply(&matrix_b, ExecutionMode::Parallel)?;
let result_fast = matrix_a.multiply(&matrix_b, ExecutionMode::ParallelSIMD)?;

```

### Utility Operations

```rust
let tensor = Tensor::random(&[3, 4], 789);

// Shape and dimension queries
let shape = tensor.shape();              // Get shape as &[usize]
let rank = tensor.rank();                // Number of dimensions
let size = tensor.size();                // Total number of elements

// Matrix-specific accessors (for 2D tensors)
let rows = tensor.rows();                // Number of rows
let cols = tensor.cols();                // Number of columns

// Mathematical operations
let transposed = tensor.transpose()?;    // Matrix transpose
let sum_all = tensor.sum();              // Sum of all elements
let squared = tensor.square();           // Element-wise square

// For column vectors
let max_index = vector.argmax()?;        // Index of maximum value

```

## Performance Optimization

### SIMD Acceleration

The crate leverages AVX2 instructions for vectorized operations:

- Processes 8 f32 values simultaneously
- Automatic fallback for remaining elements
- Significant speedup for large tensors

### Parallel Processing

Multi-threaded execution with work distribution:

```rust
// The number of threads is configurable (default: 6)
// Work is automatically distributed across available threads
let result = matrix.multiply(&vector, ExecutionMode::Parallel)?;

```

### Memory Layout

- Row-major storage for cache-friendly access patterns
- Matrix transpose optimization for better SIMD utilization
- Minimal memory allocations during operations

## Architecture

### Module Structure

```
├── tensor.rs          # Core Tensor struct and basic operations
├── simd.rs           # SIMD-optimized operations using AVX2
├── ops.rs            # Operator overloading and execution modes
├── error.rs          # Error types and handling
└── lib.rs            # Public API and re-exports

```

### Safety

The crate uses unsafe code for:

- SIMD intrinsics (AVX2 instructions)
- Raw pointer manipulation in parallel contexts
- Thread-safe data sharing with proper synchronization

All unsafe operations are carefully encapsulated and tested.

## Examples

### Neural Network Layer

```rust
fn forward_pass(weights: &Tensor, inputs: &Tensor, bias: &Tensor) -> TensorResult<Tensor> {
    // Matrix multiplication with SIMD acceleration
    let linear = weights.multiply(inputs, ExecutionMode::SIMD)?;

    // Add bias
    let result = (&linear + bias)?;

    Ok(result)
}

```

### Batch Processing

```rust
fn process_batch(data: &Tensor, weights: &Tensor) -> TensorResult<Vec<Tensor>> {
    let mut results = Vec::new();

    // Process each sample in the batch
    for i in 0..data.rows() {
        let sample = extract_row(data, i)?;
        let result = weights.multiply(&sample, ExecutionMode::ParallelSIMD)?;
        results.push(result);
    }

    Ok(results)
}

```

## Requirements

- **Rust 1.70+** with stable toolchain
- **x86_64 architecture** with AVX2 support for SIMD operations
- **std library** for threading and collections

## Performance Benchmarks

Typical performance improvements over sequential operations:

| Operation | SIMD Speedup | Parallel Speedup | ParallelSIMD Speedup |
| --- | --- | --- | --- |
| Matrix×Vector (1000×1000) | ~3.2x | ~5.1x | ~12.8x |
| Matrix×Matrix (500×500) | ~2.8x | ~4.7x | ~11.2x |
| Element-wise ops | ~3.5x | ~4.2x | ~13.1x |

*Benchmarks run on Intel i7-12700K with 8 cores*

## Contributing

Contributions are welcome! Please ensure:

1. All tests pass: `cargo test`
2. Code is formatted: `cargo fmt`
3. No clippy warnings: `cargo clippy`
4. Benchmarks show no regressions

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Roadmap

- [ ]  GPU acceleration with CUDA/OpenCL
- [ ]  Additional tensor operations (convolution, pooling)
- [ ]  Automatic differentiation support
- [ ]  Memory mapping for large tensors
- [ ]  ARM NEON SIMD support