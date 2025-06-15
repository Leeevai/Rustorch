pub mod error;
pub mod tensor;
pub mod simd;
pub mod ops;
use std::fmt;

pub use error::{TensorError, TensorResult};
pub use tensor::Tensor;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    Sequential,
    Parallel,
    SIMD,
    ParallelSIMD,
}
impl fmt::Display for ExecutionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mode_str = match self {
            ExecutionMode::Sequential => "Sequential",
            ExecutionMode::Parallel => "Parallel",
            ExecutionMode::SIMD => "SIMD",
            ExecutionMode::ParallelSIMD => "ParallelSIMD",
        };
        write!(f, "{}", mode_str)
    }
}