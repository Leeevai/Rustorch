pub mod error;
pub mod activation;
pub mod nn;
pub mod training;
pub mod display;
pub mod cost;


pub use error::{NeuralNetworkError, NeuralNetworkResult};
pub use activation::*;
pub use nn::*;


#[cfg(test)]
mod tests {
    use super::*;
    use matrix::Matrix;
    
}