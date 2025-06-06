use matrix::{Matrix, MatrixResult};
use crate::error::{NeuralNetworkError, NeuralNetworkResult};

/// Trait for activation functions that can be applied to matrices
pub trait ActivationFunction<T>: Send + Sync + Clone
where
    T: Default + Copy + Clone + Send + Sync,
{
    fn activate(&self, input: &Matrix<T>) -> NeuralNetworkResult<Matrix<T>>;
    fn derivative(&self, input: &Matrix<T>) -> NeuralNetworkResult<Matrix<T>>;
    fn name(&self) -> &'static str;
}

/// Sigmoid activation function
#[derive(Debug, Clone)]
pub struct Sigmoid;

impl ActivationFunction<f64> for Sigmoid {
    fn activate(&self, input: &Matrix<f64>) -> NeuralNetworkResult<Matrix<f64>> {
        let (rows, cols) = input.dimensions();
        let mut result = Matrix::new(rows, cols)?;
        result.set_concurrent(input.is_concurrent());

        for i in 0..rows {
            for j in 0..cols {
                let val = input.get(i, j)?;
                let sigmoid_val = 1.0 / (1.0 + (-val).exp());
                result.set(i, j, sigmoid_val)?;
            }
        }

        Ok(result)
    }

    fn derivative(&self, input: &Matrix<f64>) -> NeuralNetworkResult<Matrix<f64>> {
        let activated = self.activate(input)?;
        let (rows, cols) = activated.dimensions();
        let mut result = Matrix::new(rows, cols)?;
        result.set_concurrent(input.is_concurrent());

        for i in 0..rows {
            for j in 0..cols {
                let val = activated.get(i, j)?;
                let derivative_val = val * (1.0 - val);
                result.set(i, j, derivative_val)?;
            }
        }

        Ok(result)
    }

    fn name(&self) -> &'static str {
        "sigmoid"
    }
}

/// ReLU activation function
#[derive(Debug, Clone)]
pub struct ReLU;

impl ActivationFunction<f64> for ReLU {
    fn activate(&self, input: &Matrix<f64>) -> NeuralNetworkResult<Matrix<f64>> {
        let (rows, cols) = input.dimensions();
        let mut result = Matrix::new(rows, cols)?;
        result.set_concurrent(input.is_concurrent());

        for i in 0..rows {
            for j in 0..cols {
                let val = *input.get(i, j)?;
                result.set(i, j, val.max(0.0))?;
            }
        }

        Ok(result)
    }

    fn derivative(&self, input: &Matrix<f64>) -> NeuralNetworkResult<Matrix<f64>> {
        let (rows, cols) = input.dimensions();
        let mut result = Matrix::new(rows, cols)?;
        result.set_concurrent(input.is_concurrent());

        for i in 0..rows {
            for j in 0..cols {
                let val = *input.get(i, j)?;
                result.set(i, j, if val > 0.0 { 1.0 } else { 0.0 })?;
            }
        }

        Ok(result)
    }

    fn name(&self) -> &'static str {
        "relu"
    }
}

/// Tanh activation function
#[derive(Debug, Clone)]
pub struct Tanh;

impl ActivationFunction<f64> for Tanh {
    fn activate(&self, input: &Matrix<f64>) -> NeuralNetworkResult<Matrix<f64>> {
        let (rows, cols) = input.dimensions();
        let mut result = Matrix::new(rows, cols)?;
        result.set_concurrent(input.is_concurrent());

        for i in 0..rows {
            for j in 0..cols {
                let val = *input.get(i, j)?;
                result.set(i, j, val.tanh())?;
            }
        }

        Ok(result)
    }

    fn derivative(&self, input: &Matrix<f64>) -> NeuralNetworkResult<Matrix<f64>> {
        let activated = self.activate(input)?;
        let (rows, cols) = activated.dimensions();
        let mut result = Matrix::new(rows, cols)?;
        result.set_concurrent(input.is_concurrent());

        for i in 0..rows {
            for j in 0..cols {
                let val = *activated.get(i, j)?;
                result.set(i, j, 1.0 - val * val)?;
            }
        }

        Ok(result)
    }

    fn name(&self) -> &'static str {
        "tanh"
    }
}

/// Linear activation function (identity)
#[derive(Debug, Clone)]
pub struct Linear;

impl ActivationFunction<f64> for Linear {
    fn activate(&self, input: &Matrix<f64>) -> NeuralNetworkResult<Matrix<f64>> {
        Ok(input.clone())
    }

    fn derivative(&self, input: &Matrix<f64>) -> NeuralNetworkResult<Matrix<f64>> {
        let (rows, cols) = input.dimensions();
        Matrix::ones(rows, cols).map_err(NeuralNetworkError::from)
    }

    fn name(&self) -> &'static str {
        "linear"
    }
}

/// Leaky ReLU activation function
#[derive(Debug, Clone)]
pub struct LeakyReLU {
    pub alpha: f64,
}

impl LeakyReLU {
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}

impl Default for LeakyReLU {
    fn default() -> Self {
        Self::new(0.01)
    }
}

impl ActivationFunction<f64> for LeakyReLU {
    fn activate(&self, input: &Matrix<f64>) -> NeuralNetworkResult<Matrix<f64>> {
        let (rows, cols) = input.dimensions();
        let mut result = Matrix::new(rows, cols)?;
        result.set_concurrent(input.is_concurrent());

        for i in 0..rows {
            for j in 0..cols {
                let val = *input.get(i, j)?;
                result.set(i, j, if val > 0.0 { val } else { self.alpha * val })?;
            }
        }

        Ok(result)
    }

    fn derivative(&self, input: &Matrix<f64>) -> NeuralNetworkResult<Matrix<f64>> {
        let (rows, cols) = input.dimensions();
        let mut result = Matrix::new(rows, cols)?;
        result.set_concurrent(input.is_concurrent());

        for i in 0..rows {
            for j in 0..cols {
                let val = *input.get(i, j)?;
                result.set(i, j, if val > 0.0 { 1.0 } else { self.alpha })?;
            }
        }

        Ok(result)
    }

    fn name(&self) -> &'static str {
        "leaky_relu"
    }
}