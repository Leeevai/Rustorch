use matrix::Matrix;
use crate::error::{NeuralNetworkError, NeuralNetworkResult};

/// Trait for cost functions
pub trait CostFunction: Send + Sync + Clone {
    fn cost(&self, predicted: &Matrix<f64>, actual: &Matrix<f64>) -> NeuralNetworkResult<f64>;
    fn derivative(&self, predicted: &Matrix<f64>, actual: &Matrix<f64>) -> NeuralNetworkResult<Matrix<f64>>;
    fn name(&self) -> &'static str;
}

/// Mean Squared Error cost function
#[derive(Debug, Clone)]
pub struct MeanSquaredError;

impl CostFunction for MeanSquaredError {
    fn cost(&self, predicted: &Matrix<f64>, actual: &Matrix<f64>) -> NeuralNetworkResult<f64> {
        if predicted.dimensions() != actual.dimensions() {
            return Err(NeuralNetworkError::InvalidOutputSize {
                expected: predicted.rows(),
                actual: actual.rows(),
            });
        }

        let mut sum = 0.0;
        let (rows, cols) = predicted.dimensions();
        
        for i in 0..rows {
            for j in 0..cols {
                let diff = predicted.get(i, j)? - actual.get(i, j)?;
                sum += diff * diff;
            }
        }
        
        Ok(sum / (2.0 * rows as f64 * cols as f64))
    }

    fn derivative(&self, predicted: &Matrix<f64>, actual: &Matrix<f64>) -> NeuralNetworkResult<Matrix<f64>> {
        if predicted.dimensions() != actual.dimensions() {
            return Err(NeuralNetworkError::InvalidOutputSize {
                expected: predicted.rows(),
                actual: actual.rows(),
            });
        }

        let (rows, cols) = predicted.dimensions();
        let mut result = Matrix::new(rows, cols)?;
        result.set_concurrent(predicted.is_concurrent());

        for i in 0..rows {
            for j in 0..cols {
                let derivative_val = (predicted.get(i, j)? - actual.get(i, j)?) / (rows as f64 * cols as f64);
                result.set(i, j, derivative_val)?;
            }
        }

        Ok(result)
    }

    fn name(&self) -> &'static str {
        "mean_squared_error"
    }
}

/// Cross Entropy cost function
#[derive(Debug, Clone)]
pub struct CrossEntropy;

impl CostFunction for CrossEntropy {
    fn cost(&self, predicted: &Matrix<f64>, actual: &Matrix<f64>) -> NeuralNetworkResult<f64> {
        if predicted.dimensions() != actual.dimensions() {
            return Err(NeuralNetworkError::InvalidOutputSize {
                expected: predicted.rows(),
                actual: actual.rows(),
            });
        }

        let mut sum = 0.0;
        let (rows, cols) = predicted.dimensions();
        
        for i in 0..rows {
            for j in 0..cols {
                let p = predicted.get(i, j)?.max(1e-15).min(1.0 - 1e-15); // Prevent log(0)
                let a = *actual.get(i, j)?;
                sum += -(a * p.ln() + (1.0 - a) * (1.0 - p).ln());
            }
        }
        
        Ok(sum / (rows as f64 * cols as f64))
    }

    fn derivative(&self, predicted: &Matrix<f64>, actual: &Matrix<f64>) -> NeuralNetworkResult<Matrix<f64>> {
        if predicted.dimensions() != actual.dimensions() {
            return Err(NeuralNetworkError::InvalidOutputSize {
                expected: predicted.rows(),
                actual: actual.rows(),
            });
        }

        let (rows, cols) = predicted.dimensions();
        let mut result = Matrix::new(rows, cols)?;
        result.set_concurrent(predicted.is_concurrent());

        for i in 0..rows {
            for j in 0..cols {
                let p = predicted.get(i, j)?.max(1e-15).min(1.0 - 1e-15);
                let a = *actual.get(i, j)?;
                let derivative_val = -(a / p - (1.0 - a) / (1.0 - p)) / (rows as f64 * cols as f64);
                result.set(i, j, derivative_val)?;
            }
        }

        Ok(result)
    }

    fn name(&self) -> &'static str {
        "cross_entropy"
    }
}

/// Mean Absolute Error cost function
#[derive(Debug, Clone)]
pub struct MeanAbsoluteError;

impl CostFunction for MeanAbsoluteError {
    fn cost(&self, predicted: &Matrix<f64>, actual: &Matrix<f64>) -> NeuralNetworkResult<f64> {
        if predicted.dimensions() != actual.dimensions() {
            return Err(NeuralNetworkError::InvalidOutputSize {
                expected: predicted.rows(),
                actual: actual.rows(),
            });
        }

        let mut sum = 0.0;
        let (rows, cols) = predicted.dimensions();
        
        for i in 0..rows {
            for j in 0..cols {
                let diff = (predicted.get(i, j)? - actual.get(i, j)?).abs();
                sum += diff;
            }
        }
        
        Ok(sum / (rows as f64 * cols as f64))
    }

    fn derivative(&self, predicted: &Matrix<f64>, actual: &Matrix<f64>) -> NeuralNetworkResult<Matrix<f64>> {
        if predicted.dimensions() != actual.dimensions() {
            return Err(NeuralNetworkError::InvalidOutputSize {
                expected: predicted.rows(),
                actual: actual.rows(),
            });
        }

        let (rows, cols) = predicted.dimensions();
        let mut result = Matrix::new(rows, cols)?;
        result.set_concurrent(predicted.is_concurrent());

        for i in 0..rows {
            for j in 0..cols {
                let diff = predicted.get(i, j)? - actual.get(i, j)?;
                let derivative_val = if diff > 0.0 { 1.0 } else if diff < 0.0 { -1.0 } else { 0.0 };
                result.set(i, j, derivative_val / (rows as f64 * cols as f64))?;
            }
        }

        Ok(result)
    }

    fn name(&self) -> &'static str {
        "mean_absolute_error"
    }
}