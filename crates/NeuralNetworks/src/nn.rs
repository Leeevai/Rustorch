use matrix::Matrix;
use crate::activation::ActivationFunction;
use crate::error::{NeuralNetworkError, NeuralNetworkResult};
use rand::prelude::*;
use rand_distr::{Normal, Distribution};
use rayon::prelude::*;

/// Layer structure containing weights, biases, and activation function
#[derive(Debug, Clone)]
pub struct Layer<T, A>
where
    T: Default + Copy + Clone + Send + Sync,
    A: ActivationFunction<T>,
{
    pub weights: Matrix<T>,
    pub biases: Matrix<T>,
    pub activation: A,
    pub concurrent: bool,
}

impl<T, A> Layer<T, A>
where
    T: Default + Copy + Clone + Send + Sync,
    A: ActivationFunction<T>,
{
    pub fn new(
        input_size: usize,
        output_size: usize,
        activation: A,
        concurrent: bool,
    ) -> NeuralNetworkResult<Self> {
        let mut weights = Matrix::new(output_size, input_size)?;
        let mut biases = Matrix::new(output_size, 1)?;
        
        weights.set_concurrent(concurrent);
        biases.set_concurrent(concurrent);

        Ok(Self {
            weights,
            biases,
            activation,
            concurrent,
        })
    }

    pub fn input_size(&self) -> usize {
        self.weights.cols()
    }

    pub fn output_size(&self) -> usize {
        self.weights.rows()
    }
}

/// Neural Network structure
#[derive(Debug, Clone)]
pub struct NeuralNetwork<T, A>
where
    T: Default + Copy + Clone + Send + Sync,
    A: ActivationFunction<T>,
{
    pub (crate) layers: Vec<Layer<T, A>>,
    pub (crate) architecture: Vec<usize>,
    pub (crate) concurrent: bool,
}

impl<A> NeuralNetwork<f64, A>
where
    A: ActivationFunction<f64>,
{
    /// Create a new neural network with given architecture
    pub fn new(architecture: Vec<usize>, activation: A, concurrent: bool) -> NeuralNetworkResult<Self> {
        if architecture.len() < 2 {
            return Err(NeuralNetworkError::InvalidArchitecture(
                "Network must have at least 2 layers (input and output)".to_string(),
            ));
        }

        if architecture.iter().any(|&size| size == 0) {
            return Err(NeuralNetworkError::InvalidArchitecture(
                "Layer sizes must be greater than 0".to_string(),
            ));
        }

        let mut layers = Vec::new();

        // Create layers between consecutive sizes in architecture
        for i in 0..architecture.len() - 1 {
            let input_size = architecture[i];
            let output_size = architecture[i + 1];
            let layer = Layer::new(input_size, output_size, activation.clone(), concurrent)?;
            layers.push(layer);
        }

        Ok(Self {
            layers,
            architecture,
            concurrent,
        })
    }

    /// Initialize weights and biases with Xavier/Glorot initialization
    pub fn xavier_initialization(&mut self) -> NeuralNetworkResult<()> {
        if self.concurrent {
            // Use parallel initialization with thread-local RNGs
            self.layers.par_iter_mut().try_for_each(|layer| -> NeuralNetworkResult<()> {
                let mut rng = rand::rng(); // Use the new rand::rng() function
                let fan_in = layer.input_size() as f64;
                let fan_out = layer.output_size() as f64;
                let limit = (6.0 / (fan_in + fan_out)).sqrt();
                
                // Initialize weights
                for i in 0..layer.weights.rows() {
                    for j in 0..layer.weights.cols() {
                        let weight = rng.random_range(-limit..limit);
                        layer.weights.set(i, j, weight)?;
                    }
                }

                // Initialize biases to zero
                for i in 0..layer.biases.rows() {
                    layer.biases.set(i, 0, 0.0)?;
                }

                Ok(())
            })?;
        } else {
            let mut rng = rand::rng();
            for layer in &mut self.layers {
                let fan_in = layer.input_size() as f64;
                let fan_out = layer.output_size() as f64;
                let limit = (6.0 / (fan_in + fan_out)).sqrt();
                
                // Initialize weights
                for i in 0..layer.weights.rows() {
                    for j in 0..layer.weights.cols() {
                        let weight = rng.random_range(-limit..limit);
                        layer.weights.set(i, j, weight)?;
                    }
                }

                // Initialize biases to zero
                for i in 0..layer.biases.rows() {
                    layer.biases.set(i, 0, 0.0)?;
                }
            }
        }

        Ok(())
    }

    /// Initialize weights and biases with He initialization (good for ReLU)
    pub fn he_initialization(&mut self) -> NeuralNetworkResult<()> {
        if self.concurrent {
            // Use parallel initialization with thread-local RNGs
            self.layers.par_iter_mut().try_for_each(|layer| -> NeuralNetworkResult<()> {
                let mut rng = rand::rng(); // Use the new rand::rng() function
                let fan_in = layer.input_size() as f64;
                let std_dev = (2.0 / fan_in).sqrt();
                let normal = Normal::new(0.0, std_dev).map_err(|_| {
                    NeuralNetworkError::InitializationError("Failed to create normal distribution".to_string())
                })?;
                
                // Initialize weights
                for i in 0..layer.weights.rows() {
                    for j in 0..layer.weights.cols() {
                        let weight = normal.sample(&mut rng);
                        layer.weights.set(i, j, weight)?;
                    }
                }

                // Initialize biases to zero
                for i in 0..layer.biases.rows() {
                    layer.biases.set(i, 0, 0.0)?;
                }

                Ok(())
            })?;
        } else {
            let mut rng = rand::rng();
            for layer in &mut self.layers {
                let fan_in = layer.input_size() as f64;
                let std_dev = (2.0 / fan_in).sqrt();
                let normal = Normal::new(0.0, std_dev).map_err(|_| {
                    NeuralNetworkError::InitializationError("Failed to create normal distribution".to_string())
                })?;
                
                // Initialize weights
                for i in 0..layer.weights.rows() {
                    for j in 0..layer.weights.cols() {
                        let weight = normal.sample(&mut rng);
                        layer.weights.set(i, j, weight)?;
                    }
                }

                // Initialize biases to zero
                for i in 0..layer.biases.rows() {
                    layer.biases.set(i, 0, 0.0)?;
                }
            }
        }

        Ok(())
    }

    /// Random initialization with given range
    pub fn random_initialization(&mut self, min: f64, max: f64) -> NeuralNetworkResult<()> {
        if self.concurrent {
            // Use parallel initialization with thread-local RNGs
            self.layers.par_iter_mut().try_for_each(|layer| -> NeuralNetworkResult<()> {
                let mut rng = rand::rng(); // Use the new rand::rng() function
                
                // Initialize weights
                for i in 0..layer.weights.rows() {
                    for j in 0..layer.weights.cols() {
                        let weight = rng.random_range(min..max);
                        layer.weights.set(i, j, weight)?;
                    }
                }

                // Initialize biases
                for i in 0..layer.biases.rows() {
                    let bias = rng.random_range(min..max);
                    layer.biases.set(i, 0, bias)?;
                }

                Ok(())
            })?;
        } else {
            let mut rng = rand::rng();
            for layer in &mut self.layers {
                // Initialize weights
                for i in 0..layer.weights.rows() {
                    for j in 0..layer.weights.cols() {
                        let weight = rng.random_range(min..max);
                        layer.weights.set(i, j, weight)?;
                    }
                }

                // Initialize biases
                for i in 0..layer.biases.rows() {
                    let bias = rng.random_range(min..max);
                    layer.biases.set(i, 0, bias)?;
                }
            }
        }

        Ok(())
    }

    /// Forward propagation through the network
    pub fn forward(&self, input: &Matrix<f64>) -> NeuralNetworkResult<Matrix<f64>> {
        if input.rows() != self.architecture[0] {
            return Err(NeuralNetworkError::InvalidInputSize {
                expected: self.architecture[0],
                actual: input.rows(),
            });
        }

        if input.cols() != 1 {
            return Err(NeuralNetworkError::InvalidInputSize {
                expected: 1,
                actual: input.cols(),
            });
        }

        let mut current_output = input.clone();

        for layer in &self.layers {
            // Linear transformation: W * x + b
            let linear_output = layer.weights.matrix_multiply(&current_output)?;
            let linear_with_bias = (linear_output + layer.biases.clone())?;
            
            // Apply activation function
            current_output = layer.activation.activate(&linear_with_bias)?;
        }

        Ok(current_output)
    }

    /// Forward propagation with intermediate outputs (useful for training)
    pub fn forward_with_intermediates(&self, input: &Matrix<f64>) -> NeuralNetworkResult<Vec<Matrix<f64>>> {
        if input.rows() != self.architecture[0] {
            return Err(NeuralNetworkError::InvalidInputSize {
                expected: self.architecture[0],
                actual: input.rows(),
            });
        }

        if input.cols() != 1 {
            return Err(NeuralNetworkError::InvalidInputSize {
                expected: 1,
                actual: input.cols(),
            });
        }

        let mut outputs = Vec::new();
        let mut current_output = input.clone();
        outputs.push(current_output.clone());

        for layer in &self.layers {
            // Linear transformation: W * x + b
            let linear_output = layer.weights.matrix_multiply(&current_output)?;
            let linear_with_bias = (linear_output + layer.biases.clone())?;
            
            // Apply activation function
            current_output = layer.activation.activate(&linear_with_bias)?;
            outputs.push(current_output.clone());
        }

        Ok(outputs)
    }

    /// Get network architecture
    pub fn architecture(&self) -> &[usize] {
        &self.architecture
    }

    /// Get number of layers (excluding input layer)
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get input size
    pub fn input_size(&self) -> usize {
        self.architecture[0]
    }

    /// Get output size
    pub fn output_size(&self) -> usize {
        self.architecture[self.architecture.len() - 1]
    }

    /// Get layer by index
    pub fn get_layer(&self, index: usize) -> NeuralNetworkResult<&Layer<f64, A>> {
        self.layers.get(index).ok_or(NeuralNetworkError::LayerIndexOutOfBounds {
            index,
            max: self.layers.len() - 1,
        })
    }

    /// Get mutable layer by index
    pub fn get_layer_mut(&mut self, index: usize) -> NeuralNetworkResult<&mut Layer<f64, A>> {
        let max = self.layers.len() - 1;
        self.layers.get_mut(index).ok_or(NeuralNetworkError::LayerIndexOutOfBounds {
            index,
            max,
        })
    }

    /// Set concurrent mode for all layers
    pub fn set_concurrent(&mut self, concurrent: bool) {
        self.concurrent = concurrent;
        for layer in &mut self.layers {
            layer.concurrent = concurrent;
            layer.weights.set_concurrent(concurrent);
            layer.biases.set_concurrent(concurrent);
        }
    }

    /// Check if the network is using concurrent operations
    pub fn is_concurrent(&self) -> bool {
        self.concurrent
    }

    /// Get total number of parameters (weights + biases)
    pub fn parameter_count(&self) -> usize {
        self.layers.iter().map(|layer| {
            layer.weights.rows() * layer.weights.cols() + layer.biases.rows()
        }).sum()
    }
}


/// Macro for creating neural networks with a simple syntax
#[macro_export]
macro_rules! nn {
    ($($size:expr),+ ; $activation:expr) => {{
        let architecture = vec![$($size),+];
        NeuralNetwork::new(architecture, $activation, true)
    }};
    ($($size:expr),+) => {{
        let architecture = vec![$($size),+];
        NeuralNetwork::new(architecture, crate::activations::Sigmoid, true)
    }};
}