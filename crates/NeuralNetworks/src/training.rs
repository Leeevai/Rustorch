use matrix::Matrix;
use crate::nn::{NeuralNetwork, Layer};
use crate::activation::ActivationFunction;
use crate::cost::CostFunction;
use crate::error::{NeuralNetworkError, NeuralNetworkResult};
use std::time::{Duration, Instant};
use rayon::prelude::*;

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub learning_rate: f64,
    pub epochs: usize,
    pub batch_size: usize,
    pub validation_split: f64,
    pub early_stopping_patience: Option<usize>,
    pub min_improvement: f64,
    pub verbose: bool,
    pub log_interval: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            epochs: 1000,
            batch_size: 32,
            validation_split: 0.2,
            early_stopping_patience: Some(10),
            min_improvement: 1e-6,
            verbose: true,
            log_interval: 100,
        }
    }
}

/// Training metrics
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    pub epoch: usize,
    pub train_loss: f64,
    pub validation_loss: Option<f64>,
    pub epoch_duration: Duration,
    pub total_duration: Duration,
}

/// Training history
#[derive(Debug, Clone)]
pub struct TrainingHistory {
    pub metrics: Vec<TrainingMetrics>,
    pub best_validation_loss: Option<f64>,
    pub best_epoch: usize,
    pub stopped_early: bool,
}

impl TrainingHistory {
    pub fn new() -> Self {
        Self {
            metrics: Vec::new(),
            best_validation_loss: None,
            best_epoch: 0,
            stopped_early: false,
        }
    }

    pub fn add_metric(&mut self, metric: TrainingMetrics) {
        if let Some(val_loss) = metric.validation_loss {
            if self.best_validation_loss.is_none() || val_loss < self.best_validation_loss.unwrap() {
                self.best_validation_loss = Some(val_loss);
                self.best_epoch = metric.epoch;
            }
        }
        self.metrics.push(metric);
    }

    pub fn print_summary(&self) {
        println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
        println!("║                                TRAINING SUMMARY                                 ║");
        println!("╠══════════════════════════════════════════════════════════════════════════════╣");
        
        if let Some(last_metric) = self.metrics.last() {
            println!("║ Total Epochs: {:>10} │ Total Time: {:>8.2}s │ Avg Time/Epoch: {:>6.2}ms ║", 
                last_metric.epoch,
                last_metric.total_duration.as_secs_f64(),
                last_metric.total_duration.as_millis() as f64 / last_metric.epoch as f64
            );
            
            println!("║ Final Train Loss: {:>12.6} │ Final Val Loss: {:>12.6} │              ║", 
                last_metric.train_loss,
                last_metric.validation_loss.unwrap_or(0.0)
            );
        }
        
        if let Some(best_loss) = self.best_validation_loss {
            println!("║ Best Validation Loss: {:>8.6} │ Best Epoch: {:>8} │                  ║", 
                best_loss, self.best_epoch
            );
        }
        
        if self.stopped_early {
            println!("║ Training stopped early due to no improvement                                    ║");
        }
        
        println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    }
}

/// Trait for training algorithms
pub trait TrainingAlgorithm<A, C>: Send + Sync
where
    A: ActivationFunction<f64>,
    C: CostFunction,
{
    fn train(
        &mut self,
        network: &mut NeuralNetwork<f64, A>,
        inputs: &[Matrix<f64>],
        targets: &[Matrix<f64>],
        config: &TrainingConfig,
    ) -> NeuralNetworkResult<TrainingHistory>;
    
    fn name(&self) -> &'static str;
}

/// Stochastic Gradient Descent with backpropagation
pub struct SGD<C: CostFunction> {
    cost_function: C,
}

impl<C: CostFunction> SGD<C> {
    pub fn new(cost_function: C) -> Self {
        Self { cost_function }
    }
}

impl<A, C> TrainingAlgorithm<A, C> for SGD<C>
where
    A: ActivationFunction<f64>,
    C: CostFunction,
{
    fn train(
        &mut self,
        network: &mut NeuralNetwork<f64, A>,
        inputs: &[Matrix<f64>],
        targets: &[Matrix<f64>],
        config: &TrainingConfig,
    ) -> NeuralNetworkResult<TrainingHistory> {
        if inputs.len() != targets.len() {
            return Err(NeuralNetworkError::InvalidInputSize {
                expected: inputs.len(),
                actual: targets.len(),
            });
        }

        let mut history = TrainingHistory::new();
        let start_time = Instant::now();

        // Split data into training and validation sets
        let split_idx = ((1.0 - config.validation_split) * inputs.len() as f64) as usize;
        let (train_inputs, val_inputs) = inputs.split_at(split_idx);
        let (train_targets, val_targets) = targets.split_at(split_idx);

        let mut patience_counter = 0;

        for epoch in 1..=config.epochs {
            let epoch_start = Instant::now();

            // Training phase
            let train_loss = self.train_epoch(network, train_inputs, train_targets, config)?;

            // Validation phase
            let validation_loss = if !val_inputs.is_empty() {
                Some(self.validate(network, val_inputs, val_targets)?)
            } else {
                None
            };

            let epoch_duration = epoch_start.elapsed();
            let total_duration = start_time.elapsed();

            let metric = TrainingMetrics {
                epoch,
                train_loss,
                validation_loss,
                epoch_duration,
                total_duration,
            };

            // Check for early stopping
            if let (Some(val_loss), Some(patience)) = (validation_loss, config.early_stopping_patience) {
                if let Some(best_loss) = history.best_validation_loss {
                    if best_loss - val_loss < config.min_improvement {
                        patience_counter += 1;
                    } else {
                        patience_counter = 0;
                    }
                }

                if patience_counter >= patience {
                    if config.verbose {
                        println!("Early stopping triggered at epoch {}", epoch);
                    }
                    history.add_metric(metric);
                    history.stopped_early = true;
                    break;
                }
            }

            history.add_metric(metric.clone());

            // Logging
            if config.verbose && (epoch % config.log_interval == 0 || epoch == 1) {
                self.log_progress(&metric);
            }
        }

        if config.verbose {
            history.print_summary();
        }

        Ok(history)
    }

    fn name(&self) -> &'static str {
        "SGD"
    }
}

impl<C: CostFunction> SGD<C> {
    fn train_epoch(
        &self,
        network: &mut NeuralNetwork<f64, impl ActivationFunction<f64>>,
        inputs: &[Matrix<f64>],
        targets: &[Matrix<f64>],
        config: &TrainingConfig,
    ) -> NeuralNetworkResult<f64> {
        let mut total_loss = 0.0;
        let mut batches_processed = 0;

        // Create batches
        for batch_start in (0..inputs.len()).step_by(config.batch_size) {
            let batch_end = (batch_start + config.batch_size).min(inputs.len());
            let batch_inputs = &inputs[batch_start..batch_end];
            let batch_targets = &targets[batch_start..batch_end];

            let batch_loss = self.train_batch(network, batch_inputs, batch_targets, config)?;
            total_loss += batch_loss;
            batches_processed += 1;
        }

        Ok(total_loss / batches_processed as f64)
    }

    fn train_batch(
        &self,
        network: &mut NeuralNetwork<f64, impl ActivationFunction<f64>>,
        batch_inputs: &[Matrix<f64>],
        batch_targets: &[Matrix<f64>],
        config: &TrainingConfig,
    ) -> NeuralNetworkResult<f64> {
        let mut total_loss = 0.0;
        let mut weight_gradients = Vec::new();
        let mut bias_gradients = Vec::new();

        // Initialize gradient accumulators
        for layer_idx in 0..network.num_layers() {
            let layer = network.get_layer(layer_idx)?;
            let weight_grad = Matrix::zeros(layer.weights.rows(), layer.weights.cols())?;
            let bias_grad = Matrix::zeros(layer.biases.rows(), layer.biases.cols())?;
            weight_gradients.push(weight_grad);
            bias_gradients.push(bias_grad);
        }

        // Process each sample in the batch
        for (input, target) in batch_inputs.iter().zip(batch_targets.iter()) {
            // Forward propagation
            let activations = network.forward_with_intermediates(input)?;
            let prediction = activations.last().unwrap();

            // Calculate loss
            let loss = self.cost_function.cost(prediction, target)?;
            total_loss += loss;

            // Backpropagation
            self.backpropagate(network, &activations, target, &mut weight_gradients, &mut bias_gradients)?;
        }

        // Apply gradients
        self.apply_gradients(network, &weight_gradients, &bias_gradients, config.learning_rate, batch_inputs.len())?;

        Ok(total_loss / batch_inputs.len() as f64)
    }

    fn backpropagate(
        &self,
        network: &NeuralNetwork<f64, impl ActivationFunction<f64>>,
        activations: &[Matrix<f64>],
        target: &Matrix<f64>,
        weight_gradients: &mut [Matrix<f64>],
        bias_gradients: &mut [Matrix<f64>],
    ) -> NeuralNetworkResult<()> {
        let num_layers = network.num_layers();
        let mut delta = self.cost_function.derivative(activations.last().unwrap(), target)?;

        // Backpropagate through each layer
        for layer_idx in (0..num_layers).rev() {
            let layer = network.get_layer(layer_idx)?;
            let layer_input = &activations[layer_idx];
            let layer_output = &activations[layer_idx + 1];

            // Calculate derivative of activation function
            let activation_derivative = layer.activation.derivative(layer_output)?;

            // Element-wise multiplication of delta and activation derivative
            for i in 0..delta.rows() {
                for j in 0..delta.cols() {
                    let current_delta = *delta.get(i, j)?;
                    let current_derivative = *activation_derivative.get(i, j)?;
                    delta.set(i, j, current_delta * current_derivative)?;
                }
            }

            // Calculate gradients for weights
            let input_transposed = layer_input.transpose()?;
            let weight_gradient = delta.matrix_multiply(&input_transposed)?;
            
            // Accumulate gradients
            for i in 0..weight_gradient.rows() {
                for j in 0..weight_gradient.cols() {
                    let current_grad = *weight_gradients[layer_idx].get(i, j)?;
                    let new_grad = *weight_gradient.get(i, j)?;
                    weight_gradients[layer_idx].set(i, j, current_grad + new_grad)?;
                }
            }

            // Bias gradients are just the delta
            for i in 0..delta.rows() {
                for j in 0..delta.cols() {
                    let current_grad = *bias_gradients[layer_idx].get(i, j)?;
                    let new_grad = *delta.get(i, j)?;
                    bias_gradients[layer_idx].set(i, j, current_grad + new_grad)?;
                }
            }

            // Calculate delta for next layer (if not the first layer)
            if layer_idx > 0 {
                let weights_transposed = layer.weights.transpose()?;
                delta = weights_transposed.matrix_multiply(&delta)?;
            }
        }

        Ok(())
    }

    fn apply_gradients(
        &self,
        network: &mut NeuralNetwork<f64, impl ActivationFunction<f64>>,
        weight_gradients: &[Matrix<f64>],
        bias_gradients: &[Matrix<f64>],
        learning_rate: f64,
        batch_size: usize,
    ) -> NeuralNetworkResult<()> {
        let batch_size_f64 = batch_size as f64;

        for layer_idx in 0..network.num_layers() {
            let layer = network.get_layer_mut(layer_idx)?;

            // Update weights
            for i in 0..layer.weights.rows() {
                for j in 0..layer.weights.cols() {
                    let current_weight = *layer.weights.get(i, j)?;
                    let gradient = *weight_gradients[layer_idx].get(i, j)?;
                    let new_weight = current_weight - learning_rate * gradient / batch_size_f64;
                    layer.weights.set(i, j, new_weight)?;
                }
            }

            // Update biases
            for i in 0..layer.biases.rows() {
                for j in 0..layer.biases.cols() {
                    let current_bias = *layer.biases.get(i, j)?;
                    let gradient = *bias_gradients[layer_idx].get(i, j)?;
                    let new_bias = current_bias - learning_rate * gradient / batch_size_f64;
                    layer.biases.set(i, j, new_bias)?;
                }
            }
        }

        Ok(())
    }

    fn validate(
        &self,
        network: &NeuralNetwork<f64, impl ActivationFunction<f64>>,
        inputs: &[Matrix<f64>],
        targets: &[Matrix<f64>],
    ) -> NeuralNetworkResult<f64> {
        let mut total_loss = 0.0;

        for (input, target) in inputs.iter().zip(targets.iter()) {
            let prediction = network.forward(input)?;
            let loss = self.cost_function.cost(&prediction, target)?;
            total_loss += loss;
        }

        Ok(total_loss / inputs.len() as f64)
    }

    fn log_progress(&self, metric: &TrainingMetrics) {
        println!("Epoch {:>4}/{} | Loss: {:>10.6} | Val Loss: {:>10.6} | Time: {:>6.2}ms",
            metric.epoch,
            "?", // We don't have total epochs here
            metric.train_loss,
            metric.validation_loss.unwrap_or(0.0),
            metric.epoch_duration.as_millis()
        );
    }
}