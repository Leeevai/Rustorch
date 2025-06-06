pub mod error;
pub mod activations;
pub mod nn;

pub use error::{NeuralNetworkError, NeuralNetworkResult};
pub use activations::*;
pub use nn::*;

#[cfg(test)]
mod tests {
    use super::*;
    use matrix::Matrix;
    use approx::assert_relative_eq;

    #[test]
    fn test_neural_network_creation() {
        let nn = NeuralNetwork::new(vec![2, 3, 2, 4], Sigmoid, true);
        assert!(nn.is_ok());
        
        let nn = nn.unwrap();
        assert_eq!(nn.architecture(), &[2, 3, 2, 4]);
        assert_eq!(nn.input_size(), 2);
        assert_eq!(nn.output_size(), 4);
        assert_eq!(nn.num_layers(), 3);
        assert!(nn.is_concurrent());
    }

    #[test]
    fn test_neural_network_creation_with_macro() {
        let nn = nn![2, 3, 2, 4];
        assert!(nn.is_ok());
        
        let nn = nn.unwrap();
        assert_eq!(nn.architecture(), &[2, 3, 2, 4]);
        assert_eq!(nn.input_size(), 2);
        assert_eq!(nn.output_size(), 4);
        assert_eq!(nn.num_layers(), 3);
    }

    #[test]
    fn test_neural_network_creation_with_custom_activation() {
        let nn = nn![2, 3, 1; ReLU];
        assert!(nn.is_ok());
        
        let nn = nn.unwrap();
        assert_eq!(nn.architecture(), &[2, 3, 1]);
    }

    #[test]
    fn test_invalid_architecture() {
        let nn = NeuralNetwork::new(vec![1], Sigmoid, true);
        assert!(nn.is_err());
        match nn.unwrap_err() {
            NeuralNetworkError::InvalidArchitecture(_) => {},
            _ => panic!("Expected InvalidArchitecture error"),
        }

        let nn = NeuralNetwork::new(vec![0, 1], Sigmoid, true);
        assert!(nn.is_err());
        match nn.unwrap_err() {
            NeuralNetworkError::InvalidArchitecture(_) => {},
            _ => panic!("Expected InvalidArchitecture error"),
        }
    }

    #[test]
    fn test_xavier_initialization() {
        let mut nn = nn![3, 5, 2].unwrap();
        let result = nn.xavier_initialization();
        assert!(result.is_ok());

        // Check that weights are non-zero after initialization
        let layer = nn.get_layer(0).unwrap();
        let mut has_non_zero = false;
        for i in 0..layer.weights.rows() {
            for j in 0..layer.weights.cols() {
                if *layer.weights.get(i, j).unwrap() != 0.0 {
                    has_non_zero = true;
                    break;
                }
            }
        }
        assert!(has_non_zero);
    }

    #[test]
    fn test_he_initialization() {
        let mut nn = nn![3, 5, 2; ReLU].unwrap();
        let result = nn.he_initialization();
        assert!(result.is_ok());

        // Check that weights are non-zero after initialization
        let layer = nn.get_layer(0).unwrap();
        let mut has_non_zero = false;
        for i in 0..layer.weights.rows() {
            for j in 0..layer.weights.cols() {
                if *layer.weights.get(i, j).unwrap() != 0.0 {
                    has_non_zero = true;
                    break;
                }
            }
        }
        assert!(has_non_zero);
    }

    #[test]
    fn test_random_initialization() {
        let mut nn = nn![2, 3, 1].unwrap();
        let result = nn.random_initialization(-1.0, 1.0);
        assert!(result.is_ok());

        // Check that all weights are within range
        for layer_idx in 0..nn.num_layers() {
            let layer = nn.get_layer(layer_idx).unwrap();
            for i in 0..layer.weights.rows() {
                for j in 0..layer.weights.cols() {
                    let weight = *layer.weights.get(i, j).unwrap();
                    assert!(weight >= -1.0 && weight <= 1.0);
                }
            }
        }
    }

    #[test]
    fn test_forward_propagation() {
        let mut nn = nn![2, 3, 1].unwrap();
        nn.random_initialization(-0.5, 0.5).unwrap();

        let input = Matrix::from_vec(2, 1, vec![1.0, 0.5]).unwrap();
        let output = nn.forward(&input);
        assert!(output.is_ok());

        let output = output.unwrap();
        assert_eq!(output.dimensions(), (1, 1));
    }

    #[test]
    fn test_forward_propagation_invalid_input() {
        let mut nn = nn![2, 3, 1].unwrap();
        nn.random_initialization(-0.5, 0.5).unwrap();

        // Wrong input size
        let input = Matrix::from_vec(3, 1, vec![1.0, 0.5, 0.2]).unwrap();
        let output = nn.forward(&input);
        assert!(output.is_err());
        match output.unwrap_err() {
            NeuralNetworkError::InvalidInputSize { expected, actual } => {
                assert_eq!(expected, 2);
                assert_eq!(actual, 3);
            },
            _ => panic!("Expected InvalidInputSize error"),
        }

        // Wrong number of columns
        let input = Matrix::from_vec(2, 2, vec![1.0, 0.5, 0.2, 0.3]).unwrap();
        let output = nn.forward(&input);
        assert!(output.is_err());
    }

    #[test]
    fn test_forward_with_intermediates() {
        let mut nn = nn![2, 3, 1].unwrap();
        nn.random_initialization(-0.5, 0.5).unwrap();

        let input = Matrix::from_vec(2, 1, vec![1.0, 0.5]).unwrap();
        let outputs = nn.forward_with_intermediates(&input);
        assert!(outputs.is_ok());

        let outputs = outputs.unwrap();
        assert_eq!(outputs.len(), 4); // input + 3 layers
        assert_eq!(outputs[0].dimensions(), (2, 1)); // input
        assert_eq!(outputs[1].dimensions(), (3, 1)); // first hidden layer
        assert_eq!(outputs[2].dimensions(), (1, 1)); // output layer
    }

    #[test]
    fn test_layer_access() {
        let nn = nn![2, 3, 1].unwrap();
        
        let layer = nn.get_layer(0);
        assert!(layer.is_ok());
        let layer = layer.unwrap();
        assert_eq!(layer.input_size(), 2);
        assert_eq!(layer.output_size(), 3);

        let layer = nn.get_layer(1);
        assert!(layer.is_ok());
        let layer = layer.unwrap();
        assert_eq!(layer.input_size(), 3);
        assert_eq!(layer.output_size(), 1);

        let layer = nn.get_layer(2);
        assert!(layer.is_err());
        match layer.unwrap_err() {
            NeuralNetworkError::LayerIndexOutOfBounds { index, max } => {
                assert_eq!(index, 2);
                assert_eq!(max, 1);
            },
            _ => panic!("Expected LayerIndexOutOfBounds error"),
        }
    }

    #[test]
    fn test_concurrent_mode() {
        let mut nn = nn![2, 3, 1].unwrap();
        assert!(nn.is_concurrent());

        nn.set_concurrent(false);
        assert!(!nn.is_concurrent());

        nn.set_concurrent(true);
        assert!(nn.is_concurrent());
    }

    #[test]
    fn test_parameter_count() {
        let nn = nn![2, 3, 1].unwrap();
        // Layer 1: (2 * 3) weights + 3 biases = 9
        // Layer 2: (3 * 1) weights + 1 bias = 4
        // Total: 13
        assert_eq!(nn.parameter_count(), 13);

        let nn = nn![4, 5, 3, 2].unwrap();
        // Layer 1: (4 * 5) weights + 5 biases = 25
        // Layer 2: (5 * 3) weights + 3 biases = 18
        // Layer 3: (3 * 2) weights + 2 biases = 8
        // Total: 51
        assert_eq!(nn.parameter_count(), 51);
    }

    // Activation function tests
    #[test]
    fn test_sigmoid_activation() {
        let sigmoid = Sigmoid;
        let input = Matrix::from_vec(2, 1, vec![0.0, 1.0]).unwrap();
        
        let output = sigmoid.activate(&input).unwrap();
        assert_relative_eq!(*output.get(0, 0).unwrap(), 0.5, epsilon = 1e-10);
        assert_relative_eq!(*output.get(1, 0).unwrap(), 1.0 / (1.0 + (-1.0_f64).exp()), epsilon = 1e-10);

        let derivative = sigmoid.derivative(&input).unwrap();
        assert!(derivative.get(0, 0).is_ok());
        assert!(derivative.get(1, 0).is_ok());
    }

    #[test]
    fn test_relu_activation() {
        let relu = ReLU;
        let input = Matrix::from_vec(3, 1, vec![-1.0, 0.0, 1.0]).unwrap();
        
        let output = relu.activate(&input).unwrap();
        assert_eq!(*output.get(0, 0).unwrap(), 0.0);
        assert_eq!(*output.get(1, 0).unwrap(), 0.0);
        assert_eq!(*output.get(2, 0).unwrap(), 1.0);

        let derivative = relu.derivative(&input).unwrap();
        assert_eq!(*derivative.get(0, 0).unwrap(), 0.0);
        assert_eq!(*derivative.get(1, 0).unwrap(), 0.0);
        assert_eq!(*derivative.get(2, 0).unwrap(), 1.0);
    }

    #[test]
    fn test_tanh_activation() {
        let tanh = Tanh;
        let input = Matrix::from_vec(2, 1, vec![0.0, 1.0]).unwrap();
        
        let output = tanh.activate(&input).unwrap();
        assert_relative_eq!(*output.get(0, 0).unwrap(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(*output.get(1, 0).unwrap(), 1.0_f64.tanh(), epsilon = 1e-10);

        let derivative = tanh.derivative(&input).unwrap();
        assert!(derivative.get(0, 0).is_ok());
        assert!(derivative.get(1, 0).is_ok());
    }

    #[test]
    fn test_linear_activation() {
        let linear = Linear;
        let input = Matrix::from_vec(2, 1, vec![2.5, -1.5]).unwrap();
        
        let output = linear.activate(&input).unwrap();
        assert_eq!(*output.get(0, 0).unwrap(), 2.5);
        assert_eq!(*output.get(1, 0).unwrap(), -1.5);

        let derivative = linear.derivative(&input).unwrap();
        assert_eq!(*derivative.get(0, 0).unwrap(), 1.0);
        assert_eq!(*derivative.get(1, 0).unwrap(), 1.0);
    }

    #[test]
    fn test_leaky_relu_activation() {
        let leaky_relu = LeakyReLU::new(0.1);
        let input = Matrix::from_vec(3, 1, vec![-1.0, 0.0, 1.0]).unwrap();
        
        let output = leaky_relu.activate(&input).unwrap();
        assert_eq!(*output.get(0, 0).unwrap(), -0.1);
        assert_eq!(*output.get(1, 0).unwrap(), 0.0);
        assert_eq!(*output.get(2, 0).unwrap(), 1.0);

        let derivative = leaky_relu.derivative(&input).unwrap();
        assert_eq!(*derivative.get(0, 0).unwrap(), 0.1);
        assert_eq!(*derivative.get(1, 0).unwrap(), 0.1);
        assert_eq!(*derivative.get(2, 0).unwrap(), 1.0);
    }

    #[test]
    fn test_sequential_vs_concurrent() {
        let mut nn_concurrent = nn![2, 3, 1].unwrap();
        let mut nn_sequential = nn![2, 3, 1].unwrap();
        
        nn_concurrent.set_concurrent(true);
        nn_sequential.set_concurrent(false);

        // Use same initialization
        nn_concurrent.random_initialization(-0.5, 0.5).unwrap();
        nn_sequential.random_initialization(-0.5, 0.5).unwrap();

        // Copy weights to make them identical
        for layer_idx in 0..nn_concurrent.num_layers() {
            let concurrent_layer = nn_concurrent.get_layer(layer_idx).unwrap();
            let sequential_layer = nn_sequential.get_layer_mut(layer_idx).unwrap();
            
            for i in 0..concurrent_layer.weights.rows() {
                for j in 0..concurrent_layer.weights.cols() {
                    let weight = *concurrent_layer.weights.get(i, j).unwrap();
                    sequential_layer.weights.set(i, j, weight).unwrap();
                }
            }
            
            for i in 0..concurrent_layer.biases.rows() {
                let bias = *concurrent_layer.biases.get(i, 0).unwrap();
                sequential_layer.biases.set(i, 0, bias).unwrap();
            }
        }

        let input = Matrix::from_vec(2, 1, vec![1.0, 0.5]).unwrap();
        
        let output_concurrent = nn_concurrent.forward(&input).unwrap();
        let output_sequential = nn_sequential.forward(&input).unwrap();

        // Results should be identical
        assert_relative_eq!(
            *output_concurrent.get(0, 0).unwrap(),
            *output_sequential.get(0, 0).unwrap(),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_complex_network() {
        let mut nn = nn![4, 8, 6, 3, 1; ReLU].unwrap();
        nn.he_initialization().unwrap();

        let input = Matrix::from_vec(4, 1, vec![1.0, -0.5, 0.8, 0.2]).unwrap();
        let output = nn.forward(&input).unwrap();
        
        assert_eq!(output.dimensions(), (1, 1));
        assert!(output.get(0, 0).is_ok());

        // Test with intermediates
        let intermediates = nn.forward_with_intermediates(&input).unwrap();
        assert_eq!(intermediates.len(), 5); // input + 4 layers
        assert_eq!(intermediates[0].dimensions(), (4, 1)); // input
        assert_eq!(intermediates[1].dimensions(), (8, 1)); // first hidden
        assert_eq!(intermediates[2].dimensions(), (6, 1)); // second hidden
        assert_eq!(intermediates[3].dimensions(), (3, 1)); // third hidden
        assert_eq!(intermediates[4].dimensions(), (1, 1)); // output
    }

    #[test]
    fn test_error_propagation() {
        let nn = nn![2, 3, 1].unwrap();
        
        // Test matrix error propagation
        let invalid_input = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).unwrap();
        let result = nn.forward(&invalid_input);
        assert!(result.is_err());
        
        match result.unwrap_err() {
            NeuralNetworkError::InvalidInputSize { .. } => {},
            _ => panic!("Expected InvalidInputSize error"),
        }
    }

    #[test]
    fn test_activation_function_names() {
        assert_eq!(Sigmoid.name(), "sigmoid");
        assert_eq!(ReLU.name(), "relu");
        assert_eq!(Tanh.name(), "tanh");
        assert_eq!(Linear.name(), "linear");
        assert_eq!(LeakyReLU::default().name(), "leaky_relu");
    }
}