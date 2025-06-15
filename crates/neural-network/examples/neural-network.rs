// use NeuralNetworks::*;
// use matrix::Matrix;

// fn main() -> Result<(), Box<dyn std::error::Error>> {
//     println!("=== Neural Network Library Examples ===\n");

//     // Example 1: Creating a simple neural network
//     println!("1. Creating a simple neural network with macro:");
//     let mut nn = nn![2, 3, 2, 4]?;
//     println!("   Architecture: {:?}", nn.architecture());
//     println!("   Input size: {}, Output size: {}", nn.input_size(), nn.output_size());
//     println!("   Number of layers: {}", nn.num_layers());
//     println!("   Total parameters: {}\n", nn.parameter_count());

//     // Example 2: Creating with custom activation function
//     println!("2. Creating with ReLU activation:");
//     let mut relu_nn = nn![3, 5, 2; ReLU]?;
//     println!("   Architecture: {:?}", relu_nn.architecture());
//     println!("   Using ReLU activation function\n");

//     // Example 3: Different initialization methods
//     println!("3. Testing different initialization methods:");
    
//     println!("   Xavier initialization...");
//     nn.xavier_initialization()?;
//     let layer = nn.get_layer(0)?;
//     println!("   First layer weight sample: {:.4}", layer.weights.get(0, 0)?);

//     println!("   He initialization...");
//     relu_nn.he_initialization()?;
//     let layer = relu_nn.get_layer(0)?;
//     println!("   First layer weight sample: {:.4}", layer.weights.get(0, 0)?);

//     println!("   Random initialization...");
//     let mut random_nn = nn![2, 4, 1; Tanh]?;
//     random_nn.random_initialization(-1.0, 1.0)?;
//     let layer = random_nn.get_layer(0)?;
//     println!("   First layer weight sample: {:.4}\n", layer.weights.get(0, 0)?);

//     // Example 4: Forward propagation
//     println!("4. Forward propagation:");
//     let input = Matrix::from_vec(2, 1, vec![1.5, -0.5])?;
//     println!("   Input: [{:.1}, {:.1}]", input.get(0, 0)?, input.get(1, 0)?);
    
//     let output = nn.forward(&input)?;
//     println!("   Output shape: {:?}", output.dimensions());
//     print!("   Output values: [");
//     for i in 0..output.rows() {
//         print!("{:.4}", output.get(i, 0)?);
//         if i < output.rows() - 1 { print!(", "); }
//     }
//     println!("]\n");

//     // Example 5: Forward propagation with intermediates
//     println!("5. Forward propagation with intermediate outputs:");
//     let intermediates = nn.forward_with_intermediates(&input)?;
//     for (i, intermediate) in intermediates.iter().enumerate() {
//         if i == 0 {
//             println!("   Input layer: {:?}", intermediate.dimensions());
//         } else {
//             println!("   Layer {} output: {:?}", i, intermediate.dimensions());
//         }
//     }
//     println!();

//     // Example 6: Testing different activation functions
//     println!("6. Testing different activation functions:");
//     let test_input = Matrix::from_vec(3, 1, vec![-1.0, 0.0, 1.0])?;
    
//     // Sigmoid
//     let sigmoid = Sigmoid;
//     let sigmoid_output = sigmoid.activate(&test_input)?;
//     println!("   Sigmoid output: [{:.4}, {:.4}, {:.4}]", 
//              sigmoid_output.get(0, 0)?, sigmoid_output.get(1, 0)?, sigmoid_output.get(2, 0)?);

//     // ReLU
//     let relu = ReLU;
//     let relu_output = relu.activate(&test_input)?;
//     println!("   ReLU output: [{:.4}, {:.4}, {:.4}]", 
//              relu_output.get(0, 0)?, relu_output.get(1, 0)?, relu_output.get(2, 0)?);

//     // Tanh
//     let tanh = Tanh;
//     let tanh_output = tanh.activate(&test_input)?;
//     println!("   Tanh output: [{:.4}, {:.4}, {:.4}]", 
//              tanh_output.get(0, 0)?, tanh_output.get(1, 0)?, tanh_output.get(2, 0)?);

//     // Leaky ReLU
//     let leaky_relu = LeakyReLU::new(0.1);
//     let leaky_output = leaky_relu.activate(&test_input)?;
//     println!("   Leaky ReLU output: [{:.4}, {:.4}, {:.4}]\n", 
//              leaky_output.get(0, 0)?, leaky_output.get(1, 0)?, leaky_output.get(2, 0)?);

//     // Example 7: Performance comparison
//     println!("7. Performance comparison (Sequential vs Concurrent):");
//     let mut sequential_nn = nn![10, 50, 50, 10]?;
//     let mut concurrent_nn = nn![10, 50, 50, 10]?;
    
//     sequential_nn.set_concurrent(false);
//     concurrent_nn.set_concurrent(true);
    
//     sequential_nn.xavier_initialization()?;
//     concurrent_nn.xavier_initialization()?;

//     let large_input = Matrix::from_vec(10, 1, vec![1.0; 10])?;
    
//     println!("   Sequential mode: {}", !sequential_nn.is_concurrent());
//     println!("   Concurrent mode: {}", concurrent_nn.is_concurrent());
    
//     // Time the operations (simplified timing)
//     use std::time::Instant;
    
//     let start = Instant::now();
//     for _ in 0..100 {
//         let _ = sequential_nn.forward(&large_input)?;
//     }
//     let sequential_time = start.elapsed();

//     let start = Instant::now();
//     for _ in 0..100 {
//         let _ = concurrent_nn.forward(&large_input)?;
//     }
//     let concurrent_time = start.elapsed();
    
//     println!("   Sequential time: {:?}", sequential_time);
//     println!("   Concurrent time: {:?}\n", concurrent_time);

//     // Example 8: Error handling
//     println!("8. Error handling examples:");
    
//     // Wrong input size
//     let wrong_input = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0])?;
//     match nn.forward(&wrong_input) {
//         Err(NeuralNetworkError::InvalidInputSize { expected, actual }) => {
//             println!("   ✓ Caught invalid input size error: expected {}, got {}", expected, actual);
//         },
//         _ => println!("   ✗ Expected InvalidInputSize error"),
//     }

//     // Invalid architecture
//     match nn![1] {
//         Err(NeuralNetworkError::InvalidArchitecture(msg)) => {
//             println!("   ✓ Caught invalid architecture error: {}", msg);
//         },
//         _ => println!("   ✗ Expected InvalidArchitecture error"),
//     }

//     // Layer index out of bounds
//     match nn.get_layer(10) {
//         Err(NeuralNetworkError::LayerIndexOutOfBounds { index, max }) => {
//             println!("   ✓ Caught layer index error: index {} out of bounds (max: {})", index, max);
//         },
//         _ => println!("   ✗ Expected LayerIndexOutOfBounds error"),
//     }

//     println!("\n=== All examples completed successfully! ===");
//     Ok(())
// }

fn main()
{}