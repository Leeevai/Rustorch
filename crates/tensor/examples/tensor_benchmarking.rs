use std::time::Instant;
use std::fmt;
use std::iter;

// Assuming these are your existing modules
use tensor::tensor::Tensor;
use tensor::error::TensorResult;
use tensor::ExecutionMode;


#[derive(Debug)]
struct BenchmarkResult {
    mode: ExecutionMode,
    duration_ms: f64,
    speedup: f64,
}

impl fmt::Display for BenchmarkResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:<15} | {:>8.2} ms | {:>6.2}x speedup", 
               format!("{}", self.mode), self.duration_ms, self.speedup)
    }
}

pub struct TensorBenchmark;

impl TensorBenchmark {
    /// Run comprehensive benchmarks on tensor operations
    pub fn run_full_benchmark() {
        println!("{}", iter::repeat("=").take(80).collect::<String>());
        println!("TENSOR OPERATIONS COMPREHENSIVE BENCHMARK");
        println!("{}", iter::repeat("=").take(80).collect::<String>());
        
        Self::benchmark_element_wise_operations();
        Self::benchmark_matrix_vector_multiplication();
        Self::benchmark_matrix_matrix_multiplication();
        Self::benchmark_scaling_analysis();
    }

    /// Benchmark element-wise operations (addition, subtraction, Hadamard product)
    fn benchmark_element_wise_operations() {
        println!("\n🧮 ELEMENT-WISE OPERATIONS BENCHMARK");
        println!("{}", iter::repeat("-").take(50).collect::<String>());
        
        let sizes = vec![
            (100, 100),      // Small matrices
            (500, 500),      // Medium matrices  
            (1000, 1000),    // Large matrices
            (2000, 2000),    // Very large matrices
        ];

        for (rows, cols) in sizes {
            println!("\nMatrix size: {}x{}", rows, cols);
            println!("{:<15} | {:>10} | {:>15}", "Operation", "Time (ms)", "Elements/sec");
            println!("{}", iter::repeat("-").take(45).collect::<String>());

            let a = Tensor::random(&[rows, cols], 42);
            let b = Tensor::random(&[rows, cols], 123);
            let elements = (rows * cols) as f64;

            // Addition benchmark
            let start = Instant::now();
            let _result = &a + &b;
            let duration = start.elapsed();
            let duration_ms = duration.as_secs_f64() * 1000.0;
            let elements_per_sec = elements / duration.as_secs_f64();
            
            println!("{:<15} | {:>8.2} ms | {:>11.0} elem/s", 
                     "Addition", duration_ms, elements_per_sec);

            // Subtraction benchmark
            let start = Instant::now();
            let _result = &a - &b;
            let duration = start.elapsed();
            let duration_ms = duration.as_secs_f64() * 1000.0;
            let elements_per_sec = elements / duration.as_secs_f64();
            
            println!("{:<15} | {:>8.2} ms | {:>11.0} elem/s", 
                     "Subtraction", duration_ms, elements_per_sec);

            // Hadamard product benchmark
            let start = Instant::now();
            let _result = a.hadamard(&b).unwrap();
            let duration = start.elapsed();
            let duration_ms = duration.as_secs_f64() * 1000.0;
            let elements_per_sec = elements / duration.as_secs_f64();
            
            println!("{:<15} | {:>8.2} ms | {:>11.0} elem/s", 
                     "Hadamard", duration_ms, elements_per_sec);

            // Scaling benchmark
            let start = Instant::now();
            let _result = a.scale(2.5);
            let duration = start.elapsed();
            let duration_ms = duration.as_secs_f64() * 1000.0;
            let elements_per_sec = elements / duration.as_secs_f64();
            
            println!("{:<15} | {:>8.2} ms | {:>11.0} elem/s", 
                     "Scaling", duration_ms, elements_per_sec);
        }
    }

    /// Benchmark matrix-vector multiplication across all execution modes
    fn benchmark_matrix_vector_multiplication() {
        println!("\n🚀 MATRIX-VECTOR MULTIPLICATION BENCHMARK");
        println!("{}", iter::repeat("-").take(60).collect::<String>());
        
        let matrix_sizes = vec![
            (512, 512),
            (1024, 1024),
            (2048, 2048),
            (4096, 1024),  // Tall matrix
            (1024, 4096),  // Wide matrix
        ];

        for (rows, cols) in matrix_sizes {
            println!("\nMatrix-Vector: {}x{} * {}x1", rows, cols, cols);
            println!("{:<15} | {:>10} | {:>15}", "Mode", "Time (ms)", "Speedup");
            println!("{}","-".repeat(45));

            let matrix = Tensor::random(&[rows, cols], 42);
            let vector = Tensor::random(&[cols, 1], 123);
            
            let results = Self::benchmark_matrix_operations(&matrix, &vector);
            Self::print_benchmark_results(&results);
        }
    }

    /// Benchmark matrix-matrix multiplication across all execution modes
    fn benchmark_matrix_matrix_multiplication() {
        println!("\n⚡ MATRIX-MATRIX MULTIPLICATION BENCHMARK");
        println!("{}","-".repeat(60));
        
        let matrix_configs = vec![
            (256, 256, 256),   // Small square matrices
            (512, 512, 512),   // Medium square matrices  
            (1024, 512, 256),  // Rectangular matrices
            (256, 1024, 512),  // Different aspect ratios
            (1024, 1024, 1024), // Large square matrices
        ];

        for (m, k, n) in matrix_configs {
            println!("\nMatrix-Matrix: {}x{} * {}x{}", m, k, k, n);
            println!("{:<15} | {:>10} | {:>15}", "Mode", "Time (ms)", "Speedup");
            println!("{}","-".repeat(45));

            let matrix_a = Tensor::random(&[m, k], 42);
            let matrix_b = Tensor::random(&[k, n], 123);
            
            let results = Self::benchmark_matrix_operations(&matrix_a, &matrix_b);
            Self::print_benchmark_results(&results);
        }
    }

    /// Analyze how performance scales with matrix size
    fn benchmark_scaling_analysis() {
        println!("\n📈 SCALING ANALYSIS");
        println!("{}","-".repeat(60));
        
        let sizes = vec![128, 256, 512, 768, 1024, 1536];
        
        println!("\nMatrix-Matrix Multiplication Scaling (NxN * NxN):");
        println!("{:<8} | {:<12} | {:<12} | {:<12} | {:<12}", 
                 "Size", "Sequential", "Parallel", "SIMD", "Par+SIMD");
        println!("{}","-".repeat(70));

        for size in sizes {
            let matrix_a = Tensor::random(&[size, size], 42);
            let matrix_b = Tensor::random(&[size, size], 123);
            
            let results = Self::benchmark_matrix_operations(&matrix_a, &matrix_b);
            
            print!("{:<8}", size);
            for result in &results {
                print!(" | {:>10.2} ms", result.duration_ms);
            }
            println!();
        }

        // Memory bandwidth analysis
        println!("\n💾 MEMORY BANDWIDTH ANALYSIS");
        println!("{}","-".repeat(50));
        
        let size = 1024;
        let matrix_a = Tensor::random(&[size, size], 42);
        let matrix_b = Tensor::random(&[size, size], 123);
        
        // Calculate theoretical memory usage
        let input_data = (size * size * 2) as f64 * 4.0; // Two matrices, 4 bytes per f32
        let output_data = (size * size) as f64 * 4.0;    // Result matrix
        let total_memory_mb = (input_data + output_data) / (1024.0 * 1024.0);
        
        println!("Matrix size: {}x{}", size, size);
        println!("Memory usage: {:.2} MB", total_memory_mb);
        println!();
        
        let results = Self::benchmark_matrix_operations(&matrix_a, &matrix_b);
        
        println!("{:<15} | {:>10} | {:>15}", "Mode", "Time (ms)", "Bandwidth (GB/s)");
        println!("{}","-".repeat(45));
        
        for result in results {
            let bandwidth_gb_s = (total_memory_mb / 1024.0) / (result.duration_ms / 1000.0);
            println!("{:<15} | {:>8.2} ms | {:>13.2} GB/s", 
                     result.mode, result.duration_ms, bandwidth_gb_s);
        }
    }

    /// Benchmark a tensor operation across all execution modes
    fn benchmark_matrix_operations(a: &Tensor, b: &Tensor) -> Vec<BenchmarkResult> {
        let modes = vec![
            ExecutionMode::Sequential,
            ExecutionMode::Parallel,
            ExecutionMode::SIMD,
            ExecutionMode::ParallelSIMD,
        ];

        let mut results = Vec::new();
        let mut baseline_time = 0.0;

        for (i, mode) in modes.iter().enumerate() {
            // Warm up
            for _ in 0..3 {
                let _ = a.multiply(b, *mode);
            }

            // Actual benchmark - run multiple times for accuracy
            let iterations = 5;
            let mut total_duration = 0.0;

            for _ in 0..iterations {
                let start = Instant::now();
                let _result = a.multiply(b, *mode).unwrap();
                total_duration += start.elapsed().as_secs_f64();
            }

            let avg_duration = total_duration / iterations as f64;
            let duration_ms = avg_duration * 1000.0;

            if i == 0 {
                baseline_time = duration_ms;
            }

            let speedup = if duration_ms > 0.0 { baseline_time / duration_ms } else { 0.0 };

            results.push(BenchmarkResult {
                mode: *mode,
                duration_ms,
                speedup,
            });
        }

        results
    }

    /// Print benchmark results in a formatted table
    fn print_benchmark_results(results: &[BenchmarkResult]) {
        for result in results {
            println!("{}", result);
        }
    }

    /// Test correctness of all execution modes
    pub fn test_correctness() {
        println!("\n✅ CORRECTNESS TESTING");
        println!("{}","-".repeat(40));

        // Test matrix-vector multiplication
        let matrix = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3]
        ).unwrap();
        let vector = Tensor::new(vec![1.0, 2.0, 3.0], &[3, 1]).unwrap();

        println!("Testing Matrix-Vector multiplication:");
        println!("Matrix: 2x3, Vector: 3x1");

        let modes = vec![
            ExecutionMode::Sequential,
            ExecutionMode::Parallel,
            ExecutionMode::SIMD,
            ExecutionMode::ParallelSIMD,
        ];

        let mut reference_result = None;
        let mut all_correct = true;

        for mode in modes.clone() {
            match matrix.multiply(&vector, mode) {
                Ok(result) => {
                    println!("{:<15}: {:?}", format!("{}", mode), result.data());
                    
                    if reference_result.is_none() {
                        reference_result = Some(result);
                    } else if let Some(ref reference) = reference_result {
                        if result != *reference {
                            println!("❌ {} produces different result!", mode);
                            all_correct = false;
                        }
                    }
                }
                Err(e) => {
                    println!("❌ {} failed: {:?}", mode, e);
                    all_correct = false;
                }
            }
        }

        // Test matrix-matrix multiplication
        println!("\nTesting Matrix-Matrix multiplication:");
        let matrix_a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let matrix_b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
        println!("Matrix A: 2x2, Matrix B: 2x2");

        reference_result = None;

        for mode in modes {
            match matrix_a.multiply(&matrix_b, mode) {
                Ok(result) => {
                    println!("{:<15}: {:?}", format!("{}", mode), result.data());
                    
                    if reference_result.is_none() {
                        reference_result = Some(result);
                    } else if let Some(ref reference) = reference_result {
                        if result != *reference {
                            println!("❌ {} produces different result!", mode);
                            all_correct = false;
                        }
                    }
                }
                Err(e) => {
                    println!("❌ {} failed: {:?}", mode, e);
                    all_correct = false;
                }
            }
        }

        if all_correct {
            println!("✅ All execution modes produce identical results!");
        } else {
            println!("❌ Some execution modes produce different results!");
        }
    }

    /// Create various tensor configurations for testing
    pub fn create_test_tensors() -> Vec<(String, Tensor, Tensor)> {
        vec![
            // Small tensors
            ("Small 4x4 matrices".to_string(),
             Tensor::random(&[4, 4], 42),
             Tensor::random(&[4, 4], 123)),
            
            // Medium tensors
            ("Medium 100x100 matrices".to_string(),
             Tensor::random(&[100, 100], 42),
             Tensor::random(&[100, 100], 123)),
            
            // Large square tensors
            ("Large 1000x1000 matrices".to_string(),
             Tensor::random(&[1000, 1000], 42),
             Tensor::random(&[1000, 1000], 123)),
            
            // Rectangular tensors
            ("Tall 2000x500 matrices".to_string(),
             Tensor::random(&[2000, 500], 42),
             Tensor::random(&[500, 200], 123)),
            
            // Matrix-vector pairs
            ("Matrix 1000x1000 with vector 1000x1".to_string(),
             Tensor::random(&[1000, 1000], 42),
             Tensor::random(&[1000, 1], 123)),
        ]
    }
}

// Main function to run the benchmark
fn main() {
    run_benchmark_suite();
}

// Example usage and main benchmark runner
pub fn run_benchmark_suite() {
    println!("Starting comprehensive tensor benchmark suite...\n");
    
    // Test correctness first
    TensorBenchmark::test_correctness();
    
    // Run full benchmark
    TensorBenchmark::run_full_benchmark();
    
    // Additional custom tests
    println!("\n🔧 CUSTOM TENSOR CONFIGURATIONS");
    println!("{}","-".repeat(50));
    
    let test_tensors = TensorBenchmark::create_test_tensors();
    
    for (description, tensor_a, tensor_b) in test_tensors {
        if tensor_a.shape()[1] == tensor_b.shape()[0] {
            println!("\n{}", description);
            println!("{:<15} | {:>10} | {:>15}", "Mode", "Time (ms)", "Speedup");
            println!("{}","-".repeat(45));
            
            let results = TensorBenchmark::benchmark_matrix_operations(&tensor_a, &tensor_b);
            TensorBenchmark::print_benchmark_results(&results);
        }
    }
    
    println!("\n🎉 Benchmark suite completed!");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_runs() {
        // Test that benchmark functions don't panic
        let matrix = Tensor::random(&[10, 10], 42);
        let vector = Tensor::random(&[10, 1], 123);
        
        let results = TensorBenchmark::benchmark_matrix_operations(&matrix, &vector);
        assert_eq!(results.len(), 4); // Should have 4 execution modes
        
        // All results should have positive durations
        for result in results {
            assert!(result.duration_ms > 0.0);
        }
    }

    #[test]
    fn test_correctness() {
        // This test ensures all modes produce the same results
        TensorBenchmark::test_correctness();
    }
}