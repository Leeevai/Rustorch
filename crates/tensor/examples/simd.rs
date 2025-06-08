use tensor::{Tensor, TensorError};
use tensor::ops::{TensorOps, ComputeMode};
use std::time::Instant;

fn create_large_tensors(size: usize) -> (Tensor, Tensor) {
    let data_a: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
    let data_b: Vec<f32> = (0..size).map(|i| (i as f32) * 0.2 + 1.0).collect();
    
    let shape = vec![size];
    let tensor_a = Tensor::new(data_a, &shape).unwrap();
    let tensor_b = Tensor::new(data_b, &shape).unwrap();
    
    (tensor_a, tensor_b)
}

fn benchmark_operation<F>(name: &str, operation: F) -> std::time::Duration 
where 
    F: FnOnce() -> Result<Tensor, TensorError>
{
    println!("Running {}", name);
    let start = Instant::now();
    let result = operation();
    let duration = start.elapsed();
    
    match result {
        Ok(tensor) => {
            println!("  ✓ Success - Duration: {:?}", duration);
            println!("  ✓ Result sum: {:.6}", tensor.data().iter().sum::<f32>());
        }
        Err(e) => {
            println!("  ✗ Error: {}", e);
        }
    }
    
    duration
}

fn main() {
    println!("🚀 High-Performance Tensor Library Benchmark");
    println!("{}", "=".repeat(50));
    
    let ops = TensorOps::new();
    let (simd_width, supports_avx2, supports_avx512) = ops.get_simd_info();
    
    println!("🔧 System Information:");
    println!("  CPU Cores: {}", num_cpus::get());
    println!("  SIMD Width: {} elements", simd_width);
    println!("  AVX2 Support: {}", supports_avx2);
    println!("  AVX512 Support: {}", supports_avx512);
    println!();

    // Test different sizes
    let sizes = vec![1_000, 100_000, 1_000_000,100_000_000];
    
    for &size in &sizes {
        println!("📊 Testing with tensor size: {}", size);
        println!("{}", "-".repeat(40));
        
        let (tensor_a, tensor_b) = create_large_tensors(size);
        
        // Addition benchmarks
        println!("\n🔢 Addition Operations:");
        let single_add = benchmark_operation("Single-threaded Add", || {
            ops.add(&tensor_a, &tensor_b, ComputeMode::Single)
        });
        
        let multi_add = benchmark_operation("Multi-threaded Add", || {
            ops.add(&tensor_a, &tensor_b, ComputeMode::MultiThread)
        });
        
        let simd_multi_add = benchmark_operation("SIMD + Multi-threaded Add", || {
            ops.add(&tensor_a, &tensor_b, ComputeMode::SimdMultiThread)
        });

        // Multiplication benchmarks
        println!("\n✖️  Multiplication Operations:");
        let single_mul = benchmark_operation("Single-threaded Multiply", || {
            ops.multiply(&tensor_a, &tensor_b, ComputeMode::Single)
        });
        
        let multi_mul = benchmark_operation("Multi-threaded Multiply", || {
            ops.multiply(&tensor_a, &tensor_b, ComputeMode::MultiThread)
        });
        
        let simd_multi_mul = benchmark_operation("SIMD + Multi-threaded Multiply", || {
            ops.multiply(&tensor_a, &tensor_b, ComputeMode::SimdMultiThread)
        });

        // Performance comparison
        println!("\n📈 Performance Comparison (Addition):");
        let speedup_multi = single_add.as_nanos() as f64 / multi_add.as_nanos() as f64;
        let speedup_simd = single_add.as_nanos() as f64 / simd_multi_add.as_nanos() as f64;
        
        println!("  Multi-thread speedup: {:.2}x", speedup_multi);
        println!("  SIMD+Multi speedup: {:.2}x", speedup_simd);
        
        println!("\n📈 Performance Comparison (Multiplication):");
        let speedup_multi_mul = single_mul.as_nanos() as f64 / multi_mul.as_nanos() as f64;
        let speedup_simd_mul = single_mul.as_nanos() as f64 / simd_multi_mul.as_nanos() as f64;
        
        println!("  Multi-thread speedup: {:.2}x", speedup_multi_mul);
        println!("  SIMD+Multi speedup: {:.2}x", speedup_simd_mul);
        
        println!("\n{}", "=".repeat(50));
    }

    // Test scalar operations
    println!("\n🔢 Scalar Operations Test:");
    let (tensor_a, _) = create_large_tensors(100_000);
    
    let scalar_single = {
        let start = Instant::now();
        let result = ops.scalar_multiply(&tensor_a, 2.5, ComputeMode::Single);
        let duration = start.elapsed();
        println!("Single-threaded Scalar Multiply: {:?}", duration);
        duration
    };
    
    let scalar_simd = {
        let start = Instant::now();
        let result = ops.scalar_multiply(&tensor_a, 2.5, ComputeMode::SimdMultiThread);
        let duration = start.elapsed();
        println!("SIMD+Multi Scalar Multiply: {:?}", duration);
        duration
    };
    
    let scalar_speedup = scalar_single.as_nanos() as f64 / scalar_simd.as_nanos() as f64;
    println!("Scalar operation speedup: {:.2}x", scalar_speedup);

    // Test tensor creation and basic operations
    println!("\n🧪 Basic Tensor Operations:");
    
    let small_tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    println!("Created tensor with shape {:?}", small_tensor.shape());
    println!("Tensor rank: {}", small_tensor.rank());
    println!("Tensor size: {}", small_tensor.size());
    
    let zeros_tensor = Tensor::zeros(&[3, 3]);
    let ones_tensor = Tensor::ones(&[3, 3]);
    
    println!("Sum of zeros tensor: {}", ops.sum(&zeros_tensor, ComputeMode::Single));
    println!("Sum of ones tensor: {}", ops.sum(&ones_tensor, ComputeMode::Single));

    println!("\n✅ All benchmarks completed!");
}