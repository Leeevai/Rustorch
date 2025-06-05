use std::time::Instant;
use std::fmt;
// Fixed import - assuming Matrix is from a crate called "matrix" or similar
// Replace "matrix" with your actual crate name
use matrix::{Matrix, MatrixError, MatrixResult};

// Helper function to format duration
fn format_duration(duration: std::time::Duration) -> String {
    let nanos = duration.as_nanos();
    if nanos < 1_000 {
        format!("{}ns", nanos)
    } else if nanos < 1_000_000 {
        format!("{:.2}μs", nanos as f64 / 1_000.0)
    } else if nanos < 1_000_000_000 {
        format!("{:.2}ms", nanos as f64 / 1_000_000.0)
    } else {
        format!("{:.2}s", duration.as_secs_f64())
    }
}

// Helper function to print matrix (only for small matrices)
// Added trait bounds for T
fn print_matrix<T>(matrix: &Matrix<T>, name: &str) -> MatrixResult<()> 
where 
    T: fmt::Display + Copy + Default + Send + Sync + std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    let (rows, cols) = matrix.dimensions();
    if rows <= 5 && cols <= 5 {
        println!("\n{} ({}x{}):", name, rows, cols);
        for i in 0..rows {
            print!("  ");
            for j in 0..cols {
                print!("{:8.2} ", matrix.get(i, j)?.to_string().parse::<f64>().unwrap_or(0.0));
            }
            println!();
        }
    } else {
        println!("\n{} ({}x{}) - Matrix too large to display", name, rows, cols);
    }
    Ok(())
}

// Performance comparison function
fn benchmark_operation<F, R>(name: &str, operation: F) -> (R, std::time::Duration)
where 
    F: FnOnce() -> R,
{
    let start = Instant::now();
    let result = operation();
    let duration = start.elapsed();
    println!("  {} took: {}", name, format_duration(duration));
    (result, duration)
}

// Compare two operations and show speedup/slowdown
fn compare_operations<F1, F2, R>(
    name: &str,
    sequential_op: F1,
    concurrent_op: F2,
) -> (R, R, f64)
where
    F1: FnOnce() -> R,
    F2: FnOnce() -> R,
{
    println!("\n{}:", name);
    
    let (seq_result, seq_duration) = benchmark_operation("Sequential", sequential_op);
    let (conc_result, conc_duration) = benchmark_operation("Concurrent ", concurrent_op);
    
    let speedup = seq_duration.as_nanos() as f64 / conc_duration.as_nanos() as f64;
    
    if speedup > 1.0 {
        println!("  → Concurrent is {:.2}x faster", speedup);
    } else if speedup < 1.0 {
        println!("  → Sequential is {:.2}x faster", 1.0 / speedup);
    } else {
        println!("  → Performance is equivalent");
    }
    
    (seq_result, conc_result, speedup)
}

fn main() -> MatrixResult<()> {
    println!("=== Concurrent Matrix Operations Demo ===\n");

    // Test 1: Basic Matrix Creation and Properties
    println!("1. Matrix Creation and Basic Properties");
    println!("=====================================");

    let matrix_3x3 = Matrix::<f64>::from_vec(3, 3, vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0
    ])?;
    print_matrix(&matrix_3x3, "3x3 Matrix")?;

    let identity_3x3 = Matrix::<f64>::identity(3)?;
    print_matrix(&identity_3x3, "3x3 Identity Matrix")?;

    let zeros_2x4 = Matrix::<f64>::zeros(2, 4)?;
    print_matrix(&zeros_2x4, "2x4 Zeros Matrix")?;

    let ones_3x2 = Matrix::<f64>::ones(3, 2)?;
    print_matrix(&ones_3x2, "3x2 Ones Matrix")?;

    println!("\nMatrix Properties:");
    println!("  3x3 Matrix - Dimensions: {:?}", matrix_3x3.dimensions());
    println!("  3x3 Matrix - Is Square: {}", matrix_3x3.is_square());
    println!("  3x3 Matrix - Is Empty: {}", matrix_3x3.is_empty());
    println!("  3x3 Matrix - Rows: {}", matrix_3x3.rows());
    println!("  3x3 Matrix - Cols: {}", matrix_3x3.cols());

    // Test 2: Element Access and Modification
    println!("\n\n2. Element Access and Modification");
    println!("=================================");

    let mut mutable_matrix = Matrix::<f64>::zeros(3, 3)?;
    mutable_matrix.set(0, 0, 10.0)?;
    mutable_matrix.set(1, 1, 20.0)?;
    mutable_matrix.set(2, 2, 30.0)?;
    mutable_matrix[(0, 2)] = 5.0; // Using index operator
    print_matrix(&mutable_matrix, "Modified Matrix")?;

    println!("\nElement access:");
    println!("  Element at (0,0): {}", mutable_matrix.get(0, 0)?);
    println!("  Element at (1,1): {}", mutable_matrix[(1, 1)]);

    // Test 3: Row and Column Operations
    println!("\n\n3. Row and Column Operations");
    println!("===========================");

    let test_matrix = Matrix::<f64>::from_vec(3, 4, vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0
    ])?;
    print_matrix(&test_matrix, "Test Matrix")?;

    let row_1 = test_matrix.row(1)?;
    println!("\nRow 1: {:?}", row_1);

    let col_2 = test_matrix.col(2)?;
    println!("Column 2: {:?}", col_2);

    // Test 4: Matrix Transpose
    println!("\n\n4. Matrix Transpose");
    println!("==================");

    let original = Matrix::<f64>::from_vec(2, 3, vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0
    ])?;
    print_matrix(&original, "Original Matrix")?;

    let transposed = original.transpose()?;
    print_matrix(&transposed, "Transposed Matrix")?;

    // Test 5: Matrix Arithmetic Operations
    println!("\n\n5. Matrix Arithmetic Operations");
    println!("==============================");

    let matrix_a = Matrix::<f64>::from_vec(2, 2, vec![
        1.0, 2.0,
        3.0, 4.0
    ])?;
    let matrix_b = Matrix::<f64>::from_vec(2, 2, vec![
        5.0, 6.0,
        7.0, 8.0
    ])?;

    print_matrix(&matrix_a, "Matrix A")?;
    print_matrix(&matrix_b, "Matrix B")?;

    // Addition
    let sum = (matrix_a.clone() + matrix_b.clone())?;
    print_matrix(&sum, "A + B")?;

    // Subtraction
    let diff = (matrix_a.clone() - matrix_b.clone())?;
    print_matrix(&diff, "A - B")?;

    // Matrix multiplication
    let product = (matrix_a.clone() * matrix_b.clone())?;
    print_matrix(&product, "A * B (Matrix Multiplication)")?;

    // Element-wise multiplication (dot product)
    let dot_prod = matrix_a.dot_product(&matrix_b)?;
    print_matrix(&dot_prod, "A ⊙ B (Element-wise Multiplication)")?;

    // Scalar operations
    let scalar_mult = (matrix_a.clone() * 2.0)?;
    print_matrix(&scalar_mult, "A * 2 (Scalar Multiplication)")?;

    let scalar_div = (matrix_a.clone() / 2.0)?;
    print_matrix(&scalar_div, "A / 2 (Scalar Division)")?;

    let negated = (-matrix_a.clone())?;
    print_matrix(&negated, "-A (Negation)")?;

    // Test 6: Advanced Matrix Operations
    println!("\n\n6. Advanced Matrix Operations");
    println!("============================");

    let square_matrix = Matrix::<f64>::from_vec(3, 3, vec![
        2.0, 1.0, 3.0,
        1.0, 4.0, 2.0,
        3.0, 2.0, 5.0
    ])?;
    print_matrix(&square_matrix, "Square Matrix")?;

    // Trace
    let trace = square_matrix.trace()?;
    println!("\nTrace: {}", trace);

    // Determinant
    let det = square_matrix.determinant()?;
    println!("Determinant: {}", det);

    // Cofactor matrix
    let cofactor = square_matrix.cofactor_matrix()?;
    print_matrix(&cofactor, "Cofactor Matrix")?;

    // Test 7: Performance Comparison - Sequential vs Concurrent
    println!("\n\n7. Performance Comparison: Sequential vs Concurrent");
    println!("=================================================");

    // Test with multiple matrix sizes to show scaling behavior
    let test_sizes = vec![100, 300, 500, 1000];
    let mut speedup_results = Vec::new();

    for &size in &test_sizes {
        println!("\n--- Testing with {}x{} matrices ---", size, size);

        // Generate test data
        let data_a: Vec<f64> = (0..(size * size)).map(|i| (i as f64) % 100.0).collect();
        let data_b: Vec<f64> = (0..(size * size)).map(|i| ((i * 2) as f64) % 100.0).collect();

        // Sequential matrices
        let matrix_seq_a = Matrix::<f64>::from_vec_sequential(size, size, data_a.clone())?;
        let matrix_seq_b = Matrix::<f64>::from_vec_sequential(size, size, data_b.clone())?;

        // Concurrent matrices
        let matrix_conc_a = Matrix::<f64>::from_vec(size, size, data_a)?;
        let matrix_conc_b = Matrix::<f64>::from_vec(size, size, data_b)?;

        // Matrix Addition
        let (_, _, add_speedup) = compare_operations(
            "Matrix Addition",
            || (matrix_seq_a.clone() + matrix_seq_b.clone()).unwrap(),
            || (matrix_conc_a.clone() + matrix_conc_b.clone()).unwrap(),
        );

        // Matrix Subtraction
        let (_, _, sub_speedup) = compare_operations(
            "Matrix Subtraction",
            || (matrix_seq_a.clone() - matrix_seq_b.clone()).unwrap(),
            || (matrix_conc_a.clone() - matrix_conc_b.clone()).unwrap(),
        );

        // Element-wise Multiplication
        let (_, _, dot_speedup) = compare_operations(
            "Element-wise Multiplication",
            || matrix_seq_a.dot_product(&matrix_seq_b).unwrap(),
            || matrix_conc_a.dot_product(&matrix_conc_b).unwrap(),
        );

        // Scalar Multiplication
        let (_, _, scalar_speedup) = compare_operations(
            "Scalar Multiplication",
            || (matrix_seq_a.clone() * 2.5).unwrap(),
            || (matrix_conc_a.clone() * 2.5).unwrap(),
        );

        // Transpose
        let (_, _, trans_speedup) = compare_operations(
            "Matrix Transpose",
            || matrix_seq_a.transpose().unwrap(),
            || matrix_conc_a.transpose().unwrap(),
        );

        // Column Extraction
        let (_, _, col_speedup) = compare_operations(
            "Column Extraction",
            || matrix_seq_a.col(size / 2).unwrap(),
            || matrix_conc_a.col(size / 2).unwrap(),
        );

        // Trace (for square matrices)
        let (_, _, trace_speedup) = compare_operations(
            "Matrix Trace",
            || matrix_seq_a.trace().unwrap(),
            || matrix_conc_a.trace().unwrap(),
        );

        speedup_results.push((
            size,
            add_speedup,
            sub_speedup,
            dot_speedup,
            scalar_speedup,
            trans_speedup,
            col_speedup,
            trace_speedup,
        ));
    }

    // Matrix Multiplication Test (separate due to computational complexity)
    println!("\n--- Matrix Multiplication Performance ---");
    let mult_sizes = vec![50, 100, 200, 300];
    let mut mult_speedups = Vec::new();

    for &mult_size in &mult_sizes {
        println!("\nMatrix Multiplication ({}x{}):", mult_size, mult_size);
        
        let mult_data_a: Vec<f64> = (0..(mult_size * mult_size)).map(|i| (i as f64) % 50.0).collect();
        let mult_data_b: Vec<f64> = (0..(mult_size * mult_size)).map(|i| ((i * 3) as f64) % 50.0).collect();

        let matrix_mult_seq_a = Matrix::<f64>::from_vec_sequential(mult_size, mult_size, mult_data_a.clone())?;
        let matrix_mult_seq_b = Matrix::<f64>::from_vec_sequential(mult_size, mult_size, mult_data_b.clone())?;
        let matrix_mult_conc_a = Matrix::<f64>::from_vec(mult_size, mult_size, mult_data_a)?;
        let matrix_mult_conc_b = Matrix::<f64>::from_vec(mult_size, mult_size, mult_data_b)?;

        let (_, _, mult_speedup) = compare_operations(
            "Matrix Multiplication",
            || matrix_mult_seq_a.matrix_multiply(&matrix_mult_seq_b).unwrap(),
            || matrix_mult_conc_a.matrix_multiply(&matrix_mult_conc_b).unwrap(),
        );

        mult_speedups.push((mult_size, mult_speedup));
    }

    // Performance Summary
    println!("\n=== PERFORMANCE SUMMARY ===");
    println!("\nSpeedup by Matrix Size (Concurrent vs Sequential):");
    println!("{:<6} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}", 
             "Size", "Add", "Sub", "Dot", "Scalar", "Trans", "Col", "Trace");
    println!("{}", "=".repeat(70));
    
    for (size, add, sub, dot, scalar, trans, col, trace) in &speedup_results {
        println!("{:<6} {:<8.2} {:<8.2} {:<8.2} {:<8.2} {:<8.2} {:<8.2} {:<8.2}", 
                 size, add, sub, dot, scalar, trans, col, trace);
    }

    println!("\nMatrix Multiplication Speedup:");
    println!("{:<6} {:<10}", "Size", "Speedup");
    println!("{}", "=".repeat(20));
    for (size, speedup) in &mult_speedups {
        println!("{:<6} {:<10.2}", size, speedup);
    }

    // Analysis
    println!("\n=== PERFORMANCE ANALYSIS ===");
    let avg_speedups: Vec<f64> = speedup_results.iter()
        .map(|(_, add, sub, dot, scalar, trans, col, trace)| {
            (add + sub + dot + scalar + trans + col + trace) / 7.0
        })
        .collect();

    let overall_avg = avg_speedups.iter().sum::<f64>() / avg_speedups.len() as f64;
    println!("Overall average speedup: {:.2}x", overall_avg);
    
    let max_speedup = speedup_results.iter()
        .flat_map(|(_, add, sub, dot, scalar, trans, col, trace)| {
            vec![*add, *sub, *dot, *scalar, *trans, *col, *trace]
        })
        .fold(0.0f64, |a, b| a.max(b));
    println!("Maximum observed speedup: {:.2}x", max_speedup);

    let mult_avg = mult_speedups.iter().map(|(_, s)| s).sum::<f64>() / mult_speedups.len() as f64;
    println!("Matrix multiplication average speedup: {:.2}x", mult_avg);

    // Test 8: Detailed Timing Analysis
    println!("\n\n8. Detailed Timing Analysis");
    println!("===========================");

    let analysis_size = 800;
    println!("\nDetailed timing breakdown for {}x{} matrices:", analysis_size, analysis_size);

    let data_large: Vec<f64> = (0..(analysis_size * analysis_size))
        .map(|i| ((i as f64) * 1.5) % 1000.0)
        .collect();

    let matrix_seq_large = Matrix::<f64>::from_vec_sequential(analysis_size, analysis_size, data_large.clone())?;
    let matrix_conc_large = Matrix::<f64>::from_vec(analysis_size, analysis_size, data_large)?;

    // Multiple runs for statistical significance
    let num_runs = 5;
    println!("\nRunning {} iterations for each operation:", num_runs);

    // Transpose timing
    let mut seq_times = Vec::new();
    let mut conc_times = Vec::new();

    println!("\n--- Transpose Operation Timing ---");
    for i in 1..=num_runs {
        print!("Run {}: ", i);
        
        let start = Instant::now();
        let _seq_result = matrix_seq_large.transpose().unwrap();
        let seq_duration = start.elapsed();
        seq_times.push(seq_duration);
        
        let start = Instant::now();
        let _conc_result = matrix_conc_large.transpose().unwrap();
        let conc_duration = start.elapsed();
        conc_times.push(conc_duration);
        
        println!("Seq: {}, Conc: {}", format_duration(seq_duration), format_duration(conc_duration));
    }

    let seq_avg = seq_times.iter().sum::<std::time::Duration>() / num_runs as u32;
    let conc_avg = conc_times.iter().sum::<std::time::Duration>() / num_runs as u32;
    let avg_speedup = seq_avg.as_nanos() as f64 / conc_avg.as_nanos() as f64;

    println!("\nTranspose Statistics:");
    println!("  Sequential average: {}", format_duration(seq_avg));
    println!("  Concurrent average: {}", format_duration(conc_avg));
    println!("  Average speedup: {:.2}x", avg_speedup);
    println!("  Best sequential: {}", format_duration(*seq_times.iter().min().unwrap()));
    println!("  Best concurrent: {}", format_duration(*conc_times.iter().min().unwrap()));

    // Element-wise operations timing
    println!("\n--- Element-wise Addition Timing ---");
    let matrix_seq_b = Matrix::<f64>::ones(analysis_size, analysis_size)?;
    let matrix_conc_b = {
        let mut m = Matrix::<f64>::ones(analysis_size, analysis_size)?;
        m.set_concurrent(true);
        m
    };

    seq_times.clear();
    conc_times.clear();

    for i in 1..=num_runs {
        print!("Run {}: ", i);
        
        let start = Instant::now();
        let _seq_result = (matrix_seq_large.clone() + matrix_seq_b.clone()).unwrap();
        let seq_duration = start.elapsed();
        seq_times.push(seq_duration);
        
        let start = Instant::now();
        let _conc_result = (matrix_conc_large.clone() + matrix_conc_b.clone()).unwrap();
        let conc_duration = start.elapsed();
        conc_times.push(conc_duration);
        
        println!("Seq: {}, Conc: {}", format_duration(seq_duration), format_duration(conc_duration));
    }

    let seq_avg = seq_times.iter().sum::<std::time::Duration>() / num_runs as u32;
    let conc_avg = conc_times.iter().sum::<std::time::Duration>() / num_runs as u32;
    let avg_speedup = seq_avg.as_nanos() as f64 / conc_avg.as_nanos() as f64;

    println!("\nAddition Statistics:");
    println!("  Sequential average: {}", format_duration(seq_avg));
    println!("  Concurrent average: {}", format_duration(conc_avg));
    println!("  Average speedup: {:.2}x", avg_speedup);

    // Memory allocation timing
    println!("\n--- Matrix Creation Timing ---");
    let creation_size = 1000;
    let large_data: Vec<f64> = (0..(creation_size * creation_size))
        .map(|i| (i as f64) % 100.0)
        .collect();

    println!("Creating {}x{} matrices:", creation_size, creation_size);

    let start = Instant::now();
    let _seq_matrix = Matrix::<f64>::from_vec_sequential(creation_size, creation_size, large_data.clone()).unwrap();
    let seq_creation_time = start.elapsed();

    let start = Instant::now();
    let _conc_matrix = Matrix::<f64>::from_vec(creation_size, creation_size, large_data).unwrap();
    let conc_creation_time = start.elapsed();

    println!("  Sequential creation: {}", format_duration(seq_creation_time));
    println!("  Concurrent creation: {}", format_duration(conc_creation_time));

    // Test 9: Concurrent Mode Toggle
    println!("\n\n9. Concurrent Mode Toggle");
    println!("========================");

    let mut toggle_matrix = Matrix::<f64>::ones(1000, 1000)?;
    println!("Initial concurrent mode: {}", toggle_matrix.is_concurrent());

    // Test performance difference by toggling
    println!("\nTesting same matrix with different modes:");
    
    toggle_matrix.set_concurrent(false);
    let start = Instant::now();
    let _result1 = toggle_matrix.transpose().unwrap();
    let sequential_time = start.elapsed();
    println!("  Sequential mode: {}", format_duration(sequential_time));

    toggle_matrix.set_concurrent(true);
    let start = Instant::now();
    let _result2 = toggle_matrix.transpose().unwrap();
    let concurrent_time = start.elapsed();
    println!("  Concurrent mode: {}", format_duration(concurrent_time));

    let toggle_speedup = sequential_time.as_nanos() as f64 / concurrent_time.as_nanos() as f64;
    println!("  Speedup from toggling: {:.2}x", toggle_speedup);

    // Test 10: Error Handling
    println!("\n\n10. Error Handling Examples");
    println!("==========================");

    // Invalid dimensions
    match Matrix::<f64>::new(0, 5) {
        Ok(_) => println!("Unexpected success"),
        Err(e) => println!("Expected error for invalid dimensions: {}", e),
    }

    // Index out of bounds
    let small_matrix = Matrix::<f64>::ones(2, 2)?;
    match small_matrix.get(5, 5) {
        Ok(_) => println!("Unexpected success"),
        Err(e) => println!("Expected error for out of bounds: {}", e),
    }

    // Incompatible dimensions for operations
    let matrix_2x2 = Matrix::<f64>::ones(2, 2)?;
    let matrix_3x3 = Matrix::<f64>::ones(3, 3)?;
    match matrix_2x2.clone() + matrix_3x3.clone() {
        Ok(_) => println!("Unexpected success"),
        Err(e) => println!("Expected error for incompatible dimensions: {}", e),
    }

    // Division by zero
    match matrix_2x2 / 0.0 {
        Ok(_) => println!("Unexpected success"),
        Err(e) => println!("Expected error for division by zero: {}", e),
    }

    // Test 11: Complex Operations Chain
    println!("\n\n11. Complex Operations Chain");
    println!("============================");

    let chain_matrix = Matrix::<f64>::from_vec(3, 3, vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0
    ])?;

    print_matrix(&chain_matrix, "Original Matrix")?;

    // Chain operations: ((A * 2) + A^T) - I
    println!("\nTiming complex operation chain: ((A * 2) + A^T) - I");
    
    let start = Instant::now();
    let doubled = (chain_matrix.clone() * 2.0)?;
    let transposed = chain_matrix.transpose()?;
    let added = (doubled + transposed)?;
    let identity = Matrix::<f64>::identity(3)?;
    let final_result = (added - identity)?;
    let chain_time = start.elapsed();

    print_matrix(&final_result, "Result of ((A * 2) + A^T) - I")?;
    println!("\nChain operation completed in: {}", format_duration(chain_time));

    // Test 12: Scaling Analysis
    println!("\n\n12. Scaling Analysis");
    println!("===================");

    println!("\nHow performance scales with matrix size:");
    let scaling_sizes = vec![100, 200, 400, 600, 800];
    
    println!("{:<6} {:<12} {:<12} {:<10}", "Size", "Sequential", "Concurrent", "Speedup");
    println!("{}", "=".repeat(45));

    for &size in &scaling_sizes {
        let data: Vec<f64> = (0..(size * size)).map(|i| (i as f64) % 100.0).collect();
        
        let matrix_seq = Matrix::<f64>::from_vec_sequential(size, size, data.clone())?;
        let matrix_conc = Matrix::<f64>::from_vec(size, size, data)?;

        // Time transpose operation (good for showing parallelization benefits)
        let start = Instant::now();
        let _seq_result = matrix_seq.transpose().unwrap();
        let seq_time = start.elapsed();

        let start = Instant::now();
        let _conc_result = matrix_conc.transpose().unwrap();
        let conc_time = start.elapsed();

        let speedup = seq_time.as_nanos() as f64 / conc_time.as_nanos() as f64;
        
        println!("{:<6} {:<12} {:<12} {:<10.2}", 
                 size, 
                 format_duration(seq_time), 
                 format_duration(conc_time), 
                 speedup);
    }

    println!("\n=== Demo Complete ===");
    println!("\nAll matrix operations have been successfully demonstrated!");
    println!("The library supports both sequential and concurrent execution modes,");
    println!("with automatic parallelization for larger matrices when concurrent mode is enabled.");

    Ok(())
}