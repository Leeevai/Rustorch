mod matrix;
mod error;

pub use matrix::*;
pub use error::{MatrixError, MatrixResult};
pub use std::time::Instant;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_creation() {
        let mat = Matrix::<i32>::new(2, 3).unwrap();
        assert_eq!(mat.dimensions(), (2, 3));
        assert!(mat.is_concurrent());

        let mat_seq = Matrix::<i32>::new_sequential(2, 3).unwrap();
        assert!(!mat_seq.is_concurrent());
    }

    #[test]
    fn test_invalid_dimensions() {
        assert!(Matrix::<i32>::new(0, 3).is_err());
        assert!(Matrix::<i32>::new(3, 0).is_err());
        assert!(Matrix::<i32>::new(0, 0).is_err());
    }

    #[test]
    fn test_matrix_indexing() {
        let mut mat = Matrix::<i32>::new(2, 3).unwrap();

        mat[(0, 0)] = 1;
        mat[(0, 1)] = 2;
        mat[(1, 2)] = 3;

        assert_eq!(mat[(0, 0)], 1);
        assert_eq!(mat[(0, 1)], 2);
        assert_eq!(mat[(1, 2)], 3);
        assert_eq!(mat[(1, 0)], 0);
    }

    #[test]  
    #[should_panic(expected = "Index (2, 0) out of bounds")]
    fn test_out_of_bounds_index() {
        let mat = Matrix::<i32>::new(2, 2).unwrap();
        let _ = mat[(2, 0)];
    }

    #[test]
    #[should_panic(expected = "Index (0, 2) out of bounds")]
    fn test_out_of_bounds_index_col() {
        let mat = Matrix::<i32>::new(2, 2).unwrap();
        let _ = mat[(0, 2)];
    }

    #[test]
    fn test_safe_indexing() {
        let mut mat = Matrix::<i32>::new(2, 2).unwrap();
        
        // Test safe get
        assert!(mat.get(0, 0).is_ok());
        assert!(mat.get(2, 0).is_err());
        assert!(mat.get(0, 2).is_err());

        // Test safe set
        assert!(mat.set(0, 0, 42).is_ok());
        assert_eq!(mat[(0, 0)], 42);
        assert!(mat.set(2, 0, 42).is_err());
        assert!(mat.set(0, 2, 42).is_err());
    }

    #[test]
    fn test_from_vec() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let mat = Matrix::from_vec(2, 3, data).unwrap();
        
        assert_eq!(mat[(0, 0)], 1);
        assert_eq!(mat[(0, 1)], 2);
        assert_eq!(mat[(0, 2)], 3);
        assert_eq!(mat[(1, 0)], 4);
        assert_eq!(mat[(1, 1)], 5);
        assert_eq!(mat[(1, 2)], 6);
    }

    #[test]
    fn test_from_vec_invalid_size() {
        let data = vec![1, 2, 3, 4, 5];
        assert!(Matrix::from_vec(2, 3, data).is_err());
    }

    #[test]
    fn test_identity_matrix() {
        let mat = Matrix::<i32>::identity(3).unwrap();
        
        assert_eq!(mat[(0, 0)], 1);
        assert_eq!(mat[(1, 1)], 1);
        assert_eq!(mat[(2, 2)], 1);
        assert_eq!(mat[(0, 1)], 0);
        assert_eq!(mat[(1, 0)], 0);
    }

    #[test]
    fn test_zeros_and_ones() {
        let zeros = Matrix::<i32>::zeros(2, 2).unwrap();
        assert_eq!(zeros[(0, 0)], 0);
        assert_eq!(zeros[(1, 1)], 0);

        let ones = Matrix::<i32>::ones(2, 2).unwrap();
        assert_eq!(ones[(0, 0)], 1);
        assert_eq!(ones[(1, 1)], 1);
    }

    #[test]
    fn test_row_and_col_access() {
        let mut mat = Matrix::<i32>::new(3, 3).unwrap();
        
        // Fill with row*10 + col values
        for r in 0..3 {
            for c in 0..3 {
                mat[(r, c)] = (r * 10 + c) as i32;
            }
        }

        // Test row access
        assert_eq!(mat.row(0).unwrap(), vec![0, 1, 2]);
        assert_eq!(mat.row(1).unwrap(), vec![10, 11, 12]);
        assert_eq!(mat.row(2).unwrap(), vec![20, 21, 22]);
        assert!(mat.row(3).is_err());

        // Test column access
        assert_eq!(mat.col(0).unwrap(), vec![0, 10, 20]);
        assert_eq!(mat.col(1).unwrap(), vec![1, 11, 21]);
        assert_eq!(mat.col(2).unwrap(), vec![2, 12, 22]);
        assert!(mat.col(3).is_err());
    }

    #[test]
    fn test_all_rows_9x9_explicit() {
        let rows = 9;
        let cols = 9;
        let mut mat = Matrix::<i32>::new(rows, cols).unwrap();

        // Fill with row*10 + col values
        for r in 0..rows {
            for c in 0..cols {
                mat[(r, c)] = (r * 10 + c) as i32;
            }
        }

        assert_eq!(mat.row(0).unwrap(), vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(mat.row(1).unwrap(), vec![10, 11, 12, 13, 14, 15, 16, 17, 18]);
        assert_eq!(mat.row(2).unwrap(), vec![20, 21, 22, 23, 24, 25, 26, 27, 28]);
        assert_eq!(mat.row(3).unwrap(), vec![30, 31, 32, 33, 34, 35, 36, 37, 38]);
        assert_eq!(mat.row(4).unwrap(), vec![40, 41, 42, 43, 44, 45, 46, 47, 48]);
        assert_eq!(mat.row(5).unwrap(), vec![50, 51, 52, 53, 54, 55, 56, 57, 58]);
        assert_eq!(mat.row(6).unwrap(), vec![60, 61, 62, 63, 64, 65, 66, 67, 68]);
        assert_eq!(mat.row(7).unwrap(), vec![70, 71, 72, 73, 74, 75, 76, 77, 78]);
        assert_eq!(mat.row(8).unwrap(), vec![80, 81, 82, 83, 84, 85, 86, 87, 88]);
    }

    #[test]
    fn test_all_cols_9x9_explicit() {
        let rows = 9;
        let cols = 9;
        let mut mat = Matrix::<i32>::new(rows, cols).unwrap();

        // Fill with row*10 + col values
        for r in 0..rows {
            for c in 0..cols {
                mat[(r, c)] = (r * 10 + c) as i32;
            }
        }

        assert_eq!(mat.col(0).unwrap(), vec![0, 10, 20, 30, 40, 50, 60, 70, 80]);
        assert_eq!(mat.col(1).unwrap(), vec![1, 11, 21, 31, 41, 51, 61, 71, 81]);
        assert_eq!(mat.col(2).unwrap(), vec![2, 12, 22, 32, 42, 52, 62, 72, 82]);
        assert_eq!(mat.col(3).unwrap(), vec![3, 13, 23, 33, 43, 53, 63, 73, 83]);
        assert_eq!(mat.col(4).unwrap(), vec![4, 14, 24, 34, 44, 54, 64, 74, 84]);
        assert_eq!(mat.col(5).unwrap(), vec![5, 15, 25, 35, 45, 55, 65, 75, 85]);
        assert_eq!(mat.col(6).unwrap(), vec![6, 16, 26, 36, 46, 56, 66, 76, 86]);
        assert_eq!(mat.col(7).unwrap(), vec![7, 17, 27, 37, 47, 57, 67, 77, 87]);
        assert_eq!(mat.col(8).unwrap(), vec![8, 18, 28, 38, 48, 58, 68, 78, 88]);
    }

    #[test]
    fn test_matrix_addition() {
        let mat1 = Matrix::from_vec(2, 2, vec![1, 2, 3, 4]).unwrap();
        let mat2 = Matrix::from_vec(2, 2, vec![5, 6, 7, 8]).unwrap();
        
        let result = (mat1 + mat2).unwrap();
        assert_eq!(result[(0, 0)], 6);
        assert_eq!(result[(0, 1)], 8);
        assert_eq!(result[(1, 0)], 10);
        assert_eq!(result[(1, 1)], 12);
    }

    #[test]
    fn test_matrix_addition_dimension_mismatch() {
        let mat1 = Matrix::from_vec(2, 2, vec![1, 2, 3, 4]).unwrap();
        let mat2 = Matrix::from_vec(2, 3, vec![1, 2, 3, 4, 5, 6]).unwrap();
        
        let result = mat1 + mat2;
        assert!(result.is_err());
        if let Err(MatrixError::IncompatibleDimensions { op, dim1, dim2 }) = result {
            assert_eq!(op, "addition");
            assert_eq!(dim1, (2, 2));
            assert_eq!(dim2, (2, 3));
        }
    }

    #[test]
    fn test_matrix_subtraction() {
        let mat1 = Matrix::from_vec(2, 2, vec![5, 6, 7, 8]).unwrap();
        let mat2 = Matrix::from_vec(2, 2, vec![1, 2, 3, 4]).unwrap();
        
        let result = (mat1 - mat2).unwrap();
        assert_eq!(result[(0, 0)], 4);
        assert_eq!(result[(0, 1)], 4);
        assert_eq!(result[(1, 0)], 4);
        assert_eq!(result[(1, 1)], 4);
    }

    #[test]
    fn test_matrix_multiplication() {
        let mat1 = Matrix::from_vec(2, 3, vec![1, 2, 3, 4, 5, 6]).unwrap();
        let mat2 = Matrix::from_vec(3, 2, vec![7, 8, 9, 10, 11, 12]).unwrap();
        
        let result = (mat1 * mat2).unwrap();
        assert_eq!(result.dimensions(), (2, 2));
        assert_eq!(result[(0, 0)], 58);  // 1*7 + 2*9 + 3*11
        assert_eq!(result[(0, 1)], 64);  // 1*8 + 2*10 + 3*12
        assert_eq!(result[(1, 0)], 139); // 4*7 + 5*9 + 6*11
        assert_eq!(result[(1, 1)], 154); // 4*8 + 5*10 + 6*12
    }

    #[test]
    fn test_matrix_multiplication_dimension_mismatch() {
        let mat1 = Matrix::from_vec(2, 3, vec![1, 2, 3, 4, 5, 6]).unwrap();
        let mat2 = Matrix::from_vec(2, 2, vec![1, 2, 3, 4]).unwrap();
        
        let result = mat1 * mat2;
        assert!(result.is_err());
    }

    #[test]
    fn test_scalar_multiplication() {
        let mat = Matrix::from_vec(2, 2, vec![1, 2, 3, 4]).unwrap();
        let result = (mat * 3).unwrap();
        
        assert_eq!(result[(0, 0)], 3);
        assert_eq!(result[(0, 1)], 6);
        assert_eq!(result[(1, 0)], 9);
        assert_eq!(result[(1, 1)], 12);
    }

    #[test]
    fn test_scalar_division() {
        let mat = Matrix::from_vec(2, 2, vec![6, 8, 10, 12]).unwrap();
        let result = (mat / 2).unwrap();
        
        assert_eq!(result[(0, 0)], 3);
        assert_eq!(result[(0, 1)], 4);
        assert_eq!(result[(1, 0)], 5);
        assert_eq!(result[(1, 1)], 6);
    }

    #[test]
    fn test_scalar_division_by_zero() {
        let mat = Matrix::from_vec(2, 2, vec![1, 2, 3, 4]).unwrap();
        let result = mat / 0;
        assert!(matches!(result, Err(MatrixError::DivisionByZero)));
    }

    #[test]
    fn test_matrix_negation() {
        let mat = Matrix::from_vec(2, 2, vec![1, -2, 3, -4]).unwrap();
        let result = (-mat).unwrap();
        
        assert_eq!(result[(0, 0)], -1);
        assert_eq!(result[(0, 1)], 2);
        assert_eq!(result[(1, 0)], -3);
        assert_eq!(result[(1, 1)], 4);
    }

    #[test]
    fn test_dot_product() {
        let mat1 = Matrix::from_vec(2, 2, vec![1, 2, 3, 4]).unwrap();
        let mat2 = Matrix::from_vec(2, 2, vec![5, 6, 7, 8]).unwrap();
        
        let result = mat1.dot_product(&mat2).unwrap();
        assert_eq!(result[(0, 0)], 5);  // 1*5
        assert_eq!(result[(0, 1)], 12); // 2*6
        assert_eq!(result[(1, 0)], 21); // 3*7
        assert_eq!(result[(1, 1)], 32); // 4*8
    }

    #[test]
    fn test_transpose() {
        let mat = Matrix::from_vec(2, 3, vec![1, 2, 3, 4, 5, 6]).unwrap();
        let transposed = mat.transpose().unwrap();
        
        assert_eq!(transposed.dimensions(), (3, 2));
        assert_eq!(transposed[(0, 0)], 1);
        assert_eq!(transposed[(0, 1)], 4);
        assert_eq!(transposed[(1, 0)], 2);
        assert_eq!(transposed[(1, 1)], 5);
        assert_eq!(transposed[(2, 0)], 3);
        assert_eq!(transposed[(2, 1)], 6);
    }

    #[test]
    fn test_trace() {
        let mat = Matrix::from_vec(3, 3, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
        let trace = mat.trace().unwrap();
        assert_eq!(trace, 15); // 1 + 5 + 9
    }

    #[test]
    fn test_trace_non_square() {
        let mat = Matrix::from_vec(2, 3, vec![1, 2, 3, 4, 5, 6]).unwrap();
        let result = mat.trace();
        assert!(matches!(result, Err(MatrixError::NotSquareMatrix { .. })));
    }

    #[test]
    fn test_determinant_2x2() {
        let mat = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let det = mat.determinant().unwrap();
        assert_eq!(det, -2.0); // 1*4 - 2*3 = -2
    }

    #[test]
    fn test_determinant_3x3() {
        let mat = Matrix::from_vec(3, 3, vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0
        ]).unwrap();
        let det = mat.determinant().unwrap();
        // Use approximate comparison for floating point values
        // This matrix is singular (determinant should be 0), but due to floating point precision
        // we get a very small number instead of exactly 0
        assert!((det).abs() < 1e-10); // Check if determinant is very close to 0
    }

    #[test]
    fn test_determinant_non_square() {
        let mat = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let result = mat.determinant();
        assert!(matches!(result, Err(MatrixError::NotSquareMatrix { .. })));
    }

    #[test]
    fn test_cofactor_matrix() {
        let mat = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let cofactor = mat.cofactor_matrix().unwrap();
        
        assert_eq!(cofactor[(0, 0)], 4.0);
        assert_eq!(cofactor[(0, 1)], -3.0);
        assert_eq!(cofactor[(1, 0)], -2.0);
        assert_eq!(cofactor[(1, 1)], 1.0);
    }

    #[test]
    fn test_concurrent_vs_sequential() {
        let data = (0..100).collect::<Vec<i32>>();
        let mut mat_concurrent = Matrix::from_vec(10, 10, data.clone()).unwrap();
        let mut mat_sequential = Matrix::from_vec_sequential(10, 10, data).unwrap();
        
        assert!(mat_concurrent.is_concurrent());
        assert!(!mat_sequential.is_concurrent());
        
        // Test that operations work with both modes
        let scalar = 2;
        let result_concurrent = (mat_concurrent.clone() * scalar).unwrap();
        let result_sequential = (mat_sequential.clone() * scalar).unwrap();
        
        // Results should be the same
        for i in 0..10 {
            for j in 0..10 {
                assert_eq!(result_concurrent[(i, j)], result_sequential[(i, j)]);
            }
        }
        
        // Test changing concurrency mode
        mat_concurrent.set_concurrent(false);
        mat_sequential.set_concurrent(true);
        assert!(!mat_concurrent.is_concurrent());
        assert!(mat_sequential.is_concurrent());
    }

    #[test]
    fn test_matrix_properties() {
        let square_mat = Matrix::<i32>::new(3, 3).unwrap();
        let rect_mat = Matrix::<i32>::new(2, 3).unwrap();
        
        assert!(square_mat.is_square());
        assert!(!rect_mat.is_square());
        assert!(!square_mat.is_empty());
        assert!(!rect_mat.is_empty());
        
        assert_eq!(square_mat.rows(), 3);
        assert_eq!(square_mat.cols(), 3);
        assert_eq!(rect_mat.rows(), 2);
        assert_eq!(rect_mat.cols(), 3);
    }

    #[test]
    fn test_error_display() {
        let error = MatrixError::IndexOutOfBounds { 
            row: 5, col: 3, max_row: 3, max_col: 3 
        };
        let error_str = format!("{}", error);
        assert!(error_str.contains("Index (5, 3) out of bounds"));
        
        let error2 = MatrixError::IncompatibleDimensions {
            op: "test".to_string(),
            dim1: (2, 3),
            dim2: (3, 2),
        };
        let error2_str = format!("{}", error2);
        assert!(error2_str.contains("Incompatible dimensions for test"));
    }

     #[test]
    fn test_complex_operations() {
        // Test chained operations
        let mat1 = Matrix::from_vec(2, 2, vec![1, 2, 3, 4]).unwrap();
        let mat2 = Matrix::from_vec(2, 2, vec![5, 6, 7, 8]).unwrap();
        let mat3 = Matrix::from_vec(2, 2, vec![1, 1, 1, 1]).unwrap();
        
        // (mat1 + mat2) * mat3
        // mat1 + mat2 = [[6, 8], [10, 12]]
        // [[6, 8], [10, 12]] * [[1, 1], [1, 1]]
        // Result: [[6*1 + 8*1, 6*1 + 8*1], [10*1 + 12*1, 10*1 + 12*1]]
        //       = [[14, 14], [22, 22]]
        let sum = (mat1 + mat2).unwrap();
        let result = (sum * mat3).unwrap();
        
        assert_eq!(result[(0, 0)], 14); // 6*1 + 8*1 = 14
        assert_eq!(result[(0, 1)], 14); // 6*1 + 8*1 = 14
        assert_eq!(result[(1, 0)], 22); // 10*1 + 12*1 = 22
        assert_eq!(result[(1, 1)], 22); // 10*1 + 12*1 = 22
    }

    #[test]
    fn test_performance_comparison() {
        use std::time::Instant;
        
        let size:usize = 50;
        let data: Vec<usize> = (0..(size * size)).collect();
        
        let mat_concurrent = Matrix::from_vec(size, size, data.clone()).unwrap();
        let mat_sequential = Matrix::from_vec_sequential(size, size, data).unwrap();
        
        // Test transpose performance
        let start = Instant::now();
        let _trans_concurrent = mat_concurrent.transpose().unwrap();
        let concurrent_time = start.elapsed();
        
        let start = Instant::now();
        let _trans_sequential = mat_sequential.transpose().unwrap();
        let sequential_time = start.elapsed();
        
        println!("Concurrent transpose: {:?}", concurrent_time);
        println!("Sequential transpose: {:?}", sequential_time);
        
        // Both should produce the same result
        let trans_concurrent = mat_concurrent.transpose().unwrap();
        let trans_sequential = mat_sequential.transpose().unwrap();
        
        for i in 0..size {
            for j in 0..size {
                assert_eq!(trans_concurrent[(i, j)], trans_sequential[(i, j)]);
            }
        }
    }

    #[test]
    fn test_edge_cases() {
        // Test 1x1 matrix
        let mat = Matrix::from_vec(1, 1, vec![42]).unwrap();
        assert_eq!(mat.determinant().unwrap(), 42);
        assert_eq!(mat.trace().unwrap(), 42);
        
        // Test identity operations
        let identity = Matrix::<i32>::identity(3).unwrap();
        let mat = Matrix::from_vec(3, 3, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
        let result = (mat.clone() * identity).unwrap();
        
        // Matrix * Identity should equal original matrix
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(result[(i, j)], mat[(i, j)]);
            }
        }
    }

    #[test]
    fn test_all_error_types() {
        // Test various error conditions
        assert!(matches!(
            Matrix::<i32>::new(0, 5),
            Err(MatrixError::InvalidDimensions)
        ));
        
        let mat = Matrix::<i32>::new(2, 2).unwrap();
        assert!(matches!(
            mat.row(5),
            Err(MatrixError::InvalidRowDimension)
        ));
        
        assert!(matches!(
            mat.col(5),
            Err(MatrixError::InvalidColumnDimension)
        ));
        
        assert!(matches!(
            mat.get(5, 0),
            Err(MatrixError::IndexOutOfBounds { .. })
        ));
        
        let non_square = Matrix::<f64>::new(2, 3).unwrap();
        assert!(matches!(
            non_square.determinant(),
            Err(MatrixError::NotSquareMatrix { .. })
        ));
    }
}