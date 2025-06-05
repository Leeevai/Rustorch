use std::ops::{Index, IndexMut, Add, Sub, Mul, Div, Neg};
use std::sync::Arc;
use std::thread;
use rayon::prelude::*;
use crate::error::{MatrixError, MatrixResult};

#[derive(Debug, Clone)]
pub struct Matrix<T> {
    rows: usize,
    cols: usize,
    mat: Vec<T>,
    concurrent: bool,
}

impl<T> Matrix<T>
where
    T: Default + Copy + Clone + Send + Sync,
{
    pub fn new(rows: usize, cols: usize) -> MatrixResult<Matrix<T>> {
        if rows == 0 || cols == 0 {
            return Err(MatrixError::InvalidDimensions);
        }
        Ok(Self {
            rows,
            cols,
            mat: vec![T::default(); rows * cols],
            concurrent: true,
        })
    }

    pub fn new_sequential(rows: usize, cols: usize) -> MatrixResult<Matrix<T>> {
        if rows == 0 || cols == 0 {
            return Err(MatrixError::InvalidDimensions);
        }
        Ok(Self {
            rows,
            cols,
            mat: vec![T::default(); rows * cols],
            concurrent: false,
        })
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<T>) -> MatrixResult<Matrix<T>> {
        if rows == 0 || cols == 0 {
            return Err(MatrixError::InvalidDimensions);
        }
        if data.len() != rows * cols {
            return Err(MatrixError::DimensionMismatch {
                expected: (rows, cols),
                actual: (data.len() / cols, data.len() % cols),
            });
        }
        Ok(Self {
            rows,
            cols,
            mat: data,
            concurrent: true,
        })
    }

    pub fn from_vec_sequential(rows: usize, cols: usize, data: Vec<T>) -> MatrixResult<Matrix<T>> {
        if rows == 0 || cols == 0 {
            return Err(MatrixError::InvalidDimensions);
        }
        if data.len() != rows * cols {
            return Err(MatrixError::DimensionMismatch {
                expected: (rows, cols),
                actual: (data.len() / cols, data.len() % cols),
            });
        }
        Ok(Self {
            rows,
            cols,
            mat: data,
            concurrent: false,
        })
    }

    pub fn identity(size: usize) -> MatrixResult<Matrix<T>>
    where
        T: From<i32>,
    {
        let mut mat = Self::new(size, size)?;
        for i in 0..size {
            mat[(i, i)] = T::from(1);
        }
        Ok(mat)
    }

    pub fn zeros(rows: usize, cols: usize) -> MatrixResult<Matrix<T>> {
        Self::new(rows, cols)
    }

    pub fn ones(rows: usize, cols: usize) -> MatrixResult<Matrix<T>>
    where
        T: From<i32>,
    {
        if rows == 0 || cols == 0 {
            return Err(MatrixError::InvalidDimensions);
        }
        Ok(Self {
            rows,
            cols,
            mat: vec![T::from(1); rows * cols],
            concurrent: true,
        })
    }

    pub fn set_concurrent(&mut self, concurrent: bool) {
        self.concurrent = concurrent;
    }

    pub fn is_concurrent(&self) -> bool {
        self.concurrent
    }

    pub fn dimensions(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    pub fn is_empty(&self) -> bool {
        self.rows == 0 || self.cols == 0
    }

    fn check_bounds(&self, row: usize, col: usize) -> MatrixResult<()> {
        if row >= self.rows || col >= self.cols {
            Err(MatrixError::IndexOutOfBounds {
                row,
                col,
                max_row: self.rows,
                max_col: self.cols,
            })
        } else {
            Ok(())
        }
    }

    pub fn get(&self, row: usize, col: usize) -> MatrixResult<&T> {
        self.check_bounds(row, col)?;
        Ok(&self.mat[row * self.cols + col])
    }

    pub fn get_mut(&mut self, row: usize, col: usize) -> MatrixResult<&mut T> {
        self.check_bounds(row, col)?;
        Ok(&mut self.mat[row * self.cols + col])
    }

    pub fn set(&mut self, row: usize, col: usize, value: T) -> MatrixResult<()> {
        self.check_bounds(row, col)?;
        self.mat[row * self.cols + col] = value;
        Ok(())
    }

    pub fn row(&self, row: usize) -> MatrixResult<Vec<T>> {
        if row >= self.rows {
            return Err(MatrixError::InvalidRowDimension);
        }
        let start = row * self.cols;
        Ok(self.mat[start..(start + self.cols)].to_vec())
    }

    pub fn col(&self, col: usize) -> MatrixResult<Vec<T>> {
        if col >= self.cols {
            return Err(MatrixError::InvalidColumnDimension);
        }
        
        if self.concurrent {
            Ok((0..self.rows)
                .into_par_iter()
                .map(|r| self.mat[r * self.cols + col])
                .collect())
        } else {
            Ok((0..self.rows)
                .map(|r| self.mat[r * self.cols + col])
                .collect())
        }
    }

    pub fn transpose(&self) -> MatrixResult<Matrix<T>> {
        let mut result = Matrix::new(self.cols, self.rows)?;
        result.set_concurrent(self.concurrent);

        if self.concurrent {
            result.mat.par_chunks_mut(self.rows)
                .enumerate()
                .for_each(|(new_row, chunk)| {
                    for (new_col, value) in chunk.iter_mut().enumerate() {
                        *value = self.mat[new_col * self.cols + new_row];
                    }
                });
        } else {
            for i in 0..self.rows {
                for j in 0..self.cols {
                    result.mat[j * self.rows + i] = self.mat[i * self.cols + j];
                }
            }
        }

        Ok(result)
    }

    pub fn trace(&self) -> MatrixResult<T>
    where
        T: std::ops::Add<Output = T>,
    {
        if !self.is_square() {
            return Err(MatrixError::NotSquareMatrix {
                rows: self.rows,
                cols: self.cols,
            });
        }

        if self.concurrent {
            Ok((0..self.rows)
                .into_par_iter()
                .map(|i| self.mat[i * self.cols + i])
                .reduce(|| T::default(), |a, b| a + b))
        } else {
            let mut sum = T::default();
            for i in 0..self.rows {
                sum = sum + self.mat[i * self.cols + i];
            }
            Ok(sum)
        }
    }
}

// Arithmetic operations for numeric types
impl<T> Matrix<T>
where
    T: Default + Copy + Clone + Send + Sync + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Mul<Output = T> + PartialEq,
{
    pub fn dot_product(&self, other: &Matrix<T>) -> MatrixResult<Matrix<T>> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(MatrixError::IncompatibleDimensions {
                op: "element-wise multiplication".to_string(),
                dim1: (self.rows, self.cols),
                dim2: (other.rows, other.cols),
            });
        }

        let mut result = Matrix::new(self.rows, self.cols)?;
        result.set_concurrent(self.concurrent || other.concurrent);

        if result.concurrent {
            result.mat.par_iter_mut()
                .enumerate()
                .for_each(|(i, val)| {
                    *val = self.mat[i] * other.mat[i];
                });
        } else {
            for i in 0..self.mat.len() {
                result.mat[i] = self.mat[i] * other.mat[i];
            }
        }

        Ok(result)
    }

    pub fn matrix_multiply(&self, other: &Matrix<T>) -> MatrixResult<Matrix<T>> {
        if self.cols != other.rows {
            return Err(MatrixError::IncompatibleDimensions {
                op: "matrix multiplication".to_string(),
                dim1: (self.rows, self.cols),
                dim2: (other.rows, other.cols),
            });
        }

        let mut result = Matrix::new(self.rows, other.cols)?;
        result.set_concurrent(self.concurrent || other.concurrent);

        if result.concurrent {
            result.mat.par_chunks_mut(other.cols)
                .enumerate()
                .for_each(|(i, row)| {
                    for (j, val) in row.iter_mut().enumerate() {
                        let mut sum = T::default();
                        for k in 0..self.cols {
                            sum = sum + self.mat[i * self.cols + k] * other.mat[k * other.cols + j];
                        }
                        *val = sum;
                    }
                });
        } else {
            for i in 0..self.rows {
                for j in 0..other.cols {
                    let mut sum = T::default();
                    for k in 0..self.cols {
                        sum = sum + self.mat[i * self.cols + k] * other.mat[k * other.cols + j];
                    }
                    result.mat[i * other.cols + j] = sum;
                }
            }
        }

        Ok(result)
    }
}

// Determinant and matrix operations for floating point types
impl<T> Matrix<T>
where
    T: Default + Copy + Clone + Send + Sync + std::ops::Add<Output = T> + std::ops::Sub<Output = T> 
        + std::ops::Mul<Output = T> + std::ops::Div<Output = T> + std::ops::Neg<Output = T> 
        + PartialEq + PartialOrd + From<i32>,
{
    pub fn determinant(&self) -> MatrixResult<T> {
        if !self.is_square() {
            return Err(MatrixError::NotSquareMatrix {
                rows: self.rows,
                cols: self.cols,
            });
        }

        if self.rows == 1 {
            return Ok(self.mat[0]);
        }

        if self.rows == 2 {
            return Ok(self.mat[0] * self.mat[3] - self.mat[1] * self.mat[2]);
        }

        // For larger matrices, use LU decomposition or cofactor expansion
        self.determinant_lu()
    }

    fn determinant_lu(&self) -> MatrixResult<T> {
        let mut matrix = self.clone();
        let mut det = T::from(1);
        let n = self.rows;

        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in (i + 1)..n {
                if matrix.mat[k * n + i] > matrix.mat[max_row * n + i] 
                    || matrix.mat[k * n + i] < T::default() - matrix.mat[max_row * n + i] {
                    max_row = k;
                }
            }

            // Swap rows if needed
            if max_row != i {
                for j in 0..n {
                    let temp = matrix.mat[i * n + j];
                    matrix.mat[i * n + j] = matrix.mat[max_row * n + j];
                    matrix.mat[max_row * n + j] = temp;
                }
                det = T::default() - det;
            }

            // Check for singular matrix
            if matrix.mat[i * n + i] == T::default() {
                return Ok(T::default());
            }

            det = det * matrix.mat[i * n + i];

            // Eliminate below diagonal
            for k in (i + 1)..n {
                let factor = matrix.mat[k * n + i] / matrix.mat[i * n + i];
                for j in i..n {
                    matrix.mat[k * n + j] = matrix.mat[k * n + j] - factor * matrix.mat[i * n + j];
                }
            }
        }

        Ok(det)
    }

    pub fn cofactor_matrix(&self) -> MatrixResult<Matrix<T>> {
        if !self.is_square() {
            return Err(MatrixError::NotSquareMatrix {
                rows: self.rows,
                cols: self.cols,
            });
        }

        let mut result = Matrix::new(self.rows, self.cols)?;
        result.set_concurrent(self.concurrent);

        if self.concurrent {
            result.mat.par_chunks_mut(self.cols)
                .enumerate()
                .for_each(|(i, row)| {
                    for (j, val) in row.iter_mut().enumerate() {
                        let minor = self.minor_matrix(i, j).unwrap();
                        let cofactor = minor.determinant().unwrap();
                        *val = if (i + j) % 2 == 0 { cofactor } else { T::default() - cofactor };
                    }
                });
        } else {
            for i in 0..self.rows {
                for j in 0..self.cols {
                    let minor = self.minor_matrix(i, j)?;
                    let cofactor = minor.determinant()?;
                    result.mat[i * self.cols + j] = if (i + j) % 2 == 0 { cofactor } else { T::default() - cofactor };
                }
            }
        }

        Ok(result)
    }

    fn minor_matrix(&self, exclude_row: usize, exclude_col: usize) -> MatrixResult<Matrix<T>> {
        if self.rows <= 1 || self.cols <= 1 {
            return Err(MatrixError::InvalidDimensions);
        }

        let mut result = Matrix::new(self.rows - 1, self.cols - 1)?;
        let mut result_row = 0;

        for i in 0..self.rows {
            if i == exclude_row {
                continue;
            }
            let mut result_col = 0;
            for j in 0..self.cols {
                if j == exclude_col {
                    continue;
                }
                result.mat[result_row * (self.cols - 1) + result_col] = self.mat[i * self.cols + j];
                result_col += 1;
            }
            result_row += 1;
        }

        Ok(result)
    }
}

// Index traits with error handling
impl<T> Index<(usize, usize)> for Matrix<T>
where
    T: Default + Copy + Clone,
{
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (row, col) = index;
        if row >= self.rows || col >= self.cols {
            panic!("Index ({}, {}) out of bounds for matrix of size {}x{}", 
                   row, col, self.rows, self.cols);
        }
        &self.mat[row * self.cols + col]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T>
where
    T: Default + Copy + Clone,
{
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (row, col) = index;
        if row >= self.rows || col >= self.cols {
            panic!("Index ({}, {}) out of bounds for matrix of size {}x{}", 
                   row, col, self.rows, self.cols);
        }
        &mut self.mat[row * self.cols + col]
    }
}

// Operator overloading for addition
impl<T> Add for Matrix<T>
where
    T: Default + Copy + Clone + Send + Sync + std::ops::Add<Output = T>,
{
    type Output = MatrixResult<Matrix<T>>;

    fn add(self, other: Matrix<T>) -> Self::Output {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(MatrixError::IncompatibleDimensions {
                op: "addition".to_string(),
                dim1: (self.rows, self.cols),
                dim2: (other.rows, other.cols),
            });
        }

        let mut result = Matrix::new(self.rows, self.cols)?;
        result.set_concurrent(self.concurrent || other.concurrent);

        if result.concurrent {
            result.mat.par_iter_mut()
                .enumerate()
                .for_each(|(i, val)| {
                    *val = self.mat[i] + other.mat[i];
                });
        } else {
            for i in 0..self.mat.len() {
                result.mat[i] = self.mat[i] + other.mat[i];
            }
        }

        Ok(result)
    }
}

// Operator overloading for subtraction
impl<T> Sub for Matrix<T>
where
    T: Default + Copy + Clone + Send + Sync + std::ops::Sub<Output = T>,
{
    type Output = MatrixResult<Matrix<T>>;

    fn sub(self, other: Matrix<T>) -> Self::Output {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(MatrixError::IncompatibleDimensions {
                op: "subtraction".to_string(),
                dim1: (self.rows, self.cols),
                dim2: (other.rows, other.cols),
            });
        }

        let mut result = Matrix::new(self.rows, self.cols)?;
        result.set_concurrent(self.concurrent || other.concurrent);

        if result.concurrent {
            result.mat.par_iter_mut()
                .enumerate()
                .for_each(|(i, val)| {
                    *val = self.mat[i] - other.mat[i];
                });
        } else {
            for i in 0..self.mat.len() {
                result.mat[i] = self.mat[i] - other.mat[i];
            }
        }

        Ok(result)
    }
}

// Operator overloading for multiplication (matrix multiplication)
impl<T> Mul for Matrix<T> 
where 
    T: Default + Copy + Clone + Send + Sync 
        + std::ops::Add<Output = T>  
        + std::ops::Mul<Output = T> 
        + PartialEq 
        + Sub<Output = T>,          
{
    type Output = MatrixResult<Matrix<T>>;
    
    fn mul(self, other: Matrix<T>) -> Self::Output {
        self.matrix_multiply(&other)
    }
}
// Scalar multiplication
impl<T> Mul<T> for Matrix<T>
where
    T: Default + Copy + Clone + Send + Sync + std::ops::Mul<Output = T>,
{
    type Output = MatrixResult<Matrix<T>>;

    fn mul(self, scalar: T) -> Self::Output {
        let mut result = Matrix::new(self.rows, self.cols)?;
        result.set_concurrent(self.concurrent);

        if result.concurrent {
            result.mat.par_iter_mut()
                .enumerate()
                .for_each(|(i, val)| {
                    *val = self.mat[i] * scalar;
                });
        } else {
            for i in 0..self.mat.len() {
                result.mat[i] = self.mat[i] * scalar;
            }
        }

        Ok(result)
    }
}

// Scalar division  
impl<T> Div<T> for Matrix<T>
where
    T: Default + Copy + Clone + Send + Sync + std::ops::Div<Output = T> + PartialEq,
{
    type Output = MatrixResult<Matrix<T>>;

    fn div(self, scalar: T) -> Self::Output {
        if scalar == T::default() {
            return Err(MatrixError::DivisionByZero);
        }

        let mut result = Matrix::new(self.rows, self.cols)?;
        result.set_concurrent(self.concurrent);

        if result.concurrent {
            result.mat.par_iter_mut()
                .enumerate()
                .for_each(|(i, val)| {
                    *val = self.mat[i] / scalar;
                });
        } else {
            for i in 0..self.mat.len() {
                result.mat[i] = self.mat[i] / scalar;
            }
        }

        Ok(result)
    }
}

// Negation
impl<T> Neg for Matrix<T>
where
    T: Default + Copy + Clone + Send + Sync + std::ops::Neg<Output = T>,
{
    type Output = MatrixResult<Matrix<T>>;

    fn neg(self) -> Self::Output {
        let mut result = Matrix::new(self.rows, self.cols)?;
        result.set_concurrent(self.concurrent);

        if result.concurrent {
            result.mat.par_iter_mut()
                .enumerate()
                .for_each(|(i, val)| {
                    *val = -self.mat[i];
                });
        } else {
            for i in 0..self.mat.len() {
                result.mat[i] = -self.mat[i];
            }
        }

        Ok(result)
    }
}