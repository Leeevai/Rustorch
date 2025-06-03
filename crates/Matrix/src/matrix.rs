use crate::MatrixError;
use std::ops::AddAssign;
use std::ops::Mul;
use rayon::prelude::*;

#[derive(Clone,Debug)]
pub struct Matrix<T:Copy+Clone>
{
    rows: usize,
    cols: usize,
    mat: Vec<Vec<T>>,
}

impl<T:Copy+Clone+Default+ AddAssign + Mul<Output = T>+Send+Sync> Matrix<T>
{
    pub fn new(rows: usize, cols: usize)->Matrix<T>
    {
        Matrix {rows, cols, mat: vec![vec![T::default();cols];rows]}
    }

    pub fn set_mat(&mut self,matrix: Vec<Vec<T>>)->Result<(),MatrixError>
    {
        if (self.rows,self.cols) != (matrix.len(),matrix[0].len())
        {
            return Err(MatrixError::InvalidDimensions);
        }
        self.mat = matrix;
        Ok(())
    }

    pub fn fill(&mut self,filler: T)
    {
        for i in 0..self.rows
        {
            for j in 0..self.cols
            {
                self.mat[i][j]=filler;
            }
        }
    }

    pub fn product(&self, other: &Matrix<T>) -> Result<Matrix<T>, MatrixError> {
        if self.cols != other.rows {
            return Err(MatrixError::InvalidDimensions);
        }

        let mut result = Matrix {
            rows: self.rows,
            cols: other.cols,
            mat: vec![vec![T::default(); other.cols]; self.rows],
        };

        for i in 0..self.rows {
            for j in 0..other.cols {
                for k in 0..self.cols {
                    result.mat[i][j] += self.mat[i][k] * other.mat[k][j];
                }
            }
        }

        Ok(result)
    }
    pub fn prods(&self, other: &Matrix<T>)-> Result<Matrix<T>, MatrixError> {
        if self.cols != other.rows {
            return Err(MatrixError::InvalidDimensions);
        }

        let mut result = Matrix {
            rows: self.rows,
            cols: other.cols,
            mat: vec![vec![T::default(); other.cols]; self.rows],
        };

        result.mat.par_iter_mut().enumerate().for_each(|(i, row)| {
            for j in 0..other.cols {
                for k in 0..self.cols {
                    row[j] += self.mat[i][k] * other.mat[k][j];
                }
            }
        });

        Ok(result)
    }
}