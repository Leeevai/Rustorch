#[derive(Clone,Debug)]
pub enum MatrixError{
    InvalidMatrixDimension,
    InvalidRowDimension,
    InvalidColumnDimension,
    InvalidDimensions
}