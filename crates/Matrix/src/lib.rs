mod matrix;
mod error;
pub use matrix::*;
pub use error::MatrixError;
pub use std::time::Instant;
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test()
    {
        let mat: Matrix<usize> = Matrix::new(3,3);
        let vb = vec![ 1 ,2 ,3];
        let va = vec![vb.clone(),vb.clone().iter().map(|x| x+3).collect(),vb.iter().map(|x| x+6).collect()];
        println!("{:?}",va);
        println!(" mat(2,1) = {}",va[2][1]);

        let mut met : Matrix<usize>= Matrix::new(1000,1000) ;
       met.fill(5);
       let mut met2 : Matrix<usize>= Matrix::new(1000,1000) ;
       met2.fill(15);

       let t = Instant::now();
       let prod1 = met.product(&met2);
       let elaps = t.elapsed();
        println!("\n\nt_seq = {:?}",elaps);

        let t = Instant::now();
       let prod1 = met.prods(&met2);
       let elaps = t.elapsed();
        println!("\n\nt_par = {:?}",elaps);

    }

}