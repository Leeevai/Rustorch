use std::time::Instant;
use Matrix::Matrix;
use rayon::ThreadPoolBuilder;
fn main() {
    // Create two 500x500 matrices filled with some dummy values
    let mut m1 :Matrix<usize> = Matrix::new(500, 500);
    let mut m2:Matrix<usize> = Matrix::new(500, 500);

    m1.set_mat((0..500).map(|i| (0..500).map(|j| (i + j) as usize).collect()).collect()).unwrap();
    m2.set_mat((0..500).map(|i| (0..500).map(|j| (i * j) as usize).collect()).collect()).unwrap();

    let t = Instant::now();
    let _prod_seq = m1.product(&m2).unwrap();
    let elapsed_seq = t.elapsed();
    println!("\nt_seq = {:?}", elapsed_seq);

    let t = Instant::now();
    let _prod_par = m1.prods(&m2).unwrap();
    let elapsed_par = t.elapsed();
    println!("\nt_par = {:?}", elapsed_par);
}