
use eigenvalues::algorithms::lanczos::HermitianLanczos;
use eigenvalues::utils::{generate_random_sparse_symmetric, sort_eigenpairs};
use eigenvalues::SpectrumTarget;
use ndarray_linalg::*;
use ndarray::prelude::*;

fn main() {
    let matrix = generate_random_sparse_symmetric(200, 5, 0.5);
    let (eigenvalues, eigenvectors): (Array1<f64>, Array2<f64>) = matrix.eigh(UPLO::Upper).unwrap();
    let (eigenvalues, eigenvectors): (Array1<f64>, Array2<f64>) = sort_eigenpairs(eigenvalues, eigenvectors, false);
    let spectrum_target = SpectrumTarget::Highest;
    let lanczos = HermitianLanczos::new(matrix.view(), 50, spectrum_target).unwrap();
    println!("Computed eigenvalues:\n{}", lanczos.eigenvalues.slice(s![0..3]));
    println!("Expected eigenvalues:\n{}", eigenvalues.slice(s![0..3]));
    let x = eigenvectors.column(0);
    let y = lanczos.eigenvectors.column(0);
    println!("parallel:{}", x.dot(&y));
}
