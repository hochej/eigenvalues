
use eigenvalues::{Davidson, DavidsonCorrection, SpectrumTarget};
use eigenvalues::utils::{generate_random_sparse_symmetric, sort_eigenpairs};
use ndarray_linalg::*;
use ndarray::prelude::*;

fn main() {
    let matrix = generate_random_sparse_symmetric(200, 5, 0.5);
    let tolerance = 1e-6;
    let (eigenvalues, eigenvectors): (Array1<f64>, Array2<f64>) = matrix.eigh(UPLO::Upper).unwrap();
    let (eigenvalues, eigenvectors): (Array1<f64>, Array2<f64>) = sort_eigenpairs(eigenvalues, eigenvectors, false);
    let spectrum_target = SpectrumTarget::Highest;
    // Compute the first 5 lowest eigenvalues/eigenvectors using the DPR method
    let dav = Davidson::new(
        matrix.view(), 5, DavidsonCorrection::DPR, spectrum_target, tolerance).unwrap();
    println!("Computed eigenvalues:\n{}", dav.eigenvalues.slice(s![0..5]));
    println!("Expected eigenvalues:\n{}", eigenvalues.slice(s![0..5]));
    let x = eigenvectors.column(0);
    let y = dav.eigenvectors.column(0);
    println!("parallel:{}", x.dot(&y));
}
