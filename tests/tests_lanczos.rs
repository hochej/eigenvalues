use eigenvalues::algorithms::lanczos::HermitianLanczos;
use eigenvalues::utils::{generate_random_sparse_symmetric, sort_eigenpairs, test_eigenpairs};
use eigenvalues::SpectrumTarget;
use ndarray::prelude::*;
use ndarray_linalg::*;

#[test]
fn test_lanczos() {
    let matrix = generate_random_sparse_symmetric(100, 5, 0.5);
    let (eigenvalues, eigenvectors): (Array1<f64>, Array2<f64>) = matrix.eigh(UPLO::Upper).unwrap();
    let (eigenvalues, eigenvectors): (Array1<f64>, Array2<f64>) = sort_eigenpairs(eigenvalues, eigenvectors, false);
    let spectrum_target = SpectrumTarget::Highest;
    let lanczos = HermitianLanczos::new(matrix.view(), 40, spectrum_target).unwrap();

    println!("Computed eigenvalues:\n{}", lanczos.eigenvalues);
    println!("Expected eigenvalues:\n{}", eigenvalues);
    test_eigenpairs((eigenvalues, eigenvectors), (lanczos.eigenvalues, lanczos.eigenvectors), 1);
}
