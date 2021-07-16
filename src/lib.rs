/*!

# Eigenvalues decomposition

This crate contains implementations of several algorithm to
diagonalize symmetric matrices.

## Davidson Usage Example
```
// Use the Davidson method
use eigenvalues::{Davidson, DavidsonCorrection, SpectrumTarget};

// Generate random symmetric matrix
let matrix = eigenvalues::utils::generate_diagonal_dominant(20, 0.005);
let tolerance = 1e-4;

// Compute the first 2 lowest eigenvalues/eigenvectors using the DPR method
let eig = Davidson::new(
    matrix.view(), 2, DavidsonCorrection::DPR, SpectrumTarget::Lowest, tolerance).unwrap();
println!("eigenvalues:{}", eig.eigenvalues);
println!("eigenvectors:{}", eig.eigenvectors);

// Compute the first 2 highest eigenvalues/eigenvectors using the GJD method
let eig = Davidson::new(
    matrix.view(), 2, DavidsonCorrection::GJD, SpectrumTarget::Highest, tolerance).unwrap();
println!("eigenvalues:{}", eig.eigenvalues);
println!("eigenvectors:{}", eig.eigenvectors);
```

## Lanczos Usage Example
```
use ndarray::prelude::*;
use ndarray_linalg::*;

use eigenvalues::algorithms::lanczos::HermitianLanczos;
use eigenvalues::utils::{generate_random_sparse_symmetric, sort_eigenpairs};
use eigenvalues::SpectrumTarget;

// Generate sparse matrix
let matrix = generate_random_sparse_symmetric(100, 5, 0.5);

// Use 20 iterations to approximate the highest part of the spectrum
let spectrum_target = SpectrumTarget::Highest;
let lanczos = HermitianLanczos::new(matrix.view(), 20, spectrum_target).unwrap();
let (eigenvalues, eigenvectors): (Array1<f64>, Array2<f64>) = matrix.eigh(UPLO::Upper).unwrap();
let (eigenvalues, eigenvectors): (Array1<f64>, Array2<f64>) = sort_eigenpairs(eigenvalues, eigenvectors, false);
// Compare against reference
println!("Computed first three eigenvalues:\n{}", lanczos.eigenvalues.slice(s![0..3]));
println!("Expected first three eigenvalues:\n{}", eigenvalues.slice(s![0..3]));
```

*/

pub mod algorithms;
pub mod matrix_operations;
pub mod modified_gram_schmidt;
pub mod utils;
pub use algorithms::{davidson::Davidson, lanczos, DavidsonCorrection, SpectrumTarget};
pub use modified_gram_schmidt::MGS;
