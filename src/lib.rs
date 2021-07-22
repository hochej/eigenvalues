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



*/
mod array_sort;
pub mod engine;
pub mod utils;
mod davidson;

pub use davidson::Davidson;
