/*!

# Eigenvalues decomposition

This crate contains implementations of several algorithm to
diagonalize symmetric matrices.

## Davidson Usage Example
```
// Use the Davidson method
use eigenvalues::{Davidson};
use eigenvalues::utils;
use ndarray::prelude::*;

fn make_guess(diag: ArrayView1<f64>, dim: usize) -> Array2<f64> {
    let order: Vec<usize> = utils::argsort(diag.view());
    let mut mtx: Array2<f64> = Array2::zeros([diag.len(), dim]);
    for (idx, i) in order.into_iter().enumerate() {
        if idx < dim {
            mtx[[i, idx]] = 1.0;
        }
    }
    mtx
}

// Generate random symmetric matrix with 300 rows and 300 columns
let matrix: Array2<f64> = eigenvalues::utils::generate_diagonal_dominant(300, 0.005);
// Set the residue tolerance for the convergence of the Davidson routine
let tolerance: f64 = 1e-6;
let max_iter: usize = 50;
let n_roots: usize = 2;
// Compute 6 orthonormal guess vectors.
let guess: Array2<f64> = make_guess(matrix.diag(), 6);

// Compute the first 2 lowest eigenvalues/eigenvectors
let eig = Davidson::new(matrix.view(), guess, n_roots, tolerance, max_iter).unwrap();
println!("eigenvalues:{}", eig.eigenvalues);
println!("eigenvectors:{}", eig.eigenvectors);
```



*/
pub mod engine;
pub mod utils;
mod davidson;

pub use davidson::Davidson;
