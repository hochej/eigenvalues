use eigenvalues::utils::{generate_diagonal_dominant, test_eigenpairs};
use eigenvalues::{Davidson, utils};
use ndarray_linalg::*;
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

#[test]
fn test_davidson() {
    let arr = generate_diagonal_dominant(200, 0.005);
    let eig: (Array1<f64>, Array2<f64>) = arr.eigh(UPLO::Upper).unwrap();
    let tolerance = 1.0e-6;
    let n_roots: usize = 10;
    let max_iter: usize = 50;
    let guess: Array2<f64> = make_guess(arr.diag(), n_roots);
    let dav = Davidson::new(
        arr.view(),
        guess,
        n_roots,
        tolerance,
        max_iter
    )
    .unwrap();
    test_eigenpairs(eig.clone(), (dav.eigenvalues, dav.eigenvectors), 10);
}


