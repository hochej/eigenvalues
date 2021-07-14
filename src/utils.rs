/*!

## Auxiliar functions to manipulate arrays

 */

use nalgebra as na;
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use std::iter::FromIterator;
use approx::relative_eq;
use na::linalg::SymmetricEigen;
use na::Dynamic;
use na::{DMatrix, DVector};

/// Generate a random highly diagonal symmetric matrix
pub fn generate_diagonal_dominant(dim: usize, sparsity: f64) -> Array2<f64> {
    let xs = 1..=dim;
    let it = xs.map(|x: usize| x as f64);
    let mut arr = Array::random((dim, dim), Uniform::new(0., 1.));
    arr += &arr.t();
    arr *= sparsity;
    arr.diag_mut().assign(&Array::from_iter(it));
    arr
}

/// Random symmetric matrix
pub fn generate_random_symmetric(dim: usize, magnitude: f64) -> Array2<f64> {
    let arr = Array::random((dim, dim), Uniform::new(0., 1.)) * magnitude;
    &arr * arr.transpose()
}

/// Random Sparse matrix
pub fn generate_random_sparse_symmetric(dim: usize, lim: usize, sparsity: f64) -> Array2<f64> {
    let arr = generate_diagonal_dominant(dim, sparsity);
    let lambda = |(i, j)| {
        if j > i + lim && i > j + lim {
            0.0
        } else {
            arr[[i, j]]
        }
    };
    Array::from_shape_fn((dim, dim), lambda)
}

/// Sort the eigenvalues and their corresponding eigenvectors in ascending order
pub fn sort_eigenpairs(
    eig: SymmetricEigen<f64, Dynamic>,
    ascending: bool,
) -> SymmetricEigen<f64, Dynamic> {
    // Sort the eigenvalues
    let mut vs: Vec<(f64, usize)> = eig
        .eigenvalues
        .iter()
        .enumerate()
        .map(|(idx, &x)| (x, idx))
        .collect();
    sort_vector(&mut vs, ascending);

    // Sorted eigenvalues
    let eigenvalues = DVector::<f64>::from_iterator(vs.len(), vs.iter().map(|t| t.0));

    // Indices of the sorted eigenvalues
    let indices: Vec<_> = vs.iter().map(|t| t.1).collect();

    // Create sorted eigenvectors
    let dim_rows = eig.eigenvectors.nrows();
    let dim_cols = eig.eigenvectors.ncols();
    let mut eigenvectors = DMatrix::<f64>::zeros(dim_rows, dim_cols);

    for (k, i) in indices.iter().enumerate() {
        eigenvectors.set_column(k, &eig.eigenvectors.column(*i));
    }
    SymmetricEigen {
        eigenvectors,
        eigenvalues,
    }
}

pub fn sort_vector<T: PartialOrd>(vs: &mut Vec<T>, ascending: bool) {
    if ascending {
        vs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    } else {
        vs.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
    }
}

pub fn test_eigenpairs(
    reference: &na::linalg::SymmetricEigen<f64, na::Dynamic>,
    eigenpair: (na::DVector<f64>, na::DMatrix<f64>),
    number: usize,
) {
    let (dav_eigenvalues, dav_eigenvectors) = eigenpair;
    for i in 0..number {
        // Test Eigenvalues
        assert!(relative_eq!(
            reference.eigenvalues[i],
            dav_eigenvalues[i],
            epsilon = 1e-6
        ));
        // Test Eigenvectors
        let x = reference.eigenvectors.column(i);
        let y = dav_eigenvectors.column(i);
        // The autovectors may different in their sign
        // They should be either parallel or antiparallel
        let dot = x.dot(&y).abs();
        assert!(relative_eq!(dot, 1.0, epsilon = 1e-6));
    }
}

#[cfg(test)]
mod test {
    use ndarray::prelude::*;
    use std::f64;

    #[test]
    fn test_random_symmetric() {
        let matrix = super::generate_random_symmetric(10, 2.5);
        test_symmetric(matrix);
    }
    #[test]
    fn test_diagonal_dominant() {
        let matrix = super::generate_diagonal_dominant(10, 0.005);
        test_symmetric(matrix);
    }

    fn test_symmetric(matrix: Array2<f64>) {
        let rs = &matrix - &matrix.transpose();
        assert!(rs.sum() < f64::EPSILON);
    }
}
