/*!

## Auxiliar functions to manipulate arrays

 */

use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use std::iter::FromIterator;
use approx::relative_eq;

/// Generate a random highly diagonal symmetric matrix
pub fn generate_diagonal_dominant(dim: usize, sparsity: f64) -> Array2<f64> {
    let xs = 1..=dim;
    let it = xs.map(|x: usize| x as f64);
    let arr = Array::random((dim, dim), Uniform::new(0.0, 1.0));
    let arr = &arr + &arr.t();
    let mut arr = &arr * sparsity;
    arr.diag_mut().assign(&Array::from_iter(it));
    arr
}

/// Random symmetric matrix
pub fn generate_random_symmetric(dim: usize, magnitude: f64) -> Array2<f64> {
    let arr = Array::random((dim, dim), Uniform::new(0.0, 1.0)) * magnitude;
    arr.dot(&arr.t())
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
    eigenvalues: Array1<f64>,
    eigenvectors: Array2<f64>,
    ascending: bool,
) -> (Array1<f64>, Array2<f64>) {
    // Sort the eigenvalues
    let mut vs: Vec<(f64, usize)> = eigenvalues
        .iter()
        .enumerate()
        .map(|(idx, &x)| (x, idx))
        .collect();
    sort_vector(&mut vs, ascending);

    // Sorted eigenvalues
    let eigenvalues: Array1<f64> = Array1::from_iter(vs.iter().map(|t| t.0));

    // Indices of the sorted eigenvalues
    let indices: Vec<usize> = vs.iter().map(|t| t.1).collect();

    // Create sorted eigenvectors
    let mut sorted_eigenvectors: Array2<f64> = Array2::zeros(eigenvectors.dim());

    for (k, i) in indices.iter().enumerate() {
        sorted_eigenvectors.slice_mut(s![.., k]).assign(&eigenvectors.column(*i));
    }

    (eigenvalues, sorted_eigenvectors)
}

pub fn sort_vector<T: PartialOrd>(vs: &mut Vec<T>, ascending: bool) {
    if ascending {
        vs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    } else {
        vs.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
    }
}

pub fn test_eigenpairs(
    ref_eigenpair: (Array1<f64>, Array2<f64>),
    eigenpair: (Array1<f64>, Array2<f64>),
    number: usize,
) {
    let (dav_eigenvalues, dav_eigenvectors) = eigenpair;
    let (ref_eigenvalues, ref_eigenvectors) = ref_eigenpair;
    for i in 0..number {
        // Test Eigenvalues
        assert!(relative_eq!(
            ref_eigenvalues[i],
            dav_eigenvalues[i],
            epsilon = 1e-6
        ));
        // Test Eigenvectors
        let x = ref_eigenvectors.column(i);
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
        let rs = &matrix - &matrix.t();
        assert!(rs.sum() < f64::EPSILON);
    }
}
