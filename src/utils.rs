/*!

## Auxiliar functions to manipulate arrays

 */

use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use std::iter::FromIterator;
use approx::relative_eq;
use log::info;
use std::time::Instant;

/// Generate a random highly diagonal symmetric matrix
pub fn generate_diagonal_dominant(dim: usize, sparsity: f64) -> Array2<f64> {
    let diag: Array1<f64> = Array::random(dim, Uniform::new(0.0, dim as f64));
    let off_diag = Array::random((dim, dim), Uniform::new(0.0, 1.0));
    let arr = &off_diag + &off_diag.t();
    let mut arr = &arr * sparsity;
    arr.diag_mut().assign(&diag);
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

pub fn print_davidson_init(max_iter: usize, nroots: usize, tolerance: f64) {
    info!("{:^80}", "");
    info!("{: ^80}", "Iterative Davidson Routine");
    info!("{:-^80}", "");
    info!("{: <25} {:4.2e}", "Energy is converged when residual is below:", tolerance);
    info!("{: <25} {}", "Maximum number of iterations:", max_iter);
    info!("{: >4} {: <25}", nroots, " Roots will be computed.");
    info!("{:-^75} ", "");
    info!(
        "{: <5} {: >12} {: >12} {: >18} {: >12}",
        "Iter.", "Roots conv.", "Roots left", "Total dev.", "Max dev."
    );
    info!("{:-^75} ", "");
}

pub fn print_davidson_iteration(iter: usize, roots_cvd: usize, roots_lft: usize, t_dev: f64, max_dev:f64) {
    info!(
        "{: >5} {:>12} {:>12} {:>18.10e} {:>12.4}",
        iter + 1,
        roots_cvd,
        roots_lft,
        t_dev,
        max_dev
    );
}

pub fn print_davidson_end(result_is_ok: bool, time: Instant) {
    info!("{:-^75} ", "");
    if result_is_ok {
        info!("Davidson routine converged")
    } else {
        info!("Davidson routine did not converge!")
    }
    info!("{:>68} {:>8.2} s",
           "elapsed time:",
           time.elapsed().as_secs_f32()
    );
    info!("{:-^80}", "");
    info!("{:^80}", "");
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
