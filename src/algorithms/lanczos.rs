/*!

# Hermitian Lanczos algorithm

The [Hermitian Lanczos](https://en.wikipedia.org/wiki/Lanczos_algorithm) is an algorithm to compute the lowest/highest
eigenvalues of an hermitian matrix using a [Krylov subspace](https://en.wikipedia.org/wiki/Krylov_subspace)

*/
use crate::utils;
use super::SpectrumTarget;
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use ndarray_linalg::*;
use std::error;
use std::fmt;

#[derive(Debug, PartialEq)]
pub struct LanczosError;

impl fmt::Display for LanczosError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Lanczos Algorithm did not converge!")
    }
}

impl error::Error for LanczosError {}

pub struct HermitianLanczos {
    pub eigenvalues: Array1<f64>,
    pub eigenvectors: Array2<f64>,
}

impl HermitianLanczos {
    /// The new static method takes the following arguments:
    /// * `h` - A highly diagonal symmetric matrix
    /// * `maximum_iterations` - Krylov subspace size
    /// * `spectrum_target` Lowest or Highest part of the spectrum

    pub fn new(
        h: ArrayView2<f64>,
        maximum_iterations: usize,
        spectrum_target: SpectrumTarget,
    ) -> Result<Self, LanczosError> {
        let tolerance = 1e-8;

        // Off-diagonal elements
        let mut betas: Array1<f64> = Array1::zeros([maximum_iterations - 1]);
        // Diagonal elements
        let mut alphas: Array1<f64> = Array1::zeros([maximum_iterations]);

        // Matrix with the orthognal vectors
        let mut vs: Array2<f64> = Array2::zeros([h.nrows(), maximum_iterations]);

        // Initial vector
        let mut xs: Array1<f64> = Array::random(h.nrows(), Uniform::new(0.0, 1.0));
        xs = &xs / xs.norm();
        vs.slice_mut(s![.., 0]).assign(&xs);

        // Compute the elements of the tridiagonal matrix
        for i in 0..maximum_iterations {
            let tmp: Array1<f64> = h.dot(&vs.column(i));
            alphas[i] = tmp.dot(&vs.column(i));
            let mut tmp = {
                if i == 0 {
                    &tmp - &(alphas[0] * &vs.column(0))
                } else {
                    &tmp - &(alphas[i] * &vs.column(i)) - &(betas[i - 1] * &vs.column(i - 1))
                }
            };
            // Orthogonalize with previous vectors
            for k in 0..i {
                let projection = tmp.dot(&vs.column(k));
                if projection.abs() > tolerance {
                    tmp -= &(projection * &vs.column(i));
                }
            }
            if i < maximum_iterations - 1 {
                betas[i] = tmp.norm();
                if betas[i] > tolerance {
                    vs.slice_mut(s![.., i + 1]).assign(&(tmp / betas[i]));
                } else {
                    vs.slice_mut(s![.., i + 1]).assign(&tmp);
                }
            }
        }
        let tridiagonal: Array2<f64> = Self::construct_tridiagonal(alphas.view(), betas.view());
        let ord_sort = !matches!(spectrum_target, SpectrumTarget::Highest);
        let (eigenvalues, eigenvectors): (Array1<f64>, Array2<f64>) = tridiagonal.eigh(UPLO::Upper).unwrap();
        let (eigenvalues, eigenvectors): (Array1<f64>, Array2<f64>) = utils::sort_eigenpairs(eigenvalues, eigenvectors, ord_sort);

        let eigenvectors: Array2<f64> = vs.dot(&eigenvectors); // Ritz vectors

        Ok(HermitianLanczos {
            eigenvalues,
            eigenvectors,
        })
    }

    fn construct_tridiagonal(alphas: ArrayView1<f64>, betas: ArrayView1<f64>) -> Array2<f64> {
        let dim = alphas.len();
        let lambda = |(i, j)| {
            if i == j {
                alphas[i]
            } else if i == j + 1 {
                betas[j]
            } else if j == i + 1 {
                betas[i]
            } else {
                0.0
            }
        };
        Array::from_shape_fn((dim, dim), lambda)
    }
}
