/*!

# Hermitian Lanczos algorithm

The [Hermitian Lanczos](https://en.wikipedia.org/wiki/Lanczos_algorithm) is an algorithm to compute the lowest/highest
eigenvalues of an hermitian matrix using a [Krylov subspace](https://en.wikipedia.org/wiki/Krylov_subspace)

*/
use super::SpectrumTarget;
use crate::utils;
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use ndarray_linalg::norm;
use nalgebra::linalg::SymmetricEigen;
use nalgebra::{DMatrix, DVector};
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
        let xs: Array1<f64> = norm::normalize(Array1::random((h.nrows()), Uniform::new(0.0, 1.0)));
        vs.slice_mut(s![.., 0]).assign(&xs);

        // Compute the elements of the tridiagonal matrix
        for i in 0..maximum_iterations {
            let tmp: Array1<f64> = h.dot(vs.column(i));
            alphas[i] = tmp.dot(&vs.column(i));
            let mut tmp = {
                if i == 0 {
                    &tmp - alphas[0] * vs.column(0)
                } else {
                    &tmp - alphas[i] * vs.column(i) - betas[i - 1] * vs.column(i - 1)
                }
            };
            // Orthogonalize with previous vectors
            for k in 0..i {
                let projection = tmp.dot(&vs.column(k));
                if projection.abs() > tolerance {
                    tmp -= projection * vs.column(i);
                }
            }
            if i < maximum_iterations - 1 {
                betas[i] = tmp.norm();
                if betas[i] > tolerance {
                    vs.set_column(i + 1, &(tmp / betas[i]));
                } else {
                    vs.set_column(i + 1, &tmp);
                }
            }
        }
        let tridiagonal = Self::construct_tridiagonal(alphas.view(), betas.view());
        let ord_sort = !matches!(spectrum_target, SpectrumTarget::Highest);
        let eig = utils::sort_eigenpairs(SymmetricEigen::new(tridiagonal), ord_sort);
        let eigenvalues = eig.eigenvalues;
        let eigenvectors = vs * eig.eigenvectors; // Ritz vectors

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
