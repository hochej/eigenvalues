use ndarray::prelude::*;
use ndarray::DataMut;

/// Abstract Trait defining the API required by solver engines.
///
/// Engines implement the correct product functions for iterative solvers that
/// do not require the target matrix be stored directly.
/// Classes intended to be used as an `engine` for `Davidson` or
/// `Hamiltonian` should implement this Trait to ensure
/// that the required methods are defined.
pub trait DavidsonEngine {
    /// Compute a Matrix * trial vector products
    /// Expected output:
    ///  The product`A x X_{i}` for each `X_{i}` in `X`, in that order.
    ///   Where `A` is the hermitian matrix to be diagonalized.
    fn compute_products(&self, x: ArrayView2<f64>) -> Array2<f64>;

    /// Apply the preconditioner to a Residual vector.
    /// The preconditioner is usually defined as :math:`(w_k - D_{i})^-1` where
    /// `D` is an approximation of the diagonal of the matrix that is being diagonalized.
    fn precondition(&self, r_k: ArrayView1<f64>, w_k:f64) -> Array1<f64>;

    /// Return the size of the matrix problem.
    fn get_size(&self) -> usize;
}

impl<S> DavidsonEngine for ArrayBase<S, Ix2>
    where
        S: DataMut<Elem = f64>,
{
    fn compute_products(&self, x: ArrayView2<'_, f64>) -> Array2<f64> {
        self.dot(&x)
    }

    fn precondition(&self, r_k: ArrayView1<'_, f64>, w_k: f64) -> Array1<f64> {
        &r_k / &(Array1::from_elem(self.nrows(), w_k) - &self.diag())
    }

    fn get_size(&self) -> usize {
        self.nrows()
    }
}