/*!

# Modified Gram-Schmidt (MGS)

The Gram-Schmidt method is a method for orthonormalising a set of vectors. see:
[Gram-Schmidt process](https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process)
The MGS method improves the orthogonality loss due to the finite numerical precision
on computers.
 */
use ndarray::prelude::*;
use ndarray_linalg::Norm;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

pub struct MGS {
    pub basis: Array2<f64>,
}

impl MGS {
    /// The orthonormalize static method takes three argument:
    /// * `vectors` to diagonalize as columns of the matrix
    /// * `start` index of the column to start orthogonalizing
    /// * `end` last index of the column to diagonalize (non-inclusive)
    pub fn orthonormalize(basis: ArrayViewMut2<f64>, start: usize, end: usize) {
        for i in start..end {
            for j in 0..i {
                let proj = MGS::project(basis.column(j), basis.column(i));
                basis.slice_mut(s![.., i]).assign(&(&basis.column(i) - &proj));
            }
            basis.slice_mut(s![.., i]).assign(&(&basis.column(i) / basis.column(i).norm()));
        }
    }

    // Project
    fn project(v1: ArrayView1<f64>, v2: ArrayView1<f64>) -> Array2<f64> {
        // THIS IS WRONG
        let magnitud = v1.dot(&v2) / v1.dot(&v1);
        v1 * magnitud
    }
}

#[cfg(test)]
mod test {
    use ndarray::prelude::*;

    #[test]
    fn test_gram_schmidt() {
        let dim = 20;
        let vectors = Array::random((dim, dim), Uniform::new(0.0, 1.0));
        fun_test(vectors, 0);
    }

    #[test]
    fn test_start_gram_schmidt() {
        let arr: Array1<f64> = array![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 2.0];
        fun_test(arr, 1);
    }

    fn fun_test(vectors: Array2<f64>, start: usize) {
        let mut basis = vectors.clone();
        super::MGS::orthonormalize(&mut basis, start, vectors.ncols());
        let result: Array2<f64> = basis.t() * &basis;
        assert!(result.is_identity(1e-8));
    }
}
