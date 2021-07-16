/*!

# Modified Gram-Schmidt (MGS)

The Gram-Schmidt method is a method for orthonormalising a set of vectors. see:
[Gram-Schmidt process](https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process)
The MGS method improves the orthogonality loss due to the finite numerical precision
on computers.
 */
use ndarray::prelude::*;
use ndarray_linalg::Norm;

pub struct MGS {
    pub basis: Array2<f64>,
}

impl MGS {
    /// The orthonormalize static method takes three argument:
    /// * `vectors` to diagonalize as columns of the matrix
    /// * `start` index of the column to start orthogonalizing
    /// * `end` last index of the column to diagonalize (non-inclusive)
    pub fn orthonormalize(mut basis: ArrayViewMut2<f64>, start: usize, end: usize) {
        for i in start..end {
            for j in 0..i {
                let proj = MGS::project(basis.column(j), basis.column(i));
                let new_column_i: Array1<f64> = &basis.column(i) - &proj;
                basis.slice_mut(s![.., i]).assign(&new_column_i);
            }
            let normalized_column: Array1<f64> = &basis.column(i) / basis.column(i).norm();
            basis.slice_mut(s![.., i]).assign(&normalized_column);
        }
    }

    // Project
    fn project(v1: ArrayView1<f64>, v2: ArrayView1<f64>) -> Array1<f64> {
        // THIS IS WRONG
        let magnitude = v1.dot(&v2) / v1.dot(&v1);
        magnitude * &v1
    }
}

#[cfg(test)]
mod test {
    use ndarray::prelude::*;

    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;

    #[test]
    fn test_gram_schmidt() {
        let dim = 20;
        let vectors = Array::random((dim, dim), Uniform::new(0.0, 1.0));
        fun_test(vectors, 0);
    }

    #[test]
    fn test_start_gram_schmidt() {
        let arr: Array2<f64> = array![[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 2.0]];
        fun_test(arr, 0);
    }

    fn fun_test(vectors: Array2<f64>, start: usize) {
        let mut basis = vectors.clone();
        super::MGS::orthonormalize(basis.view_mut(), start, vectors.ncols());
        let result: Array2<f64> = basis.t().dot(&basis);
        assert!((result.diag().sum() - (result.nrows() as f64)) < 1e-8);
    }
}
